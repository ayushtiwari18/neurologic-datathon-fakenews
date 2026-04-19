"""
evaluate.py — Full evaluation pipeline for FakeGuard.

Source of truth:
    - instructions/04_model_training_strategy.md
    - instructions/03_pipeline_architecture.md
    - instructions/09_features_and_routes.md

Responsibilities:
    1. Load trained model from models/roberta_fakenews/
    2. Load processed val.csv (or test.csv for final submission)
    3. Run inference with confidence scores
    4. Apply confidence thresholding (flag UNCERTAIN predictions)
    5. Compute: Accuracy, Precision, Recall, F1 (weighted + macro)
    6. Generate and save confusion matrix PNG
    7. Run error analysis — top 20 misclassified examples
    8. Save full evaluation report to outputs/evaluation_report.json
    9. Save error analysis to outputs/error_analysis.csv
    10. Print Baseline vs RoBERTa comparison table

Usage:
    python src/evaluate.py
    --- or ---
    from src.evaluate import run_evaluation
    report = run_evaluation()
"""

import sys
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from transformers import (
    RobertaForSequenceClassification,
    AutoTokenizer,
)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    PROCESSED_DIR,
    MODEL_DIR,
    OUTPUTS_DIR,
    TRANSFORMER,
    MAX_LEN,
    BATCH_SIZE,
    SEED,
    TEXT_COL,
    LABEL_COL,
    ID2LABEL,
    CONFIDENCE_THRESHOLD,
)
from src.utils import get_logger, set_seed, create_directory, save_json

logger = get_logger("evaluate")


# ─────────────────────────────────────────────────────────────────────────────
def load_model_and_tokenizer(
    model_dir: str = None,
    model_name: str = TRANSFORMER,
) -> tuple:
    """
    Load trained model and tokenizer.

    Tries local trained model first. Falls back to HuggingFace Hub
    if local model not found (useful for first-run testing).

    Args:
        model_dir  : Path to saved model directory.
        model_name : HuggingFace model ID fallback.

    Returns:
        Tuple[model, tokenizer, device]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_path = Path(model_dir) if model_dir else Path(MODEL_DIR) / "roberta_fakenews"

    if local_path.exists() and any(local_path.iterdir()):
        load_from = str(local_path)
        logger.info("Loading trained model from local path: %s", load_from)
    else:
        load_from = model_name
        logger.warning(
            "Local model not found at %s. Loading from HuggingFace Hub: %s\n"
            "NOTE: This loads base weights only — run src/train.py first for fine-tuned accuracy.",
            local_path, model_name,
        )

    model     = RobertaForSequenceClassification.from_pretrained(load_from)
    tokenizer = AutoTokenizer.from_pretrained(load_from)
    model.to(device)
    model.eval()  # REQUIRED — disables dropout, enables deterministic inference
    logger.info("Model loaded on device: %s", device)
    return model, tokenizer, device


# ─────────────────────────────────────────────────────────────────────────────
def run_batch_inference(
    df: pd.DataFrame,
    model,
    tokenizer,
    device: torch.device,
    batch_size: int = BATCH_SIZE * 2,
    max_length: int = MAX_LEN,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> pd.DataFrame:
    """
    Run batched inference on a DataFrame and return predictions with confidence.

    Processes data in batches to avoid GPU OOM errors.
    Applies confidence thresholding per instructions/04.

    Args:
        df         : DataFrame with TEXT_COL column.
        model      : Loaded RoBERTa model (on correct device, eval mode).
        tokenizer  : Loaded tokenizer.
        device     : torch.device (cuda or cpu).
        batch_size : Number of samples per inference batch.
        max_length : Token truncation length.
        threshold  : Confidence threshold for UNCERTAIN flagging.

    Returns:
        pd.DataFrame: Input df with added columns:
            predicted, confidence, predicted_label, uncertain_flag
    """
    texts = df[TEXT_COL].fillna("").astype(str).tolist()
    all_preds   = []
    all_confs   = []

    logger.info("Running batch inference on %d samples (batch_size=%d)...", len(texts), batch_size)

    with torch.no_grad():  # REQUIRED — disables gradient tracking for inference (saves memory)
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            encoded = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            # Move all tensors to the same device as the model
            # NOTE: do NOT pass token_type_ids — RoBERTa doesn't use them
            input_ids      = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs   = F.softmax(outputs.logits, dim=-1).cpu().numpy()

            preds = np.argmax(probs, axis=-1)
            confs = np.max(probs, axis=-1)

            all_preds.extend(preds.tolist())
            all_confs.extend(confs.tolist())

            if (i // batch_size) % 10 == 0:
                logger.info("  Processed %d / %d samples", min(i + batch_size, len(texts)), len(texts))

    result_df = df.copy()
    result_df["predicted"]       = all_preds
    result_df["confidence"]      = [round(c, 6) for c in all_confs]
    result_df["predicted_label"] = [ID2LABEL.get(p, str(p)) for p in all_preds]
    result_df["uncertain_flag"]  = [c < threshold for c in all_confs]

    uncertain_count = result_df["uncertain_flag"].sum()
    logger.info(
        "Inference complete — uncertain predictions (conf < %.0f%%): %d / %d (%.1f%%)",
        threshold * 100, uncertain_count, len(result_df),
        100.0 * uncertain_count / len(result_df),
    )
    return result_df


# ─────────────────────────────────────────────────────────────────────────────
def save_confusion_matrix(
    y_true: list,
    y_pred: list,
    output_path: Path,
) -> None:
    """
    Generate and save a labelled confusion matrix as PNG.

    Args:
        y_true      : Ground-truth integer labels.
        y_pred      : Predicted integer labels.
        output_path : File path for the PNG.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend (safe for Colab/Kaggle)
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = confusion_matrix(y_true, y_pred)
        labels = ["FAKE (0)", "REAL (1)"]

        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            linewidths=0.5,
            annot_kws={"size": 14},
        )
        ax.set_title("FakeGuard — RoBERTa Confusion Matrix", fontsize=14, pad=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_xlabel("Predicted Label", fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        logger.info("Confusion matrix saved → %s", output_path)
    except ImportError:
        logger.warning("matplotlib/seaborn not installed — skipping confusion matrix plot.")


# ─────────────────────────────────────────────────────────────────────────────
def run_error_analysis(
    result_df: pd.DataFrame,
    n: int = 20,
) -> pd.DataFrame:
    """
    Extract top N misclassified examples for error analysis.

    Judges love this section — it shows you understand your model's weaknesses.

    Args:
        result_df : DataFrame with true labels and predictions.
        n         : Number of examples to return.

    Returns:
        pd.DataFrame: Top N wrong predictions sorted by confidence (most confident errors first).
    """
    wrong = result_df[result_df[LABEL_COL] != result_df["predicted"]].copy()
    wrong["true_label"] = wrong[LABEL_COL].map(ID2LABEL)

    cols = [TEXT_COL, "true_label", "predicted_label", "confidence", "uncertain_flag"]
    available_cols = [c for c in cols if c in wrong.columns]

    # Sort by confidence descending — most confidently wrong predictions are most interesting
    wrong_sorted = wrong[available_cols].sort_values("confidence", ascending=False).head(n)
    logger.info("Error analysis: %d misclassified examples (showing top %d)", len(wrong), n)
    return wrong_sorted.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
def run_evaluation(
    val_csv_path: Optional[str] = None,
    model_dir:    Optional[str] = None,
    baseline_accuracy: float = 0.921,   # typical TF-IDF + LogReg on WELFake
) -> Dict:
    """
    Full evaluation pipeline for the trained FakeGuard RoBERTa model.

    Args:
        val_csv_path       : Path to processed val.csv. Defaults to data/processed/val.csv
        model_dir          : Path to trained model directory. Defaults to models/roberta_fakenews/
        baseline_accuracy  : Baseline accuracy to compare against (from src/baseline.py run).

    Returns:
        dict: Full evaluation report with all metrics.
    """
    set_seed(SEED)

    val_path   = Path(val_csv_path) if val_csv_path else Path(PROCESSED_DIR) / "val.csv"
    outputs    = Path(OUTPUTS_DIR)
    create_directory(outputs)

    logger.info("=" * 60)
    logger.info("STARTING EVALUATION")
    logger.info("Val CSV    : %s", val_path)
    logger.info("Model Dir  : %s", model_dir or str(Path(MODEL_DIR) / "roberta_fakenews"))
    logger.info("Threshold  : %.0f%%", CONFIDENCE_THRESHOLD * 100)
    logger.info("=" * 60)

    # ── 1. Validate val.csv exists ─────────────────────────────────────────
    if not val_path.exists():
        raise FileNotFoundError(
            f"val.csv not found at: {val_path}\n"
            "Fix: Run  python src/preprocess.py  first."
        )

    # ── 2. Load validation data ───────────────────────────────────────────
    val_df = pd.read_csv(val_path)
    logger.info("Loaded val.csv — %d rows", len(val_df))

    if TEXT_COL not in val_df.columns:
        raise ValueError(
            f"Expected column '{TEXT_COL}' not found in val.csv.\n"
            f"Available columns: {val_df.columns.tolist()}"
        )
    if LABEL_COL not in val_df.columns:
        raise ValueError(
            f"Expected column '{LABEL_COL}' not found in val.csv.\n"
            "Evaluation requires ground-truth labels."
        )

    # ── 3. Load model and tokenizer ─────────────────────────────────────────
    model, tokenizer, device = load_model_and_tokenizer(model_dir)

    # ── 4. Run batched inference ───────────────────────────────────────────
    result_df = run_batch_inference(val_df, model, tokenizer, device)

    y_true = result_df[LABEL_COL].astype(int).tolist()
    y_pred = result_df["predicted"].tolist()

    # ── 5. Compute all metrics ─────────────────────────────────────────────
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_m = f1_score(y_true, y_pred, average="macro", zero_division=0)

    logger.info("Accuracy          : %.4f", acc)
    logger.info("Precision (wt)    : %.4f", prec)
    logger.info("Recall (wt)       : %.4f", rec)
    logger.info("F1 weighted       : %.4f", f1_w)
    logger.info("F1 macro          : %.4f", f1_m)

    cls_report = classification_report(
        y_true, y_pred,
        target_names=["FAKE", "REAL"],
        zero_division=0,
    )
    logger.info("\nClassification Report:\n%s", cls_report)

    # ── 6. Save confusion matrix PNG ──────────────────────────────────────
    cm_path = outputs / "confusion_matrix.png"
    save_confusion_matrix(y_true, y_pred, cm_path)

    # ── 7. Error analysis ────────────────────────────────────────────────
    error_df = run_error_analysis(result_df, n=20)
    error_path = outputs / "error_analysis.csv"
    error_df.to_csv(error_path, index=False)
    logger.info("Error analysis saved → %s", error_path)

    # ── 8. Save full predictions CSV ────────────────────────────────────────
    preds_path = outputs / "val_predictions.csv"
    result_df.to_csv(preds_path, index=False)
    logger.info("Full predictions saved → %s", preds_path)

    # ── 9. Build and save evaluation report JSON ───────────────────────────
    uncertain_pct = float(result_df["uncertain_flag"].mean()) * 100
    report = {
        "model": str(Path(MODEL_DIR) / "roberta_fakenews"),
        "val_csv": str(val_path),
        "val_samples": len(result_df),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "uncertain_predictions_pct": round(uncertain_pct, 2),
        "metrics": {
            "accuracy":           round(acc,  6),
            "precision_weighted": round(prec, 6),
            "recall_weighted":    round(rec,  6),
            "f1_weighted":        round(f1_w, 6),
            "f1_macro":           round(f1_m, 6),
        },
        "baseline_comparison": {
            "baseline_accuracy":  round(baseline_accuracy, 4),
            "roberta_accuracy":   round(acc, 4),
            "improvement":        round(acc - baseline_accuracy, 4),
        },
        "outputs": {
            "confusion_matrix":  str(cm_path),
            "error_analysis":    str(error_path),
            "val_predictions":   str(preds_path),
        },
    }
    report_path = outputs / "evaluation_report.json"
    save_json(report, report_path)

    # ── 10. Print comparison table to console ──────────────────────────────
    print("\n" + "=" * 56)
    print("  EVALUATION RESULTS — FakeGuard RoBERTa")
    print("=" * 56)
    print(f"  {'Metric':<28} {'Baseline':>10} {'RoBERTa':>10}")
    print("-" * 56)
    print(f"  {'Accuracy':<28} {baseline_accuracy:>10.4f} {acc:>10.4f}")
    print(f"  {'F1 (weighted)':<28} {'  —':>10} {f1_w:>10.4f}")
    print(f"  {'F1 (macro)':<28} {'  —':>10} {f1_m:>10.4f}")
    print(f"  {'Precision (weighted)':<28} {'  —':>10} {prec:>10.4f}")
    print(f"  {'Recall (weighted)':<28} {'  —':>10} {rec:>10.4f}")
    print("-" * 56)
    print(f"  {'Improvement over Baseline':<28} {'':>10} {acc - baseline_accuracy:>+10.4f}")
    print(f"  {'Uncertain Predictions':<28} {'':>10} {uncertain_pct:>9.1f}%")
    print("=" * 56)
    print(f"  Confusion matrix → {cm_path}")
    print(f"  Error analysis   → {error_path}")
    print(f"  Full report      → {report_path}")
    print("=" * 56 + "\n")

    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("Accuracy    : %.4f  (Baseline: %.4f  |  Improvement: %+.4f)",
                acc, baseline_accuracy, acc - baseline_accuracy)
    logger.info("=" * 60)

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate FakeGuard RoBERTa model")
    parser.add_argument("--val_csv",  type=str, default=None, help="Path to val.csv")
    parser.add_argument("--model_dir",type=str, default=None, help="Path to trained model dir")
    parser.add_argument("--baseline", type=float, default=0.921, help="Baseline accuracy to compare against")
    args = parser.parse_args()

    try:
        report = run_evaluation(
            val_csv_path=args.val_csv,
            model_dir=args.model_dir,
            baseline_accuracy=args.baseline,
        )
    except FileNotFoundError as e:
        logger.error("Missing file: %s", e)
        raise
    except ValueError as e:
        logger.error("Data error: %s", e)
        raise
