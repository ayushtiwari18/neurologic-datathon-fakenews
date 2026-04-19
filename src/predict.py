"""
predict.py — Competition submission generator for FakeGuard.

Source of truth:
    - instructions/08_key_failure_points.md  (CRITICAL: test set may have no labels)
    - instructions/04_model_training_strategy.md
    - instructions/09_features_and_routes.md

Responsibilities:
    1. Load processed test.csv from data/processed/
    2. Detect whether test.csv has labels (labeled = eval mode, unlabeled = submission mode)
    3. Run batched inference using trained RoBERTa model
    4. Apply confidence thresholding (UNCERTAIN flag for low-confidence predictions)
    5. Save outputs/predictions.csv in submission-ready format
    6. If labels exist: also print accuracy + F1 as a sanity check

CRITICAL WARNING (from instructions/08_key_failure_points.md):
    The competition test set may NOT have a label column.
    NEVER try to compute accuracy on the unlabeled test set.
    Always split your OWN train data into train/val for metric reporting.
    Use test.csv ONLY for generating the submission file.

Output format (outputs/predictions.csv):
    - id         : row index (0-based)
    - combined   : original combined text (for audit)
    - predicted  : integer prediction (0=FAKE, 1=REAL)
    - label      : string label (FAKE / REAL)
    - confidence : model confidence score (0.0 – 1.0)
    - uncertain  : True if confidence < CONFIDENCE_THRESHOLD

Usage:
    python src/predict.py
    python src/predict.py --test_csv data/processed/test.csv
    python src/predict.py --test_csv data/processed/test.csv --model_dir models/roberta_fakenews
    --- or ---
    from src.predict import run_predict
    predictions_df = run_predict()
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import (
    RobertaForSequenceClassification,
    AutoTokenizer,
)
from sklearn.metrics import accuracy_score, f1_score

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
from src.utils import get_logger, set_seed, create_directory

logger = get_logger("predict")


# ─────────────────────────────────────────────────────────────────────────────
def _load_model_and_tokenizer(model_dir: Optional[str]) -> tuple:
    """
    Load trained model and tokenizer from local path or HuggingFace Hub.

    Args:
        model_dir: Path to saved model directory. If None, uses default.

    Returns:
        Tuple[model, tokenizer, device]
    """
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_path = Path(model_dir) if model_dir else Path(MODEL_DIR) / "roberta_fakenews"

    if local_path.exists() and any(local_path.iterdir()):
        load_from = str(local_path)
        logger.info("Loading trained model from: %s", load_from)
    else:
        load_from = TRANSFORMER
        logger.warning(
            "Trained model not found at '%s'.\n"
            "Falling back to base '%s' weights — predictions will NOT be accurate.\n"
            "Run  python src/train.py  first to get competition-grade accuracy.",
            local_path, TRANSFORMER,
        )

    model     = RobertaForSequenceClassification.from_pretrained(load_from)
    tokenizer = AutoTokenizer.from_pretrained(load_from)
    model.to(device)
    model.eval()   # REQUIRED — disables dropout for deterministic inference
    logger.info("Model ready on device: %s", device)
    return model, tokenizer, device


# ─────────────────────────────────────────────────────────────────────────────
def _batch_inference(
    texts: list,
    model,
    tokenizer,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> tuple:
    """
    Run batched inference and return (predictions array, confidences array).

    Processes in batches to avoid GPU OOM. Returns numpy arrays.

    Args:
        texts      : List of raw text strings.
        model      : Loaded model in eval mode.
        tokenizer  : Loaded tokenizer.
        device     : Target device.
        batch_size : Samples per batch.
        max_length : Token truncation length.

    Returns:
        Tuple[np.ndarray, np.ndarray]: predictions, confidences
    """
    all_preds = []
    all_confs = []
    total     = len(texts)

    with torch.no_grad():   # REQUIRED — prevents silent memory leaks during inference
        for start in range(0, total, batch_size):
            batch = texts[start : start + batch_size]

            encoded = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            # NOTE: Do NOT pass token_type_ids — RoBERTa does not use them
            input_ids      = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs   = F.softmax(outputs.logits, dim=-1).cpu().numpy()

            all_preds.extend(np.argmax(probs, axis=-1).tolist())
            all_confs.extend(np.max(probs, axis=-1).tolist())

            processed = min(start + batch_size, total)
            if processed % (batch_size * 20) == 0 or processed == total:
                logger.info("  Progress: %d / %d  (%.1f%%)", processed, total, 100.0 * processed / total)

    return np.array(all_preds), np.array(all_confs)


# ─────────────────────────────────────────────────────────────────────────────
def run_predict(
    test_csv_path: Optional[str] = None,
    model_dir:     Optional[str] = None,
    output_path:   Optional[str] = None,
    threshold:     float = CONFIDENCE_THRESHOLD,
) -> pd.DataFrame:
    """
    Generate competition predictions from the test set.

    Automatically detects whether test.csv has labels:
        - If labels exist   : also prints accuracy + F1 as a sanity check
        - If no labels exist: skips all metric computation (correct competition behaviour)

    Args:
        test_csv_path : Path to processed test.csv.
                        Defaults to data/processed/test.csv
        model_dir     : Path to trained model directory.
                        Defaults to models/roberta_fakenews/
        output_path   : Where to save predictions CSV.
                        Defaults to outputs/predictions.csv
        threshold     : Confidence threshold for UNCERTAIN flag.

    Returns:
        pd.DataFrame: Predictions DataFrame saved to output_path.
    """
    set_seed(SEED)

    test_path  = Path(test_csv_path) if test_csv_path else Path(PROCESSED_DIR) / "test.csv"
    save_path  = Path(output_path)   if output_path   else Path(OUTPUTS_DIR)   / "predictions.csv"
    outputs    = save_path.parent
    create_directory(outputs)

    logger.info("=" * 60)
    logger.info("STARTING PREDICTION")
    logger.info("Test CSV   : %s", test_path)
    logger.info("Save path  : %s", save_path)
    logger.info("Threshold  : %.0f%% confidence", threshold * 100)
    logger.info("=" * 60)

    # ── 1. Validate test.csv exists ─────────────────────────────────────────
    if not test_path.exists():
        raise FileNotFoundError(
            f"test.csv not found at: {test_path}\n"
            "Fix: Run  python src/preprocess.py  first."
        )

    # ── 2. Load test data ────────────────────────────────────────────────
    test_df = pd.read_csv(test_path)
    logger.info("Loaded test.csv — %d rows | columns: %s", len(test_df), test_df.columns.tolist())

    if TEXT_COL not in test_df.columns:
        raise ValueError(
            f"Expected column '{TEXT_COL}' not found in test.csv.\n"
            f"Available columns: {test_df.columns.tolist()}\n"
            "Fix: Re-run  python src/preprocess.py"
        )

    # CRITICAL CHECK (instructions/08): test set may have NO label column in competition
    has_labels = LABEL_COL in test_df.columns
    if has_labels:
        logger.info("Label column detected — will compute accuracy as sanity check.")
    else:
        logger.info(
            "No label column in test.csv — running in submission mode (predictions only).\n"
            "This is expected for competition holdout test sets."
        )

    # ── 3. Load model ─────────────────────────────────────────────────────
    model, tokenizer, device = _load_model_and_tokenizer(model_dir)

    # ── 4. Run batched inference ───────────────────────────────────────────
    texts = test_df[TEXT_COL].fillna("").astype(str).tolist()
    logger.info("Running inference on %d samples...", len(texts))

    preds, confs = _batch_inference(
        texts     = texts,
        model     = model,
        tokenizer = tokenizer,
        device    = device,
        batch_size= BATCH_SIZE * 2,   # inference can use 2x train batch size safely
        max_length= MAX_LEN,
    )

    # ── 5. Build output DataFrame ───────────────────────────────────────────
    result_df = pd.DataFrame({
        "id":         range(len(preds)),
        TEXT_COL:     texts,
        "predicted":  preds,
        "label":      [ID2LABEL.get(int(p), str(p)) for p in preds],
        "confidence": [round(float(c), 6) for c in confs],
        "uncertain":  [bool(c < threshold) for c in confs],
    })

    # ── 6. Sanity-check metrics (only if labels exist) ─────────────────────
    # NEVER report these as final competition metrics — use val.csv metrics only.
    # This is only a sanity check on the WELFake 15% holdout test split.
    if has_labels:
        y_true = test_df[LABEL_COL].astype(int).tolist()
        acc    = accuracy_score(y_true, preds)
        f1     = f1_score(y_true, preds, average="weighted", zero_division=0)
        result_df["true_label"] = y_true
        logger.info("Sanity check (test split) — Accuracy: %.4f | F1: %.4f", acc, f1)
        print(f"\n  ⚠️  Sanity check on labeled test split:")
        print(f"     Accuracy : {acc:.4f}")
        print(f"     F1 (wt)  : {f1:.4f}")
        print(f"  (These are from your WELFake test split, NOT official competition metrics)\n")

    # ── 7. Save predictions CSV ────────────────────────────────────────────
    result_df.to_csv(save_path, index=False)
    logger.info("Predictions saved → %s (%d rows)", save_path, len(result_df))

    # ── 8. Print summary ───────────────────────────────────────────────────
    fake_count      = int((result_df["predicted"] == 0).sum())
    real_count      = int((result_df["predicted"] == 1).sum())
    uncertain_count = int(result_df["uncertain"].sum())
    avg_confidence  = float(result_df["confidence"].mean())

    print("\n" + "=" * 52)
    print("  PREDICTION COMPLETE — FakeGuard")
    print("=" * 52)
    print(f"  Total samples      : {len(result_df):>8,}")
    print(f"  Predicted FAKE (0) : {fake_count:>8,}  ({100.0*fake_count/len(result_df):.1f}%)")
    print(f"  Predicted REAL (1) : {real_count:>8,}  ({100.0*real_count/len(result_df):.1f}%)")
    print(f"  Uncertain (<{threshold:.0%}) : {uncertain_count:>8,}  ({100.0*uncertain_count/len(result_df):.1f}%)")
    print(f"  Avg confidence     : {avg_confidence:>8.4f}")
    print("-" * 52)
    print(f"  Saved to           : {save_path}")
    print("=" * 52 + "\n")

    logger.info("=" * 60)
    logger.info("PREDICTION COMPLETE — %d rows saved to %s", len(result_df), save_path)
    logger.info("=" * 60)

    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate FakeGuard competition predictions from test.csv"
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default=None,
        help="Path to processed test.csv (default: data/processed/test.csv)",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Path to trained model directory (default: models/roberta_fakenews/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for predictions CSV (default: outputs/predictions.csv)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold for UNCERTAIN flag (default: {CONFIDENCE_THRESHOLD})",
    )
    args = parser.parse_args()

    try:
        predictions_df = run_predict(
            test_csv_path = args.test_csv,
            model_dir     = args.model_dir,
            output_path   = args.output,
            threshold     = args.threshold,
        )
    except FileNotFoundError as e:
        logger.error("Missing file: %s", e)
        raise
    except ValueError as e:
        logger.error("Data error: %s", e)
        raise
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logger.error(
                "GPU OOM during inference! Reduce BATCH_SIZE in config.py and retry."
            )
        raise
