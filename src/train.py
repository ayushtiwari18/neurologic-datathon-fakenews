"""
train.py — RoBERTa fine-tuning pipeline for FakeGuard.

Source of truth:
    - instructions/04_model_training_strategy.md
    - instructions/03_pipeline_architecture.md
    - config.py

Responsibilities:
    1. Load processed train.csv and val.csv from data/processed/
    2. Build FakeNewsDataset instances
    3. Load RobertaForSequenceClassification (roberta-base, num_labels=2)
    4. Configure HuggingFace Trainer with exact settings from instructions/04
    5. Train for 3 epochs with fp16, AdamW, linear warmup
    6. Evaluate on val set every epoch — report accuracy + F1
    7. Save best model checkpoint to models/roberta_fakenews/
    8. Save training metrics to outputs/training_metrics.json

IMPORTANT RoBERTa rules (from instructions/03 and 04):
    - NEVER pass token_type_ids — RoBERTa does not use them
    - [SEP] in combined text is PLAIN TEXT — tokenizer handles real separator
    - fp16=True is REQUIRED for speed on GPU
    - eval_dataset MUST be set or evaluation never runs
    - DataCollatorWithPadding is REQUIRED to prevent padding crashes
    - compute_metrics is REQUIRED — without it Trainer shows NO accuracy

Usage:
    python src/train.py
    --- or ---
    from src.train import run_training
    trainer, metrics = run_training()
"""

import sys
import json
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import torch
from transformers import (
    AutoTokenizer,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
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
    EPOCHS,
    LEARNING_RATE,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    SEED,
    NUM_LABELS,
    ID2LABEL,
    LABEL2ID,
)
from src.dataset import FakeNewsDataset, load_tokenizer
from src.utils import get_logger, set_seed, create_directory

logger = get_logger("train")


# ─────────────────────────────────────────────────────────────────────────────
# compute_metrics — REQUIRED by Trainer
# Without this, Trainer prints NO accuracy or F1 during evaluation
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute accuracy and weighted F1 from Trainer evaluation predictions.

    Called automatically by HuggingFace Trainer after each eval epoch.

    Args:
        eval_pred: EvalPrediction namedtuple with fields (logits, label_ids).

    Returns:
        dict: {"accuracy": float, "f1": float}
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1  = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}


def run_training(
    train_path: str = None,
    val_path:   str = None,
    model_name: str = TRANSFORMER,
    output_dir: str = None,
) -> Tuple[Trainer, Dict]:
    """
    Full RoBERTa fine-tuning pipeline for fake news classification.

    Args:
        train_path : Path to processed train.csv.
                     Defaults to config.PROCESSED_DIR/train.csv
        val_path   : Path to processed val.csv.
                     Defaults to config.PROCESSED_DIR/val.csv
        model_name : HuggingFace model identifier.
                     Defaults to config.TRANSFORMER (roberta-base)
        output_dir : Directory to save trained model.
                     Defaults to config.MODEL_DIR/roberta_fakenews

    Returns:
        Tuple[Trainer, dict]:
            - Trained HuggingFace Trainer instance
            - Final evaluation metrics dictionary
    """
    set_seed(SEED)

    # ── 1. Resolve all file paths ─────────────────────────────────────────────
    if train_path:
        train_csv = Path(train_path)
    else:
        train_csv = Path(PROCESSED_DIR) / "train.csv"

    if val_path:
        val_csv = Path(val_path)
    else:
        val_csv = Path(PROCESSED_DIR) / "val.csv"

    model_save = Path(output_dir) / "roberta_fakenews" if output_dir else Path(MODEL_DIR) / "roberta_fakenews"
    outputs    = Path(OUTPUTS_DIR)

    create_directory(model_save)
    create_directory(outputs)

    logger.info("=" * 60)
    logger.info("STARTING ROBERTA FINE-TUNING")
    logger.info("Model     : %s", model_name)
    logger.info("Train CSV : %s", train_csv)
    logger.info("Val CSV   : %s", val_csv)
    logger.info("Save Dir  : %s", model_save)
    logger.info("Device    : %s", "GPU (fp16)" if torch.cuda.is_available() else "CPU")
    logger.info("=" * 60)

    # ── 2. Validate input files exist ─────────────────────────────────────────
    if not train_csv.exists():
        raise FileNotFoundError(
            f"train.csv not found at: {train_csv}\n"
            "Fix: Run  python src/preprocess.py  first."
        )
    if not val_csv.exists():
        raise FileNotFoundError(
            f"val.csv not found at: {val_csv}\n"
            "Fix: Run  python src/preprocess.py  first."
        )

    # ── 3. Load tokenizer ─────────────────────────────────────────────────────
    # The tokenizer converts raw text into token IDs that RoBERTa understands.
    # roberta-base vocabulary has 50,265 tokens.
    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = load_tokenizer(model_name)

    # ── 4. Build PyTorch datasets ─────────────────────────────────────────────
    # FakeNewsDataset wraps CSV → tokenizes on-the-fly → returns tensors.
    # NOTE: token_type_ids are intentionally NOT returned (RoBERTa doesn't use them).
    logger.info("Building FakeNewsDataset objects...")
    train_dataset = FakeNewsDataset(
        data=train_csv,
        tokenizer=tokenizer,
        max_length=MAX_LEN,
        has_labels=True,
    )
    val_dataset = FakeNewsDataset(
        data=val_csv,
        tokenizer=tokenizer,
        max_length=MAX_LEN,
        has_labels=True,
    )
    logger.info("Train samples : %d", len(train_dataset))
    logger.info("Val samples   : %d", len(val_dataset))

    # ── 5. Load pre-trained model ─────────────────────────────────────────────
    # RobertaForSequenceClassification = roberta-base (125M params)
    # + linear classification head (768 → num_labels=2)
    # id2label / label2id allow the model to output "REAL" / "FAKE" strings.
    logger.info("Loading pre-trained model: %s", model_name)
    model = RobertaForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        id2label={int(k): v for k, v in ID2LABEL.items()},
        label2id={k: int(v) for k, v in LABEL2ID.items() if isinstance(k, str)},
    )

    # ── 6. DataCollatorWithPadding — REQUIRED ─────────────────────────────────
    # Pads each batch dynamically to the longest sequence in that batch.
    # Without this, fixed-length padding wastes memory and can crash on mismatched shapes.
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ── 7. TrainingArguments — exact settings from instructions/04 ────────────
    # fp16 is auto-enabled only when a GPU is detected (safe on CPU too).
    training_args = TrainingArguments(
        output_dir=str(model_save),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,    # eval can safely use 2x batch size
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        learning_rate=LEARNING_RATE,
        evaluation_strategy="epoch",                   # evaluate after each epoch
        save_strategy="epoch",                         # save checkpoint after each epoch
        load_best_model_at_end=True,                   # auto-restore best accuracy checkpoint
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),                # mixed precision — 2x speed on GPU
        report_to="none",                              # disable wandb / tensorboard
        logging_steps=100,
        seed=SEED,
        dataloader_pin_memory=torch.cuda.is_available(),
    )

    # ── 8. Build Trainer ──────────────────────────────────────────────────────
    # eval_dataset MUST be passed — without it, Trainer never evaluates.
    # compute_metrics MUST be passed — without it, Trainer shows no accuracy.
    # data_collator MUST be passed — without it, batch padding crashes.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,           # ← REQUIRED
        tokenizer=tokenizer,
        data_collator=data_collator,        # ← REQUIRED
        compute_metrics=compute_metrics,    # ← REQUIRED
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2)  # stop if val accuracy doesn't improve
        ],
    )

    # ── 9. Train ──────────────────────────────────────────────────────────────
    logger.info("Training started — expected duration: 25-40 min on Colab T4 / 20-30 min on Kaggle P100")
    train_result = trainer.train()
    logger.info("Training finished.")

    # ── 10. Final evaluation on validation set ────────────────────────────────
    logger.info("Running final evaluation on validation set...")
    eval_metrics = trainer.evaluate()
    logger.info("Final val metrics: %s", eval_metrics)

    # ── 11. Save model and tokenizer ──────────────────────────────────────────
    # Saves pytorch_model.bin + config.json + tokenizer files.
    # NOTE: Do NOT commit models/ to GitHub — the files are 400MB+.
    # Upload to HuggingFace Hub instead for sharing.
    logger.info("Saving model and tokenizer → %s", model_save)
    trainer.save_model(str(model_save))
    tokenizer.save_pretrained(str(model_save))

    # ── 12. Save metrics to outputs/training_metrics.json ────────────────────
    metrics_out = {
        "model": model_name,
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_length": MAX_LEN,
        "warmup_steps": WARMUP_STEPS,
        "weight_decay": WEIGHT_DECAY,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "train_runtime_seconds": round(train_result.metrics.get("train_runtime", 0), 2),
        "train_samples_per_second": round(train_result.metrics.get("train_samples_per_second", 0), 2),
        "final_val_loss": round(eval_metrics.get("eval_loss", 0), 6),
        "final_val_accuracy": round(eval_metrics.get("eval_accuracy", 0), 6),
        "final_val_f1": round(eval_metrics.get("eval_f1", 0), 6),
    }
    metrics_path = outputs / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    logger.info("Training metrics saved → %s", metrics_path)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("Val Accuracy : %.4f", eval_metrics.get("eval_accuracy", 0))
    logger.info("Val F1       : %.4f", eval_metrics.get("eval_f1", 0))
    logger.info("Model saved  : %s", model_save)
    logger.info("=" * 60)

    return trainer, metrics_out


if __name__ == "__main__":
    try:
        trainer, metrics = run_training()
        print("\n" + "=" * 50)
        print("✅  TRAINING COMPLETE")
        print(f"   Val Accuracy : {metrics['final_val_accuracy']:.4f}")
        print(f"   Val F1       : {metrics['final_val_f1']:.4f}")
        print(f"   Model saved  : models/roberta_fakenews/")
        print(f"   Metrics      : outputs/training_metrics.json")
        print("=" * 50)
    except FileNotFoundError as e:
        logger.error("Missing data file — %s", e)
        raise
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logger.error(
                "GPU out of memory! Open config.py and reduce BATCH_SIZE from %d to 8, then retry.",
                BATCH_SIZE,
            )
        raise
