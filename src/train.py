"""
train.py — RoBERTa fine-tuning pipeline for FakeGuard.

Source of truth:
    - instructions/04_model_training_strategy.md
    - instructions/03_pipeline_architecture.md
    - config.py
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


def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute accuracy and weighted F1.
    Called automatically by HuggingFace Trainer after each eval epoch.
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
    Full RoBERTa fine-tuning pipeline.
    Returns trained Trainer and metrics dict.
    """
    set_seed(SEED)

    # ── 1. Paths ──────────────────────────────────────────────────────────────
    train_csv  = Path(train_path) if train_path else Path(PROCESSED_DIR) / "train.csv"
    val_csv    = Path(val_path)   if val_path   else Path(PROCESSED_DIR) / "val.csv"
    model_save = Path(output_dir) / "roberta_fakenews" if output_dir else Path(MODEL_DIR) / "roberta_fakenews"
    outputs    = Path(OUTPUTS_DIR)

    create_directory(model_save)
    create_directory(outputs)

    logger.info("=" * 60)
    logger.info("STARTING ROBERTA FINE-TUNING")
    logger.info("Model     : %s", model_name)
    logger.info("Train CSV : %s", train_csv)
    logger.info("Val CSV   : %s", val_csv)
    logger.info("Device    : %s", "GPU (fp16)" if torch.cuda.is_available() else "CPU")
    logger.info("=" * 60)

    # ── 2. Validate files ─────────────────────────────────────────────────────
    if not train_csv.exists():
        raise FileNotFoundError(f"train.csv not found: {train_csv}")
    if not val_csv.exists():
        raise FileNotFoundError(f"val.csv not found: {val_csv}")

    # ── 3. Tokenizer ──────────────────────────────────────────────────────────
    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = load_tokenizer(model_name)

    # ── 4. Datasets ───────────────────────────────────────────────────────────
    logger.info("Building FakeNewsDataset objects...")
    train_dataset = FakeNewsDataset(data=train_csv, tokenizer=tokenizer, max_length=MAX_LEN, has_labels=True)
    val_dataset   = FakeNewsDataset(data=val_csv,   tokenizer=tokenizer, max_length=MAX_LEN, has_labels=True)
    logger.info("Train samples : %d", len(train_dataset))
    logger.info("Val samples   : %d", len(val_dataset))

    # ── 5. Model ──────────────────────────────────────────────────────────────
    logger.info("Loading pre-trained model: %s", model_name)
    model = RobertaForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        id2label={int(k): v for k, v in ID2LABEL.items()},
        label2id={k: int(v) for k, v in LABEL2ID.items() if isinstance(k, str)},
    )

    # ── 6. Data collator ──────────────────────────────────────────────────────
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ── 7. TrainingArguments ──────────────────────────────────────────────────
    # NOTE: 'evaluation_strategy' was renamed to 'eval_strategy' in transformers>=4.45
    # Kaggle's PyTorch 2.10 ships with newer transformers — use 'eval_strategy'.
    training_args = TrainingArguments(
        output_dir=str(model_save),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch",                         # ← renamed from evaluation_strategy
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        logging_steps=100,
        seed=SEED,
        dataloader_pin_memory=torch.cuda.is_available(),
    )

    # ── 8. Trainer ────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # ── 9. Train ──────────────────────────────────────────────────────────────
    logger.info("Training started — ~25-35 min on T4")
    train_result = trainer.train()
    logger.info("Training finished.")

    # ── 10. Final evaluation ──────────────────────────────────────────────────
    eval_metrics = trainer.evaluate()
    logger.info("Final val metrics: %s", eval_metrics)

    # ── 11. Save model + tokenizer ────────────────────────────────────────────
    logger.info("Saving model → %s", model_save)
    trainer.save_model(str(model_save))
    tokenizer.save_pretrained(str(model_save))

    # ── 12. Save metrics JSON ─────────────────────────────────────────────────
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
    logger.info("Metrics saved → %s", metrics_path)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE — Val Accuracy: %.4f", eval_metrics.get("eval_accuracy", 0))
    logger.info("=" * 60)

    return trainer, metrics_out


if __name__ == "__main__":
    try:
        trainer, metrics = run_training()
        print(f"\nVal Accuracy : {metrics['final_val_accuracy']:.4f}")
        print(f"Val F1       : {metrics['final_val_f1']:.4f}")
        print("Model saved  : models/roberta_fakenews/")
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logger.error("OOM — reduce BATCH_SIZE to 8 in config.py and retry.")
        raise
