"""
baseline.py — TF-IDF + Logistic Regression baseline for FakeGuard.

Source of truth:
    - instructions/04_model_training_strategy.md
    - instructions/07_development_steps.md
    - instructions/09_features_and_routes.md
    - instructions/12_points_to_remember.md

Responsibilities:
    1. Load processed train/val CSVs
    2. Fit TF-IDF on train only (NO data leakage)
    3. Train Logistic Regression baseline
    4. Evaluate on validation set only
    5. Save metrics JSON and confusion matrix PNG

Usage:
    python src/baseline.py
"""

import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import LABEL_COL, OUTPUTS_DIR, PROCESSED_DIR, SEED, TEXT_COL
from src.utils import compute_metrics, create_directory, get_logger, save_json, set_seed

logger = get_logger("baseline")


class BaselineTrainer:
    """
    TF-IDF + Logistic Regression baseline trainer.

    This class encapsulates baseline training and evaluation so that logic is
    reusable and testable.
    """

    def __init__(
        self,
        processed_dir: Path,
        outputs_dir: Path,
        text_col: str = TEXT_COL,
        label_col: str = LABEL_COL,
        seed: int = SEED,
    ) -> None:
        """
        Initialize baseline trainer.

        Args:
            processed_dir: Directory containing processed train/val CSVs.
            outputs_dir: Directory to save metrics and plots.
            text_col: Combined text column name.
            label_col: Encoded label column name.
            seed: Random seed for reproducibility.
        """
        self.processed_dir = Path(processed_dir)
        self.outputs_dir = create_directory(outputs_dir)
        self.text_col = text_col
        self.label_col = label_col
        self.seed = seed
        self.vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
        self.model = LogisticRegression(max_iter=1000, random_state=self.seed)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load processed train and validation CSVs.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (train_df, val_df)

        Raises:
            FileNotFoundError: If expected files are missing.
        """
        train_path = self.processed_dir / "train.csv"
        val_path = self.processed_dir / "val.csv"

        if not train_path.exists():
            raise FileNotFoundError(f"Missing processed file: {train_path}")
        if not val_path.exists():
            raise FileNotFoundError(f"Missing processed file: {val_path}")

        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        logger.info("Loaded train: %s | val: %s", train_df.shape, val_df.shape)
        return train_df, val_df

    def fit_vectorizer(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """
        Fit TF-IDF on training data only and transform train/val.

        Args:
            train_df: Processed training DataFrame.
            val_df: Processed validation DataFrame.

        Returns:
            Tuple of sparse matrices: (X_train, X_val)
        """
        X_train = self.vectorizer.fit_transform(train_df[self.text_col])
        X_val = self.vectorizer.transform(val_df[self.text_col])
        logger.info("TF-IDF complete — train shape: %s | val shape: %s", X_train.shape, X_val.shape)
        return X_train, X_val

    def train(self, X_train, y_train) -> None:
        """
        Train Logistic Regression baseline.

        Args:
            X_train: Sparse TF-IDF matrix for training.
            y_train: Ground-truth training labels.
        """
        self.model.fit(X_train, y_train)
        logger.info("Baseline model trained successfully.")

    def evaluate(self, X_val, y_val) -> Dict[str, float]:
        """
        Evaluate model on validation data.

        Args:
            X_val: Sparse TF-IDF matrix for validation.
            y_val: Ground-truth validation labels.

        Returns:
            Dict[str, float]: Computed evaluation metrics.
        """
        preds = self.model.predict(X_val)
        metrics = compute_metrics(labels=list(y_val), predictions=list(preds))
        self._save_confusion_matrix(y_true=list(y_val), y_pred=list(preds))
        return metrics

    def _save_confusion_matrix(self, y_true, y_pred) -> None:
        """
        Save confusion matrix PNG.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
        """
        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["REAL", "FAKE"],
            yticklabels=["REAL", "FAKE"],
        )
        plt.title("Baseline Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        filepath = self.outputs_dir / "baseline_confusion_matrix.png"
        plt.savefig(filepath, dpi=150)
        plt.close()
        logger.info("Saved confusion matrix → %s", filepath)


def run_baseline(
    processed_dir: Path = Path(PROCESSED_DIR),
    outputs_dir: Path = Path(OUTPUTS_DIR),
) -> Dict[str, float]:
    """
    End-to-end baseline run.

    Args:
        processed_dir: Directory containing processed CSVs.
        outputs_dir: Directory for plots and metrics.

    Returns:
        Dict[str, float]: Baseline evaluation metrics.
    """
    set_seed(SEED)
    trainer = BaselineTrainer(processed_dir=processed_dir, outputs_dir=outputs_dir)
    train_df, val_df = trainer.load_data()

    X_train, X_val = trainer.fit_vectorizer(train_df=train_df, val_df=val_df)
    y_train = train_df[LABEL_COL]
    y_val = val_df[LABEL_COL]

    trainer.train(X_train=X_train, y_train=y_train)
    metrics = trainer.evaluate(X_val=X_val, y_val=y_val)

    save_json(metrics, Path(outputs_dir) / "baseline_metrics.json")
    logger.info("Baseline complete — metrics saved.")
    return metrics


if __name__ == "__main__":
    try:
        run_baseline()
    except Exception as exc:
        logger.error("Baseline run failed: %s", exc)
        raise
