"""
preprocess.py — Data preprocessing pipeline for FakeGuard.

Responsibilities:
    1. Load raw train.csv and test.csv
    2. Validate required columns
    3. Handle missing values
    4. Clean text (HTML, URLs, special chars, lowercase)
    5. Create combined_text = title + " [SEP] " + text
    6. Encode labels using LABEL2ID from config
    7. Stratified train / validation split
    8. Save processed CSVs to data/processed/
    9. Log dataset statistics at every step

Design rules:
    - No hardcoded paths — all paths from config.py
    - Compatible with Colab, Kaggle, and local
    - Deterministic (seed=42)
    - All outputs saved to PROCESSED_DIR from config

Usage:
    python src/preprocess.py
    — or —
    from src.preprocess import run_preprocessing
    run_preprocessing()
"""

import re
import sys
import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Path bootstrap — works in Colab, Kaggle, and local without modification
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    TRAIN_FILE,
    TEST_FILE,
    PROCESSED_DIR,
    TITLE_COL,
    BODY_COL,
    TEXT_COL,
    LABEL_COL,
    LABEL2ID,
    VAL_SPLIT,
    SEED,
)
from src.utils import get_logger, set_seed, create_directory, save_json

logger = get_logger("preprocess")


# ---------------------------------------------------------------------------
# HTML / text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Clean a single text string.

    Steps:
        1. Strip HTML tags using regex (no BS4 dependency at import time)
        2. Remove URLs (http / https / www)
        3. Remove email addresses
        4. Remove non-alphanumeric characters (keep spaces)
        5. Collapse multiple whitespace
        6. Lowercase

    Args:
        text: Raw input string.

    Returns:
        str: Cleaned lowercase string.
    """
    if not isinstance(text, str):
        return ""

    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove URLs
    text = re.sub(r"http\S+|https\S+|www\.\S+", " ", text)
    # Remove emails
    text = re.sub(r"\S+@\S+", " ", text)
    # Remove special characters — keep alphanumeric + spaces
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


# ---------------------------------------------------------------------------
# Column validation
# ---------------------------------------------------------------------------

def validate_columns(
    df: pd.DataFrame,
    required: list,
    name: str = "DataFrame",
) -> None:
    """
    Assert that all required columns exist in a DataFrame.

    Args:
        df       : DataFrame to validate.
        required : List of required column names.
        name     : Human-readable name for logging.

    Raises:
        ValueError: If any required column is missing.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{name}] Missing columns: {missing}. "
            f"Found: {df.columns.tolist()}"
        )
    logger.info("[%s] Column validation passed: %s", name, required)


# ---------------------------------------------------------------------------
# Missing value handling
# ---------------------------------------------------------------------------

def handle_missing(
    df: pd.DataFrame,
    name: str = "DataFrame",
) -> pd.DataFrame:
    """
    Handle null values:
        - Fill null TITLE_COL with empty string
        - Drop rows where BODY_COL is null (no text = unusable)
        - Log before / after counts

    Args:
        df   : Input DataFrame.
        name : Human-readable name for logging.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    before = len(df)
    null_title = df[TITLE_COL].isnull().sum() if TITLE_COL in df.columns else 0
    null_body  = df[BODY_COL].isnull().sum()  if BODY_COL  in df.columns else 0

    logger.info("[%s] Nulls before — title: %d | body: %d", name, null_title, null_body)

    if TITLE_COL in df.columns:
        df[TITLE_COL] = df[TITLE_COL].fillna("")

    if BODY_COL in df.columns:
        df = df.dropna(subset=[BODY_COL])

    after = len(df)
    logger.info("[%s] Rows after null handling: %d (dropped %d)", name, after, before - after)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Core preprocessing
# ---------------------------------------------------------------------------

def preprocess_dataframe(
    df: pd.DataFrame,
    is_train: bool = True,
    name: str = "DataFrame",
) -> pd.DataFrame:
    """
    Apply full preprocessing to a DataFrame.

    Steps:
        1. Validate columns
        2. Handle missing values
        3. Clean title and body text
        4. Create combined_text = clean_title + " [SEP] " + clean_body
        5. Encode labels (train only)

    Args:
        df       : Raw input DataFrame.
        is_train : If True, encode labels. If False (test), skip label encoding.
        name     : Human-readable name for logging.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with TEXT_COL (and LABEL_COL if train).
    """
    required_cols = [TITLE_COL, BODY_COL]
    if is_train:
        required_cols.append(LABEL_COL)

    validate_columns(df, required_cols, name)
    df = handle_missing(df, name)

    logger.info("[%s] Cleaning text columns...", name)
    df["_clean_title"] = df[TITLE_COL].apply(clean_text)
    df["_clean_body"]  = df[BODY_COL].apply(clean_text)

    # Fuse title + body with plain-text [SEP] marker
    # NOTE: [SEP] here is plain text, NOT a special token.
    # RoBERTa's real separator </s> is inserted automatically by the tokenizer.
    df[TEXT_COL] = df["_clean_title"] + " [SEP] " + df["_clean_body"]

    if is_train:
        # Map string labels to integers via LABEL2ID from config
        # If labels are already integers, keep them
        if df[LABEL_COL].dtype == object:
            df[LABEL_COL] = df[LABEL_COL].str.upper().map(LABEL2ID)
            null_labels = df[LABEL_COL].isnull().sum()
            if null_labels > 0:
                raise ValueError(
                    f"[{name}] {null_labels} labels could not be mapped. "
                    f"Check LABEL2ID in config.py. Found values: {df[LABEL_COL].unique()}"
                )
        df[LABEL_COL] = df[LABEL_COL].astype(int)

    # Drop intermediate columns
    df = df.drop(columns=["_clean_title", "_clean_body"])

    logger.info(
        "[%s] Preprocessing complete. Shape: %s | combined_text sample: '%s'",
        name,
        df.shape,
        df[TEXT_COL].iloc[0][:80] if len(df) > 0 else "EMPTY",
    )
    return df


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def stratified_split(
    df: pd.DataFrame,
    val_split: float = VAL_SPLIT,
    seed: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform a stratified train / validation split.

    Args:
        df        : Preprocessed training DataFrame (must have LABEL_COL).
        val_split : Fraction for validation set. Default from config.
        seed      : Random seed. Default from config.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, val_df)
    """
    train_df, val_df = train_test_split(
        df,
        test_size=val_split,
        stratify=df[LABEL_COL],
        random_state=seed,
        shuffle=True,
    )

    logger.info(
        "Split — Train: %d rows | Val: %d rows | Val fraction: %.2f",
        len(train_df), len(val_df), val_split,
    )

    for split_name, split_df in [("Train", train_df), ("Val", val_df)]:
        dist = split_df[LABEL_COL].value_counts(normalize=True).to_dict()
        logger.info(
            "[%s] Label distribution: %s",
            split_name,
            {k: f"{v:.2%}" for k, v in dist.items()},
        )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_split(
    df: pd.DataFrame,
    filepath: Path,
    name: str = "split",
) -> None:
    """
    Save a DataFrame to CSV.

    Args:
        df       : DataFrame to save.
        filepath : Target .csv path.
        name     : Human-readable name for logging.

    Returns:
        None
    """
    create_directory(filepath.parent)
    df.to_csv(filepath, index=False)
    logger.info("Saved [%s] → %s  (%d rows)", name, filepath, len(df))


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def run_preprocessing(
    train_path: Optional[str] = None,
    test_path: Optional[str] = None,
    processed_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full preprocessing pipeline: load → clean → split → save.

    Args:
        train_path    : Path to raw train CSV. Defaults to config.TRAIN_FILE.
        test_path     : Path to raw test CSV. Defaults to config.TEST_FILE.
        processed_dir : Output directory. Defaults to config.PROCESSED_DIR.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            (train_df, val_df, test_df) — all preprocessed.

    Raises:
        FileNotFoundError: If train or test CSV is not found.
    """
    set_seed(SEED)

    train_src = Path(train_path or TRAIN_FILE)
    test_src  = Path(test_path  or TEST_FILE)
    out_dir   = Path(processed_dir or PROCESSED_DIR)

    # --- Validate source files ---
    if not train_src.exists():
        raise FileNotFoundError(
            f"train.csv not found at: {train_src}\n"
            f"Place your dataset in the data/ directory before running preprocessing."
        )
    if not test_src.exists():
        raise FileNotFoundError(
            f"test.csv not found at: {test_src}\n"
            f"Place your dataset in the data/ directory before running preprocessing."
        )

    # --- Load ---
    logger.info("Loading train data from: %s", train_src)
    raw_train = pd.read_csv(train_src)
    logger.info("Raw train shape: %s", raw_train.shape)
    logger.info("Train columns: %s", raw_train.columns.tolist())

    logger.info("Loading test data from: %s", test_src)
    raw_test = pd.read_csv(test_src)
    logger.info("Raw test shape: %s", raw_test.shape)
    logger.info("Test columns: %s", raw_test.columns.tolist())

    # --- Preprocess ---
    logger.info("=" * 60)
    logger.info("PREPROCESSING TRAIN SET")
    logger.info("=" * 60)
    processed_train = preprocess_dataframe(raw_train, is_train=True, name="Train")

    logger.info("=" * 60)
    logger.info("PREPROCESSING TEST SET")
    logger.info("=" * 60)
    processed_test = preprocess_dataframe(raw_test, is_train=False, name="Test")

    # --- Split ---
    logger.info("=" * 60)
    logger.info("STRATIFIED SPLIT")
    logger.info("=" * 60)
    train_df, val_df = stratified_split(processed_train, val_split=VAL_SPLIT, seed=SEED)

    # --- Save ---
    logger.info("=" * 60)
    logger.info("SAVING PROCESSED FILES")
    logger.info("=" * 60)
    save_split(train_df, out_dir / "train_clean.csv", "train")
    save_split(val_df,   out_dir / "val_clean.csv",   "val")
    save_split(processed_test, out_dir / "test_clean.csv", "test")

    # --- Dataset statistics report ---
    stats = {
        "raw_train_rows":       int(len(raw_train)),
        "raw_test_rows":        int(len(raw_test)),
        "processed_train_rows": int(len(train_df)),
        "processed_val_rows":   int(len(val_df)),
        "processed_test_rows":  int(len(processed_test)),
        "val_split":            float(VAL_SPLIT),
        "seed":                 int(SEED),
        "train_label_dist":     train_df[LABEL_COL].value_counts().to_dict(),
        "val_label_dist":       val_df[LABEL_COL].value_counts().to_dict(),
        "combined_text_avg_len": round(
            float(train_df[TEXT_COL].str.len().mean()), 2
        ),
    }
    save_json(stats, out_dir / "preprocessing_stats.json")

    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("  Train rows : %d", len(train_df))
    logger.info("  Val rows   : %d", len(val_df))
    logger.info("  Test rows  : %d", len(processed_test))
    logger.info("  Output dir : %s", out_dir)
    logger.info("=" * 60)

    return train_df, val_df, processed_test


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        run_preprocessing()
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(1)
    except ValueError as e:
        logger.error("Preprocessing failed: %s", e)
        sys.exit(1)
