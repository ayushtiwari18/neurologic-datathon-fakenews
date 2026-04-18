"""
preprocess.py — Data preprocessing pipeline for FakeGuard.

Source of truth: instructions/06_data_preprocessing.md
                 instructions/03_pipeline_architecture.md

Responsibilities:
    1. Load raw train.csv and test.csv from data/raw/
       OR load a single WELFake_Dataset.csv and create train/val/test splits
    2. Validate required columns
    3. Handle missing values
    4. Clean text (HTML entities, URLs, special chars, lowercase)
    5. Create combined = title_clean + " [SEP] " + text_clean
       NOTE: [SEP] is PLAIN TEXT — NOT a special token.
             RoBERTa's real separator </s> is inserted automatically
             by the tokenizer. Do NOT change [SEP] to </s> manually.
    6. Encode labels: REAL=1, FAKE=0 (per instructions/06)
    7. Stratified split:
         - Official hackathon mode: 80/20 train/val, keep provided test.csv
         - Single-file WELFake mode: 70/15/15 train/val/test
    8. Save processed CSVs to data/processed/
    9. Log dataset statistics

Usage:
    python src/preprocess.py
    --- or ---
    from src.preprocess import run_preprocessing
    train_df, val_df, test_df = run_preprocessing()
"""

import re
import sys
import html
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    TRAIN_FILE,
    TEST_FILE,
    PROCESSED_DIR,
    OUTPUTS_DIR,
    TITLE_COL,
    BODY_COL,
    TEXT_COL,
    LABEL_COL,
    LABEL2ID,
    ID2LABEL,
    VAL_SPLIT,
    SEED,
)
from src.utils import get_logger, set_seed, create_directory, save_json

logger = get_logger("preprocess")


def clean_text(text: str) -> str:
    """
    Clean a single text string per instructions/06_data_preprocessing.md.

    Args:
        text: Raw input string.

    Returns:
        str: Cleaned lowercase string.
    """
    if not isinstance(text, str):
        return ""

    text = html.unescape(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z0-9\s.,!?']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def validate_columns(df: pd.DataFrame, required: list, name: str = "DataFrame") -> None:
    """
    Assert all required columns exist in DataFrame.

    Args:
        df: DataFrame to check.
        required: Required column names.
        name: Dataset label for logging.

    Raises:
        ValueError: If required columns are missing.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{name}] Missing columns: {missing}. Found: {df.columns.tolist()}"
        )
    logger.info("[%s] Column validation passed.", name)


def handle_missing(df: pd.DataFrame, name: str = "DataFrame") -> pd.DataFrame:
    """
    Handle null values per instructions/06.

    Rules:
        - Drop rows where BOTH title and text are null
        - Fill individual nulls with empty strings

    Args:
        df: Input DataFrame.
        name: Dataset label for logging.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    before = len(df)
    df = df.dropna(subset=[TITLE_COL, BODY_COL], how="all")
    df[TITLE_COL] = df[TITLE_COL].fillna("")
    df[BODY_COL] = df[BODY_COL].fillna("")
    logger.info("[%s] Rows after missing-value handling: %d (dropped %d)", name, len(df), before - len(df))
    return df.reset_index(drop=True)


def remove_duplicates(df: pd.DataFrame, name: str = "DataFrame") -> pd.DataFrame:
    """
    Remove duplicate rows based on title + text combination.

    Args:
        df: Input DataFrame.
        name: Dataset label for logging.

    Returns:
        pd.DataFrame: Deduplicated DataFrame.
    """
    before = len(df)
    df = df.drop_duplicates(subset=[TITLE_COL, BODY_COL])
    logger.info("[%s] Duplicates removed: %d", name, before - len(df))
    return df.reset_index(drop=True)


def encode_labels(df: pd.DataFrame, name: str = "DataFrame") -> pd.DataFrame:
    """
    Encode label column using config.LABEL2ID.

    Args:
        df: Input DataFrame with label column.
        name: Dataset label for logging.

    Returns:
        pd.DataFrame: Label-encoded DataFrame.

    Raises:
        AssertionError: If label mapping fails.
    """
    df[LABEL_COL] = df[LABEL_COL].map(LABEL2ID)
    null_labels = df[LABEL_COL].isnull().sum()
    assert null_labels == 0, f"[{name}] Label encoding failed for {null_labels} rows."
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    logger.info("[%s] Label distribution: %s", name, df[LABEL_COL].value_counts().to_dict())
    return df


def preprocess_dataframe(
    df: pd.DataFrame,
    is_train: bool = True,
    name: str = "DataFrame",
) -> pd.DataFrame:
    """
    Full preprocessing pipeline for one DataFrame.

    Args:
        df: Raw input DataFrame.
        is_train: Whether labels should be encoded.
        name: Dataset label for logging.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    required = [TITLE_COL, BODY_COL]
    if is_train:
        required.append(LABEL_COL)

    validate_columns(df, required, name)
    df = handle_missing(df, name)
    df = remove_duplicates(df, name)

    df["_title_clean"] = df[TITLE_COL].apply(clean_text)
    df["_body_clean"] = df[BODY_COL].apply(clean_text)
    df[TEXT_COL] = (df["_title_clean"] + " [SEP] " + df["_body_clean"]).apply(lambda x: x[:2000])
    df = df.drop(columns=["_title_clean", "_body_clean"])

    if is_train:
        df = encode_labels(df, name)

    logger.info("[%s] Preprocessed shape: %s", name, df.shape)
    return df


def stratified_split(
    df: pd.DataFrame,
    val_split: float = VAL_SPLIT,
    seed: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/validation split.

    Args:
        df: Labeled processed DataFrame.
        val_split: Validation ratio.
        seed: Random seed.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: train_df, val_df
    """
    train_df, val_df = train_test_split(
        df,
        test_size=val_split,
        stratify=df[LABEL_COL],
        random_state=seed,
        shuffle=True,
    )
    logger.info("Official split complete — train: %d | val: %d", len(train_df), len(val_df))
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def split_single_welfake(
    df: pd.DataFrame,
    seed: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split single WELFake file into stratified 70/15/15 train/val/test.

    Args:
        df: Preprocessed labeled DataFrame.
        seed: Random seed.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            train_df, val_df, test_df
    """
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df[LABEL_COL],
        random_state=seed,
        shuffle=True,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df[LABEL_COL],
        random_state=seed,
        shuffle=True,
    )
    logger.info(
        "WELFake split complete — train: %d | val: %d | test: %d",
        len(train_df), len(val_df), len(test_df),
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_split(df: pd.DataFrame, filepath: Path, name: str = "split") -> None:
    """
    Save DataFrame to CSV.

    Args:
        df: DataFrame to save.
        filepath: Output file path.
        name: Dataset label for logging.
    """
    create_directory(filepath.parent)
    df.to_csv(filepath, index=False)
    logger.info("Saved [%s] → %s (%d rows)", name, filepath, len(df))


def save_eda_plots(df: pd.DataFrame, outputs_dir: Path) -> None:
    """
    Save required EDA plots.

    Args:
        df: Labeled processed DataFrame.
        outputs_dir: Directory for output images.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        create_directory(outputs_dir)

        fig, ax = plt.subplots(figsize=(6, 4))
        df[LABEL_COL].map(ID2LABEL).value_counts().plot(kind="bar", ax=ax, color=["#2196F3", "#F44336"])
        ax.set_title("Class Distribution")
        ax.set_xlabel("Label")
        ax.set_ylabel("Count")
        plt.tight_layout()
        plt.savefig(outputs_dir / "class_distribution.png", dpi=150)
        plt.close()

        lengths = df[TEXT_COL].str.len()
        fig, ax = plt.subplots(figsize=(8, 4))
        lengths.hist(bins=50, ax=ax, color="#4CAF50", edgecolor="white")
        ax.set_title("Combined Text Length Distribution")
        ax.set_xlabel("Character Count")
        ax.set_ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(outputs_dir / "text_length_dist.png", dpi=150)
        plt.close()

        logger.info("Saved EDA plots to %s", outputs_dir)
    except ImportError:
        logger.warning("matplotlib not installed — skipping EDA plots.")


def _build_stats(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, mode: str) -> dict:
    """
    Build preprocessing statistics dictionary.

    Args:
        train_df: Train split.
        val_df: Validation split.
        test_df: Test split.
        mode: Processing mode label.

    Returns:
        dict: Summary statistics.
    """
    return {
        "mode": mode,
        "seed": int(SEED),
        "text_col": TEXT_COL,
        "label_encoding": {"REAL": 1, "FAKE": 0},
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_label_dist": {ID2LABEL.get(k, str(k)): int(v) for k, v in train_df[LABEL_COL].value_counts().items()},
        "val_label_dist": {ID2LABEL.get(k, str(k)): int(v) for k, v in val_df[LABEL_COL].value_counts().items()},
        "test_label_dist": {ID2LABEL.get(k, str(k)): int(v) for k, v in test_df[LABEL_COL].value_counts().items()} if LABEL_COL in test_df.columns else {},
        "avg_combined_len": round(float(train_df[TEXT_COL].str.len().mean()), 2),
        "max_combined_len": int(train_df[TEXT_COL].str.len().max()),
    }


def run_preprocessing(
    train_path: Optional[str] = None,
    test_path: Optional[str] = None,
    processed_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run preprocessing in one of two modes:

    Mode A — Official hackathon:
        train.csv + test.csv available in data/raw/
        Output: processed train.csv, val.csv, test.csv

    Mode B — Single-file WELFake:
        data/raw/WELFake_Dataset.csv only
        Output: processed train.csv, val.csv, test.csv via 70/15/15 split

    Args:
        train_path: Optional train CSV path override.
        test_path: Optional test CSV path override.
        processed_dir: Optional processed dir override.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            train_df, val_df, test_df
    """
    set_seed(SEED)

    out_dir = Path(processed_dir or PROCESSED_DIR)
    outputs_dir = Path(OUTPUTS_DIR)
    train_src = Path(train_path or TRAIN_FILE)
    test_src = Path(test_path or TEST_FILE)
    welfake_src = train_src.parent / "WELFake_Dataset.csv"

    logger.info("=" * 60)
    logger.info("STARTING PREPROCESSING")
    logger.info("=" * 60)

    if welfake_src.exists() and not train_src.exists() and not test_src.exists():
        logger.info("Detected single-file mode: %s", welfake_src)
        raw_df = pd.read_csv(welfake_src)
        logger.info("Raw WELFake shape: %s | columns: %s", raw_df.shape, raw_df.columns.tolist())

        processed_df = preprocess_dataframe(raw_df, is_train=True, name="WELFake")
        train_df, val_df, test_df = split_single_welfake(processed_df, seed=SEED)
        mode = "single_file_welfake"
    else:
        if not train_src.exists():
            raise FileNotFoundError(f"train.csv not found at: {train_src}")
        if not test_src.exists():
            raise FileNotFoundError(f"test.csv not found at: {test_src}")

        logger.info("Detected official mode: train.csv + test.csv")
        raw_train = pd.read_csv(train_src)
        raw_test = pd.read_csv(test_src)
        logger.info("Raw train shape: %s | columns: %s", raw_train.shape, raw_train.columns.tolist())
        logger.info("Raw test shape : %s | columns: %s", raw_test.shape, raw_test.columns.tolist())

        processed_train = preprocess_dataframe(raw_train, is_train=True, name="Train")
        has_test_labels = LABEL_COL in raw_test.columns
        test_df = preprocess_dataframe(raw_test, is_train=has_test_labels, name="Test")
        train_df, val_df = stratified_split(processed_train, val_split=VAL_SPLIT, seed=SEED)
        mode = "official_train_test"

    save_eda_plots(train_df, outputs_dir)
    save_split(train_df, out_dir / "train.csv", "train")
    save_split(val_df, out_dir / "val.csv", "val")
    save_split(test_df, out_dir / "test.csv", "test")

    stats = _build_stats(train_df, val_df, test_df, mode=mode)
    save_json(stats, out_dir / "preprocessing_stats.json")

    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("Mode  : %s", mode)
    logger.info("Train : %d rows", len(train_df))
    logger.info("Val   : %d rows", len(val_df))
    logger.info("Test  : %d rows", len(test_df))
    logger.info("=" * 60)

    return train_df, val_df, test_df


if __name__ == "__main__":
    try:
        run_preprocessing()
    except Exception as exc:
        logger.error("Preprocessing failed: %s", exc)
        raise
