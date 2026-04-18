"""
preprocess.py — Data preprocessing pipeline for FakeGuard.

Source of truth: instructions/06_data_preprocessing.md
                 instructions/03_pipeline_architecture.md

Responsibilities:
    1. Load raw train.csv and test.csv from data/raw/
    2. Validate required columns
    3. Handle missing values
    4. Clean text (HTML entities, URLs, special chars, lowercase)
    5. Create combined = title_clean + " [SEP] " + text_clean
       NOTE: [SEP] is PLAIN TEXT — NOT a special token.
             RoBERTa's real separator </s> is inserted automatically
             by the tokenizer. Do NOT change [SEP] to </s> manually.
    6. Encode labels: REAL=1, FAKE=0 (per instructions/06)
    7. Stratified 80/20 train/val split
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
    TEXT_COL,      # = "combined" per instructions
    LABEL_COL,
    LABEL2ID,
    ID2LABEL,
    VAL_SPLIT,     # = 0.20 per instructions
    SEED,
)
from src.utils import get_logger, set_seed, create_directory, save_json

logger = get_logger("preprocess")


# ---------------------------------------------------------------------------
# Text cleaning — matches instructions/06_data_preprocessing.md exactly
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Clean a single text string per instructions/06_data_preprocessing.md.

    Steps (exact order from instructions):
        1. html.unescape — decode HTML entities (&amp; → &, etc.)
        2. Lowercase
        3. Remove URLs (http / https / www)
        4. Remove HTML tags
        5. Keep only letters, numbers, spaces, basic punctuation
        6. Collapse multiple whitespace

    Args:
        text: Raw input string.

    Returns:
        str: Cleaned lowercase string.
    """
    if not isinstance(text, str):
        return ""

    text = html.unescape(text)                           # decode &amp; &lt; etc.
    text = text.lower()                                  # lowercase first
    text = re.sub(r"http\S+|www\S+", "", text)          # remove URLs
    text = re.sub(r"<.*?>", "", text)                    # remove HTML tags
    text = re.sub(r"[^a-z0-9\s.,!?']", " ", text)       # keep alphanum + basic punct
    text = re.sub(r"\s+", " ", text).strip()             # normalize whitespace
    return text


# ---------------------------------------------------------------------------
# Column validation
# ---------------------------------------------------------------------------

def validate_columns(
    df: pd.DataFrame,
    required: list,
    name: str = "DataFrame",
) -> None:
    """
    Assert all required columns exist in DataFrame.

    Args:
        df       : DataFrame to check.
        required : List of required column names.
        name     : Label for logging.

    Raises:
        ValueError: If any required column is missing.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{name}] Missing columns: {missing}. "
            f"Found columns: {df.columns.tolist()}"
        )
    logger.info("[%s] Column validation passed. Columns: %s", name, df.columns.tolist())


# ---------------------------------------------------------------------------
# Missing value handling — per instructions/06
# ---------------------------------------------------------------------------

def handle_missing(
    df: pd.DataFrame,
    name: str = "DataFrame",
) -> pd.DataFrame:
    """
    Handle null values per instructions/06_data_preprocessing.md.

    Rules:
        - Fill null title with empty string
        - Fill null text with empty string
        - Drop rows where BOTH title AND text are null

    Args:
        df   : Input DataFrame.
        name : Label for logging.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    before = len(df)

    null_title = df[TITLE_COL].isnull().sum() if TITLE_COL in df.columns else 0
    null_body  = df[BODY_COL].isnull().sum()  if BODY_COL  in df.columns else 0
    logger.info("[%s] Nulls — title: %d | body: %d", name, null_title, null_body)

    # Drop rows where BOTH title AND text are null (per instructions/06 Step 2)
    if TITLE_COL in df.columns and BODY_COL in df.columns:
        df = df.dropna(subset=[TITLE_COL, BODY_COL], how="all")

    # Fill individual nulls with empty string
    if TITLE_COL in df.columns:
        df[TITLE_COL] = df[TITLE_COL].fillna("")
    if BODY_COL in df.columns:
        df[BODY_COL] = df[BODY_COL].fillna("")

    after = len(df)
    logger.info("[%s] Rows after null handling: %d (dropped %d)", name, after, before - after)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Duplicate removal
# ---------------------------------------------------------------------------

def remove_duplicates(
    df: pd.DataFrame,
    name: str = "DataFrame",
) -> pd.DataFrame:
    """
    Remove duplicate rows based on body text column.
    Per instructions/06: df.drop_duplicates('combined').

    Args:
        df   : Input DataFrame.
        name : Label for logging.

    Returns:
        pd.DataFrame: De-duplicated DataFrame.
    """
    before = len(df)
    if BODY_COL in df.columns:
        df = df.drop_duplicates(subset=[BODY_COL])
    after = len(df)
    logger.info("[%s] Duplicates removed: %d (remaining: %d)", name, before - after, after)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Label encoding — REAL=1, FAKE=0 per instructions/06
# ---------------------------------------------------------------------------

def encode_labels(
    df: pd.DataFrame,
    name: str = "DataFrame",
) -> pd.DataFrame:
    """
    Encode label column to integers using LABEL2ID from config.

    Per instructions/06_data_preprocessing.md:
        REAL = 1,  FAKE = 0

    Handles: string labels ('REAL','FAKE'), int labels (1,0),
             float labels (1.0, 0.0) — pandas may load ints as floats.

    Args:
        df   : DataFrame with LABEL_COL.
        name : Label for logging.

    Returns:
        pd.DataFrame: DataFrame with integer LABEL_COL.

    Raises:
        ValueError: If any label cannot be mapped.
        AssertionError: If null labels remain after mapping.
    """
    # Map using LABEL2ID (handles str, int, float variants)
    df[LABEL_COL] = df[LABEL_COL].map(LABEL2ID)

    null_labels = df[LABEL_COL].isnull().sum()
    assert null_labels == 0, (
        f"[{name}] Label encoding failed — {null_labels} values could not be mapped. "
        f"Check LABEL2ID in config.py."
    )

    df[LABEL_COL] = df[LABEL_COL].astype(int)
    dist = df[LABEL_COL].value_counts().to_dict()
    readable = {ID2LABEL.get(k, k): v for k, v in dist.items()}
    logger.info("[%s] Label distribution: %s", name, readable)
    return df


# ---------------------------------------------------------------------------
# Core preprocessing
# ---------------------------------------------------------------------------

def preprocess_dataframe(
    df: pd.DataFrame,
    is_train: bool = True,
    name: str = "DataFrame",
) -> pd.DataFrame:
    """
    Full preprocessing pipeline for one DataFrame.

    Steps per instructions/06_data_preprocessing.md:
        1. Validate columns
        2. Handle missing values (drop both-null, fill individual)
        3. Remove duplicates
        4. Clean title and body text (html.unescape + regex)
        5. Combine: combined = title_clean + " [SEP] " + text_clean
        6. Trim combined to 2000 chars (approx 512 tokens upper bound)
        7. Encode labels (train only)

    Args:
        df       : Raw input DataFrame.
        is_train : If True, encode labels. If False (test set), skip.
        name     : Label for logging.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    required = [TITLE_COL, BODY_COL]
    if is_train:
        required.append(LABEL_COL)

    validate_columns(df, required, name)
    df = handle_missing(df, name)
    df = remove_duplicates(df, name)

    logger.info("[%s] Cleaning text...", name)
    df["_title_clean"] = df[TITLE_COL].apply(clean_text)
    df["_body_clean"]  = df[BODY_COL].apply(clean_text)

    # Combined text — plain text [SEP] marker, NOT a special token
    # RoBERTa tokenizer handles </s> separator automatically
    df[TEXT_COL] = df["_title_clean"] + " [SEP] " + df["_body_clean"]

    # Trim to ~2000 chars per instructions/06 Step 4
    df[TEXT_COL] = df[TEXT_COL].apply(lambda x: x[:2000])

    # Drop temp columns
    df = df.drop(columns=["_title_clean", "_body_clean"])

    if is_train:
        df = encode_labels(df, name)

    avg_len = df[TEXT_COL].str.len().mean()
    logger.info(
        "[%s] Done. Shape: %s | Avg combined length: %.0f chars",
        name, df.shape, avg_len
    )
    return df


# ---------------------------------------------------------------------------
# Stratified split — 80/20 per instructions/06
# ---------------------------------------------------------------------------

def stratified_split(
    df: pd.DataFrame,
    val_split: float = VAL_SPLIT,
    seed: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified 80/20 train/val split.

    Per instructions/06 Step 6:
        test_size=0.20, stratify=df['label'], random_state=42

    Args:
        df        : Preprocessed training DataFrame (must have LABEL_COL).
        val_split : Validation fraction. Default 0.20 from config.
        seed      : Random seed. Default 42.

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
        "Split — Train: %d rows | Val: %d rows (%.0f%% / %.0f%%)",
        len(train_df), len(val_df),
        (1 - val_split) * 100, val_split * 100,
    )

    for split_name, split_df in [("Train", train_df), ("Val", val_df)]:
        dist = split_df[LABEL_COL].value_counts().to_dict()
        readable = {ID2LABEL.get(k, k): v for k, v in dist.items()}
        logger.info("[%s] Label distribution: %s", split_name, readable)

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_split(
    df: pd.DataFrame,
    filepath: Path,
    name: str = "split",
) -> None:
    """
    Save DataFrame to CSV.

    Args:
        df       : DataFrame to save.
        filepath : Target .csv path.
        name     : Label for logging.
    """
    create_directory(filepath.parent)
    df.to_csv(filepath, index=False)
    logger.info("Saved [%s] → %s  (%d rows)", name, filepath, len(df))


# ---------------------------------------------------------------------------
# EDA outputs — per instructions/06 Step 7
# ---------------------------------------------------------------------------

def save_eda_plots(
    df: pd.DataFrame,
    outputs_dir: Path,
) -> None:
    """
    Save EDA plots per instructions/06 Step 7.

    Outputs:
        outputs/class_distribution.png
        outputs/text_length_dist.png

    Args:
        df          : Full preprocessed training DataFrame.
        outputs_dir : Directory to save plots.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend for server/Colab
        import matplotlib.pyplot as plt

        create_directory(outputs_dir)

        # Class distribution
        fig, ax = plt.subplots(figsize=(6, 4))
        labels_named = df[LABEL_COL].map(ID2LABEL)
        labels_named.value_counts().plot(kind="bar", ax=ax, color=["#2196F3", "#F44336"])
        ax.set_title("Class Distribution (REAL vs FAKE)")
        ax.set_xlabel("Label")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=0)
        plt.tight_layout()
        plt.savefig(outputs_dir / "class_distribution.png", dpi=150)
        plt.close()
        logger.info("Saved class_distribution.png")

        # Text length distribution
        df["_text_len"] = df[TEXT_COL].str.len()
        fig, ax = plt.subplots(figsize=(8, 4))
        df["_text_len"].hist(bins=50, ax=ax, color="#4CAF50", edgecolor="white")
        ax.set_title("Combined Text Length Distribution")
        ax.set_xlabel("Character Count")
        ax.set_ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(outputs_dir / "text_length_dist.png", dpi=150)
        plt.close()
        df = df.drop(columns=["_text_len"])
        logger.info("Saved text_length_dist.png")
        logger.info(
            "Text stats — Avg: %.0f chars | Max: %d chars",
            df[TEXT_COL].str.len().mean(),
            df[TEXT_COL].str.len().max(),
        )

    except ImportError:
        logger.warning("matplotlib not available — skipping EDA plots.")


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

    Per instructions/06_data_preprocessing.md Steps 1–7.

    Args:
        train_path    : Path to raw train CSV. Defaults to config.TRAIN_FILE.
        test_path     : Path to raw test CSV. Defaults to config.TEST_FILE.
        processed_dir : Output directory. Defaults to config.PROCESSED_DIR.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            (train_df, val_df, test_df)

    Raises:
        FileNotFoundError: If train or test CSV does not exist.
    """
    set_seed(SEED)

    train_src = Path(train_path  or TRAIN_FILE)
    test_src  = Path(test_path   or TEST_FILE)
    out_dir   = Path(processed_dir or PROCESSED_DIR)

    # --- Validate source files ---
    if not train_src.exists():
        raise FileNotFoundError(
            f"train.csv not found at: {train_src}\n"
            f"Place dataset in data/raw/ before running preprocessing.\n"
            f"See SETUP.md for dataset download instructions."
        )
    if not test_src.exists():
        raise FileNotFoundError(
            f"test.csv not found at: {test_src}\n"
            f"Place dataset in data/raw/ before running preprocessing."
        )

    # --- Load ---
    logger.info("=" * 60)
    logger.info("LOADING DATA")
    logger.info("=" * 60)
    raw_train = pd.read_csv(train_src)
    raw_test  = pd.read_csv(test_src)
    logger.info("Raw train: %s | columns: %s", raw_train.shape, raw_train.columns.tolist())
    logger.info("Raw test : %s | columns: %s", raw_test.shape,  raw_test.columns.tolist())

    # --- Preprocess ---
    logger.info("=" * 60)
    logger.info("PREPROCESSING TRAIN SET")
    logger.info("=" * 60)
    processed_train = preprocess_dataframe(raw_train, is_train=True, name="Train")

    logger.info("=" * 60)
    logger.info("PREPROCESSING TEST SET")
    logger.info("=" * 60)
    # Test set may not have label column — handle gracefully per instructions/08
    has_labels = LABEL_COL in raw_test.columns
    processed_test = preprocess_dataframe(raw_test, is_train=has_labels, name="Test")
    if not has_labels:
        logger.info("[Test] No label column detected — inference mode only.")

    # --- Split ---
    logger.info("=" * 60)
    logger.info("STRATIFIED 80/20 SPLIT")
    logger.info("=" * 60)
    train_df, val_df = stratified_split(processed_train)

    # --- EDA plots ---
    save_eda_plots(processed_train, Path(out_dir).parent.parent / "outputs")

    # --- Save processed CSVs ---
    logger.info("=" * 60)
    logger.info("SAVING OUTPUTS")
    logger.info("=" * 60)
    save_split(train_df,       out_dir / "train.csv",  "train")
    save_split(val_df,         out_dir / "val.csv",    "val")
    save_split(processed_test, out_dir / "test.csv",   "test")

    # --- Statistics report ---
    stats = {
        "raw_train_rows":        int(len(raw_train)),
        "raw_test_rows":         int(len(raw_test)),
        "processed_train_rows":  int(len(train_df)),
        "processed_val_rows":    int(len(val_df)),
        "processed_test_rows":   int(len(processed_test)),
        "val_split":             float(VAL_SPLIT),
        "seed":                  int(SEED),
        "text_col":              TEXT_COL,
        "label_encoding":        {"REAL": 1, "FAKE": 0},
        "train_label_dist":      {ID2LABEL.get(k, str(k)): int(v)
                                  for k, v in train_df[LABEL_COL].value_counts().items()},
        "val_label_dist":        {ID2LABEL.get(k, str(k)): int(v)
                                  for k, v in val_df[LABEL_COL].value_counts().items()},
        "combined_avg_len":      round(float(train_df[TEXT_COL].str.len().mean()), 2),
        "combined_max_len":      int(train_df[TEXT_COL].str.len().max()),
    }
    save_json(stats, out_dir / "preprocessing_stats.json")

    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("  Train : %d rows", len(train_df))
    logger.info("  Val   : %d rows", len(val_df))
    logger.info("  Test  : %d rows", len(processed_test))
    logger.info("  Out   : %s",      out_dir)
    logger.info("=" * 60)

    return train_df, val_df, processed_test


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys as _sys
    try:
        run_preprocessing()
    except FileNotFoundError as e:
        logger.error("%s", e)
        _sys.exit(1)
    except (ValueError, AssertionError) as e:
        logger.error("Preprocessing failed: %s", e)
        _sys.exit(1)
