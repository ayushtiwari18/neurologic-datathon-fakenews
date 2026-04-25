"""
preprocess.py — Data preprocessing pipeline for FakeGuard.

Dataset: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Input  : Fake.csv + True.csv  (passed in as paths from the notebook)
Output : data/processed/train.csv, val.csv, test.csv

Pipeline steps:
    1.  Read Fake.csv  → add label='FALSE'
        Read True.csv  → add label='TRUE'
    2.  Merge + shuffle
    3.  Validate columns: title, text, subject, label
    4.  Drop date column (noise, not useful for classification)
    5.  Handle missing values (fill empty with "")
    6.  Remove exact duplicates on (title, text)
    7.  Clean text: HTML unescape → lowercase → strip URLs/HTML tags/special chars
    8.  Build combined = subject_clean + " [SEP] " + title_clean + " [SEP] " + text_clean
        NOTE: [SEP] is plain text, NOT a special token.
              RoBERTa's tokenizer adds </s> separators automatically.
    9.  Encode labels: TRUE→1, FALSE→0
    10. Stratified 70 / 15 / 15 split  (train / val / test)
    11. Save processed CSVs to data/processed/
    12. Save EDA plots + stats JSON to outputs/
"""

import re
import sys
import html
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    PROCESSED_DIR,
    OUTPUTS_DIR,
    TITLE_COL,
    BODY_COL,
    SUBJECT_COL,
    DATE_COL,
    TEXT_COL,
    LABEL_COL,
    LABEL2ID,
    ID2LABEL,
    SEED,
)
from src.utils import get_logger, set_seed, create_directory, save_json

logger = get_logger("preprocess")

# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Clean a single text string."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z0-9\s.,!?']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Validation & cleaning helpers
# ---------------------------------------------------------------------------

def validate_columns(df: pd.DataFrame, required: list, name: str = "DataFrame") -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{name}] Missing columns: {missing}.\n"
            f"Found columns: {df.columns.tolist()}\n"
            f"Expected: {required}"
        )
    logger.info("[%s] Column validation passed.", name)


def handle_missing(df: pd.DataFrame, name: str = "DataFrame") -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=[TITLE_COL, BODY_COL], how="all")
    df[TITLE_COL]   = df[TITLE_COL].fillna("")
    df[BODY_COL]    = df[BODY_COL].fillna("")
    df[SUBJECT_COL] = df[SUBJECT_COL].fillna("") if SUBJECT_COL in df.columns else ""
    dropped = before - len(df)
    if dropped:
        logger.info("[%s] Dropped %d rows with both title+text empty.", name, dropped)
    return df.reset_index(drop=True)


def remove_duplicates(df: pd.DataFrame, name: str = "DataFrame") -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=[TITLE_COL, BODY_COL])
    removed = before - len(df)
    if removed:
        logger.info("[%s] Removed %d duplicate rows.", name, removed)
    return df.reset_index(drop=True)


def encode_labels(df: pd.DataFrame, name: str = "DataFrame") -> pd.DataFrame:
    """Map TRUE/FALSE string labels to 1/0 integers."""
    df[LABEL_COL] = df[LABEL_COL].map(LABEL2ID)
    null_count = df[LABEL_COL].isnull().sum()
    if null_count > 0:
        raise ValueError(
            f"[{name}] Label encoding failed for {null_count} rows.\n"
            f"Unique raw labels were: {df[LABEL_COL].unique().tolist()}"
        )
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    dist = df[LABEL_COL].value_counts().to_dict()
    logger.info("[%s] Label distribution (0=FALSE/fake, 1=TRUE/real): %s", name, dist)
    return df


# ---------------------------------------------------------------------------
# Core preprocessing
# ---------------------------------------------------------------------------

def preprocess_dataframe(df: pd.DataFrame, name: str = "DataFrame") -> pd.DataFrame:
    """
    Full preprocessing for the merged Fake+True DataFrame.

    Expects columns: title, text, subject, label
    Produces column: combined
    """
    validate_columns(df, [TITLE_COL, BODY_COL, SUBJECT_COL, LABEL_COL], name)

    # Drop date — adds noise, not useful for text classification
    if DATE_COL in df.columns:
        df = df.drop(columns=[DATE_COL])
        logger.info("[%s] Dropped '%s' column.", name, DATE_COL)

    df = handle_missing(df, name)
    df = remove_duplicates(df, name)

    title_clean   = df[TITLE_COL].apply(clean_text)
    body_clean    = df[BODY_COL].apply(clean_text)
    subject_clean = df[SUBJECT_COL].apply(clean_text)

    # subject is a strong discriminator (e.g. "politicsNews" vs "News")
    # combining all three fields gives RoBERTa maximum context
    df[TEXT_COL] = (
        subject_clean + " [SEP] " + title_clean + " [SEP] " + body_clean
    ).str[:2000]  # cap at 2000 chars — tokenizer will truncate to 512 tokens anyway

    df = encode_labels(df, name)

    logger.info("[%s] Final shape after preprocessing: %s", name, df.shape)
    return df


# ---------------------------------------------------------------------------
# 70 / 15 / 15 stratified split
# ---------------------------------------------------------------------------

def stratified_three_way_split(
    df: pd.DataFrame,
    seed: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the full preprocessed DataFrame into train / val / test.

    Ratios: 70% train | 15% val | 15% test
    Stratified on label so both classes are equally represented in every split.
    """
    # Step 1: carve out 30% (will become val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df[LABEL_COL],
        random_state=seed,
        shuffle=True,
    )
    # Step 2: split the 30% evenly into 15% val + 15% test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df[LABEL_COL],
        random_state=seed,
        shuffle=True,
    )
    logger.info(
        "70/15/15 split — train: %d | val: %d | test: %d",
        len(train_df), len(val_df), len(test_df),
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_split(df: pd.DataFrame, filepath: Path, name: str) -> None:
    create_directory(filepath.parent)
    df.to_csv(filepath, index=False)
    logger.info("Saved [%s] → %s  (%d rows)", name, filepath, len(df))


def save_eda_plots(df: pd.DataFrame, outputs_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        create_directory(outputs_dir)

        # Class distribution bar chart
        fig, ax = plt.subplots(figsize=(6, 4))
        df[LABEL_COL].map(ID2LABEL).value_counts().plot(
            kind="bar", ax=ax, color=["#2196F3", "#F44336"]
        )
        ax.set_title("Class Distribution (FALSE=Fake, TRUE=Real)")
        ax.set_xlabel("Label")
        ax.set_ylabel("Count")
        plt.tight_layout()
        plt.savefig(outputs_dir / "class_distribution.png", dpi=150)
        plt.close()

        # Combined text length histogram
        lengths = df[TEXT_COL].str.len()
        fig, ax = plt.subplots(figsize=(8, 4))
        lengths.hist(bins=50, ax=ax, color="#4CAF50", edgecolor="white")
        ax.set_title("Combined Text Length Distribution")
        ax.set_xlabel("Character Count")
        ax.set_ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(outputs_dir / "text_length_dist.png", dpi=150)
        plt.close()

        logger.info("Saved EDA plots → %s", outputs_dir)
    except ImportError:
        logger.warning("matplotlib not installed — skipping EDA plots.")


def _build_stats(train_df, val_df, test_df):
    return {
        "dataset": "clmentbisaillon/fake-and-real-news-dataset",
        "kaggle_url": "https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset",
        "split_ratio": "70 / 15 / 15",
        "seed": int(SEED),
        "text_col": TEXT_COL,
        "label_encoding": {"TRUE": 1, "FALSE": 0},
        "train_rows": int(len(train_df)),
        "val_rows":   int(len(val_df)),
        "test_rows":  int(len(test_df)),
        "train_label_dist": {
            ID2LABEL.get(k, str(k)): int(v)
            for k, v in train_df[LABEL_COL].value_counts().items()
        },
        "val_label_dist": {
            ID2LABEL.get(k, str(k)): int(v)
            for k, v in val_df[LABEL_COL].value_counts().items()
        },
        "test_label_dist": {
            ID2LABEL.get(k, str(k)): int(v)
            for k, v in test_df[LABEL_COL].value_counts().items()
        },
        "avg_combined_len": round(float(train_df[TEXT_COL].str.len().mean()), 2),
        "max_combined_len": int(train_df[TEXT_COL].str.len().max()),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_preprocessing(
    fake_csv_path: str = None,
    true_csv_path: str = None,
    processed_dir: str = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full preprocessing pipeline for the Clément Bisaillon fake-and-real-news dataset.

    Parameters
    ----------
    fake_csv_path : str
        Absolute path to Fake.csv on disk (e.g. '/kaggle/input/fake-and-real-news-dataset/Fake.csv')
    true_csv_path : str
        Absolute path to True.csv on disk (e.g. '/kaggle/input/fake-and-real-news-dataset/True.csv')
    processed_dir : str, optional
        Where to write train.csv / val.csv / test.csv.  Defaults to config.PROCESSED_DIR.

    Returns
    -------
    train_df, val_df, test_df  — processed DataFrames
    """
    set_seed(SEED)

    out_dir     = Path(processed_dir or PROCESSED_DIR)
    outputs_dir = Path(OUTPUTS_DIR)

    logger.info("=" * 60)
    logger.info("STARTING PREPROCESSING")
    logger.info("Dataset : clmentbisaillon/fake-and-real-news-dataset")
    logger.info("=" * 60)

    # ── 1. Validate inputs ──────────────────────────────────────────────────
    if not fake_csv_path or not true_csv_path:
        raise ValueError(
            "Both fake_csv_path and true_csv_path must be provided.\n"
            "Example:\n"
            "  run_preprocessing(\n"
            "      fake_csv_path='/kaggle/input/fake-and-real-news-dataset/Fake.csv',\n"
            "      true_csv_path='/kaggle/input/fake-and-real-news-dataset/True.csv'\n"
            "  )"
        )

    fake_path = Path(fake_csv_path)
    true_path = Path(true_csv_path)

    if not fake_path.exists():
        raise FileNotFoundError(
            f"Fake.csv not found at: {fake_path}\n"
            "Add the dataset in Kaggle: Notebook → Add Data → "
            "search 'clmentbisaillon fake and real news dataset'"
        )
    if not true_path.exists():
        raise FileNotFoundError(
            f"True.csv not found at: {true_path}\n"
            "Add the dataset in Kaggle: Notebook → Add Data → "
            "search 'clmentbisaillon fake and real news dataset'"
        )

    # ── 2. Load ──────────────────────────────────────────────────────────────
    logger.info("Loading Fake.csv from: %s", fake_path)
    fake_df = pd.read_csv(fake_path)
    logger.info("Fake.csv  — shape: %s | columns: %s", fake_df.shape, fake_df.columns.tolist())

    logger.info("Loading True.csv from: %s", true_path)
    true_df = pd.read_csv(true_path)
    logger.info("True.csv  — shape: %s | columns: %s", true_df.shape, true_df.columns.tolist())

    # ── 3. Assign labels ────────────────────────────────────────────────────
    fake_df[LABEL_COL] = "FALSE"   # Fake news = FALSE = 0
    true_df[LABEL_COL] = "TRUE"    # Real news = TRUE  = 1

    # ── 4. Merge + shuffle ──────────────────────────────────────────────────
    merged_df = pd.concat([fake_df, true_df], ignore_index=True)
    merged_df = merged_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    logger.info(
        "Merged shape: %s | FALSE: %d | TRUE: %d",
        merged_df.shape,
        (merged_df[LABEL_COL] == "FALSE").sum(),
        (merged_df[LABEL_COL] == "TRUE").sum(),
    )

    # ── 5. Preprocess ────────────────────────────────────────────────────────
    processed_df = preprocess_dataframe(merged_df, name="Merged")

    # ── 6. Split 70 / 15 / 15 ───────────────────────────────────────────────
    train_df, val_df, test_df = stratified_three_way_split(processed_df, seed=SEED)

    # ── 7. Save ──────────────────────────────────────────────────────────────
    save_eda_plots(train_df, outputs_dir)
    save_split(train_df, out_dir / "train.csv", "train")
    save_split(val_df,   out_dir / "val.csv",   "val")
    save_split(test_df,  out_dir / "test.csv",  "test")

    stats = _build_stats(train_df, val_df, test_df)
    save_json(stats, out_dir / "preprocessing_stats.json")

    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("Train : %d rows", len(train_df))
    logger.info("Val   : %d rows", len(val_df))
    logger.info("Test  : %d rows", len(test_df))
    logger.info("=" * 60)

    return train_df, val_df, test_df


if __name__ == "__main__":
    import sys
    # CLI usage: python src/preprocess.py <fake_csv_path> <true_csv_path>
    if len(sys.argv) == 3:
        run_preprocessing(fake_csv_path=sys.argv[1], true_csv_path=sys.argv[2])
    else:
        logger.error(
            "Usage: python src/preprocess.py <path/to/Fake.csv> <path/to/True.csv>\n"
            "Example:\n"
            "  python src/preprocess.py /kaggle/input/fake-and-real-news-dataset/Fake.csv "
            "/kaggle/input/fake-and-real-news-dataset/True.csv"
        )
        sys.exit(1)
