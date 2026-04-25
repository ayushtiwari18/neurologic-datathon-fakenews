# ============================================================
# config.py — Central configuration for FakeGuard pipeline
# NeuroLogic '26 Datathon — Fake News Detection
# ============================================================
# SOURCE OF TRUTH: All values below match instructions/ exactly.
# Do NOT change values here without updating instructions/.
# ============================================================

import os

# --- Paths ---
DATA_DIR        = "data/"
RAW_DIR         = "data/raw/"              # original dataset — DO NOT MODIFY
PROCESSED_DIR   = "data/processed/"       # cleaned outputs from preprocess.py
MODEL_DIR       = "models/"               # saved model weights (never commit)
OUTPUTS_DIR     = "outputs/"              # plots, metrics, confusion matrix

# --- Dataset ---
# Source: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
# Kaggle path: /kaggle/input/fake-and-real-news-dataset/
# Two files:
#   Fake.csv  — 23,481 fake news articles  → label = FALSE → 0
#   True.csv  — 21,417 real news articles  → label = TRUE  → 1
# Total: ~44,898 rows. Nearly balanced (1.1x ratio).
# Columns in both files: title, text, subject, date
FAKE_CSV        = "Fake.csv"              # filename only — path resolved in notebook
TRUE_CSV        = "True.csv"              # filename only — path resolved in notebook

# Processed output files (created by preprocess.py)
TRAIN_FILE      = os.path.join(RAW_DIR, "train.csv")    # NOT used — preprocessing reads Fake/True directly
TEST_FILE       = os.path.join(RAW_DIR, "test.csv")     # NOT used — no separate test file

# --- Column Names ---
# Both Fake.csv and True.csv have these exact columns:
LABEL_COL       = "label"                 # added by Cell 6 — does not exist in raw files
TITLE_COL       = "title"                 # article headline
BODY_COL        = "text"                  # article body
SUBJECT_COL     = "subject"               # topic category (e.g. politicsNews, worldnews)
DATE_COL        = "date"                  # publication date — dropped during preprocessing
TEXT_COL        = "combined"              # CREATED during preprocessing
                                           # = subject_clean + " [SEP] " + title_clean + " [SEP] " + text_clean

# --- Label Encoding ---
# Fake.csv  → we assign label = 'FALSE' → encodes to 0
# True.csv  → we assign label = 'TRUE'  → encodes to 1
LABEL2ID        = {
    "TRUE":  1,
    "FALSE": 0,
    "true":  1,
    "false": 0,
    "1":     1,
    "0":     0,
     1:      1,
     0:      0,
     1.0:    1,
     0.0:    0,
}
ID2LABEL        = {1: "TRUE", 0: "FALSE"}
NUM_LABELS      = 2

# --- Class Balance ---
# Fake.csv: 23,481 rows | True.csv: 21,417 rows
# Ratio: 1.1x — nearly balanced. Class weights not required.
# USE_CLASS_WEIGHTS left True as a safety net (negligible impact on balanced data).
USE_CLASS_WEIGHTS = True

# --- Model ---
BASELINE_MODEL  = "tfidf_logreg"
TRANSFORMER     = "roberta-base"
MAX_LEN         = 512

# --- Training Hyperparameters ---
BATCH_SIZE      = 16
EPOCHS          = 3
LEARNING_RATE   = 2e-5
WARMUP_STEPS    = 500
WEIGHT_DECAY    = 0.01
VAL_SPLIT       = 0.15                    # 70/15/15 split for single-file dataset
SEED            = 42

# --- Inference ---
CONFIDENCE_THRESHOLD = 0.70

# --- API ---
API_HOST        = "0.0.0.0"
API_PORT        = 8000
