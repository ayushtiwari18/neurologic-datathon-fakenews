# ============================================================
# config.py — Central configuration for FakeGuard pipeline
# ============================================================

import os

# --- Paths ---
DATA_DIR        = "data/"                      # raw data goes here
PROCESSED_DIR   = "data/processed/"           # cleaned data output
MODEL_DIR       = "models/"                   # saved model checkpoints
OUTPUTS_DIR     = "outputs/"                  # plots, metrics, confusion matrix

# --- Dataset ---
TRAIN_FILE      = os.path.join(DATA_DIR, "train.csv")
TEST_FILE       = os.path.join(DATA_DIR, "test.csv")
LABEL_COL       = "label"                     # column name: 0=Real, 1=Fake
TEXT_COL        = "combined_text"             # created during preprocessing
TITLE_COL       = "title"
BODY_COL        = "text"

# --- Model ---
BASELINE_MODEL  = "tfidf_logreg"              # fast baseline
TRANSFORMER     = "roberta-base"              # main model from HuggingFace
MAX_LEN         = 512                          # max token length for RoBERTa
NUM_LABELS      = 2                            # Real / Fake

# --- Training Hyperparameters ---
BATCH_SIZE      = 16
EPOCHS          = 3
LEARNING_RATE   = 2e-5
WARMUP_STEPS    = 100
WEIGHT_DECAY    = 0.01
VAL_SPLIT       = 0.15                        # 15% held out for validation
SEED            = 42

# --- Inference ---
CONFIDENCE_THRESHOLD = 0.70                   # below this → "Uncertain"

# --- Labels ---
ID2LABEL        = {0: "REAL", 1: "FAKE"}
LABEL2ID        = {"REAL": 0, "FAKE": 1}

# --- API ---
API_HOST        = "0.0.0.0"
API_PORT        = 8000
