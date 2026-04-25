# ============================================================
# config.py — Central configuration for FakeGuard pipeline
# NeuroLogic '26 Datathon — Fake News Detection
# ============================================================
# SOURCE OF TRUTH: All values below match instructions/ exactly.
# Do NOT change values here without updating instructions/.
# ============================================================

import os

# --- Paths ---
# Instructions (02_project_structure.md): raw data in data/raw/
DATA_DIR        = "data/"
RAW_DIR         = "data/raw/"              # original dataset — DO NOT MODIFY
PROCESSED_DIR   = "data/processed/"       # cleaned outputs from preprocess.py
MODEL_DIR       = "models/"               # saved model weights (never commit)
OUTPUTS_DIR     = "outputs/"              # plots, metrics, confusion matrix

# --- Dataset Files ---
# Instructions (02_project_structure.md): data/raw/dataset.csv
TRAIN_FILE      = os.path.join(RAW_DIR, "train.csv")
TEST_FILE       = os.path.join(RAW_DIR, "test.csv")

# --- Column Names ---
# Instructions (06_data_preprocessing.md, 03_pipeline_architecture.md)
LABEL_COL       = "label"                 # raw label column in CSV
TITLE_COL       = "title"                 # raw title column in CSV
BODY_COL        = "text"                  # raw body column in CSV
SUBJECT_COL     = "subject"               # topic/category column — used in combined text
TEXT_COL        = "combined"              # CREATED during preprocessing
                                           # = subject_clean + " [SEP] " + title_clean + " [SEP] " + text_clean

# --- Label Encoding ---
# Competition dataset uses TRUE/FALSE labels (not REAL/FAKE)
# TRUE  = Real news = 1
# FALSE = Fake news = 0
LABEL2ID        = {
    "TRUE":  1, "FALSE": 0,   # competition dataset format
    "true":  1, "false": 0,
    "REAL":  1, "FAKE":  0,   # fallback for WELFake / internal use
    "real":  1, "fake":  0,
    "1":     1, "0":     0,
     1:      1,  0:      0,
     1.0:    1,  0.0:    0,   # handles pandas float-loaded labels
}
ID2LABEL        = {1: "TRUE", 0: "FALSE"}
NUM_LABELS      = 2

# --- Class Imbalance ---
# Dataset: TRUE=15438 (64.6%), FALSE=8455 (35.4%) — imbalanced 1.8x
# USE_CLASS_WEIGHTS=True tells train.py to compute and apply class weights
# to the loss function so the model doesn't just predict TRUE for everything.
USE_CLASS_WEIGHTS = True

# --- Model ---
# Instructions (04_model_training_strategy.md, 10_resources.md)
BASELINE_MODEL  = "tfidf_logreg"          # Phase 3 fast baseline
TRANSFORMER     = "roberta-base"          # Phase 4 main model (HuggingFace Hub)
MAX_LEN         = 512                     # max token length — never reduce to 128/256

# --- Training Hyperparameters ---
# Instructions (04_model_training_strategy.md)
BATCH_SIZE      = 16                      # safe for Colab T4; reduce to 8 if OOM
EPOCHS          = 3                       # sufficient for convergence on ~40k samples
LEARNING_RATE   = 2e-5                    # AdamW default for RoBERTa fine-tuning
WARMUP_STEPS    = 500                     # 10% of total steps
WEIGHT_DECAY    = 0.01
VAL_SPLIT       = 0.20                    # 80/20 split — per instructions/06
SEED            = 42

# --- Inference ---
# Instructions (04_model_training_strategy.md)
CONFIDENCE_THRESHOLD = 0.70              # below this → "UNCERTAIN — needs human review"

# --- API (Phase 8 FastAPI / Gradio) ---
API_HOST        = "0.0.0.0"
API_PORT        = 8000
