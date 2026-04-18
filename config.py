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
TEXT_COL        = "combined"              # CREATED during preprocessing
                                           # = title_clean + " [SEP] " + text_clean
                                           # Instructions use 'combined' — NOT 'combined_text'

# --- Label Encoding ---
# Instructions (06_data_preprocessing.md, 04_model_training_strategy.md):
#   REAL = 1,  FAKE = 0
# WARNING: This is counter-intuitive but matches the instructions exactly.
# Do NOT swap — any change here will invert all model predictions.
LABEL2ID        = {"REAL": 1, "FAKE": 0,
                   "real": 1, "fake": 0,
                   "1":    1, "0":    0,
                    1:     1,  0:     0,
                    1.0:   1,  0.0:   0}  # handles pandas float-loaded labels
ID2LABEL        = {1: "REAL", 0: "FAKE"}
NUM_LABELS      = 2

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
WARMUP_STEPS    = 500                     # 10% of total steps (~40k/16*3 = ~7500 steps)
WEIGHT_DECAY    = 0.01
VAL_SPLIT       = 0.20                    # 80/20 split — per instructions/06
SEED            = 42

# --- Inference ---
# Instructions (04_model_training_strategy.md)
CONFIDENCE_THRESHOLD = 0.70              # below this → "UNCERTAIN — needs human review"

# --- API (Phase 8 FastAPI / Gradio) ---
API_HOST        = "0.0.0.0"
API_PORT        = 8000
