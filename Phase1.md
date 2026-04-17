# Phase 1 — Foundational Source Code Layer
**Cycle:** 1 of N  
**Status:** ACTIVE  
**Goal:** Build the complete executable `src/` pipeline from scratch.

---

## Scope

Phase 1 covers all files required to go from raw CSV data to a trained model and submission file.  
No API, no Gradio, no deployment. Core ML pipeline only.

---

## Files to Create (in order)

### Batch A — Data Layer (Cycle 2)
| File | Purpose |
|------|---------|
| `src/__init__.py` | Makes src a Python package |
| `src/preprocess.py` | Clean text, combine title+body, split train/val |
| `src/dataset.py` | PyTorch Dataset class for RoBERTa tokenization |

### Batch B — Baseline (Cycle 3)
| File | Purpose |
|------|---------|
| `src/baseline.py` | TF-IDF + Logistic Regression fast baseline |
| `src/utils.py` | Shared helpers: seed, metrics, save/load |
| `notebooks/00_setup_and_baseline.ipynb` | Colab notebook for Batch A+B |

### Batch C — Transformer Training (Cycle 4)
| File | Purpose |
|------|---------|
| `src/train.py` | Full RoBERTa fine-tuning with HuggingFace Trainer |
| `src/evaluate.py` | Accuracy, F1, confusion matrix, classification report |
| `src/predict.py` | Inference on test.csv → submission.csv |

### Batch D — Notebook + Submission (Cycle 5)
| File | Purpose |
|------|---------|
| `notebooks/01_train_roberta.ipynb` | End-to-end Colab training notebook |
| `notebooks/02_evaluate_and_submit.ipynb` | Evaluation + submission generation |
| `outputs/.gitkeep` | Ensures outputs/ directory exists in repo |

---

## Acceptance Criteria for Phase 1 Completion

- [ ] `src/preprocess.py` runs without error on valid train.csv
- [ ] `src/baseline.py` prints accuracy + F1 score
- [ ] `src/train.py` trains RoBERTa for 3 epochs without crash
- [ ] `src/evaluate.py` generates classification report
- [ ] `src/predict.py` outputs `outputs/submission.csv`
- [ ] At least one Colab notebook runs end-to-end

---

## Constraints

- All code must be compatible with Google Colab T4 (CUDA 12.x)
- Do NOT pin torch version
- MAX_LEN = 512, BATCH_SIZE = 16 (from config.py)
- SEED = 42 everywhere
- Use `roberta-base` from HuggingFace hub
