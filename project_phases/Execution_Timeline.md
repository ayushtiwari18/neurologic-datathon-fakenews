# Execution Timeline
**Project:** NeuroLogic '26 Datathon — Challenge 2: Fake News Detection  
**Model:** Fine-tuned RoBERTa (`roberta-base`)  
**Target Environment:** Google Colab T4 GPU  
**Last Updated:** 2026-04-17

---

## Pre-Hackathon Preparation (Before Event Starts)

| Task | Time Required | Owner | Status |
|------|--------------|-------|--------|
| Read all `instructions/` files | 20 min | Human | ✅ Done |
| Confirm Colab T4 GPU access | 5 min | Human | ⬜ |
| Pre-download `roberta-base` weights | 5 min | Human | ⬜ |
| Confirm dataset format (columns) | 10 min | Human | ⬜ |
| Clone repo into Colab | 2 min | Human | ⬜ |
| Run `pip install -r requirements.txt` | 3 min | Human | ⬜ |
| Verify all imports + GPU check | 2 min | Human | ⬜ |
| **Pre-Hackathon Total** | **~47 min** | | |

---

## Hackathon Day Timeline

### Hour 0:00 — Phase 1: Environment Stabilization
| Step | Duration | Critical? |
|------|----------|-----------|
| Clone + install + GPU verify | 5 min | 🔴 YES |
| Create directory structure | 2 min | 🔴 YES |
| Confirm config.py loads | 1 min | 🔴 YES |
| **Phase 1 Total** | **~8 min** | |

### Hour 0:08 — Phase 2: Data Pipeline
| Step | Duration | Critical? |
|------|----------|-----------|
| Load + validate train.csv / test.csv | 3 min | 🔴 YES |
| Clean text (lowercase, HTML, URLs) | 5 min | 🔴 YES |
| Create combined_text column | 2 min | 🔴 YES |
| Stratified 80/20 split | 2 min | 🔴 YES |
| Save processed CSVs | 1 min | 🔴 YES |
| **Phase 2 Total** | **~13 min** | |

### Hour 0:21 — Phase 3: Baseline Model
| Step | Duration | Critical? |
|------|----------|-----------|
| Fit TF-IDF + train LogReg | 3 min | 🟡 HIGH |
| Evaluate + print classification report | 1 min | 🟡 HIGH |
| Save confusion matrix + metrics JSON | 2 min | 🟡 HIGH |
| **Phase 3 Total** | **~6 min** | |

### Hour 0:27 — Phase 4: RoBERTa Fine-Tuning ⚡ LONGEST STEP
| Step | Duration | Critical? |
|------|----------|-----------|
| Create FakeNewsDataset class | 5 min | 🔴 YES |
| Configure TrainingArguments | 3 min | 🔴 YES |
| trainer.train() — 3 epochs | 30 min | 🔴 YES |
| trainer.evaluate() + save model | 3 min | 🔴 YES |
| **Phase 4 Total** | **~41 min** | |

**⚠️ RISK BUFFER:** If T4 is slow, allow up to 50 min. Do NOT reduce epochs below 3.

### Hour 1:08 — Phase 5: Evaluation & Error Analysis
| Step | Duration | Critical? |
|------|----------|-----------|
| Batched inference on val set | 5 min | 🔴 YES |
| Generate classification report | 1 min | 🔴 YES |
| Plot + save confusion matrix | 2 min | 🔴 YES |
| Build error analysis table | 3 min | 🟡 HIGH |
| Save model_comparison.json | 1 min | 🟡 HIGH |
| **Phase 5 Total** | **~12 min** | |

### Hour 1:20 — Phase 6: Optimization (TIME-BOXED)
| Step | Duration | Critical? |
|------|----------|-----------|
| LR sweep (3 values × 1 epoch) | 30 min | 🟡 HIGH |
| Epoch extension if improving | 15 min | 🟢 OPTIONAL |
| Soft ensemble test | 5 min | 🟢 OPTIONAL |
| Save best model + hyperparam log | 2 min | 🟡 HIGH |
| **Phase 6 Total** | **~32 min (core)** | |

**⚠️ TIME BOX: If running late, skip ensemble test. LR sweep is mandatory.**

### Hour 1:52 — Phase 7: Inference & Submission
| Step | Duration | Critical? |
|------|----------|-----------|
| Load best model | 2 min | 🔴 YES |
| Run inference on test.csv | 8 min | 🔴 YES |
| Build + validate submission.csv | 3 min | 🔴 YES |
| **Phase 7 Total** | **~13 min** | |

### Hour 2:05 — Phase 8: Demo & Documentation
| Step | Duration | Critical? |
|------|----------|-----------|
| Build Gradio demo + launch | 10 min | 🟡 HIGH |
| Screenshot demo | 2 min | 🟡 HIGH |
| Expand README.md | 15 min | 🟡 HIGH |
| Create REPRODUCIBILITY.md | 5 min | 🟢 OPTIONAL |
| Commit all final files | 5 min | 🔴 YES |
| **Phase 8 Total** | **~37 min** | |

---

## Full Timeline Summary

| Phase | Start | Duration | End | Status |
|-------|-------|----------|-----|--------|
| Phase 1: Setup | 0:00 | 8 min | 0:08 | ⬜ |
| Phase 2: Data Pipeline | 0:08 | 13 min | 0:21 | ⬜ |
| Phase 3: Baseline | 0:21 | 6 min | 0:27 | ⬜ |
| Phase 4: RoBERTa Training | 0:27 | 41 min | 1:08 | ⬜ |
| Phase 5: Evaluation | 1:08 | 12 min | 1:20 | ⬜ |
| Phase 6: Optimization | 1:20 | 32 min | 1:52 | ⬜ |
| Phase 7: Submission | 1:52 | 13 min | 2:05 | ⬜ |
| Phase 8: Demo + Docs | 2:05 | 37 min | 2:42 | ⬜ |
| **TOTAL** | | **~2h 42min** | | |

---

## Risk Buffer Allocation

| Scenario | Buffer Needed | Action |
|----------|--------------|--------|
| T4 slow / OOM | +15 min | Reduce batch_size to 8, add grad accum |
| Colab disconnect | +10 min | Reconnect + reload from checkpoint |
| Data column mismatch | +10 min | Rename columns in preprocess.py |
| Phase 6 skipped | -32 min | Submit Phase 4 model directly |
| Demo fails | -10 min | Skip Gradio, submit CSV only |

**Total safe buffer: ~45 minutes if all optional steps are skipped.**

---

## Critical Checkpoints (Must Pass)

| Checkpoint | Condition | If Failed |
|-----------|-----------|----------|
| CP1: GPU | `torch.cuda.is_available()` = True | Switch runtime |
| CP2: Data | `train_clean.csv` rows > 0 | Fix preprocessing |
| CP3: Baseline | Accuracy ≥ 0.91 | Recheck label encoding |
| CP4: Training | `trainer.train()` completes | Reduce batch_size |
| CP5: Eval | Val accuracy ≥ 0.97 | Add 1 more epoch |
| CP6: Submission | Row count matches test.csv | Fix predict.py |

---

## Submission Deadline Protocol

**T-30 min:** `outputs/submission.csv` must exist. Everything else is bonus.

**T-15 min:** Commit all files. Push to GitHub. Verify repo is public.

**T-5 min:** Submit CSV to judge portal. Do not make further commits.

**T-0:** Stop all work. Do not push broken commits after deadline.
