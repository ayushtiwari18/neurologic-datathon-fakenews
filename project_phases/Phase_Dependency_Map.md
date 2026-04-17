# Phase Dependency Map
**Project:** NeuroLogic '26 Datathon — Fake News Detection  
**Version:** 2.0 (Full Phase Map) | **Updated:** 2026-04-17

---

## Phase Execution Order

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1: Environment Setup                                   │
│  Outputs: GPU confirmed, dirs created, imports verified       │
└────────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2: Data Pipeline                                       │
│  Outputs: train_clean.csv, val_clean.csv, test_clean.csv      │
└────────────────────────────────┬───────────────────────────────┘
                               │
               ┌────────────┴────────────┐
               │                           │
               ▼                           ▼
┌───────────────┐         ┌──────────────────────────┐
│  PHASE 3       │         │  PHASE 4: RoBERTa Training      │
│  Baseline      │         │  Outputs: models/roberta_*      │
│  (Parallel)    │         │  roberta_train_metrics.json     │
└───────┬──────┘         └────────────┬─────────────┘
       │                           │
       └────────────┬────────────┘
                       │
                       ▼
          ┌────────────────────────┐
          │  PHASE 5: Evaluation         │
          │  Outputs: confusion matrix,  │
          │  eval_metrics, error_analysis │
          └────────────┬───────────┘
                       │
                       ▼
          ┌────────────────────────┐
          │  PHASE 6: Optimization       │
          │  Outputs: best model,        │
          │  hyperparam_log.csv          │
          └────────────┬───────────┘
                       │
                       ▼
          ┌────────────────────────┐
          │  PHASE 7: Inference          │
          │  Outputs: submission.csv     │
          └────────────┬───────────┘
                       │
                       ▼
          ┌────────────────────────┐
          │  PHASE 8: Demo + Docs        │
          │  Outputs: app.py, README,    │
          │  REPRODUCIBILITY.md          │
          └────────────────────────┘
```

---

## Phase-to-File Dependency Table

| Phase | Requires From | Produces For |
|-------|--------------|-------------|
| Phase 1 | `config.py`, `requirements.txt` | Stable env for all phases |
| Phase 2 | Phase 1 env, `data/train.csv`, `data/test.csv` | `train_clean.csv`, `val_clean.csv`, `test_clean.csv` |
| Phase 3 | Phase 2 processed CSVs | `baseline_metrics.json`, `baseline_confusion_matrix.png` |
| Phase 4 | Phase 2 processed CSVs, Phase 1 GPU | `models/roberta_fakenews/`, `roberta_train_metrics.json` |
| Phase 5 | Phase 4 model, Phase 2 val CSV, Phase 3 metrics | `roberta_eval_metrics.json`, `error_analysis.csv`, `model_comparison.json` |
| Phase 6 | Phase 4 model, Phase 5 metrics, Phase 2 CSVs | `models/roberta_fakenews_best/`, `hyperparam_log.csv` |
| Phase 7 | Phase 6 best model (or Phase 4), Phase 2 test CSV | `submission.csv` |
| Phase 8 | Phase 7 submission, Phase 5 outputs, all `src/*.py` | `app.py`, `README.md`, `REPRODUCIBILITY.md` |

---

## Blocking Conditions

| Condition | Blocks | Resolution |
|-----------|--------|------------|
| GPU not available | Phase 4, 6, 7 | Switch to T4 runtime |
| `train.csv` missing | Phase 2, 3, 4 | Place data in `data/` folder |
| Phase 2 fails | All downstream phases | Fix preprocessing before continuing |
| Phase 4 crashes | Phase 5, 6, 7 | Reduce batch_size, fix token_type_ids |
| Phase 5 accuracy < 0.93 | Phase 6 (re-train needed) | Add epoch, check label encoding |
| `submission.csv` missing | Phase 8, submission | Re-run Phase 7 before any docs work |

---

## Parallel Execution Opportunities

| Parallel Work | Condition |
|--------------|----------|
| Phase 3 (baseline) can run while Phase 4 trains | Both depend on Phase 2 only |
| Phase 8 README drafting can start after Phase 5 | Does not need Phase 6/7 complete |
| `src/evaluate.py` can be written during Phase 4 training wait | Code does not need model to be written |
| Gradio demo template can be drafted during any training phase | Only needs model path at runtime |

---

## Critical Path (Minimum Viable Submission)

```
Phase 1 → Phase 2 → Phase 4 → Phase 7 → submission.csv
```

This 4-phase critical path is the **absolute minimum** to produce a valid submission.  
All other phases add score but are not blockers for submission existence.

---

## File Inventory by Phase

### Planning Layer (COMPLETE ✅)
```
Repository_Scan_Report.md
Phase1.md, Phase_Dependency_Map.md    ← Cycle 1
project_phases/Phase1.md – Phase8.md  ← Cycles 2–5
project_phases/Execution_Timeline.md  ← Cycle 4
project_phases/Risk_Management.md     ← Cycle 5
project_phases/Winning_Strategy.md    ← Cycle 5
project_phases/Phase_Dependency_Map.md ← Cycle 5
```

### Source Code Layer (NEXT ⏳)
```
src/__init__.py
src/utils.py
src/preprocess.py
src/dataset.py
src/baseline.py
src/train.py
src/evaluate.py
src/predict.py
src/app.py
```

### Notebook Layer (AFTER SOURCE ⏳)
```
notebooks/00_setup_and_baseline.ipynb
notebooks/01_train_roberta.ipynb
notebooks/02_evaluate_and_submit.ipynb
```

### Output Layer (GENERATED AT RUNTIME ⏳)
```
data/processed/train_clean.csv
data/processed/val_clean.csv
data/processed/test_clean.csv
outputs/baseline_metrics.json
outputs/baseline_confusion_matrix.png
outputs/roberta_train_metrics.json
outputs/roberta_eval_metrics.json
outputs/roberta_confusion_matrix.png
outputs/error_analysis.csv
outputs/model_comparison.json
outputs/optimized_metrics.json
outputs/hyperparam_log.csv
outputs/submission.csv
```
