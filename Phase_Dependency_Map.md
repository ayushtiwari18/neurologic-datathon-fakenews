# Phase Dependency Map
**Version:** 1.0  
**Updated:** 2026-04-17

---

## Execution Dependency Chain

```
config.py
    └── src/__init__.py
            ├── src/utils.py
            │       ├── src/preprocess.py
            │       │       └── src/dataset.py
            │       │               ├── src/baseline.py   (no dataset.py needed)
            │       │               └── src/train.py
            │       │                       └── src/evaluate.py
            │       │                               └── src/predict.py
            │       │                                       └── outputs/submission.csv
            │       └── (shared by all modules)
            └── notebooks/ (depend on all src/ modules)
```

---

## File-Level Dependencies

| File | Depends On | Blocks |
|------|-----------|--------|
| `config.py` | nothing | everything |
| `src/__init__.py` | config.py | all src imports |
| `src/utils.py` | config.py | train, evaluate, predict |
| `src/preprocess.py` | config.py, utils.py | dataset.py, baseline.py |
| `src/dataset.py` | config.py, preprocess.py | train.py |
| `src/baseline.py` | config.py, preprocess.py | (standalone branch) |
| `src/train.py` | config.py, dataset.py, utils.py | evaluate.py |
| `src/evaluate.py` | config.py, utils.py | predict.py |
| `src/predict.py` | config.py, utils.py | submission.csv |
| `notebooks/*.ipynb` | all src modules | final execution |

---

## Critical Path (Minimum Viable Pipeline)

```
config.py → utils.py → preprocess.py → dataset.py → train.py → predict.py → submission.csv
```

This is the shortest path to a valid submission. Every file on this path is a hard blocker.

---

## Parallel Work Possible

- `baseline.py` can be built in parallel with `dataset.py` (both depend on `preprocess.py`)
- `evaluate.py` can be written while `train.py` is being built
- Notebooks can be drafted while src files are being committed

---

## Risk Map

| Risk | Affected Files | Mitigation |
|------|---------------|------------|
| torch CUDA mismatch | train.py, dataset.py | Never pin torch; use Colab default |
| OOM on T4 (16GB) | train.py | BATCH_SIZE=16, MAX_LEN=512, gradient_checkpointing |
| Label mismatch | predict.py, evaluate.py | Always use ID2LABEL from config.py |
| Missing data columns | preprocess.py | Add column existence checks + fallback |
| tokenizer slow | dataset.py | Use fast tokenizer (default in transformers 4.40) |

---

## Cycle-to-File Mapping

| Cycle | Files | Phase |
|-------|-------|-------|
| 1 | Repository_Scan_Report.md, Phase1.md, Phase_Dependency_Map.md | Planning |
| 2 | src/__init__.py, src/preprocess.py, src/dataset.py | Data Layer |
| 3 | src/baseline.py, src/utils.py, notebooks/00_setup_and_baseline.ipynb | Baseline |
| 4 | src/train.py, src/evaluate.py, src/predict.py | Transformer |
| 5 | notebooks/01_train_roberta.ipynb, notebooks/02_evaluate_and_submit.ipynb, outputs/.gitkeep | Execution |
