# Risk Management Plan
**Project:** NeuroLogic '26 Datathon — Fake News Detection  
**Version:** 1.0 | **Updated:** 2026-04-17

---

## Risk Severity Scale
| Level | Label | Action |
|-------|-------|--------|
| P1 | 🔴 CRITICAL | Stop. Fix before proceeding. |
| P2 | 🟡 HIGH | Fix within current phase. |
| P3 | 🟢 MEDIUM | Log and monitor. Workaround available. |
| P4 | ⚪ LOW | Document. No immediate action needed. |

---

## 1. Technical Risks

### T1 — GPU Unavailable / CUDA Error
- **Probability:** Medium (15%)
- **Impact:** 🔴 CRITICAL — RoBERTa training is 4–8x slower on CPU (4–8 hours vs 40 min)
- **Detection:** `torch.cuda.is_available()` returns False at Phase 1
- **Mitigation:**
  - Always start by switching Colab to T4 GPU runtime before any code
  - Pre-check: Runtime → Change runtime type → T4 GPU
  - If T4 unavailable: use Kaggle P100 (free, 30h/week quota)
- **Recovery:** Switch runtime, re-run Phase 1 setup cell

### T2 — GPU Out-of-Memory (OOM)
- **Probability:** Medium (25%) at batch_size=16 with long articles
- **Impact:** 🔴 CRITICAL — training crashes mid-epoch, checkpoint may be lost
- **Detection:** `RuntimeError: CUDA out of memory`
- **Mitigation:**
  - Default batch_size=16 in config.py is safe for most datasets
  - Enable `fp16=True` in TrainingArguments (halves memory use)
  - Use `DataCollatorWithPadding` (no fixed-length padding waste)
- **Recovery:**
  ```python
  per_device_train_batch_size = 8
  gradient_accumulation_steps = 2  # simulates batch_size=16
  ```

### T3 — token_type_ids Passed to RoBERTa
- **Probability:** High (40%) if BERT code is copy-pasted
- **Impact:** 🔴 CRITICAL — silent wrong results or runtime crash
- **Detection:** Model accuracy stuck at ~50%, or `ValueError` on model forward pass
- **Mitigation:** Dataset `__getitem__` must NEVER include `token_type_ids` key
- **Recovery:** Remove `token_type_ids` from tokenizer output dict in Dataset class

### T4 — Label Encoding Flip
- **Probability:** Medium (20%)
- **Impact:** 🔴 CRITICAL — model predicts REAL as FAKE and vice versa
- **Detection:** Confusion matrix is inverted (high false positive rate on wrong class)
- **Mitigation:** Always use `LABEL2ID` and `ID2LABEL` from `config.py`. Never hardcode 0/1.
- **Recovery:** Print `df['label'].value_counts()`. Verify 0=REAL, 1=FAKE. Remap if needed.

### T5 — Data Leakage (Vectorizer Fitted on Val)
- **Probability:** Low (10%) but catastrophic
- **Impact:** 🔴 CRITICAL — inflated baseline metrics, invalid comparison
- **Detection:** Baseline accuracy > 99% (suspiciously high)
- **Mitigation:** Always use `fit_transform` on train only, `transform` on val/test
- **Recovery:** Delete all baseline outputs, re-run Phase 3 from scratch

### T6 — compute_metrics Not Passed to Trainer
- **Probability:** Medium (30%) if setup rushed
- **Impact:** 🟡 HIGH — Trainer shows no accuracy/F1 metrics during training
- **Detection:** Training log shows only `loss`, no `eval_accuracy`
- **Mitigation:** Always pass `compute_metrics=compute_metrics` to Trainer constructor
- **Recovery:** Stop training, add `compute_metrics`, restart

### T7 — eval_dataset Not Passed to Trainer
- **Probability:** Medium (25%)
- **Impact:** 🟡 HIGH — evaluation never runs, best model not selected
- **Detection:** No `eval_` keys in training log
- **Mitigation:** Always pass `eval_dataset=val_dataset` to Trainer
- **Recovery:** Restart training with `eval_dataset` set

### T8 — FP16 NaN Loss
- **Probability:** Low (8%)
- **Impact:** 🟢 MEDIUM — training diverges, loss becomes NaN
- **Detection:** Loss prints `nan` after first few steps
- **Mitigation:** `fp16=True` is generally safe on T4; use `fp16_opt_level='O1'`
- **Recovery:** Set `fp16=False`, restart training (will be ~30% slower)

---

## 2. Data Risks

### D1 — Missing Required Columns
- **Probability:** Medium (20%) — contest CSVs often have different column names
- **Impact:** 🔴 CRITICAL — preprocessing crashes immediately
- **Detection:** `KeyError` on `df['title']` or `df['text']`
- **Mitigation:** Print `df.columns.tolist()` as first step in Phase 2
- **Recovery:** Use `df.rename(columns={'headline': 'title', 'body': 'text'})` to remap

### D2 — High Null Rate in Text Column
- **Probability:** Low-Medium (15%)
- **Impact:** 🟡 HIGH — dropping too many rows reduces training data
- **Detection:** `df.isnull().sum()` shows > 5% nulls in `text`
- **Mitigation:** Fill null `title` with empty string; drop only if `text` is null
- **Recovery:** `df['title'] = df['title'].fillna('')`; `df = df.dropna(subset=['text'])`

### D3 — Severe Class Imbalance
- **Probability:** Low (10%)
- **Impact:** 🟢 MEDIUM — model biased toward majority class
- **Detection:** `df['label'].value_counts()` shows > 70/30 split
- **Mitigation:** Use `compute_class_weight` + weighted CrossEntropyLoss (Phase 6)
- **Recovery:** Oversample minority class with `resample` from sklearn

### D4 — Test CSV Missing Title Column
- **Probability:** Medium (20%)
- **Impact:** 🟡 HIGH — `combined_text` cannot be created for test set
- **Detection:** `KeyError` in `preprocess.py` on test data
- **Mitigation:** Add column existence check in `preprocess.py`; fall back to body-only if title missing
- **Recovery:** `df['title'] = df.get('title', '')` — use empty string as title

---

## 3. Operational Risks

### O1 — Colab Session Disconnects Mid-Training
- **Probability:** High (35%) for sessions > 60 minutes
- **Impact:** 🟡 HIGH — training lost if no checkpoints saved
- **Detection:** Session expires, kernel restarts
- **Mitigation:** `save_strategy='epoch'` in TrainingArguments saves after every epoch
- **Recovery:** Reload from last checkpoint: `Trainer.train(resume_from_checkpoint='./models/roberta_fakenews/checkpoint-XXX')`

### O2 — HuggingFace Hub Download Timeout
- **Probability:** Low-Medium (15%) on slow Colab connections
- **Impact:** 🟢 MEDIUM — model download stalls or fails
- **Detection:** Download hangs > 5 minutes
- **Mitigation:** Pre-download `roberta-base` in Phase 1 before training starts
- **Recovery:** Re-run download cell; use `TRANSFORMERS_CACHE` env var to persist cache

### O3 — Wrong Checkpoint Loaded for Submission
- **Probability:** Low (10%)
- **Impact:** 🟡 HIGH — submitting predictions from epoch 1 model instead of best model
- **Detection:** Submission accuracy lower than expected
- **Mitigation:** Always set `load_best_model_at_end=True` + `metric_for_best_model='accuracy'`
- **Recovery:** Manually load the checkpoint with highest eval_accuracy from `hyperparam_log.csv`

---

## 4. Time Risks

### TM1 — Phase 4 Training Exceeds 50 Minutes
- **Probability:** Medium (20%) on slow T4 or large dataset
- **Impact:** 🟡 HIGH — compresses time for Phase 6 optimization
- **Mitigation:** Start training immediately after Phase 3 baseline confirmed
- **Recovery:** Skip Phase 6 LR sweep; submit Phase 4 model directly

### TM2 — Deadline Reached Before Submission File Generated
- **Probability:** Low (5%) if timeline followed
- **Impact:** 🔴 CRITICAL — no submission = disqualification
- **Mitigation:** `outputs/submission.csv` must exist by T-30 min
- **Recovery:** If Phase 6/8 not done, skip them. Phase 7 (submission) is always mandatory.

### TM3 — Optimization Phase Takes Too Long
- **Probability:** Medium (25%)
- **Impact:** 🟢 MEDIUM — less time for documentation
- **Mitigation:** Hard time-box Phase 6 to 32 minutes; skip ensemble if behind schedule
- **Recovery:** Stop LR sweep after 2 values if time is short

---

## 5. Hardware Risks

### H1 — Colab Free Tier GPU Quota Exhausted
- **Probability:** Low-Medium (15%) if multiple sessions run same day
- **Impact:** 🟡 HIGH — no GPU available, training falls back to CPU
- **Mitigation:** Use Kaggle (30h/week free P100) as backup
- **Recovery:** Submit notebook to Kaggle, run there with GPU enabled

### H2 — Colab RAM Exceeded (System RAM, not GPU)
- **Probability:** Low (8%) on large datasets with TF-IDF
- **Impact:** 🟢 MEDIUM — kernel crashes during TF-IDF vectorization
- **Mitigation:** `max_features=50000` is safe for most datasets; reduce to 30000 if needed
- **Recovery:** `del vectorizer; gc.collect()` after saving baseline model

---

## Risk Priority Matrix

| Risk | Probability | Impact | Priority |
|------|------------|--------|----------|
| T1 GPU unavailable | Medium | Critical | 🔴 P1 |
| T2 OOM | Medium | Critical | 🔴 P1 |
| T3 token_type_ids | High | Critical | 🔴 P1 |
| T4 Label flip | Medium | Critical | 🔴 P1 |
| T5 Data leakage | Low | Critical | 🔴 P1 |
| TM2 No submission | Low | Critical | 🔴 P1 |
| D1 Missing columns | Medium | Critical | 🔴 P1 |
| O1 Colab disconnect | High | High | 🟡 P2 |
| T6 No compute_metrics | Medium | High | 🟡 P2 |
| D2 High nulls | Low-Med | High | 🟡 P2 |
| D3 Class imbalance | Low | Medium | 🟢 P3 |
| T8 FP16 NaN | Low | Medium | 🟢 P3 |
| O2 Hub timeout | Low-Med | Medium | 🟢 P3 |
