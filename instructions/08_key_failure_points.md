# ⚠️ Key Points of Failure — Read Before Starting

## 🔴 Critical Failures (Will Break Everything)

### 1. Colab Session Disconnects Mid-Training
**Problem:** Training takes 30 min. If Colab disconnects, you lose everything.
**Fix:**
- Save checkpoint every epoch: `save_strategy='epoch'` in TrainingArguments
- Mount Google Drive and set `output_dir` to Drive path
- Use `load_best_model_at_end=True` to auto-restore best checkpoint

### 2. Running Out of GPU Memory (CUDA OOM Error)
**Problem:** `RuntimeError: CUDA out of memory`
**Fix:**
```python
# Reduce batch size
per_device_train_batch_size=8   # down from 16
# Enable gradient checkpointing
model.gradient_checkpointing_enable()
# Use fp16
fp16=True  # in TrainingArguments
```

### 3. Label Encoding Mismatch
**Problem:** Model predicts 0/1 but labels are 'REAL'/'FAKE' strings → all predictions wrong
**Fix:** Always run `assert df['label'].dtype in [int, float]` before training

### 4. Dataset Column Name Mismatch
**Problem:** Code expects column `text` but dataset has `content` or `body`
**Fix:** Always check `df.columns.tolist()` first and update column names at top of notebook:
```python
TEXT_COL = 'text'    # ← change this if needed
TITLE_COL = 'title'  # ← change this if needed
LABEL_COL = 'label'  # ← change this if needed
```

---

## 🟡 Medium Failures (Will Hurt Score)

### 5. Not Using Stratified Split
**Problem:** Random split may put 90% FAKE in val — metrics look artificially high/low
**Fix:** Always use `stratify=df['label']` in train_test_split

### 6. Data Leakage
**Problem:** Preprocessing (e.g., TF-IDF fit) done on full dataset before split → inflated accuracy
**Fix:** Fit vectorizer ONLY on training data, transform val separately
```python
vectorizer.fit(train_df['combined'])          # fit on train only
X_train = vectorizer.transform(train_df['combined'])
X_val = vectorizer.transform(val_df['combined'])  # transform only
```

### 7. Reporting Accuracy on Training Set
**Problem:** Training accuracy is always high — means nothing
**Fix:** Only report metrics computed on `val_df` (held-out data)

### 8. Forgetting fp16=True
**Problem:** Training takes 2x longer without mixed precision
**Fix:** Always set `fp16=True` in TrainingArguments when using GPU

---

## 🟢 Common Mistakes (Will Slow You Down)

### 9. Not Setting Random Seeds
**Problem:** Results differ between runs — can't reproduce your best accuracy
**Fix:** Set seeds at top of every notebook (see 05_colab_setup.md)

### 10. Large Model Files in Git
**Problem:** `pytorch_model.bin` is 500MB+ — GitHub will reject
**Fix:** Add to `.gitignore`:
```
models/
*.bin
*.pt
*.pth
__pycache__/
.ipynb_checkpoints/
data/raw/
```

### 11. Notebook Not Running Top-to-Bottom
**Problem:** Cells run out of order leave hidden state — other team members can't reproduce
**Fix:** Before submitting, do **Runtime → Restart and run all** and verify it completes

### 12. Missing Screenshots
**Problem:** README has no visuals — judges can't quickly validate your results
**Fix:** At minimum, screenshot: (1) training loss curve, (2) confusion matrix, (3) final accuracy print
