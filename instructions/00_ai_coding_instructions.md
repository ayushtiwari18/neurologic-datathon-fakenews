# 🤖 AI Coding Instructions — Rules for AI Assistants Working on This Repo

> This file is for AI coding assistants (GitHub Copilot, Cursor, Claude, ChatGPT, etc.)
> Read this BEFORE generating any code for this project.

---

## 🛑 NEVER Do These

### 1. Never use `bert-base-uncased`
- This project uses **`roberta-base`** exclusively
- BERT and RoBERTa have different tokenizers, different configs, different separator tokens
- Substituting BERT will silently produce wrong results

### 2. Never pass `token_type_ids` to RoBERTa
```python
# ❌ WRONG — copied from a BERT example
inputs = tokenizer(text, return_tensors='pt')
outputs = model(input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                token_type_ids=inputs['token_type_ids'])  # ❌ CRASHES or silently fails

# ✅ CORRECT — RoBERTa only needs input_ids + attention_mask
outputs = model(input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'])
```
RoBERTa has `type_vocab_size=1` — `token_type_ids` is not in its config.

### 3. Never hardcode column names
Always define column names as variables at the top of every script:
```python
# ✅ CORRECT — define once, use everywhere
TEXT_COL  = 'text'
TITLE_COL = 'title'
LABEL_COL = 'label'

# ❌ WRONG — hardcoded everywhere
df['text'].apply(...)   # if column is 'content', this crashes silently
```

### 4. Never commit model weights
Files matching these patterns must be in `.gitignore`:
```
models/
*.bin
*.pt
*.pth
*.safetensors
```
Model folders are 400MB–1GB. GitHub will reject pushes above 100MB.
Upload models to HuggingFace Hub instead.

### 5. Never run inference without `torch.no_grad()`
```python
# ❌ WRONG — builds computation graph, wastes memory, slows inference 3x
outputs = model(**inputs)

# ✅ CORRECT
with torch.no_grad():
    outputs = model(**inputs)
```

### 6. Never ignore device placement
```python
# ❌ WRONG — model is on GPU, inputs are on CPU → RuntimeError
inputs = tokenizer(text, return_tensors='pt')
outputs = model(**inputs)

# ✅ CORRECT
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.no_grad():
    outputs = model(**inputs)
```

---

## ✅ Always Do These

- Always define `compute_metrics` and pass it to `Trainer`
- Always use `DataCollatorWithPadding` in `Trainer`
- Always use `eval_dataset=val_dataset` in `Trainer` — or evaluation never runs
- Always use `stratify=df['label']` in `train_test_split`
- Always fit TF-IDF/scalers on `train_df` only, transform `val_df` separately
- Always report metrics on the **validation set**, never on the training set
- Always set `fp16=True` in `TrainingArguments` when using GPU
- Always set `TOKENIZERS_PARALLELISM=false` before importing tokenizers in Colab

---

## 📌 Project-Specific Constants

```python
MODEL_NAME = 'roberta-base'      # Never change to bert-*
SEED       = 42                   # Use for all random states
MAX_LEN    = 512                  # RoBERTa token limit
BATCH_SIZE = 16                   # For Colab T4; reduce to 8 if OOM
EPOCHS     = 3                    # Sufficient for convergence
LR         = 2e-5                 # Standard for RoBERTa fine-tuning
```
