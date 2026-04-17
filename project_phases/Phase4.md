# Phase 4 — RoBERTa Fine-Tuning

## GOAL
Fine-tune `roberta-base` on the cleaned fake news dataset for 3 epochs using HuggingFace Trainer API, achieving ≥ 97% validation accuracy.

## WHY THIS PHASE EXISTS
- **Risk reduced:** Transformer training is the highest-risk step — GPU OOM, tokenizer bugs, wrong eval setup all silently kill accuracy
- **Capability created:** A production-quality fine-tuned model that far outperforms the TF-IDF baseline
- **Competitive advantage:** RoBERTa trained on title+body fusion with confidence thresholding targets the 40% accuracy judging weight and the 15% innovation weight simultaneously

## PREREQUISITES
- Phase 1 complete (GPU confirmed, environment stable)
- Phase 2 complete (`data/processed/train_clean.csv`, `val_clean.csv` exist)
- Phase 3 complete (baseline metrics confirmed ≥ 91%)
- `src/preprocess.py` importable
- `src/dataset.py` exists (PyTorch Dataset class)

## INPUTS
- `data/processed/train_clean.csv`
- `data/processed/val_clean.csv`
- `config.py` (TRANSFORMER, MAX_LEN, BATCH_SIZE, EPOCHS, LEARNING_RATE, WARMUP_STEPS, WEIGHT_DECAY, SEED, NUM_LABELS, ID2LABEL, LABEL2ID)

## TASKS
1. Load tokenizer: `RobertaTokenizer.from_pretrained('roberta-base')`
2. Create `FakeNewsDataset` PyTorch class (tokenize on-the-fly, return `input_ids`, `attention_mask`, `labels`)
3. **NEVER pass `token_type_ids`** — RoBERTa has `type_vocab_size=1`, it will error or silently break
4. Load model: `RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID)`
5. Define `compute_metrics` function returning accuracy + weighted F1
6. Create `DataCollatorWithPadding(tokenizer=tokenizer)`
7. Configure `TrainingArguments` (see snippets)
8. Instantiate `Trainer` with train_dataset, eval_dataset, data_collator, compute_metrics
9. Call `trainer.train()`
10. Call `trainer.evaluate()` — print final val metrics
11. Save model + tokenizer to `models/roberta_fakenews/`
12. Save training metrics to `outputs/roberta_train_metrics.json`

## AI EXECUTION PROMPTS
- "Load roberta-base tokenizer. Create a PyTorch Dataset class that tokenizes combined_text with max_length=512, padding=True, truncation=True. Do NOT include token_type_ids in the returned dict."
- "Load RobertaForSequenceClassification with num_labels=2. Set fp16=True in TrainingArguments. Pass eval_dataset to Trainer. Pass data_collator=DataCollatorWithPadding. Pass compute_metrics. Call trainer.train()."
- "After training, call trainer.evaluate(). Save model with model.save_pretrained('./models/roberta_fakenews'). Save tokenizer with tokenizer.save_pretrained('./models/roberta_fakenews'). Save final accuracy and f1 to outputs/roberta_train_metrics.json."

## ALGORITHMS
- **Model:** RoBERTa-base (125M params), HuggingFace `transformers==4.40.0`
- **Optimizer:** AdamW (built into Trainer)
- **Scheduler:** Linear warmup over first 500 steps
- **Loss:** CrossEntropyLoss (default for SequenceClassification)
- **Mixed Precision:** FP16 (essential for Colab T4 speed)

## CODE SNIPPETS
```python
from torch.utils.data import Dataset
from transformers import RobertaTokenizer

class FakeNewsDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512):
        self.texts  = df['combined_text'].tolist()
        self.labels = df['label'].tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True, max_length=self.max_len,
            padding=False  # DataCollatorWithPadding handles this
        )
        # NO token_type_ids — RoBERTa does not use them
        return {
            'input_ids':      enc['input_ids'],
            'attention_mask': enc['attention_mask'],
            'labels':         self.labels[idx]
        }
```
```python
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1':       f1_score(labels, preds, average='weighted')
    }
```
```python
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding

training_args = TrainingArguments(
    output_dir='./models/roberta_fakenews',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    learning_rate=2e-5,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    fp16=True,
    report_to='none',
    logging_steps=100,
    seed=42,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,          # REQUIRED
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),  # REQUIRED
    compute_metrics=compute_metrics,   # REQUIRED
)
trainer.train()
```

## OUTPUTS
- `src/dataset.py` — FakeNewsDataset class
- `src/train.py` — full training script
- `models/roberta_fakenews/` — model weights + tokenizer
- `outputs/roberta_train_metrics.json` — epoch-wise accuracy + F1

## EXPECTED RESULTS
- Epoch 1 val accuracy: ≥ 95%
- Epoch 2 val accuracy: ≥ 97%
- Epoch 3 val accuracy: ≥ 97.5%
- Training time on Colab T4: 25–40 minutes
- No OOM errors at batch_size=16

## VALIDATION CHECKS
- [ ] `trainer.train()` completes without crash
- [ ] `trainer.evaluate()` prints accuracy ≥ 0.97
- [ ] `models/roberta_fakenews/config.json` exists after save
- [ ] `outputs/roberta_train_metrics.json` exists with accuracy key
- [ ] No `token_type_ids` in dataset `__getitem__` output

## FAILURE CONDITIONS
- CUDA OOM → reduce `per_device_train_batch_size` to 8
- Accuracy stuck at ~50% → label encoding flipped, check LABEL2ID
- Training crashes at step 0 → tokenizer output contains `token_type_ids`, remove it
- `compute_metrics` not called → `eval_dataset` not passed to Trainer

## RECOVERY ACTIONS
- OOM: set `per_device_train_batch_size=8`, `gradient_accumulation_steps=2`
- Label flip: print `df['label'].value_counts()`, verify 0=REAL, 1=FAKE matches config
- token_type_ids error: explicitly pop from tokenizer output in `__getitem__`
- No eval metrics: confirm `eval_dataset=val_dataset` in Trainer constructor

## PERFORMANCE TARGETS
- Training time: < 40 minutes on Colab T4
- GPU memory: < 14 GB (leaves buffer on 16 GB T4)
- Val accuracy: ≥ 97%
- Val F1 weighted: ≥ 0.97

## RISKS
- GPU OOM at batch_size=16 if articles are very long
- Colab session disconnect mid-training (save checkpoints each epoch)
- HuggingFace Hub download slow → pre-download tokenizer in Phase 1
- FP16 NaN loss on some hardware → disable fp16 if loss becomes NaN

## DELIVERABLES
- ✅ `src/dataset.py`
- ✅ `src/train.py`
- ✅ `models/roberta_fakenews/` (model + tokenizer saved)
- ✅ `outputs/roberta_train_metrics.json` (accuracy ≥ 0.97)
- ✅ Training log printed with per-epoch accuracy
