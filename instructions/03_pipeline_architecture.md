# 🔧 Pipeline & Architecture

## Full Data-to-Prediction Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT LAYER                          │
│  Raw CSV (title + text columns, label column)           │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                 PREPROCESSING LAYER                     │
│  1. Drop null rows                                      │
│  2. Lowercase all text                                  │
│  3. Strip HTML tags                                     │
│  4. Remove URLs, emails, special characters             │
│  5. Combine: combined = title + " [SEP] " + text        │
│  6. Encode labels: REAL=1, FAKE=0                       │
│  7. Train/Val split (80/20, stratified)                 │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                 TOKENIZATION LAYER                      │
│  RobertaTokenizer.from_pretrained('roberta-base')       │
│  max_length=512, padding=True, truncation=True          │
│  Output: input_ids, attention_mask tensors              │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                   MODEL LAYER                           │
│  RobertaForSequenceClassification (num_labels=2)        │
│  Loaded from: roberta-base (HuggingFace Hub)            │
│  Fine-tuned on our dataset for 3 epochs                 │
│  Optimizer: AdamW, LR: 2e-5                             │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                 EVALUATION LAYER                        │
│  Metrics: Accuracy, Precision, Recall, F1               │
│  Confusion Matrix (saved as PNG)                        │
│  Error Analysis (top 10 misclassified samples)          │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                  OUTPUT LAYER                           │
│  predictions.csv (article, true_label, pred, confidence)│
│  Saved model weights in models/roberta_fakenews/        │
│  Optional: Gradio/FastAPI demo app                      │
└─────────────────────────────────────────────────────────┘
```

## Model Architecture Detail
- **Base Model:** `roberta-base` (125M parameters)
- **Added Head:** Linear classifier (768 → 2 classes)
- **Loss Function:** CrossEntropyLoss
- **Batch Size:** 16 (Colab T4), 8 (lower GPU)
- **Epochs:** 3 (sufficient for convergence on ~40k samples)
- **Scheduler:** Linear warmup (10% of steps)

## Why RoBERTa over BERT?
| Feature | BERT | RoBERTa |
|---|---|---|
| Training data | 16GB | 160GB |
| Next Sentence Prediction | Yes (weakens performance) | Removed |
| Dynamic Masking | No | Yes |
| Typical NLP accuracy | ~95% | ~98%+ |
