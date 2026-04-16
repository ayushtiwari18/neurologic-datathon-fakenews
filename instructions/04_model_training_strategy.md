# 🧠 Model Training Strategy

## Phase 1: Baseline (Do This First — Takes 2 Minutes)
Before touching transformers, build a fast baseline:
```python
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(train_df['combined'])
X_val_tfidf = vectorizer.transform(val_df['combined'])

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, train_df['label'])
preds = model.predict(X_val_tfidf)
print(f"Baseline Accuracy: {accuracy_score(val_df['label'], preds):.4f}")
```
- Expected baseline: ~91–93%
- This is your comparison point for the report

## Phase 2: RoBERTa Fine-Tuning

### Step-by-Step
```
1. Install: pip install transformers datasets torch accelerate
2. Load tokenizer: RobertaTokenizer.from_pretrained('roberta-base')
3. Tokenize dataset (combined = title + [SEP] + text)
4. Create PyTorch Dataset class
5. Load model: RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
6. Use HuggingFace Trainer API (simplest approach)
7. Train for 3 epochs
8. Evaluate and save
```

### HuggingFace Trainer Config
```python
from transformers import TrainingArguments, Trainer

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
    fp16=True,              # Essential for Colab T4 speed
    report_to='none',       # Disable wandb
    logging_steps=100,
)
```

## Phase 3: Innovation Add-ons (Do After Baseline Works)

### Confidence Thresholding
```python
import torch.nn.functional as F

def predict_with_confidence(text, threshold=0.70):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1)
    confidence = probs.max().item()
    pred = probs.argmax().item()
    
    if confidence < threshold:
        return "UNCERTAIN — needs human review"
    return "REAL" if pred == 1 else "FAKE", f"{confidence:.1%}"
```

### Error Analysis
```python
# Find misclassified examples
wrong = val_df[val_df['true_label'] != val_df['predicted']]
print(wrong[['combined', 'true_label', 'predicted', 'confidence']].head(10))
# Add this table to your README and notebook — judges love it
```

## Training Time Estimates
| Environment | Estimated Time |
|---|---|
| Google Colab Free T4 | 25–40 minutes |
| Kaggle P100 | 20–30 minutes |
| Local CPU only | 4–8 hours (avoid) |
| Colab Pro A100 | 8–12 minutes |

## Model Saving
```python
model.save_pretrained('./models/roberta_fakenews')
tokenizer.save_pretrained('./models/roberta_fakenews')
# Do NOT commit this folder to GitHub — too large
# Upload to HuggingFace Hub instead for sharing
```
