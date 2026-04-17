# Phase 5 — Evaluation & Error Analysis

## GOAL
Generate a comprehensive, judge-ready evaluation report: accuracy, F1, confusion matrix, per-class metrics, and a curated error analysis table of the top 10 misclassified samples.

## WHY THIS PHASE EXISTS
- **Risk reduced:** Training metrics alone are insufficient — judges require evidence of methodology rigor (20% of score)
- **Capability created:** Full evaluation suite that quantifies exactly where and why the model fails
- **Competitive advantage:** Error analysis table in README is a visible differentiator — most teams skip it. It directly scores Methodology + Innovation points.

## PREREQUISITES
- Phase 4 complete
- `models/roberta_fakenews/` exists with saved weights
- `data/processed/val_clean.csv` exists
- `src/train.py` and `src/dataset.py` importable

## INPUTS
- `models/roberta_fakenews/` (saved model + tokenizer)
- `data/processed/val_clean.csv`
- `config.py` (ID2LABEL, CONFIDENCE_THRESHOLD)

## TASKS
1. Load saved model and tokenizer from `models/roberta_fakenews/`
2. Run inference on full val set
3. Collect: `true_label`, `predicted_label`, `confidence`, `combined_text`
4. Compute: Accuracy, Precision, Recall, F1 (macro + weighted)
5. Print full `classification_report` (per-class breakdown)
6. Plot confusion matrix (seaborn heatmap, normalized) → save to `outputs/roberta_confusion_matrix.png`
7. Identify top 10 misclassified samples (lowest confidence on wrong predictions)
8. Save error analysis table to `outputs/error_analysis.csv`
9. Save all metrics to `outputs/roberta_eval_metrics.json`
10. Print side-by-side comparison: Baseline vs RoBERTa metrics

## AI EXECUTION PROMPTS
- "Load the saved RoBERTa model from models/roberta_fakenews/. Run inference on the full val_clean.csv. Collect true labels, predicted labels, and softmax confidence for each sample."
- "Compute accuracy, precision, recall, F1 (macro and weighted) using sklearn. Print classification_report with target_names=['REAL','FAKE']. Plot normalized confusion matrix and save to outputs/roberta_confusion_matrix.png."
- "Find the top 10 misclassified rows (predicted != true_label), sorted by confidence descending. Save as outputs/error_analysis.csv with columns: combined_text, true_label, predicted_label, confidence."
- "Print a comparison table: Baseline accuracy vs RoBERTa accuracy, Baseline F1 vs RoBERTa F1. Save to outputs/model_comparison.json."

## ALGORITHMS
- **Inference:** `model.eval()` + `torch.no_grad()` + `F.softmax(logits, dim=-1)`
- **Metrics:** `sklearn.metrics` — accuracy_score, classification_report, confusion_matrix
- **Confidence Thresholding:** `config.CONFIDENCE_THRESHOLD = 0.70` — samples below flagged as UNCERTAIN
- **Error Analysis:** Sort misclassified by confidence, select top 10

## CODE SNIPPETS
```python
import torch, torch.nn.functional as F
from transformers import RobertaForSequenceClassification, RobertaTokenizer

model = RobertaForSequenceClassification.from_pretrained('./models/roberta_fakenews')
tokenizer = RobertaTokenizer.from_pretrained('./models/roberta_fakenews')
model.eval().to(device)

def predict_batch(texts, batch_size=32):
    all_preds, all_confs = [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, truncation=True, max_length=512,
                        padding=True, return_tensors='pt').to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = F.softmax(logits, dim=-1)
        all_preds.extend(probs.argmax(dim=-1).cpu().tolist())
        all_confs.extend(probs.max(dim=-1).values.cpu().tolist())
    return all_preds, all_confs
```
```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns, matplotlib.pyplot as plt

preds, confs = predict_batch(val_df['combined_text'].tolist())
print(classification_report(val_df['label'], preds, target_names=['REAL','FAKE']))

cm = confusion_matrix(val_df['label'], preds, normalize='true')
sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=['REAL','FAKE'], yticklabels=['REAL','FAKE'])
plt.title('RoBERTa — Normalized Confusion Matrix')
plt.savefig('outputs/roberta_confusion_matrix.png', dpi=150, bbox_inches='tight')
```

## OUTPUTS
- `src/evaluate.py` — importable evaluation module
- `outputs/roberta_confusion_matrix.png`
- `outputs/roberta_eval_metrics.json`
- `outputs/error_analysis.csv`
- `outputs/model_comparison.json` — Baseline vs RoBERTa
- Console: full classification_report

## EXPECTED RESULTS
- RoBERTa Accuracy: ≥ 97%
- RoBERTa F1 (weighted): ≥ 0.97
- Confusion matrix saved successfully
- Error analysis table: exactly 10 rows
- Improvement over baseline: ≥ +5% accuracy

## VALIDATION CHECKS
- [ ] `outputs/roberta_eval_metrics.json` accuracy ≥ 0.97
- [ ] `outputs/roberta_confusion_matrix.png` exists, non-zero size
- [ ] `outputs/error_analysis.csv` has exactly 10 rows
- [ ] `outputs/model_comparison.json` has both baseline and roberta keys
- [ ] No data leakage — val set was never seen during training

## FAILURE CONDITIONS
- Model accuracy on val < 0.93 → re-check Phase 4, may need another epoch
- Confusion matrix blank → matplotlib backend issue in Colab (add `%matplotlib inline`)
- Error analysis empty → all predictions correct (unlikely; check label alignment)

## RECOVERY ACTIONS
- If accuracy < 0.93: reload best checkpoint from `models/roberta_fakenews/` (not last epoch)
- If plot fails: use `plt.switch_backend('Agg')` before plotting
- If error analysis empty: lower confidence threshold to 0.60 to find uncertain samples

## PERFORMANCE TARGETS
- Inference on full val set: < 5 minutes
- All output files generated: < 8 minutes total

## RISKS
- Loading wrong checkpoint (last vs best) → always use `load_best_model_at_end=True`
- Confidence values not calibrated → for judging, raw softmax is sufficient
- Memory leak during batch inference → use `torch.no_grad()` always

## DELIVERABLES
- ✅ `src/evaluate.py`
- ✅ `outputs/roberta_eval_metrics.json` (accuracy ≥ 0.97)
- ✅ `outputs/roberta_confusion_matrix.png`
- ✅ `outputs/error_analysis.csv` (10 rows)
- ✅ `outputs/model_comparison.json` (Baseline vs RoBERTa)
- ✅ Printed classification report ready for README inclusion
