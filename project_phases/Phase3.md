# Phase 3 — Baseline Model (TF-IDF + Logistic Regression)

## GOAL
Train a fast, interpretable baseline classifier in under 3 minutes that establishes a performance floor and validates the data pipeline.

## WHY THIS PHASE EXISTS
- **Risk reduced:** If baseline fails, data pipeline is broken — catch this before investing 40 minutes in RoBERTa training
- **Capability created:** A working prediction pipeline and evaluation framework
- **Competitive advantage:** Judges require baseline comparison. Teams without baselines lose Methodology points (20% of score)

## PREREQUISITES
- Phase 2 complete
- `data/processed/train_clean.csv` exists
- `data/processed/val_clean.csv` exists
- `src/preprocess.py` importable

## INPUTS
- `data/processed/train_clean.csv`
- `data/processed/val_clean.csv`
- `config.py` (TEXT_COL, LABEL_COL, SEED)

## TASKS
1. Load `train_clean.csv` and `val_clean.csv`
2. Extract `combined_text` as features, `label` as target
3. Fit `TfidfVectorizer(max_features=50000, ngram_range=(1,2))` on train only
4. Transform train and val sets
5. Train `LogisticRegression(max_iter=1000, random_state=42)`
6. Predict on val set
7. Compute: Accuracy, Precision, Recall, F1 (weighted)
8. Print full `classification_report`
9. Plot and save confusion matrix to `outputs/baseline_confusion_matrix.png`
10. Save metrics to `outputs/baseline_metrics.json`
11. Save model to `outputs/baseline_tfidf_logreg.pkl`

## AI EXECUTION PROMPTS
- "Load train_clean.csv and val_clean.csv. Fit TF-IDF vectorizer with max_features=50000 and ngram_range=(1,2) on train only. NEVER fit on val."
- "Train LogisticRegression(max_iter=1000, random_state=42). Predict on val. Print accuracy_score, f1_score(weighted), and full classification_report."
- "Plot confusion matrix using seaborn heatmap. Save to outputs/baseline_confusion_matrix.png. Save accuracy and f1 to outputs/baseline_metrics.json."

## ALGORITHMS
- **Vectorizer:** TF-IDF, unigrams + bigrams, top 50k features
- **Classifier:** Logistic Regression, L2 regularization, max_iter=1000
- **Evaluation:** sklearn classification_report, confusion matrix

## CODE SNIPPETS
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train_df['combined_text'])
X_val   = vectorizer.transform(val_df['combined_text'])  # transform only

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train, train_df['label'])
preds = clf.predict(X_val)

print(f"Accuracy: {accuracy_score(val_df['label'], preds):.4f}")
print(f"F1:       {f1_score(val_df['label'], preds, average='weighted'):.4f}")
print(classification_report(val_df['label'], preds, target_names=['REAL','FAKE']))
```
```python
import json, joblib
metrics = {"accuracy": float(acc), "f1_weighted": float(f1)}
with open('outputs/baseline_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
joblib.dump(clf, 'outputs/baseline_tfidf_logreg.pkl')
```

## OUTPUTS
- `src/baseline.py` — importable baseline module
- `outputs/baseline_confusion_matrix.png`
- `outputs/baseline_metrics.json`
- `outputs/baseline_tfidf_logreg.pkl`
- Console: classification_report

## EXPECTED RESULTS
- Baseline Accuracy: **≥ 91%**
- Baseline F1 (weighted): **≥ 0.91**
- Training completes in < 3 minutes
- All output files saved successfully

## VALIDATION CHECKS
- [ ] `outputs/baseline_metrics.json` exists and accuracy ≥ 0.91
- [ ] `outputs/baseline_confusion_matrix.png` exists and non-zero size
- [ ] No data leakage: vectorizer fitted ONLY on train
- [ ] `classification_report` shows both REAL and FAKE class metrics

## FAILURE CONDITIONS
- Accuracy < 0.85 → data pipeline error, re-check preprocessing
- `fit_transform` called on val set → data leakage, restart
- Memory error on TF-IDF → reduce `max_features` to 30000

## RECOVERY ACTIONS
- If accuracy < 0.85: print 5 sample rows, check label encoding
- If memory error: `max_features=30000`, `ngram_range=(1,1)`
- If fit on val detected: delete all outputs, rerun from Phase 2

## PERFORMANCE TARGETS
- Training time < 3 minutes
- Memory < 2 GB
- Accuracy ≥ 91%

## RISKS
- Data leakage if vectorizer fitted on full dataset
- Label encoding mismatch (0=REAL vs 0=FAKE)
- Class imbalance reducing minority-class recall

## DELIVERABLES
- ✅ `src/baseline.py`
- ✅ `outputs/baseline_metrics.json` (accuracy ≥ 0.91)
- ✅ `outputs/baseline_confusion_matrix.png`
- ✅ Printed classification report
- ✅ Baseline established as comparison point for RoBERTa
