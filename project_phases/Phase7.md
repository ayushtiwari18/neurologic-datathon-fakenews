# Phase 7 — Inference & Submission Generation

## GOAL
Run the best trained model on `data/test.csv` and produce a clean, correctly formatted `outputs/submission.csv` ready for datathon judges.

## WHY THIS PHASE EXISTS
- **Risk reduced:** Many teams finish training but submit wrong-format CSVs or forget to apply preprocessing to test data — instant disqualification
- **Capability created:** A one-command inference pipeline that goes from raw test CSV to submission file
- **Competitive advantage:** Reproducible, documented inference with confidence scores signals engineering maturity to judges

## PREREQUISITES
- Phase 4 or Phase 6 complete (`models/roberta_fakenews_best/` exists)
- Phase 2 complete (`data/processed/test_clean.csv` exists)
- `src/evaluate.py` importable
- `config.py` loaded

## INPUTS
- `models/roberta_fakenews_best/` (or `models/roberta_fakenews/` if Phase 6 skipped)
- `data/processed/test_clean.csv` — columns: `combined_text`
- `data/test.csv` — original (for ID/index preservation)
- `config.py` (ID2LABEL, CONFIDENCE_THRESHOLD)

## TASKS
1. Load best model + tokenizer from `models/roberta_fakenews_best/`
2. Load `data/processed/test_clean.csv`
3. Preserve original row index / ID column from `data/test.csv`
4. Run batched inference on all test rows (batch_size=32)
5. Apply `config.CONFIDENCE_THRESHOLD = 0.70` — flag uncertain predictions
6. Map predicted integer labels back to string labels using `ID2LABEL`
7. Build submission DataFrame: columns = `[id, label, confidence]`
8. Save to `outputs/submission.csv`
9. Print: total rows, label distribution, uncertain count, sample 5 rows
10. Validate submission format matches expected contest format

## AI EXECUTION PROMPTS
- "Load the best saved model from models/roberta_fakenews_best/. Load data/processed/test_clean.csv. Run batched inference with batch_size=32 using torch.no_grad(). Collect predicted labels and softmax confidence for each row."
- "Map integer predictions back to string labels using ID2LABEL from config.py. Apply confidence threshold: if confidence < 0.70, flag as UNCERTAIN but still output the predicted label (do not leave blank)."
- "Build a DataFrame with columns: id (from original test.csv index), label (predicted string), confidence (float 0-1). Save to outputs/submission.csv. Print shape, value_counts of label column, and 5 sample rows."

## ALGORITHMS
- **Inference:** Batched `model.eval()` + `torch.no_grad()` + `F.softmax`
- **Label Mapping:** `ID2LABEL = {0: 'REAL', 1: 'FAKE'}` from config.py
- **Confidence Thresholding:** Flag rows below 0.70 for manual review (log only, do not exclude from submission)
- **Batch Size:** 32 for inference (2x train batch — no gradient memory needed)

## CODE SNIPPETS
```python
import torch, torch.nn.functional as F
import pandas as pd
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from config import ID2LABEL, CONFIDENCE_THRESHOLD

model = RobertaForSequenceClassification.from_pretrained('./models/roberta_fakenews_best')
tokenizer = RobertaTokenizer.from_pretrained('./models/roberta_fakenews_best')
model.eval().to(device)

test_df = pd.read_csv('data/processed/test_clean.csv')
texts = test_df['combined_text'].tolist()

preds, confs = [], []
for i in range(0, len(texts), 32):
    batch = texts[i:i+32]
    enc = tokenizer(batch, truncation=True, max_length=512,
                    padding=True, return_tensors='pt').to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    probs = F.softmax(logits, dim=-1)
    preds.extend(probs.argmax(dim=-1).cpu().tolist())
    confs.extend(probs.max(dim=-1).values.cpu().tolist())
```
```python
orig_test = pd.read_csv('data/test.csv')
submission = pd.DataFrame({
    'id':         orig_test.index,
    'label':      [ID2LABEL[p] for p in preds],
    'confidence': [round(c, 4) for c in confs]
})

uncertain = (submission['confidence'] < CONFIDENCE_THRESHOLD).sum()
print(f"Uncertain predictions: {uncertain} / {len(submission)}")
print(submission['label'].value_counts())
print(submission.head())

submission.to_csv('outputs/submission.csv', index=False)
print("Saved: outputs/submission.csv")
```

## OUTPUTS
- `src/predict.py` — importable inference + submission module
- `outputs/submission.csv` — columns: `id`, `label`, `confidence`
- Console: label distribution, uncertain count, 5 sample rows

## EXPECTED RESULTS
- `outputs/submission.csv` row count = `test.csv` row count (no rows dropped)
- Label distribution: reasonable REAL/FAKE split (not 100% one class)
- Uncertain predictions: < 5% of total rows
- File saved without error
- Inference completes in < 10 minutes

## VALIDATION CHECKS
- [ ] `outputs/submission.csv` exists
- [ ] Row count matches `data/test.csv` exactly
- [ ] Columns are exactly: `id`, `label`, `confidence`
- [ ] `label` column contains only `REAL` or `FAKE` (no integers, no nulls)
- [ ] No row has null confidence
- [ ] Uncertain count printed and < 5%

## FAILURE CONDITIONS
- Row count mismatch → test preprocessing dropped rows, re-check Phase 2
- All labels same class → label encoding flipped, re-check LABEL2ID
- Null values in label → ID2LABEL mapping failed, check prediction integers
- Memory error during inference → reduce inference batch_size to 16

## RECOVERY ACTIONS
- Row mismatch: re-run `src/preprocess.py` on test.csv with `dropna=False` (fill nulls instead of dropping)
- All-same-class: print `preds[:20]`, verify ID2LABEL, check model loaded correctly
- Memory error: reduce batch_size from 32 to 16
- Wrong columns: rename DataFrame columns before saving

## PERFORMANCE TARGETS
- Inference time: < 10 minutes for 10k test rows on T4
- Memory: < 8 GB GPU
- Submission file size: < 5 MB

## RISKS
- Test CSV has different column names than train → preprocess.py must handle both
- Original test IDs lost if index not preserved → always read orig test.csv for IDs
- Tokenizer from wrong checkpoint → always load tokenizer from same dir as model

## DELIVERABLES
- ✅ `src/predict.py`
- ✅ `outputs/submission.csv` (correct format, complete rows)
- ✅ Printed label distribution and uncertain count
- ✅ Submission validated: row count, columns, no nulls
