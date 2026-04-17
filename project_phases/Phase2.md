# Phase 2 — Data Pipeline Setup

## GOAL
Build a robust, validated data preprocessing pipeline that converts raw CSV inputs into clean, tokenized, split-ready tensors for model training.

## WHY THIS PHASE EXISTS
- **Risk reduced:** Dirty data (nulls, HTML, encoding errors) silently degrades model accuracy by 3–5%
- **Capability created:** Reproducible data pipeline that runs identically every time
- **Competitive advantage:** Title+Body text fusion gives the model richer signal than body-only approaches used by most teams

## PREREQUISITES
- Phase 1 complete (environment stable, dirs exist)
- `data/train.csv` present with columns: `title`, `text`, `label`
- `data/test.csv` present with columns: `title`, `text`
- `config.py` loaded
- `src/__init__.py` exists

## INPUTS
- `data/train.csv`
- `data/test.csv`
- `config.py` (TEXT_COL, LABEL_COL, TITLE_COL, BODY_COL, VAL_SPLIT, SEED)

## TASKS
1. Load `train.csv` and `test.csv` with pandas
2. Validate required columns exist (`title`, `text`, `label`)
3. Log and drop rows where `title` or `text` is null
4. Lowercase all text
5. Strip HTML tags using BeautifulSoup
6. Remove URLs (`http://...`), emails, and special characters
7. Create `combined_text` column: `title + " [SEP] " + text`
8. Encode labels: use `config.LABEL2ID` mapping
9. Stratified train/validation split (80/20, seed=42)
10. Save `data/processed/train_clean.csv` and `data/processed/val_clean.csv`
11. Save `data/processed/test_clean.csv`
12. Print shape, label distribution, null count after each step

## AI EXECUTION PROMPTS
- "Load data/train.csv. Check for columns title, text, label. Print shape, null counts per column, and label value counts."
- "Clean the text: lowercase, strip HTML with BeautifulSoup, remove URLs and special chars. Create combined_text = title + ' [SEP] ' + text. Do NOT replace [SEP] with </s> — RoBERTa handles its own separator token."
- "Perform stratified 80/20 train/val split with random_state=42. Save train_clean.csv and val_clean.csv to data/processed/. Print label distribution of each split."

## ALGORITHMS
- **Text Cleaning:** Regex + BeautifulSoup
- **Text Fusion:** Concatenation with plain-text `[SEP]` marker
- **Label Encoding:** Dict map from config.LABEL2ID
- **Splitting:** `sklearn.model_selection.train_test_split` with `stratify=labels`

## CODE SNIPPETS
```python
from bs4 import BeautifulSoup
import re

def clean_text(text):
    text = BeautifulSoup(str(text), 'html.parser').get_text()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return text.lower().strip()
```
```python
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(
    df, test_size=0.20, stratify=df['label'], random_state=42
)
```

## OUTPUTS
- `data/processed/train_clean.csv` — columns: `combined_text`, `label`
- `data/processed/val_clean.csv` — columns: `combined_text`, `label`
- `data/processed/test_clean.csv` — columns: `combined_text`
- `src/preprocess.py` — importable preprocessing module

## EXPECTED RESULTS
- Zero null rows in processed files
- Label distribution in train and val within ±2% of original
- `combined_text` column present in all files
- Files saved successfully to `data/processed/`

## VALIDATION CHECKS
- [ ] `train_clean.csv` shape printed and non-zero
- [ ] `val_clean.csv` label distribution matches stratification
- [ ] No null values in `combined_text`
- [ ] `combined_text` contains ` [SEP] ` substring (spot-check 5 rows)
- [ ] Labels are integers (0 or 1), not strings

## FAILURE CONDITIONS
- CSV missing required columns → STOP, report column names found
- >10% null rows → log warning, investigate data quality
- Label distribution skew >5% after split → re-run with different seed

## RECOVERY ACTIONS
- If column names differ: remap using `df.rename(columns={...})`
- If high nulls: fill title with empty string, drop if text is null
- If split skew: use `StratifiedKFold` and pick best fold

## PERFORMANCE TARGETS
- Preprocessing completes in < 2 minutes for 40k rows
- Memory usage < 1 GB

## RISKS
- Column name mismatch between train and test CSV
- Unicode encoding errors in text
- Extremely long articles causing tokenizer truncation loss

## DELIVERABLES
- ✅ `src/preprocess.py` (importable, tested)
- ✅ `data/processed/train_clean.csv`
- ✅ `data/processed/val_clean.csv`
- ✅ `data/processed/test_clean.csv`
- ✅ Printed shape + label distribution report
