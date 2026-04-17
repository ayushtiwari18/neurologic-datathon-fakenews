# 🧹 Data Preprocessing Guide

## Expected Dataset Format
The hackathon dataset will likely have these columns:
```
id | title | text | subject | date | label
```
Where `label` = REAL or FAKE (or 1/0)

## Step-by-Step Preprocessing

### Step 1: Load and Inspect
```python
import pandas as pd

df = pd.read_csv('data/raw/dataset.csv')
print(df.shape)           # How many rows?
print(df.dtypes)          # Column types
print(df.isnull().sum())  # Missing values
print(df['label'].value_counts())  # Class balance
```

### Step 2: Handle Missing Values
```python
# Drop rows where BOTH title and text are missing
df = df.dropna(subset=['title', 'text'], how='all')

# Fill individual missing with empty string
df['title'] = df['title'].fillna('')
df['text'] = df['text'].fillna('')

print(f"Rows after cleaning: {len(df)}")
```

### Step 3: Clean Text
```python
import re
import html  # REQUIRED for html.unescape — do not skip this import

def clean_text(text):
    text = str(text)
    text = html.unescape(text)                            # Decode HTML entities (&amp; → &, etc.)
    text = text.lower()                                   # Lowercase
    text = re.sub(r'http\S+|www\S+', '', text)            # Remove URLs
    text = re.sub(r'<.*?>', '', text)                     # Remove HTML tags
    text = re.sub(r'[^a-z0-9\s.,!?\']', '', text)        # Keep only letters/numbers/basic punctuation
    text = re.sub(r'\s+', ' ', text).strip()              # Normalize whitespace
    return text

df['title_clean'] = df['title'].apply(clean_text)
df['text_clean'] = df['text'].apply(clean_text)
```

### Step 4: Combine Title + Text (Key Innovation)
```python
# Using [SEP] tells RoBERTa these are two distinct segments
df['combined'] = df['title_clean'] + ' [SEP] ' + df['text_clean']

# Trim to ~2000 chars — approximate equivalent of 512 tokens.
# Actual token count varies by text; this is a safe upper bound.
# The tokenizer will still truncate at max_length=512 tokens.
df['combined'] = df['combined'].apply(lambda x: x[:2000])
```

### Step 5: Encode Labels
```python
# NOTE: pandas may load numeric labels as float (1.0, 0.0) —
# the label_map below includes float keys to handle this case.
label_map = {'REAL': 1, 'real': 1, '1': 1, 1: 1, 1.0: 1,
             'FAKE': 0, 'fake': 0, '0': 0, 0: 0, 0.0: 0}
df['label'] = df['label'].map(label_map)

# Verify no unmapped labels
assert df['label'].isnull().sum() == 0, "Label encoding failed — check label column values"
print(df['label'].value_counts())
```

### Step 6: Train/Validation Split
```python
from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(
    df[['combined', 'label']],
    test_size=0.20,
    random_state=42,
    stratify=df['label']   # Ensures same Real/Fake ratio in both splits
)

print(f"Train: {len(train_df)} | Val: {len(val_df)}")
train_df.to_csv('data/processed/train.csv', index=False)
val_df.to_csv('data/processed/val.csv', index=False)
```

### Step 7: EDA Checks to Report
```python
import matplotlib.pyplot as plt

# Class distribution
df['label'].value_counts().plot(kind='bar', title='Class Distribution')
plt.savefig('outputs/class_distribution.png')

# Text length distribution
df['text_len'] = df['combined'].apply(len)
df['text_len'].hist(bins=50)
plt.title('Combined Text Length Distribution')
plt.savefig('outputs/text_length_dist.png')

print(f"Average text length: {df['text_len'].mean():.0f} chars")
print(f"Max text length: {df['text_len'].max()} chars")
```

## Common Dirty Data Issues & Fixes
| Issue | Detection | Fix |
|---|---|---|
| Duplicate articles | `df.duplicated('combined').sum()` | `df.drop_duplicates('combined')` |
| Labels as strings | `df['label'].dtype == object` | Map using `label_map` |
| HTML entities (`&amp;`) | Check sample text | `html.unescape(text)` — already in clean_text |
| Non-English articles | Spot-check text | `langdetect` library filter |
| Extreme class imbalance | `value_counts()` ratio > 4:1 | Report it, use `class_weight='balanced'` |
