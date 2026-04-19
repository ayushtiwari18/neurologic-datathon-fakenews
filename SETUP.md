# 🛠️ FakeGuard — Local & Colab Setup Guide
**NeuroLogic '26 Datathon — Fake News Detection**

> Follow this guide exactly. Every command is tested and ordered correctly.

---

## 💻 SECTION 1 — Local Setup (Ubuntu / Debian / WSL)

### Step 1 — Clone the Repository

```bash
cd /your/working/directory
git clone https://github.com/ayushtiwari18/neurologic-datathon-fakenews.git
cd neurologic-datathon-fakenews
```

---

### Step 2 — Create Python Virtual Environment

> ⚠️ **Never use system Python directly** on Ubuntu/Debian — it is externally managed.

```bash
# Ensure python3-full and venv are installed
sudo apt update
sudo apt install python3-full python3-venv -y

# Create the virtual environment inside the repo folder
python3 -m venv venv
```

---

### Step 3 — Activate the Virtual Environment

```bash
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal prompt:
```
(venv) ayush5410@Ayush-PC:/media/.../neurologic-datathon-fakenews$
```

> ⚠️ **You must activate the venv every time you open a new terminal session.**

---

### Step 4 — Upgrade pip

```bash
pip install --upgrade pip
```

---

### Step 5 — Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ `torch` is **not** in requirements.txt for Colab compatibility.
> Install it separately for local use:

```bash
# For CPU-only (safe on any machine)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For NVIDIA GPU (CUDA 12.x)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

### Step 6 — Verify Installation

```bash
python -c "
import torch, transformers, sklearn, pandas, numpy, gradio
print('torch      :', torch.__version__)
print('transformers:', transformers.__version__)
print('sklearn    :', sklearn.__version__)
print('pandas     :', pandas.__version__)
print('numpy      :', numpy.__version__)
print('CUDA available:', torch.cuda.is_available())
print('ALL OK')
"
```

Expected output:
```
torch      : 2.x.x
transformers: 4.40.0
sklearn    : 1.4.2
pandas     : 2.2.2
numpy      : 1.26.4
CUDA available: False   (True if NVIDIA GPU present)
ALL OK
```

---

### Step 7 — Create Required Directories

```bash
mkdir -p data/processed models outputs
```

---

### Step 8 — Deactivate (when done working)

```bash
deactivate
```

---

## 📂 SECTION 2 — Dataset Setup

> ⚠️ **Datasets are NEVER committed to GitHub.** They are placed locally only.

### Required Files

```
neurologic-datathon-fakenews/
└── data/
    ├── train.csv     ← training data with labels
    └── test.csv      ← test data (no labels — for submission)
```

### Required Columns

| File | Required Columns | Notes |
|------|-----------------|-------|
| `train.csv` | `title`, `text`, `label` | label: 0=REAL, 1=FAKE |
| `test.csv` | `title`, `text` | no label column needed |

---

### Option A — NeuroLogic '26 Official Dataset (Primary)

1. Log into the NeuroLogic '26 datathon portal
2. Navigate to **Challenge 2: Fake News Detection**
3. Download `train.csv` and `test.csv`
4. Place them in the `data/` folder:
```bash
mv ~/Downloads/train.csv data/train.csv
mv ~/Downloads/test.csv  data/test.csv
```

---

### Option B — WELFake Dataset (Kaggle, Public Benchmark)

> Use this if you don't have the official dataset yet.

```bash
# Install Kaggle CLI (inside venv)
pip install kaggle

# Place your kaggle.json API key at ~/.kaggle/kaggle.json
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download WELFake dataset
kaggle datasets download -d saurabhshahane/fake-news-classification
unzip fake-news-classification.zip -d data/

# WELFake has 'title', 'text', 'label' columns — directly compatible
```

---

### Option C — ISOT Fake News Dataset (Direct Download)

```bash
# Download from University of Victoria
# https://www.uvic.ca/ecs/ece/isot/datasets/fake-news/index.php
# Download Fake.csv and True.csv, then combine:

python3 - <<'EOF'
import pandas as pd

fake = pd.read_csv('data/Fake.csv')
true = pd.read_csv('data/True.csv')
fake['label'] = 1  # FAKE
true['label'] = 0  # REAL

combined = pd.concat([fake, true]).sample(frac=1, random_state=42).reset_index(drop=True)

# Split 90/10 for train/test
train = combined.iloc[:int(0.9*len(combined))]
test  = combined.iloc[int(0.9*len(combined)):].drop(columns=['label'])

train.to_csv('data/train.csv', index=False)
test.to_csv('data/test.csv',  index=False)
print(f'train: {len(train)} rows | test: {len(test)} rows')
EOF
```

---

## ▶️ SECTION 3 — Run the Pipeline

### Step 1 — Activate venv
```bash
source venv/bin/activate
```

### Step 2 — Run Preprocessing
```bash
python src/preprocess.py
```

Expected output:
```
2026-04-19 00:00:00 | INFO     | preprocess | Loading train data from: data/train.csv
2026-04-19 00:00:00 | INFO     | preprocess | Raw train shape: (XXXXX, 3)
...
2026-04-19 00:00:00 | INFO     | preprocess | PREPROCESSING COMPLETE
2026-04-19 00:00:00 | INFO     | preprocess |   Train rows : XXXXX
2026-04-19 00:00:00 | INFO     | preprocess |   Val rows   : XXXXX
2026-04-19 00:00:00 | INFO     | preprocess |   Test rows  : XXXXX
```

### Step 3 — Verify Outputs
```bash
ls -lh data/processed/
# Expected:
# train_clean.csv
# val_clean.csv
# test_clean.csv
# preprocessing_stats.json
```

### Step 4 — Check Stats
```bash
python -c "from src.utils import load_json; import json; print(json.dumps(load_json('data/processed/preprocessing_stats.json'), indent=2))"
```

---

## ☁️ SECTION 4 — Google Colab Setup

```python
# Cell 1 — Clone repo
!git clone https://github.com/ayushtiwari18/neurologic-datathon-fakenews.git
%cd neurologic-datathon-fakenews

# Cell 2 — Install dependencies (torch is pre-installed on Colab)
!pip install -r requirements.txt -q

# Cell 3 — Upload data files
from google.colab import files
uploaded = files.upload()   # select train.csv and test.csv

import shutil
shutil.move('train.csv', 'data/train.csv')
shutil.move('test.csv',  'data/test.csv')

# Cell 4 — Run preprocessing
from src.preprocess import run_preprocessing
train_df, val_df, test_df = run_preprocessing()
print(f'Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}')

# Cell 5 — Verify GPU
import torch
print('CUDA:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
```

---

## 🔒 SECTION 5 — What is NEVER Committed to GitHub

| Type | Examples | Reason |
|------|---------|--------|
| Datasets | `data/*.csv` | Too large, contains contest data |
| Model weights | `models/`, `*.bin`, `*.pt` | 400MB+, use HuggingFace Hub |
| Virtual env | `venv/` | OS-specific, reproducible via requirements.txt |
| Outputs | `outputs/*.json`, `*.png` | Generated at runtime |
| Secrets | `.env`, `*.secret` | Security |
| Cache | `__pycache__/`, `.ipynb_checkpoints/` | Temporary files |

All of the above are blocked by `.gitignore`.

---

## ✅ SECTION 6 — Quick Verification Checklist

Run this to confirm everything is ready:

```bash
source venv/bin/activate

python - <<'EOF'
import sys
from pathlib import Path

checks = [
    (Path('config.py').exists(),              'config.py exists'),
    (Path('requirements.txt').exists(),       'requirements.txt exists'),
    (Path('src/__init__.py').exists(),        'src/__init__.py exists'),
    (Path('src/utils.py').exists(),           'src/utils.py exists'),
    (Path('src/preprocess.py').exists(),      'src/preprocess.py exists'),
    (Path('data/train.csv').exists(),         'data/train.csv exists'),
    (Path('data/test.csv').exists(),          'data/test.csv exists'),
]

all_pass = True
for result, label in checks:
    status = '✅' if result else '❌'
    print(f'{status}  {label}')
    if not result:
        all_pass = False

print()
print('SETUP READY' if all_pass else 'SETUP INCOMPLETE — fix ❌ items above')
EOF
```

---

## 🚀 NEXT STEP after Setup is Verified

Once all checks pass, run:
```bash
python src/preprocess.py
```
Then tell the AI: **"Setup verified, run Cycle 7"**


# This is the correct way — launch python first, then type the commands
python3 -c "import torch; print('GPU:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU found — use Colab/Kaggle')"
