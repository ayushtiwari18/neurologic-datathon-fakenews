# ☁️ Google Colab Setup & Configuration

## For ALL Team Members — Follow These Steps Exactly

### Step 1: Enable GPU
1. Open Colab notebook
2. Go to **Runtime → Change runtime type**
3. Set **Hardware accelerator: T4 GPU**
4. Click Save
5. Verify: `!nvidia-smi` → should show Tesla T4

### Step 2: Install All Dependencies (Run First Cell)
```python
# Cell 1 — Run this FIRST every session
# NOTE: Do NOT pin torch with --index-url https://download.pytorch.org/whl/cu118
# Colab T4 uses CUDA 12.x — pinning cu118 creates version conflicts with
# the pre-installed torch. Let Colab manage its own torch installation.
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Suppress tokenizer fork warnings

!pip install -q transformers==4.40.0 datasets==2.19.0 accelerate==0.29.0
!pip install -q scikit-learn pandas numpy matplotlib seaborn
# torch is already installed correctly in Colab — do NOT reinstall it
```

### Step 3: Mount Google Drive (for saving model/data)
```python
# Cell 2 — Mount Drive to persist files between sessions
from google.colab import drive
drive.mount('/content/drive')

# Set your working directory
IMPORT_PATH = '/content/drive/MyDrive/neurologic_datathon/'
import os
os.makedirs(IMPORT_PATH, exist_ok=True)
```

### Step 4: Verify GPU Memory
```python
# Cell 3 — Check available GPU RAM
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
# T4 = ~15GB, good for batch_size=16
# If < 8GB, reduce batch_size to 8
```

### Step 5: Set Random Seeds (Everyone Must Use Same Seeds)
```python
# Cell 4 — Reproducibility — ALL members use these exact seeds
import random, numpy as np, torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

print(f"Seeds set to {SEED} — results will be reproducible")
```

### Step 6: Upload Dataset
```python
# Option A: Upload from local machine
from google.colab import files
uploaded = files.upload()  # Select your dataset.csv

# Option B: Load directly if on Kaggle dataset
# !kaggle datasets download -d dataset-name
# !unzip dataset-name.zip
```

## ⚠️ Colab Session Rules
- Colab disconnects after **90 minutes of inactivity** — save model to Drive frequently
- Colab Pro disconnects after **24 hours** — save checkpoints every epoch
- **Free Colab** GPU quota resets every ~12 hours — if you get "GPU not available", wait or use Kaggle
- If Colab crashes mid-training, reload from last saved checkpoint in Drive

## Colab vs Kaggle Quick Decision
| Situation | Use |
|---|---|
| First time running | Colab (easier UI) |
| Colab GPU quota exhausted | Kaggle Notebooks |
| Need more than 15GB GPU RAM | Kaggle P100 (free) |
| Need persistent storage | Mount Google Drive |

## Sharing Notebook with Team
1. Save notebook to Drive
2. Click **Share** → Anyone with link can view
3. Team members: **File → Save a copy in Drive** to get their own editable version
4. Everyone runs **Cell 1–4** first to ensure identical environment
