# Phase 1 — Repository Stabilization & Environment Setup

## GOAL
Establish a clean, reproducible, crash-free project foundation before any code is written or model is trained.

## WHY THIS PHASE EXISTS
- **Risk reduced:** Broken imports, missing directories, and environment mismatches crash everything downstream
- **Capability created:** A confirmed working environment where every subsequent phase can execute without setup failures
- **Competitive advantage:** Teams that skip stabilization lose time debugging environment issues during the hackathon

## PREREQUISITES
- GitHub repository access confirmed
- Google Colab T4 GPU runtime available
- `config.py` and `requirements.txt` present in repo root

## INPUTS
- `config.py` (existing)
- `requirements.txt` (existing)
- Google Colab runtime

## TASKS
1. Clone repository into Colab environment
2. Install all dependencies from `requirements.txt` (except torch — Colab-managed)
3. Verify all imports load without error
4. Create missing directory structure: `data/`, `data/processed/`, `models/`, `outputs/`
5. Confirm `config.py` values load correctly
6. Verify GPU is available (`torch.cuda.is_available()` → True)
7. Set global random seed = 42 everywhere
8. Create `src/__init__.py` to make src a Python package
9. Commit directory scaffolding files (`.gitkeep`) for `data/processed/`, `models/`, `outputs/`

## AI EXECUTION PROMPTS
- "Clone the repo, install requirements excluding torch, then run a Python import check for transformers, datasets, sklearn, pandas, numpy, torch, gradio. Print OK or FAIL for each."
- "Create directories data/, data/processed/, models/, outputs/ if they do not exist. Add .gitkeep to each. Verify torch.cuda.is_available() returns True."
- "Load config.py and print every variable. Confirm no KeyError or AttributeError."

## ALGORITHMS
- None (environment phase only)

## CODE SNIPPETS
```python
import torch, transformers, datasets, sklearn, pandas, numpy
print("CUDA:", torch.cuda.is_available())
print("Transformers:", transformers.__version__)
```
```python
import os, sys
sys.path.insert(0, '/content/neurologic-datathon-fakenews')
from config import *
print(TRANSFORMER, BATCH_SIZE, EPOCHS, LEARNING_RATE)
```

## OUTPUTS
- `src/__init__.py`
- `data/.gitkeep`
- `data/processed/.gitkeep`
- `models/.gitkeep`
- `outputs/.gitkeep`
- Console: all imports GREEN, CUDA: True

## EXPECTED RESULTS
- All 7 libraries import successfully
- `torch.cuda.is_available()` returns `True`
- All config variables load without error
- Directory structure matches `config.py` paths exactly

## VALIDATION CHECKS
- [ ] `import transformers` → no error
- [ ] `import torch; torch.cuda.is_available()` → True
- [ ] `from config import TRANSFORMER` → prints `roberta-base`
- [ ] `os.path.exists('data/')` → True
- [ ] `os.path.exists('outputs/')` → True

## FAILURE CONDITIONS
- `torch.cuda.is_available()` returns False → switch to GPU runtime
- Import error on transformers → re-run pip install
- config.py throws error → fix syntax before proceeding

## RECOVERY ACTIONS
- If CUDA unavailable: Runtime → Change runtime type → T4 GPU → Reconnect
- If transformers fails: `!pip install transformers==4.40.0 --quiet`
- If config error: open config.py, check for syntax issues

## PERFORMANCE TARGETS
- Setup completes in < 5 minutes
- Zero import errors
- Zero missing directories

## RISKS
- Colab session timeout during setup → reconnect and re-run cell
- Package version conflict → use exact versions from requirements.txt
- Git clone auth failure → use HTTPS with token

## DELIVERABLES
- ✅ Working Colab environment
- ✅ All directories created
- ✅ `src/__init__.py` committed
- ✅ Config loads cleanly
- ✅ GPU confirmed
