"""
utils.py — Shared utility functions for the FakeGuard pipeline.

Provides:
    - set_seed()          : Reproducible random seed across all libraries
    - create_directory()  : Safe directory creation
    - save_json()         : Save dict to JSON file
    - load_json()         : Load JSON file to dict
    - compute_metrics()   : Accuracy + F1 from predictions and labels
    - get_logger()        : Configured Python logger

Design rules:
    - No hardcoded paths
    - All functions accept Path or str
    - Deterministic output
    - Compatible with Colab, Kaggle, and local environments
"""

import os
import json
import random
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def get_logger(name: str = "fakeguard", level: int = logging.INFO) -> logging.Logger:
    """
    Return a configured logger with console output.

    Args:
        name  : Logger name (default: 'fakeguard').
        level : Logging level (default: logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:  # avoid duplicate handlers on re-import
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


_logger = get_logger()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds across Python, NumPy, and PyTorch for full reproducibility.

    Args:
        seed: Integer seed value. Default is 42.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        _logger.warning("PyTorch not found — skipping torch seed setting.")

    _logger.info("Random seed set to %d", seed)


# ---------------------------------------------------------------------------
# File System
# ---------------------------------------------------------------------------

def create_directory(path: Union[str, Path]) -> Path:
    """
    Create a directory (and all parents) if it does not already exist.

    Args:
        path: Directory path to create.

    Returns:
        Path: The resolved directory path.
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    _logger.debug("Directory ensured: %s", dir_path)
    return dir_path


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------

def save_json(data: Dict, filepath: Union[str, Path], indent: int = 2) -> None:
    """
    Serialize a dictionary to a JSON file.

    Args:
        data     : Dictionary to serialize.
        filepath : Target file path (.json).
        indent   : JSON indentation level. Default is 2.

    Returns:
        None
    """
    filepath = Path(filepath)
    create_directory(filepath.parent)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    _logger.info("Saved JSON → %s", filepath)


def load_json(filepath: Union[str, Path]) -> Dict:
    """
    Load a JSON file into a dictionary.

    Args:
        filepath: Path to the JSON file.

    Returns:
        Dict: Parsed JSON content.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    _logger.info("Loaded JSON ← %s", filepath)
    return data


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    labels: List[int],
    predictions: List[int],
    target_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute classification metrics: accuracy, precision, recall, F1.

    Args:
        labels       : Ground-truth integer labels.
        predictions  : Predicted integer labels.
        target_names : Optional class name list e.g. ['REAL', 'FAKE'].

    Returns:
        Dict with keys: accuracy, precision_weighted, recall_weighted,
                        f1_weighted, f1_macro.
    """
    if target_names is None:
        target_names = [str(c) for c in sorted(set(labels))]

    metrics = {
        "accuracy":           round(float(accuracy_score(labels, predictions)), 6),
        "precision_weighted": round(float(precision_score(labels, predictions, average="weighted", zero_division=0)), 6),
        "recall_weighted":    round(float(recall_score(labels, predictions, average="weighted", zero_division=0)), 6),
        "f1_weighted":        round(float(f1_score(labels, predictions, average="weighted", zero_division=0)), 6),
        "f1_macro":           round(float(f1_score(labels, predictions, average="macro", zero_division=0)), 6),
    }

    _logger.info(
        "Metrics — Accuracy: %.4f | F1 (weighted): %.4f | F1 (macro): %.4f",
        metrics["accuracy"],
        metrics["f1_weighted"],
        metrics["f1_macro"],
    )
    return metrics
