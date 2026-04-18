"""
dataset.py — PyTorch dataset wrapper for FakeGuard.

Source of truth:
    - instructions/03_pipeline_architecture.md
    - instructions/04_model_training_strategy.md
    - instructions/09_features_and_routes.md

Responsibilities:
    1. Load processed CSVs containing `combined` and `label`
    2. Tokenize text using RobertaTokenizer
    3. Return tensors required by RoBERTa:
         - input_ids
         - attention_mask
         - labels (for train/val)
    4. NEVER return token_type_ids — RoBERTa does not use them

Compatible with:
    - Local Python
    - Kaggle
    - Google Colab
"""

from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from config import MAX_LEN, SEED, TEXT_COL, LABEL_COL, TRANSFORMER
from src.utils import get_logger, set_seed

logger = get_logger("dataset")


class FakeNewsDataset(Dataset):
    """
    PyTorch Dataset for FakeGuard fake-news classification.

    This dataset wraps a processed pandas DataFrame or CSV path and tokenizes
    the `combined` text column for RoBERTa sequence classification.

    IMPORTANT:
        RoBERTa does NOT use `token_type_ids`. This class intentionally does
        not return them, even if another model family might.

    Args:
        data         : Processed pandas DataFrame or CSV path.
        tokenizer    : HuggingFace tokenizer instance. If None, loads from config.
        text_col     : Column containing preprocessed combined text.
        label_col    : Column containing encoded labels.
        max_length   : Maximum token length for truncation.
        has_labels   : If False, dataset returns inference-only features.
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, str, Path],
        tokenizer: Optional[AutoTokenizer] = None,
        text_col: str = TEXT_COL,
        label_col: str = LABEL_COL,
        max_length: int = MAX_LEN,
        has_labels: bool = True,
    ) -> None:
        set_seed(SEED)

        self.data = self._load_data(data)
        self.text_col = text_col
        self.label_col = label_col
        self.max_length = max_length
        self.has_labels = has_labels
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(TRANSFORMER)

        self._validate()
        logger.info(
            "Dataset initialized — rows: %d | text_col: %s | labels: %s | max_length: %d",
            len(self.data),
            self.text_col,
            self.has_labels,
            self.max_length,
        )

    def _load_data(self, data: Union[pd.DataFrame, str, Path]) -> pd.DataFrame:
        """
        Load dataset from DataFrame or CSV path.

        Args:
            data: DataFrame or path to CSV.

        Returns:
            pd.DataFrame: Loaded dataset.

        Raises:
            FileNotFoundError: If the CSV path does not exist.
            ValueError: If the loaded data is empty.
        """
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            path = Path(data)
            if not path.exists():
                raise FileNotFoundError(f"Dataset file not found: {path}")
            df = pd.read_csv(path)

        if df.empty:
            raise ValueError("Dataset is empty.")
        return df.reset_index(drop=True)

    def _validate(self) -> None:
        """
        Validate required columns and ensure no null text remains.

        Raises:
            ValueError: If required columns are missing.
        """
        required = [self.text_col]
        if self.has_labels:
            required.append(self.label_col)

        missing = [col for col in required if col not in self.data.columns]
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}. "
                f"Available columns: {self.data.columns.tolist()}"
            )

        self.data[self.text_col] = self.data[self.text_col].fillna("").astype(str)
        if self.has_labels:
            self.data[self.label_col] = self.data[self.label_col].astype(int)

    def __len__(self) -> int:
        """
        Return number of samples.

        Returns:
            int: Dataset length.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Return one tokenized example.

        Args:
            idx: Row index.

        Returns:
            Dict[str, torch.Tensor]: Tokenized tensors.
                For train/val:
                    {
                        'input_ids': Tensor,
                        'attention_mask': Tensor,
                        'labels': Tensor,
                    }
                For test/inference:
                    {
                        'input_ids': Tensor,
                        'attention_mask': Tensor,
                    }
        """
        row = self.data.iloc[idx]
        text = row[self.text_col]

        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

        if self.has_labels:
            item["labels"] = torch.tensor(int(row[self.label_col]), dtype=torch.long)

        return item


def load_tokenizer(model_name: str = TRANSFORMER) -> AutoTokenizer:
    """
    Load and return the HuggingFace tokenizer.

    Args:
        model_name: Pretrained model identifier.

    Returns:
        AutoTokenizer: Loaded tokenizer instance.
    """
    logger.info("Loading tokenizer: %s", model_name)
    return AutoTokenizer.from_pretrained(model_name)
