"""
Genzon — PyTorch Dataset for BERT fine-tuning.
Tokenizes review text and prepares it for BERT input.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import BertTokenizer


class ReviewDataset(Dataset):
    """
    PyTorch Dataset that tokenizes review text for BERT.
    
    Each item returns:
        - input_ids: token IDs
        - attention_mask: 1 for real tokens, 0 for padding
        - label: 0 (genuine) or 1 (fake)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        text_col: str = "text_clean",
        label_col: str = "label_encoded",
        model_name: str = "bert-base-uncased",
        max_length: int = 512,
        tokenizer: BertTokenizer | None = None,
    ):
        self.texts = df[text_col].fillna("").tolist()
        self.labels = df[label_col].tolist()

        # Reuse tokenizer if passed (saves loading time)
        self.tokenizer = tokenizer or BertTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    text_col: str = "text_clean",
    label_col: str = "label_encoded",
    model_name: str = "bert-base-uncased",
    max_length: int = 512,
    batch_size: int = 16,
    num_workers: int = 0,
) -> tuple:
    """
    Create train and validation DataLoaders.
    
    Returns:
        (train_loader, val_loader, tokenizer)
    """
    tokenizer = BertTokenizer.from_pretrained(model_name)

    train_dataset = ReviewDataset(
        train_df, text_col, label_col,
        model_name, max_length, tokenizer,
    )
    val_dataset = ReviewDataset(
        val_df, text_col, label_col,
        model_name, max_length, tokenizer,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"  ✓ Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  ✓ Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  ✓ Batch size: {batch_size}, Max length: {max_length}")

    return train_loader, val_loader, tokenizer


# ─── Quick test ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from data.config import PROCESSED_DIR

    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    val = pd.read_csv(PROCESSED_DIR / "val.csv")

    # Test with small subset
    train_loader, val_loader, tokenizer = create_dataloaders(
        train.head(100), val.head(50),
        batch_size=8, max_length=128,
    )

    # Check one batch
    batch = next(iter(train_loader))
    print(f"\n  Sample batch:")
    print(f"    input_ids shape:     {batch['input_ids'].shape}")
    print(f"    attention_mask shape: {batch['attention_mask'].shape}")
    print(f"    labels shape:        {batch['label'].shape}")
    print(f"    labels:              {batch['label'].tolist()}")