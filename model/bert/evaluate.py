"""
Genzon — BERT Evaluation Script
Load a saved BERT model and evaluate on test set.

Usage:
    python -m model.bert.evaluate
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import BertForSequenceClassification, BertTokenizer

from model.bert.dataset import ReviewDataset
from model.utils.metrics import (
    compute_metrics, print_metrics, print_classification_report,
    print_confusion_matrix, find_best_threshold,
)


def load_model(
    model_dir: str | Path = "model/checkpoints/bert_best",
    device: str = "auto",
) -> tuple:
    """
    Load a saved BERT model and tokenizer.
    
    Returns:
        (model, tokenizer, device)
    """
    model_dir = Path(model_dir)

    if device == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(device)

    print(f"  Loading model from {model_dir}...")
    model = BertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model.to(dev)
    model.eval()
    print(f"  ✓ Model loaded on {dev}")

    return model, tokenizer, dev


@torch.no_grad()
def evaluate_on_dataset(
    model,
    tokenizer,
    device,
    df: pd.DataFrame,
    text_col: str = "text_clean",
    label_col: str = "label_encoded",
    batch_size: int = 32,
    max_length: int = 512,
) -> dict:
    """
    Run full evaluation on a dataset.
    
    Returns:
        dict with metrics + predictions
    """
    dataset = ReviewDataset(df, text_col, label_col, max_length=max_length, tokenizer=tokenizer)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_labels = []
    all_probs = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"]

        outputs = model(input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_proba = np.array(all_probs)

    metrics = compute_metrics(y_true, y_pred, y_proba)

    return {
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def full_evaluation(model_dir: str = "model/checkpoints/bert_best"):
    """Run complete evaluation pipeline on test set."""
    print("=" * 60)
    print("  Genzon — BERT Model Evaluation")
    print("=" * 60)

    # Load model
    model, tokenizer, device = load_model(model_dir)

    # Load test data
    data_dir = Path("data/processed")
    if not data_dir.exists():
        data_dir = Path("../data/processed")

    test = pd.read_csv(data_dir / "test.csv")
    print(f"\n  Test set: {len(test)} reviews")

    # Evaluate
    results = evaluate_on_dataset(model, tokenizer, device, test)

    print_metrics(results["metrics"], title="BERT Test Set Results")
    print_classification_report(results["y_true"], results["y_pred"])
    print_confusion_matrix(results["y_true"], results["y_pred"])

    # Threshold tuning
    print("\n  Threshold tuning:")
    find_best_threshold(results["y_true"], results["y_proba"], metric="f1")

    return results


if __name__ == "__main__":
    full_evaluation()