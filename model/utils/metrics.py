"""
Genzon — Shared Metric Functions
Used by all model components (rule engine, baseline, BERT, fusion).
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> dict:
    """
    Compute all evaluation metrics.
    
    Args:
        y_true: ground truth labels (0 or 1)
        y_pred: predicted labels (0 or 1)
        y_proba: predicted probabilities for class 1 (optional, needed for AUC)
    
    Returns:
        dict of metric_name → value
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }

    if y_proba is not None:
        metrics["auc_roc"] = roc_auc_score(y_true, y_proba)
        metrics["auc_pr"] = average_precision_score(y_true, y_proba)

    return metrics


def print_metrics(metrics: dict, title: str = "Evaluation"):
    """Pretty-print metrics with bar chart."""
    print(f"\n  {title}:")
    print("  " + "-" * 45)
    for k, v in metrics.items():
        bar = "█" * int(v * 30)
        print(f"    {k:<12} {v:.4f}  {bar}")


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray):
    """Print sklearn classification report."""
    print("\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Genuine", "Fake"]))


def print_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
    """Print confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    print("\n  Confusion Matrix:")
    print(f"                  Predicted")
    print(f"                  Genuine  Fake")
    print(f"    Actual Genuine  {cm[0][0]:>5}  {cm[0][1]:>5}")
    print(f"    Actual Fake     {cm[1][0]:>5}  {cm[1][1]:>5}")


def find_best_threshold(y_true: np.ndarray, y_proba: np.ndarray, metric: str = "f1") -> tuple[float, float]:
    """
    Find the optimal classification threshold.
    
    Args:
        y_true: ground truth labels
        y_proba: predicted probabilities
        metric: which metric to optimize ("f1", "precision", "recall")
    
    Returns:
        (best_threshold, best_metric_value)
    """
    best_thresh = 0.5
    best_score = 0.0

    for thresh in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_proba >= thresh).astype(int)

        if metric == "f1":
            score = f1_score(y_true, y_pred)
        elif metric == "precision":
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            score = f1_score(y_true, y_pred)

        if score > best_score:
            best_score = score
            best_thresh = thresh

    print(f"  Best threshold for {metric}: {best_thresh:.2f} (score: {best_score:.4f})")
    return best_thresh, best_score