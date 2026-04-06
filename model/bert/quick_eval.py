"""
Genzon — Quick BERT Evaluation (CPU-friendly)
Evaluates on a random subset for fast results, then tests sample reviews.

Usage:
    python -m model.bert.quick_eval              # default 500 samples
    python -m model.bert.quick_eval --samples 1000
    python -m model.bert.quick_eval --full        # full test set (slow on CPU)
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=500, help="Number of test samples (default 500)")
    parser.add_argument("--full", action="store_true", help="Use full test set (slow on CPU)")
    parser.add_argument("--model-dir", type=str, default="model/checkpoints/bert_best")
    args = parser.parse_args()

    # ── Device ──
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Load model ──
    model_dir = Path(args.model_dir).resolve()
    print(f"Loading model from {model_dir}...")
    model = BertForSequenceClassification.from_pretrained(str(model_dir))
    tokenizer = BertTokenizer.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()
    print("✓ Model loaded\n")

    # ── Load test data ──
    data_dir = Path("data/processed")
    if not data_dir.exists():
        data_dir = Path("../data/processed")

    test_df = pd.read_csv(data_dir / "test.csv")
    print(f"Full test set: {len(test_df)} reviews")

    if not args.full:
        n = min(args.samples, len(test_df))
        test_df = test_df.sample(n=n, random_state=42)
        print(f"Using random subset: {n} reviews (use --full for all)")
    print()

    # ── Evaluate ──
    print("Running evaluation...")
    start = time.time()

    all_preds = []
    all_labels = []
    all_probs = []

    texts = test_df["text_clean"].fillna("").tolist()
    labels = test_df["label_encoded"].tolist()

    # Process in small batches for progress updates
    batch_size = 16
    total = len(texts)

    with torch.no_grad():
        for i in range(0, total, batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]

            encoding = tokenizer(
                batch_texts,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels)
            all_probs.extend(probs[:, 1].cpu().numpy())

            # Progress
            done = min(i + batch_size, total)
            elapsed = time.time() - start
            rate = done / elapsed
            remaining = (total - done) / rate if rate > 0 else 0
            print(f"\r  {done}/{total} reviews ({done/total*100:.0f}%) | {elapsed:.0f}s elapsed | ~{remaining:.0f}s remaining", end="", flush=True)

    elapsed = time.time() - start
    print(f"\n\n✓ Evaluation done in {elapsed:.1f}s ({total/elapsed:.1f} reviews/sec)\n")

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_proba = np.array(all_probs)

    # ── Results ──
    print("=" * 55)
    print(f"  BERT Results ({total} reviews)")
    print("=" * 55)

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "AUC-ROC": roc_auc_score(y_true, y_proba),
    }

    for name, val in metrics.items():
        bar = "█" * int(val * 30)
        print(f"  {name:<12} {val:.4f}  {bar}")

    print("\n" + classification_report(y_true, y_pred, target_names=["Genuine", "Fake"]))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(f"                  Predicted")
    print(f"                  Genuine  Fake")
    print(f"  Actual Genuine  {cm[0][0]:>5}  {cm[0][1]:>5}")
    print(f"  Actual Fake     {cm[1][0]:>5}  {cm[1][1]:>5}")

    # ── Sample predictions ──
    print("\n" + "=" * 55)
    print("  Sample Predictions")
    print("=" * 55)

    test_reviews = [
        "I bought this blender 3 months ago and it still works great. Motor is quiet, blends smoothly. Worth every penny.",
        "AMAZING PRODUCT!!! BEST THING EVER BUY NOW!!! 5 STARS!!!",
        "It's okay. Does what it says but the build quality could be better for the price point.",
        "I received this product for free in exchange for my honest review. That said it is truly the best product I have ever used.",
        "Stopped working after 2 weeks. Customer service was unhelpful. Returning it.",
    ]

    with torch.no_grad():
        for i, text in enumerate(test_reviews):
            enc = tokenizer(text, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)

            out = model(ids, attention_mask=mask)
            probs = torch.softmax(out.logits, dim=1)
            genuine_prob = probs[0, 0].item()
            fake_prob = probs[0, 1].item()
            score = round(genuine_prob * 10, 1)

            if score >= 8:
                label = "✅ Likely Genuine"
            elif score >= 5:
                label = "⚠️  Uncertain"
            else:
                label = "🚨 Likely Fake"

            preview = text[:65] + "..." if len(text) > 65 else text
            print(f"\n  {i+1}. \"{preview}\"")
            print(f"     Score: {score}/10  {label}  (fake: {fake_prob:.1%})")

    print()


if __name__ == "__main__":
    main()