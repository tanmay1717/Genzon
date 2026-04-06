"""
Genzon — Diagnose BERT prediction issue
Run: python -m model.bert.diagnose
"""

from pathlib import Path
import pandas as pd
import torch
from transformers import BertForSequenceClassification, BertTokenizer


def main():
    # Load model
    model_dir = Path("model/checkpoints/bert_best").resolve()
    print(f"Loading model from {model_dir}...")
    model = BertForSequenceClassification.from_pretrained(str(model_dir))
    tokenizer = BertTokenizer.from_pretrained(str(model_dir))

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model.to(device)
    model.eval()
    print(f"Device: {device}\n")

    # Check model config
    print("=" * 55)
    print("  Model Config Check")
    print("=" * 55)
    print(f"  num_labels: {model.config.num_labels}")
    print(f"  id2label:   {model.config.id2label}")
    print(f"  label2id:   {model.config.label2id}")

    # Load some known fake and genuine reviews from test set
    data_dir = Path("data/processed")
    if not data_dir.exists():
        data_dir = Path("../data/processed")

    test_df = pd.read_csv(data_dir / "test.csv")

    genuine_samples = test_df[test_df["label_encoded"] == 0].head(5)
    fake_samples = test_df[test_df["label_encoded"] == 1].head(5)

    print(f"\n{'=' * 55}")
    print("  Testing 5 KNOWN GENUINE reviews (label=0)")
    print("=" * 55)

    with torch.no_grad():
        for _, row in genuine_samples.iterrows():
            text = str(row["text_clean"])[:200]
            enc = tokenizer(text, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
            out = model(enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device))
            probs = torch.softmax(out.logits, dim=1)

            p0 = probs[0, 0].item()
            p1 = probs[0, 1].item()
            preview = text[:60] + "..."
            print(f"\n  \"{preview}\"")
            print(f"    class_0 prob: {p0:.4f}  |  class_1 prob: {p1:.4f}  |  argmax: {torch.argmax(probs, dim=1).item()}")

    print(f"\n{'=' * 55}")
    print("  Testing 5 KNOWN FAKE reviews (label=1)")
    print("=" * 55)

    with torch.no_grad():
        for _, row in fake_samples.iterrows():
            text = str(row["text_clean"])[:200]
            enc = tokenizer(text, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
            out = model(enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device))
            probs = torch.softmax(out.logits, dim=1)

            p0 = probs[0, 0].item()
            p1 = probs[0, 1].item()
            preview = text[:60] + "..."
            print(f"\n  \"{preview}\"")
            print(f"    class_0 prob: {p0:.4f}  |  class_1 prob: {p1:.4f}  |  argmax: {torch.argmax(probs, dim=1).item()}")

    # Summary
    print(f"\n{'=' * 55}")
    print("  Diagnosis")
    print("=" * 55)
    print("""
  Look at the output above:

  IF genuine reviews have HIGH class_0 AND fake reviews have HIGH class_1:
    → Labels are CORRECT. Model works. predict.py is fine.

  IF genuine reviews have HIGH class_1 AND fake reviews have HIGH class_0:
    → Labels are SWAPPED. class_0=fake, class_1=genuine.
    → Fix: swap the probabilities in predict.py

  IF ALL reviews have HIGH class_0 (or all class_1):
    → Model didn't train properly or checkpoint didn't save right.
    → Fix: retrain on Colab.
    """)


if __name__ == "__main__":
    main()