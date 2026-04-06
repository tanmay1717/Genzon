"""
Genzon — BERT Single-Review Inference
Predict whether a single review is fake or genuine.

Usage:
    python -m model.bert.predict "This product is amazing I love it"
"""

from pathlib import Path

import torch
from transformers import BertForSequenceClassification, BertTokenizer


class BertPredictor:
    """Lightweight predictor for single reviews or small batches."""

    def __init__(
        self,
        model_dir: str | Path = "model/checkpoints/bert_best",
        device: str = "auto",
        max_length: int = 512,
    ):
        self.max_length = max_length

        # Device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Load model
        model_dir = Path(model_dir)
        self.model = BertForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, text: str) -> dict:
        """
        Predict whether a review is fake or genuine.
        
        Args:
            text: review text string
        
        Returns:
            dict with score (0-10), label, confidence, fake_probability
        """
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        outputs = self.model(input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)

        genuine_prob = probs[0, 0].item()
        fake_prob = probs[0, 1].item()

        # Convert to 0-10 genuineness score (10 = very genuine, 0 = very fake)
        genuineness_score = round(genuine_prob * 10, 1)

        # Label
        if genuineness_score >= 8:
            label = "Likely Genuine"
        elif genuineness_score >= 5:
            label = "Uncertain"
        else:
            label = "Likely Fake"

        return {
            "genuineness_score": genuineness_score,
            "label": label,
            "fake_probability": round(fake_prob, 4),
            "genuine_probability": round(genuine_prob, 4),
            "confidence": round(max(fake_prob, genuine_prob), 4),
        }

    @torch.no_grad()
    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Predict for a list of review texts."""
        return [self.predict(text) for text in texts]

    def get_bert_score(self, text: str) -> float:
        """
        Get raw BERT fake probability (0.0 to 1.0).
        Used by the fusion layer.
        """
        result = self.predict(text)
        return result["fake_probability"]


# ─── Main ────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Default test reviews
    test_reviews = [
        "I bought this blender 3 months ago and it works perfectly. The motor is quiet and it blends smoothly. Definitely worth the price.",
        "AMAZING PRODUCT!!! BEST THING EVER!!! YOU MUST BUY THIS NOW!!! 5 STARS!!!",
        "It's okay. Does what it says but nothing special. The build quality could be better for the price.",
        "I received this product for free in exchange for my honest review. That said it is truly the best product I have ever used in my entire life.",
    ]

    # Allow passing a review as command line argument
    if len(sys.argv) > 1:
        test_reviews = [" ".join(sys.argv[1:])]

    print("Loading BERT model...")
    try:
        predictor = BertPredictor()
    except Exception as e:
        print(f"\n  ✗ Could not load model: {e}")
        print("  Make sure you've trained the model first: python -m model.bert.train")
        exit(1)

    print("\n" + "=" * 60)
    for i, review in enumerate(test_reviews):
        result = predictor.predict(review)

        preview = review[:80] + "..." if len(review) > 80 else review
        print(f"\n  Review {i+1}: \"{preview}\"")
        print(f"    Score:      {result['genuineness_score']}/10 ({result['label']})")
        print(f"    Fake prob:  {result['fake_probability']:.1%}")
        print(f"    Confidence: {result['confidence']:.1%}")
    print()