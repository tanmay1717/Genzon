"""
Genzon — Inference Service
Loads all models at startup, runs hybrid predictions.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import BertForSequenceClassification, BertTokenizer

from backend.app.config import settings
from backend.app.services.preprocessing import preprocess_reviews

# Import model components
import sys
sys.path.insert(0, str(settings.project_root))

from model.rule_engine.manual_rules import compute_manual_rule_score
from model.rule_engine.learned_rules import LearnedRuleScorer


class InferenceEngine:
    """
    Loads and manages all ML models.
    Provides hybrid predictions combining rules + BERT.
    """

    def __init__(self):
        self.bert_model = None
        self.tokenizer = None
        self.learned_rules = None
        self.device = None
        self.is_loaded = False

    def load_models(self):
        """Load all models into memory. Called once at server startup."""
        start = time.time()
        print("\n🔧 Loading ML models...")

        # Device
        if settings.device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(settings.device)
        print(f"  Device: {self.device}")

        # BERT
        bert_path = settings.bert_model_full_path
        if bert_path.exists():
            print(f"  Loading BERT from {bert_path}...")
            self.bert_model = BertForSequenceClassification.from_pretrained(str(bert_path))
            self.tokenizer = BertTokenizer.from_pretrained(str(bert_path))
            self.bert_model.to(self.device)
            self.bert_model.eval()
            print("  ✓ BERT loaded")
        else:
            print(f"  ⚠ BERT model not found at {bert_path} — BERT scoring disabled")

        # Learned rules
        rules_path = settings.rules_model_full_path
        if rules_path.exists():
            self.learned_rules = LearnedRuleScorer()
            self.learned_rules.load(rules_path)
            print("  ✓ Learned rules loaded")
        else:
            print(f"  ⚠ Learned rules not found at {rules_path} — using manual rules only")

        self.is_loaded = True
        elapsed = time.time() - start
        print(f"  ✓ All models loaded in {elapsed:.1f}s\n")

    @torch.no_grad()
    def _bert_score(self, text: str) -> float:
        """Get BERT fake probability for a single review. Returns 0.0-1.0."""
        if self.bert_model is None:
            return 0.5  # neutral if BERT unavailable

        encoding = self.tokenizer(
            text,
            max_length=settings.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        ids = encoding["input_ids"].to(self.device)
        mask = encoding["attention_mask"].to(self.device)

        out = self.bert_model(ids, attention_mask=mask)
        probs = torch.softmax(out.logits, dim=1)

        return probs[0, 1].item()  # class 1 = fake

    @torch.no_grad()
    def _bert_score_batch(self, texts: list[str]) -> list[float]:
        """Batch BERT scoring for efficiency."""
        if self.bert_model is None:
            return [0.5] * len(texts)

        results = []
        for i in range(0, len(texts), settings.batch_size):
            batch = texts[i:i + settings.batch_size]
            encoding = self.tokenizer(
                batch,
                max_length=settings.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            ids = encoding["input_ids"].to(self.device)
            mask = encoding["attention_mask"].to(self.device)

            out = self.bert_model(ids, attention_mask=mask)
            probs = torch.softmax(out.logits, dim=1)
            results.extend(probs[:, 1].cpu().numpy().tolist())

        return results

    def _manual_rule_score(self, features: dict) -> float:
        """Get manual rule-based fake probability. Returns 0.0-1.0."""
        scores = compute_manual_rule_score(
            text=features.get("text_clean", ""),
            star_rating=features.get("rating", 3),
            sentiment_polarity=features.get("sentiment_polarity", 0.0),
            verified_purchase=features.get("verified_purchase", True),
            has_media=features.get("has_media", False),
            helpful_votes=features.get("helpful_votes", 0),
            word_count=features.get("word_count"),
        )
        return scores["combined_manual_score"]

    def _learned_rule_score(self, features: dict) -> float:
        """Get learned rule fake probability. Returns 0.0-1.0."""
        if self.learned_rules is None:
            return 0.5

        return self.learned_rules.score_single(features)

    def predict(self, reviews: list[dict]) -> dict:
        """
        Full hybrid prediction pipeline.
        
        Args:
            reviews: list of dicts from Chrome extension, each with:
                - review_text: str
                - star_rating: int
                - verified_purchase: bool (optional)
                - helpful_votes: int (optional)
                - has_media: bool (optional)
        
        Returns:
            dict with review_scores and aggregate_score
        """
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        # Preprocess
        processed = preprocess_reviews(reviews)

        # BERT scores (batch)
        texts = [p["text_clean"] for p in processed]
        bert_scores = self._bert_score_batch(texts)

        # Rule scores + fusion
        review_scores = []
        for i, features in enumerate(processed):
            # Rule scores
            manual_score = self._manual_rule_score(features)
            learned_score = self._learned_rule_score(features)
            rule_score = 0.4 * manual_score + 0.6 * learned_score

            # BERT score
            bert_score = bert_scores[i]

            # Fusion: weighted average
            fused_fake_prob = (
                settings.rule_weight * rule_score
                + settings.bert_weight * bert_score
            )

            # Convert to 0-10 genuineness score
            genuineness = round((1 - fused_fake_prob) * 10, 1)
            genuineness = max(0.0, min(10.0, genuineness))

            # Scores on 0-10 scale for response
            rule_10 = round((1 - rule_score) * 10, 1)
            bert_10 = round((1 - bert_score) * 10, 1)

            # Divergence check
            divergence = abs(rule_10 - bert_10)
            flags = []
            if divergence > settings.divergence_threshold:
                flags.append("rule_ml_divergence")

            # Label
            if flags and "rule_ml_divergence" in flags:
                label = "Uncertain"
            elif genuineness >= 8:
                label = "Likely Genuine"
            elif genuineness >= 5:
                label = "Uncertain"
            else:
                label = "Likely Fake"

            review_scores.append({
                "score": genuineness,
                "label": label,
                "rule_score": rule_10,
                "bert_score": bert_10,
                "confidence": round(1 - (divergence / 10), 2),
                "flags": flags,
            })

        # Aggregate product score
        all_scores = [r["score"] for r in review_scores]
        avg_score = round(float(np.mean(all_scores)), 1) if all_scores else 5.0

        if avg_score >= 8:
            agg_label = "Likely Genuine"
        elif avg_score >= 5:
            agg_label = "Uncertain"
        else:
            agg_label = "Likely Fake"

        return {
            "review_scores": review_scores,
            "aggregate_score": avg_score,
            "aggregate_label": agg_label,
            "total_reviews_analyzed": len(review_scores),
        }


# Singleton — created once, shared across all requests
engine = InferenceEngine()