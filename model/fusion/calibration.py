"""
Genzon — Fusion Calibration & Threshold Tuning
Finds optimal fusion weights, decision threshold, and calibrates probabilities.

Usage:
    python -m model.fusion.calibration
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.utils.metrics import compute_metrics, print_metrics


class FusionCalibrator:
    """
    Calibrates the hybrid fusion model by:
    1. Finding optimal fusion weights (rule vs BERT)
    2. Finding optimal decision threshold
    3. Calibrating probabilities using isotonic regression
    """

    def __init__(self):
        self.best_rule_weight = 0.35
        self.best_bert_weight = 0.65
        self.best_threshold = 5.0  # on 0-10 scale
        self.calibrator = None     # isotonic regression
        self.is_calibrated = False

    def find_best_weights(
        self,
        rule_scores: np.ndarray,
        bert_scores: np.ndarray,
        y_true: np.ndarray,
        metric: str = "f1",
    ) -> dict:
        """
        Grid search over fusion weights to find the best combination.

        Args:
            rule_scores: rule-based fake probabilities (0.0 to 1.0)
            bert_scores: BERT fake probabilities (0.0 to 1.0)
            y_true: ground truth labels (0=genuine, 1=fake)
            metric: which metric to optimize ("f1", "auc_roc", "precision", "recall")

        Returns:
            dict with best weights and scores at each combination
        """
        print("\n  Searching for optimal fusion weights...")

        best_score = 0.0
        best_rw = 0.35
        results = []

        # Try all weight combinations in steps of 0.05
        for rule_w in np.arange(0.0, 1.05, 0.05):
            bert_w = 1.0 - rule_w

            # Fuse
            fused = rule_w * rule_scores + bert_w * bert_scores

            # Convert to predictions using current threshold (0.5 on probability scale)
            y_pred = (fused >= 0.5).astype(int)

            # Compute metric
            if metric == "f1":
                score = f1_score(y_true, y_pred)
            elif metric == "auc_roc":
                score = roc_auc_score(y_true, fused)
            elif metric == "precision":
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == "recall":
                score = recall_score(y_true, y_pred, zero_division=0)
            else:
                score = f1_score(y_true, y_pred)

            results.append({
                "rule_weight": round(rule_w, 2),
                "bert_weight": round(bert_w, 2),
                metric: round(score, 4),
            })

            if score > best_score:
                best_score = score
                best_rw = rule_w

        self.best_rule_weight = round(best_rw, 2)
        self.best_bert_weight = round(1.0 - best_rw, 2)

        print(f"  ✓ Best weights: rule={self.best_rule_weight}, bert={self.best_bert_weight}")
        print(f"    Best {metric}: {best_score:.4f}")

        # Show top 5
        results_df = pd.DataFrame(results).sort_values(metric, ascending=False)
        print(f"\n  Top 5 weight combinations:")
        print(results_df.head().to_string(index=False))

        return {
            "best_rule_weight": self.best_rule_weight,
            "best_bert_weight": self.best_bert_weight,
            "best_score": best_score,
            "all_results": results,
        }

    def find_best_threshold(
        self,
        fused_scores: np.ndarray,
        y_true: np.ndarray,
        metric: str = "f1",
    ) -> dict:
        """
        Find optimal decision threshold on the 0-10 genuineness scale.

        Args:
            fused_scores: fused fake probabilities (0.0 to 1.0)
            y_true: ground truth labels (0=genuine, 1=fake)

        Returns:
            dict with best threshold and metrics at each threshold
        """
        print("\n  Searching for optimal threshold...")

        best_score = 0.0
        best_thresh = 0.5
        results = []

        for thresh in np.arange(0.1, 0.9, 0.01):
            y_pred = (fused_scores >= thresh).astype(int)

            f1 = f1_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)

            results.append({
                "threshold_prob": round(thresh, 2),
                "threshold_10_scale": round((1 - thresh) * 10, 1),
                "f1": round(f1, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
            })

            current = f1 if metric == "f1" else prec if metric == "precision" else rec
            if current > best_score:
                best_score = current
                best_thresh = thresh

        # Convert to 0-10 scale (genuineness)
        self.best_threshold = round((1 - best_thresh) * 10, 1)

        print(f"  ✓ Best threshold: {best_thresh:.2f} (probability) = {self.best_threshold}/10 (genuineness)")
        print(f"    Best {metric}: {best_score:.4f}")

        return {
            "best_threshold_prob": round(best_thresh, 2),
            "best_threshold_genuineness": self.best_threshold,
            "best_score": best_score,
            "all_results": results,
        }

    def calibrate_probabilities(
        self,
        fused_scores: np.ndarray,
        y_true: np.ndarray,
    ):
        """
        Fit isotonic regression to calibrate model probabilities.
        
        After calibration, if the model says "70% fake", it actually
        IS fake ~70% of the time (not just a raw softmax number).

        Args:
            fused_scores: fused fake probabilities (0.0 to 1.0)
            y_true: ground truth labels
        """
        print("\n  Calibrating probabilities (isotonic regression)...")

        self.calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        self.calibrator.fit(fused_scores, y_true)
        self.is_calibrated = True

        # Show before vs after
        calibrated = self.calibrator.predict(fused_scores)

        print(f"  ✓ Calibration fitted on {len(fused_scores)} samples")
        print(f"    Raw scores     — mean: {fused_scores.mean():.3f}, std: {fused_scores.std():.3f}")
        print(f"    Calibrated     — mean: {calibrated.mean():.3f}, std: {calibrated.std():.3f}")

    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        """Apply calibration to new scores."""
        if not self.is_calibrated:
            return scores
        return self.calibrator.predict(scores)

    def full_calibration(
        self,
        rule_scores: np.ndarray,
        bert_scores: np.ndarray,
        y_true: np.ndarray,
    ) -> dict:
        """
        Run the complete calibration pipeline:
        1. Find best weights
        2. Fuse with best weights
        3. Calibrate probabilities
        4. Find best threshold
        5. Compute final metrics
        """
        print("=" * 55)
        print("  Genzon — Fusion Calibration")
        print("=" * 55)

        # Step 1: Best weights
        weight_results = self.find_best_weights(rule_scores, bert_scores, y_true)

        # Step 2: Fuse with best weights
        fused = self.best_rule_weight * rule_scores + self.best_bert_weight * bert_scores

        # Step 3: Calibrate
        self.calibrate_probabilities(fused, y_true)
        fused_calibrated = self.calibrate(fused)

        # Step 4: Best threshold (on calibrated scores)
        thresh_results = self.find_best_threshold(fused_calibrated, y_true)

        # Step 5: Final metrics with optimal settings
        best_thresh_prob = thresh_results["best_threshold_prob"]
        y_pred = (fused_calibrated >= best_thresh_prob).astype(int)
        final_metrics = compute_metrics(y_true, y_pred, fused_calibrated)

        print_metrics(final_metrics, title="Final Calibrated Results")

        return {
            "weights": weight_results,
            "threshold": thresh_results,
            "final_metrics": final_metrics,
            "config": {
                "rule_weight": self.best_rule_weight,
                "bert_weight": self.best_bert_weight,
                "threshold_genuineness": self.best_threshold,
                "threshold_probability": best_thresh_prob,
            },
        }

    def save(self, path: str | Path):
        """Save calibration config to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "best_rule_weight": self.best_rule_weight,
            "best_bert_weight": self.best_bert_weight,
            "best_threshold": self.best_threshold,
            "calibrator": self.calibrator,
            "is_calibrated": self.is_calibrated,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"  ✓ Saved calibration to {path}")

        # Also save human-readable config
        config = {
            "rule_weight": self.best_rule_weight,
            "bert_weight": self.best_bert_weight,
            "threshold_genuineness_scale": self.best_threshold,
        }
        config_path = path.with_suffix(".json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"  ✓ Saved config to {config_path}")

    def load(self, path: str | Path):
        """Load calibration from disk."""
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.best_rule_weight = data["best_rule_weight"]
        self.best_bert_weight = data["best_bert_weight"]
        self.best_threshold = data["best_threshold"]
        self.calibrator = data["calibrator"]
        self.is_calibrated = data["is_calibrated"]
        print(f"  ✓ Loaded calibration from {path}")
        return self


# ─── Main ────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch
    from transformers import BertForSequenceClassification, BertTokenizer
    from data.config import PROCESSED_DIR
    from model.rule_engine.manual_rules import compute_manual_rule_score
    from model.rule_engine.learned_rules import LearnedRuleScorer

    # Load validation data
    val = pd.read_csv(PROCESSED_DIR / "val.csv")
    print(f"Validation set: {len(val)} reviews\n")

    y_true = val["label_encoded"].values

    # ── Get rule scores ──
    print("Computing rule scores...")

    # Manual rules
    manual_scores = []
    for _, row in val.iterrows():
        s = compute_manual_rule_score(
            text=str(row.get("text_clean", "")),
            star_rating=int(row.get("rating", 3)),
            sentiment_polarity=float(row.get("sentiment_polarity", 0.0)),
            word_count=int(row.get("word_count", 0)),
        )
        manual_scores.append(s["combined_manual_score"])
    manual_scores = np.array(manual_scores)

    # Learned rules
    rules_path = PROJECT_ROOT / "model" / "checkpoints" / "learned_rules.pkl"
    if rules_path.exists():
        learned = LearnedRuleScorer()
        learned.load(rules_path)
        learned_scores = learned.predict_proba(val)
    else:
        print("  ⚠ No learned rules found, using manual only")
        learned_scores = manual_scores

    rule_scores = 0.4 * manual_scores + 0.6 * learned_scores

    # ── Get BERT scores ──
    print("Computing BERT scores (this may take a while on CPU)...")

    bert_path = PROJECT_ROOT / "model" / "checkpoints" / "bert_best"
    if bert_path.exists():
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = BertForSequenceClassification.from_pretrained(str(bert_path))
        tokenizer = BertTokenizer.from_pretrained(str(bert_path))
        model.to(device)
        model.eval()

        bert_scores = []
        texts = val["text_clean"].fillna("").tolist()
        batch_size = 16

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                enc = tokenizer(batch, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
                out = model(enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device))
                probs = torch.softmax(out.logits, dim=1)
                bert_scores.extend(probs[:, 1].cpu().numpy().tolist())

                done = min(i + batch_size, len(texts))
                print(f"\r  {done}/{len(texts)} ({done/len(texts)*100:.0f}%)", end="", flush=True)

        bert_scores = np.array(bert_scores)
        print()
    else:
        print("  ⚠ No BERT model found, using rule scores only")
        bert_scores = rule_scores

    # ── Run calibration ──
    calibrator = FusionCalibrator()
    results = calibrator.full_calibration(rule_scores, bert_scores, y_true)

    # Save
    save_path = PROJECT_ROOT / "model" / "checkpoints" / "calibration.pkl"
    calibrator.save(save_path)

    # Print final config
    print(f"\n{'=' * 55}")
    print("  Optimal Configuration")
    print(f"{'=' * 55}")
    config = results["config"]
    print(f"  Rule weight:  {config['rule_weight']}")
    print(f"  BERT weight:  {config['bert_weight']}")
    print(f"  Threshold:    {config['threshold_genuineness']}/10 genuineness")
    print(f"\n  Update these values in backend/app/config.py:")
    print(f"    rule_weight = {config['rule_weight']}")
    print(f"    bert_weight = {config['bert_weight']}")
    print(f"    # threshold = {config['threshold_genuineness']} (on 0-10 scale)")