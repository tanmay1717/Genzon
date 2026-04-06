"""
Genzon — Preprocessing Service
Cleans incoming review data from the Chrome extension before inference.
"""

import re
import string

from textblob import TextBlob


def clean_text(text: str) -> str:
    """Clean review text for model input."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    return text


def extract_features(text: str, star_rating: int = 3) -> dict:
    """
    Extract all features from a single review.
    Returns a dict matching the columns the model expects.
    """
    cleaned = clean_text(text)
    words = cleaned.split()
    word_count = len(words)

    # Sentiment
    blob = TextBlob(cleaned)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    # Star-sentiment gap
    rating_normalized = (star_rating - 3) / 2
    gap = abs(rating_normalized - polarity)

    # Text features
    features = {
        "text": text,
        "text_clean": cleaned,
        "char_count": len(cleaned),
        "word_count": word_count,
        "avg_word_length": len(cleaned) / max(word_count, 1),
        "caps_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
        "exclamation_count": text.count("!"),
        "question_count": text.count("?"),
        "punct_ratio": sum(1 for c in text if c in string.punctuation) / max(len(text), 1),
        "unique_word_ratio": len(set(w.lower() for w in words)) / max(word_count, 1),
        "sentence_count": max(len(re.split(r"[.!?]+", text)), 1),
        "avg_sentence_length": word_count / max(len(re.split(r"[.!?]+", text)), 1),
        "first_person_ratio": sum(
            1 for w in words if w.lower() in {"i", "me", "my", "mine", "myself", "we", "our", "ours"}
        ) / max(word_count, 1),
        "sentiment_polarity": polarity,
        "sentiment_subjectivity": subjectivity,
        "star_sentiment_gap": min(gap / 2.0, 1.0),
        "rating": star_rating,
    }

    return features


def preprocess_reviews(reviews: list[dict]) -> list[dict]:
    """
    Process a batch of reviews from the Chrome extension.
    
    Each review dict should have at least:
        - review_text: str
        - star_rating: int (1-5)
    Optional:
        - verified_purchase: bool
        - helpful_votes: int
        - has_media: bool
    """
    processed = []
    for review in reviews:
        features = extract_features(
            text=review.get("review_text", ""),
            star_rating=review.get("star_rating", 3),
        )
        # Pass through extra fields from the extension
        features["verified_purchase"] = review.get("verified_purchase", True)
        features["helpful_votes"] = review.get("helpful_votes", 0)
        features["has_media"] = review.get("has_media", False)
        features["review_date"] = review.get("review_date")

        processed.append(features)

    return processed