"""
Genzon — API Request Schemas
"""

from pydantic import BaseModel


class ReviewInput(BaseModel):
    review_text: str
    star_rating: int = 3
    verified_purchase: bool = True
    helpful_votes: int = 0
    has_media: bool = False
    review_date: str | None = None


class PredictRequest(BaseModel):
    reviews: list[ReviewInput]