"""
Genzon — Predict Endpoint
POST /api/v1/predict — accepts reviews, returns genuineness scores.
"""

from fastapi import APIRouter, HTTPException

from backend.app.schemas.request import PredictRequest
from backend.app.schemas.response import PredictResponse, ReviewScore
from backend.app.services.inference import engine

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
async def predict_reviews(request: PredictRequest):
    """
    Analyze reviews and return genuineness scores.
    
    Accepts a list of reviews from the Chrome extension.
    Returns per-review scores (0-10) and an aggregate product score.
    """
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet. Server is starting up.")

    if not request.reviews:
        raise HTTPException(status_code=400, detail="No reviews provided.")

    if len(request.reviews) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 reviews per request.")

    # Convert Pydantic models to dicts for the inference engine
    reviews_data = [r.model_dump() for r in request.reviews]

    try:
        result = engine.predict(reviews_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    # Convert to response schema
    review_scores = [
        ReviewScore(**score) for score in result["review_scores"]
    ]

    return PredictResponse(
        review_scores=review_scores,
        aggregate_score=result["aggregate_score"],
        aggregate_label=result["aggregate_label"],
        total_reviews_analyzed=result["total_reviews_analyzed"],
    )