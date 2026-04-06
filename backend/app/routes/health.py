"""
Genzon — Health Check Endpoint
GET /health — check if server and models are ready.
"""

from fastapi import APIRouter
from backend.app.services.inference import engine

router = APIRouter()


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": engine.is_loaded,
        "bert_available": engine.bert_model is not None,
        "rules_available": engine.learned_rules is not None,
        "device": str(engine.device) if engine.device else "not set",
    }