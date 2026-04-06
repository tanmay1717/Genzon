"""
Genzon — FastAPI Backend
Amazon Fake Review Detector API

Usage:
    cd genzon
    uvicorn backend.app.main:app --reload --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.routes import predict, health
from backend.app.services.inference import engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models when server starts, cleanup when it stops."""
    print("\n🚀 Starting Genzon API server...")
    engine.load_models()
    yield
    print("\n👋 Shutting down Genzon API server...")


app = FastAPI(
    title="Genzon — Fake Review Detector API",
    version="0.1.0",
    description="ML-powered fake review detection for Amazon products",
    lifespan=lifespan,
)

# CORS — allow Chrome extension and local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "chrome-extension://*",
        "http://localhost:*",
        "http://127.0.0.1:*",
        "*",  # for development — restrict in production
    ],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# Routes
app.include_router(health.router)
app.include_router(predict.router, prefix="/api/v1")


# Root
@app.get("/")
async def root():
    return {
        "name": "Genzon — Fake Review Detector",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/api/v1/predict",
    }