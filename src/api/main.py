"""FastAPI app. Run with: uvicorn src.api.main:app --reload"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.api.middleware import register_middleware
from src.api.routes.transcription import router as transcription_router
from src.api.routes.soap import router as soap_router
from src.api.routes.patients import router as patients_router
from src.api.routes.sessions import router as sessions_router
from src.api.routes.review import router as review_router
from src.api.routes.features import router as features_router
from src.api.routes.features2 import router as features2_router
from src.api.schemas import HealthResponse

app = FastAPI(
    title="PRAXIS Clinical Scribe API",
    description="STT + SOAP generation API for GP consultations",
    version="0.3.0",
)

register_middleware(app)
app.include_router(transcription_router)
app.include_router(soap_router)
app.include_router(patients_router)
app.include_router(sessions_router)
app.include_router(review_router)
app.include_router(features_router)
app.include_router(features2_router)

# Serve the website from /website as static files
website_dir = Path(__file__).resolve().parent.parent.parent / "website"
if website_dir.exists():
    app.mount("/site", StaticFiles(directory=str(website_dir), html=True), name="website")


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check():
    """Health check."""
    return HealthResponse()
