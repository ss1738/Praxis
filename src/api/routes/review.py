"""GP blind-review API routes - Likert scoring + inter-rater kappa."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.clinical.gp_scoring import (
    GPScoringSession,
    ScoringCriteria,
    compute_cohens_kappa,
)

router = APIRouter(prefix="/api/v1", tags=["review"])

_OUTPUT_DIR = Path("data/gp_scores")
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class ReviewRequest(BaseModel):
    gp_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    model_id: str = Field(default="unknown")
    soap_text: str = Field(default="")
    transcript: str = Field(default="")
    clinical_accuracy: int = Field(..., ge=1, le=5)
    completeness: int = Field(..., ge=1, le=5)
    structure_quality: int = Field(..., ge=1, le=5)
    language_appropriateness: int = Field(..., ge=1, le=5)
    safety_netting: int = Field(..., ge=1, le=5)
    coding_quality: int = Field(..., ge=1, le=5)
    overall_acceptability: int = Field(..., ge=1, le=5)
    comments: str = ""
    would_edit: bool = False
    critical_errors: list[str] = Field(default_factory=list)


def _load_gp_file(gp_id: str) -> dict:
    path = _OUTPUT_DIR / f"gp_{gp_id}_scores.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"gp_id": gp_id, "notes_scored": 0, "averages": {}, "scores": []}


@router.post("/reviews")
async def submit_review(req: ReviewRequest):
    """Record a GP Likert review for one SOAP note."""
    criteria = ScoringCriteria(
        clinical_accuracy=req.clinical_accuracy,
        completeness=req.completeness,
        structure_quality=req.structure_quality,
        language_appropriateness=req.language_appropriateness,
        safety_netting=req.safety_netting,
        coding_quality=req.coding_quality,
        overall_acceptability=req.overall_acceptability,
        comments=req.comments,
        would_edit=req.would_edit,
        critical_errors=req.critical_errors,
    )

    session = GPScoringSession(gp_id=req.gp_id, output_dir=_OUTPUT_DIR)
    existing = _load_gp_file(req.gp_id)
    session.scored_notes = list(existing.get("scores", []))

    record = session.score_note(
        session_id=req.session_id,
        soap_text=req.soap_text,
        model_id=req.model_id,
        transcript=req.transcript,
        scores=criteria,
    )
    session.export_scores()
    return {"saved": True, "record": record, "averages": session.compute_averages()}


@router.get("/reviews")
async def list_reviews(gp_id: str | None = None):
    """List all reviews, optionally filtered by gp_id, with aggregate averages."""
    files = sorted(_OUTPUT_DIR.glob("gp_*_scores.json"))
    all_reviews: list[dict] = []
    per_gp: dict[str, dict] = {}
    for f in files:
        data = json.loads(f.read_text())
        per_gp[data["gp_id"]] = {
            "notes_scored": data.get("notes_scored", 0),
            "averages": data.get("averages", {}),
        }
        if gp_id is None or data["gp_id"] == gp_id:
            all_reviews.extend(data.get("scores", []))

    return {
        "total_reviews": len(all_reviews),
        "per_gp": per_gp,
        "reviews": all_reviews,
    }


@router.get("/reviews/kappa")
async def inter_rater_kappa(gp_a: str, gp_b: str):
    """Compute weighted Cohen's kappa between two GPs' scoring files."""
    file_a = _OUTPUT_DIR / f"gp_{gp_a}_scores.json"
    file_b = _OUTPUT_DIR / f"gp_{gp_b}_scores.json"
    if not file_a.exists() or not file_b.exists():
        raise HTTPException(status_code=404, detail="One or both GP score files not found")
    return compute_cohens_kappa(file_a, file_b)
