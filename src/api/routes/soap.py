"""POST /api/v1/soap - generate clinical notes from transcripts."""

from __future__ import annotations

from fastapi import APIRouter
from loguru import logger
from pydantic import BaseModel, Field

from src.soap_generation.constitutional import ConstitutionalSOAPGenerator
from src.soap_generation.generator import SOAPGenerator, SOAPModel
from src.soap_generation.templates import NoteTemplate, TEMPLATE_DISPLAY_NAMES, TEMPLATE_DESCRIPTIONS

router = APIRouter(prefix="/api/v1", tags=["soap"])

_generator_cache: dict[str, SOAPGenerator] = {}
_constitutional_cache: dict[str, ConstitutionalSOAPGenerator] = {}


def _get_generator(model: SOAPModel) -> SOAPGenerator:
    if model.value not in _generator_cache:
        _generator_cache[model.value] = SOAPGenerator(model=model)
    return _generator_cache[model.value]


def _get_constitutional(model: SOAPModel, template: NoteTemplate) -> ConstitutionalSOAPGenerator:
    key = f"{model.value}:{template.value}"
    if key not in _constitutional_cache:
        _constitutional_cache[key] = ConstitutionalSOAPGenerator(model=model, template=template)
    return _constitutional_cache[key]


class SOAPRequest(BaseModel):
    transcript: str = Field(..., description="GP consultation transcript")
    model: SOAPModel = Field(default=SOAPModel.PRAXIS_PRIMARY, description="Model to use")
    template: NoteTemplate = Field(default=NoteTemplate.SOAP, description="Note template format")


class SOAPResponse(BaseModel):
    soap_note: str
    model: str
    template: str
    generation_time_secs: float
    token_count: int


class ConstitutionalRequest(BaseModel):
    transcript: str = Field(..., description="GP consultation transcript")
    model: SOAPModel = Field(default=SOAPModel.PRAXIS_PRIMARY, description="Model to use")
    template: NoteTemplate = Field(default=NoteTemplate.SOAP, description="Note template format")
    max_iterations: int = Field(default=3, ge=1, le=5, description="Max self-correction iterations")


class ConstitutionalIterationResponse(BaseModel):
    iteration: int
    trust_score: int
    verdict: str
    checks_passed: int
    checks_total: int
    issues: list[str]
    repair_targets: list[str]


class ConstitutionalResponse(BaseModel):
    soap_note: str
    model: str
    template: str
    trust_score: int
    verdict: str
    converged: bool
    num_iterations: int
    score_trajectory: list[int]
    iterations: list[ConstitutionalIterationResponse]
    total_time_secs: float
    token_count: int


@router.post("/soap", response_model=SOAPResponse)
async def generate_soap(request: SOAPRequest):
    """Generate a clinical note from a consultation transcript."""
    generator = _get_generator(request.model)
    result = generator.generate(request.transcript, template=request.template)

    return SOAPResponse(
        soap_note=result.soap_note,
        model=result.model,
        template=result.template,
        generation_time_secs=result.generation_time_secs,
        token_count=result.token_count,
    )


@router.post("/soap/constitutional", response_model=ConstitutionalResponse)
async def generate_constitutional(request: ConstitutionalRequest):
    """Generate a clinical note with iterative constitutional self-correction.

    Runs TrustScore checks after each generation attempt and surgically repairs
    failing sections until the note reaches 'approved' (TrustScore ≥ 70) or
    max_iterations is exhausted. Returns the best-scoring note seen.
    """
    gen = ConstitutionalSOAPGenerator(
        model=request.model,
        template=request.template,
        max_iterations=request.max_iterations,
    )
    result = gen.generate(request.transcript)

    return ConstitutionalResponse(
        soap_note=result.final_note,
        model=result.model,
        template=result.template,
        trust_score=result.final_trust_score,
        verdict=result.final_verdict,
        converged=result.converged,
        num_iterations=result.num_iterations,
        score_trajectory=result.score_trajectory,
        iterations=[
            ConstitutionalIterationResponse(
                iteration=it.iteration,
                trust_score=it.trust_score,
                verdict=it.verdict,
                checks_passed=it.checks_passed,
                checks_total=it.checks_total,
                issues=it.issues,
                repair_targets=it.repair_targets,
            )
            for it in result.iterations
        ],
        total_time_secs=result.total_time_secs,
        token_count=result.token_count,
    )


@router.get("/soap/models")
async def list_models():
    """List available models."""
    gen = SOAPGenerator()
    available = gen.list_models()
    return {
        "models": available,
        "supported": [m.value for m in SOAPModel],
    }


@router.get("/soap/templates")
async def list_templates():
    """List available note templates."""
    return {
        "templates": [
            {
                "id": t.value,
                "name": TEMPLATE_DISPLAY_NAMES[t.value],
                "description": TEMPLATE_DESCRIPTIONS[t.value],
            }
            for t in NoteTemplate
        ]
    }
