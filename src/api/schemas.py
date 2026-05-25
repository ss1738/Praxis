"""API request/response models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.speech_to_text.engine_factory import EngineType


class TranscribeRequest(BaseModel):
    """Transcription request params."""

    engine: EngineType = Field(default=EngineType.MEDASR, description="ASR engine to use")
    apply_spell_check: bool = Field(default=True, description="Apply medical spell-checker")


class TranscribeResponse(BaseModel):
    """Transcription result."""

    text: str = Field(..., description="Transcribed text")
    engine: str = Field(..., description="Engine used for transcription")
    duration_secs: float = Field(..., description="Audio duration in seconds")
    processing_time_secs: float = Field(..., description="Inference wall-clock time")
    spell_checked: bool = Field(..., description="Whether medical spell-check was applied")


class HealthResponse(BaseModel):
    """Health check."""

    status: str = "ok"
    version: str = "0.1.0"


class ErrorResponse(BaseModel):
    """Error response."""

    detail: str
