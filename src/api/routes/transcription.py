"""POST /api/v1/transcribe - upload audio, get text back."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, File, Query, UploadFile
from loguru import logger

from src.api.schemas import TranscribeResponse
from src.speech_to_text.audio_preprocessor import AudioPreprocessor
from src.speech_to_text.engine_factory import EngineType, create_engine
from src.speech_to_text.medical_spell_checker import MedicalSpellChecker

router = APIRouter(prefix="/api/v1", tags=["transcription"])

# cache engines so we don't reload on every request
_engine_cache: dict[EngineType, object] = {}
_spell_checker = MedicalSpellChecker()
_preprocessor = AudioPreprocessor()


def _get_engine(engine_type: EngineType):
    if engine_type not in _engine_cache:
        engine = create_engine(engine_type)
        engine.load_model()
        _engine_cache[engine_type] = engine
    return _engine_cache[engine_type]


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file (WAV, MP3, FLAC, OGG)"),
    engine: EngineType = Query(
        default=EngineType.MEDASR,
        description="ASR engine. Default is a baseline (MedASR) so the endpoint is "
        "turn-key without the private PRAXIS-AUSCULT checkpoint; pass "
        "'praxis_conformer' to run the contributed model (needs best_primock57.pt).",
    ),
    apply_spell_check: bool = Query(default=True, description="Apply medical spell-checker"),
):
    """Upload audio, get transcription back."""
    # Save upload to a temp file
    suffix = Path(file.filename).suffix if file.filename else ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_path = tmp.name
    try:
        content = await file.read()
        tmp.write(content)
        tmp.close()
        # Preprocess
        preprocessed = _preprocessor.process_file(tmp_path)

        # Transcribe
        asr_engine = _get_engine(engine)
        t0 = time.perf_counter()
        text = asr_engine.transcribe(preprocessed.audio, preprocessed.sample_rate)
        inference_time = time.perf_counter() - t0

        # Post-process
        if apply_spell_check:
            text = _spell_checker.correct(text)

        logger.info(
            f"Transcribed {file.filename} via {engine.value}: "
            f"{len(text)} chars in {inference_time:.2f}s"
        )

        return TranscribeResponse(
            text=text,
            engine=engine.value,
            duration_secs=round(preprocessed.duration_secs, 2),
            processing_time_secs=round(inference_time, 3),
            spell_checked=apply_spell_check,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)
