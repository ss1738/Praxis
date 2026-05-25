"""Pick which STT engine to use."""

from __future__ import annotations

from enum import Enum
from typing import Protocol

import numpy as np


class ASREngine(Protocol):

    def load_model(self) -> None: ...
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16_000) -> str: ...
    def transcribe_file(self, path: str) -> str: ...


class EngineType(str, Enum):

    MEDASR = "medasr"
    WHISPER = "whisper"
    FASTER_WHISPER = "faster_whisper"
    # praxis_conformer is the internal engine identifier for the model branded
    # PRAXIS-AUSCULT in the paper and dissertation. Do not rename - many call sites,
    # registries, and result JSONs depend on this string.
    PRAXIS_CONFORMER = "praxis_conformer"


def create_engine(
    engine_type: EngineType = EngineType.MEDASR,
    **kwargs,
) -> ASREngine:
    # The default is a baseline (MedASR) rather than PRAXIS_CONFORMER on purpose:
    # baselines auto-download from Hugging Face, so the system is turn-key without
    # the private PRAXIS-AUSCULT checkpoint. Pass EngineType.PRAXIS_CONFORMER to run
    # the contributed model (requires best_primock57.pt locally; see README).
    if engine_type == EngineType.MEDASR:
        from src.speech_to_text.medasr_engine import MedASREngine
        return MedASREngine(**kwargs)

    if engine_type == EngineType.WHISPER:
        from src.speech_to_text.whisper_engine import WhisperEngine
        return WhisperEngine(use_faster_whisper=False, **kwargs)

    if engine_type == EngineType.FASTER_WHISPER:
        from src.speech_to_text.whisper_engine import WhisperEngine
        return WhisperEngine(use_faster_whisper=True, **kwargs)

    if engine_type == EngineType.PRAXIS_CONFORMER:
        from src.speech_to_text.conformer_engine import ConformerEngine
        return ConformerEngine(**kwargs)

    raise ValueError(f"Unknown engine type: {engine_type}")
