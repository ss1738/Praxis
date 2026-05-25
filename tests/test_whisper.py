"""Whisper engine tests (mocked)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from src.speech_to_text.whisper_engine import WhisperEngine


@pytest.fixture
def engine():
    """Engine without loading weights."""
    return WhisperEngine(model_size="tiny", device="cpu")


class TestWhisperEngine:
    """Whisper tests."""

    def test_init_defaults(self, engine):
        assert engine._model_size == "tiny"
        assert engine._device == "cpu"
        assert engine._use_faster is False
        assert engine._model is None

    def test_faster_whisper_flag(self):
        engine = WhisperEngine(use_faster_whisper=True, device="cpu")
        assert engine._use_faster is True

    def test_adapter_path_none_by_default(self, engine):
        assert engine._adapter_path is None

    def test_transcribe_file_missing(self, engine):
        with pytest.raises(FileNotFoundError):
            engine.transcribe_file("/nonexistent/audio.wav")
