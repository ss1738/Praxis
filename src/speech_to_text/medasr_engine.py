"""Google MedASR engine. CTC-based, 105M params. Needs HF_TOKEN (gated model)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from src.config.settings import get_settings
from src.speech_to_text.audio_preprocessor import AudioPreprocessor


def _ctc_greedy_decode(logits: torch.Tensor, tokenizer, blank_id: int = 0) -> str:
    """argmax -> collapse repeats -> remove blanks -> decode tokens"""
    pred_ids = torch.argmax(logits, dim=-1)[0]  # (T,)
    collapsed = []
    prev = -1
    for idx in pred_ids.tolist():
        if idx != prev:
            if idx != blank_id:
                collapsed.append(idx)
        prev = idx
    return tokenizer.decode(collapsed)


def normalize_medasr_output(text: str) -> str:
    """Clean up MedASR output for WER comparison (same as their eval notebook)."""
    text = text.lower()
    text = text.replace("</s>", "")
    text = re.sub(r"[^ a-z0-9']", " ", text)
    return " ".join(text.split())


class MedASREngine:
    """Google MedASR wrapper."""

    def __init__(self, model_id: str | None = None, device: str | None = None):
        self._model_id = model_id or get_settings().medasr_model_id
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = None
        self._model = None
        self._preprocessor = AudioPreprocessor()
        logger.info(f"MedASR engine initialised (model={self._model_id}, device={self._device})")

    def load_model(self) -> None:
        from transformers import AutoModelForCTC, AutoProcessor

        token = get_settings().hf_token or None
        logger.info(f"Loading MedASR model: {self._model_id}")
        self._processor = AutoProcessor.from_pretrained(self._model_id, token=token)
        self._model = AutoModelForCTC.from_pretrained(self._model_id, token=token).to(self._device)
        self._model.eval()
        logger.info("MedASR model loaded successfully")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16_000) -> str:
        """Run CTC greedy decode on audio, return raw text."""
        if self._model is None:
            self.load_model()

        preprocessed = self._preprocessor.process_array(audio, sample_rate)
        inputs = self._processor(
            preprocessed.audio,
            sampling_rate=preprocessed.sample_rate,
            return_tensors="pt",
            padding=True,
        ).to(self._device)

        with torch.no_grad():
            logits = self._model(**inputs).logits

        text = _ctc_greedy_decode(logits, self._processor.tokenizer, blank_id=0)
        logger.debug(f"MedASR transcription ({preprocessed.duration_secs:.1f}s): {text[:80]}...")
        return text.strip()

    def transcribe_file(self, path: str | Path) -> str:
        preprocessed = self._preprocessor.process_file(str(path))
        return self.transcribe(preprocessed.audio, preprocessed.sample_rate)
