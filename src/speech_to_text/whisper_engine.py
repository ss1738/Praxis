"""Whisper engine - supports base models and LoRA fine-tuned versions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from loguru import logger

from src.config.settings import get_settings
from src.speech_to_text.audio_preprocessor import AudioPreprocessor


class WhisperEngine:
    """Whisper ASR, can optionally merge LoRA adapters."""

    def __init__(
        self,
        model_size: str | None = None,
        adapter_path: str | Path | None = None,
        device: str | None = None,
        use_faster_whisper: bool = False,
    ):
        settings = get_settings()
        self._model_size = model_size or settings.whisper_model_size
        self._adapter_path = Path(adapter_path) if adapter_path else None
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._use_faster = use_faster_whisper
        self._model = None
        self._processor = None
        self._preprocessor = AudioPreprocessor()
        logger.info(
            f"Whisper engine initialised (size={self._model_size}, "
            f"adapter={'yes' if self._adapter_path else 'none'}, "
            f"backend={'faster-whisper' if self._use_faster else 'openai-whisper'})"
        )

    def load_model(self) -> None:
        """Load the model (and merge LoRA weights if we have them)."""
        if self._use_faster:
            self._load_faster_whisper()
        else:
            self._load_hf_whisper()

    # HF transformers backend

    def _load_hf_whisper(self) -> None:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        model_id = f"openai/whisper-{self._model_size}"
        logger.info(f"Loading HF Whisper: {model_id}")

        self._processor = WhisperProcessor.from_pretrained(model_id)
        model = WhisperForConditionalGeneration.from_pretrained(model_id).to(self._device)

        if self._adapter_path and self._adapter_path.exists():
            logger.info(f"Merging QLoRA adapter from {self._adapter_path}")
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(self._adapter_path))
            model = model.merge_and_unload()

        model.eval()
        self._model = model
        logger.info("HF Whisper loaded")

    # faster-whisper backend (CTranslate2)

    def _load_faster_whisper(self) -> None:
        from faster_whisper import WhisperModel

        compute_type = "float16" if self._device == "cuda" else "int8"
        logger.info(f"Loading faster-whisper {self._model_size} ({compute_type})")
        self._model = WhisperModel(
            self._model_size,
            device=self._device,
            compute_type=compute_type,
        )
        logger.info("faster-whisper loaded")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16_000) -> str:
        """Transcribe audio array, returns text."""
        if self._model is None:
            self.load_model()

        preprocessed = self._preprocessor.process_array(audio, sample_rate)

        if self._use_faster:
            return self._transcribe_faster(preprocessed.audio)
        return self._transcribe_hf(preprocessed.audio, preprocessed.sample_rate)

    def _transcribe_hf(self, audio: np.ndarray, sr: int) -> str:
        inputs = self._processor(audio, sampling_rate=sr, return_tensors="pt").to(self._device)
        forced_ids = self._processor.get_decoder_prompt_ids(language="en", task="transcribe")
        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                num_beams=5,
                forced_decoder_ids=forced_ids,
                max_new_tokens=440,
            )
        text: str = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()

    def _transcribe_faster(self, audio: np.ndarray) -> str:
        segments, _ = self._model.transcribe(audio, beam_size=5)
        return " ".join(seg.text.strip() for seg in segments)

    def transcribe_file(self, path: str | Path) -> str:
        preprocessed = self._preprocessor.process_file(str(path))
        return self.transcribe(preprocessed.audio, preprocessed.sample_rate)
