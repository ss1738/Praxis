"""Audio preprocessing - resample, noise reduction, trim silence, normalise volume."""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np
from loguru import logger

from src.config.settings import get_settings
from src.utils.audio_utils import audio_to_int16, load_audio


@dataclass
class PreprocessedAudio:
    """Result of preprocessing an audio file."""

    audio: np.ndarray
    sample_rate: int
    duration_secs: float
    was_trimmed: bool
    was_resampled: bool


class AudioPreprocessor:
    """Resample to 16kHz, trim silence, normalise. Optional VAD filtering."""

    def __init__(self, target_sr: int | None = None, vad_aggressiveness: int = 2):
        self._target_sr = target_sr or get_settings().audio_sample_rate
        self._vad_aggressiveness = vad_aggressiveness

    def process_file(self, path: str) -> PreprocessedAudio:
        audio, sr = load_audio(path, target_sr=self._target_sr)
        return self._pipeline(audio, sr, resampled=True)

    def process_array(self, audio: np.ndarray, sr: int) -> PreprocessedAudio:
        import librosa

        resampled = False
        if sr != self._target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self._target_sr)
            sr = self._target_sr
            resampled = True
        return self._pipeline(audio, sr, resampled=resampled)


    def _pipeline(self, audio: np.ndarray, sr: int, resampled: bool) -> PreprocessedAudio:
        denoised = self._reduce_noise(audio, sr)
        trimmed = self._trim_silence(denoised, sr)
        was_trimmed = len(trimmed) < len(audio)
        normalised = self._normalise(trimmed)
        duration = len(normalised) / sr
        logger.info(f"Preprocessed: {duration:.1f}s, trimmed={was_trimmed}, resampled={resampled}")
        return PreprocessedAudio(
            audio=normalised,
            sample_rate=sr,
            duration_secs=duration,
            was_trimmed=was_trimmed,
            was_resampled=resampled,
        )

    @staticmethod
    def _reduce_noise(audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply spectral-gating noise reduction to remove background noise."""
        # Spectral gating needs a buffer at least one FFT window long; very short
        # clips make noisereduce/scipy raise "noverlap must be less than nperseg".
        if len(audio) < 2048:
            return audio
        try:
            import noisereduce as nr

            reduced = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.6, stationary=True)
            logger.info("Noise reduction applied (spectral gating)")
            return reduced
        except ImportError:
            logger.warning("noisereduce not installed, skipping noise reduction")
            return audio
        except ValueError as exc:
            logger.warning(f"Noise reduction skipped (short/degenerate buffer): {exc}")
            return audio

    @staticmethod
    def _trim_silence(audio: np.ndarray, sr: int, top_db: int = 30) -> np.ndarray:
        import librosa

        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed

    @staticmethod
    def _normalise(audio: np.ndarray, target_dbfs: float = -1.0) -> np.ndarray:
        peak = np.max(np.abs(audio))
        if peak == 0:
            return audio
        target_peak = 10 ** (target_dbfs / 20)
        return audio * (target_peak / peak)

    def vad_filter(self, audio: np.ndarray, sr: int, frame_ms: int = 30) -> np.ndarray:
        """Strip silence using webrtcvad. frame_ms must be 10, 20, or 30."""
        import webrtcvad

        vad = webrtcvad.Vad(self._vad_aggressiveness)
        int16 = audio_to_int16(audio)
        frame_len = int(sr * frame_ms / 1000)
        voiced_frames: list[bytes] = []

        for start in range(0, len(int16) - frame_len, frame_len):
            frame_bytes = struct.pack(f"{frame_len}h", *int16[start : start + frame_len])
            if vad.is_speech(frame_bytes, sr):
                voiced_frames.append(frame_bytes)

        if not voiced_frames:
            logger.warning("VAD removed all frames, returning original audio")
            return audio

        raw = b"".join(voiced_frames)
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767
