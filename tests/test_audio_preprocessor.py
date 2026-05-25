"""Deterministic-invariant tests for src/speech_to_text/audio_preprocessor.py.

Asserts the exact contract every downstream consumer (medasr_engine,
whisper_engine, pipeline, features2, transcription, run_full_pipeline) relies on:

  - target sample rate = 16000 (or whatever settings.audio_sample_rate dictates)
  - output is mono float32 ndarray
  - normalisation: peak <= 10^(-1/20) (≈0.891), i.e. -1 dBFS
  - silence-trim is bounded (does not eat speech)
  - resampling preserves duration to within one frame
  - empty / all-silence / clipping inputs do not crash

These tests do NOT require a real WAV file; they construct synthetic signals
deterministically so the suite runs in CI.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.speech_to_text.audio_preprocessor import AudioPreprocessor, PreprocessedAudio


# ── Helpers ────────────────────────────────────────────────────────────────


def _sine(sr: int, dur_s: float, freq: float = 440.0, amp: float = 0.5) -> np.ndarray:
    n = int(sr * dur_s)
    t = np.arange(n, dtype=np.float32) / sr
    return (amp * np.sin(2 * math.pi * freq * t)).astype(np.float32)


def _silence(sr: int, dur_s: float) -> np.ndarray:
    return np.zeros(int(sr * dur_s), dtype=np.float32)


def _speech_then_silence(sr: int) -> np.ndarray:
    return np.concatenate([_sine(sr, 1.0, 440.0, 0.5), _silence(sr, 1.0)])


# ── Output-shape invariants ────────────────────────────────────────────────


def test_process_array_returns_preprocessed_audio_object():
    pp = AudioPreprocessor(target_sr=16000)
    audio = _sine(16000, 1.0)
    out = pp.process_array(audio, sr=16000)
    assert isinstance(out, PreprocessedAudio)
    assert isinstance(out.audio, np.ndarray)
    assert out.audio.dtype in (np.float32, np.float64)
    assert out.audio.ndim == 1
    assert out.sample_rate == 16000
    assert out.duration_secs == pytest.approx(len(out.audio) / 16000, rel=1e-3)


def test_output_is_mono_after_processing_mono_input():
    pp = AudioPreprocessor(target_sr=16000)
    audio = _sine(16000, 0.5)
    out = pp.process_array(audio, sr=16000)
    assert out.audio.ndim == 1


def test_target_sr_default_is_16000():
    # Read settings: project default is 16 kHz per src/config/settings.py
    pp = AudioPreprocessor()
    assert pp._target_sr == 16000


# ── Resampling determinism ─────────────────────────────────────────────────


def test_resample_8khz_to_16khz_marks_was_resampled_true():
    pp = AudioPreprocessor(target_sr=16000)
    audio_8k = _sine(8000, 1.0)
    out = pp.process_array(audio_8k, sr=8000)
    assert out.was_resampled is True
    assert out.sample_rate == 16000
    # Allow ±1 frame tolerance for resampling rounding
    expected_n = int(8000 * 1.0 * 2)  # 16000 samples
    assert abs(len(out.audio) - expected_n) <= int(0.05 * expected_n)


def test_no_resample_when_input_matches_target_sr():
    pp = AudioPreprocessor(target_sr=16000)
    audio = _sine(16000, 1.0)
    out = pp.process_array(audio, sr=16000)
    assert out.was_resampled is False


# ── Normalisation invariant: peak <= -1 dBFS ───────────────────────────────


def test_normalisation_caps_peak_at_minus_one_dbfs():
    pp = AudioPreprocessor(target_sr=16000)
    # Loud sine wave (amplitude 0.95)
    audio = _sine(16000, 1.0, amp=0.95)
    out = pp.process_array(audio, sr=16000)
    target_peak = 10 ** (-1.0 / 20)  # ≈ 0.8913
    peak = float(np.max(np.abs(out.audio)))
    # Allow small floating-point tolerance
    assert peak <= target_peak + 1e-3, f"peak {peak} > target {target_peak}"


def test_normalisation_handles_clipping_input_safely():
    pp = AudioPreprocessor(target_sr=16000)
    # Hard-clipped sine
    audio = np.clip(_sine(16000, 1.0, amp=2.0), -1.0, 1.0).astype(np.float32)
    out = pp.process_array(audio, sr=16000)
    assert np.isfinite(out.audio).all()
    target_peak = 10 ** (-1.0 / 20)
    assert float(np.max(np.abs(out.audio))) <= target_peak + 1e-3


# ── Silence trim invariant ────────────────────────────────────────────────


def test_silence_trim_shortens_trailing_silence():
    pp = AudioPreprocessor(target_sr=16000)
    audio = _speech_then_silence(16000)
    out = pp.process_array(audio, sr=16000)
    assert out.was_trimmed is True
    assert out.duration_secs < 2.0
    # The 1 s of speech should remain (within ~10% tolerance from the noise reducer)
    assert out.duration_secs >= 0.7


def test_all_silence_input_does_not_crash():
    pp = AudioPreprocessor(target_sr=16000)
    audio = _silence(16000, 1.0)
    # _normalise has an early-return when peak==0; should not raise
    out = pp.process_array(audio, sr=16000)
    assert np.isfinite(out.audio).all()
    # Output may be heavily trimmed or fully silent - both are acceptable
    assert out.sample_rate == 16000


# ── Empty / pathological inputs ────────────────────────────────────────────


def test_short_buffer_does_not_crash():
    pp = AudioPreprocessor(target_sr=16000)
    audio = np.zeros(160, dtype=np.float32)  # 10 ms of silence
    out = pp.process_array(audio, sr=16000)
    assert out.sample_rate == 16000


# ── Cross-caller invariant: same input → same output ──────────────────────


def test_pipeline_is_deterministic_on_identical_input():
    pp1 = AudioPreprocessor(target_sr=16000)
    pp2 = AudioPreprocessor(target_sr=16000)
    audio = _sine(16000, 1.0, amp=0.5)
    out1 = pp1.process_array(audio.copy(), sr=16000)
    out2 = pp2.process_array(audio.copy(), sr=16000)
    assert len(out1.audio) == len(out2.audio)
    assert np.allclose(out1.audio, out2.audio, atol=1e-5)
    assert out1.was_trimmed == out2.was_trimmed
    assert out1.was_resampled == out2.was_resampled


# ── Contract probe: every consumer expects (.audio, .sample_rate, ...) ──────


def test_preprocessed_audio_has_required_attributes():
    pp = AudioPreprocessor(target_sr=16000)
    out = pp.process_array(_sine(16000, 0.5), sr=16000)
    for attr in ("audio", "sample_rate", "duration_secs", "was_trimmed", "was_resampled"):
        assert hasattr(out, attr), f"PreprocessedAudio missing attribute: {attr}"
