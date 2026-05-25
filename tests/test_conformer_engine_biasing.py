"""Tests for ConformerEngine SNOMED biasing integration.

All tests are mocked - no model checkpoint required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.speech_to_text.conformer_engine import ConformerEngine


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_engine(bias: bool = False, bias_terms=None, bias_weight=8.0) -> ConformerEngine:
    return ConformerEngine(
        model_dir="/nonexistent",
        device="cpu",
        bias=bias,
        bias_terms=bias_terms,
        bias_weight=bias_weight,
    )


def _fake_log_probs(T: int = 10, V: int = 50) -> "torch.Tensor":
    import torch
    rng = np.random.default_rng(42)
    lp = rng.random((1, T, V)).astype(np.float32)
    lp = lp / lp.sum(axis=-1, keepdims=True)
    return torch.from_numpy(np.log(lp + 1e-10))


# ── Construction ───────────────────────────────────────────────────────────────

def test_engine_initialises_without_model():
    engine = _make_engine()
    assert engine._model is None
    assert engine._biaser is None


def test_engine_bias_flag_stored():
    engine = _make_engine(bias=True)
    assert engine._bias is True


def test_engine_bias_weight_stored():
    engine = _make_engine(bias=True, bias_weight=10.0)
    assert engine._bias_weight == 10.0


def test_engine_bias_terms_stored():
    terms = ["amoxicillin", "metformin"]
    engine = _make_engine(bias=True, bias_terms=terms)
    assert engine._bias_terms == terms


def test_engine_no_bias_biaser_is_none():
    engine = _make_engine(bias=False)
    assert engine._biaser is None


# ── _build_bias_decoder uses SnomedBiaser ─────────────────────────────────────

def test_build_bias_decoder_uses_snomed_biaser(monkeypatch):
    """_build_bias_decoder should create a SnomedBiaser and call build_decoder."""
    from src.speech_to_text import snomed_biasing as sb

    mock_biaser = MagicMock()
    mock_biaser.build_decoder.return_value = True

    mock_biaser_cls = MagicMock(return_value=mock_biaser)

    with patch.object(sb, "SnomedBiaser", mock_biaser_cls):
        engine = _make_engine(bias=True)
        engine._sp = MagicMock()
        engine._sp.get_piece_size.return_value = 10
        engine._sp.id_to_piece.side_effect = lambda i: f"▁word{i}"

        engine._build_bias_decoder()

    mock_biaser_cls.assert_called_once()
    mock_biaser.build_decoder.assert_called_once()
    assert engine._biaser is mock_biaser


def test_build_bias_decoder_augments_with_extra_terms():
    """Caller-supplied bias_terms should be added to the SnomedLexicon."""
    engine = _make_engine(bias=True, bias_terms=["bespoke_drug_xyz"])
    engine._sp = MagicMock()
    engine._sp.get_piece_size.return_value = 5
    engine._sp.id_to_piece.side_effect = lambda i: f"word{i}"

    # Patch SnomedBiaser to avoid pyctcdecode dependency
    with patch("src.speech_to_text.snomed_biasing.SnomedBiaser") as MockBiaser:
        instance = MagicMock()
        instance.build_decoder.return_value = True
        MockBiaser.return_value = instance

        from src.speech_to_text.snomed_biasing import SnomedLexicon
        with patch("src.speech_to_text.snomed_biasing.SnomedLexicon") as MockLex:
            lex_instance = MagicMock()
            lex_instance._terms = set()
            MockLex.return_value = lex_instance

            engine._build_bias_decoder()

        assert "bespoke_drug_xyz" in lex_instance._terms


def test_build_bias_decoder_disables_bias_when_pyctcdecode_missing():
    """When build_decoder returns False (pyctcdecode absent), bias is disabled."""
    engine = _make_engine(bias=True)
    engine._sp = MagicMock()
    engine._sp.get_piece_size.return_value = 5
    engine._sp.id_to_piece.side_effect = lambda i: f"tok{i}"

    with patch("src.speech_to_text.snomed_biasing.SnomedBiaser") as MockBiaser:
        instance = MagicMock()
        instance.build_decoder.return_value = False  # pyctcdecode absent
        MockBiaser.return_value = instance

        engine._build_bias_decoder()

    assert engine._bias is False
    assert engine._biaser is None


# ── _ctc_biased_decode delegates to SnomedBiaser ──────────────────────────────

def test_ctc_biased_decode_calls_biaser(monkeypatch):
    import torch
    engine = _make_engine(bias=True)
    mock_biaser = MagicMock()
    mock_biaser.decode.return_value = "amoxicillin 500mg"
    engine._biaser = mock_biaser

    log_probs = _fake_log_probs()
    result = engine._ctc_biased_decode(log_probs)

    mock_biaser.decode.assert_called_once()
    assert result == "amoxicillin 500mg"


def test_ctc_biased_decode_passes_numpy_array(monkeypatch):
    import torch
    engine = _make_engine(bias=True)
    captured = {}

    def capture_decode(arr):
        captured["arr"] = arr
        return "test"

    engine._biaser = MagicMock()
    engine._biaser.decode.side_effect = capture_decode

    log_probs = _fake_log_probs(T=8, V=30)
    engine._ctc_biased_decode(log_probs)

    assert isinstance(captured["arr"], np.ndarray)
    assert captured["arr"].ndim == 2  # (T, V) after squeeze


# ── transcribe path selection ──────────────────────────────────────────────────

def test_transcribe_uses_greedy_when_bias_false():
    """With bias=False, transcribe should call _ctc_greedy_decode."""
    engine = _make_engine(bias=False)

    # Inject a fake loaded model
    import torch
    fake_log_probs = _fake_log_probs()
    engine._model = MagicMock(return_value=fake_log_probs)
    engine._sp = MagicMock()
    engine._sp.get_piece_size.return_value = 5
    engine._sp.decode.return_value = "chest pain"
    engine._mel_fb = torch.ones(80, 201)
    engine._window = torch.hann_window(400)

    with patch.object(engine, "_ctc_greedy_decode", return_value="chest pain") as mock_greedy, \
         patch.object(engine, "_ctc_biased_decode", return_value="biased") as mock_biased, \
         patch.object(engine, "_audio_to_mel", return_value=torch.zeros(1, 80, 50)):
        audio = np.zeros(TARGET_SR * 2, dtype=np.float32)
        engine.transcribe(audio)

    mock_greedy.assert_called()
    mock_biased.assert_not_called()


def test_transcribe_uses_biased_when_bias_true_and_biaser_set():
    """With bias=True and biaser loaded, transcribe should call _ctc_biased_decode."""
    import torch
    engine = _make_engine(bias=True)

    fake_log_probs = _fake_log_probs()
    engine._model = MagicMock(return_value=fake_log_probs)
    engine._sp = MagicMock()
    engine._sp.get_piece_size.return_value = 5
    engine._biaser = MagicMock()
    engine._mel_fb = torch.ones(80, 201)
    engine._window = torch.hann_window(400)

    with patch.object(engine, "_ctc_greedy_decode", return_value="greedy") as mock_greedy, \
         patch.object(engine, "_ctc_biased_decode", return_value="amoxicillin") as mock_biased, \
         patch.object(engine, "_audio_to_mel", return_value=torch.zeros(1, 80, 50)):
        audio = np.zeros(TARGET_SR * 2, dtype=np.float32)
        engine.transcribe(audio)

    mock_biased.assert_called()
    mock_greedy.assert_not_called()


def test_transcribe_falls_back_to_greedy_when_biaser_none():
    """bias=True but biaser is None → greedy (build failed)."""
    import torch
    engine = _make_engine(bias=True)
    engine._biaser = None  # build failed

    fake_log_probs = _fake_log_probs()
    engine._model = MagicMock(return_value=fake_log_probs)
    engine._sp = MagicMock()
    engine._sp.get_piece_size.return_value = 5
    engine._mel_fb = torch.ones(80, 201)
    engine._window = torch.hann_window(400)

    with patch.object(engine, "_ctc_greedy_decode", return_value="greedy") as mock_greedy, \
         patch.object(engine, "_ctc_biased_decode", return_value="biased") as mock_biased, \
         patch.object(engine, "_audio_to_mel", return_value=torch.zeros(1, 80, 50)):
        audio = np.zeros(TARGET_SR * 2, dtype=np.float32)
        engine.transcribe(audio)

    mock_greedy.assert_called()
    mock_biased.assert_not_called()


# ── import for TARGET_SR ───────────────────────────────────────────────────────

from src.speech_to_text.conformer_engine import TARGET_SR
