"""Tests for streaming generation-time TrustScore veto (N16).

These tests mock the Ollama HTTP stream and the NLI scoring primitive so they
run without external dependencies. The behaviours under test:

- Sentence boundary detection (pure regex / helper functions).
- Streaming event ordering for an all-pass run.
- Veto + rewrite cycle when NLI scores fall below threshold.
- Give-up placeholder after max retries.
- Veto-inactive fallback when the NLI model fails to load.
- The all_shipped_passed_veto invariant (the hard guarantee).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from src.soap_generation.streaming_constitutional import (
    EventKind,
    StreamEvent,
    StreamingConstitutionalGenerator,
    StreamingResult,
    _is_skippable,
    _split_buffer,
)


# ── Pure helpers ──────────────────────────────────────────────────────


class TestSentenceBoundary:
    def test_completes_after_terminal_punctuation_then_space(self):
        completed, residual = _split_buffer("The patient has cough. They report fever.")
        assert completed == ["The patient has cough."]
        assert residual == "They report fever."

    def test_no_split_without_trailing_whitespace(self):
        completed, residual = _split_buffer("The patient has cough.")
        assert completed == []
        assert residual == "The patient has cough."

    def test_paragraph_break_splits(self):
        completed, residual = _split_buffer(
            "Paragraph one ends here.\n\nParagraph two begins")
        assert "Paragraph one ends here." in completed[0]
        assert residual == "Paragraph two begins"

    def test_section_colon_then_newline_splits(self):
        completed, residual = _split_buffer("PLAN:\nGive paracetamol")
        assert completed[0].rstrip() == "PLAN:"
        assert residual.strip() == "Give paracetamol"

    def test_empty_buffer(self):
        completed, residual = _split_buffer("")
        assert completed == []
        assert residual == ""

    def test_multiple_sentences(self):
        completed, residual = _split_buffer(
            "First sentence. Second sentence. Third sentence")
        assert len(completed) == 2
        assert residual == "Third sentence"


class TestSkippable:
    @pytest.mark.parametrize("s", [
        "", "   ", "\n", "\t",
        "SUBJECTIVE:", "OBJECTIVE:", "ASSESSMENT:", "PLAN:",
        "S:", "O:", "A:", "P:",
        "subjective:", "PLAN",
        "Hello.", "Yes.", "OK now.",   # too few words
    ])
    def test_skips_structural_or_short(self, s):
        assert _is_skippable(s) is True

    @pytest.mark.parametrize("s", [
        "Patient reports fever and cough for three days.",
        "On examination, chest is clear bilaterally.",
        "Paracetamol 1g four times daily as required.",
    ])
    def test_does_not_skip_real_sentences(self, s):
        assert _is_skippable(s) is False


# ── Mocked streaming behaviour ────────────────────────────────────────


def _fake_chunks(text: str, chunk_size: int = 12):
    """Split a string into deterministic chunks to simulate Ollama streaming."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


@pytest.fixture
def gen_no_nli():
    """Generator with NLI model deliberately unavailable -> veto inactive."""
    g = StreamingConstitutionalGenerator()
    # Force the lazy load to skip the model and report unavailable.
    g._nli_loaded = True
    g._nli_tokenizer = None
    g._nli_model = None
    return g


@pytest.fixture
def gen_with_nli():
    """Generator with NLI faked-loaded; tests can stub _score_sentence."""
    g = StreamingConstitutionalGenerator()
    g._nli_loaded = True
    # Sentinels so _ensure_nli() returns True
    g._nli_tokenizer = object()
    g._nli_model = object()
    return g


class TestStreamingNoVeto:
    """Veto inactive: every candidate sentence ships."""

    def test_three_sentences_all_ship(self, gen_no_nli, monkeypatch):
        text = ("Patient reports fever and cough. "
                "On examination chest is clear. "
                "Plan paracetamol PRN and safety net.")

        def fake_stream(messages, **kwargs):
            yield from _fake_chunks(text)

        monkeypatch.setattr(gen_no_nli, "_stream_chunks", fake_stream)
        result = gen_no_nli.generate("Patient said they have fever.")

        assert isinstance(result, StreamingResult)
        assert result.veto_active is False
        assert result.accepted_count >= 2
        assert result.vetoed_count == 0
        assert result.rewritten_count == 0
        assert "fever" in result.soap_note.lower()

    def test_emits_complete_event_last(self, gen_no_nli, monkeypatch):
        def fake_stream(messages, **kwargs):
            yield from _fake_chunks("First valid clinical sentence here. Second.")

        monkeypatch.setattr(gen_no_nli, "_stream_chunks", fake_stream)
        events = list(gen_no_nli.generate_streaming("transcript"))
        assert events[-1].kind == EventKind.COMPLETE


class TestStreamingWithVeto:
    """Veto active: low entailment scores trigger rewrite or give-up."""

    def test_high_score_passes(self, gen_with_nli, monkeypatch):
        text = "Patient reports fever and cough today. Examination is clear."

        def fake_stream(messages, **kwargs):
            yield from _fake_chunks(text)

        # Always-high entailment
        def fake_score(sentence, transcript):
            return 0.95, transcript[:200]

        monkeypatch.setattr(gen_with_nli, "_stream_chunks", fake_stream)
        monkeypatch.setattr(gen_with_nli, "_score_sentence", fake_score)
        result = gen_with_nli.generate("Patient said fever and cough.")

        assert result.veto_active is True
        assert result.vetoed_count == 0
        assert result.rewritten_count == 0
        assert result.accepted_count >= 1
        assert result.all_shipped_passed_veto is True

    def test_low_score_triggers_rewrite(self, gen_with_nli, monkeypatch):
        # First call: bad sentence -> low score -> veto + rewrite
        # Second call (after rewrite): good sentence -> high score -> ship
        call_idx = {"i": 0}
        bad_text = "Patient takes morphine for chronic back pain."
        good_text = "Patient reports a cough that began this morning."

        def fake_stream(messages, **kwargs):
            call_idx["i"] += 1
            if call_idx["i"] == 1:
                yield from _fake_chunks(bad_text)
            else:
                yield from _fake_chunks(good_text)

        score_idx = {"i": 0}

        def fake_score(sentence, transcript):
            score_idx["i"] += 1
            # First call (bad sentence) low; subsequent (good) high
            if "morphine" in sentence:
                return 0.10, transcript[:200]
            return 0.90, transcript[:200]

        monkeypatch.setattr(gen_with_nli, "_stream_chunks", fake_stream)
        monkeypatch.setattr(gen_with_nli, "_score_sentence", fake_score)
        result = gen_with_nli.generate("Patient said cough this morning.")

        assert result.vetoed_count >= 1
        assert result.rewritten_count >= 1
        assert "morphine" not in result.soap_note
        assert "cough" in result.soap_note.lower()
        assert result.all_shipped_passed_veto is True

    def test_max_retries_emits_placeholder(self, gen_with_nli, monkeypatch):
        # Every sentence scores low forever -> after max_retries -> placeholder
        def fake_stream(messages, **kwargs):
            yield from _fake_chunks("Patient takes diazepam every evening.")

        def fake_score(sentence, transcript):
            return 0.05, transcript[:200]

        monkeypatch.setattr(gen_with_nli, "_stream_chunks", fake_stream)
        monkeypatch.setattr(gen_with_nli, "_score_sentence", fake_score)
        gen_with_nli.max_retries = 2
        result = gen_with_nli.generate("Patient said cough.")

        assert result.gave_up_count >= 1
        assert gen_with_nli.give_up_placeholder in result.soap_note

    def test_skippable_lines_pass_without_score(self, gen_with_nli, monkeypatch):
        # Headers should ship without ever calling _score_sentence
        def fake_stream(messages, **kwargs):
            yield from _fake_chunks("PLAN:\n\nSection content here is long enough.")

        score_calls = {"n": 0}

        def fake_score(sentence, transcript):
            score_calls["n"] += 1
            return 0.95, transcript[:200]

        monkeypatch.setattr(gen_with_nli, "_stream_chunks", fake_stream)
        monkeypatch.setattr(gen_with_nli, "_score_sentence", fake_score)
        result = gen_with_nli.generate("transcript")

        # PLAN: should ship without calling NLI; only the "Section content" sentence
        # (if it qualifies) might call NLI.
        assert "PLAN:" in result.soap_note
        assert score_calls["n"] <= 1


class TestHardGuarantee:
    """The all_shipped_passed_veto invariant under various stress patterns."""

    def test_invariant_holds_after_one_veto_then_rewrite(self, gen_with_nli, monkeypatch):
        cnt = {"stream": 0, "score": 0}

        def fake_stream(messages, **kwargs):
            cnt["stream"] += 1
            text = ("First fabrication that fails. " if cnt["stream"] == 1
                    else "Patient reports cough this morning. ")
            yield from _fake_chunks(text)

        def fake_score(sentence, transcript):
            cnt["score"] += 1
            return (0.10 if "fabrication" in sentence else 0.92), transcript[:200]

        monkeypatch.setattr(gen_with_nli, "_stream_chunks", fake_stream)
        monkeypatch.setattr(gen_with_nli, "_score_sentence", fake_score)

        result = gen_with_nli.generate("transcript")
        assert result.all_shipped_passed_veto is True
        assert "fabrication" not in result.soap_note

    def test_invariant_holds_when_give_up_fires(self, gen_with_nli, monkeypatch):
        # Persistent low score -> give-up; placeholder counts as accepted by
        # invariant (it is explicitly flagged for human review).
        # Sentence must be long enough (>= _MIN_VETO_WORDS) to trigger the veto
        # path rather than being treated as a structural skip.
        def fake_stream(messages, **kwargs):
            yield from _fake_chunks(
                "This is a fabricated clinical sentence about treatment.")

        def fake_score(sentence, transcript):
            return 0.05, transcript[:200]

        monkeypatch.setattr(gen_with_nli, "_stream_chunks", fake_stream)
        monkeypatch.setattr(gen_with_nli, "_score_sentence", fake_score)
        gen_with_nli.max_retries = 1

        result = gen_with_nli.generate("transcript")
        assert result.all_shipped_passed_veto is True
        assert gen_with_nli.give_up_placeholder in result.soap_note


class TestEventEmission:
    def test_events_have_dict_serialisation(self, gen_no_nli, monkeypatch):
        def fake_stream(messages, **kwargs):
            yield from _fake_chunks("Long enough valid clinical sentence here.")

        monkeypatch.setattr(gen_no_nli, "_stream_chunks", fake_stream)
        result = gen_no_nli.generate("transcript")
        for ev in result.events:
            d = ev.to_dict()
            assert d["kind"] in {k.value for k in EventKind}
            assert "sentence" in d
            assert "entailment_score" in d

    def test_streaming_result_to_dict(self, gen_no_nli, monkeypatch):
        def fake_stream(messages, **kwargs):
            yield from _fake_chunks("Long enough valid clinical sentence here.")

        monkeypatch.setattr(gen_no_nli, "_stream_chunks", fake_stream)
        result = gen_no_nli.generate("transcript")
        d = result.to_dict()
        assert d["model"]
        assert "events" in d
        assert "all_shipped_passed_veto" in d
        assert d["veto_active"] is False
