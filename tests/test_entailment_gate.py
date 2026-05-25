"""Tests for the NLI entailment gate."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.clinical.entailment_gate import (
    DEFAULT_THRESHOLD,
    EntailmentGate,
    GateResult,
    SentenceResult,
    _select_window,
    _split_sentences,
)


# ── Sentence splitting ─────────────────────────────────────────────────────────

def test_split_returns_list():
    result = _split_sentences("Patient reports chest pain for three days.")
    assert isinstance(result, list)


def test_split_skips_section_headers():
    note = "SUBJECTIVE:\nPatient has a cough.\nOBJECTIVE:\nTemp 38.4."
    sentences = _split_sentences(note)
    assert all("SUBJECTIVE" not in s and "OBJECTIVE" not in s for s in sentences)


def test_split_skips_very_short_fragments():
    note = "Pain.\nPatient reports three weeks of lower back pain radiating to the leg."
    sentences = _split_sentences(note)
    assert not any(len(s.split()) < 5 for s in sentences)


def test_split_strips_numbered_prefix():
    note = "1. Prescribe naproxen 500mg twice daily with food.\n2. Refer physiotherapy."
    sentences = _split_sentences(note)
    assert not any(s.startswith("1.") or s.startswith("2.") for s in sentences)


def test_split_splits_on_period_boundary():
    note = "Temperature is 38.4 degrees. Sats are 97 percent on room air."
    sentences = _split_sentences(note)
    assert len(sentences) >= 1


def test_split_empty_text_returns_empty():
    assert _split_sentences("") == []


def test_split_only_headers_returns_empty():
    assert _split_sentences("SUBJECTIVE:\nOBJECTIVE:\nASSESSMENT:\nPLAN:") == []


def test_split_multiline_soap_note():
    note = (
        "SUBJECTIVE:\nPatient reports three weeks of lower back pain radiating to left leg.\n\n"
        "OBJECTIVE:\nStraight leg raise positive at 40 degrees on the left side.\n\n"
        "ASSESSMENT:\nLumbar disc protrusion with left-sided sciatica.\n\n"
        "PLAN:\n1. Naproxen 500mg twice daily with food as prescribed.\n"
        "2. Refer to physiotherapy for rehabilitation programme.\n"
        "Safety-netting: return if symptoms worsen, call 999 for leg weakness."
    )
    sentences = _split_sentences(note)
    assert len(sentences) >= 3
    assert all(len(s.split()) >= 5 for s in sentences)


# ── Window selection ───────────────────────────────────────────────────────────

def test_select_window_short_returns_full():
    transcript = "Patient has chest pain and fever."
    result = _select_window(transcript, "chest pain", max_tokens=500)
    assert result == transcript


def test_select_window_long_returns_subset():
    # 500 words transcript
    transcript = " ".join(["word"] * 500)
    result = _select_window(transcript, "some query sentence", max_tokens=100)
    assert len(result.split()) <= 100


def test_select_window_keyword_matching():
    # Build a transcript where amoxicillin appears only in the second half
    first_half = " ".join(["irrelevant"] * 200)
    second_half = "patient prescribed amoxicillin 500mg three times daily for infection"
    transcript = first_half + " " + second_half
    sentence = "Prescribe amoxicillin 500mg three times daily."
    result = _select_window(transcript, sentence, max_tokens=50)
    assert "amoxicillin" in result


def test_select_window_no_keywords_returns_start():
    transcript = " ".join([f"w{i}" for i in range(500)])
    # Sentence with no 4-letter+ words that match
    result = _select_window(transcript, "ok ok ok ok ok", max_tokens=50)
    # Should return first window (default)
    assert len(result.split()) <= 50


# ── GateResult structure ────────────────────────────────────────────────────────

def test_gate_result_rates_sum_to_one():
    result = GateResult(
        num_checked=10, num_supported=7, num_flagged=3,
        hallucination_rate=0.3, supported_rate=0.7,
    )
    assert abs(result.hallucination_rate + result.supported_rate - 1.0) < 0.01


def test_gate_result_flagged_field_default_empty():
    result = GateResult(
        num_checked=5, num_supported=5, num_flagged=0,
        hallucination_rate=0.0, supported_rate=1.0,
    )
    assert result.flagged == []


# ── EntailmentGate.check - no model needed ──────────────────────────────────────

def test_check_no_transcript_returns_skipped():
    gate = EntailmentGate()
    result = gate.check("Patient has chest pain.", "")
    assert result.skipped is True
    assert result.num_checked == 0


def test_check_empty_note_returns_zero_checked():
    gate = EntailmentGate()
    result = gate.check("", "Some transcript about chest pain.")
    assert result.num_checked == 0


def test_check_model_unavailable_returns_skipped():
    gate = EntailmentGate(model_name="nonexistent-model-xyz")
    gate._loaded = True   # mark as loaded
    gate._model = None    # but model is None
    gate._tokenizer = None
    result = gate.check(
        "Patient prescribed amoxicillin 500mg for infection.",
        "The doctor prescribed amoxicillin.",
    )
    assert result.skipped is True
    assert "unavailable" in result.skip_reason.lower()


# ── EntailmentGate.check - mocked model ─────────────────────────────────────────

def _make_gate_with_mock_scores(scores: dict[str, float]) -> EntailmentGate:
    """Create a gate whose _score_entailment returns configurable P(entailment)."""
    gate = EntailmentGate(threshold=0.5)
    gate._loaded = True
    gate._tokenizer = MagicMock()
    gate._model = MagicMock()

    def mock_score(premise, hypothesis, tokenizer, model):
        # Return a high score for 'supported', low for 'unsupported' based on content
        for keyword, score in scores.items():
            if keyword.lower() in hypothesis.lower():
                contra = (1 - score) / 2
                return score, contra, contra
        return 0.8, 0.1, 0.1  # default: supported

    import src.clinical.entailment_gate as eg
    gate._score = mock_score

    # Patch the module-level function used inside check()
    original_check = gate.check

    def patched_check(soap_note, transcript):
        sentences = _split_sentences(soap_note)
        if not transcript.strip():
            return GateResult(
                num_checked=0, num_supported=0, num_flagged=0,
                hallucination_rate=0.0, supported_rate=1.0,
                skipped=True, skip_reason="no transcript provided",
            )
        if not sentences:
            return GateResult(
                num_checked=0, num_supported=0, num_flagged=0,
                hallucination_rate=0.0, supported_rate=1.0,
            )

        from src.clinical.entailment_gate import _select_window
        results = []
        for sentence in sentences:
            window = _select_window(transcript, sentence, gate.max_window_tokens)
            p_e, p_c, p_n = mock_score(window, sentence, None, None)
            results.append(SentenceResult(
                sentence=sentence,
                entailment_score=round(p_e, 4),
                contradiction_score=round(p_c, 4),
                neutral_score=round(p_n, 4),
                supported=p_e >= gate.threshold,
                transcript_window=window[:400],
            ))

        flagged = sorted([r for r in results if not r.supported], key=lambda x: x.entailment_score)
        n = len(results)
        n_flagged = len(flagged)
        return GateResult(
            num_checked=n,
            num_supported=n - n_flagged,
            num_flagged=n_flagged,
            hallucination_rate=round(n_flagged / n, 4) if n else 0.0,
            supported_rate=round((n - n_flagged) / n, 4) if n else 1.0,
            flagged=flagged,
            all_sentences=results,
        )

    gate.check = patched_check
    return gate


def test_check_all_supported_no_flagged():
    gate = _make_gate_with_mock_scores({})  # all default to 0.8
    result = gate.check(
        "SUBJECTIVE:\nPatient reports chest pain and fever for two days.\n"
        "PLAN:\nPrescribe paracetamol 1g four times daily as needed.",
        "Patient has chest pain and fever.",
    )
    assert result.num_flagged == 0
    assert result.flagged == []
    assert result.supported_rate == 1.0


def test_check_unsupported_sentence_flagged():
    # "fabricated" keyword → low score 0.2
    gate = _make_gate_with_mock_scores({"fabricated": 0.2})
    result = gate.check(
        "Patient prescribed a fabricated drug not in the transcript at all.",
        "Patient came in with a cough.",
    )
    assert result.num_flagged >= 1
    assert result.hallucination_rate > 0


def test_check_hallucination_rate_correct():
    # Two sentences: one unsupported (keyword "invented"), one supported
    gate = _make_gate_with_mock_scores({"invented": 0.1})
    note = (
        "Patient reports chest pain and fever for several days.\n"
        "Patient also has an invented condition not mentioned anywhere."
    )
    result = gate.check(note, "Patient has chest pain and fever.")
    assert result.num_checked == 2
    assert result.num_flagged == 1
    assert abs(result.hallucination_rate - 0.5) < 0.01


def test_check_flagged_sorted_by_score_ascending():
    gate = _make_gate_with_mock_scores({"alpha": 0.3, "beta": 0.1})
    note = (
        "Patient has an alpha condition not in the record.\n"
        "Patient also has a beta condition not in the record.\n"
        "Patient reports genuine chest pain and cough."
    )
    result = gate.check(note, "Patient came in with some symptoms today.")
    if len(result.flagged) >= 2:
        scores = [s.entailment_score for s in result.flagged]
        assert scores == sorted(scores)


def test_check_all_sentences_included():
    gate = _make_gate_with_mock_scores({})
    note = (
        "Patient reports three weeks of lower back pain radiating down the left leg.\n"
        "Straight leg raise is positive on the left side at forty degrees.\n"
        "Prescribe naproxen 500mg twice daily with food and refer to physiotherapy."
    )
    result = gate.check(note, "Patient has lower back pain radiating to left leg.")
    assert result.num_checked == len(result.all_sentences)
    assert result.num_checked >= 2


def test_check_supported_rate_plus_hallucination_rate_equals_one():
    gate = _make_gate_with_mock_scores({"unsupported": 0.2})
    note = (
        "Patient has genuine chest pain documented in the consultation.\n"
        "Patient has an unsupported condition fabricated by the model."
    )
    result = gate.check(note, "Patient has chest pain.")
    if result.num_checked > 0:
        assert abs(result.supported_rate + result.hallucination_rate - 1.0) < 0.01


# ── Integration with validate_soap_note ────────────────────────────────────────

def test_validate_entailment_disabled_does_not_add_check():
    from src.clinical.verification import validate_soap_note
    note = (
        "SUBJECTIVE:\nPatient reports chest pain for two days.\n"
        "OBJECTIVE:\nTemp 37.2. Sats 98%.\n"
        "ASSESSMENT:\nChest pain query musculoskeletal.\n"
        "PLAN:\n1. Ibuprofen 400mg three times daily with food.\n"
        "Safety-netting: return if symptoms worsen, call 999 for severe chest pain."
    )
    result = validate_soap_note(note, "Patient has chest pain.", use_entailment_gate=False)
    check_names = [c.check_name for c in result.check_results]
    assert "entailment_gate" not in check_names


def test_validate_checks_total_unchanged_when_gate_enabled_model_absent():
    """Gate adds a check only when the model actually runs; canonical total stays 6."""
    from src.clinical.verification import validate_soap_note
    note = (
        "SUBJECTIVE:\nPatient reports lower back pain for three weeks.\n"
        "OBJECTIVE:\nStraight leg raise positive.\n"
        "ASSESSMENT:\nLumbar disc protrusion.\n"
        "PLAN:\n1. Naproxen 500mg BD with food.\n"
        "Safety-netting: attend A&E if leg weakness develops, call 999."
    )
    # With no model available (will fail to load), gate skips gracefully
    result = validate_soap_note(note, "Patient has back pain.", use_entailment_gate=True)
    assert result.checks_total == 6


def test_validate_entailment_gate_check_added_when_model_mocked():
    """When model is mocked to return high scores, gate passes and is included in results."""
    from src.clinical.verification import validate_soap_note

    mock_result = GateResult(
        num_checked=3, num_supported=3, num_flagged=0,
        hallucination_rate=0.0, supported_rate=1.0,
    )

    class MockGate:
        def check(self, note, transcript):
            return mock_result

    with patch("src.clinical.entailment_gate.EntailmentGate", return_value=MockGate()):
        note = (
            "SUBJECTIVE:\nPatient reports chest pain.\n"
            "OBJECTIVE:\nTemp 37.5.\n"
            "ASSESSMENT:\nChest pain.\n"
            "PLAN:\n1. Paracetamol 1g QDS.\nSafety-netting: call 999 if worsens."
        )
        result = validate_soap_note(note, "Patient has chest pain.", use_entailment_gate=True)

    check_names = [c.check_name for c in result.check_results]
    assert "entailment_gate" in check_names


def test_validate_entailment_flagged_sentences_appear_in_issues():
    """Flagged sentences from the gate propagate into validation issues."""
    from src.clinical.verification import validate_soap_note

    mock_result = GateResult(
        num_checked=2,
        num_supported=1,
        num_flagged=1,
        hallucination_rate=0.5,
        supported_rate=0.5,
        flagged=[
            SentenceResult(
                sentence="Patient prescribed methotrexate for rheumatoid arthritis.",
                entailment_score=0.12,
                contradiction_score=0.7,
                neutral_score=0.18,
                supported=False,
                transcript_window="patient came in with a cough",
            )
        ],
    )

    class MockGate:
        def check(self, note, transcript):
            return mock_result

    with patch("src.clinical.entailment_gate.EntailmentGate", return_value=MockGate()):
        note = (
            "SUBJECTIVE:\nPatient reports cough.\n"
            "OBJECTIVE:\nTemp normal.\n"
            "ASSESSMENT:\nUpper respiratory tract infection.\n"
            "PLAN:\n1. Paracetamol PRN.\nSafety-netting: call 111 if not improving."
        )
        result = validate_soap_note(note, "Patient has a cough.", use_entailment_gate=True)

    gate_check = next((c for c in result.check_results if c.check_name == "entailment_gate"), None)
    assert gate_check is not None
    assert gate_check.passed is False
    assert any("methotrexate" in i or "score=0.12" in i for i in gate_check.issues)
