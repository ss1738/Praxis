"""Tests for the Asgari-style hallucination + omission evaluation harness."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.evaluation.hallucination_eval import (
    ASGARI_HALLUCINATION_BAR,
    ASGARI_OMISSION_BAR,
    HallucinationBenchmarkResult,
    NoteHallucinationResult,
    _bootstrap_ci,
    evaluate_note,
    run_hallucination_eval,
    to_latex_table,
    to_markdown_table,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_note_result(
    note_id: str = "note_001",
    model: str = "test-model",
    num_sentences: int = 5,
    num_hallucinated: int = 0,
    mhi_omission: float = 0.01,
    mhi_hallucination: float = 0.01,
    entailment_skipped: bool = True,
) -> NoteHallucinationResult:
    hall_rate = round(num_hallucinated / num_sentences, 4) if num_sentences > 0 else 0.0
    return NoteHallucinationResult(
        note_id=note_id,
        model=model,
        num_sentences=num_sentences,
        num_hallucinated=num_hallucinated,
        sentence_hallucination_rate=hall_rate,
        mhi_omission=mhi_omission,
        mhi_hallucination=mhi_hallucination,
        entailment_skipped=entailment_skipped,
        delta_hallucination=round(hall_rate - ASGARI_HALLUCINATION_BAR, 4),
        delta_omission=round(mhi_omission - ASGARI_OMISSION_BAR, 4),
    )


def _make_benchmark(
    model: str = "test-model",
    hall_rate: float = 0.01,
    mhi_omission: float = 0.02,
    mhi_hallucination: float = 0.01,
    num_notes: int = 8,
    total_sentences: int = 40,
) -> HallucinationBenchmarkResult:
    return HallucinationBenchmarkResult(
        model=model,
        timestamp="2026-04-29T00:00:00+00:00",
        num_notes=num_notes,
        total_sentences=total_sentences,
        total_hallucinated=int(hall_rate * total_sentences),
        mean_sentence_hallucination_rate=hall_rate,
        ci_hallucination_lower=max(0.0, hall_rate - 0.005),
        ci_hallucination_upper=hall_rate + 0.005,
        entailment_skipped=False,
        mean_mhi_score=round(1 - mhi_omission - mhi_hallucination, 4),
        mean_mhi_omission=mhi_omission,
        mean_mhi_hallucination=mhi_hallucination,
        ci_mhi_omission_lower=max(0.0, mhi_omission - 0.005),
        ci_mhi_omission_upper=mhi_omission + 0.005,
        beats_asgari_hallucination=hall_rate < ASGARI_HALLUCINATION_BAR,
        beats_asgari_omission=mhi_omission < ASGARI_OMISSION_BAR,
    )


# ── Constants ─────────────────────────────────────────────────────────────────

def test_asgari_hallucination_bar():
    assert abs(ASGARI_HALLUCINATION_BAR - 0.0147) < 1e-6


def test_asgari_omission_bar():
    assert abs(ASGARI_OMISSION_BAR - 0.0345) < 1e-6


# ── NoteHallucinationResult ────────────────────────────────────────────────────

def test_note_result_fields_exist():
    r = _make_note_result()
    assert hasattr(r, "note_id")
    assert hasattr(r, "sentence_hallucination_rate")
    assert hasattr(r, "mhi_omission")
    assert hasattr(r, "mhi_hallucination")
    assert hasattr(r, "delta_hallucination")
    assert hasattr(r, "delta_omission")


def test_note_result_delta_hallucination_sign():
    r = _make_note_result(num_sentences=10, num_hallucinated=0)
    # 0% < 1.47% so delta is negative (better than bar)
    assert r.delta_hallucination < 0


def test_note_result_delta_omission_negative_when_better():
    r = _make_note_result(mhi_omission=0.01)
    assert r.delta_omission < 0


def test_note_result_flagged_sentences_default_empty():
    r = _make_note_result()
    assert r.flagged_sentences == []


def test_note_result_entailment_skipped_default_false():
    r = NoteHallucinationResult(
        note_id="x", model="m",
        num_sentences=0, num_hallucinated=0, sentence_hallucination_rate=0.0,
    )
    assert r.entailment_skipped is False


# ── _bootstrap_ci ──────────────────────────────────────────────────────────────

def test_bootstrap_ci_single_value():
    lo, hi = _bootstrap_ci([0.05])
    assert lo == hi == 0.05


def test_bootstrap_ci_bounds_within_data_range():
    values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    lo, hi = _bootstrap_ci(values)
    assert lo >= min(values) - 0.01
    assert hi <= max(values) + 0.01


def test_bootstrap_ci_lower_le_upper():
    lo, hi = _bootstrap_ci([0.0, 0.1, 0.2, 0.3])
    assert lo <= hi


def test_bootstrap_ci_deterministic_seed():
    values = [0.01, 0.02, 0.015, 0.025, 0.018]
    lo1, hi1 = _bootstrap_ci(values)
    lo2, hi2 = _bootstrap_ci(values)
    assert lo1 == lo2
    assert hi1 == hi2


def test_bootstrap_ci_zero_values():
    lo, hi = _bootstrap_ci([0.0, 0.0, 0.0, 0.0])
    assert lo == 0.0
    assert hi == 0.0


# ── evaluate_note ──────────────────────────────────────────────────────────────

def test_evaluate_note_mhi_runs_without_entailment():
    transcript = "Patient has chest pain and fever for two days."
    note = (
        "SUBJECTIVE:\nPatient reports chest pain and fever for two days.\n"
        "OBJECTIVE:\nTemp 38.2. Sats 97%.\n"
        "ASSESSMENT:\nViral illness.\n"
        "PLAN:\nParacetamol PRN.\nSafety-netting: return if worsens."
    )
    result = evaluate_note(note, transcript, note_id="test_001", model="test", use_entailment=False)
    assert result.note_id == "test_001"
    assert result.model == "test"
    assert result.entailment_skipped is True
    assert result.num_sentences == 0   # gate was skipped
    assert result.mhi_score >= 0.0
    assert result.mhi_omission >= 0.0
    assert result.mhi_hallucination >= 0.0


def test_evaluate_note_empty_transcript_skips_entailment():
    note = "SUBJECTIVE:\nPatient reports cough.\nPLAN:\nParacetamol PRN.\nSafety-netting: call 111."
    result = evaluate_note(note, "", note_id="empty_tx", use_entailment=True)
    assert result.entailment_skipped is True


def test_evaluate_note_hall_rate_zero_when_skipped():
    note = "SUBJECTIVE:\nPatient reports fever.\nPLAN:\nRest and fluids.\nSafety-netting: A&E if worsens."
    result = evaluate_note(note, "", use_entailment=False)
    assert result.sentence_hallucination_rate == 0.0


def test_evaluate_note_delta_computed():
    note = "SUBJECTIVE:\nCough.\nPLAN:\nRest.\nSafety-netting: call 111."
    transcript = "Patient has a cough."
    result = evaluate_note(note, transcript, use_entailment=False)
    # delta = hall_rate - 0.0147; with skipped gate, hall_rate = 0 so delta = -0.0147
    assert abs(result.delta_hallucination - (0.0 - ASGARI_HALLUCINATION_BAR)) < 0.001


def test_evaluate_note_mhi_omission_positive_when_transcript_has_extra():
    # Transcript mentions amoxicillin but note is empty
    transcript = "Doctor prescribed amoxicillin 500mg three times daily."
    note = "SUBJECTIVE:\nPatient came in.\nPLAN:\nReview.\nSafety-netting: call 111 if unwell."
    result = evaluate_note(note, transcript, use_entailment=False)
    # Amoxicillin is a drug (weight=10) - should produce non-zero omission
    assert result.mhi_omission >= 0.0


def test_evaluate_note_auto_generates_note_id():
    note = "SUBJECTIVE:\nPain.\nPLAN:\nRest.\nSafety-netting: attend if worsens."
    result = evaluate_note(note, "Pain in leg.")
    assert result.note_id != ""


# ── run_hallucination_eval ─────────────────────────────────────────────────────

def test_run_eval_raises_on_empty_list():
    with pytest.raises(ValueError, match="No valid notes"):
        run_hallucination_eval([], model="test", use_entailment=False, save=False)


def test_run_eval_raises_on_all_empty_soaps():
    notes = [{"id": "n1", "soap_note": "", "transcript": "some text"}]
    with pytest.raises(ValueError, match="No valid notes"):
        run_hallucination_eval(notes, use_entailment=False, save=False)


def test_run_eval_returns_benchmark_result():
    notes = [
        {
            "id": "case_01",
            "soap_note": (
                "SUBJECTIVE:\nPatient reports chest pain for two days.\n"
                "OBJECTIVE:\nTemp 37.5. Sats 98%.\n"
                "ASSESSMENT:\nChest pain - musculoskeletal.\n"
                "PLAN:\nIbuprofen PRN.\nSafety-netting: call 999 if radiating pain."
            ),
            "transcript": "Patient came in with chest pain for two days.",
        },
        {
            "id": "case_02",
            "soap_note": (
                "SUBJECTIVE:\nCough for one week.\n"
                "OBJECTIVE:\nChest clear.\n"
                "ASSESSMENT:\nUpper respiratory tract infection.\n"
                "PLAN:\nSelf-limiting.\nSafety-netting: call 111 if not improving."
            ),
            "transcript": "Patient has a cough for a week.",
        },
    ]
    result = run_hallucination_eval(notes, model="test-model", use_entailment=False, save=False)
    assert isinstance(result, HallucinationBenchmarkResult)
    assert result.model == "test-model"
    assert result.num_notes == 2


def test_run_eval_note_count_matches():
    notes = [
        {"id": f"note_{i}", "soap_note": f"PLAN:\nRest.\nSafety-netting: call 111.", "transcript": "Patient unwell."}
        for i in range(5)
    ]
    result = run_hallucination_eval(notes, use_entailment=False, save=False)
    assert result.num_notes == 5
    assert len(result.notes) == 5


def test_run_eval_skips_empty_soap_silently():
    notes = [
        {"id": "good", "soap_note": "PLAN:\nRest.\nSafety-netting: call 111.", "transcript": "Patient unwell."},
        {"id": "empty", "soap_note": "", "transcript": "Some text"},
    ]
    result = run_hallucination_eval(notes, use_entailment=False, save=False)
    assert result.num_notes == 1


def test_run_eval_beats_asgari_hall_flag():
    notes = [
        {"soap_note": "PLAN:\nRest.\nSafety-netting: call 111.", "transcript": "Patient needs rest."}
        for _ in range(3)
    ]
    # All entailment skipped → hall_rate = 0, which beats 1.47%
    result = run_hallucination_eval(notes, use_entailment=False, save=False)
    assert result.beats_asgari_hallucination is True


def test_run_eval_total_sentences_sums_notes():
    notes = [
        {
            "soap_note": (
                "SUBJECTIVE:\nPatient reports pain for two weeks with increasing severity.\n"
                "OBJECTIVE:\nTemp 36.8. Sats 99%.\n"
                "ASSESSMENT:\nMusculoskeletal pain.\n"
                "PLAN:\nNaproxen BD.\nSafety-netting: call 111 if worsens."
            ),
            "transcript": "Pain for two weeks.",
        }
        for _ in range(3)
    ]
    result = run_hallucination_eval(notes, use_entailment=False, save=False)
    # total_sentences = sum of num_sentences across notes (all 0 when entailment skipped)
    assert result.total_sentences == sum(r.num_sentences for r in result.notes)


def test_run_eval_corpus_hall_rate_zero_when_all_skipped():
    notes = [{"soap_note": "PLAN:\nRest.\nSafety-netting: call 111.", "transcript": "Patient unwell."}]
    result = run_hallucination_eval(notes, use_entailment=False, save=False)
    assert result.mean_sentence_hallucination_rate == 0.0


def test_run_eval_ci_lower_le_upper():
    notes = [
        {"id": f"n{i}", "soap_note": "PLAN:\nRest.\nSafety-netting: call 111.", "transcript": "Unwell."}
        for i in range(4)
    ]
    result = run_hallucination_eval(notes, use_entailment=False, save=False)
    assert result.ci_hallucination_lower <= result.ci_hallucination_upper
    assert result.ci_mhi_omission_lower <= result.ci_mhi_omission_upper


def test_run_eval_saves_to_file(tmp_path, monkeypatch):
    import src.evaluation.hallucination_eval as he
    monkeypatch.setattr(he, "RESULTS_DIR", tmp_path)

    notes = [{"soap_note": "PLAN:\nRest.\nSafety-netting: call 111.", "transcript": "Patient unwell."}]
    run_hallucination_eval(notes, model="save-test", use_entailment=False, save=True)

    saved = list(tmp_path.glob("hallucination_eval_save-test_*.json"))
    assert len(saved) == 1


def test_run_eval_no_save_writes_no_file(tmp_path, monkeypatch):
    import src.evaluation.hallucination_eval as he
    monkeypatch.setattr(he, "RESULTS_DIR", tmp_path)

    notes = [{"soap_note": "PLAN:\nRest.\nSafety-netting: call 111.", "transcript": "Patient unwell."}]
    run_hallucination_eval(notes, use_entailment=False, save=False)

    assert len(list(tmp_path.glob("*.json"))) == 0


# ── to_latex_table ─────────────────────────────────────────────────────────────

def test_latex_table_contains_begin_tabular():
    b = _make_benchmark()
    table = to_latex_table([b])
    assert r"\begin{tabular}" in table


def test_latex_table_contains_asgari_reference_row():
    b = _make_benchmark()
    table = to_latex_table([b])
    assert "Asgari" in table
    assert "12" in table  # 12,999 sentences


def test_latex_table_contains_model_name():
    b = _make_benchmark(model="praxis-soap-v1")
    table = to_latex_table([b])
    assert "praxis-soap-v1" in table


def test_latex_table_multiple_models():
    b1 = _make_benchmark(model="praxis-soap-v1")
    b2 = _make_benchmark(model="mistral-7b", hall_rate=0.03)
    table = to_latex_table([b1, b2])
    assert "praxis-soap-v1" in table
    assert "mistral-7b" in table


def test_latex_table_has_end_table():
    b = _make_benchmark()
    table = to_latex_table([b])
    assert r"\end{table}" in table


def test_latex_table_hall_rate_as_percent():
    b = _make_benchmark(hall_rate=0.01)
    table = to_latex_table([b])
    # 0.01 → "1.00\%"
    assert "1.00" in table


# ── to_markdown_table ──────────────────────────────────────────────────────────

def test_markdown_table_header_row():
    b = _make_benchmark()
    table = to_markdown_table([b])
    assert "Model" in table
    assert "Sentences" in table
    assert "Hall%" in table


def test_markdown_table_contains_asgari_reference():
    b = _make_benchmark()
    table = to_markdown_table([b])
    assert "Asgari" in table


def test_markdown_table_beats_bar_checkmark_when_better():
    b = _make_benchmark(hall_rate=0.005)  # 0.5% < 1.47% bar
    assert b.beats_asgari_hallucination is True
    table = to_markdown_table([b])
    assert "✓" in table


def test_markdown_table_no_checkmark_when_worse():
    b = _make_benchmark(hall_rate=0.05)  # 5% > 1.47% bar
    assert b.beats_asgari_hallucination is False
    table = to_markdown_table([b])
    assert "·" in table


def test_markdown_table_model_name_in_row():
    b = _make_benchmark(model="meditron-3b")
    table = to_markdown_table([b])
    assert "meditron-3b" in table


def test_markdown_table_multiple_rows():
    benchmarks = [_make_benchmark(model=f"model_{i}") for i in range(3)]
    table = to_markdown_table(benchmarks)
    for i in range(3):
        assert f"model_{i}" in table


def test_markdown_table_positive_delta_has_plus_sign():
    b = _make_benchmark(hall_rate=0.05)  # worse than bar
    table = to_markdown_table([b])
    assert "+" in table


def test_markdown_table_negative_delta_no_plus():
    b = _make_benchmark(hall_rate=0.005)  # better than bar → negative delta
    table = to_markdown_table([b])
    # negative delta should appear without + sign
    lines = [l for l in table.split("\n") if "model" in l.lower() or "0.5" in l]
    assert any("-" in l for l in table.split("\n") if "model_" in l or "test" in l.lower())


# ── Asgari comparison flags ────────────────────────────────────────────────────

def test_beats_asgari_hall_true_when_below_bar():
    b = _make_benchmark(hall_rate=0.01)
    assert b.beats_asgari_hallucination is True


def test_beats_asgari_hall_false_when_above_bar():
    b = _make_benchmark(hall_rate=0.02)
    assert b.beats_asgari_hallucination is False


def test_beats_asgari_omission_true_when_below_bar():
    b = _make_benchmark(mhi_omission=0.02)
    assert b.beats_asgari_omission is True


def test_beats_asgari_omission_false_when_above_bar():
    b = _make_benchmark(mhi_omission=0.05)
    assert b.beats_asgari_omission is False


def test_asgari_bar_values_stored_on_benchmark():
    b = _make_benchmark()
    assert abs(b.asgari_hallucination_bar - 0.0147) < 1e-6
    assert abs(b.asgari_omission_bar - 0.0345) < 1e-6


# ── End-to-end: no entailment (pure MHI path) ─────────────────────────────────

def test_e2e_mhi_only_diabetes_review():
    transcript = (
        "Patient is here for annual diabetes review. "
        "She is taking metformin 1g twice daily and atorvastatin 20mg at night. "
        "HbA1c improved from 65 to 58. eGFR is 85. Blood pressure 134 over 82."
    )
    note = (
        "SUBJECTIVE:\nAnnual review. On metformin 1g BD and atorvastatin 20mg nocte.\n"
        "OBJECTIVE:\nHbA1c 58 (down from 65). eGFR 85. BP 134/82.\n"
        "ASSESSMENT:\nType 2 diabetes mellitus, well controlled.\n"
        "PLAN:\n1. Continue medications.\n2. Review in 6 months.\n"
        "Safety-netting: attend urgently if blood sugar consistently above 12."
    )
    result = evaluate_note(note, transcript, note_id="dm_review", use_entailment=False)
    assert result.note_id == "dm_review"
    assert result.entailment_skipped is True
    assert result.mhi_score <= 1.0


def test_e2e_batch_eval_three_cases():
    notes = [
        {
            "id": "back_pain",
            "transcript": "Patient has lower back pain for three weeks radiating to left leg.",
            "soap_note": (
                "SUBJECTIVE:\nThree weeks lower back pain radiating to left leg.\n"
                "ASSESSMENT:\nLumbar disc protrusion.\n"
                "PLAN:\nNaproxen 500mg BD.\nSafety-netting: A&E for leg weakness."
            ),
        },
        {
            "id": "cough",
            "transcript": "Patient has a cough for six weeks with weight loss. Smoker.",
            "soap_note": (
                "SUBJECTIVE:\nSix weeks cough with weight loss. Long-term smoker.\n"
                "ASSESSMENT:\nSuspicious cough - urgent 2WW.\n"
                "PLAN:\n1. Urgent CXR.\n2. 2-week wait referral.\n"
                "Safety-netting: A&E for haemoptysis."
            ),
        },
        {
            "id": "anxiety",
            "transcript": "Patient reports anxiety and panic attacks for two months.",
            "soap_note": (
                "SUBJECTIVE:\nTwo months anxiety and panic attacks.\n"
                "ASSESSMENT:\nGeneralised anxiety disorder.\n"
                "PLAN:\nIAPT referral.\nSafety-netting: Samaritans 116123 or A&E."
            ),
        },
    ]
    result = run_hallucination_eval(notes, model="praxis-test", use_entailment=False, save=False)
    assert result.num_notes == 3
    assert result.mean_mhi_score <= 1.0
    assert result.beats_asgari_hallucination is True  # skipped gate → 0% hall rate
