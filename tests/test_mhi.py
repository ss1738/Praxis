"""Tests for Medical Hallucination Index (MHI)."""

from __future__ import annotations

import pytest

from src.evaluation.mhi import (
    MHILexicon,
    MHIResult,
    _normalise,
    compute_mhi,
    run_mhi_benchmark,
    to_latex_table,
    to_markdown_table,
)


# ── Lexicon weights ───────────────────────────────────────────────────────────

def test_drug_name_weight_ten():
    lex = MHILexicon()
    assert lex.weight("amoxicillin") == 10
    assert lex.weight("warfarin") == 10
    assert lex.weight("metformin") == 10


def test_red_flag_weight_ten():
    lex = MHILexicon()
    assert lex.weight("melaena") == 10
    assert lex.weight("haemoptysis") == 10
    assert lex.weight("suicidal") == 10


def test_negation_weight_nine():
    lex = MHILexicon()
    assert lex.weight("no") == 9
    assert lex.weight("denies") == 9
    assert lex.weight("not") == 9


def test_lab_term_weight_eight():
    lex = MHILexicon()
    assert lex.weight("hba1c") == 8
    assert lex.weight("egfr") == 8
    assert lex.weight("troponin") == 8


def test_diagnosis_weight_seven():
    lex = MHILexicon()
    assert lex.weight("diabetes") == 7
    assert lex.weight("hypertension") == 7


def test_general_clinical_weight_four():
    lex = MHILexicon()
    assert lex.weight("pain") == 4
    assert lex.weight("fever") == 4
    assert lex.weight("cough") == 4


def test_stopword_weight_one():
    lex = MHILexicon()
    assert lex.weight("the") == 1
    assert lex.weight("patient") == 1
    assert lex.weight("has") == 1


def test_dose_pattern_weight_nine():
    lex = MHILexicon()
    assert lex.weight("500mg") == 9
    assert lex.weight("10mg") == 9
    assert lex.weight("bd") == 9
    assert lex.weight("tds") == 9
    assert lex.weight("stat") == 9


def test_case_insensitive():
    lex = MHILexicon()
    assert lex.weight("Amoxicillin") == 10
    assert lex.weight("WARFARIN") == 10
    assert lex.weight("HbA1c") == 8


# ── Normalisation ─────────────────────────────────────────────────────────────

def test_normalise_lowercases():
    tokens = _normalise("PATIENT has Fever")
    assert all(t == t.lower() for t in tokens)


def test_normalise_strips_html_tags():
    tokens = _normalise("<b>chest pain</b> patient")
    assert "b" not in tokens
    assert "patient" in tokens


def test_normalise_collapses_multi_word():
    tokens = _normalise("patient reports chest pain and shortness of breath")
    assert "chest_pain" in tokens
    assert "shortness_of_breath" in tokens


def test_normalise_returns_list():
    assert isinstance(_normalise("hello world"), list)


# ── compute_mhi ───────────────────────────────────────────────────────────────

def test_perfect_match_score_one():
    ref = "patient prescribed amoxicillin 500mg for chest infection"
    result = compute_mhi(ref, ref)
    assert result.mhi_score == 1.0
    assert result.mhi_omission == 0.0


def test_missing_drug_penalised_heavily():
    ref = "patient prescribed amoxicillin 500mg"
    hyp = "patient prescribed 500mg"
    result = compute_mhi(ref, hyp)
    assert result.mhi_omission > 0
    assert "amoxicillin" in [t for t, _ in result.top_omissions]


def test_missing_stopword_low_impact():
    ref = "the patient has a cough"
    hyp = "patient has cough"
    result = compute_mhi(ref, hyp)
    # "the" and "a" are weight-1, ignored; "cough" weight-4 present - omission should be low
    assert result.mhi_omission < 0.2


def test_hallucination_detected():
    ref = "patient has a cough"
    hyp = "patient has amoxicillin for chest infection"
    result = compute_mhi(ref, hyp)
    assert result.mhi_hallucination > 0
    fabricated_terms = [t for t, _ in result.top_hallucinations]
    assert "amoxicillin" in fabricated_terms or result.mhi_hallucination > 0


def test_red_flag_omission_high_cost():
    ref = "patient with melaena and weight loss"
    hyp = "patient with some symptoms"
    result = compute_mhi(ref, hyp)
    # melaena=10, weight_loss (as "weight" + "loss" split) - omission should be significant
    assert result.mhi_omission > 0.5


def test_mhi_score_between_zero_and_one():
    result = compute_mhi("amoxicillin warfarin hba1c", "completely different text here now")
    assert 0.0 <= result.mhi_score <= 1.0
    assert 0.0 <= result.mhi_omission <= 1.0
    assert 0.0 <= result.mhi_hallucination <= 1.0


def test_empty_hypothesis_max_omission():
    ref = "amoxicillin 500mg bd for chest infection"
    result = compute_mhi(ref, "")
    assert result.mhi_omission == 1.0
    assert result.mhi_score == 0.0


def test_empty_reference_zero_omission():
    result = compute_mhi("", "amoxicillin 500mg bd")
    assert result.mhi_omission == 0.0
    assert result.mhi_score == 1.0


def test_negation_flip_penalised():
    # "no chest pain" in reference - "no" is weight-9
    # hypothesis drops "no" - this is a negation flip (could hallucinate chest pain)
    ref = "no chest pain no breathlessness"
    hyp = "chest pain breathlessness"
    result = compute_mhi(ref, hyp)
    assert result.mhi_omission > 0
    negation_terms = [t for t, _ in result.top_omissions]
    assert "no" in negation_terms


def test_top_omissions_sorted_by_weight():
    ref = "amoxicillin melaena paracetamol cough fever"
    hyp = "some other content here"
    result = compute_mhi(ref, hyp)
    if len(result.top_omissions) >= 2:
        weights = [w for _, w in result.top_omissions]
        assert weights == sorted(weights, reverse=True)


def test_dose_in_reference_penalised_if_missing():
    ref = "prescribe amoxicillin 500mg tds for 5 days"
    hyp = "prescribe amoxicillin for infection"
    result = compute_mhi(ref, hyp)
    # 500mg and tds are weight-9 and missing from hyp
    dose_omitted = any(t in ("500mg", "tds") for t, _ in result.top_omissions)
    assert dose_omitted


# ── run_mhi_benchmark ─────────────────────────────────────────────────────────

def test_benchmark_returns_correct_count():
    refs = ["amoxicillin for infection", "warfarin anticoagulation"]
    hyps = ["amoxicillin prescribed", "warfarin started"]
    bench = run_mhi_benchmark(refs, hyps, model_name="test", save=False)
    assert bench.num_samples == 2


def test_benchmark_mean_score_in_range():
    refs = ["patient has chest pain and fever"] * 5
    hyps = ["patient has chest pain"] * 5
    bench = run_mhi_benchmark(refs, hyps, model_name="test", save=False)
    assert 0.0 <= bench.mean_mhi_score <= 1.0
    assert 0.0 <= bench.mean_omission <= 1.0


def test_benchmark_with_wer_computes_delta():
    refs = ["amoxicillin 500mg for infection"] * 3
    hyps = ["amoxicillin prescribed"] * 3
    wer_vals = [0.40, 0.35, 0.45]
    bench = run_mhi_benchmark(refs, hyps, model_name="test", wer_values=wer_vals, save=False)
    assert bench.mean_wer is not None
    assert bench.mhi_vs_wer_delta is not None


def test_benchmark_raises_on_mismatched_lengths():
    with pytest.raises(ValueError):
        run_mhi_benchmark(["ref1", "ref2"], ["hyp1"], save=False)


def test_benchmark_perfect_match():
    text = ["amoxicillin 500mg tds warfarin hba1c egfr"]
    bench = run_mhi_benchmark(text, text, model_name="test", save=False)
    assert bench.mean_mhi_score == 1.0
    assert bench.mean_omission == 0.0


# ── Output formatting ─────────────────────────────────────────────────────────

def test_latex_table_contains_model_name():
    refs = ["amoxicillin"]
    hyps = ["amoxicillin"]
    bench = run_mhi_benchmark(refs, hyps, model_name="praxis-conformer", save=False)
    latex = to_latex_table([bench])
    assert "praxis-conformer" in latex
    assert r"\begin{table}" in latex
    assert r"\end{table}" in latex


def test_markdown_table_contains_headers():
    refs = ["amoxicillin"]
    hyps = ["amoxicillin"]
    bench = run_mhi_benchmark(refs, hyps, model_name="test-model", save=False)
    md = to_markdown_table([bench])
    assert "MHI Score" in md
    assert "Omission" in md
    assert "Hallucination" in md
