"""Tests for Constitutional vs Baseline evaluation framework."""

from __future__ import annotations

import pytest

from src.evaluation.constitutional_eval import (
    DEMO_TRANSCRIPTS,
    CaseResult,
    ConstitutionalBenchmarkResult,
    _aggregate,
    dry_run,
    to_latex_table,
    to_markdown_table,
)


# ── DEMO_TRANSCRIPTS ──────────────────────────────────────────────────────────

def test_demo_transcripts_has_eight_cases():
    assert len(DEMO_TRANSCRIPTS) == 8


def test_demo_transcripts_keys():
    expected = {
        "Lower Back Pain",
        "Persistent Cough",
        "Diabetes Review",
        "Mental Health - Anxiety",
        "Chest Infection",
        "Hypertension Review",
        "Paediatric Ear Infection",
        "Skin Lesion Assessment",
    }
    assert set(DEMO_TRANSCRIPTS.keys()) == expected


def test_demo_transcripts_non_empty():
    for name, text in DEMO_TRANSCRIPTS.items():
        assert len(text) > 100, f"Transcript for '{name}' is too short"


# ── CaseResult ────────────────────────────────────────────────────────────────

def _make_case(
    name="Test Case",
    baseline=55,
    const_final=75,
    converged=True,
    n_iter=2,
) -> CaseResult:
    return CaseResult(
        case_name=name,
        baseline_trust_score=baseline,
        baseline_verdict="review",
        constitutional_initial_score=baseline,
        constitutional_final_score=const_final,
        constitutional_verdict="approved" if converged else "review",
        improvement=const_final - baseline,
        score_trajectory=[baseline, const_final],
        converged=converged,
        num_iterations=n_iter,
        repair_targets=["clinical_safety"],
        baseline_generation_secs=2.0,
        constitutional_total_secs=6.0,
    )


def test_case_result_improvement_computed():
    c = _make_case(baseline=55, const_final=75)
    assert c.improvement == 20


def test_case_result_converged_flag():
    c = _make_case(const_final=72, converged=True)
    assert c.converged is True


def test_case_result_trajectory_length():
    c = _make_case()
    assert len(c.score_trajectory) == 2


# ── _aggregate ────────────────────────────────────────────────────────────────

def test_aggregate_mean_baseline():
    cases = [_make_case(baseline=50), _make_case(baseline=60)]
    result = _aggregate("test-model", cases)
    assert result.mean_baseline_score == 55.0


def test_aggregate_mean_constitutional():
    cases = [_make_case(const_final=70), _make_case(const_final=80)]
    result = _aggregate("test-model", cases)
    assert result.mean_constitutional_score == 75.0


def test_aggregate_mean_improvement():
    cases = [
        _make_case(baseline=50, const_final=70),
        _make_case(baseline=60, const_final=80),
    ]
    result = _aggregate("test-model", cases)
    assert result.mean_improvement == 20.0


def test_aggregate_converged_count():
    cases = [
        _make_case(converged=True),
        _make_case(converged=True),
        _make_case(converged=False),
    ]
    result = _aggregate("test-model", cases)
    assert result.converged_count == 2


def test_aggregate_baseline_approved_count():
    cases = [
        _make_case(baseline=75, converged=True),   # baseline ≥70
        _make_case(baseline=55, converged=True),   # baseline <70
        _make_case(baseline=65, converged=False),  # baseline <70
    ]
    result = _aggregate("test-model", cases)
    assert result.baseline_approved_count == 1


def test_aggregate_num_cases():
    cases = [_make_case() for _ in range(5)]
    result = _aggregate("test-model", cases)
    assert result.num_cases == 5


def test_aggregate_mean_iterations():
    cases = [_make_case(n_iter=1), _make_case(n_iter=3)]
    result = _aggregate("test-model", cases)
    assert result.mean_iterations == 2.0


def test_aggregate_stores_cases():
    cases = [_make_case("Case A"), _make_case("Case B")]
    result = _aggregate("test-model", cases)
    assert len(result.cases) == 2
    assert result.cases[0].case_name == "Case A"


# ── dry_run ───────────────────────────────────────────────────────────────────

def test_dry_run_returns_benchmark_result():
    result = dry_run()
    assert isinstance(result, ConstitutionalBenchmarkResult)


def test_dry_run_has_eight_cases():
    result = dry_run()
    assert result.num_cases == 8


def test_dry_run_scores_in_range():
    result = dry_run()
    assert 0 <= result.mean_baseline_score <= 100
    assert 0 <= result.mean_constitutional_score <= 100


def test_dry_run_mean_improvement_non_negative():
    result = dry_run()
    # constitutional should never do worse than baseline
    assert result.mean_constitutional_score >= result.mean_baseline_score


def test_dry_run_converged_count_plausible():
    result = dry_run()
    assert 0 <= result.converged_count <= 8


def test_dry_run_case_names_match_transcripts():
    result = dry_run()
    dry_names = {c.case_name for c in result.cases}
    assert dry_names == set(DEMO_TRANSCRIPTS.keys())


# ── to_latex_table ────────────────────────────────────────────────────────────

def test_latex_table_structure():
    result = dry_run()
    latex = to_latex_table(result)
    assert r"\begin{table}" in latex
    assert r"\end{table}" in latex
    assert r"\toprule" in latex
    assert r"\bottomrule" in latex


def test_latex_table_has_all_case_names():
    result = dry_run()
    latex = to_latex_table(result)
    for case in result.cases:
        assert case.case_name in latex


def test_latex_table_has_mean_row():
    result = dry_run()
    latex = to_latex_table(result)
    assert r"\textbf{Mean}" in latex


def test_latex_table_has_converged_fraction():
    result = dry_run()
    latex = to_latex_table(result)
    assert f"{result.converged_count}/{result.num_cases}" in latex


# ── to_markdown_table ─────────────────────────────────────────────────────────

def test_markdown_table_has_headers():
    result = dry_run()
    md = to_markdown_table(result)
    assert "Baseline" in md
    assert "Constitutional" in md
    assert "Iterations" in md


def test_markdown_table_has_all_case_names():
    result = dry_run()
    md = to_markdown_table(result)
    for case in result.cases:
        assert case.case_name in md


def test_markdown_table_shows_model_name():
    result = dry_run()
    md = to_markdown_table(result)
    assert result.model in md


def test_markdown_table_shows_mean_improvement():
    result = dry_run()
    md = to_markdown_table(result)
    assert "Mean improvement" in md


def test_markdown_table_shows_converged_fraction():
    result = dry_run()
    md = to_markdown_table(result)
    assert f"{result.converged_count}/{result.num_cases}" in md
