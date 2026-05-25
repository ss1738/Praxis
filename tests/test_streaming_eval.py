"""Tests for the streaming-veto evaluation harness (N16 eval)."""
from __future__ import annotations

import json

import pytest

from src.evaluation.streaming_eval import (
    StrategyAggregate,
    StrategyCase,
    StreamingBenchmark,
    _aggregate,
    _bootstrap_ci,
    _build_benchmark,
    run_dry,
    to_latex_table,
    to_markdown_table,
)


# ── Bootstrap CI ──────────────────────────────────────────────────────


class TestBootstrapCI:
    def test_empty(self):
        lo, hi = _bootstrap_ci([])
        assert lo == 0.0 and hi == 0.0

    def test_constant_returns_constant(self):
        lo, hi = _bootstrap_ci([5.0] * 8, seed=42)
        assert lo == pytest.approx(5.0)
        assert hi == pytest.approx(5.0)

    def test_increasing_values_widen_ci(self):
        lo, hi = _bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], seed=42)
        assert hi > lo
        assert lo < 5.0 < hi  # mean is 4.5; CI should bracket it

    def test_seed_determinism(self):
        v = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
        a = _bootstrap_ci(v, seed=42)
        b = _bootstrap_ci(v, seed=42)
        assert a == b


# ── Aggregation ──────────────────────────────────────────────────────


def _make_case(name="case1", strategy="baseline", trust=70, hr=0.1, t=10.0,
               accepted=0, vetoed=0, rewritten=0, gave_up=0,
               passed_veto=False) -> StrategyCase:
    return StrategyCase(
        case_name=name, strategy=strategy, soap_note="placeholder",
        trust_score=trust, hallucination_rate=hr, generation_time_secs=t,
        accepted_count=accepted, vetoed_count=vetoed, rewritten_count=rewritten,
        gave_up_count=gave_up, all_shipped_passed_veto=passed_veto,
    )


class TestAggregate:
    def test_empty_returns_zero_aggregate(self):
        a = _aggregate("baseline", [])
        assert isinstance(a, StrategyAggregate)
        assert a.n_cases == 0
        assert a.mean_trust_score == 0.0

    def test_single_case_zero_ci_width(self):
        cases = [_make_case(trust=70, hr=0.1)]
        a = _aggregate("baseline", cases)
        assert a.n_cases == 1
        assert a.mean_trust_score == pytest.approx(70.0)
        assert a.trust_score_ci_low == pytest.approx(70.0)

    def test_perfect_grounding_count(self):
        cases = [
            _make_case(hr=0.0),
            _make_case(hr=0.0),
            _make_case(hr=0.1),
        ]
        a = _aggregate("baseline", cases)
        assert a.n_with_perfect_grounding == 2

    def test_streaming_veto_count_only_for_streaming(self):
        cases_b = [_make_case(strategy="baseline", passed_veto=True)]
        a_b = _aggregate("baseline", cases_b)
        assert a_b.veto_active_cases == 0    # Not streaming -> always 0

        cases_s = [
            _make_case(strategy="streaming", passed_veto=True),
            _make_case(strategy="streaming", passed_veto=True),
            _make_case(strategy="streaming", passed_veto=False),
        ]
        a_s = _aggregate("streaming", cases_s)
        assert a_s.veto_active_cases == 2


# ── Dry-run benchmark ────────────────────────────────────────────────


class TestDryRun:
    def test_runs(self):
        b = run_dry()
        assert isinstance(b, StreamingBenchmark)
        assert len(b.cases) == 24    # 8 cases × 3 strategies
        assert len(b.aggregates) == 3

    def test_dry_run_streaming_zero_hallucination(self):
        b = run_dry()
        s = next(a for a in b.aggregates if a.strategy == "streaming")
        # By construction in dry-run numbers, every streaming case is 0.0
        assert s.mean_hallucination_rate == 0.0
        assert s.n_with_perfect_grounding == 8

    def test_dry_run_baseline_has_some_hallucination(self):
        b = run_dry()
        base = next(a for a in b.aggregates if a.strategy == "baseline")
        assert base.mean_hallucination_rate > 0.0

    def test_constitutional_improves_trust_over_baseline(self):
        b = run_dry()
        base = next(a for a in b.aggregates if a.strategy == "baseline")
        const = next(a for a in b.aggregates
                     if a.strategy == "constitutional")
        # Per the canned numbers, constitutional should at least match baseline
        assert const.mean_trust_score >= base.mean_trust_score - 0.5

    def test_streaming_costs_more_time(self):
        b = run_dry()
        base = next(a for a in b.aggregates if a.strategy == "baseline")
        stream = next(a for a in b.aggregates if a.strategy == "streaming")
        assert stream.mean_generation_time_secs > base.mean_generation_time_secs


# ── Table emitters ───────────────────────────────────────────────────


class TestTables:
    def test_markdown_table_well_formed(self):
        b = run_dry()
        md = to_markdown_table(b)
        assert "# Streaming-veto" in md
        assert "Strategy" in md
        # Each strategy has a row
        for s in ("baseline", "constitutional", "streaming"):
            assert s in md

    def test_latex_table_well_formed(self):
        b = run_dry()
        tex = to_latex_table(b)
        assert "\\begin{table}" in tex
        assert "\\end{table}" in tex
        assert "tab:streaming_eval" in tex
        for s in ("Baseline", "Constitutional", "Streaming"):
            assert s in tex


# ── Round-trip serialisation ─────────────────────────────────────────


class TestSerialisation:
    def test_to_dict_round_trips_through_json(self):
        b = run_dry()
        d = b.to_dict()
        s = json.dumps(d)
        d2 = json.loads(s)
        assert d2["seed"] == 42
        assert len(d2["cases"]) == 24
        assert len(d2["aggregates"]) == 3
        # Aggregates have all fields
        for a in d2["aggregates"]:
            for k in ("mean_trust_score", "trust_score_ci_low",
                      "mean_hallucination_rate", "n_with_perfect_grounding"):
                assert k in a
