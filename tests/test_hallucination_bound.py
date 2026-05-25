"""Tests for the streaming-veto hallucination bound (N17)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.evaluation.hallucination_bound import (
    HallucinationBoundResult,
    bound_grid,
    compute_bound,
    estimate_delta_default,
    estimate_p_fab_from_baseline,
    verify_bound,
)


# ── Pure-math bound ──────────────────────────────────────────────────


class TestComputeBound:
    def test_returns_bound_record(self):
        b = compute_bound(p=0.10, delta=0.05, k=3, N=30)
        assert isinstance(b, HallucinationBoundResult)
        assert b.p == 0.10
        assert b.delta == 0.05
        assert b.k == 3
        assert b.N == 30

    def test_zero_fabrication_propensity_gives_zero_bound(self):
        b = compute_bound(p=0.0, delta=0.05, k=3, N=30)
        assert b.per_position_bound == 0.0
        assert b.note_level_bound == 0.0
        assert b.is_meaningful is True

    def test_zero_delta_gives_zero_bound(self):
        # If the gate is perfect (delta = 0), no fabrication ever ships
        b = compute_bound(p=0.5, delta=0.0, k=3, N=30)
        assert b.per_position_bound == 0.0
        assert b.note_level_bound == 0.0

    def test_bound_increases_with_p(self):
        a = compute_bound(p=0.05, delta=0.10, k=3, N=30)
        b = compute_bound(p=0.20, delta=0.10, k=3, N=30)
        assert b.note_level_bound > a.note_level_bound

    def test_bound_increases_with_delta(self):
        a = compute_bound(p=0.10, delta=0.01, k=3, N=30)
        b = compute_bound(p=0.10, delta=0.20, k=3, N=30)
        assert b.note_level_bound > a.note_level_bound

    def test_finite_k_strictly_below_asymptotic(self):
        b = compute_bound(p=0.30, delta=0.10, k=3, N=30)
        # For finite k, geometric tail is positive
        assert b.per_position_bound < b.per_position_asymptotic

    def test_asymptotic_limit_as_k_grows(self):
        large_k = compute_bound(p=0.10, delta=0.10, k=100, N=30)
        # Should be very close to the asymptotic value
        assert math.isclose(large_k.per_position_bound,
                            large_k.per_position_asymptotic,
                            rel_tol=1e-6)

    def test_note_level_bound_capped_at_one(self):
        # With nearly-saturated p and delta, the union bound clamps at 1.0
        b = compute_bound(p=0.99, delta=0.99, k=3, N=30)
        assert b.note_level_bound == 1.0
        assert b.is_meaningful is False

    def test_meaningful_regime(self):
        # Modest p and delta give a meaningful bound
        b = compute_bound(p=0.10, delta=0.05, k=3, N=30)
        assert b.is_meaningful is True
        # Sanity: 30 * 0.10 * 0.05 / (1 - 0.10*0.95) ~ 0.166
        assert 0.10 < b.note_level_bound < 0.25

    @pytest.mark.parametrize("bad", [-0.1, 1.5])
    def test_invalid_p_rejected(self, bad):
        with pytest.raises(ValueError):
            compute_bound(p=bad, delta=0.1, k=3, N=30)

    @pytest.mark.parametrize("bad", [-0.1, 1.5])
    def test_invalid_delta_rejected(self, bad):
        with pytest.raises(ValueError):
            compute_bound(p=0.1, delta=bad, k=3, N=30)

    def test_invalid_k_or_N_rejected(self):
        with pytest.raises(ValueError):
            compute_bound(p=0.1, delta=0.1, k=-1, N=30)
        with pytest.raises(ValueError):
            compute_bound(p=0.1, delta=0.1, k=3, N=-1)


# ── Grid sweep ───────────────────────────────────────────────────────


class TestBoundGrid:
    def test_dimensions(self):
        p_grid = [0.05, 0.10, 0.50]
        delta_grid = [0.01, 0.05, 0.10, 0.20]
        g = bound_grid(p_grid, delta_grid, k=3, N=30)
        assert g["p_grid"] == p_grid
        assert g["delta_grid"] == delta_grid
        assert g["k"] == 3 and g["N"] == 30
        m = g["note_level_bound_matrix"]
        assert len(m) == len(p_grid)
        assert len(m[0]) == len(delta_grid)

    def test_matrix_values_match_compute_bound(self):
        p_grid = [0.05, 0.50]
        delta_grid = [0.01, 0.20]
        g = bound_grid(p_grid, delta_grid, k=3, N=30)
        for i, p in enumerate(p_grid):
            for j, d in enumerate(delta_grid):
                expected = compute_bound(p, d, k=3, N=30).note_level_bound
                assert math.isclose(g["note_level_bound_matrix"][i][j],
                                    expected, rel_tol=1e-9)


# ── Empirical estimators ─────────────────────────────────────────────


class TestEstimateP:
    def test_no_cases(self):
        p, diag = estimate_p_fab_from_baseline({"cases": []})
        assert p == 0.0
        assert "no cases" in diag.get("reason", "")

    def test_baseline_only(self):
        data = {"cases": [
            {"strategy": "baseline", "hallucination_rate": 0.10},
            {"strategy": "baseline", "hallucination_rate": 0.20},
            {"strategy": "constitutional", "hallucination_rate": 0.05},
        ]}
        p, diag = estimate_p_fab_from_baseline(data)
        assert math.isclose(p, 0.15)
        assert diag["n_cases"] == 2
        assert diag["min"] == 0.10 and diag["max"] == 0.20

    def test_strategy_filter(self):
        data = {"cases": [
            {"strategy": "baseline", "hallucination_rate": 0.10},
            {"strategy": "constitutional", "hallucination_rate": 0.30},
        ]}
        p_b, _ = estimate_p_fab_from_baseline(data, strategy="baseline")
        p_c, _ = estimate_p_fab_from_baseline(data, strategy="constitutional")
        assert math.isclose(p_b, 0.10)
        assert math.isclose(p_c, 0.30)


class TestEstimateDelta:
    def test_default_returns_conservative_value(self):
        d, diag = estimate_delta_default()
        assert 0.0 < d < 1.0
        assert "default" in diag["source"].lower()


# ── End-to-end verifier ──────────────────────────────────────────────


class TestVerifyBound:
    def test_against_synthetic_streaming_eval(self, tmp_path):
        # Build a small synthetic streaming_eval payload
        synthetic = {
            "cases": [
                {"strategy": "baseline", "hallucination_rate": 0.10},
                {"strategy": "baseline", "hallucination_rate": 0.10},
                {"strategy": "streaming", "hallucination_rate": 0.0},
                {"strategy": "streaming", "hallucination_rate": 0.0},
            ]
        }
        f = tmp_path / "stream_eval.json"
        f.write_text(json.dumps(synthetic))

        result = verify_bound(f, delta=0.05, k=3, N=30)
        assert math.isclose(result["estimated_p"], 0.10)
        assert result["delta"] == 0.05
        assert result["theoretical_epsilon"] > 0.0
        assert result["empirical_streaming_rate"] == 0.0
        assert result["bound_holds"] is True
        # 30 * 0.10 * 0.05 / (1 - 0.10*0.95) ~ 0.166
        assert 0.10 < result["theoretical_epsilon"] < 0.25

    def test_meaningful_flag_propagates(self, tmp_path):
        # Adversarial regime where the bound is not meaningful (epsilon=1)
        synthetic = {
            "cases": [
                {"strategy": "baseline", "hallucination_rate": 0.95},
                {"strategy": "streaming", "hallucination_rate": 0.0},
            ]
        }
        f = tmp_path / "stream_eval.json"
        f.write_text(json.dumps(synthetic))
        result = verify_bound(f, delta=0.5, k=3, N=30)
        assert result["is_meaningful"] is False
        assert result["theoretical_epsilon"] == 1.0
