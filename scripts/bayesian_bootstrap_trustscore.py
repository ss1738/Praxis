#!/usr/bin/env python3
"""Bayesian bootstrap (Rubin 1981) for TrustScore per-model means.

Given a scored SOAP outputs JSON (from score_soap_outputs.py), compute the 95%
Bayesian credible interval on the mean TrustScore for each model. The Bayesian
bootstrap re-weights each observation by Dirichlet(1, ..., 1) draws rather than
the frequentist re-sampling-with-replacement, which is more appropriate for
small-N (n=8 cases per model) inference.

Output: per-model {mean, ci_low, ci_high, n} table + a paper-grade summary.

Usage:
    python3 scripts/bayesian_bootstrap_trustscore.py \
        --input data/evaluation_results/saturation_8model_scored.json \
        --output data/evaluation_results/bayesian_ci_trustscore.json \
        --n-iters 10000 --seed 42

Reference: Rubin, D. B. (1981). The Bayesian Bootstrap. Annals of Statistics 9(1).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def bayesian_bootstrap_mean(values: list[float], n_iters: int, rng: np.random.Generator) -> np.ndarray:
    """Return n_iters Bayesian-bootstrap draws of the mean of `values`."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return np.array([])
    weights = rng.dirichlet(np.ones(n), size=n_iters)  # shape (n_iters, n)
    return weights @ arr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="Scored SOAP outputs JSON (from score_soap_outputs.py)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n-iters", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ci-level", type=float, default=0.95)
    args = parser.parse_args()

    with args.input.open() as f:
        records = json.load(f)

    # Group TrustScores by model
    by_model: dict[str, list[float]] = {}
    by_model_auto: dict[str, list[float]] = {}
    for rec in records:
        if rec.get("trustscore") is None:
            continue
        m = rec["model_id"]
        by_model.setdefault(m, []).append(float(rec["trustscore"]))
        # Auto-rubric: try several common keys
        auto = rec.get("overall_score") or rec.get("auto_rubric") or rec.get("content_score")
        if auto is not None:
            by_model_auto.setdefault(m, []).append(float(auto))

    if not by_model:
        raise SystemExit("No TrustScore values found in input.")

    rng = np.random.default_rng(args.seed)
    alpha = (1.0 - args.ci_level) / 2.0
    summary: dict[str, dict] = {}
    for model, scores in by_model.items():
        ts_draws = bayesian_bootstrap_mean(scores, args.n_iters, rng)
        ts_low, ts_high = np.percentile(ts_draws, [alpha * 100, (1 - alpha) * 100])
        rec = {
            "n": len(scores),
            "trustscore_mean": float(np.mean(scores)),
            "trustscore_bayes_ci_low": float(ts_low),
            "trustscore_bayes_ci_high": float(ts_high),
        }
        if model in by_model_auto:
            auto_draws = bayesian_bootstrap_mean(by_model_auto[model], args.n_iters, rng)
            auto_low, auto_high = np.percentile(auto_draws, [alpha * 100, (1 - alpha) * 100])
            rec["auto_rubric_mean"] = float(np.mean(by_model_auto[model]))
            rec["auto_rubric_bayes_ci_low"] = float(auto_low)
            rec["auto_rubric_bayes_ci_high"] = float(auto_high)
        summary[model] = rec

    # Pairwise non-overlap analysis (which model pairs have non-overlapping 95% CIs?)
    pairs = []
    models = sorted(summary.keys(), key=lambda m: -summary[m]["trustscore_mean"])
    for i, m1 in enumerate(models):
        for m2 in models[i + 1:]:
            non_overlap_ts = (
                summary[m1]["trustscore_bayes_ci_low"] > summary[m2]["trustscore_bayes_ci_high"]
                or summary[m2]["trustscore_bayes_ci_low"] > summary[m1]["trustscore_bayes_ci_high"]
            )
            non_overlap_auto = None
            if "auto_rubric_bayes_ci_low" in summary[m1] and "auto_rubric_bayes_ci_low" in summary[m2]:
                non_overlap_auto = (
                    summary[m1]["auto_rubric_bayes_ci_low"] > summary[m2]["auto_rubric_bayes_ci_high"]
                    or summary[m2]["auto_rubric_bayes_ci_low"] > summary[m1]["auto_rubric_bayes_ci_high"]
                )
            pairs.append({
                "m1": m1,
                "m2": m2,
                "trustscore_non_overlap": bool(non_overlap_ts),
                "auto_rubric_non_overlap": non_overlap_auto,
            })

    out = {
        "config": {
            "n_iters": args.n_iters,
            "seed": args.seed,
            "ci_level": args.ci_level,
        },
        "per_model": summary,
        "pairwise_non_overlap": pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.output}")

    # Console summary
    print(f"\nBayesian bootstrap, {args.n_iters} iterations, seed={args.seed}\n")
    print(f"{'model':<35} {'n':<4} {'TS mean':<8} {'TS 95% CI':<22} {'Auto mean':<11} {'Auto 95% CI':<22}")
    for m in models:
        d = summary[m]
        ts_ci = f"[{d['trustscore_bayes_ci_low']:.1f}, {d['trustscore_bayes_ci_high']:.1f}]"
        auto_mean = f"{d.get('auto_rubric_mean', float('nan')):.3f}" if "auto_rubric_mean" in d else "-"
        auto_ci = f"[{d['auto_rubric_bayes_ci_low']:.3f}, {d['auto_rubric_bayes_ci_high']:.3f}]" if "auto_rubric_bayes_ci_low" in d else "-"
        print(f"{m:<35} {d['n']:<4} {d['trustscore_mean']:<8.1f} {ts_ci:<22} {auto_mean:<11} {auto_ci:<22}")
    print()
    n_ts_pairs = sum(1 for p in pairs if p["trustscore_non_overlap"])
    n_auto_pairs = sum(1 for p in pairs if p["auto_rubric_non_overlap"]) if any(p["auto_rubric_non_overlap"] is not None for p in pairs) else None
    print(f"Pairs with non-overlapping TrustScore 95% CIs: {n_ts_pairs}/{len(pairs)}")
    if n_auto_pairs is not None:
        print(f"Pairs with non-overlapping Auto-rubric 95% CIs: {n_auto_pairs}/{len(pairs)}")


if __name__ == "__main__":
    main()
