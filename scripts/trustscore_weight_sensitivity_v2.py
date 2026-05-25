#!/usr/bin/env python3
"""TrustScore weight-sensitivity analysis.

Question: how much does the model ranking depend on the specific deduction weights
in the _DEDUCTION_TABLE? If a reviewer asks "you chose those weights, what if they
were different?", we need a sensitivity result.

Method:
    1. Recompute per-check per-case pass/fail across all 32 SOAP outputs.
    2. For N=1000 random perturbations of the per-check deduction values
       (uniform within ±X% of the nominal mid-point), recompute the aggregate
       TrustScore per model and re-rank.
    3. Report:
       - Fraction of perturbations that preserve the exact 4-way model ranking
       - Kendall's tau between perturbed ranking and nominal ranking
       - Distribution of mean TrustScore per model under perturbation

Output: data/evaluation_results/trustscore_weight_sensitivity.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Suppress noisy logger output
from loguru import logger
logger.remove()

from src.clinical.verification import validate_soap_note
from scripts.run_model_comparison import TRANSCRIPTS

# Nominal deduction table (from src/clinical/verification.py)
NOMINAL_DEDUCTIONS = {
    "clinical_safety": (15, 35),
    "transcript_fidelity": (10, 30),
    "snomed_validation": (5, 15),
    "nice_guidelines": (5, 20),
    "completeness": (5, 15),
    "drug_interactions": (5, 20),
}


def compute_trust_score_from_checks(check_pass: dict[str, bool], deductions: dict[str, tuple[float, float]]) -> float:
    """Aggregate a single note's score from its per-check pass/fail."""
    score = 100.0
    for check_name, passed in check_pass.items():
        if passed:
            continue
        d_min, d_max = deductions.get(check_name, (5, 15))
        # Use mid-point as the deterministic deduction (matches typical TrustScore behaviour)
        score -= (d_min + d_max) / 2.0
    return max(0.0, min(100.0, score))


def main() -> int:
    records = json.load(open(ROOT / "docs" / "model_comparison_results.json"))
    print(f"Loaded {len(records)} records.")

    # Step 1: cache per-record check outcomes
    per_record_checks: list[dict] = []
    print("Re-scoring each record (this caches per-check pass/fail, runs once)...")
    for r in records:
        result = validate_soap_note(r["soap_note"], TRANSCRIPTS[r["case"]])
        check_pass = {cr.check_name: cr.passed for cr in result.check_results}
        per_record_checks.append({
            "case": r["case"],
            "model": r["model_id"],
            "check_pass": check_pass,
            "nominal_trust_score": result.trust_score,
        })

    # Step 2: nominal model means
    nominal_per_model: dict[str, list[float]] = defaultdict(list)
    for rec in per_record_checks:
        ts = compute_trust_score_from_checks(rec["check_pass"], NOMINAL_DEDUCTIONS)
        nominal_per_model[rec["model"]].append(ts)
    nominal_means = {m: float(np.mean(v)) for m, v in nominal_per_model.items()}
    nominal_ranking = sorted(nominal_means.keys(), key=lambda m: -nominal_means[m])
    print(f"\nNominal model ranking: {nominal_ranking}")
    print(f"Nominal means: {nominal_means}")

    # Step 3: Monte Carlo perturbation
    N_TRIALS = 1000
    PERTURB_PCT = 0.30  # ±30%
    rng = np.random.default_rng(seed=42)

    perturbed_rankings = []
    perturbed_kendall_tau = []
    perturbed_means_dist: dict[str, list[float]] = defaultdict(list)

    nominal_rank_vector = {m: i for i, m in enumerate(nominal_ranking)}

    for trial in range(N_TRIALS):
        # Perturb each check's (d_min, d_max) by a uniform multiplicative factor in [1-X, 1+X]
        perturbed_deductions = {}
        for check_name, (d_min, d_max) in NOMINAL_DEDUCTIONS.items():
            factor = rng.uniform(1.0 - PERTURB_PCT, 1.0 + PERTURB_PCT)
            perturbed_deductions[check_name] = (d_min * factor, d_max * factor)

        per_model_perturbed: dict[str, list[float]] = defaultdict(list)
        for rec in per_record_checks:
            ts = compute_trust_score_from_checks(rec["check_pass"], perturbed_deductions)
            per_model_perturbed[rec["model"]].append(ts)
        perturbed_means = {m: float(np.mean(v)) for m, v in per_model_perturbed.items()}

        for m, v in perturbed_means.items():
            perturbed_means_dist[m].append(v)

        # Re-rank under perturbation
        perturbed_ranking = sorted(perturbed_means.keys(), key=lambda m: -perturbed_means[m])
        perturbed_rankings.append(perturbed_ranking)

        # Kendall tau between nominal and perturbed
        perturbed_rank_vector = {m: i for i, m in enumerate(perturbed_ranking)}
        ks = [nominal_rank_vector[m] for m in nominal_ranking]
        ps = [perturbed_rank_vector[m] for m in nominal_ranking]
        tau, _ = kendalltau(ks, ps)
        perturbed_kendall_tau.append(float(tau))

    # Step 4: aggregate
    n_preserved = sum(1 for r in perturbed_rankings if r == nominal_ranking)
    pct_preserved = 100.0 * n_preserved / N_TRIALS

    tau_mean = float(np.mean(perturbed_kendall_tau))
    tau_min = float(np.min(perturbed_kendall_tau))
    tau_p5 = float(np.percentile(perturbed_kendall_tau, 5))

    print(f"\n=== Weight-sensitivity result ({N_TRIALS} trials, ±{int(PERTURB_PCT*100)}%) ===")
    print(f"Exact 4-way ranking preserved: {n_preserved}/{N_TRIALS} = {pct_preserved:.1f}%")
    print(f"Kendall's tau (nominal vs perturbed): mean={tau_mean:.4f} min={tau_min:.4f} 5th-pct={tau_p5:.4f}")

    print("\nPer-model TrustScore distribution under weight perturbation:")
    print(f"  {'model':<22} {'nominal':<10} {'mean':<10} {'5th-pct':<10} {'95th-pct':<10}")
    for m in nominal_ranking:
        dist = perturbed_means_dist[m]
        print(f"  {m:<22} {nominal_means[m]:<10.2f} {np.mean(dist):<10.2f} {np.percentile(dist, 5):<10.2f} {np.percentile(dist, 95):<10.2f}")

    # Step 5: save
    out = {
        "n_trials": N_TRIALS,
        "perturb_pct": PERTURB_PCT,
        "n_records": len(per_record_checks),
        "nominal_ranking": nominal_ranking,
        "nominal_means": nominal_means,
        "pct_ranking_preserved": pct_preserved,
        "n_ranking_preserved": n_preserved,
        "kendall_tau": {
            "mean": tau_mean,
            "min": tau_min,
            "p5": tau_p5,
            "p25": float(np.percentile(perturbed_kendall_tau, 25)),
            "p50": float(np.percentile(perturbed_kendall_tau, 50)),
            "p75": float(np.percentile(perturbed_kendall_tau, 75)),
        },
        "per_model_distribution": {
            m: {
                "nominal": nominal_means[m],
                "mean": float(np.mean(perturbed_means_dist[m])),
                "std": float(np.std(perturbed_means_dist[m])),
                "p5": float(np.percentile(perturbed_means_dist[m], 5)),
                "p25": float(np.percentile(perturbed_means_dist[m], 25)),
                "p50": float(np.percentile(perturbed_means_dist[m], 50)),
                "p75": float(np.percentile(perturbed_means_dist[m], 75)),
                "p95": float(np.percentile(perturbed_means_dist[m], 95)),
            }
            for m in nominal_means.keys()
        },
    }
    out_file = ROOT / "data" / "evaluation_results" / "trustscore_weight_sensitivity.json"
    json.dump(out, open(out_file, "w"), indent=2)
    print(f"\nSaved: {out_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
