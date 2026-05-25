"""Re-run the ±30% weight-sensitivity Monte Carlo against current verification.py.

This regenerates Figure 6 of the paper using the same TrustScore implementation
snapshot that produces Table 1 (post-2026-05-24 re-run), restoring internal
consistency between the headline numbers and the robustness analysis.

Output:
- data/evaluation_results/trustscore_weight_sensitivity_2026_05.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.clinical.verification import _DEDUCTION_TABLE, validate_soap_note  # noqa: E402

OUTPUTS_PATH = ROOT / "docs" / "model_comparison_results.json"
TRANSCRIPTS_PATH = ROOT / "scripts" / "run_model_comparison.py"
OUT_PATH = ROOT / "data" / "evaluation_results" / "trustscore_weight_sensitivity_2026_05.json"

N_TRIALS = 1000
PERTURB_PCT = 0.30
SEED = 42


def _load_transcripts() -> dict[str, str]:
    spec: dict = {}
    src = TRANSCRIPTS_PATH.read_text()
    start = src.index("TRANSCRIPTS = {")
    end = src.index("\n}\n", start) + 3
    exec(src[start:end], spec)
    return spec["TRANSCRIPTS"]


def _collect_per_case_failure_profile() -> dict[str, list[dict]]:
    """For each model, return a list (per case) of {check_name: (n_issues, base_min, base_max)} for failing checks."""
    outputs = json.loads(OUTPUTS_PATH.read_text())
    transcripts = _load_transcripts()

    by_model: dict[str, list[dict]] = defaultdict(list)
    for entry in outputs:
        case = entry["case"]
        model_id = entry["model_id"]
        validation = validate_soap_note(
            entry["soap_note"], transcript=transcripts[case], use_entailment_gate=False
        )
        per_case: dict[str, tuple[int, int, int]] = {}
        for cr in validation.check_results:
            if cr.check_name == "entailment_gate":
                continue
            if not cr.passed:
                d_min, d_max = _DEDUCTION_TABLE.get(cr.check_name, (5, 15))
                per_case[cr.check_name] = (len(cr.issues), d_min, d_max)
        by_model[model_id].append({"case": case, "failures": per_case})
    return by_model


def _score_with_perturbed_weights(
    profile: list[dict], deduction_overrides: dict[str, tuple[int, int]]
) -> float:
    """Recompute mean TrustScore across cases using the perturbed deduction table."""
    scores = []
    for case in profile:
        deduction = 0
        for check_name, (n_issues, _, _) in case["failures"].items():
            d_min, d_max = deduction_overrides[check_name]
            d = min(d_max, d_min + (n_issues - 1) * 5)
            deduction += d
        score = max(0, min(100, 100 - deduction))
        scores.append(score)
    return float(np.mean(scores))


def main() -> None:
    rng = np.random.default_rng(SEED)
    profiles = _collect_per_case_failure_profile()

    nominal_means = {
        m: _score_with_perturbed_weights(profiles[m], _DEDUCTION_TABLE) for m in profiles
    }
    nominal_ranking = sorted(nominal_means, key=lambda m: nominal_means[m], reverse=True)

    n_models = len(profiles)
    n_preserved = 0
    kendall_taus = []
    per_model_scores: dict[str, list[float]] = {m: [] for m in profiles}

    base_table = {k: v for k, v in _DEDUCTION_TABLE.items() if k != "entailment_gate"}

    for trial in range(N_TRIALS):
        perturbed = {}
        for check_name, (d_min, d_max) in base_table.items():
            jitter_min = float(d_min) * (1 + rng.uniform(-PERTURB_PCT, PERTURB_PCT))
            jitter_max = float(d_max) * (1 + rng.uniform(-PERTURB_PCT, PERTURB_PCT))
            if jitter_min > jitter_max:
                jitter_min, jitter_max = jitter_max, jitter_min
            perturbed[check_name] = (jitter_min, jitter_max)

        trial_means = {
            m: _score_with_perturbed_weights(profiles[m], perturbed) for m in profiles
        }
        trial_ranking = sorted(trial_means, key=lambda m: trial_means[m], reverse=True)
        if trial_ranking == nominal_ranking:
            n_preserved += 1
        # Kendall tau: count concordant / total pairs
        nominal_rank = {m: i for i, m in enumerate(nominal_ranking)}
        trial_rank = {m: i for i, m in enumerate(trial_ranking)}
        concordant = 0
        discordant = 0
        for i, m1 in enumerate(nominal_ranking):
            for m2 in nominal_ranking[i + 1:]:
                if (nominal_rank[m1] < nominal_rank[m2]) == (trial_rank[m1] < trial_rank[m2]):
                    concordant += 1
                else:
                    discordant += 1
        total = concordant + discordant
        tau = (concordant - discordant) / total if total else 1.0
        kendall_taus.append(tau)

        for m in profiles:
            per_model_scores[m].append(trial_means[m])

    pct_preserved = n_preserved / N_TRIALS * 100

    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_trials": N_TRIALS,
        "perturb_pct": PERTURB_PCT,
        "seed": SEED,
        "implementation_snapshot": "current (verification.py 2026-05-24)",
        "nominal_means": nominal_means,
        "nominal_ranking": nominal_ranking,
        "n_ranking_preserved": n_preserved,
        "pct_ranking_preserved": pct_preserved,
        "kendall_tau": {
            "mean": float(np.mean(kendall_taus)),
            "min": float(np.min(kendall_taus)),
            "p5": float(np.percentile(kendall_taus, 5)),
            "p25": float(np.percentile(kendall_taus, 25)),
            "p50": float(np.percentile(kendall_taus, 50)),
            "p75": float(np.percentile(kendall_taus, 75)),
        },
        "per_model_distribution": {
            m: {
                "nominal": nominal_means[m],
                "mean": float(np.mean(per_model_scores[m])),
                "std": float(np.std(per_model_scores[m])),
                "p5": float(np.percentile(per_model_scores[m], 5)),
                "p25": float(np.percentile(per_model_scores[m], 25)),
                "p50": float(np.percentile(per_model_scores[m], 50)),
                "p75": float(np.percentile(per_model_scores[m], 75)),
                "p95": float(np.percentile(per_model_scores[m], 95)),
            }
            for m in profiles
        },
    }

    OUT_PATH.write_text(json.dumps(out, indent=2))
    print("=" * 60)
    print(f"Weight sensitivity re-run ({N_TRIALS} trials, ±{int(PERTURB_PCT*100)}%)")
    print("=" * 60)
    print(f"\nNominal ranking: {' > '.join(nominal_ranking)}")
    print(f"Nominal means:")
    for m in nominal_ranking:
        print(f"  {m:25s} {nominal_means[m]:6.2f}")
    print(f"\nRanking preserved: {n_preserved}/{N_TRIALS} ({pct_preserved:.1f}%)")
    print(f"Kendall τ mean: {np.mean(kendall_taus):.4f}  min: {np.min(kendall_taus):.4f}")
    print(f"\nWritten: {OUT_PATH}")


if __name__ == "__main__":
    main()
