#!/usr/bin/env python3
"""TrustScore deduction-table weight-sensitivity analysis.

Perturbs the per-check deduction weights randomly within a defined band and
recomputes TrustScore for each perturbation. Reports:
  - Stability of the per-model ordering across perturbations (Kendall's tau)
  - Per-model TrustScore distribution under perturbation (mean, std)
  - Whether the qualitative finding (saturation in auto-rubric, stratification
    in TrustScore) is robust to weight choices

This addresses the "weights calibrated against author-curated examples"
limitation flagged in chapter 5 §5.10 and chapter 6 §6.4. Adds substantive
content to thesis Appendix B.

Usage:
    python3 scripts/trustscore_weight_sensitivity.py \
        --input data/evaluation_results/saturation_8model_scored.json \
        --output data/evaluation_results/weight_sensitivity.json \
        --n-perturbations 200 --perturbation-strength 0.30 --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.clinical.verification import (
    validate_soap_note,
    _DEDUCTION_TABLE,
)


def perturbed_deduction_table(rng: np.random.Generator, strength: float) -> dict:
    """Return a copy of _DEDUCTION_TABLE with each (primary, max) deduction
    multiplied by a uniform [(1-strength), (1+strength)] factor."""
    out = {}
    for check_name, (primary, maximum) in _DEDUCTION_TABLE.items():
        factor1 = float(rng.uniform(1 - strength, 1 + strength))
        factor2 = float(rng.uniform(1 - strength, 1 + strength))
        new_primary = max(0, int(round(primary * factor1)))
        new_max = max(new_primary, int(round(maximum * factor2)))
        out[check_name] = (new_primary, new_max)
    return out


def compute_perturbed_trustscore(record: dict, transcripts: dict[str, str], deduction_table: dict) -> int | None:
    """Re-score a single SOAP record with a perturbed deduction table."""
    soap = record.get("soap_note", "")
    case = record.get("case", "").strip()
    transcript = transcripts.get(case, "")
    try:
        # Monkey-patch the deduction table for this scoring pass.
        from src.clinical import verification as _v
        original = _v._DEDUCTION_TABLE
        _v._DEDUCTION_TABLE = deduction_table
        try:
            result = validate_soap_note(
                note_text=soap, transcript=transcript, use_entailment_gate=False
            )
        finally:
            _v._DEDUCTION_TABLE = original
        return int(result.trust_score)
    except Exception:
        return None


def kendall_tau(a: list, b: list) -> float:
    """Lightweight Kendall's tau (no scipy dependency)."""
    n = len(a)
    if n < 2:
        return 1.0
    concord = discord = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = a[i] - a[j]
            db = b[i] - b[j]
            sign = da * db
            if sign > 0: concord += 1
            elif sign < 0: discord += 1
    total = n * (n - 1) // 2
    return (concord - discord) / total if total else 1.0


def _load_transcripts() -> dict[str, str]:
    try:
        from scripts.run_model_comparison import TRANSCRIPTS  # type: ignore
        return dict(TRANSCRIPTS)
    except Exception:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="Scored SOAP outputs JSON")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n-perturbations", type=int, default=200)
    parser.add_argument("--perturbation-strength", type=float, default=0.30,
                        help="Uniform perturbation band: weights scaled by [1-s, 1+s]")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with args.input.open() as f:
        records = json.load(f)
    transcripts = _load_transcripts()

    rng = np.random.default_rng(args.seed)

    # Reference (unperturbed) ordering: per-model mean of original TrustScore in input
    ref_means: dict[str, float] = {}
    by_model: dict[str, list[dict]] = {}
    for rec in records:
        if rec.get("trustscore") is None:
            continue
        by_model.setdefault(rec["model_id"], []).append(rec)
    for m, recs in by_model.items():
        ref_means[m] = float(np.mean([r["trustscore"] for r in recs]))
    ref_order = sorted(ref_means.keys(), key=lambda m: -ref_means[m])
    print(f"Reference order: {ref_order}")

    per_perturbation_means: list[dict[str, float]] = []
    for p_idx in range(args.n_perturbations):
        table = perturbed_deduction_table(rng, args.perturbation_strength)
        per_model: dict[str, list[int]] = {}
        for m, recs in by_model.items():
            for rec in recs:
                ts = compute_perturbed_trustscore(rec, transcripts, table)
                if ts is not None:
                    per_model.setdefault(m, []).append(ts)
        means = {m: float(np.mean(scores)) for m, scores in per_model.items() if scores}
        per_perturbation_means.append(means)
        if (p_idx + 1) % 20 == 0:
            print(f"  perturbation {p_idx + 1}/{args.n_perturbations}")

    # Stability analysis
    perturbation_orders = [
        sorted(m.keys(), key=lambda k: -m[k]) for m in per_perturbation_means
    ]
    taus = []
    ref_indices = {m: i for i, m in enumerate(ref_order)}
    for order in perturbation_orders:
        common = [m for m in order if m in ref_indices]
        a = [ref_indices[m] for m in common]
        b = list(range(len(common)))
        if len(common) >= 2:
            taus.append(kendall_tau(a, b))

    # Per-model summary across perturbations
    per_model_summary: dict[str, dict] = {}
    for m in ref_means:
        scores = [pm.get(m) for pm in per_perturbation_means if m in pm]
        if scores:
            per_model_summary[m] = {
                "ref_mean": ref_means[m],
                "perturbed_mean": float(np.mean(scores)),
                "perturbed_std": float(np.std(scores)),
                "perturbed_min": float(np.min(scores)),
                "perturbed_max": float(np.max(scores)),
                "n_perturbations": len(scores),
            }

    out = {
        "config": {
            "n_perturbations": args.n_perturbations,
            "perturbation_strength": args.perturbation_strength,
            "seed": args.seed,
        },
        "reference_order": ref_order,
        "kendall_tau_mean": float(np.mean(taus)) if taus else 1.0,
        "kendall_tau_std": float(np.std(taus)) if taus else 0.0,
        "kendall_tau_min": float(np.min(taus)) if taus else 1.0,
        "fraction_perturbations_preserve_top": float(
            sum(1 for o in perturbation_orders if o[:1] == ref_order[:1]) / len(perturbation_orders)
        ),
        "fraction_perturbations_preserve_bottom": float(
            sum(1 for o in perturbation_orders if o[-1:] == ref_order[-1:]) / len(perturbation_orders)
        ),
        "per_model": per_model_summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.output}")
    print(f"\nKendall tau (ordering stability): mean {out['kendall_tau_mean']:.3f}, "
          f"min {out['kendall_tau_min']:.3f} across {len(taus)} perturbations")
    print(f"Fraction preserving top-ranked model: {out['fraction_perturbations_preserve_top']:.2f}")
    print(f"Fraction preserving bottom-ranked model: {out['fraction_perturbations_preserve_bottom']:.2f}")


if __name__ == "__main__":
    main()
