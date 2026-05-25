#!/usr/bin/env python3
"""Judge-to-heuristic meta-alignment: are LLM judges ranking the same way TrustScore does?

Compares per-case rankings produced by:
  - LLM judges    (Mistral-7B, Llama 3.1 8B)
  - TrustScore    (6-check deterministic rubric)
  - rule_based_scorer  (from soap_significance.json)

Reports:
  1. Per-case Kendall τ between each judge and TrustScore (over the 4 models).
  2. Macro Kendall τ across all 8 cases.
  3. Cohen κ on verdict agreement (3-state: approved / needs_review / rejected),
     using ≥0.7 / 0.4-0.69 / <0.4 thresholds for the LLM-judge continuous scores.
  4. Per-criterion saturation: stddev of judge means across the 4 models,
     for each of the six judge dimensions. Low stddev = saturation.
  5. Per-check stratification: TrustScore's per-check pass rate variance.

Output: data/evaluation_results/judge_meta_alignment.json (+ .tex + .md).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

EVAL = REPO_ROOT / "data" / "evaluation_results"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _kendall_tau(x: list[float], y: list[float]) -> float:
    """Kendall τ via pairwise concordant/discordant counting (no ties)."""
    n = len(x)
    if n < 2:
        return float("nan")
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx * dy > 0:
                concordant += 1
            elif dx * dy < 0:
                discordant += 1
    denom = n * (n - 1) / 2
    return (concordant - discordant) / denom if denom else float("nan")


def _cohen_kappa(rater1: list[str], rater2: list[str]) -> dict:
    """Cohen κ for two rater categorical sequences."""
    assert len(rater1) == len(rater2)
    categories = sorted(set(rater1) | set(rater2))
    n = len(rater1)
    if n == 0:
        return {"kappa": float("nan"), "observed_agreement": 0.0, "expected_agreement": 0.0}
    po = sum(1 for a, b in zip(rater1, rater2) if a == b) / n
    pe = sum(
        (rater1.count(c) / n) * (rater2.count(c) / n)
        for c in categories
    )
    kappa = (po - pe) / (1 - pe) if pe != 1.0 else float("nan")
    return {
        "kappa": round(kappa, 4),
        "observed_agreement": round(po, 4),
        "expected_agreement": round(pe, 4),
        "categories": categories,
    }


def _verdict_from_judge(score: float) -> str:
    if score >= 0.7:
        return "approved"
    if score >= 0.4:
        return "needs_review"
    return "rejected"


def _verdict_from_trustscore(score: float) -> str:
    if score >= 70:
        return "approved"
    if score >= 40:
        return "needs_review"
    return "rejected"


def main():
    # Inputs
    ts_path = EVAL / "trustscore_rerun_2026_05.json"
    mistral_path = EVAL / "llm_judge_mistral_latest_20260522_115849.json"
    llama_path = EVAL / "llm_judge_llama3.1_8b_20260522_123941.json"
    rule_path = EVAL / "soap_significance.json"

    ts = json.loads(ts_path.read_text())
    mistral = json.loads(mistral_path.read_text())
    llama = json.loads(llama_path.read_text())
    rule = json.loads(rule_path.read_text())

    # Per-case score arrays {case_name: {model: score}}
    def _index_judge(judge_doc):
        out: dict[str, dict[str, float]] = {}
        for rec in judge_doc["per_record"]:
            out.setdefault(rec["case"], {})[rec["model"]] = rec["judge_score"]["overall"]
        return out

    mistral_by_case = _index_judge(mistral)
    llama_by_case = _index_judge(llama)
    rule_by_case = rule["per_case"]  # {case: {model: score}}

    trust_by_case: dict[str, dict[str, float]] = {}
    for model, payload in ts["summary_per_model"].items():
        for case_name, score in payload["per_case"].items():
            trust_by_case.setdefault(case_name, {})[model] = float(score)

    models = ["praxis-soap-v1", "med42", "mistral", "meditron:7b"]
    cases = sorted(trust_by_case.keys())
    assert len(cases) == 8, f"expected 8 cases, got {len(cases)}"

    # ── Per-case Kendall τ ──
    per_case_tau = []
    for c in cases:
        ts_vals = [trust_by_case[c][m] for m in models]
        mist_vals = [mistral_by_case.get(c, {}).get(m, 0.0) for m in models]
        llam_vals = [llama_by_case.get(c, {}).get(m, 0.0) for m in models]
        rule_vals = [rule_by_case.get(c, {}).get(m, 0.0) for m in models]
        per_case_tau.append({
            "case": c,
            "trustscore_vector": ts_vals,
            "mistral_judge_vector": mist_vals,
            "llama_judge_vector": llam_vals,
            "rule_based_vector": rule_vals,
            "tau_mistral_vs_trustscore": round(_kendall_tau(mist_vals, ts_vals), 4),
            "tau_llama_vs_trustscore":   round(_kendall_tau(llam_vals, ts_vals), 4),
            "tau_rule_vs_trustscore":    round(_kendall_tau(rule_vals, ts_vals), 4),
            "tau_mistral_vs_llama":      round(_kendall_tau(mist_vals, llam_vals), 4),
        })

    def _mean_tau(field):
        vals = [r[field] for r in per_case_tau if not np.isnan(r[field])]
        return round(float(np.mean(vals)), 4) if vals else float("nan")

    macro_tau = {
        "mistral_vs_trustscore": _mean_tau("tau_mistral_vs_trustscore"),
        "llama_vs_trustscore":   _mean_tau("tau_llama_vs_trustscore"),
        "rule_vs_trustscore":    _mean_tau("tau_rule_vs_trustscore"),
        "mistral_vs_llama":      _mean_tau("tau_mistral_vs_llama"),
    }

    # ── Verdict-state Cohen κ ──
    # For each of the 32 (case, model) records, compute the verdict from each scorer.
    ts_verdicts: list[str] = []
    mistral_verdicts: list[str] = []
    llama_verdicts: list[str] = []
    rule_verdicts: list[str] = []
    record_index: list[tuple[str, str]] = []
    for c in cases:
        for m in models:
            ts_verdicts.append(_verdict_from_trustscore(trust_by_case[c][m]))
            mistral_verdicts.append(_verdict_from_judge(mistral_by_case.get(c, {}).get(m, 0.0)))
            llama_verdicts.append(_verdict_from_judge(llama_by_case.get(c, {}).get(m, 0.0)))
            rule_verdicts.append(_verdict_from_judge(rule_by_case.get(c, {}).get(m, 0.0)))
            record_index.append((c, m))

    kappa = {
        "mistral_vs_trustscore": _cohen_kappa(mistral_verdicts, ts_verdicts),
        "llama_vs_trustscore":   _cohen_kappa(llama_verdicts, ts_verdicts),
        "rule_vs_trustscore":    _cohen_kappa(rule_verdicts, ts_verdicts),
        "mistral_vs_llama":      _cohen_kappa(mistral_verdicts, llama_verdicts),
    }

    # ── Per-criterion saturation analysis ──
    # For each judge: stddev of per-model mean across the 6 criteria.
    def _per_criterion_means(judge_doc):
        # {criterion: {model: list[per-case]}}
        crits = ("structure", "content", "snomed", "safety_netting", "uk_english", "budget")
        accum: dict[str, dict[str, list[float]]] = {c: {} for c in crits}
        for rec in judge_doc["per_record"]:
            m = rec["model"]
            for c in crits:
                accum[c].setdefault(m, []).append(rec["judge_score"][c])
        # Average per (criterion, model), then compute model-spread per criterion
        spread = {}
        for c in crits:
            model_means = [float(np.mean(accum[c][m])) for m in models if m in accum[c]]
            spread[c] = {
                "model_means": {m: round(float(np.mean(accum[c][m])), 4)
                                for m in models if m in accum[c]},
                "mean": round(float(np.mean(model_means)), 4),
                "std": round(float(np.std(model_means, ddof=1)), 4) if len(model_means) > 1 else 0.0,
                "max_minus_min": round(float(np.max(model_means) - np.min(model_means)), 4)
                                  if model_means else 0.0,
            }
        return spread

    mistral_crit = _per_criterion_means(mistral)
    llama_crit = _per_criterion_means(llama)

    # TrustScore per-check pass rate stratification
    per_check = ts.get("per_check_pass_rate", {})
    trust_check_stratification = {}
    if per_check:
        # Each check has 4 model values (pass rate ∈ [0,1])
        any_model = next(iter(per_check.values()))
        check_names = list(any_model.keys())
        for check in check_names:
            vals = [per_check[m][check] for m in models if m in per_check]
            trust_check_stratification[check] = {
                "model_pass_rates": {m: per_check[m][check] for m in models if m in per_check},
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else 0.0,
                "max_minus_min": round(float(np.max(vals) - np.min(vals)), 4) if vals else 0.0,
            }

    out = {
        "is_synthetic": False,
        "host": socket.gethostname(),
        "git_commit": _git_sha(),
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "inputs": {
            "trustscore": str(ts_path.relative_to(REPO_ROOT)),
            "mistral_judge": str(mistral_path.relative_to(REPO_ROOT)),
            "llama_judge": str(llama_path.relative_to(REPO_ROOT)),
            "rule_based_scorer": str(rule_path.relative_to(REPO_ROOT)),
        },
        "ranking_correlation": {
            "macro_kendall_tau": macro_tau,
            "per_case": per_case_tau,
        },
        "verdict_agreement_cohen_kappa": kappa,
        "criterion_saturation": {
            "mistral_judge": mistral_crit,
            "llama_judge": llama_crit,
        },
        "trustscore_check_stratification": trust_check_stratification,
        "interpretation_notes": {
            "verdict_thresholds_llm": "≥0.7 → approved, 0.4 to 0.69 → needs_review, <0.4 → rejected",
            "verdict_thresholds_trustscore": "≥70 → approved, 40 to 69 → needs_review, <40 → rejected",
            "saturated_criterion": "std < 0.05 (across the 4 models)",
        },
    }

    out_path = EVAL / "judge_meta_alignment.json"
    out_path.write_text(json.dumps(out, indent=2))

    # ── LaTeX tables (numerical only) ──
    latex_lines = [
        r"% Macro Kendall τ - per-pair, mean over 8 cases",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Comparison & Macro Kendall $\tau$ \\",
        r"\midrule",
        f"Mistral judge vs TrustScore & {macro_tau['mistral_vs_trustscore']:.3f} \\\\",
        f"Llama judge vs TrustScore   & {macro_tau['llama_vs_trustscore']:.3f} \\\\",
        f"Rule-based vs TrustScore    & {macro_tau['rule_vs_trustscore']:.3f} \\\\",
        f"Mistral vs Llama            & {macro_tau['mistral_vs_llama']:.3f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"% Cohen κ - verdict agreement on 32 (case, model) pairs",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Comparison & Cohen $\kappa$ & Observed agreement & Expected agreement \\",
        r"\midrule",
    ]
    for label in ("mistral_vs_trustscore", "llama_vs_trustscore",
                  "rule_vs_trustscore", "mistral_vs_llama"):
        k = kappa[label]
        latex_lines.append(
            f"{label.replace('_', r'\_')} & {k['kappa']:.3f} & "
            f"{k['observed_agreement']:.3f} & {k['expected_agreement']:.3f} \\\\"
        )
    latex_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"% Per-criterion saturation (Llama judge) - std of model means across 4 models",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Criterion & Mean across models & Std across models & Max -- Min \\",
        r"\midrule",
    ]
    for crit, payload in llama_crit.items():
        latex_lines.append(
            f"{crit.replace('_', r'\_')} & {payload['mean']:.3f} & "
            f"{payload['std']:.3f} & {payload['max_minus_min']:.3f} \\\\"
        )
    latex_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"% TrustScore per-check stratification (pass rate variance across 4 models)",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Check & Mean pass rate & Std across models & Max -- Min \\",
        r"\midrule",
    ]
    for check, payload in trust_check_stratification.items():
        latex_lines.append(
            f"{check.replace('_', r'\_')} & {payload['mean']:.3f} & "
            f"{payload['std']:.3f} & {payload['max_minus_min']:.3f} \\\\"
        )
    latex_lines += [r"\bottomrule", r"\end{tabular}"]
    out_path.with_suffix(".tex").write_text("\n".join(latex_lines) + "\n")

    # Markdown summary
    md = [
        "## Macro Kendall τ\n",
        "| Comparison | τ |", "|---|---|",
        f"| Mistral judge vs TrustScore | {macro_tau['mistral_vs_trustscore']:.3f} |",
        f"| Llama judge vs TrustScore | {macro_tau['llama_vs_trustscore']:.3f} |",
        f"| Rule-based vs TrustScore | {macro_tau['rule_vs_trustscore']:.3f} |",
        f"| Mistral vs Llama | {macro_tau['mistral_vs_llama']:.3f} |",
        "\n## Verdict Cohen κ (3-state)\n",
        "| Comparison | κ | Observed | Expected |", "|---|---|---|---|",
    ]
    for label in ("mistral_vs_trustscore", "llama_vs_trustscore",
                  "rule_vs_trustscore", "mistral_vs_llama"):
        k = kappa[label]
        md.append(f"| {label} | {k['kappa']:.3f} | {k['observed_agreement']:.3f} | {k['expected_agreement']:.3f} |")
    md += [
        "\n## Per-criterion saturation (Llama judge)\n",
        "| Criterion | Mean | Std | Max−Min |", "|---|---|---|---|",
    ]
    for crit, payload in llama_crit.items():
        md.append(f"| {crit} | {payload['mean']:.3f} | {payload['std']:.3f} | {payload['max_minus_min']:.3f} |")
    md += [
        "\n## TrustScore per-check stratification\n",
        "| Check | Mean | Std | Max−Min |", "|---|---|---|---|",
    ]
    for check, payload in trust_check_stratification.items():
        md.append(f"| {check} | {payload['mean']:.3f} | {payload['std']:.3f} | {payload['max_minus_min']:.3f} |")
    out_path.with_suffix(".md").write_text("\n".join(md) + "\n")

    print(f"Written: {out_path}")
    print(f"Written: {out_path.with_suffix('.tex')}")
    print(f"Written: {out_path.with_suffix('.md')}")
    print("\n== Macro Kendall τ ==")
    for k, v in macro_tau.items():
        print(f"  {k:<30}  τ = {v:.3f}")
    print("\n== Cohen κ (verdict, 3-state) ==")
    for k, v in kappa.items():
        print(f"  {k:<30}  κ = {v['kappa']:.3f}  (observed {v['observed_agreement']:.2f}, expected {v['expected_agreement']:.2f})")
    print("\n== Llama judge saturation (std across 4 models) ==")
    for c, p in sorted(llama_crit.items(), key=lambda kv: kv[1]["std"]):
        print(f"  {c:<15}  std={p['std']:.3f}  spread={p['max_minus_min']:.3f}")
    print("\n== TrustScore per-check stratification ==")
    for c, p in sorted(trust_check_stratification.items(), key=lambda kv: kv[1]["std"], reverse=True):
        print(f"  {c:<22}  std={p['std']:.3f}  spread={p['max_minus_min']:.3f}")


if __name__ == "__main__":
    main()
