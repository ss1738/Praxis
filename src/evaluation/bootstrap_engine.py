"""Bootstrap 95% CI engine for TrustScore (and any other per-case scalar).

Reads canonical per-case score arrays from existing evaluation JSONs and produces
percentile bootstrap CIs around the mean of each model's distribution. Fixed
seed=42, 1,000 iterations (configurable). Output: a single JSON ledger plus a
clean LaTeX table (numerical cells only - captions are author's hand).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class BootstrapResult:
    model: str
    n: int
    mean: float
    median: float
    std: float
    ci_lower: float
    ci_upper: float
    ci_method: str
    n_iter: int
    seed: int


def bootstrap_mean_ci(
    values: list[float],
    n_iter: int = 1000,
    ci_pct: float = 95.0,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    """Return (mean, ci_lower, ci_upper, std) via percentile bootstrap."""
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(n_iter, dtype=np.float64)
    n = len(arr)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        means[i] = arr[idx].mean()
    half = (100.0 - ci_pct) / 2.0
    return (
        float(arr.mean()),
        float(np.percentile(means, half)),
        float(np.percentile(means, 100.0 - half)),
        float(arr.std(ddof=1)) if n > 1 else 0.0,
    )


def trustscore_per_case(json_path: Path) -> dict[str, list[float]]:
    """Read trustscore_rerun_2026_05.json → {model: [score_per_case]}."""
    data = json.loads(json_path.read_text())
    out = {}
    for model, payload in data.get("summary_per_model", {}).items():
        per_case = payload.get("per_case", {})
        out[model] = [float(v) for v in per_case.values()]
    return out


def llm_judge_per_case(json_path: Path) -> dict[str, list[float]]:
    """Read llm_judge_*.json → {model: [overall_per_case]}."""
    data = json.loads(json_path.read_text())
    per_case = data.get("per_case", {})
    if not per_case and "per_record" in data:
        # The mistral/llama judge JSONs use per_record; bin by model
        per_model: dict[str, list[float]] = {}
        for rec in data["per_record"]:
            per_model.setdefault(rec["model"], []).append(rec["judge_score"]["overall"])
        return per_model
    by_model: dict[str, list[float]] = {}
    for case_name, scores in per_case.items():
        for model, val in scores.items():
            by_model.setdefault(model, []).append(float(val))
    return by_model


def render_latex_table(rows: list[BootstrapResult], caption_key: str) -> str:
    """Minimal numerical LaTeX table - no narrative captions added by the engine."""
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        "Model & $n$ & Mean & Std & 95\\% CI lower & 95\\% CI upper \\\\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r.model.replace('_', r'\\_')} & {r.n} & "
            f"{r.mean:.3f} & {r.std:.3f} & "
            f"{r.ci_lower:.3f} & {r.ci_upper:.3f} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def render_markdown_table(rows: list[BootstrapResult]) -> str:
    out = ["| Model | n | Mean | Std | 95% CI lower | 95% CI upper |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        out.append(
            f"| `{r.model}` | {r.n} | {r.mean:.3f} | {r.std:.3f} | "
            f"{r.ci_lower:.3f} | {r.ci_upper:.3f} |"
        )
    return "\n".join(out) + "\n"


def run_for_input(label: str, per_model: dict[str, list[float]],
                  n_iter: int, seed: int) -> list[BootstrapResult]:
    results = []
    for model, vals in per_model.items():
        if not vals:
            continue
        mean, ci_lo, ci_hi, std = bootstrap_mean_ci(vals, n_iter=n_iter, seed=seed)
        results.append(BootstrapResult(
            model=model, n=len(vals), mean=round(mean, 4),
            median=round(float(np.median(vals)), 4),
            std=round(std, 4), ci_lower=round(ci_lo, 4), ci_upper=round(ci_hi, 4),
            ci_method="percentile_bootstrap", n_iter=n_iter, seed=seed,
        ))
    return results


def main():
    REPO_ROOT = Path(__file__).resolve().parents[2]
    EVAL = REPO_ROOT / "data" / "evaluation_results"

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out",
                    default=str(EVAL / "bootstrap_ci_latest.json"))
    args = ap.parse_args()

    inputs = []
    if (EVAL / "trustscore_rerun_2026_05.json").exists():
        inputs.append(("trustscore_rerun_2026_05",
                       trustscore_per_case(EVAL / "trustscore_rerun_2026_05.json")))
    if (EVAL / "soap_significance.json").exists():
        sig = json.loads((EVAL / "soap_significance.json").read_text())
        by_model: dict[str, list[float]] = {}
        for case_name, scores in sig.get("per_case", {}).items():
            for m, v in scores.items():
                by_model.setdefault(m, []).append(float(v))
        inputs.append(("rule_based_scorer", by_model))
    for fname in ("llm_judge_mistral_latest_20260522_115849.json",
                  "llm_judge_llama3.1_8b_20260522_123941.json"):
        path = EVAL / fname
        if path.exists():
            inputs.append((fname.replace(".json", ""), llm_judge_per_case(path)))

    ledger = {
        "n_iter": args.n_iter,
        "seed": args.seed,
        "ci_method": "percentile_bootstrap",
        "ci_pct": 95.0,
        "results": {},
        "tables": {},
    }
    for label, per_model in inputs:
        rows = run_for_input(label, per_model, args.n_iter, args.seed)
        ledger["results"][label] = [r.__dict__ for r in rows]
        ledger["tables"][label] = {
            "markdown": render_markdown_table(rows),
            "latex": render_latex_table(rows, caption_key=label),
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(ledger, f, indent=2)

    # Also emit the LaTeX as a standalone file for direct \input usage
    latex_path = Path(args.out).with_suffix(".tex")
    latex_blocks = []
    for label, payload in ledger["tables"].items():
        latex_blocks.append(f"% --- {label} ---")
        latex_blocks.append(payload["latex"])
    latex_path.write_text("\n".join(latex_blocks))

    md_path = Path(args.out).with_suffix(".md")
    md_blocks = []
    for label, payload in ledger["tables"].items():
        md_blocks.append(f"### {label}\n")
        md_blocks.append(payload["markdown"])
    md_path.write_text("\n".join(md_blocks))

    print(f"Wrote {args.out}")
    print(f"Wrote {latex_path}")
    print(f"Wrote {md_path}")
    for label, payload in ledger["results"].items():
        print(f"\n[{label}]  n_models={len(payload)}")
        for r in payload:
            print(f"  {r['model']:<30}  mean={r['mean']:.3f}  95% CI [{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]")


if __name__ == "__main__":
    main()
