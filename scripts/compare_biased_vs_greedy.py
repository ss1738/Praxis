"""Compare biased vs greedy CTC decoding on PriMock57.

Loads the two latest WER benchmark JSONs (praxis-conformer-biased and
praxis-conformer), pairs samples by audio_file, and reports:
    - Per-sample WER deltas
    - Wilcoxon signed-rank significance
    - Medical-term recall delta

Usage:
    python3 -m scripts.compare_biased_vs_greedy
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "data" / "evaluation_results"


def _latest(prefix: str) -> Path | None:
    """Return the most recent benchmark JSON for a given prefix."""
    files = sorted(RESULTS_DIR.glob(f"{prefix}_*.json"), reverse=True)
    return files[0] if files else None


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    biased_path = _latest("wer_praxis-conformer-biased")
    greedy_path = _latest("wer_praxis-conformer")

    if not biased_path or not greedy_path:
        print(f"Missing benchmark files (biased={biased_path}, greedy={greedy_path})")
        return

    print(f"Biased: {biased_path.name}")
    print(f"Greedy: {greedy_path.name}\n")

    biased = _load(biased_path)
    greedy = _load(greedy_path)

    # Pair by filename
    g_by_file = {s["audio_file"]: s["wer"] for s in greedy["samples"]}
    b_by_file = {s["audio_file"]: s["wer"] for s in biased["samples"]}
    common = sorted(set(g_by_file) & set(b_by_file))

    g_wers = np.array([g_by_file[f] for f in common])
    b_wers = np.array([b_by_file[f] for f in common])
    diff = b_wers - g_wers  # negative = biasing helped

    print(f"{'File':<55}{'Greedy':>10}{'Biased':>10}{'Δ':>10}")
    for f in common:
        d = b_by_file[f] - g_by_file[f]
        marker = " ↓" if d < -0.005 else (" ↑" if d > 0.005 else "")
        print(f"{f:<55}{g_by_file[f]*100:>9.2f}%{b_by_file[f]*100:>9.2f}%{d*100:>+9.2f}{marker}")

    print()
    print(f"Mean WER greedy:   {greedy['mean_wer']*100:.2f}%  (95% CI [{greedy['ci_lower']*100:.2f}, {greedy['ci_upper']*100:.2f}])")
    print(f"Mean WER biased:   {biased['mean_wer']*100:.2f}%  (95% CI [{biased['ci_lower']*100:.2f}, {biased['ci_upper']*100:.2f}])")
    print(f"Mean delta:        {diff.mean()*100:+.2f} pp")

    if not np.allclose(diff, 0):
        try:
            stat, p = wilcoxon(g_wers, b_wers, zero_method="wilcox", alternative="two-sided")
            print(f"Wilcoxon W={stat:.2f}  p={p:.4f}  significant_at_0.05={p < 0.05}")
        except ValueError as exc:
            print(f"Wilcoxon undefined: {exc}")

    # Medical-term recall delta
    g_mc = greedy.get("medical_term_recall")
    b_mc = biased.get("medical_term_recall")
    if g_mc and b_mc:
        print()
        print(f"Medical-term recall (greedy):  {g_mc['recovered']:>4}/{g_mc['ref_hits']:>4} = {g_mc['mc_recall']*100:.2f}%")
        print(f"Medical-term recall (biased):  {b_mc['recovered']:>4}/{b_mc['ref_hits']:>4} = {b_mc['mc_recall']*100:.2f}%")
        print(f"MC-recall delta:               {(b_mc['mc_recall'] - g_mc['mc_recall'])*100:+.2f} pp")

        # Save comparison
        out = {
            "biased_file": biased_path.name,
            "greedy_file": greedy_path.name,
            "n_samples": len(common),
            "mean_wer_greedy": greedy["mean_wer"],
            "mean_wer_biased": biased["mean_wer"],
            "mean_delta": float(diff.mean()),
            "wilcoxon_p": float(p) if 'p' in dir() else None,
            "mc_recall_greedy": g_mc["mc_recall"],
            "mc_recall_biased": b_mc["mc_recall"],
            "mc_recall_delta": b_mc["mc_recall"] - g_mc["mc_recall"],
        }
        out_path = RESULTS_DIR / "biased_vs_greedy.json"
        out_path.write_text(json.dumps(out, indent=2))
        print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
