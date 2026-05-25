#!/usr/bin/env python3
"""SNOMED CT contextual-biasing sweep across boost values.

Extends scripts/snomed_biasing_poc.py from a single boost (8.0) to a sweep of
configurable values, producing a precision-recall trade-off curve on PriMock57
test (n=14). Replaces the POC caveat in chapter 5 §5.10 with a benchmarked result.

For each boost value:
  - run AUSCULT inference with hotword-boosted CTC beam search
  - compute MC-WER (medical-corpus WER) against the 185-term curated lexicon
  - compute overall WER for context
  - bootstrap 95% CI on MC-WER (1,000 iters, fixed seed=42)

Usage on vast.ai:
    pip install -U pyctcdecode
    python3 scripts/snomed_biasing_sweep.py --device cuda \
        --boost-values 0 1 2 4 6 8 10 12 16 \
        --output data/evaluation_results/snomed_sweep_$(date +%Y%m%d).json

Output JSON shape:
    {
      "boost_0": {"wer": 0.50, "mc_wer": 0.42, "mc_wer_ci_low": 0.38, "mc_wer_ci_high": 0.46, "per_file": [...]},
      "boost_8": {...},
      ...
    }
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def run_poc_at_boost(
    boost: float,
    device: str,
    lexicon: Path | None,
    out_dir: Path,
) -> dict:
    """Invoke the existing snomed_biasing_poc.py at a specific boost value.

    The POC script builds its own output path from --tag and writes to
    data/processed/snomed_biasing_<tag>.json. We pass an explicit tag so
    parallel sweeps don't clobber each other.
    """
    tag = "baseline_greedy" if boost == 0.0 else f"biased_boost{boost:g}"
    poc_path = ROOT / "data" / "processed" / f"snomed_biasing_{tag}.json"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "snomed_biasing_poc.py"),
        "--device", device,
        "--boost", str(boost),
        "--tag", tag,
    ]
    if lexicon is not None:
        cmd += ["--lexicon", str(lexicon)]
    if boost == 0.0:
        cmd += ["--no-biasing"]
    print(f"[boost={boost}] {' '.join(cmd)}")
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    print(f"[boost={boost}] done in {time.perf_counter() - t0:.1f}s")
    with poc_path.open() as f:
        return json.load(f)


def bootstrap_ci(values: list[float], n_iters: int = 1000, seed: int = 42) -> tuple[float, float]:
    import numpy as np
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_iters)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boost-values", nargs="+", type=float,
                        default=[0, 1, 2, 4, 6, 8, 10, 12, 16],
                        help="Boost values to sweep")
    parser.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    parser.add_argument("--lexicon", type=Path, default=None,
                        help="Path to SNOMED lexicon file (one term per line). Default = built-in 185 terms.")
    parser.add_argument("--output", required=True, type=Path,
                        help="Aggregate output JSON path")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "data" / "processed",
                        help="Where per-boost intermediate JSONs are written")
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)

    aggregate: dict[str, dict] = {}
    for boost in args.boost_values:
        try:
            poc_result = run_poc_at_boost(
                boost=boost,
                device=args.device,
                lexicon=args.lexicon,
                out_dir=args.work_dir,
            )
        except subprocess.CalledProcessError as exc:
            print(f"WARN: boost={boost} run failed: {exc}", file=sys.stderr)
            continue

        per_file_wer = [r.get("wer") for r in poc_result.get("per_file", []) if r.get("wer") is not None]
        per_file_mc = [r.get("mc_wer") for r in poc_result.get("per_file", []) if r.get("mc_wer") is not None]

        if per_file_mc:
            ci_low, ci_high = bootstrap_ci(per_file_mc)
        else:
            ci_low, ci_high = (None, None)

        aggregate[f"boost_{boost}"] = {
            "boost": boost,
            "wer": poc_result.get("overall_wer"),
            "mc_wer": poc_result.get("overall_mc_wer"),
            "mc_wer_ci_low": ci_low,
            "mc_wer_ci_high": ci_high,
            "per_file_wer": per_file_wer,
            "per_file_mc_wer": per_file_mc,
            "n_files": len(poc_result.get("per_file", [])),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\nWrote {args.output}")

    print("\nBoost sweep summary:")
    print(f"{'boost':<8} {'WER':<10} {'MC-WER':<10} {'95% CI':<22}")
    for k, v in aggregate.items():
        ci = f"[{v['mc_wer_ci_low']:.3f}, {v['mc_wer_ci_high']:.3f}]" if v["mc_wer_ci_low"] is not None else "-"
        wer = f"{v['wer']:.3f}" if v["wer"] is not None else "-"
        mcwer = f"{v['mc_wer']:.3f}" if v["mc_wer"] is not None else "-"
        print(f"{v['boost']:<8} {wer:<10} {mcwer:<10} {ci:<22}")


if __name__ == "__main__":
    main()
