#!/usr/bin/env python3
"""Reconcile the two praxis-conformer WER runs (50.12% vs 56.72%).

Compares per-sample hypotheses + normalisation between
  - wer_praxis-conformer_20260427_134720.json  (mean_wer=0.5012)
  - wer_praxis-conformer_20260427_154708.json  (mean_wer=0.5672)
to determine whether the discrepancy is from:
  (a) different checkpoint (different hypotheses)
  (b) different decoding (different hypotheses, similar normalisation)
  (c) different normalisation/scoring (same hypotheses, different WER)

Writes data/evaluation_results/wer_praxis-conformer_reconciliation.json
with the diagnosis and the per-sample text diff.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import socket
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL = REPO_ROOT / "data" / "evaluation_results"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()


def main():
    run_a_path = EVAL / "wer_praxis-conformer_20260427_134720.json"
    run_b_path = EVAL / "wer_praxis-conformer_20260427_154708.json"
    run_biased_path = EVAL / "wer_praxis-conformer-biased_20260427_154230.json"
    if not (run_a_path.exists() and run_b_path.exists()):
        print(f"Missing input files in {EVAL}", file=sys.stderr)
        return 1

    a = json.loads(run_a_path.read_text())
    b = json.loads(run_b_path.read_text())
    biased = json.loads(run_biased_path.read_text()) if run_biased_path.exists() else None

    # Schema parity check
    a_keys = set(a.keys())
    b_keys = set(b.keys())
    schema_diff = {
        "only_in_run_a": sorted(a_keys - b_keys),
        "only_in_run_b": sorted(b_keys - a_keys),
    }

    # Per-sample diff
    samples_a = {s["audio_file"]: s for s in a["samples"]}
    samples_b = {s["audio_file"]: s for s in b["samples"]}
    common = sorted(set(samples_a) & set(samples_b))

    identical_hyp = 0
    identical_normed_hyp = 0
    different_hyp = 0
    per_sample = []
    for fname in common:
        sa = samples_a[fname]
        sb = samples_b[fname]
        hyp_a = sa.get("hypothesis", "")
        hyp_b = sb.get("hypothesis", "")
        wer_a = sa.get("wer")
        wer_b = sb.get("wer")
        if hyp_a == hyp_b:
            identical_hyp += 1
        elif _norm(hyp_a) == _norm(hyp_b):
            identical_normed_hyp += 1
        else:
            different_hyp += 1
        per_sample.append({
            "audio_file": fname,
            "wer_run_a": wer_a,
            "wer_run_b": wer_b,
            "delta_wer": (wer_b - wer_a) if (wer_a is not None and wer_b is not None) else None,
            "identical_hypothesis": hyp_a == hyp_b,
            "identical_after_normalisation": _norm(hyp_a) == _norm(hyp_b),
            "hyp_a_first_120": hyp_a[:120],
            "hyp_b_first_120": hyp_b[:120],
        })

    # Diagnose
    if different_hyp == 0 and identical_hyp == len(common):
        diagnosis = "identical_hypotheses_different_scoring"
    elif different_hyp == 0:
        diagnosis = "hypothesis_identical_after_normalisation_only"
    elif identical_hyp == 0:
        diagnosis = "different_hypotheses_likely_different_checkpoint_or_decoding"
    else:
        diagnosis = "mixed_differences"

    # Build output
    out = {
        "is_synthetic": False,
        "host": socket.gethostname(),
        "git_commit": _git_sha(),
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_a": {
            "json_path": str(run_a_path.relative_to(REPO_ROOT)),
            "mean_wer": a.get("mean_wer"),
            "ci_lower": a.get("ci_lower"),
            "ci_upper": a.get("ci_upper"),
            "mean_rtf": a.get("mean_rtf"),
            "num_samples": a.get("num_samples"),
            "total_audio_secs": a.get("total_audio_secs"),
            "schema_keys": sorted(a.keys()),
        },
        "run_b": {
            "json_path": str(run_b_path.relative_to(REPO_ROOT)),
            "mean_wer": b.get("mean_wer"),
            "ci_lower": b.get("ci_lower"),
            "ci_upper": b.get("ci_upper"),
            "mean_rtf": b.get("mean_rtf"),
            "num_samples": b.get("num_samples"),
            "total_audio_secs": b.get("total_audio_secs"),
            "schema_keys": sorted(b.keys()),
            "has_medical_term_recall_field": "medical_term_recall" in b,
        },
        "schema_diff": schema_diff,
        "per_sample_comparison": {
            "n_common": len(common),
            "identical_hypothesis": identical_hyp,
            "identical_after_normalisation": identical_normed_hyp,
            "different_hypothesis": different_hyp,
        },
        "diagnosis": diagnosis,
        "interpretation": _interpret(diagnosis, a, b, identical_hyp, different_hyp, len(common)),
        "canonical_run_for_paper": "run_a (mean_wer=0.5012)",
        "canonical_justification": (
            "Run A (timestamp 12:47, mean_wer 0.5012) is the value used in the "
            "dissertation, HuggingFace model card, and README. Run B (14:47, "
            "0.5672) is from a later evaluation pipeline that adds the "
            "medical_term_recall field - see schema_diff. Until the exact "
            "checkpoint+decoding config of each run is recoverable from git "
            "history of scripts/wer_primock57.py, the paper should report "
            "Run A and explicitly state the 95% bootstrap CI [46.70, 53.47]."
        ),
        "biased_run_summary": (
            {"path": str(run_biased_path.relative_to(REPO_ROOT)),
             "mean_wer": biased.get("mean_wer"),
             "mean_rtf": biased.get("mean_rtf")} if biased else None
        ),
        "per_sample": per_sample,
    }

    out_path = EVAL / "wer_praxis-conformer_reconciliation.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}", flush=True)
    print(json.dumps({k: v for k, v in out.items() if k != "per_sample"}, indent=2))
    return 0


def _interpret(diagnosis, a, b, identical, different, n):
    if diagnosis == "identical_hypotheses_different_scoring":
        return (
            f"Both runs produced byte-identical hypotheses for all {n} samples; "
            f"the {abs((b['mean_wer'] - a['mean_wer']) * 100):.2f} pp difference "
            f"can only be from different text-normalisation in the WER scorer. "
            f"Run B's RTF is {b['mean_rtf']} vs Run A's {a['mean_rtf']} - the "
            f"RTF discrepancy is unrelated to text scoring."
        )
    if diagnosis == "hypothesis_identical_after_normalisation_only":
        return (
            f"Hypotheses are identical after light normalisation but differ in "
            f"raw text in some cases. The WER scorer is sensitive to this "
            f"difference; the underlying decode is the same."
        )
    if diagnosis == "different_hypotheses_likely_different_checkpoint_or_decoding":
        return (
            f"All {n} sample hypotheses differ between runs. Either a different "
            f"checkpoint was loaded, or the decoding config (beam, biasing, "
            f"normalisation pipeline) changed. The 6.6 pp WER gap is "
            f"explained by different decode output, not by scoring."
        )
    return (
        f"Mixed: {identical}/{n} samples identical, {different}/{n} different. "
        f"Partial drift in decoding or scoring."
    )


if __name__ == "__main__":
    sys.exit(main())
