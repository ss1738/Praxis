#!/usr/bin/env python3
"""Score existing SOAP-generation outputs with TrustScore (the deterministic rubric).

Input: a model_comparison_results.json file (list of records each containing
       'case', 'model_id', 'soap_note', plus auto-rubric scores from
       run_model_comparison.py).

Output: same records augmented with TrustScore fields:
        - trustscore: integer 0-100 aggregate score
        - trustscore_verdict: 'approved' | 'needs_review' | 'rejected'
        - trustscore_checks: dict of per-check {passed, deduction, details}

Usage:
    python3 scripts/score_soap_outputs.py \
        --input docs/model_comparison_results.json \
        --output data/evaluation_results/saturation_8model_scored.json

The transcript for each case is loaded from run_model_comparison.py via TRANSCRIPTS
import; this script does not duplicate the transcript content.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.clinical.verification import validate_soap_note


def _load_transcripts() -> dict[str, str]:
    """Pull TRANSCRIPTS dict from run_model_comparison.py without re-running it."""
    try:
        from scripts.run_model_comparison import TRANSCRIPTS  # type: ignore
        return dict(TRANSCRIPTS)
    except Exception as exc:
        print(f"WARN: could not import TRANSCRIPTS from run_model_comparison: {exc}", file=sys.stderr)
        return {}


def _normalise_case_name(case: str) -> str:
    """Map case names from comparison records to TRANSCRIPTS keys (best effort)."""
    return case.strip()


def score_record(record: dict, transcripts: dict[str, str], use_entailment: bool = False) -> dict:
    """Augment one comparison record with TrustScore fields."""
    soap_text = record.get("soap_note", "")
    case_name = _normalise_case_name(record.get("case", ""))
    transcript = transcripts.get(case_name, "")

    result = validate_soap_note(
        note_text=soap_text,
        transcript=transcript,
        use_entailment_gate=use_entailment,
    )

    out = dict(record)  # don't mutate input
    out["trustscore"] = int(result.trust_score)
    out["trustscore_verdict"] = result.verdict
    out["trustscore_checks"] = {}
    for c in result.check_results:
        name = getattr(c, "check_name", getattr(c, "name", "unknown"))
        out["trustscore_checks"][name] = {
            "passed": getattr(c, "passed", None),
            "deduction": getattr(c, "deduction", None),
            "details": getattr(c, "details", getattr(c, "message", None)),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="Path to model_comparison_results.json")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output JSON path")
    parser.add_argument("--use-entailment", action="store_true",
                        help="Enable the optional NLI entailment gate (slow, requires DeBERTa-v3 weights)")
    args = parser.parse_args()

    with args.input.open() as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise SystemExit(f"Expected list of records in {args.input}, got {type(records).__name__}")

    transcripts = _load_transcripts()
    print(f"Loaded {len(transcripts)} transcripts; will score {len(records)} records.")

    scored = []
    for i, rec in enumerate(records, 1):
        try:
            out = score_record(rec, transcripts, use_entailment=args.use_entailment)
        except Exception as exc:
            print(f"WARN: scoring failed for record {i} ({rec.get('case')} / {rec.get('model_id')}): {exc}",
                  file=sys.stderr)
            out = dict(rec)
            out["trustscore"] = None
            out["trustscore_verdict"] = "error"
            out["trustscore_error"] = str(exc)
        scored.append(out)
        if i % 8 == 0 or i == len(records):
            print(f"  scored {i}/{len(records)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(scored, f, indent=2)
    print(f"Wrote {args.output}")

    # Per-model summary
    by_model: dict[str, list[int]] = {}
    for rec in scored:
        if rec.get("trustscore") is None:
            continue
        by_model.setdefault(rec["model_id"], []).append(rec["trustscore"])
    print("\nPer-model mean TrustScore:")
    for model_id, scores in sorted(by_model.items(), key=lambda x: -sum(x[1]) / max(len(x[1]), 1)):
        mean = sum(scores) / len(scores)
        print(f"  {model_id}: {mean:.1f} (n={len(scores)})")


if __name__ == "__main__":
    main()
