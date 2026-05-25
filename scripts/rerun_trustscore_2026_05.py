"""Re-run TrustScore with current verification.py implementation on the 32 SOAP outputs.

Fixes the Table-1-vs-Figure-6 inconsistency by regenerating Table 1 and the
per-check ablation from a single consistent snapshot.

Outputs:
- data/evaluation_results/trustscore_rerun_2026_05_<model>.json (per-model details)
- data/evaluation_results/trustscore_rerun_2026_05_summary.json (paper Table 1 source)
- data/evaluation_results/trustscore_per_check_ablation_2026_05.json (radar source)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.clinical.verification import validate_soap_note

OUTPUTS_PATH = ROOT / "docs" / "model_comparison_results.json"
TRANSCRIPTS_PATH = ROOT / "scripts" / "run_model_comparison.py"
OUT_DIR = ROOT / "data" / "evaluation_results"


def _load_transcripts() -> dict[str, str]:
    """Extract the TRANSCRIPTS dict from run_model_comparison.py."""
    spec_globals: dict = {}
    src = TRANSCRIPTS_PATH.read_text()
    start = src.index("TRANSCRIPTS = {")
    end = src.index("\n}\n", start) + 3
    exec(src[start:end], spec_globals)
    return spec_globals["TRANSCRIPTS"]


def _per_check_passed(result) -> dict[str, bool]:
    """Map check_name → passed for the 6 deterministic checks."""
    return {
        cr.check_name: cr.passed
        for cr in result.check_results
        if cr.check_name != "entailment_gate"
    }


def main() -> None:
    outputs = json.loads(OUTPUTS_PATH.read_text())
    transcripts = _load_transcripts()

    by_model: dict[str, list[dict]] = defaultdict(list)
    per_check_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    per_check_total: dict[str, int] = defaultdict(int)

    for entry in outputs:
        case = entry["case"]
        model_id = entry["model_id"]
        soap_note = entry["soap_note"]
        transcript = transcripts[case]

        validation = validate_soap_note(
            soap_note, transcript=transcript, use_entailment_gate=False
        )
        passed = _per_check_passed(validation)

        for check_name, ok in passed.items():
            if ok:
                per_check_counts[model_id][check_name] += 1
            per_check_total[model_id] += 1 if check_name == "transcript_fidelity" else 0

        by_model[model_id].append({
            "case": case,
            "trust_score": validation.trust_score,
            "verdict": validation.verdict,
            "checks_passed_n": validation.checks_passed,
            "passed": passed,
        })

    summary: dict[str, dict] = {}
    for model_id, records in by_model.items():
        scores = [r["trust_score"] for r in records]
        summary[model_id] = {
            "n_cases": len(scores),
            "mean_trust_score": sum(scores) / len(scores),
            "min": min(scores),
            "max": max(scores),
            "per_case": {r["case"]: r["trust_score"] for r in records},
            "per_case_verdict": {r["case"]: r["verdict"] for r in records},
        }

    n_per_model = len(by_model[next(iter(by_model))])
    per_check_pass_rate: dict[str, dict[str, float]] = {}
    for model_id in by_model:
        per_check_pass_rate[model_id] = {
            check: per_check_counts[model_id][check] / n_per_model
            for check in [
                "transcript_fidelity",
                "snomed_validation",
                "clinical_safety",
                "nice_guidelines",
                "completeness",
                "drug_interactions",
            ]
        }

    ranking = sorted(
        summary.keys(),
        key=lambda m: summary[m]["mean_trust_score"],
        reverse=True,
    )
    means = [summary[m]["mean_trust_score"] for m in ranking]

    timestamp = datetime.now(timezone.utc).isoformat()

    full_out = {
        "timestamp": timestamp,
        "source_outputs": str(OUTPUTS_PATH.relative_to(ROOT)),
        "implementation_snapshot": "current (verification.py with 6-check default, no NLI gate)",
        "summary_per_model": summary,
        "per_check_pass_rate": per_check_pass_rate,
        "ranking": ranking,
        "range": means[0] - means[-1],
    }

    (OUT_DIR / "trustscore_rerun_2026_05.json").write_text(
        json.dumps(full_out, indent=2)
    )
    (OUT_DIR / "trustscore_per_check_ablation_2026_05.json").write_text(
        json.dumps(per_check_pass_rate, indent=2)
    )

    print("=" * 60)
    print("TrustScore re-run - current implementation (no NLI gate)")
    print("=" * 60)
    print(f"\nRanking (high → low):")
    for m in ranking:
        s = summary[m]
        print(
            f"  {m:25s} mean={s['mean_trust_score']:6.2f}  "
            f"range=[{s['min']:.0f}, {s['max']:.0f}]"
        )
    print(f"\nHeadline range (top − bottom): {means[0] - means[-1]:.1f}")
    print(f"\nVerdicts crossed: ", end="")
    verdicts = {m: summary[m]["per_case_verdict"] for m in ranking}
    n_approved = {m: sum(1 for v in verdicts[m].values() if v == "approved") for m in ranking}
    n_review = {m: sum(1 for v in verdicts[m].values() if v == "needs_review") for m in ranking}
    n_reject = {m: sum(1 for v in verdicts[m].values() if v == "rejected") for m in ranking}
    print()
    for m in ranking:
        print(f"  {m:25s} approved={n_approved[m]} needs_review={n_review[m]} rejected={n_reject[m]}")

    print("\nPer-check pass rate (radar source):")
    print(f"{'Check':25s}", end="")
    for m in ranking:
        print(f"  {m[:12]:>12s}", end="")
    print()
    for check in [
        "transcript_fidelity",
        "snomed_validation",
        "clinical_safety",
        "nice_guidelines",
        "completeness",
        "drug_interactions",
    ]:
        print(f"{check:25s}", end="")
        for m in ranking:
            print(f"  {per_check_pass_rate[m][check]:>12.3f}", end="")
        print()

    print(f"\nWritten: {OUT_DIR / 'trustscore_rerun_2026_05.json'}")
    print(f"Written: {OUT_DIR / 'trustscore_per_check_ablation_2026_05.json'}")


if __name__ == "__main__":
    main()
