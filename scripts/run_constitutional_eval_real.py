#!/usr/bin/env python3
"""Real Constitutional Clinical AI evaluation - no dry_run mock.

Runs ConstitutionalSOAPGenerator against the 8 GP case transcripts via Ollama
on the local box (assumes praxis-soap-v1 is registered). For each iteration:

  - Records baseline TrustScore + verdict + per-check pass map
  - Records the target check chosen by _REPAIR_PRIORITY
  - Records the section text BEFORE and AFTER the repair splice
  - Records the post-iteration TrustScore + verdict
  - Records wall time per call

Writes the full trajectory to data/evaluation_results/constitutional_eval_real_latest.json
with explicit metadata: is_dry_run=false, hardware_target, ollama_endpoint, git commit.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Pull the canonical 8 transcripts from the comparison driver.
from scripts.run_model_comparison import TRANSCRIPTS  # noqa: E402

from src.clinical.verification import validate_soap_note  # noqa: E402
from src.soap_generation.constitutional import (  # noqa: E402
    _REPAIR_INSTRUCTIONS,
    _REPAIR_PRIORITY,
    _extract_section,
    _splice_section,
    ConstitutionalSOAPGenerator,
)
from src.soap_generation.generator import SOAPGenerator, SOAPModel  # noqa: E402


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _validation_to_dict(v) -> dict:
    return {
        "trust_score": v.trust_score,
        "verdict": v.verdict,
        "checks_passed": v.checks_passed,
        "checks_total": v.checks_total,
        "per_check": {
            r.check_name: {"passed": r.passed, "n_issues": len(r.issues), "issues": r.issues[:6]}
            for r in v.check_results
        },
        "issues_summary": v.issues[:8],
    }


def run_one_case(case_name: str, transcript: str, gen: SOAPGenerator,
                 max_iterations: int, target_score: int) -> dict:
    """Execute the constitutional loop on a single case, capturing every step."""
    print(f"\n[CASE] {case_name}", flush=True)
    case_t0 = time.perf_counter()

    # Initial generation
    t0 = time.perf_counter()
    initial = gen.generate(transcript)
    init_secs = round(time.perf_counter() - t0, 2)
    note = initial.soap_note
    total_tokens = initial.token_count

    validation = validate_soap_note(note, transcript)
    print(
        f"  baseline: score={validation.trust_score} verdict={validation.verdict}"
        f"  ({validation.checks_passed}/{validation.checks_total})  {init_secs}s",
        flush=True,
    )

    iterations = [{
        "iteration": 0,
        "kind": "baseline",
        "soap_note": note,
        "trust_score": validation.trust_score,
        "verdict": validation.verdict,
        "checks_passed": validation.checks_passed,
        "checks_total": validation.checks_total,
        "per_check": _validation_to_dict(validation)["per_check"],
        "issues_summary": validation.issues[:8],
        "repair_target": None,
        "repair_instruction": None,
        "section_text_before": None,
        "section_text_after": None,
        "wall_time_s": init_secs,
        "token_count": initial.token_count,
    }]

    best_note = note
    best_score = validation.trust_score
    best_iteration_index = 0

    # Repair loop
    for i in range(1, max_iterations + 1):
        if validation.trust_score >= target_score:
            print(f"  converged at i={i - 1}", flush=True)
            break

        failing = {r.check_name for r in validation.check_results if not r.passed}
        target_check = next((c for c in _REPAIR_PRIORITY if c in failing), None)
        if not target_check:
            print(f"  no repairable check; stopping", flush=True)
            break
        section, instruction = _REPAIR_INSTRUCTIONS[target_check]
        before = _extract_section(note, section)
        issues_for_target = [
            iss for r in validation.check_results
            if r.check_name == target_check
            for iss in r.issues
        ]

        # Build the same repair prompt the production class uses
        issues_text = ""
        if issues_for_target:
            issues_text = "\n\nSpecific issues to fix:\n" + "\n".join(
                f"- {iss}" for iss in issues_for_target
            )
        repair_prompt = (
            f"You are a UK GP documentation assistant. Below is a clinical note that needs "
            f"a targeted correction.\n\n"
            f"CONSULTATION TRANSCRIPT:\n{transcript}\n\n"
            f"CURRENT {section} SECTION:\n{before}\n\n"
            f"CORRECTION REQUIRED:\n{instruction}{issues_text}\n\n"
            f"Output ONLY the corrected {section} section content - "
            f"no section header, no other sections, no explanation. "
            f"Start directly with the corrected content."
        )

        t0 = time.perf_counter()
        # Direct Ollama call so we capture the raw critique/response without re-pasting
        import httpx
        with httpx.Client(timeout=180.0) as client:
            resp = client.post(
                f"{gen.ollama_url}/api/chat",
                json={
                    "model": gen.model.value,
                    "messages": [{"role": "user", "content": repair_prompt}],
                    "stream": False,
                    "options": {"temperature": 0.1},
                },
            )
        repair_secs = round(time.perf_counter() - t0, 2)
        repair_payload = resp.json()
        after = (repair_payload.get("message") or {}).get("content", "").strip()
        if not after:
            after = repair_payload.get("response", "").strip()

        note = _splice_section(note, section, after)
        validation = validate_soap_note(note, transcript)
        total_tokens += repair_payload.get("eval_count", 0) + repair_payload.get("prompt_eval_count", 0)

        print(
            f"  i={i} repair[{target_check}]: score={validation.trust_score}"
            f" verdict={validation.verdict}  ({repair_secs}s)",
            flush=True,
        )

        iterations.append({
            "iteration": i,
            "kind": "repair",
            "soap_note": note,
            "trust_score": validation.trust_score,
            "verdict": validation.verdict,
            "checks_passed": validation.checks_passed,
            "checks_total": validation.checks_total,
            "per_check": _validation_to_dict(validation)["per_check"],
            "issues_summary": validation.issues[:8],
            "repair_target": target_check,
            "repair_instruction": instruction,
            "repair_prompt": repair_prompt,
            "section_text_before": before,
            "section_text_after": after,
            "wall_time_s": repair_secs,
        })

        if validation.trust_score > best_score:
            best_score = validation.trust_score
            best_note = note
            best_iteration_index = i

    converged = best_score >= target_score
    case_secs = round(time.perf_counter() - case_t0, 2)
    improvement = best_score - iterations[0]["trust_score"]
    print(
        f"  FINAL: best={best_score} (from baseline {iterations[0]['trust_score']}, "
        f"Δ={improvement:+d}) converged={converged} total={case_secs}s",
        flush=True,
    )

    return {
        "case_name": case_name,
        "baseline_trust_score": iterations[0]["trust_score"],
        "baseline_verdict": iterations[0]["verdict"],
        "best_trust_score": best_score,
        "best_iteration_index": best_iteration_index,
        "best_verdict": ("approved" if best_score >= 70
                         else ("needs_review" if best_score >= 40 else "rejected")),
        "improvement": improvement,
        "converged": converged,
        "n_iterations_used": len(iterations) - 1,
        "score_trajectory": [it["trust_score"] for it in iterations],
        "iterations": iterations,
        "total_wall_time_s": case_secs,
        "total_tokens": total_tokens,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    ap.add_argument("--model", default="praxis-soap-v1")
    ap.add_argument("--max-iterations", type=int, default=3)
    ap.add_argument("--target-score", type=int, default=70)
    ap.add_argument("--out", default=str(REPO_ROOT / "data/evaluation_results/constitutional_eval_real_latest.json"))
    args = ap.parse_args()

    gen = SOAPGenerator(
        model=SOAPModel(args.model),
        ollama_url=args.ollama_url,
    )
    if not gen.check_ollama():
        print(f"ERROR: cannot reach Ollama at {args.ollama_url}", file=sys.stderr)
        return 1

    print(f"Running {len(TRANSCRIPTS)} cases on {args.model} via {args.ollama_url}", flush=True)
    print(f"max_iterations={args.max_iterations} target_score={args.target_score}", flush=True)

    t_start = time.perf_counter()
    cases = []
    for case_name, transcript in TRANSCRIPTS.items():
        case = run_one_case(case_name, transcript, gen,
                            max_iterations=args.max_iterations,
                            target_score=args.target_score)
        cases.append(case)

    total_secs = round(time.perf_counter() - t_start, 2)

    # Aggregate
    baseline_scores = [c["baseline_trust_score"] for c in cases]
    final_scores = [c["best_trust_score"] for c in cases]
    improvements = [c["improvement"] for c in cases]
    mean_baseline = sum(baseline_scores) / len(baseline_scores)
    mean_final = sum(final_scores) / len(final_scores)
    mean_improvement = sum(improvements) / len(improvements)
    n_converged = sum(1 for c in cases if c["converged"])
    n_regressed = sum(1 for c in cases if c["improvement"] < 0)
    n_unchanged = sum(1 for c in cases if c["improvement"] == 0)
    n_improved = sum(1 for c in cases if c["improvement"] > 0)
    mean_iterations = sum(c["n_iterations_used"] for c in cases) / len(cases)

    output = {
        "is_dry_run": False,
        "model": args.model,
        "ollama_endpoint": args.ollama_url,
        "hardware_target": "RTX 5090 32GB Linux (satyawan-1)",
        "host": socket.gethostname(),
        "git_commit": _git_sha(),
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "max_iterations": args.max_iterations,
        "target_score": args.target_score,
        "trust_score_implementation": "src/clinical/verification.py (current)",
        "n_cases": len(cases),
        "aggregate": {
            "mean_baseline_score": round(mean_baseline, 3),
            "mean_final_score": round(mean_final, 3),
            "mean_improvement": round(mean_improvement, 3),
            "n_converged": n_converged,
            "n_improved": n_improved,
            "n_unchanged": n_unchanged,
            "n_regressed": n_regressed,
            "mean_iterations_used": round(mean_iterations, 3),
            "total_wall_time_s": total_secs,
        },
        "cases": cases,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWritten: {args.out}", flush=True)
    print(f"  mean baseline:  {mean_baseline:.2f}", flush=True)
    print(f"  mean final:     {mean_final:.2f}", flush=True)
    print(f"  mean Δ:         {mean_improvement:+.2f}", flush=True)
    print(f"  converged:      {n_converged}/{len(cases)}", flush=True)
    print(f"  regressed:      {n_regressed}/{len(cases)}", flush=True)
    print(f"  total wall:     {total_secs}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
