#!/usr/bin/env python3
"""LLM-judge evaluation of 32 SOAP outputs from 4 generation models.

Uses a locally hosted instruction-tuned LLM (default: mistral:7b via Ollama)
as the judge. Each note is scored on six criteria matching the paper's
described rubric: structure, content fidelity, SNOMED code count,
safety-netting language, UK English idiom, time/token budget.

Output: data/evaluation_results/llm_judge_<model>_<timestamp>.json
        + aggregate llm_judge_significance.json (Wilcoxon vs praxis-soap-v1)

Usage:
    python3 scripts/llm_judge_eval.py [--judge-model mistral:latest]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
OLLAMA_URL = "http://localhost:11434/api/generate"

# Transcripts (verbatim from scripts/run_model_comparison.py - kept here so the
# judge sees the same source-of-truth the auto-rubric had access to).
sys.path.insert(0, str(ROOT))
from scripts.run_model_comparison import TRANSCRIPTS  # type: ignore  # noqa: E402

JUDGE_PROMPT = """You are an expert NHS GP reviewer evaluating an AI-generated SOAP consultation note.

Score the note on six criteria, each on a 0.0-1.0 scale where 1.0 = perfect, 0.0 = completely missing or wrong:

1. structure: Are all four SOAP sections (Subjective, Objective, Assessment, Plan) clearly present and well-organised?
2. content: Does the note accurately reflect what was discussed in the consultation transcript, without omitting key clinical information or fabricating content?
3. snomed: Are appropriate SNOMED CT codes present (numeric concept IDs in the note where clinical concepts are mentioned)?
4. safety_netting: Does the Plan section include appropriate safety-netting advice (e.g., "return if symptoms worsen", "seek urgent help if...", "call 111 if...")?
5. uk_english: Does the note use British clinical terminology (e.g., "GP", "A&E", "paracetamol") rather than American (e.g., "PCP", "ER", "acetaminophen")?
6. budget: Is the note appropriately concise and well-paced - neither truncated nor excessively verbose?

CONSULTATION TRANSCRIPT:
---
{transcript}
---

GENERATED SOAP NOTE:
---
{soap_note}
---

Return ONLY a single JSON object on one line, with keys structure, content, snomed, safety_netting, uk_english, budget (each 0.0-1.0 float to 2 decimal places) and overall (the unweighted mean of the six). Example: {{"structure": 0.95, "content": 0.85, "snomed": 0.70, "safety_netting": 0.90, "uk_english": 1.00, "budget": 0.90, "overall": 0.88}}

JSON:"""


def call_judge(judge_model: str, transcript: str, soap_note: str, max_retries: int = 2) -> dict:
    """Single LLM-judge call, returns parsed score dict or {} on failure."""
    prompt = JUDGE_PROMPT.format(transcript=transcript, soap_note=soap_note)
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model": judge_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 200},
                },
                timeout=120,
            )
            resp.raise_for_status()
            text = resp.json().get("response", "").strip()
            # Extract first JSON object from response
            match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
            if not match:
                if attempt < max_retries:
                    continue
                return {"_error": "no_json", "_raw": text[:300]}
            obj = json.loads(match.group(0))
            # Ensure overall is computed if missing
            keys = ("structure", "content", "snomed", "safety_netting", "uk_english", "budget")
            if all(k in obj for k in keys):
                if "overall" not in obj or not isinstance(obj["overall"], (int, float)):
                    obj["overall"] = sum(float(obj[k]) for k in keys) / 6.0
                return obj
            if attempt < max_retries:
                continue
            return {"_error": "missing_keys", "_raw": text[:300]}
        except Exception as exc:
            if attempt < max_retries:
                time.sleep(1)
                continue
            return {"_error": str(exc)[:200]}
    return {"_error": "exhausted_retries"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-model", default="mistral:latest")
    ap.add_argument(
        "--input",
        default=str(ROOT / "docs" / "model_comparison_results.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "data" / "evaluation_results"),
    )
    args = ap.parse_args()

    records = json.load(open(args.input))
    print(f"Loaded {len(records)} records. Judge model: {args.judge_model}")

    # Verify judge model is loaded
    tags = requests.get("http://localhost:11434/api/tags").json()
    available = {m["name"] for m in tags.get("models", [])}
    if args.judge_model not in available:
        print(f"ERROR: judge model {args.judge_model} not in Ollama. Available: {sorted(available)}", file=sys.stderr)
        return 2

    scored: list[dict] = []
    t0 = time.time()
    for i, rec in enumerate(records, 1):
        case = rec["case"]
        model = rec["model_id"]
        soap_note = rec["soap_note"]
        transcript = TRANSCRIPTS.get(case, "")
        if not transcript:
            print(f"  [{i}/{len(records)}] {model} / {case} -- NO TRANSCRIPT, skipping")
            continue
        ts = time.time()
        score = call_judge(args.judge_model, transcript, soap_note)
        elapsed = time.time() - ts
        scored.append(
            {
                "case": case,
                "model": model,
                "judge_model": args.judge_model,
                "judge_score": score,
                "elapsed_secs": round(elapsed, 2),
            }
        )
        ov = score.get("overall", "?")
        ov_str = f"{ov:.3f}" if isinstance(ov, (int, float)) else str(ov)
        print(f"  [{i}/{len(records)}] {model:<18} / {case:<25} overall={ov_str} ({elapsed:.1f}s)")

    total_elapsed = time.time() - t0
    print(f"\nTotal: {total_elapsed:.1f}s ({total_elapsed/len(records):.1f}s/note)")

    # Aggregate per-model
    per_model: dict[str, list[float]] = defaultdict(list)
    per_model_per_case: dict[str, dict[str, float]] = defaultdict(dict)
    for s in scored:
        ov = s["judge_score"].get("overall")
        if isinstance(ov, (int, float)):
            per_model[s["model"]].append(float(ov))
            per_model_per_case[s["case"]][s["model"]] = float(ov)

    means = {m: sum(v) / len(v) if v else float("nan") for m, v in per_model.items()}
    print("\n=== Per-model means (LLM-judge) ===")
    for m, mean in sorted(means.items(), key=lambda kv: -kv[1]):
        n = len(per_model[m])
        print(f"  {m:<22} mean={mean:.4f} n={n}")

    # Wilcoxon signed-rank, paired by case, vs praxis-soap-v1
    reference = "praxis-soap-v1"
    cases = sorted(per_model_per_case.keys())
    ref_scores = [per_model_per_case[c].get(reference, float("nan")) for c in cases]

    sig: dict[str, dict] = {}
    for m in sorted(per_model.keys()):
        if m == reference:
            continue
        m_scores = [per_model_per_case[c].get(m, float("nan")) for c in cases]
        # Only paired cases where both have scores
        paired = [
            (a, b) for a, b in zip(ref_scores, m_scores)
            if not (a != a or b != b)  # NaN check
        ]
        if len(paired) < 2:
            sig[m] = {"wilcoxon_p": None, "n_paired": len(paired)}
            continue
        ref_p = [a for a, _ in paired]
        m_p = [b for _, b in paired]
        try:
            stat = wilcoxon(ref_p, m_p, zero_method="pratt")
            p = float(stat.pvalue)
        except Exception as e:
            p = None  # type: ignore
        sig[m] = {"wilcoxon_p": p, "n_paired": len(paired)}

    print("\n=== Wilcoxon vs praxis-soap-v1 (paired by case) ===")
    for m, info in sig.items():
        p = info["wilcoxon_p"]
        p_str = f"{p:.4f}" if p is not None else "n/a"
        print(f"  {m:<22} p={p_str} n_paired={info['n_paired']}")

    # Holm correction
    p_values = [(m, info["wilcoxon_p"]) for m, info in sig.items() if info["wilcoxon_p"] is not None]
    p_sorted = sorted(p_values, key=lambda kv: kv[1])
    k = len(p_sorted)
    holm = {}
    for rank, (m, p) in enumerate(p_sorted, start=1):
        holm_p = min(1.0, p * (k - rank + 1))
        holm[m] = holm_p

    print("\n=== Holm-corrected (α=0.05) ===")
    for m, p_corr in sorted(holm.items(), key=lambda kv: kv[1]):
        marker = " *" if p_corr < 0.05 else ""
        print(f"  {m:<22} p_holm={p_corr:.4f}{marker}")

    # Write outputs
    ts_iso = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detailed_file = out_dir / f"llm_judge_{args.judge_model.replace(':', '_').replace('/', '_')}_{ts_iso}.json"
    json.dump(
        {
            "timestamp": ts_iso,
            "judge_model": args.judge_model,
            "n_records": len(records),
            "n_scored": len(scored),
            "per_record": scored,
            "per_model_means": means,
            "wilcoxon_vs_praxis_soap_v1": sig,
            "holm_corrected": holm,
            "judge_prompt": JUDGE_PROMPT,
        },
        open(detailed_file, "w"),
        indent=2,
    )
    print(f"\nWrote: {detailed_file}")

    sig_file = out_dir / "llm_judge_significance.json"
    json.dump(
        {
            "timestamp": ts_iso,
            "judge_model": args.judge_model,
            "n_cases": len(cases),
            "cases": cases,
            "per_case": dict(per_model_per_case),
            "per_model_means": means,
            "wilcoxon_vs_praxis_soap_v1": sig,
            "holm_corrected": holm,
        },
        open(sig_file, "w"),
        indent=2,
    )
    print(f"Wrote: {sig_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
