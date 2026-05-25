"""Constitutional vs Baseline evaluation - empirical comparison for the paper.

Runs all 8 demo GP consultation transcripts through both:
  - Baseline: SOAPGenerator (single-pass generation)
  - Constitutional: ConstitutionalSOAPGenerator (iterative self-correction)

Records per-case and aggregate:
  - baseline_trust_score         - TrustScore after single generation
  - constitutional_initial_score - TrustScore before any repairs (same as baseline)
  - constitutional_final_score   - TrustScore after constitutional repairs
  - improvement                  - constitutional_final - baseline_trust_score
  - score_trajectory             - per-iteration scores showing convergence
  - converged                    - whether ≥70 was reached
  - num_iterations               - repairs needed

This produces Table N in the dissertation: Constitutional AI improves
TrustScore by mean X points across 8 GP consultation types, with Y/8
notes reaching 'approved' verdict vs Z/8 from baseline.

Usage (with Ollama running):
    python3 -m src.evaluation.constitutional_eval

Usage (dry run - shows framework without calling Ollama):
    python3 -m src.evaluation.constitutional_eval --dry-run
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "data" / "evaluation_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── The 8 demo GP consultation transcripts (same as run_model_comparison.py) ─

DEMO_TRANSCRIPTS: dict[str, str] = {
    "Lower Back Pain": (
        "Doctor: Good morning, what brings you in today?\n"
        "Patient: I've been having terrible lower back pain for about three weeks now. "
        "It started after I was lifting some heavy boxes at work.\n"
        "Doctor: Can you describe the pain? Is it sharp, dull, or aching?\n"
        "Patient: It's a dull ache most of the time but gets sharp when I bend forward. "
        "It sometimes goes down my left leg.\n"
        "Doctor: Any numbness or tingling in your legs?\n"
        "Patient: Yes, I get pins and needles in my left foot sometimes.\n"
        "Doctor: Any problems with your bladder or bowels?\n"
        "Patient: No, nothing like that.\n"
        "Doctor: Are you taking anything for the pain?\n"
        "Patient: Just paracetamol, two tablets four times a day, but it's not really helping.\n"
        "Doctor: Let me examine you. I can see some muscle spasm on the left side. "
        "Straight leg raise is positive on the left at 40 degrees. "
        "Reflexes and power are normal. Temperature is 36.8.\n"
        "Doctor: This looks like a lumbar disc protrusion with left-sided sciatica. "
        "I'm prescribing naproxen 500mg twice daily with food, and referring you for physiotherapy. "
        "Come back in four weeks. If the pain gets significantly worse, or you develop any weakness "
        "in your legs, or bladder or bowel problems, please come to A&E immediately or call 999."
    ),
    "Persistent Cough": (
        "Doctor: Hello, how can I help you today?\n"
        "Patient: I've had this cough that won't go away. It's been about six weeks now.\n"
        "Doctor: Any blood in the phlegm?\n"
        "Patient: No, no blood.\n"
        "Doctor: Any fever, night sweats, or weight loss?\n"
        "Patient: I have lost a bit of weight actually, maybe half a stone.\n"
        "Doctor: Are you a smoker?\n"
        "Patient: Yes, I've smoked about 20 a day for 25 years.\n"
        "Doctor: There are some crackles at the right base. Oxygen saturation is 96%.\n"
        "Doctor: Given your smoking history, the persistent cough, weight loss, and what I'm "
        "hearing on examination, I want to arrange an urgent chest X-ray and refer you on the "
        "two-week wait pathway. I'll also do blood tests - FBC, CRP, and LFTs. "
        "If you develop haemoptysis or worsening breathlessness, please call 111 or attend A&E."
    ),
    "Diabetes Review": (
        "Doctor: You're here for your annual diabetes review.\n"
        "Patient: I'm taking metformin 1g twice daily and atorvastatin 20mg at night. "
        "Sometimes the metformin upsets my stomach.\n"
        "Doctor: Your HbA1c has come down to 58 mmol/mol from 65 last time. "
        "Cholesterol is 4.2. eGFR is 85. Blood pressure today is 134 over 82.\n"
        "Doctor: Overall you're doing well. Keep medications the same. "
        "Take metformin with food for the stomach issues. Review in six months. "
        "If blood sugars run consistently above 12, book an urgent appointment."
    ),
    "Mental Health - Anxiety": (
        "Doctor: Hello, I can see you'd like to talk about how you've been feeling.\n"
        "Patient: I've been really struggling. I feel anxious all the time, "
        "I can't sleep, and I'm finding it hard to concentrate at work.\n"
        "Doctor: How long has this been going on?\n"
        "Patient: About two months. I've had a few panic attacks.\n"
        "Doctor: Any thoughts of harming yourself?\n"
        "Patient: My mood is low but no thoughts of harming myself.\n"
        "Doctor: I think you're experiencing generalised anxiety disorder. "
        "I'm referring you to IAPT for CBT therapy. If things don't improve in four to six weeks, "
        "we can discuss sertraline. If you experience any thoughts of self-harm, "
        "please call 111, the Samaritans on 116 123, or attend A&E."
    ),
    "Chest Infection": (
        "Doctor: Good morning. What's the matter today?\n"
        "Patient: Terrible cough with green phlegm and I feel hot and cold for five days.\n"
        "Doctor: Any chest pain or shortness of breath?\n"
        "Patient: A bit of chest pain when I cough, and I'm a bit wheezy.\n"
        "Patient: I've got asthma. I use a salbutamol inhaler and a brown preventer.\n"
        "Doctor: Temperature is 38.4. Sats 97%. Coarse crackles and wheeze at the right lower zone.\n"
        "Doctor: This looks like a lower respiratory tract infection. "
        "Prescribing amoxicillin 500mg three times daily for five days. "
        "Keep using your salbutamol. Take paracetamol for the fever. "
        "Come back in one week. If you become more breathless, fever doesn't settle in 48 hours, "
        "or you cough up blood, call 111 or go to A&E."
    ),
    "Hypertension Review": (
        "Doctor: Your blood pressure is 158 over 96 today. Ambulatory readings averaged 152 over 92.\n"
        "Patient: Any family history of heart disease?\n"
        "Patient: My dad had a heart attack at 55.\n"
        "Doctor: I'd like to start you on ramipril 2.5mg once daily. "
        "We'll check kidney function in two weeks. Reduce salt intake, 30 minutes exercise most days. "
        "Come back in four weeks to recheck. If you get a persistent dry cough, let us know. "
        "If you develop sudden severe headache, chest pain, or visual changes, call 999 immediately."
    ),
    "Paediatric Ear Infection": (
        "Patient's mother: He's been really unsettled for two days. "
        "Pulling at his right ear and running a temperature up to 38.9.\n"
        "Doctor: Temperature now is 38.2. Right tympanic membrane is red and bulging. "
        "Left ear normal.\n"
        "Doctor: This is acute otitis media. Continue paracetamol and ibuprofen for pain and fever. "
        "I'll give a delayed prescription for amoxicillin - use it if not improving after three days. "
        "If he develops a high fever that won't come down, becomes very drowsy, or you see fluid "
        "from the ear, bring him straight back or call 111."
    ),
    "Skin Lesion Assessment": (
        "Patient: I've got a mole on my back that's got bigger and the colour has changed.\n"
        "Doctor: Any itching, bleeding, or crusting?\n"
        "Patient: It's been a bit itchy. No bleeding.\n"
        "Doctor: It's about 8mm, asymmetric, irregular borders, multiple colours including a bluish area. "
        "This meets several ABCDE criteria for a suspicious mole. "
        "I'm referring urgently on the two-week wait pathway to dermatology for excision biopsy. "
        "If it starts bleeding or changes rapidly before your appointment, please call us or attend A&E."
    ),
}


# ── Data types ────────────────────────────────────────────────────────────────


@dataclass
class CaseResult:
    """Comparison result for one consultation type."""
    case_name: str
    baseline_trust_score: int
    baseline_verdict: str
    constitutional_initial_score: int   # score before any repairs
    constitutional_final_score: int     # score after all repairs
    constitutional_verdict: str
    improvement: int                    # constitutional_final - baseline
    score_trajectory: list[int]
    converged: bool
    num_iterations: int
    repair_targets: list[str]           # which checks were repaired
    baseline_generation_secs: float
    constitutional_total_secs: float
    baseline_note: str = ""
    constitutional_note: str = ""


@dataclass
class ConstitutionalBenchmarkResult:
    """Aggregate comparison across all 8 consultation types."""
    model: str
    timestamp: str
    num_cases: int
    # TrustScore stats
    mean_baseline_score: float
    mean_constitutional_score: float
    mean_improvement: float
    # Convergence
    converged_count: int          # notes that reached ≥70
    baseline_approved_count: int  # baseline notes already ≥70
    mean_iterations: float
    # Per-case
    cases: list[CaseResult] = field(default_factory=list)


# ── Run comparison ────────────────────────────────────────────────────────────


def run_comparison(
    model_name: str = "praxis-soap-v1",
    max_iterations: int = 3,
    transcripts: dict[str, str] | None = None,
    save: bool = True,
) -> ConstitutionalBenchmarkResult:
    """Run Constitutional vs Baseline comparison across all demo consultations.

    Requires Ollama to be running with the target model loaded.
    """
    from src.soap_generation.constitutional import ConstitutionalSOAPGenerator
    from src.soap_generation.generator import SOAPGenerator, SOAPModel
    from src.clinical.verification import validate_soap_note

    cases_to_run = transcripts or DEMO_TRANSCRIPTS
    model = SOAPModel(model_name)
    case_results: list[CaseResult] = []

    baseline_gen = SOAPGenerator(model=model)
    const_gen = ConstitutionalSOAPGenerator(model=model, max_iterations=max_iterations)

    for i, (case_name, transcript) in enumerate(cases_to_run.items()):
        logger.info(f"\n[{i+1}/{len(cases_to_run)}] {case_name}")

        # ── Baseline ──
        logger.info("  Baseline generation...")
        t0 = time.perf_counter()
        baseline_result = baseline_gen.generate(transcript)
        baseline_time = round(time.perf_counter() - t0, 2)
        baseline_val = validate_soap_note(baseline_result.soap_note, transcript)
        logger.info(f"  Baseline TrustScore: {baseline_val.trust_score} ({baseline_val.verdict})")

        # ── Constitutional ──
        logger.info("  Constitutional generation...")
        const_result = const_gen.generate(transcript)
        logger.info(
            f"  Constitutional TrustScore: {const_result.final_trust_score} "
            f"({const_result.final_verdict}) in {const_result.num_iterations} iterations"
        )

        repair_targets = [
            target
            for it in const_result.iterations[1:]  # skip iteration 0
            for target in it.repair_targets
        ]

        case_results.append(CaseResult(
            case_name=case_name,
            baseline_trust_score=baseline_val.trust_score,
            baseline_verdict=baseline_val.verdict,
            constitutional_initial_score=const_result.iterations[0].trust_score,
            constitutional_final_score=const_result.final_trust_score,
            constitutional_verdict=const_result.final_verdict,
            improvement=const_result.final_trust_score - baseline_val.trust_score,
            score_trajectory=const_result.score_trajectory,
            converged=const_result.converged,
            num_iterations=const_result.num_iterations,
            repair_targets=repair_targets,
            baseline_generation_secs=baseline_time,
            constitutional_total_secs=const_result.total_time_secs,
            baseline_note=baseline_result.soap_note,
            constitutional_note=const_result.final_note,
        ))

    benchmark = _aggregate(model_name, case_results)

    if save:
        out = RESULTS_DIR / f"constitutional_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out.write_text(json.dumps(asdict(benchmark), indent=2))
        logger.info(f"Results saved: {out}")

    _log_summary(benchmark)
    return benchmark


def _aggregate(model_name: str, cases: list[CaseResult]) -> ConstitutionalBenchmarkResult:
    baseline_scores = [c.baseline_trust_score for c in cases]
    constitutional_scores = [c.constitutional_final_score for c in cases]
    improvements = [c.improvement for c in cases]

    return ConstitutionalBenchmarkResult(
        model=model_name,
        timestamp=datetime.now(timezone.utc).isoformat(),
        num_cases=len(cases),
        mean_baseline_score=round(float(np.mean(baseline_scores)), 1),
        mean_constitutional_score=round(float(np.mean(constitutional_scores)), 1),
        mean_improvement=round(float(np.mean(improvements)), 1),
        converged_count=sum(1 for c in cases if c.converged),
        baseline_approved_count=sum(1 for c in cases if c.baseline_trust_score >= 70),
        mean_iterations=round(float(np.mean([c.num_iterations for c in cases])), 1),
        cases=cases,
    )


def _log_summary(benchmark: ConstitutionalBenchmarkResult) -> None:
    logger.info("\n" + "=" * 60)
    logger.info("CONSTITUTIONAL vs BASELINE - SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Consultations:          {benchmark.num_cases}")
    logger.info(f"  Baseline approved:      {benchmark.baseline_approved_count}/{benchmark.num_cases}")
    logger.info(f"  Constitutional approved:{benchmark.converged_count}/{benchmark.num_cases}")
    logger.info(f"  Mean baseline score:    {benchmark.mean_baseline_score:.1f}")
    logger.info(f"  Mean constitutional:    {benchmark.mean_constitutional_score:.1f}")
    logger.info(f"  Mean improvement:       {benchmark.mean_improvement:+.1f} points")
    logger.info(f"  Mean iterations:        {benchmark.mean_iterations:.1f}")
    for c in benchmark.cases:
        marker = "✓" if c.converged else "·"
        logger.info(
            f"  {marker} {c.case_name:<30} "
            f"{c.baseline_trust_score:>3} → {c.constitutional_final_score:>3} "
            f"({c.improvement:+d}) [{c.num_iterations} iter]"
        )


# ── Output formatting ─────────────────────────────────────────────────────────


def to_latex_table(benchmark: ConstitutionalBenchmarkResult) -> str:
    """Full LaTeX comparison table for the dissertation."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Constitutional Clinical AI: TrustScore Improvement over Baseline}",
        r"\label{tab:constitutional-eval}",
        r"\begin{tabular}{lrrrrl}",
        r"\toprule",
        r"Consultation & Baseline & Constitutional & $\Delta$ & Iterations & Converged \\",
        r"\midrule",
    ]
    for c in benchmark.cases:
        conv = r"\checkmark" if c.converged else "-"
        sign = "+" if c.improvement >= 0 else ""
        lines.append(
            f"  {c.case_name} & {c.baseline_trust_score} & "
            f"{c.constitutional_final_score} & "
            f"{sign}{c.improvement} & {c.num_iterations} & {conv} \\\\"
        )
    lines.extend([
        r"\midrule",
        f"  \\textbf{{Mean}} & "
        f"\\textbf{{{benchmark.mean_baseline_score:.1f}}} & "
        f"\\textbf{{{benchmark.mean_constitutional_score:.1f}}} & "
        f"\\textbf{{{benchmark.mean_improvement:+.1f}}} & "
        f"\\textbf{{{benchmark.mean_iterations:.1f}}} & "
        f"{benchmark.converged_count}/{benchmark.num_cases} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def to_markdown_table(benchmark: ConstitutionalBenchmarkResult) -> str:
    """Markdown comparison table."""
    lines = [
        "### Constitutional Clinical AI vs Baseline",
        "",
        f"Model: `{benchmark.model}` | "
        f"Mean improvement: **{benchmark.mean_improvement:+.1f} points** | "
        f"Converged: **{benchmark.converged_count}/{benchmark.num_cases}**",
        "",
        "| Consultation | Baseline | Constitutional | Δ | Iterations | Approved |",
        "|---|---|---|---|---|---|",
    ]
    for c in benchmark.cases:
        sign = "+" if c.improvement >= 0 else ""
        approved = "✓" if c.converged else "·"
        lines.append(
            f"| {c.case_name} | {c.baseline_trust_score} | "
            f"{c.constitutional_final_score} | {sign}{c.improvement} | "
            f"{c.num_iterations} | {approved} |"
        )
    lines.extend([
        f"| **Mean** | **{benchmark.mean_baseline_score:.1f}** | "
        f"**{benchmark.mean_constitutional_score:.1f}** | "
        f"**{benchmark.mean_improvement:+.1f}** | "
        f"**{benchmark.mean_iterations:.1f}** | "
        f"**{benchmark.converged_count}/{benchmark.num_cases}** |",
    ])
    return "\n".join(lines)


# ── Dry-run demo (no Ollama needed) ──────────────────────────────────────────


def dry_run() -> ConstitutionalBenchmarkResult:
    """Simulate the evaluation using TrustScore-only (no LLM calls).

    Demonstrates the evaluation framework structure with synthetic notes
    that reflect realistic performance gaps between baseline and constitutional.
    """
    from src.clinical.verification import validate_soap_note

    logger.info("DRY RUN - using synthetic notes to demonstrate framework")

    # Synthetic baseline notes (realistic - missing safety-netting or NICE issues)
    BASELINE_NOTES = {
        "Lower Back Pain": (
            "SUBJECTIVE:\nPatient reports 3-week history of lower back pain radiating to left leg.\n\n"
            "OBJECTIVE:\nStraight leg raise positive left 40°. Power and reflexes normal. Temp 36.8.\n\n"
            "ASSESSMENT:\nLumbar disc protrusion with left sciatica [SNOMED: 161891005]\n\n"
            "PLAN:\n1. Naproxen 500mg BD with food\n2. Paracetamol 1g QDS\n3. Refer physiotherapy"
        ),
        "Persistent Cough": (
            "SUBJECTIVE:\nSix-week persistent cough with weight loss. Smoker 20/day x 25 years.\n\n"
            "OBJECTIVE:\nCrackles right base. Sats 96%. BP 138/85.\n\n"
            "ASSESSMENT:\nSuspicious cough - 2WW referral indicated [SNOMED: 49727002]\n\n"
            "PLAN:\n1. Urgent CXR\n2. 2-week wait referral\n3. FBC, CRP, LFTs"
        ),
        "Diabetes Review": (
            "SUBJECTIVE:\nAnnual diabetes review. On metformin 1g BD, atorvastatin 20mg nocte.\n\n"
            "OBJECTIVE:\nHbA1c 58, eGFR 85, BP 134/82, BMI 31.2.\n\n"
            "ASSESSMENT:\nType 2 diabetes, well controlled [SNOMED: 44054006]\n\n"
            "PLAN:\n1. Continue current medications\n2. Take metformin with food\n3. Review 6 months"
        ),
        "Mental Health - Anxiety": (
            "SUBJECTIVE:\nTwo months of anxiety, panic attacks, insomnia, low mood.\n\n"
            "OBJECTIVE:\nAppropriately distressed. No evidence of psychosis.\n\n"
            "ASSESSMENT:\nGeneralised anxiety disorder [SNOMED: 197480006]\n\n"
            "PLAN:\n1. IAPT referral for CBT\n2. Alcohol reduction advice\n3. Review 4-6 weeks"
        ),
        "Chest Infection": (
            "SUBJECTIVE:\nFive-day history of cough with green sputum and fever. Asthmatic.\n\n"
            "OBJECTIVE:\nTemp 38.4. Sats 97%. Crackles and wheeze right lower zone.\n\n"
            "ASSESSMENT:\nLower respiratory tract infection [SNOMED: 10509002]\n\n"
            "PLAN:\n1. Amoxicillin 500mg TDS x 5 days\n2. Continue salbutamol\n3. Paracetamol PRN"
        ),
        "Hypertension Review": (
            "SUBJECTIVE:\nHigh BP noted. Family history of MI at 55.\n\n"
            "OBJECTIVE:\nBP 158/96. Ambulatory average 152/92.\n\n"
            "ASSESSMENT:\nHypertension [SNOMED: 38341003]\n\n"
            "PLAN:\n1. Ramipril 2.5mg OD\n2. Renal function check in 2 weeks\n3. Salt reduction"
        ),
        "Paediatric Ear Infection": (
            "SUBJECTIVE:\nTwo-day history of right ear pain, temp 38.9 in child.\n\n"
            "OBJECTIVE:\nTemp 38.2. Right tympanic membrane red and bulging.\n\n"
            "ASSESSMENT:\nAcute otitis media [SNOMED: 65363002]\n\n"
            "PLAN:\n1. Paracetamol and ibuprofen PRN\n2. Delayed amoxicillin prescription"
        ),
        "Skin Lesion Assessment": (
            "SUBJECTIVE:\nChanging mole - 8mm, asymmetric, irregular, multiple colours.\n\n"
            "OBJECTIVE:\nABCDE criteria met. Irregular pigment network on dermatoscopy.\n\n"
            "ASSESSMENT:\nSuspicious pigmented lesion [SNOMED: 32343005]\n\n"
            "PLAN:\n1. Urgent 2WW dermatology referral for excision biopsy"
        ),
    }

    case_results: list[CaseResult] = []

    for case_name, transcript in DEMO_TRANSCRIPTS.items():
        baseline_note = BASELINE_NOTES.get(case_name, "")
        baseline_val = validate_soap_note(baseline_note, transcript)

        # Simulate constitutional improvement: add safety-netting to plans that lack it
        const_note = baseline_note
        if "safety" not in const_note.lower() and "111" not in const_note and "999" not in const_note:
            const_note = const_note.rstrip() + (
                "\n4. Safety-netting: return if symptoms worsen, call 111 if concerned, "
                "call 999 if emergency symptoms develop"
            )
        const_val = validate_soap_note(const_note, transcript)

        improvement = const_val.trust_score - baseline_val.trust_score
        converged = const_val.trust_score >= 70
        n_iters = 2 if improvement > 0 else 1

        case_results.append(CaseResult(
            case_name=case_name,
            baseline_trust_score=baseline_val.trust_score,
            baseline_verdict=baseline_val.verdict,
            constitutional_initial_score=baseline_val.trust_score,
            constitutional_final_score=const_val.trust_score,
            constitutional_verdict=const_val.verdict,
            improvement=improvement,
            score_trajectory=[baseline_val.trust_score, const_val.trust_score],
            converged=converged,
            num_iterations=n_iters,
            repair_targets=["clinical_safety"] if improvement > 0 else [],
            baseline_generation_secs=2.0,
            constitutional_total_secs=6.0 if improvement > 0 else 2.0,
        ))

    benchmark = _aggregate("praxis-soap-v1 (dry-run)", case_results)
    _log_summary(benchmark)

    out = RESULTS_DIR / f"constitutional_eval_dryrun_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(asdict(benchmark), indent=2))
    logger.info(f"Dry-run results saved: {out}")
    return benchmark


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Run without Ollama (synthetic notes)")
    ap.add_argument("--model", default="praxis-soap-v1", help="Ollama model to use")
    ap.add_argument("--max-iterations", type=int, default=3)
    args = ap.parse_args()

    if args.dry_run:
        result = dry_run()
    else:
        result = run_comparison(model_name=args.model, max_iterations=args.max_iterations)

    print("\n" + to_markdown_table(result))
    print("\n--- LaTeX ---")
    print(to_latex_table(result))
