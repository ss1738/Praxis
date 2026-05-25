"""Asgari-style hallucination + omission evaluation harness.

Evaluates SOAP notes against two complementary metrics:

  1. Sentence-level entailment hallucination rate  (Asgari-style)
     Each sentence in the note is checked against the source transcript using
     the NLI entailment gate. Flagged sentences (P_entailment < threshold) are
     counted as hallucinations.

  2. Medical Hallucination Index (MHI) omission rate  (consequence-weighted)
     Consequence-weighted fraction of clinically important transcript terms
     absent from the note.

Target: beat Asgari et al. npj Digital Medicine 2025 (s41746-025-01670-7)
    - Hallucination rate across 4 commercial scribes: 1.47%
    - Omission rate:                                  3.45%

Usage (dry-run, no models needed):
    python3 -m src.evaluation.hallucination_eval --dry-run

Usage (with Ollama + entailment model downloaded):
    python3 -m src.evaluation.hallucination_eval --notes-dir data/pipeline_runs/
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "data" / "evaluation_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Asgari et al. npj Digital Medicine 2025 - published bars
ASGARI_HALLUCINATION_BAR = 0.0147
ASGARI_OMISSION_BAR = 0.0345


# ── Data types ────────────────────────────────────────────────────────────────


@dataclass
class NoteHallucinationResult:
    """Hallucination evaluation for a single SOAP note."""
    note_id: str
    model: str
    # Sentence-level (Asgari-style)
    num_sentences: int
    num_hallucinated: int
    sentence_hallucination_rate: float
    flagged_sentences: list[str] = field(default_factory=list)
    entailment_skipped: bool = False
    # MHI consequence-weighted
    mhi_score: float = 1.0
    mhi_omission: float = 0.0
    mhi_hallucination: float = 0.0
    # Delta vs Asgari bar
    delta_hallucination: float = 0.0   # our rate − 1.47% (negative = better)
    delta_omission: float = 0.0        # our MHI omission − 3.45% (negative = better)


@dataclass
class HallucinationBenchmarkResult:
    """Aggregate hallucination benchmark across all evaluated notes."""
    model: str
    timestamp: str
    num_notes: int
    # Sentence-level aggregate
    total_sentences: int
    total_hallucinated: int
    mean_sentence_hallucination_rate: float
    ci_hallucination_lower: float
    ci_hallucination_upper: float
    entailment_skipped: bool
    # MHI aggregate
    mean_mhi_score: float
    mean_mhi_omission: float
    mean_mhi_hallucination: float
    ci_mhi_omission_lower: float
    ci_mhi_omission_upper: float
    # Comparison
    asgari_hallucination_bar: float = ASGARI_HALLUCINATION_BAR
    asgari_omission_bar: float = ASGARI_OMISSION_BAR
    beats_asgari_hallucination: bool = False
    beats_asgari_omission: bool = False
    notes: list[NoteHallucinationResult] = field(default_factory=list)


# ── Bootstrap CI ──────────────────────────────────────────────────────────────


def _bootstrap_ci(
    values: list[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if len(values) < 2:
        v = values[0] if values else 0.0
        return v, v
    rng = np.random.default_rng(42)
    arr = np.array(values)
    means = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)])
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


# ── Core evaluation ───────────────────────────────────────────────────────────


def evaluate_note(
    soap_note: str,
    transcript: str,
    note_id: str = "",
    model: str = "unknown",
    use_entailment: bool = True,
) -> NoteHallucinationResult:
    """Evaluate hallucination and omission for a single (note, transcript) pair."""
    from src.evaluation.mhi import compute_mhi

    # MHI - no model required
    mhi = compute_mhi(transcript, soap_note)

    # Sentence-level entailment gate
    num_sentences = 0
    num_hallucinated = 0
    flagged: list[str] = []
    entailment_skipped = True

    if use_entailment and transcript.strip():
        from src.clinical.entailment_gate import EntailmentGate
        gate = EntailmentGate()
        gate_result = gate.check(soap_note, transcript)
        entailment_skipped = gate_result.skipped
        num_sentences = gate_result.num_checked
        num_hallucinated = gate_result.num_flagged
        flagged = [s.sentence for s in gate_result.flagged]

    hall_rate = (
        round(num_hallucinated / num_sentences, 4)
        if num_sentences > 0 else 0.0
    )

    return NoteHallucinationResult(
        note_id=note_id or datetime.now().strftime("%Y%m%d_%H%M%S"),
        model=model,
        num_sentences=num_sentences,
        num_hallucinated=num_hallucinated,
        sentence_hallucination_rate=hall_rate,
        flagged_sentences=flagged,
        entailment_skipped=entailment_skipped,
        mhi_score=round(mhi.mhi_score, 4),
        mhi_omission=round(mhi.mhi_omission, 4),
        mhi_hallucination=round(mhi.mhi_hallucination, 4),
        delta_hallucination=round(hall_rate - ASGARI_HALLUCINATION_BAR, 4),
        delta_omission=round(mhi.mhi_omission - ASGARI_OMISSION_BAR, 4),
    )


def run_hallucination_eval(
    notes: list[dict],
    model: str = "unknown",
    use_entailment: bool = True,
    save: bool = True,
) -> HallucinationBenchmarkResult:
    """Evaluate hallucination across a list of (soap_note, transcript) dicts.

    Each dict must have:
        soap_note  (str) - generated SOAP note
        transcript (str) - source consultation transcript
        id         (str, optional) - note identifier
    """
    note_results: list[NoteHallucinationResult] = []

    for i, entry in enumerate(notes):
        soap = entry.get("soap_note", "")
        transcript = entry.get("transcript", "")
        note_id = entry.get("id", f"note_{i:03d}")

        if not soap:
            logger.warning(f"Skipping {note_id}: empty SOAP note")
            continue

        result = evaluate_note(soap, transcript, note_id, model, use_entailment)
        note_results.append(result)

        logger.info(
            f"  [{i+1}/{len(notes)}] {note_id}: "
            f"hall={result.sentence_hallucination_rate:.1%} "
            f"mhi_omit={result.mhi_omission:.3f} "
            f"mhi_hall={result.mhi_hallucination:.3f}"
        )

    if not note_results:
        raise ValueError("No valid notes to evaluate")

    hall_rates = [r.sentence_hallucination_rate for r in note_results]
    omissions = [r.mhi_omission for r in note_results]
    mhi_halls = [r.mhi_hallucination for r in note_results]
    mhi_scores = [r.mhi_score for r in note_results]

    hall_ci = _bootstrap_ci(hall_rates)
    omit_ci = _bootstrap_ci(omissions)

    total_sent = sum(r.num_sentences for r in note_results)
    total_hall = sum(r.num_hallucinated for r in note_results)
    corpus_hall_rate = round(total_hall / total_sent, 4) if total_sent > 0 else 0.0
    mean_omission = round(float(np.mean(omissions)), 4)
    any_skipped = any(r.entailment_skipped for r in note_results)

    benchmark = HallucinationBenchmarkResult(
        model=model,
        timestamp=datetime.now(timezone.utc).isoformat(),
        num_notes=len(note_results),
        total_sentences=total_sent,
        total_hallucinated=total_hall,
        mean_sentence_hallucination_rate=corpus_hall_rate,
        ci_hallucination_lower=round(hall_ci[0], 4),
        ci_hallucination_upper=round(hall_ci[1], 4),
        entailment_skipped=any_skipped,
        mean_mhi_score=round(float(np.mean(mhi_scores)), 4),
        mean_mhi_omission=mean_omission,
        mean_mhi_hallucination=round(float(np.mean(mhi_halls)), 4),
        ci_mhi_omission_lower=round(omit_ci[0], 4),
        ci_mhi_omission_upper=round(omit_ci[1], 4),
        beats_asgari_hallucination=corpus_hall_rate < ASGARI_HALLUCINATION_BAR,
        beats_asgari_omission=mean_omission < ASGARI_OMISSION_BAR,
        notes=note_results,
    )

    if save:
        out = RESULTS_DIR / f"hallucination_eval_{model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out.write_text(json.dumps(asdict(benchmark), indent=2))
        logger.info(f"Results saved: {out}")

    _log_summary(benchmark)
    return benchmark


# ── Output formatting ─────────────────────────────────────────────────────────


def to_latex_table(benchmarks: list[HallucinationBenchmarkResult]) -> str:
    """Dissertation table: hallucination comparison including Asgari bar."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Hallucination and Omission Evaluation (cf.\ Asgari et al.\ 2025)}",
        r"\label{tab:hallucination-eval}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Model & $N_\text{sent}$ & Hall.\% & 95\%~CI & MHI Omit & MHI Hall & $\Delta$Hall \\",
        r"\midrule",
    ]
    for b in benchmarks:
        beats = r"\checkmark" if b.beats_asgari_hallucination else "-"
        sign = "+" if b.mean_sentence_hallucination_rate >= ASGARI_HALLUCINATION_BAR else ""
        lines.append(
            f"  {b.model} & {b.total_sentences} & "
            f"{b.mean_sentence_hallucination_rate*100:.2f}\\% & "
            f"[{b.ci_hallucination_lower*100:.2f}, {b.ci_hallucination_upper*100:.2f}] & "
            f"{b.mean_mhi_omission:.3f} & "
            f"{b.mean_mhi_hallucination:.3f} & "
            f"{sign}{(b.mean_sentence_hallucination_rate - ASGARI_HALLUCINATION_BAR)*100:.2f}pp \\\\"
        )
    lines.extend([
        r"\midrule",
        "  Asgari et al.\\ 2025 (4 scribes) & 12{,}999 & "
        f"{ASGARI_HALLUCINATION_BAR*100:.2f}\\% & --- & "
        f"{ASGARI_OMISSION_BAR:.3f} & --- & --- \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def to_markdown_table(benchmarks: list[HallucinationBenchmarkResult]) -> str:
    """Markdown comparison table including Asgari bar."""
    lines = [
        "### Hallucination & Omission Evaluation",
        "",
        f"*Asgari et al. npj Digital Medicine 2025 bar: "
        f"{ASGARI_HALLUCINATION_BAR:.1%} hallucination / "
        f"{ASGARI_OMISSION_BAR:.1%} omission (12,999 sentences, 4 commercial scribes)*",
        "",
        "| Model | Sentences | Hall% | 95% CI | MHI Omit | MHI Hall | Δ vs Asgari | Beats bar |",
        "|-------|-----------|-------|--------|----------|----------|-------------|-----------|",
    ]
    for b in benchmarks:
        delta = (b.mean_sentence_hallucination_rate - ASGARI_HALLUCINATION_BAR) * 100
        sign = "+" if delta >= 0 else ""
        beats = "✓" if b.beats_asgari_hallucination else "·"
        lines.append(
            f"| {b.model} | {b.total_sentences} | "
            f"{b.mean_sentence_hallucination_rate:.1%} | "
            f"[{b.ci_hallucination_lower:.1%}, {b.ci_hallucination_upper:.1%}] | "
            f"{b.mean_mhi_omission:.3f} | {b.mean_mhi_hallucination:.3f} | "
            f"{sign}{delta:.2f}pp | {beats} |"
        )
    return "\n".join(lines)


# ── Logging ───────────────────────────────────────────────────────────────────


def _log_summary(b: HallucinationBenchmarkResult) -> None:
    logger.info("\n" + "=" * 62)
    logger.info("HALLUCINATION EVALUATION SUMMARY")
    logger.info("=" * 62)
    logger.info(f"  Model:              {b.model}")
    logger.info(f"  Notes evaluated:    {b.num_notes}")
    logger.info(f"  Sentences checked:  {b.total_sentences}")
    logger.info(f"  Hallucination rate: {b.mean_sentence_hallucination_rate:.2%}"
                f"  (Asgari bar: {ASGARI_HALLUCINATION_BAR:.2%})")
    logger.info(f"  95% CI:             [{b.ci_hallucination_lower:.2%}, {b.ci_hallucination_upper:.2%}]")
    logger.info(f"  MHI omission:       {b.mean_mhi_omission:.3f}"
                f"  (Asgari bar: {ASGARI_OMISSION_BAR:.3f})")
    logger.info(f"  MHI hallucination:  {b.mean_mhi_hallucination:.3f}")
    logger.info(f"  Beats hall bar:     {'YES' if b.beats_asgari_hallucination else 'NO'}")
    logger.info(f"  Beats omit bar:     {'YES' if b.beats_asgari_omission else 'NO'}")
    if b.entailment_skipped:
        logger.warning("  Entailment gate was SKIPPED - hallucination rates are 0 (MHI only)")


# ── CLI ───────────────────────────────────────────────────────────────────────


def _dry_run() -> HallucinationBenchmarkResult:
    """Demo using the 8 GP scoring pack notes (no model required for MHI)."""
    from src.evaluation.constitutional_eval import DEMO_TRANSCRIPTS

    # Synthetic notes that mirror the GP scoring pack quality
    _DEMO_NOTES = {
        "Lower Back Pain": (
            "SUBJECTIVE:\nPatient reports 3-week lower back pain radiating to left leg "
            "after lifting boxes. Pins and needles in left foot. Paracetamol ineffective.\n\n"
            "OBJECTIVE:\nSLR positive left 40°. Power and reflexes normal. Temp 36.8.\n\n"
            "ASSESSMENT:\nLumbar disc protrusion with left sciatica [SNOMED: 161891005]\n\n"
            "PLAN:\n1. Naproxen 500mg BD with food\n2. Refer physiotherapy\n"
            "Safety-netting: A&E immediately for leg weakness or bladder/bowel symptoms, call 999."
        ),
        "Persistent Cough": (
            "SUBJECTIVE:\nSix-week cough, half-stone weight loss, 20/day smoker x25 years.\n\n"
            "OBJECTIVE:\nCrackles right base. Sats 96%. BP 138/85.\n\n"
            "ASSESSMENT:\nSuspicious cough - urgent 2WW [SNOMED: 49727002]\n\n"
            "PLAN:\n1. Urgent CXR\n2. 2-week wait referral\n3. FBC, CRP, LFTs\n"
            "Safety-netting: call 111 or A&E if haemoptysis or severe breathlessness."
        ),
        "Diabetes Review": (
            "SUBJECTIVE:\nAnnual review. Metformin 1g BD, atorvastatin 20mg nocte. GI upset.\n\n"
            "OBJECTIVE:\nHbA1c 58 (from 65). eGFR 85. BP 134/82. Cholesterol 4.2.\n\n"
            "ASSESSMENT:\nType 2 diabetes mellitus, well controlled [SNOMED: 44054006]\n\n"
            "PLAN:\n1. Continue medications\n2. Metformin with food\n3. Review 6 months\n"
            "Safety-netting: urgent appointment if BM consistently >12."
        ),
        "Mental Health - Anxiety": (
            "SUBJECTIVE:\nTwo months anxiety, panic attacks, insomnia. Low mood, no suicidal ideation.\n\n"
            "OBJECTIVE:\nAppropriately distressed. No psychosis.\n\n"
            "ASSESSMENT:\nGeneralised anxiety disorder [SNOMED: 197480006]\n\n"
            "PLAN:\n1. IAPT referral for CBT\n2. Review 4-6 weeks\n"
            "Safety-netting: call 111 or Samaritans 116123 or A&E if self-harm thoughts."
        ),
        "Chest Infection": (
            "SUBJECTIVE:\nFive-day cough, green sputum, fever. Asthmatic on salbutamol and preventer.\n\n"
            "OBJECTIVE:\nTemp 38.4. Sats 97%. Crackles and wheeze right lower zone.\n\n"
            "ASSESSMENT:\nLRTI [SNOMED: 10509002]\n\n"
            "PLAN:\n1. Amoxicillin 500mg TDS x5 days\n2. Continue salbutamol\n3. Paracetamol PRN\n"
            "Safety-netting: return in 1 week; call 111/A&E if more breathless or haemoptysis."
        ),
        "Hypertension Review": (
            "SUBJECTIVE:\nElevated BP. Family history of MI at 55.\n\n"
            "OBJECTIVE:\nBP 158/96. Ambulatory average 152/92.\n\n"
            "ASSESSMENT:\nHypertension [SNOMED: 38341003]\n\n"
            "PLAN:\n1. Ramipril 2.5mg OD\n2. Renal function in 2 weeks\n3. Salt reduction\n"
            "Safety-netting: call 999 for sudden severe headache, chest pain, or visual changes."
        ),
        "Paediatric Ear Infection": (
            "SUBJECTIVE:\nTwo-day right ear pain, temp up to 38.9.\n\n"
            "OBJECTIVE:\nTemp 38.2. Right TM red and bulging. Left normal.\n\n"
            "ASSESSMENT:\nAcute otitis media [SNOMED: 65363002]\n\n"
            "PLAN:\n1. Paracetamol and ibuprofen PRN\n2. Delayed amoxicillin if not improving in 3 days\n"
            "Safety-netting: bring back or call 111 for persistent fever, drowsiness, ear discharge."
        ),
        "Skin Lesion Assessment": (
            "SUBJECTIVE:\n8mm changing mole, itchy, multiple colours. Three months history.\n\n"
            "OBJECTIVE:\nAsymmetric, irregular borders, bluish area. ABCDE criteria met.\n\n"
            "ASSESSMENT:\nSuspicious pigmented lesion [SNOMED: 32343005]\n\n"
            "PLAN:\n1. Urgent 2WW dermatology referral for excision biopsy\n"
            "Safety-netting: call practice or attend A&E if bleeding or rapid change before appointment."
        ),
    }

    notes_input = [
        {
            "id": name.lower().replace(" ", "_").replace("-", "_"),
            "soap_note": note,
            "transcript": DEMO_TRANSCRIPTS[name],
        }
        for name, note in _DEMO_NOTES.items()
    ]

    logger.info("DRY RUN - entailment gate will attempt model load (may be skipped)")
    return run_hallucination_eval(
        notes_input,
        model="praxis-soap-v1 (dry-run)",
        use_entailment=True,
        save=True,
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="PRAXIS Hallucination Evaluation Harness")
    ap.add_argument("--dry-run", action="store_true", help="Use 8 demo GP notes (no Ollama)")
    ap.add_argument("--notes-dir", type=str, help="Directory of pipeline run JSONs")
    ap.add_argument("--model", default="praxis-soap-v1")
    ap.add_argument("--no-entailment", action="store_true", help="Skip entailment gate (MHI only)")
    args = ap.parse_args()

    if args.dry_run:
        result = _dry_run()
    elif args.notes_dir:
        notes_path = Path(args.notes_dir)
        notes = []
        for f in sorted(notes_path.glob("*/soap_note.json")):
            data = json.loads(f.read_text())
            notes.append(data)
        if not notes:
            raise SystemExit(f"No soap_note.json files found in {notes_path}")
        result = run_hallucination_eval(
            notes,
            model=args.model,
            use_entailment=not args.no_entailment,
        )
    else:
        raise SystemExit("Provide --dry-run or --notes-dir")

    print("\n" + to_markdown_table([result]))
    print("\n--- LaTeX ---")
    print(to_latex_table([result]))
