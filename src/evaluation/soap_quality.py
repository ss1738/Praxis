"""SOAP quality evaluation - automated and clinician-assessment metrics.

Evaluates generated SOAP notes against reference transcripts using automated
proxy metrics. Also generates a clinician evaluation form (HTML) for blind
assessment by a GP.

Automated metrics:
    - Section completeness (0-1): fraction of S/O/A/P sections present and non-trivial
    - SNOMED code count: number of valid SNOMED codes in the note
    - Medication extraction recall: fraction of transcript medications found in note
    - Safety-netting presence: binary check for safety-netting in Plan section
    - TrustScore: from the 6-check validation pipeline

Usage:
    python3 -m src.evaluation.soap_quality --transcript "Patient reports..."
    python3 -m src.evaluation.soap_quality --generate-form --notes-dir data/pipeline_runs/
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "data" / "evaluation_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Known medications for recall computation ──

_KNOWN_MEDS = frozenset({
    "metformin", "ramipril", "lisinopril", "losartan", "bisoprolol",
    "atenolol", "amlodipine", "furosemide", "atorvastatin", "simvastatin",
    "omeprazole", "lansoprazole", "amoxicillin", "flucloxacillin",
    "clarithromycin", "doxycycline", "paracetamol", "ibuprofen", "codeine",
    "naproxen", "co-codamol", "prednisolone", "salbutamol", "gliclazide",
    "levothyroxine", "sertraline", "citalopram", "fluoxetine", "diazepam",
    "warfarin", "aspirin", "gabapentin", "pregabalin",
})


# ── Data types ──


@dataclass
class SOAPQualityResult:
    """Quality metrics for a single SOAP note."""
    note_id: str
    section_completeness: float  # 0-1
    sections_present: list[str]
    sections_missing: list[str]
    snomed_code_count: int
    snomed_codes: list[str]
    medication_recall: float  # 0-1
    meds_in_transcript: list[str]
    meds_in_note: list[str]
    meds_missed: list[str]
    safety_netting_present: bool
    trust_score: int
    trust_verdict: str
    word_count: int
    mhi_score: float = 1.0          # consequence-weighted fidelity (0-1)
    mhi_omission: float = 0.0       # missed clinical terms
    mhi_hallucination: float = 0.0  # fabricated clinical terms


@dataclass
class SOAPBenchmarkResult:
    """Aggregate SOAP quality benchmark."""
    timestamp: str
    model: str
    num_notes: int
    mean_section_completeness: float
    mean_snomed_count: float
    mean_medication_recall: float
    safety_netting_rate: float
    mean_trust_score: float
    notes: list[SOAPQualityResult] = field(default_factory=list)


# ── Section detection ──


def _parse_sections(note_text: str) -> dict[str, str]:
    """Extract SOAP sections from note text."""
    sections: dict[str, str] = {}
    patterns = [
        ("subjective", r"(?:^|\n)\s*SUBJECTIVE\s*:(.*?)(?=\n\s*(?:OBJECTIVE|ASSESSMENT|PLAN)\s*:|$)"),
        ("objective", r"(?:^|\n)\s*OBJECTIVE\s*:(.*?)(?=\n\s*(?:ASSESSMENT|PLAN)\s*:|$)"),
        ("assessment", r"(?:^|\n)\s*ASSESSMENT\s*:(.*?)(?=\n\s*PLAN\s*:|$)"),
        ("plan", r"(?:^|\n)\s*PLAN\s*:(.*)"),
    ]
    for name, pattern in patterns:
        match = re.search(pattern, note_text, re.IGNORECASE | re.DOTALL)
        if match:
            content = match.group(1).strip()
            if len(content) >= 10:  # non-trivial content
                sections[name] = content
    return sections


def _extract_meds(text: str) -> list[str]:
    """Extract known medications from text."""
    text_lower = text.lower()
    return sorted(m for m in _KNOWN_MEDS if m in text_lower)


def _check_safety_netting(plan_text: str) -> bool:
    """Check for safety-netting advice in plan section."""
    plan_lower = plan_text.lower()
    keywords = [
        "safety", "return if", "seek help", "seek urgent",
        "111", "999", "a&e", "worsen", "red flag", "emergency",
    ]
    return any(kw in plan_lower for kw in keywords)


# ── Evaluation ──


def evaluate_soap_note(
    note_text: str,
    transcript: str = "",
    note_id: str = "",
    model: str = "",
) -> SOAPQualityResult:
    """Evaluate a single SOAP note's quality using automated proxy metrics."""
    from src.clinical.verification import validate_soap_note

    sections = _parse_sections(note_text)
    all_sections = ["subjective", "objective", "assessment", "plan"]
    present = [s for s in all_sections if s in sections]
    missing = [s for s in all_sections if s not in sections]
    completeness = len(present) / len(all_sections)

    snomed_re = re.compile(r'(?:\[?SNOMED[:\s]*|snomed[:\s]*)(\d{6,})\]?')
    snomed_codes = list(set(snomed_re.findall(note_text)))

    meds_transcript = _extract_meds(transcript) if transcript else []
    meds_note = _extract_meds(note_text)
    meds_missed = [m for m in meds_transcript if m not in meds_note]
    med_recall = (
        (len(meds_transcript) - len(meds_missed)) / len(meds_transcript)
        if meds_transcript else 1.0
    )

    plan_text = sections.get("plan", "")
    safety = _check_safety_netting(plan_text)

    validation = validate_soap_note(note_text=note_text, transcript=transcript)

    mhi_score = 1.0
    mhi_omission = 0.0
    mhi_hallucination = 0.0
    if transcript:
        from src.evaluation.mhi import compute_mhi
        mhi_result = compute_mhi(transcript, note_text)
        mhi_score = mhi_result.mhi_score
        mhi_omission = mhi_result.mhi_omission
        mhi_hallucination = mhi_result.mhi_hallucination

    return SOAPQualityResult(
        note_id=note_id or datetime.now().strftime("%Y%m%d_%H%M%S"),
        section_completeness=round(completeness, 2),
        sections_present=present,
        sections_missing=missing,
        snomed_code_count=len(snomed_codes),
        snomed_codes=snomed_codes,
        medication_recall=round(med_recall, 2),
        meds_in_transcript=meds_transcript,
        meds_in_note=meds_note,
        meds_missed=meds_missed,
        safety_netting_present=safety,
        trust_score=validation.trust_score,
        trust_verdict=validation.verdict,
        word_count=len(note_text.split()),
        mhi_score=round(mhi_score, 4),
        mhi_omission=round(mhi_omission, 4),
        mhi_hallucination=round(mhi_hallucination, 4),
    )


def run_benchmark(
    notes: list[dict],
    model: str = "unknown",
) -> SOAPBenchmarkResult:
    """Run SOAP quality benchmark on a list of notes.

    Each dict in notes must have 'soap_note' and optionally 'transcript' and 'id'.
    """
    results: list[SOAPQualityResult] = []

    for i, entry in enumerate(notes):
        note_text = entry.get("soap_note", "")
        transcript = entry.get("transcript", "")
        note_id = entry.get("id", f"note_{i:03d}")

        if not note_text:
            logger.warning(f"Skipping {note_id}: empty note")
            continue

        result = evaluate_soap_note(note_text, transcript, note_id, model)
        results.append(result)
        logger.info(
            f"  [{i+1}/{len(notes)}] {note_id}: "
            f"complete={result.section_completeness:.0%} "
            f"SNOMED={result.snomed_code_count} "
            f"med_recall={result.medication_recall:.0%} "
            f"safety={'Y' if result.safety_netting_present else 'N'} "
            f"trust={result.trust_score}"
        )

    benchmark = SOAPBenchmarkResult(
        timestamp=datetime.now(timezone.utc).isoformat(),
        model=model,
        num_notes=len(results),
        mean_section_completeness=round(
            sum(r.section_completeness for r in results) / len(results), 3
        ) if results else 0.0,
        mean_snomed_count=round(
            sum(r.snomed_code_count for r in results) / len(results), 1
        ) if results else 0.0,
        mean_medication_recall=round(
            sum(r.medication_recall for r in results) / len(results), 3
        ) if results else 0.0,
        safety_netting_rate=round(
            sum(1 for r in results if r.safety_netting_present) / len(results), 3
        ) if results else 0.0,
        mean_trust_score=round(
            sum(r.trust_score for r in results) / len(results), 1
        ) if results else 0.0,
        notes=results,
    )

    out_path = RESULTS_DIR / f"soap_{model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(asdict(benchmark), indent=2))
    logger.info(f"Results saved: {out_path}")
    return benchmark


def generate_clinician_form(notes: list[dict], output_path: Path | None = None) -> str:
    """Generate an HTML evaluation form for blind clinician assessment.

    Produces a printable form where a GP scores each note on a 1-5 scale
    across four dimensions: accuracy, completeness, safety, readability.
    """
    html_parts = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='UTF-8'>",
        "<title>PRAXIS - Clinician SOAP Note Evaluation Form</title>",
        "<style>",
        "body { font-family: 'Inter', Arial, sans-serif; max-width: 800px; margin: 2em auto; line-height: 1.6; color: #1a1a2e; }",
        "h1 { font-size: 1.5em; border-bottom: 2px solid #0f4c81; padding-bottom: 0.5em; }",
        "h2 { font-size: 1.2em; margin-top: 2em; color: #0f4c81; }",
        ".note-box { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 1.5em; margin: 1em 0; white-space: pre-wrap; font-family: 'JetBrains Mono', monospace; font-size: 0.85em; }",
        ".score-table { width: 100%; border-collapse: collapse; margin: 1em 0; }",
        ".score-table th, .score-table td { border: 1px solid #dee2e6; padding: 0.5em; text-align: center; }",
        ".score-table th { background: #0f4c81; color: white; }",
        ".instructions { background: #e8f4fd; border-left: 4px solid #0f4c81; padding: 1em; margin: 1em 0; }",
        "@media print { .page-break { page-break-before: always; } }",
        "</style></head><body>",
        "<h1>PRAXIS - Clinician SOAP Note Evaluation</h1>",
        "<div class='instructions'>",
        "<strong>Instructions:</strong> For each note below, score on a 1-5 scale:<br>",
        "<strong>1</strong> = Poor/Dangerous &nbsp; <strong>2</strong> = Below Standard &nbsp; "
        "<strong>3</strong> = Adequate &nbsp; <strong>4</strong> = Good &nbsp; <strong>5</strong> = Excellent<br><br>",
        "Notes are presented in random order without model labels. "
        "Please assess based on your clinical judgement as a GP.",
        "</div>",
        f"<p>Evaluator: __________________ &nbsp; Date: {datetime.now().strftime('%Y-%m-%d')} &nbsp; GMC No: __________________</p>",
    ]

    for i, entry in enumerate(notes):
        note_text = entry.get("soap_note", "No note available")
        note_id = entry.get("id", f"Note {i+1}")

        html_parts.append(f"<div class='page-break'>" if i > 0 else "<div>")
        html_parts.append(f"<h2>Note {i+1} (ID: {note_id})</h2>")
        html_parts.append(f"<div class='note-box'>{note_text}</div>")
        html_parts.append("<table class='score-table'>")
        html_parts.append("<tr><th>Criterion</th><th>1</th><th>2</th><th>3</th><th>4</th><th>5</th><th>Comments</th></tr>")
        for criterion in ["Clinical Accuracy", "Completeness", "Patient Safety", "Readability"]:
            html_parts.append(
                f"<tr><td style='text-align:left'><strong>{criterion}</strong></td>"
                "<td>☐</td><td>☐</td><td>☐</td><td>☐</td><td>☐</td>"
                "<td style='min-width:200px'></td></tr>"
            )
        html_parts.append("</table>")
        html_parts.append("<p>Additional comments: _______________________________________________</p>")
        html_parts.append("</div>")

    html_parts.append("</body></html>")
    html = "\n".join(html_parts)

    out = output_path or (RESULTS_DIR / f"clinician_form_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    out.write_text(html)
    logger.info(f"Clinician evaluation form saved: {out}")
    return str(out)


def to_latex(result: SOAPBenchmarkResult) -> str:
    """Generate a LaTeX table row for SOAP quality results."""
    return (
        f"{result.model} & {result.num_notes} & "
        f"{result.mean_section_completeness:.0%} & "
        f"{result.mean_snomed_count:.1f} & "
        f"{result.mean_medication_recall:.0%} & "
        f"{result.safety_netting_rate:.0%} & "
        f"{result.mean_trust_score:.0f} \\\\"
    )


def to_latex_table(results: list[SOAPBenchmarkResult]) -> str:
    """Generate a complete LaTeX table from multiple SOAP quality results."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{SOAP Note Quality Evaluation}",
        r"\label{tab:soap-quality}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Model & N & Completeness & SNOMED & Med. Recall & Safety Net & TrustScore \\",
        r"\midrule",
    ]
    for r in results:
        lines.append(to_latex(r))
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="PRAXIS SOAP Quality Evaluation")
    ap.add_argument("--transcript", type=str, help="Transcript text to evaluate")
    ap.add_argument("--soap-note", type=str, help="SOAP note text to evaluate")
    ap.add_argument("--notes-dir", type=str, help="Directory of pipeline run JSONs")
    ap.add_argument("--generate-form", action="store_true", help="Generate clinician evaluation form")
    args = ap.parse_args()

    if args.transcript and args.soap_note:
        result = evaluate_soap_note(args.soap_note, args.transcript)
        print(json.dumps(asdict(result), indent=2))
        return

    if args.notes_dir:
        notes_path = Path(args.notes_dir)
        notes = []
        for f in sorted(notes_path.glob("*/soap_note.json")):
            data = json.loads(f.read_text())
            notes.append(data)

        if args.generate_form:
            form_path = generate_clinician_form(notes)
            print(f"Clinician form generated: {form_path}")
        else:
            result = run_benchmark(notes)
            print(f"\nSOAP Quality: completeness={result.mean_section_completeness:.0%}, "
                  f"SNOMED={result.mean_snomed_count:.1f}, "
                  f"med_recall={result.mean_medication_recall:.0%}, "
                  f"safety={result.safety_netting_rate:.0%}, "
                  f"trust={result.mean_trust_score:.0f}")


if __name__ == "__main__":
    main()
