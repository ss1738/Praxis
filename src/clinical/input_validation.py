"""Pre-pipeline input validation - gates non-medical audio before it wastes compute."""

from __future__ import annotations

from dataclasses import dataclass, field


_MEDICAL_KEYWORDS = frozenset({
    "pain", "cough", "fever", "headache", "nausea", "breathless",
    "medication", "prescribed", "diagnosis", "symptoms", "examination",
    "blood pressure", "diabetes", "asthma", "infection", "antibiotic",
    "doctor", "patient", "gp", "hospital", "appointment", "refer",
})


@dataclass
class InputValidation:
    valid: bool = True
    issues: list[str] = field(default_factory=list)
    medical_keyword_count: int = 0
    word_count: int = 0


def validate_transcript(transcript: str) -> InputValidation:
    """Check that a transcript contains genuine medical content.

    Run this BEFORE the pipeline to reject non-medical audio early.
    """
    issues: list[str] = []
    word_count = len(transcript.split())

    if word_count < 20:
        issues.append(f"Transcript too short for clinical analysis ({word_count} words)")

    medical_hits = 0
    if transcript:
        transcript_lower = transcript.lower()
        medical_hits = sum(1 for kw in _MEDICAL_KEYWORDS if kw in transcript_lower)
        if medical_hits == 0 and word_count >= 20:
            issues.append("No medical keywords detected - content may not be a medical consultation")

    return InputValidation(
        valid=len(issues) == 0,
        issues=issues,
        medical_keyword_count=medical_hits,
        word_count=word_count,
    )
