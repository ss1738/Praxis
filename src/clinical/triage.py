"""Clinical triage - risk-level assessment from note content.

Separate from note verification: this is a clinical feature, not a quality check.
"""

from __future__ import annotations

from dataclasses import dataclass, field


_RISK_CRITERIA: dict[str, dict] = {
    "CRITICAL": {
        "patterns": [
            "chest pain", "cardiac arrest", "myocardial infarction", "heart attack",
            "stroke", "cerebrovascular", "facial droop", "arm weakness",
            "anaphylaxis", "anaphylactic", "severe allergic reaction",
            "sepsis", "septic shock", "meningitis",
            "suicidal ideation", "suicidal thoughts", "self-harm",
        ],
        "timeframe": "Call 999 now - immediate emergency assessment required",
    },
    "HIGH": {
        "patterns": [
            "uncontrolled hypertension", "hypertensive crisis", "bp 180", "bp 200",
            "new neurological", "sudden weakness", "sudden numbness", "vision loss",
            "unexplained weight loss", "significant weight loss",
            "gastrointestinal bleeding", "haematemesis", "melaena", "rectal bleeding",
            "pulmonary embolism", "deep vein thrombosis", "dvt",
            "acute psychosis", "hallucinations",
        ],
        "timeframe": "Same-day GP review - urgent assessment within 24 hours",
    },
    "MEDIUM": {
        "patterns": [
            "chronic disease review", "medication change", "dose adjustment",
            "new symptoms", "worsening symptoms", "follow-up required",
            "blood test results", "abnormal results",
            "anxiety", "depression", "insomnia",
            "infection", "antibiotic", "uti",
        ],
        "timeframe": "Routine appointment - within 1-2 weeks",
    },
    "LOW": {
        "patterns": [
            "routine follow-up", "stable", "well-controlled",
            "annual review", "check-up", "minor complaint",
            "repeat prescription", "medication review",
            "vaccination", "immunisation",
        ],
        "timeframe": "Routine follow-up - standard appointment",
    },
}


@dataclass
class TriageResult:
    risk_level: str = "LOW"
    timeframe: str = ""
    matched_patterns: list[str] = field(default_factory=list)
    hallucinated_patterns: list[str] = field(default_factory=list)


def assess_risk(note_text: str, transcript: str = "") -> TriageResult:
    """Assess clinical risk level from note content: CRITICAL / HIGH / MEDIUM / LOW."""
    note_lower = note_text.lower()
    transcript_lower = transcript.lower() if transcript else ""

    risk_matches: dict[str, list[str]] = {}
    hallucinated: list[str] = []

    for level, criteria in _RISK_CRITERIA.items():
        matched = [p for p in criteria["patterns"] if p in note_lower]
        if matched:
            if transcript_lower:
                verified = [p for p in matched if p in transcript_lower]
                hallucinated.extend(p for p in matched if p not in transcript_lower)
                if verified:
                    risk_matches[level] = verified
            else:
                risk_matches[level] = matched

    detected_level = "LOW"
    all_matched: list[str] = []
    for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        if level in risk_matches:
            detected_level = level
            all_matched = risk_matches[level]
            break

    return TriageResult(
        risk_level=detected_level,
        timeframe=_RISK_CRITERIA[detected_level]["timeframe"],
        matched_patterns=all_matched,
        hallucinated_patterns=hallucinated,
    )
