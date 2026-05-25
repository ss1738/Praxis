"""Clinical quality evaluator for clinical notes.

Every note is scored on objective criteria. No hand-waving.
Supports SOAP, SBAR, and other NHS/GP note formats.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class QualityScore:
    """Quality assessment of a single clinical note."""
    has_subjective: bool = False
    has_objective: bool = False
    has_assessment: bool = False
    has_plan: bool = False

    has_chief_complaint: bool = False
    has_history: bool = False
    has_medications: bool = False
    has_examination_findings: bool = False
    has_diagnosis: bool = False
    has_treatment_plan: bool = False
    has_safety_netting: bool = False
    has_follow_up: bool = False

    has_snomed_codes: bool = False
    snomed_code_count: int = 0

    has_hallucinated_content: bool = False
    uses_uk_english: bool = False

    template_used: str = "soap"
    model_used: str = ""
    generation_time_secs: float = 0.0
    token_count: int = 0

    @property
    def structure_score(self) -> float:
        checks = [self.has_subjective, self.has_objective, self.has_assessment, self.has_plan]
        return sum(checks) / len(checks)

    @property
    def content_score(self) -> float:
        checks = [
            self.has_chief_complaint, self.has_history, self.has_medications,
            self.has_examination_findings, self.has_diagnosis, self.has_treatment_plan,
            self.has_safety_netting, self.has_follow_up,
        ]
        return sum(checks) / len(checks)

    @property
    def overall_score(self) -> float:
        coding = 1.0 if self.has_snomed_codes else 0.0
        safety = 0.0 if self.has_hallucinated_content else 1.0
        return (
            0.15 * self.structure_score
            + 0.35 * self.content_score
            + 0.10 * coding
            + 0.40 * safety
        )

    def to_dict(self) -> dict:
        return {
            "structure_score": round(self.structure_score, 2),
            "content_score": round(self.content_score, 2),
            "overall_score": round(self.overall_score, 2),
            "has_snomed_codes": self.has_snomed_codes,
            "snomed_code_count": self.snomed_code_count,
            "has_safety_netting": self.has_safety_netting,
            "uses_uk_english": self.uses_uk_english,
            "template_used": self.template_used,
            "model_used": self.model_used,
            "generation_time_secs": self.generation_time_secs,
            "token_count": self.token_count,
        }


# Template-specific structure detection patterns
STRUCTURE_PATTERNS = {
    "soap": {
        "section_1": ["SUBJECTIVE", "\nS:"],
        "section_2": ["OBJECTIVE", "\nO:"],
        "section_3": ["ASSESSMENT", "\nA:"],
        "section_4": ["PLAN", "\nP:"],
    },
    "sbar": {
        "section_1": ["SITUATION"],
        "section_2": ["BACKGROUND"],
        "section_3": ["ASSESSMENT"],
        "section_4": ["RECOMMENDATION"],
    },
    "clinical_summary": {
        "section_1": ["PRESENTING COMPLAINT", "PRESENTING PROBLEM"],
        "section_2": ["HISTORY", "HISTORY OF PRESENTING"],
        "section_3": ["DIAGNOSIS", "ASSESSMENT"],
        "section_4": ["MANAGEMENT", "MANAGEMENT PLAN", "SAFETY-NETTING", "SAFETY NETTING"],
    },
    "referral_letter": {
        "section_1": ["DEAR COLLEAGUE", "DEAR DR", "RE:"],
        "section_2": ["HISTORY", "CLINICAL DETAILS"],
        "section_3": ["EXAMINATION", "INVESTIGATION"],
        "section_4": ["YOURS FAITHFULLY", "YOURS SINCERELY", "GRATEFUL", "REQUEST"],
    },
    "discharge_summary": {
        "section_1": ["DIAGNOSIS", "DISCHARGE"],
        "section_2": ["TREATMENT", "MEDICATION"],
        "section_3": ["FOLLOW-UP", "FOLLOW UP"],
        "section_4": ["RED FLAG", "WHEN TO SEEK", "SAFETY", "URGENT"],
    },
    "emis_consultation": {
        "section_1": ["PROBLEM"],
        "section_2": ["HISTORY"],
        "section_3": ["EXAMINATION"],
        "section_4": ["MEDICATION ISSUED", "FOLLOW-UP", "FOLLOW UP", "COMMENT"],
    },
}


class ClinicalEvaluator:
    """Automated quality evaluation for clinical notes."""

    UK_TERMS = [
        "paracetamol", "adrenaline", "salbutamol", "ramipril", "metformin",
        "atorvastatin", "omeprazole", "amoxicillin", "co-codamol", "diclofenac",
        "physiotherapy", "a&e", "gp", "111", "999",
        "mmol", "hba1c", "esr", "crp",
    ]

    US_TERMS = [
        "acetaminophen", "epinephrine", "albuterol", "tylenol", "advil",
        "emergency room", "primary care physician", "mg/dl",
    ]

    def evaluate(self, note_text: str, model: str = "", time_secs: float = 0.0,
                 tokens: int = 0, template: str = "soap") -> QualityScore:
        score = QualityScore()
        score.model_used = model
        score.generation_time_secs = time_secs
        score.token_count = tokens
        score.template_used = template

        upper = note_text.upper()
        lower = note_text.lower()

        # Structure detection - template-aware
        patterns = STRUCTURE_PATTERNS.get(template, STRUCTURE_PATTERNS["soap"])
        score.has_subjective = any(p in upper or p in note_text for p in patterns["section_1"])
        score.has_objective = any(p in upper or p in note_text for p in patterns["section_2"])
        score.has_assessment = any(p in upper or p in note_text for p in patterns["section_3"])
        score.has_plan = any(p in upper or p in note_text for p in patterns["section_4"])

        # Content detection (universal across all templates)
        score.has_chief_complaint = any(t in lower for t in [
            "chief complaint", "presenting complaint", "reason for visit", "complain",
            "presenting problem", "situation", "reports", "presents with"])
        score.has_history = any(t in lower for t in [
            "history", "hpi", "presenting illness", "duration", "onset",
            "background", "pmh", "past medical", "pack-year", "smoking"])
        score.has_medications = any(t in lower for t in [
            "medication", "drug", "prescription", "mg ", "tablet", "daily", "twice daily",
            "tds", "bd", "od", "capsule", "inhaler", "prescribe", "issued"])
        score.has_examination_findings = any(t in lower for t in [
            "examination", "findings", "auscultation", "palpation", "inspection",
            "observation", "temperature", "blood pressure", "bp ", "spo2",
            "sats", "pulse", "heart rate", "bmi"])
        score.has_diagnosis = any(t in lower for t in [
            "diagnosis", "assessment", "impression", "consistent with", "likely",
            "clinical picture", "differential"])
        score.has_treatment_plan = any(t in lower for t in [
            "prescribe", "treatment", "refer", "advise", "recommend", "start",
            "increase", "continue", "management", "plan", "issued"])
        score.has_safety_netting = any(t in lower for t in [
            "safety", "worsen", "urgent", "emergency", "return if", "come back",
            "seek", "111", "999", "a&e", "red flag", "when to seek"])
        score.has_follow_up = any(t in lower for t in [
            "follow-up", "follow up", "review", "weeks", "months", "appointment",
            "return", "recheck"])

        # SNOMED code detection - look for both [SNOMED: 123456] and SNOMED: 123456 formats
        snomed_matches = re.findall(r'(?:SNOMED|snomed)[:\s]*(\d{6,})', note_text)
        # Also catch [SNOMED: 123456789] bracket format
        bracket_matches = re.findall(r'\[SNOMED:\s*(\d{6,})\]', note_text)
        all_codes = set(snomed_matches + bracket_matches)
        score.has_snomed_codes = len(all_codes) > 0
        score.snomed_code_count = len(all_codes)

        # UK English detection
        uk_count = sum(1 for t in self.UK_TERMS if t in lower)
        us_count = sum(1 for t in self.US_TERMS if t in lower)
        score.uses_uk_english = uk_count > 0 and us_count == 0

        return score
