"""Post-generation clinical note validation - 6 checks + deduction scoring.

Each check is a plain function that inspects the parsed SOAP note and returns
issues/suggestions. No classes, no "agents" - these are deterministic rule-based
validators. The scoring function aggregates results into a single TrustScore.

Checks:
    1. Transcript fidelity  - hallucination detection
    2. SNOMED validation    - code verification
    3. Clinical safety      - red flag coverage + safety-netting
    4. NICE guidelines      - guideline compliance
    5. Documentation completeness - SOAP structure
    6. Drug interactions    - BNF interaction warnings + missing doses
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from loguru import logger

from src.clinical.drug_interactions import check_interactions
from src.clinical.nice_guidelines import check_nice_guidelines


# ── Data types ──


@dataclass
class CheckResult:
    """Result from a single validation check."""
    check_name: str
    passed: bool
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Final TrustScore from the validation pipeline."""
    trust_score: int = 0
    verdict: str = "needs_review"
    checks_passed: int = 0
    checks_total: int = 6
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    check_results: list[CheckResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "trust_score": self.trust_score,
            "verdict": self.verdict,
            "checks_passed": self.checks_passed,
            "checks_total": self.checks_total,
            "issues": self.issues,
            "suggestions": self.suggestions,
            "check_results": [
                {
                    "check": r.check_name,
                    "passed": r.passed,
                    "issues": r.issues,
                    "suggestions": r.suggestions,
                }
                for r in self.check_results
            ],
        }


@dataclass
class ClinicalNote:
    """Parsed clinical note for validation."""
    transcript: str = ""
    chief_complaint: str = ""
    history_of_presenting_illness: str = ""
    examination_findings: str = ""
    assessment: str = ""
    plan: str = ""
    raw_text: str = ""


# ── Known medications for extraction ──

_KNOWN_MEDS = frozenset({
    "metformin", "ramipril", "lisinopril", "losartan", "candesartan", "bisoprolol",
    "atenolol", "amlodipine", "indapamide", "bendroflumethiazide", "furosemide",
    "spironolactone", "digoxin", "atorvastatin", "simvastatin", "rosuvastatin",
    "ezetimibe", "omeprazole", "lansoprazole", "pantoprazole", "amoxicillin",
    "flucloxacillin", "clarithromycin", "doxycycline", "trimethoprim",
    "nitrofurantoin", "co-amoxiclav", "paracetamol", "ibuprofen", "codeine",
    "naproxen", "co-codamol", "tramadol", "prednisolone", "salbutamol",
    "beclometasone", "fluticasone", "montelukast", "tiotropium", "gliclazide",
    "sitagliptin", "empagliflozin", "levothyroxine", "sertraline", "citalopram",
    "fluoxetine", "mirtazapine", "amitriptyline", "diazepam", "zopiclone",
    "warfarin", "apixaban", "rivaroxaban", "edoxaban", "dabigatran", "aspirin",
    "clopidogrel", "gabapentin", "pregabalin", "carbamazepine", "phenytoin",
    "lamotrigine", "valproate", "levetiracetam", "lithium", "methotrexate",
    "azathioprine", "allopurinol", "colchicine", "morphine",
})


def _extract_medications(text: str) -> list[str]:
    text_lower = text.lower()
    return [m for m in _KNOWN_MEDS if m in text_lower]


# ── 6 validation checks ──


def check_transcript_fidelity(note: ClinicalNote) -> CheckResult:
    """Check that the note faithfully represents the transcript (hallucination detection)."""
    issues: list[str] = []
    suggestions: list[str] = []

    if not note.transcript:
        return CheckResult(
            check_name="transcript_fidelity", passed=True,
            suggestions=["No transcript provided - fidelity check skipped"],
        )

    transcript_lower = note.transcript.lower()
    note_text = f"{note.assessment} {note.plan}".lower()

    note_meds = _extract_medications(note_text)
    for med in note_meds:
        if med not in transcript_lower:
            issues.append(f"Medication '{med}' in note but not in transcript - possible fabrication")

    assessment_terms = re.findall(r'[A-Z][\w\s]{2,25}(?:\s*\[SNOMED)', note.assessment)
    for term in assessment_terms:
        term_clean = term.replace("[SNOMED", "").strip().lower()
        words = [w for w in term_clean.split() if len(w) >= 4]
        if words and not any(w in transcript_lower for w in words):
            issues.append(f"Assessment term '{term_clean}' not found in transcript")

    passed = len(issues) == 0
    if not passed:
        suggestions.append("Review flagged terms against original transcript")

    return CheckResult(
        check_name="transcript_fidelity", passed=passed,
        issues=issues, suggestions=suggestions,
    )


def check_snomed_codes(note: ClinicalNote) -> CheckResult:
    """Validate SNOMED CT codes in the clinical note."""
    _VALID_CODES: dict[str, str] = {
        "267036007": "Dyspnoea", "49727002": "Cough", "25064002": "Headache",
        "161891005": "Backache", "29857009": "Chest pain", "386661006": "Fever",
        "21522001": "Abdominal pain", "422587007": "Nausea",
        "44054006": "Type 2 diabetes mellitus", "38341003": "Hypertensive disorder",
        "195967001": "Asthma", "13645005": "COPD", "35489007": "Depressive disorder",
        "197480006": "Anxiety disorder", "65363002": "Otitis media",
        "36971009": "Sinusitis", "10509002": "Acute bronchitis",
        "233604007": "Pneumonia", "126851005": "Urinary tract infection",
        "449868002": "Acute tonsillitis", "387517004": "Paracetamol",
        "372756006": "Amoxicillin", "387207008": "Ibuprofen",
        "373254001": "Naproxen", "109081006": "Metformin",
        "108600003": "Atorvastatin", "372913009": "Ramipril",
        "372897005": "Salbutamol", "372767007": "Omeprazole",
        "373225007": "Prednisolone", "1003403007": "Sertraline",
    }

    issues: list[str] = []
    suggestions: list[str] = []

    snomed_re = re.compile(r'(?:\[?SNOMED[:\s]*|snomed[:\s]*)(\d{6,})\]?')
    found_codes = set(snomed_re.findall(note.raw_text))

    verified = 0
    for code in found_codes:
        if code in _VALID_CODES:
            verified += 1
        else:
            issues.append(f"SNOMED code {code} not in verified database - requires review")

    if not found_codes and note.raw_text:
        suggestions.append("No SNOMED codes found in note - consider adding coded diagnoses")

    if verified > 0:
        suggestions.append(f"{verified} SNOMED codes verified against local database")

    return CheckResult(
        check_name="snomed_validation", passed=len(issues) == 0,
        issues=issues, suggestions=suggestions,
    )


def check_clinical_safety(note: ClinicalNote) -> CheckResult:
    """Check for clinical safety issues - red flags and safety-netting."""
    _RED_FLAG_PATTERNS = [
        ("chest pain", ["ecg", "troponin", "aspirin"]),
        ("headache", ["red flag", "neuroimag", "ct head"]),
        ("suicidal", ["safety", "risk assessment", "crisis"]),
        ("melaena", ["urgent", "refer", "endoscop"]),
        ("haemoptysis", ["chest x-ray", "refer", "urgent"]),
    ]

    issues: list[str] = []
    suggestions: list[str] = []

    note_text = f"{note.chief_complaint} {note.assessment} {note.plan}".lower()

    for symptom, required_actions in _RED_FLAG_PATTERNS:
        if symptom in note_text:
            missing = [a for a in required_actions if a not in note_text]
            if missing:
                issues.append(
                    f"Red flag '{symptom}' detected but missing: {', '.join(missing)}"
                )

    plan_lower = note.plan.lower()
    safety_keywords = [
        "safety", "return if", "seek help", "seek urgent",
        "111", "999", "a&e", "worsen", "red flag",
    ]
    if note.plan and not any(kw in plan_lower for kw in safety_keywords):
        issues.append("No safety-netting advice in plan")
        suggestions.append("Add safety-netting: when to return, when to call 111/999")

    full_text = f"{note.transcript} {note.raw_text}".lower()
    allergy_terms = ["allergy", "allergies", "allergic", "no known allergies", "nkda"]
    if not any(t in full_text for t in allergy_terms):
        suggestions.append("No allergy information documented")

    return CheckResult(
        check_name="clinical_safety", passed=len(issues) == 0,
        issues=issues, suggestions=suggestions,
    )


def check_nice_compliance(note: ClinicalNote) -> CheckResult:
    """Check alignment with NICE clinical guidelines."""
    issues: list[str] = []
    suggestions: list[str] = []

    results = check_nice_guidelines(
        chief_complaint=note.chief_complaint,
        hpi=note.history_of_presenting_illness,
        assessment=note.assessment,
        plan=note.plan,
    )

    for result in results:
        for req in result["not_met"]:
            issues.append(f"{result['nice_ref']}: {req} - NOT MET")
        for req in result["met"]:
            suggestions.append(f"{result['nice_ref']}: {req} - met")
        for rf in result["red_flags"]:
            issues.append(f"{result['nice_ref']} red flag: {rf}")

    if not results:
        suggestions.append("No NICE guidelines matched the conditions in this note")

    return CheckResult(
        check_name="nice_guidelines", passed=len(issues) == 0,
        issues=issues, suggestions=suggestions,
    )


def check_completeness(note: ClinicalNote) -> CheckResult:
    """Ensure all required SOAP sections are present and non-trivial."""
    issues: list[str] = []
    suggestions: list[str] = []

    required = {
        "chief_complaint": note.chief_complaint,
        "assessment": note.assessment,
        "plan": note.plan,
    }
    recommended = {
        "history_of_presenting_illness": note.history_of_presenting_illness,
        "examination_findings": note.examination_findings,
    }

    for field_name, value in required.items():
        if not value or len(value.strip()) < 5:
            issues.append(f"Required section '{field_name}' is missing or too short")

    for field_name, value in recommended.items():
        if not value or len(value.strip()) < 5:
            suggestions.append(f"Recommended section '{field_name}' is missing")

    if note.plan and len(note.plan.split()) < 10:
        issues.append("Plan section appears too brief for a clinical encounter")

    return CheckResult(
        check_name="completeness", passed=len(issues) == 0,
        issues=issues, suggestions=suggestions,
    )


def check_drug_interactions(note: ClinicalNote) -> CheckResult:
    """Check for dangerous drug interactions and missing dosing."""
    issues: list[str] = []
    suggestions: list[str] = []

    full_text = f"{note.transcript} {note.raw_text}".lower()
    found_meds = _extract_medications(full_text)

    if len(found_meds) >= 2:
        interactions = check_interactions(found_meds)
        for ix in interactions:
            severity = ix["severity"]
            msg = (
                f"{ix['drug_a']} + {ix['drug_b']} ({severity}): "
                f"{ix['effect']} - {ix['clinical_action']}"
            )
            if severity in ("critical", "high"):
                issues.append(msg)
            else:
                suggestions.append(msg)

    plan_lower = note.plan.lower()
    dose_re = re.compile(r'\d+\s*(?:mg|mcg|ml|units?|g)\b')
    plan_meds = _extract_medications(plan_lower)
    for med in plan_meds:
        med_idx = plan_lower.find(med)
        if med_idx >= 0:
            context = plan_lower[med_idx:med_idx + 50]
            if not dose_re.search(context):
                suggestions.append(f"Medication '{med}' in plan without documented dose")

    if not found_meds:
        suggestions.append("No medications detected in note")

    passed = not any("critical" in i.lower() or "high" in i.lower() for i in issues)
    return CheckResult(
        check_name="drug_interactions", passed=passed,
        issues=issues, suggestions=suggestions,
    )


# ── Scoring ──


_DEDUCTION_TABLE: dict[str, tuple[int, int]] = {
    "clinical_safety": (15, 35),
    "transcript_fidelity": (10, 30),
    "snomed_validation": (5, 15),
    "nice_guidelines": (5, 20),
    "completeness": (5, 15),
    "drug_interactions": (5, 20),
    "entailment_gate": (10, 25),
}

_ALL_CHECKS = [
    check_transcript_fidelity,
    check_snomed_codes,
    check_clinical_safety,
    check_nice_compliance,
    check_completeness,
    check_drug_interactions,
]


def validate_soap_note(
    note_text: str,
    transcript: str = "",
    use_entailment_gate: bool = False,
) -> ValidationResult:
    """Post-generation clinical validation. Returns score 0-100 + issues.

    Runs all 6 deterministic checks plus an optional 7th NLI entailment gate.
    Set use_entailment_gate=True to activate per-sentence hallucination scoring
    (requires cross-encoder/nli-deberta-v3-small; skipped gracefully if unavailable).
    """
    sections = _parse_soap_sections(note_text)

    note = ClinicalNote(
        transcript=transcript,
        chief_complaint=sections.get("subjective", ""),
        history_of_presenting_illness=sections.get("subjective", ""),
        examination_findings=sections.get("objective", ""),
        assessment=sections.get("assessment", ""),
        plan=sections.get("plan", ""),
        raw_text=note_text,
    )

    results: list[CheckResult] = []
    for check_fn in _ALL_CHECKS:
        try:
            result = check_fn(note)
            results.append(result)
        except Exception as e:
            logger.warning(f"Check {check_fn.__name__} failed: {e}")
            results.append(CheckResult(
                check_name=check_fn.__name__.removeprefix("check_"),
                passed=False,
                issues=[f"Check error: {str(e)[:200]}"],
            ))

    # Optional check 7: NLI entailment gate
    if use_entailment_gate and transcript:
        try:
            from src.clinical.entailment_gate import EntailmentGate
            gate = EntailmentGate()
            gate_result = gate.check(note_text, transcript)
            if gate_result.skipped:
                logger.debug(f"Entailment gate skipped: {gate_result.skip_reason}")
            else:
                issues_eg: list[str] = []
                suggestions_eg: list[str] = []
                for s in gate_result.flagged[:5]:
                    issues_eg.append(
                        f"Unsupported sentence (score={s.entailment_score:.2f}): "
                        f"{s.sentence[:100]}"
                    )
                if gate_result.hallucination_rate > 0:
                    suggestions_eg.append(
                        f"Entailment gate: {gate_result.num_flagged}/{gate_result.num_checked} "
                        f"sentences unsupported ({gate_result.hallucination_rate:.1%})"
                    )
                results.append(CheckResult(
                    check_name="entailment_gate",
                    passed=gate_result.num_flagged == 0,
                    issues=issues_eg,
                    suggestions=suggestions_eg,
                ))
        except Exception as e:
            logger.warning(f"Entailment gate check failed: {e}")

    score = 100
    all_issues: list[str] = []
    all_suggestions: list[str] = []

    for result in results:
        all_suggestions.extend(result.suggestions)
        if not result.passed:
            min_ded, max_ded = _DEDUCTION_TABLE.get(result.check_name, (5, 15))
            n_issues = len(result.issues)
            deduction = min(max_ded, min_ded + (n_issues - 1) * 5)
            score -= deduction
            all_issues.extend(result.issues)

    score = max(0, min(100, score))
    checks_passed = sum(1 for r in results if r.passed)

    if score >= 70:
        verdict_str = "approved"
    elif score >= 40:
        verdict_str = "needs_review"
    else:
        verdict_str = "rejected"

    validation = ValidationResult(
        trust_score=score,
        verdict=verdict_str,
        checks_passed=checks_passed,
        checks_total=6,  # canonical count; entailment gate is supplementary
        issues=all_issues[:10],
        suggestions=all_suggestions[:10],
        check_results=results,
    )

    logger.info(
        f"Validation: TrustScore={score}, verdict={verdict_str}, "
        f"{checks_passed}/{len(results)} checks passed"
    )
    return validation


def _parse_soap_sections(text: str) -> dict[str, str]:
    """Extract SOAP sections from raw note text."""
    sections: dict[str, str] = {}
    patterns = [
        ("subjective", r"(?:^|\n)\s*SUBJECTIVE\s*:(.*?)(?=\n\s*(?:OBJECTIVE|ASSESSMENT|PLAN)\s*:|$)"),
        ("objective", r"(?:^|\n)\s*OBJECTIVE\s*:(.*?)(?=\n\s*(?:ASSESSMENT|PLAN)\s*:|$)"),
        ("assessment", r"(?:^|\n)\s*ASSESSMENT\s*:(.*?)(?=\n\s*PLAN\s*:|$)"),
        ("plan", r"(?:^|\n)\s*PLAN\s*:(.*)"),
    ]
    for name, pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            sections[name] = match.group(1).strip()
    return sections
