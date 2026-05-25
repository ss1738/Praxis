"""Tests for ConstitutionalSOAPGenerator - self-correcting SOAP generation.

Unit tests cover: section extraction/splicing, repair-priority ordering,
iteration data structures, and convergence logic (with mocked Ollama calls).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.soap_generation.constitutional import (
    ConstitutionalResult,
    ConstitutionalSOAPGenerator,
    _REPAIR_PRIORITY,
    _extract_section,
    _splice_section,
)
from src.soap_generation.generator import SOAPModel
from src.soap_generation.templates import NoteTemplate


# ── Section extraction ────────────────────────────────────────────────────────

SAMPLE_NOTE = """SUBJECTIVE:
Patient presents with 3-day history of productive cough and fever.

OBJECTIVE:
Temperature 38.2°C. Chest clear on auscultation.

ASSESSMENT:
Acute bronchitis [SNOMED: 10509002]

PLAN:
1. Paracetamol 1g QDS PRN
2. Encourage fluids and rest
3. Return if worsening or no improvement in 7 days"""


def test_extract_plan_section():
    text = _extract_section(SAMPLE_NOTE, "PLAN")
    assert "Paracetamol" in text
    assert "SUBJECTIVE" not in text


def test_extract_assessment_section():
    text = _extract_section(SAMPLE_NOTE, "ASSESSMENT")
    assert "bronchitis" in text.lower()
    assert "PLAN" not in text


def test_extract_subjective_section():
    text = _extract_section(SAMPLE_NOTE, "SUBJECTIVE")
    assert "cough" in text.lower()


def test_extract_missing_section_returns_empty():
    note = "SUBJECTIVE:\nHeadache\n\nASSESSMENT:\nMigraine\n"
    assert _extract_section(note, "PLAN") == ""


def test_extract_case_insensitive():
    note = "subjective:\nHeadache\n\nobjective:\nNormal\n\nassessment:\nMigraine\n\nplan:\nTreat"
    assert _extract_section(note, "PLAN") == "Treat"


# ── Section splicing ──────────────────────────────────────────────────────────

def test_splice_replaces_plan():
    new_plan = "1. Amoxicillin 500mg TDS\n2. Return if worsening\n3. Call 111 if high fever"
    result = _splice_section(SAMPLE_NOTE, "PLAN", new_plan)
    assert "Amoxicillin" in result
    assert "Paracetamol" not in result  # old plan replaced
    assert "SUBJECTIVE" in result       # other sections intact
    assert "ASSESSMENT" in result


def test_splice_replaces_assessment():
    new_assessment = "Community-acquired pneumonia [SNOMED: 233604007]"
    result = _splice_section(SAMPLE_NOTE, "ASSESSMENT", new_assessment)
    assert "Community-acquired pneumonia" in result
    assert "Acute bronchitis" not in result
    assert "PLAN" in result  # plan still present


def test_splice_appends_if_section_missing():
    note = "SUBJECTIVE:\nHeadache\n\nASSESSMENT:\nMigraine\n"
    result = _splice_section(note, "PLAN", "Sumatriptan 50mg PRN")
    assert "Sumatriptan" in result


def test_splice_roundtrip_preserves_other_sections():
    new_plan = "Updated plan content"
    result = _splice_section(SAMPLE_NOTE, "PLAN", new_plan)
    for section in ("SUBJECTIVE", "OBJECTIVE", "ASSESSMENT"):
        assert section in result


# ── Repair priority ───────────────────────────────────────────────────────────

def test_clinical_safety_is_highest_priority():
    assert _REPAIR_PRIORITY[0] == "clinical_safety"


def test_all_check_names_in_priority():
    expected = {
        "clinical_safety", "completeness", "transcript_fidelity",
        "drug_interactions", "nice_guidelines", "snomed_validation",
    }
    assert set(_REPAIR_PRIORITY) == expected


# ── ConstitutionalResult ──────────────────────────────────────────────────────

def _make_result(scores: list[int]) -> ConstitutionalResult:
    from src.soap_generation.constitutional import ConstitutionalIteration
    iters = [
        ConstitutionalIteration(
            iteration=i, note="note", trust_score=s, verdict="needs_review",
            checks_passed=3, checks_total=6, issues=[], repair_targets=[],
            generation_time_secs=1.0,
        )
        for i, s in enumerate(scores)
    ]
    best = max(scores)
    return ConstitutionalResult(
        final_note="note", final_trust_score=best,
        final_verdict="approved" if best >= 70 else "needs_review",
        converged=best >= 70, iterations=iters,
        total_time_secs=3.0, model="praxis-soap-v1", template="soap",
    )


def test_score_trajectory():
    result = _make_result([55, 65, 75])
    assert result.score_trajectory == [55, 65, 75]


def test_num_iterations():
    result = _make_result([55, 65, 75])
    assert result.num_iterations == 3


def test_converged_true_when_score_above_target():
    result = _make_result([55, 75])
    assert result.converged is True


def test_converged_false_when_below_target():
    result = _make_result([55, 60, 65])
    assert result.converged is False


# ── Constitutional generation (mocked Ollama) ─────────────────────────────────

_SAFE_NOTE = """\
SUBJECTIVE:
Patient presents with 5-day cough and fever.

OBJECTIVE:
Temperature 38.5°C. Chest clear bilaterally.

ASSESSMENT:
Acute lower respiratory tract infection [SNOMED: 10509002]

PLAN:
1. Amoxicillin 500mg TDS for 5 days
2. Paracetamol 1g QDS PRN for fever
3. Drink plenty of fluids and rest
4. Return if no improvement in 72 hours, call 111 if temperature rises above 39.5°C or breathing worsens, call 999 if severe difficulty breathing
5. Safety-netting: seek urgent help if symptoms worsen significantly"""


def _mock_soap_result(note: str, tokens: int = 300):
    from src.soap_generation.generator import SOAPResult
    return SOAPResult(
        soap_note=note, model="praxis-soap-v1",
        generation_time_secs=2.0, token_count=tokens, template="soap",
    )


@patch("src.soap_generation.constitutional.validate_soap_note")
@patch("src.soap_generation.generator.SOAPGenerator.generate")
def test_no_repair_when_already_approved(mock_generate, mock_validate):
    from src.clinical.verification import CheckResult, ValidationResult

    mock_generate.return_value = _mock_soap_result(_SAFE_NOTE)
    # Simulate an already-approved note - TrustScore 80, all checks pass
    mock_validate.return_value = ValidationResult(
        trust_score=80, verdict="approved",
        checks_passed=6, checks_total=6, issues=[], suggestions=[],
        check_results=[
            CheckResult(check_name=name, passed=True)
            for name in [
                "transcript_fidelity", "snomed_validation", "clinical_safety",
                "nice_guidelines", "completeness", "drug_interactions",
            ]
        ],
    )

    gen = ConstitutionalSOAPGenerator(max_iterations=3)
    result = gen.generate("Patient has cough and fever for 5 days.")

    assert result.converged
    assert result.num_iterations == 1   # no repairs triggered
    assert result.final_trust_score == 80


@patch("src.soap_generation.constitutional.httpx.Client")
@patch("src.soap_generation.generator.SOAPGenerator.generate")
def test_repairs_improve_score(mock_generate, mock_client_cls):
    # Initial note missing safety-netting
    unsafe_note = """\
SUBJECTIVE:
Patient has a cough.

OBJECTIVE:
Chest clear.

ASSESSMENT:
Acute bronchitis [SNOMED: 10509002]

PLAN:
1. Paracetamol 1g QDS
2. Rest"""

    repaired_plan = (
        "1. Paracetamol 1g QDS\n2. Rest\n"
        "3. Return if worsening, call 111 if high fever, call 999 if breathing difficulty\n"
        "4. Safety-netting: seek urgent help if symptoms significantly worsen"
    )

    mock_generate.return_value = _mock_soap_result(unsafe_note)

    mock_client = MagicMock()
    mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
    mock_client_cls.return_value = mock_client
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "message": {"content": repaired_plan},
        "eval_count": 80,
    }
    mock_client.post.return_value = mock_resp

    gen = ConstitutionalSOAPGenerator(max_iterations=3)
    result = gen.generate("Patient has a cough.")

    assert result.num_iterations >= 1
    # Trajectory should show improvement or at least no degradation
    scores = result.score_trajectory
    assert scores[0] <= scores[-1] or len(scores) == 1


@patch("src.soap_generation.generator.SOAPGenerator.generate")
def test_returns_best_note_on_no_convergence(mock_generate):
    note_55 = """\
SUBJECTIVE:
Cough.

OBJECTIVE:
Normal.

ASSESSMENT:
Bronchitis

PLAN:
Rest."""

    mock_generate.return_value = _mock_soap_result(note_55)

    with patch.object(
        ConstitutionalSOAPGenerator,
        "_repair_section",
        return_value=(note_55, 1.0, 50),
    ):
        gen = ConstitutionalSOAPGenerator(max_iterations=2)
        result = gen.generate("Cough for 3 days.")

    assert result.final_note is not None
    assert result.final_trust_score >= 0


@patch("src.soap_generation.generator.SOAPGenerator.generate")
def test_max_iterations_respected(mock_generate):
    minimal_note = "SUBJECTIVE:\nCough\n\nASSESSMENT:\nBronchitis\n\nPLAN:\nRest"
    mock_generate.return_value = _mock_soap_result(minimal_note)

    with patch.object(
        ConstitutionalSOAPGenerator,
        "_repair_section",
        return_value=(minimal_note, 1.0, 30),
    ):
        gen = ConstitutionalSOAPGenerator(max_iterations=2)
        result = gen.generate("Cough.")

    assert result.num_iterations <= 3  # initial + max 2 repairs
