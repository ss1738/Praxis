"""Tests for src/clinical/evaluator.py - Clinical quality evaluator."""

from __future__ import annotations

import pytest

from src.clinical.evaluator import ClinicalEvaluator, QualityScore


FULL_SOAP = """
SUBJECTIVE:
Chief complaint: Patient presents with persistent cough and history of asthma.
Duration: 2 weeks. Onset: gradual. Currently taking salbutamol inhaler.

OBJECTIVE:
Examination findings: Auscultation reveals mild wheeze bilaterally.
Blood pressure: 128/82. Temperature: 37.1C.

ASSESSMENT:
Diagnosis: Acute exacerbation of asthma, consistent with viral trigger.
SNOMED: 195967001

PLAN:
Prescribe prednisolone 40mg OD for 5 days.
Treatment: increase salbutamol to QDS.
Safety-netting: return if symptoms worsen or breathing difficulty increases.
Seek urgent care via 111 or 999 if severe breathlessness.
Follow-up appointment in 2 weeks.
"""


class TestQualityScoreDefaults:
    """QualityScore default values and properties."""

    def test_defaults_all_false(self):
        q = QualityScore()
        assert q.has_subjective is False
        assert q.has_objective is False
        assert q.has_assessment is False
        assert q.has_plan is False
        assert q.has_snomed_codes is False
        assert q.has_hallucinated_content is False

    def test_structure_score_none(self):
        q = QualityScore()
        assert q.structure_score == 0.0

    def test_structure_score_all(self):
        q = QualityScore(has_subjective=True, has_objective=True, has_assessment=True, has_plan=True)
        assert q.structure_score == 1.0

    def test_structure_score_partial(self):
        q = QualityScore(has_subjective=True, has_plan=True)
        assert q.structure_score == 0.5

    def test_content_score_none(self):
        q = QualityScore()
        assert q.content_score == 0.0

    def test_content_score_all(self):
        q = QualityScore(
            has_chief_complaint=True, has_history=True, has_medications=True,
            has_examination_findings=True, has_diagnosis=True, has_treatment_plan=True,
            has_safety_netting=True, has_follow_up=True,
        )
        assert q.content_score == 1.0

    def test_overall_score_zero(self):
        q = QualityScore()
        # No SNOMED (0), no hallucination (safety=1.0)
        # 0.15*0 + 0.35*0 + 0.10*0 + 0.40*1 = 0.4
        assert q.overall_score == pytest.approx(0.4)

    def test_overall_score_perfect(self):
        q = QualityScore(
            has_subjective=True, has_objective=True, has_assessment=True, has_plan=True,
            has_chief_complaint=True, has_history=True, has_medications=True,
            has_examination_findings=True, has_diagnosis=True, has_treatment_plan=True,
            has_safety_netting=True, has_follow_up=True,
            has_snomed_codes=True, has_hallucinated_content=False,
        )
        assert q.overall_score == pytest.approx(1.0)

    def test_overall_score_with_hallucination(self):
        q = QualityScore(has_hallucinated_content=True)
        # safety = 0, no structure/content/snomed
        # 0.15*0 + 0.35*0 + 0.10*0 + 0.40*0 = 0.0
        assert q.overall_score == pytest.approx(0.0)

    def test_overall_score_safety_dominates(self):
        """Hallucination should heavily penalise even an otherwise good note."""
        q = QualityScore(
            has_subjective=True, has_objective=True, has_assessment=True, has_plan=True,
            has_chief_complaint=True, has_history=True, has_medications=True,
            has_examination_findings=True, has_diagnosis=True, has_treatment_plan=True,
            has_safety_netting=True, has_follow_up=True,
            has_snomed_codes=True, has_hallucinated_content=True,
        )
        # 0.15*1 + 0.35*1 + 0.10*1 + 0.40*0 = 0.6
        assert q.overall_score == pytest.approx(0.6)

    def test_to_dict_keys(self):
        q = QualityScore()
        d = q.to_dict()
        expected = {
            "structure_score", "content_score", "overall_score",
            "has_snomed_codes", "snomed_code_count", "has_safety_netting",
            "uses_uk_english", "template_used", "model_used", "generation_time_secs", "token_count",
        }
        assert set(d.keys()) == expected

    def test_to_dict_rounds_scores(self):
        q = QualityScore(has_subjective=True)
        d = q.to_dict()
        # structure = 0.25, content = 0.0
        # overall = 0.15*0.25 + 0.35*0 + 0.10*0 + 0.40*1.0 = 0.4375
        assert d["structure_score"] == 0.25
        assert d["overall_score"] == round(0.4375, 2)


class TestClinicalEvaluator:
    """ClinicalEvaluator.evaluate() with various inputs."""

    @pytest.fixture
    def evaluator(self):
        return ClinicalEvaluator()

    def test_full_soap_structure(self, evaluator):
        score = evaluator.evaluate(FULL_SOAP)
        assert score.has_subjective is True
        assert score.has_objective is True
        assert score.has_assessment is True
        assert score.has_plan is True
        assert score.structure_score == 1.0

    def test_full_soap_content(self, evaluator):
        score = evaluator.evaluate(FULL_SOAP)
        assert score.has_chief_complaint is True
        assert score.has_history is True
        assert score.has_medications is True
        assert score.has_examination_findings is True
        assert score.has_diagnosis is True
        assert score.has_treatment_plan is True
        assert score.has_safety_netting is True
        assert score.has_follow_up is True

    def test_full_soap_snomed(self, evaluator):
        score = evaluator.evaluate(FULL_SOAP)
        assert score.has_snomed_codes is True
        assert score.snomed_code_count >= 1

    def test_full_soap_uk_english(self, evaluator):
        score = evaluator.evaluate(FULL_SOAP)
        assert score.uses_uk_english is True

    def test_us_english_detected(self, evaluator):
        text = "SUBJECTIVE: Patient took acetaminophen for the pain. Visited the emergency room."
        score = evaluator.evaluate(text)
        assert score.uses_uk_english is False

    def test_mixed_uk_us_not_uk(self, evaluator):
        text = "Prescribed paracetamol. Also mentioned acetaminophen."
        score = evaluator.evaluate(text)
        # UK terms present but US terms also present => not UK English
        assert score.uses_uk_english is False

    def test_safety_netting_detected(self, evaluator):
        text = "PLAN: Return if symptoms worsen. Call 999 in emergency."
        score = evaluator.evaluate(text)
        assert score.has_safety_netting is True

    def test_empty_text(self, evaluator):
        score = evaluator.evaluate("")
        assert score.has_subjective is False
        assert score.has_objective is False
        assert score.structure_score == 0.0
        assert score.content_score == 0.0

    def test_partial_soap(self, evaluator):
        text = "SUBJECTIVE: Headache.\nPLAN: Prescribe paracetamol."
        score = evaluator.evaluate(text)
        assert score.has_subjective is True
        assert score.has_plan is True
        assert score.has_objective is False
        assert score.has_assessment is False
        assert score.structure_score == 0.5

    def test_metadata_passthrough(self, evaluator):
        score = evaluator.evaluate("some text", model="med42", time_secs=1.5, tokens=200)
        assert score.model_used == "med42"
        assert score.generation_time_secs == 1.5
        assert score.token_count == 200

    def test_snomed_multiple_codes(self, evaluator):
        text = "SNOMED: 195967001 and SNOMED: 386661006"
        score = evaluator.evaluate(text)
        assert score.snomed_code_count == 2

    def test_snomed_no_codes(self, evaluator):
        text = "No coding present."
        score = evaluator.evaluate(text)
        assert score.has_snomed_codes is False
        assert score.snomed_code_count == 0
