"""Tests for src/clinical/entity_extraction.py (no network/model calls)."""

from __future__ import annotations

import pytest

from src.clinical.entity_extraction import (
    ClinicalEntityExtractor,
    ExtractionResult,
)
from src.clinical.ontology import Confidence, EntityType


class TestExtractionResult:
    """ExtractionResult dataclass."""

    def test_empty_result(self):
        r = ExtractionResult()
        assert r.entities == []
        assert r.method == "regex"
        assert r.num_snomed_linked == 0
        assert r.num_negated == 0

    def test_with_data(self):
        r = ExtractionResult(method="medcat", num_snomed_linked=5, num_negated=1)
        assert r.method == "medcat"
        assert r.num_snomed_linked == 5


class TestRegexExtraction:
    """Regex-based extraction (fallback path)."""

    def setup_method(self):
        self.extractor = ClinicalEntityExtractor()
        # Force regex fallback
        self.extractor._medcat_available = False

    def test_extracts_symptoms(self):
        result = self.extractor.extract("I've been having a cough and headache for days.")
        texts = [e.text for e in result.entities]
        assert "cough" in texts
        assert "headache" in texts

    def test_symptom_confidence_high_for_exact_match(self):
        result = self.extractor.extract("I have a fever and nausea.")
        fever = [e for e in result.entities if e.text == "fever"]
        assert len(fever) == 1
        assert fever[0].confidence == Confidence.HIGH

    def test_extracts_medications(self):
        result = self.extractor.extract("I take paracetamol 500mg and ibuprofen.")
        texts = [e.text for e in result.entities]
        assert any("paracetamol" in t for t in texts)
        assert any("ibuprofen" in t for t in texts)

    def test_medication_confidence_high(self):
        result = self.extractor.extract("I'm on metformin.")
        meds = [e for e in result.entities if e.entity_type == EntityType.MEDICATION]
        assert len(meds) >= 1
        assert meds[0].confidence == Confidence.HIGH

    def test_extracts_negations(self):
        result = self.extractor.extract("No allergies. Denies chest pain.")
        negated = [e for e in result.entities if e.negated]
        assert len(negated) >= 1

    def test_negation_entity_type(self):
        result = self.extractor.extract("No fever reported.")
        negated = [e for e in result.entities if e.negated]
        for e in negated:
            assert e.entity_type == EntityType.NEGATED_FINDING

    def test_deduplication(self):
        result = self.extractor.extract("Cough. The cough has been persistent.")
        coughs = [e for e in result.entities if "cough" in e.text]
        assert len(coughs) == 1

    def test_method_is_regex(self):
        result = self.extractor.extract("I have a headache.")
        assert result.method == "regex"

    def test_empty_transcript(self):
        result = self.extractor.extract("")
        assert result.entities == []

    def test_pain_pattern(self):
        result = self.extractor.extract("I have pain in my lower back.")
        texts = [e.text for e in result.entities]
        assert any("pain" in t and "back" in t for t in texts)

    def test_provenance_set(self):
        result = self.extractor.extract("I have a cough.")
        for e in result.entities:
            assert e.provenance.extraction_method == "regex_ner"
            assert e.provenance.source_text != ""


class TestRegexNegationCheck:
    """Static negation detection method."""

    def test_no_pattern(self):
        assert ClinicalEntityExtractor._regex_negation_check(
            "no fever reported", "fever"
        ) is True

    def test_denies_pattern(self):
        assert ClinicalEntityExtractor._regex_negation_check(
            "patient denies chest pain", "chest pain"
        ) is True

    def test_without_pattern(self):
        assert ClinicalEntityExtractor._regex_negation_check(
            "without swelling", "swelling"
        ) is True

    def test_negative_for_pattern(self):
        assert ClinicalEntityExtractor._regex_negation_check(
            "negative for diabetes", "diabetes"
        ) is True

    def test_not_negated(self):
        assert ClinicalEntityExtractor._regex_negation_check(
            "patient has a cough", "cough"
        ) is False

    def test_ruled_out(self):
        assert ClinicalEntityExtractor._regex_negation_check(
            "ruled out pneumonia", "pneumonia"
        ) is True


class TestExtractorAvailability:
    """Availability checks."""

    def test_medcat_availability_check(self):
        extractor = ClinicalEntityExtractor()
        # Should return True since medcat is installed
        assert isinstance(extractor.medcat_available, bool)

    def test_medspacy_availability_check(self):
        extractor = ClinicalEntityExtractor()
        assert isinstance(extractor.medspacy_available, bool)

    def test_fallback_when_medcat_disabled(self):
        extractor = ClinicalEntityExtractor()
        extractor._medcat_available = False
        result = extractor.extract("Patient has a cough.")
        assert result.method == "regex"
