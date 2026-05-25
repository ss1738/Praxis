"""Tests for src/soap_generation/pipeline.py - Clinical scribe pipeline."""

from __future__ import annotations

import pytest

from src.clinical.ontology import ClinicalEntity, EntityType
from src.soap_generation.pipeline import ClinicalScribePipeline, PipelineResult


class TestPipelineResult:
    """PipelineResult dataclass."""

    def test_defaults(self):
        r = PipelineResult()
        assert r.transcript == ""
        assert r.speakers == []
        assert r.entities == []
        assert r.soap_result is None
        assert r.session is None
        assert r.total_time_secs == 0.0

    def test_custom_values(self):
        r = PipelineResult(transcript="hello", stt_engine="whisper", speakers=["Doctor"])
        assert r.transcript == "hello"
        assert r.stt_engine == "whisper"
        assert r.speakers == ["Doctor"]


class TestDiarise:
    """ClinicalScribePipeline._diarise() speaker labelling."""

    def test_doctor_cue(self):
        text = "How can I help you today?"
        result = ClinicalScribePipeline._diarise(text)
        assert "[DOCTOR]" in result

    def test_patient_cue(self):
        text = "I've been having chest pain for a week."
        result = ClinicalScribePipeline._diarise(text)
        assert "[PATIENT]" in result

    def test_already_labelled_doctor(self):
        text = "[DOCTOR]: Please sit down."
        result = ClinicalScribePipeline._diarise(text)
        assert result == "[DOCTOR]: Please sit down."

    def test_already_labelled_patient(self):
        text = "PATIENT: I feel dizzy."
        result = ClinicalScribePipeline._diarise(text)
        assert result == "PATIENT: I feel dizzy."

    def test_no_cue_unchanged(self):
        text = "The weather is nice."
        result = ClinicalScribePipeline._diarise(text)
        assert result == "The weather is nice."

    def test_multiline(self):
        text = "How can I help you?\nI've been having headaches.\nThe weather is nice."
        result = ClinicalScribePipeline._diarise(text)
        lines = result.split("\n")
        assert len(lines) == 3
        assert "[DOCTOR]" in lines[0]
        assert "[PATIENT]" in lines[1]

    def test_empty_lines_stripped(self):
        text = "How can I help?\n\n\nI feel tired."
        result = ClinicalScribePipeline._diarise(text)
        lines = result.split("\n")
        assert all(line.strip() for line in lines)


class TestDetectSpeakers:
    """ClinicalScribePipeline._detect_speakers()."""

    def test_both_speakers_labelled(self):
        text = "[DOCTOR]: Hello\n[PATIENT]: Hi"
        speakers = ClinicalScribePipeline._detect_speakers(text)
        assert "Doctor" in speakers
        assert "Patient" in speakers

    def test_doctor_only(self):
        text = "DOCTOR: Some monologue"
        speakers = ClinicalScribePipeline._detect_speakers(text)
        assert "Doctor" in speakers

    def test_no_labels_returns_default(self):
        text = "Just some plain text without speaker labels."
        speakers = ClinicalScribePipeline._detect_speakers(text)
        assert speakers == ["Doctor", "Patient"]

    def test_sorted_output(self):
        text = "[PATIENT]: First\n[DOCTOR]: Second"
        speakers = ClinicalScribePipeline._detect_speakers(text)
        assert speakers == ["Doctor", "Patient"]


class TestExtractEntities:
    """ClinicalScribePipeline._extract_entities() - returns ClinicalEntity objects."""

    def test_returns_clinical_entity_objects(self):
        text = "I have a persistent cough."
        entities = ClinicalScribePipeline._extract_entities(text)
        assert all(isinstance(e, ClinicalEntity) for e in entities)

    def test_symptom_detection(self):
        text = "I have a persistent cough and headache."
        entities = ClinicalScribePipeline._extract_entities(text)
        types = {e.entity_type for e in entities}
        assert EntityType.SYMPTOM in types
        texts = {e.text for e in entities}
        assert "cough" in texts
        assert "headache" in texts

    def test_medication_detection(self):
        text = "Patient is taking metformin 500mg and paracetamol."
        entities = ClinicalScribePipeline._extract_entities(text)
        med_entities = [e for e in entities if e.entity_type == EntityType.MEDICATION]
        assert len(med_entities) >= 1
        med_texts = {e.text for e in med_entities}
        assert any("metformin" in t for t in med_texts)

    def test_negation_detection(self):
        text = "No allergies. Denies chest pain."
        entities = ClinicalScribePipeline._extract_entities(text)
        neg_entities = [e for e in entities if e.entity_type == EntityType.NEGATED_FINDING]
        assert len(neg_entities) >= 1
        assert all(e.negated is True for e in neg_entities)

    def test_deduplication(self):
        text = "Patient has a cough. The cough is persistent. Cough for 2 weeks."
        entities = ClinicalScribePipeline._extract_entities(text)
        cough_entities = [e for e in entities if e.text == "cough" and e.entity_type == EntityType.SYMPTOM]
        assert len(cough_entities) == 1

    def test_empty_text(self):
        entities = ClinicalScribePipeline._extract_entities("")
        assert entities == []

    def test_entity_has_provenance(self):
        text = "I have a headache."
        entities = ClinicalScribePipeline._extract_entities(text)
        for e in entities:
            assert e.provenance.extraction_method == "regex_ner"
            assert e.provenance.source_text == e.text

    def test_entity_has_confidence(self):
        text = "I have a headache."
        entities = ClinicalScribePipeline._extract_entities(text)
        for e in entities:
            assert e.confidence.value in ("high", "medium", "low", "negated")

    def test_medication_gets_high_confidence(self):
        text = "Patient is taking metformin."
        entities = ClinicalScribePipeline._extract_entities(text)
        med = [e for e in entities if e.entity_type == EntityType.MEDICATION]
        assert len(med) >= 1
        assert med[0].confidence.value == "high"

    def test_exact_symptom_gets_high_confidence(self):
        text = "I have a headache."
        entities = ClinicalScribePipeline._extract_entities(text)
        symptom = [e for e in entities if e.text == "headache"]
        assert len(symptom) >= 1
        assert symptom[0].confidence.value == "high"

    def test_negation_gets_negated_confidence(self):
        text = "No allergies."
        entities = ClinicalScribePipeline._extract_entities(text)
        neg = [e for e in entities if e.negated]
        assert len(neg) >= 1
        assert neg[0].confidence.value == "negated"

    def test_multiple_entity_types(self):
        text = "I've been having chest pain. Taking paracetamol. No fever."
        entities = ClinicalScribePipeline._extract_entities(text)
        types = {e.entity_type for e in entities}
        assert len(types) >= 2
