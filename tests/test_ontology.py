"""Tests for src/clinical/ontology.py - Clinical Ontology dataclasses and enums."""

from __future__ import annotations

import json

import pytest

from src.clinical.ontology import (
    ClinicalEntity,
    ClinicalSession,
    Confidence,
    EntityType,
    Provenance,
    SOAPNote,
    Speaker,
)


class TestEnums:
    """Confidence, EntityType, Speaker enum values."""

    def test_confidence_values(self):
        assert Confidence.HIGH == "high"
        assert Confidence.MEDIUM == "medium"
        assert Confidence.LOW == "low"
        assert Confidence.NEGATED == "negated"

    def test_entity_type_values(self):
        assert EntityType.SYMPTOM == "symptom"
        assert EntityType.CONDITION == "condition"
        assert EntityType.MEDICATION == "medication"
        assert EntityType.PROCEDURE == "procedure"
        assert EntityType.MEASUREMENT == "measurement"
        assert EntityType.ALLERGY == "allergy"
        assert EntityType.NEGATED_FINDING == "negated_finding"

    def test_speaker_values(self):
        assert Speaker.DOCTOR == "doctor"
        assert Speaker.PATIENT == "patient"
        assert Speaker.UNKNOWN == "unknown"

    def test_enums_are_str_subclass(self):
        assert isinstance(Confidence.HIGH, str)
        assert isinstance(EntityType.SYMPTOM, str)
        assert isinstance(Speaker.DOCTOR, str)


class TestProvenance:
    """Provenance dataclass."""

    def test_defaults(self):
        p = Provenance()
        assert p.source_text == ""
        assert p.speaker == Speaker.UNKNOWN
        assert p.timestamp_start == 0.0
        assert p.model_used == ""
        assert p.created_at  # should be a non-empty ISO string

    def test_custom_values(self):
        p = Provenance(source_text="I have a headache", speaker=Speaker.PATIENT, model_used="whisper")
        assert p.source_text == "I have a headache"
        assert p.speaker == Speaker.PATIENT
        assert p.model_used == "whisper"


class TestClinicalEntity:
    """ClinicalEntity creation and serialization."""

    def test_defaults(self):
        e = ClinicalEntity()
        assert len(e.id) == 8
        assert e.text == ""
        assert e.entity_type == EntityType.SYMPTOM
        assert e.confidence == Confidence.MEDIUM
        assert e.negated is False

    def test_custom_entity(self):
        e = ClinicalEntity(
            text="headache",
            entity_type=EntityType.SYMPTOM,
            confidence=Confidence.HIGH,
            snomed_code="25064002",
            snomed_term="Headache",
            severity="moderate",
        )
        assert e.text == "headache"
        assert e.snomed_code == "25064002"
        assert e.severity == "moderate"

    def test_to_dict_keys(self):
        e = ClinicalEntity(text="cough", entity_type=EntityType.SYMPTOM)
        d = e.to_dict()
        expected_keys = {
            "id", "text", "type", "confidence", "snomed_code", "snomed_term",
            "negated", "duration", "severity", "body_site", "source_text",
            "speaker", "model",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values(self):
        prov = Provenance(source_text="patient said cough", speaker=Speaker.PATIENT, model_used="nlp-v1")
        e = ClinicalEntity(text="cough", entity_type=EntityType.SYMPTOM, provenance=prov, negated=True)
        d = e.to_dict()
        assert d["text"] == "cough"
        assert d["type"] == "symptom"
        assert d["negated"] is True
        assert d["source_text"] == "patient said cough"
        assert d["speaker"] == "patient"
        assert d["model"] == "nlp-v1"

    def test_unique_ids(self):
        e1 = ClinicalEntity()
        e2 = ClinicalEntity()
        assert e1.id != e2.id


class TestSOAPNote:
    """SOAPNote creation and serialization."""

    def test_defaults(self):
        s = SOAPNote()
        assert s.subjective == ""
        assert s.reviewed is False
        assert s.accepted is None
        assert s.snomed_codes == []

    def test_custom_soap(self):
        s = SOAPNote(
            subjective="Patient reports headache for 3 days",
            objective="BP 130/85",
            assessment="Tension headache",
            plan="Paracetamol 1g QDS",
            model_used="praxis-soap-v1",
            has_safety_netting=True,
        )
        assert s.subjective.startswith("Patient")
        assert s.has_safety_netting is True

    def test_to_dict_keys(self):
        s = SOAPNote()
        d = s.to_dict()
        expected_keys = {
            "subjective", "objective", "assessment", "plan", "raw_text",
            "model_used", "generation_time_secs", "token_count",
            "has_safety_netting", "has_snomed_codes", "snomed_codes",
            "reviewed", "reviewed_by", "reviewed_at", "accepted", "edits_made",
        }
        assert set(d.keys()) == expected_keys


class TestClinicalSession:
    """ClinicalSession: the root ontological object."""

    def test_creation_defaults(self):
        s = ClinicalSession()
        assert s.session_id  # UUID, non-empty
        assert s.created_at  # ISO timestamp
        assert s.entities == []
        assert s.audit_log == []
        assert s.soap_note is None
        assert s.pipeline_version == "1.0.0"

    def test_log_event(self):
        s = ClinicalSession()
        s.log_event("test_event", details="some detail", model="whisper")
        assert len(s.audit_log) == 1
        entry = s.audit_log[0]
        assert entry["event"] == "test_event"
        assert entry["details"] == "some detail"
        assert entry["model"] == "whisper"
        assert entry["session_id"] == s.session_id
        assert "timestamp" in entry

    def test_log_event_multiple(self):
        s = ClinicalSession()
        s.log_event("e1")
        s.log_event("e2")
        s.log_event("e3")
        assert len(s.audit_log) == 3

    def test_add_entity(self):
        s = ClinicalSession()
        e = ClinicalEntity(text="cough", entity_type=EntityType.SYMPTOM)
        s.add_entity(e)
        assert len(s.entities) == 1
        assert s.entities[0].text == "cough"
        # add_entity should also log an event
        assert len(s.audit_log) == 1
        assert s.audit_log[0]["event"] == "entity_added"

    def test_set_soap_note(self):
        s = ClinicalSession()
        soap = SOAPNote(subjective="Headache", model_used="med42")
        s.set_soap_note(soap)
        assert s.soap_note is soap
        assert len(s.audit_log) == 1
        assert s.audit_log[0]["event"] == "soap_generated"

    def test_review_soap_accepted(self):
        s = ClinicalSession()
        soap = SOAPNote(subjective="Headache")
        s.set_soap_note(soap)
        s.review_soap("Dr Smith", accepted=True)
        assert s.soap_note.reviewed is True
        assert s.soap_note.reviewed_by == "Dr Smith"
        assert s.soap_note.accepted is True
        assert s.soap_note.reviewed_at != ""

    def test_review_soap_rejected_with_edits(self):
        s = ClinicalSession()
        soap = SOAPNote(subjective="Headache")
        s.set_soap_note(soap)
        s.review_soap("Dr Jones", accepted=False, edits=["fix plan", "add safety netting"])
        assert s.soap_note.accepted is False
        assert s.soap_note.edits_made == ["fix plan", "add safety netting"]

    def test_review_soap_no_soap_set(self):
        s = ClinicalSession()
        s.review_soap("Dr Smith", accepted=True)
        # Should not crash; soap_note remains None
        assert s.soap_note is None

    def test_hash_audio(self):
        s = ClinicalSession()
        audio = b"fake audio bytes"
        h = s.hash_audio(audio)
        assert len(h) == 64  # SHA-256 hex digest
        assert s.audio_hash == h

    def test_hash_audio_deterministic(self):
        s = ClinicalSession()
        assert s.hash_audio(b"abc") == s.hash_audio(b"abc")

    def test_to_dict_structure(self):
        s = ClinicalSession()
        d = s.to_dict()
        assert "session_id" in d
        assert "entities" in d
        assert isinstance(d["entities"], list)
        assert d["soap_note"] is None

    def test_to_dict_with_entities_and_soap(self):
        s = ClinicalSession()
        s.add_entity(ClinicalEntity(text="cough"))
        s.set_soap_note(SOAPNote(subjective="Cough for 1 week"))
        d = s.to_dict()
        assert len(d["entities"]) == 1
        assert d["entities"][0]["text"] == "cough"
        assert d["soap_note"]["subjective"] == "Cough for 1 week"

    def test_to_json_valid(self):
        s = ClinicalSession()
        s.add_entity(ClinicalEntity(text="fever"))
        j = s.to_json()
        parsed = json.loads(j)
        assert parsed["session_id"] == s.session_id

    def test_to_json_indent(self):
        s = ClinicalSession()
        j4 = s.to_json(indent=4)
        # 4-space indent should produce longer output than 2-space
        j2 = s.to_json(indent=2)
        assert len(j4) >= len(j2)
