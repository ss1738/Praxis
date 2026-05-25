"""Clinical Ontology - structured representation of a GP consultation.

Every entity has identity, provenance, confidence, and relationships.
Nothing exists without an audit trail.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


@dataclass
class Patient:
    """A registered patient in the system."""
    patient_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    nhs_number: str = ""
    first_name: str = ""
    last_name: str = ""
    date_of_birth: str = ""
    registered_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}".strip()

    def to_dict(self) -> dict:
        return {
            "patient_id": self.patient_id,
            "nhs_number": self.nhs_number,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "date_of_birth": self.date_of_birth,
            "registered_at": self.registered_at,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @staticmethod
    def from_dict(d: dict) -> "Patient":
        return Patient(
            patient_id=d.get("patient_id", str(uuid.uuid4())),
            nhs_number=d.get("nhs_number", ""),
            first_name=d.get("first_name", ""),
            last_name=d.get("last_name", ""),
            date_of_birth=d.get("date_of_birth", ""),
            registered_at=d.get("registered_at", datetime.now(timezone.utc).isoformat()),
        )


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGATED = "negated"


class EntityType(str, Enum):
    SYMPTOM = "symptom"
    CONDITION = "condition"
    MEDICATION = "medication"
    PROCEDURE = "procedure"
    MEASUREMENT = "measurement"
    ALLERGY = "allergy"
    SOCIAL_HISTORY = "social_history"
    FAMILY_HISTORY = "family_history"
    NEGATED_FINDING = "negated_finding"


class Speaker(str, Enum):
    DOCTOR = "doctor"
    PATIENT = "patient"
    UNKNOWN = "unknown"


@dataclass
class Provenance:
    """Where did this piece of information come from?"""
    source_text: str = ""
    speaker: Speaker = Speaker.UNKNOWN
    timestamp_start: float = 0.0
    timestamp_end: float = 0.0
    model_used: str = ""
    extraction_method: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class ClinicalEntity:
    """A single clinical fact with full provenance and confidence."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    text: str = ""
    entity_type: EntityType = EntityType.SYMPTOM
    confidence: Confidence = Confidence.MEDIUM
    snomed_code: str = ""
    snomed_term: str = ""
    provenance: Provenance = field(default_factory=Provenance)
    duration: str = ""
    severity: str = ""
    laterality: str = ""
    body_site: str = ""
    negated: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "type": self.entity_type.value,
            "confidence": self.confidence.value,
            "snomed_code": self.snomed_code,
            "snomed_term": self.snomed_term,
            "negated": self.negated,
            "duration": self.duration,
            "severity": self.severity,
            "body_site": self.body_site,
            "source_text": self.provenance.source_text,
            "speaker": self.provenance.speaker.value,
            "model": self.provenance.model_used,
        }


@dataclass
class SOAPNote:
    """Structured SOAP note with provenance tracking."""
    subjective: str = ""
    objective: str = ""
    assessment: str = ""
    plan: str = ""
    model_used: str = ""
    generation_time_secs: float = 0.0
    token_count: int = 0
    has_safety_netting: bool = False
    has_snomed_codes: bool = False
    snomed_codes: list = field(default_factory=list)
    reviewed: bool = False
    reviewed_by: str = ""
    reviewed_at: str = ""
    edits_made: list = field(default_factory=list)
    accepted: Optional[bool] = None
    raw_text: str = ""

    def to_dict(self) -> dict:
        return {
            "subjective": self.subjective,
            "objective": self.objective,
            "assessment": self.assessment,
            "plan": self.plan,
            "raw_text": self.raw_text,
            "model_used": self.model_used,
            "generation_time_secs": self.generation_time_secs,
            "token_count": self.token_count,
            "has_safety_netting": self.has_safety_netting,
            "has_snomed_codes": self.has_snomed_codes,
            "snomed_codes": self.snomed_codes,
            "reviewed": self.reviewed,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at,
            "accepted": self.accepted,
            "edits_made": self.edits_made,
        }


@dataclass
class ClinicalSession:
    """A complete consultation session - the root ontological object."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    patient_id: str = ""
    patient_name: str = ""

    audio_hash: str = ""
    audio_duration_secs: float = 0.0

    raw_transcript: str = ""
    stt_model: str = ""
    stt_time_secs: float = 0.0

    diarised_transcript: str = ""
    speakers_detected: list = field(default_factory=list)

    entities: list[ClinicalEntity] = field(default_factory=list)

    soap_note: Optional[SOAPNote] = None
    alternative_soap_notes: list[SOAPNote] = field(default_factory=list)

    audit_log: list[dict] = field(default_factory=list)
    pipeline_version: str = "1.0.0"
    total_time_secs: float = 0.0

    def log_event(self, event: str, details: str = "", model: str = ""):
        self.audit_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "details": details,
            "model": model,
            "session_id": self.session_id,
        })

    def hash_audio(self, audio_bytes: bytes) -> str:
        self.audio_hash = hashlib.sha256(audio_bytes).hexdigest()
        return self.audio_hash

    def add_entity(self, entity: ClinicalEntity):
        self.entities.append(entity)
        self.log_event("entity_added", f"{entity.entity_type.value}: {entity.text}", entity.provenance.model_used)

    def set_soap_note(self, soap: SOAPNote):
        self.soap_note = soap
        self.log_event("soap_generated", f"model={soap.model_used}, time={soap.generation_time_secs}s", soap.model_used)

    def review_soap(self, reviewer: str, accepted: bool, edits: list[str] = None):
        if self.soap_note:
            self.soap_note.reviewed = True
            self.soap_note.reviewed_by = reviewer
            self.soap_note.reviewed_at = datetime.now(timezone.utc).isoformat()
            self.soap_note.accepted = accepted
            if edits:
                self.soap_note.edits_made = edits
            self.log_event("soap_reviewed", f"accepted={accepted}, edits={len(edits or [])}")

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "patient_id": self.patient_id,
            "patient_name": self.patient_name,
            "audio_hash": self.audio_hash,
            "audio_duration_secs": self.audio_duration_secs,
            "raw_transcript": self.raw_transcript,
            "stt_model": self.stt_model,
            "stt_time_secs": self.stt_time_secs,
            "diarised_transcript": self.diarised_transcript,
            "speakers_detected": self.speakers_detected,
            "entities": [e.to_dict() for e in self.entities],
            "soap_note": self.soap_note.to_dict() if self.soap_note else None,
            "alternative_soap_notes": [s.to_dict() for s in self.alternative_soap_notes],
            "audit_log": self.audit_log,
            "pipeline_version": self.pipeline_version,
            "total_time_secs": self.total_time_secs,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
