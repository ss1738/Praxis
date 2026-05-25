"""Clinical intelligence layer - structured clinical reasoning for GP consultations."""

from src.clinical.ontology import (
    ClinicalEntity,
    ClinicalSession,
    Confidence,
    EntityType,
    Provenance,
    SOAPNote,
    Speaker,
)
from src.clinical.verification import ValidationResult, validate_soap_note

__all__ = [
    "ClinicalSession",
    "ClinicalEntity",
    "SOAPNote",
    "Provenance",
    "Confidence",
    "EntityType",
    "Speaker",
    "ValidationResult",
    "validate_soap_note",
]
