"""Clinical entity extraction using MedCAT + medspaCy.

Extracts symptoms, medications, conditions, and procedures from GP transcripts,
links them to SNOMED CT codes, and detects negations.

Falls back to rule-based extraction if MedCAT/medspaCy are unavailable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from loguru import logger

from src.clinical.ontology import (
    ClinicalEntity,
    Confidence,
    EntityType,
    Provenance,
)


# MedCAT semantic type -> our EntityType mapping
_MEDCAT_TYPE_MAP: dict[str, EntityType] = {
    "T-11": EntityType.SYMPTOM,         # Sign or Symptom
    "T-47": EntityType.SYMPTOM,         # Finding
    "T-33": EntityType.CONDITION,        # Disease or Syndrome
    "T-37": EntityType.CONDITION,        # Injury or Poisoning
    "T-61": EntityType.MEDICATION,       # Pharmacological Substance
    "T-20": EntityType.MEDICATION,       # Clinical Drug
    "T-60": EntityType.PROCEDURE,        # Diagnostic Procedure
    "T-59": EntityType.PROCEDURE,        # Therapeutic or Preventive Procedure
    "T-32": EntityType.MEASUREMENT,      # Laboratory or Test Result
}

# Fallback keyword -> EntityType for MedCAT entities without semantic types
_KEYWORD_TYPE_MAP: dict[str, EntityType] = {
    "disorder": EntityType.CONDITION,
    "disease": EntityType.CONDITION,
    "finding": EntityType.SYMPTOM,
    "symptom": EntityType.SYMPTOM,
    "drug": EntityType.MEDICATION,
    "product": EntityType.MEDICATION,
    "procedure": EntityType.PROCEDURE,
    "observable": EntityType.MEASUREMENT,
}


@dataclass
class ExtractionResult:
    """Result of clinical entity extraction."""
    entities: list[ClinicalEntity] = field(default_factory=list)
    method: str = "regex"  # "medcat", "medcat+medspacy", or "regex"
    num_snomed_linked: int = 0
    num_negated: int = 0


class ClinicalEntityExtractor:
    """Extract clinical entities from GP consultation transcripts.

    Uses MedCAT for SNOMED CT concept recognition and medspaCy for
    negation detection. Falls back to regex-based extraction if
    dependencies are unavailable.
    """

    def __init__(self):
        self._medcat_available = None
        self._medspacy_available = None
        self._cat = None
        self._nlp = None

    @property
    def medcat_available(self) -> bool:
        if self._medcat_available is None:
            try:
                import medcat  # noqa: F401
                self._medcat_available = True
            except ImportError:
                self._medcat_available = False
        return self._medcat_available

    @property
    def medspacy_available(self) -> bool:
        if self._medspacy_available is None:
            try:
                import medspacy  # noqa: F401
                self._medspacy_available = True
            except ImportError:
                self._medspacy_available = False
        return self._medspacy_available

    def _load_medcat(self):
        """Load MedCAT with SNOMED CT model."""
        if self._cat is not None:
            return

        from medcat.cat import CAT

        try:
            # Try loading a pre-downloaded model pack
            from pathlib import Path
            model_paths = [
                Path("data/models/medcat_snomed"),
                Path.home() / ".medcat" / "snomed",
            ]
            for path in model_paths:
                if path.exists():
                    logger.info(f"Loading MedCAT from {path}")
                    self._cat = CAT.load_model_pack(str(path))
                    return

            # No local model - use MedCAT's small built-in vocabulary
            logger.info("No SNOMED model pack found, using MedCAT with default vocab")
            self._cat = CAT.load_model_pack(
                "data/models/medcat_snomed"
            )
        except Exception as e:
            logger.warning(f"MedCAT model loading failed: {e}")
            logger.info("Falling back to regex-based extraction")
            self._medcat_available = False

    def _load_medspacy(self):
        """Load medspaCy for negation detection."""
        if self._nlp is not None:
            return

        import medspacy
        self._nlp = medspacy.load(
            enable=["medspacy_pyrush", "medspacy_context"],
        )
        logger.info("medspaCy loaded (sentence splitting + context/negation)")

    def extract(self, transcript: str) -> ExtractionResult:
        """Extract clinical entities from a transcript.

        Tries MedCAT + medspaCy first, falls back to regex.

        Args:
            transcript: GP consultation transcript text.

        Returns:
            ExtractionResult with entities and metadata.
        """
        # Try MedCAT extraction
        if self.medcat_available and self._medcat_available is not False:
            try:
                self._load_medcat()
                if self._cat is not None:
                    return self._extract_medcat(transcript)
            except Exception as e:
                logger.warning(f"MedCAT extraction failed: {e}")
                self._medcat_available = False

        # Fallback to regex
        return self._extract_regex(transcript)

    def _extract_medcat(self, transcript: str) -> ExtractionResult:
        """Extract entities using MedCAT with SNOMED CT linking."""
        doc = self._cat.get_entities(transcript)
        entities_dict = doc.get("entities", {})

        entities = []
        seen: set[str] = set()

        for ent_id, ent in entities_dict.items():
            cui = ent.get("cui", "")
            text = ent.get("source_value", ent.get("detected_name", ""))
            pretty_name = ent.get("pretty_name", text)

            # Skip duplicates
            key = f"{cui}:{text.lower()}"
            if key in seen:
                continue
            seen.add(key)

            # Map semantic type to our EntityType
            type_ids = ent.get("type_ids", [])
            entity_type = EntityType.SYMPTOM
            for tid in type_ids:
                if tid in _MEDCAT_TYPE_MAP:
                    entity_type = _MEDCAT_TYPE_MAP[tid]
                    break
            else:
                # Try keyword matching on the pretty name
                name_lower = pretty_name.lower()
                for keyword, etype in _KEYWORD_TYPE_MAP.items():
                    if keyword in name_lower:
                        entity_type = etype
                        break

            # Confidence from MedCAT's context score
            context_similarity = ent.get("context_similarity", 0.0)
            if context_similarity >= 0.7:
                confidence = Confidence.HIGH
            elif context_similarity >= 0.4:
                confidence = Confidence.MEDIUM
            else:
                confidence = Confidence.LOW

            # Check negation via medspaCy
            negated = False
            if self.medspacy_available:
                negated = self._check_negation_medspacy(transcript, text)
                if negated:
                    confidence = Confidence.NEGATED
                    entity_type = EntityType.NEGATED_FINDING

            entities.append(ClinicalEntity(
                text=text,
                entity_type=entity_type,
                confidence=confidence,
                snomed_code=cui,
                snomed_term=pretty_name,
                negated=negated,
                provenance=Provenance(
                    source_text=text,
                    extraction_method="medcat",
                ),
            ))

        method = "medcat+medspacy" if self.medspacy_available else "medcat"
        return ExtractionResult(
            entities=entities,
            method=method,
            num_snomed_linked=sum(1 for e in entities if e.snomed_code),
            num_negated=sum(1 for e in entities if e.negated),
        )

    def _check_negation_medspacy(self, transcript: str, entity_text: str) -> bool:
        """Check if an entity is negated using medspaCy context detection."""
        try:
            self._load_medspacy()
            doc = self._nlp(transcript)

            entity_lower = entity_text.lower()
            for ent in doc.ents:
                if ent.text.lower() == entity_lower:
                    if hasattr(ent, "is_negated") and ent.is_negated:
                        return True
                    # medspaCy ConText stores negation in ent._.is_negated
                    if hasattr(ent._, "is_negated") and ent._.is_negated:
                        return True
        except Exception as e:
            logger.debug(f"medspaCy negation check failed: {e}")

        # Fallback: simple regex negation check
        return self._regex_negation_check(transcript, entity_text)

    @staticmethod
    def _regex_negation_check(text: str, entity_text: str) -> bool:
        """Simple regex-based negation detection."""
        text_lower = text.lower()
        entity_lower = entity_text.lower()
        neg_patterns = [
            rf"\bno\s+{re.escape(entity_lower)}",
            rf"\bdenies?\s+{re.escape(entity_lower)}",
            rf"\bwithout\s+{re.escape(entity_lower)}",
            rf"\bnegative\s+for\s+{re.escape(entity_lower)}",
            rf"\bruled\s+out\s+{re.escape(entity_lower)}",
            rf"\bno\s+evidence\s+of\s+{re.escape(entity_lower)}",
            rf"\bunlikely\s+to\s+be\s+{re.escape(entity_lower)}",
            rf"\bexcluded\s+{re.escape(entity_lower)}",
            rf"\bcannot\s+rule\s+out\s+{re.escape(entity_lower)}",
            rf"\bnot\s+{re.escape(entity_lower)}",
        ]
        return any(re.search(p, text_lower) for p in neg_patterns)

    def _extract_regex(self, transcript: str) -> ExtractionResult:
        """Regex-based entity extraction (fallback).

        This preserves the original pipeline logic for environments
        where MedCAT is not installed or its model is unavailable.
        """
        raw_entities: list[dict] = []
        text_lower = transcript.lower()

        # Symptoms (UK primary-care expanded)
        symptom_patterns = [
            (r"(?:pain|ache|discomfort)\s+(?:in\s+)?(?:my\s+)?(\w+(?:\s+\w+)?)", "SYMPTOM"),
            (r"(cough|headache|fever|nausea|vomiting|diarrhoea|fatigue|dizziness|lightheaded|lightheadedness)", "SYMPTOM"),
            (r"(shortness of breath|breathless|chest pain|palpitations|wheeze|wheezing)", "SYMPTOM"),
            (r"(numbness|tingling|swelling|rash|itching|bleeding|bruising)", "SYMPTOM"),
            (r"(insomnia|anxiety|depression|weight loss|weight gain|low mood)", "SYMPTOM"),
            (r"(hayfever|hay fever|eczema|psoriasis|gout|frozen shoulder|sciatica)", "CONDITION"),
            (r"(atrial fibrillation|af|afib|dvt|pe|pulmonary embolism|stroke|tia|mi|heart attack)", "CONDITION"),
            (r"(asthma|copd|diabetes|hypertension|high blood pressure|migraine|uti|tonsillitis|otitis media)", "CONDITION"),
            (r"(sore throat|runny nose|blocked nose|sinusitis|earache|toothache|back pain|neck pain|joint pain)", "SYMPTOM"),
            (r"(heartburn|reflux|indigestion|constipation|bloating|dysuria|frequency|urgency)", "SYMPTOM"),
        ]
        for pattern, etype in symptom_patterns:
            for match in re.finditer(pattern, text_lower):
                raw_entities.append({
                    "text": match.group(0),
                    "type": etype,
                    "start": match.start(),
                    "end": match.end(),
                })

        # Medications (UK BNF primary-care expanded)
        med_patterns = [
            r"((?:metformin|ramipril|lisinopril|losartan|candesartan|bisoprolol|atenolol|"
            r"amlodipine|indapamide|bendroflumethiazide|furosemide|spironolactone|digoxin|"
            r"atorvastatin|simvastatin|rosuvastatin|ezetimibe|omeprazole|lansoprazole|"
            r"pantoprazole|amoxicillin|flucloxacillin|clarithromycin|doxycycline|"
            r"trimethoprim|nitrofurantoin|co-amoxiclav|paracetamol|ibuprofen|codeine|"
            r"naproxen|co-codamol|tramadol|prednisolone|salbutamol|beclometasone|"
            r"fluticasone|montelukast|tiotropium|gliclazide|sitagliptin|empagliflozin|"
            r"levothyroxine|sertraline|citalopram|fluoxetine|mirtazapine|amitriptyline|"
            r"diazepam|zopiclone|warfarin|apixaban|rivaroxaban|edoxaban|dabigatran|"
            r"aspirin|clopidogrel|gtn|glyceryl trinitrate|isosorbide|alendronic acid|"
            r"calcium|vitamin d|colecalciferol|ferrous sulfate|folic acid)"
            r"\s*\d*\s*(?:mg|mcg|ml|units?)?)",
        ]
        for pattern in med_patterns:
            for match in re.finditer(pattern, text_lower):
                raw_entities.append({
                    "text": match.group(0).strip(),
                    "type": "MEDICATION",
                    "start": match.start(),
                    "end": match.end(),
                })

        # Negations
        neg_patterns = [
            r"no\s+([\w\s]+?)(?:\.|,|$)",
            r"denies\s+([\w\s]+?)(?:\.|,|$)",
            r"without\s+([\w\s]+?)(?:\.|,|$)",
        ]
        for pattern in neg_patterns:
            for match in re.finditer(pattern, text_lower):
                raw_entities.append({
                    "text": match.group(1).strip(),
                    "type": "NEGATED",
                    "start": match.start(),
                    "end": match.end(),
                })

        # Map to EntityType
        entity_type_map = {
            "SYMPTOM": EntityType.SYMPTOM,
            "MEDICATION": EntityType.MEDICATION,
            "NEGATED": EntityType.NEGATED_FINDING,
        }

        exact_symptoms = {
            "cough", "headache", "fever", "nausea", "vomiting", "diarrhoea",
            "fatigue", "dizziness", "shortness of breath", "breathless",
            "chest pain", "palpitations", "numbness", "tingling", "swelling",
            "rash", "itching", "bleeding", "insomnia", "anxiety", "depression",
            "weight loss", "weight gain",
        }

        # Deduplicate and convert
        seen: set[tuple[str, str]] = set()
        entities: list[ClinicalEntity] = []
        for e in raw_entities:
            key = (e["text"], e["type"])
            if key in seen:
                continue
            seen.add(key)

            etype = entity_type_map.get(e["type"], EntityType.SYMPTOM)

            if e["type"] == "MEDICATION":
                confidence = Confidence.HIGH
            elif e["type"] == "NEGATED":
                confidence = Confidence.NEGATED
            elif e["type"] == "SYMPTOM" and e["text"] in exact_symptoms:
                confidence = Confidence.HIGH
            else:
                confidence = Confidence.MEDIUM

            entities.append(ClinicalEntity(
                text=e["text"],
                entity_type=etype,
                confidence=confidence,
                negated=(e["type"] == "NEGATED"),
                provenance=Provenance(
                    source_text=e["text"],
                    extraction_method="regex_ner",
                ),
            ))

        return ExtractionResult(
            entities=entities,
            method="regex",
            num_snomed_linked=0,
            num_negated=sum(1 for e in entities if e.negated),
        )
