"""Full clinical scribe pipeline: STT -> Diarisation -> NLP -> SOAP Generation."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from src.clinical.ontology import (
    ClinicalEntity, ClinicalSession, Confidence, EntityType,
    Provenance, SOAPNote, Speaker,
)
from src.soap_generation.generator import SOAPGenerator, SOAPModel, SOAPResult
from src.soap_generation.templates import COMMON_SNOMED_CODES, NoteTemplate
from src.clinical.entity_extraction import ClinicalEntityExtractor, ExtractionResult
from src.speech_to_text.speaker_diarisation import (
    DiarisationResult,
    SpeakerSegment,
    diarise_with_fallback,
)
from src.clinical.verification import ValidationResult, validate_soap_note
from src.clinical.source_linking import SourceMap, build_source_map
from src.clinical.nlp_pipeline import NLPResult, run_nlp_pipeline, format_nlp_context


# ── Valid SNOMED CT codes (extracted from our reference list) ────────────────
_VALID_SNOMED_CODES: set[str] = set()
for _line in COMMON_SNOMED_CODES.strip().splitlines():
    _m = re.match(r"- (\d{6,}):", _line.strip())
    if _m:
        _VALID_SNOMED_CODES.add(_m.group(1))

# ── US → UK English medical term mapping ─────────────────────────────────────
_US_TO_UK_TERMS: dict[str, str] = {
    "acetaminophen": "paracetamol",
    "epinephrine": "adrenaline",
    "albuterol": "salbutamol",
    "physical therapy": "physiotherapy",
    "emergency room": "A&E",
    " er ": " A&E ",
    " pcp ": " GP ",
    "primary care physician": "GP",
    "primary care provider": "GP",
    "behavioral": "behavioural",
    "anesthesia": "anaesthesia",
    "anesthetic": "anaesthetic",
    "pediatric": "paediatric",
    "gynecology": "gynaecology",
    "hemorrhage": "haemorrhage",
    "hemoptysis": "haemoptysis",
    "edema": "oedema",
    "esophagus": "oesophagus",
    "fetus": "foetus",
    "tumor": "tumour",
    "color": "colour",
    "fiber": "fibre",
    "center": "centre",
    "liter": "litre",
    "meter": "metre",
}


# Map regex entity types to ontology EntityType
_ENTITY_TYPE_MAP = {
    "SYMPTOM": EntityType.SYMPTOM,
    "MEDICATION": EntityType.MEDICATION,
    "NEGATED": EntityType.NEGATED_FINDING,
}

# Exact symptom names - HIGH confidence when matched directly
_EXACT_SYMPTOMS = {
    "cough", "headache", "fever", "nausea", "vomiting", "diarrhoea",
    "fatigue", "dizziness", "shortness of breath", "breathless",
    "chest pain", "palpitations", "numbness", "tingling", "swelling",
    "rash", "itching", "bleeding", "insomnia", "anxiety", "depression",
    "weight loss", "weight gain",
}

# Regex to extract SNOMED codes from generated SOAP text
_SNOMED_RE = re.compile(r'(?:\[?SNOMED[:\s]*|snomed[:\s]*)(\d{6,})\]?')


@dataclass
class PipelineResult:
    """Complete pipeline output."""
    # STT stage
    transcript: str = ""
    stt_engine: str = ""
    stt_time_secs: float = 0.0

    # Diarisation stage
    diarised_transcript: str = ""
    speakers: list[str] = field(default_factory=list)
    diarisation: DiarisationResult | None = None

    # NLP / Entity extraction stage
    entities: list[ClinicalEntity] = field(default_factory=list)
    extraction_result: ExtractionResult | None = None
    nlp_result: NLPResult | None = None

    # SOAP generation stage
    soap_result: SOAPResult | None = None

    # Verification stage
    verification: ValidationResult | None = None

    # Source linking stage
    source_map: SourceMap | None = None

    # Full session (ontology object)
    session: ClinicalSession | None = None

    # Overall
    total_time_secs: float = 0.0


class ClinicalScribePipeline:
    """End-to-end pipeline from audio/transcript to SOAP note.

    Currently supports:
    - STT: via existing Whisper/MedASR engines (when audio provided)
    - Diarisation: placeholder (labels doctor/patient turns from text cues)
    - Entity extraction: rule-based medical entity detection
    - SOAP generation: via fine-tuned LLMs through Ollama
    """

    def __init__(self, soap_model: SOAPModel = SOAPModel.PRAXIS_PRIMARY,
                 template: NoteTemplate = NoteTemplate.SOAP):
        self.soap_generator = SOAPGenerator(model=soap_model, template=template)
        self.template = template
        self.entity_extractor = ClinicalEntityExtractor()

    def run_from_transcript(self, transcript: str,
                            template: NoteTemplate | None = None) -> PipelineResult:
        """Run the pipeline starting from an existing transcript."""
        t0 = time.perf_counter()
        active_template = template or self.template

        # Build session from the start
        session = ClinicalSession()
        session.raw_transcript = transcript
        session.log_event("pipeline_started", f"template={active_template.value}")

        result = PipelineResult(transcript=transcript, session=session)

        # Stage 1: Diarisation (rule-based fallback for text-only input)
        logger.info("Stage 1: Speaker diarisation...")
        diar = diarise_with_fallback(audio_path=None, transcript=transcript)
        result.diarisation = diar
        result.diarised_transcript = diar.diarised_transcript or self._diarise(transcript)
        result.speakers = diar.speakers or self._detect_speakers(transcript)
        session.diarised_transcript = result.diarised_transcript
        session.speakers_detected = result.speakers

        # Stage 2: Entity extraction (MedCAT + medspaCy, or regex fallback)
        logger.info("Stage 2: Entity extraction...")
        extraction = self.entity_extractor.extract(transcript)
        result.extraction_result = extraction
        result.entities = extraction.entities
        logger.info(
            f"  Extracted {len(extraction.entities)} entities "
            f"(method={extraction.method}, "
            f"snomed={extraction.num_snomed_linked}, "
            f"negated={extraction.num_negated})"
        )
        for entity in result.entities:
            session.add_entity(entity)

        # Stage 2.5: NLP pipeline (SNOMED mapping, negation, section hints)
        logger.info("Stage 2.5: NLP pipeline (SNOMED + negation + sections)...")
        nlp = run_nlp_pipeline(transcript)
        result.nlp_result = nlp
        logger.info(
            f"  NLP: {len(nlp.entities)} entities, "
            f"{sum(1 for e in nlp.entities if e.snomed_code)} SNOMED codes, "
            f"{len(nlp.medications)} medications"
        )

        # Build enriched transcript with NLP context for the LLM
        nlp_context = format_nlp_context(nlp)
        enriched_transcript = transcript
        if nlp_context:
            enriched_transcript = (
                f"[CLINICAL NLP CONTEXT - use to inform the note]\n"
                f"{nlp_context}\n\n"
                f"[CONSULTATION TRANSCRIPT]\n"
                f"{transcript}"
            )

        # Stage 3: SOAP generation
        logger.info("Stage 3: Clinical note generation...")
        result.soap_result = self.soap_generator.generate(enriched_transcript, template=active_template)

        if result.soap_result:
            note_text = result.soap_result.soap_note

            # Post-processing: enforce UK English
            note_text = self._enforce_uk_english(note_text)
            result.soap_result.soap_note = note_text

            # SNOMED validation: only keep codes in our reference list
            all_codes = list(set(_SNOMED_RE.findall(note_text)))
            valid_codes = [c for c in all_codes if c in _VALID_SNOMED_CODES]
            invalid_codes = [c for c in all_codes if c not in _VALID_SNOMED_CODES]
            if invalid_codes:
                logger.warning(f"Removed {len(invalid_codes)} unvalidated SNOMED codes: {invalid_codes}")

            # Safety-netting: check specifically in Plan section
            has_safety = self._check_safety_netting_in_plan(note_text)

            # Hallucination grounding: check SOAP content against transcript
            grounding_warnings = self._check_grounding(note_text, transcript)
            if grounding_warnings:
                logger.warning(f"Grounding check: {len(grounding_warnings)} potential issues")
                session.log_event("grounding_warnings", "; ".join(grounding_warnings))

            soap = SOAPNote(
                raw_text=note_text,
                model_used=result.soap_result.model,
                generation_time_secs=result.soap_result.generation_time_secs,
                token_count=result.soap_result.token_count,
                has_safety_netting=has_safety,
                has_snomed_codes=len(valid_codes) > 0,
                snomed_codes=valid_codes,
            )
            session.set_soap_note(soap)

        # Stage 4: Clinical validation (6 checks + TrustScore)
        logger.info("Stage 4: Clinical validation (6 checks)...")
        note_text = result.soap_result.soap_note if result.soap_result else ""
        if note_text:
            result.verification = validate_soap_note(
                note_text=note_text,
                transcript=transcript,
            )
            logger.info(
                f"  TrustScore={result.verification.trust_score}, "
                f"verdict={result.verification.verdict}, "
                f"{result.verification.checks_passed}/{result.verification.checks_total} passed"
            )
            session.log_event(
                "verification_complete",
                f"TrustScore={result.verification.trust_score}, verdict={result.verification.verdict}",
            )

        # Stage 5: Source linking
        logger.info("Stage 5: Source linking...")
        if note_text:
            soap_sections = self._extract_soap_sections(note_text)
            result.source_map = build_source_map(transcript, soap_sections)
            logger.info(f"  {len(result.source_map.links)} source links created")

        result.total_time_secs = round(time.perf_counter() - t0, 2)
        session.total_time_secs = result.total_time_secs
        logger.info(f"Pipeline complete in {result.total_time_secs}s")
        return result

    def run_from_audio(self, audio_path: str | Path) -> PipelineResult:
        """Run full pipeline from audio file (STT -> Diarisation -> NLP -> SOAP)."""
        from src.speech_to_text.audio_preprocessor import AudioPreprocessor
        from src.speech_to_text.engine_factory import EngineType, create_engine
        from src.speech_to_text.medical_spell_checker import MedicalSpellChecker

        t0 = time.perf_counter()

        # Stage 0: STT
        # Uses the Whisper baseline so the audio->SOAP path is turn-key without the
        # private PRAXIS-AUSCULT checkpoint. Switch to EngineType.PRAXIS_CONFORMER to
        # run the contributed model (requires best_primock57.pt locally; see README).
        logger.info("Stage 0: Speech-to-text...")
        preprocessor = AudioPreprocessor()
        engine = create_engine(EngineType.WHISPER)
        engine.load_model()
        spell_checker = MedicalSpellChecker()

        processed = preprocessor.process_file(str(audio_path))
        stt_t0 = time.perf_counter()
        text = engine.transcribe(processed.audio, processed.sample_rate)
        stt_time = round(time.perf_counter() - stt_t0, 2)
        transcript = spell_checker.correct(text)

        # Stage 1: Speaker diarisation
        # Tries PRAXIS-TIMBRE (custom 5.80M ECAPA-TDNN) first, falls back to
        # pyannote 3.1, then to text-cue heuristics. See speaker_diarisation.py.
        logger.info("Stage 1: Speaker diarisation (TIMBRE → pyannote → heuristic)...")
        diar = diarise_with_fallback(
            audio_path=str(audio_path),
            transcript=transcript,
        )

        # Continue with transcript pipeline
        result = self.run_from_transcript(transcript)

        # Override transcript-only heuristic with audio-based diarisation if we got segments
        if diar.segments:
            result.diarisation = diar
            result.diarised_transcript = diar.diarised_transcript
            result.speakers = diar.speakers

        result.stt_engine = "whisper"
        result.stt_time_secs = stt_time
        result.total_time_secs = round(time.perf_counter() - t0, 2)

        if result.session:
            result.session.stt_model = "whisper-small"
            result.session.stt_time_secs = stt_time
            result.session.total_time_secs = result.total_time_secs

        return result

    @staticmethod
    def _diarise(transcript: str) -> str:
        """Simple rule-based speaker labelling from text cues."""
        lines = transcript.strip().split("\n")
        output = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.upper().startswith(("[DOCTOR]", "[PATIENT]", "DOCTOR:", "PATIENT:")):
                output.append(stripped)
            elif any(kw in stripped.lower() for kw in [
                "how can i help", "let me examine", "i'll prescribe", "let me listen",
                "i'd like to", "i think this is", "i'll refer", "your blood pressure",
            ]):
                output.append(f"[DOCTOR]: {stripped}")
            elif any(kw in stripped.lower() for kw in [
                "i've been having", "i've had", "my pain", "i feel", "i noticed",
                "it started", "i take", "i'm on", "no allergies",
            ]):
                output.append(f"[PATIENT]: {stripped}")
            else:
                output.append(stripped)
        return "\n".join(output)

    @staticmethod
    def _detect_speakers(transcript: str) -> list[str]:
        """Detect unique speakers in transcript."""
        speakers = set()
        for line in transcript.split("\n"):
            upper = line.strip().upper()
            if upper.startswith(("[DOCTOR]", "DOCTOR:")):
                speakers.add("Doctor")
            elif upper.startswith(("[PATIENT]", "PATIENT:")):
                speakers.add("Patient")
        return sorted(speakers) or ["Doctor", "Patient"]

    @staticmethod
    def _enforce_uk_english(text: str) -> str:
        """Replace US English medical terms with UK equivalents."""
        result = text
        for us_term, uk_term in _US_TO_UK_TERMS.items():
            pattern = re.compile(re.escape(us_term), re.IGNORECASE)
            result = pattern.sub(uk_term, result)
        return result

    @staticmethod
    def _check_safety_netting_in_plan(note_text: str) -> bool:
        """Check for safety-netting advice specifically in the Plan section."""
        plan_match = re.search(
            r"(?:^|\n)\s*PLAN\s*:(.*)",
            note_text, re.IGNORECASE | re.DOTALL,
        )
        if not plan_match:
            return False
        plan_text = plan_match.group(1).lower()
        safety_keywords = [
            "safety", "return if", "seek help", "seek urgent",
            "111", "999", "a&e", "worsen", "red flag",
            "attend", "emergency", "call if",
        ]
        return any(kw in plan_text for kw in safety_keywords)

    @staticmethod
    def _extract_soap_sections(text: str) -> dict[str, str]:
        """Extract SOAP sections from raw note text for source linking."""
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

    @staticmethod
    def _check_grounding(soap_text: str, transcript: str) -> list[str]:
        """Check that key clinical terms in SOAP output appear in the transcript.

        Returns a list of warnings for terms found in the SOAP note
        that may not be grounded in the original transcript.
        """
        warnings = []
        transcript_lower = transcript.lower()

        # Extract diagnoses/conditions from ASSESSMENT section
        assess_match = re.search(
            r"(?:^|\n)\s*ASSESSMENT\s*:(.*?)(?:(?:^|\n)\s*PLAN\s*:|$)",
            soap_text, re.IGNORECASE | re.DOTALL,
        )
        if assess_match:
            assessment = assess_match.group(1)
            # Extract key clinical terms (multi-word capitalized phrases or terms before SNOMED)
            clinical_terms = re.findall(
                r"([A-Z][\w\s]{2,30})(?:\s*\[SNOMED)", assessment,
            )
            for term in clinical_terms:
                term_clean = term.strip().lower()
                # Check if any significant word (4+ chars) from the term appears in transcript
                words = [w for w in term_clean.split() if len(w) >= 4]
                if words and not any(w in transcript_lower for w in words):
                    warnings.append(f"Assessment term '{term.strip()}' not found in transcript")

        # Check medications in PLAN that weren't mentioned by patient
        plan_match = re.search(
            r"(?:^|\n)\s*PLAN\s*:(.*)", soap_text, re.IGNORECASE | re.DOTALL,
        )
        if plan_match:
            plan = plan_match.group(1).lower()
            med_pattern = (
                r"(?:prescribe|start|commence|initiate)\s+"
                r"((?:metformin|ramipril|amlodipine|atorvastatin|omeprazole|amoxicillin|"
                r"paracetamol|ibuprofen|codeine|prednisolone|salbutamol|gliclazide|"
                r"levothyroxine|sertraline|citalopram|fluoxetine|diazepam|warfarin|"
                r"aspirin|lansoprazole|naproxen|co-codamol)\w*)"
            )
            for match in re.finditer(med_pattern, plan):
                med = match.group(1)
                if med not in transcript_lower:
                    warnings.append(f"New medication '{med}' in Plan not discussed in transcript")

        return warnings

    @staticmethod
    def _extract_entities(transcript: str) -> list[ClinicalEntity]:
        """Rule-based medical entity extraction. Returns typed ClinicalEntity objects."""
        raw_entities: list[dict] = []
        text_lower = transcript.lower()

        # Symptoms
        symptom_patterns = [
            (r"(?:pain|ache|discomfort)\s+(?:in\s+)?(?:my\s+)?(\w+(?:\s+\w+)?)", "SYMPTOM"),
            (r"(cough|headache|fever|nausea|vomiting|diarrhoea|fatigue|dizziness)", "SYMPTOM"),
            (r"(shortness of breath|breathless|chest pain|palpitations)", "SYMPTOM"),
            (r"(numbness|tingling|swelling|rash|itching|bleeding)", "SYMPTOM"),
            (r"(insomnia|anxiety|depression|weight loss|weight gain)", "SYMPTOM"),
        ]
        for pattern, etype in symptom_patterns:
            for match in re.finditer(pattern, text_lower):
                raw_entities.append({
                    "text": match.group(0),
                    "type": etype,
                    "start": match.start(),
                    "end": match.end(),
                })

        # Medications
        med_patterns = [
            r"((?:metformin|ramipril|amlodipine|atorvastatin|omeprazole|amoxicillin|"
            r"paracetamol|ibuprofen|codeine|prednisolone|salbutamol|gliclazide|"
            r"levothyroxine|sertraline|citalopram|fluoxetine|diazepam|warfarin|"
            r"aspirin|lansoprazole|naproxen|co-codamol)\s*\d*\s*(?:mg|mcg|ml)?)",
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

        # Deduplicate and convert to ClinicalEntity
        seen: set[tuple[str, str]] = set()
        entities: list[ClinicalEntity] = []
        for e in raw_entities:
            key = (e["text"], e["type"])
            if key in seen:
                continue
            seen.add(key)

            etype = _ENTITY_TYPE_MAP.get(e["type"], EntityType.SYMPTOM)

            # Confidence calibration based on match type
            if e["type"] == "MEDICATION":
                confidence = Confidence.HIGH  # Named drugs are unambiguous
            elif e["type"] == "NEGATED":
                confidence = Confidence.NEGATED
            elif e["type"] == "SYMPTOM" and e["text"] in _EXACT_SYMPTOMS:
                confidence = Confidence.HIGH  # Exact symptom name match
            else:
                confidence = Confidence.MEDIUM  # Pattern-based match

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

        return entities
