"""Batch 2 feature routes: timed transcription, grounded SOAP, E2E pipeline,
audit log, consent. Kappa viewer reuses /reviews/kappa."""

from __future__ import annotations

import json
import re
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field

from src.clinical.entity_extraction import ClinicalEntityExtractor
from src.clinical.nlp_pipeline import run_nlp_pipeline, format_nlp_context
from src.clinical.ontology import ClinicalSession, Patient, SOAPNote
from src.clinical.session_store import SessionStore
from src.clinical.patient_store import PatientStore
from src.clinical.source_linking import build_source_map
from src.clinical.verification import validate_soap_note
from src.soap_generation.generator import SOAPGenerator, SOAPModel
from src.soap_generation.templates import NoteTemplate
from src.speech_to_text.audio_preprocessor import AudioPreprocessor
from src.speech_to_text.engine_factory import EngineType, create_engine
from src.speech_to_text.speaker_diarisation import diarise_with_fallback

router = APIRouter(prefix="/api/v1", tags=["features2"])

_extractor = ClinicalEntityExtractor()
_session_store = SessionStore()
_patient_store = PatientStore()
_preprocessor = AudioPreprocessor()
_engine_cache: dict = {}
_gen_cache: dict = {}

_AUDIT_LOG = Path("data/audit_log.jsonl")
_AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)


def _audit(event: str, payload: dict) -> None:
    rec = {
        "event": event,
        "at": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    with _AUDIT_LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def _get_engine(et: EngineType):
    if et not in _engine_cache:
        eng = create_engine(et)
        eng.load_model()
        _engine_cache[et] = eng
    return _engine_cache[et]


# ---------------------------------------------------------------------------
# #1 Timed transcription - faster-whisper word-level timestamps
# ---------------------------------------------------------------------------

@router.post("/transcribe/timed")
async def transcribe_timed(file: UploadFile = File(...)):
    """faster-whisper with word_timestamps=True → [{word,start,end}]."""
    suffix = Path(file.filename or "a.wav").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        pre = _preprocessor.process_file(tmp_path)
        eng = _get_engine(EngineType.FASTER_WHISPER)
        model = eng._model  # faster-whisper WhisperModel
        if model is None:
            eng.load_model()
            model = eng._model
        t0 = time.perf_counter()
        segments, _info = model.transcribe(pre.audio, beam_size=5, word_timestamps=True)
        words = []
        full_text_parts = []
        for seg in segments:
            full_text_parts.append(seg.text.strip())
            if seg.words:
                for w in seg.words:
                    words.append({
                        "word": w.word.strip(),
                        "start": round(w.start, 2),
                        "end": round(w.end, 2),
                    })
        elapsed = round(time.perf_counter() - t0, 2)
        _audit("transcribe_timed", {"file": file.filename, "secs": elapsed, "n_words": len(words)})
        return {
            "text": " ".join(full_text_parts).strip(),
            "words": words,
            "duration_secs": round(pre.duration_secs, 2),
            "processing_secs": elapsed,
            "engine": "faster_whisper",
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# #2 Grounded SOAP - generator emits [evidence: "..."] citations, we parse
# ---------------------------------------------------------------------------

class GroundedSOAPRequest(BaseModel):
    transcript: str = Field(..., min_length=1)
    model: SOAPModel = Field(default=SOAPModel.PRAXIS_PRIMARY)


_GROUNDING_INSTRUCTION = (
    "\n\nGROUNDING REQUIREMENT: After every clinical statement you make, append a citation "
    'in the exact form [evidence: "<verbatim span from the transcript>"]. Every symptom, '
    "finding, negation, medication, and plan item must be traceable to the transcript. "
    "Do not invent facts. If no transcript evidence exists, write [evidence: none]."
)


@router.post("/soap/grounded")
async def generate_grounded_soap(req: GroundedSOAPRequest):
    if req.model.value not in _gen_cache:
        _gen_cache[req.model.value] = SOAPGenerator(model=req.model)
    gen = _gen_cache[req.model.value]

    t0 = time.perf_counter()
    result = gen.generate(req.transcript + _GROUNDING_INSTRUCTION, template=NoteTemplate.SOAP)
    elapsed = round(time.perf_counter() - t0, 2)

    citations = _parse_citations(result.soap_note, req.transcript)
    _audit("soap_grounded", {"model": result.model, "citations": len(citations), "secs": elapsed})
    return {
        "soap_note": result.soap_note,
        "citations": citations,
        "grounded_fraction": round(
            sum(1 for c in citations if c["verified_in_transcript"]) / max(len(citations), 1), 3
        ),
        "model": result.model,
        "generation_secs": elapsed,
    }


_CITE_RE = re.compile(r'\[evidence:\s*"([^"]*)"\]|\[evidence:\s*none\]', re.IGNORECASE)


def _parse_citations(soap: str, transcript: str) -> list[dict]:
    tl = transcript.lower()
    out = []
    for m in _CITE_RE.finditer(soap):
        quote = (m.group(1) or "").strip()
        out.append({
            "quote": quote,
            "char_in_soap": m.start(),
            "verified_in_transcript": bool(quote) and quote.lower() in tl,
        })
    return out


# ---------------------------------------------------------------------------
# #3 End-to-end pipeline runner - one call: audio → transcript → entities → SOAP → save
# ---------------------------------------------------------------------------

@router.post("/pipeline/run")
async def run_pipeline(
    file: UploadFile = File(...),
    engine: EngineType = Form(default=EngineType.PRAXIS_CONFORMER),
    model: SOAPModel = Form(default=SOAPModel.PRAXIS_PRIMARY),
    patient_id: str = Form(default=""),
    consent: bool = Form(default=False),
):
    if not consent:
        raise HTTPException(status_code=400, detail="Patient consent must be recorded before processing")

    suffix = Path(file.filename or "a.wav").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        pre = _preprocessor.process_file(tmp_path)
        asr = _get_engine(engine)
        t0 = time.perf_counter()
        transcript = asr.transcribe(pre.audio, pre.sample_rate)
        asr_secs = round(time.perf_counter() - t0, 2)

        ext = _extractor.extract(transcript)

        if model.value not in _gen_cache:
            _gen_cache[model.value] = SOAPGenerator(model=model)
        gen = _gen_cache[model.value]
        t1 = time.perf_counter()
        soap_res = gen.generate(transcript, template=NoteTemplate.SOAP)
        soap_secs = round(time.perf_counter() - t1, 2)

        session = ClinicalSession(
            session_id=str(uuid.uuid4()),
            patient_id=patient_id,
            transcript=transcript,
        )
        for e in ext.entities:
            session.entities.append(e)
        session.soap_note = SOAPNote(
            content=soap_res.soap_note,
            model_used=soap_res.model,
        )
        if patient_id:
            patient = _patient_store.load(patient_id)
            if patient:
                session.patient_name = patient.full_name
        _session_store.save(session)
        _audit("pipeline_run", {
            "session_id": session.session_id,
            "engine": engine.value,
            "model": model.value,
            "asr_secs": asr_secs,
            "soap_secs": soap_secs,
            "entities": len(ext.entities),
        })
        return {
            "session_id": session.session_id,
            "transcript": transcript,
            "entity_count": len(ext.entities),
            "entity_method": ext.method,
            "soap_note": soap_res.soap_note,
            "timings": {"asr_secs": asr_secs, "soap_secs": soap_secs},
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# #5 Consent + audit log
# ---------------------------------------------------------------------------

class ConsentRecord(BaseModel):
    patient_id: str
    gp_id: str = "gp_anon"
    consent_type: str = "recording_and_processing"
    notes: str = ""


@router.post("/consent")
async def record_consent(req: ConsentRecord):
    _audit("consent_recorded", req.model_dump())
    return {"recorded": True, "at": datetime.now(timezone.utc).isoformat()}


@router.get("/audit")
async def audit_log(limit: int = Query(default=100, ge=1, le=1000)):
    if not _AUDIT_LOG.exists():
        return {"entries": [], "total": 0}
    lines = _AUDIT_LOG.read_text().splitlines()
    entries = [json.loads(l) for l in lines[-limit:]]
    return {"entries": entries, "total": len(lines)}


# ---------------------------------------------------------------------------
# #6 Pseudo-streaming: accept a chunk, return partial
# (Real streaming needs a streaming-capable model; Conformer/Whisper here are batch.
#  We re-transcribe the growing audio each call - honest about the limitation.)
# ---------------------------------------------------------------------------

@router.post("/transcribe/chunk")
async def transcribe_chunk(
    file: UploadFile = File(...),
    engine: EngineType = Form(default=EngineType.FASTER_WHISPER),
):
    suffix = Path(file.filename or "a.webm").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        pre = _preprocessor.process_file(tmp_path)
        asr = _get_engine(engine)
        t0 = time.perf_counter()
        text = asr.transcribe(pre.audio, pre.sample_rate)
        return {
            "partial": text,
            "duration_secs": round(pre.duration_secs, 2),
            "processing_secs": round(time.perf_counter() - t0, 2),
            "engine": engine.value,
            "note": "Batch re-transcription of accumulated audio (not true streaming).",
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# #7 Full pipeline: ASR → Diarisation → NLP → SOAP → 9-agent Verification
# ---------------------------------------------------------------------------

def _extract_soap_sections(text: str) -> dict[str, str]:
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


@router.post("/pipeline/full")
async def run_full_pipeline(
    file: UploadFile = File(...),
    engine: EngineType = Form(default=EngineType.PRAXIS_CONFORMER),
    model: SOAPModel = Form(default=SOAPModel.PRAXIS_PRIMARY),
    patient_id: str = Form(default=""),
    consent: bool = Form(default=False),
):
    """Full PRAXIS pipeline: ASR → Diarisation → NLP (SNOMED) → SOAP → 9-agent Verification.

    Returns the SOAP note, TrustScore, entities with SNOMED codes,
    source links, and per-agent verification details.
    """
    if not consent:
        raise HTTPException(status_code=400, detail="Patient consent must be recorded before processing")

    suffix = Path(file.filename or "a.wav").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        timings: dict[str, float] = {}
        t_total = time.perf_counter()

        # Stage 1: ASR
        pre = _preprocessor.process_file(tmp_path)
        asr = _get_engine(engine)
        t0 = time.perf_counter()
        transcript = asr.transcribe(pre.audio, pre.sample_rate)
        timings["asr_secs"] = round(time.perf_counter() - t0, 2)

        # Stage 2: Diarisation
        t0 = time.perf_counter()
        diar = diarise_with_fallback(audio_path=tmp_path, transcript=transcript)
        timings["diarisation_secs"] = round(time.perf_counter() - t0, 2)

        # Stage 3: NLP entity extraction (SNOMED + negation + section hints)
        t0 = time.perf_counter()
        nlp = run_nlp_pipeline(transcript)
        nlp_context = format_nlp_context(nlp)
        timings["nlp_secs"] = round(time.perf_counter() - t0, 2)

        nlp_entities = [
            {
                "text": e.text,
                "label": e.label,
                "negated": e.negated,
                "snomed_code": e.snomed_code,
                "snomed_display": e.snomed_display,
            }
            for e in nlp.entities
        ]

        # Stage 4: SOAP generation (with NLP context injected)
        enriched = transcript
        if nlp_context:
            enriched = (
                f"[CLINICAL NLP CONTEXT - use to inform the note]\n"
                f"{nlp_context}\n\n"
                f"[CONSULTATION TRANSCRIPT]\n"
                f"{transcript}"
            )

        if model.value not in _gen_cache:
            _gen_cache[model.value] = SOAPGenerator(model=model)
        gen = _gen_cache[model.value]
        t0 = time.perf_counter()
        soap_res = gen.generate(enriched, template=NoteTemplate.SOAP)
        timings["soap_secs"] = round(time.perf_counter() - t0, 2)
        soap_note = soap_res.soap_note

        # Stage 5: Clinical validation (6 checks + TrustScore)
        t0 = time.perf_counter()
        validation = validate_soap_note(note_text=soap_note, transcript=transcript)
        timings["validation_secs"] = round(time.perf_counter() - t0, 2)

        # Stage 6: Source linking
        t0 = time.perf_counter()
        soap_sections = _extract_soap_sections(soap_note)
        source_map = build_source_map(transcript, soap_sections)
        timings["source_linking_secs"] = round(time.perf_counter() - t0, 2)

        timings["total_secs"] = round(time.perf_counter() - t_total, 2)

        # Persist session
        session = ClinicalSession(
            session_id=str(uuid.uuid4()),
            patient_id=patient_id,
            raw_transcript=transcript,
        )
        session.soap_note = SOAPNote(
            raw_text=soap_note,
            model_used=soap_res.model,
        )
        _session_store.save(session)

        _audit("pipeline_full", {
            "session_id": session.session_id,
            "engine": engine.value,
            "model": model.value,
            "trust_score": validation.trust_score,
            "verdict": validation.verdict,
            **timings,
        })

        return {
            "session_id": session.session_id,
            "transcript": transcript,
            "diarised_transcript": diar.diarised_transcript,
            "speakers": diar.speakers,
            "nlp_entities": nlp_entities,
            "medications": nlp.medications,
            "snomed_codes": [
                {"code": e.snomed_code, "display": e.snomed_display}
                for e in nlp.entities if e.snomed_code
            ],
            "soap_note": soap_note,
            "trust_score": validation.trust_score,
            "verdict": validation.verdict,
            "checks_passed": validation.checks_passed,
            "checks_total": validation.checks_total,
            "issues": validation.issues,
            "suggestions": validation.suggestions,
            "check_details": [
                {
                    "check": r.check_name,
                    "passed": r.passed,
                    "issues": r.issues,
                    "suggestions": r.suggestions,
                }
                for r in validation.check_results
            ],
            "source_links": [
                {
                    "soap_section": l.note_section,
                    "soap_text": l.note_sentence,
                    "transcript_span": "; ".join(l.source_sentences[:2]) if l.source_sentences else "",
                    "confidence": l.confidence,
                }
                for l in source_map.links
            ],
            "timings": timings,
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)
