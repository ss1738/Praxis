"""Extra feature routes: entities, status, benchmark, export."""

from __future__ import annotations

import html
import json
import time
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.clinical.entity_extraction import ClinicalEntityExtractor
from src.clinical.session_store import SessionStore

router = APIRouter(prefix="/api/v1", tags=["features"])

_REPO_ROOT = Path(__file__).resolve().parents[3]

_extractor = ClinicalEntityExtractor()
_session_store = SessionStore()


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

class EntitiesRequest(BaseModel):
    transcript: str = Field(..., min_length=1)


@router.post("/entities")
async def extract_entities(req: EntitiesRequest):
    result = _extractor.extract(req.transcript)
    return {
        "method": result.method,
        "num_snomed_linked": result.num_snomed_linked,
        "num_negated": result.num_negated,
        "entities": [
            {
                "text": e.text,
                "type": e.entity_type.value,
                "confidence": e.confidence.value,
                "snomed_code": e.snomed_code,
                "snomed_term": e.snomed_term,
                "negated": e.negated,
            }
            for e in result.entities
        ],
    }


# ---------------------------------------------------------------------------
# Pipeline status dashboard
# ---------------------------------------------------------------------------

@router.get("/status")
async def pipeline_status():
    """Live health of each model/service. Truthful - no green lies."""
    status: dict = {}

    # Conformer
    best = _REPO_ROOT / "data/models/conformer_v2/best_model_v2_80M.pt"
    final = _REPO_ROOT / "data/models/conformer_v2/final_model_v2_80M.pt"
    active = best if best.exists() else final
    status["praxis_conformer"] = {
        "loaded": active.exists(),
        "path": str(active),
        "checkpoint": "best_val_loss" if active == best else "final_epoch",
        "note": "117M custom from-scratch Conformer-CTC (trained on UK English + LibriSpeech)",
    }

    # Whisper dirs
    for name, d in [
        ("whisper_small_british", "data/models/whisper-small-british-medical-v2"),
        ("whisper_large_v3_british", "data/models/whisper-british-medical-v3"),
        ("medasr_british", "data/models/medasr-british-medical"),
    ]:
        p = _REPO_ROOT / d
        status[name] = {"loaded": p.exists(), "path": str(p)}

    # MedCAT
    medcat_pack = _REPO_ROOT / "data/models/medcat_snomed"
    status["medcat"] = {
        "loaded": medcat_pack.exists(),
        "path": str(medcat_pack),
        "fallback": "regex" if not medcat_pack.exists() else None,
        "note": "Awaiting UMLS licence for SNOMED pack" if not medcat_pack.exists() else "Active",
    }

    # medspaCy
    try:
        import medspacy  # noqa: F401
        status["medspacy"] = {"loaded": True}
    except ImportError:
        status["medspacy"] = {"loaded": False}

    # Pyannote
    try:
        import pyannote.audio  # noqa: F401
        import os
        status["pyannote"] = {
            "installed": True,
            "hf_token_set": bool(os.environ.get("HF_TOKEN")),
            "note": "Licence acceptance required on HF for 3.1 model",
        }
    except ImportError:
        status["pyannote"] = {"installed": False, "fallback": "lexical keywords"}

    # Ollama
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=1.5)
        models = [m["name"] for m in r.json().get("models", [])]
        status["ollama"] = {"loaded": True, "models": models}
    except Exception as e:
        status["ollama"] = {"loaded": False, "error": str(e)[:80]}

    return status


# ---------------------------------------------------------------------------
# Benchmark (cached report or quick live run)
# ---------------------------------------------------------------------------

_BENCHMARK_CACHE = _REPO_ROOT / "data/benchmarks/latest_report.json"


@router.get("/benchmark")
async def get_benchmark():
    """Return last benchmark report. Run via scripts/run_benchmark.py to refresh."""
    if _BENCHMARK_CACHE.exists():
        return json.loads(_BENCHMARK_CACHE.read_text())
    return {
        "status": "no_report",
        "note": "Run scripts/benchmark_engines.py to produce data/benchmarks/latest_report.json",
        "placeholder": [
            {"engine": "praxis_conformer", "wer": None, "cer": None, "rtf": None},
            {"engine": "whisper_small", "wer": None, "cer": None, "rtf": None},
            {"engine": "faster_whisper", "wer": None, "cer": None, "rtf": None},
            {"engine": "medasr", "wer": None, "cer": None, "rtf": None},
        ],
    }


class QuickBenchRequest(BaseModel):
    sample_count: int = Field(default=3, ge=1, le=10)
    engines: list[str] = Field(default_factory=lambda: ["praxis_conformer", "whisper"])


@router.post("/benchmark/quick")
async def run_quick_benchmark(req: QuickBenchRequest):
    """Quick live WER comparison on a handful of primock57 samples.

    Expensive - loads each engine. Cap at 10 samples.
    """
    import jiwer, re as _re
    from src.speech_to_text.engine_factory import EngineType, create_engine

    def _norm(t):
        t = t.lower()
        t = _re.sub(r"[^\w\s]", "", t)
        return _re.sub(r"\s+", " ", t).strip()

    audio_dir = _REPO_ROOT / "data/raw/primock57/audio"
    transcript_dir = _REPO_ROOT / "data/raw/primock57/transcripts"
    if not audio_dir.exists():
        raise HTTPException(status_code=404, detail="primock57 audio not found")

    wavs = sorted(audio_dir.glob("*_doctor.wav"))[: req.sample_count]
    samples = []
    for wav in wavs:
        tg = transcript_dir / f"{wav.stem}.TextGrid"
        if not tg.exists():
            continue
        reference = _textgrid_to_text(tg)
        samples.append((str(wav), reference))

    if not samples:
        raise HTTPException(status_code=404, detail="No aligned samples found")

    results = []
    for engine_name in req.engines:
        try:
            et = EngineType(engine_name)
        except ValueError:
            continue
        eng = create_engine(et)
        eng.load_model()
        wers, cers = [], []
        t0 = time.perf_counter()
        for wav_path, ref in samples:
            hyp = eng.transcribe_file(wav_path)
            rn, hn = _norm(ref), _norm(hyp)
            if rn:
                wers.append(jiwer.wer(rn, hn))
                cers.append(jiwer.cer(rn, hn))
        total = time.perf_counter() - t0
        results.append({
            "engine": engine_name,
            "samples": len(samples),
            "mean_wer": round(sum(wers)/max(len(wers),1), 4),
            "mean_cer": round(sum(cers)/max(len(cers),1), 4),
            "total_inference_secs": round(total, 2),
        })

    _BENCHMARK_CACHE.parent.mkdir(parents=True, exist_ok=True)
    payload = {"sample_count": len(samples), "results": results, "ran_at": time.time()}
    _BENCHMARK_CACHE.write_text(json.dumps(payload, indent=2))
    return payload


def _textgrid_to_text(tg_path: Path) -> str:
    """Crude TextGrid → plain text."""
    txt = tg_path.read_text(errors="ignore")
    import re
    words = re.findall(r'text = "([^"]+)"', txt)
    return " ".join(w for w in words if w and w.strip() not in {".", ","})


# ---------------------------------------------------------------------------
# Session export as printable HTML (browser can save to PDF)
# ---------------------------------------------------------------------------

@router.get("/sessions/{session_id}/export", response_class=HTMLResponse)
async def export_session(session_id: str):
    session = _session_store.load(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    transcript = html.escape(session.get("transcript", ""))
    soap = html.escape(session.get("soap_note", ""))
    patient_name = html.escape(session.get("patient_name", "Unknown"))
    created = html.escape(session.get("created_at", ""))
    entities = session.get("entities", [])

    ent_rows = "".join(
        f"<tr><td>{html.escape(str(e.get('text','')))}</td>"
        f"<td>{html.escape(str(e.get('type','')))}</td>"
        f"<td>{html.escape(str(e.get('snomed_code','')))}</td>"
        f"<td>{'Yes' if e.get('negated') else ''}</td></tr>"
        for e in entities
    )

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>PRAXIS Session {session_id}</title>
<style>
body {{ font-family: Georgia, serif; max-width: 780px; margin: 40px auto; color: #111; line-height: 1.5; }}
h1 {{ border-bottom: 2px solid #111; padding-bottom: 8px; }}
h2 {{ margin-top: 28px; color: #1e3a8a; }}
pre {{ white-space: pre-wrap; background: #f5f5f5; padding: 12px; border-radius: 4px; }}
table {{ border-collapse: collapse; width: 100%; }}
td, th {{ border: 1px solid #ccc; padding: 6px 8px; text-align: left; font-size: 13px; }}
.meta {{ color: #555; font-size: 14px; }}
@media print {{ body {{ margin: 20px; }} }}
</style></head><body>
<h1>PRAXIS Clinical Note</h1>
<p class="meta"><strong>Patient:</strong> {patient_name} &nbsp;|&nbsp;
<strong>Session:</strong> {session_id} &nbsp;|&nbsp;
<strong>Recorded:</strong> {created}</p>

<h2>Transcript</h2>
<pre>{transcript}</pre>

<h2>SOAP Note</h2>
<pre>{soap}</pre>

<h2>Clinical Entities ({len(entities)})</h2>
<table><thead><tr><th>Term</th><th>Type</th><th>SNOMED</th><th>Negated</th></tr></thead>
<tbody>{ent_rows or '<tr><td colspan="4"><em>No entities recorded</em></td></tr>'}</tbody></table>

<p class="meta" style="margin-top:40px;">Generated by PRAXIS · University of Leicester MSc dissertation ·
<em>Use your browser's Print → Save as PDF.</em></p>
</body></html>"""
