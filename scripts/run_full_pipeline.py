"""Run the COMPLETE PRAXIS pipeline on an audio file and save all artifacts.

Executes: Audio → Preprocessing → ASR → Diarization → NLP → SOAP → Validation
Saves every intermediate result to a timestamped directory.

Usage:
    python3 scripts/run_full_pipeline.py path/to/audio.wav
    python3 scripts/run_full_pipeline.py path/to/audio.wav --offline
    python3 scripts/run_full_pipeline.py path/to/audio.wav --engine praxis_conformer --model praxis-soap-v1
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "pipeline_runs"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _make_run_dir() -> Path:
    """Create a timestamped run directory."""
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = DATA_DIR / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_pipeline(
    audio_path: str,
    engine_name: str = "praxis_conformer",
    model_name: str = "praxis-soap-v1",
    offline: bool = False,
) -> Path:
    """Run the full PRAXIS pipeline and save all artifacts.

    Returns the path to the run directory containing all outputs.
    """
    console = Console()
    run_dir = _make_run_dir()
    timings: dict[str, float] = {}
    metadata: dict = {
        "input_audio": str(audio_path),
        "engine": engine_name,
        "soap_model": model_name,
        "offline_mode": offline,
        "started_at": datetime.now().isoformat(),
    }

    console.print(Panel(
        f"[bold]PRAXIS Full Pipeline[/bold]\n"
        f"Audio: {audio_path}\n"
        f"Engine: {engine_name} | SOAP: {model_name}\n"
        f"Output: {run_dir}",
        title="Pipeline Run",
        border_style="blue",
    ))

    t_total = time.perf_counter()

    # ── Stage 0: Copy input audio ──
    console.print("\n[bold cyan]Stage 0:[/bold cyan] Copying input audio...")
    input_copy = run_dir / f"input_audio{Path(audio_path).suffix}"
    shutil.copy2(audio_path, input_copy)

    # ── Stage 1: Audio preprocessing ──
    console.print("[bold cyan]Stage 1:[/bold cyan] Audio preprocessing...")
    from src.speech_to_text.audio_preprocessor import AudioPreprocessor

    t0 = time.perf_counter()
    preprocessor = AudioPreprocessor()
    processed = preprocessor.process_file(audio_path)
    timings["preprocessing_secs"] = round(time.perf_counter() - t0, 3)

    preprocessed_path = run_dir / "preprocessed_audio.wav"
    sf.write(str(preprocessed_path), processed.audio, processed.sample_rate)
    console.print(f"  Duration: {processed.duration_secs:.1f}s, SR: {processed.sample_rate}Hz")

    # ── Stage 2: ASR ──
    console.print(f"[bold cyan]Stage 2:[/bold cyan] Speech-to-text ({engine_name})...")
    from src.speech_to_text.engine_factory import EngineType, create_engine

    t0 = time.perf_counter()
    engine = create_engine(EngineType(engine_name))
    engine.load_model()
    transcript = engine.transcribe(processed.audio, processed.sample_rate)
    timings["asr_secs"] = round(time.perf_counter() - t0, 3)

    asr_result = {
        "text": transcript,
        "engine": engine_name,
        "audio_duration_secs": round(processed.duration_secs, 2),
        "inference_secs": timings["asr_secs"],
        "rtf": round(timings["asr_secs"] / max(processed.duration_secs, 0.01), 3),
    }
    (run_dir / "asr_transcript.json").write_text(json.dumps(asr_result, indent=2))
    console.print(f"  Transcript: {transcript[:120]}...")
    console.print(f"  Time: {timings['asr_secs']:.2f}s (RTF: {asr_result['rtf']:.3f})")

    # ── Stage 3: Diarization ──
    console.print("[bold cyan]Stage 3:[/bold cyan] Speaker diarization...")
    from src.speech_to_text.speaker_diarisation import diarise_with_fallback

    t0 = time.perf_counter()
    diar = diarise_with_fallback(audio_path=audio_path, transcript=transcript)
    timings["diarization_secs"] = round(time.perf_counter() - t0, 3)

    diar_result = {
        "speakers": diar.speakers,
        "num_segments": len(diar.segments) if diar.segments else 0,
        "diarised_transcript": diar.diarised_transcript,
        "method": diar.method,
        "inference_secs": timings["diarization_secs"],
        "segments": [
            {
                "speaker": s.speaker,
                "start": s.start,
                "end": s.end,
                "text": s.text,
            }
            for s in (diar.segments or [])
        ],
    }
    (run_dir / "diarization.json").write_text(json.dumps(diar_result, indent=2))
    console.print(f"  Speakers: {diar.speakers}, Segments: {diar_result['num_segments']}, Method: {diar.method}")

    # ── Stage 4: NLP entity extraction ──
    console.print("[bold cyan]Stage 4:[/bold cyan] Clinical NLP (SNOMED + negation)...")
    from src.clinical.nlp_pipeline import run_nlp_pipeline, format_nlp_context

    t0 = time.perf_counter()
    nlp = run_nlp_pipeline(transcript)
    timings["nlp_secs"] = round(time.perf_counter() - t0, 3)

    nlp_result = {
        "entities": [
            {
                "text": e.text,
                "label": e.label,
                "negated": e.negated,
                "snomed_code": e.snomed_code,
                "snomed_display": e.snomed_display,
            }
            for e in nlp.entities
        ],
        "medications": nlp.medications,
        "section_hints": nlp.section_hints,
        "num_entities": len(nlp.entities),
        "num_snomed": sum(1 for e in nlp.entities if e.snomed_code),
        "num_negated": sum(1 for e in nlp.entities if e.negated),
        "inference_secs": timings["nlp_secs"],
    }
    (run_dir / "nlp_entities.json").write_text(json.dumps(nlp_result, indent=2))
    console.print(
        f"  Entities: {nlp_result['num_entities']}, "
        f"SNOMED: {nlp_result['num_snomed']}, "
        f"Medications: {len(nlp.medications)}"
    )

    # ── Stage 5: SOAP generation ──
    console.print(f"[bold cyan]Stage 5:[/bold cyan] SOAP note generation ({model_name})...")
    from src.soap_generation.generator import SOAPGenerator, SOAPModel
    from src.soap_generation.templates import NoteTemplate

    nlp_context = format_nlp_context(nlp)
    enriched = transcript
    if nlp_context:
        enriched = (
            f"[CLINICAL NLP CONTEXT - use to inform the note]\n"
            f"{nlp_context}\n\n"
            f"[CONSULTATION TRANSCRIPT]\n"
            f"{transcript}"
        )

    soap_note_text = ""
    soap_model_used = model_name

    if offline:
        console.print("  [yellow]Offline mode - using template-based generation[/yellow]")
        soap_note_text = _generate_offline_soap(transcript, nlp)
        soap_model_used = "template-based (offline)"
        timings["soap_secs"] = 0.0
    else:
        try:
            t0 = time.perf_counter()
            soap_model = SOAPModel(model_name)
            gen = SOAPGenerator(model=soap_model)
            if not gen.check_ollama():
                console.print("  [yellow]Ollama not available - falling back to offline mode[/yellow]")
                soap_note_text = _generate_offline_soap(transcript, nlp)
                soap_model_used = "template-based (offline fallback)"
                timings["soap_secs"] = round(time.perf_counter() - t0, 3)
            else:
                result = gen.generate(enriched, template=NoteTemplate.SOAP)
                soap_note_text = result.soap_note
                timings["soap_secs"] = round(time.perf_counter() - t0, 3)
        except Exception as e:
            console.print(f"  [red]SOAP generation failed: {e}[/red]")
            soap_note_text = _generate_offline_soap(transcript, nlp)
            soap_model_used = "template-based (error fallback)"
            timings["soap_secs"] = 0.0

    soap_result = {
        "soap_note": soap_note_text,
        "model": soap_model_used,
        "inference_secs": timings.get("soap_secs", 0.0),
    }
    (run_dir / "soap_note.json").write_text(json.dumps(soap_result, indent=2))
    (run_dir / "soap_note_rendered.txt").write_text(soap_note_text)
    console.print(f"  Model: {soap_model_used}")
    console.print(f"  Note length: {len(soap_note_text)} chars")

    # ── Stage 6: Validation ──
    console.print("[bold cyan]Stage 6:[/bold cyan] Clinical validation (6 checks)...")
    from src.clinical.verification import validate_soap_note

    t0 = time.perf_counter()
    validation = validate_soap_note(note_text=soap_note_text, transcript=transcript)
    timings["validation_secs"] = round(time.perf_counter() - t0, 3)

    val_result = validation.to_dict()
    val_result["inference_secs"] = timings["validation_secs"]
    (run_dir / "validation_result.json").write_text(json.dumps(val_result, indent=2))

    trust_color = "green" if validation.trust_score >= 70 else ("yellow" if validation.trust_score >= 40 else "red")
    console.print(
        f"  TrustScore: [{trust_color}]{validation.trust_score}[/{trust_color}] "
        f"({validation.verdict}) - "
        f"{validation.checks_passed}/{validation.checks_total} checks passed"
    )

    # ── Save metadata ──
    timings["total_secs"] = round(time.perf_counter() - t_total, 3)
    metadata["completed_at"] = datetime.now().isoformat()
    metadata["timings"] = timings
    (run_dir / "pipeline_metadata.json").write_text(json.dumps(metadata, indent=2))

    # ── Summary ──
    console.print()
    table = Table(title="Pipeline Summary", border_style="blue")
    table.add_column("Stage", style="cyan")
    table.add_column("Time", justify="right")
    table.add_column("Key Result")

    table.add_row("Preprocessing", f"{timings['preprocessing_secs']:.3f}s", f"{processed.duration_secs:.1f}s audio")
    table.add_row("ASR", f"{timings['asr_secs']:.3f}s", f"{len(transcript.split())} words")
    table.add_row("Diarization", f"{timings['diarization_secs']:.3f}s", f"{len(diar.speakers)} speakers")
    table.add_row("NLP", f"{timings['nlp_secs']:.3f}s", f"{nlp_result['num_entities']} entities, {nlp_result['num_snomed']} SNOMED")
    table.add_row("SOAP", f"{timings.get('soap_secs', 0):.3f}s", f"{len(soap_note_text)} chars")
    table.add_row("Validation", f"{timings['validation_secs']:.3f}s", f"TrustScore={validation.trust_score}")
    table.add_row("[bold]Total[/bold]", f"[bold]{timings['total_secs']:.3f}s[/bold]", "")

    console.print(table)
    console.print(f"\n[bold green]All artifacts saved to:[/bold green] {run_dir}")

    return run_dir


def _generate_offline_soap(transcript: str, nlp) -> str:
    """Generate a template-based SOAP note without an LLM (offline mode).

    Fills in SOAP sections using NLP-extracted entities and transcript content.
    """
    entities_text = ", ".join(
        f"{e.text} [SNOMED: {e.snomed_code}]" if e.snomed_code else e.text
        for e in nlp.entities if not e.negated
    ) or "See transcript"

    negated_text = ", ".join(
        e.text for e in nlp.entities if e.negated
    )

    meds_text = ", ".join(nlp.medications) if nlp.medications else "None documented"

    # Truncate transcript for subjective section
    words = transcript.split()
    subjective = " ".join(words[:100]) + ("..." if len(words) > 100 else "")

    return (
        f"SUBJECTIVE:\n"
        f"  {subjective}\n\n"
        f"OBJECTIVE:\n"
        f"  [Template-based - no examination findings extracted from audio]\n"
        f"  Medications: {meds_text}\n\n"
        f"ASSESSMENT:\n"
        f"  Clinical findings: {entities_text}\n"
        + (f"  Negated: {negated_text}\n" if negated_text else "")
        + f"\nPLAN:\n"
        f"  [Template-based - requires clinician review]\n"
        f"  Safety-netting: Return if symptoms worsen or new symptoms develop. "
        f"Call 111 for urgent advice out of hours.\n"
    )


def main():
    ap = argparse.ArgumentParser(
        description="Run the full PRAXIS pipeline on an audio file"
    )
    ap.add_argument("audio", type=str, help="Path to audio file (.wav)")
    ap.add_argument("--engine", type=str, default="praxis_conformer",
                     choices=["medasr", "whisper", "faster_whisper", "praxis_conformer"])
    ap.add_argument("--model", type=str, default="praxis-soap-v1")
    ap.add_argument("--offline", action="store_true",
                     help="Use template-based SOAP (no Ollama required)")
    args = ap.parse_args()

    if not Path(args.audio).exists():
        logger.error(f"Audio file not found: {args.audio}")
        raise SystemExit(1)

    run_pipeline(args.audio, args.engine, args.model, args.offline)


if __name__ == "__main__":
    main()
