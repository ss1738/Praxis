"""Run MedASR vs Whisper on PriMock57 test set, compare WER side by side."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import jiwer
import librosa
import torch
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console(width=100)


def _normalise(text: str) -> str:
    text = text.lower()
    text = text.replace("</s>", "")
    text = re.sub(r"[^ a-z0-9']", " ", text)
    return " ".join(text.split())


def load_test_samples() -> list[dict]:
    with open("data/processed/primock57/test.json") as f:
        return json.load(f)


def transcribe_whisper(audio: "np.ndarray", processor, model) -> tuple[str, float]:
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    t0 = time.perf_counter()
    with torch.no_grad():
        generated_ids = model.generate(**inputs, language="en", task="transcribe")
    elapsed = time.perf_counter() - t0
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip(), elapsed


def transcribe_medasr(audio: "np.ndarray", processor, model) -> tuple[str, float]:
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(**inputs).logits
    elapsed = time.perf_counter() - t0

    # CTC decode: argmax, collapse repeats, skip blanks
    pred_ids = torch.argmax(logits, dim=-1)[0]
    collapsed = []
    prev = -1
    for idx in pred_ids.tolist():
        if idx != prev and idx != 0:  # 0 = <epsilon> = CTC blank
            collapsed.append(idx)
        prev = idx
    text = processor.tokenizer.decode(collapsed)
    return text.strip(), elapsed


def main():
    console.print("\n[bold cyan]═══ AI Medical Notes: Baseline Validation ═══[/bold cyan]")
    console.print("[bold cyan]═══ MedASR vs Whisper Medium on PriMock57 (British GP) ═══[/bold cyan]\n")

    records = load_test_samples()

    # --- Load both models once ---
    console.print("[bold]Loading Whisper Medium...[/bold]")
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    w_processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
    w_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
    console.print("[green]Whisper Medium loaded[/green]")

    console.print("[bold]Loading MedASR...[/bold]")
    from transformers import AutoModelForCTC, AutoProcessor
    m_processor = AutoProcessor.from_pretrained("google/medasr")
    m_model = AutoModelForCTC.from_pretrained("google/medasr")
    console.print("[green]MedASR loaded[/green]\n")

    # --- Process each test sample (first 30s to keep it fast) ---
    MAX_DUR = 30.0
    results = []

    for i, rec in enumerate(records):
        audio_path = rec["audio_path"]
        fname = Path(audio_path).name
        full_ref = rec["text"]
        full_dur = librosa.get_duration(path=audio_path)
        segment_dur = min(MAX_DUR, full_dur)

        console.print(f"\n[bold]─── Sample {i+1}/{len(records)}: {fname} ({full_dur:.0f}s, using {segment_dur:.0f}s) ───[/bold]")

        # Load audio segment
        audio, sr = librosa.load(audio_path, sr=16000, duration=MAX_DUR)

        # Approximate reference for the segment
        words_per_sec = max(len(full_ref.split()) / full_dur, 1)
        approx_words = int(words_per_sec * segment_dur)
        ref_segment = " ".join(full_ref.split()[:approx_words])

        # Whisper
        w_text, w_time = transcribe_whisper(audio, w_processor, w_model)
        w_ref_norm = _normalise(ref_segment)
        w_hyp_norm = _normalise(w_text)
        w_wer = jiwer.wer(w_ref_norm, w_hyp_norm) if w_ref_norm and w_hyp_norm else 1.0
        w_cer = jiwer.cer(w_ref_norm, w_hyp_norm) if w_ref_norm and w_hyp_norm else 1.0

        # MedASR
        m_text, m_time = transcribe_medasr(audio, m_processor, m_model)
        m_ref_norm = _normalise(ref_segment)
        m_hyp_norm = _normalise(m_text)
        m_wer = jiwer.wer(m_ref_norm, m_hyp_norm) if m_ref_norm and m_hyp_norm else 1.0
        m_cer = jiwer.cer(m_ref_norm, m_hyp_norm) if m_ref_norm and m_hyp_norm else 1.0

        results.append({
            "file": fname,
            "duration": segment_dur,
            "ref_preview": ref_segment[:100],
            "whisper": {"text": w_text, "wer": w_wer, "cer": w_cer, "time": w_time},
            "medasr": {"text": m_text, "wer": m_wer, "cer": m_cer, "time": m_time},
        })

        # Side-by-side preview
        console.print(f"  [green]REF:[/green] {ref_segment[:120]}...")
        console.print(f"  [yellow]WHI:[/yellow] {w_text[:120]}...")
        console.print(f"  [magenta]MED:[/magenta] {m_text[:120]}...")
        console.print(f"  WER: Whisper={w_wer:.1%}  MedASR={m_wer:.1%}  |  Time: Whisper={w_time:.1f}s  MedASR={m_time:.1f}s")

    # --- Summary table ---
    console.print("\n")
    table = Table(title="Per-Sample WER Comparison", show_header=True, show_lines=True)
    table.add_column("File", style="bold", max_width=35)
    table.add_column("Dur", justify="right")
    table.add_column("Whisper WER", justify="right")
    table.add_column("MedASR WER", justify="right")
    table.add_column("Winner", justify="center")

    for r in results:
        w_wer = r["whisper"]["wer"]
        m_wer = r["medasr"]["wer"]
        if w_wer < m_wer:
            winner = "[yellow]Whisper[/yellow]"
        elif m_wer < w_wer:
            winner = "[magenta]MedASR[/magenta]"
        else:
            winner = "Tie"
        table.add_row(
            r["file"],
            f"{r['duration']:.0f}s",
            f"{w_wer:.1%}",
            f"{m_wer:.1%}",
            winner,
        )

    # Averages
    avg_w = sum(r["whisper"]["wer"] for r in results) / len(results)
    avg_m = sum(r["medasr"]["wer"] for r in results) / len(results)
    avg_w_time = sum(r["whisper"]["time"] for r in results) / len(results)
    avg_m_time = sum(r["medasr"]["time"] for r in results) / len(results)

    table.add_row(
        "[bold]AVERAGE[/bold]",
        "",
        f"[bold]{avg_w:.1%}[/bold]",
        f"[bold]{avg_m:.1%}[/bold]",
        "[bold yellow]Whisper[/bold yellow]" if avg_w < avg_m else "[bold magenta]MedASR[/bold magenta]",
    )
    console.print(table)

    # Timing summary
    console.print(f"\n[dim]Avg inference time (30s audio): Whisper={avg_w_time:.2f}s, MedASR={avg_m_time:.2f}s[/dim]")

    # Key insight
    console.print(Panel(
        f"[bold]Whisper Medium avg WER: {avg_w:.1%}[/bold]\n"
        f"[bold]MedASR avg WER: {avg_m:.1%}[/bold]\n\n"
        "Note: MedASR is trained on US medical dictation (radiology, clinical notes).\n"
        "PriMock57 contains British English GP conversations, a domain mismatch.\n"
        "Whisper is a general-purpose multilingual model with strong British English support.\n\n"
        "The QLoRA fine-tuning of Whisper on PriMock57 should further reduce Whisper's WER.",
        title="Baseline Summary",
        border_style="cyan",
    ))

    # Save results
    out = Path("data/processed/baseline_results.json")
    out.write_text(json.dumps(results, indent=2, default=str))
    console.print(f"[dim]Full results saved to {out}[/dim]")


if __name__ == "__main__":
    main()
