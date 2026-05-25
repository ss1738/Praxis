"""Generate a synthetic 2-speaker GP consultation audio for demos.

Uses edge-tts (Microsoft Edge TTS, free, no API key) to generate a ~2 minute
consultation between a doctor and patient discussing a hypertension review
with medication adjustment.

Output: data/samples/sample_consultation.wav

Usage:
    python3 scripts/generate_sample.py
    python3 scripts/generate_sample.py --output path/to/output.wav

Requires: pip install edge-tts soundfile numpy
"""

from __future__ import annotations

import argparse
import asyncio
import io
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "data" / "samples" / "sample_consultation.wav"

# British English voices - doctor (male) and patient (female)
DOCTOR_VOICE = "en-GB-RyanNeural"
PATIENT_VOICE = "en-GB-SoniaNeural"
TARGET_SR = 16000

# Realistic GP consultation: hypertension review with medication adjustment
CONSULTATION_SCRIPT = [
    ("doctor", "Good morning, Mrs. Thompson. Please take a seat. How are you doing today?"),
    ("patient", "Morning, Doctor. I'm alright, thanks. I've come in for my blood pressure check."),
    ("doctor", "Of course. I can see from your records that you're due for your annual hypertension review. How have you been feeling generally? Any headaches, dizziness, or visual disturbances?"),
    ("patient", "No headaches or anything like that. I've been feeling a bit more tired than usual over the last few weeks though. Nothing major, just not quite myself."),
    ("doctor", "OK. And are you still taking the ramipril five milligrams once daily?"),
    ("patient", "Yes, every morning with breakfast. I haven't missed any doses."),
    ("doctor", "That's good. Any side effects you've noticed? Dry cough, or any swelling in your ankles?"),
    ("patient", "Actually, I have had a bit of a dry cough. It started maybe six weeks ago. I thought it was just a cold but it hasn't gone away."),
    ("doctor", "That's quite a common side effect of ramipril, the ACE inhibitor. We may need to consider switching you to a different medication. Let me check your blood pressure now."),
    ("patient", "OK, sure."),
    ("doctor", "Right. Your blood pressure today is one hundred and forty-eight over ninety-two. That's a bit higher than we'd like. Our target for you is below one forty over ninety."),
    ("patient", "Oh, that's not great, is it?"),
    ("doctor", "It's not at a dangerous level, but it does suggest we need to adjust your treatment. Given the cough, which is likely from the ramipril, I'd like to switch you to losartan. It's an ARB, a similar type of medication but much less likely to cause a cough."),
    ("patient", "What dose would that be?"),
    ("doctor", "I'll start you on losartan fifty milligrams once daily. Same time as you were taking the ramipril, with breakfast. We'll review your blood pressure in four weeks to see if we need to increase the dose."),
    ("patient", "And the cough should go away?"),
    ("doctor", "Yes, it typically resolves within a few weeks of stopping the ACE inhibitor. Now, can I also check - are you still taking atorvastatin twenty milligrams for your cholesterol?"),
    ("patient", "Yes, I take that at night."),
    ("doctor", "Good. And your last blood tests showed your cholesterol is well controlled, so we'll continue that. I also notice you're due for an HbA1c check. Your last one was six point eight, which is in the pre-diabetic range. Have you made any changes to your diet?"),
    ("patient", "I've been trying to cut down on sugar and eat more vegetables. My daughter's been helping me with meal planning."),
    ("doctor", "That's excellent. We'll repeat the HbA1c today along with your renal function, which we need to monitor with the new medication. The blood test form will be at reception."),
    ("patient", "Is there anything else I should be doing?"),
    ("doctor", "Regular exercise is very important. Even a thirty-minute walk five times a week can make a real difference to your blood pressure. Try to keep your salt intake low as well. And if the fatigue continues or gets worse, do come back and see us."),
    ("patient", "OK, I will do. Thank you, Doctor."),
    ("doctor", "You're welcome. So to summarise: stop the ramipril today, start losartan fifty milligrams tomorrow morning with breakfast. Blood tests today, and I'll see you back in four weeks. If you develop any swelling, dizziness, or the cough gets worse rather than better, please call the surgery or ring one-one-one."),
    ("patient", "Thank you. I'll book the follow-up on the way out."),
    ("doctor", "Perfect. Take care, Mrs. Thompson."),
]


async def _synthesise_line(text: str, voice: str) -> np.ndarray:
    """Synthesise a single line using edge-tts."""
    import edge_tts

    communicate = edge_tts.Communicate(text, voice, rate="-5%")
    audio_bytes = io.BytesIO()

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_bytes.write(chunk["data"])

    audio_bytes.seek(0)

    # edge-tts outputs mp3 - decode via soundfile (needs ffmpeg or use temporary file)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp:
        tmp.write(audio_bytes.read())
        tmp.flush()
        audio, sr = sf.read(tmp.name, dtype="float32")

    # Resample to target SR if needed
    if sr != TARGET_SR:
        ratio = TARGET_SR / sr
        n_samples = int(len(audio) * ratio)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, n_samples),
            np.arange(len(audio)),
            audio if audio.ndim == 1 else audio.mean(axis=1),
        ).astype(np.float32)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    return audio


async def generate_consultation(output_path: Path) -> None:
    """Generate the full consultation audio from the script."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    segments: list[np.ndarray] = []

    # Short silence between turns (0.5s)
    silence = np.zeros(int(TARGET_SR * 0.5), dtype=np.float32)
    # Longer pause for examination moments (1.5s)
    long_pause = np.zeros(int(TARGET_SR * 1.5), dtype=np.float32)

    print(f"Generating {len(CONSULTATION_SCRIPT)} dialogue turns...")

    for i, (speaker, text) in enumerate(CONSULTATION_SCRIPT):
        voice = DOCTOR_VOICE if speaker == "doctor" else PATIENT_VOICE
        print(f"  [{i+1}/{len(CONSULTATION_SCRIPT)}] {speaker}: {text[:60]}...")

        audio = await _synthesise_line(text, voice)
        segments.append(audio)

        # Add pause between turns
        if "check your blood pressure" in text.lower():
            segments.append(long_pause)
        else:
            segments.append(silence)

    full_audio = np.concatenate(segments)

    # Normalise to -3dB peak
    peak = np.abs(full_audio).max()
    if peak > 0:
        full_audio = full_audio * (0.707 / peak)

    sf.write(str(output_path), full_audio, TARGET_SR)
    duration = len(full_audio) / TARGET_SR
    print(f"\nSaved: {output_path}")
    print(f"Duration: {duration:.1f}s ({duration/60:.1f} min)")
    print(f"Sample rate: {TARGET_SR}Hz")


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic GP consultation audio")
    ap.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()

    try:
        import edge_tts  # noqa: F401
    except ImportError:
        print("edge-tts not installed. Install with: pip install edge-tts")
        print("Falling back to silence-only placeholder...")
        _generate_placeholder(Path(args.output))
        return

    asyncio.run(generate_consultation(Path(args.output)))


def _generate_placeholder(output_path: Path):
    """Generate a silent placeholder WAV if edge-tts is unavailable."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = 10.0
    audio = np.zeros(int(TARGET_SR * duration), dtype=np.float32)
    sf.write(str(output_path), audio, TARGET_SR)
    print(f"Placeholder (silent) saved: {output_path}")
    print("Install edge-tts for real consultation audio: pip install edge-tts")


if __name__ == "__main__":
    main()
