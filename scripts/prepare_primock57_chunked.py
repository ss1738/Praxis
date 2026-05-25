"""Chunk PriMock57 into 30s segments. The old script wasted 97% of the audio."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from loguru import logger

RAW_DIR = Path("data/raw/primock57")
AUDIO_DIR = RAW_DIR / "audio"
TRANSCRIPT_DIR = RAW_DIR / "transcripts"
CHUNKED_DIR = Path("data/processed/primock57_chunked")
CHUNKS_AUDIO_DIR = CHUNKED_DIR / "audio"

CHUNK_DURATION = 30.0  # seconds
MIN_CHUNK_DURATION = 5.0  # skip chunks shorter than this
SAMPLE_RATE = 16000

def parse_textgrid_with_timestamps(path: Path) -> list[dict]:
    """Get timestamped text intervals from TextGrid."""
    content = path.read_text(encoding="utf-8")

    intervals = []
    # Match each interval block
    pattern = re.compile(
        r'xmin\s*=\s*([\d.]+)\s*\n\s*xmax\s*=\s*([\d.]+)\s*\n\s*text\s*=\s*"([^"]*)"',
        re.MULTILINE,
    )

    for match in pattern.finditer(content):
        xmin = float(match.group(1))
        xmax = float(match.group(2))
        text = match.group(3).strip()

        if not text:
            continue

        # Clean tags
        text = re.sub(r"<UNSURE>(.*?)</UNSURE>", r"\1", text)
        text = re.sub(r"<UNIN\s*/>", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        if text:
            intervals.append({"xmin": xmin, "xmax": xmax, "text": text})

    return intervals

def chunk_audio_with_transcript(
    audio_path: Path,
    intervals: list[dict],
    chunk_duration: float = CHUNK_DURATION,
) -> list[dict]:
    """Split audio into chunks, match up transcript text by overlap."""
    audio, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    total_duration = len(audio) / sr

    chunks = []
    t = 0.0

    while t < total_duration:
        t_end = min(t + chunk_duration, total_duration)
        actual_duration = t_end - t

        if actual_duration < MIN_CHUNK_DURATION:
            break

        # Collect transcript text overlapping this chunk
        chunk_text_parts = []
        for iv in intervals:
            # Check overlap
            overlap_start = max(t, iv["xmin"])
            overlap_end = min(t_end, iv["xmax"])
            if overlap_end > overlap_start:
                chunk_text_parts.append(iv["text"])

        chunk_text = " ".join(chunk_text_parts)

        if chunk_text.strip():
            # Extract audio segment
            start_sample = int(t * sr)
            end_sample = int(t_end * sr)
            chunk_audio = audio[start_sample:end_sample]

            chunks.append({
                "audio": chunk_audio,
                "text": chunk_text,
                "start": t,
                "end": t_end,
                "duration": actual_duration,
            })

        t = t_end

    return chunks

def save_chunk(chunk_audio: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), chunk_audio, SAMPLE_RATE)

def speed_augment(audio: np.ndarray, rate: float) -> np.ndarray:
    return librosa.effects.time_stretch(audio, rate=1.0 / rate)

def main():
    if not AUDIO_DIR.exists():
        logger.error(f"Audio dir not found: {AUDIO_DIR}")
        return

    CHUNKS_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Parse all audio files and their TextGrid transcripts
    audio_files = sorted(AUDIO_DIR.glob("*.wav"))
    logger.info(f"Found {len(audio_files)} audio files")

    # Group by consultation
    consultations: dict[str, list[dict]] = {}
    for wav_path in audio_files:
        stem = wav_path.stem
        tg_path = TRANSCRIPT_DIR / f"{stem}.TextGrid"

        if not tg_path.exists():
            logger.warning(f"No transcript for {wav_path.name}, skipping")
            continue

        intervals = parse_textgrid_with_timestamps(tg_path)
        if not intervals:
            logger.warning(f"Empty transcript for {wav_path.name}, skipping")
            continue

        # Parse consultation ID and speaker
        parts = stem.rsplit("_", 1)
        consultation_id = parts[0]
        speaker = parts[1] if len(parts) > 1 else "unknown"

        if consultation_id not in consultations:
            consultations[consultation_id] = []

        consultations[consultation_id].append({
            "audio_path": wav_path,
            "intervals": intervals,
            "speaker": speaker,
            "stem": stem,
        })

    logger.info(f"Found {len(consultations)} consultations")

    # Step 2: Use the same test consultations as before
    # Load original test.json to get test consultation IDs
    orig_test_path = Path("data/processed/primock57/test.json")
    if orig_test_path.exists():
        with open(orig_test_path) as f:
            orig_test = json.load(f)
        test_consults = set(r["consultation"] for r in orig_test)
        logger.info(f"Keeping {len(test_consults)} test consultations from original split")
    else:
        # Fallback: random split
        all_consults = sorted(consultations.keys())
        random.seed(42)
        random.shuffle(all_consults)
        n_test = max(7, int(len(all_consults) * 0.1))
        test_consults = set(all_consults[-n_test:])
        logger.info(f"No original test.json found, using random {n_test} test consultations")

    # Split remaining into train/val (85/15)
    remaining = sorted(c for c in consultations if c not in test_consults)
    random.seed(42)
    random.shuffle(remaining)
    n_val = max(4, int(len(remaining) * 0.15))
    val_consults = set(remaining[:n_val])
    train_consults = set(remaining[n_val:])

    logger.info(f"Split: train={len(train_consults)}, val={len(val_consults)}, test={len(test_consults)}")

    # Step 3: Chunk all audio
    all_records = {"train": [], "val": [], "test": []}

    for consult_id, files in sorted(consultations.items()):
        if consult_id in test_consults:
            split = "test"
        elif consult_id in val_consults:
            split = "val"
        else:
            split = "train"

        for file_info in files:
            wav_path = file_info["audio_path"]
            intervals = file_info["intervals"]
            speaker = file_info["speaker"]
            stem = file_info["stem"]

            if split == "test":
                # For test: keep as single full file (matches original benchmark)
                full_text = " ".join(iv["text"] for iv in intervals)
                all_records["test"].append({
                    "audio_path": str(wav_path.resolve()),
                    "text": full_text,
                    "consultation": consult_id,
                    "speaker": speaker,
                    "source": "primock57_original",
                })
                continue

            # For train/val: chunk into 30s segments
            chunks = chunk_audio_with_transcript(wav_path, intervals, CHUNK_DURATION)

            for ci, chunk in enumerate(chunks):
                chunk_name = f"{stem}_chunk{ci:03d}"
                chunk_path = CHUNKS_AUDIO_DIR / f"{chunk_name}.wav"

                if not chunk_path.exists():
                    save_chunk(chunk["audio"], chunk_path)

                record = {
                    "audio_path": str(chunk_path.resolve()),
                    "text": chunk["text"],
                    "consultation": consult_id,
                    "speaker": speaker,
                    "source": "primock57_chunked",
                    "chunk_start": chunk["start"],
                    "chunk_end": chunk["end"],
                }
                all_records[split].append(record)

    logger.info(f"Chunked: train={len(all_records['train'])}, val={len(all_records['val'])}, test={len(all_records['test'])}")

    # Step 4: Speed augmentation on training chunks only
    logger.info("Applying speed augmentation to training chunks...")
    augmented = []
    for i, rec in enumerate(all_records["train"]):
        audio_path = rec["audio_path"]
        stem = Path(audio_path).stem

        for speed, label in [(0.9, "slow"), (1.1, "fast")]:
            aug_name = f"aug_{label}_{stem}"
            aug_path = CHUNKS_AUDIO_DIR / f"{aug_name}.wav"

            if not aug_path.exists():
                audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
                stretched = speed_augment(audio, speed)
                save_chunk(stretched, aug_path)

            augmented.append({
                "audio_path": str(aug_path.resolve()),
                "text": rec["text"],
                "consultation": rec["consultation"],
                "speaker": rec["speaker"],
                "source": f"primock57_aug_{label}",
            })

        if (i + 1) % 100 == 0:
            logger.info(f"  Augmented {i + 1}/{len(all_records['train'])} chunks")

    all_records["train"].extend(augmented)
    logger.info(f"After augmentation: train={len(all_records['train'])}")

    # Step 5: Save splits
    for split_name, records in all_records.items():
        out_path = CHUNKED_DIR / f"{split_name}.json"
        out_path.write_text(json.dumps(records, indent=2))

        # Compute duration
        total_dur = 0
        for r in records:
            try:
                info = sf.info(r["audio_path"])
                total_dur += info.duration
            except Exception:
                pass

        logger.info(
            f"  {split_name}: {len(records)} samples, "
            f"{total_dur/3600:.2f}h"
        )

    # Summary
    train_original = sum(1 for r in all_records["train"] if r["source"] == "primock57_chunked")
    train_aug = sum(1 for r in all_records["train"] if "aug" in r["source"])
    logger.info(f"\nSummary:")
    logger.info(f"  Original chunks: {train_original}")
    logger.info(f"  Augmented (speed): {train_aug}")
    logger.info(f"  Total train: {len(all_records['train'])}")
    logger.info(f"  Val: {len(all_records['val'])}")
    logger.info(f"  Test: {len(all_records['test'])} (same consultations as before)")

if __name__ == "__main__":
    main()
