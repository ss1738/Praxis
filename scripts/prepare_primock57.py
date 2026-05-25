"""
Parse PriMock57 into train/val/test splits.
Pairs each WAV with its TextGrid transcript.
Split by consultation (not file) to avoid data leakage.
"""

from __future__ import annotations

import json
import os
import re
import random
from pathlib import Path

from loguru import logger

RAW_DIR = Path("data/raw/primock57")
AUDIO_DIR = RAW_DIR / "audio"
TRANSCRIPT_DIR = RAW_DIR / "transcripts"
PROCESSED_DIR = Path("data/processed/primock57")


def parse_textgrid(path: Path) -> str:
    """Get all text from a TextGrid file, clean up UNSURE/UNIN tags."""
    content = path.read_text(encoding="utf-8")

    # grab all text fields
    texts = re.findall(r'text\s*=\s*"([^"]*)"', content)

    # clean up tags
    cleaned = []
    for t in texts:
        t = t.strip()
        if not t:
            continue
        # keep the text inside UNSURE tags
        t = re.sub(r"<UNSURE>(.*?)</UNSURE>", r"\1", t)
        # drop unintelligible markers
        t = re.sub(r"<UNIN\s*/>", "", t)
        # clean up whitespace
        t = re.sub(r"\s+", " ", t).strip()
        if t:
            cleaned.append(t)

    return " ".join(cleaned)


def main():
    if not AUDIO_DIR.exists():
        logger.error(f"Audio dir not found: {AUDIO_DIR}")
        logger.error("Clone first: git clone https://github.com/babylonhealth/primock57.git data/raw/primock57")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # pair each audio file with its transcript
    records = []
    audio_files = sorted(AUDIO_DIR.glob("*.wav"))
    logger.info(f"Found {len(audio_files)} audio files")

    for wav_path in audio_files:
        stem = wav_path.stem  # e.g. "day1_consultation01_doctor"
        tg_path = TRANSCRIPT_DIR / f"{stem}.TextGrid"

        if not tg_path.exists():
            logger.warning(f"No transcript for {wav_path.name}, skipping")
            continue

        text = parse_textgrid(tg_path)
        if not text:
            logger.warning(f"Empty transcript for {wav_path.name}, skipping")
            continue

        # stem format: dayN_consultationNN_speaker
        parts = stem.rsplit("_", 1)
        consultation_id = parts[0]
        speaker = parts[1] if len(parts) > 1 else "unknown"

        records.append({
            "audio_path": str(wav_path.resolve()),
            "text": text,
            "consultation": consultation_id,
            "speaker": speaker,
        })

    logger.info(f"Total valid records: {len(records)}")

    # split by consultation so doctor+patient from same visit stay together
    consultations = sorted(set(r["consultation"] for r in records))
    random.seed(42)
    random.shuffle(consultations)

    n = len(consultations)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    train_consults = set(consultations[:n_train])
    val_consults = set(consultations[n_train : n_train + n_val])
    test_consults = set(consultations[n_train + n_val :])

    splits = {
        "train": [r for r in records if r["consultation"] in train_consults],
        "val": [r for r in records if r["consultation"] in val_consults],
        "test": [r for r in records if r["consultation"] in test_consults],
    }

    # save splits and log durations
    import soundfile as sf

    for name, split_records in splits.items():
        total_dur = 0
        for r in split_records:
            info = sf.info(r["audio_path"])
            total_dur += info.duration

        out_path = PROCESSED_DIR / f"{name}.json"
        out_path.write_text(json.dumps(split_records, indent=2))
        logger.info(
            f"  {name}: {len(split_records)} files, "
            f"{len(set(r['consultation'] for r in split_records))} consultations, "
            f"{total_dur/60:.1f} min → {out_path}"
        )

    logger.info("PriMock57 preparation complete")


if __name__ == "__main__":
    main()
