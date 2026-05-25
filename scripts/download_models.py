"""Download model weights from HF Hub. Usage: python scripts/download_models.py --all"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download
from loguru import logger

MODELS_DIR = Path("data/models")


def download_medasr(model_id: str = "google/medasr") -> None:
    dest = MODELS_DIR / "medasr"
    logger.info(f"Downloading MedASR ({model_id}) → {dest}")
    snapshot_download(repo_id=model_id, local_dir=str(dest))
    logger.info("MedASR download complete")


def download_whisper(model_size: str = "medium") -> None:
    model_id = f"openai/whisper-{model_size}"
    dest = MODELS_DIR / f"whisper-{model_size}"
    logger.info(f"Downloading Whisper ({model_id}) → {dest}")
    snapshot_download(repo_id=model_id, local_dir=str(dest))
    logger.info("Whisper download complete")


def main():
    parser = argparse.ArgumentParser(description="Download model weights")
    parser.add_argument("--medasr", action="store_true", help="Download MedASR")
    parser.add_argument("--whisper", action="store_true", help="Download Whisper")
    parser.add_argument("--all", action="store_true", help="Download all models")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.all or args.medasr:
        download_medasr()
    if args.all or args.whisper:
        download_whisper()

    if not (args.all or args.medasr or args.whisper):
        parser.print_help()


if __name__ == "__main__":
    main()
