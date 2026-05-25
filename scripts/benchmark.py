"""Run WER benchmarks. Usage: python scripts/benchmark.py"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger

from src.speech_to_text.benchmarker import Benchmarker
from src.speech_to_text.engine_factory import EngineType, create_engine


def main():
    parser = argparse.ArgumentParser(description="Benchmark ASR engines")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed/primock57",
        help="Directory with test.json",
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        default=["medasr", "whisper"],
        help="Engines to benchmark",
    )
    args = parser.parse_args()

    # Load test references
    test_path = Path(args.data_dir) / "test.json"
    if not test_path.exists():
        logger.error(f"Test data not found: {test_path}")
        return

    with open(test_path) as f:
        records = json.load(f)

    references = {}
    for rec in records:
        stem = Path(rec["audio_path"]).stem
        references[stem] = rec["text"]

    audio_dir = Path(records[0]["audio_path"]).parent
    logger.info(f"Benchmarking on {len(references)} samples from {audio_dir}")

    benchmarker = Benchmarker()
    reports = []

    for engine_name in args.engines:
        engine_type = EngineType(engine_name)
        logger.info(f"\n{'='*60}\nBenchmarking: {engine_name}\n{'='*60}")
        engine = create_engine(engine_type)
        report = benchmarker.evaluate_engine(engine, engine_name, audio_dir, references)
        reports.append(report)

    # Summary comparison
    logger.info(f"\n{'='*60}\nFINAL COMPARISON\n{'='*60}")
    for report in reports:
        logger.info(report.summary())


if __name__ == "__main__":
    main()
