"""WER/CER benchmarking for comparing STT engines."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import jiwer
from loguru import logger

from src.speech_to_text.engine_factory import ASREngine


@dataclass
class BenchmarkResult:
    """Single sample result."""

    file_name: str
    reference: str
    hypothesis: str
    wer: float
    cer: float
    inference_time_secs: float


@dataclass
class BenchmarkReport:
    """Aggregated results for one engine."""

    engine_name: str
    results: list[BenchmarkResult] = field(default_factory=list)

    @property
    def mean_wer(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.wer for r in self.results) / len(self.results)

    @property
    def mean_cer(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.cer for r in self.results) / len(self.results)

    @property
    def total_inference_time(self) -> float:
        return sum(r.inference_time_secs for r in self.results)

    def summary(self) -> str:
        return (
            f"Engine: {self.engine_name} | "
            f"Samples: {len(self.results)} | "
            f"Mean WER: {self.mean_wer:.2%} | "
            f"Mean CER: {self.mean_cer:.2%} | "
            f"Total time: {self.total_inference_time:.1f}s"
        )


class Benchmarker:
    """Run WER/CER evaluation on test data."""

    def __init__(self):
        pass

    @staticmethod
    def _normalise(text: str) -> str:
        import re as _re
        text = text.lower()
        text = _re.sub(r"[^\w\s]", "", text)
        text = _re.sub(r"\s+", " ", text).strip()
        return text

    def evaluate_engine(
        self,
        engine: ASREngine,
        engine_name: str,
        audio_dir: str | Path,
        references: dict[str, str],
    ) -> BenchmarkReport:
        """Transcribe every wav in audio_dir and compute WER against references."""
        audio_dir = Path(audio_dir)
        report = BenchmarkReport(engine_name=engine_name)

        for wav_path in sorted(audio_dir.glob("*.wav")):
            stem = wav_path.stem
            if stem not in references:
                logger.warning(f"No reference for {wav_path.name}, skipping")
                continue

            ref = references[stem]
            t0 = time.perf_counter()
            hyp = engine.transcribe_file(str(wav_path))
            elapsed = time.perf_counter() - t0

            ref_norm = self._normalise(ref)
            hyp_norm = self._normalise(hyp)
            wer = jiwer.wer(ref_norm, hyp_norm)
            cer = jiwer.cer(ref_norm, hyp_norm)

            result = BenchmarkResult(
                file_name=wav_path.name,
                reference=ref,
                hypothesis=hyp,
                wer=wer,
                cer=cer,
                inference_time_secs=elapsed,
            )
            report.results.append(result)
            logger.info(f"  {wav_path.name}: WER={wer:.2%}, CER={cer:.2%}, time={elapsed:.2f}s")

        logger.info(report.summary())
        return report
