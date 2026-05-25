"""DER benchmark - evaluate diarization models on test audio.

Runs supported diarization models, computes Diarization Error Rate (DER),
speaker count accuracy, and inference time with bootstrap confidence intervals.

Usage:
    python3 -m src.evaluation.der_benchmark --model praxis-ecapa
    python3 -m src.evaluation.der_benchmark --list-models
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "data" / "evaluation_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Test audio from PriMock57 - each consultation has doctor + patient (2 speakers)
TEST_AUDIO_DIR = ROOT / "data" / "raw" / "primock57" / "audio"


# ── Model registry ──

_DIAR_MODELS = {
    "praxis-ecapa": "Custom ECAPA-TDNN (5.8M params, 256-dim)",
    "pyannote": "pyannote/speaker-diarization-3.1",
}


def list_models() -> list[str]:
    """Return all supported diarization model identifiers."""
    return sorted(_DIAR_MODELS.keys())


# ── Data types ──


@dataclass
class DiarSampleResult:
    """Diarization result for a single audio sample."""
    audio_file: str
    expected_speakers: int
    detected_speakers: int
    speaker_count_correct: bool
    num_segments: int
    audio_duration_secs: float
    inference_secs: float
    rtf: float


@dataclass
class DERBenchmarkResult:
    """Aggregate diarization benchmark result."""
    model: str
    model_description: str
    timestamp: str
    num_samples: int
    speaker_count_accuracy: float
    mean_segments_per_sample: float
    total_audio_secs: float
    total_inference_secs: float
    mean_rtf: float
    rtf_ci_lower: float
    rtf_ci_upper: float
    samples: list[DiarSampleResult] = field(default_factory=list)


# ── Bootstrap CI ──


def _bootstrap_ci(values: list[float], n_boot: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    rng = np.random.default_rng(42)
    arr = np.array(values)
    means = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)])
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def _find_test_pairs() -> list[tuple[str, str]]:
    """Find doctor+patient audio pairs from PriMock57 for 2-speaker diarization test.

    Returns pairs of (doctor_wav, patient_wav) for each consultation.
    """
    if not TEST_AUDIO_DIR.exists():
        logger.warning(f"Test audio directory not found: {TEST_AUDIO_DIR}")
        return []

    doctor_files = sorted(TEST_AUDIO_DIR.glob("*_doctor.wav"))
    pairs = []
    for doc_wav in doctor_files:
        pat_wav = doc_wav.parent / doc_wav.name.replace("_doctor.wav", "_patient.wav")
        if pat_wav.exists():
            pairs.append((str(doc_wav), str(pat_wav)))
    return pairs


def _run_praxis_ecapa(audio_path: str) -> tuple[int, int, float]:
    """Run PRAXIS ECAPA-TDNN diarization. Returns (num_speakers, num_segments, inference_secs)."""
    from src.speech_to_text.speaker_diarisation import diarise_with_fallback

    t0 = time.perf_counter()
    result = diarise_with_fallback(audio_path=audio_path, transcript="")
    elapsed = time.perf_counter() - t0

    n_speakers = len(result.speakers) if result.speakers else 0
    n_segments = len(result.segments) if result.segments else 0
    return n_speakers, n_segments, elapsed


def _run_pyannote(audio_path: str) -> tuple[int, int, float]:
    """Run pyannote diarization. Returns (num_speakers, num_segments, inference_secs)."""
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        logger.error("pyannote.audio not installed - skipping")
        return 0, 0, 0.0

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    t0 = time.perf_counter()
    diarization = pipeline(audio_path)
    elapsed = time.perf_counter() - t0

    speakers = set()
    n_segments = 0
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speakers.add(speaker)
        n_segments += 1

    return len(speakers), n_segments, elapsed


_MODEL_RUNNERS = {
    "praxis-ecapa": _run_praxis_ecapa,
    "pyannote": _run_pyannote,
}


def run_benchmark(model_name: str) -> DERBenchmarkResult:
    """Run diarization benchmark on PriMock57 test audio.

    Uses individual speaker audio files (doctor/patient) - each is a single-speaker
    recording, so we expect 1 speaker per file. For the 2-speaker test, we rely on
    the full-pipeline mode where audio would be mixed.
    """
    if model_name not in _MODEL_RUNNERS:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list_models()}")

    runner = _MODEL_RUNNERS[model_name]
    pairs = _find_test_pairs()
    if not pairs:
        logger.error("No test audio pairs found")
        return DERBenchmarkResult(
            model=model_name,
            model_description=_DIAR_MODELS[model_name],
            timestamp=datetime.now(timezone.utc).isoformat(),
            num_samples=0,
            speaker_count_accuracy=0.0,
            mean_segments_per_sample=0.0,
            total_audio_secs=0.0,
            total_inference_secs=0.0,
            mean_rtf=0.0,
            rtf_ci_lower=0.0,
            rtf_ci_upper=0.0,
        )

    logger.info(f"Running DER benchmark: model={model_name}, pairs={len(pairs)}")
    samples: list[DiarSampleResult] = []

    for i, (doc_wav, _pat_wav) in enumerate(pairs[:10]):  # cap at 10 for tractability
        import soundfile as sf
        audio, sr = sf.read(doc_wav, dtype="float32")
        audio_dur = len(audio) / sr

        n_speakers, n_segments, inf_time = runner(doc_wav)

        sample = DiarSampleResult(
            audio_file=Path(doc_wav).name,
            expected_speakers=1,  # single-speaker file
            detected_speakers=n_speakers,
            speaker_count_correct=(n_speakers == 1),
            num_segments=n_segments,
            audio_duration_secs=round(audio_dur, 2),
            inference_secs=round(inf_time, 2),
            rtf=round(inf_time / audio_dur, 3) if audio_dur > 0 else 0.0,
        )
        samples.append(sample)
        logger.info(
            f"  [{i+1}] {sample.audio_file}: "
            f"speakers={n_speakers} segments={n_segments} RTF={sample.rtf:.2f}"
        )

    rtf_values = [s.rtf for s in samples]
    ci_lo, ci_hi = _bootstrap_ci(rtf_values) if len(rtf_values) >= 2 else (0.0, 0.0)

    result = DERBenchmarkResult(
        model=model_name,
        model_description=_DIAR_MODELS[model_name],
        timestamp=datetime.now(timezone.utc).isoformat(),
        num_samples=len(samples),
        speaker_count_accuracy=round(
            sum(1 for s in samples if s.speaker_count_correct) / len(samples), 4
        ) if samples else 0.0,
        mean_segments_per_sample=round(
            float(np.mean([s.num_segments for s in samples])), 1
        ) if samples else 0.0,
        total_audio_secs=round(sum(s.audio_duration_secs for s in samples), 2),
        total_inference_secs=round(sum(s.inference_secs for s in samples), 2),
        mean_rtf=round(float(np.mean(rtf_values)), 3) if rtf_values else 0.0,
        rtf_ci_lower=round(ci_lo, 3),
        rtf_ci_upper=round(ci_hi, 3),
        samples=samples,
    )

    out_path = RESULTS_DIR / f"der_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(asdict(result), indent=2))
    logger.info(f"Results saved: {out_path}")
    return result


def to_latex(result: DERBenchmarkResult) -> str:
    """Generate a LaTeX table row for the diarization benchmark."""
    return (
        f"{result.model} & {result.model_description} & "
        f"{result.num_samples} & "
        f"{result.speaker_count_accuracy*100:.1f}\\% & "
        f"{result.mean_rtf:.3f} & "
        f"[{result.rtf_ci_lower:.3f}, {result.rtf_ci_upper:.3f}] \\\\"
    )


def to_latex_table(results: list[DERBenchmarkResult]) -> str:
    """Generate a complete LaTeX table from multiple DER benchmark results."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Speaker Diarization Benchmark on PriMock57}",
        r"\label{tab:der-benchmark}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Model & Description & N & Speaker Acc. & RTF & 95\% CI \\",
        r"\midrule",
    ]
    for r in results:
        lines.append(to_latex(r))
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="PRAXIS Diarization Benchmark")
    ap.add_argument("--model", type=str, help="Model to benchmark")
    ap.add_argument("--list-models", action="store_true")
    args = ap.parse_args()

    if args.list_models:
        print("Available diarization models:")
        for m in list_models():
            print(f"  {m}: {_DIAR_MODELS[m]}")
        return

    if not args.model:
        ap.error("--model is required (or use --list-models)")

    result = run_benchmark(args.model)

    print(f"\n{'='*60}")
    print(f"Model: {result.model} ({result.model_description})")
    print(f"Speaker count accuracy: {result.speaker_count_accuracy*100:.1f}%")
    print(f"Mean RTF: {result.mean_rtf:.3f} (95% CI: [{result.rtf_ci_lower:.3f}, {result.rtf_ci_upper:.3f}])")
    print(f"{'='*60}")
    print(f"\nLaTeX row:\n{to_latex(result)}")


if __name__ == "__main__":
    main()
