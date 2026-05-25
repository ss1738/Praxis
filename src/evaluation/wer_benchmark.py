"""WER benchmark - evaluate ASR models on PriMock57 test set.

Runs any supported ASR engine on the PriMock57 test split (14 samples),
computing per-sample and aggregate WER with bootstrap confidence intervals.
Supports pairwise statistical comparison via Wilcoxon signed-rank test.

Usage:
    python3 -m src.evaluation.wer_benchmark --model whisper-small
    python3 -m src.evaluation.wer_benchmark --model praxis-conformer --compare whisper-small
    python3 -m src.evaluation.wer_benchmark --list-models
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from jiwer import wer
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent.parent
TEST_JSON = ROOT / "data" / "processed" / "primock57_chunked" / "test.json"
RESULTS_DIR = ROOT / "data" / "evaluation_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Text normalisation (same as wer_primock57.py) ──

_TAG_RE = re.compile(r"<[^>]+/?>|</[^>]+>")
_NONWORD_RE = re.compile(r"[^a-z0-9' ]+")
_WS_RE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Normalise transcript text for WER computation."""
    t = text.lower()
    t = _TAG_RE.sub(" ", t)
    t = _NONWORD_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


# ── Model registry ──

_MODEL_MAP = {
    "whisper-small": ("whisper", {"model_size": "small"}),
    "whisper-medium": ("whisper", {"model_size": "medium"}),
    "whisper-large-v3": ("whisper", {"model_size": "large-v3"}),
    "faster-whisper-small": ("faster_whisper", {"model_size": "small"}),
    "faster-whisper-medium": ("faster_whisper", {"model_size": "medium"}),
    "medasr-base": ("medasr", {}),
    "praxis-conformer": ("praxis_conformer", {}),
    "praxis-conformer-biased": ("praxis_conformer", {"bias": True}),
}


def list_models() -> list[str]:
    """Return all supported model identifiers."""
    return sorted(_MODEL_MAP.keys())


def _load_engine(model_name: str):
    """Load and return an ASR engine for the given model name."""
    from src.speech_to_text.engine_factory import EngineType, create_engine

    if model_name not in _MODEL_MAP:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list_models()}")

    engine_key, kwargs = _MODEL_MAP[model_name]
    engine_type = EngineType(engine_key)
    engine = create_engine(engine_type, **kwargs)
    engine.load_model()
    return engine


# ── Data types ──


@dataclass
class SampleResult:
    """WER result for a single audio sample."""
    audio_file: str
    reference: str
    hypothesis: str
    wer: float
    audio_duration_secs: float
    inference_secs: float
    rtf: float  # real-time factor: inference_secs / audio_duration_secs


@dataclass
class BenchmarkResult:
    """Aggregate WER benchmark result."""
    model: str
    timestamp: str
    num_samples: int
    mean_wer: float
    median_wer: float
    ci_lower: float  # 95% bootstrap CI lower bound
    ci_upper: float  # 95% bootstrap CI upper bound
    total_audio_secs: float
    total_inference_secs: float
    mean_rtf: float
    samples: list[SampleResult] = field(default_factory=list)


@dataclass
class ComparisonResult:
    """Pairwise statistical comparison between two models."""
    model_a: str
    model_b: str
    mean_wer_a: float
    mean_wer_b: float
    wer_diff: float
    wilcoxon_statistic: float
    p_value: float
    significant: bool  # p < 0.05


# ── Bootstrap CI ──


def _bootstrap_ci(values: list[float], n_boot: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    rng = np.random.default_rng(42)
    arr = np.array(values)
    means = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)])
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


# ── Benchmark runner ──


def run_benchmark(model_name: str, test_json: Path | None = None) -> BenchmarkResult:
    """Run WER benchmark for a single model on PriMock57 test set.

    Loads the model, transcribes each test sample, computes WER, and
    returns aggregate statistics with bootstrap 95% CI.
    """
    import soundfile as sf

    test_path = test_json or TEST_JSON
    entries = json.loads(test_path.read_text())
    logger.info(f"Running WER benchmark: model={model_name}, samples={len(entries)}")

    engine = _load_engine(model_name)
    samples: list[SampleResult] = []
    full_refs: list[str] = []
    full_hyps: list[str] = []

    for i, entry in enumerate(entries):
        audio_path = str(pathlib.Path(entry["audio_path"]).resolve()) if os.path.isabs(entry["audio_path"]) else str(pathlib.Path(os.environ.get("PRAXIS_DATA_ROOT", ".")).joinpath(entry["audio_path"]))
        ref_text = normalise(entry.get("text", ""))
        if not ref_text:
            logger.warning(f"Skipping {audio_path}: no reference text")
            continue

        audio, sr = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio_dur = len(audio) / sr

        t0 = time.perf_counter()
        hyp_raw = engine.transcribe(audio, sr)
        inf_time = time.perf_counter() - t0

        hyp_text = normalise(hyp_raw)
        sample_wer = wer(ref_text, hyp_text) if hyp_text else 1.0

        full_refs.append(ref_text)
        full_hyps.append(hyp_text)

        sample = SampleResult(
            audio_file=Path(audio_path).name,
            reference=ref_text[:200],
            hypothesis=hyp_text[:200],
            wer=round(sample_wer, 4),
            audio_duration_secs=round(audio_dur, 2),
            inference_secs=round(inf_time, 2),
            rtf=round(inf_time / audio_dur, 3) if audio_dur > 0 else 0.0,
        )
        samples.append(sample)
        logger.info(
            f"  [{i+1}/{len(entries)}] {sample.audio_file}: "
            f"WER={sample_wer*100:.1f}% RTF={sample.rtf:.2f}"
        )

    wer_values = [s.wer for s in samples]
    ci_lo, ci_hi = _bootstrap_ci(wer_values) if len(wer_values) >= 2 else (0.0, 1.0)

    # Medical-term recall - computed on FULL reference and hypothesis (not the
    # 200-char truncated copies stored in SampleResult).
    try:
        from src.clinical.medical_lexicon import build_lexicon, in_lexicon_recall
        lex = set(build_lexicon())
        mc = in_lexicon_recall(full_refs, full_hyps, lex)
        logger.info(
            f"Medical-term recall: {mc['recovered']}/{mc['ref_hits']} = "
            f"{mc['mc_recall']*100:.2f}% (err {mc['mc_err']*100:.2f}%)"
        )
    except Exception as exc:
        logger.warning(f"Medical-term recall skipped: {exc}")
        mc = None

    # MHI - consequence-weighted hallucination / omission rates
    try:
        from src.evaluation.mhi import run_mhi_benchmark
        mhi_bench = run_mhi_benchmark(
            full_refs, full_hyps,
            model_name=model_name,
            wer_values=wer_values,
            save=True,
        )
        logger.info(
            f"MHI: score={mhi_bench.mean_mhi_score:.3f} "
            f"omission={mhi_bench.mean_omission:.3f} "
            f"hallucination={mhi_bench.mean_hallucination:.3f} "
            f"delta(WER-omission)={mhi_bench.mhi_vs_wer_delta:+.3f}"
        )
    except Exception as exc:
        logger.warning(f"MHI skipped: {exc}")
        mhi_bench = None

    result = BenchmarkResult(
        model=model_name,
        timestamp=datetime.now(timezone.utc).isoformat(),
        num_samples=len(samples),
        mean_wer=round(float(np.mean(wer_values)), 4) if wer_values else 1.0,
        median_wer=round(float(np.median(wer_values)), 4) if wer_values else 1.0,
        ci_lower=round(ci_lo, 4),
        ci_upper=round(ci_hi, 4),
        total_audio_secs=round(sum(s.audio_duration_secs for s in samples), 2),
        total_inference_secs=round(sum(s.inference_secs for s in samples), 2),
        mean_rtf=round(float(np.mean([s.rtf for s in samples])), 3) if samples else 0.0,
        samples=samples,
    )

    # Save results (with optional medical-term recall and MHI blocks)
    payload = asdict(result)
    if mc is not None:
        payload["medical_term_recall"] = mc
    if mhi_bench is not None:
        payload["mhi"] = {
            "mhi_score": mhi_bench.mean_mhi_score,
            "mhi_omission": mhi_bench.mean_omission,
            "mhi_hallucination": mhi_bench.mean_hallucination,
            "mhi_ci_lower": mhi_bench.mhi_ci_lower,
            "mhi_ci_upper": mhi_bench.mhi_ci_upper,
            "mhi_vs_wer_delta": mhi_bench.mhi_vs_wer_delta,
        }
    out_path = RESULTS_DIR / f"wer_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info(f"Results saved: {out_path}")
    logger.info(
        f"WER: {result.mean_wer*100:.2f}% "
        f"(95% CI: [{result.ci_lower*100:.2f}%, {result.ci_upper*100:.2f}%])"
    )
    return result


def compare_models(result_a: BenchmarkResult, result_b: BenchmarkResult) -> ComparisonResult:
    """Pairwise Wilcoxon signed-rank test between two models' WER results.

    Tests whether the difference in per-sample WER between two models is
    statistically significant (p < 0.05).
    """
    from scipy.stats import wilcoxon

    wers_a = [s.wer for s in result_a.samples]
    wers_b = [s.wer for s in result_b.samples]

    if len(wers_a) != len(wers_b):
        raise ValueError("Both models must be evaluated on the same samples")

    stat, p = wilcoxon(wers_a, wers_b)

    comp = ComparisonResult(
        model_a=result_a.model,
        model_b=result_b.model,
        mean_wer_a=result_a.mean_wer,
        mean_wer_b=result_b.mean_wer,
        wer_diff=round(result_a.mean_wer - result_b.mean_wer, 4),
        wilcoxon_statistic=round(float(stat), 4),
        p_value=round(float(p), 6),
        significant=p < 0.05,
    )

    out_path = RESULTS_DIR / f"wer_comparison_{result_a.model}_vs_{result_b.model}.json"
    out_path.write_text(json.dumps(asdict(comp), indent=2))
    logger.info(
        f"Comparison: {comp.model_a} vs {comp.model_b}: "
        f"diff={comp.wer_diff*100:+.2f}pp, p={comp.p_value:.4f}, "
        f"significant={comp.significant}"
    )
    return comp


def to_latex(result: BenchmarkResult) -> str:
    """Generate a LaTeX table row for the benchmark result."""
    return (
        f"{result.model} & {result.num_samples} & "
        f"{result.mean_wer*100:.2f}\\% & {result.median_wer*100:.2f}\\% & "
        f"[{result.ci_lower*100:.2f}, {result.ci_upper*100:.2f}] & "
        f"{result.mean_rtf:.2f} \\\\"
    )


def to_latex_table(results: list[BenchmarkResult]) -> str:
    """Generate a complete LaTeX table from multiple benchmark results."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Word Error Rate (WER) on PriMock57 Test Set}",
        r"\label{tab:wer-benchmark}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Model & N & Mean WER & Median WER & 95\% CI & RTF \\",
        r"\midrule",
    ]
    for r in sorted(results, key=lambda x: x.mean_wer):
        lines.append(to_latex(r))
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="PRAXIS WER Benchmark on PriMock57")
    ap.add_argument("--model", type=str, help="Model to benchmark")
    ap.add_argument("--compare", type=str, help="Second model for pairwise comparison")
    ap.add_argument("--test-json", type=str, default=None, help="Path to test manifest")
    ap.add_argument("--list-models", action="store_true", help="List available models")
    args = ap.parse_args()

    if args.list_models:
        print("Available models:")
        for m in list_models():
            print(f"  {m}")
        return

    if not args.model:
        ap.error("--model is required (or use --list-models)")

    test_json = Path(args.test_json) if args.test_json else None
    result_a = run_benchmark(args.model, test_json)

    print(f"\n{'='*60}")
    print(f"Model: {result_a.model}")
    print(f"Mean WER: {result_a.mean_wer*100:.2f}%")
    print(f"Median WER: {result_a.median_wer*100:.2f}%")
    print(f"95% CI: [{result_a.ci_lower*100:.2f}%, {result_a.ci_upper*100:.2f}%]")
    print(f"Mean RTF: {result_a.mean_rtf:.3f}")
    print(f"{'='*60}")

    if args.compare:
        result_b = run_benchmark(args.compare, test_json)
        comp = compare_models(result_a, result_b)
        print(f"\nComparison: {comp.model_a} vs {comp.model_b}")
        print(f"  WER diff: {comp.wer_diff*100:+.2f} percentage points")
        print(f"  Wilcoxon p-value: {comp.p_value:.6f}")
        print(f"  Significant (p<0.05): {comp.significant}")

    print(f"\nLaTeX row:\n{to_latex(result_a)}")


if __name__ == "__main__":
    main()
