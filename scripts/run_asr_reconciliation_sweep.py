#!/usr/bin/env python3
"""ASR decoding-configuration sweep to reconcile the 50.12% vs 56.72% AUSCULT WER gap.

Runs PRAXIS-AUSCULT (`best_primock57.pt`) over the 14 PriMock57 test consultations
under two isolated decoding conditions, captures hypotheses for each, and compares
them character-for-character against the two historical hypothesis arrays
stored in:

  - data/evaluation_results/wer_praxis-conformer_20260427_134720.json (Run A, 50.12%)
  - data/evaluation_results/wer_praxis-conformer_20260427_154708.json (Run B, 56.72%)

Conditions:
  1. greedy  : argmax over the emission matrix per frame, then collapse blanks.
  2. beam_nobias : pyctcdecode beam search (beam_width default, no LM, no hotwords).

Output:
  data/evaluation_results/wer_decoding_sweep_final.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.speech_to_text.conformer_engine import ConformerEngine  # noqa: E402


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()


def _wer(ref: str, hyp: str) -> float:
    try:
        import jiwer
        return float(jiwer.wer(_norm(ref), _norm(hyp)))
    except Exception:
        ref_toks = _norm(ref).split()
        hyp_toks = _norm(hyp).split()
        if not ref_toks:
            return 1.0 if hyp_toks else 0.0
        m, n = len(ref_toks), len(hyp_toks)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if ref_toks[i - 1] == hyp_toks[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
        return dp[m][n] / m


def _bootstrap_ci(values: list[float], n_iter: int = 1000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(values)
    means = np.array([arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n_iter)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _identical_or_normed(a: str, b: str) -> dict:
    return {
        "byte_identical": a == b,
        "identical_after_normalisation": _norm(a) == _norm(b),
    }


def transcribe_greedy(engine: ConformerEngine, audio_path: str) -> tuple[str, float]:
    """Plain argmax decode. Uses engine's existing greedy path."""
    t0 = time.perf_counter()
    text = engine.transcribe_file(audio_path)
    return text, time.perf_counter() - t0


def transcribe_beam_nobias(engine: ConformerEngine, audio_path: str,
                            beam_width: int = 100) -> tuple[str, float]:
    """pyctcdecode beam search with NO hot-word biasing and NO LM.

    Replicates the engine's exact forward path (audio → mel → model log_probs)
    and feeds the emission matrix to pyctcdecode with an empty hotword list.
    """
    from pyctcdecode import build_ctcdecoder

    # Build a one-shot decoder over the engine's full vocab (blank + sp pieces)
    if not hasattr(engine, "_beam_nobias_decoder"):
        sp = engine._sp
        labels = [""] + [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
        engine._beam_nobias_decoder = build_ctcdecoder(labels)

    t0 = time.perf_counter()
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    mel = engine._audio_to_mel(audio, sr).to(engine._device)
    with torch.no_grad():
        log_probs = engine._model(mel)
    # log_probs shape: [B=1, T, V] -> [T, V] float on CPU for pyctcdecode
    lp_np = log_probs.squeeze(0).float().cpu().numpy()

    text = engine._beam_nobias_decoder.decode(
        lp_np,
        beam_width=beam_width,
        hotwords=[],
        hotword_weight=0.0,
    )
    return text, time.perf_counter() - t0


def load_engine(checkpoint_dir: Path, device: str) -> ConformerEngine:
    eng = ConformerEngine(model_dir=str(checkpoint_dir), bias=False, device=device)
    eng.load_model()
    return eng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir",
                    default=str(REPO_ROOT / "data/models/conformer_v2"))
    ap.add_argument("--test-json",
                    default=str(REPO_ROOT / "data/processed/primock57_chunked/test.json"))
    ap.add_argument("--historical-run-a",
                    default=str(REPO_ROOT / "data/evaluation_results/wer_praxis-conformer_20260427_134720.json"))
    ap.add_argument("--historical-run-b",
                    default=str(REPO_ROOT / "data/evaluation_results/wer_praxis-conformer_20260427_154708.json"))
    ap.add_argument("--out",
                    default=str(REPO_ROOT / "data/evaluation_results/wer_decoding_sweep_final.json"))
    ap.add_argument("--beam-width", type=int, default=100)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device={device}", flush=True)

    test_entries = json.loads(Path(args.test_json).read_text())
    # Index historicals by basename
    def _by_filename(d):
        return {Path(s["audio_file"]).name if "audio_file" in s else "?": s
                for s in d.get("samples", [])}
    run_a_full = json.loads(Path(args.historical_run_a).read_text())
    run_b_full = json.loads(Path(args.historical_run_b).read_text())
    run_a_samples = _by_filename(run_a_full)
    run_b_samples = _by_filename(run_b_full)

    print(f"Loading AUSCULT from {args.checkpoint_dir}...", flush=True)
    engine = load_engine(Path(args.checkpoint_dir), device=device)
    canonical_ckpt = Path(args.checkpoint_dir) / "best_primock57.pt"
    ckpt_sha = _sha256_of(canonical_ckpt) if canonical_ckpt.exists() else None
    print(f"  canonical_checkpoint_sha256: {ckpt_sha}", flush=True)

    per_sample = []
    greedy_wers: list[float] = []
    beam_wers: list[float] = []
    total_greedy_sec = 0.0
    total_beam_sec = 0.0
    total_audio_sec = 0.0

    for i, entry in enumerate(test_entries, 1):
        audio_path = entry["audio_path"]
        ref = entry["text"]
        basename = Path(audio_path).name
        audio_dur = sf.info(audio_path).duration
        total_audio_sec += audio_dur

        try:
            greedy_hyp, t_g = transcribe_greedy(engine, audio_path)
        except Exception as exc:
            print(f"  [{i}/{len(test_entries)}] {basename} GREEDY FAIL: {exc}", flush=True)
            greedy_hyp, t_g = "", 0.0

        try:
            beam_hyp, t_b = transcribe_beam_nobias(engine, audio_path,
                                                    beam_width=args.beam_width)
        except Exception as exc:
            print(f"  [{i}/{len(test_entries)}] {basename} BEAM FAIL: {exc}", flush=True)
            beam_hyp, t_b = "", 0.0

        wer_g = _wer(ref, greedy_hyp) if greedy_hyp else 1.0
        wer_b = _wer(ref, beam_hyp) if beam_hyp else 1.0
        greedy_wers.append(wer_g)
        beam_wers.append(wer_b)
        total_greedy_sec += t_g
        total_beam_sec += t_b

        # Compare against historicals (by basename)
        hist_a = run_a_samples.get(basename, {})
        hist_b = run_b_samples.get(basename, {})
        cmp_a = _identical_or_normed(greedy_hyp, hist_a.get("hypothesis", ""))
        cmp_b = _identical_or_normed(greedy_hyp, hist_b.get("hypothesis", ""))
        cmp_a_beam = _identical_or_normed(beam_hyp, hist_a.get("hypothesis", ""))
        cmp_b_beam = _identical_or_normed(beam_hyp, hist_b.get("hypothesis", ""))

        print(
            f"  [{i:2d}/{len(test_entries)}] {basename:<40} "
            f"WER g={wer_g:.3f} b={wer_b:.3f}  "
            f"greedy↔RunA={'Y' if cmp_a['byte_identical'] else ('y' if cmp_a['identical_after_normalisation'] else 'n')} "
            f"greedy↔RunB={'Y' if cmp_b['byte_identical'] else ('y' if cmp_b['identical_after_normalisation'] else 'n')}",
            flush=True,
        )

        per_sample.append({
            "audio_file": basename,
            "audio_duration_s": round(audio_dur, 2),
            "reference_first_120": ref[:120],
            "greedy": {
                "hypothesis": greedy_hyp,
                "hypothesis_first_120": greedy_hyp[:120],
                "wer": wer_g,
                "wall_time_s": round(t_g, 3),
                "rtf": round(t_g / max(audio_dur, 1e-6), 4),
                "vs_run_a": cmp_a,
                "vs_run_b": cmp_b,
            },
            "beam_nobias": {
                "hypothesis": beam_hyp,
                "hypothesis_first_120": beam_hyp[:120],
                "wer": wer_b,
                "wall_time_s": round(t_b, 3),
                "rtf": round(t_b / max(audio_dur, 1e-6), 4),
                "vs_run_a": cmp_a_beam,
                "vs_run_b": cmp_b_beam,
            },
            "historical_run_a": {
                "wer": hist_a.get("wer"),
                "hypothesis_first_120": hist_a.get("hypothesis", "")[:120],
            },
            "historical_run_b": {
                "wer": hist_b.get("wer"),
                "hypothesis_first_120": hist_b.get("hypothesis", "")[:120],
            },
        })

    # Aggregates
    mean_greedy = float(np.mean(greedy_wers))
    mean_beam = float(np.mean(beam_wers))
    ci_g_lo, ci_g_hi = _bootstrap_ci(greedy_wers)
    ci_b_lo, ci_b_hi = _bootstrap_ci(beam_wers)

    # Counts of identical matches across the 14 samples
    n_greedy_match_a = sum(1 for r in per_sample if r["greedy"]["vs_run_a"]["byte_identical"])
    n_greedy_match_a_normed = sum(1 for r in per_sample if r["greedy"]["vs_run_a"]["identical_after_normalisation"])
    n_greedy_match_b = sum(1 for r in per_sample if r["greedy"]["vs_run_b"]["byte_identical"])
    n_greedy_match_b_normed = sum(1 for r in per_sample if r["greedy"]["vs_run_b"]["identical_after_normalisation"])
    n_beam_match_a = sum(1 for r in per_sample if r["beam_nobias"]["vs_run_a"]["byte_identical"])
    n_beam_match_a_normed = sum(1 for r in per_sample if r["beam_nobias"]["vs_run_a"]["identical_after_normalisation"])
    n_beam_match_b = sum(1 for r in per_sample if r["beam_nobias"]["vs_run_b"]["byte_identical"])
    n_beam_match_b_normed = sum(1 for r in per_sample if r["beam_nobias"]["vs_run_b"]["identical_after_normalisation"])

    # Verdict
    def _verdict():
        if n_greedy_match_a == len(per_sample):
            return ("greedy ON best_primock57.pt EXACTLY reproduces historical Run A (50.12%). "
                    "Run B used a different config/checkpoint.")
        if n_greedy_match_b == len(per_sample):
            return ("greedy ON best_primock57.pt EXACTLY reproduces historical Run B (56.72%). "
                    "Run A used a different config/checkpoint.")
        if n_beam_match_a == len(per_sample):
            return ("beam-nobias ON best_primock57.pt EXACTLY reproduces historical Run A (50.12%). "
                    "Run B used a different config/checkpoint.")
        if n_beam_match_b == len(per_sample):
            return ("beam-nobias ON best_primock57.pt EXACTLY reproduces historical Run B (56.72%). "
                    "Run A used a different config/checkpoint.")
        if n_greedy_match_a_normed == len(per_sample) or n_greedy_match_b_normed == len(per_sample):
            return "normalisation-only difference identified for greedy decode."
        if n_beam_match_a_normed == len(per_sample) or n_beam_match_b_normed == len(per_sample):
            return "normalisation-only difference identified for beam decode."
        return ("NEITHER greedy NOR beam ON best_primock57.pt reproduces EITHER historical run. "
                "Both historical runs almost certainly used a different checkpoint "
                "(best_model_v2_120M.pt or another) that is no longer on this box.")

    aggregate = {
        "n_samples": len(per_sample),
        "total_audio_sec": round(total_audio_sec, 2),
        "greedy": {
            "mean_wer": round(mean_greedy, 4),
            "ci_95_lower": round(ci_g_lo, 4),
            "ci_95_upper": round(ci_g_hi, 4),
            "total_wall_time_s": round(total_greedy_sec, 2),
            "mean_rtf": round(total_greedy_sec / total_audio_sec, 4) if total_audio_sec else None,
            "exact_match_run_a": n_greedy_match_a,
            "exact_match_run_a_normed": n_greedy_match_a_normed,
            "exact_match_run_b": n_greedy_match_b,
            "exact_match_run_b_normed": n_greedy_match_b_normed,
        },
        "beam_nobias": {
            "mean_wer": round(mean_beam, 4),
            "ci_95_lower": round(ci_b_lo, 4),
            "ci_95_upper": round(ci_b_hi, 4),
            "beam_width": args.beam_width,
            "total_wall_time_s": round(total_beam_sec, 2),
            "mean_rtf": round(total_beam_sec / total_audio_sec, 4) if total_audio_sec else None,
            "exact_match_run_a": n_beam_match_a,
            "exact_match_run_a_normed": n_beam_match_a_normed,
            "exact_match_run_b": n_beam_match_b,
            "exact_match_run_b_normed": n_beam_match_b_normed,
        },
        "verdict": _verdict(),
    }

    output = {
        "is_synthetic": False,
        "hardware_target": "RTX 5090 32GB Linux (satyawan-1)",
        "host": socket.gethostname(),
        "git_commit": _git_sha(),
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "canonical_checkpoint": str(canonical_ckpt),
        "canonical_checkpoint_sha256": ckpt_sha,
        "historical_run_a": {
            "path": str(Path(args.historical_run_a).relative_to(REPO_ROOT)),
            "mean_wer": run_a_full.get("mean_wer"),
            "mean_rtf": run_a_full.get("mean_rtf"),
        },
        "historical_run_b": {
            "path": str(Path(args.historical_run_b).relative_to(REPO_ROOT)),
            "mean_wer": run_b_full.get("mean_wer"),
            "mean_rtf": run_b_full.get("mean_rtf"),
        },
        "decoding_conditions": {
            "greedy": "torch.argmax over emission matrix, blank-collapse",
            "beam_nobias": f"pyctcdecode beam_width={args.beam_width}, no LM, no hotwords",
        },
        "aggregate": aggregate,
        "per_sample": per_sample,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(output, indent=2))
    print(f"\nWritten: {args.out}", flush=True)
    print(json.dumps(aggregate, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
