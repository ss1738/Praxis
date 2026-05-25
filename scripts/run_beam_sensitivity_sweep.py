#!/usr/bin/env python3
"""Beam-width sensitivity sweep on AUSCULT (best_primock57.pt).

Walks beam_width ∈ {10, 50, 100, 200, 500} over the 14 PriMock57 test
consultations and reports the accuracy/efficiency trade-off curve.

For efficiency, the model forward pass runs ONCE per audio file (caching
emission log-probs in memory), then each beam width decodes from the cached
emissions. This isolates the decoder cost from the model cost and lets the
sweep run in a single shot.

Output:
  data/evaluation_results/asr_beam_sensitivity_sweep.json
  data/evaluation_results/asr_beam_sensitivity_sweep.tex  (numerical matrix only)
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


def _sha256(path: Path) -> str:
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
        r = _norm(ref).split()
        h = _norm(hyp).split()
        if not r:
            return 1.0 if h else 0.0
        m, n = len(r), len(h)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                c = 0 if r[i - 1] == h[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + c)
        return dp[m][n] / m


def _bootstrap_ci(values: list[float], n_iter: int = 1000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(values)
    means = np.array([arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n_iter)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _load_audio_to_16k_mono(path: str) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    return audio, sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir",
                    default=str(REPO_ROOT / "data/models/conformer_v2"))
    ap.add_argument("--test-json",
                    default=str(REPO_ROOT / "data/processed/primock57_chunked/test.json"))
    ap.add_argument("--out",
                    default=str(REPO_ROOT / "data/evaluation_results/asr_beam_sensitivity_sweep.json"))
    ap.add_argument("--beam-widths", type=int, nargs="+",
                    default=[10, 50, 100, 200, 500])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device={device}", flush=True)

    test_entries = json.loads(Path(args.test_json).read_text())
    print(f"Loaded {len(test_entries)} test entries.", flush=True)

    # ── Load engine + decoder ──
    engine = ConformerEngine(model_dir=str(args.checkpoint_dir), bias=False, device=device)
    engine.load_model()
    canonical_ckpt = Path(args.checkpoint_dir) / "best_primock57.pt"
    ckpt_sha = _sha256(canonical_ckpt) if canonical_ckpt.exists() else None
    print(f"Checkpoint sha256: {ckpt_sha}", flush=True)

    from pyctcdecode import build_ctcdecoder
    sp = engine._sp
    labels = [""] + [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    decoder = build_ctcdecoder(labels)
    print(f"Decoder built: {len(labels)} labels.", flush=True)

    # ── Forward pass + emission caching ──
    print("\n[1/2] Forward pass - caching emissions for 14 audios...", flush=True)
    cached = []  # list of (basename, ref, audio_dur, log_probs_np)
    forward_total = 0.0
    total_audio = 0.0
    for i, entry in enumerate(test_entries, 1):
        path = entry["audio_path"]
        ref = entry["text"]
        basename = Path(path).name
        t0 = time.perf_counter()
        audio, sr = _load_audio_to_16k_mono(path)
        mel = engine._audio_to_mel(audio, sr).to(engine._device)
        with torch.no_grad():
            log_probs = engine._model(mel)
        lp_np = log_probs.squeeze(0).float().cpu().numpy()
        forward_t = time.perf_counter() - t0
        forward_total += forward_t
        audio_dur = len(audio) / sr
        total_audio += audio_dur
        cached.append((basename, ref, audio_dur, lp_np))
        print(f"  [{i:2d}/14] {basename:<40} forward={forward_t:.2f}s "
              f"audio={audio_dur:.1f}s lp_shape={lp_np.shape}", flush=True)

    print(f"\nForward total: {forward_total:.2f}s over {total_audio:.1f}s audio "
          f"(RTF_forward = {forward_total/total_audio:.4f})", flush=True)

    # ── Greedy baseline from cached emissions ──
    print("\n[2/2] Decoding sweep...", flush=True)
    sweep_results = []

    print("\n-- greedy (argmax) --", flush=True)
    greedy_wers = []
    greedy_decode_t = 0.0
    greedy_per_sample = []
    for basename, ref, audio_dur, lp in cached:
        t0 = time.perf_counter()
        ids = np.argmax(lp, axis=-1).tolist()
        # collapse blanks + dedupe consecutive
        out = []
        prev = None
        for d in ids:
            if d != prev and d != 0:
                out.append(d - 1)  # TOKEN_OFFSET=1
            prev = d
        out = [d for d in out if 0 <= d < sp.get_piece_size()]
        hyp = sp.decode(out) if out else ""
        dt_s = time.perf_counter() - t0
        greedy_decode_t += dt_s
        w = _wer(ref, hyp)
        greedy_wers.append(w)
        greedy_per_sample.append({"audio_file": basename, "wer": w,
                                  "decode_s": round(dt_s, 4),
                                  "hypothesis_first_120": hyp[:120]})
    g_lo, g_hi = _bootstrap_ci(greedy_wers)
    sweep_results.append({
        "decoder": "greedy",
        "beam_width": None,
        "mean_wer": round(float(np.mean(greedy_wers)), 4),
        "ci_95_lower": round(g_lo, 4),
        "ci_95_upper": round(g_hi, 4),
        "decode_total_s": round(greedy_decode_t, 3),
        "decode_rtf": round(greedy_decode_t / total_audio, 6),
        "forward_plus_decode_rtf": round((forward_total + greedy_decode_t) / total_audio, 6),
        "per_sample": greedy_per_sample,
    })
    print(f"  mean WER = {np.mean(greedy_wers)*100:.2f}%  "
          f"decode={greedy_decode_t:.2f}s  RTF_decode={greedy_decode_t/total_audio:.6f}",
          flush=True)

    # ── Beam sweep ──
    for w_beam in args.beam_widths:
        print(f"\n-- beam w={w_beam} --", flush=True)
        wers = []
        decode_total = 0.0
        per_sample = []
        for basename, ref, audio_dur, lp in cached:
            t0 = time.perf_counter()
            hyp = decoder.decode(lp, beam_width=w_beam, hotwords=[], hotword_weight=0.0)
            dt_s = time.perf_counter() - t0
            decode_total += dt_s
            wer_v = _wer(ref, hyp)
            wers.append(wer_v)
            per_sample.append({"audio_file": basename, "wer": wer_v,
                               "decode_s": round(dt_s, 4),
                               "hypothesis_first_120": hyp[:120]})
        ci_lo, ci_hi = _bootstrap_ci(wers)
        sweep_results.append({
            "decoder": "beam_nobias",
            "beam_width": w_beam,
            "mean_wer": round(float(np.mean(wers)), 4),
            "ci_95_lower": round(ci_lo, 4),
            "ci_95_upper": round(ci_hi, 4),
            "decode_total_s": round(decode_total, 3),
            "decode_rtf": round(decode_total / total_audio, 6),
            "forward_plus_decode_rtf": round((forward_total + decode_total) / total_audio, 6),
            "per_sample": per_sample,
        })
        print(f"  mean WER = {np.mean(wers)*100:.2f}%  "
              f"decode={decode_total:.2f}s  RTF_decode={decode_total/total_audio:.6f}  "
              f"RTF_total={(forward_total+decode_total)/total_audio:.6f}",
              flush=True)

    # ── LaTeX matrix (numerical only - captions are author's hand) ──
    latex_lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Decoder & Beam $w$ & Mean WER (\%) & 95\% CI (\%) & Decode time (s) & RTF (forward+decode) \\",
        r"\midrule",
    ]
    for r in sweep_results:
        decoder = r["decoder"]
        bw = r["beam_width"] if r["beam_width"] is not None else "--"
        latex_lines.append(
            f"{decoder.replace('_', r'\\_')} & {bw} & "
            f"{r['mean_wer']*100:.2f} & "
            f"[{r['ci_95_lower']*100:.2f}, {r['ci_95_upper']*100:.2f}] & "
            f"{r['decode_total_s']:.2f} & {r['forward_plus_decode_rtf']:.4f} \\\\"
        )
    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_blob = "\n".join(latex_lines) + "\n"

    # Markdown mirror
    md_lines = [
        "| Decoder | Beam w | Mean WER | 95% CI | Decode (s) | RTF (forward+decode) |",
        "|---|---|---|---|---|---|",
    ]
    for r in sweep_results:
        bw = r["beam_width"] if r["beam_width"] is not None else "-"
        md_lines.append(
            f"| `{r['decoder']}` | {bw} | "
            f"{r['mean_wer']*100:.2f}% | "
            f"[{r['ci_95_lower']*100:.2f}%, {r['ci_95_upper']*100:.2f}%] | "
            f"{r['decode_total_s']:.2f} | {r['forward_plus_decode_rtf']:.4f} |"
        )
    md_blob = "\n".join(md_lines) + "\n"

    out = {
        "is_synthetic": False,
        "hardware_target": "RTX 5090 32GB Linux (satyawan-1)",
        "host": socket.gethostname(),
        "git_commit": _git_sha(),
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "canonical_checkpoint": str(canonical_ckpt),
        "canonical_checkpoint_sha256": ckpt_sha,
        "n_samples": len(test_entries),
        "total_audio_s": round(total_audio, 2),
        "forward_total_s": round(forward_total, 3),
        "forward_rtf": round(forward_total / total_audio, 6),
        "beam_widths_swept": args.beam_widths,
        "results": sweep_results,
        "tables": {"markdown": md_blob, "latex": latex_blob},
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    out_path.with_suffix(".tex").write_text(latex_blob)
    out_path.with_suffix(".md").write_text(md_blob)
    print(f"\nWritten: {out_path}", flush=True)
    print(f"Written: {out_path.with_suffix('.tex')}", flush=True)
    print(f"Written: {out_path.with_suffix('.md')}", flush=True)
    print("\nSummary:")
    print(md_blob)
    return 0


if __name__ == "__main__":
    sys.exit(main())
