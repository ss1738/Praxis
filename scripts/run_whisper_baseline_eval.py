#!/usr/bin/env python3
"""Whisper-large-v3 baseline on PriMock57 test (n=14).

Targets the AUSCULT 88.64% CE-WER bottleneck by swapping the upstream ASR step
for openai/whisper-large-v3. Runs on RTX 5090 with fp16, chunked transcription
(30 s windows) via the transformers ASR pipeline. Reuses the existing CE-WER /
MTER scorers in scripts/evaluate_ce_wer.py.

Output:
  data/evaluation_results/whisper_baseline_metrics.json
  data/evaluation_results/whisper_baseline_metrics.tex   (numerical only)
  data/evaluation_results/whisper_baseline_metrics.md
"""

from __future__ import annotations

import argparse
import datetime as dt
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

# Reuse CE-WER + MTER scorers
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from evaluate_ce_wer import (  # noqa: E402
    _build_clinical_mask, ce_wer_for_pair, mter_for_pair, _bootstrap_ci, _norm
)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-json",
                    default=str(REPO_ROOT / "data/processed/primock57_chunked/test.json"))
    ap.add_argument("--model-id", default="openai/whisper-large-v3")
    ap.add_argument("--out",
                    default=str(REPO_ROOT / "data/evaluation_results/whisper_baseline_metrics.json"))
    ap.add_argument("--chunk-length-s", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"device={device}  dtype={dtype}", flush=True)

    from transformers import (AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline)

    print(f"Loading {args.model_id} ...", flush=True)
    t0 = time.perf_counter()
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)
    processor = AutoProcessor.from_pretrained(args.model_id)
    print(f"  loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=device,
        chunk_length_s=args.chunk_length_s,
        batch_size=args.batch_size,
        return_timestamps=False,
        generate_kwargs={"language": "english", "task": "transcribe"},
    )

    test_entries = json.loads(Path(args.test_json).read_text())
    print(f"Test entries: {len(test_entries)}", flush=True)

    mask = _build_clinical_mask()

    per_sample = []
    wers = []
    ce_wers = []
    total_audio_sec = 0.0
    total_wall_sec = 0.0

    for i, entry in enumerate(test_entries, 1):
        path = entry["audio_path"]
        ref = entry["text"]
        basename = Path(path).name

        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        audio_dur = len(audio) / sr
        total_audio_sec += audio_dur

        t0 = time.perf_counter()
        result = asr(audio.copy())
        wall = time.perf_counter() - t0
        total_wall_sec += wall

        hyp = (result.get("text") if isinstance(result, dict) else result).strip()
        overall_wer = _wer(ref, hyp)
        cew = ce_wer_for_pair(ref, hyp, mask)
        mter = mter_for_pair(ref, hyp, mask["suffixes"])
        wers.append(overall_wer)
        if cew["ce_wer"] is not None:
            ce_wers.append(cew["ce_wer"])

        print(
            f"[{i:2d}/{len(test_entries)}] {basename:<40} "
            f"WER={overall_wer:.3f}  CE-WER={cew['ce_wer'] if cew['ce_wer'] is not None else 'NA'}  "
            f"audio={audio_dur:.1f}s wall={wall:.1f}s RTF={wall/audio_dur:.3f}",
            flush=True,
        )

        per_sample.append({
            "audio_file": basename,
            "audio_dur_s": round(audio_dur, 2),
            "wall_s": round(wall, 2),
            "rtf": round(wall / audio_dur, 4),
            "wer": round(overall_wer, 4),
            "ce_wer": round(cew["ce_wer"], 4) if cew["ce_wer"] is not None else None,
            "n_ref_tokens": cew["n_ref_tokens"],
            "n_ref_clinical": cew["n_ref_clinical"],
            "wer_breakdown": cew["wer_overall_breakdown"],
            "ce_wer_breakdown": cew["ce_wer_breakdown"],
            "hypothesis_first_300": hyp[:300],
            "reference_first_300": ref[:300],
            "missed_clinical_tokens": cew["missed_clinical_tokens"],
            "mter": mter,
        })

    # Aggregate
    wer_ci_lo, wer_ci_hi = _bootstrap_ci(wers)
    cew_ci_lo, cew_ci_hi = _bootstrap_ci(ce_wers)

    mter_agg = {}
    for r in per_sample:
        for suffix, b in r["mter"].items():
            acc = mter_agg.setdefault(suffix, {"ref_hits": 0, "exact_match": 0,
                                                "suffix_only_match": 0,
                                                "complete_omission": 0})
            acc["ref_hits"] += b["ref_hits"]
            acc["exact_match"] += b["exact_match"]
            acc["suffix_only_match"] += b["suffix_only_match"]
            acc["complete_omission"] += b["complete_omission"]

    output = {
        "is_synthetic": False,
        "hardware_target": "RTX 5090 32GB Linux (satyawan-1)",
        "host": socket.gethostname(),
        "git_commit": _git_sha(),
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_id": args.model_id,
        "dtype": str(dtype),
        "chunk_length_s": args.chunk_length_s,
        "batch_size": args.batch_size,
        "n_samples": len(per_sample),
        "total_audio_sec": round(total_audio_sec, 2),
        "total_wall_sec": round(total_wall_sec, 2),
        "mean_rtf": round(total_wall_sec / total_audio_sec, 4),
        "aggregate": {
            "mean_wer": round(float(np.mean(wers)), 4),
            "median_wer": round(float(np.median(wers)), 4),
            "wer_ci_95": [round(wer_ci_lo, 4), round(wer_ci_hi, 4)],
            "mean_ce_wer": round(float(np.mean(ce_wers)), 4) if ce_wers else None,
            "median_ce_wer": round(float(np.median(ce_wers)), 4) if ce_wers else None,
            "ce_wer_ci_95": [round(cew_ci_lo, 4), round(cew_ci_hi, 4)] if ce_wers else None,
        },
        "mter_aggregate": mter_agg,
        "per_sample": per_sample,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(output, indent=2))

    agg = output["aggregate"]
    # LaTeX (numerical only)
    latex_lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Model & $n$ & Mean WER (\%) & Mean CE-WER (\%) & WER 95\% CI (\%) & Mean RTF \\",
        r"\midrule",
        f"{args.model_id.replace('_', r'\\_')} & {len(per_sample)} & "
        f"{agg['mean_wer']*100:.2f} & {agg['mean_ce_wer']*100:.2f} & "
        f"[{agg['wer_ci_95'][0]*100:.2f}, {agg['wer_ci_95'][1]*100:.2f}] & "
        f"{output['mean_rtf']:.4f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    Path(args.out).with_suffix(".tex").write_text("\n".join(latex_lines) + "\n")

    md_lines = [
        "| Model | n | Mean WER | Mean CE-WER | WER 95% CI | Mean RTF |",
        "|---|---|---|---|---|---|",
        f"| `{args.model_id}` | {len(per_sample)} | {agg['mean_wer']*100:.2f}% | "
        f"{agg['mean_ce_wer']*100:.2f}% | "
        f"[{agg['wer_ci_95'][0]*100:.2f}%, {agg['wer_ci_95'][1]*100:.2f}%] | "
        f"{output['mean_rtf']:.4f} |",
    ]
    Path(args.out).with_suffix(".md").write_text("\n".join(md_lines) + "\n")

    print(f"\nWritten: {args.out}", flush=True)
    print(json.dumps(agg, indent=2), flush=True)
    print(f"\nMTER per suffix (with hits):")
    for s, b in sorted(mter_agg.items(), key=lambda kv: kv[1]["ref_hits"], reverse=True):
        if b["ref_hits"] == 0:
            continue
        rec = (b["exact_match"] / b["ref_hits"] * 100) if b["ref_hits"] else 0.0
        print(f"  -{s:<10} hits={b['ref_hits']:>3}  exact={b['exact_match']:>3}  "
              f"suffix_only={b['suffix_only_match']:>3}  omit={b['complete_omission']:>3}  "
              f"recovery={rec:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
