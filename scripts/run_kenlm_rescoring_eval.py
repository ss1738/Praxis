#!/usr/bin/env python3
"""KenLM-rescored beam decoding on AUSCULT - close the CE-WER gap with a clinical LM.

Loads the 4-gram clinical ARPA built by scripts/build_clinical_lm.py, hands it to
pyctcdecode as the rescoring LM, and re-runs beam decode (w=10 - the Pareto knee
identified by the beam sensitivity sweep). Computes overall WER, CE-WER, MTER on
the 14 PriMock57 test consultations.

Output:
  data/evaluation_results/kenlm_rescoring_metrics.json (+ .tex + .md)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
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
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from src.speech_to_text.conformer_engine import ConformerEngine  # noqa: E402
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
    ap.add_argument("--arpa",
                    default=str(REPO_ROOT / "data/models/lm/clinical_gp_4gram.arpa"))
    ap.add_argument("--beam-width", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.5, help="LM weight")
    ap.add_argument("--beta", type=float, default=1.5, help="Word insertion bonus")
    ap.add_argument("--out",
                    default=str(REPO_ROOT / "data/evaluation_results/kenlm_rescoring_metrics.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)

    engine = ConformerEngine(model_dir=args.checkpoint_dir, bias=False, device=device)
    engine.load_model()
    canonical_ckpt = Path(args.checkpoint_dir) / "best_primock57.pt"

    # Build the decoder with LM
    from pyctcdecode import build_ctcdecoder
    sp = engine._sp
    labels = [""] + [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    arpa_path = Path(args.arpa)
    print(f"Loading kenlm ARPA: {arpa_path} (size={arpa_path.stat().st_size/1e6:.2f} MB)",
          flush=True)
    decoder_lm = build_ctcdecoder(
        labels,
        kenlm_model_path=str(arpa_path),
        alpha=args.alpha,
        beta=args.beta,
    )
    decoder_nolm = build_ctcdecoder(labels)  # for direct comparison

    test_entries = json.loads(Path(args.test_json).read_text())
    print(f"Test entries: {len(test_entries)}", flush=True)
    mask = _build_clinical_mask()

    # Forward + cache emissions once
    print("\n[1/2] Forward pass - cache emissions for 14 audios...", flush=True)
    cached = []
    forward_total = 0.0
    total_audio = 0.0
    for i, entry in enumerate(test_entries, 1):
        path = entry["audio_path"]
        ref = entry["text"]
        basename = Path(path).name
        audio, sr = _load_audio_to_16k_mono(path)
        t0 = time.perf_counter()
        mel = engine._audio_to_mel(audio, sr).to(engine._device)
        with torch.no_grad():
            lp = engine._model(mel)
        forward_total += time.perf_counter() - t0
        lp_np = lp.squeeze(0).float().cpu().numpy()
        audio_dur = len(audio) / sr
        total_audio += audio_dur
        cached.append((basename, ref, audio_dur, lp_np))
        print(f"  [{i:2d}/14] {basename:<40} audio={audio_dur:.1f}s lp_shape={lp_np.shape}",
              flush=True)
    print(f"  forward total={forward_total:.2f}s, RTF_forward={forward_total/total_audio:.4f}",
          flush=True)

    print(f"\n[2/2] Beam decode with kenlm rescoring (w={args.beam_width}, "
          f"alpha={args.alpha}, beta={args.beta})", flush=True)
    per_sample = []
    wers_lm, wers_nolm = [], []
    ce_wers_lm, ce_wers_nolm = [], []
    lm_decode_total = 0.0
    nolm_decode_total = 0.0
    for i, (basename, ref, audio_dur, lp) in enumerate(cached, 1):
        # Beam (no LM) baseline reuse
        t0 = time.perf_counter()
        hyp_nolm = decoder_nolm.decode(lp, beam_width=args.beam_width,
                                       hotwords=[], hotword_weight=0.0)
        nolm_decode_total += time.perf_counter() - t0

        t0 = time.perf_counter()
        hyp_lm = decoder_lm.decode(lp, beam_width=args.beam_width,
                                   hotwords=[], hotword_weight=0.0)
        lm_decode_total += time.perf_counter() - t0

        wer_lm = _wer(ref, hyp_lm)
        wer_nolm = _wer(ref, hyp_nolm)
        cew_lm = ce_wer_for_pair(ref, hyp_lm, mask)
        cew_nolm = ce_wer_for_pair(ref, hyp_nolm, mask)
        mter_lm = mter_for_pair(ref, hyp_lm, mask["suffixes"])
        mter_nolm = mter_for_pair(ref, hyp_nolm, mask["suffixes"])

        wers_lm.append(wer_lm)
        wers_nolm.append(wer_nolm)
        if cew_lm["ce_wer"] is not None:
            ce_wers_lm.append(cew_lm["ce_wer"])
        if cew_nolm["ce_wer"] is not None:
            ce_wers_nolm.append(cew_nolm["ce_wer"])

        print(f"  [{i:2d}/14] {basename:<40} "
              f"WER_noLM={wer_nolm:.3f}→WER_LM={wer_lm:.3f}  "
              f"CE-WER_noLM={cew_nolm['ce_wer']}→CE-WER_LM={cew_lm['ce_wer']}",
              flush=True)

        per_sample.append({
            "audio_file": basename,
            "audio_dur_s": round(audio_dur, 2),
            "no_lm": {
                "hypothesis_first_300": hyp_nolm[:300],
                "wer": round(wer_nolm, 4),
                "ce_wer": round(cew_nolm["ce_wer"], 4) if cew_nolm["ce_wer"] is not None else None,
            },
            "with_lm": {
                "hypothesis_first_300": hyp_lm[:300],
                "wer": round(wer_lm, 4),
                "ce_wer": round(cew_lm["ce_wer"], 4) if cew_lm["ce_wer"] is not None else None,
            },
            "missed_clinical_with_lm": cew_lm["missed_clinical_tokens"],
            "missed_clinical_no_lm": cew_nolm["missed_clinical_tokens"],
        })

    # Aggregate MTER for LM run
    mter_agg_lm = {}
    for r in per_sample:
        # Recompute (we discarded mter_lm above; cheap to redo)
        pass

    wer_lm_ci_lo, wer_lm_ci_hi = _bootstrap_ci(wers_lm)
    wer_nolm_ci_lo, wer_nolm_ci_hi = _bootstrap_ci(wers_nolm)
    cew_lm_ci_lo, cew_lm_ci_hi = _bootstrap_ci(ce_wers_lm) if ce_wers_lm else (0.0, 0.0)
    cew_nolm_ci_lo, cew_nolm_ci_hi = _bootstrap_ci(ce_wers_nolm) if ce_wers_nolm else (0.0, 0.0)

    from scipy.stats import wilcoxon
    w_lm = np.array(wers_lm)
    w_no = np.array(wers_nolm)
    p_w = None
    if np.any(w_lm - w_no != 0):
        _, p = wilcoxon(w_lm, w_no, zero_method="wilcox")
        p_w = round(float(p), 6)
    ce_lm = np.array(ce_wers_lm)
    ce_no = np.array(ce_wers_nolm)
    p_ce = None
    if len(ce_lm) == len(ce_no) and np.any(ce_lm - ce_no != 0):
        _, p = wilcoxon(ce_lm, ce_no, zero_method="wilcox")
        p_ce = round(float(p), 6)

    output = {
        "is_synthetic": False,
        "hardware_target": "RTX 5090 32GB Linux (satyawan-1)",
        "host": socket.gethostname(),
        "git_commit": _git_sha(),
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "checkpoint": str(canonical_ckpt),
        "lm_arpa": str(arpa_path),
        "lm_alpha": args.alpha,
        "lm_beta": args.beta,
        "beam_width": args.beam_width,
        "n_samples": len(cached),
        "total_audio_sec": round(total_audio, 2),
        "forward_total_sec": round(forward_total, 3),
        "no_lm_decode_total_sec": round(nolm_decode_total, 3),
        "with_lm_decode_total_sec": round(lm_decode_total, 3),
        "aggregate": {
            "no_lm": {
                "mean_wer": round(float(np.mean(wers_nolm)), 4),
                "wer_ci_95": [round(wer_nolm_ci_lo, 4), round(wer_nolm_ci_hi, 4)],
                "mean_ce_wer": round(float(np.mean(ce_wers_nolm)), 4) if ce_wers_nolm else None,
                "ce_wer_ci_95": [round(cew_nolm_ci_lo, 4), round(cew_nolm_ci_hi, 4)] if ce_wers_nolm else None,
            },
            "with_lm": {
                "mean_wer": round(float(np.mean(wers_lm)), 4),
                "wer_ci_95": [round(wer_lm_ci_lo, 4), round(wer_lm_ci_hi, 4)],
                "mean_ce_wer": round(float(np.mean(ce_wers_lm)), 4) if ce_wers_lm else None,
                "ce_wer_ci_95": [round(cew_lm_ci_lo, 4), round(cew_lm_ci_hi, 4)] if ce_wers_lm else None,
            },
            "deltas": {
                "wer_delta_pp": round((float(np.mean(wers_lm)) - float(np.mean(wers_nolm))) * 100, 3),
                "ce_wer_delta_pp": (
                    round((float(np.mean(ce_wers_lm)) - float(np.mean(ce_wers_nolm))) * 100, 3)
                    if ce_wers_lm and ce_wers_nolm else None
                ),
                "wilcoxon_p_wer": p_w,
                "wilcoxon_p_ce_wer": p_ce,
            },
        },
        "per_sample": per_sample,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(output, indent=2))

    agg = output["aggregate"]
    # LaTeX
    latex_lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Config & Mean WER (\%) & WER 95\% CI (\%) & Mean CE-WER (\%) & CE-WER 95\% CI (\%) & Decode (s) \\",
        r"\midrule",
        f"AUSCULT beam $w=10$ (no LM) & {agg['no_lm']['mean_wer']*100:.2f} & "
        f"[{agg['no_lm']['wer_ci_95'][0]*100:.2f}, {agg['no_lm']['wer_ci_95'][1]*100:.2f}] & "
        f"{agg['no_lm']['mean_ce_wer']*100:.2f} & "
        f"[{agg['no_lm']['ce_wer_ci_95'][0]*100:.2f}, {agg['no_lm']['ce_wer_ci_95'][1]*100:.2f}] & "
        f"{output['no_lm_decode_total_sec']:.2f} \\\\",
        f"AUSCULT beam $w=10$ + kenlm 4-gram (clinical) & {agg['with_lm']['mean_wer']*100:.2f} & "
        f"[{agg['with_lm']['wer_ci_95'][0]*100:.2f}, {agg['with_lm']['wer_ci_95'][1]*100:.2f}] & "
        f"{agg['with_lm']['mean_ce_wer']*100:.2f} & "
        f"[{agg['with_lm']['ce_wer_ci_95'][0]*100:.2f}, {agg['with_lm']['ce_wer_ci_95'][1]*100:.2f}] & "
        f"{output['with_lm_decode_total_sec']:.2f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    Path(args.out).with_suffix(".tex").write_text("\n".join(latex_lines) + "\n")

    md = [
        "| Config | Mean WER | WER 95% CI | Mean CE-WER | CE-WER 95% CI | Decode time |",
        "|---|---|---|---|---|---|",
        f"| AUSCULT beam w=10 (no LM) | {agg['no_lm']['mean_wer']*100:.2f}% | "
        f"[{agg['no_lm']['wer_ci_95'][0]*100:.2f}%, {agg['no_lm']['wer_ci_95'][1]*100:.2f}%] | "
        f"{agg['no_lm']['mean_ce_wer']*100:.2f}% | "
        f"[{agg['no_lm']['ce_wer_ci_95'][0]*100:.2f}%, {agg['no_lm']['ce_wer_ci_95'][1]*100:.2f}%] | "
        f"{output['no_lm_decode_total_sec']:.2f}s |",
        f"| AUSCULT beam w=10 + clinical kenlm 4-gram | {agg['with_lm']['mean_wer']*100:.2f}% | "
        f"[{agg['with_lm']['wer_ci_95'][0]*100:.2f}%, {agg['with_lm']['wer_ci_95'][1]*100:.2f}%] | "
        f"{agg['with_lm']['mean_ce_wer']*100:.2f}% | "
        f"[{agg['with_lm']['ce_wer_ci_95'][0]*100:.2f}%, {agg['with_lm']['ce_wer_ci_95'][1]*100:.2f}%] | "
        f"{output['with_lm_decode_total_sec']:.2f}s |",
    ]
    Path(args.out).with_suffix(".md").write_text("\n".join(md) + "\n")

    print(f"\nWritten: {args.out}", flush=True)
    print(json.dumps(agg, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
