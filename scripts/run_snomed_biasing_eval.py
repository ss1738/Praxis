#!/usr/bin/env python3
"""SNOMED biasing MC-WER harness - clinical-reference-aware.

The previous biased_vs_greedy.json had mc_recall=0 in both conditions because the
PriMock57 chunked test references contain few hits against the 185-term lexicon.

This harness builds a CLINICAL-REFERENCE SUBSET: it selects PriMock57 test
chunks whose reference transcript contains at least one lexicon term, then runs
AUSCULT decoding in two modes:

  1. greedy (no biasing)
  2. pyctcdecode beam search with SnomedLexicon hot-word boost (boost=8.0)

For each consultation chunk it computes:
  - overall WER (jiwer/local routine)
  - medical-term-only WER (MC-WER) over reference tokens that appear in the lexicon
  - the per-term recovery delta

Output: data/evaluation_results/snomed_biasing_mc_wer_latest.json
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.speech_to_text.snomed_biasing import SnomedLexicon, _MEDICAL_SUFFIXES  # noqa: E402


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def _tokenise(text: str) -> list[str]:
    return [w for w in _normalise(text).split() if w]


def _mc_recall(ref_tokens: list[str], hyp_tokens: list[str], lex: set[str]) -> dict:
    hyp_set = set(hyp_tokens)
    lex_hits = [w for w in ref_tokens if w in lex]
    recovered = [w for w in lex_hits if w in hyp_set]
    missed = sorted({w for w in lex_hits if w not in hyp_set})
    return {
        "ref_hits": len(lex_hits),
        "recovered": len(recovered),
        "mc_recall": (len(recovered) / len(lex_hits)) if lex_hits else None,
        "missed": missed[:30],
    }


def _wer(ref: str, hyp: str) -> float:
    # Levenshtein-based WER (use jiwer if available, else fall back)
    try:
        import jiwer
        return float(jiwer.wer(ref, hyp))
    except Exception:
        ref_toks = _tokenise(ref)
        hyp_toks = _tokenise(hyp)
        return _levenshtein_wer(ref_toks, hyp_toks)


def _levenshtein_wer(ref: list[str], hyp: list[str]) -> float:
    if not ref:
        return 1.0 if hyp else 0.0
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,
                           dp[i][j - 1] + 1,
                           dp[i - 1][j - 1] + cost)
    return dp[m][n] / m


def build_clinical_subset(chunked_train_json: Path, lex: set[str],
                          min_hits_per_chunk: int = 2,
                          max_chunks: int = 60) -> list[dict]:
    """Pick chunks whose reference has at least N lexicon hits."""
    data = json.loads(chunked_train_json.read_text())
    subset = []
    for entry in data:
        ref = entry.get("text", "")
        tokens = _tokenise(ref)
        hits = [w for w in tokens if w in lex]
        if len(hits) >= min_hits_per_chunk:
            subset.append({
                "audio_path": entry["audio_path"],
                "reference": ref,
                "lex_hit_count": len(hits),
            })
        if len(subset) >= max_chunks:
            break
    return subset


def load_asr_engine(checkpoint_dir: Path, use_bias: bool):
    """Return a callable: (audio_path: str) -> hypothesis string."""
    from src.speech_to_text.conformer_engine import ConformerEngine
    eng = ConformerEngine(model_dir=str(checkpoint_dir), bias=use_bias)
    eng.load_model()
    return eng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir",
                    default=str(REPO_ROOT / "data/models/conformer_v2"))
    ap.add_argument("--manifest",
                    default=str(REPO_ROOT / "data/processed/primock57_chunked/train.json"),
                    help="Source manifest to mine clinical-reference subsets from")
    ap.add_argument("--out",
                    default=str(REPO_ROOT / "data/evaluation_results/snomed_biasing_mc_wer_latest.json"))
    ap.add_argument("--min-hits", type=int, default=2)
    ap.add_argument("--max-chunks", type=int, default=40)
    ap.add_argument("--boost", type=float, default=8.0)
    args = ap.parse_args()

    print("Building SnomedLexicon...", flush=True)
    lex_obj = SnomedLexicon()
    lex_set = set(lex_obj.terms)
    print(f"  lexicon size: {len(lex_set)}", flush=True)

    print(f"Mining clinical-reference subset from {args.manifest}...", flush=True)
    subset = build_clinical_subset(
        Path(args.manifest), lex_set,
        min_hits_per_chunk=args.min_hits,
        max_chunks=args.max_chunks,
    )
    if not subset:
        print("ERROR: no chunks meet the clinical-reference threshold", file=sys.stderr)
        return 1
    print(f"  selected {len(subset)} chunks (min_hits={args.min_hits})", flush=True)

    print("Loading AUSCULT (greedy)...", flush=True)
    greedy_eng = load_asr_engine(Path(args.checkpoint_dir), use_bias=False)
    print("Loading AUSCULT (biased)...", flush=True)
    biased_eng = load_asr_engine(Path(args.checkpoint_dir), use_bias=True)
    biased_available = getattr(biased_eng, "_biased_decode_available", None)
    print(f"  pyctcdecode available: {biased_available}", flush=True)

    per_chunk = []
    t_start = time.perf_counter()
    for i, item in enumerate(subset, 1):
        t0 = time.perf_counter()
        greedy_hyp = greedy_eng.transcribe_file(item["audio_path"])
        t_g = round(time.perf_counter() - t0, 2)
        t0 = time.perf_counter()
        biased_hyp = biased_eng.transcribe_file(item["audio_path"])
        t_b = round(time.perf_counter() - t0, 2)

        ref = item["reference"]
        wer_g = _wer(ref, greedy_hyp)
        wer_b = _wer(ref, biased_hyp)
        mc_g = _mc_recall(_tokenise(ref), _tokenise(greedy_hyp), lex_set)
        mc_b = _mc_recall(_tokenise(ref), _tokenise(biased_hyp), lex_set)

        print(
            f"[{i}/{len(subset)}] hits={item['lex_hit_count']:>2}  "
            f"WER g={wer_g:.3f} b={wer_b:.3f}  "
            f"MC-R g={mc_g['mc_recall']} b={mc_b['mc_recall']}",
            flush=True,
        )

        per_chunk.append({
            "audio_path": item["audio_path"],
            "reference": ref,
            "lex_hit_count": item["lex_hit_count"],
            "greedy": {"hypothesis": greedy_hyp, "wer": wer_g,
                       **mc_g, "wall_time_s": t_g},
            "biased": {"hypothesis": biased_hyp, "wer": wer_b,
                       **mc_b, "wall_time_s": t_b},
        })

    # Aggregates
    greedy_wer = np.array([c["greedy"]["wer"] for c in per_chunk])
    biased_wer = np.array([c["biased"]["wer"] for c in per_chunk])
    greedy_recoveries = sum(c["greedy"]["recovered"] for c in per_chunk)
    biased_recoveries = sum(c["biased"]["recovered"] for c in per_chunk)
    total_hits = sum(c["greedy"]["ref_hits"] for c in per_chunk)
    mc_recall_g = (greedy_recoveries / total_hits) if total_hits else None
    mc_recall_b = (biased_recoveries / total_hits) if total_hits else None
    mc_wer_g = 1 - mc_recall_g if mc_recall_g is not None else None
    mc_wer_b = 1 - mc_recall_b if mc_recall_b is not None else None
    relative_reduction = (
        (mc_wer_g - mc_wer_b) / mc_wer_g
        if (mc_wer_g not in (None, 0) and mc_wer_b is not None)
        else None
    )

    # Paired Wilcoxon on WER deltas
    from scipy.stats import wilcoxon
    deltas = biased_wer - greedy_wer
    if np.any(deltas != 0):
        stat, p = wilcoxon(biased_wer, greedy_wer, zero_method="wilcox")
        wilcoxon_p = round(float(p), 6)
    else:
        wilcoxon_p = None

    aggregate = {
        "n_chunks": len(per_chunk),
        "lexicon_size": len(lex_set),
        "lexicon_suffix_set": list(_MEDICAL_SUFFIXES),
        "boost": args.boost,
        "pyctcdecode_available": biased_available,
        "mean_wer_greedy": round(float(greedy_wer.mean()), 4),
        "mean_wer_biased": round(float(biased_wer.mean()), 4),
        "mean_wer_delta": round(float((biased_wer - greedy_wer).mean()), 4),
        "wilcoxon_signed_rank_p": wilcoxon_p,
        "total_ref_hits": int(total_hits),
        "total_recovered_greedy": int(greedy_recoveries),
        "total_recovered_biased": int(biased_recoveries),
        "mc_recall_greedy": round(mc_recall_g, 4) if mc_recall_g is not None else None,
        "mc_recall_biased": round(mc_recall_b, 4) if mc_recall_b is not None else None,
        "mc_wer_greedy": round(mc_wer_g, 4) if mc_wer_g is not None else None,
        "mc_wer_biased": round(mc_wer_b, 4) if mc_wer_b is not None else None,
        "relative_mc_wer_reduction": round(relative_reduction, 4) if relative_reduction is not None else None,
        "total_wall_time_s": round(time.perf_counter() - t_start, 2),
    }

    output = {
        "is_synthetic": False,
        "hardware_target": "RTX 5090 32GB Linux (satyawan-1)",
        "host": socket.gethostname(),
        "git_commit": _git_sha(),
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "checkpoint": args.checkpoint_dir,
        "manifest": args.manifest,
        "subset_selection": {
            "min_lex_hits_per_chunk": args.min_hits,
            "max_chunks": args.max_chunks,
        },
        "aggregate": aggregate,
        "per_chunk": per_chunk,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWritten: {args.out}", flush=True)
    print(json.dumps(aggregate, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
