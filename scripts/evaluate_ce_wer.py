#!/usr/bin/env python3
"""Clinical Entity WER (CE-WER) + Medical Term Error Rate (MTER).

Re-uses the per-sample hypotheses already captured for greedy and beam decoders
in data/evaluation_results/wer_decoding_sweep_final.json. Computes:

  1. CE-WER: standard Levenshtein WER restricted to the Clinical Entity Pool
             (lexicon hits ∪ drug-interaction names ∪ suffix-mined morphemes).
  2. MTER : exact / suffix-only / omission rate per pharmacological suffix
             class (-cillin, -azole, -olol, -pril, -statin, ...).

Output: data/evaluation_results/ce_wer_metrics.json
        data/evaluation_results/ce_wer_metrics.tex   (numerical matrix only)
        data/evaluation_results/ce_wer_metrics.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import socket
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.speech_to_text.snomed_biasing import SnomedLexicon, _MEDICAL_SUFFIXES  # noqa: E402


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()


def _tokens(text: str) -> list[str]:
    return [w for w in _norm(text).split() if w]


def _levenshtein_word(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """Word-level Levenshtein. Returns (substitutions, deletions, insertions)."""
    m, n = len(ref), len(hyp)
    # Build DP with backpointer info: each cell stores the (cost, op) chain
    # Op encoding: 0=match, 1=sub, 2=del, 3=ins
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    back = [[None] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
        back[i][0] = 2 if i > 0 else None
    for j in range(n + 1):
        dp[0][j] = j
        back[0][j] = 3 if j > 0 else None
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                back[i][j] = 0
                continue
            sub = dp[i - 1][j - 1] + 1
            dele = dp[i - 1][j] + 1
            ins = dp[i][j - 1] + 1
            best = min(sub, dele, ins)
            dp[i][j] = best
            back[i][j] = 1 if best == sub else (2 if best == dele else 3)
    # Walk back
    s_count = d_count = i_count = 0
    i, j = m, n
    while i > 0 or j > 0:
        op = back[i][j]
        if op == 0:
            i -= 1
            j -= 1
        elif op == 1:
            s_count += 1
            i -= 1
            j -= 1
        elif op == 2:
            d_count += 1
            i -= 1
        elif op == 3:
            i_count += 1
            j -= 1
        else:
            break
    return s_count, d_count, i_count


def _bootstrap_ci(values: list[float], seed: int = 42, n_iter: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(values)
    means = np.array([arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n_iter)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _drug_interaction_terms() -> set[str]:
    """Pull unique drug names from src/clinical/drug_interactions.py:INTERACTION_DB."""
    from src.clinical.drug_interactions import INTERACTION_DB
    s = set()
    for row in INTERACTION_DB:
        for k in ("drug_a", "drug_b"):
            v = row.get(k)
            if isinstance(v, str):
                s.add(v.lower())
    return s


def _build_clinical_mask() -> dict:
    lex = set(SnomedLexicon().terms)
    drugs = _drug_interaction_terms()
    mask = {
        "snomed_lexicon": lex,
        "drug_interaction_terms": drugs,
        "combined": lex | drugs,
        "suffixes": tuple(_MEDICAL_SUFFIXES),
    }
    return mask


def _is_clinical(token: str, mask: dict) -> bool:
    if token in mask["combined"]:
        return True
    if len(token) >= 5 and token.endswith(mask["suffixes"]):
        return True
    return False


def _partition_clinical(tokens: list[str], mask: dict) -> tuple[list[str], list[str]]:
    clinical, functional = [], []
    for t in tokens:
        (clinical if _is_clinical(t, mask) else functional).append(t)
    return clinical, functional


def ce_wer_for_pair(ref: str, hyp: str, mask: dict) -> dict:
    ref_toks = _tokens(ref)
    hyp_toks = _tokens(hyp)
    ref_clin, ref_func = _partition_clinical(ref_toks, mask)
    hyp_clin, hyp_func = _partition_clinical(hyp_toks, mask)

    if ref_clin:
        s, d, i = _levenshtein_word(ref_clin, hyp_clin)
        ce_wer = (s + d + i) / len(ref_clin)
    else:
        s = d = i = 0
        ce_wer = None

    if ref_toks:
        s_overall, d_overall, i_overall = _levenshtein_word(ref_toks, hyp_toks)
        wer_overall = (s_overall + d_overall + i_overall) / len(ref_toks)
    else:
        s_overall = d_overall = i_overall = 0
        wer_overall = None

    return {
        "n_ref_tokens": len(ref_toks),
        "n_ref_clinical": len(ref_clin),
        "n_ref_functional": len(ref_func),
        "n_hyp_tokens": len(hyp_toks),
        "n_hyp_clinical": len(hyp_clin),
        "wer_overall": wer_overall,
        "wer_overall_breakdown": {"subs": s_overall, "dels": d_overall, "ins": i_overall},
        "ce_wer": ce_wer,
        "ce_wer_breakdown": {"subs": s, "dels": d, "ins": i},
        "missed_clinical_tokens": sorted(set(ref_clin) - set(hyp_clin))[:30],
        "extra_clinical_in_hyp": sorted(set(hyp_clin) - set(ref_clin))[:30],
    }


def mter_for_pair(ref: str, hyp: str, suffixes: tuple[str, ...]) -> dict:
    """For each suffix, measure exact-match / suffix-only-match / complete-omission."""
    ref_toks = _tokens(ref)
    hyp_set = set(_tokens(hyp))
    by_suffix: dict[str, dict[str, int | list[str]]] = {}
    for s in suffixes:
        bucket = {"ref_hits": 0, "exact_match": 0, "suffix_only_match": 0,
                  "complete_omission": 0, "missed_terms": []}
        for tok in ref_toks:
            if len(tok) >= 5 and tok.endswith(s):
                bucket["ref_hits"] += 1
                if tok in hyp_set:
                    bucket["exact_match"] += 1
                elif any(h.endswith(s) and len(h) >= 5 for h in hyp_set):
                    bucket["suffix_only_match"] += 1
                    bucket["missed_terms"].append(tok)
                else:
                    bucket["complete_omission"] += 1
                    bucket["missed_terms"].append(tok)
        by_suffix[s] = bucket
    return by_suffix


def _latex_table(per_decoder: dict, agg: dict) -> str:
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Decoder & $n$ & Mean WER (\%) & Mean CE-WER (\%) & CE-WER 95\% CI (\%) & Clinical tokens / sample \\",
        r"\midrule",
    ]
    for decoder, payload in per_decoder.items():
        a = agg[decoder]
        lines.append(
            f"{decoder.replace('_', r'\_')} & {a['n_samples']} & "
            f"{a['mean_wer_overall']*100:.2f} & {a['mean_ce_wer']*100:.2f} & "
            f"[{a['ce_wer_ci_lower']*100:.2f}, {a['ce_wer_ci_upper']*100:.2f}] & "
            f"{a['mean_clinical_tokens_per_sample']:.1f} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def _mter_latex(mter_agg: dict) -> str:
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Suffix & Ref hits & Exact match & Suffix-only match & Complete omission & Exact-recovery rate (\%) \\",
        r"\midrule",
    ]
    for suffix, row in mter_agg.items():
        rec = (row["exact_match"] / row["ref_hits"] * 100) if row["ref_hits"] else 0.0
        lines.append(
            f"{suffix} & {row['ref_hits']} & {row['exact_match']} & "
            f"{row['suffix_only_match']} & {row['complete_omission']} & {rec:.1f} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-json",
                    default=str(REPO_ROOT / "data/evaluation_results/wer_decoding_sweep_final.json"))
    ap.add_argument("--out",
                    default=str(REPO_ROOT / "data/evaluation_results/ce_wer_metrics.json"))
    args = ap.parse_args()

    sweep = json.loads(Path(args.sweep_json).read_text())
    test_entries_lookup = {}
    # test_json gives us references
    test_json = REPO_ROOT / "data/processed/primock57_chunked/test.json"
    test_data = json.loads(test_json.read_text())
    refs_by_basename = {Path(e["audio_path"]).name: e["text"] for e in test_data}

    mask = _build_clinical_mask()
    print(f"Clinical Entity Mask:", flush=True)
    print(f"  snomed_lexicon       : {len(mask['snomed_lexicon'])}", flush=True)
    print(f"  drug_interaction_terms: {len(mask['drug_interaction_terms'])}", flush=True)
    print(f"  combined (union)     : {len(mask['combined'])}", flush=True)
    print(f"  medical suffixes     : {len(mask['suffixes'])}", flush=True)

    decoders = {}  # decoder_label -> list of per_sample dicts
    for entry in sweep["per_sample"]:
        basename = entry["audio_file"]
        ref = refs_by_basename.get(basename, "")
        if not ref:
            continue
        for key in ("greedy", "beam_nobias"):
            payload = entry.get(key, {})
            hyp = payload.get("hypothesis")
            if hyp is None:
                continue
            label = key
            cew = ce_wer_for_pair(ref, hyp, mask)
            mter = mter_for_pair(ref, hyp, mask["suffixes"])
            row = {
                "audio_file": basename,
                "wer_overall": cew["wer_overall"],
                "ce_wer": cew["ce_wer"],
                "wer_breakdown": cew["wer_overall_breakdown"],
                "ce_wer_breakdown": cew["ce_wer_breakdown"],
                "n_ref_clinical": cew["n_ref_clinical"],
                "n_ref_functional": cew["n_ref_functional"],
                "missed_clinical_tokens": cew["missed_clinical_tokens"],
                "extra_clinical_in_hyp": cew["extra_clinical_in_hyp"],
                "mter": mter,
            }
            decoders.setdefault(label, []).append(row)

    # Aggregate per decoder
    agg = {}
    mter_agg_per_decoder = {}
    for label, rows in decoders.items():
        ce_wers = [r["ce_wer"] for r in rows if r["ce_wer"] is not None]
        wers = [r["wer_overall"] for r in rows if r["wer_overall"] is not None]
        ci_lo, ci_hi = _bootstrap_ci(ce_wers)
        agg[label] = {
            "n_samples": len(rows),
            "mean_wer_overall": round(float(np.mean(wers)), 4),
            "mean_ce_wer": round(float(np.mean(ce_wers)), 4),
            "ce_wer_ci_lower": round(ci_lo, 4),
            "ce_wer_ci_upper": round(ci_hi, 4),
            "median_ce_wer": round(float(np.median(ce_wers)), 4),
            "mean_clinical_tokens_per_sample": round(
                float(np.mean([r["n_ref_clinical"] for r in rows])), 2
            ),
            "total_clinical_ref_tokens": int(sum(r["n_ref_clinical"] for r in rows)),
        }

        # Aggregate MTER across all 14 samples for this decoder
        per_suffix = {}
        for r in rows:
            for suffix, b in r["mter"].items():
                acc = per_suffix.setdefault(suffix, {"ref_hits": 0, "exact_match": 0,
                                                     "suffix_only_match": 0,
                                                     "complete_omission": 0,
                                                     "sample_missed_terms": []})
                acc["ref_hits"] += b["ref_hits"]
                acc["exact_match"] += b["exact_match"]
                acc["suffix_only_match"] += b["suffix_only_match"]
                acc["complete_omission"] += b["complete_omission"]
                acc["sample_missed_terms"].extend(b["missed_terms"][:5])
        mter_agg_per_decoder[label] = per_suffix

    # Print summary
    print("\n== CE-WER aggregates ==", flush=True)
    for label, a in agg.items():
        print(f"  {label:<15}  overall WER {a['mean_wer_overall']*100:.2f}%  "
              f"CE-WER {a['mean_ce_wer']*100:.2f}%  "
              f"95% CI [{a['ce_wer_ci_lower']*100:.2f}, {a['ce_wer_ci_upper']*100:.2f}]  "
              f"n_clin/sample {a['mean_clinical_tokens_per_sample']:.1f}",
              flush=True)
    print("\n== MTER per suffix (greedy decoder) ==", flush=True)
    g_mter = mter_agg_per_decoder.get("greedy", {})
    suffixes_w_hits = [(s, b) for s, b in g_mter.items() if b["ref_hits"] > 0]
    suffixes_w_hits.sort(key=lambda kv: kv[1]["ref_hits"], reverse=True)
    for s, b in suffixes_w_hits[:15]:
        rec = (b["exact_match"] / b["ref_hits"] * 100) if b["ref_hits"] else 0.0
        print(f"  -{s:<10} hits={b['ref_hits']:>3}  exact={b['exact_match']:>3}  "
              f"suffix_only={b['suffix_only_match']:>3}  omit={b['complete_omission']:>3}  "
              f"recovery={rec:.1f}%", flush=True)

    out = {
        "is_synthetic": False,
        "host": socket.gethostname(),
        "git_commit": _git_sha(),
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_sweep_json": args.sweep_json,
        "clinical_entity_mask": {
            "n_snomed_lexicon": len(mask["snomed_lexicon"]),
            "n_drug_interaction_terms": len(mask["drug_interaction_terms"]),
            "n_combined": len(mask["combined"]),
            "medical_suffixes": list(mask["suffixes"]),
        },
        "per_decoder": {
            label: {"per_sample": rows, "aggregate": agg[label],
                    "mter_aggregate": mter_agg_per_decoder[label]}
            for label, rows in decoders.items()
        },
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    latex_main = _latex_table(decoders, agg)
    latex_mter = "\n".join([f"% --- MTER per suffix ({label}) ---\n" + _mter_latex(mter_agg_per_decoder[label])
                            for label in decoders.keys()])
    Path(args.out).with_suffix(".tex").write_text(latex_main + "\n\n" + latex_mter)

    md_lines = ["## CE-WER aggregates\n",
                "| Decoder | n | Mean WER | Mean CE-WER | CE-WER 95% CI | Clinical tokens / sample |",
                "|---|---|---|---|---|---|"]
    for label, a in agg.items():
        md_lines.append(
            f"| `{label}` | {a['n_samples']} | {a['mean_wer_overall']*100:.2f}% | "
            f"{a['mean_ce_wer']*100:.2f}% | "
            f"[{a['ce_wer_ci_lower']*100:.2f}%, {a['ce_wer_ci_upper']*100:.2f}%] | "
            f"{a['mean_clinical_tokens_per_sample']:.1f} |"
        )
    for label, per_suffix in mter_agg_per_decoder.items():
        md_lines.append(f"\n## MTER per suffix ({label})\n")
        md_lines.append("| Suffix | Ref hits | Exact | Suffix-only | Omission | Recovery |")
        md_lines.append("|---|---|---|---|---|---|")
        for s, b in sorted(per_suffix.items(), key=lambda kv: kv[1]["ref_hits"], reverse=True):
            if b["ref_hits"] == 0:
                continue
            rec = (b["exact_match"] / b["ref_hits"] * 100)
            md_lines.append(f"| -{s} | {b['ref_hits']} | {b['exact_match']} | "
                            f"{b['suffix_only_match']} | {b['complete_omission']} | {rec:.1f}% |")
    Path(args.out).with_suffix(".md").write_text("\n".join(md_lines) + "\n")

    print(f"\nWritten:")
    print(f"  {args.out}")
    print(f"  {Path(args.out).with_suffix('.tex')}")
    print(f"  {Path(args.out).with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
