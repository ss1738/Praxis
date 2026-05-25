#!/usr/bin/env python3
"""Build a clinical-domain 4-gram KenLM ARPA from the in-repo clinical sources.

Corpus sources (no test-set leakage):
  - data/processed/primock57_chunked/train.json (transcripts of training chunks)
  - data/processed/primock57_chunked/val.json (transcripts of val chunks)
  - src/clinical/drug_interactions.py:INTERACTION_DB (drug names + effect strings)
  - src/clinical/nice_guidelines.py:NICE_CKS (condition names + key_checks)
  - src/speech_to_text/snomed_biasing.py:SnomedLexicon (185 UK primary-care terms)

Pipeline:
  1. Harvest text, lower-case + light normalisation.
  2. Write to data/models/lm/clinical_gp_corpus.txt
  3. Call `lmplz -o 4 --discount_fallback` to build ARPA → data/models/lm/clinical_gp_4gram.arpa
  4. Also build a binary via `build_binary` for faster pyctcdecode load.

Requires kenlm tools (lmplz, build_binary) on PATH.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _norm(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)  # strip <INAUDIBLE_SPEECH/> etc.
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def harvest_corpus(out_path: Path) -> dict:
    counts = {}
    lines: list[str] = []

    # 1. Training + validation transcripts
    for split in ("train", "val"):
        p = REPO_ROOT / "data/processed/primock57_chunked" / f"{split}.json"
        if not p.exists():
            print(f"  WARN: {p} missing, skipping")
            continue
        data = json.loads(p.read_text())
        for entry in data:
            txt = _norm(entry.get("text", ""))
            if txt:
                lines.append(txt)
        counts[f"primock57_{split}"] = len(data)

    # 2. Drug interaction terms
    try:
        from src.clinical.drug_interactions import INTERACTION_DB
        for row in INTERACTION_DB:
            for k in ("drug_a", "drug_b"):
                v = row.get(k)
                if isinstance(v, str):
                    lines.append(_norm(v))
            for k in ("effect", "clinical_action"):
                v = row.get(k)
                if isinstance(v, str):
                    lines.append(_norm(v))
        counts["drug_interaction_db_rows"] = len(INTERACTION_DB)
    except Exception as exc:
        print(f"  WARN: drug_interactions import failed: {exc}")

    # 3. NICE CKS conditions
    try:
        from src.clinical.nice_guidelines import NICE_CKS
        for row in NICE_CKS:
            lines.append(_norm(row.get("condition", "")))
            lines.append(_norm(row.get("nice_ref", "")))
            for kc in row.get("key_checks", []):
                if isinstance(kc, str):
                    lines.append(_norm(kc))
            for rf in row.get("red_flags", []):
                if isinstance(rf, str):
                    lines.append(_norm(rf))
        counts["nice_cks_conditions"] = len(NICE_CKS)
    except Exception as exc:
        print(f"  WARN: nice_guidelines import failed: {exc}")

    # 4. SNOMED biasing lexicon
    try:
        from src.speech_to_text.snomed_biasing import SnomedLexicon
        terms = SnomedLexicon().terms
        for term in terms:
            lines.append(_norm(term))
        counts["snomed_lexicon_terms"] = len(terms)
    except Exception as exc:
        print(f"  WARN: SnomedLexicon import failed: {exc}")

    lines = [ln for ln in lines if ln]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    counts["total_lines_written"] = len(lines)
    counts["total_tokens"] = sum(len(ln.split()) for ln in lines)
    return counts


def build_arpa(corpus_path: Path, arpa_path: Path, order: int = 4) -> dict:
    """Build a 4-gram ARPA. Prefers lmplz if available, else uses a Python
    Stupid-Backoff builder (Brants et al. 2007) so the pipeline runs in
    environments without the kenlm C++ tools.
    """
    arpa_path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("lmplz"):
        cmd = ["lmplz", "-o", str(order), "--discount_fallback"]
        print(f"running: {' '.join(cmd)} < {corpus_path}", flush=True)
        with open(corpus_path) as fin, open(arpa_path, "w") as fout:
            proc = subprocess.run(cmd, stdin=fin, stdout=fout, stderr=subprocess.PIPE)
        stderr = proc.stderr.decode("utf-8", errors="replace")
        return {
            "builder": "lmplz",
            "returncode": proc.returncode,
            "stderr_tail": stderr.splitlines()[-8:],
            "arpa_size_bytes": arpa_path.stat().st_size if arpa_path.exists() else 0,
        }

    # Python fallback: Stupid Backoff with MLE log-probabilities
    print("lmplz unavailable - using Python Stupid-Backoff ARPA builder",
          flush=True)
    info = _python_arpa_stupid_backoff(corpus_path, arpa_path, order=order)
    return info


# ── Python Stupid-Backoff ARPA builder ─────────────────────────────────────

import math
from collections import defaultdict


def _python_arpa_stupid_backoff(corpus_path: Path, arpa_path: Path,
                                  order: int = 4,
                                  backoff_weight: float = 0.4) -> dict:
    """Build an ARPA file using Stupid Backoff (Brants et al. 2007).

    For each n-gram of orders 1..N:
       logprob = log10(count(ngram) / count(prefix))   if count > 0
    Backoff weight for non-highest orders is fixed log10(0.4) ≈ -0.398.

    Adds <s>, </s>, and <unk> special tokens. Lower-cases everything.
    """
    BOS = "<s>"
    EOS = "</s>"
    UNK = "<unk>"

    counts: dict[int, dict[tuple, int]] = {n: defaultdict(int) for n in range(1, order + 1)}
    total_tokens = 0
    n_sentences = 0

    with open(corpus_path) as f:
        for line in f:
            toks = line.strip().split()
            if not toks:
                continue
            toks = [BOS] * (order - 1) + toks + [EOS]
            n_sentences += 1
            total_tokens += len(toks)
            for n in range(1, order + 1):
                for i in range(len(toks) - n + 1):
                    ngram = tuple(toks[i:i + n])
                    counts[n][ngram] += 1

    # Add UNK with a small count so it appears in the unigram table
    counts[1][(UNK,)] = max(counts[1].get((UNK,), 0), 1)

    # Probability for an n-gram: count(ngram) / count(prefix)
    def _logprob(ngram: tuple) -> float:
        n = len(ngram)
        c = counts[n][ngram]
        if c == 0:
            return -99.0
        if n == 1:
            denom = sum(counts[1].values())
        else:
            denom = counts[n - 1][ngram[:-1]]
        if denom == 0:
            return -99.0
        return math.log10(c / denom)

    log_bo = math.log10(backoff_weight)

    # Write ARPA
    with open(arpa_path, "w") as f:
        f.write("\n\\data\\\n")
        for n in range(1, order + 1):
            f.write(f"ngram {n}={len(counts[n])}\n")
        f.write("\n")

        for n in range(1, order + 1):
            f.write(f"\\{n}-grams:\n")
            # Sort by token sequence for stable output
            for ngram in sorted(counts[n].keys()):
                lp = _logprob(ngram)
                # backoff weight applies to all orders < highest
                if n < order:
                    f.write(f"{lp:.6f}\t{' '.join(ngram)}\t{log_bo:.6f}\n")
                else:
                    f.write(f"{lp:.6f}\t{' '.join(ngram)}\n")
            f.write("\n")

        f.write("\\end\\\n")

    return {
        "builder": "python_stupid_backoff",
        "order": order,
        "backoff_weight": backoff_weight,
        "n_sentences": n_sentences,
        "total_tokens": total_tokens,
        "ngram_counts": {n: len(counts[n]) for n in range(1, order + 1)},
        "arpa_size_bytes": arpa_path.stat().st_size,
        "returncode": 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-out",
                    default=str(REPO_ROOT / "data/models/lm/clinical_gp_corpus.txt"))
    ap.add_argument("--arpa-out",
                    default=str(REPO_ROOT / "data/models/lm/clinical_gp_4gram.arpa"))
    ap.add_argument("--order", type=int, default=4)
    args = ap.parse_args()

    print("== Harvesting corpus ==")
    counts = harvest_corpus(Path(args.corpus_out))
    print(json.dumps(counts, indent=2))

    print(f"\n== Building {args.order}-gram ARPA ==")
    build_info = build_arpa(Path(args.corpus_out), Path(args.arpa_out), order=args.order)
    print(json.dumps(build_info, indent=2))


if __name__ == "__main__":
    main()
