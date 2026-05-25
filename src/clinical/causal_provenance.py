"""Causal provenance via O(log T) deterministic bisection (N18).

Adapts the PRAXIS-VOX audio-bisection algorithm
(`praxis_vox/provenance/bisection.py`) to operate on a transcript and a
generated SOAP sentence. For each SOAP sentence, find the minimal contiguous
transcript window that "causes" the sentence - defined as the smallest token
span S* such that an entailment scorer on (S*, sentence) preserves the
sentence's full-transcript entailment score within tolerance ε.

Why bisection rather than gradient saliency:
  - Gradient saliency depends on baseline choice and is non-replayable.
  - Bisection is deterministic. Same input ⇒ same span, every time.

Why bisection rather than the existing N2 source-linking (Dice overlap):
  - N2 finds spans that *correlate* with a sentence (lexical overlap).
  - Bisection finds spans that *entail* the sentence (causal sufficiency).
  - The two coincide on simple sentences; they diverge on summarised /
    paraphrased clinical claims, which is where forensic accuracy matters.

API:
    causal_provenance(transcript, sentence, nli_fn, ...)
        -> ProvenanceSpan(start_token, end_token, ...)

    replayability_test(...) -> dict   # zero-variance check across reruns
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np


# ── Data type ────────────────────────────────────────────────────────


@dataclass
class ProvenanceSpan:
    """Minimal causal transcript span for one SOAP sentence."""
    sentence: str
    full_transcript_score: float
    span_start_token: int
    span_end_token: int          # exclusive
    span_text: str
    span_score: float
    span_n_tokens: int
    full_n_tokens: int
    n_bisection_steps: int
    epsilon: float
    threshold: float
    span_fraction: float         # span_n_tokens / full_n_tokens

    def to_dict(self) -> dict:
        return asdict(self)


# ── Tokenisation ─────────────────────────────────────────────────────

_WHITESPACE = re.compile(r"\s+")


def _tokenize(text: str) -> list[str]:
    """Whitespace tokenisation. Deterministic, replayable, language-agnostic
    enough for English clinical prose."""
    return [t for t in _WHITESPACE.split(text.strip()) if t]


def _join(tokens: list[str], start: int, end: int) -> str:
    return " ".join(tokens[start:end])


# ── Bisection core ───────────────────────────────────────────────────


def causal_provenance(
    transcript: str,
    sentence: str,
    nli_fn: Callable[[str, str], float],
    *,
    epsilon: float = 0.05,
    threshold: float = 0.5,
    min_span_tokens: int = 5,
    max_iters: int = 16,
) -> ProvenanceSpan:
    """Find the minimal contiguous transcript span that preserves the
    sentence's NLI entailment score within tolerance ε.

    Args:
        transcript: full consultation transcript (string).
        sentence: one SOAP sentence to attribute.
        nli_fn: deterministic callable(premise, hypothesis) -> P(entailment).
            Must return the same value for the same arguments across calls.
            For PRAXIS we use the cached EntailmentGate cross-encoder.
        epsilon: tolerance - span is "preserving" iff its score is within
            ε of the full-transcript score.
        threshold: minimum P(entailment) required to consider the sentence
            attributable at all. If even the full transcript scores below
            threshold, returns the full span as the trivial answer.
        min_span_tokens: floor on span size (avoid degenerate single-token spans).
        max_iters: bisection budget. 16 steps gives sub-token resolution on
            transcripts up to ~65k tokens.

    Returns:
        ProvenanceSpan with span_start_token, span_end_token, and diagnostics.

    Algorithm (mirrors praxis_vox.provenance.bisection.provenance_span):
        1. Score the full transcript.
        2. Coarse scan: 16 candidate windows at half-overlap; pick the
           highest-scoring centre.
        3. Symmetric bisection outward from that centre: shrink the radius
           while the score stays within ε of full; grow it otherwise.

    Complexity: O(max_iters) calls to nli_fn. Deterministic.
    """
    if max_iters < 1:
        raise ValueError(f"max_iters must be >= 1, got {max_iters}")
    if min_span_tokens < 1:
        raise ValueError(f"min_span_tokens must be >= 1, got {min_span_tokens}")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    if epsilon < 0.0:
        raise ValueError(f"epsilon must be >= 0, got {epsilon}")

    tokens = _tokenize(transcript)
    n = len(tokens)

    if n == 0:
        return ProvenanceSpan(
            sentence=sentence,
            full_transcript_score=0.0,
            span_start_token=0, span_end_token=0,
            span_text="", span_score=0.0, span_n_tokens=0,
            full_n_tokens=0, n_bisection_steps=0,
            epsilon=epsilon, threshold=threshold, span_fraction=0.0,
        )

    p_full = float(nli_fn(transcript, sentence))

    # If even the full transcript doesn't entail the sentence, return the
    # whole-transcript span as the trivial answer.
    if p_full < threshold:
        return ProvenanceSpan(
            sentence=sentence,
            full_transcript_score=p_full,
            span_start_token=0, span_end_token=n,
            span_text=transcript, span_score=p_full, span_n_tokens=n,
            full_n_tokens=n, n_bisection_steps=0,
            epsilon=epsilon, threshold=threshold, span_fraction=1.0,
        )

    # Coarse localisation: scan 16 candidate windows at half-overlap.
    n_windows = min(16, max(1, n // max(1, min_span_tokens)))
    win = max(min_span_tokens, n // n_windows)
    centres_t = np.linspace(win / 2, n - win / 2, n_windows).astype(int)
    scores = np.full(n_windows, -np.inf, dtype=float)
    for i, c in enumerate(centres_t):
        a = max(0, c - win // 2)
        b = min(n, c + win // 2)
        if b - a < min_span_tokens:
            continue
        scores[i] = float(nli_fn(_join(tokens, a, b), sentence))
    centre = int(centres_t[int(np.argmax(scores))])

    # Symmetric bisection outward
    radius_lo = max(1, min_span_tokens // 2)
    radius_hi = max(centre, n - centre)
    best_a = max(0, centre - radius_hi)
    best_b = min(n, centre + radius_hi)
    best_p = p_full
    n_steps = 0
    for _ in range(max_iters):
        n_steps += 1
        radius_mid = (radius_lo + radius_hi) // 2
        if radius_mid == radius_lo:
            break
        a = max(0, centre - radius_mid)
        b = min(n, centre + radius_mid)
        if b - a < min_span_tokens:
            radius_lo = radius_mid
            continue
        p_seg = float(nli_fn(_join(tokens, a, b), sentence))
        if abs(p_seg - p_full) <= epsilon:
            # Preserved - try smaller
            best_a, best_b = a, b
            best_p = p_seg
            radius_hi = radius_mid
        else:
            radius_lo = radius_mid

    span_text = _join(tokens, best_a, best_b)
    span_n = best_b - best_a
    return ProvenanceSpan(
        sentence=sentence,
        full_transcript_score=p_full,
        span_start_token=int(best_a),
        span_end_token=int(best_b),
        span_text=span_text,
        span_score=float(best_p),
        span_n_tokens=int(span_n),
        full_n_tokens=int(n),
        n_bisection_steps=int(n_steps),
        epsilon=float(epsilon),
        threshold=float(threshold),
        span_fraction=float(span_n / n) if n else 0.0,
    )


# ── Replayability ────────────────────────────────────────────────────


def replayability_test(
    transcript: str,
    sentence: str,
    nli_fn: Callable[[str, str], float],
    *,
    n_runs: int = 5,
    **kwargs,
) -> dict:
    """Empirical replayability: identical input must produce identical span.

    For a deterministic nli_fn, all n_runs must return the same span. Any
    non-zero variance indicates non-determinism somewhere in the stack
    (e.g., a stochastic feature function or a non-deterministic NLI model
    run on GPU).

    Returns a dict suitable for serialisation into the dissertation
    replayability table.
    """
    if n_runs < 2:
        raise ValueError(f"n_runs must be >= 2, got {n_runs}")
    starts: list[int] = []
    ends: list[int] = []
    scores: list[float] = []
    for _ in range(n_runs):
        r = causal_provenance(transcript, sentence, nli_fn, **kwargs)
        starts.append(r.span_start_token)
        ends.append(r.span_end_token)
        scores.append(r.span_score)
    s_arr = np.asarray(starts)
    e_arr = np.asarray(ends)
    sc_arr = np.asarray(scores)
    return {
        "n_runs": n_runs,
        "start_std": float(s_arr.std()),
        "end_std": float(e_arr.std()),
        "score_std": float(sc_arr.std()),
        "all_starts": starts,
        "all_ends": ends,
        "exact_replayable": bool(s_arr.std() < 1e-9 and e_arr.std() < 1e-9),
    }


# ── Comparison to N2 source linking (Dice) ───────────────────────────


def dice_baseline_span(
    transcript: str,
    sentence: str,
    *,
    min_span_tokens: int = 5,
    window_tokens: int = 50,
) -> ProvenanceSpan:
    """Sliding-window Dice-overlap baseline matching N2 source linking.

    For each candidate window, compute |W ∩ S| / (|W| + |S|) where W and S
    are token sets. Return the highest-scoring window of size window_tokens.
    Used to compare bisection (causal) against Dice (correlational) in the
    dissertation table.

    The returned ProvenanceSpan repurposes `span_score` as the Dice score
    so the same data structure can be used for both methods.
    """
    tokens = _tokenize(transcript)
    n = len(tokens)
    s_tokens = set(_tokenize(sentence))
    if n == 0 or not s_tokens:
        return ProvenanceSpan(
            sentence=sentence, full_transcript_score=0.0,
            span_start_token=0, span_end_token=0,
            span_text="", span_score=0.0, span_n_tokens=0,
            full_n_tokens=n, n_bisection_steps=0,
            epsilon=0.0, threshold=0.0, span_fraction=0.0,
        )
    w = min(window_tokens, n)
    best = (0, w, 0.0)
    for a in range(0, max(1, n - w + 1)):
        b = a + w
        chunk = set(t.lower() for t in tokens[a:b])
        s_low = set(t.lower() for t in s_tokens)
        intersection = len(chunk & s_low)
        if intersection == 0:
            score = 0.0
        else:
            score = (2 * intersection) / (len(chunk) + len(s_low))
        if score > best[2]:
            best = (a, b, score)
    a, b, score = best
    return ProvenanceSpan(
        sentence=sentence, full_transcript_score=score,
        span_start_token=a, span_end_token=b,
        span_text=_join(tokens, a, b),
        span_score=float(score),
        span_n_tokens=b - a, full_n_tokens=n,
        n_bisection_steps=0, epsilon=0.0, threshold=0.0,
        span_fraction=(b - a) / n if n else 0.0,
    )
