"""Tests for causal-provenance bisection on SOAP sentences (N18)."""
from __future__ import annotations

import math

import pytest

from src.clinical.causal_provenance import (
    ProvenanceSpan,
    _tokenize,
    causal_provenance,
    dice_baseline_span,
    replayability_test,
)


# ── Tokenisation ──────────────────────────────────────────────────────


class TestTokenize:
    def test_simple(self):
        assert _tokenize("Hello world") == ["Hello", "world"]

    def test_extra_whitespace(self):
        assert _tokenize("  Hello   world  \t\n") == ["Hello", "world"]

    def test_empty(self):
        assert _tokenize("") == []
        assert _tokenize("   ") == []


# ── Synthetic NLI scorer ──────────────────────────────────────────────


def make_keyword_scorer(keywords: list[str]):
    """Synthetic deterministic scorer.

    Returns 1.0 if all keywords are in premise (case-insensitive),
    fraction-of-keywords-present otherwise. Replayable.
    """
    kws = [k.lower() for k in keywords]

    def score(premise: str, hypothesis: str) -> float:
        if not kws:
            return 0.0
        p_low = premise.lower()
        present = sum(1 for k in kws if k in p_low)
        return present / len(kws)

    return score


# ── Core bisection ────────────────────────────────────────────────────


class TestCausalProvenance:
    def test_returns_span_dataclass(self):
        nli = make_keyword_scorer(["fever", "cough"])
        r = causal_provenance(
            "patient says fever and cough today",
            "patient has fever and cough",
            nli,
        )
        assert isinstance(r, ProvenanceSpan)
        assert r.full_transcript_score >= 0.99   # all keywords present

    def test_empty_transcript_returns_empty_span(self):
        nli = make_keyword_scorer(["x"])
        r = causal_provenance("", "any sentence", nli)
        assert r.span_n_tokens == 0
        assert r.span_text == ""

    def test_below_threshold_returns_full_span(self):
        # No keywords present in transcript -> score 0 -> below threshold
        nli = make_keyword_scorer(["nonexistent"])
        transcript = ("a b c d e f g h i j k l m n o p q r s t u v w x y z "
                      "a b c d e f g h")
        r = causal_provenance(transcript, "sentence", nli, threshold=0.5)
        # score is 0; below threshold -> trivial answer is full span
        assert r.span_n_tokens == r.full_n_tokens

    def test_finds_minimal_span(self):
        # Bury the keywords in a long irrelevant prefix and suffix
        prefix = " ".join(["filler"] * 50)
        relevant = "patient reports fever and cough"
        suffix = " ".join(["padding"] * 50)
        transcript = f"{prefix} {relevant} {suffix}"
        nli = make_keyword_scorer(["fever", "cough"])

        r = causal_provenance(transcript, "fever and cough",
                              nli, epsilon=0.1)
        # The bisected span should be much smaller than the full transcript
        # while still containing both keywords.
        assert r.span_fraction < 0.6
        assert "fever" in r.span_text.lower()
        assert "cough" in r.span_text.lower()
        assert r.span_score >= 0.99    # both keywords still present

    def test_increasing_epsilon_widens_acceptable_spans(self):
        # Larger eps tolerates larger drift, may pick a smaller span
        prefix = " ".join(["x"] * 30)
        suffix = " ".join(["y"] * 30)
        transcript = f"{prefix} fever cough {suffix}"
        nli = make_keyword_scorer(["fever", "cough"])

        r_tight = causal_provenance(transcript, "f c", nli, epsilon=0.01)
        r_loose = causal_provenance(transcript, "f c", nli, epsilon=0.5)
        assert r_tight.span_n_tokens >= r_loose.span_n_tokens

    @pytest.mark.parametrize("bad_eps", [-0.01])
    def test_invalid_epsilon_rejected(self, bad_eps):
        with pytest.raises(ValueError):
            causal_provenance("a b c d e", "x", make_keyword_scorer(["a"]),
                              epsilon=bad_eps)

    def test_invalid_threshold_rejected(self):
        with pytest.raises(ValueError):
            causal_provenance("a b c d e", "x", make_keyword_scorer(["a"]),
                              threshold=1.5)

    def test_invalid_iters_rejected(self):
        with pytest.raises(ValueError):
            causal_provenance("a b c d e", "x", make_keyword_scorer(["a"]),
                              max_iters=0)


# ── Replayability - the hard guarantee ────────────────────────────────


class TestReplayability:
    def test_zero_variance_across_runs(self):
        transcript = ("doctor good morning what brings you in today "
                      "patient i have had this terrible cough for two weeks "
                      "and lots of fever above thirty eight degrees")
        nli = make_keyword_scorer(["cough", "fever"])
        result = replayability_test(transcript, "cough and fever", nli,
                                    n_runs=5)
        assert result["n_runs"] == 5
        assert result["start_std"] == 0.0
        assert result["end_std"] == 0.0
        assert result["exact_replayable"] is True

    def test_n_runs_validation(self):
        with pytest.raises(ValueError):
            replayability_test("a b c", "x", make_keyword_scorer(["a"]),
                               n_runs=1)


# ── Dice baseline ────────────────────────────────────────────────────


class TestDiceBaseline:
    def test_finds_overlap_window(self):
        transcript = ("a b c d e f fever cough back pain g h i j k l m n o p")
        r = dice_baseline_span(transcript, "patient with fever and cough",
                               window_tokens=8)
        assert r.span_n_tokens == 8
        assert "fever" in r.span_text.lower()
        assert "cough" in r.span_text.lower()
        assert r.span_score > 0.0

    def test_no_overlap_returns_zero_score(self):
        transcript = "alpha beta gamma delta epsilon"
        r = dice_baseline_span(transcript, "kappa lambda mu nu",
                               window_tokens=3)
        assert r.span_score == 0.0

    def test_empty_inputs(self):
        r = dice_baseline_span("", "anything")
        assert r.span_n_tokens == 0


# ── Bisection vs Dice - comparison the dissertation cites ────────────


class TestBisectionVsDice:
    def test_bisection_finds_tighter_span_on_paraphrase(self):
        # Paraphrased sentence: keywords don't match literally but
        # entailment is preserved by a synthetic scorer.
        transcript = ("doctor good morning patient describes sharp pain "
                      "in lower back after lifting boxes at work three weeks ago")
        # Synthetic scorer that rewards "lower back" mentions specifically
        nli = make_keyword_scorer(["lower", "back"])
        bisection = causal_provenance(transcript,
                                      "patient has lumbar disc pain",
                                      nli)
        dice = dice_baseline_span(transcript, "patient has lumbar disc pain",
                                  window_tokens=10)
        # Bisection should locate the "lower back" region; Dice will
        # underperform because there's no literal overlap with "lumbar disc".
        assert bisection.span_score > dice.span_score
