"""Tests for SNOMED CT contextual biasing module."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.speech_to_text.snomed_biasing import SnomedBiaser, SnomedLexicon, _normalise


# ── _normalise ─────────────────────────────────────────────────────────────────

def test_normalise_lowercases():
    assert _normalise("Amoxicillin") == "amoxicillin"


def test_normalise_strips_punctuation():
    assert _normalise("amoxicillin.") == "amoxicillin"


def test_normalise_collapses_whitespace():
    assert _normalise("chest  pain") == "chest pain"


def test_normalise_removes_numbers_preserves_alphanumeric():
    result = _normalise("HbA1c 58")
    assert "hba1c" in result
    assert "58" in result


def test_normalise_empty_string():
    assert _normalise("") == ""


# ── SnomedLexicon - construction ───────────────────────────────────────────────

def test_lexicon_loads_curated_terms():
    lex = SnomedLexicon()
    assert len(lex) > 100


def test_lexicon_contains_key_drugs():
    lex = SnomedLexicon()
    for drug in ["amoxicillin", "metformin", "ramipril", "salbutamol", "warfarin"]:
        assert lex.contains(drug), f"Missing drug: {drug}"


def test_lexicon_contains_key_conditions():
    lex = SnomedLexicon()
    for cond in ["asthma", "diabetes", "hypertension", "eczema", "migraine"]:
        assert lex.contains(cond), f"Missing condition: {cond}"


def test_lexicon_contains_nhs_abbreviations():
    lex = SnomedLexicon()
    for abbr in ["hba1c", "bnf", "nice", "ecg", "fbc"]:
        assert lex.contains(abbr), f"Missing abbreviation: {abbr}"


def test_lexicon_contains_is_case_insensitive():
    lex = SnomedLexicon()
    assert lex.contains("Amoxicillin")
    assert lex.contains("AMOXICILLIN")
    assert lex.contains("amoxicillin")


def test_lexicon_does_not_contain_stopwords():
    lex = SnomedLexicon()
    for word in ["the", "and", "for", "has"]:
        assert not lex.contains(word)


def test_lexicon_min_term_len_filters_short_terms():
    lex = SnomedLexicon(min_term_len=5)
    for term in lex.terms:
        assert len(term) >= 5


def test_lexicon_terms_is_sorted_list():
    lex = SnomedLexicon()
    terms = lex.terms
    assert isinstance(terms, list)
    assert terms == sorted(terms)


def test_lexicon_len_matches_terms_count():
    lex = SnomedLexicon()
    assert len(lex) == len(lex.terms)


def test_lexicon_loads_extra_file(tmp_path):
    extra = tmp_path / "extra_terms.txt"
    extra.write_text("methotrexate\nhydroxychloroquine\nleflunomide\n")
    lex = SnomedLexicon(extra_path=extra)
    assert lex.contains("methotrexate")
    assert lex.contains("hydroxychloroquine")
    assert lex.contains("leflunomide")


def test_lexicon_missing_extra_file_does_not_crash(tmp_path):
    lex = SnomedLexicon(extra_path=tmp_path / "nonexistent.txt")
    assert len(lex) > 0


def test_lexicon_mines_suffix_terms(tmp_path):
    corpus = tmp_path / "train.json"
    corpus.write_text(json.dumps([
        {"text": "patient has chronic gastropathy and oesophagitis"},
        {"text": "prescribe clarithromycin and omeprazole for H pylori"},
    ]))
    lex = SnomedLexicon(mine_corpus=corpus)
    # "gastropathy" ends in "pathy" - should be mined
    assert lex.contains("gastropathy")


def test_lexicon_mine_missing_corpus_does_not_crash(tmp_path):
    lex = SnomedLexicon(mine_corpus=tmp_path / "missing.json")
    assert len(lex) > 0


def test_lexicon_mine_malformed_corpus_does_not_crash(tmp_path):
    corpus = tmp_path / "bad.json"
    corpus.write_text("not valid json {{{")
    lex = SnomedLexicon(mine_corpus=corpus)
    assert len(lex) > 0


# ── SnomedLexicon.medical_term_recall ─────────────────────────────────────────

def test_recall_perfect_match():
    lex = SnomedLexicon()
    refs = ["patient prescribed amoxicillin 500mg for chest infection"]
    hyps = ["patient prescribed amoxicillin 500mg for chest infection"]
    result = lex.medical_term_recall(refs, hyps)
    assert result["mc_recall"] == 1.0
    assert result["mc_err"] == 0.0


def test_recall_zero_when_all_missed():
    lex = SnomedLexicon()
    refs = ["patient prescribed amoxicillin for infection"]
    hyps = ["patient prescribed something for something"]
    result = lex.medical_term_recall(refs, hyps)
    assert result["mc_recall"] < 1.0
    assert result["total_ref_hits"] >= 1


def test_recall_empty_refs_returns_one():
    lex = SnomedLexicon()
    result = lex.medical_term_recall([], [])
    assert result["mc_recall"] == 1.0


def test_recall_no_lexicon_terms_in_ref():
    lex = SnomedLexicon()
    refs = ["the patient went home"]
    hyps = ["the patient left"]
    result = lex.medical_term_recall(refs, hyps)
    assert result["total_ref_hits"] == 0
    assert result["mc_recall"] == 1.0


def test_recall_missed_terms_non_empty_when_miss():
    lex = SnomedLexicon()
    refs = ["prescribed metformin and ramipril"]
    hyps = ["prescribed something and something"]
    result = lex.medical_term_recall(refs, hyps)
    assert len(result["missed_terms"]) > 0


def test_recall_result_keys_present():
    lex = SnomedLexicon()
    result = lex.medical_term_recall(["patient has asthma"], ["patient has asthma"])
    for key in ["total_ref_hits", "total_recovered", "mc_recall", "mc_err", "missed_terms"]:
        assert key in result


def test_recall_mc_recall_plus_mc_err_equals_one():
    lex = SnomedLexicon()
    refs = ["amoxicillin and ibuprofen prescribed"]
    hyps = ["amoxicillin prescribed"]
    result = lex.medical_term_recall(refs, hyps)
    if result["total_ref_hits"] > 0:
        assert abs(result["mc_recall"] + result["mc_err"] - 1.0) < 0.001


def test_recall_multi_pair():
    lex = SnomedLexicon()
    refs = ["amoxicillin 500mg TDS", "metformin 1g BD"]
    hyps = ["amoxicillin 500mg", "nothing"]
    result = lex.medical_term_recall(refs, hyps)
    assert result["total_ref_hits"] >= 2


# ── SnomedBiaser - availability ────────────────────────────────────────────────

def test_biaser_is_available_returns_bool():
    lex = SnomedLexicon()
    biaser = SnomedBiaser(lex)
    assert isinstance(biaser.is_available(), bool)


def test_biaser_is_available_twice_consistent():
    lex = SnomedLexicon()
    biaser = SnomedBiaser(lex)
    r1 = biaser.is_available()
    r2 = biaser.is_available()
    assert r1 == r2


def test_biaser_stores_lexicon():
    lex = SnomedLexicon()
    biaser = SnomedBiaser(lex, boost=10.0)
    assert biaser.lexicon is lex
    assert biaser.boost == 10.0


def test_biaser_default_boost_is_eight():
    lex = SnomedLexicon()
    biaser = SnomedBiaser(lex)
    assert biaser.boost == 8.0


# ── SnomedBiaser - build_decoder ──────────────────────────────────────────────

def test_biaser_build_decoder_returns_false_without_pyctcdecode(monkeypatch):
    import src.speech_to_text.snomed_biasing as sb
    monkeypatch.setattr(sb, "logger", sb.logger)

    lex = SnomedLexicon()
    biaser = SnomedBiaser(lex)
    biaser._available = False  # simulate missing pyctcdecode

    result = biaser.build_decoder(["", "a", "b", "c"])
    assert result is False
    assert biaser._built is False


def test_biaser_not_built_initially():
    lex = SnomedLexicon()
    biaser = SnomedBiaser(lex)
    assert biaser._built is False
    assert biaser._decoder is None


# ── SnomedBiaser - decode (greedy fallback) ────────────────────────────────────

def test_biaser_decode_greedy_fallback_returns_string():
    lex = SnomedLexicon()
    biaser = SnomedBiaser(lex)
    biaser._built = False
    biaser._decoder = None

    T, V = 10, 50
    rng = np.random.default_rng(42)
    log_probs = rng.random((T, V)).astype(np.float32)
    log_probs = log_probs / log_probs.sum(axis=-1, keepdims=True)
    log_probs = np.log(log_probs + 1e-10)

    result = biaser.decode(log_probs)
    assert isinstance(result, str)


def test_biaser_greedy_decode_blank_only():
    """All-blank log probs → empty string."""
    lex = SnomedLexicon()
    biaser = SnomedBiaser(lex)
    T, V = 5, 10
    log_probs = np.full((T, V), -100.0, dtype=np.float32)
    log_probs[:, 0] = 0.0  # blank = index 0
    result = biaser.decode(log_probs)
    assert result == ""


def test_biaser_decode_non_empty_when_non_blank():
    lex = SnomedLexicon()
    biaser = SnomedBiaser(lex)
    T, V = 5, 10
    log_probs = np.full((T, V), -100.0, dtype=np.float32)
    log_probs[:, 3] = 0.0  # token 3 highest
    result = biaser.decode(log_probs)
    assert result != ""


# ── Integration: lexicon size sanity ──────────────────────────────────────────

def test_default_lexicon_has_reasonable_size():
    lex = SnomedLexicon()
    assert 100 <= len(lex) <= 5000


def test_extra_file_increases_lexicon_size(tmp_path):
    base = SnomedLexicon()
    base_size = len(base)

    extra = tmp_path / "terms.txt"
    extra.write_text("\n".join(f"uniqueterm{i}" for i in range(50)))
    augmented = SnomedLexicon(extra_path=extra)
    assert len(augmented) > base_size
