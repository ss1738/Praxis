"""Tests for the UK GP medical lexicon used at ASR decode time."""

from __future__ import annotations

from src.clinical.medical_lexicon import (
    build_lexicon,
    in_lexicon_recall,
    load_curated,
)


def test_curated_contains_key_uk_gp_terms():
    terms = load_curated()
    for must_have in ["amoxicillin", "ramipril", "metformin", "salbutamol", "snomed", "qof"]:
        assert must_have in terms, f"{must_have} missing from curated lexicon"


def test_curated_only_lowercase():
    terms = load_curated()
    assert all(t == t.lower() for t in terms)


def test_curated_minimum_length_three():
    terms = load_curated()
    assert all(len(t) >= 3 for t in terms)


def test_build_lexicon_at_least_curated_size():
    full = build_lexicon(mine_train=False)
    curated = load_curated()
    assert len(full) >= len(curated)


def test_build_lexicon_returns_sorted_unique():
    full = build_lexicon(mine_train=False)
    assert full == sorted(set(full))


def test_in_lexicon_recall_perfect():
    refs = ["patient prescribed amoxicillin and paracetamol"]
    hyps = ["patient prescribed amoxicillin and paracetamol"]
    lex = {"amoxicillin", "paracetamol"}
    out = in_lexicon_recall(refs, hyps, lex)
    assert out["ref_hits"] == 2
    assert out["recovered"] == 2
    assert out["mc_recall"] == 1.0


def test_in_lexicon_recall_partial():
    refs = ["start metformin and atorvastatin"]
    hyps = ["start metformin and chocolate"]  # atorvastatin lost
    lex = {"metformin", "atorvastatin"}
    out = in_lexicon_recall(refs, hyps, lex)
    assert out["ref_hits"] == 2
    assert out["recovered"] == 1
    assert out["mc_recall"] == 0.5
    assert "atorvastatin" in out["missed_top30"]


def test_in_lexicon_recall_no_lex_terms():
    refs = ["good morning how are you today"]
    hyps = ["good morning how are you today"]
    lex = {"amoxicillin"}
    out = in_lexicon_recall(refs, hyps, lex)
    assert out["ref_hits"] == 0
    assert out["mc_recall"] == 0.0
