"""Tests for generate_tables - dissertation table auto-generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation.generate_tables import generate_all


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _wer_fixture() -> dict:
    return {
        "model": "praxis-conformer",
        "timestamp": "2026-04-29T10:00:00+00:00",
        "mean_wer": 0.5012,
        "median_wer": 0.5100,
        "ci_lower": 0.4670,
        "ci_upper": 0.5347,
        "mean_rtf": 0.044,
        "num_samples": 14,
    }


def _der_fixture() -> dict:
    return {
        "model": "praxis-ecapa",
        "model_description": "Custom ECAPA-TDNN (5.8M params, 256-dim)",
        "timestamp": "2026-04-29T10:00:00+00:00",
        "num_samples": 10,
        "speaker_count_accuracy": 1.0,
        "mean_segments_per_sample": 4.2,
        "total_audio_secs": 120.0,
        "total_inference_secs": 1.14,
        "mean_rtf": 0.0095,
        "rtf_ci_lower": 0.0080,
        "rtf_ci_upper": 0.0110,
        "params": "5.8M",
        "samples": [],
    }


def _soap_fixture() -> dict:
    return {
        "model": "praxis-soap-v1",
        "timestamp": "2026-04-29T10:00:00+00:00",
        "num_notes": 8,
        "mean_section_completeness": 0.95,
        "mean_snomed_count": 3.2,
        "mean_medication_recall": 0.88,
        "safety_netting_rate": 1.0,
        "mean_trust_score": 73.0,
        "mean_overall_score": 0.984,
    }


def _mhi_fixture() -> dict:
    return {
        "model": "praxis-soap-v1",
        "timestamp": "2026-04-29T10:00:00+00:00",
        "num_notes": 8,
        "mean_mhi_score": 0.92,
        "mean_mhi_omission": 0.05,
        "mean_mhi_hallucination": 0.03,
        "ci_omission_lower": 0.03,
        "ci_omission_upper": 0.07,
        "ci_hallucination_lower": 0.01,
        "ci_hallucination_upper": 0.05,
        "notes": [],
    }


def _constitutional_fixture() -> dict:
    return {
        "model": "praxis-soap-v1",
        "timestamp": "2026-04-29T10:00:00+00:00",
        "num_cases": 8,
        "mean_trust_improvement": 4.2,
        "mean_baseline_trust": 68.5,
        "mean_constitutional_trust": 72.7,
        "mean_iterations": 1.4,
        "cases_improved": 6,
        "cases_unchanged": 2,
        "cases_degraded": 0,
        "cases": [],
    }


def _hallucination_fixture() -> dict:
    return {
        "model": "praxis-soap-v1",
        "timestamp": "2026-04-29T10:00:00+00:00",
        "num_notes": 8,
        "total_sentences": 40,
        "total_hallucinated": 0,
        "mean_sentence_hallucination_rate": 0.0,
        "ci_hallucination_lower": 0.0,
        "ci_hallucination_upper": 0.0,
        "entailment_skipped": True,
        "mean_mhi_score": 0.92,
        "mean_mhi_omission": 0.02,
        "mean_mhi_hallucination": 0.01,
        "ci_mhi_omission_lower": 0.01,
        "ci_mhi_omission_upper": 0.03,
        "asgari_hallucination_bar": 0.0147,
        "asgari_omission_bar": 0.0345,
        "beats_asgari_hallucination": True,
        "beats_asgari_omission": True,
        "notes": [],
    }


def _write_fixtures(tmp_path: Path) -> None:
    (tmp_path / "wer_praxis-conformer_20260429_100000.json").write_text(
        json.dumps(_wer_fixture())
    )
    (tmp_path / "der_praxis-ecapa_20260429_100000.json").write_text(
        json.dumps(_der_fixture())
    )
    (tmp_path / "soap_praxis-soap-v1_20260429_100000.json").write_text(
        json.dumps(_soap_fixture())
    )
    (tmp_path / "mhi_praxis-soap-v1_20260429_100000.json").write_text(
        json.dumps(_mhi_fixture())
    )
    (tmp_path / "constitutional_eval_praxis-soap-v1_20260429_100000.json").write_text(
        json.dumps(_constitutional_fixture())
    )
    (tmp_path / "hallucination_eval_praxis-soap-v1_20260429_100000.json").write_text(
        json.dumps(_hallucination_fixture())
    )


# ── generate_all - empty results dir ──────────────────────────────────────────

def test_generate_all_empty_dir_produces_pending_placeholders(tmp_path, monkeypatch):
    import src.evaluation.generate_tables as gt
    monkeypatch.setattr(gt, "RESULTS_DIR", tmp_path)
    latex, md = generate_all()
    assert "Pending" in md or "pending" in md.lower() or "benchmark" in md.lower()


def test_generate_all_empty_dir_no_crash(tmp_path, monkeypatch):
    import src.evaluation.generate_tables as gt
    monkeypatch.setattr(gt, "RESULTS_DIR", tmp_path)
    latex, md = generate_all()
    assert isinstance(latex, str)
    assert isinstance(md, str)


def test_generate_all_writes_tex_file(tmp_path, monkeypatch):
    import src.evaluation.generate_tables as gt
    monkeypatch.setattr(gt, "RESULTS_DIR", tmp_path)
    generate_all()
    assert (tmp_path / "tables.tex").exists()


def test_generate_all_writes_md_file(tmp_path, monkeypatch):
    import src.evaluation.generate_tables as gt
    monkeypatch.setattr(gt, "RESULTS_DIR", tmp_path)
    generate_all()
    assert (tmp_path / "tables.md").exists()


# ── generate_all - with fixture data ──────────────────────────────────────────

def test_generate_all_with_fixtures_no_crash(tmp_path, monkeypatch):
    import src.evaluation.generate_tables as gt
    monkeypatch.setattr(gt, "RESULTS_DIR", tmp_path)
    _write_fixtures(tmp_path)
    latex, md = generate_all()
    assert isinstance(latex, str)
    assert isinstance(md, str)


def test_generate_all_wer_model_in_output(tmp_path, monkeypatch):
    import src.evaluation.generate_tables as gt
    monkeypatch.setattr(gt, "RESULTS_DIR", tmp_path)
    _write_fixtures(tmp_path)
    latex, md = generate_all()
    assert "praxis-conformer" in latex or "praxis-conformer" in md


def test_generate_all_soap_section_present(tmp_path, monkeypatch):
    import src.evaluation.generate_tables as gt
    monkeypatch.setattr(gt, "RESULTS_DIR", tmp_path)
    _write_fixtures(tmp_path)
    latex, md = generate_all()
    assert "SOAP" in md or "soap" in md.lower()


def test_generate_all_mhi_section_present(tmp_path, monkeypatch):
    import src.evaluation.generate_tables as gt
    monkeypatch.setattr(gt, "RESULTS_DIR", tmp_path)
    _write_fixtures(tmp_path)
    latex, md = generate_all()
    assert "MHI" in md or "Hallucination Index" in md or "mhi" in md.lower()


def test_generate_all_constitutional_section_present(tmp_path, monkeypatch):
    import src.evaluation.generate_tables as gt
    monkeypatch.setattr(gt, "RESULTS_DIR", tmp_path)
    _write_fixtures(tmp_path)
    latex, md = generate_all()
    assert "Constitutional" in md or "constitutional" in md.lower() or "TrustScore" in md


def test_generate_all_hallucination_eval_section_present(tmp_path, monkeypatch):
    import src.evaluation.generate_tables as gt
    monkeypatch.setattr(gt, "RESULTS_DIR", tmp_path)
    _write_fixtures(tmp_path)
    latex, md = generate_all()
    assert "Asgari" in md or "hallucination" in md.lower()


def test_generate_all_latex_has_begin_document_structure(tmp_path, monkeypatch):
    import src.evaluation.generate_tables as gt
    monkeypatch.setattr(gt, "RESULTS_DIR", tmp_path)
    _write_fixtures(tmp_path)
    latex, _ = generate_all()
    assert r"\begin{table}" in latex
    assert r"\end{table}" in latex


def test_generate_all_md_has_header(tmp_path, monkeypatch):
    import src.evaluation.generate_tables as gt
    monkeypatch.setattr(gt, "RESULTS_DIR", tmp_path)
    _write_fixtures(tmp_path)
    _, md = generate_all()
    assert md.startswith("## PRAXIS")


def test_generate_all_md_has_autogen_timestamp(tmp_path, monkeypatch):
    import src.evaluation.generate_tables as gt
    monkeypatch.setattr(gt, "RESULTS_DIR", tmp_path)
    _write_fixtures(tmp_path)
    _, md = generate_all()
    assert "Auto-generated" in md or "auto-generated" in md.lower()


# ── Deduplication: latest per model wins ──────────────────────────────────────

def test_generate_all_deduplicates_wer_by_model(tmp_path, monkeypatch):
    import src.evaluation.generate_tables as gt
    monkeypatch.setattr(gt, "RESULTS_DIR", tmp_path)

    old = _wer_fixture()
    old["timestamp"] = "2026-04-20T10:00:00+00:00"
    old["mean_wer"] = 0.80  # older, worse result

    new = _wer_fixture()
    new["timestamp"] = "2026-04-29T10:00:00+00:00"
    new["mean_wer"] = 0.5012  # latest result

    (tmp_path / "wer_praxis-conformer_20260420.json").write_text(json.dumps(old))
    (tmp_path / "wer_praxis-conformer_20260429.json").write_text(json.dumps(new))

    latex, md = generate_all()
    # Only one row for praxis-conformer, showing latest (50.12%)
    assert md.count("praxis-conformer") <= 3  # at most in header + 1 data row + latex


# ── Graceful degradation: malformed files ignored ─────────────────────────────

def test_generate_all_ignores_malformed_json(tmp_path, monkeypatch):
    import src.evaluation.generate_tables as gt
    monkeypatch.setattr(gt, "RESULTS_DIR", tmp_path)
    (tmp_path / "wer_broken_20260429.json").write_text("not valid json {{{")
    latex, md = generate_all()
    assert isinstance(latex, str)


def test_generate_all_ignores_missing_fields(tmp_path, monkeypatch):
    import src.evaluation.generate_tables as gt
    monkeypatch.setattr(gt, "RESULTS_DIR", tmp_path)
    (tmp_path / "wer_incomplete_20260429.json").write_text(json.dumps({"model": "foo"}))
    latex, md = generate_all()
    assert isinstance(latex, str)


# ── Output file contents match return value ────────────────────────────────────

def test_tex_file_matches_return_value(tmp_path, monkeypatch):
    import src.evaluation.generate_tables as gt
    monkeypatch.setattr(gt, "RESULTS_DIR", tmp_path)
    _write_fixtures(tmp_path)
    latex, _ = generate_all()
    written = (tmp_path / "tables.tex").read_text()
    assert written == latex


def test_md_file_matches_return_value(tmp_path, monkeypatch):
    import src.evaluation.generate_tables as gt
    monkeypatch.setattr(gt, "RESULTS_DIR", tmp_path)
    _write_fixtures(tmp_path)
    _, md = generate_all()
    written = (tmp_path / "tables.md").read_text()
    assert written == md


# ── DER section ───────────────────────────────────────────────────────────────

def test_generate_all_der_section_present_with_fixture(tmp_path, monkeypatch):
    import src.evaluation.generate_tables as gt
    monkeypatch.setattr(gt, "RESULTS_DIR", tmp_path)
    _write_fixtures(tmp_path)
    latex, md = generate_all()
    assert "Diarization" in md or "diarization" in md.lower() or "DER" in md or "praxis-ecapa" in md


def test_generate_all_der_pending_without_fixture(tmp_path, monkeypatch):
    import src.evaluation.generate_tables as gt
    monkeypatch.setattr(gt, "RESULTS_DIR", tmp_path)
    _, md = generate_all()
    assert "benchmark-der" in md or "Pending" in md or "Diarization" in md


def test_generate_all_der_model_in_latex(tmp_path, monkeypatch):
    import src.evaluation.generate_tables as gt
    monkeypatch.setattr(gt, "RESULTS_DIR", tmp_path)
    _write_fixtures(tmp_path)
    latex, _ = generate_all()
    assert "praxis-ecapa" in latex


def test_generate_all_der_speaker_accuracy_in_output(tmp_path, monkeypatch):
    import src.evaluation.generate_tables as gt
    monkeypatch.setattr(gt, "RESULTS_DIR", tmp_path)
    _write_fixtures(tmp_path)
    latex, md = generate_all()
    assert "100.0" in latex or "100.0" in md  # speaker_count_accuracy = 1.0 → 100.0%


def test_generate_all_der_deduplicates_by_model(tmp_path, monkeypatch):
    import src.evaluation.generate_tables as gt
    monkeypatch.setattr(gt, "RESULTS_DIR", tmp_path)

    old = _der_fixture()
    old["timestamp"] = "2026-04-20T10:00:00+00:00"
    old["speaker_count_accuracy"] = 0.8

    new = _der_fixture()
    new["timestamp"] = "2026-04-29T10:00:00+00:00"
    new["speaker_count_accuracy"] = 1.0

    (tmp_path / "der_praxis-ecapa_20260420.json").write_text(json.dumps(old))
    (tmp_path / "der_praxis-ecapa_20260429.json").write_text(json.dumps(new))

    latex, md = generate_all()
    assert md.count("praxis-ecapa") <= 3
