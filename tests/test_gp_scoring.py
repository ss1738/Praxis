"""Tests for src/clinical/gp_scoring.py - GP blind-scoring system."""

from __future__ import annotations

import json

import pytest

from src.clinical.gp_scoring import (
    GPScoringSession,
    ScoringCriteria,
    _interpret_kappa,
    _manual_cohen_kappa,
)


class TestScoringCriteriaDefaults:
    """ScoringCriteria default values and validation."""

    def test_defaults_all_zero(self):
        sc = ScoringCriteria()
        assert sc.clinical_accuracy == 0
        assert sc.completeness == 0
        assert sc.structure_quality == 0
        assert sc.language_appropriateness == 0
        assert sc.safety_netting == 0
        assert sc.coding_quality == 0
        assert sc.overall_acceptability == 0

    def test_defaults_metadata(self):
        sc = ScoringCriteria()
        assert sc.comments == ""
        assert sc.would_edit is False
        assert sc.critical_errors == []

    def test_valid_values_accepted(self):
        sc = ScoringCriteria(
            clinical_accuracy=1, completeness=2, structure_quality=3,
            language_appropriateness=4, safety_netting=5,
            coding_quality=3, overall_acceptability=4,
        )
        assert sc.clinical_accuracy == 1
        assert sc.safety_netting == 5

    def test_value_too_high_raises(self):
        with pytest.raises(ValueError):
            ScoringCriteria(clinical_accuracy=6)

    def test_negative_value_raises(self):
        with pytest.raises(ValueError):
            ScoringCriteria(completeness=-1)

    def test_zero_is_allowed(self):
        sc = ScoringCriteria(clinical_accuracy=0)
        assert sc.clinical_accuracy == 0


class TestScoringCriteriaToDict:
    """ScoringCriteria.to_dict() output."""

    def test_returns_correct_keys(self):
        sc = ScoringCriteria()
        d = sc.to_dict()
        expected_keys = {
            "clinical_accuracy", "completeness", "structure_quality",
            "language_appropriateness", "safety_netting", "coding_quality",
            "overall_acceptability", "comments", "would_edit", "critical_errors",
        }
        assert set(d.keys()) == expected_keys

    def test_values_match(self):
        sc = ScoringCriteria(clinical_accuracy=4, comments="Good note")
        d = sc.to_dict()
        assert d["clinical_accuracy"] == 4
        assert d["comments"] == "Good note"


class TestScoringCriteriaLikertFields:
    """ScoringCriteria.LIKERT_FIELDS metadata."""

    def test_has_seven_entries(self):
        sc = ScoringCriteria()
        assert len(sc.LIKERT_FIELDS) == 7

    def test_contains_expected_fields(self):
        sc = ScoringCriteria()
        expected = [
            "clinical_accuracy", "completeness", "structure_quality",
            "language_appropriateness", "safety_netting", "coding_quality",
            "overall_acceptability",
        ]
        assert sc.LIKERT_FIELDS == expected


class TestGPScoringSession:
    """GPScoringSession lifecycle tests."""

    def test_creation_with_gp_id(self, tmp_path):
        session = GPScoringSession(gp_id="GP001", output_dir=tmp_path)
        assert session.gp_id == "GP001"
        assert session.scored_notes == []

    def test_score_note_records_entry(self, tmp_path):
        session = GPScoringSession(gp_id="GP001", output_dir=tmp_path)
        scores = ScoringCriteria(clinical_accuracy=4, completeness=3, overall_acceptability=4)
        record = session.score_note(
            session_id="S001", soap_text="SUBJECTIVE: Cough.",
            model_id="med42", transcript="Patient has a cough.", scores=scores,
        )
        assert len(session.scored_notes) == 1
        assert record["session_id"] == "S001"
        assert record["model_id"] == "med42"
        assert record["gp_id"] == "GP001"
        assert record["scores"]["clinical_accuracy"] == 4

    def test_compute_averages_with_notes(self, tmp_path):
        session = GPScoringSession(gp_id="GP002", output_dir=tmp_path)
        scores_a = ScoringCriteria(
            clinical_accuracy=4, completeness=2, structure_quality=4,
            language_appropriateness=3, safety_netting=5, coding_quality=3,
            overall_acceptability=4,
        )
        scores_b = ScoringCriteria(
            clinical_accuracy=2, completeness=4, structure_quality=2,
            language_appropriateness=3, safety_netting=3, coding_quality=5,
            overall_acceptability=2,
        )
        session.score_note("S001", "note1", "m1", "t1", scores_a)
        session.score_note("S002", "note2", "m2", "t2", scores_b)

        avgs = session.compute_averages()
        assert avgs["clinical_accuracy"] == 3.0
        assert avgs["completeness"] == 3.0
        assert avgs["notes_scored"] == 2

    def test_compute_averages_empty_returns_empty(self, tmp_path):
        session = GPScoringSession(gp_id="GP003", output_dir=tmp_path)
        assert session.compute_averages() == {}

    def test_export_scores_creates_file(self, tmp_path):
        session = GPScoringSession(gp_id="GP004", output_dir=tmp_path)
        scores = ScoringCriteria(clinical_accuracy=3, overall_acceptability=3)
        session.score_note("S001", "note", "m1", "transcript", scores)

        filepath = session.export_scores()
        assert filepath.exists()
        assert filepath.name == "gp_GP004_scores.json"

        data = json.loads(filepath.read_text())
        assert data["gp_id"] == "GP004"
        assert data["notes_scored"] == 1
        assert len(data["scores"]) == 1


class TestManualCohenKappa:
    """_manual_cohen_kappa() statistical computation."""

    def test_perfect_agreement_returns_one(self):
        ratings = [1, 2, 3, 4, 5, 3, 2, 4]
        assert _manual_cohen_kappa(ratings, ratings) == 1.0

    def test_random_disagreement_below_one(self):
        r1 = [1, 2, 3, 4, 5]
        r2 = [5, 4, 3, 2, 1]
        assert _manual_cohen_kappa(r1, r2) < 1.0

    def test_empty_ratings_returns_zero(self):
        assert _manual_cohen_kappa([], []) == 0.0

    def test_linear_weights(self):
        r1 = [1, 2, 3, 4, 5]
        assert _manual_cohen_kappa(r1, r1, weights="linear") == 1.0


class TestInterpretKappa:
    """_interpret_kappa() label assignment (Landis & Koch 1977)."""

    def test_poor(self):
        assert _interpret_kappa(-0.1) == "poor"

    def test_slight(self):
        assert _interpret_kappa(0.10) == "slight"

    def test_fair(self):
        assert _interpret_kappa(0.30) == "fair"

    def test_moderate(self):
        assert _interpret_kappa(0.50) == "moderate"

    def test_substantial(self):
        assert _interpret_kappa(0.70) == "substantial"

    def test_almost_perfect(self):
        assert _interpret_kappa(0.90) == "almost_perfect"

    def test_perfect_one(self):
        assert _interpret_kappa(1.0) == "almost_perfect"
