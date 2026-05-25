"""Tests for src/clinical/session_store.py - File-based session persistence."""

from __future__ import annotations

import json

import pytest

from src.clinical.ontology import ClinicalEntity, ClinicalSession, EntityType, SOAPNote
from src.clinical.session_store import SessionStore


@pytest.fixture
def store(tmp_path):
    """SessionStore using a temporary directory."""
    return SessionStore(store_dir=tmp_path / "sessions")


@pytest.fixture
def sample_session():
    """A populated ClinicalSession for testing."""
    s = ClinicalSession()
    s.raw_transcript = "Doctor: How can I help? Patient: I have a cough."
    s.stt_model = "whisper"
    s.audio_hash = "abc123"
    s.total_time_secs = 5.0
    s.add_entity(ClinicalEntity(text="cough", entity_type=EntityType.SYMPTOM))
    s.set_soap_note(SOAPNote(subjective="Cough for 1 week", model_used="med42"))
    return s


class TestSessionStoreSave:
    """Saving sessions to disk."""

    def test_save_creates_file(self, store, sample_session):
        path = store.save(sample_session)
        assert path.exists()
        assert path.suffix == ".json"

    def test_save_file_is_valid_json(self, store, sample_session):
        path = store.save(sample_session)
        data = json.loads(path.read_text())
        assert data["session_id"] == sample_session.session_id

    def test_save_creates_directory(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c"
        store = SessionStore(store_dir=deep)
        s = ClinicalSession()
        path = store.save(s)
        assert path.exists()


class TestSessionStoreLoad:
    """Loading sessions from disk."""

    def test_load_returns_session(self, store, sample_session):
        store.save(sample_session)
        loaded = store.load(sample_session.session_id)
        assert loaded is not None
        assert loaded.session_id == sample_session.session_id

    def test_load_preserves_fields(self, store, sample_session):
        store.save(sample_session)
        loaded = store.load(sample_session.session_id)
        assert loaded.raw_transcript == sample_session.raw_transcript
        assert loaded.stt_model == "whisper"
        assert loaded.audio_hash == "abc123"
        assert loaded.total_time_secs == 5.0

    def test_load_nonexistent_returns_none(self, store):
        assert store.load("nonexistent-id") is None

    def test_round_trip(self, store):
        s = ClinicalSession()
        s.raw_transcript = "Hello world"
        s.log_event("test")
        store.save(s)
        loaded = store.load(s.session_id)
        assert loaded.session_id == s.session_id
        assert loaded.raw_transcript == "Hello world"
        assert len(loaded.audit_log) == 1

    def test_load_preserves_entities(self, store, sample_session):
        """Entities must survive save/load round-trip (was a critical bug)."""
        store.save(sample_session)
        loaded = store.load(sample_session.session_id)
        assert len(loaded.entities) == 1
        assert loaded.entities[0].text == "cough"
        assert loaded.entities[0].entity_type == EntityType.SYMPTOM

    def test_load_preserves_soap_note(self, store, sample_session):
        """SOAP note must survive save/load round-trip (was a critical bug)."""
        store.save(sample_session)
        loaded = store.load(sample_session.session_id)
        assert loaded.soap_note is not None
        assert loaded.soap_note.subjective == "Cough for 1 week"
        assert loaded.soap_note.model_used == "med42"

    def test_load_preserves_speakers(self, store):
        s = ClinicalSession()
        s.speakers_detected = ["Doctor", "Patient"]
        store.save(s)
        loaded = store.load(s.session_id)
        assert loaded.speakers_detected == ["Doctor", "Patient"]

    def test_load_preserves_pipeline_version(self, store):
        s = ClinicalSession()
        s.pipeline_version = "2.0.0"
        store.save(s)
        loaded = store.load(s.session_id)
        assert loaded.pipeline_version == "2.0.0"


class TestSessionStoreLoadFull:
    """load_full returns raw dict."""

    def test_load_full_returns_dict(self, store, sample_session):
        store.save(sample_session)
        data = store.load_full(sample_session.session_id)
        assert isinstance(data, dict)
        assert data["session_id"] == sample_session.session_id
        assert data["soap_note"] is not None

    def test_load_full_nonexistent(self, store):
        assert store.load_full("nope") is None


class TestSessionStoreListSessions:
    """list_sessions returns summaries."""

    def test_list_empty_store(self, store):
        assert store.list_sessions() == []

    def test_list_after_saves(self, store):
        for _ in range(3):
            store.save(ClinicalSession())
        sessions = store.list_sessions()
        assert len(sessions) == 3

    def test_list_limit(self, store):
        for _ in range(5):
            store.save(ClinicalSession())
        sessions = store.list_sessions(limit=2)
        assert len(sessions) == 2

    def test_list_session_summary_keys(self, store, sample_session):
        store.save(sample_session)
        sessions = store.list_sessions()
        assert len(sessions) == 1
        summary = sessions[0]
        assert "session_id" in summary
        assert "created_at" in summary
        assert "has_soap" in summary
        assert "entity_count" in summary
        assert summary["has_soap"] is True
        assert summary["entity_count"] >= 1


class TestSessionStoreExportAll:
    """export_all writes JSONL."""

    def test_export_all(self, store, tmp_path):
        for _ in range(3):
            store.save(ClinicalSession())
        out = tmp_path / "export.jsonl"
        result_path = store.export_all(out)
        assert result_path.exists()
        lines = result_path.read_text().strip().split("\n")
        assert len(lines) == 3
        # Each line is valid JSON
        for line in lines:
            json.loads(line)

    def test_export_empty_store(self, store, tmp_path):
        out = tmp_path / "empty.jsonl"
        store.export_all(out)
        assert out.read_text() == ""
