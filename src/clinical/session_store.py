"""Session store - persistent storage for clinical sessions.

Every session is stored. Every decision is auditable.
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from src.clinical.ontology import (
    ClinicalEntity, ClinicalSession, Confidence, EntityType,
    Provenance, SOAPNote, Speaker,
)


class SessionStore:
    """File-based session store. Each session = one JSON file."""

    def __init__(self, store_dir: str | Path = "data/sessions"):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def save(self, session: ClinicalSession) -> Path:
        filepath = self.store_dir / f"{session.session_id}.json"
        filepath.write_text(session.to_json())
        logger.info(f"Session saved: {filepath.name}")
        return filepath

    @staticmethod
    def _load_entity(d: dict) -> ClinicalEntity:
        """Reconstruct a ClinicalEntity from its dict representation."""
        speaker_map = {s.value: s for s in Speaker}
        etype_map = {e.value: e for e in EntityType}
        conf_map = {c.value: c for c in Confidence}
        return ClinicalEntity(
            id=d.get("id", ""),
            text=d.get("text", ""),
            entity_type=etype_map.get(d.get("type", "symptom"), EntityType.SYMPTOM),
            confidence=conf_map.get(d.get("confidence", "medium"), Confidence.MEDIUM),
            snomed_code=d.get("snomed_code", ""),
            snomed_term=d.get("snomed_term", ""),
            negated=d.get("negated", False),
            duration=d.get("duration", ""),
            severity=d.get("severity", ""),
            body_site=d.get("body_site", ""),
            provenance=Provenance(
                source_text=d.get("source_text", ""),
                speaker=speaker_map.get(d.get("speaker", "unknown"), Speaker.UNKNOWN),
                model_used=d.get("model", ""),
                extraction_method=d.get("extraction_method", ""),
            ),
        )

    @staticmethod
    def _load_soap_note(d: dict) -> SOAPNote:
        """Reconstruct a SOAPNote from its dict representation."""
        return SOAPNote(
            subjective=d.get("subjective", ""),
            objective=d.get("objective", ""),
            assessment=d.get("assessment", ""),
            plan=d.get("plan", ""),
            raw_text=d.get("raw_text", ""),
            model_used=d.get("model_used", ""),
            generation_time_secs=d.get("generation_time_secs", 0.0),
            token_count=d.get("token_count", 0),
            has_safety_netting=d.get("has_safety_netting", False),
            has_snomed_codes=d.get("has_snomed_codes", False),
            snomed_codes=d.get("snomed_codes", []),
            reviewed=d.get("reviewed", False),
            reviewed_by=d.get("reviewed_by", ""),
            reviewed_at=d.get("reviewed_at", ""),
            edits_made=d.get("edits_made", []),
            accepted=d.get("accepted"),
        )

    def load(self, session_id: str) -> ClinicalSession | None:
        filepath = self.store_dir / f"{session_id}.json"
        if not filepath.exists():
            return None
        data = json.loads(filepath.read_text())
        session = ClinicalSession()
        session.session_id = data["session_id"]
        session.created_at = data["created_at"]
        session.raw_transcript = data.get("raw_transcript", "")
        session.diarised_transcript = data.get("diarised_transcript", "")
        session.speakers_detected = data.get("speakers_detected", [])
        session.audit_log = data.get("audit_log", [])
        session.total_time_secs = data.get("total_time_secs", 0.0)
        session.stt_model = data.get("stt_model", "")
        session.stt_time_secs = data.get("stt_time_secs", 0.0)
        session.audio_hash = data.get("audio_hash", "")
        session.audio_duration_secs = data.get("audio_duration_secs", 0.0)
        session.patient_id = data.get("patient_id", "")
        session.patient_name = data.get("patient_name", "")
        session.pipeline_version = data.get("pipeline_version", "1.0.0")

        # Restore entities
        for e_data in data.get("entities", []):
            session.entities.append(self._load_entity(e_data))

        # Restore SOAP note
        soap_data = data.get("soap_note")
        if soap_data:
            session.soap_note = self._load_soap_note(soap_data)

        # Restore alternative SOAP notes
        for alt_data in data.get("alternative_soap_notes", []):
            session.alternative_soap_notes.append(self._load_soap_note(alt_data))

        return session

    def list_sessions(self, limit: int = 50) -> list[dict]:
        sessions = []
        files = sorted(self.store_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
        for f in files[:limit]:
            try:
                data = json.loads(f.read_text())
                sessions.append({
                    "session_id": data["session_id"],
                    "created_at": data["created_at"],
                    "patient_id": data.get("patient_id", ""),
                    "patient_name": data.get("patient_name", ""),
                    "has_soap": data.get("soap_note") is not None,
                    "entity_count": len(data.get("entities", [])),
                    "reviewed": data.get("soap_note", {}).get("reviewed", False) if data.get("soap_note") else False,
                    "transcript_preview": data.get("raw_transcript", "")[:100],
                    "total_time_secs": data.get("total_time_secs", 0.0),
                })
            except Exception:
                continue
        return sessions

    def get_patient_sessions(self, patient_id: str) -> list[dict]:
        """Get all sessions for a specific patient, newest first."""
        all_sessions = self.list_sessions(limit=500)
        return [s for s in all_sessions if s.get("patient_id") == patient_id]

    def load_full(self, session_id: str) -> dict | None:
        """Load the raw JSON dict for display purposes."""
        filepath = self.store_dir / f"{session_id}.json"
        if not filepath.exists():
            return None
        return json.loads(filepath.read_text())

    def export_all(self, output_path: str | Path) -> Path:
        output_path = Path(output_path)
        count = 0
        with open(output_path, "w") as f:
            for fp in sorted(self.store_dir.glob("*.json")):
                try:
                    data = json.loads(fp.read_text())
                    f.write(json.dumps(data) + "\n")
                    count += 1
                except Exception:
                    continue
        logger.info(f"Exported {count} sessions to {output_path}")
        return output_path
