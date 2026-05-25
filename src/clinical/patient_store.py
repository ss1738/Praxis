"""Patient store - persistent storage for registered patients.

Each patient is stored as a JSON file. Patients can be looked up by ID,
NHS number, or name.
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from src.clinical.ontology import Patient


class PatientStore:
    """File-based patient registry. Each patient = one JSON file."""

    def __init__(self, store_dir: str | Path = "data/patients"):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def save(self, patient: Patient) -> Path:
        filepath = self.store_dir / f"{patient.patient_id}.json"
        filepath.write_text(patient.to_json())
        logger.info(f"Patient saved: {patient.full_name} ({filepath.name})")
        return filepath

    def load(self, patient_id: str) -> Patient | None:
        filepath = self.store_dir / f"{patient_id}.json"
        if not filepath.exists():
            return None
        data = json.loads(filepath.read_text())
        return Patient.from_dict(data)

    def list_patients(self) -> list[Patient]:
        patients = []
        for fp in sorted(self.store_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
            try:
                data = json.loads(fp.read_text())
                patients.append(Patient.from_dict(data))
            except Exception:
                continue
        return patients

    def find_by_nhs_number(self, nhs_number: str) -> Patient | None:
        for fp in self.store_dir.glob("*.json"):
            try:
                data = json.loads(fp.read_text())
                if data.get("nhs_number", "").strip() == nhs_number.strip():
                    return Patient.from_dict(data)
            except Exception:
                continue
        return None

    def search_by_name(self, query: str) -> list[Patient]:
        query_lower = query.lower().strip()
        results = []
        for fp in self.store_dir.glob("*.json"):
            try:
                data = json.loads(fp.read_text())
                full_name = f"{data.get('first_name', '')} {data.get('last_name', '')}".lower()
                if query_lower in full_name:
                    results.append(Patient.from_dict(data))
            except Exception:
                continue
        return results

    def delete(self, patient_id: str) -> bool:
        filepath = self.store_dir / f"{patient_id}.json"
        if filepath.exists():
            filepath.unlink()
            logger.info(f"Patient deleted: {patient_id}")
            return True
        return False
