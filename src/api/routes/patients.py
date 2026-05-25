"""Patient management API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.clinical.ontology import Patient
from src.clinical.patient_store import PatientStore
from src.clinical.session_store import SessionStore

router = APIRouter(prefix="/api/v1", tags=["patients"])

_patient_store = PatientStore()
_session_store = SessionStore()


class PatientCreateRequest(BaseModel):
    first_name: str = Field(..., min_length=1)
    last_name: str = Field(..., min_length=1)
    nhs_number: str = Field(default="")
    date_of_birth: str = Field(default="")


class PatientResponse(BaseModel):
    patient_id: str
    first_name: str
    last_name: str
    full_name: str
    nhs_number: str
    date_of_birth: str
    registered_at: str


@router.post("/patients", response_model=PatientResponse)
async def register_patient(req: PatientCreateRequest):
    """Register a new patient."""
    if req.nhs_number:
        existing = _patient_store.find_by_nhs_number(req.nhs_number)
        if existing:
            raise HTTPException(
                status_code=409,
                detail=f"Patient with NHS number {req.nhs_number} already exists",
            )

    patient = Patient(
        first_name=req.first_name,
        last_name=req.last_name,
        nhs_number=req.nhs_number,
        date_of_birth=req.date_of_birth,
    )
    _patient_store.save(patient)
    return PatientResponse(
        patient_id=patient.patient_id,
        first_name=patient.first_name,
        last_name=patient.last_name,
        full_name=patient.full_name,
        nhs_number=patient.nhs_number,
        date_of_birth=patient.date_of_birth,
        registered_at=patient.registered_at,
    )


@router.get("/patients", response_model=list[PatientResponse])
async def list_patients():
    """List all registered patients."""
    patients = _patient_store.list_patients()
    return [
        PatientResponse(
            patient_id=p.patient_id,
            first_name=p.first_name,
            last_name=p.last_name,
            full_name=p.full_name,
            nhs_number=p.nhs_number,
            date_of_birth=p.date_of_birth,
            registered_at=p.registered_at,
        )
        for p in patients
    ]


@router.get("/patients/{patient_id}", response_model=PatientResponse)
async def get_patient(patient_id: str):
    """Get a single patient by ID."""
    patient = _patient_store.load(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return PatientResponse(
        patient_id=patient.patient_id,
        first_name=patient.first_name,
        last_name=patient.last_name,
        full_name=patient.full_name,
        nhs_number=patient.nhs_number,
        date_of_birth=patient.date_of_birth,
        registered_at=patient.registered_at,
    )


@router.get("/patients/search/{query}")
async def search_patients(query: str):
    """Search patients by name."""
    results = _patient_store.search_by_name(query)
    return [
        PatientResponse(
            patient_id=p.patient_id,
            first_name=p.first_name,
            last_name=p.last_name,
            full_name=p.full_name,
            nhs_number=p.nhs_number,
            date_of_birth=p.date_of_birth,
            registered_at=p.registered_at,
        )
        for p in results
    ]


@router.get("/patients/{patient_id}/sessions")
async def get_patient_sessions(patient_id: str):
    """Get all sessions for a patient."""
    patient = _patient_store.load(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    sessions = _session_store.get_patient_sessions(patient_id)
    return {
        "patient_id": patient_id,
        "patient_name": patient.full_name,
        "total_visits": len(sessions),
        "sessions": sessions,
    }


@router.delete("/patients/{patient_id}")
async def delete_patient(patient_id: str):
    """Delete a patient."""
    deleted = _patient_store.delete(patient_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Patient not found")
    return {"deleted": True, "patient_id": patient_id}
