"""Session history API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.clinical.session_store import SessionStore

router = APIRouter(prefix="/api/v1", tags=["sessions"])

_session_store = SessionStore()


@router.get("/sessions")
async def list_sessions(limit: int = 50):
    """List recent sessions."""
    return _session_store.list_sessions(limit=limit)


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get full session data."""
    data = _session_store.load_full(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="Session not found")
    return data
