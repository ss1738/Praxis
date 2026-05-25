"""API endpoint tests (mocked, no GPU)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.main import app
from src.soap_generation.generator import SOAPResult


@pytest.fixture
async def client():
    """Async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestHealthEndpoint:
    """Health endpoint."""

    @pytest.mark.asyncio
    async def test_health_returns_ok(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestTranscribeEndpoint:
    """Transcribe endpoint."""

    @pytest.mark.asyncio
    async def test_transcribe_no_file_returns_422(self, client):
        """No file = 422."""
        resp = await client.post("/api/v1/transcribe")
        assert resp.status_code == 422

    @pytest.mark.asyncio
    @patch("src.api.routes.transcription._get_engine")
    @patch("src.api.routes.transcription._preprocessor")
    async def test_transcribe_returns_text(self, mock_preprocessor, mock_get_engine, client):
        """Should return transcription text."""
        # Mock preprocessor
        mock_result = MagicMock()
        mock_result.audio = b"\x00" * 16_000
        mock_result.sample_rate = 16_000
        mock_result.duration_secs = 1.0
        mock_preprocessor.process_file.return_value = mock_result

        # Mock engine
        mock_engine = MagicMock()
        mock_engine.transcribe.return_value = "patient reports chest pain"
        mock_get_engine.return_value = mock_engine

        resp = await client.post(
            "/api/v1/transcribe",
            files={"file": ("test.wav", b"\x00" * 1000, "audio/wav")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "text" in data
        assert "engine" in data


class TestSOAPEndpoint:
    """POST /api/v1/soap - SOAP generation endpoint."""

    @pytest.mark.asyncio
    @patch("src.api.routes.soap._get_generator")
    async def test_soap_returns_note(self, mock_get_gen, client):
        """SOAP endpoint returns generated note."""
        mock_gen = MagicMock()
        mock_gen.generate.return_value = SOAPResult(
            soap_note="SUBJECTIVE:\nCough.\n\nOBJECTIVE:\nCrackles.",
            model="praxis-soap-v1",
            generation_time_secs=1.5,
            token_count=50,
            template="soap",
        )
        mock_get_gen.return_value = mock_gen

        resp = await client.post("/api/v1/soap", json={
            "transcript": "Patient has a cough for two weeks.",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "soap_note" in data
        assert "SUBJECTIVE" in data["soap_note"]
        assert data["model"] == "praxis-soap-v1"
        assert data["template"] == "soap"
        assert data["token_count"] == 50

    @pytest.mark.asyncio
    @patch("src.api.routes.soap._get_generator")
    async def test_soap_with_template(self, mock_get_gen, client):
        """SOAP endpoint accepts template parameter."""
        mock_gen = MagicMock()
        mock_gen.generate.return_value = SOAPResult(
            soap_note="SITUATION:\nCough.",
            model="med42",
            generation_time_secs=2.0,
            token_count=30,
            template="sbar",
        )
        mock_get_gen.return_value = mock_gen

        resp = await client.post("/api/v1/soap", json={
            "transcript": "Patient has a cough.",
            "template": "sbar",
        })
        assert resp.status_code == 200
        assert resp.json()["template"] == "sbar"

    @pytest.mark.asyncio
    async def test_soap_missing_transcript_returns_422(self, client):
        """Missing transcript field returns 422."""
        resp = await client.post("/api/v1/soap", json={})
        assert resp.status_code == 422


class TestSOAPTemplatesEndpoint:
    """GET /api/v1/soap/templates - template listing."""

    @pytest.mark.asyncio
    async def test_templates_returns_list(self, client):
        resp = await client.get("/api/v1/soap/templates")
        assert resp.status_code == 200
        data = resp.json()
        assert "templates" in data
        assert len(data["templates"]) == 6
        for t in data["templates"]:
            assert "id" in t
            assert "name" in t
            assert "description" in t


class TestSOAPModelsEndpoint:
    """GET /api/v1/soap/models - model listing."""

    @pytest.mark.asyncio
    @patch("src.api.routes.soap.SOAPGenerator")
    async def test_models_returns_supported(self, mock_gen_cls, client):
        mock_gen = MagicMock()
        mock_gen.list_models.return_value = ["praxis-soap-v1:latest", "med42:latest"]
        mock_gen_cls.return_value = mock_gen

        resp = await client.get("/api/v1/soap/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert "supported" in data
        assert len(data["supported"]) == 4
