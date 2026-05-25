"""Tests for src/soap_generation/generator.py - SOAP generator (no network calls)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.soap_generation.generator import (
    MODEL_DISPLAY_NAMES,
    SOAPGenerator,
    SOAPModel,
    SOAPResult,
    display_name,
)


class TestSOAPModel:
    """SOAPModel enum values."""

    def test_praxis_primary(self):
        assert SOAPModel.PRAXIS_PRIMARY.value == "praxis-soap-v1"

    def test_med42(self):
        assert SOAPModel.MED42.value == "med42"

    def test_meditron(self):
        assert SOAPModel.MEDITRON.value == "meditron:7b"

    def test_mistral(self):
        assert SOAPModel.MISTRAL.value == "mistral"

    def test_all_models_in_display_names(self):
        for model in SOAPModel:
            assert model.value in MODEL_DISPLAY_NAMES


class TestDisplayName:
    """display_name() helper function."""

    def test_known_model(self):
        assert display_name("praxis-soap-v1") == "PRAXIS-SOAP-v1 (QLoRA Fine-tuned Med42)"

    def test_med42(self):
        assert display_name("med42") == "Med42-8B (Prompted Baseline)"

    def test_unknown_model_returns_id(self):
        assert display_name("some-unknown-model") == "some-unknown-model"


class TestModelDisplayNames:
    """MODEL_DISPLAY_NAMES mapping."""

    def test_has_four_entries(self):
        assert len(MODEL_DISPLAY_NAMES) == 4

    def test_all_values_are_strings(self):
        for k, v in MODEL_DISPLAY_NAMES.items():
            assert isinstance(k, str)
            assert isinstance(v, str)


class TestSOAPResult:
    """SOAPResult dataclass."""

    def test_creation(self):
        r = SOAPResult(soap_note="S: headache", model="med42", generation_time_secs=1.5)
        assert r.soap_note == "S: headache"
        assert r.model == "med42"
        assert r.generation_time_secs == 1.5
        assert r.token_count == 0
        assert r.raw_response == {}

    def test_with_all_fields(self):
        r = SOAPResult(
            soap_note="SOAP text",
            model="praxis-soap-v1",
            generation_time_secs=2.0,
            token_count=512,
            raw_response={"message": {"content": "SOAP text"}},
        )
        assert r.token_count == 512
        assert "message" in r.raw_response


class TestSOAPGeneratorInit:
    """SOAPGenerator initialization (no network)."""

    @patch("src.soap_generation.generator.httpx.Client")
    def test_defaults(self, mock_client_cls):
        gen = SOAPGenerator()
        assert gen.model == SOAPModel.PRAXIS_PRIMARY
        assert gen.ollama_url == "http://localhost:11434"
        assert gen.timeout == 120.0

    @patch("src.soap_generation.generator.httpx.Client")
    def test_custom_model(self, mock_client_cls):
        gen = SOAPGenerator(model=SOAPModel.MED42)
        assert gen.model == SOAPModel.MED42

    @patch("src.soap_generation.generator.httpx.Client")
    def test_custom_url(self, mock_client_cls):
        gen = SOAPGenerator(ollama_url="http://myserver:11434/")
        assert gen.ollama_url == "http://myserver:11434"

    @patch("src.soap_generation.generator.httpx.Client")
    def test_custom_timeout(self, mock_client_cls):
        gen = SOAPGenerator(timeout=60.0)
        assert gen.timeout == 60.0

    @patch("src.soap_generation.generator.httpx.Client")
    def test_check_ollama_failure(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get.side_effect = ConnectionError("refused")
        mock_client_cls.return_value = mock_client
        gen = SOAPGenerator()
        assert gen.check_ollama() is False

    @patch("src.soap_generation.generator.httpx.Client")
    def test_list_models_failure(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get.side_effect = ConnectionError("refused")
        mock_client_cls.return_value = mock_client
        gen = SOAPGenerator()
        assert gen.list_models() == []


class TestSOAPGeneratorGenerate:
    """SOAPGenerator.generate() - the core generation method."""

    @patch("src.soap_generation.generator.httpx.Client")
    def test_generate_returns_soap_result(self, mock_client_cls):
        """generate() returns a SOAPResult with note text from Ollama response."""
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {"content": "SUBJECTIVE:\nCough for 2 weeks.\n\nOBJECTIVE:\nCrackles."},
            "eval_count": 42,
        }
        mock_resp.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        gen = SOAPGenerator()
        result = gen.generate("Patient has a cough.")

        assert isinstance(result, SOAPResult)
        assert "SUBJECTIVE" in result.soap_note
        assert result.model == "praxis-soap-v1"
        assert result.token_count == 42
        assert result.generation_time_secs >= 0

    @patch("src.soap_generation.generator.httpx.Client")
    def test_generate_with_template(self, mock_client_cls):
        """generate() passes template config to the model."""
        from src.soap_generation.templates import NoteTemplate

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"content": "SITUATION: Cough."},
            "eval_count": 10,
        }
        mock_resp.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        gen = SOAPGenerator()
        result = gen.generate("Patient has a cough.", template=NoteTemplate.SBAR)

        assert result.template == "sbar"
        # Verify system prompt was sent to Ollama
        call_args = mock_client.post.call_args
        payload = call_args[1]["json"] if "json" in call_args[1] else call_args[0][1]
        messages = payload["messages"]
        assert any("SBAR" in m.get("content", "") or "Situation" in m.get("content", "") for m in messages)

    @patch("src.soap_generation.generator.httpx.Client")
    def test_generate_connection_error_raises_runtime(self, mock_client_cls):
        """Connection error raises RuntimeError with helpful message."""
        import httpx as _httpx

        mock_client = MagicMock()
        mock_client.post.side_effect = _httpx.ConnectError("Connection refused")
        mock_client_cls.return_value = mock_client

        gen = SOAPGenerator()
        with pytest.raises(RuntimeError, match="Cannot connect to Ollama"):
            gen.generate("test")

    @patch("src.soap_generation.generator.httpx.Client")
    def test_generate_timeout_retries_then_raises(self, mock_client_cls):
        """Timeout triggers retry and then raises RuntimeError."""
        import httpx as _httpx

        mock_client = MagicMock()
        mock_client.post.side_effect = _httpx.ReadTimeout("timed out")
        mock_client_cls.return_value = mock_client

        gen = SOAPGenerator()
        with pytest.raises(RuntimeError, match="timed out"):
            gen.generate("test")
        # Should have retried (2 attempts)
        assert mock_client.post.call_count == 2

    @patch("src.soap_generation.generator.httpx.Client")
    def test_generate_http_error_raises_runtime(self, mock_client_cls):
        """HTTP 404/500 raises RuntimeError with status code."""
        import httpx as _httpx

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.text = "model not found"
        mock_resp.raise_for_status.side_effect = _httpx.HTTPStatusError(
            "Not Found", request=MagicMock(), response=mock_resp
        )
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        gen = SOAPGenerator()
        with pytest.raises(RuntimeError, match="HTTP 404"):
            gen.generate("test")
