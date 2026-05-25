"""Clinical note generator using fine-tuned medical LLMs via Ollama.

Supports multiple NHS/GP note templates (SOAP, SBAR, consultation summary, etc.)
with structured prompts that enforce SNOMED CT coding and UK English.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import httpx
from loguru import logger

from src.soap_generation.templates import (
    NoteTemplate,
    TEMPLATE_CONFIGS,
    TEMPLATE_DISPLAY_NAMES,
    TEMPLATE_DESCRIPTIONS,
    get_template_config,
)


class SOAPModel(str, Enum):
    """Available SOAP generation models in Ollama."""
    PRAXIS_PRIMARY = "praxis-soap-v1"     # Fine-tuned Med42 - primary model
    MED42 = "med42"                       # Med42 baseline
    MEDITRON = "meditron:7b"              # Meditron baseline
    MISTRAL = "mistral"                   # Mistral zero-shot baseline


# Display names for the UI (academic, no commercial branding)
MODEL_DISPLAY_NAMES = {
    "praxis-soap-v1": "PRAXIS-SOAP-v1 (QLoRA Fine-tuned Med42)",
    "med42": "Med42-8B (Prompted Baseline)",
    "meditron:7b": "Meditron3-8B (Prompted Baseline)",
    "mistral": "Mistral-7B (Zero-shot Baseline)",
}


def display_name(model_id: str) -> str:
    """Get the academic display name for a model."""
    return MODEL_DISPLAY_NAMES.get(model_id, model_id)


def template_display_name(template_id: str) -> str:
    """Get the display name for a template."""
    return TEMPLATE_DISPLAY_NAMES.get(template_id, template_id)


# Legacy constants kept for backward compatibility with existing tests
SYSTEM_PROMPT = TEMPLATE_CONFIGS[NoteTemplate.SOAP].system_prompt
INSTRUCTION = TEMPLATE_CONFIGS[NoteTemplate.SOAP].instruction


@dataclass
class SOAPResult:
    """Result of clinical note generation."""
    soap_note: str
    model: str
    generation_time_secs: float
    token_count: int = 0
    template: str = "soap"
    raw_response: dict = field(default_factory=dict)


class SOAPGenerator:
    """Generate clinical notes from transcripts using Ollama models."""

    def __init__(
        self,
        model: SOAPModel = SOAPModel.PRAXIS_PRIMARY,
        template: NoteTemplate = NoteTemplate.SOAP,
        ollama_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ):
        self.model = model
        self.template = template
        self.ollama_url = ollama_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def check_ollama(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            resp = self._client.get(f"{self.ollama_url}/api/tags")
            if resp.status_code != 200:
                return False
            models = [m["name"] for m in resp.json().get("models", [])]
            return any(
                self.model.value in m or m.startswith(self.model.value)
                for m in models
            )
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """List available Ollama models."""
        try:
            resp = self._client.get(f"{self.ollama_url}/api/tags")
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []

    def generate(self, transcript: str, template: NoteTemplate | None = None) -> SOAPResult:
        """Generate a clinical note from a consultation transcript.

        Args:
            transcript: The consultation transcript text.
            template: Override the default template for this generation.
        """
        active_template = template or self.template
        config = get_template_config(active_template)

        t0 = time.perf_counter()

        # Build the user message with few-shot example
        user_content = (
            f"{config.instruction}\n\n"
            f"Here is an example of the expected output format:\n\n"
            f"{config.example_output}\n\n"
            f"---\n\n"
            f"Now generate the note for this consultation transcript:\n\n"
            f"{transcript}"
        )

        payload = {
            "model": self.model.value,
            "messages": [
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9,
                "num_predict": 1500,
            },
        }

        logger.info(f"Generating {active_template.value} note with {self.model.value}...")

        max_retries = 2
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = self._client.post(
                    f"{self.ollama_url}/api/chat",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                break
            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(
                    f"Timeout on attempt {attempt}/{max_retries} "
                    f"(model={self.model.value}): {e}"
                )
                if attempt == max_retries:
                    raise RuntimeError(
                        f"SOAP generation timed out after {max_retries} attempts "
                        f"(model={self.model.value}, timeout={self.timeout}s)"
                    ) from last_error
            except httpx.ConnectError as e:
                raise RuntimeError(
                    f"Cannot connect to Ollama at {self.ollama_url}. "
                    f"Is the server running? Error: {e}"
                ) from e
            except httpx.HTTPStatusError as e:
                raise RuntimeError(
                    f"Ollama returned HTTP {e.response.status_code} for model "
                    f"{self.model.value}: {e.response.text[:200]}"
                ) from e
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Unexpected error on attempt {attempt}/{max_retries} "
                    f"(model={self.model.value}): {e}"
                )
                if attempt == max_retries:
                    raise RuntimeError(
                        f"SOAP generation failed after {max_retries} attempts "
                        f"(model={self.model.value}): {e}"
                    ) from last_error

        generation_time = time.perf_counter() - t0
        note_text = data.get("message", {}).get("content", "")
        token_count = data.get("eval_count", 0)

        logger.info(
            f"{active_template.value} note generated in {generation_time:.1f}s "
            f"({token_count} tokens, model={self.model.value})"
        )

        return SOAPResult(
            soap_note=note_text,
            model=self.model.value,
            generation_time_secs=round(generation_time, 2),
            token_count=token_count,
            template=active_template.value,
            raw_response=data,
        )

    def generate_comparison(self, transcript: str, template: NoteTemplate | None = None) -> list[SOAPResult]:
        """Generate notes from all available models for comparison."""
        results = []
        available = self.list_models()

        for model in SOAPModel:
            if any(model.value in m for m in available):
                try:
                    old_model = self.model
                    self.model = model
                    result = self.generate(transcript, template=template)
                    self.model = old_model
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to generate with {model.value}: {e}")

        return results
