"""Tests for src/soap_generation/templates.py - Clinical note templates."""

from __future__ import annotations

import pytest

from src.soap_generation.templates import (
    COMMON_SNOMED_CODES,
    TEMPLATE_CONFIGS,
    TEMPLATE_DESCRIPTIONS,
    TEMPLATE_DISPLAY_NAMES,
    NoteTemplate,
    TemplateConfig,
)


class TestNoteTemplateEnum:
    """NoteTemplate enum values and type."""

    def test_has_six_members(self):
        assert len(NoteTemplate) == 6

    def test_is_str_subclass(self):
        assert issubclass(NoteTemplate, str)

    def test_member_values(self):
        expected = {
            "SOAP": "soap",
            "SBAR": "sbar",
            "CLINICAL_SUMMARY": "clinical_summary",
            "REFERRAL_LETTER": "referral_letter",
            "DISCHARGE_SUMMARY": "discharge_summary",
            "EMIS_CONSULTATION": "emis_consultation",
        }
        for name, value in expected.items():
            assert NoteTemplate[name].value == value


class TestTemplateDisplayNames:
    """TEMPLATE_DISPLAY_NAMES mapping coverage."""

    def test_has_entry_for_every_template(self):
        for template in NoteTemplate:
            assert template.value in TEMPLATE_DISPLAY_NAMES

    def test_has_six_entries(self):
        assert len(TEMPLATE_DISPLAY_NAMES) == 6

    def test_values_are_non_empty_strings(self):
        for name in TEMPLATE_DISPLAY_NAMES.values():
            assert isinstance(name, str) and len(name) > 0


class TestTemplateDescriptions:
    """TEMPLATE_DESCRIPTIONS mapping coverage."""

    def test_has_entry_for_every_template(self):
        for template in NoteTemplate:
            assert template.value in TEMPLATE_DESCRIPTIONS

    def test_has_six_entries(self):
        assert len(TEMPLATE_DESCRIPTIONS) == 6

    def test_values_are_non_empty_strings(self):
        for desc in TEMPLATE_DESCRIPTIONS.values():
            assert isinstance(desc, str) and len(desc) > 0


class TestCommonSnomedCodes:
    """COMMON_SNOMED_CODES constant."""

    def test_is_non_empty_string(self):
        assert isinstance(COMMON_SNOMED_CODES, str)
        assert len(COMMON_SNOMED_CODES) > 0

    def test_contains_snomed_codes(self):
        assert "SNOMED" in COMMON_SNOMED_CODES
        assert any(char.isdigit() for char in COMMON_SNOMED_CODES)


class TestTemplateConfigs:
    """TEMPLATE_CONFIGS mapping and TemplateConfig structure."""

    def test_has_entry_for_every_template(self):
        for template in NoteTemplate:
            assert template in TEMPLATE_CONFIGS

    def test_has_six_entries(self):
        assert len(TEMPLATE_CONFIGS) == 6

    @pytest.mark.parametrize("template", list(NoteTemplate))
    def test_config_is_template_config(self, template):
        assert isinstance(TEMPLATE_CONFIGS[template], TemplateConfig)

    @pytest.mark.parametrize("template", list(NoteTemplate))
    def test_system_prompt_non_empty(self, template):
        cfg = TEMPLATE_CONFIGS[template]
        assert isinstance(cfg.system_prompt, str) and len(cfg.system_prompt) > 0

    @pytest.mark.parametrize("template", list(NoteTemplate))
    def test_instruction_non_empty(self, template):
        cfg = TEMPLATE_CONFIGS[template]
        assert isinstance(cfg.instruction, str) and len(cfg.instruction) > 0

    @pytest.mark.parametrize("template", list(NoteTemplate))
    def test_example_output_non_empty(self, template):
        cfg = TEMPLATE_CONFIGS[template]
        assert isinstance(cfg.example_output, str) and len(cfg.example_output) > 0

    @pytest.mark.parametrize("template", list(NoteTemplate))
    def test_system_prompt_contains_snomed(self, template):
        cfg = TEMPLATE_CONFIGS[template]
        assert "SNOMED" in cfg.system_prompt
