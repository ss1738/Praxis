"""Tests for src/speech_to_text/speaker_diarisation.py (no network calls)."""

from __future__ import annotations

import pytest

from src.speech_to_text.speaker_diarisation import (
    DiarisationResult,
    SpeakerSegment,
    _rule_based_diarise,
    diarise_with_fallback,
)


class TestSpeakerSegment:
    """SpeakerSegment dataclass."""

    def test_creation(self):
        seg = SpeakerSegment(speaker="Doctor", start=0.0, end=5.0, text="Hello")
        assert seg.speaker == "Doctor"
        assert seg.duration == 5.0

    def test_duration_calculation(self):
        seg = SpeakerSegment(speaker="Patient", start=1.5, end=4.5)
        assert seg.duration == 3.0

    def test_default_text_empty(self):
        seg = SpeakerSegment(speaker="Doctor", start=0, end=1)
        assert seg.text == ""


class TestDiarisationResult:
    """DiarisationResult dataclass."""

    def test_empty_result(self):
        result = DiarisationResult()
        assert result.segments == []
        assert result.num_speakers == 0
        assert result.diarised_transcript == ""
        assert result.speakers == []

    def test_diarised_transcript_format(self):
        result = DiarisationResult(segments=[
            SpeakerSegment(speaker="Doctor", start=0, end=5, text="How can I help?"),
            SpeakerSegment(speaker="Patient", start=5, end=10, text="I have a cough."),
        ])
        transcript = result.diarised_transcript
        assert "[Doctor]: How can I help?" in transcript
        assert "[Patient]: I have a cough." in transcript

    def test_speakers_list(self):
        result = DiarisationResult(segments=[
            SpeakerSegment(speaker="Doctor", start=0, end=5, text="Hi"),
            SpeakerSegment(speaker="Patient", start=5, end=10, text="Hi"),
            SpeakerSegment(speaker="Doctor", start=10, end=15, text="More"),
        ])
        assert result.speakers == ["Doctor", "Patient"]

    def test_empty_text_segments_excluded_from_transcript(self):
        result = DiarisationResult(segments=[
            SpeakerSegment(speaker="Doctor", start=0, end=5, text="Hello"),
            SpeakerSegment(speaker="Patient", start=5, end=6, text=""),
            SpeakerSegment(speaker="Doctor", start=6, end=10, text="Next"),
        ])
        lines = result.diarised_transcript.split("\n")
        assert len(lines) == 2


class TestRuleBasedDiarise:
    """Rule-based fallback diarisation."""

    def test_labelled_input_preserved(self):
        transcript = "[DOCTOR]: How can I help?\n[PATIENT]: I have a headache."
        result = _rule_based_diarise(transcript)
        assert len(result.segments) == 2
        assert result.segments[0].speaker == "Doctor"
        assert result.segments[1].speaker == "Patient"

    def test_doctor_keywords_detected(self):
        transcript = "Let me examine your throat."
        result = _rule_based_diarise(transcript)
        assert result.segments[0].speaker == "Doctor"

    def test_patient_keywords_detected(self):
        transcript = "I've been having headaches for a week."
        result = _rule_based_diarise(transcript)
        assert result.segments[0].speaker == "Patient"

    def test_unknown_speaker_for_ambiguous(self):
        transcript = "The weather is nice today."
        result = _rule_based_diarise(transcript)
        assert result.segments[0].speaker == "Unknown"

    def test_empty_lines_skipped(self):
        transcript = "Hello\n\n\nI feel sick"
        result = _rule_based_diarise(transcript)
        assert len(result.segments) == 2

    def test_colon_format_preserved(self):
        transcript = "DOCTOR: Take this medicine.\nPATIENT: OK."
        result = _rule_based_diarise(transcript)
        assert result.segments[0].speaker == "Doctor"
        assert "Take this medicine" in result.segments[0].text


class TestDiariseWithFallback:
    """diarise_with_fallback() convenience function."""

    def test_text_only_uses_rule_based(self):
        transcript = "[DOCTOR]: Hello\n[PATIENT]: Hi doctor"
        result = diarise_with_fallback(audio_path=None, transcript=transcript)
        assert len(result.segments) == 2
        assert result.segments[0].speaker == "Doctor"

    def test_nonexistent_audio_falls_back(self):
        result = diarise_with_fallback(
            audio_path="/nonexistent/file.wav",
            transcript="I've been having a cough.",
        )
        # Should fall back to rule-based (Pyannote either unavailable or fails)
        assert len(result.segments) >= 1
