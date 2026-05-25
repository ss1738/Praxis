"""Source linking - map clinical note sentences back to transcript segments.

The #1 trust feature: a GP can hover over any sentence in the note and
instantly see the exact transcript text that generated it.

Uses token overlap + medical term boosting (Dice coefficient). No embeddings,
no model - runs in <50ms on any hardware.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class SourceLink:
    """A link from a note sentence to its source transcript segment(s)."""
    note_section: str
    note_sentence: str
    note_sentence_idx: int
    source_sentences: list[str]
    source_indices: list[int]
    confidence: float
    segment_ids: list[str] | None = None
    start_times: list[float] | None = None
    end_times: list[float] | None = None
    speakers: list[str] | None = None


@dataclass
class StreamingSegment:
    """A transcript segment from a streaming session."""
    segment_id: str
    text: str
    speaker: str
    start_time: float
    end_time: float
    confidence: float


@dataclass
class SourceMap:
    """Complete source mapping for a clinical note."""
    links: list[SourceLink] = field(default_factory=list)
    transcript_sentences: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "links": [
                {
                    "note_section": l.note_section,
                    "note_sentence": l.note_sentence,
                    "note_sentence_idx": l.note_sentence_idx,
                    "source_sentences": l.source_sentences,
                    "source_indices": l.source_indices,
                    "confidence": l.confidence,
                    "segment_ids": l.segment_ids,
                    "start_times": l.start_times,
                    "end_times": l.end_times,
                    "speakers": l.speakers,
                }
                for l in self.links
            ],
            "transcript_sentences": self.transcript_sentences,
        }


_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+|(?<=\n)')

_MEDICAL_TERMS = frozenset({
    "pain", "ache", "fever", "cough", "headache", "nausea", "vomiting",
    "diarrhoea", "diarrhea", "breathless", "dizzy", "fatigue", "tired",
    "swelling", "rash", "bleeding", "discharge", "tender", "stiff",
    "prescribed", "medication", "dose", "tablet", "inhaler", "injection",
    "blood pressure", "heart rate", "temperature", "oxygen", "bmi",
    "diabetes", "asthma", "copd", "hypertension", "depression", "anxiety",
    "infection", "inflammation", "fracture", "sprain", "strain",
    "examination", "auscultation", "palpation", "percussion",
    "chest", "abdomen", "throat", "ear", "eye", "skin", "joint", "back",
    "refer", "referral", "follow up", "review", "blood test", "x-ray",
    "scan", "physio", "specialist", "hospital", "a&e", "emergency",
    "antibiotic", "paracetamol", "ibuprofen", "steroid", "insulin",
    "denied", "denies", "no", "not", "without", "normal", "abnormal",
    "elevated", "reduced", "improved", "worse", "stable", "acute", "chronic",
})


def _split_sentences(text: str) -> list[str]:
    if not text:
        return []
    raw = _SENTENCE_RE.split(text)
    return [s.strip() for s in raw if len(s.strip()) > 10]


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r'\b[a-z]{2,}\b', text.lower()))


def _score_match(note_tokens: set[str], transcript_tokens: set[str]) -> float:
    if not note_tokens or not transcript_tokens:
        return 0.0

    common = note_tokens & transcript_tokens
    if not common:
        return 0.0

    base_score = 2 * len(common) / (len(note_tokens) + len(transcript_tokens))

    medical_overlap = common & _MEDICAL_TERMS
    medical_boost = min(0.3, len(medical_overlap) * 0.1)

    return min(1.0, base_score + medical_boost)


def build_source_map(
    transcript: str,
    note_sections: dict[str, str],
    min_confidence: float = 0.15,
    max_sources_per_sentence: int = 3,
) -> SourceMap:
    """Build a source map linking note sentences to transcript sentences.

    Args:
        transcript: The full consultation transcript.
        note_sections: Dict of section_name -> section_text.
        min_confidence: Minimum match score to include a link.
        max_sources_per_sentence: Max transcript sources per note sentence.
    """
    transcript_sentences = _split_sentences(transcript)
    if not transcript_sentences:
        return SourceMap()

    transcript_tokens = [_tokenize(s) for s in transcript_sentences]
    source_map = SourceMap(transcript_sentences=transcript_sentences)

    for section_name, section_text in note_sections.items():
        if not section_text or section_text == "Not documented":
            continue

        note_sentences = _split_sentences(section_text)

        for note_idx, note_sent in enumerate(note_sentences):
            note_toks = _tokenize(note_sent)

            scores: list[tuple[int, float]] = []
            for t_idx, t_toks in enumerate(transcript_tokens):
                score = _score_match(note_toks, t_toks)
                if score >= min_confidence:
                    scores.append((t_idx, score))

            scores.sort(key=lambda x: x[1], reverse=True)
            top = scores[:max_sources_per_sentence]

            if top:
                source_map.links.append(SourceLink(
                    note_section=section_name,
                    note_sentence=note_sent,
                    note_sentence_idx=note_idx,
                    source_sentences=[transcript_sentences[i] for i, _ in top],
                    source_indices=[i for i, _ in top],
                    confidence=round(top[0][1], 3),
                ))

    logger.info(
        f"Source map built: {len(transcript_sentences)} transcript sentences, "
        f"{len(source_map.links)} links, "
        f"avg confidence {sum(l.confidence for l in source_map.links) / max(1, len(source_map.links)):.3f}"
    )
    return source_map


def build_source_map_from_segments(
    segments: list[StreamingSegment],
    note_sections: dict[str, str],
    min_confidence: float = 0.15,
    max_sources_per_sentence: int = 3,
) -> SourceMap:
    """Build a source map from streaming transcript segments with timestamps."""
    if not segments:
        return SourceMap()

    segment_texts = [seg.text for seg in segments]
    segment_tokens = [_tokenize(seg.text) for seg in segments]
    source_map = SourceMap(transcript_sentences=segment_texts)

    for section_name, section_text in note_sections.items():
        if not section_text or section_text == "Not documented":
            continue

        note_sentences = _split_sentences(section_text)

        for note_idx, note_sent in enumerate(note_sentences):
            note_toks = _tokenize(note_sent)

            scores: list[tuple[int, float]] = []
            for s_idx, s_toks in enumerate(segment_tokens):
                score = _score_match(note_toks, s_toks)
                if score >= min_confidence:
                    scores.append((s_idx, score))

            scores.sort(key=lambda x: x[1], reverse=True)
            top = scores[:max_sources_per_sentence]

            if top:
                matched_indices = [i for i, _ in top]
                source_map.links.append(SourceLink(
                    note_section=section_name,
                    note_sentence=note_sent,
                    note_sentence_idx=note_idx,
                    source_sentences=[segment_texts[i] for i in matched_indices],
                    source_indices=matched_indices,
                    confidence=round(top[0][1], 3),
                    segment_ids=[segments[i].segment_id for i in matched_indices],
                    start_times=[round(segments[i].start_time, 3) for i in matched_indices],
                    end_times=[round(segments[i].end_time, 3) for i in matched_indices],
                    speakers=[segments[i].speaker for i in matched_indices],
                ))

    logger.info(
        f"Source map (streaming): {len(segments)} segments, "
        f"{len(source_map.links)} links"
    )
    return source_map
