"""Per-sentence NLI entailment gate - hallucination detection for SOAP notes.

For each sentence in a generated SOAP note, checks whether it is entailed by
the source transcript using a DeBERTa-v3 cross-encoder NLI model. Sentences
with entailment score below the threshold are flagged as potentially unsupported
or hallucinated.

Target: beat Asgari et al. npj Digital Medicine 2025 bar of
    - 1.47 % hallucination rate
    - 3.45 % omission rate
on UK primary-care consultations (PriMock57 + Leicester pilot).

Default model: cross-encoder/nli-deberta-v3-small (~180 MB, MIT, CPU-viable)
The model is lazy-loaded on first call and cached for the process lifetime.
Falls back gracefully when the model is unavailable (no crash, gate skipped).

Usage:
    gate = EntailmentGate()
    result = gate.check(soap_note, transcript)
    print(f"Hallucination rate: {result.hallucination_rate:.1%}")
    for s in result.flagged:
        print(f"  [{s.entailment_score:.2f}] {s.sentence}")

Wire into full validation:
    from src.clinical.verification import validate_soap_note
    result = validate_soap_note(note, transcript, use_entailment_gate=True)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-small"
DEFAULT_THRESHOLD = 0.50  # P(entailment) below this → flagged
MAX_WINDOW_TOKENS = 350   # words; keeps total input within 512 tokens

_tokenizer_cache: dict[str, Any] = {}
_model_cache: dict[str, Any] = {}


# ── Data types ────────────────────────────────────────────────────────────────


@dataclass
class SentenceResult:
    """Entailment result for one SOAP sentence."""
    sentence: str
    entailment_score: float       # P(entailment | premise=transcript, hyp=sentence)
    contradiction_score: float
    neutral_score: float
    supported: bool               # entailment_score >= threshold
    transcript_window: str        # excerpt used as NLI premise (truncated for storage)


@dataclass
class GateResult:
    """Aggregate entailment gate result for one SOAP note."""
    num_checked: int
    num_supported: int
    num_flagged: int
    hallucination_rate: float     # num_flagged / num_checked
    supported_rate: float         # num_supported / num_checked
    flagged: list[SentenceResult] = field(default_factory=list)
    all_sentences: list[SentenceResult] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""


# ── Sentence splitting ────────────────────────────────────────────────────────

_SECTION_RE = re.compile(r'^\s*(SUBJECTIVE|OBJECTIVE|ASSESSMENT|PLAN)\s*:', re.I)
_NUMBERED_BARE = re.compile(r'^\d+\.\s*$')
_NUMBERED_PREFIX = re.compile(r'^\d+\.\s+')
_MIN_WORDS = 5


def _split_sentences(note_text: str) -> list[str]:
    """Split a SOAP note into individually assessable clinical sentences."""
    sentences: list[str] = []
    for raw_line in note_text.split('\n'):
        line = raw_line.strip()
        if not line:
            continue
        # Strip section header prefix, keep remainder
        line = _SECTION_RE.sub('', line).strip()
        if not line:
            continue
        if _NUMBERED_BARE.match(line):
            continue
        # Strip leading "N. " from list items
        line = _NUMBERED_PREFIX.sub('', line).strip()
        # Split on sentence boundaries within long lines
        for part in re.split(r'(?<=[.!?])\s+', line):
            part = part.strip()
            if len(part.split()) >= _MIN_WORDS:
                sentences.append(part)
    return sentences


# ── Transcript window selection ───────────────────────────────────────────────


def _select_window(transcript: str, sentence: str, max_tokens: int = MAX_WINDOW_TOKENS) -> str:
    """Return the most relevant excerpt of transcript for the given sentence.

    Scores sliding windows by keyword overlap; returns full transcript when
    short enough.
    """
    words = transcript.split()
    if len(words) <= max_tokens:
        return transcript

    keywords = set(re.findall(r'\b[a-z]{4,}\b', sentence.lower()))
    if not keywords:
        return ' '.join(words[:max_tokens])

    step = max(1, max_tokens // 3)
    max_valid_start = max(0, len(words) - max_tokens)
    best_start = 0
    best_score = -1

    # Sweep with step, then always include the final valid window if not covered
    starts = list(range(0, max_valid_start + 1, step))
    if not starts or starts[-1] < max_valid_start:
        starts.append(max_valid_start)

    for start in starts:
        chunk = words[start:start + max_tokens]
        overlap = len(keywords & {w.lower() for w in chunk})
        if overlap > best_score:
            best_score = overlap
            best_start = start

    return ' '.join(words[best_start:best_start + max_tokens])


# ── Model loading ─────────────────────────────────────────────────────────────


def _load_model(model_name: str = DEFAULT_MODEL) -> tuple[Any, Any]:
    """Lazy-load the NLI model and tokenizer. Returns (tokenizer, model) or (None, None)."""
    if model_name in _tokenizer_cache:
        return _tokenizer_cache[model_name], _model_cache[model_name]

    try:
        import torch  # noqa: F401  (confirm torch available)
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
        mdl.eval()

        _tokenizer_cache[model_name] = tok
        _model_cache[model_name] = mdl
        logger.info(f"NLI entailment model loaded: {model_name}")
        return tok, mdl

    except Exception as exc:
        logger.warning(f"NLI model unavailable ({model_name}): {exc}")
        _tokenizer_cache[model_name] = None
        _model_cache[model_name] = None
        return None, None


# ── Scoring ───────────────────────────────────────────────────────────────────


def _score_entailment(
    premise: str,
    hypothesis: str,
    tokenizer: Any,
    model: Any,
) -> tuple[float, float, float]:
    """Run NLI and return (P_entailment, P_contradiction, P_neutral).

    premise   = transcript window (ground truth)
    hypothesis = SOAP sentence (claim being checked)
    """
    import torch

    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)[0].tolist()
    id2label: dict[int, str] = {k: v.lower() for k, v in model.config.id2label.items()}

    # Build a label→prob map; fall back to positional guesses for unusual orderings
    label_prob: dict[str, float] = {id2label.get(i, str(i)): p for i, p in enumerate(probs)}

    p_e = label_prob.get("entailment", probs[1] if len(probs) > 1 else 1 / 3)
    p_c = label_prob.get("contradiction", probs[0])
    p_n = label_prob.get("neutral", probs[2] if len(probs) > 2 else 1 / 3)
    return float(p_e), float(p_c), float(p_n)


# ── Main gate class ───────────────────────────────────────────────────────────


class EntailmentGate:
    """NLI-based per-sentence entailment check for SOAP note hallucination detection.

    Instantiate once; the underlying model is lazy-loaded and cached.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        threshold: float = DEFAULT_THRESHOLD,
        max_window_tokens: int = MAX_WINDOW_TOKENS,
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self.max_window_tokens = max_window_tokens
        self._tokenizer: Any = None
        self._model: Any = None
        self._loaded = False

    # Allow patching in tests without going through the module-level cache
    def _load(self) -> bool:
        if not self._loaded:
            self._tokenizer, self._model = _load_model(self.model_name)
            self._loaded = True
        return self._model is not None

    def check(self, soap_note: str, transcript: str) -> GateResult:
        """Check all assessable sentences in soap_note against transcript.

        Returns a GateResult with per-sentence scores and aggregate stats.
        Gracefully skips if the NLI model is not available.
        """
        if not transcript.strip():
            return GateResult(
                num_checked=0, num_supported=0, num_flagged=0,
                hallucination_rate=0.0, supported_rate=1.0,
                skipped=True, skip_reason="no transcript provided",
            )

        sentences = _split_sentences(soap_note)
        if not sentences:
            return GateResult(
                num_checked=0, num_supported=0, num_flagged=0,
                hallucination_rate=0.0, supported_rate=1.0,
            )

        if not self._load():
            return GateResult(
                num_checked=len(sentences),
                num_supported=len(sentences),
                num_flagged=0,
                hallucination_rate=0.0,
                supported_rate=1.0,
                skipped=True,
                skip_reason="NLI model unavailable - entailment gate skipped",
            )

        results: list[SentenceResult] = []
        for sentence in sentences:
            window = _select_window(transcript, sentence, self.max_window_tokens)
            p_e, p_c, p_n = _score_entailment(window, sentence, self._tokenizer, self._model)
            results.append(SentenceResult(
                sentence=sentence,
                entailment_score=round(p_e, 4),
                contradiction_score=round(p_c, 4),
                neutral_score=round(p_n, 4),
                supported=p_e >= self.threshold,
                transcript_window=window[:400],
            ))

        flagged = sorted(
            [r for r in results if not r.supported],
            key=lambda x: x.entailment_score,
        )
        n = len(results)
        n_flagged = len(flagged)

        return GateResult(
            num_checked=n,
            num_supported=n - n_flagged,
            num_flagged=n_flagged,
            hallucination_rate=round(n_flagged / n, 4) if n else 0.0,
            supported_rate=round((n - n_flagged) / n, 4) if n else 1.0,
            flagged=flagged,
            all_sentences=results,
        )
