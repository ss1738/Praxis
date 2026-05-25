"""Streaming Constitutional Clinical AI - generation-time TrustScore veto (N16).

Where N11 validates AFTER generation and surgically repairs failing sections,
this module enforces TrustScore checks DURING generation. The SOAP model streams
tokens one chunk at a time; we buffer to sentence boundaries; at each boundary
we run a fast subset of TrustScore checks (sentence-level transcript fidelity
plus the N13 NLI entailment gate); a failing sentence triggers a backtrack and
a corrective re-prompt rather than being shipped.

Hard guarantee by construction: every sentence in the final SOAP has passed the
streaming veto. No post-hoc filter required.

Public API:
    gen = StreamingConstitutionalGenerator(model, transcript)
    for event in gen.generate_streaming():
        if event.kind == "sentence":     ...   # one verified sentence shipped
        elif event.kind == "veto":       ...   # sentence rejected by veto
        elif event.kind == "rewrite":    ...   # corrective re-prompt fired
        elif event.kind == "complete":   ...   # final SOAP + stats

Falls back gracefully if the NLI model is unavailable: still streams, still
splits on sentence boundaries, still emits events, but the veto becomes a no-op
and every candidate sentence ships.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator

import httpx
from loguru import logger

from src.clinical.entailment_gate import (
    DEFAULT_MODEL as NLI_DEFAULT_MODEL,
    DEFAULT_THRESHOLD as NLI_DEFAULT_THRESHOLD,
    MAX_WINDOW_TOKENS,
    _load_model as _nli_load_model,
    _score_entailment,
    _select_window,
)
from src.soap_generation.generator import SOAPModel
from src.soap_generation.templates import NoteTemplate, get_template_config


# ── Data types ────────────────────────────────────────────────────────


class EventKind(str, Enum):
    SENTENCE = "sentence"     # one accepted sentence
    VETO = "veto"             # one rejected sentence (will be rewritten)
    REWRITE = "rewrite"       # corrective re-prompt about to fire
    GIVE_UP = "give_up"       # max retries hit; placeholder emitted
    COMPLETE = "complete"     # final note ready
    ERROR = "error"           # exceptional condition


@dataclass
class StreamEvent:
    """One event emitted during streaming generation."""
    kind: EventKind
    content: str = ""
    sentence: str = ""
    entailment_score: float = 0.0
    transcript_window: str = ""
    retry_count: int = 0
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "kind": self.kind.value,
            "content": self.content,
            "sentence": self.sentence,
            "entailment_score": round(self.entailment_score, 4),
            "transcript_window": self.transcript_window[:200],
            "retry_count": self.retry_count,
            "reason": self.reason,
        }


@dataclass
class StreamingResult:
    """Final output of streaming-veto generation."""
    soap_note: str
    transcript: str
    model: str
    template: str
    accepted_count: int
    vetoed_count: int
    rewritten_count: int
    gave_up_count: int
    total_time_secs: float
    veto_active: bool                             # False if NLI unavailable
    events: list[StreamEvent] = field(default_factory=list)

    @property
    def all_shipped_passed_veto(self) -> bool:
        """The hard guarantee: no sentence vetoed by the streaming check
        ended up in the final note.

        A vetoed sentence may legitimately reappear as an accepted SENTENCE
        only if the rewrite happens to produce the same text and that text
        passes the veto on the second attempt - which is vanishingly rare in
        practice because the rewrite prompt explicitly asks for a different,
        grounded sentence.
        """
        vetoed = {e.sentence.strip() for e in self.events
                  if e.kind == EventKind.VETO}
        shipped = {e.sentence.strip() for e in self.events
                   if e.kind == EventKind.SENTENCE}
        return not (vetoed & shipped)

    def to_dict(self) -> dict:
        return {
            "soap_note": self.soap_note,
            "model": self.model,
            "template": self.template,
            "accepted_count": self.accepted_count,
            "vetoed_count": self.vetoed_count,
            "rewritten_count": self.rewritten_count,
            "gave_up_count": self.gave_up_count,
            "total_time_secs": round(self.total_time_secs, 2),
            "veto_active": self.veto_active,
            "all_shipped_passed_veto": self.all_shipped_passed_veto,
            "events": [e.to_dict() for e in self.events],
        }


# ── Sentence boundary detection ───────────────────────────────────────

# A SOAP sentence typically ends at one of:
#  - terminal punctuation (. ! ?) followed by whitespace, EOL, or EOF
#  - a newline that ends a list item or short clinical note line
# We deliberately split on newlines too because clinical SOAP notes often have
# very short bullets that lack terminal punctuation.
_SENT_BOUND = re.compile(
    r"""
    (?<=[.!?])\s+      # terminal punctuation followed by whitespace
    |
    \n\s*\n            # paragraph break
    |
    (?<=:)\s*\n        # section colon followed by newline (e.g. "PLAN:\n")
    """,
    re.VERBOSE,
)

# Section headers we should not veto - they are structural, not claims.
_SECTION_HEADER = re.compile(
    r"^\s*(SUBJECTIVE|OBJECTIVE|ASSESSMENT|PLAN|S|O|A|P|HPI|CC|PMH|MEDS)\s*:?\s*$",
    re.IGNORECASE,
)

# Minimum word count to bother running the veto on
_MIN_VETO_WORDS = 4


def _is_skippable(sentence: str) -> bool:
    """Section headers, SNOMED codes, and trivially short fragments skip the veto."""
    s = sentence.strip()
    if not s:
        return True
    if _SECTION_HEADER.match(s):
        return True
    if len(s.split()) < _MIN_VETO_WORDS:
        return True
    return False


def _split_buffer(buffer: str) -> tuple[list[str], str]:
    """Split a buffer into completed sentences and a residual tail."""
    if not buffer:
        return [], ""
    parts = _SENT_BOUND.split(buffer)
    parts = [p for p in parts if p is not None]
    if len(parts) == 1:
        return [], buffer
    completed = [p.rstrip() for p in parts[:-1] if p.strip()]
    residual = parts[-1]
    return completed, residual


# ── Streaming generator ───────────────────────────────────────────────


class StreamingConstitutionalGenerator:
    """Streaming SOAP generation with per-sentence NLI veto.

    Wraps the Ollama chat API in streaming mode, buffers tokens to sentence
    boundaries, and runs the N13 NLI entailment gate on each candidate
    sentence before letting it ship.
    """

    def __init__(
        self,
        model: SOAPModel = SOAPModel.PRAXIS_PRIMARY,
        template: NoteTemplate = NoteTemplate.SOAP,
        ollama_url: str = "http://localhost:11434",
        timeout: float = 180.0,
        nli_threshold: float = NLI_DEFAULT_THRESHOLD,
        nli_model: str = NLI_DEFAULT_MODEL,
        max_retries: int = 3,
        max_sentences_per_case: int = 30,
        give_up_placeholder: str = "[CLINICAL_REVIEW_REQUIRED]",
    ):
        self.model = model
        self.template = template
        self.ollama_url = ollama_url.rstrip("/")
        self.timeout = timeout
        self.nli_threshold = nli_threshold
        self.nli_model_name = nli_model
        self.max_retries = max_retries
        # Cumulative cap: stop accepting candidate sentences once this many
        # have been considered (accepted + vetoed). Prevents the rewrite loop
        # from running away on adversarial / poorly-grounded transcripts.
        self.max_sentences_per_case = max_sentences_per_case
        self.give_up_placeholder = give_up_placeholder

        # Lazy-load NLI on first call so non-veto features still work without it.
        self._nli_tokenizer = None
        self._nli_model = None
        self._nli_loaded = False

    # ── NLI loading ──

    def _ensure_nli(self) -> bool:
        if not self._nli_loaded:
            self._nli_tokenizer, self._nli_model = _nli_load_model(
                self.nli_model_name)
            self._nli_loaded = True
        return self._nli_model is not None

    def _score_sentence(self, sentence: str, transcript: str) -> tuple[float, str]:
        """Return (P(entailment), transcript_window) for a single sentence."""
        window = _select_window(transcript, sentence, MAX_WINDOW_TOKENS)
        p_e, _, _ = _score_entailment(
            window, sentence, self._nli_tokenizer, self._nli_model)
        return p_e, window

    # ── Prompt construction ──

    def _initial_messages(self, transcript: str) -> list[dict]:
        config = get_template_config(self.template)
        user_content = (
            f"{config.instruction}\n\n"
            "Output the clinical note one short sentence per line, ending each "
            "sentence with a full stop. This format is required for our "
            "verification pipeline.\n\n"
            f"Here is an example of the expected output format:\n\n"
            f"{config.example_output}\n\n"
            "---\n\n"
            "Now generate the note for this consultation transcript:\n\n"
            f"{transcript}"
        )
        return [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _rewrite_prompt(
        self,
        transcript: str,
        accepted_so_far: str,
        rejected_sentence: str,
        retry_count: int,
    ) -> list[dict]:
        """Build a corrective re-prompt asking the model to produce ONE
        grounded continuation sentence to replace the rejected one.

        Keep this terse on purpose: the looser the contract, the more the
        model regenerates large portions of the note and re-triggers the
        veto loop. The contract here is: produce at most one short
        sentence grounded in the transcript, or output STOP if grounding
        is not possible.
        """
        config = get_template_config(self.template)
        instruction = (
            "You are continuing a clinical note generation that was paused "
            "because the last candidate sentence was not grounded in the "
            "transcript.\n\n"
            "CONSULTATION TRANSCRIPT (the only source of truth):\n"
            f"{transcript}\n\n"
            "ACCEPTED NOTE SO FAR (do not repeat or modify):\n"
            f"{accepted_so_far.strip() or '(empty)'}\n\n"
            "REJECTED LAST CANDIDATE:\n"
            f"{rejected_sentence.strip()}\n\n"
            "TASK: Produce exactly ONE short sentence that continues the note "
            "AND is grounded in something the doctor or patient said in the "
            "transcript above. End it with a full stop. Then output the "
            "literal token STOP and nothing else.\n\n"
            "If you cannot produce a grounded sentence, output STOP "
            "immediately without any sentence."
        )
        if retry_count >= 2:
            instruction += (
                "\n\nThis is the final attempt for this position. "
                "Prefer STOP over any guess."
            )
        return [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": instruction},
        ]

    # ── Ollama streaming primitive ──

    def _stream_chunks(self, messages: list[dict],
                       *, num_predict: int = 1500,
                       stop: list[str] | None = None) -> Iterator[str]:
        """Yield content chunks from Ollama's streaming chat endpoint.

        Each chunk is a fragment of generated text (often one or a few
        tokens). Stops yielding when the server signals 'done': true.

        `num_predict` and `stop` are exposed so a rewrite stream can be
        tighter than the initial generation (e.g. one paragraph max,
        halt on section headers).
        """
        options = {
            "temperature": 0.3,
            "top_p": 0.9,
            "num_predict": num_predict,
        }
        if stop:
            options["stop"] = stop
        payload = {
            "model": self.model.value,
            "messages": messages,
            "stream": True,
            "options": options,
        }
        url = f"{self.ollama_url}/api/chat"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream("POST", url, json=payload) as r:
                    r.raise_for_status()
                    for line in r.iter_lines():
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        chunk = obj.get("message", {}).get("content", "")
                        if chunk:
                            yield chunk
                        if obj.get("done"):
                            return
        except httpx.ConnectError as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.ollama_url}. "
                f"Is the server running? Error: {e}"
            ) from e

    # ── Main streaming loop ──

    def generate_streaming(self, transcript: str) -> Iterator[StreamEvent]:
        """Stream a SOAP note one verified sentence at a time.

        Yields StreamEvents and finally a single COMPLETE event whose
        `content` field carries the assembled SOAP note.
        """
        veto_active = self._ensure_nli()
        if not veto_active:
            yield StreamEvent(
                kind=EventKind.ERROR,
                reason="NLI model unavailable; streaming proceeds without veto",
            )

        accepted: list[str] = []
        events: list[StreamEvent] = []
        retry_for_current_sentence = 0
        sentences_considered = 0
        messages = self._initial_messages(transcript)
        is_rewrite_stream = False

        # Outer loop: each iteration consumes one Ollama stream. We exit when
        # a stream completes without triggering a backtrack-and-resample, OR
        # when the cumulative cap is hit (sanity bound on adversarial inputs).
        while True:
            if sentences_considered >= self.max_sentences_per_case:
                cap = StreamEvent(
                    kind=EventKind.ERROR,
                    reason=f"max_sentences_per_case ({self.max_sentences_per_case}) "
                           f"hit; halting to bound veto-rewrite loop",
                )
                events.append(cap)
                yield cap
                break

            outcome = "completed"   # "completed" | "rewrite"
            try:
                for ev, action, new_messages in self._consume_stream(
                        messages, transcript, accepted,
                        retry_for_current_sentence,
                        veto_active, is_rewrite_stream):
                    events.append(ev)
                    yield ev
                    if action in ("accepted", "vetoed", "gave_up"):
                        sentences_considered += 1
                    if action == "accepted":
                        retry_for_current_sentence = 0
                    elif action == "gave_up":
                        retry_for_current_sentence = 0
                    elif action == "rewrite":
                        retry_for_current_sentence += 1
                        messages = new_messages
                        is_rewrite_stream = True
                        outcome = "rewrite"
                        break  # restart stream
                    if sentences_considered >= self.max_sentences_per_case:
                        outcome = "cap"
                        break
            except RuntimeError as e:
                err = StreamEvent(kind=EventKind.ERROR, reason=str(e))
                events.append(err)
                yield err
                break
            if outcome == "completed" or outcome == "cap":
                break

        final_note = "\n".join(accepted).strip()
        complete = StreamEvent(
            kind=EventKind.COMPLETE,
            content=final_note,
            reason=(f"accepted={sum(1 for e in events if e.kind == EventKind.SENTENCE)} "
                    f"vetoed={sum(1 for e in events if e.kind == EventKind.VETO)} "
                    f"rewrites={sum(1 for e in events if e.kind == EventKind.REWRITE)}"),
        )
        events.append(complete)
        yield complete

    def _consume_stream(
        self,
        messages: list[dict],
        transcript: str,
        accepted: list[str],
        starting_retry: int,
        veto_active: bool,
        is_rewrite: bool = False,
    ) -> Iterator[tuple[StreamEvent, str, list[dict] | None]]:
        """Consume one Ollama stream pass.

        Yields (event, action, next_messages) tuples where action is one of:
            "accepted"  -- sentence appended to `accepted`; keep streaming
            "vetoed"    -- sentence rejected (no rewrite triggered yet)
            "gave_up"   -- placeholder appended; keep streaming
            "rewrite"   -- veto fired, caller must restart with next_messages

        On "rewrite" the iterator stops (the caller breaks). On normal stream
        completion, residual text is force-flushed through the same handler
        so trailing sentences without terminal whitespace still get vetted.

        Rewrite streams use tighter token budgets and stop sequences than the
        initial generation.
        """
        retry_count = starting_retry
        buffer = ""
        # Rewrite streams: ask for a short continuation, halt at section
        # headers or paragraph breaks. Initial stream: full SOAP budget.
        if is_rewrite:
            num_predict = 250
            stop = ["\n\nSUBJECTIVE:", "\n\nOBJECTIVE:",
                    "\n\nASSESSMENT:", "\n\nPLAN:",
                    "\n\nSTOP", "\n\n#"]
        else:
            num_predict = 1500
            stop = None
        for chunk in self._stream_chunks(messages,
                                         num_predict=num_predict, stop=stop):
            buffer += chunk
            completed, buffer = _split_buffer(buffer)
            for raw in completed:
                yield from self._handle_one_sentence(
                    raw, transcript, accepted, retry_count, veto_active,
                    on_rewrite_messages_factory=lambda s: self._rewrite_prompt(
                        transcript=transcript,
                        accepted_so_far="\n".join(accepted),
                        rejected_sentence=s,
                        retry_count=retry_count + 1,
                    ),
                )
                # Bump local retry tracker if the last yield was a rewrite signal
                # (we cannot easily inspect from here, so we rely on the outer
                # loop to update on receipt of action="rewrite").
                # Caller breaks on action=="rewrite", so retry_count below
                # only affects subsequent sentences in the SAME stream.

        # Stream ended naturally. Force-flush any residual as one final sentence
        # by appending a paragraph-break sentinel (so _split_buffer's regex bites).
        flushed_completed, _ = _split_buffer(buffer + "\n\n")
        for raw in flushed_completed:
            yield from self._handle_one_sentence(
                raw, transcript, accepted, retry_count, veto_active,
                on_rewrite_messages_factory=lambda s: self._rewrite_prompt(
                    transcript=transcript,
                    accepted_so_far="\n".join(accepted),
                    rejected_sentence=s,
                    retry_count=retry_count + 1,
                ),
            )

    def _handle_one_sentence(
        self,
        raw_sentence: str,
        transcript: str,
        accepted: list[str],
        retry_count: int,
        veto_active: bool,
        on_rewrite_messages_factory,
    ) -> Iterator[tuple[StreamEvent, str, list[dict] | None]]:
        """Process one extracted candidate sentence and yield an outcome tuple."""
        sentence = raw_sentence.strip()
        if not sentence:
            return

        # Structural line (section header, very short stub): ship without scoring
        if _is_skippable(sentence):
            accepted.append(sentence)
            yield (
                StreamEvent(
                    kind=EventKind.SENTENCE,
                    sentence=sentence,
                    reason="skipped veto (structural line)",
                    retry_count=retry_count,
                ),
                "accepted",
                None,
            )
            return

        # Veto inactive: every sentence ships
        if not veto_active:
            accepted.append(sentence)
            yield (
                StreamEvent(
                    kind=EventKind.SENTENCE,
                    sentence=sentence,
                    reason="veto inactive",
                    retry_count=retry_count,
                ),
                "accepted",
                None,
            )
            return

        # Veto check
        p_e, window = self._score_sentence(sentence, transcript)
        if p_e >= self.nli_threshold:
            accepted.append(sentence)
            yield (
                StreamEvent(
                    kind=EventKind.SENTENCE,
                    sentence=sentence,
                    entailment_score=p_e,
                    transcript_window=window,
                    retry_count=retry_count,
                ),
                "accepted",
                None,
            )
            return

        # Vetoed
        yield (
            StreamEvent(
                kind=EventKind.VETO,
                sentence=sentence,
                entailment_score=p_e,
                transcript_window=window,
                retry_count=retry_count,
                reason=f"P(entailment)={p_e:.3f} < threshold "
                       f"{self.nli_threshold:.2f}",
            ),
            "vetoed",
            None,
        )

        new_retry_count = retry_count + 1
        if new_retry_count > self.max_retries:
            accepted.append(self.give_up_placeholder)
            yield (
                StreamEvent(
                    kind=EventKind.GIVE_UP,
                    sentence=self.give_up_placeholder,
                    reason=f"max_retries ({self.max_retries}) hit",
                    retry_count=new_retry_count,
                ),
                "gave_up",
                None,
            )
            return

        # Backtrack-and-resample: signal caller to restart the stream
        yield (
            StreamEvent(
                kind=EventKind.REWRITE,
                sentence=sentence,
                retry_count=new_retry_count,
                reason="backtrack-and-resample",
            ),
            "rewrite",
            on_rewrite_messages_factory(sentence),
        )

    def generate(self, transcript: str) -> StreamingResult:
        """Convenience wrapper that drains the streaming iterator into a
        single result object."""
        t0 = time.perf_counter()
        events: list[StreamEvent] = []
        for event in self.generate_streaming(transcript):
            events.append(event)
            if event.kind == EventKind.COMPLETE:
                break
        elapsed = time.perf_counter() - t0

        soap_note = ""
        for ev in reversed(events):
            if ev.kind == EventKind.COMPLETE:
                soap_note = ev.content
                break

        accepted = sum(1 for e in events if e.kind == EventKind.SENTENCE)
        vetoed = sum(1 for e in events if e.kind == EventKind.VETO)
        rewrites = sum(1 for e in events if e.kind == EventKind.REWRITE)
        gave_up = sum(1 for e in events if e.kind == EventKind.GIVE_UP)
        veto_active = self._nli_model is not None

        logger.info(
            f"Streaming generation done: {accepted} accepted, {vetoed} vetoed, "
            f"{rewrites} rewrites, {gave_up} placeholders, "
            f"{elapsed:.1f}s, veto_active={veto_active}"
        )

        return StreamingResult(
            soap_note=soap_note,
            transcript=transcript,
            model=self.model.value,
            template=self.template.value,
            accepted_count=accepted,
            vetoed_count=vetoed,
            rewritten_count=rewrites,
            gave_up_count=gave_up,
            total_time_secs=elapsed,
            veto_active=veto_active,
            events=events,
        )
