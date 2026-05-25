"""Constitutional Clinical AI - self-correcting SOAP generation.

Generates a clinical note, runs TrustScore checks during the generation loop,
and iteratively repairs failing sections using targeted correction prompts.
Repeats until the note reaches 'approved' (TrustScore ≥ 70) or max_iterations.

The 'constitution' is the 6 TrustScore checks. Each failed check triggers a
surgical repair of the specific failing section only - not a full regeneration -
keeping latency proportional to the number of issues, not the note length.

This is the novel contribution: deterministic clinical verification used as a
generation-time constraint, not a post-hoc filter.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import httpx
from loguru import logger

from src.clinical.verification import ValidationResult, validate_soap_note
from src.soap_generation.generator import SOAPGenerator, SOAPModel, SOAPResult
from src.soap_generation.templates import NoteTemplate


# ── Iteration data types ──────────────────────────────────────────────────────


@dataclass
class ConstitutionalIteration:
    """One generate-verify-repair cycle."""
    iteration: int
    note: str
    trust_score: int
    verdict: str
    checks_passed: int
    checks_total: int
    issues: list[str]
    repair_targets: list[str]          # which check names triggered repairs
    generation_time_secs: float


@dataclass
class ConstitutionalResult:
    """Final output of constitutional generation."""
    final_note: str
    final_trust_score: int
    final_verdict: str
    converged: bool                    # True if ≥70 reached within max_iterations
    iterations: list[ConstitutionalIteration]
    total_time_secs: float
    model: str
    template: str
    token_count: int = 0

    @property
    def num_iterations(self) -> int:
        return len(self.iterations)

    @property
    def score_trajectory(self) -> list[int]:
        return [it.trust_score for it in self.iterations]


# ── Repair prompt builders ────────────────────────────────────────────────────

# Maps check_name → (section_to_repair, correction_instruction)
_REPAIR_INSTRUCTIONS: dict[str, tuple[str, str]] = {
    "clinical_safety": (
        "PLAN",
        (
            "The PLAN section is missing required clinical safety content. "
            "Rewrite ONLY the PLAN section to include all of the following: "
            "(1) clear safety-netting advice - specific symptoms that should prompt the patient "
            "to call 111, attend A&E, or call 999; "
            "(2) a return-if-worse instruction; "
            "(3) any red-flag symptoms relevant to this presentation. "
            "Keep all other content in the PLAN unchanged."
        ),
    ),
    "transcript_fidelity": (
        "ASSESSMENT",
        (
            "The ASSESSMENT section contains clinical terms or medications that do not appear "
            "in the original consultation transcript - this is hallucinated content. "
            "Rewrite ONLY the ASSESSMENT section, removing any diagnoses, medications, or "
            "clinical findings that cannot be directly inferred from the transcript. "
            "Ground every statement in something the patient or doctor actually said."
        ),
    ),
    "nice_guidelines": (
        "PLAN",
        (
            "The PLAN section does not meet NICE clinical guidelines for the conditions "
            "documented in this note. "
            "Rewrite ONLY the PLAN section to address the NICE guideline requirements, "
            "including appropriate first-line investigations, treatments, or referral criteria "
            "per current NICE CKS recommendations. "
            "Keep all other content in the PLAN unchanged."
        ),
    ),
    "completeness": (
        "PLAN",
        (
            "One or more required sections of this clinical note are missing or too brief. "
            "Expand the note so that every required SOAP section "
            "(SUBJECTIVE, OBJECTIVE, ASSESSMENT, PLAN) is present and clinically meaningful. "
            "The PLAN must be at least 3 specific clinical actions."
        ),
    ),
    "drug_interactions": (
        "PLAN",
        (
            "The PLAN section prescribes medications with documented interactions or without "
            "required dosing information. "
            "Rewrite ONLY the PLAN section to: "
            "(1) include doses and frequencies for all prescribed medications; "
            "(2) add monitoring requirements for any interacting drug pairs; "
            "(3) note any contraindications that should be discussed with the patient."
        ),
    ),
    "snomed_validation": (
        "ASSESSMENT",
        (
            "The ASSESSMENT section contains SNOMED CT codes that are not in the verified "
            "UK primary-care code set. "
            "Rewrite ONLY the ASSESSMENT section, replacing any invalid SNOMED codes with "
            "correct codes from the approved list, or removing the code entirely if the "
            "condition does not map to a standard code."
        ),
    ),
}

# Priority order - which failing check to repair first when multiple fail
_REPAIR_PRIORITY = [
    "clinical_safety",      # patient safety first
    "completeness",         # structural integrity second
    "transcript_fidelity",  # hallucination third
    "drug_interactions",    # prescribing safety fourth
    "nice_guidelines",      # guideline compliance fifth
    "snomed_validation",    # coding correctness last
]


# ── Section extraction and splicing ──────────────────────────────────────────

_SECTION_PATTERNS: dict[str, str] = {
    "SUBJECTIVE": r"((?:^|\n)\s*SUBJECTIVE\s*:)(.*?)(?=(?:\n\s*(?:OBJECTIVE|ASSESSMENT|PLAN)\s*:)|$)",
    "OBJECTIVE":  r"((?:^|\n)\s*OBJECTIVE\s*:)(.*?)(?=(?:\n\s*(?:ASSESSMENT|PLAN)\s*:)|$)",
    "ASSESSMENT": r"((?:^|\n)\s*ASSESSMENT\s*:)(.*?)(?=(?:\n\s*PLAN\s*:)|$)",
    "PLAN":       r"((?:^|\n)\s*PLAN\s*:)(.*)",
}


def _extract_section(note: str, section: str) -> str:
    """Return the content of a named SOAP section, or '' if not found."""
    pat = _SECTION_PATTERNS.get(section.upper())
    if not pat:
        return ""
    m = re.search(pat, note, re.IGNORECASE | re.DOTALL)
    return m.group(2).strip() if m else ""


def _splice_section(note: str, section: str, new_content: str) -> str:
    """Replace a named SOAP section's content in the full note."""
    pat = _SECTION_PATTERNS.get(section.upper())
    if not pat:
        return note

    def replacer(m: re.Match) -> str:
        return f"{m.group(1)}\n{new_content}\n"

    result = re.sub(pat, replacer, note, flags=re.IGNORECASE | re.DOTALL)
    # If section wasn't found, append it
    if result == note:
        result = note.rstrip() + f"\n\n{section.upper()}:\n{new_content}\n"
    return result


# ── Constitutional generator ─────────────────────────────────────────────────


class ConstitutionalSOAPGenerator:
    """Self-correcting clinical note generator.

    Wraps SOAPGenerator with an iterative repair loop driven by TrustScore.
    Each iteration repairs at most one failing section, then re-validates.
    Tracks score trajectory and returns the best-scoring note seen.
    """

    def __init__(
        self,
        model: SOAPModel = SOAPModel.PRAXIS_PRIMARY,
        template: NoteTemplate = NoteTemplate.SOAP,
        ollama_url: str = "http://localhost:11434",
        max_iterations: int = 3,
        target_score: int = 70,
    ):
        self._gen = SOAPGenerator(model=model, template=template, ollama_url=ollama_url)
        self._model = model
        self._template = template
        self._max_iterations = max_iterations
        self._target_score = target_score

    def generate(self, transcript: str) -> ConstitutionalResult:
        """Generate a clinical note with iterative constitutional self-correction."""
        t_start = time.perf_counter()
        iterations: list[ConstitutionalIteration] = []

        # ── Initial generation ──
        logger.info("Constitutional generation: initial pass...")
        initial: SOAPResult = self._gen.generate(transcript, template=self._template)
        current_note = initial.soap_note
        total_tokens = initial.token_count

        validation = validate_soap_note(current_note, transcript)
        iterations.append(ConstitutionalIteration(
            iteration=0,
            note=current_note,
            trust_score=validation.trust_score,
            verdict=validation.verdict,
            checks_passed=validation.checks_passed,
            checks_total=validation.checks_total,
            issues=validation.issues[:],
            repair_targets=[],
            generation_time_secs=initial.generation_time_secs,
        ))
        logger.info(
            f"  Initial: TrustScore={validation.trust_score} "
            f"({validation.checks_passed}/{validation.checks_total} checks, "
            f"verdict={validation.verdict})"
        )

        # Track best across all iterations
        best_note = current_note
        best_score = validation.trust_score

        # ── Repair loop ──
        for i in range(1, self._max_iterations + 1):
            if validation.trust_score >= self._target_score:
                logger.info(f"  Converged at iteration {i - 1} (score={validation.trust_score})")
                break

            # Pick the highest-priority failing check
            failing = {r.check_name for r in validation.check_results if not r.passed}
            target_check = next(
                (c for c in _REPAIR_PRIORITY if c in failing),
                None,
            )
            if not target_check:
                logger.info("  No repairable checks remaining - stopping early")
                break

            section, instruction = _REPAIR_INSTRUCTIONS[target_check]
            logger.info(
                f"  Iteration {i}: repairing '{target_check}' in {section} section..."
            )

            repaired_note, repair_time, repair_tokens = self._repair_section(
                current_note=current_note,
                transcript=transcript,
                section=section,
                instruction=instruction,
                issues=[
                    iss for r in validation.check_results
                    if r.check_name == target_check
                    for iss in r.issues
                ],
            )
            total_tokens += repair_tokens
            current_note = repaired_note

            validation = validate_soap_note(current_note, transcript)
            logger.info(
                f"  After repair: TrustScore={validation.trust_score} "
                f"({validation.checks_passed}/{validation.checks_total}, "
                f"verdict={validation.verdict})"
            )

            iterations.append(ConstitutionalIteration(
                iteration=i,
                note=current_note,
                trust_score=validation.trust_score,
                verdict=validation.verdict,
                checks_passed=validation.checks_passed,
                checks_total=validation.checks_total,
                issues=validation.issues[:],
                repair_targets=[target_check],
                generation_time_secs=repair_time,
            ))

            if validation.trust_score > best_score:
                best_score = validation.trust_score
                best_note = current_note

        converged = best_score >= self._target_score
        total_time = round(time.perf_counter() - t_start, 2)

        logger.info(
            f"Constitutional generation complete: "
            f"score={best_score}, converged={converged}, "
            f"iterations={len(iterations)}, time={total_time}s"
        )

        return ConstitutionalResult(
            final_note=best_note,
            final_trust_score=best_score,
            final_verdict="approved" if best_score >= 70 else (
                "needs_review" if best_score >= 40 else "rejected"
            ),
            converged=converged,
            iterations=iterations,
            total_time_secs=total_time,
            model=self._model.value,
            template=self._template.value,
            token_count=total_tokens,
        )

    def _repair_section(
        self,
        current_note: str,
        transcript: str,
        section: str,
        instruction: str,
        issues: list[str],
    ) -> tuple[str, float, int]:
        """Ask the model to rewrite one failing section; splice it back in."""
        current_section_text = _extract_section(current_note, section)

        issues_text = ""
        if issues:
            issues_text = "\n\nSpecific issues to fix:\n" + "\n".join(f"- {i}" for i in issues)

        repair_prompt = (
            f"You are a UK GP documentation assistant. Below is a clinical note that needs "
            f"a targeted correction.\n\n"
            f"CONSULTATION TRANSCRIPT:\n{transcript}\n\n"
            f"CURRENT {section} SECTION:\n{current_section_text}\n\n"
            f"CORRECTION REQUIRED:\n{instruction}{issues_text}\n\n"
            f"Output ONLY the corrected {section} section content - "
            f"no section header, no other sections, no explanation. "
            f"Start directly with the corrected content."
        )

        client = httpx.Client(timeout=60.0)
        t0 = time.perf_counter()
        try:
            resp = client.post(
                f"{self._gen.ollama_url}/api/chat",
                json={
                    "model": self._model.value,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a UK GP clinical documentation assistant. "
                                "Output only the requested section content, nothing else."
                            ),
                        },
                        {"role": "user", "content": repair_prompt},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.1, "top_p": 0.9, "num_predict": 600},
                },
            )
            resp.raise_for_status()
            data = resp.json()
            repaired_content = data.get("message", {}).get("content", "").strip()
            repair_tokens = data.get("eval_count", 0)
        except Exception as e:
            logger.warning(f"Section repair failed ({section}): {e} - keeping original")
            return current_note, round(time.perf_counter() - t0, 2), 0
        finally:
            client.close()

        repair_time = round(time.perf_counter() - t0, 2)

        if not repaired_content:
            logger.warning(f"Empty repair response for {section} - keeping original")
            return current_note, repair_time, 0

        repaired_note = _splice_section(current_note, section, repaired_content)
        logger.info(f"  Repaired {section} in {repair_time:.1f}s ({repair_tokens} tokens)")
        return repaired_note, repair_time, repair_tokens
