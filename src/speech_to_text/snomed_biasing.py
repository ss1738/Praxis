"""SNOMED CT UK contextual biasing at CTC decode time.

Passes UK primary-care terms (drugs, conditions, abbreviations) as hotwords to
a pyctcdecode beam-search decoder, boosting the log-probability of any candidate
path that produces a known medical term. Zero retraining required.

POC measurement: 31% relative MC-WER reduction on PriMock57 test (n=14),
boost=8.0, 185-term curated lexicon + suffix-mined train terms.
See scripts/snomed_biasing_poc.py for the original measurement script.

Usage:
    from src.speech_to_text.snomed_biasing import SnomedLexicon, SnomedBiaser

    lexicon = SnomedLexicon()
    biaser = SnomedBiaser(lexicon, boost=8.0)
    biaser.build_decoder(vocab_labels)          # once per session
    text = biaser.decode(log_probs_np)          # per utterance
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent.parent

# 185 curated UK primary-care terms: drugs, conditions, anatomy, NHS abbreviations
_CURATED = """
paracetamol ibuprofen aspirin amoxicillin flucloxacillin doxycycline clarithromycin
erythromycin azithromycin metronidazole ciprofloxacin trimethoprim nitrofurantoin
omeprazole lansoprazole ranitidine simvastatin atorvastatin rosuvastatin
ramipril lisinopril perindopril amlodipine bisoprolol atenolol bendroflumethiazide
furosemide spironolactone warfarin apixaban rivaroxaban clopidogrel ticagrelor
metformin gliclazide sitagliptin empagliflozin dapagliflozin insulin levothyroxine
prednisolone hydrocortisone beclometasone fluticasone salbutamol salmeterol
tiotropium ipratropium montelukast cetirizine loratadine fexofenadine piriton
chlorphenamine diazepam lorazepam sertraline citalopram fluoxetine mirtazapine
venlafaxine amitriptyline nortriptyline pregabalin gabapentin tramadol codeine
morphine naproxen diclofenac tamsulosin finasteride sildenafil tadalafil
levonorgestrel microgynon cerazette noriday depo-provera mirena nexplanon
emollient oilatum aveeno cetraben diprobase epaderm betnovate
dermovate eumovate hydromol zerobase
eczema dermatitis psoriasis urticaria impetigo cellulitis shingles tinea
asthma copd bronchitis pneumonia sinusitis pharyngitis tonsillitis laryngitis
otitis conjunctivitis blepharitis glaucoma cataract
hypertension hypotension angina arrhythmia palpitations atrial fibrillation
diabetes hypoglycaemia hyperglycaemia thyrotoxicosis hypothyroidism
migraine vertigo tinnitus bppv sciatica fibromyalgia osteoarthritis
gastritis oesophagitis peptic ulcer gord reflux constipation diarrhoea haemorrhoids
urticaria allergic rhinitis hayfever herpes varicella
cystitis pyelonephritis prostatitis endometriosis dysmenorrhoea menorrhagia
bnf nice cks qof snomed gp ecg fbc crp esr hba1c tsh ldl hdl egfr inr
sbp dbp bmi bp mmhg spo2 sats prn bd tds qds od nocte mane
snomed icd hrt cocp pop larc iud ius
"""

# Suffixes characteristic of medical terms - used for corpus mining
_MEDICAL_SUFFIXES = (
    "itis", "osis", "emia", "aemia", "oma", "rrhoea", "phagia",
    "pathy", "ectomy", "otomy", "plasty", "algia", "gram", "scopy",
    "cillin", "azole", "olol", "pril", "sartan", "statin", "zepam",
    "profen", "caine", "mycin", "cycline", "pramine", "dipine",
)

_NONWORD_RE = re.compile(r"[^a-z0-9' ]+")
_WS_RE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    text = text.lower()
    text = _NONWORD_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


class SnomedLexicon:
    """UK primary-care medical term lexicon for contextual biasing.

    Combines:
    - 185 curated UK primary-care terms (drugs, conditions, abbreviations)
    - Optional external term file (one term per line, e.g. SNOMED CT UK dump)
    - Optional suffix-based corpus mining from a training manifest
    """

    def __init__(
        self,
        extra_path: str | Path | None = None,
        mine_corpus: str | Path | None = None,
        min_term_len: int = 3,
    ) -> None:
        self._min_len = min_term_len
        self._terms: set[str] = self._load_curated()

        if extra_path is not None:
            self._load_file(Path(extra_path))

        if mine_corpus is not None:
            self._mine_suffixes(Path(mine_corpus))

    # ── Construction ──────────────────────────────────────────────────────────

    def _load_curated(self) -> set[str]:
        terms: set[str] = set()
        for line in _CURATED.strip().splitlines():
            for w in line.split():
                w = w.strip().lower()
                if w and len(w) >= self._min_len:
                    terms.add(w)
        return terms

    def _load_file(self, path: Path) -> None:
        if not path.exists():
            logger.warning(f"Lexicon file not found: {path}")
            return
        added = 0
        with path.open() as f:
            for line in f:
                w = line.strip().lower()
                if w and len(w) >= self._min_len and w not in self._terms:
                    self._terms.add(w)
                    added += 1
        logger.info(f"Loaded {added} additional terms from {path}")

    def _mine_suffixes(self, corpus_path: Path) -> None:
        """Mine medical-suffix terms from a JSON training manifest."""
        if not corpus_path.exists():
            logger.warning(f"Corpus not found for suffix mining: {corpus_path}")
            return
        import json
        mined = 0
        try:
            entries = json.loads(corpus_path.read_text())
            for entry in entries:
                text = _normalise(entry.get("text", ""))
                for w in text.split():
                    if (
                        len(w) >= 5
                        and w.endswith(_MEDICAL_SUFFIXES)
                        and w not in self._terms
                    ):
                        self._terms.add(w)
                        mined += 1
        except Exception as exc:
            logger.warning(f"Suffix mining failed: {exc}")
        if mined:
            logger.info(f"Mined {mined} suffix-matched terms from {corpus_path.name}")

    # ── Public interface ──────────────────────────────────────────────────────

    @property
    def terms(self) -> list[str]:
        return sorted(self._terms)

    def __len__(self) -> int:
        return len(self._terms)

    def contains(self, word: str) -> bool:
        return word.lower() in self._terms

    def medical_term_recall(
        self,
        references: list[str],
        hypotheses: list[str],
    ) -> dict[str, Any]:
        """Compute medical-term recall (MC-WER) across ref/hyp pairs.

        Returns dict with: total_ref_hits, total_recovered, mc_recall,
        mc_err, missed_terms.
        """
        total_hits = 0
        total_recovered = 0
        missed: list[str] = []

        for ref, hyp in zip(references, hypotheses):
            ref_words = _normalise(ref).split()
            hyp_words = set(_normalise(hyp).split())
            for w in ref_words:
                if w in self._terms:
                    total_hits += 1
                    if w in hyp_words:
                        total_recovered += 1
                    else:
                        missed.append(w)

        mc_recall = total_recovered / total_hits if total_hits else 1.0
        return {
            "total_ref_hits": total_hits,
            "total_recovered": total_recovered,
            "mc_recall": round(mc_recall, 4),
            "mc_err": round(1.0 - mc_recall, 4),
            "missed_terms": sorted(set(missed)),
        }


class SnomedBiaser:
    """Hotword-boosted CTC beam search using SNOMED CT UK terms.

    Wraps pyctcdecode when available; falls back to greedy decode otherwise.
    Lazy-loads the decoder - call build_decoder(vocab_labels) once per session
    before decoding.
    """

    def __init__(self, lexicon: SnomedLexicon, boost: float = 8.0, beam_width: int = 100) -> None:
        self.lexicon = lexicon
        self.boost = boost
        self.beam_width = beam_width
        self._decoder: Any = None
        self._built = False
        self._available: bool | None = None

    # ── Availability ──────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """True if pyctcdecode is importable."""
        if self._available is None:
            try:
                import pyctcdecode  # noqa: F401
                self._available = True
            except ImportError:
                self._available = False
                logger.warning("pyctcdecode not installed - SnomedBiaser will use greedy fallback")
        return self._available

    # ── Decoder setup ─────────────────────────────────────────────────────────

    def build_decoder(self, vocab_labels: list[str]) -> bool:
        """Build pyctcdecode decoder from CTC vocab labels.

        vocab_labels: list where index 0 = blank token, remaining = subword labels.
        Returns True if decoder was built, False if pyctcdecode unavailable.
        """
        if not self.is_available():
            return False
        try:
            from pyctcdecode import build_ctcdecoder
            self._decoder = build_ctcdecoder(vocab_labels)
            self._built = True
            logger.info(
                f"SnomedBiaser decoder built: {len(vocab_labels)} labels, "
                f"{len(self.lexicon)} hotwords, boost={self.boost}"
            )
            return True
        except Exception as exc:
            logger.warning(f"Failed to build pyctcdecode decoder: {exc}")
            return False

    # ── Decode ────────────────────────────────────────────────────────────────

    def decode(self, log_probs: np.ndarray) -> str:
        """Hotword-boosted beam search decode.

        log_probs: (T, vocab_size) numpy float32 array.
        Falls back to greedy if decoder not built or pyctcdecode unavailable.
        """
        if self._built and self._decoder is not None:
            try:
                return self._decoder.decode(
                    log_probs,
                    hotwords=self.lexicon.terms,
                    hotword_weight=self.boost,
                    beam_width=self.beam_width,
                ).strip()
            except Exception as exc:
                logger.warning(f"Biased decode failed, falling back to greedy: {exc}")

        # Greedy fallback
        return self._greedy_decode(log_probs)

    @staticmethod
    def _greedy_decode(log_probs: np.ndarray) -> str:
        """CTC greedy decode - blank=0, no vocabulary (returns raw token ids as string)."""
        ids = log_probs.argmax(axis=-1).tolist()
        out, prev = [], -1
        for i in ids:
            if i != 0 and i != prev:
                out.append(i)
            prev = i
        return " ".join(str(i) for i in out)
