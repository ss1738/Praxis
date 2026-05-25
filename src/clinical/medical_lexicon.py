"""UK primary-care medical lexicon for ASR contextual biasing.

Curated seed of ~180 UK GP terms (drugs, conditions, anatomy, abbreviations)
plus optional loaders for full SNOMED CT UK / dm+d / BNF dumps and corpus
mining via medical-suffix heuristics.

Used by `src.speech_to_text.conformer_engine.ConformerEngine.transcribe(..., bias=True)`
to boost recognition of medical terms at decode time via pyctcdecode.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parents[2]


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
otitis conjunctivitis blepharitis glaucoma cataract stye chalazion
hypertension hypotension angina arrhythmia palpitations
diabetes hypoglycaemia hyperglycaemia thyrotoxicosis hypothyroidism
migraine vertigo tinnitus bppv sciatica fibromyalgia
gastritis oesophagitis constipation diarrhoea haemorrhoids
allergic rhinitis hayfever herpes varicella
cystitis pyelonephritis prostatitis endometriosis dysmenorrhoea menorrhagia
bnf nice cks qof snomed ecg fbc crp esr hba1c tsh ldl hdl
"""


_SUFFIXES = (
    "itis", "osis", "emia", "aemia", "oma", "rrhoea", "phagia",
    "pathy", "ectomy", "otomy", "plasty", "algia", "gram", "scopy",
    "cillin", "azole", "olol", "pril", "sartan", "statin", "zepam",
    "profen", "caine", "mycin", "cycline", "pramine", "dipine",
)


def load_curated() -> set[str]:
    """Return the curated seed lexicon (~180 UK GP terms)."""
    out: set[str] = set()
    for line in _CURATED.strip().splitlines():
        for w in line.split():
            w = w.strip().lower()
            if w and len(w) >= 3:
                out.add(w)
    return out


def load_from_file(path: Path | str) -> set[str]:
    """Load a SNOMED CT UK / dm+d / BNF lexicon dump (one term per line)."""
    out: set[str] = set()
    p = Path(path)
    if not p.exists():
        logger.warning(f"Lexicon file not found: {p}")
        return out
    for line in p.read_text().splitlines():
        w = line.strip().lower()
        if w and len(w) >= 3:
            out.add(w)
    return out


def mine_corpus(corpus_path: Path | str, suffixes: tuple[str, ...] = _SUFFIXES) -> set[str]:
    """Mine medical-suffix terms from a chunked manifest's transcripts."""
    out: set[str] = set()
    p = Path(corpus_path)
    if not p.exists():
        return out
    entries = json.loads(p.read_text())
    pat = re.compile(r"[a-z][a-z\-']{4,}")
    for e in entries:
        text = (e.get("text") or "").lower()
        for w in pat.findall(text):
            if w.endswith(suffixes):
                out.add(w)
    return out


def build_lexicon(
    extra_path: Path | str | None = None,
    mine_train: bool = True,
    train_path: Path | str | None = None,
) -> list[str]:
    """Build the full lexicon: curated + optional file dump + mined train terms."""
    terms = load_curated()

    if extra_path:
        loaded = load_from_file(extra_path)
        logger.info(f"Loaded {len(loaded)} terms from {extra_path}")
        terms |= loaded

    if mine_train:
        train = Path(train_path) if train_path else (
            ROOT / "data" / "processed" / "primock57_chunked" / "train.json"
        )
        mined = mine_corpus(train)
        logger.info(f"Mined {len(mined)} suffix-matched terms from {train.name}")
        terms |= mined

    logger.info(f"Final medical lexicon: {len(terms)} terms")
    return sorted(terms)


def in_lexicon_recall(refs: list[str], hyps: list[str], lexicon: set[str]) -> dict[str, float]:
    """Compute medical-term recall between reference/hypothesis pairs.

    Returns dict with: ref_hits, recovered, mc_recall, mc_err, missed (top 30).
    """
    total_hits = 0
    recovered = 0
    missed: list[str] = []
    for ref, hyp in zip(refs, hyps):
        hyp_words = set(hyp.split())
        for w in ref.split():
            if w in lexicon:
                total_hits += 1
                if w in hyp_words:
                    recovered += 1
                else:
                    missed.append(w)
    mc_recall = recovered / total_hits if total_hits else 0.0
    return {
        "ref_hits": total_hits,
        "recovered": recovered,
        "mc_recall": round(mc_recall, 4),
        "mc_err": round(1.0 - mc_recall, 4) if total_hits else 0.0,
        "missed_top30": sorted(set(missed))[:30],
    }
