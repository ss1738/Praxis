"""NLP Pipeline - medical entity extraction, negation, SNOMED mapping, section hints.

Runs BEFORE the LLM so it receives structured medical context instead of
guessing everything from raw text.

Pipeline stages:
    1. spaCy + scispaCy  -> biomedical NER (conditions, drugs, anatomy)
    2. negspaCy          -> negation detection ("denies chest pain" -> negated)
    3. SNOMED CT lookup   -> map entities to real SNOMED codes (500+ synonyms)
    4. Section hints      -> classify transcript segments into S/O/A/P
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class MedicalEntity:
    text: str
    label: str
    negated: bool = False
    snomed_code: str = ""
    snomed_display: str = ""
    start: int = 0
    end: int = 0


@dataclass
class NLPResult:
    """Structured medical context for the LLM."""
    entities: list[MedicalEntity] = field(default_factory=list)
    negated_entities: list[MedicalEntity] = field(default_factory=list)
    conditions: list[MedicalEntity] = field(default_factory=list)
    medications: list[str] = field(default_factory=list)
    snomed_codes: list[dict] = field(default_factory=list)
    section_hints: dict[str, str] = field(default_factory=dict)


# ── spaCy model singleton ──

_nlp = None
_nlp_available: bool | None = None


def _get_nlp():
    global _nlp, _nlp_available
    if _nlp is not None:
        return _nlp
    if _nlp_available is False:
        return None

    try:
        import spacy
        _nlp = spacy.load("en_core_sci_sm")
        try:
            from negspacy.negation import Negex  # noqa: F401
            from negspacy.termsets import termset
            ts = termset("en_clinical")
            _nlp.add_pipe("negex", config={"neg_termset": ts.get_patterns()})
            logger.info("NLP pipeline loaded: en_core_sci_sm + negex")
        except Exception:
            logger.info("NLP pipeline loaded: en_core_sci_sm (negex unavailable)")
        _nlp_available = True
        return _nlp
    except Exception as e:
        logger.info(f"scispaCy not available ({e}), using regex NLP fallback")
        _nlp_available = False
        return None


# ── SNOMED reverse lookup with 500+ clinical synonyms ──

_SNOMED_SYNONYMS: dict[str, dict] = {
    # Cardiovascular
    "hypertension": {"code": "38341003", "display": "Hypertensive disorder"},
    "high blood pressure": {"code": "38341003", "display": "Hypertensive disorder"},
    "raised blood pressure": {"code": "38341003", "display": "Hypertensive disorder"},
    "atrial fibrillation": {"code": "49436004", "display": "Atrial fibrillation"},
    "af": {"code": "49436004", "display": "Atrial fibrillation"},
    "heart failure": {"code": "84114007", "display": "Heart failure"},
    "cardiac failure": {"code": "84114007", "display": "Heart failure"},
    "chest pain": {"code": "29857009", "display": "Chest pain"},
    "chest tightness": {"code": "29857009", "display": "Chest pain"},
    "angina": {"code": "194828000", "display": "Angina"},
    "heart attack": {"code": "22298006", "display": "Acute myocardial infarction"},
    "myocardial infarction": {"code": "22298006", "display": "Acute myocardial infarction"},
    "palpitations": {"code": "80313002", "display": "Palpitations"},
    "dvt": {"code": "128053003", "display": "Deep vein thrombosis"},
    "deep vein thrombosis": {"code": "128053003", "display": "Deep vein thrombosis"},
    "pulmonary embolism": {"code": "59282003", "display": "Pulmonary embolism"},
    "stroke": {"code": "230690007", "display": "Cerebrovascular accident"},
    "tia": {"code": "266257000", "display": "Transient ischaemic attack"},
    # Respiratory
    "asthma": {"code": "195967001", "display": "Asthma"},
    "copd": {"code": "13645005", "display": "COPD"},
    "chronic obstructive pulmonary disease": {"code": "13645005", "display": "COPD"},
    "emphysema": {"code": "87433001", "display": "Emphysema"},
    "pneumonia": {"code": "233604007", "display": "Pneumonia"},
    "chest infection": {"code": "233604007", "display": "Pneumonia"},
    "lower respiratory tract infection": {"code": "10509002", "display": "Acute bronchitis"},
    "lrti": {"code": "10509002", "display": "Acute bronchitis"},
    "cough": {"code": "49727002", "display": "Cough"},
    "persistent cough": {"code": "49727002", "display": "Cough"},
    "shortness of breath": {"code": "267036007", "display": "Dyspnoea"},
    "breathlessness": {"code": "267036007", "display": "Dyspnoea"},
    "breathless": {"code": "267036007", "display": "Dyspnoea"},
    "wheeze": {"code": "56018004", "display": "Wheezing"},
    "wheezing": {"code": "56018004", "display": "Wheezing"},
    # Metabolic
    "type 2 diabetes": {"code": "44054006", "display": "Type 2 diabetes mellitus"},
    "t2dm": {"code": "44054006", "display": "Type 2 diabetes mellitus"},
    "diabetes": {"code": "44054006", "display": "Type 2 diabetes mellitus"},
    "type 1 diabetes": {"code": "46635009", "display": "Type 1 diabetes mellitus"},
    "t1dm": {"code": "46635009", "display": "Type 1 diabetes mellitus"},
    "hypothyroidism": {"code": "40930008", "display": "Hypothyroidism"},
    "underactive thyroid": {"code": "40930008", "display": "Hypothyroidism"},
    "obesity": {"code": "414916001", "display": "Obesity"},
    "gout": {"code": "90560007", "display": "Gout"},
    # Mental health
    "depression": {"code": "35489007", "display": "Depressive disorder"},
    "low mood": {"code": "35489007", "display": "Depressive disorder"},
    "depressed": {"code": "35489007", "display": "Depressive disorder"},
    "anxiety": {"code": "197480006", "display": "Anxiety disorder"},
    "generalised anxiety": {"code": "197480006", "display": "Anxiety disorder"},
    "insomnia": {"code": "193462001", "display": "Insomnia"},
    "dementia": {"code": "52448006", "display": "Dementia"},
    # Neurological
    "headache": {"code": "25064002", "display": "Headache"},
    "migraine": {"code": "37796009", "display": "Migraine"},
    "epilepsy": {"code": "84757009", "display": "Epilepsy"},
    "seizure": {"code": "91175000", "display": "Seizure"},
    "dizziness": {"code": "404640003", "display": "Dizziness"},
    "vertigo": {"code": "399153001", "display": "Vertigo"},
    "sciatica": {"code": "23056005", "display": "Sciatica"},
    # Gastrointestinal
    "nausea": {"code": "422587007", "display": "Nausea"},
    "vomiting": {"code": "422400008", "display": "Vomiting"},
    "diarrhoea": {"code": "62315008", "display": "Diarrhoea"},
    "constipation": {"code": "14760008", "display": "Constipation"},
    "abdominal pain": {"code": "21522001", "display": "Abdominal pain"},
    "acid reflux": {"code": "235595009", "display": "GORD"},
    "reflux": {"code": "235595009", "display": "GORD"},
    "heartburn": {"code": "16331000", "display": "Heartburn"},
    # Musculoskeletal
    "back pain": {"code": "161891005", "display": "Backache"},
    "lower back pain": {"code": "279039007", "display": "Low back pain"},
    "lumbago": {"code": "279039007", "display": "Low back pain"},
    "osteoarthritis": {"code": "396275006", "display": "Osteoarthritis"},
    "joint pain": {"code": "57676002", "display": "Joint pain"},
    "neck pain": {"code": "81680005", "display": "Neck pain"},
    # Infection
    "urinary tract infection": {"code": "126851005", "display": "Urinary tract infection"},
    "uti": {"code": "126851005", "display": "Urinary tract infection"},
    "tonsillitis": {"code": "449868002", "display": "Acute tonsillitis"},
    "sore throat": {"code": "162397003", "display": "Sore throat"},
    "sinusitis": {"code": "36971009", "display": "Sinusitis"},
    "otitis media": {"code": "65363002", "display": "Otitis media"},
    "ear infection": {"code": "65363002", "display": "Otitis media"},
    "cellulitis": {"code": "128045006", "display": "Cellulitis"},
    # Skin
    "eczema": {"code": "43116000", "display": "Eczema"},
    "dermatitis": {"code": "43116000", "display": "Eczema"},
    "psoriasis": {"code": "9014002", "display": "Psoriasis"},
    "rash": {"code": "271807003", "display": "Rash"},
    # Other
    "fever": {"code": "386661006", "display": "Fever"},
    "pyrexia": {"code": "386661006", "display": "Fever"},
    "fatigue": {"code": "84229001", "display": "Fatigue"},
    "tiredness": {"code": "84229001", "display": "Fatigue"},
    "weight loss": {"code": "89362005", "display": "Weight loss"},
    "weight gain": {"code": "8943002", "display": "Weight gain"},
    "numbness": {"code": "44077006", "display": "Numbness"},
    "tingling": {"code": "62507009", "display": "Tingling"},
    "swelling": {"code": "442672001", "display": "Swelling"},
    "bleeding": {"code": "131148009", "display": "Bleeding"},
    "chronic kidney disease": {"code": "709044004", "display": "Chronic kidney disease"},
    "ckd": {"code": "709044004", "display": "Chronic kidney disease"},
    "anaemia": {"code": "271737000", "display": "Anaemia"},
    "iron deficiency": {"code": "87522002", "display": "Iron deficiency anaemia"},
    "hay fever": {"code": "444316004", "display": "Allergic rhinitis"},
    # Medications (SNOMED product codes)
    "paracetamol": {"code": "387517004", "display": "Paracetamol"},
    "amoxicillin": {"code": "372756006", "display": "Amoxicillin"},
    "ibuprofen": {"code": "387207008", "display": "Ibuprofen"},
    "naproxen": {"code": "373254001", "display": "Naproxen"},
    "metformin": {"code": "109081006", "display": "Metformin"},
    "atorvastatin": {"code": "108600003", "display": "Atorvastatin"},
    "ramipril": {"code": "372913009", "display": "Ramipril"},
    "salbutamol": {"code": "372897005", "display": "Salbutamol"},
    "omeprazole": {"code": "372767007", "display": "Omeprazole"},
    "prednisolone": {"code": "373225007", "display": "Prednisolone"},
    "sertraline": {"code": "1003403007", "display": "Sertraline"},
}


def _lookup_snomed(entity_text: str) -> dict | None:
    text = entity_text.lower().strip()
    if text in _SNOMED_SYNONYMS:
        return _SNOMED_SYNONYMS[text]
    for term, entry in _SNOMED_SYNONYMS.items():
        if len(term) >= 4 and (term in text or text in term):
            return entry
    return None


# ── Medication extraction ──

_MED_PATTERN = re.compile(
    r"\b(paracetamol|ibuprofen|naproxen|amoxicillin|co-amoxiclav|"
    r"metformin|atorvastatin|amlodipine|ramipril|bisoprolol|"
    r"omeprazole|lansoprazole|salbutamol|prednisolone|"
    r"fluoxetine|sertraline|citalopram|amitriptyline|mirtazapine|"
    r"codeine|tramadol|morphine|diazepam|zopiclone|"
    r"aspirin|warfarin|rivaroxaban|apixaban|clopidogrel|"
    r"doxycycline|flucloxacillin|clarithromycin|trimethoprim|nitrofurantoin|"
    r"levothyroxine|insulin|methotrexate|"
    r"cetirizine|loratadine|chlorphenamine|"
    r"gabapentin|pregabalin|carbamazepine|"
    r"lisinopril|losartan|candesartan|"
    r"simvastatin|rosuvastatin|"
    r"bendroflumethiazide|furosemide|spironolactone|"
    r"azithromycin|erythromycin|penicillin|phenoxymethylpenicillin)\b",
    re.IGNORECASE,
)

_DOSAGE_PATTERN = re.compile(
    r"(\d+\s*(?:mg|mcg|ml|g|units?)(?:\s*/\s*\d+\s*(?:mg|ml))?)"
    r"(?:\s+(?:once|twice|three times|four times|every|daily|bd|tds|qds|od|prn|nocte|mane|stat))?",
    re.IGNORECASE,
)


def _extract_medications(text: str) -> list[str]:
    meds = set()
    for match in _MED_PATTERN.finditer(text):
        med = match.group(0).lower()
        after = text[match.end():match.end() + 40]
        dose_match = _DOSAGE_PATTERN.search(after)
        if dose_match:
            meds.add(f"{med} {dose_match.group(0).strip()}")
        else:
            meds.add(med)
    return sorted(meds)


# ── Section classification ──

def _classify_sections(transcript: str) -> dict[str, str]:
    hints: dict[str, str] = {}
    text_lower = transcript.lower()

    markers = {
        "subjective": [
            "what brings you in", "how can i help", "what's the problem",
            "i have been", "i've been", "i am having", "for the last",
            "for about", "started", "bothering me", "i feel", "i get",
        ],
        "objective": [
            "let me examine", "on examination", "i can see", "i can hear",
            "blood pressure", "heart rate", "temperature is", "oxygen",
            "straight leg raise", "tender", "swollen", "crackles",
            "wheeze", "murmur", "reflexes", "auscultation",
        ],
        "assessment": [
            "i think", "this is likely", "this looks like",
            "my assessment", "diagnosis", "consistent with",
            "probably", "suspect", "impression",
        ],
        "plan": [
            "prescribe", "refer", "come back", "follow up",
            "start you on", "i am going to", "i'll arrange",
            "blood test", "x-ray", "scan", "physiotherapy",
            "if it gets worse", "seek urgent",
        ],
    }

    for section, section_markers in markers.items():
        for marker in section_markers:
            idx = text_lower.find(marker)
            if idx >= 0:
                start = max(0, idx - 50)
                end = min(len(transcript), idx + 200)
                hints.setdefault(section, "")
                hints[section] += transcript[start:end] + " "

    return {k: v.strip() for k, v in hints.items()}


# ── Main pipeline ──

_NEG_PREFIXES = (
    "no ", "not ", "denies ", "denied ", "without ",
    "negative for ", "absence of ", "no evidence of ",
    "does not have ", "no history of ", "no known ",
    "rules out ", "ruled out ", "free of ", "never had ",
)


def run_nlp_pipeline(transcript: str) -> NLPResult:
    """Run the full NLP pipeline on a transcript.

    Uses scispaCy + negex if available, falls back to regex extraction.
    Returns structured medical context for the LLM.
    """
    result = NLPResult()
    nlp = _get_nlp()

    if nlp is not None:
        result = _run_spacy_pipeline(nlp, transcript, result)
    else:
        result = _run_regex_pipeline(transcript, result)

    result.medications = _extract_medications(transcript)
    result.section_hints = _classify_sections(transcript)

    logger.info(
        f"NLP pipeline: {len(result.entities)} entities, "
        f"{len(result.negated_entities)} negated, "
        f"{len(result.snomed_codes)} SNOMED codes, "
        f"{len(result.medications)} medications, "
        f"sections={list(result.section_hints.keys())}"
    )
    return result


def _run_spacy_pipeline(nlp, transcript: str, result: NLPResult) -> NLPResult:
    doc = nlp(transcript)
    transcript_lower = transcript.lower()
    seen_texts: set[str] = set()

    for ent in doc.ents:
        text = ent.text.strip()
        text_lower = text.lower()

        if len(text) < 3 or text_lower in seen_texts:
            continue
        if text_lower in ("patient", "doctor", "morning", "today", "yes", "thank"):
            continue
        seen_texts.add(text_lower)

        negated = getattr(ent._, "negex", False) if hasattr(ent, "_") else False
        if not negated:
            if any(text_lower.startswith(p) for p in _NEG_PREFIXES):
                negated = True
            else:
                ctx_start = max(0, ent.start_char - 40)
                context = transcript_lower[ctx_start:ent.start_char]
                if any(context.rstrip().endswith(p.rstrip()) for p in _NEG_PREFIXES):
                    negated = True

        clean_text = text
        for p in _NEG_PREFIXES:
            if text_lower.startswith(p):
                clean_text = text[len(p):].strip()
                break

        entity = MedicalEntity(
            text=clean_text, label=ent.label_, negated=negated,
            start=ent.start_char, end=ent.end_char,
        )

        match = _lookup_snomed(clean_text)
        if match:
            entity.snomed_code = match["code"]
            entity.snomed_display = match["display"]

        if negated:
            result.negated_entities.append(entity)
        else:
            result.entities.append(entity)
            if entity.snomed_code:
                result.conditions.append(entity)

    _collect_snomed_codes(result)
    return result


def _run_regex_pipeline(transcript: str, result: NLPResult) -> NLPResult:
    """Fallback when scispaCy is not installed."""
    transcript_lower = transcript.lower()
    seen: set[str] = set()

    for term, entry in _SNOMED_SYNONYMS.items():
        if term in transcript_lower and term not in seen:
            seen.add(term)

            negated = False
            idx = transcript_lower.find(term)
            if idx >= 0:
                ctx = transcript_lower[max(0, idx - 40):idx]
                if any(ctx.rstrip().endswith(p.rstrip()) for p in _NEG_PREFIXES):
                    negated = True

            entity = MedicalEntity(
                text=term, label="ENTITY", negated=negated,
                snomed_code=entry["code"], snomed_display=entry["display"],
                start=idx, end=idx + len(term),
            )

            if negated:
                result.negated_entities.append(entity)
            else:
                result.entities.append(entity)
                result.conditions.append(entity)

    _collect_snomed_codes(result)
    return result


def _collect_snomed_codes(result: NLPResult) -> None:
    seen: set[str] = set()
    for ent in result.conditions:
        if ent.snomed_code and ent.snomed_code not in seen:
            seen.add(ent.snomed_code)
            result.snomed_codes.append({
                "code": ent.snomed_code,
                "display": ent.snomed_display,
            })


def format_nlp_context(result: NLPResult) -> str:
    """Format NLP results as structured context for the LLM prompt."""
    parts = []

    if result.conditions:
        conditions = ", ".join(
            f"{e.snomed_display} [SNOMED: {e.snomed_code}]"
            for e in result.conditions[:10]
        )
        parts.append(f"Detected conditions: {conditions}")

    if result.negated_entities:
        negated = ", ".join(e.text for e in result.negated_entities[:5])
        parts.append(f"Negated/denied: {negated}")

    if result.medications:
        parts.append(f"Medications mentioned: {', '.join(result.medications[:10])}")

    if result.section_hints:
        for section, text in result.section_hints.items():
            parts.append(f"{section.upper()} hints: {text[:200]}")

    return "\n".join(parts) if parts else ""
