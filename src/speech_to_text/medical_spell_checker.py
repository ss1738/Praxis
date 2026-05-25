"""Fix common medical spelling errors in ASR output using fuzzy matching."""

from __future__ import annotations

import re

from loguru import logger
from rapidfuzz import fuzz, process

# top UK prescriptions from BNF
UK_DRUG_NAMES: list[str] = [
    "amoxicillin", "paracetamol", "ibuprofen", "omeprazole", "lansoprazole",
    "metformin", "atorvastatin", "simvastatin", "amlodipine", "ramipril",
    "bisoprolol", "levothyroxine", "salbutamol", "sertraline", "citalopram",
    "fluoxetine", "codeine", "tramadol", "naproxen", "doxycycline",
    "prednisolone", "warfarin", "apixaban", "rivaroxaban", "methotrexate",
    "adalimumab", "infliximab", "diazepam", "lorazepam", "gabapentin",
    "pregabalin", "co-amoxiclav", "cefalexin", "ciprofloxacin",
    "azithromycin", "clarithromycin", "trimethoprim", "nitrofurantoin",
    "mirtazapine", "venlafaxine", "duloxetine", "insulin", "gliclazide",
    "empagliflozin", "dapagliflozin", "semaglutide", "liraglutide",
    "tiotropium", "budesonide", "beclometasone", "fluticasone",
]

# spoken form -> written form
NHS_ABBREVIATIONS: dict[str, str] = {
    "a and e": "A&E",
    "gp": "GP",
    "nhs": "NHS",
    "bmj": "BMJ",
    "ecg": "ECG",
    "eeg": "EEG",
    "ct scan": "CT scan",
    "mri": "MRI",
    "bp": "BP",
    "bmi": "BMI",
    "copd": "COPD",
    "dvt": "DVT",
    "pe": "PE",
    "uti": "UTI",
    "ent": "ENT",
    "obs and gynae": "Obs & Gynae",
    "hba1c": "HbA1c",
    "crp": "CRP",
    "fbc": "FBC",
    "lfts": "LFTs",
    "u and es": "U&Es",
    "tfts": "TFTs",
    "prn": "PRN",
    "od": "OD",
    "bd": "BD",
    "tds": "TDS",
    "qds": "QDS",
}


class MedicalSpellChecker:
    """Fix drug names and NHS abbreviations in ASR output."""

    def __init__(self, drug_score_threshold: float = 85.0):
        self._drug_threshold = drug_score_threshold

    def correct(self, text: str) -> str:
        text = self._fix_abbreviations(text)
        text = self._fix_drug_names(text)
        logger.debug(f"Spell-checked: {text[:80]}...")
        return text

    def _fix_abbreviations(self, text: str) -> str:
        for spoken, canonical in NHS_ABBREVIATIONS.items():
            pattern = re.compile(re.escape(spoken), re.IGNORECASE)
            text = pattern.sub(canonical, text)
        return text

    def _fix_drug_names(self, text: str) -> str:
        # fuzzy match words against drug list
        words = text.split()
        corrected: list[str] = []
        for word in words:
            clean = word.strip(".,;:!?")
            if len(clean) < 5:
                corrected.append(word)
                continue
            match = process.extractOne(
                clean.lower(), UK_DRUG_NAMES, scorer=fuzz.ratio
            )
            if match and match[1] >= self._drug_threshold:
                replacement = match[0]
                corrected.append(word.replace(clean, replacement))
            else:
                corrected.append(word)
        return " ".join(corrected)
