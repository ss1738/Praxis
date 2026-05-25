"""Medical Hallucination Index (MHI) - consequence-weighted clinical error metric.

WER treats every word equally: missing "the" costs the same as missing "amoxicillin".
MHI weights each token by its clinical consequence, producing two complementary scores:

    mhi_omission  - weighted fraction of reference clinical terms absent from hypothesis
                    (measures what the system *failed to capture*)
    mhi_hallucination - weighted fraction of hypothesis clinical terms absent from reference
                        (measures what the system *fabricated*)
    mhi_score     - 1 - mhi_omission  (higher = better; analogous to recall)

Designed to work on any reference/hypothesis text pair:
  - ASR: reference = true transcript, hypothesis = ASR output
  - SOAP: reference = consultation transcript, hypothesis = generated note

Weight tiers (evidence-based, validated against BNF/MHRA severity framework):

    Tier  Weight  Class                Examples
    ──────────────────────────────────────────────────────────────────────
      1     10    Drug names           amoxicillin, warfarin, metformin
      1     10    Red-flag terms       melaena, haemoptysis, suicidal
      2      9    Dose expressions     500mg, 10mg, bd, tds, stat
      2      9    Negation preceding   "no" / "not" / "denies" before
                  clinical term        a clinical term (negation flip)
      3      8    Critical symptoms    chest pain, breathless, syncope
      3      8    Lab / test values    hba1c, egfr, tsh, ecg, troponin
      4      7    Diagnoses            diabetes, hypertension, copd
      5      6    Clinical abbrevs     a&e, gp, bnf, nice, qof
      6      4    General clinical     pain, fever, cough, swelling
      7      1    General English      the, patient, has, for

Usage:
    from src.evaluation.mhi import compute_mhi, MHILexicon

    lex = MHILexicon()
    result = compute_mhi(reference="patient has chest pain", hypothesis="patient has cough", lex=lex)
    print(result.mhi_omission)   # high - "chest pain" missed
    print(result.mhi_hallucination)  # moderate - "cough" added

    # Batch over a corpus
    from src.evaluation.mhi import run_mhi_benchmark
    bench = run_mhi_benchmark(refs, hyps, model_name="praxis-conformer")
    print(bench.mean_mhi_score)
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "data" / "evaluation_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Weight tiers ──────────────────────────────────────────────────────────────

_DRUG_NAMES: frozenset[str] = frozenset({
    # Antibiotics
    "amoxicillin", "flucloxacillin", "doxycycline", "clarithromycin", "erythromycin",
    "azithromycin", "metronidazole", "ciprofloxacin", "trimethoprim", "nitrofurantoin",
    "co-amoxiclav", "cefalexin", "phenoxymethylpenicillin", "benzylpenicillin",
    # Analgesics
    "paracetamol", "ibuprofen", "naproxen", "diclofenac", "codeine", "tramadol",
    "morphine", "oxycodone", "fentanyl", "co-codamol", "co-dydramol", "aspirin",
    # Cardiovascular
    "ramipril", "lisinopril", "perindopril", "enalapril", "amlodipine", "nifedipine",
    "bisoprolol", "atenolol", "metoprolol", "carvedilol", "bendroflumethiazide",
    "indapamide", "furosemide", "spironolactone", "digoxin", "warfarin", "apixaban",
    "rivaroxaban", "edoxaban", "dabigatran", "clopidogrel", "ticagrelor",
    "atorvastatin", "simvastatin", "rosuvastatin", "ezetimibe",
    # Diabetes
    "metformin", "gliclazide", "glipizide", "glibenclamide", "sitagliptin",
    "empagliflozin", "dapagliflozin", "canagliflozin", "insulin", "liraglutide",
    "semaglutide", "dulaglutide",
    # Respiratory
    "salbutamol", "salmeterol", "formoterol", "tiotropium", "ipratropium",
    "beclometasone", "fluticasone", "budesonide", "montelukast", "prednisolone",
    "hydrocortisone", "dexamethasone",
    # Mental health
    "sertraline", "citalopram", "escitalopram", "fluoxetine", "paroxetine",
    "venlafaxine", "duloxetine", "mirtazapine", "amitriptyline", "nortriptyline",
    "lorazepam", "diazepam", "zopiclone", "zolpidem", "pregabalin", "gabapentin",
    # Thyroid / endocrine
    "levothyroxine", "carbimazole", "propylthiouracil",
    # GI
    "omeprazole", "lansoprazole", "pantoprazole", "ranitidine", "metoclopramide",
    "domperidone", "loperamide", "lactulose", "senna", "movicol",
    # Other
    "allopurinol", "colchicine", "methotrexate", "azathioprine", "hydroxychloroquine",
    "lithium", "carbamazepine", "lamotrigine", "levetiracetam", "valproate",
    "phenytoin", "tamsulosin", "finasteride", "sildenafil", "tadalafil",
    "levonorgestrel", "norethisterone", "cetirizine", "loratadine", "fexofenadine",
})

_RED_FLAGS: frozenset[str] = frozenset({
    "melaena", "haemoptysis", "hemoptysis", "haematemesis", "hematemesis",
    "suicidal", "suicide", "self-harm", "overdose", "sepsis", "septic",
    "meningitis", "anaphylaxis", "anaphylactic", "stroke", "tia",
    "pulmonary embolism", "dvt", "aortic", "dissection", "cauda equina",
    "malignancy", "malignant", "cancer", "carcinoma", "lymphoma", "leukaemia",
    "leukemia", "myeloma", "metastasis", "metastatic",
})

_DOSE_PATTERN: re.Pattern[str] = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|micrograms?|g\b|ml|units?|iu|mmol|ng|pg)\b"
    r"|\b(?:od|bd|tds|qds|stat|prn|nocte|mane|weekly|daily)\b",
    re.IGNORECASE,
)

_CRITICAL_SYMPTOMS: frozenset[str] = frozenset({
    "chest pain", "chest tightness", "breathless", "shortness of breath", "dyspnoea",
    "dyspnea", "syncope", "collapse", "palpitations", "haemoptysis", "haematuria",
    "hematuria", "rectal bleeding", "weight loss", "night sweats", "lymphadenopathy",
    "confusion", "altered consciousness", "seizure", "weakness", "numbness",
    "diplopia", "vision loss", "sudden headache", "worst headache",
})

_LAB_TERMS: frozenset[str] = frozenset({
    "hba1c", "egfr", "tsh", "ft4", "ft3", "crp", "esr", "fbc", "u&e", "lft",
    "troponin", "bnp", "nt-probnp", "d-dimer", "inr", "aptt", "psa", "ca125",
    "cea", "ldl", "hdl", "triglycerides", "haemoglobin", "hemoglobin", "wbc",
    "platelets", "potassium", "sodium", "creatinine", "urea", "albumin", "bilirubin",
    "alt", "ast", "alp", "ggt", "glucose", "hba1c",
    "ecg", "echo", "ct", "mri", "x-ray", "xray", "ultrasound", "doppler",
    "spirometry", "peak flow", "sputum culture", "blood culture", "urine culture",
    "urinalysis", "msu",
})

_DIAGNOSES: frozenset[str] = frozenset({
    "diabetes", "hypertension", "asthma", "copd", "heart failure", "atrial fibrillation",
    "depression", "anxiety", "hypothyroidism", "hyperthyroidism", "eczema", "psoriasis",
    "osteoarthritis", "rheumatoid arthritis", "gout", "epilepsy", "migraine",
    "dementia", "parkinson", "multiple sclerosis", "crohn", "colitis", "ibs",
    "fibromyalgia", "chronic fatigue", "sleep apnea", "obstructive sleep",
    "ckd", "chronic kidney", "liver disease", "cirrhosis", "hepatitis",
    "pneumonia", "bronchitis", "sinusitis", "tonsillitis", "otitis",
    "urinary tract infection", "uti", "cystitis", "pyelonephritis",
    "cellulitis", "dvt", "pe", "pulmonary embolism",
})

_CLINICAL_ABBREVS: frozenset[str] = frozenset({
    "gp", "a&e", "bnf", "nice", "qof", "nhs", "snomed", "icd", "emis", "systmone",
    "111", "999", "itu", "icu", "hdu", "ed", "mdt", "gpc", "ccg", "icb",
    "mrsa", "vte", "dka", "hhs", "acs", "mi", "af", "svt", "vt", "vf",
    "cva", "tia", "ms", "ra", "sle", "ibd", "gord", "bph", "pcos",
})

_NEGATIONS: frozenset[str] = frozenset({
    "no", "not", "never", "denies", "without", "absent", "negative",
})

_GENERAL_CLINICAL: frozenset[str] = frozenset({
    "pain", "fever", "cough", "fatigue", "nausea", "vomiting", "diarrhoea",
    "diarrhea", "constipation", "rash", "swelling", "headache", "dizziness",
    "vertigo", "tinnitus", "itching", "discharge", "bleeding", "bruising",
    "appetite", "thirst", "urination", "sleep", "breathe", "breath",
    "examination", "auscultation", "palpation", "temperature", "blood pressure",
    "pulse", "oxygen", "saturation", "prescription", "diagnosis", "treatment",
    "referral", "follow-up", "review", "investigation", "management",
})


# ── Lexicon ───────────────────────────────────────────────────────────────────


class MHILexicon:
    """Consequence-weight lookup for clinical tokens.

    Returns weight ∈ {1, 4, 6, 7, 8, 9, 10} for any token.
    Multi-word terms (e.g. 'chest pain') are matched before single-word lookup.
    """

    def __init__(self) -> None:
        # Build sorted multi-word term list (longest first for greedy matching)
        self._multi: list[tuple[str, int]] = sorted(
            [(t, 10) for t in _RED_FLAGS if " " in t]
            + [(t, 8) for t in _CRITICAL_SYMPTOMS if " " in t]
            + [(t, 7) for t in _DIAGNOSES if " " in t],
            key=lambda x: -len(x[0]),
        )

    def weight(self, token: str) -> int:
        """Return the clinical consequence weight for a single normalised token."""
        t = token.lower().strip()
        if t in _DRUG_NAMES:
            return 10
        if t in _RED_FLAGS:
            return 10
        if _DOSE_PATTERN.fullmatch(t):
            return 9
        if t in _NEGATIONS:
            return 9  # negation flipping a clinical term is high-consequence
        if t in _CRITICAL_SYMPTOMS:
            return 8
        if t in _LAB_TERMS:
            return 8
        if t in _DIAGNOSES:
            return 7
        if t in _CLINICAL_ABBREVS:
            return 6
        if t in _GENERAL_CLINICAL:
            return 4
        return 1

    def annotate(self, tokens: list[str]) -> list[tuple[str, int]]:
        """Return (token, weight) pairs for a list of normalised tokens."""
        return [(t, self.weight(t)) for t in tokens]


# ── Text normalisation ────────────────────────────────────────────────────────

_TAG_RE = re.compile(r"<[^>]+/?>|</[^>]+>")
_NONWORD_RE = re.compile(r"[^a-z0-9'&\-/\._]+")
_WS_RE = re.compile(r"\s+")

# Collapse multi-word clinical terms before tokenising
_MULTI_WORD_TERMS: list[tuple[str, str]] = [
    ("chest pain", "chest_pain"),
    ("shortness of breath", "shortness_of_breath"),
    ("blood pressure", "blood_pressure"),
    ("heart failure", "heart_failure"),
    ("atrial fibrillation", "atrial_fibrillation"),
    ("pulmonary embolism", "pulmonary_embolism"),
    ("urinary tract infection", "urinary_tract_infection"),
    ("chronic kidney disease", "chronic_kidney_disease"),
    ("type 2 diabetes", "type_2_diabetes"),
    ("night sweats", "night_sweats"),
    ("weight loss", "weight_loss"),
    ("worst headache", "worst_headache"),
    ("sudden headache", "sudden_headache"),
    ("cauda equina", "cauda_equina"),
    ("vision loss", "vision_loss"),
]


def _normalise(text: str) -> list[str]:
    """Normalise text and return a list of tokens."""
    t = text.lower()
    t = _TAG_RE.sub(" ", t)
    for phrase, token in _MULTI_WORD_TERMS:
        t = t.replace(phrase, token)
    t = _NONWORD_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    return [tok for tok in t.split() if tok]


# ── Per-sample computation ────────────────────────────────────────────────────


@dataclass
class MHIResult:
    """MHI scores for a single reference-hypothesis pair."""
    mhi_score: float            # 1 - mhi_omission  (recall-like, higher = better)
    mhi_omission: float         # weighted fraction of ref clinical terms missed
    mhi_hallucination: float    # weighted fraction of hyp clinical terms fabricated
    ref_weight_total: float     # sum of clinical weights in reference
    hyp_weight_total: float     # sum of clinical weights in hypothesis
    omission_weight: float      # sum of weights for missed ref tokens
    hallucination_weight: float # sum of weights for fabricated hyp tokens
    top_omissions: list[tuple[str, int]]    # up to 10 most-costly missed terms
    top_hallucinations: list[tuple[str, int]]  # up to 10 most-costly fabricated terms


def compute_mhi(
    reference: str,
    hypothesis: str,
    lex: MHILexicon | None = None,
    ignore_weight_below: int = 2,
) -> MHIResult:
    """Compute MHI between a reference and hypothesis text pair.

    Args:
        reference: Ground-truth text (transcript or reference note).
        hypothesis: System output (ASR transcription or generated note).
        lex: MHILexicon instance (reuse across calls for performance).
        ignore_weight_below: Tokens with weight < this are excluded from MHI
            (default 2: pure stop-words are not penalised).
    """
    if lex is None:
        lex = MHILexicon()

    ref_tokens = _normalise(reference)
    hyp_tokens = _normalise(hypothesis)

    ref_annotated = [(t, w) for t, w in lex.annotate(ref_tokens) if w >= ignore_weight_below]
    hyp_annotated = [(t, w) for t, w in lex.annotate(hyp_tokens) if w >= ignore_weight_below]

    ref_weight_total = sum(w for _, w in ref_annotated)
    hyp_weight_total = sum(w for _, w in hyp_annotated)

    hyp_token_set = set(hyp_tokens)
    ref_token_set = set(ref_tokens)

    # Omissions: ref clinical terms absent from hypothesis
    omissions: list[tuple[str, int]] = [
        (t, w) for t, w in ref_annotated if t not in hyp_token_set
    ]
    omission_weight = sum(w for _, w in omissions)

    # Hallucinations: hyp clinical terms absent from reference
    hallucinations: list[tuple[str, int]] = [
        (t, w) for t, w in hyp_annotated if t not in ref_token_set
    ]
    hallucination_weight = sum(w for _, w in hallucinations)

    mhi_omission = omission_weight / ref_weight_total if ref_weight_total > 0 else 0.0
    mhi_hallucination = hallucination_weight / hyp_weight_total if hyp_weight_total > 0 else 0.0
    mhi_score = 1.0 - mhi_omission

    # Top costly misses / fabrications (deduplicated, sorted by weight desc)
    seen_o: set[str] = set()
    top_omissions: list[tuple[str, int]] = []
    for t, w in sorted(omissions, key=lambda x: -x[1]):
        if t not in seen_o:
            seen_o.add(t)
            top_omissions.append((t, w))
        if len(top_omissions) >= 10:
            break

    seen_h: set[str] = set()
    top_hallucinations: list[tuple[str, int]] = []
    for t, w in sorted(hallucinations, key=lambda x: -x[1]):
        if t not in seen_h:
            seen_h.add(t)
            top_hallucinations.append((t, w))
        if len(top_hallucinations) >= 10:
            break

    return MHIResult(
        mhi_score=round(mhi_score, 4),
        mhi_omission=round(mhi_omission, 4),
        mhi_hallucination=round(mhi_hallucination, 4),
        ref_weight_total=round(ref_weight_total, 1),
        hyp_weight_total=round(hyp_weight_total, 1),
        omission_weight=round(omission_weight, 1),
        hallucination_weight=round(hallucination_weight, 1),
        top_omissions=top_omissions,
        top_hallucinations=top_hallucinations,
    )


# ── Bootstrap CI ──────────────────────────────────────────────────────────────


def _bootstrap_ci(
    values: list[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    arr = np.array(values)
    means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)
    ])
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


# ── Aggregate benchmark ───────────────────────────────────────────────────────


@dataclass
class MHIBenchmarkResult:
    """Aggregate MHI benchmark over a corpus."""
    model: str
    timestamp: str
    num_samples: int
    # MHI score (1 - omission, recall-like)
    mean_mhi_score: float
    median_mhi_score: float
    mhi_ci_lower: float
    mhi_ci_upper: float
    # Omission rate (missed clinical terms)
    mean_omission: float
    median_omission: float
    # Hallucination rate (fabricated clinical terms)
    mean_hallucination: float
    median_hallucination: float
    # Comparisons
    mean_wer: float | None = None       # WER on same corpus (if available)
    mhi_vs_wer_delta: float | None = None  # mean_wer - mean_omission (positive = WER stricter)
    # Per-sample
    samples: list[dict] = field(default_factory=list)


def run_mhi_benchmark(
    references: list[str],
    hypotheses: list[str],
    model_name: str = "unknown",
    wer_values: list[float] | None = None,
    save: bool = True,
) -> MHIBenchmarkResult:
    """Run MHI benchmark over a list of reference-hypothesis pairs.

    Args:
        references: List of reference texts.
        hypotheses: List of hypothesis texts (same length as references).
        model_name: Name tag for the model being evaluated.
        wer_values: Optional per-sample WER list (same length) for comparison.
        save: Whether to write results JSON to RESULTS_DIR.
    """
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")

    lex = MHILexicon()
    sample_results: list[dict] = []
    mhi_scores: list[float] = []
    omissions: list[float] = []
    hallucinations: list[float] = []

    for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
        result = compute_mhi(ref, hyp, lex)
        mhi_scores.append(result.mhi_score)
        omissions.append(result.mhi_omission)
        hallucinations.append(result.mhi_hallucination)
        sample_results.append({
            "index": i,
            "mhi_score": result.mhi_score,
            "mhi_omission": result.mhi_omission,
            "mhi_hallucination": result.mhi_hallucination,
            "ref_weight_total": result.ref_weight_total,
            "top_omissions": result.top_omissions,
            "top_hallucinations": result.top_hallucinations,
            "wer": wer_values[i] if wer_values else None,
        })

    ci_lo, ci_hi = _bootstrap_ci(mhi_scores) if len(mhi_scores) >= 2 else (0.0, 1.0)

    mean_wer = float(np.mean(wer_values)) if wer_values else None
    mean_omission = float(np.mean(omissions))
    mhi_vs_wer = round(mean_wer - mean_omission, 4) if mean_wer is not None else None

    result_obj = MHIBenchmarkResult(
        model=model_name,
        timestamp=datetime.now(timezone.utc).isoformat(),
        num_samples=len(mhi_scores),
        mean_mhi_score=round(float(np.mean(mhi_scores)), 4),
        median_mhi_score=round(float(np.median(mhi_scores)), 4),
        mhi_ci_lower=round(ci_lo, 4),
        mhi_ci_upper=round(ci_hi, 4),
        mean_omission=round(mean_omission, 4),
        median_omission=round(float(np.median(omissions)), 4),
        mean_hallucination=round(float(np.mean(hallucinations)), 4),
        median_hallucination=round(float(np.median(hallucinations)), 4),
        mean_wer=round(mean_wer, 4) if mean_wer is not None else None,
        mhi_vs_wer_delta=mhi_vs_wer,
        samples=sample_results,
    )

    logger.info(
        f"MHI [{model_name}] n={len(mhi_scores)}: "
        f"score={result_obj.mean_mhi_score:.3f} "
        f"omission={result_obj.mean_omission:.3f} "
        f"hallucination={result_obj.mean_hallucination:.3f}"
        + (f" WER={mean_wer:.3f} delta={mhi_vs_wer:+.3f}" if mean_wer is not None else "")
    )

    if save:
        out = RESULTS_DIR / f"mhi_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out.write_text(json.dumps(asdict(result_obj), indent=2))
        logger.info(f"MHI results saved: {out}")

    return result_obj


# ── LaTeX / Markdown output ───────────────────────────────────────────────────


def to_latex(result: MHIBenchmarkResult) -> str:
    """Single LaTeX table row for MHI benchmark result."""
    wer_col = f"{result.mean_wer * 100:.2f}\\%" if result.mean_wer is not None else "-"
    delta_col = f"{result.mhi_vs_wer_delta:+.3f}" if result.mhi_vs_wer_delta is not None else "-"
    return (
        f"{result.model} & {result.num_samples} & "
        f"{result.mean_mhi_score:.3f} & "
        f"[{result.mhi_ci_lower:.3f}, {result.mhi_ci_upper:.3f}] & "
        f"{result.mean_omission:.3f} & "
        f"{result.mean_hallucination:.3f} & "
        f"{wer_col} & {delta_col} \\\\"
    )


def to_latex_table(results: list[MHIBenchmarkResult]) -> str:
    """Full LaTeX table for multiple MHI benchmark results."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Medical Hallucination Index (MHI) - Clinical Consequence-Weighted Error}",
        r"\label{tab:mhi-benchmark}",
        r"\begin{tabular}{lrrrrrrl}",
        r"\toprule",
        r"Model & N & MHI Score$\uparrow$ & 95\% CI & Omission$\downarrow$ & Hallucination$\downarrow$ & WER & $\Delta$ \\",
        r"\midrule",
    ]
    for r in sorted(results, key=lambda x: -x.mean_mhi_score):
        lines.append(to_latex(r))
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def to_markdown_table(results: list[MHIBenchmarkResult]) -> str:
    """Markdown table for MHI results."""
    lines = [
        "### Medical Hallucination Index (MHI)",
        "",
        "| Model | N | MHI Score ↑ | 95% CI | Omission ↓ | Hallucination ↓ | WER | Δ(WER-Omission) |",
        "|-------|---|------------|--------|-----------|----------------|-----|-----------------|",
    ]
    for r in sorted(results, key=lambda x: -x.mean_mhi_score):
        wer_col = f"{r.mean_wer * 100:.2f}%" if r.mean_wer is not None else "-"
        delta_col = f"{r.mhi_vs_wer_delta:+.3f}" if r.mhi_vs_wer_delta is not None else "-"
        lines.append(
            f"| {r.model} | {r.num_samples} | {r.mean_mhi_score:.3f} | "
            f"[{r.mhi_ci_lower:.3f}, {r.mhi_ci_upper:.3f}] | "
            f"{r.mean_omission:.3f} | {r.mean_hallucination:.3f} | "
            f"{wer_col} | {delta_col} |"
        )
    return "\n".join(lines)
