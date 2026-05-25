"""GP blind-scoring system for dissertation evaluation.

Real GPs score SOAP notes on standardised criteria. Every score is
recorded, every inter-rater statistic is computable. No shortcuts.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger


# ---------------------------------------------------------------------------
# Scoring criteria dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScoringCriteria:
    """A single GP's assessment of one SOAP note (1-5 Likert scale)."""

    clinical_accuracy: int = 0       # Are the documented findings correct?
    completeness: int = 0            # Is anything clinically important missing?
    structure_quality: int = 0       # Is the note well-organised?
    language_appropriateness: int = 0  # UK English, professional tone?
    safety_netting: int = 0          # Are safety-netting instructions appropriate?
    coding_quality: int = 0          # Are SNOMED codes relevant and correct?
    overall_acceptability: int = 0   # Would you sign this note as-is?
    comments: str = ""               # Free text feedback
    would_edit: bool = False         # Would you need to edit before signing?
    critical_errors: list[str] = field(default_factory=list)  # Dangerous omissions/errors

    LIKERT_FIELDS: list[str] = field(
        default=None, init=False, repr=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "LIKERT_FIELDS", [
            "clinical_accuracy",
            "completeness",
            "structure_quality",
            "language_appropriateness",
            "safety_netting",
            "coding_quality",
            "overall_acceptability",
        ])
        for f in self.LIKERT_FIELDS:
            val = getattr(self, f)
            if val != 0 and not (1 <= val <= 5):
                raise ValueError(f"{f} must be between 1 and 5, got {val}")

    def to_dict(self) -> dict:
        return {
            "clinical_accuracy": self.clinical_accuracy,
            "completeness": self.completeness,
            "structure_quality": self.structure_quality,
            "language_appropriateness": self.language_appropriateness,
            "safety_netting": self.safety_netting,
            "coding_quality": self.coding_quality,
            "overall_acceptability": self.overall_acceptability,
            "comments": self.comments,
            "would_edit": self.would_edit,
            "critical_errors": self.critical_errors,
        }


# ---------------------------------------------------------------------------
# Scoring session
# ---------------------------------------------------------------------------

class GPScoringSession:
    """Manages one GP's scoring of multiple SOAP notes."""

    def __init__(self, gp_id: str, output_dir: str | Path = "data/gp_scores") -> None:
        self.gp_id = gp_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scored_notes: list[dict] = []
        logger.info(f"GP scoring session started: gp_id={gp_id}")

    def score_note(
        self,
        session_id: str,
        soap_text: str,
        model_id: str,
        transcript: str,
        scores: ScoringCriteria,
    ) -> dict:
        """Record a GP's score for one SOAP note."""
        record = {
            "session_id": session_id,
            "model_id": model_id,
            "soap_text": soap_text,
            "transcript_preview": transcript[:500],
            "scores": scores.to_dict(),
            "scored_at": datetime.now(timezone.utc).isoformat(),
            "gp_id": self.gp_id,
        }
        self.scored_notes.append(record)
        logger.info(
            f"Note scored: session={session_id}, gp={self.gp_id}, "
            f"overall={scores.overall_acceptability}"
        )
        return record

    def compute_averages(self) -> dict:
        """Compute mean score for each Likert criterion across all rated notes."""
        if not self.scored_notes:
            return {}

        likert_fields = [
            "clinical_accuracy",
            "completeness",
            "structure_quality",
            "language_appropriateness",
            "safety_netting",
            "coding_quality",
            "overall_acceptability",
        ]

        totals = {f: 0.0 for f in likert_fields}
        count = len(self.scored_notes)

        for record in self.scored_notes:
            for f in likert_fields:
                totals[f] += record["scores"][f]

        averages = {f: round(totals[f] / count, 2) for f in likert_fields}
        averages["notes_scored"] = count
        averages["would_edit_pct"] = round(
            sum(1 for r in self.scored_notes if r["scores"]["would_edit"]) / count * 100, 1
        )
        averages["critical_error_count"] = sum(
            len(r["scores"]["critical_errors"]) for r in self.scored_notes
        )
        return averages

    def export_scores(self) -> Path:
        """Save all scores to a JSON file, keyed by GP ID."""
        filepath = self.output_dir / f"gp_{self.gp_id}_scores.json"
        payload = {
            "gp_id": self.gp_id,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "notes_scored": len(self.scored_notes),
            "averages": self.compute_averages(),
            "scores": self.scored_notes,
        }
        filepath.write_text(json.dumps(payload, indent=2))
        logger.info(f"Scores exported: {filepath} ({len(self.scored_notes)} notes)")
        return filepath


# ---------------------------------------------------------------------------
# Inter-rater reliability (Cohen's kappa)
# ---------------------------------------------------------------------------

def _manual_cohen_kappa(
    ratings1: list[int],
    ratings2: list[int],
    weights: str = "quadratic",
    min_val: int = 1,
    max_val: int = 5,
) -> float:
    """Compute weighted Cohen's kappa without sklearn.

    Uses quadratic weights by default, appropriate for ordinal Likert data.
    """
    n = len(ratings1)
    if n == 0:
        return 0.0

    num_categories = max_val - min_val + 1

    # Build observed agreement matrix
    observed = [[0] * num_categories for _ in range(num_categories)]
    for r1, r2 in zip(ratings1, ratings2):
        i = r1 - min_val
        j = r2 - min_val
        observed[i][j] += 1

    # Row and column marginals
    row_sums = [sum(observed[i]) for i in range(num_categories)]
    col_sums = [sum(observed[i][j] for i in range(num_categories)) for j in range(num_categories)]

    # Build weight matrix
    w = [[0.0] * num_categories for _ in range(num_categories)]
    for i in range(num_categories):
        for j in range(num_categories):
            if weights == "quadratic":
                w[i][j] = ((i - j) ** 2) / ((num_categories - 1) ** 2)
            else:
                w[i][j] = abs(i - j) / (num_categories - 1)

    # Observed and expected disagreement
    obs_disagreement = 0.0
    exp_disagreement = 0.0
    for i in range(num_categories):
        for j in range(num_categories):
            obs_disagreement += w[i][j] * observed[i][j] / n
            exp_disagreement += w[i][j] * row_sums[i] * col_sums[j] / (n * n)

    if exp_disagreement == 0.0:
        return 1.0

    return round(1.0 - (obs_disagreement / exp_disagreement), 4)


def compute_cohens_kappa(scorer1_file: Path, scorer2_file: Path) -> dict:
    """Compute Cohen's kappa between two GP scoring files.

    Loads both files, matches by session_id, and computes weighted kappa
    (quadratic weights) for each Likert criterion.

    Returns a dict with kappa per criterion and an overall (mean) kappa.
    """
    scorer1_file = Path(scorer1_file)
    scorer2_file = Path(scorer2_file)

    data1 = json.loads(scorer1_file.read_text())
    data2 = json.loads(scorer2_file.read_text())

    # Index by session_id for fast lookup
    scores1 = {r["session_id"]: r["scores"] for r in data1["scores"]}
    scores2 = {r["session_id"]: r["scores"] for r in data2["scores"]}

    common_ids = sorted(set(scores1.keys()) & set(scores2.keys()))
    if not common_ids:
        logger.warning("No overlapping session IDs between the two scorers.")
        return {"error": "no_overlap", "common_sessions": 0}

    likert_fields = [
        "clinical_accuracy",
        "completeness",
        "structure_quality",
        "language_appropriateness",
        "safety_netting",
        "coding_quality",
        "overall_acceptability",
    ]

    kappas: dict[str, float] = {}

    for field_name in likert_fields:
        r1 = [scores1[sid][field_name] for sid in common_ids]
        r2 = [scores2[sid][field_name] for sid in common_ids]

        # Prefer sklearn if available
        try:
            from sklearn.metrics import cohen_kappa_score
            kappa = round(cohen_kappa_score(r1, r2, weights="quadratic"), 4)
        except ImportError:
            kappa = _manual_cohen_kappa(r1, r2, weights="quadratic")

        kappas[field_name] = kappa

    overall_kappa = round(sum(kappas.values()) / len(kappas), 4)

    result = {
        "scorer1": data1["gp_id"],
        "scorer2": data2["gp_id"],
        "common_sessions": len(common_ids),
        "kappa_per_criterion": kappas,
        "overall_kappa": overall_kappa,
        "weights": "quadratic",
        "interpretation": _interpret_kappa(overall_kappa),
    }

    logger.info(
        f"Inter-rater kappa: {result['scorer1']} vs {result['scorer2']} = "
        f"{overall_kappa} ({result['interpretation']})"
    )
    return result


def _interpret_kappa(kappa: float) -> str:
    """Landis & Koch (1977) interpretation of kappa values."""
    if kappa < 0.0:
        return "poor"
    elif kappa < 0.21:
        return "slight"
    elif kappa < 0.41:
        return "fair"
    elif kappa < 0.61:
        return "moderate"
    elif kappa < 0.81:
        return "substantial"
    else:
        return "almost_perfect"
