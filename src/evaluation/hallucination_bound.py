"""Streaming-veto hallucination bound (N17).

Implements the per-position and note-level bounds proved in dissertation
Chapter 5 §5.8 (sec:hallucination-bound):

    Per-position:  P[ship_fab] <= p*delta / (1 - p*(1-delta))
    Note-level:    P[any fab in note] <= N * p*delta / (1 - p*(1-delta))

Where:
    p       = per-attempt fabrication propensity of the SOAP model
    delta   = false-negative rate of the NLI gate at chosen threshold
    k       = retry budget (each rewrite is one trial)
    N       = max sentence positions per note

The module exposes:
    compute_bound(p, delta, k=3, N=30)        -> exact + asymptotic bound
    estimate_p_fab(notes, transcripts, gate)  -> empirical p estimate
    bound_grid(p_grid, delta_grid, ...)       -> heatmap of epsilon
    verify_bound(streaming_eval_json_path)    -> compare empirical vs bound

Pure-math helpers have no Ollama, no NLI, no Torch dependency. Empirical
helpers re-use the existing EntailmentGate from src.clinical.entailment_gate.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "data" / "evaluation_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Pure-math bound ──────────────────────────────────────────────────


@dataclass
class HallucinationBoundResult:
    """One evaluation of the bound at fixed (p, delta, k, N)."""
    p: float
    delta: float
    k: int
    N: int
    per_position_bound: float          # exact: p*delta * (1 - (p*(1-delta))^(k+1)) / (1 - p*(1-delta))
    per_position_asymptotic: float     # k -> infinity: p*delta / (1 - p*(1-delta))
    note_level_bound: float            # N * per_position_bound (union bound)
    note_level_asymptotic: float       # N * per_position_asymptotic
    is_meaningful: bool                # True if note_level_bound < 1.0

    def to_dict(self) -> dict:
        return asdict(self)


def compute_bound(p: float, delta: float, k: int = 3, N: int = 30) -> HallucinationBoundResult:
    """Compute per-position and note-level fabrication bounds.

    Args:
        p: Per-attempt fabrication propensity of the underlying SOAP model.
           In [0, 1]. Smaller is better.
        delta: False-negative rate of the NLI entailment gate at the chosen
           threshold. In [0, 1]. Smaller is better.
        k: Maximum retry budget. Default 3 matches StreamingConstitutionalGenerator.
        N: Maximum number of sentence positions per note. Default 30 matches
           max_sentences_per_case.

    Returns:
        HallucinationBoundResult with both finite-k and k -> infinity values.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0, 1], got {p}")
    if not (0.0 <= delta <= 1.0):
        raise ValueError(f"delta must be in [0, 1], got {delta}")
    if k < 0 or N < 0:
        raise ValueError(f"k and N must be non-negative, got k={k}, N={N}")

    denom = 1.0 - p * (1.0 - delta)
    if denom <= 0.0:
        # Degenerate case: every attempt is a retry and succeeds; geometric
        # series diverges. The placeholder give-up handles this in code; the
        # bound here is trivially 1.0 (no useful guarantee).
        per_pos = 1.0
        per_pos_asym = 1.0
    else:
        # Finite-k exact bound (geometric series with k+1 terms).
        per_pos = p * delta * (1.0 - (p * (1.0 - delta)) ** (k + 1)) / denom
        per_pos_asym = p * delta / denom
    note_bound = min(N * per_pos, 1.0)
    note_asym = min(N * per_pos_asym, 1.0)

    return HallucinationBoundResult(
        p=p, delta=delta, k=k, N=N,
        per_position_bound=per_pos,
        per_position_asymptotic=per_pos_asym,
        note_level_bound=note_bound,
        note_level_asymptotic=note_asym,
        is_meaningful=note_bound < 1.0,
    )


def bound_grid(
    p_grid: list[float],
    delta_grid: list[float],
    k: int = 3,
    N: int = 30,
) -> dict:
    """Sweep the bound across a (p, delta) grid. Returns a dict suitable for
    heatmap plotting and dissertation appendix."""
    matrix = np.zeros((len(p_grid), len(delta_grid)))
    for i, p in enumerate(p_grid):
        for j, d in enumerate(delta_grid):
            matrix[i, j] = compute_bound(p, d, k=k, N=N).note_level_bound
    return {
        "p_grid": list(p_grid),
        "delta_grid": list(delta_grid),
        "k": k,
        "N": N,
        "note_level_bound_matrix": matrix.tolist(),
    }


# ── Empirical estimators ─────────────────────────────────────────────


def estimate_p_fab_from_baseline(
    streaming_eval_json: dict,
    strategy: str = "baseline",
) -> tuple[float, dict]:
    """Estimate p_fab from a streaming_eval JSON output.

    Defines p_fab as the per-sentence NLI-failure rate observed when the
    veto is NOT in the loop (i.e., the baseline strategy). This is an
    empirical lower bound on the true generation propensity, since the
    baseline ships every sentence and we count the fraction the gate
    flags as unsupported.

    Returns (point_estimate, diagnostic_dict).
    """
    cases = [c for c in streaming_eval_json.get("cases", [])
             if c.get("strategy") == strategy]
    if not cases:
        return 0.0, {"reason": f"no cases with strategy={strategy}"}
    rates = [c.get("hallucination_rate", 0.0) for c in cases]
    return float(np.mean(rates)), {
        "n_cases": len(cases),
        "rates": rates,
        "mean": float(np.mean(rates)),
        "min": float(np.min(rates)),
        "max": float(np.max(rates)),
    }


def estimate_delta_default() -> tuple[float, dict]:
    """Default delta estimate when no calibration set is available.

    DeBERTa-v3-small NLI on clinical summarisation has not been
    calibrated against gold-standard clinician labels in this work; the
    Asgari paper reports clinician-annotated entailment but on a different
    SOAP corpus. We adopt a conservative default delta = 0.10 reflecting
    the assumption that the gate misses one in ten genuine fabrications.

    Replace with a calibrated estimate post-clinician-panel.
    """
    return 0.10, {
        "source": "default conservative estimate",
        "note": "calibrate against clinician labels post-panel",
    }


# ── Verifier: empirical vs theoretical ───────────────────────────────


def verify_bound(
    streaming_eval_json_path: str | Path,
    delta: float | None = None,
    k: int = 3,
    N: int = 30,
) -> dict:
    """Compare the theoretical bound against the empirical streaming
    fabrication rate from a streaming_eval JSON output.

    Returns a dict ready for serialisation that includes:
      - estimated p (from baseline strategy)
      - delta (from argument or default)
      - theoretical epsilon
      - empirical rate (from streaming strategy)
      - whether the bound holds (empirical <= theoretical)
    """
    path = Path(streaming_eval_json_path)
    data = json.loads(path.read_text())

    p_hat, p_diag = estimate_p_fab_from_baseline(data)
    if delta is None:
        delta, delta_diag = estimate_delta_default()
    else:
        delta_diag = {"source": "user-supplied"}

    bound = compute_bound(p_hat, delta, k=k, N=N)

    streaming_cases = [c for c in data.get("cases", [])
                       if c.get("strategy") == "streaming"]
    empirical = (
        float(np.mean([c.get("hallucination_rate", 0.0)
                       for c in streaming_cases]))
        if streaming_cases else 0.0
    )

    return {
        "source_file": str(path),
        "estimated_p": p_hat,
        "p_diagnostic": p_diag,
        "delta": delta,
        "delta_diagnostic": delta_diag,
        "k": k,
        "N": N,
        "theoretical_epsilon": bound.note_level_bound,
        "theoretical_epsilon_asymptotic": bound.note_level_asymptotic,
        "is_meaningful": bound.is_meaningful,
        "empirical_streaming_rate": empirical,
        "bound_holds": empirical <= bound.note_level_bound,
        "bound_record": bound.to_dict(),
    }


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--streaming-eval",
        help="Path to a streaming_eval real-mode JSON. Default: latest.",
        default=None,
    )
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--N", type=int, default=30)
    parser.add_argument("--grid", action="store_true",
                        help="Also emit a (p, delta) bound grid")
    args = parser.parse_args()

    if args.streaming_eval is None:
        candidates = sorted(RESULTS_DIR.glob("streaming_eval_real_*.json"))
        if not candidates:
            print("no streaming_eval real-mode output found; pass --streaming-eval")
            return 1
        path = candidates[-1]
    else:
        path = Path(args.streaming_eval)

    result = verify_bound(path, delta=args.delta, k=args.k, N=args.N)

    if args.grid:
        p_grid = [0.01, 0.05, 0.10, 0.20, 0.50, 0.90]
        delta_grid = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
        result["bound_grid"] = bound_grid(p_grid, delta_grid,
                                          k=args.k, N=args.N)

    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"hallucination_bound_{stamp}.json"
    out.write_text(json.dumps(result, indent=2))

    print("=" * 64)
    print(f"  estimated p:  {result['estimated_p']:.4f} "
          f"(from baseline strategy NLI fail rate)")
    print(f"  delta:        {result['delta']:.4f} "
          f"({result['delta_diagnostic']['source']})")
    print(f"  k = {result['k']}, N = {result['N']}")
    print(f"  theoretical epsilon (note-level):  "
          f"{result['theoretical_epsilon']:.4f}")
    print(f"  empirical streaming rate:          "
          f"{result['empirical_streaming_rate']:.4f}")
    print(f"  bound holds (empirical <= theoretical): "
          f"{result['bound_holds']}")
    print(f"  bound is meaningful (epsilon < 1):      "
          f"{result['is_meaningful']}")
    print("=" * 64)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
