"""Generate publication-ready LaTeX and Markdown tables from evaluation results.

Reads all JSON result files from data/evaluation_results/ and produces:
    - LaTeX tables for the dissertation (data/evaluation_results/tables.tex)
    - Markdown tables for the README (data/evaluation_results/tables.md)

Usage:
    python3 -m src.evaluation.generate_tables
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "data" / "evaluation_results"


def _load_latest(prefix: str) -> dict | None:
    """Load the most recent result file matching a prefix."""
    files = sorted(RESULTS_DIR.glob(f"{prefix}*.json"), reverse=True)
    if not files:
        return None
    return json.loads(files[0].read_text())


def _wer_latex(results: list[dict]) -> str:
    """Generate WER comparison LaTeX table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Word Error Rate (WER) on PriMock57 Test Set (N=14)}",
        r"\label{tab:wer-benchmark}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model & Mean WER (\%) & Median WER (\%) & 95\% CI & RTF \\",
        r"\midrule",
    ]
    for r in sorted(results, key=lambda x: x["mean_wer"]):
        lines.append(
            f"  {r['model']} & {r['mean_wer']*100:.2f} & {r['median_wer']*100:.2f} & "
            f"[{r['ci_lower']*100:.2f}, {r['ci_upper']*100:.2f}] & {r['mean_rtf']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def _wer_markdown(results: list[dict]) -> str:
    """Generate WER comparison Markdown table."""
    lines = [
        "### ASR Word Error Rate (WER) - PriMock57 Test Set",
        "",
        "| Model | Mean WER | Median WER | 95% CI | RTF |",
        "|-------|----------|------------|--------|-----|",
    ]
    for r in sorted(results, key=lambda x: x["mean_wer"]):
        lines.append(
            f"| {r['model']} | {r['mean_wer']*100:.2f}% | {r['median_wer']*100:.2f}% | "
            f"[{r['ci_lower']*100:.2f}, {r['ci_upper']*100:.2f}] | {r['mean_rtf']:.3f} |"
        )
    return "\n".join(lines)


def _soap_latex(results: list[dict]) -> str:
    """Generate SOAP quality LaTeX table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{SOAP Note Quality Evaluation}",
        r"\label{tab:soap-quality}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Model & N & Completeness & SNOMED & Med. Recall & Safety Net & TrustScore \\",
        r"\midrule",
    ]
    for r in results:
        lines.append(
            f"  {r['model']} & {r['num_notes']} & "
            f"{r['mean_section_completeness']*100:.0f}\\% & "
            f"{r['mean_snomed_count']:.1f} & "
            f"{r['mean_medication_recall']*100:.0f}\\% & "
            f"{r['safety_netting_rate']*100:.0f}\\% & "
            f"{r['mean_trust_score']:.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def _soap_markdown(results: list[dict]) -> str:
    """Generate SOAP quality Markdown table."""
    lines = [
        "### SOAP Note Quality",
        "",
        "| Model | N | Completeness | SNOMED Codes | Med. Recall | Safety Net | TrustScore |",
        "|-------|---|-------------|-------------|-------------|------------|------------|",
    ]
    for r in results:
        lines.append(
            f"| {r['model']} | {r['num_notes']} | "
            f"{r['mean_section_completeness']*100:.0f}% | "
            f"{r['mean_snomed_count']:.1f} | "
            f"{r['mean_medication_recall']*100:.0f}% | "
            f"{r['safety_netting_rate']*100:.0f}% | "
            f"{r['mean_trust_score']:.1f} |"
        )
    return "\n".join(lines)


def _der_latex(results: list[dict]) -> str:
    """Generate speaker diarization LaTeX table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Speaker Diarization on PriMock57 (2-speaker GP consultations)}",
        r"\label{tab:der-benchmark}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model & Params & Speaker Acc.\ (\%) & Mean RTF & 95\%~CI RTF \\",
        r"\midrule",
    ]
    for r in sorted(results, key=lambda x: -x["speaker_count_accuracy"]):
        lines.append(
            f"  {r['model']} & {r.get('params', '-')} & "
            f"{r['speaker_count_accuracy']*100:.1f} & "
            f"{r['mean_rtf']:.4f} & "
            f"[{r['rtf_ci_lower']:.4f}, {r['rtf_ci_upper']:.4f}] \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def _der_markdown(results: list[dict]) -> str:
    """Generate speaker diarization Markdown table."""
    lines = [
        "### Speaker Diarization",
        "",
        "| Model | Params | Speaker Acc. | Mean RTF | 95% CI RTF |",
        "|-------|--------|-------------|----------|------------|",
    ]
    for r in sorted(results, key=lambda x: -x["speaker_count_accuracy"]):
        lines.append(
            f"| {r['model']} | {r.get('params', '-')} | "
            f"{r['speaker_count_accuracy']*100:.1f}% | "
            f"{r['mean_rtf']:.4f} | "
            f"[{r['rtf_ci_lower']:.4f}, {r['rtf_ci_upper']:.4f}] |"
        )
    return "\n".join(lines)


def generate_all() -> tuple[str, str]:
    """Generate all tables from existing evaluation results.

    Returns (latex_content, markdown_content).
    """
    latex_parts = [
        f"% PRAXIS Evaluation Tables - auto-generated {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"% Do not edit manually. Regenerate with: python3 -m src.evaluation.generate_tables",
        "",
    ]
    md_parts = [
        f"## PRAXIS Benchmark Results",
        f"",
        f"*Auto-generated {datetime.now().strftime('%Y-%m-%d %H:%M')}. "
        f"Regenerate with `make eval-tables`.*",
        "",
    ]

    # WER results
    wer_results = []
    for f in sorted(RESULTS_DIR.glob("wer_*.json")):
        if "comparison" in f.name:
            continue
        try:
            data = json.loads(f.read_text())
            if "mean_wer" in data:
                wer_results.append(data)
        except Exception:
            continue

    if wer_results:
        # Deduplicate: keep latest per model
        latest_by_model: dict[str, dict] = {}
        for r in wer_results:
            model = r["model"]
            if model not in latest_by_model or r["timestamp"] > latest_by_model[model]["timestamp"]:
                latest_by_model[model] = r
        wer_results = list(latest_by_model.values())

        latex_parts.append(_wer_latex(wer_results))
        latex_parts.append("")
        md_parts.append(_wer_markdown(wer_results))
        md_parts.append("")
    else:
        md_parts.append("### ASR WER Benchmark")
        md_parts.append("")
        md_parts.append("*Pending - run `make benchmark-asr` to generate.*")
        md_parts.append("")

    # DER results
    der_results = []
    for f in sorted(RESULTS_DIR.glob("der_*.json")):
        try:
            data = json.loads(f.read_text())
            if "speaker_count_accuracy" in data:
                der_results.append(data)
        except Exception:
            continue

    if der_results:
        latest_der: dict[str, dict] = {}
        for r in der_results:
            model = r["model"]
            if model not in latest_der or r["timestamp"] > latest_der[model]["timestamp"]:
                latest_der[model] = r
        der_results = list(latest_der.values())
        latex_parts.append(_der_latex(der_results))
        latex_parts.append("")
        md_parts.append(_der_markdown(der_results))
        md_parts.append("")
    else:
        md_parts.append("### Speaker Diarization (DER)")
        md_parts.append("")
        md_parts.append("*Pending - run `make benchmark-der` to generate.*")
        md_parts.append("")

    # SOAP quality results
    soap_results = []
    for f in sorted(RESULTS_DIR.glob("soap_*.json")):
        try:
            data = json.loads(f.read_text())
            if "mean_section_completeness" in data:
                soap_results.append(data)
        except Exception:
            continue

    if soap_results:
        latest_by_model = {}
        for r in soap_results:
            model = r["model"]
            if model not in latest_by_model or r["timestamp"] > latest_by_model[model]["timestamp"]:
                latest_by_model[model] = r
        soap_results = list(latest_by_model.values())

        latex_parts.append(_soap_latex(soap_results))
        latex_parts.append("")
        md_parts.append(_soap_markdown(soap_results))
        md_parts.append("")
    else:
        md_parts.append("### SOAP Note Quality")
        md_parts.append("")
        md_parts.append("*Pending - run `make benchmark-soap` to generate.*")
        md_parts.append("")

    # MHI results
    from src.evaluation.mhi import MHIBenchmarkResult, to_latex_table as mhi_latex_table, to_markdown_table as mhi_md_table
    mhi_results: list[MHIBenchmarkResult] = []
    for f in sorted(RESULTS_DIR.glob("mhi_*.json")):
        try:
            data = json.loads(f.read_text())
            if "mean_mhi_score" in data:
                r = MHIBenchmarkResult(**{
                    k: v for k, v in data.items()
                    if k in MHIBenchmarkResult.__dataclass_fields__
                })
                mhi_results.append(r)
        except Exception:
            continue

    if mhi_results:
        # Deduplicate: latest per model
        latest: dict[str, MHIBenchmarkResult] = {}
        for r in mhi_results:
            if r.model not in latest or r.timestamp > latest[r.model].timestamp:
                latest[r.model] = r
        mhi_results = list(latest.values())
        latex_parts.append(mhi_latex_table(mhi_results))
        latex_parts.append("")
        md_parts.append(mhi_md_table(mhi_results))
        md_parts.append("")
    else:
        md_parts.append("### Medical Hallucination Index (MHI)")
        md_parts.append("")
        md_parts.append("*Pending - run `make benchmark-asr` to generate.*")
        md_parts.append("")

    # Constitutional eval results
    from src.evaluation.constitutional_eval import (
        ConstitutionalBenchmarkResult,
        to_latex_table as const_latex_table,
        to_markdown_table as const_md_table,
    )
    const_results: list[ConstitutionalBenchmarkResult] = []
    for f in sorted(RESULTS_DIR.glob("constitutional_eval_*.json")):
        try:
            data = json.loads(f.read_text())
            if "cases" in data and "mean_trust_improvement" in data:
                r = ConstitutionalBenchmarkResult(**{
                    k: v for k, v in data.items()
                    if k in ConstitutionalBenchmarkResult.__dataclass_fields__
                })
                const_results.append(r)
        except Exception:
            continue

    if const_results:
        latest_const: dict[str, ConstitutionalBenchmarkResult] = {}
        for r in const_results:
            if r.model not in latest_const or r.timestamp > latest_const[r.model].timestamp:
                latest_const[r.model] = r
        const_results = list(latest_const.values())
        latex_parts.append(const_latex_table(const_results))
        latex_parts.append("")
        md_parts.append(const_md_table(const_results))
        md_parts.append("")
    else:
        md_parts.append("### Constitutional Clinical AI - Trust Score Improvement")
        md_parts.append("")
        md_parts.append("*Pending - run `python3 -m src.evaluation.constitutional_eval --dry-run` to generate.*")
        md_parts.append("")

    # Hallucination eval results (Asgari-style)
    from src.evaluation.hallucination_eval import (
        HallucinationBenchmarkResult,
        to_latex_table as hall_latex_table,
        to_markdown_table as hall_md_table,
    )
    hall_results: list[HallucinationBenchmarkResult] = []
    for f in sorted(RESULTS_DIR.glob("hallucination_eval_*.json")):
        try:
            data = json.loads(f.read_text())
            if "mean_sentence_hallucination_rate" in data:
                notes_raw = data.pop("notes", [])
                r = HallucinationBenchmarkResult(**{
                    k: v for k, v in data.items()
                    if k in HallucinationBenchmarkResult.__dataclass_fields__
                })
                hall_results.append(r)
        except Exception:
            continue

    if hall_results:
        latest_hall: dict[str, HallucinationBenchmarkResult] = {}
        for r in hall_results:
            if r.model not in latest_hall or r.timestamp > latest_hall[r.model].timestamp:
                latest_hall[r.model] = r
        hall_results = list(latest_hall.values())
        latex_parts.append(hall_latex_table(hall_results))
        latex_parts.append("")
        md_parts.append(hall_md_table(hall_results))
        md_parts.append("")
    else:
        md_parts.append("### Hallucination & Omission Evaluation (vs Asgari et al. 2025)")
        md_parts.append("")
        md_parts.append("*Pending - run `python3 -m src.evaluation.hallucination_eval --dry-run` to generate.*")
        md_parts.append("")

    latex_content = "\n".join(latex_parts)
    md_content = "\n".join(md_parts)

    # Write output files
    tex_path = RESULTS_DIR / "tables.tex"
    tex_path.write_text(latex_content)

    md_path = RESULTS_DIR / "tables.md"
    md_path.write_text(md_content)

    logger.info(f"LaTeX tables written: {tex_path}")
    logger.info(f"Markdown tables written: {md_path}")

    return latex_content, md_content


def main():
    latex, md = generate_all()

    print("=== LaTeX ===")
    print(latex)
    print()
    print("=== Markdown ===")
    print(md)


if __name__ == "__main__":
    main()
