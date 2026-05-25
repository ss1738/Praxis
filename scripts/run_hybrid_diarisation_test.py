#!/usr/bin/env python3
"""Hybrid diarisation diagnostic - isolates the clustering vs embedding contribution.

Three diariser conditions on the same 14 PriMock57 test consultations:

  A. TIMBRE-only (current baseline)
       sliding-window ECAPA-TDNN → agglomerative clustering(cosine, k=2)
       (reference: data/evaluation_results/h2h_praxis-timbre_vs_pyannote-3.1.json)

  B. Hybrid v1 - pyannote VAD + TIMBRE + KMeans
       neural VAD (pyannote/voice-activity-detection) → TIMBRE embed each
       speech region → KMeans(k=2) clustering

  C. Pyannote full (reference)
       pyannote/speaker-diarization-3.1 end-to-end
       (reference: data/evaluation_results/h2h_praxis-timbre_vs_pyannote-3.1.json)

Test isolation:
  Hybrid v1 − TIMBRE-only  isolates the contribution of replacing our blind
  frame-walking + agglomerative cluster with a neural-VAD-guided pipeline.
  Pyannote-full − Hybrid v1  isolates the remaining contribution of pyannote's
  end-to-end segmentation transformer over a vanilla VAD+embed+KMeans pipeline.

Reference RTTM: same energy-VAD-per-channel construction as the existing h2h
(doctor channel → SPEAKER_DOCTOR; patient channel → SPEAKER_PATIENT).

Output: data/evaluation_results/hybrid_diarisation_analysis.json
        data/evaluation_results/hybrid_diarisation_analysis.tex (numerical only)
        data/evaluation_results/hybrid_diarisation_analysis.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _consultation_pairs(test_json_path: Path) -> list[tuple[str, Path, Path]]:
    data = json.loads(test_json_path.read_text())
    by_consult = {}
    for entry in data:
        cid = entry.get("consultation") or Path(entry["audio_path"]).name.split("_doctor")[0].split("_patient")[0]
        spk = "doctor" if "_doctor" in entry["audio_path"] else "patient"
        by_consult.setdefault(cid, {})[spk] = Path(entry["audio_path"])
    pairs = []
    for cid in sorted(by_consult.keys()):
        d = by_consult[cid].get("doctor")
        p = by_consult[cid].get("patient")
        if d and p and d.exists() and p.exists():
            pairs.append((cid, d, p))
    return pairs


def _vad_segments(wav_path: Path, frame_ms: int = 30, energy_pct: float = 30.0,
                  min_dur_s: float = 0.3) -> list[tuple[float, float]]:
    audio, sr = sf.read(str(wav_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    frame_len = int(sr * frame_ms / 1000)
    n_frames = len(audio) // frame_len
    if n_frames == 0:
        return []
    energies = np.array([
        np.sqrt(np.mean(audio[i * frame_len:(i + 1) * frame_len] ** 2))
        for i in range(n_frames)
    ])
    threshold = np.percentile(energies, energy_pct)
    voiced = energies > threshold
    segs = []
    i = 0
    while i < len(voiced):
        if voiced[i]:
            j = i
            while j < len(voiced) and voiced[j]:
                j += 1
            start_s = i * frame_ms / 1000.0
            end_s = j * frame_ms / 1000.0
            if end_s - start_s >= min_dur_s:
                segs.append((start_s, end_s))
            i = j
        else:
            i += 1
    return segs


def _mix_stereo_to_mono(doctor_wav: Path, patient_wav: Path) -> tuple[np.ndarray, int]:
    doc, sr_d = sf.read(str(doctor_wav))
    pat, sr_p = sf.read(str(patient_wav))
    if doc.ndim > 1:
        doc = doc.mean(axis=1)
    if pat.ndim > 1:
        pat = pat.mean(axis=1)
    if sr_d != sr_p:
        raise ValueError(f"sample-rate mismatch")
    n = max(len(doc), len(pat))
    if len(doc) < n:
        doc = np.pad(doc, (0, n - len(doc)))
    if len(pat) < n:
        pat = np.pad(pat, (0, n - len(pat)))
    mixed = 0.5 * (doc + pat)
    peak = np.max(np.abs(mixed))
    if peak > 0:
        mixed = mixed / peak * 0.95
    return mixed.astype(np.float32), sr_d


def _build_reference(doctor_wav: Path, patient_wav: Path):
    from pyannote.core import Annotation, Segment
    ann = Annotation()
    for s, e in _vad_segments(doctor_wav):
        ann[Segment(s, e)] = "DOCTOR"
    for s, e in _vad_segments(patient_wav):
        ann[Segment(s, e)] = "PATIENT"
    return ann


def _compute_der(reference, hypothesis):
    from pyannote.metrics.diarization import DiarizationErrorRate
    metric = DiarizationErrorRate(collar=0.5, skip_overlap=True)
    detail = metric(reference, hypothesis, detailed=True)
    return {
        "der": round(float(detail.get("diarization error rate",
                                       metric(reference, hypothesis))), 4),
        "miss": round(float(detail.get("missed detection", 0.0)), 4),
        "false_alarm": round(float(detail.get("false alarm", 0.0)), 4),
        "confusion": round(float(detail.get("confusion", 0.0)), 4),
    }


# ───────────────────── Hybrid v1: pyannote VAD + TIMBRE + KMeans ─────────────────────


_PYANNOTE_PIPELINE = None  # module-level cache


def _get_pyannote_pipeline(hf_token: str | None):
    global _PYANNOTE_PIPELINE
    if _PYANNOTE_PIPELINE is not None:
        return _PYANNOTE_PIPELINE
    from pyannote.audio import Pipeline
    try:
        p = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=hf_token)
    except TypeError:
        p = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
    if torch.cuda.is_available():
        p.to(torch.device("cuda"))
    _PYANNOTE_PIPELINE = p
    return p


def _pyannote_turns(audio_path: Path, hf_token: str | None) -> list[tuple[float, float, str]]:
    """Run pyannote 3.1 and return its turn boundaries: [(start_s, end_s, pyannote_label)].

    The pipeline assigns its own SPEAKER_00 / SPEAKER_01 labels; we ignore those
    and only keep the turn boundaries. The hybrid then re-clusters using TIMBRE
    embeddings, isolating the embedding+cluster contribution from pyannote's
    segmentation.
    """
    pipeline = _get_pyannote_pipeline(hf_token)
    raw = pipeline({"uri": "mixed", "audio": str(audio_path)}, num_speakers=2)
    ann = raw.speaker_diarization if hasattr(raw, "speaker_diarization") else raw
    turns: list[tuple[float, float, str]] = []
    for segment, _, label in ann.itertracks(yield_label=True):
        turns.append((float(segment.start), float(segment.end), str(label)))
    return turns


def _hybrid_v1_diarise(mixed_wav: Path, ecapa, hf_token: str | None,
                        window_s: float = 1.5, step_s: float = 0.75):
    """pyannote 3.1 segmentation → TIMBRE per-turn embed → KMeans(k=2).

    Uses pyannote's own turn boundaries (the strong neural segmentation), discards
    pyannote's speaker labels, and reassigns speakers via TIMBRE-embedded
    KMeans(k=2). Isolates the contribution of TIMBRE-vs-wespeaker embeddings.
    """
    from pyannote.core import Annotation, Segment
    from sklearn.cluster import KMeans

    t0 = time.perf_counter()

    # 1. Strong neural segmentation (use pyannote 3.1's turn boundaries)
    turns = _pyannote_turns(mixed_wav, hf_token)
    if len(turns) < 2:
        ann = Annotation()
        for (s, e, _) in turns:
            ann[Segment(s, e)] = "Speaker_0"
        return ann, round(time.perf_counter() - t0, 2), {
            "n_turns_from_pyannote": len(turns), "n_embeddings": 0,
        }

    # 2. Load audio
    audio, sr = sf.read(str(mixed_wav), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    waveform = torch.from_numpy(audio)

    # 3. Per-turn TIMBRE embedding
    ecapa._load()
    turn_embeddings = []
    valid_turns = []
    for (start_s, end_s, _) in turns:
        s_idx = int(start_s * sr)
        e_idx = int(end_s * sr)
        if e_idx - s_idx < int(0.4 * sr):  # skip <0.4s turns (unreliable)
            continue
        # If turn is short, embed the whole turn; if long, average several windows
        win = int(window_s * sr)
        step = int(step_s * sr)
        window_embeddings = []
        cur = s_idx
        while cur + win <= e_idx:
            segment = waveform[cur:cur + win]
            mel = ecapa._audio_to_mel(segment).unsqueeze(0).to(ecapa._device)
            with torch.inference_mode():
                emb = ecapa._model.get_embedding(mel).cpu().numpy()[0]
            window_embeddings.append(emb)
            cur += step
        if not window_embeddings:
            # Turn shorter than window - embed the full turn padded to win length
            segment = waveform[s_idx:e_idx]
            if segment.shape[0] < win:
                segment = torch.nn.functional.pad(segment, (0, win - segment.shape[0]))
            mel = ecapa._audio_to_mel(segment[:win]).unsqueeze(0).to(ecapa._device)
            with torch.inference_mode():
                emb = ecapa._model.get_embedding(mel).cpu().numpy()[0]
            window_embeddings.append(emb)
        # Mean-pool per turn
        turn_emb = np.mean(np.stack(window_embeddings, 0), axis=0)
        turn_embeddings.append(turn_emb)
        valid_turns.append((start_s, end_s))

    if len(turn_embeddings) < 2:
        ann = Annotation()
        for s, e in valid_turns:
            ann[Segment(s, e)] = "Speaker_0"
        return ann, round(time.perf_counter() - t0, 2), {
            "n_turns_from_pyannote": len(turns), "n_embeddings": len(turn_embeddings),
        }

    embeddings = np.vstack(turn_embeddings)
    # L2-normalise for cosine-equivalent KMeans clustering
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    km = KMeans(n_clusters=2, n_init=10, random_state=42)
    labels = km.fit_predict(embeddings)

    # Map clusters → Doctor / Patient by speaking-time totals
    durations = {0: 0.0, 1: 0.0}
    for (s, e), lab in zip(valid_turns, labels):
        durations[lab] += (e - s)
    sorted_clusters = sorted(durations, key=lambda c: durations[c], reverse=True)
    label_map = {sorted_clusters[0]: "Doctor", sorted_clusters[1]: "Patient"}

    ann = Annotation()
    for (s, e), lab in zip(valid_turns, labels):
        ann[Segment(s, e)] = label_map[lab]

    return ann, round(time.perf_counter() - t0, 2), {
        "n_turns_from_pyannote": len(turns),
        "n_turns_with_embedding": int(len(turn_embeddings)),
        "cluster_durations_s": {int(k): round(v, 2) for k, v in durations.items()},
    }


# ───────────────────── Driver ─────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-json",
                    default=str(REPO_ROOT / "data/processed/primock57_chunked/test.json"))
    ap.add_argument("--out",
                    default=str(REPO_ROOT / "data/evaluation_results/hybrid_diarisation_analysis.json"))
    ap.add_argument("--baseline-h2h",
                    default=str(REPO_ROOT / "data/evaluation_results/h2h_praxis-timbre_vs_pyannote-3.1.json"))
    args = ap.parse_args()

    hf_token = None
    tok_path = Path.home() / ".cache" / "huggingface" / "token"
    if tok_path.exists():
        hf_token = tok_path.read_text().strip()

    pairs = _consultation_pairs(Path(args.test_json))
    print(f"Found {len(pairs)} consultations.", flush=True)

    # Load baseline numbers for direct comparison
    baseline_doc = json.loads(Path(args.baseline_h2h).read_text())
    baseline_by_id = {c["consultation_id"]: c for c in baseline_doc["per_consultation"]}

    # Load TIMBRE
    from src.speech_to_text.speaker_diarisation import ECAPADiariser
    ecapa = ECAPADiariser(
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_speakers=2,
    )

    per_consult = []
    hybrid_ders: list[float] = []
    timbre_ders_baseline: list[float] = []
    pyannote_ders_baseline: list[float] = []

    for i, (cid, doc_wav, pat_wav) in enumerate(pairs, 1):
        print(f"\n[{i}/{len(pairs)}] {cid}", flush=True)
        ref = _build_reference(doc_wav, pat_wav)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            mixed, sr = _mix_stereo_to_mono(doc_wav, pat_wav)
            sf.write(tmp.name, mixed, sr)
            mixed_path = Path(tmp.name)

        try:
            hybrid_ann, hybrid_secs, hybrid_meta = _hybrid_v1_diarise(
                mixed_path, ecapa, hf_token
            )
            hybrid_metrics = _compute_der(ref, hybrid_ann)
            print(f"  HYBRID (pyannote-VAD + TIMBRE-embed + KMeans):  "
                  f"DER={hybrid_metrics['der']:.4f}  wall={hybrid_secs}s  "
                  f"speech_segs={hybrid_meta.get('n_speech_segs')}  "
                  f"n_emb={hybrid_meta.get('n_embeddings')}",
                  flush=True)
        finally:
            mixed_path.unlink(missing_ok=True)

        baseline = baseline_by_id.get(cid, {})
        timbre_baseline_der = (baseline.get("praxis_timbre") or {}).get("der")
        pyannote_baseline_der = (baseline.get("pyannote_3_1") or {}).get("der")

        hybrid_ders.append(hybrid_metrics["der"])
        if timbre_baseline_der is not None:
            timbre_ders_baseline.append(timbre_baseline_der)
        if pyannote_baseline_der is not None:
            pyannote_ders_baseline.append(pyannote_baseline_der)

        # Per-consultation winner among {timbre, hybrid, pyannote}
        candidates = {
            "timbre_baseline": timbre_baseline_der,
            "hybrid": hybrid_metrics["der"],
            "pyannote_baseline": pyannote_baseline_der,
        }
        valid = {k: v for k, v in candidates.items() if v is not None}
        winner = min(valid, key=valid.get) if valid else None

        per_consult.append({
            "consultation_id": cid,
            "timbre_baseline": ({"der": timbre_baseline_der} if timbre_baseline_der is not None else None),
            "hybrid": {**hybrid_metrics, "wall_time_s": hybrid_secs, **hybrid_meta},
            "pyannote_baseline": ({"der": pyannote_baseline_der} if pyannote_baseline_der is not None else None),
            "winner": winner,
            "hybrid_vs_timbre_pp_improvement": (
                round((timbre_baseline_der - hybrid_metrics["der"]) * 100, 2)
                if timbre_baseline_der is not None else None
            ),
            "hybrid_vs_pyannote_pp_gap": (
                round((hybrid_metrics["der"] - pyannote_baseline_der) * 100, 2)
                if pyannote_baseline_der is not None else None
            ),
        })

    # Aggregate
    aggregate = {
        "n_consultations": len(per_consult),
        "hybrid_mean_der": round(float(np.mean(hybrid_ders)), 4) if hybrid_ders else None,
        "hybrid_median_der": round(float(np.median(hybrid_ders)), 4) if hybrid_ders else None,
        "timbre_baseline_mean_der": round(float(np.mean(timbre_ders_baseline)), 4) if timbre_ders_baseline else None,
        "pyannote_baseline_mean_der": round(float(np.mean(pyannote_ders_baseline)), 4) if pyannote_ders_baseline else None,
    }
    # Pairwise win/loss vs timbre baseline (the contribution of replacing clustering)
    h_vs_t_wins = sum(1 for c in per_consult if c.get("hybrid_vs_timbre_pp_improvement", -1) > 0)
    h_vs_t_losses = sum(1 for c in per_consult if c.get("hybrid_vs_timbre_pp_improvement") is not None
                        and c["hybrid_vs_timbre_pp_improvement"] < 0)
    h_vs_t_ties = len(per_consult) - h_vs_t_wins - h_vs_t_losses
    aggregate["hybrid_vs_timbre_baseline_tally"] = {
        "hybrid_wins": h_vs_t_wins,
        "hybrid_losses": h_vs_t_losses,
        "ties": h_vs_t_ties,
    }
    h_vs_p_wins = sum(1 for c in per_consult
                       if c.get("hybrid_vs_pyannote_pp_gap") is not None
                       and c["hybrid_vs_pyannote_pp_gap"] < 0)
    h_vs_p_losses = sum(1 for c in per_consult
                        if c.get("hybrid_vs_pyannote_pp_gap") is not None
                        and c["hybrid_vs_pyannote_pp_gap"] > 0)
    aggregate["hybrid_vs_pyannote_tally"] = {
        "hybrid_wins": h_vs_p_wins,
        "pyannote_wins": h_vs_p_losses,
        "ties": len(per_consult) - h_vs_p_wins - h_vs_p_losses,
    }

    # Paired Wilcoxon
    from scipy.stats import wilcoxon
    if timbre_ders_baseline and len(timbre_ders_baseline) == len(hybrid_ders):
        t_arr = np.array(timbre_ders_baseline)
        h_arr = np.array(hybrid_ders)
        if np.any(t_arr - h_arr != 0):
            stat, p = wilcoxon(t_arr, h_arr, zero_method="wilcox")
            aggregate["wilcoxon_hybrid_vs_timbre_p"] = round(float(p), 6)
    if pyannote_ders_baseline and len(pyannote_ders_baseline) == len(hybrid_ders):
        p_arr = np.array(pyannote_ders_baseline)
        h_arr = np.array(hybrid_ders)
        if np.any(h_arr - p_arr != 0):
            stat, p = wilcoxon(h_arr, p_arr, zero_method="wilcox")
            aggregate["wilcoxon_hybrid_vs_pyannote_p"] = round(float(p), 6)

    output = {
        "is_synthetic": False,
        "hardware_target": "RTX 5090 32GB Linux (satyawan-1)",
        "host": socket.gethostname(),
        "git_commit": _git_sha(),
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "test_set": "primock57_test",
        "reference_protocol": "energy-VAD-per-channel (same as h2h)",
        "der_protocol": "pyannote.metrics.DiarizationErrorRate(collar=0.5, skip_overlap=True)",
        "diagnostic_design": (
            "Three diariser conditions on identical references. "
            "(A) TIMBRE-only baseline from h2h JSON. "
            "(B) Hybrid: pyannote/voice-activity-detection + TIMBRE-ECAPA "
            "per-window embed + KMeans(k=2, seed=42) clustering. "
            "(C) Pyannote 3.1 full pipeline from h2h JSON. "
            "Comparison (B - A) isolates the cost of TIMBRE's blind frame-walking "
            "and agglomerative clustering vs neural-VAD + KMeans. "
            "Comparison (C - B) isolates the remaining contribution of pyannote's "
            "end-to-end segmentation transformer beyond VAD+embed+cluster."
        ),
        "aggregate": aggregate,
        "per_consultation": per_consult,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(output, indent=2))

    # LaTeX table (numerical only)
    latex_lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Condition & Mean DER (\%) & Median DER (\%) \\",
        r"\midrule",
    ]
    if aggregate["timbre_baseline_mean_der"] is not None:
        latex_lines.append(
            f"TIMBRE baseline & {aggregate['timbre_baseline_mean_der']*100:.2f} & "
            f"{round(float(np.median(timbre_ders_baseline))*100, 2):.2f} \\\\"
        )
    latex_lines.append(
        f"Hybrid (pyannote VAD + TIMBRE + KMeans) & "
        f"{aggregate['hybrid_mean_der']*100:.2f} & "
        f"{aggregate['hybrid_median_der']*100:.2f} \\\\"
    )
    if aggregate["pyannote_baseline_mean_der"] is not None:
        latex_lines.append(
            f"Pyannote 3.1 full pipeline & "
            f"{aggregate['pyannote_baseline_mean_der']*100:.2f} & "
            f"{round(float(np.median(pyannote_ders_baseline))*100, 2):.2f} \\\\"
        )
    latex_lines += [r"\bottomrule", r"\end{tabular}"]
    Path(args.out).with_suffix(".tex").write_text("\n".join(latex_lines) + "\n")

    # Markdown
    md = [
        "| Condition | Mean DER | Median DER |", "|---|---|---|",
        f"| TIMBRE baseline | {aggregate['timbre_baseline_mean_der']*100:.2f}% | "
        f"{np.median(timbre_ders_baseline)*100:.2f}% |" if timbre_ders_baseline else "",
        f"| Hybrid (pyannote VAD + TIMBRE + KMeans) | "
        f"{aggregate['hybrid_mean_der']*100:.2f}% | "
        f"{aggregate['hybrid_median_der']*100:.2f}% |",
        f"| Pyannote 3.1 full pipeline | {aggregate['pyannote_baseline_mean_der']*100:.2f}% | "
        f"{np.median(pyannote_ders_baseline)*100:.2f}% |" if pyannote_ders_baseline else "",
    ]
    Path(args.out).with_suffix(".md").write_text("\n".join(md) + "\n")

    print(f"\nWritten: {args.out}", flush=True)
    print(json.dumps(aggregate, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
