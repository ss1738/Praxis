#!/usr/bin/env python3
"""Head-to-head diarisation: PRAXIS-TIMBRE vs pyannote/speaker-diarization-3.1.

For each of the 14 PriMock57 test consultations:
  1. Mix the doctor + patient channels into a 2-speaker stereo→mono signal.
  2. Build a reference RTTM from VAD-derived speech regions per channel
     (doctor channel → SPEAKER_DOCTOR, patient channel → SPEAKER_PATIENT).
  3. Run pyannote 3.1 on the mixed audio.
  4. Run PRAXIS-TIMBRE (custom ECAPA + spectral clustering) on the mixed audio.
  5. Compute DER via pyannote.metrics.diarization.DiarizationErrorRate.
  6. Aggregate per-consultation wins, paired Wilcoxon signed-rank test.

Writes data/evaluation_results/h2h_praxis-timbre_vs_pyannote-3.1.json with
the full per-consultation breakdown and the aggregate tally.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
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
    """Return [(consultation_id, doctor_wav, patient_wav), ...] from the test split."""
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
    """Cheap energy-based VAD. Returns [(start_s, end_s), ...].

    Used only for reference RTTM construction. PriMock57 channels are
    isolated-speaker so a single-side energy gate works adequately.
    """
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
    # Coalesce consecutive voiced frames into intervals
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
    """Mix doctor + patient channels into a single mono stream at 16 kHz."""
    doc, sr_d = sf.read(str(doctor_wav))
    pat, sr_p = sf.read(str(patient_wav))
    if doc.ndim > 1:
        doc = doc.mean(axis=1)
    if pat.ndim > 1:
        pat = pat.mean(axis=1)
    if sr_d != sr_p:
        raise ValueError(f"sample-rate mismatch: doctor={sr_d}, patient={sr_p}")
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


def _build_reference_annotation(doctor_wav: Path, patient_wav: Path):
    """Build a pyannote Annotation reference labelling DOCTOR/PATIENT speech regions."""
    from pyannote.core import Annotation, Segment
    ann = Annotation()
    for start, end in _vad_segments(doctor_wav):
        ann[Segment(start, end)] = "DOCTOR"
    for start, end in _vad_segments(patient_wav):
        ann[Segment(start, end)] = "PATIENT"
    return ann


def _pyannote_hypothesis(mixed_wav_path: Path, hf_token: str | None):
    """Run pyannote 3.1 on the mixed audio. Returns (annotation, wall_time_s)."""
    from pyannote.audio import Pipeline
    t0 = time.perf_counter()
    # pyannote-audio 4.x: from_pretrained takes `token=` not `use_auth_token=`
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token,
        )
    except TypeError:
        # Older pyannote-audio (3.x)
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    raw = pipeline({"uri": "mixed", "audio": str(mixed_wav_path)}, num_speakers=2)
    secs = round(time.perf_counter() - t0, 2)
    # pyannote-audio 4.x returns DiarizeOutput; the Annotation lives under
    # .speaker_diarization. Older 3.x returned the Annotation directly.
    if hasattr(raw, "speaker_diarization"):
        return raw.speaker_diarization, secs
    return raw, secs


def _timbre_hypothesis(mixed_wav_path: Path):
    """Run PRAXIS-TIMBRE (custom ECAPADiariser) on the mixed audio.

    Returns (annotation, wall_time_s).
    """
    from pyannote.core import Annotation, Segment
    from src.speech_to_text.speaker_diarisation import ECAPADiariser

    t0 = time.perf_counter()
    diar = ECAPADiariser(
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_speakers=2,
    )
    result = diar.diarise_audio(str(mixed_wav_path))
    secs = round(time.perf_counter() - t0, 2)

    ann = Annotation()
    for seg in result.segments:
        ann[Segment(seg.start, seg.end)] = seg.speaker
    return ann, secs


def _compute_der(reference, hypothesis) -> dict:
    from pyannote.metrics.diarization import DiarizationErrorRate
    metric = DiarizationErrorRate(collar=0.5, skip_overlap=True)
    der = metric(reference, hypothesis, detailed=True)
    return {
        "der": round(float(der.get("diarization error rate",
                                   metric(reference, hypothesis))), 4),
        "miss": round(float(der.get("missed detection", 0.0)), 4),
        "false_alarm": round(float(der.get("false alarm", 0.0)), 4),
        "confusion": round(float(der.get("confusion", 0.0)), 4),
        "total": round(float(der.get("total", 0.0)), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-json",
                    default=str(REPO_ROOT / "data/processed/primock57_chunked/test.json"))
    ap.add_argument("--out",
                    default=str(REPO_ROOT / "data/evaluation_results/h2h_praxis-timbre_vs_pyannote-3.1.json"))
    ap.add_argument("--skip-pyannote", action="store_true",
                    help="Run TIMBRE only (for environments without pyannote auth)")
    args = ap.parse_args()

    hf_token = None
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        hf_token = token_path.read_text().strip()

    pairs = _consultation_pairs(Path(args.test_json))
    print(f"Found {len(pairs)} consultations with both channels.", flush=True)

    per_consult = []
    timbre_der_list: list[float] = []
    pyannote_der_list: list[float] = []

    for i, (cid, doctor_wav, patient_wav) in enumerate(pairs, 1):
        print(f"\n[{i}/{len(pairs)}] {cid}", flush=True)
        reference = _build_reference_annotation(doctor_wav, patient_wav)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpf:
            mixed, sr = _mix_stereo_to_mono(doctor_wav, patient_wav)
            sf.write(tmpf.name, mixed, sr)
            mixed_path = Path(tmpf.name)

        try:
            timbre_ann, timbre_secs = _timbre_hypothesis(mixed_path)
            timbre_metrics = _compute_der(reference, timbre_ann)
            print(f"  TIMBRE   DER={timbre_metrics['der']:.4f}  rtf-ish={timbre_secs}s",
                  flush=True)

            if args.skip_pyannote:
                pyannote_metrics = None
                pyannote_secs = None
            else:
                try:
                    py_ann, pyannote_secs = _pyannote_hypothesis(mixed_path, hf_token)
                    pyannote_metrics = _compute_der(reference, py_ann)
                    print(f"  pyannote DER={pyannote_metrics['der']:.4f}  rtf-ish={pyannote_secs}s",
                          flush=True)
                except Exception as exc:
                    print(f"  pyannote FAILED: {exc}", flush=True)
                    pyannote_metrics = None
                    pyannote_secs = None
        finally:
            mixed_path.unlink(missing_ok=True)

        if pyannote_metrics is not None:
            winner = (
                "praxis_timbre"
                if timbre_metrics["der"] < pyannote_metrics["der"] - 1e-6
                else ("pyannote_3_1"
                      if pyannote_metrics["der"] < timbre_metrics["der"] - 1e-6
                      else "tie")
            )
        else:
            winner = None

        per_consult.append({
            "consultation_id": cid,
            "audio_duration_s": round(sf.info(str(doctor_wav)).duration, 2),
            "reference_segments": [
                {"speaker": label,
                 "start": round(seg.start, 3),
                 "end": round(seg.end, 3)}
                for seg, _, label in reference.itertracks(yield_label=True)
            ][:50],
            "praxis_timbre": {
                **timbre_metrics,
                "wall_time_s": timbre_secs,
            },
            "pyannote_3_1": ({
                **pyannote_metrics,
                "wall_time_s": pyannote_secs,
            } if pyannote_metrics is not None else None),
            "winner": winner,
        })
        timbre_der_list.append(timbre_metrics["der"])
        if pyannote_metrics is not None:
            pyannote_der_list.append(pyannote_metrics["der"])

    aggregate = {
        "n_consultations": len(per_consult),
        "praxis_timbre_mean_der": round(float(np.mean(timbre_der_list)), 4),
        "praxis_timbre_median_der": round(float(np.median(timbre_der_list)), 4),
    }
    if pyannote_der_list:
        aggregate["pyannote_3_1_mean_der"] = round(float(np.mean(pyannote_der_list)), 4)
        aggregate["pyannote_3_1_median_der"] = round(float(np.median(pyannote_der_list)), 4)
        # Wilcoxon paired
        from scipy.stats import wilcoxon
        timbre_arr = np.array(timbre_der_list[:len(pyannote_der_list)])
        py_arr = np.array(pyannote_der_list)
        if np.any(timbre_arr - py_arr != 0):
            stat, p = wilcoxon(timbre_arr, py_arr, zero_method="wilcox")
            aggregate["wilcoxon_signed_rank_p"] = round(float(p), 6)
            aggregate["wilcoxon_statistic"] = round(float(stat), 4)
        else:
            aggregate["wilcoxon_signed_rank_p"] = None
        wins = sum(1 for c in per_consult if c["winner"] == "praxis_timbre")
        losses = sum(1 for c in per_consult if c["winner"] == "pyannote_3_1")
        ties = sum(1 for c in per_consult if c["winner"] == "tie")
        aggregate["tally"] = {
            "praxis_timbre_wins": wins,
            "pyannote_3_1_wins": losses,
            "ties": ties,
        }

    output = {
        "is_synthetic": False,
        "hardware_target": "RTX 5090 32GB Linux (satyawan-1)",
        "host": socket.gethostname(),
        "git_commit": _git_sha(),
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "test_set": "primock57_test",
        "reference_protocol": "energy-VAD-per-channel; PriMock57 channels are speaker-isolated",
        "vad_params": {"frame_ms": 30, "energy_pct_threshold": 30.0, "min_dur_s": 0.3},
        "der_protocol": "pyannote.metrics.DiarizationErrorRate(collar=0.5, skip_overlap=True)",
        "pyannote_pipeline": "pyannote/speaker-diarization-3.1",
        "aggregate": aggregate,
        "per_consultation": per_consult,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWritten: {args.out}", flush=True)
    print(json.dumps(aggregate, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
