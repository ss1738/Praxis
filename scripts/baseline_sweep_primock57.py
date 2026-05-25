"""
Baseline sweep on PriMock57 test split - establishes the WER numbers the v2 paper must beat.

Runs one model at a time against the same 14-track eval used for `wer_primock57_ft.json`.
Output JSON schema matches the v1 FT result so rows can be concatenated into one Table 1.

Usage (on vast.ai 4090):
    pip install -U transformers accelerate soundfile jiwer librosa
    python3 scripts/baseline_sweep_primock57.py --model openai/whisper-small      --device cuda
    python3 scripts/baseline_sweep_primock57.py --model openai/whisper-large-v3   --device cuda
    python3 scripts/baseline_sweep_primock57.py --model google/medasr             --device cuda  # HAI-DEF gate - accept terms on HF first

Writes: data/processed/baselines_primock57_<slug>.json
"""
import os, sys, json, re, argparse, time
import numpy as np
import torch
import soundfile as sf
from jiwer import wer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST = os.path.join(ROOT, "data/processed/primock57_chunked/test.json")
OUT_DIR = os.path.join(ROOT, "data/processed")
TARGET_SR = 16000

TAG_RE = re.compile(r"<[^>]+/?>|</[^>]+>")
NONWORD_RE = re.compile(r"[^a-z0-9' ]+")
WS_RE = re.compile(r"\s+")
def normalize(t: str) -> str:
    t = t.lower()
    t = TAG_RE.sub(" ", t)
    t = NONWORD_RE.sub(" ", t)
    t = WS_RE.sub(" ", t).strip()
    return t

def load_audio(path):
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        ratio = TARGET_SR / sr
        audio = np.interp(
            np.linspace(0, len(audio) - 1, int(len(audio) * ratio)),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak
    return audio

def build_pipeline(model_id: str, device: str):
    """HF ASR pipeline - handles both Whisper and MedASR-style Conformer-CTC."""
    from transformers import pipeline
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    kwargs = dict(
        task="automatic-speech-recognition",
        model=model_id,
        device=0 if device == "cuda" else -1,
        torch_dtype=dtype,
        chunk_length_s=30,
        stride_length_s=5,
        return_timestamps=False,
    )
    if "whisper" in model_id.lower():
        kwargs["generate_kwargs"] = {
            "language": "en",
            "task": "transcribe",
            "no_repeat_ngram_size": 3,
            "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
        }
    return pipeline(**kwargs)

def transcribe(pipe, audio: np.ndarray) -> str:
    out = pipe({"raw": audio, "sampling_rate": TARGET_SR})
    if isinstance(out, dict):
        return out.get("text", "")
    if isinstance(out, list) and out and isinstance(out[0], dict):
        return " ".join(o.get("text", "") for o in out)
    return str(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="HF model id (e.g. openai/whisper-small, openai/whisper-large-v3, google/medasr)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    slug = args.model.replace("/", "__").replace(".", "_")
    out_path = args.out or os.path.join(OUT_DIR, f"baselines_primock57_{slug}.json")

    print(f"=== PriMock57 baseline sweep ===")
    print(f"Model:  {args.model}")
    print(f"Device: {args.device}")
    print(f"Out:    {out_path}")

    pipe = build_pipeline(args.model, args.device)

    entries = json.load(open(TEST))
    if args.limit:
        entries = entries[:args.limit]
    print(f"Tracks: {len(entries)}")

    refs, hyps, per_file = [], [], []
    t0 = time.time()
    for i, e in enumerate(entries):
        audio = load_audio(e["audio_path"])
        dur = len(audio) / TARGET_SR
        t_s = time.time()
        hyp = transcribe(pipe, audio)
        dt = time.time() - t_s

        ref_n = normalize(e["text"])
        hyp_n = normalize(hyp)
        if not ref_n:
            continue
        w = wer(ref_n, hyp_n) if hyp_n else 1.0
        refs.append(ref_n)
        hyps.append(hyp_n)
        per_file.append(round(w * 100, 2))

        print(f"[{i+1:2d}/{len(entries)}] {os.path.basename(e['audio_path']):55s} "
              f"audio={dur:6.1f}s wall={dt:6.1f}s WER={w*100:5.1f}%")
        print(f"   REF: {ref_n[:140]}")
        print(f"   HYP: {hyp_n[:140]}")

    overall = wer(refs, hyps) if refs else float("nan")
    wall_min = (time.time() - t0) / 60
    print(f"\n{'='*60}")
    print(f"Overall WER: {overall*100:.2f}%  |  {len(refs)} tracks  |  {wall_min:.1f} min")
    print(f"{'='*60}")

    result = {
        "date": time.strftime("%Y-%m-%d"),
        "dataset": "PriMock57 test split (14 per-speaker tracks across 7 consultations)",
        "model": args.model,
        "decode": "HF ASR pipeline, chunk_length_s=30, stride_length_s=5, bf16 on CUDA",
        "device": args.device,
        "per_file_wer_pct": per_file,
        "overall_corpus_wer_pct": round(overall * 100, 2),
        "wall_time_min": round(wall_min, 2),
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
