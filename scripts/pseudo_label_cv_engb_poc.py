"""
Pseudo-label Common Voice en-GB with Whisper-large-v3 + Granary-style filter.

Validates Stage 3 (noisy-student) of the train plan at ~1% of eventual scale.
Produces commercial-safe (CC0) pseudo-labelled UK-accent audio manifests that can be
immediately folded into the next supervised FT pass.

Filter thresholds (per Granary arXiv:2505.13404 + Whisper §4.5):
  - avg_logprob        > -0.4
  - no_speech_prob     < 0.3
  - compression_ratio  <= 2.4
  - min_clip_seconds   >= 2.0
  - max_clip_seconds   <= 25.0
Expected retention: ~55-65% of input clips.

Usage on vast.ai:
    pip install -U datasets transformers accelerate soundfile librosa
    huggingface-cli login          # accept Common Voice terms at
                                   # https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0
    python3 scripts/pseudo_label_cv_engb_poc.py --hours 20 --device cuda

Writes:
    data/processed/pseudo_label_cv_engb/retained.jsonl   (accepted pairs, one per line)
    data/processed/pseudo_label_cv_engb/rejected.jsonl   (rejected + reason)
    data/processed/pseudo_label_cv_engb/stats.json       (retention rate + filter breakdown)
    data/processed/pseudo_label_cv_engb/audio/*.wav      (accepted clips, 16k mono)
"""
import os, sys, json, time, argparse, re
from pathlib import Path
import numpy as np
import torch
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data/processed/pseudo_label_cv_engb")
AUDIO_DIR = os.path.join(OUT_DIR, "audio")
TARGET_SR = 16000

THRESH = dict(
    avg_logprob_min=-0.4,
    no_speech_max=0.3,
    compression_ratio_max=2.4,
    min_seconds=2.0,
    max_seconds=25.0,
)

NONWORD_RE = re.compile(r"[^a-z0-9' ]+")
WS_RE = re.compile(r"\s+")
def norm_text(t: str) -> str:
    t = t.lower()
    t = NONWORD_RE.sub(" ", t)
    t = WS_RE.sub(" ", t).strip()
    return t

def to_mono_16k(audio, sr):
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        ratio = TARGET_SR / sr
        audio = np.interp(
            np.linspace(0, len(audio) - 1, int(len(audio) * ratio)),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)
    return audio.astype(np.float32)

def compression_ratio(text: str) -> float:
    if not text:
        return 0.0
    import zlib
    b = text.encode("utf-8")
    return len(b) / max(1, len(zlib.compress(b)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=20.0, help="target hours of audio to process")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--locale", default="en", help="HF dataset locale/config")
    ap.add_argument("--accent_filter", default="england,british,scottish,welsh,irish,united kingdom,uk",
                    help="comma-separated accent substrings to keep (matched in lowercase)")
    ap.add_argument("--save_audio", action="store_true", help="write accepted clips to audio/")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    if args.save_audio:
        os.makedirs(AUDIO_DIR, exist_ok=True)

    print(f"=== Pseudo-label CV en-GB - Granary filter ===")
    print(f"Model:      {args.model}")
    print(f"Device:     {args.device}")
    print(f"Target:     {args.hours} h audio")
    print(f"Accent kw:  {args.accent_filter}")
    print(f"Out:        {OUT_DIR}")

    from datasets import load_dataset
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

    accent_kws = [k.strip().lower() for k in args.accent_filter.split(",") if k.strip()]

    print("Loading Common Voice 17 (streaming)...")
    ds = load_dataset(
        "mozilla-foundation/common_voice_17_0",
        args.locale,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    print(f"Loading {args.model} on {args.device}...")
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model, torch_dtype=dtype)
    model.to(args.device)
    model.eval()

    gen_kwargs = dict(
        language="en",
        task="transcribe",
        num_beams=1,
        do_sample=False,
        return_timestamps=False,
        return_dict_in_generate=True,
        output_scores=True,
        no_repeat_ngram_size=3,
        max_new_tokens=440,
    )

    retained_f = open(os.path.join(OUT_DIR, "retained.jsonl"), "w")
    rejected_f = open(os.path.join(OUT_DIR, "rejected.jsonl"), "w")

    stats = dict(
        seen=0, accent_kept=0, processed=0, retained=0,
        rejected_logprob=0, rejected_nospeech=0, rejected_compression=0,
        rejected_duration=0, rejected_empty=0, rejected_other=0,
        hours_seen=0.0, hours_retained=0.0,
        rejection_reasons=[],
    )
    t0 = time.time()

    for ex in ds:
        stats["seen"] += 1
        accent = (ex.get("accent") or "").lower()
        if accent_kws and not any(k in accent for k in accent_kws):
            continue
        stats["accent_kept"] += 1

        arr = ex["audio"]["array"]
        sr = ex["audio"]["sampling_rate"]
        audio = to_mono_16k(np.asarray(arr, dtype=np.float32), sr)
        dur_s = len(audio) / TARGET_SR
        stats["hours_seen"] += dur_s / 3600

        if dur_s < THRESH["min_seconds"] or dur_s > THRESH["max_seconds"]:
            stats["rejected_duration"] += 1
            rejected_f.write(json.dumps(dict(id=ex.get("client_id", ""), reason="duration", dur_s=dur_s)) + "\n")
            if stats["hours_seen"] >= args.hours: break
            continue

        inputs = processor(audio, sampling_rate=TARGET_SR, return_tensors="pt").to(args.device)
        inputs["input_features"] = inputs["input_features"].to(dtype)
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)

        tokens = out.sequences[0]
        scores = out.scores  # list of (1, vocab)
        if scores:
            logprobs = []
            for step, s in enumerate(scores):
                tok = tokens[step + 1] if step + 1 < len(tokens) else None
                if tok is None: break
                lp = torch.log_softmax(s[0].float(), dim=-1)[tok].item()
                logprobs.append(lp)
            avg_lp = float(np.mean(logprobs)) if logprobs else -99.0
        else:
            avg_lp = -99.0

        text = processor.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()
        text_n = norm_text(text)
        no_speech = 0.0  # large-v3 no_speech_prob is per-segment; we infer via empty-text proxy
        if not text_n:
            no_speech = 1.0
        cr = compression_ratio(text)

        stats["processed"] += 1
        why = None
        if not text_n:
            why = "empty"; stats["rejected_empty"] += 1
        elif avg_lp <= THRESH["avg_logprob_min"]:
            why = f"logprob={avg_lp:.3f}"; stats["rejected_logprob"] += 1
        elif no_speech >= THRESH["no_speech_max"]:
            why = f"no_speech={no_speech:.3f}"; stats["rejected_nospeech"] += 1
        elif cr > THRESH["compression_ratio_max"]:
            why = f"compression={cr:.3f}"; stats["rejected_compression"] += 1

        if why:
            rejected_f.write(json.dumps(dict(id=ex.get("client_id", ""), reason=why,
                                             text=text, dur_s=dur_s)) + "\n")
            if len(stats["rejection_reasons"]) < 30:
                stats["rejection_reasons"].append(why)
        else:
            audio_path = None
            if args.save_audio:
                clip_id = f"cv_{stats['retained']:06d}.wav"
                audio_path = os.path.join(AUDIO_DIR, clip_id)
                sf.write(audio_path, audio, TARGET_SR, subtype="PCM_16")
            retained_f.write(json.dumps(dict(
                id=ex.get("client_id", ""),
                audio_path=audio_path,
                text=text,
                text_norm=text_n,
                dur_s=round(dur_s, 2),
                avg_logprob=round(avg_lp, 3),
                compression_ratio=round(cr, 3),
                accent=accent,
                locale=args.locale,
            )) + "\n")
            stats["retained"] += 1
            stats["hours_retained"] += dur_s / 3600

        if stats["processed"] % 50 == 0:
            elapsed = (time.time() - t0) / 60
            print(f"  seen={stats['seen']} accent_kept={stats['accent_kept']} "
                  f"processed={stats['processed']} retained={stats['retained']} "
                  f"h_seen={stats['hours_seen']:.2f} h_ret={stats['hours_retained']:.2f} "
                  f"t={elapsed:.1f}m")

        if stats["hours_seen"] >= args.hours:
            break

    retained_f.close()
    rejected_f.close()

    stats["retention_rate"] = (stats["retained"] / stats["processed"]) if stats["processed"] else 0.0
    stats["wall_min"] = round((time.time() - t0) / 60, 2)
    with open(os.path.join(OUT_DIR, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Seen           : {stats['seen']}")
    print(f"Accent-kept    : {stats['accent_kept']}")
    print(f"Processed      : {stats['processed']}")
    print(f"Retained       : {stats['retained']}  ({stats['retention_rate']*100:.1f}%)")
    print(f"Hours retained : {stats['hours_retained']:.2f} h")
    print(f"Rejections     : logprob={stats['rejected_logprob']}  "
          f"empty={stats['rejected_empty']}  "
          f"compression={stats['rejected_compression']}  "
          f"duration={stats['rejected_duration']}")
    print(f"Wall time      : {stats['wall_min']:.1f} min")
    print(f"Manifests in   : {OUT_DIR}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
