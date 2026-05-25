"""WER on LibriSpeech test-clean using the 117M ConformerASR checkpoint.

Read speech, in-domain (part of the training set), mostly American.
Diagnostic: if WER is low here, the model works and PriMock57's high WER is
domain mismatch. If WER is also high, the model is undertrained.
"""
import os, sys, json, math, re, time, random, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import soundfile as sf, sentencepiece as spm
from jiwer import wer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wer_primock57 import (ConformerASR, mel_fb, normalize, ctc_greedy,
                           N_MELS, N_FFT, HOP, MAX_FRAMES, TARGET_SR)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "data/models/conformer_v2/final_model_v2_80M.pt")
TOK  = os.path.join(ROOT, "data/models/conformer_v2/praxis_tokenizer_v2.model")
LS   = os.path.join(ROOT, "data/librispeech/LibriSpeech/test-clean")

def scan(root):
    items = []
    for spk in os.listdir(root):
        sp_dir = os.path.join(root, spk)
        if not os.path.isdir(sp_dir): continue
        for chap in os.listdir(sp_dir):
            ch_dir = os.path.join(sp_dir, chap)
            if not os.path.isdir(ch_dir): continue
            tr = os.path.join(ch_dir, f"{spk}-{chap}.trans.txt")
            if not os.path.exists(tr): continue
            trans = {}
            for line in open(tr):
                parts = line.strip().split(" ", 1)
                if len(parts) == 2: trans[parts[0]] = parts[1]
            for fn in os.listdir(ch_dir):
                if fn.endswith(".flac"):
                    u = fn.replace(".flac", "")
                    if u in trans:
                        items.append({"audio_path": os.path.join(ch_dir, fn),
                                      "text": trans[u].lower()})
    return items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt", default=CKPT)
    args = ap.parse_args()

    device = args.device
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    sp = spm.SentencePieceProcessor(model_file=TOK)
    model = ConformerASR(N_MELS, cfg["vocab_size"], cfg["d_model"], cfg["n_heads"],
                         cfg["ff_dim"], cfg["n_layers"], cfg["conv_kernel"], 0.1).to(device)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

    fb = mel_fb(TARGET_SR, N_FFT, N_MELS); win = torch.hann_window(N_FFT)

    all_items = scan(LS)
    print(f"LibriSpeech test-clean: {len(all_items):,} utterances")
    random.seed(args.seed)
    items = random.sample(all_items, min(args.limit, len(all_items)))
    print(f"Evaluating random {len(items)} (seed={args.seed})")

    refs, hyps = [], []
    t0 = time.time()
    for i, e in enumerate(items):
        audio, sr = sf.read(e["audio_path"], dtype="float32")
        if audio.ndim > 1: audio = audio.mean(axis=1)
        if sr != TARGET_SR:
            r = TARGET_SR/sr
            audio = np.interp(np.linspace(0, len(audio)-1, int(len(audio)*r)),
                              np.arange(len(audio)), audio).astype(np.float32)
        peak = np.abs(audio).max()
        if peak > 0: audio = audio/peak
        wav = torch.from_numpy(audio).float()
        stft = torch.stft(wav, N_FFT, HOP, window=win, return_complex=True)
        mel = torch.log(torch.clamp(fb @ (stft.abs()**2), min=1e-10))
        if mel.shape[1] > MAX_FRAMES: mel = mel[:, :MAX_FRAMES]
        mel = mel.unsqueeze(0).to(device)
        with torch.no_grad():
            lp, _ = model(mel, torch.tensor([mel.shape[2]], device=device))
        hyp = ctc_greedy(lp[0].cpu(), sp)
        ref_n, hyp_n = normalize(e["text"]), normalize(hyp)
        if not ref_n: continue
        refs.append(ref_n); hyps.append(hyp_n)
        w = wer(ref_n, hyp_n) if hyp_n else 1.0
        if (i+1) % 10 == 0 or i < 5:
            print(f"[{i+1}/{len(items)}] WER={w*100:5.1f}%  REF: {ref_n[:60]}  | HYP: {hyp_n[:60]}")

    overall = wer(refs, hyps)
    per_utt = [wer(r, h) if h else 1.0 for r, h in zip(refs, hyps)]
    per_utt_pct = [round(x*100, 2) for x in per_utt]
    out = {
        "date": time.strftime("%Y-%m-%d"),
        "dataset": f"LibriSpeech test-clean (random {len(refs)} utterances, seed {args.seed})",
        "model": f"PRAXIS-AUSCULT Conformer-CTC, {sum(p.numel() for p in model.parameters())/1e6:.2f}M params",
        "decode": "CTC greedy, no LM",
        "device": device,
        "overall_corpus_wer_pct": round(overall*100, 2),
        "mean_utt_wer_pct": round(sum(per_utt_pct)/len(per_utt_pct), 2),
        "min_utt_wer_pct": min(per_utt_pct),
        "max_utt_wer_pct": max(per_utt_pct),
        "wall_time_min": round((time.time()-t0)/60, 2),
    }
    os.makedirs(os.path.join(ROOT, "data/processed"), exist_ok=True)
    with open(os.path.join(ROOT, "data/processed/wer_librispeech_testclean_v2.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n{'='*60}\nLibriSpeech test-clean WER: {overall*100:.2f}% ({len(refs)} utts, {(time.time()-t0)/60:.1f} min)\n{'='*60}")
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
