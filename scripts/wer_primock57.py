# NOTE: ConformerASR/ConvSubsampling/RelativePositionalEncoding are duplicated here for standalone vast.ai execution.
# Canonical version: src/speech_to_text/conformer_engine.py
"""
WER evaluation for the 119.6M ConformerASR v2 checkpoint on PriMock57 test split.

Loads the checkpoint, reconstructs the model from its saved config, chunks each
full consultation audio into ~15s windows, runs CTC greedy decoding, and reports
WER against the reference transcript. CPU-only inference so it runs on the MacBook.

Usage:
    python3 scripts/wer_primock57.py
    python3 scripts/wer_primock57.py --limit 3      # quick smoke test
    python3 scripts/wer_primock57.py --device cuda  # on GPU if available
"""
import os, json, math, re, argparse, time
import pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import sentencepiece as spm
from jiwer import wer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "data/models/conformer_v2/final_model_v2_80M.pt")
TOK  = os.path.join(ROOT, "data/models/conformer_v2/praxis_tokenizer_v2.model")
TEST = os.path.join(ROOT, "data/processed/primock57_chunked/test.json")

TARGET_SR = 16000
N_MELS, N_FFT, HOP = 80, 400, 160
MAX_FRAMES = 1500            # ~15s - matches training cap
CTC_BLANK = 0
TOKEN_OFFSET = 1

# --- model (verbatim from vast_train.py) ---
class ConvSubsampling(nn.Module):
    def __init__(self, n_mels, d_model):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, d_model//2, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(d_model//2, d_model, 3, stride=2, padding=1), nn.ReLU())
        self.linear = nn.Linear(d_model * (n_mels // 4), d_model)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        x = self.conv(x.unsqueeze(1))
        b, c, f, t = x.shape
        return self.dropout(self.linear(x.permute(0, 3, 1, 2).reshape(b, t, c * f)))

class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos*div); pe[:, 1::2] = torch.cos(pos*div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.shape[1], :]

class ConformerBlock(nn.Module):
    def __init__(self, d, h, ff, k=31, p=0.1):
        super().__init__()
        self.ff1 = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, ff), nn.SiLU(), nn.Dropout(p), nn.Linear(ff, d), nn.Dropout(p))
        self.attn_norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        self.attn_dropout = nn.Dropout(p)
        self.conv_norm = nn.LayerNorm(d)
        self.conv = nn.Sequential(
            nn.Conv1d(d, 2*d, 1), nn.GLU(dim=1),
            nn.Conv1d(d, d, k, padding=k//2, groups=d),
            nn.BatchNorm1d(d), nn.SiLU(), nn.Conv1d(d, d, 1), nn.Dropout(p))
        self.ff2 = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, ff), nn.SiLU(), nn.Dropout(p), nn.Linear(ff, d), nn.Dropout(p))
        self.final_norm = nn.LayerNorm(d)
    def forward(self, x):
        x = x + 0.5*self.ff1(x)
        r = x; xn = self.attn_norm(x); xa,_ = self.attn(xn, xn, xn); x = r + self.attn_dropout(xa)
        r = x; xn = self.conv_norm(x); x = r + self.conv(xn.transpose(1,2)).transpose(1,2)
        x = x + 0.5*self.ff2(x)
        return self.final_norm(x)

class ConformerASR(nn.Module):
    def __init__(self, n_mels, vocab, d=512, h=8, ff=2048, n=18, k=31, p=0.1):
        super().__init__()
        self.subsampling = ConvSubsampling(n_mels, d)
        self.pos_enc = RelativePositionalEncoding(d)
        self.layers = nn.ModuleList([ConformerBlock(d, h, ff, k, p) for _ in range(n)])
        self.output = nn.Linear(d, vocab)
    def forward(self, x, x_lens):
        x = self.subsampling(x); x = self.pos_enc(x)
        for l in self.layers: x = l(x)
        return F.log_softmax(self.output(x), dim=-1), x_lens // 4

# --- mel extraction ---
def mel_fb(sr, n_fft, n_mels, fmin=20.0, fmax=8000.0):
    def h2m(h): return 2595.0 * np.log10(1.0 + h/700.0)
    def m2h(m): return 700.0 * (10.0**(m/2595.0) - 1.0)
    pts = np.linspace(h2m(fmin), h2m(fmax), n_mels+2); hz = m2h(pts)
    bins = np.floor((n_fft+1)*hz/sr).astype(int)
    fb = np.zeros((n_mels, n_fft//2+1))
    for i in range(n_mels):
        l, c, r = bins[i], bins[i+1], bins[i+2]
        for j in range(l, c):
            if c != l: fb[i, j] = (j-l)/(c-l)
        for j in range(c, r):
            if r != c: fb[i, j] = (r-j)/(r-c)
    return torch.FloatTensor(fb)

# --- text normalisation ---
TAG_RE = re.compile(r"<[^>]+/?>|</[^>]+>")
NONWORD_RE = re.compile(r"[^a-z0-9' ]+")
WS_RE = re.compile(r"\s+")
def normalize(t: str) -> str:
    t = t.lower()
    t = TAG_RE.sub(" ", t)          # strip <UNSURE>, <INAUDIBLE_SPEECH/>, etc
    t = NONWORD_RE.sub(" ", t)
    t = WS_RE.sub(" ", t).strip()
    return t

# --- decoding ---
def ctc_greedy(log_probs, sp):
    ids = log_probs.argmax(-1).tolist()
    out, prev = [], -1
    for i in ids:
        if i != CTC_BLANK and i != prev: out.append(i - TOKEN_OFFSET)
        prev = i
    out = [d for d in out if 0 <= d < sp.get_piece_size()]
    return sp.decode(out) if out else ""

def chunked_transcribe(model, audio, sr, fb_cpu, window_cpu, device, chunk_s=14.0, overlap_s=0.0, sp=None):
    # audio: float32 np array @ 16k; chunk in 14s windows, decode independently, concatenate.
    samples_per_chunk = int(chunk_s * sr)
    step = samples_per_chunk - int(overlap_s * sr)
    pieces = []
    for start in range(0, len(audio), step):
        seg = audio[start:start+samples_per_chunk]
        if len(seg) < sr * 0.5:   # skip <0.5s tail
            continue
        wav = torch.from_numpy(seg).float()
        stft = torch.stft(wav, N_FFT, HOP, window=window_cpu, return_complex=True)
        power = stft.abs()**2
        mel = torch.log(torch.clamp(fb_cpu @ power, min=1e-10))
        if mel.shape[1] > MAX_FRAMES: mel = mel[:, :MAX_FRAMES]
        mel = mel.unsqueeze(0).to(device)
        with torch.no_grad():
            lp, _ = model(mel, torch.tensor([mel.shape[2]], device=device))
        pieces.append(ctc_greedy(lp[0].cpu(), sp))
    return " ".join(p for p in pieces if p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="Only eval first N entries (0 = all)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ckpt", default=CKPT)
    args = ap.parse_args()

    device = args.device
    print(f"Device: {device}")
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    print(f"Config: {cfg}")

    sp = spm.SentencePieceProcessor(model_file=TOK)
    print(f"Tokenizer: {sp.get_piece_size()} pieces")

    model = ConformerASR(N_MELS, cfg["vocab_size"], cfg["d_model"], cfg["n_heads"],
                         cfg["ff_dim"], cfg["n_layers"], cfg["conv_kernel"], 0.1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    total = sum(p.numel() for p in model.parameters())
    print(f"Model: {total/1e6:.2f}M params")

    fb_cpu = mel_fb(TARGET_SR, N_FFT, N_MELS)
    window_cpu = torch.hann_window(N_FFT)

    entries = json.load(open(TEST))
    if args.limit: entries = entries[:args.limit]
    print(f"Evaluating {len(entries)} entries from PriMock57 test split")

    refs, hyps = [], []
    t0 = time.time()
    for i, e in enumerate(entries):
        audio_path = str(pathlib.Path(e["audio_path"]).resolve()) if os.path.isabs(e["audio_path"]) else str(pathlib.Path(os.environ.get("PRAXIS_DATA_ROOT", ".")).joinpath(e["audio_path"]))
        audio, sr = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1: audio = audio.mean(axis=1)
        if sr != TARGET_SR:
            ratio = TARGET_SR / sr
            audio = np.interp(np.linspace(0, len(audio)-1, int(len(audio)*ratio)),
                              np.arange(len(audio)), audio).astype(np.float32)
        peak = np.abs(audio).max()
        if peak > 0: audio = audio/peak

        t_start = time.time()
        hyp = chunked_transcribe(model, audio, TARGET_SR, fb_cpu, window_cpu, device, sp=sp)
        dur_audio = len(audio)/TARGET_SR
        dur_wall = time.time() - t_start

        ref_n = normalize(e["text"])
        hyp_n = normalize(hyp)
        if not ref_n: continue
        refs.append(ref_n); hyps.append(hyp_n)
        w = wer(ref_n, hyp_n) if hyp_n else 1.0
        print(f"[{i+1}/{len(entries)}] {os.path.basename(e['audio_path']):40s} "
              f"audio={dur_audio:6.1f}s wall={dur_wall:6.1f}s WER={w*100:5.1f}%")
        print(f"   REF: {ref_n[:120]}")
        print(f"   HYP: {hyp_n[:120]}")

    overall = wer(refs, hyps) if refs else float("nan")
    print(f"\n{'='*60}")
    print(f"Overall WER on PriMock57 test ({len(refs)} files): {overall*100:.2f}%")
    print(f"Wall time: {(time.time()-t0)/60:.1f} min")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
