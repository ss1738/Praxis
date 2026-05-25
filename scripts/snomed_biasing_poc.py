# NOTE: ConformerASR/ConvSubsampling/RelativePositionalEncoding are duplicated here for standalone vast.ai execution.
# Canonical version: src/speech_to_text/conformer_engine.py
"""
SNOMED-CT-UK / dm+d / BNF contextual-biasing proof-of-concept on PriMock57 test.

Runs the v1 FT Conformer-CTC 117M checkpoint (best_primock57.pt) with and without
hotword-boosted CTC beam search, and reports:
  - Overall WER
  - Medical-term WER (MC-WER): recall of lexicon terms present in the reference
  - Per-file WER array (14 tracks, same layout as wer_primock57_ft.json)

Default lexicon = 180 curated UK primary-care terms (drugs/conditions/anatomy) embedded
below + any extra terms mined from PriMock57 train transcripts that match common medical
patterns. Pass --lexicon PATH to override with a real SNOMED-CT-UK / dm+d / BNF dump
(plaintext, one term per line).

Usage on vast.ai:
    pip install -U pyctcdecode
    python3 scripts/snomed_biasing_poc.py --device cuda                    # biased decode
    python3 scripts/snomed_biasing_poc.py --device cuda --no-biasing       # baseline (plain beam)
    python3 scripts/snomed_biasing_poc.py --device cuda --lexicon data/lexicons/snomed_uk.txt
    python3 scripts/snomed_biasing_poc.py --device cuda --boost 10.0       # sweep boost strength

Writes: data/processed/snomed_biasing_<tag>.json
"""
import os, sys, json, re, math, argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import sentencepiece as spm
from jiwer import wer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "data/models/conformer_v2/best_primock57.pt")
TOK  = os.path.join(ROOT, "data/models/conformer_v2/praxis_tokenizer_v2.model")
TEST = os.path.join(ROOT, "data/processed/primock57_chunked/test.json")
TRAIN = os.path.join(ROOT, "data/processed/primock57_chunked/train.json")
OUT_DIR = os.path.join(ROOT, "data/processed")

TARGET_SR = 16000
N_MELS, N_FFT, HOP = 80, 400, 160
MAX_FRAMES = 1500
CTC_BLANK = 0
TOKEN_OFFSET = 1
CHUNK_S = 14.0

# --- curated UK primary-care lexicon (seed; override via --lexicon)
CURATED = """
paracetamol ibuprofen aspirin amoxicillin flucloxacillin doxycycline clarithromycin
erythromycin azithromycin metronidazole ciprofloxacin trimethoprim nitrofurantoin
omeprazole lansoprazole ranitidine simvastatin atorvastatin rosuvastatin
ramipril lisinopril perindopril amlodipine bisoprolol atenolol bendroflumethiazide
furosemide spironolactone warfarin apixaban rivaroxaban clopidogrel ticagrelor
metformin gliclazide sitagliptin empagliflozin dapagliflozin insulin levothyroxine
prednisolone hydrocortisone beclometasone fluticasone salbutamol salmeterol
tiotropium ipratropium montelukast cetirizine loratadine fexofenadine piriton
chlorphenamine diazepam lorazepam sertraline citalopram fluoxetine mirtazapine
venlafaxine amitriptyline nortriptyline pregabalin gabapentin tramadol codeine
morphine naproxen diclofenac tamsulosin finasteride sildenafil tadalafil
levonorgestrel microgynon cerazette noriday depo-provera mirena nexplanon
loratadine piriton emollient oilatum aveeno cetraben diprobase epaderm betnovate
dermovate eumovate hydromol zerobase
eczema dermatitis psoriasis urticaria impetigo cellulitis shingles tinea
asthma copd bronchitis pneumonia sinusitis pharyngitis tonsillitis laryngitis
otitis conjunctivitis blepharitis glaucoma cataract stye chalazion
hypertension hypotension angina arrhythmia atrial fibrillation palpitations
diabetes hypoglycaemia hyperglycaemia thyrotoxicosis hypothyroidism
migraine tension headache vertigo tinnitus bppv sciatica fibromyalgia
gastritis oesophagitis peptic ulcer gord reflux constipation diarrhoea haemorrhoids
urticaria allergic rhinitis hayfever eczema dermatitis herpes varicella
cystitis pyelonephritis prostatitis endometriosis dysmenorrhoea menorrhagia
bnf nice cks qof snomed gp ecg fbc u and e crp esr hba1c tsh ldl hdl
"""

def load_curated():
    out = set()
    for line in CURATED.strip().splitlines():
        for w in line.split():
            w = w.strip().lower()
            if w and len(w) >= 3:
                out.add(w)
    return out

# --- text normalisation (verbatim from wer_primock57.py) ---
TAG_RE = re.compile(r"<[^>]+/?>|</[^>]+>")
NONWORD_RE = re.compile(r"[^a-z0-9' ]+")
WS_RE = re.compile(r"\s+")
def normalize(t: str) -> str:
    t = t.lower()
    t = TAG_RE.sub(" ", t)
    t = NONWORD_RE.sub(" ", t)
    t = WS_RE.sub(" ", t).strip()
    return t

# --- model (verbatim from wer_primock57.py) ---
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

# --- decoding ---
def ctc_greedy(log_probs, sp):
    ids = log_probs.argmax(-1).tolist()
    out, prev = [], -1
    for i in ids:
        if i != CTC_BLANK and i != prev:
            out.append(i - TOKEN_OFFSET)
        prev = i
    out = [d for d in out if 0 <= d < sp.get_piece_size()]
    return sp.decode(out) if out else ""

def build_labels(sp):
    vocab = [""]
    for i in range(sp.get_piece_size()):
        p = sp.id_to_piece(i)
        vocab.append(p.replace("\u2581", " "))
    return vocab

def ctc_hotword_beam(log_probs_np, decoder, hotwords, boost, sp):
    """pyctcdecode hotword-boosted beam search. log_probs_np: (T, V) numpy."""
    text = decoder.decode(
        log_probs_np,
        hotwords=hotwords,
        hotword_weight=boost,
        beam_width=100,
    )
    return text.strip()

def transcribe(model, audio, fb, win, device, sp, decoder, hotwords, boost, biasing):
    samples_per_chunk = int(CHUNK_S * TARGET_SR)
    pieces = []
    for start in range(0, len(audio), samples_per_chunk):
        seg = audio[start:start + samples_per_chunk]
        if len(seg) < TARGET_SR * 0.5:
            continue
        wav = torch.from_numpy(seg).float()
        stft = torch.stft(wav, N_FFT, HOP, window=win, return_complex=True)
        power = stft.abs() ** 2
        mel = torch.log(torch.clamp(fb @ power, min=1e-10))
        if mel.shape[1] > MAX_FRAMES: mel = mel[:, :MAX_FRAMES]
        mel = mel.unsqueeze(0).to(device)
        with torch.no_grad():
            lp, _ = model(mel, torch.tensor([mel.shape[2]], device=device))
        if biasing:
            piece = ctc_hotword_beam(lp[0].cpu().numpy(), decoder, hotwords, boost, sp)
        else:
            piece = ctc_greedy(lp[0].cpu(), sp)
        pieces.append(piece)
    return " ".join(p for p in pieces if p)

# --- medical-term WER (MC-WER): per-term recall, case-insensitive, whole-word ---
def medical_term_recall(refs, hyps, lexicon):
    """For each ref-hyp pair, count lexicon terms present in ref and how many appear in hyp."""
    total_ref_hits = 0
    total_recovered = 0
    missed = []
    for ref, hyp in zip(refs, hyps):
        ref_words = ref.split()
        hyp_words = set(hyp.split())
        for w in ref_words:
            if w in lexicon:
                total_ref_hits += 1
                if w in hyp_words:
                    total_recovered += 1
                else:
                    missed.append(w)
    return total_ref_hits, total_recovered, missed

def build_lexicon(path=None, mine_train=True):
    terms = load_curated()
    if path and os.path.exists(path):
        with open(path) as f:
            for line in f:
                w = line.strip().lower()
                if w and len(w) >= 3:
                    terms.add(w)
        print(f"Loaded {len(terms)} terms (curated + {path})")
    if mine_train and os.path.exists(TRAIN):
        mined = 0
        suffixes = ("itis", "osis", "emia", "aemia", "oma", "rrhoea", "phagia",
                    "pathy", "ectomy", "otomy", "plasty", "algia", "gram", "scopy",
                    "cillin", "azole", "olol", "pril", "sartan", "statin", "zepam",
                    "profen", "caine", "mycin", "cycline", "pramine", "dipine")
        for e in json.load(open(TRAIN)):
            for w in normalize(e["text"]).split():
                if len(w) >= 5 and w.endswith(suffixes) and w not in terms:
                    terms.add(w); mined += 1
        print(f"Mined {mined} additional suffix-matched terms from train set")
    print(f"Final lexicon size: {len(terms)}")
    return sorted(terms)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lexicon", default="", help="optional plaintext lexicon (one term/line)")
    ap.add_argument("--boost", type=float, default=8.0, help="hotword boost weight (pyctcdecode)")
    ap.add_argument("--no-biasing", action="store_true", help="baseline: plain greedy, no biasing")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    tag = args.tag or ("baseline_greedy" if args.no_biasing else f"biased_boost{args.boost:g}")
    out_path = os.path.join(OUT_DIR, f"snomed_biasing_{tag}.json")

    device = args.device
    print(f"=== SNOMED biasing POC - tag={tag} ===")
    print(f"Device: {device}  |  Checkpoint: {args.ckpt}")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    print(f"Config: {cfg}")

    sp = spm.SentencePieceProcessor(model_file=TOK)
    model = ConformerASR(N_MELS, cfg["vocab_size"], cfg["d_model"], cfg["n_heads"],
                         cfg["ff_dim"], cfg["n_layers"], cfg["conv_kernel"], 0.1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    total = sum(p.numel() for p in model.parameters())
    print(f"Model: {total/1e6:.2f}M params")

    lex_terms = build_lexicon(args.lexicon or None, mine_train=True)
    lex_set = set(lex_terms)

    decoder = None
    if not args.no_biasing:
        from pyctcdecode import build_ctcdecoder
        labels = build_labels(sp)
        decoder = build_ctcdecoder(labels)
        print(f"pyctcdecode decoder built | labels={len(labels)} | hotwords={len(lex_terms)} | boost={args.boost}")

    fb = mel_fb(TARGET_SR, N_FFT, N_MELS)
    win = torch.hann_window(N_FFT)

    entries = json.load(open(TEST))
    if args.limit: entries = entries[:args.limit]
    print(f"Evaluating {len(entries)} tracks")

    refs, hyps, per_file = [], [], []
    t0 = time.time()
    for i, e in enumerate(entries):
        audio, sr = sf.read(e["audio_path"], dtype="float32")
        if audio.ndim > 1: audio = audio.mean(axis=1)
        if sr != TARGET_SR:
            ratio = TARGET_SR / sr
            audio = np.interp(np.linspace(0, len(audio)-1, int(len(audio)*ratio)),
                              np.arange(len(audio)), audio).astype(np.float32)
        peak = np.abs(audio).max()
        if peak > 0: audio = audio / peak

        t_s = time.time()
        hyp = transcribe(model, audio, fb, win, device, sp, decoder,
                         lex_terms, args.boost, biasing=not args.no_biasing)
        dt = time.time() - t_s

        ref_n = normalize(e["text"])
        hyp_n = normalize(hyp)
        if not ref_n: continue
        refs.append(ref_n); hyps.append(hyp_n)
        w = wer(ref_n, hyp_n) if hyp_n else 1.0
        per_file.append(round(w*100, 2))
        print(f"[{i+1:2d}/{len(entries)}] {os.path.basename(e['audio_path']):55s} "
              f"wall={dt:6.1f}s WER={w*100:5.1f}%")

    overall = wer(refs, hyps) if refs else float("nan")
    hits, recovered, missed = medical_term_recall(refs, hyps, lex_set)
    mc_recall = recovered / hits if hits else float("nan")
    mc_err = 1.0 - mc_recall if hits else float("nan")
    wall_min = (time.time() - t0) / 60

    print(f"\n{'='*60}")
    print(f"Tag:            {tag}")
    print(f"Overall WER:    {overall*100:.2f}%  ({len(refs)} tracks)")
    print(f"Lexicon size:   {len(lex_terms)}")
    print(f"Ref lex-hits:   {hits}   recovered: {recovered}   MC-recall: {mc_recall*100:.2f}%   MC-err: {mc_err*100:.2f}%")
    print(f"Top 30 missed:  {sorted(set(missed))[:30]}")
    print(f"Wall time:      {wall_min:.1f} min")
    print(f"{'='*60}")

    result = {
        "date": time.strftime("%Y-%m-%d"),
        "dataset": "PriMock57 test split (14 per-speaker tracks)",
        "checkpoint": os.path.basename(args.ckpt),
        "tag": tag,
        "biasing_enabled": not args.no_biasing,
        "boost": args.boost if not args.no_biasing else None,
        "lexicon_size": len(lex_terms),
        "lexicon_source": args.lexicon or "curated+mined",
        "per_file_wer_pct": per_file,
        "overall_corpus_wer_pct": round(overall*100, 2),
        "mc_recall_pct": round(mc_recall*100, 2) if hits else None,
        "mc_err_pct":    round(mc_err*100, 2)    if hits else None,
        "ref_lexicon_hits": hits,
        "recovered_hits":   recovered,
        "wall_time_min": round(wall_min, 2),
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
