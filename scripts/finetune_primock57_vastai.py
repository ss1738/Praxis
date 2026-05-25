# NOTE: ConformerASR/ConvSubsampling/RelativePositionalEncoding are duplicated here for standalone vast.ai execution.
# Canonical version: src/speech_to_text/conformer_engine.py
"""
Fine-tune the 117M ConformerASR v2 checkpoint on PriMock57 train split.

Designed to run on vast.ai (RTX 4090 24GB or A100 40GB).

Expected layout on the vast.ai box (all at /workspace):
  /workspace/best_model_v2_80M.pt          (starting weights, 1.42GB)
  /workspace/praxis_tokenizer_v2.model
  /workspace/train.json                     (primock57 chunked train manifest)
  /workspace/val.json
  /workspace/audio/                         (the 5063 chunk wavs)

Rewrites audio_path entries that start with the Mac path prefix to /workspace/audio/.

Run:
  python3 finetune_primock57_vastai.py \
      --epochs 25 --lr 1e-5 --batch 16

Outputs:
  /workspace/checkpoints/best_primock57.pt      (best val loss)
  /workspace/checkpoints/final_primock57.pt
  /workspace/checkpoints/train_log.json
"""
import os, json, time, math, random, argparse, re
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import soundfile as sf
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader

# -------- config --------
TARGET_SR = 16000
N_MELS, N_FFT, HOP = 80, 400, 160
MAX_AUDIO_FRAMES = 1500
CTC_BLANK = 0
TOKEN_OFFSET = 1
WORKSPACE = "/workspace"
CKPT_IN   = os.path.join(WORKSPACE, "best_model_v2_80M.pt")
TOK_PATH  = os.path.join(WORKSPACE, "praxis_tokenizer_v2.model")
TRAIN_JS  = os.path.join(WORKSPACE, "train.json")
VAL_JS    = os.path.join(WORKSPACE, "val.json")
AUDIO_DIR = os.path.join(WORKSPACE, "audio")
SAVE_DIR  = os.path.join(WORKSPACE, "checkpoints")
os.makedirs(SAVE_DIR, exist_ok=True)

# -------- model (verbatim from vast_train.py) --------
class ConvSubsampling(nn.Module):
    def __init__(self, n_mels, d):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, d//2, 3, 2, 1), nn.ReLU(),
                                   nn.Conv2d(d//2, d, 3, 2, 1), nn.ReLU())
        self.linear = nn.Linear(d * (n_mels // 4), d)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        x = self.conv(x.unsqueeze(1)); b,c,f,t = x.shape
        return self.dropout(self.linear(x.permute(0,3,1,2).reshape(b, t, c*f)))

class RelativePositionalEncoding(nn.Module):
    def __init__(self, d, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0)/d))
        pe[:,0::2] = torch.sin(pos*div); pe[:,1::2] = torch.cos(pos*div)
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
        self.conv = nn.Sequential(nn.Conv1d(d, 2*d, 1), nn.GLU(dim=1),
            nn.Conv1d(d, d, k, padding=k//2, groups=d), nn.BatchNorm1d(d), nn.SiLU(),
            nn.Conv1d(d, d, 1), nn.Dropout(p))
        self.ff2 = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, ff), nn.SiLU(), nn.Dropout(p), nn.Linear(ff, d), nn.Dropout(p))
        self.final_norm = nn.LayerNorm(d)
    def forward(self, x):
        x = x + 0.5*self.ff1(x)
        r=x; xn=self.attn_norm(x); xa,_=self.attn(xn,xn,xn); x=r+self.attn_dropout(xa)
        r=x; xn=self.conv_norm(x); x=r+self.conv(xn.transpose(1,2)).transpose(1,2)
        return self.final_norm(x + 0.5*self.ff2(x))

class ConformerASR(nn.Module):
    def __init__(self, n_mels, vocab, d=512, h=8, ff=2048, n=18, k=31, p=0.1):
        super().__init__()
        self.subsampling = ConvSubsampling(n_mels, d)
        self.pos_enc = RelativePositionalEncoding(d)
        self.layers = nn.ModuleList([ConformerBlock(d,h,ff,k,p) for _ in range(n)])
        self.output = nn.Linear(d, vocab)
    def forward(self, x, xl):
        x = self.subsampling(x); x = self.pos_enc(x)
        for l in self.layers: x = l(x)
        return F.log_softmax(self.output(x), dim=-1), xl // 4

# -------- audio features --------
def mel_fb(sr, n_fft, n_mels, fmin=20.0, fmax=8000.0):
    def h2m(h): return 2595.0*np.log10(1.0+h/700.0)
    def m2h(m): return 700.0*(10.0**(m/2595.0)-1.0)
    pts = np.linspace(h2m(fmin), h2m(fmax), n_mels+2); hz = m2h(pts)
    bins = np.floor((n_fft+1)*hz/sr).astype(int)
    fb = np.zeros((n_mels, n_fft//2+1))
    for i in range(n_mels):
        l,c,r = bins[i], bins[i+1], bins[i+2]
        for j in range(l,c):
            if c!=l: fb[i,j] = (j-l)/(c-l)
        for j in range(c,r):
            if r!=c: fb[i,j] = (r-j)/(r-c)
    return torch.FloatTensor(fb)

class FTDataset(Dataset):
    def __init__(self, samples, sp, fb_cpu, win_cpu, augment=False):
        self.samples = samples; self.sp = sp; self.fb = fb_cpu; self.win = win_cpu
        self.augment = augment
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            audio, sr = sf.read(s['audio_path'], dtype='float32')
            if audio.ndim > 1: audio = audio.mean(axis=1)
            if sr != TARGET_SR:
                r = TARGET_SR/sr
                audio = np.interp(np.linspace(0, len(audio)-1, int(len(audio)*r)),
                                  np.arange(len(audio)), audio).astype(np.float32)
            if self.augment:
                f = random.choice([0.9, 0.95, 1.0, 1.05, 1.1])
                if f != 1.0:
                    new = int(len(audio)/f)
                    audio = np.interp(np.linspace(0,len(audio)-1,new), np.arange(len(audio)), audio).astype(np.float32)
            peak = np.abs(audio).max()
            if peak > 0: audio = audio/peak
            wav = torch.from_numpy(audio).float()
            stft = torch.stft(wav, N_FFT, HOP, window=self.win, return_complex=True)
            mel = torch.log(torch.clamp(self.fb @ (stft.abs()**2), min=1e-10))
            if mel.shape[1] > MAX_AUDIO_FRAMES: mel = mel[:, :MAX_AUDIO_FRAMES]
            if self.augment:
                for _ in range(2):
                    f = random.randint(0, 27); f0 = random.randint(0, max(0, N_MELS-f))
                    mel[f0:f0+f, :] = 0.0
                for _ in range(2):
                    t = random.randint(0, min(70, mel.shape[1]//4)); t0 = random.randint(0, max(0, mel.shape[1]-t))
                    mel[:, t0:t0+t] = 0.0
            text = (s.get('text') or s.get('transcript') or '').lower()
            text = re.sub(r'<[^>]+/?>|</[^>]+>', ' ', text)   # strip primock tags
            text = re.sub(r'\s+', ' ', text).strip()
            ids = [i + TOKEN_OFFSET for i in self.sp.encode(text, out_type=int)]
            if not ids: return None, None
            return mel, torch.LongTensor(ids)
        except Exception:
            return None, None

def collate(batch):
    batch = [(m,l) for m,l in batch if m is not None]
    if not batch: return None, None, None, None
    mels, labels = zip(*batch)
    mel_lens = torch.tensor([m.shape[1] for m in mels])
    lab_lens = torch.tensor([len(l) for l in labels])
    maxm, maxl = max(m.shape[1] for m in mels), max(len(l) for l in labels)
    pm = torch.zeros(len(mels), N_MELS, maxm)
    pl = torch.zeros(len(labels), maxl, dtype=torch.long)
    for i,(m,l) in enumerate(zip(mels, labels)):
        pm[i,:,:m.shape[1]] = m; pl[i,:len(l)] = l
    return pm, pl, mel_lens, lab_lens

def remap(samples):
    out = []
    for s in samples:
        p = s['audio_path']
        fn = os.path.basename(p)
        s2 = dict(s); s2['audio_path'] = os.path.join(AUDIO_DIR, fn)
        if os.path.isfile(s2['audio_path']): out.append(s2)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    args = ap.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB")
    random.seed(42); torch.manual_seed(42)

    sp = spm.SentencePieceProcessor(model_file=TOK_PATH)
    fb = mel_fb(TARGET_SR, N_FFT, N_MELS); win = torch.hann_window(N_FFT)

    train = remap(json.load(open(TRAIN_JS)))
    val   = remap(json.load(open(VAL_JS)))
    print(f"Train: {len(train)} | Val: {len(val)}")

    tr_loader = DataLoader(FTDataset(train, sp, fb, win, augment=True),
        batch_size=args.batch, shuffle=True, num_workers=4, collate_fn=collate, pin_memory=True, drop_last=True)
    va_loader = DataLoader(FTDataset(val, sp, fb, win, augment=False),
        batch_size=args.batch, shuffle=False, num_workers=4, collate_fn=collate, pin_memory=True)

    ckpt = torch.load(CKPT_IN, map_location='cuda', weights_only=False)
    cfg = ckpt['config']
    print(f"Loading base checkpoint, config: {cfg}")
    model = ConformerASR(N_MELS, cfg['vocab_size'], cfg['d_model'], cfg['n_heads'],
                         cfg['ff_dim'], cfg['n_layers'], cfg['conv_kernel'], 0.1).cuda()
    model.load_state_dict(ckpt['model_state'])
    print(f"Base model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
    scaler = torch.amp.GradScaler('cuda')
    ctc = nn.CTCLoss(blank=CTC_BLANK, zero_infinity=True)

    steps_per_epoch = len(tr_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup = int(total_steps * args.warmup_frac)
    def get_lr(step):
        if step < warmup: return args.lr * step / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return args.lr * max(0.02, 0.5 * (1 + math.cos(math.pi * prog)))

    def evaluate():
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for mels, labels, ml, ll in va_loader:
                if mels is None: continue
                mels, labels = mels.cuda(), labels.cuda()
                with torch.amp.autocast('cuda'):
                    lp, ol = model(mels, ml)
                    loss = ctc(lp.transpose(0,1), labels, ol, ll)
                if torch.isfinite(loss) and loss.item() > 0.01:
                    tot += loss.item(); n += 1
        model.train()
        return tot/n if n else float('inf')

    print(f"Steps/epoch: {steps_per_epoch}  Total steps: {total_steps}  Warmup: {warmup}")
    best_val = float('inf'); step = 0; log = {'train_loss': [], 'val_loss': []}
    t0 = time.time()
    model.train()
    for epoch in range(args.epochs):
        ep_loss, ep_n = 0.0, 0
        for mels, labels, ml, ll in tr_loader:
            if mels is None: continue
            mels, labels = mels.cuda(), labels.cuda()
            with torch.amp.autocast('cuda'):
                lp, ol = model(mels, ml)
                loss = ctc(lp.transpose(0,1), labels, ol, ll)
            if not torch.isfinite(loss): continue
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            lr = get_lr(step)
            for pg in opt.param_groups: pg['lr'] = lr
            scaler.step(opt); scaler.update()
            ep_loss += loss.item(); ep_n += 1; step += 1
            log['train_loss'].append(loss.item())
            if step % 100 == 0:
                avg = sum(log['train_loss'][-100:]) / 100
                print(f"  step {step}/{total_steps}  loss={avg:.4f}  lr={lr:.2e}  {(time.time()-t0)/60:.1f}m")
        vl = evaluate()
        log['val_loss'].append((step, vl))
        print(f"  epoch {epoch+1}/{args.epochs}  train={ep_loss/max(ep_n,1):.4f}  val={vl:.4f}  best={best_val:.4f}")
        if vl < best_val:
            best_val = vl
            torch.save({'model_state': model.state_dict(), 'config': cfg, 'step': step,
                        'epoch': epoch, 'val_loss': vl},
                       os.path.join(SAVE_DIR, 'best_primock57.pt'))
            print("  >>> saved best_primock57.pt")

    torch.save({'model_state': model.state_dict(), 'config': cfg, 'step': step,
                'epoch': args.epochs, 'train_losses': log['train_loss'], 'val_losses': log['val_loss']},
               os.path.join(SAVE_DIR, 'final_primock57.pt'))
    with open(os.path.join(SAVE_DIR, 'train_log.json'), 'w') as f:
        json.dump({'best_val': best_val, 'wall_min': (time.time()-t0)/60,
                   'val_curve': log['val_loss'], 'args': vars(args)}, f, indent=2)
    print(f"\nDONE. Best val loss: {best_val:.4f}  Wall: {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
