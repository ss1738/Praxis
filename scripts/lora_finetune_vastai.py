# NOTE: ConformerASR/ConvSubsampling/RelativePositionalEncoding are duplicated here for standalone vast.ai execution.
# Canonical version: src/speech_to_text/conformer_engine.py
"""
LoRA fine-tune on PriMock57 (+ optional LibriSpeech replay buffer).

Starts from /workspace/best_model_v2_120M.pt (the ~1,200h pre-trained base, NOT
the already-fine-tuned best_primock57.pt) so the LoRA delta is cleanly
attributable.

Freezes all 117M base params. Adds rank-16 LoRA adapters to every Conformer
block's MultiheadAttention (q/k/v fused in_proj + out_proj). Only ~2M params
become trainable.

If --replay_dir is given (pointing to a LibriSpeech-style flac tree, e.g.
dev-clean), 30% of each training batch is replaced by LibriSpeech utterances -
addresses catastrophic forgetting.

Run on vast.ai (RTX 4090 24GB):

    # LoRA only:
    python3 lora_finetune_vastai.py --epochs 25 --lr 5e-4 --batch 16

    # LoRA + replay (after uploading LibriSpeech dev-clean to /workspace/librispeech_replay):
    python3 lora_finetune_vastai.py --epochs 25 --lr 5e-4 --batch 16 \
        --replay_dir /workspace/librispeech_replay --replay_frac 0.3

Outputs:
  /workspace/checkpoints/best_lora.pt       (full merged state, convenient for eval)
  /workspace/checkpoints/lora_only.pt       (just LoRA deltas, ~8 MB)
  /workspace/checkpoints/final_lora.pt
  /workspace/checkpoints/lora_train_log.json
"""
import os, json, time, math, random, argparse, re, glob
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import soundfile as sf
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader

TARGET_SR = 16000
N_MELS, N_FFT, HOP = 80, 400, 160
MAX_AUDIO_FRAMES = 1500
CTC_BLANK = 0
TOKEN_OFFSET = 1
WORKSPACE = "/workspace"
CKPT_IN   = os.path.join(WORKSPACE, "best_model_v2_120M.pt")
TOK_PATH  = os.path.join(WORKSPACE, "praxis_tokenizer_v2.model")
TRAIN_JS  = os.path.join(WORKSPACE, "train.json")
VAL_JS    = os.path.join(WORKSPACE, "val.json")
AUDIO_DIR = os.path.join(WORKSPACE, "audio")
SAVE_DIR  = os.path.join(WORKSPACE, "checkpoints")
os.makedirs(SAVE_DIR, exist_ok=True)


# -------- model (verbatim, must match base checkpoint) --------
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


# -------- LoRA wrapper around nn.MultiheadAttention --------
class LoRAMultiheadAttention(nn.Module):
    """Wraps a frozen nn.MultiheadAttention; adds rank-r LoRA on in_proj (q,k,v fused) and out_proj."""
    def __init__(self, base: nn.MultiheadAttention, r: int = 16, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        d = base.embed_dim
        self.d, self.h = d, base.num_heads
        self.r = r
        self.scale = alpha / r
        self.lora_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.A_qkv = nn.Parameter(torch.zeros(r, d))
        self.B_qkv = nn.Parameter(torch.zeros(3 * d, r))
        self.A_o   = nn.Parameter(torch.zeros(r, d))
        self.B_o   = nn.Parameter(torch.zeros(d, r))
        nn.init.kaiming_uniform_(self.A_qkv, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.A_o,   a=math.sqrt(5))

    def forward(self, query, key, value, **kwargs):
        dW_qkv = (self.B_qkv @ self.lora_drop(self.A_qkv)) * self.scale
        dW_o   = (self.B_o   @ self.lora_drop(self.A_o))   * self.scale
        W_in  = self.base.in_proj_weight + dW_qkv
        W_out = self.base.out_proj.weight + dW_o
        q = query.transpose(0, 1); k = key.transpose(0, 1); v = value.transpose(0, 1)
        out, _ = F.multi_head_attention_forward(
            q, k, v,
            embed_dim_to_check=self.d,
            num_heads=self.h,
            in_proj_weight=W_in,
            in_proj_bias=self.base.in_proj_bias,
            bias_k=None, bias_v=None,
            add_zero_attn=False,
            dropout_p=0.0,
            out_proj_weight=W_out,
            out_proj_bias=self.base.out_proj.bias,
            training=self.training,
            key_padding_mask=None,
            need_weights=False,
            attn_mask=None,
        )
        return out.transpose(0, 1), None


def inject_lora(model: ConformerASR, r=16, alpha=16):
    for blk in model.layers:
        blk.attn = LoRAMultiheadAttention(blk.attn, r=r, alpha=alpha)
    for p in model.parameters():
        p.requires_grad = False
    for blk in model.layers:
        blk.attn.A_qkv.requires_grad = True
        blk.attn.B_qkv.requires_grad = True
        blk.attn.A_o.requires_grad   = True
        blk.attn.B_o.requires_grad   = True
    return model


def extract_lora_state(model: ConformerASR):
    out = {}
    for i, blk in enumerate(model.layers):
        out[f"layers.{i}.attn.A_qkv"] = blk.attn.A_qkv.data.clone()
        out[f"layers.{i}.attn.B_qkv"] = blk.attn.B_qkv.data.clone()
        out[f"layers.{i}.attn.A_o"]   = blk.attn.A_o.data.clone()
        out[f"layers.{i}.attn.B_o"]   = blk.attn.B_o.data.clone()
    return out


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


class SpeechDataset(Dataset):
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
            text = re.sub(r'<[^>]+/?>|</[^>]+>', ' ', text)
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


def remap_primock(samples):
    out = []
    for s in samples:
        fn = os.path.basename(s['audio_path'])
        s2 = dict(s); s2['audio_path'] = os.path.join(AUDIO_DIR, fn)
        if os.path.isfile(s2['audio_path']): out.append(s2)
    return out


def scan_librispeech(root):
    """Scan a LibriSpeech-style directory (speaker/chapter/*.flac + *.trans.txt)."""
    items = []
    if not os.path.isdir(root): return items
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


class MixedLoader:
    """Alternates batches from primary and replay loaders at target replay fraction."""
    def __init__(self, primary_loader, replay_loader, replay_frac):
        self.primary = primary_loader
        self.replay = replay_loader
        self.replay_frac = replay_frac
    def __len__(self):
        return len(self.primary)
    def __iter__(self):
        p_iter = iter(self.primary)
        r_iter = iter(self.replay) if self.replay is not None else None
        for i in range(len(self.primary)):
            if r_iter is not None and random.random() < self.replay_frac:
                try:
                    yield next(r_iter); continue
                except StopIteration:
                    r_iter = iter(self.replay)
                    yield next(r_iter); continue
            yield next(p_iter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--replay_dir", type=str, default="", help="LibriSpeech-style dir for replay")
    ap.add_argument("--replay_frac", type=float, default=0.3)
    args = ap.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB")
    random.seed(42); torch.manual_seed(42)

    sp = spm.SentencePieceProcessor(model_file=TOK_PATH)
    fb = mel_fb(TARGET_SR, N_FFT, N_MELS); win = torch.hann_window(N_FFT)

    train = remap_primock(json.load(open(TRAIN_JS)))
    val   = remap_primock(json.load(open(VAL_JS)))
    print(f"PriMock Train: {len(train)} | Val: {len(val)}")

    replay = []
    if args.replay_dir:
        replay = scan_librispeech(args.replay_dir)
        print(f"Replay (LibriSpeech) utterances: {len(replay)}")

    tr_loader = DataLoader(SpeechDataset(train, sp, fb, win, augment=True),
        batch_size=args.batch, shuffle=True, num_workers=4, collate_fn=collate, pin_memory=True, drop_last=True)
    va_loader = DataLoader(SpeechDataset(val, sp, fb, win, augment=False),
        batch_size=args.batch, shuffle=False, num_workers=4, collate_fn=collate, pin_memory=True)

    replay_loader = None
    if replay:
        replay_loader = DataLoader(SpeechDataset(replay, sp, fb, win, augment=True),
            batch_size=args.batch, shuffle=True, num_workers=2, collate_fn=collate, pin_memory=True, drop_last=True)

    mixed = MixedLoader(tr_loader, replay_loader, args.replay_frac)

    ckpt = torch.load(CKPT_IN, map_location='cuda', weights_only=False)
    cfg = ckpt['config']
    print(f"Loading base checkpoint, config: {cfg}")
    model = ConformerASR(N_MELS, cfg['vocab_size'], cfg['d_model'], cfg['n_heads'],
                         cfg['ff_dim'], cfg['n_layers'], cfg['conv_kernel'], 0.1).cuda()
    model.load_state_dict(ckpt['model_state'])
    total = sum(p.numel() for p in model.parameters())
    print(f"Base model params: {total/1e6:.2f}M")

    inject_lora(model, r=args.rank, alpha=args.alpha)
    model = model.cuda()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA rank={args.rank} alpha={args.alpha} trainable: {trainable/1e6:.3f}M ({100*trainable/total:.2f}% of base)")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.98))
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
        for mels, labels, ml, ll in mixed:
            if mels is None: continue
            mels, labels = mels.cuda(), labels.cuda()
            with torch.amp.autocast('cuda'):
                lp, ol = model(mels, ml)
                loss = ctc(lp.transpose(0,1), labels, ol, ll)
            if not torch.isfinite(loss): continue
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
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
            torch.save({'model_state': model.state_dict(), 'lora_state': extract_lora_state(model),
                        'config': cfg, 'lora': {'rank': args.rank, 'alpha': args.alpha},
                        'step': step, 'epoch': epoch, 'val_loss': vl,
                        'args': vars(args)},
                       os.path.join(SAVE_DIR, 'best_lora.pt'))
            torch.save({'lora_state': extract_lora_state(model),
                        'config': cfg, 'lora': {'rank': args.rank, 'alpha': args.alpha},
                        'step': step, 'epoch': epoch, 'val_loss': vl},
                       os.path.join(SAVE_DIR, 'lora_only.pt'))
            print("  >>> saved best_lora.pt + lora_only.pt")

    torch.save({'model_state': model.state_dict(), 'lora_state': extract_lora_state(model),
                'config': cfg, 'lora': {'rank': args.rank, 'alpha': args.alpha},
                'step': step, 'epoch': args.epochs,
                'train_losses': log['train_loss'], 'val_losses': log['val_loss'],
                'args': vars(args)},
               os.path.join(SAVE_DIR, 'final_lora.pt'))
    with open(os.path.join(SAVE_DIR, 'lora_train_log.json'), 'w') as f:
        json.dump({'best_val': best_val, 'wall_min': (time.time()-t0)/60,
                   'val_curve': log['val_loss'], 'args': vars(args)}, f, indent=2)
    print(f"\nDONE. Best val loss: {best_val:.4f}  Wall: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
