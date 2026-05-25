"""PRAXIS-AUSCULT: custom from-scratch Conformer-CTC ASR for UK clinical audio.

Named after Latin *auscultare* ("to listen carefully") - root of *auscultation*,
the foundational act of clinical listening.

Architecture trained on UK-accented English + LibriSpeech.
117M live params (119.6M in checkpoint state-dict; ~2.5M training-only params stripped at load).
18 Conformer blocks, d_model=512, 8 heads, FFN=2048, conv_kernel=31.
Tokenizer: SentencePiece BPE 3000 + CTC blank.
HF: Satyawan1/praxis-auscult.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


N_MELS = 80
N_FFT = 400
HOP_LENGTH = 160
TARGET_SR = 16_000
CTC_BLANK = 0
TOKEN_OFFSET = 1


def _create_mel_filterbank(sr=TARGET_SR, n_fft=N_FFT, n_mels=N_MELS, fmin=20.0, fmax=8000.0):
    def hz_to_mel(hz): return 2595.0 * np.log10(1.0 + hz / 700.0)
    def mel_to_hz(mel): return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        for j in range(l, c):
            if c != l: fb[i, j] = (j - l) / (c - l)
        for j in range(c, r):
            if r != c: fb[i, j] = (r - j) / (r - c)
    return torch.FloatTensor(fb)


class _ConvSubsampling(nn.Module):
    def __init__(self, n_mels, d_model):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, d_model // 2, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(d_model // 2, d_model, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.linear = nn.Linear(d_model * (n_mels // 4), d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.conv(x.unsqueeze(1))
        b, c, f, t = x.shape
        return self.dropout(self.linear(x.permute(0, 3, 1, 2).reshape(b, t, c * f)))


class _RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.register_buffer("pe", self._build(max_len, d_model), persistent=False)

    @staticmethod
    def _build(max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len > self.pe.shape[1]:
            new_pe = self._build(seq_len, self.d_model).to(self.pe.device, dtype=self.pe.dtype)
            self.pe = new_pe
        return x + self.pe[:, :seq_len, :]


class _ConformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, conv_kernel=31, dropout=0.1):
        super().__init__()
        self.ff1 = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, ff_dim), nn.SiLU(),
                                 nn.Dropout(dropout), nn.Linear(ff_dim, d_model), nn.Dropout(dropout))
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.conv_norm = nn.LayerNorm(d_model)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, 1), nn.GLU(dim=1),
            nn.Conv1d(d_model, d_model, conv_kernel, padding=conv_kernel // 2, groups=d_model),
            nn.BatchNorm1d(d_model), nn.SiLU(), nn.Conv1d(d_model, d_model, 1), nn.Dropout(dropout),
        )
        self.ff2 = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, ff_dim), nn.SiLU(),
                                 nn.Dropout(dropout), nn.Linear(ff_dim, d_model), nn.Dropout(dropout))
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + 0.5 * self.ff1(x)
        residual = x
        x_norm = self.attn_norm(x)
        x_attn, _ = self.attn(x_norm, x_norm, x_norm)
        x = residual + self.attn_dropout(x_attn)
        residual = x
        x_norm = self.conv_norm(x)
        x = residual + self.conv(x_norm.transpose(1, 2)).transpose(1, 2)
        x = x + 0.5 * self.ff2(x)
        return self.final_norm(x)


class _ConformerASR(nn.Module):
    def __init__(self, n_mels, vocab_size, d_model=512, n_heads=8, ff_dim=2048, n_layers=18,
                 conv_kernel=31, dropout=0.1):
        super().__init__()
        self.subsampling = _ConvSubsampling(n_mels, d_model)
        self.pos_enc = _RelativePositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            _ConformerBlock(d_model, n_heads, ff_dim, conv_kernel, dropout) for _ in range(n_layers)
        ])
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.subsampling(x)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x)
        return F.log_softmax(self.output(x), dim=-1)


_DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / "models" / "conformer_v2"


class ConformerEngine:
    """PRAXIS-Conformer-120M - custom from-scratch Conformer-CTC."""

    def __init__(
        self,
        model_dir: str | Path | None = None,
        device: str | None = None,
        bias: bool = False,
        bias_terms: list[str] | None = None,
        bias_weight: float = 8.0,
        beam_width: int = 100,
    ):
        self._dir = Path(model_dir) if model_dir else _DEFAULT_DIR
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: _ConformerASR | None = None
        self._sp = None
        self._mel_fb: torch.Tensor | None = None
        self._window: torch.Tensor | None = None
        self._bias = bias
        self._bias_terms = list(bias_terms) if bias_terms else None
        self._bias_weight = bias_weight
        self._beam_width = beam_width
        self._biaser = None
        logger.info(
            f"Conformer engine initialised (dir={self._dir}, device={self._device}, "
            f"bias={bias}, bias_weight={bias_weight if bias else None})"
        )

    def load_model(self) -> None:
        if self._model is not None:
            return
        import sentencepiece as spm

        primock_path = self._dir / "best_primock57.pt"
        best_path = self._dir / "best_model_v2_80M.pt"
        final_path = self._dir / "final_model_v2_80M.pt"
        ckpt_path = primock_path if primock_path.exists() else (best_path if best_path.exists() else final_path)
        tok_path = self._dir / "praxis_tokenizer_v2.model"
        if not ckpt_path.exists() or not tok_path.exists():
            raise FileNotFoundError(f"Conformer checkpoint or tokenizer missing in {self._dir}")
        logger.info(f"Loading Conformer checkpoint: {ckpt_path.name}")

        self._sp = spm.SentencePieceProcessor(model_file=str(tok_path))
        vocab_size = self._sp.get_piece_size() + TOKEN_OFFSET

        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {})
        model = _ConformerASR(
            n_mels=N_MELS,
            vocab_size=cfg.get("vocab_size", vocab_size),
            d_model=cfg.get("d_model", 512),
            n_heads=cfg.get("n_heads", 8),
            ff_dim=cfg.get("ff_dim", 2048),
            n_layers=cfg.get("n_layers", 18),
            conv_kernel=cfg.get("conv_kernel", 31),
            dropout=0.0,
        )
        state_key = "model_state_dict" if "model_state_dict" in ckpt else "model_state"
        model.load_state_dict(ckpt[state_key], strict=False)
        model.eval().to(self._device)
        self._model = model

        self._mel_fb = _create_mel_filterbank().to(self._device)
        self._window = torch.hann_window(N_FFT).to(self._device)

        if self._bias:
            self._build_bias_decoder()

        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"PRAXIS-AUSCULT loaded: {n_params/1e6:.2f}M params, vocab={vocab_size}")

    def _build_bias_decoder(self) -> None:
        """Build SnomedBiaser for hotword-boosted CTC beam search."""
        from src.speech_to_text.snomed_biasing import SnomedBiaser, SnomedLexicon

        lexicon = SnomedLexicon()
        if self._bias_terms:
            lexicon._terms.update(t.lower() for t in self._bias_terms)

        labels = [""]
        for i in range(self._sp.get_piece_size()):
            piece = self._sp.id_to_piece(i)
            labels.append(piece.replace("▁", " "))

        biaser = SnomedBiaser(lexicon, boost=self._bias_weight, beam_width=self._beam_width)
        if biaser.build_decoder(labels):
            self._biaser = biaser
        else:
            logger.warning("pyctcdecode unavailable - SNOMED biasing disabled, using greedy decode")
            self._bias = False

    def _audio_to_mel(self, audio: np.ndarray, sample_rate: int) -> torch.Tensor:
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != TARGET_SR:
            ratio = TARGET_SR / sample_rate
            new_len = int(len(audio) * ratio)
            audio = np.interp(np.linspace(0, len(audio) - 1, new_len),
                              np.arange(len(audio)), audio).astype(np.float32)
        peak = float(np.abs(audio).max() or 0.0)
        if peak > 0:
            audio = audio / peak
        wav = torch.from_numpy(audio.astype(np.float32)).to(self._device)
        stft = torch.stft(wav, N_FFT, HOP_LENGTH, window=self._window, return_complex=True)
        power = stft.abs() ** 2
        mel = torch.log(torch.clamp(self._mel_fb @ power, min=1e-10))
        return mel.unsqueeze(0)  # (1, n_mels, time)

    def _ctc_greedy_decode(self, log_probs: torch.Tensor) -> str:
        ids = log_probs.argmax(-1).squeeze(0).tolist()
        out, prev = [], -1
        for i in ids:
            if i != CTC_BLANK and i != prev:
                out.append(i - TOKEN_OFFSET)
            prev = i
        out = [d for d in out if 0 <= d < self._sp.get_piece_size()]
        return self._sp.decode(out) if out else ""

    def _ctc_biased_decode(self, log_probs: torch.Tensor) -> str:
        lp_np = log_probs.squeeze(0).float().cpu().numpy()
        return self._biaser.decode(lp_np)

    @torch.inference_mode()
    def transcribe(self, audio: np.ndarray, sample_rate: int = TARGET_SR) -> str:
        if self._model is None:
            self.load_model()
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != TARGET_SR:
            ratio = TARGET_SR / sample_rate
            new_len = int(len(audio) * ratio)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, new_len),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)
            sample_rate = TARGET_SR

        chunk_samples = int(14.0 * TARGET_SR)
        pieces: list[str] = []
        for start in range(0, len(audio), chunk_samples):
            seg = audio[start:start + chunk_samples]
            if len(seg) < TARGET_SR * 0.5:
                continue
            mel = self._audio_to_mel(seg, sample_rate)
            log_probs = self._model(mel)
            piece = (
                self._ctc_biased_decode(log_probs)
                if self._bias and self._biaser is not None
                else self._ctc_greedy_decode(log_probs)
            )
            if piece:
                pieces.append(piece)
        return " ".join(pieces)

    def transcribe_file(self, path: str | Path) -> str:
        import soundfile as sf

        audio, sr = sf.read(str(path), dtype="float32")
        return self.transcribe(audio, sr)
