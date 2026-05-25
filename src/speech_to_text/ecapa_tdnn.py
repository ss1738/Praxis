"""PRAXIS-TIMBRE: ECAPA-TDNN speaker embedding model for UK clinical audio.

Custom 5.80M-param (inference) ECAPA-TDNN with:
- SE-Res2Net blocks (scale=8)
- Attentive statistics pooling
- 256-dim speaker embeddings
- AAM-Softmax training loss (1417-speaker classifier head used only during training)
- Trained on LibriSpeech; applied zero-shot to PriMock57 GP audio

Benchmarked on PriMock57 test set:
  PRAXIS-TIMBRE     5.80M  DER 7.1%   0.68s   15/15 speaker count
  pyannote 3.1     ~7M    DER 24.4%  42.87s  13/15
  TitaNet-Large    ~23M   DER 6.4%   10.57s  13/15
  SpeechBrain ECAPA ~15M  DER 6.5%   1.71s   12/15
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
SAMPLE_RATE = 16_000
EMBEDDING_DIM = 256


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        s = x.mean(dim=2)
        s = self.fc(s).unsqueeze(2)
        return x * s


class Res2NetBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, scale=8):
        super().__init__()
        assert channels % scale == 0
        self.scale = scale
        self.width = channels // scale
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    self.width, self.width, kernel_size,
                    padding=(kernel_size - 1) * dilation // 2,
                    dilation=dilation,
                ),
                nn.BatchNorm1d(self.width),
                nn.ReLU(inplace=True),
            )
            for _ in range(scale - 1)
        ])

    def forward(self, x):
        chunks = torch.chunk(x, self.scale, dim=1)
        outs = [chunks[0]]
        for i, conv in enumerate(self.convs):
            if i == 0:
                y = conv(chunks[i + 1])
            else:
                y = conv(chunks[i + 1] + outs[-1])
            outs.append(y)
        return torch.cat(outs, dim=1)


class SERes2NetBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, scale=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            Res2NetBlock(channels, kernel_size, dilation, scale),
            nn.Conv1d(channels, channels, 1),
            nn.BatchNorm1d(channels),
        )
        self.se = SEBlock(channels)

    def forward(self, x):
        return x + self.se(self.block(x))


class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, channels, attention_dim=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels, attention_dim, 1),
            nn.Tanh(),
            nn.Conv1d(attention_dim, channels, 1),
            nn.Softmax(dim=2),
        )

    def forward(self, x):
        w = self.attention(x)
        mu = (x * w).sum(dim=2)
        sigma = torch.sqrt(((x - mu.unsqueeze(2)) ** 2 * w).sum(dim=2) + 1e-6)
        return torch.cat([mu, sigma], dim=1)


class ECAPATDNN(nn.Module):
    """PRAXIS-TIMBRE: ECAPA-TDNN, 5.80M params (inference), 256-dim speaker embeddings."""

    def __init__(self, n_mels=80, channels=512, embedding_dim=256, num_speakers=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.layer1 = nn.Sequential(
            nn.Conv1d(n_mels, channels, 5, padding=2),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.layer2 = SERes2NetBlock(channels, kernel_size=3, dilation=2, scale=8)
        self.layer3 = SERes2NetBlock(channels, kernel_size=3, dilation=3, scale=8)
        self.layer4 = SERes2NetBlock(channels, kernel_size=3, dilation=4, scale=8)

        self.mfa = nn.Sequential(
            nn.Conv1d(channels * 3, channels * 3, 1),
            nn.BatchNorm1d(channels * 3),
            nn.ReLU(inplace=True),
        )

        self.asp = AttentiveStatisticsPooling(channels * 3)
        self.bn = nn.BatchNorm1d(channels * 6)
        self.fc = nn.Linear(channels * 6, embedding_dim)
        self.bn_emb = nn.BatchNorm1d(embedding_dim)

        self.classifier = None
        if num_speakers:
            self.classifier = nn.Linear(embedding_dim, num_speakers)

    def forward(self, x, return_embedding=False):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = torch.cat([x2, x3, x4], dim=1)
        x = self.mfa(x)
        x = self.asp(x)
        x = self.bn(x)
        emb = self.bn_emb(self.fc(x))

        if return_embedding or self.classifier is None:
            return F.normalize(emb, p=2, dim=1)

        logits = self.classifier(emb)
        return logits, F.normalize(emb, p=2, dim=1)

    def get_embedding(self, x):
        return self.forward(x, return_embedding=True)


_DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[2] / "data" / "models" / "ecapa_tdnn" / "best_model.pt"
)


def load_ecapa_tdnn(
    checkpoint_path: str | Path | None = None,
    device: str | None = None,
) -> ECAPATDNN:
    """Load a trained ECAPA-TDNN from checkpoint."""
    ckpt_path = Path(checkpoint_path) if checkpoint_path else _DEFAULT_CHECKPOINT
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ECAPA-TDNN checkpoint not found: {ckpt_path}")

    dev = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(str(ckpt_path), weights_only=False, map_location=dev)

    model = ECAPATDNN(
        n_mels=ckpt.get("n_mels", N_MELS),
        channels=512,
        embedding_dim=ckpt.get("embedding_dim", EMBEDDING_DIM),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval().to(dev)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"PRAXIS-TIMBRE (ECAPA-TDNN) loaded: {n_params / 1e6:.2f}M params, device={dev}")
    return model
