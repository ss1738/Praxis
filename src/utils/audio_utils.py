"""Audio loading and saving helpers."""

from pathlib import Path

import numpy as np
import soundfile as sf
from loguru import logger


def load_audio(path: str | Path, target_sr: int = 16_000) -> tuple[np.ndarray, int]:
    """Load audio file, resample to target_sr, return (audio, sr)."""
    import librosa

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio, sr = librosa.load(str(path), sr=target_sr, mono=True)
    logger.debug(f"Loaded {path.name}: {len(audio)/sr:.1f}s @ {sr}Hz")
    return audio, sr


def save_audio(path: str | Path, audio: np.ndarray, sr: int = 16_000) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)
    logger.debug(f"Saved {path.name}: {len(audio)/sr:.1f}s @ {sr}Hz")
    return path


def get_duration(path: str | Path) -> float:
    import librosa

    return librosa.get_duration(path=str(path))


def audio_to_int16(audio: np.ndarray) -> np.ndarray:
    # webrtcvad needs int16
    return (audio * 32767).astype(np.int16)
