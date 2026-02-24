from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T


def save_pcm16_wav(wav: torch.Tensor, sr: int, out_path: Path) -> None:
    """Save wav tensor to 16-bit PCM wav.
    wav: [C, T] or [T] float tensor in [-1, 1].
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = wav.detach().cpu().float().numpy().astype(np.float32)
    if x.ndim == 1:
        x = x[None, :]
    x = np.clip(x, -1.0, 1.0)
    sf.write(str(out_path), x.T, sr, subtype="PCM_16")


def basic_audio_stats(wav: np.ndarray) -> dict:
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    peak = float(np.max(np.abs(wav)))
    rms = float(np.sqrt(np.mean(np.square(wav)) + 1e-12))
    return {"peak": peak, "rms": rms}


def save_melspec_png(wav: np.ndarray, sr: int, out_png: Path, n_mels: int = 128) -> None:
    """Quick mel-spectrogram using torchaudio (no librosa/numba dependency)."""
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    waveform = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
    mel_transform = T.MelSpectrogram(
        sample_rate=sr, n_mels=n_mels, n_fft=2048, hop_length=512, f_max=sr / 2,
    )
    S = mel_transform(waveform)
    S_db = T.AmplitudeToDB(stype="power", top_db=80)(S)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = ax.imshow(
        S_db[0].numpy(), origin="lower", aspect="auto",
        extent=[0, wav.shape[0] / sr, 0, sr / 2],
    )
    ax.set_ylabel("Freq (Hz)")
    ax.set_xlabel("Time (s)")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
