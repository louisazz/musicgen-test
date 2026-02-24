from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import gc

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from utils_io import read_jsonl, sha1_file, write_json, append_jsonl
from utils_audio import save_pcm16_wav, save_melspec_png, basic_audio_stats

MUSICGEN_TOKENS_PER_SECOND = 50

try:
    from audiocraft.models import MusicGen as ACMusicGen
    from audiocraft.data.audio import audio_write
    HAS_AUDIOCRAFT = True
except ImportError:
    HAS_AUDIOCRAFT = False

try:
    from audiocraft.models import MultiBandDiffusion
    HAS_MBD = True
except ImportError:
    HAS_MBD = False

try:
    from transformers import MusicgenForConditionalGeneration, AutoProcessor
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

HF_MODEL_MAP = {
    "small": "facebook/musicgen-small",
    "medium": "facebook/musicgen-medium",
    "large": "facebook/musicgen-large",
    "melody": "facebook/musicgen-melody",
    "melody-large": "facebook/musicgen-melody-large",
    "stereo-small": "facebook/musicgen-stereo-small",
    "stereo-medium": "facebook/musicgen-stereo-medium",
    "stereo-large": "facebook/musicgen-stereo-large",
    "stereo-melody": "facebook/musicgen-stereo-melody",
    "stereo-melody-large": "facebook/musicgen-stereo-melody-large",
    "style": "facebook/musicgen-style",
}


def resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return requested


def resolve_backend(requested: str) -> str:
    """Pick inference backend: audiocraft (preferred for stereo/MBD) or transformers."""
    if requested == "auto":
        return "audiocraft" if HAS_AUDIOCRAFT else "transformers"
    if requested == "audiocraft" and not HAS_AUDIOCRAFT:
        raise RuntimeError("backend=audiocraft but audiocraft is not installed")
    if requested == "transformers" and not HAS_TRANSFORMERS:
        raise RuntimeError("backend=transformers but transformers is not installed")
    return requested


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def loudness_normalize(wav_np: np.ndarray, sr: int, target_lufs: float = -14.0) -> np.ndarray:
    wav_np = wav_np.astype(np.float32)
    if wav_np.ndim > 1:
        mono = wav_np.mean(axis=-1) if wav_np.shape[-1] > wav_np.shape[0] else wav_np.mean(axis=0)
    else:
        mono = wav_np
    rms = float(np.sqrt(np.mean(mono ** 2) + 1e-12))
    if rms < 1e-8:
        return wav_np
    target_rms = 10 ** (target_lufs / 20)
    gain = target_rms / rms
    return np.clip(wav_np * gain, -1.0, 1.0)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default=None)
    ap.add_argument('--prompts', type=str, required=True)
    ap.add_argument('--run_name', type=str, default=None)
    ap.add_argument('--out_root', type=str, default=None)
    ap.add_argument('--device', type=str, default=None, help='cuda|mps|cpu|auto')
    ap.add_argument('--backend', type=str, default=None, help='auto|audiocraft|transformers')
    ap.add_argument('--model', type=str, default=None, help='variant name or full HF id')
    ap.add_argument('--duration', type=float, default=None)
    ap.add_argument('--num_samples_per_seed', type=int, default=None)
    ap.add_argument('--max_prompts', type=int, default=None)
    ap.add_argument('--use_mbd', action='store_true', help='Enable Multi-Band Diffusion decoder (audiocraft mono only)')
    ap.add_argument('--dry_run', action='store_true')
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Backend: audiocraft
# ---------------------------------------------------------------------------
class AudiocraftBackend:
    def __init__(self, model_id: str, device: str, use_mbd: bool = False):
        print(f"[audiocraft] Loading '{model_id}' on {device}")
        self.model = ACMusicGen.get_pretrained(model_id, device=device)
        self.sample_rate = self.model.sample_rate
        self.device = device
        self.use_mbd = use_mbd
        self.mbd = None
        if use_mbd:
            if not HAS_MBD:
                raise RuntimeError("use_mbd=True but MultiBandDiffusion not available")
            print("[audiocraft] Loading MultiBandDiffusion decoder …")
            _orig_load = torch.load
            torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})
            try:
                self.mbd = MultiBandDiffusion.get_mbd_musicgen()
            finally:
                torch.load = _orig_load

    def set_params(self, gen: Dict[str, Any], duration: float):
        self.model.set_generation_params(
            use_sampling=bool(gen["use_sampling"]),
            top_k=int(gen["top_k"]),
            top_p=float(gen["top_p"]),
            temperature=float(gen["temperature"]),
            duration=duration,
            cfg_coef=float(gen["cfg_coef"]),
        )

    def generate(self, prompt: str) -> torch.Tensor:
        """Returns [C, T] tensor (C=1 mono, C=2 stereo)."""
        wav = self.model.generate([prompt], progress=False)
        return wav[0]  # [C, T]

    def generate_with_mbd(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (wav_encodec [C,T], wav_mbd [C,T])."""
        wav, tokens = self.model.generate([prompt], progress=False, return_tokens=True)
        wav_out = wav[0].cpu()
        del wav
        torch.cuda.empty_cache()
        wav_mbd = self.mbd.tokens_to_wav(tokens)
        wav_mbd_out = wav_mbd[0].cpu()
        del wav_mbd, tokens
        torch.cuda.empty_cache()
        return wav_out, wav_mbd_out

    def save_loudness_wav(self, wav: torch.Tensor, out_dir: Path):
        audio_write(str(out_dir / "loudness"), wav.cpu(), self.sample_rate, strategy="loudness")
        return out_dir / "loudness.wav"


# ---------------------------------------------------------------------------
# Backend: transformers
# ---------------------------------------------------------------------------
class TransformersBackend:
    def __init__(self, model_id: str, device: str):
        print(f"[transformers] Loading '{model_id}' on {device}")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = MusicgenForConditionalGeneration.from_pretrained(model_id).to(device)
        self.sample_rate = self.model.config.audio_encoder.sampling_rate
        self.device = device
        self._gen_kwargs: Dict[str, Any] = {}

    def set_params(self, gen: Dict[str, Any], duration: float):
        self._gen_kwargs = {
            "max_new_tokens": int(duration * MUSICGEN_TOKENS_PER_SECOND),
            "do_sample": bool(gen["use_sampling"]),
            "guidance_scale": float(gen["cfg_coef"]),
        }
        if int(gen["top_k"]) > 0:
            self._gen_kwargs["top_k"] = int(gen["top_k"])
        if float(gen["top_p"]) > 0:
            self._gen_kwargs["top_p"] = float(gen["top_p"])
        if float(gen["temperature"]) != 1.0:
            self._gen_kwargs["temperature"] = float(gen["temperature"])

    def generate(self, prompt: str) -> torch.Tensor:
        inputs = self.processor(text=[prompt], padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            audio_values = self.model.generate(**inputs, **self._gen_kwargs)
        return audio_values[0].float()  # [C, T], ensure float32

    def save_loudness_wav(self, wav: torch.Tensor, out_dir: Path):
        wav_np = wav.cpu().numpy()
        if wav_np.ndim == 2:
            wav_np = wav_np.T  # [T, C] for soundfile
        wav_loud = loudness_normalize(wav_np, self.sample_rate)
        out_path = out_dir / "loudness.wav"
        sf.write(str(out_path), wav_loud, self.sample_rate, subtype="PCM_16")
        return out_path


def main() -> None:
    args = parse_args()
    cfg = load_yaml(Path(args.config) if args.config else None)

    cfg = deep_update({
        "model": {"variant": "small", "device": "auto", "backend": "auto"},
        "generation": {
            "use_sampling": True, "top_k": 250, "top_p": 0.0, "temperature": 1.0,
            "cfg_coef": 3.0, "duration_default": 10.0, "num_samples_per_seed": 1,
            "use_mbd": False,
        },
        "output": {
            "out_root": "runs", "run_name": "musicgen_run",
            "save_raw_wav": True, "save_loudness_wav": True,
            "save_spectrogram": True, "spectrogram_type": "mel",
        }
    }, cfg)

    if args.run_name is not None:
        cfg["output"]["run_name"] = args.run_name
    if args.out_root is not None:
        cfg["output"]["out_root"] = args.out_root
    if args.device is not None:
        cfg["model"]["device"] = args.device
    if args.backend is not None:
        cfg["model"]["backend"] = args.backend
    if args.model is not None:
        cfg["model"]["variant"] = args.model
    if args.duration is not None:
        cfg["generation"]["duration_default"] = float(args.duration)
    if args.num_samples_per_seed is not None:
        cfg["generation"]["num_samples_per_seed"] = int(args.num_samples_per_seed)
    if args.use_mbd:
        cfg["generation"]["use_mbd"] = True

    prompts_path = Path(args.prompts)
    prompt_rows = read_jsonl(prompts_path)
    if args.max_prompts is not None:
        prompt_rows = prompt_rows[:args.max_prompts]

    run_dir = Path(cfg["output"]["out_root"]) / cfg["output"]["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prompts_path": str(prompts_path),
        "prompts_sha1": sha1_file(prompts_path),
        "cfg": cfg,
    }
    write_json(run_dir / "run_meta.json", run_meta)

    if args.dry_run:
        print(f"[DRY RUN] Loaded {len(prompt_rows)} prompts. Run dir: {run_dir}")
        return

    device = resolve_device(cfg["model"]["device"])
    backend_name = resolve_backend(cfg["model"]["backend"])
    model_variant = cfg["model"]["variant"]
    hf_model_id = HF_MODEL_MAP.get(model_variant, model_variant)

    use_mbd = bool(cfg["generation"].get("use_mbd", False))
    if use_mbd and backend_name != "audiocraft":
        raise RuntimeError("MBD requires backend=audiocraft (not transformers)")
    if use_mbd and "stereo" in model_variant:
        print("WARNING: MBD pretrained weights are mono-only. Stereo models may produce artifacts.")

    if backend_name == "audiocraft":
        backend = AudiocraftBackend(hf_model_id, device, use_mbd=use_mbd)
    else:
        backend = TransformersBackend(hf_model_id, device)

    sample_rate = backend.sample_rate
    gen = cfg["generation"]
    duration_default = float(gen["duration_default"])

    index_jsonl = run_dir / "index.jsonl"
    if index_jsonl.exists():
        index_jsonl.unlink()

    for row in tqdm(prompt_rows, desc="MusicGen inference"):
        pid = str(row.get("id"))
        prompt = str(row["prompt"])
        duration = float(row.get("duration", duration_default))
        seeds = row.get("seeds", [0])

        backend.set_params(gen, duration)

        for seed in seeds:
            for sidx in range(int(gen["num_samples_per_seed"])):
                set_seed(int(seed) + 1000 * sidx)

                out_dir = run_dir / "samples" / pid / f"seed{int(seed):04d}" / f"s{sidx:02d}"
                out_dir.mkdir(parents=True, exist_ok=True)

                wav_mbd = None
                if use_mbd and hasattr(backend, "generate_with_mbd"):
                    wav, wav_mbd = backend.generate_with_mbd(prompt)
                else:
                    wav = backend.generate(prompt)  # [C, T]
                wav = torch.clamp(wav, -1.0, 1.0)

                raw_path = out_dir / "raw.wav"
                if cfg["output"]["save_raw_wav"]:
                    save_pcm16_wav(wav, sample_rate, raw_path)

                mbd_path = out_dir / "mbd.wav"
                if wav_mbd is not None:
                    wav_mbd = torch.clamp(wav_mbd, -1.0, 1.0)
                    save_pcm16_wav(wav_mbd, sample_rate, mbd_path)

                loud_path = out_dir / "loudness.wav"
                if cfg["output"]["save_loudness_wav"]:
                    src = wav_mbd if wav_mbd is not None else wav
                    loud_path = backend.save_loudness_wav(src, out_dir)

                spec_path = out_dir / "spec_mel.png"
                if cfg["output"]["save_spectrogram"]:
                    spec_src = str(mbd_path) if mbd_path.exists() else str(raw_path)
                    x, sr = sf.read(spec_src)
                    save_melspec_png(x, sr, spec_path)

                meta = {
                    "prompt_id": pid,
                    "sample_id": f"{pid}_seed{int(seed):04d}_s{sidx:02d}",
                    "prompt": prompt,
                    "seed": int(seed),
                    "sample_idx": int(sidx),
                    "duration": duration,
                    "sample_rate": int(sample_rate),
                    "model_variant": model_variant,
                    "hf_model_id": hf_model_id,
                    "backend": backend_name,
                    "use_mbd": use_mbd,
                    "generation_params": {
                        "use_sampling": bool(gen["use_sampling"]),
                        "top_k": int(gen["top_k"]),
                        "top_p": float(gen["top_p"]),
                        "temperature": float(gen["temperature"]),
                        "cfg_coef": float(gen["cfg_coef"]),
                    },
                    "prompt_fields": dict(row),
                    "paths": {
                        "raw_wav": str(raw_path.relative_to(run_dir)) if raw_path.exists() else None,
                        "mbd_wav": str(mbd_path.relative_to(run_dir)) if mbd_path.exists() else None,
                        "loudness_wav": str(loud_path.relative_to(run_dir)) if loud_path.exists() else None,
                        "spec_png": str(spec_path.relative_to(run_dir)) if spec_path.exists() else None,
                    },
                }
                write_json(out_dir / "meta.json", meta)

                append_jsonl(index_jsonl, {
                    "sample_id": meta["sample_id"],
                    "prompt_id": pid,
                    "seed": int(seed),
                    "sample_idx": int(sidx),
                    "raw_wav": str(raw_path.relative_to(run_dir)),
                    "loudness_wav": str(loud_path.relative_to(run_dir)) if loud_path.exists() else None,
                    "meta": str((out_dir / "meta.json").relative_to(run_dir)),
                })

                del wav
                if wav_mbd is not None:
                    del wav_mbd
                gc.collect()
                torch.cuda.empty_cache()

    print(f"Done. Samples written to: {run_dir}")


if __name__ == "__main__":
    main()
