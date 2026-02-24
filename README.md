# SHAYI - Task 5: MusicGen Baseline Inference

Reproducible MusicGen inference pipeline with unified output format for downstream evaluation.

## Model Configurations

| Config | Model | Params | Channels | MBD | VRAM | Notes |
|--------|-------|--------|----------|-----|------|-------|
| `musicgen_small.yaml` | small | 300M | mono | no | ~2GB | Debug only |
| `musicgen_medium_mbd.yaml` | medium | 1.5B | mono | yes | ~5GB | T4 16GB compatible |
| `musicgen_large_mbd.yaml` | large | 3.3B | mono | yes | ~12GB | A100 required, primary baseline |
| `musicgen_stereo_large.yaml` | stereo-large | 3.3B | stereo | no | ~8GB | Stereo output, no MBD support |

All configs use official default generation params: `top_k=250, top_p=0, temperature=1.0, cfg_coef=3.0, duration=30s`.

**Multi-Band Diffusion (MBD)** replaces the standard EnCodec decoder to reduce high-frequency artifacts. Pretrained weights are mono-only and require the audiocraft backend.

## Backends

- **audiocraft**: Official Meta library. Required for MBD, stereo, and melody conditioning.
- **transformers**: HuggingFace implementation. Easier to install, no MBD support.

Default is `auto` (prefers audiocraft, falls back to transformers).

## Installation

Core dependencies:
```bash
pip install -r requirements.txt
```

For audiocraft backend (MBD / stereo):
```bash
pip install audiocraft --no-deps
pip install einops flashy xformers encodec num2words julius hydra-core hydra-colorlog torchmetrics
apt-get install -y ffmpeg pkg-config libavformat-dev libavcodec-dev libavutil-dev libswresample-dev libavfilter-dev
pip install av
```

`audiocraft 1.3.0` declares `torch==2.1.0`; `--no-deps` bypasses this. Runtime is compatible with PyTorch 2.x.

## Usage

```bash
python scripts/infer_musicgen.py \
  --config configs/musicgen_large_mbd.yaml \
  --prompts prompts/prompts_v2_instrumental_disentangle.jsonl
```

## Output Structure

```
runs/<run_name>/
  run_meta.json                         # Config snapshot + prompts SHA1
  index.jsonl                           # Sample manifest for eval scripts
  samples/<prompt_id>/seed<NNNN>/s<NN>/
    raw.wav                             # Standard EnCodec decode
    mbd.wav                             # MBD decode (when enabled)
    loudness.wav                        # Loudness-normalized to -14 LUFS
    spec_mel.png                        # Mel spectrogram
    meta.json                           # Full provenance
```

All future models (custom, ablations, other baselines) should write this same structure so eval scripts are reusable.

## CLI Reference

| Flag | Description |
|------|-------------|
| `--config` | YAML config path |
| `--prompts` | JSONL prompt file path |
| `--backend` | Force `audiocraft` or `transformers` |
| `--device` | Force `cuda`, `mps`, or `cpu` |
| `--model` | Override model variant |
| `--duration` | Override generation duration (seconds) |
| `--max_prompts` | Limit number of prompts to process |
| `--use_mbd` | Enable MBD decoder (audiocraft backend only) |
| `--dry_run` | Validate config and prompts without running inference |
