from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from utils_io import read_jsonl, write_json
from utils_audio import save_melspec_png, basic_audio_stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="runs/<run_name> directory")
    ap.add_argument("--use_loudness", action="store_true",
                    help="use loudness.wav if available; otherwise raw.wav")
    ap.add_argument("--max_items", type=int, default=None, help="debug: limit number of samples")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    index_path = run_dir / "index.jsonl"
    rows = read_jsonl(index_path)
    if args.max_items is not None:
        rows = rows[:args.max_items]

    out_dir = run_dir / "eval_basic"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "specs").mkdir(parents=True, exist_ok=True)

    stats = []
    for r in tqdm(rows, desc="eval_basic"):
        wav_rel = r.get("loudness_wav") if args.use_loudness else r.get("raw_wav")
        if wav_rel is None:
            wav_rel = r["raw_wav"]
        wav_path = run_dir / wav_rel

        audio, sr = sf.read(str(wav_path))
        st = basic_audio_stats(audio)
        duration = float(audio.shape[0] / sr)

        # Save mel-spectrogram png
        out_png = out_dir / "specs" / (Path(wav_rel).stem + ".png")
        save_melspec_png(audio, sr, out_png)

        stats.append({
            "sample_id": r["sample_id"],
            "prompt_id": r["prompt_id"],
            "seed": r["seed"],
            "sample_idx": r["sample_idx"],
            "wav": wav_rel,
            "sr": sr,
            "duration_s": duration,
            **st
        })

    df = pd.DataFrame(stats)
    df.to_csv(out_dir / "basic_stats.csv", index=False)
    write_json(out_dir / "basic_stats.json", stats)

    # quick summary
    summary = {
        "num_samples": int(len(df)),
        "sr_mode": int(df["sr"].mode().iloc[0]) if len(df) else None,
        "duration_mean": float(df["duration_s"].mean()) if len(df) else None,
        "peak_p95": float(df["peak"].quantile(0.95)) if len(df) else None,
        "rms_mean": float(df["rms"].mean()) if len(df) else None,
    }
    write_json(out_dir / "summary.json", summary)
    print("Saved:", out_dir)


if __name__ == "__main__":
    main()
