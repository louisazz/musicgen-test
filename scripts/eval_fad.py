from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from utils_io import read_jsonl, write_json


def collect_wavs(run_dir: Path, out_dir: Path, use_loudness: bool) -> int:
    """Flatten wavs into a single folder for fadtk CLI."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(run_dir / "index.jsonl")
    for i, r in enumerate(rows):
        wav_rel = r.get("loudness_wav") if use_loudness else r.get("raw_wav")
        if wav_rel is None:
            wav_rel = r["raw_wav"]
        src = run_dir / wav_rel
        dst = out_dir / f"{i:06d}.wav"
        shutil.copy2(src, dst)
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--fad_model", type=str, default="vggish",
                    help="fadtk embedding model name, e.g., vggish / clap-laion-music / encodec-emb")
    ap.add_argument("--baseline", type=str, default="fma_pop",
                    help="fadtk baseline name OR a path to a baseline audio directory")
    ap.add_argument("--mode", type=str, default="inf", choices=["inf", "indiv"])
    ap.add_argument("--use_loudness", action="store_true",
                    help="use loudness.wav for evaluation (recommended if baseline is also normalized)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    eval_dir = run_dir / "_fad_eval_wavs"
    n = collect_wavs(run_dir, eval_dir, args.use_loudness)

    cmd = ["fadtk", args.fad_model, args.baseline, str(eval_dir)]
    if args.mode == "inf":
        cmd += ["--inf"]
    else:
        cmd += ["--indiv", str(run_dir / "eval_fad_indiv.csv")]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    summary = {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "num_samples": n,
        "fad_model": args.fad_model,
        "baseline": args.baseline,
        "mode": args.mode,
        "use_loudness": bool(args.use_loudness),
    }
    write_json(run_dir / "eval_fad_summary.json", summary)
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr)


if __name__ == "__main__":
    main()
