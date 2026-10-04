"""Verify the 15 published Hugging Face models end-to-end.

For every llaa33219/wskan-<size>-<dataset> repo: download config.yaml +
model.safetensors from the Hub, build WSKAN-11, regenerate the three card
prompts with the documented settings (temperature 0.8, seed 1234), and require
a byte-exact match against generations.json (also downloaded from the Hub).

Run: .venv/bin/python experiments/RELEASE_verify.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SIZES = ["1k", "10k", "100k", "1m", "10m"]
DATASETS = ["tinystories", "ultrachat", "wikitext"]
NS = "llaa33219"


def pick_gpu() -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"]
    ).decode()
    rows = [l.split(",") for l in out.strip().splitlines()]
    return int(min(rows, key=lambda r: int(r[1].strip().split()[0]))[0])


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(pick_gpu())
    from huggingface_hub import hf_hub_download

    from experiments.RELEASE_infer import load_model

    ok, fail = 0, 0
    for dataset in DATASETS:
        for size in SIZES:
            repo = f"{NS}/wskan-{size}-{dataset}"
            expected = json.loads(
                Path(hf_hub_download(repo, "generations.json")).read_text()
            )
            model, _ = load_model(repo)
            mismatches = []
            for g in expected:
                torch.manual_seed(g["seed"])
                got = model.generate(
                    g["prompt"].encode("utf-8"),
                    max_new=g["max_new"],
                    temperature=g["temperature"],
                ).decode("utf-8", errors="replace")
                if got != g["output"]:
                    mismatches.append(g["prompt"][:40])
            if mismatches:
                fail += 1
                print(f"FAIL {repo}: {len(mismatches)} mismatch(es) {mismatches}")
            else:
                ok += 1
                print(f"OK   {repo}: 3/3 outputs byte-identical")
            del model
            torch.cuda.empty_cache()
    print(f"\n{ok}/15 repos verified, {fail} failed")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
