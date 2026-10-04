"""WSKAN-11 release driver: convert the 3-epoch campaign checkpoints (seed 42)
to safetensors + config.yaml, and generate real samples per model for the
Hugging Face model cards.

Scope: models/wskan11 only, 5 sizes x 3 datasets, seed 42 (canonical seed).

Outputs (staging, not committed):
  /tmp/opencode/hf_release/wskan-<size>-<dataset>/
      model.safetensors   fp32 weights
      config.yaml         architecture + tokenizer + training metadata
      generations.json    real generation outputs (prompt, text, settings)
      meta.json           params, final eval loss/ppl, steps, checkpoint step

Run: .venv/bin/python experiments/RELEASE_publish.py
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
STAGING = Path("/tmp/opencode/hf_release")

SIZES = ["1k", "10k", "100k", "1m", "10m"]
DATASETS = ["tinystories", "ultrachat", "wikitext"]
SIZE_CFG = {"1k": (4, 1), "10k": (12, 1), "100k": (40, 2), "1m": (80, 6), "10m": (512, 2)}
SEED = 42

# campaign protocol (experiments/CAMPAIGN_orchestrate.py)
STEPS = {
    ("tinystories", "1k"): 30_000, ("tinystories", "10k"): 60_000, ("tinystories", "100k"): 300_000,
    ("tinystories", "1m"): 100_000, ("tinystories", "10m"): 33_000,
    ("ultrachat", "1k"): 36_000, ("ultrachat", "10k"): 54_000, ("ultrachat", "100k"): 108_000,
    ("ultrachat", "1m"): 54_000, ("ultrachat", "10m"): 36_000,
    ("wikitext", "1k"): 33_000, ("wikitext", "10k"): 50_000, ("wikitext", "100k"): 99_000,
    ("wikitext", "1m"): 66_000, ("wikitext", "10m"): 33_000,
}
BLOCK = {"tinystories": 256, "ultrachat": 512, "wikitext": 256}
BATCH = 64
HF_DATASET = {
    "tinystories": "roneneldan/TinyStories",
    "ultrachat": "HuggingFaceH4/ultrachat_200k (train_sft)",
    "wikitext": "Salesforce/wikitext wikitext-103-raw-v1",
}

PROMPTS = {
    "tinystories": [
        "Once upon a time",
        "One day, a little girl named Lily found a",
        "Tom was very sad because he lost his",
    ],
    "ultrachat": [
        "User: What is the capital of France?\nAssistant:",
        "User: Can you explain what photosynthesis is in simple terms?\nAssistant:",
        "User: Write a short poem about the ocean.\nAssistant:",
    ],
    "wikitext": [
        "The history of the city began",
        "In the early 20th century, scientists discovered",
        "The film was directed by",
    ],
}
GEN_MAX_NEW = 400
GEN_TEMPERATURE = 0.8
GEN_SEED = 1234


def pick_gpu() -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"]
    ).decode()
    rows = [l.split(",") for l in out.strip().splitlines()]
    return int(min(rows, key=lambda r: int(r[1].strip().split()[0]))[0])


def final_eval(ckpt_dir: Path) -> tuple[float, int]:
    rows = list(csv.reader((ckpt_dir / "train_log.csv").read_text().splitlines()[1:]))
    rows = [r for r in rows if r and r[0].strip()]
    last = rows[-1]
    return float(last[2]), int(last[0])


def main() -> None:
    import os

    gpu = pick_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    device = "cuda"
    print(f"using GPU {gpu}")

    from experiments.RELEASE_infer import build_model

    summary = []
    for dataset in DATASETS:
        for size in SIZES:
            name = f"wskan-{size}-{dataset}"
            tag = f"wskan11_{dataset}_{size}_3ep_s{SEED}"
            ckpt_dir = ROOT / "checkpoints" / tag
            out_dir = STAGING / name
            out_dir.mkdir(parents=True, exist_ok=True)

            d, L = SIZE_CFG[size]
            cfg_arch = dict(
                vocab_size=256, d_model=d, n_layers=L, n_states=6,
                use_feature_bc=True, bc_rank=min(32, d), wz_diag=False,
                g_rank=None, oscillatory=True, bf16_scan=True,
            )
            lr = 3e-3 if size in ("1k", "10k", "100k") else 1e-3
            steps = STEPS[(dataset, size)]

            ckpt = torch.load(ckpt_dir / "latest.pt", map_location="cpu", weights_only=False)
            ckpt_step = int(ckpt["step"])
            if ckpt_step != steps:
                print(f"WARNING: {name} ckpt step {ckpt_step} != expected {steps}")

            model = build_model(cfg_arch).to(device).eval()
            model.load_state_dict(ckpt["state_dict"])
            n_params = sum(p.numel() for p in model.parameters())

            state = {k: v.detach().cpu().float().contiguous() for k, v in model.state_dict().items()}
            save_file(state, str(out_dir / "model.safetensors"), metadata={"format": "pt"})

            eval_loss, _ = final_eval(ckpt_dir)
            config = {
                "model_name": name,
                "architecture_family": "WSKAN-11 (Wavelet-like State-Space KAN, fused Triton scan)",
                "architecture": cfg_arch,
                "tokenizer": {
                    "type": "byte-level",
                    "description": "raw UTF-8 bytes; vocab ids 0-255; no BPE, no special tokens",
                },
                "training": {
                    "dataset": HF_DATASET[dataset],
                    "steps": steps,
                    "checkpoint_step": ckpt_step,
                    "batch_size": BATCH,
                    "block_size": BLOCK[dataset],
                    "lr": lr,
                    "lr_schedule": "cosine",
                    "optimizer": "AdamW (weight_decay=0.0, grad clip 1.0)",
                    "seed": SEED,
                    "epochs": "~3",
                    "precision": "fp32 weights, bf16 scan",
                },
                "evaluation": {
                    "final_eval_loss": round(eval_loss, 4),
                    "final_eval_byte_ppl": round(float(torch.exp(torch.tensor(eval_loss))), 2),
                },
                "generation_defaults": {
                    "max_new": GEN_MAX_NEW,
                    "temperature": GEN_TEMPERATURE,
                    "seed": GEN_SEED,
                },
            }
            (out_dir / "config.yaml").write_text(
                yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
            )

            gens = []
            for prompt in PROMPTS[dataset]:
                torch.manual_seed(GEN_SEED)
                text = model.generate(
                    prompt.encode("utf-8"), max_new=GEN_MAX_NEW, temperature=GEN_TEMPERATURE
                ).decode("utf-8", errors="replace")
                gens.append({
                    "prompt": prompt,
                    "output": text,
                    "max_new": GEN_MAX_NEW,
                    "temperature": GEN_TEMPERATURE,
                    "seed": GEN_SEED,
                })
            (out_dir / "generations.json").write_text(json.dumps(gens, indent=2, ensure_ascii=False))

            meta = {
                "name": name, "params": n_params, "dataset": dataset, "size": size,
                "ckpt_step": ckpt_step, "eval_loss": eval_loss,
                "eval_byte_ppl": config["evaluation"]["final_eval_byte_ppl"],
            }
            (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
            summary.append(meta)
            print(f"{name}: {n_params:,} params, eval {eval_loss:.4f}, "
                  f"ppl {meta['eval_byte_ppl']}, ckpt step {ckpt_step}")

            del model
            torch.cuda.empty_cache()

    (STAGING / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nstaged {len(summary)} models at {STAGING}")


if __name__ == "__main__":
    main()
