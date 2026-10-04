"""WSKAN-11 release inference: load a published model (local dir or HF repo)
and generate byte-level text.

Each released model repo contains:
  config.yaml         architecture + tokenizer + training metadata
  model.safetensors   fp32 weights (state_dict of WaveletStateKANLMV11)

Requirements: torch, safetensors, pyyaml, triton + a CUDA GPU (the V11 scan
is a fused Triton kernel; there is no CPU path).

Usage:
  .venv/bin/python experiments/RELEASE_infer.py \
      --model llaa33219/wskan-100k-ultrachat \
      --prompt "User: What is the capital of France?\nAssistant:" \
      --max-new 400 --temperature 0.8 --seed 1234

  # or from a local directory:
  .venv/bin/python experiments/RELEASE_infer.py --model /path/to/wskan-100k-ultrachat
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def build_model(cfg: dict) -> torch.nn.Module:
    """Reconstruct WaveletStateKANLMV11 exactly as the release trainer did."""
    from models.V11_WSKAN import WaveletStateKANLMV11

    return WaveletStateKANLMV11(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_states=cfg.get("n_states", 6),
        use_feature_bc=cfg.get("use_feature_bc", True),
        bc_rank=cfg.get("bc_rank", 32),
        wz_diag=cfg.get("wz_diag", False),
        g_rank=cfg.get("g_rank"),
        bf16_scan=cfg.get("bf16_scan", False),
        oscillatory=cfg.get("oscillatory", True),
    )


def resolve_files(source: str) -> tuple[Path, Path]:
    """Return (config.yaml, model.safetensors) from a local dir or HF repo."""
    p = Path(source)
    if p.is_dir():
        return p / "config.yaml", p / "model.safetensors"
    from huggingface_hub import hf_hub_download

    cfg = Path(hf_hub_download(source, "config.yaml"))
    weights = Path(hf_hub_download(source, "model.safetensors"))
    return cfg, weights


def load_model(source: str, device: str = "cuda") -> tuple[torch.nn.Module, dict]:
    cfg_path, weights_path = resolve_files(source)
    cfg = yaml.safe_load(cfg_path.read_text())
    model = build_model(cfg["architecture"])
    state = load_file(weights_path)
    model.load_state_dict(state)
    model = model.to(device).eval()
    return model, cfg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF repo id or local directory")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max-new", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required: the V11 scan is a fused Triton kernel.")

    torch.manual_seed(args.seed)
    model, cfg = load_model(args.model)
    prompt = args.prompt.encode("utf-8")
    out = model.generate(
        prompt, max_new=args.max_new, temperature=args.temperature
    ).decode("utf-8", errors="replace")
    print(out)


if __name__ == "__main__":
    main()
