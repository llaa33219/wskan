"""V1 WSKAN byte-level LM training on TinyStories.

Trains WaveletStateKANLM (~94k params) on a subset of TinyStories.
With --model mamba2, trains a matched-size HF Mamba-2 baseline under the
identical protocol (same data order, batch, block, lr, schedule, seed).
Checkpoints (the saved bundle of edge *functions*: lambda, g, s, mu, w_base,
w_wav per edge) and qualitative generation samples are written to
checkpoints/<model>_tinystories_lm/.

Run:  .venv/bin/python experiments/V1_train_tinystories_lm.py
      .venv/bin/python experiments/V1_train_tinystories_lm.py --model mamba2 --compile
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.V1_LM import WaveletStateKANLM, count_parameters

PROMPT = b"Once upon a time"


class Mamba2ByteLM(torch.nn.Module):
    """Adapter giving HF Mamba2ForCausalLM the same .loss(idx)/.generate(bytes)
    interface as WaveletStateKANLM so both train under one loop."""

    def __init__(self, scale: str = "100k"):
        super().__init__()
        from transformers import Mamba2Config, Mamba2ForCausalLM

        table = {
            "1k": (8, 1, 8, 8), "10k": (16, 3, 16, 8), "100k": (56, 4, 16, 8),
            "1m": (256, 2, 64, 64), "10m": (512, 6, 64, 64),
            "interp": (64, 3, 16, 8),
        }
        d, L, hd, st = table[scale]
        cfg = Mamba2Config(
            vocab_size=256, hidden_size=d, num_hidden_layers=L,
            num_heads=max(1, (d * 2) // hd), head_dim=hd, state_size=st, expand=2,
            n_groups=1, conv_kernel=4, tie_word_embeddings=True,
            chunk_size=32 if st > 8 else 16,
        )
        self.model = Mamba2ForCausalLM(cfg)
        if scale in ("1m", "10m"):
            self.model.gradient_checkpointing_enable()
        self.vocab_size = 256

    def loss(self, idx: torch.Tensor) -> torch.Tensor:
        next_byte_logits = self.model(input_ids=idx[:, :-1]).logits
        next_byte_targets = idx[:, 1:].reshape(-1)
        return torch.nn.functional.cross_entropy(
            next_byte_logits.reshape(-1, self.vocab_size), next_byte_targets
        )

    @torch.no_grad()
    def generate(self, prompt: bytes, max_new: int = 200, temperature: float = 1.0) -> bytes:
        device = next(self.parameters()).device
        idx = torch.tensor(list(prompt), dtype=torch.long, device=device).unsqueeze(0)
        for _ in range(max_new):
            logits = self.model(input_ids=idx).logits[:, -1, :] / temperature
            nxt = torch.multinomial(torch.softmax(logits, -1), 1)
            idx = torch.cat([idx, nxt], dim=1)
        return bytes(idx[0].tolist())


def load_data(max_train_stories: int, n_eval_stories: int, seed: int, dataset: str = "tinystories"):
    from datasets import load_dataset

    if dataset == "tinystories":
        train_ds = load_dataset("roneneldan/TinyStories", split="train")
        eval_ds = load_dataset("roneneldan/TinyStories", split="validation")
        train_text = "\n".join(train_ds["text"][:max_train_stories])
        eval_text = "\n".join(eval_ds["text"][:n_eval_stories])
    elif dataset == "ultrachat":
        train_ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
        eval_ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")

        def render(conv):
            return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in conv)

        parts, total = [], 0
        for conv in train_ds["messages"]:
            t = render(conv)
            parts.append(t)
            total += len(t)
            if total >= max_train_stories * 900:  # ~900 bytes/story parity with TinyStories runs
                break
        train_text = "\n\n".join(parts)
        eval_text = "\n\n".join(render(c) for c in eval_ds["messages"][:n_eval_stories])
    else:  # wikitext
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
        eval_ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
        parts, total = [], 0
        for t in ds["text"]:
            parts.append(t)
            total += len(t)
            if total >= max_train_stories * 900:
                break
        train_text = "\n\n".join(parts)
        eval_text = "\n\n".join(eval_ds["text"][:n_eval_stories])
    train_ids = torch.tensor(list(train_text.encode("utf-8", errors="ignore")), dtype=torch.uint8)
    eval_ids = torch.tensor(list(eval_text.encode("utf-8", errors="ignore")), dtype=torch.uint8)
    print(f"train bytes: {len(train_ids):,}  eval bytes: {len(eval_ids):,}")
    return train_ids, eval_ids


def get_batch(ids: torch.Tensor, batch: int, block: int, device: str) -> torch.Tensor:
    starts = torch.randint(0, len(ids) - block - 1, (batch,))
    return torch.stack([ids[s : s + block + 1] for s in starts]).long().to(device)


def save_checkpoint(model, step, loss, out_dir: Path, is_wskan: bool):
    ckpt = {"step": step, "train_loss": loss, "state_dict": model.state_dict()}
    torch.save(ckpt, out_dir / f"ckpt_step{step}.pt")
    torch.save(ckpt, out_dir / "latest.pt")
    if is_wskan:
        # human-readable summary of the learned edge functions (layer, edge -> dominant mode)
        summary = {
            f"layer{i}": {
                "sigma_dominant": layer.spectral_signature()["sigma"].tolist(),
                "omega_dominant": layer.spectral_signature()["omega"].tolist(),
            }
            for i, layer in enumerate(model.layers)
        }
        (out_dir / f"edge_functions_step{step}.json").write_text(json.dumps(summary))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["wskan", "wskan2", "wskan3", "wskan3real", "wskan4",
                                       "wskan5", "wskan5nc", "wskan5lin", "wskan6", "wskan7",
                                       "wskan7bc", "wskan7bc16", "wskan7g", "wskan7z",
                                       "mamba2", "tf", "conv", "lstm"], default="wskan")
    p.add_argument("--dataset", choices=["tinystories", "ultrachat", "wikitext"], default="tinystories")
    p.add_argument("--scale", choices=["1k", "10k", "100k", "1m", "10m", "interp"], default="100k")
    p.add_argument("--compile", action="store_true", help="torch.compile the loss step")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--block", type=int, default=256)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant",
                   help="cosine decays lr to 0 over --steps; recommended for long runs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-stories", type=int, default=20000)
    p.add_argument("--eval-stories", type=int, default=1000)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--eval-every", type=int, default=250)
    args = p.parse_args()

    if args.steps is None:
        args.steps = {"1k": 20000, "10k": 20000, "100k": 20000, "1m": 10000, "10m": 5000, "interp": 100000}[args.scale]
    if args.lr is None:
        args.lr = 3e-3 if args.scale in ("1k", "10k", "100k") else 1e-3
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = (
        Path(__file__).resolve().parent.parent
        / "checkpoints"
        / f"{args.model}_{args.dataset}_{args.scale}_s{args.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ids, eval_ids = load_data(args.train_stories, args.eval_stories, args.seed, args.dataset)

    if args.scale == "10m":
        wskan_cfg = dict(vocab_size=256, d_model=256, n_layers=5, n_states=6)
    else:
        wskan_cfg = dict(vocab_size=256, d_model=32, n_layers=3, n_states=6)
    v3_extra = dict(chunk_size=16, grad_checkpoint=True, compile_chunk=True) if args.scale == "10m" else dict()

    SIZE_CFG = {
        "wskan7bc": {"1k": (4, 1), "10k": (12, 1), "100k": (40, 2), "1m": (80, 6), "10m": (512, 2),
                     "interp": (32, 3)},
        "tf": {"1k": (4, 1), "10k": (12, 2), "100k": (40, 4), "1m": (160, 3), "10m": (448, 4)},
        "conv": {"1k": (4, 1), "10k": (32, 1), "100k": (112, 5), "1m": (384, 6), "10m": (1280, 6)},
        "lstm": {"1k": (4, 1), "10k": (12, 6), "100k": (96, 1), "1m": (192, 3), "10m": (448, 6)},
    }

    if args.model in ("tf", "conv", "lstm"):
        from experiments.baselines import GatedConvLM, LSTMLM, TinyTransformerLM

        d, L = SIZE_CFG[args.model][args.scale]
        cls = {"tf": TinyTransformerLM, "conv": GatedConvLM, "lstm": LSTMLM}[args.model]
        model = cls(d_model=d, n_layers=L).to(device)
    elif args.model in ("wskan7bc", "wskan7bc16") and args.scale in SIZE_CFG["wskan7bc"]:
        from models.V7_WSKAN import WaveletStateKANLMV7

        d, L = SIZE_CFG["wskan7bc"][args.scale]
        extra10m = dict(chunk_size=8, grad_checkpoint=True, compile_chunk=True) if args.scale in ("1m", "10m") else dict()
        model = WaveletStateKANLMV7(
            vocab_size=256, d_model=d, n_layers=L, use_feature_bc=True,
            wz_diag=False, g_rank=None, bc_rank=min(32, d), **extra10m,
        ).to(device)
    elif args.model == "wskan":
        model = WaveletStateKANLM(**wskan_cfg).to(device)
    elif args.model == "wskan2":
        from models.V2_WSKAN import WaveletStateKANLMV2

        model = WaveletStateKANLMV2(**wskan_cfg).to(device)
    elif args.model == "wskan3":
        from models.V3_WSKAN import WaveletStateKANLMV3

        model = WaveletStateKANLMV3(**wskan_cfg, **v3_extra).to(device)
    elif args.model == "wskan3real":
        from models.V3_WSKAN import WaveletStateKANLMV3

        model = WaveletStateKANLMV3(**wskan_cfg, **v3_extra, oscillatory=False).to(device)
    elif args.model == "wskan4":
        from models.V4_WSKAN import WaveletStateKANLMV4

        model = WaveletStateKANLMV4(**wskan_cfg, **v3_extra).to(device)
    elif args.model in ("wskan5", "wskan5nc", "wskan5lin"):
        from models.V5_WSKAN import WaveletStateKANLMV5

        model = WaveletStateKANLMV5(
            **wskan_cfg, **v3_extra,
            use_conv=args.model != "wskan5nc",
            omega_init="linear" if args.model == "wskan5lin" else "geometric",
        ).to(device)
    elif args.model == "wskan6":
        from models.V6_WSKAN import WaveletStateKANLMV6

        model = WaveletStateKANLMV6(**wskan_cfg, **v3_extra).to(device)
    elif args.model.startswith("wskan7"):
        from models.V7_WSKAN import WaveletStateKANLMV7

        flags = dict(
            use_feature_bc=args.model in ("wskan7", "wskan7bc", "wskan7bc16"),
            bc_rank=16 if args.model == "wskan7bc16" else 32,
            wz_diag=args.model in ("wskan7", "wskan7z"),
            g_rank=32 if args.model in ("wskan7", "wskan7g") else None,
        )
        model = WaveletStateKANLMV7(**wskan_cfg, **flags).to(device)
    else:
        model = Mamba2ByteLM(scale=args.scale).to(device)
    n_params = count_parameters(model)
    print(f"model: {args.model}  params: {n_params:,}  device: {device}")
    budget = {"1k": 5_000, "10k": 20_000, "100k": 130_000, "1m": 1_100_000, "10m": 11_000_000, "interp": 130_000}[args.scale]
    assert n_params < budget, f"budget check: {n_params:,} >= {budget:,}"

    loss_fn = model.loss
    if args.compile:
        loss_fn = torch.compile(model.loss)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
        if args.lr_schedule == "cosine"
        else None
    )

    log_path = out_dir / "train_log.csv"
    with log_path.open("w", newline="") as f:
        csv.writer(f).writerow(["step", "train_loss", "eval_loss", "param_abs_max"])

    t0 = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        loss = loss_fn(get_batch(train_ids, args.batch, args.block, device))
        if not torch.isfinite(loss):
            print(f"NON-FINITE LOSS at step {step}: {loss.item()} - aborting, checkpoint preserved")
            save_checkpoint(model, step, float("nan"), out_dir, args.model.startswith("wskan"))
            break
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if sched is not None:
            sched.step()

        if step % args.eval_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                ev = loss_fn(get_batch(eval_ids, args.batch, args.block, device)).item()
            pmax = max(p.abs().max().item() for p in model.parameters())
            with log_path.open("a", newline="") as f:
                csv.writer(f).writerow([step, f"{loss.item():.4f}", f"{ev:.4f}", f"{pmax:.2f}"])
            print(f"step {step:5d}  train {loss.item():.4f}  eval {ev:.4f}  pmax {pmax:.1f}  ({time.time()-t0:.0f}s)")

        if step % args.ckpt_every == 0 or step == args.steps:
            save_checkpoint(model, step, loss.item(), out_dir, args.model.startswith("wskan"))
            model.eval()
            sample = model.generate(PROMPT, max_new=200, temperature=0.8)
            (out_dir / f"samples_step{step}.txt").write_bytes(sample)
            print(f"  [checkpoint + sample saved at step {step}]")

    print("done. artifacts in", out_dir)


if __name__ == "__main__":
    main()
