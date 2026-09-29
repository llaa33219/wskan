"""Ablation RETRAIN battery: task-requirement test, not solution-load test.

Round-6 review: surgical interventions on the trained checkpoint measure
"does THIS solution need the structure", not "does the TASK require it".
The decisive test is retraining with the structure impossible from the
start. Variants (100k tier, UltraChat, campaign 3ep protocol):

  n1        n_states=1        - no mode ladder is possible at all
  nofeat    use_feature_bc=False - no named structure/content tables
  rhofrozen log_rho frozen at 0  - modes share one timescale multiplier
  (reference: wskan11real freezes omega=0 - frequency structure removed)

Protocol matches CAMPAIGN_orchestrate.py for (ultrachat, 100k):
batch 64, block 512, lr 3e-3 cosine, 108k steps, bf16 scan, grad clip 1.0.

Run: .venv/bin/python experiments/W11_ablation_retrain.py --variant n1 --seed 42
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, ".")
from models.V11_WSKAN import WaveletStateKANLMV11


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["base", "n1", "n1wide", "nofeat", "rhofrozen", "flatomega"],
                   required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset", choices=["ultrachat", "tinystories"], default="ultrachat")
    p.add_argument("--tier", choices=["1k", "10k", "100k", "1m", "10m"], default="100k")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--eval-every", type=int, default=2000)
    args = p.parse_args()
    TIER_D = {"1k": 4, "10k": 12, "100k": 40, "1m": 80, "10m": 512}
    TIER_L = {"1k": 1, "10k": 1, "100k": 2, "1m": 6, "10m": 2}
    TIER_STEPS = {"1k": 36000, "10k": 54000, "100k": 108000, "1m": 54000, "10m": 36000}
    if args.steps is None:
        args.steps = TIER_STEPS[args.tier]
    if args.lr is None:
        args.lr = 3e-3 if args.tier in ("1k", "10k", "100k") else 1e-3

    from experiments.V1_train_tinystories_lm import load_data
    train_ids, eval_ids = load_data(1400000, 500, args.seed, args.dataset)
    eval_ids = eval_ids.long().cuda()

    torch.manual_seed(args.seed)
    kw = dict(vocab_size=256, d_model=TIER_D[args.tier], n_layers=TIER_L[args.tier],
              use_feature_bc=True, wz_diag=False, g_rank=None,
              bc_rank=min(32, TIER_D[args.tier]), bf16_scan=True)
    if args.variant == "n1":
        kw["n_states"] = 1
    if args.variant == "n1wide":
        kw["n_states"] = 1
        kw["d_model"] = {"10k": 18, "100k": 67, "1m": 132, "10m": 779}[args.tier]
        kw["bc_rank"] = min(32, kw["d_model"])
    if args.variant == "nofeat":
        kw["use_feature_bc"] = False
    m = WaveletStateKANLMV11(**kw).cuda()
    if args.variant == "rhofrozen":
        for layer in m.layers:
            layer.log_rho.requires_grad_(False)
    if args.variant == "flatomega":
        for layer in m.layers:
            layer.omega.data.zero_()   # pi-harmonic prior removed; omega stays trainable
    print(f"variant {args.variant} seed {args.seed} params {sum(q.numel() for q in m.parameters()):,}",
          flush=True)

    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    def get_batch(src, bs=None, blk=512):
        bs = bs or args.batch
        ix = torch.randint(len(src) - blk - 1, (bs,))
        return torch.stack([src[i:i + blk + 1] for i in ix]).long().cuda()

    ev_chunks = [eval_ids[s:s + 513] for s in range(0, 16 * 512, 512)]
    log = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        m.train()
        batch = get_batch(train_ids)
        lg = m(batch[:, :-1])
        loss = F.cross_entropy(lg.reshape(-1, 256), batch[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.eval_every == 0 or step == args.steps:
            m.eval()
            with torch.no_grad():
                ev = float(np_mean([F.cross_entropy(m(c[:-1].unsqueeze(0))[0], c[1:]).item()
                                    for c in ev_chunks]))
            log.append(dict(step=step, train=round(float(loss.item()), 4), eval=round(ev, 4)))
            print(f"step {step:6d} train {loss.item():.4f} eval {ev:.4f} ({time.time() - t0:.0f}s)",
                  flush=True)

    tag = "3ep" if (args.dataset == "ultrachat" and args.steps == 108000) else f"{args.steps // 1000}k"
    out_dir = Path(f"checkpoints/wskan11abl-{args.variant}_{args.dataset}_{args.tier}_{tag}_s{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": m.state_dict(), "variant": args.variant, "seed": args.seed},
               out_dir / "latest.pt")
    final_ce = sum(r["eval"] for r in log[-3:]) / 3
    with open(out_dir / "result.json", "w") as f:
        json.dump(dict(variant=args.variant, seed=args.seed, final_eval_ce=final_ce, log=log), f, indent=2)
    print(f"FINAL {args.variant} s{args.seed}: eval CE {final_ce:.4f} -> {out_dir}", flush=True)


def np_mean(xs):
    return sum(xs) / len(xs)


if __name__ == "__main__":
    main()
