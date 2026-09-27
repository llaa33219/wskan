"""Benchmark-substrate demonstration (round-9): post-hoc methods vs exact
ground truth on the canonical 100k checkpoint.

The claimed value "a benchmark on which post-hoc interpretability methods
can be validated against exact answers" was aspirational; this script makes
it concrete.

  Demo 1: decision attribution. Exact per-position margin contributions
  (the model's own arithmetic, W11_hand_simulation) vs two standard
  post-hoc attributions: gradient x input saliency, and leave-one-out byte
  replacement. Spearman rank correlation + top-3 overlap per method.
  Demo 2: the linear-probe false negative. A linear probe on the write
  path concludes "no word identity" (R2 0.15); the exact machinery
  (nonlinear probe + readout analysis, round-5) shows the information is
  present but linearly inaccessible. The substrate adjudicates a wrong
  conclusion a practitioner would have drawn.

Outputs: figures/w11_benchmark_demo.json.
Run: PYTHONPATH=.:experiments .venv/bin/python experiments/W11_benchmark_demo.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "experiments")
from W11_hand_simulation import load_model, hand_forward, margin_decomposition, per_history_terms

DEV = "cuda"
CKPT = "checkpoints/wskan11_ultrachat_100k_3ep_s42/latest.pt"

CONTEXTS = [
    b"User: Hello! How are you today, my dear frien",
    b"The committee shall review the annual repor",
    b"Ingredients: 2 cups of flour, 1 cup of suga",
]


def exact_per_position(m, idx):
    hf = hand_forward(m, idx)
    parts, logits, std = margin_decomposition(m, idx, hf)
    t1, t2 = np.argsort(-logits)[:2]
    per_pos = np.zeros(idx.shape[1])
    for li in range(len(hf["layers"])):
        full = per_history_terms(m, hf, li, std)          # (L,I,N,vocab)
        per_pos += (full[..., t1] - full[..., t2]).sum(axis=(1, 2))
    return per_pos, logits, (int(t1), int(t2))


def saliency_per_position(m, idx):
    m.zero_grad()
    emb = m.tok_emb(idx).detach().requires_grad_(True)
    x = emb
    for norm, layer in zip(m.prenorms, m.layers):
        x = x + layer(norm(x), idx)
    logits = m.head(m.norm(x))[0, -1]
    t1, t2 = torch.topk(logits, 2).indices
    margin = logits[t1] - logits[t2]
    margin.backward()
    return emb.grad[0].norm(dim=-1).cpu().numpy()


def loo_per_position(m, idx, logits_ref, t12):
    t1, t2 = t12
    drops = np.zeros(idx.shape[1])
    for p in range(idx.shape[1]):
        idx2 = idx.clone()
        idx2[0, p] = 32   # replace with space
        with torch.no_grad():
            lg = m(idx2)[0, -1].cpu().numpy()
        drops[p] = (logits_ref[t1] - logits_ref[t2]) - (lg[t1] - lg[t2])
    return drops


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    m = load_model(CKPT, 40, 2)
    out = {}
    for ctx in CONTEXTS:
        idx = torch.tensor(list(ctx), device=DEV).unsqueeze(0)
        exact, logits, t12 = exact_per_position(m, idx)
        sal = saliency_per_position(m, idx)
        loo = loo_per_position(m, idx, logits, t12)
        top3_exact = set(np.argsort(-np.abs(exact))[:3].tolist())
        row = dict(
            context=ctx.decode("latin1"), pair=(chr(t12[0]), chr(t12[1])),
            saliency_spearman=spearman(np.abs(exact), sal),
            loo_spearman=spearman(np.abs(exact), np.abs(loo)),
            saliency_top3_overlap=len(top3_exact & set(np.argsort(-sal)[:3].tolist())),
            loo_top3_overlap=len(top3_exact & set(np.argsort(-np.abs(loo))[:3].tolist())),
        )
        tag = ctx[-8:].decode("latin1")
        out[tag] = row
        print(f"  ...{tag!r} ({row['pair'][0]!r}/{row['pair'][1]!r}): "
              f"saliency rho {row['saliency_spearman']:+.2f} top3 {row['saliency_top3_overlap']}/3 | "
              f"LOO rho {row['loo_spearman']:+.2f} top3 {row['loo_top3_overlap']}/3")
    with open("experiments/figures/w11_benchmark_demo.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved experiments/figures/w11_benchmark_demo.json")


if __name__ == "__main__":
    main()
