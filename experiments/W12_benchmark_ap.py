"""W12 attribution-patching benchmark row (review round 2, 2026-10-04).

The benchmark scored gradient saliency, IG, replacement, and LOO against
the exact arithmetic decomposition - but the method family that actually
targets causal contribution, attribution patching (EAP; Syed et al. 2023),
was only cited, not run. This adds the missing row.

Per-position single-corruption EAP: for each position p, corrupt that
byte with a random corpus byte, and approximate the effect of patching
the clean byte back as (emb_clean[p] - emb_corr[p]) . d margin/d emb[p]
evaluated on the corrupted run (standard EAP sign convention). Scored
against the same exact per-position arithmetic on the same 3x50 contexts
as W11_benchmark_demo.py (identical selection rules and seeds).

Outputs: figures/w12_benchmark_ap.json
Run: PYTHONPATH=.:experiments .venv/bin/python experiments/W12_benchmark_ap.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "experiments")
from W11_benchmark_demo import exact_per_position, spearman
from W11_hand_simulation import load_model

DEV = "cuda"
CKPT = "checkpoints/wskan11_ultrachat_100k_3ep_s42/latest.pt"


def ap_per_position(m, idx, t12, rng):
    t1, t2 = t12
    emb_clean = m.tok_emb(idx).detach()
    attrs = np.zeros(idx.shape[1])
    for p in range(idx.shape[1]):
        idx2 = idx.clone()
        idx2[0, p] = int(rng.integers(0, 256))
        emb2 = m.tok_emb(idx2).detach().requires_grad_(True)
        x = emb2
        for norm, layer in zip(m.prenorms, m.layers):
            x = x + layer(norm(x), idx2)
        logits = m.head(m.norm(x))[0, -1]
        margin = logits[t1] - logits[t2]
        m.zero_grad()
        margin.backward()
        g = emb2.grad[0, p]
        attrs[p] = float(((emb_clean[0, p] - emb2[0, p].detach()) * g).sum())
    return attrs


def main():
    m = load_model(CKPT, 40, 2)
    from experiments.V1_train_tinystories_lm import load_data
    tr, ev = load_data(1400000, 500, 42, "ultrachat")
    tr_arr = tr.numpy()[:2_000_000]
    ev_arr = ev.numpy()
    from collections import defaultdict
    stats = defaultdict(lambda: np.zeros(256, dtype=np.int64))
    for s in range(len(tr_arr) - 5):
        stats[bytes(tr_arr[s:s + 4])][tr_arr[s + 4]] += 1
    rng = np.random.default_rng(3)
    cand = rng.choice(len(ev_arr) - 40, size=8000, replace=False)
    contexts = []
    for s in cand:
        ctxb = ev_arr[s:s + 24]
        key = bytes(ctxb[-4:])
        if not all(chr(c).isalpha() for c in key):
            continue
        cnt = stats.get(key)
        if cnt is None or cnt.sum() < 8:
            continue
        if cnt.max() / cnt.sum() < 0.75:
            continue
        contexts.append(ctxb)
        if len(contexts) >= 50:
            break
    rng2 = np.random.default_rng(11)
    cand2 = rng2.choice(len(ev_arr) - 40, size=4000, replace=False)
    bctx = []
    for s in cand2:
        ctxb = ev_arr[s:s + 24]
        if ctxb[-1] == 32 and all(chr(c).isalpha() for c in ctxb[-4:-1]):
            bctx.append(ctxb)
        if len(bctx) >= 50:
            break
    rng3 = np.random.default_rng(17)
    cand3 = rng3.choice(len(ev_arr) - 40, size=4000, replace=False)
    pctx = []
    for s in cand3:
        ctxb = ev_arr[s:s + 24]
        if ctxb[-1] == 46 and all(chr(c).isalpha() or c == 32 for c in ctxb[-6:-1]):
            pctx.append(ctxb)
        if len(pctx) >= 50:
            break

    out = {}
    for name, ctx_list in (("morphology_50", contexts), ("word_initial_50", bctx),
                           ("post_punct_50", pctx)):
        rhos = []
        rng = np.random.default_rng(5)
        for ctxb in ctx_list:
            idx = torch.tensor([int(c) for c in ctxb], device=DEV).unsqueeze(0)
            exact, logits, t12 = exact_per_position(m, idx)
            ap = ap_per_position(m, idx, t12, rng)
            rhos.append(spearman(np.abs(exact), np.abs(ap)))
        out[name] = dict(ap_mean=round(float(np.mean(rhos)), 3),
                         ap_std=round(float(np.std(rhos)), 3))
        print(f"  {name}: AP rho {out[name]['ap_mean']:+.3f} +- {out[name]['ap_std']:.3f}", flush=True)
    with open("experiments/figures/w12_benchmark_ap.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved experiments/figures/w12_benchmark_ap.json")


if __name__ == "__main__":
    main()
