"""Deep anatomy at every scale (round-19): the analyses that were
100k-centric, run at all five tiers (UltraChat, 3ep, seed 42).

  A. Exclusion battery per tier: linear R^2 families + held-out MLP word
     identity (+ shuffled-label control) on the last-layer write path.
  B. frien->d exact decision decomposition per tier (the same canonical
     context; where the model fails to produce 'd', we decompose the
     decision it actually makes).
  C. Hand-simulation truncation at 1m/10m (margin-targeted per-term
     ranking, memory-safe).

Outputs: figures/w11_deep_all_sizes.json
Run: PYTHONPATH=.:experiments .venv/bin/python experiments/W11_deep_all_sizes.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
sys.path.insert(0, "experiments")
from W11_excess_structure import load_model
from W11_hand_simulation import hand_forward, margin_decomposition, ctxs_from_eval
from W11_round5_probes import extract_signals, word_labels, mlp_probe_acc, linear_r2

DEV = "cuda"
CKPT = {t: f"checkpoints/wskan11_ultrachat_{t}_3ep_s42/latest.pt" for t in ("1k", "10k", "100k", "1m", "10m")}
DIMS = {"1k": (4, 1), "10k": (12, 1), "100k": (40, 2), "1m": (80, 6), "10m": (512, 2)}


def part_a(m, arr):
    sig = extract_signals(m, arr)
    X = sig["resB"].reshape(len(arr), -1)
    gid, ncls = word_labels(arr)
    sent = np.zeros(len(arr), dtype=np.int64)
    d = 0
    for i, c in enumerate(arr):
        sent[i] = min(d, 60)
        d = 0 if c in b".!?\n" else d + 1
    byte32 = arr.astype(np.int64) % 32
    bigram = np.array([arr[i - 1] % 16 * 16 + arr[i] % 16 if i > 0 else -1 for i in range(len(arr))])
    def reg_r2(y):
        Xd = torch.tensor(X, dtype=torch.float32, device=DEV)
        yv = torch.tensor(y.astype(np.float32), device=DEV)
        A = torch.cat([Xd, torch.ones(len(Xd), 1, device=DEV)], 1)
        sol = torch.linalg.lstsq(A, yv.unsqueeze(-1)).solution
        pred = (A @ sol).squeeze(-1)
        return float(1 - ((pred - yv) ** 2).sum() / ((yv - yv.mean()) ** 2).sum())
    mlp = mlp_probe_acc(X, gid, ncls, seed=0)
    rng = np.random.default_rng(0)
    sh = gid.copy(); keep = sh >= 0
    sh[keep] = rng.permutation(sh[keep])
    ctl = mlp_probe_acc(X, sh, ncls, seed=1)
    return dict(byte_linear_r2=round(linear_r2(X, byte32, 32), 3),
                bigram_linear_r2=round(linear_r2(X, bigram, 256), 3),
                word_linear_r2=round(linear_r2(X, gid, ncls), 3),
                sentpos_linear_r2=round(reg_r2(sent), 4),
                word_mlp_heldout=round(mlp, 3), word_mlp_shuffled=round(ctl, 3))


def part_b(m):
    ctx = b"User: Hello! How are you today, my dear frien"
    idx = torch.tensor(list(ctx), device=DEV).unsqueeze(0)
    hf = hand_forward(m, idx)
    parts, logits, std = margin_decomposition(m, idx, hf)
    W = m.head.weight.detach().cpu().numpy()
    t1, t2 = np.argsort(-logits)[:2]
    margin = float(logits[t1] - logits[t2])
    row = {"pred": chr(int(t1)), "runner_up": chr(int(t2)), "margin": round(margin, 2)}
    for k in ("emb", "norm-const"):
        row[k] = round(float((W[t1] - W[t2]) @ parts[k]), 2)
    for li in range(len(hf["layers"])):
        for p in ("base", "wave"):
            row[f"L{li}-{p}"] = round(float((W[t1] - W[t2]) @ parts[f"L{li}-{p}"]), 2)
    row["produces_d"] = bool(chr(int(t1)) == "d")
    return row


def part_c(m, ds, tier):
    from W11_hand_simulation import per_history_terms
    ks, k0 = [], 0
    for w in ctxs_from_eval(ds, 50, 24, seed=7):
        idx = torch.tensor([int(c) for c in w[:-1]], device=DEV).unsqueeze(0)
        hf = hand_forward(m, idx)
        parts, logits, std = margin_decomposition(m, idx, hf)
        v_full = int(np.argmax(logits))
        flats = []
        for li in range(len(hf["layers"])):
            full = per_history_terms(m, hf, li, std)      # (L,I,N,vocab)
            flats.append(full.reshape(-1, full.shape[-1]))
            logits = logits - full.sum(axis=(0, 1, 2))    # strip this layer's history
        flat = np.concatenate(flats, axis=0)
        base_logits = logits
        order_t = np.argsort(-np.abs(flat[:, v_full]))
        k0 += int(int(np.argmax(base_logits)) == v_full)
        for k in range(1, len(order_t) + 1):
            if int(np.argmax(base_logits + flat[order_t[:k]].sum(0))) == v_full:
                ks.append(k)
                break
        else:
            ks.append(len(order_t))
    return dict(median_k=float(np.median(ks)), p90_k=float(np.percentile(ks, 90)),
                frac_le25=float(np.mean(np.array(ks) <= 25)), n=len(ks),
                frac_k0=round(k0 / len(ks), 3), total_terms=int(flat.size))


def main():
    from experiments.V1_train_tinystories_lm import load_data
    tr, _ = load_data(1400000, 500, 42, "ultrachat")
    arr = tr[:4096].numpy()
    out = {"A_exclusion": {}, "B_friend": {}, "C_truncation": {}}
    for tier, (d, L) in DIMS.items():
        print(f"== {tier} (d={d}, L={L})")
        m = load_model(CKPT[tier], d, L)
        a = part_a(m, arr)
        out["A_exclusion"][tier] = a
        print(f"   A: byte {a['byte_linear_r2']} bigram {a['bigram_linear_r2']} "
              f"word-lin {a['word_linear_r2']} word-mlp {a['word_mlp_heldout']} (ctl {a['word_mlp_shuffled']}) "
              f"sentpos {a['sentpos_linear_r2']}")
        b = part_b(m)
        out["B_friend"][tier] = b
        print(f"   B: frien->... pred {b['pred']!r} (margin {b['margin']}) produces_d={b['produces_d']}")
        if tier in ("1m", "10m"):
            c = part_c(m, "ultrachat", tier)
            out["C_truncation"][tier] = c
            print(f"   C: median {c['median_k']:.0f} p90 {c['p90_k']:.0f} of {c['total_terms']} terms, "
                  f"<=25 in {c['frac_le25']*100:.0f}%")
    with open("experiments/figures/w11_deep_all_sizes.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("saved experiments/figures/w11_deep_all_sizes.json")


if __name__ == "__main__":
    main()
