"""W12 probe null-calibration battery (review response, 2026-10-04).

The per-tier linear R2 table (paper, exclusion section) used an in-sample
class-conditional-mean R2 with no null calibration; for a k-class task on n
positions the in-sample null is ~(k-1)/(n-1), which for word identity
(~500 classes, 4096 positions) is ~0.12 - at the reported values. This
script re-measures the battery with null calibration:

  Per tier x label family (byte-32 / bigram-256 / word-500):
    A. in-sample class-mean R2 (the original statistic, for comparability)
       + shuffled-label null (mean of 5 permutations)
    B. held-out ridge regression on class one-hots, alpha sweep, 3 splits:
       best held-out R2, held-out argmax accuracy, shuffled control

  C. Readout alignment (100k only): cosine between the ridge word-identity
     direction w in the write path and the model's own sensitivity to the
     write path, r = mean_n d(sum CE)/d(resB_n) - "does the model's actual
     readout point where the word-identity information is". Null 1/sqrt(dim)
     plus random-direction control.

Outputs: figures/w12_probe_nulls.json
Run: PYTHONPATH=.:experiments .venv/bin/python experiments/W12_probe_nulls.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "experiments")
from W11_excess_structure import load_model
from W11_round5_probes import extract_signals, word_labels, linear_r2

DEV = "cuda"
CKPT = {t: f"checkpoints/wskan11_ultrachat_{t}_3ep_s42/latest.pt" for t in ("1k", "10k", "100k", "1m", "10m")}
DIMS = {"1k": (4, 1), "10k": (12, 1), "100k": (40, 2), "1m": (80, 6), "10m": (512, 2)}
ALPHAS = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]


def make_labels(arr):
    byte32 = arr.astype(np.int64) % 32
    bigram = np.array([arr[i - 1] % 16 * 16 + arr[i] % 16 if i > 0 else -1 for i in range(len(arr))])
    gid, nw = word_labels(arr)
    return {"byte32": (byte32, 32), "bigram256": (bigram, 256), "word500": (gid, nw)}


def insample_with_null(X, y, ncls, n_shuffles=5):
    r2 = linear_r2(X, y, ncls)
    rng = np.random.default_rng(0)
    nulls = []
    for _ in range(n_shuffles):
        ys = y.copy()
        keep = ys >= 0
        ys[keep] = rng.permutation(ys[keep])
        nulls.append(linear_r2(X, ys, ncls))
    return r2, float(np.mean(nulls))


def ridge_sweep(X, y, ncls, n_splits=3):
    keep = y >= 0
    Xk = torch.tensor(X[keep], dtype=torch.float32, device=DEV)
    yk = torch.tensor(y[keep], dtype=torch.long, device=DEV)
    Y = torch.zeros(len(yk), ncls, device=DEV)
    Y[torch.arange(len(yk)), yk] = 1.0
    n = len(yk)
    best = None
    for a in ALPHAS:
        r2s, accs = [], []
        for sp in range(n_splits):
            g = torch.Generator(device=DEV).manual_seed(sp)
            perm = torch.randperm(n, device=DEV, generator=g)
            ntr = int(n * 0.8)
            tri, tei = perm[:ntr], perm[ntr:]
            Xtr, Xte = Xk[tri], Xk[tei]
            Ytr, Yte = Y[tri], Y[tei]
            Xm, Ym = Xtr.mean(0), Ytr.mean(0)
            Xc, Yc = Xtr - Xm, Ytr - Ym
            p = Xc.shape[1]
            W = torch.linalg.solve(Xc.T @ Xc + a * torch.eye(p, device=DEV), Xc.T @ Yc)
            pred = (Xte - Xm) @ W + Ym
            ss_res = ((pred - Yte) ** 2).sum()
            ss_tot = ((Yte - Yte.mean(0)) ** 2).sum()
            r2s.append(float(1 - ss_res / ss_tot))
            accs.append(float((pred.argmax(-1) == yk[tei]).float().mean()))
        r2, acc = float(np.mean(r2s)), float(np.mean(accs))
        if best is None or r2 > best["r2"]:
            best = dict(alpha=a, r2=r2, acc=acc)
    ys = y.copy()
    ys[keep] = np.random.default_rng(1).permutation(ys[keep])
    keep2 = ys >= 0
    Xs = torch.tensor(X[keep2], dtype=torch.float32, device=DEV)
    ysT = torch.tensor(ys[keep2], dtype=torch.long, device=DEV)
    Ys = torch.zeros(len(ysT), ncls, device=DEV)
    Ys[torch.arange(len(ysT)), ysT] = 1.0
    ns = len(ysT)
    g = torch.Generator(device=DEV).manual_seed(0)
    perm = torch.randperm(ns, device=DEV, generator=g)
    ntr = int(ns * 0.8)
    Xtr, Xte = Xs[perm[:ntr]], Xs[perm[ntr:]]
    Ytr, Yte = Ys[perm[:ntr]], Ys[perm[ntr:]]
    Xm, Ym = Xtr.mean(0), Ytr.mean(0)
    Xc, Yc = Xtr - Xm, Ytr - Ym
    W = torch.linalg.solve(Xc.T @ Xc + best["alpha"] * torch.eye(Xc.shape[1], device=DEV), Xc.T @ Yc)
    pred = (Xte - Xm) @ W + Ym
    null_acc = float((pred.argmax(-1) == ysT[perm[ntr:]]).float().mean())
    best["null_acc"] = null_acc
    return best


def ridge_direction(X, y, ncls, alpha):
    keep = y >= 0
    Xk = torch.tensor(X[keep], dtype=torch.float32, device=DEV)
    yk = torch.tensor(y[keep], dtype=torch.long, device=DEV)
    Y = torch.zeros(len(yk), ncls, device=DEV)
    Y[torch.arange(len(yk)), yk] = 1.0
    Xc, Yc = Xk - Xk.mean(0), Y - Y.mean(0)
    W = torch.linalg.solve(Xc.T @ Xc + alpha * torch.eye(Xc.shape[1], device=DEV), Xc.T @ Yc)
    return W  # (p, ncls)


def readout_alignment(m, arr, X, y, ncls, alpha, n_positions=2048):
    W = ridge_direction(X, y, ncls, alpha)  # (p, ncls) probe weight matrix
    U, S, _ = torch.linalg.svd(W, full_matrices=False)
    u1 = U[:, 0]
    u1 = u1 / u1.norm()
    k = 10
    Uk = U[:, :k]
    idx = torch.tensor(arr[: n_positions + 1].astype(np.int64), device=DEV).unsqueeze(0)
    saved = {}

    def hook(mod, inp, out):
        out.retain_grad()
        saved["resB"] = out

    hnd = m.layers[-1].res_B.register_forward_hook(hook)
    logits = m(idx[:, :-1])
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 256), idx[:, 1:].reshape(-1))
    loss.backward()
    hnd.remove()
    G = saved["resB"].grad[0]  # (L, I*N) total-sensitivity per position
    r = G.mean(0)
    r = r / r.norm()
    cos_u1 = float((u1 * r).sum())
    overlap_k = float(((Uk.T @ r) ** 2).sum())
    dim = r.numel()
    rng = np.random.default_rng(0)
    rn = r.cpu().numpy()
    rand_cos = []
    for _ in range(1000):
        v = rng.standard_normal(dim)
        v /= np.linalg.norm(v)
        rand_cos.append(abs(float(v @ rn)))
    return dict(cos_r_top1_singular=round(cos_u1, 4),
                null_1_over_sqrt_dim=round(1 / dim ** 0.5, 4),
                overlap_r_top10_subspace=round(overlap_k, 4),
                null_overlap_top10=round(k / dim, 4),
                random_control_mean_abscos=round(float(np.mean(rand_cos)), 4),
                dim=dim, n_positions=n_positions)


def main():
    from experiments.V1_train_tinystories_lm import load_data
    tr, _ = load_data(1400000, 500, 42, "ultrachat")
    out = {"A_insample_null": {}, "B_ridge_heldout": {}, "C_readout_alignment": {}}
    for tier, (d, L) in DIMS.items():
        print(f"== {tier} (d={d}, L={L})", flush=True)
        m = load_model(CKPT[tier], d, L)
        arr = tr[:4096].numpy()
        sig = extract_signals(m, arr)
        X = sig["resB"].reshape(len(arr), -1)
        labels = make_labels(arr)
        out["A_insample_null"][tier] = {}
        out["B_ridge_heldout"][tier] = {}
        for name, (y, ncls) in labels.items():
            r2, null = insample_with_null(X, y, ncls)
            out["A_insample_null"][tier][name] = dict(r2=round(r2, 4), shuffled_null=round(null, 4))
            best = ridge_sweep(X, y, ncls)
            out["B_ridge_heldout"][tier][name] = {k: (round(v, 4) if isinstance(v, float) else v)
                                                  for k, v in best.items()}
            print(f"   {name:>9}: in-sample R2 {r2:.3f} (null {null:.3f}) | "
                  f"ridge heldout R2 {best['r2']:.3f} acc {best['acc']:.3f} "
                  f"(null {best['null_acc']:.3f}, alpha {best['alpha']})", flush=True)
        if tier == "100k":
            y500, ncls = labels["word500"]
            al = readout_alignment(m, arr, X, y500, ncls,
                                   alpha=out["B_ridge_heldout"][tier]["word500"]["alpha"])
            out["C_readout_alignment"][tier] = al
            print(f"   readout alignment: cos(r, u1) {al['cos_r_top1_singular']:.4f} "
                  f"(null {al['null_1_over_sqrt_dim']:.4f}, rand {al['random_control_mean_abscos']:.4f}) "
                  f"top10-overlap {al['overlap_r_top10_subspace']:.4f} (null {al['null_overlap_top10']:.4f})",
                  flush=True)
        del m
        torch.cuda.empty_cache()
    with open("experiments/figures/w12_probe_nulls.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved experiments/figures/w12_probe_nulls.json")


if __name__ == "__main__":
    main()
