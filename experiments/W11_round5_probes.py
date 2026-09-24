"""W11 round-5 probes: (1) the usefulness ladder for word identity,
(2) identifying the 10M non-local write component.

Part 1 answers the round-5 distinction "what the model computes vs what it
can use": a nonlinear probe recovers word identity from the write path
(round-4), but the model's own readout is linear - so probe the signal the
model actually USES (post-gate wave output y) and the residual stream the
head sees, with the same 500-way word-identity task, held-out split.

Part 2 attacks "non-local context component (~25-40%) at 10M, unidentified
yet": sequential family-R2 decomposition of the last-layer residual-B write
at the 10M UltraChat tier - byte identity, bigram, trigram, word identity,
previous word, position-in-word, sentence position - what explains the
non-local share, and how much remains genuinely unidentified.

Outputs: figures/w11_round5_probes.json.
Run: .venv/bin/python experiments/W11_round5_probes.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from models.V11_WSKAN import WaveletStateKANLMV11

DEV = "cuda"


def load_model(ckpt: str, d: int, L: int) -> WaveletStateKANLMV11:
    m = WaveletStateKANLMV11(vocab_size=256, d_model=d, n_layers=L, use_feature_bc=True,
                             wz_diag=False, g_rank=None, bc_rank=min(32, d), bf16_scan=False)
    sd = torch.load(ckpt, weights_only=False)
    m.load_state_dict(sd["state_dict"] if "state_dict" in sd else sd)
    return m.eval().to(DEV)


@torch.no_grad()
def extract_signals(m, arr: np.ndarray) -> dict:
    """Forward over the byte stream; collect per-position signals at the last
    layer: residual-B write, post-gate wave output y, residual stream pre-norm."""
    idx = torch.tensor(arr.astype(np.int64), device=DEV).unsqueeze(0)
    x = m.tok_emb(idx)
    sig = {}
    for li, (norm, layer) in enumerate(zip(m.prenorms, m.layers)):
        h = norm(x)
        if li == len(m.layers) - 1:
            sig["resB"] = layer.res_B(h)[0].detach().cpu().numpy()
            Bsz, Ls, I, N = h.shape[0], h.shape[1], layer.in_dim, layer.n_states
            sigma = torch.exp(torch.clamp(layer.log_sigma, layer.log_sigma_min, layer.log_sigma_max))
            rho = layer._rho()
            lam = torch.complex(-sigma * rho, layer.omega * rho)
            dt = torch.clamp(F.softplus(layer.W_dt(h)), max=1.0)
            from models.V7_WSKAN import byte_features
            Bn = torch.einsum("blf,fin->blin", byte_features(idx), layer.M_B) \
                + layer.res_B(h).view(Bsz, Ls, I, N)
            Cn = torch.einsum("blf,fin->blin", byte_features(idx), layer.M_C) \
                + layer.res_C(h).view(Bsz, Ls, I, N)
            u = Bn * h.unsqueeze(-1) * dt.unsqueeze(-1)
            hst = torch.zeros(Bsz, I, N, dtype=torch.complex64, device=DEV)
            ys = []
            for n in range(Ls):
                a_n = torch.exp(lam * dt[:, n].unsqueeze(-1))
                hst = a_n * hst + u[:, n].to(torch.complex64)
                rp = Cn[:, n] * hst.real
                ip = Cn[:, n] * hst.imag
                yp = torch.einsum("bik,iok->bo", rp, layer.a) - torch.einsum(
                    "bik,iok->bo", ip, layer.b)
                ys.append(yp * F.silu(layer.W_z(h[:, n])))
            sig["y_used"] = torch.stack(ys, dim=1)[0].detach().cpu().numpy()
        x = x + layer(h, idx)
    sig["stream"] = x[0].detach().cpu().numpy()
    return sig


def word_labels(arr: np.ndarray, top: int = 500):
    words, cur = [], []
    for c in arr:
        if c in (32, 10):
            if cur:
                words.append(bytes(cur))
            cur = []
        else:
            cur.append(int(c))
    fids = {w: k for k, (w, _) in enumerate(Counter(words).most_common(top))}
    gid = np.full(len(arr), -1)
    start, buf = 0, []
    for i, c in enumerate(arr):
        if c in (32, 10):
            gid[start:i + 1] = fids.get(bytes(buf), -1)
            buf, start = [], i + 1
        else:
            buf.append(int(c))
    gid[start:] = fids.get(bytes(buf), -1)
    return gid, len(fids)


def mlp_probe_acc(X: np.ndarray, y: np.ndarray, ncls: int, seed: int = 0) -> float:
    """1-hidden-layer MLP, 80/20 split, HELD-OUT accuracy."""
    torch.manual_seed(seed)
    keep = y >= 0
    Xk = torch.tensor(X[keep], dtype=torch.float32, device=DEV)
    yk = torch.tensor(y[keep], dtype=torch.long, device=DEV)
    n = len(Xk)
    perm = torch.randperm(n, device=DEV)
    ntr = int(n * 0.8)
    tri, tei = perm[:ntr], perm[ntr:]
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 256), torch.nn.ReLU(),
                              torch.nn.Linear(256, ncls)).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    ce = torch.nn.CrossEntropyLoss()
    for _ in range(40):
        p2 = tri[torch.randperm(len(tri), device=DEV)]
        for i in range(0, len(p2), 4096):
            opt.zero_grad()
            loss = ce(net(Xk[p2[i:i + 4096]]), yk[p2[i:i + 4096]])
            loss.backward()
            opt.step()
    with torch.no_grad():
        pred = net(Xk[tei]).argmax(-1)
    return float((pred == yk[tei]).float().mean())


def linear_r2(X: np.ndarray, y: np.ndarray, ncls: int) -> float:
    keep = y >= 0
    Xk = torch.tensor(X[keep], dtype=torch.float32, device=DEV)
    yk = torch.tensor(y[keep], dtype=torch.long, device=DEV)
    gm = torch.stack([Xk[yk == g].mean(0) for g in range(ncls)])
    pred = gm[yk]
    r2 = 1 - ((Xk - pred) ** 2).sum(0) / ((Xk - Xk.mean(0)) ** 2).sum(0)
    return float(r2.mean())


def part1():
    print("=" * 70)
    print("Part 1: usefulness ladder (100k canonical, 500-way word identity)")
    from experiments.V1_train_tinystories_lm import load_data
    tr, _ = load_data(1400000, 500, 42, "ultrachat")
    arr = tr[:4096].numpy()
    m = load_model("checkpoints/wskan11_ultrachat_100k_3ep_s42/latest.pt", 40, 2)
    sig = extract_signals(m, arr)
    gid, ncls = word_labels(arr)
    out = {}
    for name, X in sig.items():
        X2 = X.reshape(len(arr), -1)
        acc = mlp_probe_acc(X2, gid, ncls)
        r2 = linear_r2(X2, gid, ncls)
        out[name] = dict(linear_r2=round(r2, 4), mlp_acc_heldout=round(acc, 4))
        print(f"  {name:>8}: linear R2 {r2:.3f} | MLP held-out acc {acc:.3f} (chance ~{1 / ncls:.3f})")
    return out


def part2():
    print("=" * 70)
    print("Part 2: identifying the 10M non-local write component (UltraChat 10m)")
    from experiments.V1_train_tinystories_lm import load_data
    tr, _ = load_data(1400000, 500, 42, "ultrachat")
    arr = tr[:8192].numpy()
    m = load_model("checkpoints/wskan11_ultrachat_10m_3ep_s42/latest.pt", 512, 2)
    sig = extract_signals(m, arr)
    X = sig["resB"].reshape(len(arr), -1)
    Xt = torch.tensor(X, dtype=torch.float32, device=DEV)

    def labels_group(keys: np.ndarray, top: int):
        cnt = Counter(keys)
        vocab = {k: i for i, (k, _) in enumerate(cnt.most_common(top))}
        return np.array([vocab.get(k, -1) for k in keys]), len(vocab)

    n = len(arr)
    arr64 = arr.astype(np.int64)
    byte32, nb = labels_group(arr64.copy(), 32)
    bigram, nbg = labels_group(np.array([arr64[i - 1] * 256 + arr64[i] if i > 0 else -1 for i in range(n)]), 128)
    trigram, ntg = labels_group(np.array([arr64[i - 2] * 65536 + arr64[i - 1] * 256 + arr64[i] if i > 1 else -1
                                          for i in range(n)]), 2000)
    gid, nw = word_labels(arr, 500)
    prevw = np.full(n, -1)
    seen = -1
    buf = []
    starts = 0
    wid = {}
    cnt = Counter()
    words, cur = [], []
    for c in arr:
        if c in (32, 10):
            if cur:
                words.append(bytes(cur))
            cur = []
        else:
            cur.append(int(c))
    wids = {w: k for k, (w, _) in enumerate(Counter(words).most_common(500))}
    prev = -1
    start, buf = 0, []
    for i, c in enumerate(arr):
        if c in (32, 10):
            curid = wids.get(bytes(buf), -1)
            prevw[start:i + 1] = prev
            prev = curid
            buf, start = [], i + 1
        else:
            buf.append(int(c))
    posw = np.zeros(n, dtype=np.int64)
    d = 0
    for i, c in enumerate(arr):
        posw[i] = min(d, 15)
        d = 0 if c in (32, 10) else d + 1
    sentp = np.zeros(n, dtype=np.int64)
    d = 0
    for i, c in enumerate(arr):
        sentp[i] = min(d, 60)
        d = 0 if c in b".!?\n" else d + 1

    families = [("byte-32", byte32, nb), ("bigram-128", bigram, nbg), ("trigram-2k", trigram, ntg),
                ("word-id-500", gid, nw), ("prev-word-500", prevw, nw), ("pos-in-word-16", posw, 16),
                ("sent-pos-60", sentp, 61)]

    def group_pred(Xt, y, ncls):
        keep = y >= 0
        Xk, yk = Xt[keep], torch.tensor(y[keep], dtype=torch.long, device=DEV)
        gm = torch.stack([Xk[yk == g].mean(0) for g in range(ncls)])
        pred = torch.zeros_like(Xt)
        pred[keep] = gm[yk]
        return pred

    resid = Xt.clone()
    total_var = ((Xt - Xt.mean(0)) ** 2).sum(0)
    out = {"total_var_mean": float(total_var.mean())}
    explained = 0.0
    for name, y, ncls in families:
        pred = group_pred(resid, y, ncls)
        r2_gain = float((1 - ((resid - pred) ** 2).sum(0) / ((resid - resid.mean(0)) ** 2).sum(0)).mean())
        resid = resid - pred
        explained = 1 - float((((resid - resid.mean(0)) ** 2).sum(0) / total_var).mean())
        out[name] = dict(r2_gain_alone=round(r2_gain, 4), cumulative_explained=round(explained, 4))
        print(f"  {name:>15}: family R2 {r2_gain:+.3f} | cumulative explained {explained:.3f}")
    out["unidentified_remainder"] = round(1 - explained, 4)
    print(f"  unidentified remainder: {1 - explained:.3f}")
    return out


def main():
    out = {"part1_usefulness_ladder": part1(), "part2_10m_families": part2()}
    with open("experiments/figures/w11_round5_probes.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved experiments/figures/w11_round5_probes.json")


if __name__ == "__main__":
    main()
