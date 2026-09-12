"""10M-scale interpretability replication on wskan7bc (d512, L2) x 3 seeds.

Replicates the canonical story at scale: word clock, causal battery,
whitespace-removal test, residual-B context battery, memory profile,
frien->d mode attribution. Plus a cross-dataset clock check (s42).

Output: figures/v7e_10m_analysis.json
Run: .venv/bin/python experiments/W7BC_10m_analysis.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.V1_train_tinystories_lm import get_batch, load_data
from models.V7_WSKAN import WaveletStateKANLMV7

device = "cuda"
torch.manual_seed(0)
OUT = Path(__file__).resolve().parent / "figures"

CLASSES = {
    "space": lambda t: t == 32, "newline": lambda t: t == 10,
    "punct": lambda t: np.isin(t, list(b".,!?;:'\"()-")),
    "upper": lambda t: (t >= 65) & (t <= 90),
    "lower": lambda t: (t >= 97) & (t <= 122),
    "digit": lambda t: (t >= 48) & (t <= 57),
}
BOUNDARY = torch.zeros(256, dtype=torch.bool)
for c in b" \n.,!?;:'\"()-":
    BOUNDARY[c] = True
LOWER = torch.tensor([(97 <= c <= 122) for c in range(256)])


def load_10m(seed):
    m = WaveletStateKANLMV7(vocab_size=256, d_model=512, n_layers=2, n_states=6,
                            chunk_size=8, use_feature_bc=True, wz_diag=False,
                            g_rank=None, bc_rank=32)
    m.load_state_dict(torch.load(f"checkpoints/wskan7bc_ultrachat_10m_s{seed}/latest.pt",
                                 weights_only=False)["state_dict"])
    return m.eval().cuda()


train_ids, eval_ids = load_data(200000, 500, 42, "ultrachat")
probe = train_ids[:2048].numpy().astype(np.int64)
B, L = 32, 512
batch = torch.cat([get_batch(eval_ids, 1, L, device) for _ in range(B)], 0)
b_eval, targets = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()


def word_gids(text):
    words, cur = [], []
    for c in text:
        if c in (32, 10):
            if cur: words.append(bytes(cur))
            cur = []
        else:
            cur.append(c)
    top = {w for w, _ in Counter(words).most_common(500)}
    fids = {w: k for k, w in enumerate(sorted(top))}
    gid = np.full(len(text), -1); start = 0; buf = []
    for i, c in enumerate(text):
        if c in (32, 10):
            gid[start:i + 1] = fids.get(bytes(buf), -1)
            buf = []; start = i + 1
        else:
            buf.append(c)
    gid[start:] = fids.get(bytes(buf), -1)
    return gid


def r2_of(R, groups):
    groups = [g for g in groups if g.sum() > 0]
    gidx = np.full(len(R), -1, dtype=int)
    for gi, m in enumerate(groups): gidx[m] = gi
    keep = gidx >= 0
    gm = np.stack([R[m].mean(0) for m in groups])
    pred = gm[gidx[keep]]
    Rk = R[keep]
    return float(np.mean(1 - ((Rk - pred) ** 2).sum(0) / ((Rk - Rk.mean(0)) ** 2).sum(0)))


def analyze(seed):
    model = load_10m(seed)
    R = {}
    idx_probe = torch.tensor(probe, device=device).unsqueeze(0)
    hiddens, dts = [], []
    with torch.no_grad():
        x = model.tok_emb(idx_probe)
        for norm, layer in zip(model.prenorms, model.layers):
            h = norm(x)
            hiddens.append(h[0].cpu())
            dts.append(layer._compute_dt(h)[0].cpu())
            x = x + layer(h, idx_probe)

    # A. word clock
    R["dt_by_class"] = {}
    for cname, pred in CLASSES.items():
        m = pred(probe)
        R["dt_by_class"][cname] = {f"L{li}": round(float(dts[li][m].mean()), 4) for li in range(2)}
    R["dt_ratio_space_over_lower"] = {f"L{li}": round(
        R["dt_by_class"]["space"][f"L{li}"] / R["dt_by_class"]["lower"][f"L{li}"], 3) for li in range(2)}

    # memory profile
    for li, layer in enumerate(model.layers):
        sigma = torch.exp(layer.log_sigma.detach().clamp(-6, 8)).cpu()
        rho = torch.exp(layer.log_rho.detach().clamp(-3, 3)).cpu()
        mean_dt = dts[li].mean(0)
        hl = (0.693 / (rho * sigma * mean_dt.unsqueeze(-1))).flatten()
        ceil = float((layer.log_sigma.detach() > 1.95).float().mean())
        R[f"memory_L{li}"] = {"sigma_mean": round(float(sigma.mean()), 3),
                              "frac_at_ceiling": round(ceil, 3),
                              "half_life_med": round(float(hl.median()), 2),
                              "half_life_p99": round(float(np.quantile(hl, .99)), 2),
                              "half_life_max": round(float(hl.max()), 2)}

    # E1 whitespace removal
    stripped = probe[probe != 32]
    j, bound_mask = 0, np.zeros(len(stripped), dtype=bool)
    for i, c in enumerate(probe):
        if c != 32:
            if i > 0 and probe[i - 1] == 32:
                bound_mask[j] = True
            j += 1
    idx_b = torch.tensor(stripped, device=device).unsqueeze(0)
    dtb = []
    with torch.no_grad():
        x = model.tok_emb(idx_b)
        for norm, layer in zip(model.prenorms, model.layers):
            h = norm(x)
            dtb.append(layer._compute_dt(h)[0].cpu())
            x = x + layer(h, idx_b)
    R["E1_nospace_ratio"] = {}
    for li in range(2):
        db = dtb[li]
        R["E1_nospace_ratio"][f"L{li}"] = round(float(db[bound_mask].mean() / db[~bound_mask].mean()), 3)

    # B. causal battery
    orig = {i: layer._compute_dt for i, layer in enumerate(model.layers)}
    for li, layer in enumerate(model.layers):
        def mk(li):
            def p(x):
                dt = orig[li](x)
                cfg = getattr(model.layers[li], "_iv", None)
                if cfg is None:
                    return dt
                kind, mask = cfg
                Lx = x.shape[1]
                m = mask[:, :Lx]
                pos = LOWER.to(device)[b_eval[:, :Lx]]
                src = pos if kind == "clamp" else BOUNDARY.to(device)[b_eval[:, :Lx]]
                val = (dt * src.unsqueeze(-1)).sum(1) / src.sum(1, keepdim=True).clamp(min=1)
                mf = m.unsqueeze(-1).float()
                return dt * (1 - mf) + val.unsqueeze(1) * mf
            return p
        model.layers[li]._compute_dt = mk(li)

    def letter_mask():
        bp = BOUNDARY.to(device)[b_eval]
        k = int(bp.float().sum())
        g = torch.Generator(device="cpu").manual_seed(7)
        sc = torch.rand(b_eval.shape, generator=g).to(device)
        sc[~LOWER.to(device)[b_eval]] = -1.0
        th = sc.flatten().kthvalue(sc.numel() - k + 1).values
        return sc >= th

    @torch.no_grad()
    def ev(cfg):
        for li, layer in enumerate(model.layers):
            layer._iv = cfg
        logits = model(b_eval)
        ce = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1), reduction="none").view(B, L)
        wi = (b_eval == 32)
        return round(ce.mean().item(), 4), round(ce[wi].mean().item(), 4)

    R["causal"] = {
        "baseline": ev(None),
        "clamp_boundary": ev(("clamp", BOUNDARY.to(device)[b_eval])),
        "clamp_letter": ev(("clamp", letter_mask())),
        "inject_tick": ev(("inject", letter_mask())),
    }
    for li, layer in enumerate(model.layers):
        layer._iv = None

    # E4-lite residual-B context
    gid = word_gids(probe)
    R["resid_B_R2"] = {}
    for li, layer in enumerate(model.layers):
        H = hiddens[li]
        Rb = layer.res_B(H.cuda()).detach().cpu().numpy().reshape(len(probe), -1)
        R["resid_B_R2"][f"L{li}"] = {
            "byte_identity": round(r2_of(Rb, [probe == bb for bb in np.unique(probe)]), 3),
            "common_bigrams": round(r2_of(Rb, [(np.roll(probe, 1) == c1) & (probe == c2)
                                               for c1, c2 in [(ord('t'), ord('h')), (ord('h'), ord('e')),
                                                              (ord('t'), ord('e')), (ord('a'), ord('n')),
                                                              (ord('i'), ord('n')), (ord('o'), ord('f')),
                                                              (ord(' '), ord('t')), (ord('e'), ord(' '))]]), 3),
            "word_identity_top500": round(r2_of(Rb, [gid == k for k in range(500)]), 3),
            "prev_word_identity_top500": round(r2_of(Rb, [np.roll(gid, 1) == k for k in range(500)]), 3),
        }

    # frien->d
    ctx = b"User: Hello! How are you today, my dear frien"
    idx = torch.tensor(list(ctx), dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        x = model.tok_emb(idx)
        hs = []
        for norm, layer in zip(model.prenorms, model.layers):
            h = norm(x)
            hs.append(h)
            x = x + layer(h, idx)
        layer = model.layers[-1]
        h_in = hs[-1]
        sigma = torch.exp(torch.clamp(layer.log_sigma, layer.log_sigma_min, layer.log_sigma_max))
        rho = layer._rho()
        lam = torch.complex(-sigma * rho, layer.omega * rho)
        dt = layer._compute_dt(h_in)
        Bn = layer._compute_B(h_in, idx)
        Cn = layer._compute_C(h_in, idx)
        hst = torch.zeros(1, layer.in_dim, layer.n_states, dtype=torch.complex64, device=device)
        for n in range(h_in.shape[1]):
            a_n = torch.exp(lam * dt[:, n].unsqueeze(-1))
            hst = a_n * hst + (Bn[:, n] * dt[:, n].unsqueeze(-1) * h_in[:, n].unsqueeze(-1)).to(torch.complex64)
        C_last = Cn[0, -1]
        g_re, g_im = layer.a, layer.b
        read_re, read_im = C_last * hst[0].real, C_last * hst[0].imag
        contrib = torch.einsum("ik,iok->iok", read_re, g_re) - torch.einsum("ik,iok->iok", read_im, g_im)
        gate = F.silu(layer.W_z(h_in[:, -1]))[0]
        head_d = model.head.weight[ord("d")]
        pair = (contrib * gate.unsqueeze(0).unsqueeze(-1) * head_d.unsqueeze(0).unsqueeze(-1)).sum(1)
        logit_modes = pair.sum(0)
        logits = model(idx)[0, -1]
        t2 = torch.topk(logits, 2)
        top5 = torch.topk(pair.abs().flatten(), 5)
        R["frien_d"] = {
            "prediction": {"top1": chr(int(t2.indices[0])), "top2": chr(int(t2.indices[1])),
                           "margin": round(float(t2.values[0] - t2.values[1]), 3)},
            "mode_contrib": [round(float(v), 3) for v in logit_modes.tolist()],
            "top_ik": {f"i{int(i)},k{int(k)}": round(float(pair[i, k]), 2)
                       for i, k in [np.divmod(t.item(), layer.n_states) for t in top5.indices]},
        }

    del model
    torch.cuda.empty_cache()
    return R


S = {}
for seed in (42, 123, 2024):
    print(f"=== seed {seed} ===", flush=True)
    S[f"s{seed}"] = analyze(seed)
    r = S[f"s{seed}"]
    print("  dt ratio space/lower:", r["dt_ratio_space_over_lower"])
    print("  causal:", r["causal"])
    print("  E1:", r["E1_nospace_ratio"], " mem:", r["memory_L0"]["half_life_med"], r["memory_L1"]["half_life_med"])
    print("  resid L1:", r["resid_B_R2"]["L1"])
    print("  frien:", r["frien_d"]["prediction"], r["frien_d"]["mode_contrib"])

(OUT / "v7e_10m_analysis.json").write_text(json.dumps(S, indent=2))
print("saved", OUT / "v7e_10m_analysis.json")
