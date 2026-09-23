"""All-sizes interpretability battery on the 3-epoch campaign checkpoints.

Per (size, seed) on UltraChat wskan11: word clock, memory profile, causal
battery (boundary/letter/random clamps + injection), residual-B variance
decomposition, mode frequencies (median + readout-weighted), Q, effective
rank, frien->d attribution. Plus a generation sample per size.

Output: figures/w11_all_sizes.json
Run: .venv/bin/python experiments/W11_all_sizes_analysis.py
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
from models.V11_WSKAN import WaveletStateKANLMV11

device = "cuda"
torch.manual_seed(0)
OUT = Path(__file__).resolve().parent / "figures"
SIZE_CFG = {"1k": (4, 1), "10k": (12, 1), "100k": (40, 2), "1m": (80, 6), "10m": (512, 2)}
SEEDS = [42, 123, 2024, 7, 31337]
SIZES = ["1k", "10k", "100k", "1m", "10m"]

CLASSES = {"space": lambda t: t == 32, "newline": lambda t: t == 10,
           "punct": lambda t: np.isin(t, list(b".,!?;:'\"()-")),
           "upper": lambda t: (t >= 65) & (t <= 90),
           "lower": lambda t: (t >= 97) & (t <= 122), "digit": lambda t: (t >= 48) & (t <= 57)}
BOUNDARY = torch.zeros(256, dtype=torch.bool)
for c in b" \n.,!?;:'\"()-":
    BOUNDARY[c] = True
LOWER = torch.tensor([(97 <= c <= 122) for c in range(256)])

train_ids, eval_ids = load_data(1400000, 500, 42, "ultrachat")
probe = train_ids[:2048].numpy().astype(np.int64)
B, L = 32, 512
batch = torch.cat([get_batch(eval_ids, 1, L, device) for _ in range(B)], 0)
b_eval, targets = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()


def load_model(size, seed):
    d, L = SIZE_CFG[size]
    m = WaveletStateKANLMV11(d_model=d, n_layers=L, n_states=6, use_feature_bc=True,
                             wz_diag=False, g_rank=None, bc_rank=min(32, d), bf16_scan=True)
    m.load_state_dict(torch.load(f"checkpoints/wskan11_ultrachat_{size}_3ep_s{seed}/latest.pt",
                                 weights_only=False)["state_dict"])
    return m.eval().cuda()


def r2_of(R, groups):
    groups = [g for g in groups if g.sum() > 0]
    gidx = np.full(len(R), -1, dtype=int)
    for gi, g in enumerate(groups): gidx[g] = gi
    keep = gidx >= 0
    gm = np.stack([R[g].mean(0) for g in groups])
    Rk = R[keep]
    return float(np.mean(1 - ((Rk - gm[gidx[keep]]) ** 2).sum(0) / ((Rk - Rk.mean(0)) ** 2).sum(0)))


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


def analyze(size, seed):
    model = load_model(size, seed)
    R = {"size": size, "seed": seed}
    idx_probe = torch.tensor(probe, device=device).unsqueeze(0)
    hiddens, dts = [], []
    with torch.no_grad():
        x = model.tok_emb(idx_probe)
        for norm, layer in zip(model.prenorms, model.layers):
            h = norm(x)
            hiddens.append(h[0].cpu())
            dts.append(layer._compute_dt(h)[0].cpu())
            x = x + layer(h, idx_probe)
    NL = len(model.layers)

    # word clock
    R["clock"] = {}
    for cname, pred in CLASSES.items():
        m = pred(probe)
        R["clock"][cname] = [round(float(dts[l][m].mean()), 4) for l in range(NL)]
    R["clock_ratio"] = [round(R["clock"]["space"][l] / R["clock"]["lower"][l], 3) for l in range(NL)]

    # memory
    R["memory"] = []
    for li, layer in enumerate(model.layers):
        sigma = torch.exp(layer.log_sigma.detach().clamp(-6, 8)).cpu()
        rho = torch.exp(layer.log_rho.detach().clamp(-3, 3)).cpu()
        sig_eff = (sigma * rho).flatten().numpy()
        om_eff = (layer.omega.detach().cpu() * rho).flatten().numpy()
        hl = 0.693 / (sig_eff * np.repeat(dts[li].mean(0).numpy(), layer.n_states))
        R["memory"].append({
            "sigma_med": round(float(np.median(sig_eff)), 2),
            "Q_med": round(float(np.median(om_eff / (2 * sig_eff))), 2),
            "ceil_frac": round(float((layer.log_sigma.detach() > 1.95).float().mean()), 3),
            "hl_med": round(float(np.median(hl)), 2), "hl_p99": round(float(np.quantile(hl, .99)), 1),
            "hl_max": round(float(hl.max()), 1),
            "cyc_tok_med": round(float(np.median(om_eff * np.repeat(dts[li].mean(0).numpy(), layer.n_states) / (2 * np.pi))), 3),
        })

    # effective rank
    er = []
    layer0 = model.layers[0]
    a, b_ = layer0.a.detach().cpu().numpy(), layer0.b.detach().cpu().numpy()
    for k in range(layer0.n_states):
        Mk = np.concatenate([a[:, :, k], b_[:, :, k]], axis=1)
        sv = np.linalg.svd(Mk, compute_uv=False); pv = sv / sv.sum()
        er.append(float(np.exp(-(pv * np.log(pv + 1e-12)).sum())))
    R["g_eff_rank_L0"] = [round(v, 1) for v in er]

    # residual-B variance decomposition (last layer)
    H = hiddens[-1]
    layer = model.layers[-1]
    Rb = layer.res_B(H.cuda()).detach().cpu().numpy().reshape(len(probe), -1)
    gid = word_gids(probe)
    bigrams = [(ord('t'), ord('h')), (ord('h'), ord('e')), (ord('t'), ord('e')), (ord('a'), ord('n')),
               (ord('i'), ord('n')), (ord('o'), ord('f')), (ord(' '), ord('t')), (ord('e'), ord(' '))]
    R["resid_B_R2_last"] = {
        "byte_identity": round(r2_of(Rb, [probe == bb for bb in np.unique(probe)]), 3),
        "common_bigrams": round(r2_of(Rb, [(np.roll(probe, 1) == c1) & (probe == c2) for c1, c2 in bigrams]), 3),
        "word_identity_top500": round(r2_of(Rb, [gid == k for k in range(500)]), 3),
    }

    # causal battery
    orig = {i: l._compute_dt for i, l in enumerate(model.layers)}
    for li, layer in enumerate(model.layers):
        def mk(li):
            def p(x):
                dt = orig[li](x)
                cfg = getattr(model.layers[li], "_iv", None)
                if cfg is None: return dt
                kind, mask = cfg
                Lx = x.shape[1]
                m = mask[:, :Lx]
                pos = LOWER.to(device)[b_eval[:, :Lx]]
                src = pos if kind == "clamp" else BOUNDARY.to(device)[b_eval[:, :Lx]]
                val = (dt * src.unsqueeze(-1)).sum(1) / src.sum(1, keepdim=True).clamp(min=1)
                return dt * (1 - m.unsqueeze(-1).float()) + val.unsqueeze(1) * m.unsqueeze(-1).float()
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
        for li, layer in enumerate(model.layers): layer._iv = cfg
        logits = model(b_eval)
        ce = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1), reduction="none").view(B, L)
        wi = (b_eval == 32)
        return round(ce.mean().item(), 4), round(ce[wi].mean().item(), 4)

    R["causal"] = {"baseline": ev(None),
                   "clamp_boundary": ev(("clamp", BOUNDARY.to(device)[b_eval])),
                   "clamp_letter": ev(("clamp", letter_mask())),
                   "inject_tick": ev(("inject", letter_mask()))}
    for li, layer in enumerate(model.layers): layer._iv = None

    # frien->d attribution
    ctx = b"User: Hello! How are you today, my dear frien"
    idx = torch.tensor(list(ctx), dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(idx)[0, -1]
        t2 = torch.topk(logits, 2)
        R["frien_d"] = {"top1": chr(int(t2.indices[0])), "top2": chr(int(t2.indices[1])),
                        "margin": round(float(t2.values[0] - t2.values[1]), 3),
                        "correct": chr(int(t2.indices[0])) == "d"}

    # generation
    R["sample"] = model.generate(b"User: Can you tell me a story?\nAssistant:",
                                 max_new=160, temperature=0.8).decode(errors="replace")
    del model
    torch.cuda.empty_cache()
    return R


S = {}
for size in SIZES:
    for seed in SEEDS:
        print(f"=== {size} s{seed} ===", flush=True)
        R = analyze(size, seed)
        S[f"{size}_s{seed}"] = R
        print(f"  clock ratio: {R['clock_ratio']} | hl med: {[m['hl_med'] for m in R['memory']]} | "
              f"causal: bnd {R['causal']['clamp_boundary'][0]:.3f} vs let {R['causal']['clamp_letter'][0]:.3f} (base {R['causal']['baseline'][0]:.3f}) | frien->d: {R['frien_d']['top1']}")

(OUT / "w11_all_sizes.json").write_text(json.dumps(S, indent=2))
print("saved", OUT / "w11_all_sizes.json")
