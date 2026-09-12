"""Canonical probe: re-measure the headline interpretability numbers on the
fresh interp checkpoint (wskan7bc d32L3, ultrachat, seed 42, clean code).

  A. word clock: Delta by byte class per layer
  B. causal battery: boundary/letter/random clamps + false-tick injection
  C. frien->d: exact path + per-mode attribution
  D. production trace stats (clock + path shares during generation)
  E. named-table strengths, rho ladder, effective rank

Output: figures/v7e_canonical_probe.json
Run: .venv/bin/python experiments/W7BC_canonical_probe.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.V1_train_tinystories_lm import get_batch, load_data
from models.V7_WSKAN import WaveletStateKANLMV7

device = "cuda"
torch.manual_seed(0)
CKPT = "checkpoints/wskan7bc_ultrachat_interp_s42/latest.pt"
OUT = Path(__file__).resolve().parent / "figures"
S = {}

model = WaveletStateKANLMV7(use_feature_bc=True, wz_diag=False, g_rank=None, bc_rank=32)
model.load_state_dict(torch.load(CKPT, weights_only=False)["state_dict"])
model.eval().cuda()

train_ids, eval_ids = load_data(200000, 500, 42, "ultrachat")
probe = train_ids[:2048].numpy().astype(np.int64)
idx_probe = torch.tensor(probe, device="cuda").unsqueeze(0)

# ---------------- A. Delta by byte class ----------------
hiddens, dts = [], []
with torch.no_grad():
    x = model.tok_emb(idx_probe)
    for norm, layer in zip(model.prenorms, model.layers):
        h = norm(x)
        hiddens.append(h)
        dts.append(layer._compute_dt(h)[0].cpu())
        x = x + layer(h, idx_probe)

CLASSES = {
    "space": lambda t: t == 32, "newline": lambda t: t == 10,
    "punct": lambda t: np.isin(t, list(b".,!?;:'\"()-")),
    "upper": lambda t: (t >= 65) & (t <= 90),
    "lower": lambda t: (t >= 97) & (t <= 122),
    "digit": lambda t: (t >= 48) & (t <= 57),
}
S["A_dt_by_class"] = {}
for cname, pred in CLASSES.items():
    m = pred(probe)
    S["A_dt_by_class"][cname] = {f"L{li}": float(dts[li][m].mean()) for li in range(3)}

# rho ladders + effective rank + named-table strengths
for li, layer in enumerate(model.layers):
    rho = torch.exp(layer.log_rho.detach().clamp(-3, 3)).cpu().numpy()
    S[f"E_rho_L{li}"] = [round(float(v), 3) for v in rho]
    MB = layer.M_B.detach().abs().cpu().numpy()
    S[f"E_MB_strength_L{li}"] = {cname: float(MB[fi].mean()) for fi, cname in enumerate(
        ["space", "newline", "punct", "upper", "lower", "digit", "vowel", "const"])}
    a = layer.a.detach().cpu().numpy(); b = layer.b.detach().cpu().numpy()
    er = []
    for k in range(a.shape[2]):
        Mk = np.concatenate([a[:, :, k], b[:, :, k]], axis=1)
        s = np.linalg.svd(Mk, compute_uv=False)
        p = s / s.sum()
        er.append(float(np.exp(-(p * np.log(p + 1e-12)).sum())))
    S[f"E_g_effective_rank_L{li}"] = [round(v, 1) for v in er]

# ---------------- B. causal battery ----------------
_, eval_ids = load_data(200000, 500, 42, "ultrachat")
B, L = 32, 512
batch = torch.cat([get_batch(eval_ids, 1, L, device) for _ in range(B)], 0)
b, targets = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()

BOUNDARY = torch.zeros(256, dtype=torch.bool)
for c in b" \n.,!?;:'\"()-":
    BOUNDARY[c] = True
LOWER = torch.tensor([(97 <= c <= 122) for c in range(256)])

orig = {i: layer._compute_dt for i, layer in enumerate(model.layers)}
def patch():
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
                pos = LOWER.to(device)[b[:, :Lx]]
                src = pos if kind == "clamp" else BOUNDARY.to(device)[b[:, :Lx]]
                val = (dt * src.unsqueeze(-1)).sum(1) / src.sum(1, keepdim=True).clamp(min=1)
                mf = m.unsqueeze(-1).float()
                return dt * (1 - mf) + val.unsqueeze(1) * mf
            return p
        model.layers[li]._compute_dt = mk(li)
patch()

def letter_matched_mask():
    bp = BOUNDARY.to(device)[b]
    k = int(bp.float().sum())
    g = torch.Generator(device="cpu").manual_seed(7)
    sc = torch.rand(b.shape, generator=g).to(device)
    sc[~LOWER.to(device)[b]] = -1.0
    th = sc.flatten().kthvalue(sc.numel() - k + 1).values
    return sc >= th

@torch.no_grad()
def evaluate(cfg):
    for i, layer in enumerate(model.layers):
        layer._iv = cfg
    logits = model(b)
    ce = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1), reduction="none").view(B, L)
    wi = (b == 32)
    return ce.mean().item(), ce[wi].mean().item()

res = {}
bnd = BOUNDARY.to(device)[b]
res["baseline"] = evaluate(None)
res["clamp_boundary"] = evaluate(("clamp", bnd))
res["clamp_letter"] = evaluate(("clamp", letter_matched_mask()))
res["inject_tick"] = evaluate(("inject", letter_matched_mask()))
for layer in model.layers:
    layer._iv = None
S["B_causal"] = res
print("causal:", res)

# ---------------- C. frien -> d ----------------
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
    top = torch.topk(logit_modes.abs().flatten(), 5)
    S["C_frien_d_modes"] = [round(float(v), 3) for v in logit_modes.tolist()]
    S["C_top_ik"] = {f"i{int(i)},k{int(k)}": round(float(pair[i, k]), 2)
                     for i, k in [np.divmod(t.item(), layer.n_states) for t in top.indices]}
    logits = model(idx)[0, -1]
    t2 = torch.topk(logits, 2)
    S["C_prediction"] = {"top1": chr(int(t2.indices[0])), "top2": chr(int(t2.indices[1])),
                         "margin": float(t2.values[0] - t2.values[1])}

# ---------------- D. production trace ----------------
PROMPT = b"User: Hello! Can you tell me a story?\nAssistant:"
gen = model.generate(PROMPT, max_new=120, temperature=0.3)
n0 = len(PROMPT)
idxg = torch.tensor(list(gen), device=device).unsqueeze(0)
dtx = []
with torch.no_grad():
    x = model.tok_emb(idxg)
    for norm, layer in zip(model.prenorms, model.layers):
        h = norm(x)
        dtx.append(layer._compute_dt(h)[0, n0:].cpu())
        x = x + layer(h, idxg)
prod_cls = {"letter": lambda c: (97 <= c <= 122) or (65 <= c <= 90),
            "space": lambda c: c == 32, "punct": lambda c: c in b".,!?;:'\"()-"}
S["D_generated"] = gen.decode(errors="replace")
S["D_dt_by_class"] = {}
for cname, pred in prod_cls.items():
    m = np.array([pred(c) for c in gen[n0:]])
    if m.sum():
        S["D_dt_by_class"][cname] = {f"L{li}": float(dtx[li][m].mean()) for li in range(3)}

(OUT / "v7e_canonical_probe.json").write_text(json.dumps(S, indent=2))
print(json.dumps(S, indent=2)[:2600])
