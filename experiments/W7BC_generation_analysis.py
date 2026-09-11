"""How does it speak? Generation-side interpretation of wskan7bc.

Traces one autoregressive generation byte-by-byte with full instrumentation:
  - per-position clock (Delta per layer) during production
  - exact per-path margin decomposition (embedding / L0-2 base / L0-2 wave)
  - exact per-(i,k) mode attribution at L2 wavelet path
  - aggregated: what drives word-internal vs boundary emission

Output: figures/fig_w7bc_generation.png + v7e_generation_summary.json
Run: .venv/bin/python experiments/W7BC_generation_analysis.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.V7_WSKAN import WaveletStateKANLMV7

OUT = Path(__file__).resolve().parent / "figures"
device = "cuda"

model = WaveletStateKANLMV7(vocab_size=256, d_model=40, n_layers=2,
                            use_feature_bc=True, wz_diag=False, g_rank=None, bc_rank=32)
model.load_state_dict(torch.load("checkpoints/wskan7bc_ultrachat_100k_s42/latest.pt", weights_only=False)["state_dict"])
model.eval().cuda()

PROMPT = b"User: Hello! Can you tell me a story?\nAssistant:"
GEN_N = 160

# ---------- 1. generate (greedy-ish) ----------
gen = model.generate(PROMPT, max_new=GEN_N, temperature=0.3)
full = (PROMPT + b"")  # prompt itself
text = gen
print("generated:", text.decode(errors="replace"))

idx = torch.tensor(list(text), dtype=torch.long, device=device).unsqueeze(0)
L = idx.shape[1]

# ---------- 2. teacher-forced pass with captures ----------
hs = []
dts = []
x = model.tok_emb(idx)
with torch.no_grad():
    for norm, layer in zip(model.prenorms, model.layers):
        h = norm(x)
        hs.append(h)
        dts.append(layer._compute_dt(h)[0].cpu())  # (L, d)
        x = x + layer(h, idx)

# logits/margins per position (prediction at position n = byte n+1)
with torch.no_grad():
    logits = model(idx)[0].cpu().numpy()  # (L, 256)
top2 = np.argsort(-logits, axis=1)[:, :2]
margin = np.array([logits[n, top2[n, 0]] - logits[n, top2[n, 1]] for n in range(L)])

# per-path exact margin contributions
W_head = model.head.weight.detach().cpu().numpy()  # (256, d)
path_vecs = {}   # name -> (L, d) component vectors at each position (hidden after all layers)
with torch.no_grad():
    x = model.tok_emb(idx)
    emb0 = x.clone()
    comp = {}
    for li, (norm, layer) in enumerate(zip(model.prenorms, model.layers)):
        h = norm(x)
        base = h @ layer.w_base                    # (L, d)
        full = layer(h, idx)                       # (L, d)
        wav = full - base
        comp[f"L{li}-base"] = base[0].cpu().numpy()
        comp[f"L{li}-wave"] = wav[0].cpu().numpy()
        x = x + full
    final_h = model.norm(x)[0].cpu().numpy()
    # exact decomposition of final_h: embedding + sum of layer deltas
    embv = model.tok_emb(idx)[0].cpu().numpy()
    final_raw = x[0].cpu().numpy()
    # norm is affine per channel; handle by un-scaling: keep final_raw, apply norm linearly
    gn = model.norm.weight.cpu().numpy()
    gmean = model.norm.bias.cpu().numpy()
    mean = final_raw.mean(-1, keepdims=True)
    var = np.var(final_raw, axis=-1)
    std = np.sqrt(var + 1e-5)
    # norm(h) = (h - mean)/std * gn + gmean  -> per component c: contribution = ((c - 0)/std * gn)
    def normed(c):
        return (c / std[:, None]) * gn
    paths = {"token-emb": normed(embv - 0)}
    NL = len(model.layers)
    for li in range(NL):
        paths[f"L{li}-base"] = normed(comp[f"L{li}-base"])
        paths[f"L{li}-wave"] = normed(comp[f"L{li}-wave"])
    # bias term constant -> excluded from margin (cancels in top1-top2)

margins_path = {}
for name, v in paths.items():  # v: (L, d)
    margins_path[name] = np.array([W_head[top2[n, 0]] @ v[n] - W_head[top2[n, 1]] @ v[n]
                                   for n in range(L)])
# ---------- 3. L2 scan replay for per-position hidden state ----------
layer = model.layers[-1]
h_in = hs[-1]
with torch.no_grad():
    sigma = torch.exp(torch.clamp(layer.log_sigma, layer.log_sigma_min, layer.log_sigma_max))
    rho = layer._rho()
    lam = torch.complex(-sigma * rho, layer.omega * rho)
    dt = layer._compute_dt(h_in)
    Bn = layer._compute_B(h_in, idx)
    Cn = layer._compute_C(h_in, idx)
    hst = torch.zeros(1, layer.in_dim, layer.n_states, dtype=torch.complex64, device=device)
    gate = F.silu(layer.W_z(h_in))[0]            # (L, o)
    headw = model.head.weight                    # (256, o)
    margin_mode = np.zeros((L, layer.n_states))
    for n in range(L):
        a_n = torch.exp(lam * dt[:, n].unsqueeze(-1))
        hst = a_n * hst + (Bn[:, n] * dt[:, n].unsqueeze(-1) * h_in[:, n].unsqueeze(-1)).to(torch.complex64)
        read_re = Cn[0, n] * hst[0].real
        read_im = Cn[0, n] * hst[0].imag
        y_ok = torch.einsum("ik,iok->iok", read_re, layer.a) - torch.einsum("ik,iok->iok", read_im, layer.b)
        per_mode_o = (y_ok.sum(0) * gate[n].unsqueeze(-1))     # (o, N)
        # margin contribution of mode k: head[t1] @ (per_mode_o[:,k]) - head[t2] @ (...)
        t1, t2 = int(top2[n, 0]), int(top2[n, 1])
        pm = per_mode_o  # (o, N)
        # apply final LayerNorm linearly: (c/std)*gn on the (o) dim... std of final_raw at position n
        pm_np = pm.cpu().numpy()
        std_n = std[n]
        gn_np = gn
        pm_normed = (pm_np / std_n) * gn_np[:, None]  # (o, N)
        margin_mode[n] = (W_head[t1] @ pm_normed) - (W_head[t2] @ pm_normed)

# ---------- 4. aggregate stats ----------
def bclass(c):
    if c == 32: return "space"
    if c == 10: return "newline"
    if (97 <= c <= 122) or (65 <= c <= 90): return "letter"
    return "punct"

n0 = len(PROMPT)  # only measure produced positions
classes = [bclass(c) for c in text[n0:]]
dt_l = np.stack([dts[l][n0:, :].mean(-1).numpy() for l in range(len(model.layers))])  # (3, T)
cls_arr = np.array(classes)
S = {"generated": text.decode(errors="replace"),
     "dt_by_class_per_layer": {c: [float(dt_l[l][cls_arr == c].mean()) if (cls_arr == c).any() else None
                                   for l in range(dt_l.shape[0])] for c in ("letter", "space", "punct", "newline")},
     "margin_by_class": {c: float(margin[n0:][cls_arr == c].mean()) for c in ("letter", "space", "punct", "newline")}}

path_arr = {k: v[n0:] for k, v in margins_path.items()}
S["path_margin_share_by_class"] = {}
for c in ("letter", "space"):
    m = cls_arr == c
    tot = {k: float(np.mean(v[m])) for k, v in path_arr.items()}
    S["path_margin_share_by_class"][c] = tot

# mode attribution stats at boundary emissions (predicting space/newline)
emit_bnd = np.array([text[n + 1] in (32, 10) for n in range(n0, L - 1)])
mm = margin_mode[n0:L - 1]
S["mode_at_boundary_emission"] = {f"k{k}": float(np.mean(np.abs(mm[emit_bnd[:, 0] if False else None]))) for k in range(6)} \
    if emit_bnd.sum() > 0 else {}
if emit_bnd.sum() > 0:
    S["mode_at_boundary_emission"] = {f"k{k}": float(np.mean(np.abs(mm[emit_bnd][:, k]))) for k in range(6)}
    S["mode_at_letter_emission"] = {f"k{k}": float(np.mean(np.abs(mm[~emit_bnd][:, k]))) for k in range(6)}

# ---------- 5. figure ----------
T = L - n0
fig, axes = plt.subplots(4, 1, figsize=(16, 11), gridspec_kw={"height_ratios": [1, 1.4, 1.4, 1.6]})

ax = axes[0]
colmap = {"letter": "#69f", "space": "#222", "punct": "#e83", "newline": "#2a2"}
for i, c in enumerate(classes):
    ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=colmap[c]))
ax.plot(range(T), margin[n0:], "w-", lw=0.8)
ax.set_ylim(-0.5, max(1.5, margin[n0:].max()))
ax.set_xlim(0, T)
ax.set_ylabel("margin")
ax.set_title("produced bytes (blue=letter, black=space, orange=punct, green=newline) + decision margin (white)")
# annotate bytes
for i in range(0, T, 8):
    ax.text(i, -0.4, repr(chr(text[n0 + i])), fontsize=6, rotation=0)

ax = axes[1]
ax.imshow(dt_l, aspect="auto", cmap="magma")
ax.set_yticks(range(len(model.layers)), [f"L{i}" for i in range(len(model.layers))])
ax.set_ylabel("clock Delta"); ax.set_title("the clock during production (mean per layer)")
ax.set_xlim(0, T)

ax = axes[2]
names = ["token-emb"] + [f"L{li}-{t}" for li in range(len(model.layers)) for t in ("base", "wave")]
M = np.stack([path_arr[k] for k in names])
ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-np.abs(M).max(), vmax=np.abs(M).max())
ax.set_yticks(range(len(names)), names, fontsize=7)
ax.set_title("exact path contributions to decision margin (per position)")
ax.set_xlim(0, T)

ax = axes[3]
ax.imshow(mm.T, aspect="auto", cmap="RdBu_r", vmin=-np.abs(mm).max(), vmax=np.abs(mm).max())
ax.set_yticks(range(6), [f"k{k}" for k in range(6)])
ax.set_title("L2 wavelet path: per-mode margin contribution (mode i x time)")
ax.set_xlim(0, T)

fig.tight_layout()
fig.savefig(OUT / "fig_w7bc_generation.png", dpi=120)
plt.close(fig)

(OUT / "v7e_generation_summary.json").write_text(json.dumps(S, indent=2))
print(json.dumps(S, indent=2)[:2200])
print("figure:", OUT / "fig_w7bc_generation.png")
