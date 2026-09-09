"""Deep interpretability battery on the adopted wskan7bc checkpoint.

Covers the four remaining interpretability debts:
  A. M_B/M_C named-table readout  (fig_v7bc_tables.png)
  B. Edge atlas: kernel clustering + effective rank (fig_v7bc_atlas.png)
  C. Channel dictionary: byte-class x position-in-word profiles (fig_v7bc_channels.png)
  D. Clock drivers: which channels drive Delta (fig_v7bc_clock.png)

Run: .venv/bin/python experiments/V7_deep_interpret.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.V1_train_tinystories_lm import load_data
from models.V7_WSKAN import WaveletStateKANLMV7, byte_features

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
CKPT = Path(__file__).resolve().parent.parent / "checkpoints/wskan7bc_ultrachat_100k_s42/latest.pt"
FEAT_NAMES = ["space", "newline", "punct", "upper", "lower", "digit", "vowel", "const"]

model = WaveletStateKANLMV7(use_feature_bc=True, wz_diag=False, g_rank=None)
model.load_state_dict(torch.load(CKPT, weights_only=False)["state_dict"])
model.eval().cuda()
S = {}

train_ids, _ = load_data(200000, 500, 42, "ultrachat")
text = train_ids[:2048].numpy().astype(np.int64)
idx_probe = torch.tensor(text, device="cuda").unsqueeze(0)

# hidden states + dt capture
hiddens, dts = [], []
x = model.tok_emb(idx_probe)
with torch.no_grad():
    for norm, layer in zip(model.prenorms, model.layers):
        h = norm(x)
        hiddens.append(h[0].cpu())          # (L, d)
        dts.append(layer._compute_dt(h)[0].cpu())  # (L, d)
        x = x + layer(h, idx_probe)

# ==================== A. M_B / M_C table readout ====================
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for li, layer in enumerate(model.layers):
    MB = layer.M_B.detach().cpu()   # (F, i, N)
    MC = layer.M_C.detach().cpu()
    axes[0][li].imshow(MB.abs().mean(dim=1).numpy(), aspect="auto", cmap="viridis")
    axes[0][li].set_yticks(range(8), FEAT_NAMES, fontsize=7)
    axes[0][li].set_xlabel("mode k"); axes[0][li].set_title(f"L{li} |M_B| write")
    axes[1][li].imshow(MC.abs().mean(dim=1).numpy(), aspect="auto", cmap="viridis")
    axes[1][li].set_yticks(range(8), FEAT_NAMES, fontsize=7)
    axes[1][li].set_xlabel("mode k"); axes[1][li].set_title(f"L{li} |M_C| read")
fig.suptitle("Named gate tables: |M| averaged over channels (feature x mode)")
fig.tight_layout()
fig.savefig(OUT / "fig_v7bc_tables.png", dpi=130)
plt.close(fig)

# does 'space' write preferentially to long-memory modes?
for li, layer in enumerate(model.layers):
    rho = torch.exp(layer.log_rho.detach().clamp(-3, 3)).cpu()
    sigma = torch.exp(layer.log_sigma.detach().clamp(-6, 8)).cpu()
    mean_dt = dts[li].mean(0)                       # (i,)
    hl = 0.693 / (rho * sigma * mean_dt.unsqueeze(-1))  # (i, N)
    hl_mean = hl.mean(0)                            # (N,)
    w_space = layer.M_B.detach().cpu()[0].abs()   # (i, N) feature=space
    corr = np.corrcoef(np.log(hl.flatten().numpy()), np.log(w_space.flatten().numpy() + 1e-12))[0, 1]
    S[f"L{li}_space_write_vs_halflife_corr"] = float(corr)
    S[f"L{li}_halflife_by_mode"] = [float(v) for v in hl_mean]

# ==================== B. edge atlas ====================
from scipy.cluster.vq import kmeans2

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
lags = np.arange(48)
S["atlas"] = {}
for li, layer in enumerate(model.layers):
    rho = torch.exp(layer.log_rho.detach().clamp(-3, 3)).cpu()
    sigma = torch.exp(layer.log_sigma.detach().clamp(-6, 8)).cpu()
    om = layer.omega.detach().cpu()
    a, b = layer.a.detach().cpu(), layer.b.detach().cpu()   # (i, o, N)
    mean_dt = dts[li].mean(0)
    I, O, N = a.shape
    # kernels per edge (i,o): (I*O, L)
    lg = torch.tensor(lags, dtype=torch.float32)
    env = torch.exp(-(sigma * rho).unsqueeze(-1) * lg * mean_dt[:, None, None])   # (i, N, L)
    ph = (om * rho).unsqueeze(-1) * lg * mean_dt[:, None, None]
    # K_{i,o}(l) = sum_k a*cos + b*sin weighted
    K = torch.einsum("ion,inl->iol", a, env * torch.cos(ph)) + \
        torch.einsum("ion,inl->iol", b, env * torch.sin(ph))     # (i, o, L)
    Kf = K.reshape(-1, len(lags)).numpy()
    Kn = Kf / (np.abs(Kf).max(axis=1, keepdims=True) + 1e-9)
    centroid, label = kmeans2(Kn, 6, minit="++", seed=0)
    counts = np.bincount(label, minlength=6)
    order = np.argsort(-counts)
    for r, ci in enumerate(order):
        sel = Kf[label == ci]
        mk = sel[np.argmax(np.abs(sel).max(axis=1))]
        axes[li].plot(lags, mk / np.abs(mk).max() + (5 - r) * 2.2, lw=0.9)
    axes[li].set_title(f"L{li}: cluster sizes {counts[order].tolist()}")
    axes[li].set_xlabel("lag (tokens)")
    # effective rank of g per mode via SVD
    er = []
    for k in range(N):
        Mk = torch.cat([a[:, :, k], b[:, :, k]], dim=1).numpy()  # (i, 2o)
        s = np.linalg.svd(Mk, compute_uv=False)
        p = s / s.sum(); er.append(float(np.exp(-(p * np.log(p + 1e-12)).sum())))
    S["atlas"][f"L{li}_effective_rank_per_mode"] = [round(e, 1) for e in er]
fig.suptitle("Edge atlas: representative kernel per cluster (offset vertically; top=largest cluster)")
fig.tight_layout()
fig.savefig(OUT / "fig_v7bc_atlas.png", dpi=130)
plt.close(fig)

# ==================== C. channel dictionary ====================
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
S["channels"] = {}
classes = [("space", lambda t: t == 32), ("upper", lambda t: (t >= 65) & (t <= 90)),
           ("lower", lambda t: (t >= 97) & (t <= 122)), ("digit", lambda t: (t >= 48) & (t <= 57)),
           ("punct", lambda t: np.isin(t, list(b".,!?;:'\"()-")))]
pos_word = np.zeros_like(text)
pw = 99
for n, t in enumerate(text):
    pw = 0 if t == 32 else pw + 1
    pos_word[n] = min(pw, 6)
for li in range(3):
    H = hiddens[li].numpy()          # (L, d)
    prof = []
    for name, pred in classes:
        prof.append(H[pred(text)].mean(0))
    for pw_bucket in range(7):
        prof.append(H[pos_word == pw_bucket].mean(0))
    P = np.stack(prof)               # (12, d)
    Pn = (P - P.mean(1, keepdims=True)) / (P.std(1, keepdims=True) + 1e-9)
    centroid, label = kmeans2(Pn.T, 5, minit="++", seed=0)
    counts = np.bincount(label, minlength=5)
    im = axes[li].imshow(centroid, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    axes[li].set_yticks(range(5), [f"c{j} (n={counts[j]})" for j in range(5)], fontsize=7)
    axes[li].set_xticks(range(12), [c[0] for c in classes] + [f"pw{j}" for j in range(7)], rotation=60, fontsize=6)
    axes[li].set_title(f"layer {li} channel clusters")
    fig.colorbar(im, ax=axes[li])
fig.suptitle("Channel dictionary: cluster-mean z-scored profiles over byte class (sp/up/lo/di/pu) and position-in-word (0..6)")
fig.tight_layout()
fig.savefig(OUT / "fig_v7bc_channels.png", dpi=130)
plt.close(fig)

# ==================== D. clock drivers ====================
fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
S["clock_drivers"] = {}
for li, layer in enumerate(model.layers):
    W = layer.W_dt.weight.detach().cpu().abs().numpy()   # (d_out, d_in)
    hmag = hiddens[li].abs().mean(0).numpy()             # (d,)
    driver = (W * hmag[None, :]).sum(0)
    driver = driver / driver.sum()
    top = np.argsort(-driver)[:8]
    axes[li].bar(range(len(driver)), driver)
    axes[li].set_title(f"L{li}: top drivers {top.tolist()}")
    axes[li].set_xlabel("input channel")
    S["clock_drivers"][f"L{li}_top8"] = top.tolist()
    S["clock_drivers"][f"L{li}_top8_share"] = float(driver[top].sum())
fig.suptitle("Which residual channels drive the clock (|W_dt| weighted by activation)")
fig.tight_layout()
fig.savefig(OUT / "fig_v7bc_clock.png", dpi=130)
plt.close(fig)

(OUT / "v7bc_deep_summary.json").write_text(json.dumps(S, indent=2))
print(json.dumps(S, indent=2)[:2500])
print("figures in", OUT)
