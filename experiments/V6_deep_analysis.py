"""V6 deep mathematical analysis: how did the network learn language?

Produces (experiments/figures/):
  fig_v6_q_structure.png    - constant-Q test: log sigma_tilde vs log omega_tilde
  fig_v6_spectrum.png       - learned mode frequencies vs English byte-stream spectrum
  fig_v6_byteclass.png      - dt / |B| / |C| gating by byte class
  fig_v6_kernels.png        - top-edge effective kernels
  v6_analysis_summary.json  - all numbers

Run: .venv/bin/python experiments/V6_deep_analysis.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.V6_WSKAN import WaveletStateKANLMV6

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
CKPT = Path(__file__).resolve().parent.parent / "checkpoints/wskan6_ultrachat_100k_s42/latest.pt"

model = WaveletStateKANLMV6()
model.load_state_dict(torch.load(CKPT, weights_only=False)["state_dict"])
model.eval().cuda()
S = {}

# ---------- text for probing / spectrum ----------
from experiments.V1_train_tinystories_lm import load_data

train_ids, _ = load_data(200000, 500, 42, "ultrachat")
text = train_ids[:65536].numpy()

# ============================== 1. constant-Q ==============================
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for li, layer in enumerate(model.layers):
    sigma = torch.exp(layer.log_sigma.detach().clamp(-6, 8)).cuda()
    omega = layer.omega.detach().cuda()
    rho = torch.exp(layer.log_rho.detach().clamp(-3, 3)).cuda()
    sig_t = (sigma * rho).flatten()  # effective decay
    om_t = (omega * rho).flatten()   # effective frequency
    lw = np.log(om_t.cpu().numpy()); ls = np.log(sig_t.cpu().numpy())
    A = np.vstack([lw, np.ones_like(lw)]).T
    slope, intercept = np.linalg.lstsq(A, ls, rcond=None)[0]
    pred = slope * lw + intercept
    r2 = 1 - ((ls - pred) ** 2).sum() / ((ls - ls.mean()) ** 2).sum()
    Q = (om_t / (2 * sig_t)).cpu().numpy()
    ax = axes[li]
    ax.scatter(lw, ls, s=6, alpha=0.5)
    xs = np.linspace(lw.min(), lw.max(), 10)
    ax.plot(xs, slope * xs + intercept, "r--", label=f"slope={slope:.2f} R2={r2:.3f}")
    ax.set_xlabel("log omega_eff"); ax.set_ylabel("log sigma_eff")
    ax.set_title(f"layer {li}  (Q med {np.median(Q):.2f})")
    ax.legend(fontsize=8)
    S[f"layer{li}_q_fit"] = {"slope": float(slope), "intercept": float(intercept), "r2": float(r2)}
    S[f"layer{li}_Q"] = {"median": float(np.median(Q)), "p10": float(np.quantile(Q, .1)), "p90": float(np.quantile(Q, .9))}
fig.suptitle("Constant-Q test: effective (omega, sigma) of learned modes (log-log)")
fig.tight_layout()
fig.savefig(OUT / "fig_v6_q_structure.png", dpi=130)
plt.close(fig)

# =================== 2. learned frequencies vs text spectrum ===================
# per-token effective frequency of each mode on real text (uses mean dt per channel)
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
idx_probe = torch.tensor(text[:2048], dtype=torch.long, device="cuda").unsqueeze(0)
x = model.tok_emb(idx_probe)
freqs_per_layer = []
with torch.no_grad():
    for li, (norm, layer) in enumerate(zip(model.prenorms, model.layers)):
        h = norm(x)
        dt = layer._compute_dt(h)                       # (B,L,i)
        mean_dt = dt.mean(dim=(0, 1))                   # (i,)
        rho = torch.exp(layer.log_rho.detach().clamp(-3, 3))
        om_t = (layer.omega.detach().cuda() * rho)      # (i,N) rad per warped-unit
        cyc_tok = (om_t * mean_dt.unsqueeze(-1) / (2 * np.pi)).flatten().cpu().numpy()
        cyc_tok = cyc_tok[cyc_tok > 0]
        freqs_per_layer.append(cyc_tok)
        x = x + layer(h)
all_freqs = np.concatenate(freqs_per_layer)
axes[0].hist(all_freqs, bins=60, density=True, alpha=0.7, label="learned modes")

# English byte-stream periodograms: byte value + space indicator
seg = text.astype(np.float64); seg -= seg.mean()
spec = np.abs(np.fft.rfft(seg)) ** 2
fr = np.fft.rfftfreq(len(seg), d=1.0)  # cycles per byte
m = fr < 1.0
axes[1].semilogy(fr[m], spec[m] / spec[m].sum(), lw=0.8, label="byte-value spectrum")
space_ind = (text == 32).astype(np.float64); space_ind -= space_ind.mean()
spec2 = np.abs(np.fft.rfft(space_ind)) ** 2
axes[1].semilogy(fr[m], spec2[m] / spec2[m].sum(), lw=0.8, alpha=0.7, label="space-indicator spectrum")
axes[1].set_xlabel("cycles per byte"); axes[1].set_ylabel("normalized power")
axes[1].legend(fontsize=8); axes[1].set_title("English text periodicity (training sample)")
axes[0].set_xlabel("cycles per token (omega_eff * dt / 2pi)"); axes[0].set_ylabel("density")
axes[0].set_title("Learned mode frequencies")
fig.tight_layout()
fig.savefig(OUT / "fig_v6_spectrum.png", dpi=130)
plt.close(fig)
S["mode_freq_cycles_per_token"] = {
    "median": float(np.median(all_freqs)), "p90": float(np.quantile(all_freqs, .9)),
    "max": float(all_freqs.max()),
}
# dominant text periodicities
top = np.argsort(spec2[m])[::-1][:5]
S["text_space_spectrum_peaks_cycles_per_byte"] = [float(fr[m][i]) for i in top]

# =================== 3. dt / B / C gating by byte class ===================
classes = {
    "space(32)": lambda b: b == 32, "newline(10)": lambda b: b == 10,
    "lowercase": lambda b: (b >= 97) & (b <= 122), "uppercase": lambda b: (b >= 65) & (b <= 90),
    "digit": lambda b: (b >= 48) & (b <= 57), "punct": lambda b: np.isin(b, list(b".,!?;:'\"()-")),
}
rows = []
for cname, pred in classes.items():
    mask = pred(text[:2048])
    entry = {"class": cname, "n": int(mask.sum())}
    x = model.tok_emb(idx_probe)
    with torch.no_grad():
        for li, (norm, layer) in enumerate(zip(model.prenorms, model.layers)):
            h = norm(x)
            dt = layer._compute_dt(h)[0]                       # (L, i)
            Bn = layer.W_B(h).view(h.shape[1], h.shape[2], layer.n_states)  # (L,i,N)
            Cn = layer.W_C(h).view(h.shape[1], h.shape[2], layer.n_states)
            entry[f"L{li}_dt"] = float(dt[mask].mean())
            entry[f"L{li}_Bnorm"] = float(Bn.abs()[mask].mean())
            entry[f"L{li}_Cnorm"] = float(Cn.abs()[mask].mean())
            x = x + layer(h)
    rows.append(entry)
S["byte_class_gating"] = rows

keys = [f"L{i}_{q}" for i in range(3) for q in ("dt", "Bnorm", "Cnorm")]
fig, axes = plt.subplots(3, 3, figsize=(13, 8))
names = [r["class"] for r in rows]
for col, q in enumerate(("dt", "Bnorm", "Cnorm")):
    for li in range(3):
        vals = [r[f"L{li}_{q}"] for r in rows]
        axes[li][col].bar(names, vals, color="steelblue")
        axes[li][col].set_title(f"L{li} {q}")
        axes[li][col].tick_params(axis="x", rotation=45, labelsize=7)
fig.suptitle("Content gating by byte class (mean over channels/modes)")
fig.tight_layout()
fig.savefig(OUT / "fig_v6_byteclass.png", dpi=130)
plt.close(fig)

# =================== 4. top-edge effective kernels + sparsity ===================
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
lags = np.arange(48)
x = model.tok_emb(idx_probe)
with torch.no_grad():
    for li, (norm, layer) in enumerate(zip(model.prenorms, model.layers)):
        h = norm(x)
        dt = layer._compute_dt(h)
        mean_dt = dt.mean(dim=(0, 1))
        rho = torch.exp(layer.log_rho.detach().clamp(-3, 3))
        sig_t = torch.exp(layer.log_sigma.detach().clamp(-6, 8)).cuda() * rho
        om_t = layer.omega.detach().cuda() * rho
        g_re, g_im = layer.a.detach(), layer.b.detach()
        mag = (g_re**2 + g_im**2).sum(-1)
        top = torch.topk(mag.flatten(), 3).indices
        cross_counts = []
        for flat in top:
            i, o = divmod(flat.item(), layer.out_dim)
            lg = torch.tensor(lags, device="cuda", dtype=torch.float32)
            env = torch.exp(-sig_t[i].unsqueeze(-1) * lg * mean_dt[i])       # (N, L)
            ph = om_t[i].unsqueeze(-1) * lg * mean_dt[i]                     # (N, L)
            K = (env * (g_re[i, o].unsqueeze(-1) * torch.cos(ph)
                        + g_im[i, o].unsqueeze(-1) * torch.sin(ph))).sum(0).cpu().numpy()
            axes[li].plot(lags, K / np.abs(K).max(), label=f"{i}->{o}")
            cross_counts.append(int(np.sum(np.diff(np.sign(K)) != 0)))
        mag_flat = mag.flatten().cpu().numpy()
        gini = float(np.abs(np.subtract.outer(mag_flat, mag_flat)).mean() / (2 * mag_flat.mean() * mag_flat.size))
        S[f"layer{li}_gini"] = gini
        S[f"layer{li}_top_edge_zero_crossings"] = cross_counts
        axes[li].legend(fontsize=7); axes[li].set_title(f"layer {li} top-|g| kernels")
        axes[li].set_xlabel("lag (tokens)")
        x = x + layer(h)
fig.suptitle("Effective per-edge kernels (normalized; oscillation = wavelet shape)")
fig.tight_layout()
fig.savefig(OUT / "fig_v6_kernels.png", dpi=130)
plt.close(fig)

(OUT / "v6_analysis_summary.json").write_text(json.dumps(S, indent=2))
print(json.dumps(S, indent=2)[:3000])
print("figures in", OUT)
