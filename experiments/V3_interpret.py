"""V3 interpretability analysis: extract and visualize learned edge functions.

Loads a trained WaveletStateKANLMV3 checkpoint and produces:
  1. fig_edge_wavelets.png   - per-edge nominal wavelet psi_io(t) samples
  2. fig_modes.png           - learned (sigma, omega) vs S4D-Lin init
  3. fig_edge_magnitude.png  - |g| heatmaps (which edges carry the function)
  4. fig_impulse.png         - nominal impulse response kernels K_l

Run: .venv/bin/python experiments/V3_interpret.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.V3_WSKAN import WaveletStateKANLMV3

CKPT = "checkpoints/wskan3_tinystories_lm_s42/latest.pt"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

model = WaveletStateKANLMV3()
model.load_state_dict(torch.load(CKPT, weights_only=False)["state_dict"])
model.eval()

summary = {}
t = np.linspace(-4, 4, 400)

# ---------- 1. per-edge wavelets (static nominal shape) ----------
fig, axes = plt.subplots(3, 4, figsize=(13, 8))
for li, layer in enumerate(model.layers):
    mag = (layer.a**2 + layer.b**2).sum(-1)  # (in, out) edge strength
    top = torch.topk(mag.flatten(), 4).indices
    for ax, flat_idx in zip(axes[li], top):
        i, o = divmod(flat_idx.item(), layer.out_dim)
        with torch.no_grad():
            tt = torch.tensor(t, dtype=torch.float32).view(-1, 1, 1)
            psi = layer.wavelet(tt)[:, i, o].numpy()
        ax.plot(t, psi)
        ax.set_title(f"L{li} edge {i}->{o} (|g|={mag[i,o]:.2f})", fontsize=8)
        ax.axhline(0, lw=0.5, c="gray")
fig.suptitle("V3 learned edge wavelets (top-4 edges by |g| per layer)")
fig.tight_layout()
fig.savefig(OUT / "fig_edge_wavelets.png", dpi=130)
plt.close(fig)

# ---------- 2. learned modes vs init ----------
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
init_omega = np.pi * np.arange(1, 7)
for li, layer in enumerate(model.layers):
    sigma = torch.exp(layer.log_sigma.clamp(-4, 2)).detach().flatten().numpy()
    omega = layer.omega.detach().flatten().numpy()
    axes[li].scatter(init_omega.tolist() * (sigma.size // 6), sigma, s=8, alpha=0.5, label="init")
    axes[li].scatter(omega, sigma, s=8, alpha=0.5, label="learned")
    axes[li].set_xlabel("omega (frequency)")
    axes[li].set_ylabel("sigma (decay)")
    axes[li].set_title(f"layer {li}")
    axes[li].legend(markerscale=2, fontsize=7)
    summary[f"layer{li}"] = {
        "omega_init": init_omega.tolist(),
        "omega_learned_mean": float(np.mean(omega)),
        "omega_learned_std": float(np.std(omega)),
        "omega_drift_mean": float(np.mean(np.abs(omega - np.tile(init_omega, sigma.size // 6)))),
        "sigma_mean": float(np.mean(sigma)),
        "sigma_std": float(np.std(sigma)),
    }
fig.suptitle("Learned (omega, sigma) vs S4D-Lin wavelet init")
fig.tight_layout()
fig.savefig(OUT / "fig_modes.png", dpi=130)
plt.close(fig)

# ---------- 3. edge magnitude heatmaps ----------
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
for li, layer in enumerate(model.layers):
    mag = (layer.a**2 + layer.b**2).sum(-1).detach().numpy()
    im = axes[li].imshow(mag, cmap="viridis", aspect="auto")
    axes[li].set_title(f"layer {li}")
    axes[li].set_xlabel("out node")
    axes[li].set_ylabel("in node")
    fig.colorbar(im, ax=axes[li])
    summary[f"layer{li}"]["edge_mag_max"] = float(mag.max())
    summary[f"layer{li}"]["edge_mag_median"] = float(np.median(mag))
    summary[f"layer{li}"]["edge_mag_p90_over_median"] = float(np.quantile(mag, 0.9) / np.median(mag))
fig.suptitle("Edge strength |g| (sum over modes)")
fig.tight_layout()
fig.savefig(OUT / "fig_edge_magnitude.png", dpi=130)
plt.close(fig)

# ---------- 4. nominal impulse response kernels ----------
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
lags = np.arange(64)
for li, layer in enumerate(model.layers):
    sigma = torch.exp(layer.log_sigma.clamp(-4, 2)).detach()  # (in, N)
    omega = layer.omega.detach()
    g = torch.complex(layer.a, layer.b).detach()  # (in, out, N)
    env = torch.exp(-sigma.unsqueeze(-1) * torch.tensor(lags))  # (in, N, L)
    modes = torch.polar(env, omega.unsqueeze(-1) * torch.tensor(lags))  # e^{lam t}
    K = torch.einsum("ion,inl->iol", g, modes).real  # nominal kernel (B=delta)
    mag = (layer.a**2 + layer.b**2).sum(-1)
    for flat_idx in torch.topk(mag.flatten(), 3).indices:
        i, o = divmod(flat_idx.item(), layer.out_dim)
        axes[li].plot(lags, K[i, o].numpy(), label=f"{i}->{o}")
    axes[li].set_title(f"layer {li}")
    axes[li].set_xlabel("lag l (tokens)")
    axes[li].legend(fontsize=7)
fig.suptitle("Nominal impulse-response kernels K_l of top edges (recurrent mode)")
fig.tight_layout()
fig.savefig(OUT / "fig_impulse.png", dpi=130)
plt.close(fig)

(OUT / "interpret_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print("figures in", OUT)
