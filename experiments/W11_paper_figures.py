"""Paper figures for the WSKAN monograph (4 schematics).

  fig_clock.pdf    - the boundary clock field over a sample text
  fig_kernels.pdf  - gallery of learned edge impulse responses (warped time)
  fig_decision.pdf - the frien->d exact decision decomposition
  fig_dynamics.pdf - emergence trajectories (20k-step dynamics runs)

Outputs: paper/figures/*.pdf
"""

from __future__ import annotations

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
sys.path.insert(0, "experiments")
from W11_excess_structure import load_model
from W11_hand_simulation import hand_forward, margin_decomposition, per_history_terms

DEV = "cuda"
CKPT = "checkpoints/wskan11_ultrachat_100k_3ep_s42/latest.pt"


def fig_clock(m):
    text = b"Hello! How are you today, my dear friend?"
    idx = torch.tensor(list(text), device=DEV).unsqueeze(0)
    with torch.no_grad():
        x = m.tok_emb(idx)
        dts = []
        for norm, layer in zip(m.prenorms, m.layers):
            h = norm(x)
            dt = torch.clamp(F.softplus(layer.W_dt(h)), max=1.0)[0]  # (L, d)
            dts.append(dt.mean(-1).cpu().numpy())
            x = x + layer(h, idx)
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    xs = np.arange(len(text))
    ax.plot(xs, dts[0], lw=2, label="L0", color="#1f77b4")
    ax.plot(xs, dts[1], lw=2, label="L1", color="#d62728")
    import matplotlib.transforms as mtransforms
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for i, c in enumerate(text):
        ch = chr(c)
        if ch == " ":
            ax.axvspan(i - 0.5, i + 0.5, color="#ffcc80", alpha=0.45, lw=0,
                       hatch="///", edgecolor="#b45309", linewidth=0.0)
            ax.plot([i], [0.035], marker="v", color="#b45309", markersize=5,
                    transform=trans, clip_on=False)
        ax.annotate(ch if ch != " " else "_", (i, -0.03), ha="center", fontsize=9,
                    color="gray", family="monospace", xycoords=trans, annotation_clip=False)
    ax.set_ylabel(r"$\Delta$ (warped-time step)")
    ax.set_xlabel("position (byte below axis; shaded = space)")
    ax.set_title("The boundary clock: $\\Delta$ pulses at spaces and punctuation")
    ax.legend()
    fig.tight_layout()
    fig.savefig("paper/figures/fig_clock.pdf")


def fig_kernels(m):
    layer = m.layers[1]
    with torch.no_grad():
        sigma = torch.exp(torch.clamp(layer.log_sigma, layer.log_sigma_min, layer.log_sigma_max))
        rho = layer._rho()
        se = (sigma * rho).cpu().numpy()
        oe = (layer.omega * rho).cpu().numpy()
        layer._compose_g()
        a = layer.a.cpu().numpy(); b = layer.b.cpu().numpy()
    t = np.linspace(0, 2.5, 500)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.4), sharex=True)
    # one representative edge (i=0, o=0) per mode, plus one aggregate
    for k in range(6):
        ax = axes[k // 3, k % 3]
        psi = np.exp(-se[0, k] * t) * (a[0, 0, k] * np.cos(oe[0, k] * t)
                                       + b[0, 0, k] * np.sin(oe[0, k] * t))
        ax.plot(t, psi, lw=1.8, color="#1f77b4")
        ax.axhline(0, color="gray", lw=0.5)
        hl = np.log(2) / se[0, k]
        ax.axvline(hl, color="#d62728", ls="--", lw=1)
        ax.set_title(f"mode {k}: $\\tilde\\sigma$={se[0,k]:.1f}, "
                     f"$\\tilde\\omega$={oe[0,k]:.1f}", fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("warped time")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"kernel value")
    fig.suptitle("Learned edge kernels, one edge, six modes (L1, channel 0; "
                 "dashed = half-life)", fontsize=9)
    fig.tight_layout()
    fig.savefig("paper/figures/fig_kernels.pdf")


def fig_decision(m):
    ctx = b"User: Hello! How are you today, my dear frien"
    idx = torch.tensor(list(ctx), device=DEV).unsqueeze(0)
    hf = hand_forward(m, idx)
    parts, logits, std = margin_decomposition(m, idx, hf)
    t1, t2 = np.argsort(-logits)[:2]
    names = ["emb", "L0-base", "L0-wave", "L1-base", "L1-wave", "norm-const"]
    vals = [float((m.head.weight.detach().cpu().numpy()[t1] - m.head.weight.detach().cpu().numpy()[t2])
                  @ parts[k]) for k in ("emb", "L0-base", "L0-wave", "L1-base", "L1-wave", "norm-const")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1, 1.2]})
    colors = ["#1f77b4" if v >= 0 else "#d62728" for v in vals]
    axes[0].bar(names, vals, color=colors)
    axes[0].set_title("frien$\\to$d: margin decomposition (sum exact)", fontsize=9)
    axes[0].set_ylabel("margin contribution")
    axes[0].tick_params(axis="x", rotation=45)
    lo = hf["layers"][1]
    terms = lo["terms"].detach().cpu().numpy()
    a, b = lo["read_gain"]
    an, bn = a.detach().cpu().numpy(), b.detach().cpu().numpy()
    Cn = lo["C_last"].detach().cpu().numpy()
    gate = lo["gate"].detach().cpu().numpy()
    gn = m.norm.weight.detach().cpu().numpy()
    W = m.head.weight.detach().cpu().numpy()
    per_mode = []
    for k in range(6):
        y_o = (terms.real[:, :, k].sum(0)[:, None] * Cn[:, k][:, None] * an[:, :, k]
               - terms.imag[:, :, k].sum(0)[:, None] * Cn[:, k][:, None] * bn[:, :, k])
        v = ((y_o.sum(0) * gate) / std) * gn
        per_mode.append(float(W[t1] @ v - W[t2] @ v))
    axes[1].bar([f"mode {k}" for k in range(6)], per_mode,
                color=["#1f77b4" if v >= 0 else "#d62728" for v in per_mode])
    axes[1].set_title("L1-wave per mode (0-indexed)", fontsize=9)
    axes[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig("paper/figures/fig_decision.pdf")


def fig_dynamics():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharex=True)
    for tier, color in (("1k", "#1f77b4"), ("100k", "#d62728")):
        traj = json.load(open(f"experiments/figures/w11_dynamics20k_{tier}.json"))
        steps = [r["step"] for r in traj]
        l0 = [r["layers"][-1] for r in traj]
        axes[0].plot(steps, [l["clock_space_letter"] for l in l0], color=color, label=tier)
        axes[1].plot(steps, [l["gate_split_named_share"] for l in l0], color=color)
        axes[2].plot(steps, [l["timescale_ladder"]["r2"] for l in l0], color=color)
    axes[0].set_title("boundary clock ratio", fontsize=9)
    axes[1].set_title("gate named-share", fontsize=9)
    axes[2].set_title("timescale ladder $R^2$", fontsize=9)
    for ax in axes:
        ax.set_xlabel("step")
    axes[0].legend()
    fig.suptitle("Emergence dynamics (from scratch, 20k steps)", fontsize=9)
    fig.tight_layout()
    fig.savefig("paper/figures/fig_dynamics.pdf")


def main():
    import os
    os.makedirs("paper/figures", exist_ok=True)
    m = load_model(CKPT, 40, 2)
    fig_clock(m)
    fig_kernels(m)
    fig_decision(m)
    fig_dynamics()
    print("saved paper/figures/fig_{clock,kernels,decision,dynamics}.pdf")


if __name__ == "__main__":
    main()
