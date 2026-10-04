"""Fact-check regeneration: recompute the paper's disputed numbers at full
precision, fp32, from the canonical checkpoints.

  1. frien->d decomposition, ALL tiers, full precision (parts, margin,
     per-mode list) -- the paper prints rounded/unrecorded values.
  2. Mean top1-top2 margin over the 50 truncation contexts per tier
     (paper App. D narrative: "0.94 -> 1.15 -> 2.37" -- unsourced).
  3. Boundary-byte share of the top-25 history terms per tier + base rate
     (paper: "17% at every tier" -- unsourced).
  4. |DC|/RMS per edge on the canonical 100k checkpoint
     (paper: "median 0.12; 36-40% below 0.1" -- unsourced).
  5. Effective rank of the readout gain [a|b] per mode, canonical 100k
     (paper: "~29 of 32" -- V7-era number; wskan11 is d=40).

Output: experiments/figures/factcheck_regen.json + stdout.
Run: PYTHONPATH=.:experiments .venv/bin/python experiments/factcheck_regen.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "experiments")
from W11_hand_simulation import (hand_forward, margin_decomposition,
                                 per_history_terms, ctxs_from_eval)
from W11_excess_structure import load_model

DEV = "cuda"
ULTRACHAT_CKPT = {t: f"checkpoints/wskan11_ultrachat_{t}_3ep_s42/latest.pt"
                  for t in ("1k", "10k", "100k", "1m", "10m")}
TRUNC_CKPT = {"1k": "checkpoints/wskan11_tinystories_1k_3ep_s42/latest.pt",
              "10k": "checkpoints/wskan11_tinystories_10k_3ep_s42/latest.pt",
              "100k": ULTRACHAT_CKPT["100k"], "1m": ULTRACHAT_CKPT["1m"],
              "10m": ULTRACHAT_CKPT["10m"]}
TRUNC_DS = {"1k": "tinystories", "10k": "tinystories", "100k": "ultrachat",
            "1m": "ultrachat", "10m": "ultrachat"}
DIMS = {"1k": (4, 1), "10k": (12, 1), "100k": (40, 2), "1m": (80, 6), "10m": (512, 2)}
BOUNDARY_BYTES = set(b" \n.,!?;:'\"-()")


def per_mode_wave(m, hf, li, std, v1, v2):
    """Per-mode decomposition of layer li's wave path into the v1-vs-v2 margin."""
    lo = hf["layers"][li]
    a, b = lo["read_gain"]
    C_last = lo["C_last"].detach().cpu().numpy()
    gate = lo["gate"].detach().cpu().numpy()
    gn = m.norm.weight.detach().cpu().numpy()
    terms = lo["terms"].detach().cpu().numpy()
    re, im = terms.real, terms.imag
    an, bn = a.detach().cpu().numpy(), b.detach().cpu().numpy()
    W = m.head.weight.detach().cpu().numpy()
    out = []
    for k in range(lo["terms"].shape[-1]):
        y_o = (re[:, :, k].sum(0)[:, None] * C_last[:, k][:, None] * an[:, :, k]
               - im[:, :, k].sum(0)[:, None] * C_last[:, k][:, None] * bn[:, :, k])
        yg = y_o.sum(0) * gate
        v = (yg / std) * gn
        out.append(float(W[v1] @ v - W[v2] @ v))
    return out


def main():
    out = {}
    # ---------- 1. frien->d full precision, all tiers ----------
    ctx = b"User: Hello! How are you today, my dear frien"
    for tier, (d, L) in DIMS.items():
        m = load_model(ULTRACHAT_CKPT[tier], d, L)
        idx = torch.tensor(list(ctx), device=DEV).unsqueeze(0)
        hf = hand_forward(m, idx)
        parts, logits, std = margin_decomposition(m, idx, hf)
        W = m.head.weight.detach().cpu().numpy()
        t1, t2 = np.argsort(-logits)[:2]
        dm = W[t1] - W[t2]
        row = {"pred": chr(int(t1)), "runner_up": chr(int(t2)),
               "margin": float(logits[t1] - logits[t2]),
               "parts": {k: float(dm @ p) for k, p in parts.items()},
               "per_mode_last_wave": per_mode_wave(m, hf, L - 1, std, t1, t2)}
        row["parts_sum"] = float(sum(row["parts"].values()))
        out.setdefault("friend", {})[tier] = row
        print(f"[friend {tier}] pred {row['pred']!r} margin {row['margin']:.4f} "
              f"sum {row['parts_sum']:.4f} parts { {k: round(v, 3) for k, v in row['parts'].items()} }")
        print(f"   per-mode last-wave: {[round(x, 3) for x in row['per_mode_last_wave']]}")
        del m
        torch.cuda.empty_cache()

    # ---------- 2+3. truncation contexts: mean margin, boundary share ----------
    for tier, (d, L) in DIMS.items():
        ds = TRUNC_DS[tier]
        m = load_model(TRUNC_CKPT[tier], d, L)
        margins, shares, bases = [], [], []
        for w in ctxs_from_eval(ds, 50, 24, seed=7):
            ctx_b = [int(c) for c in w[:-1]]
            idx = torch.tensor(ctx_b, device=DEV).unsqueeze(0)
            hf = hand_forward(m, idx)
            parts, logits, std = margin_decomposition(m, idx, hf)
            o = np.argsort(-logits)
            margins.append(float(logits[o[0]] - logits[o[1]]))
            v_full = int(o[0])
            flats = []
            for li in range(L):
                full = per_history_terms(m, hf, li, std)
                flats.append(full.reshape(-1, full.shape[-1]))
            flat = np.concatenate(flats, axis=0)     # (L*I*N summed layers, vocab)
            top25 = np.argsort(-np.abs(flat[:, v_full]))[:25]
            # position index of each term: terms stacked layer-major; per layer L*I*N
            per_layer = 24 * d * 6
            pos = [(t % per_layer) // (d * 6) for t in top25]
            shares.append(float(np.mean([ctx_b[p] in BOUNDARY_BYTES for p in pos])))
            bases.append(float(np.mean([c in BOUNDARY_BYTES for c in ctx_b])))
        out.setdefault("trunc50", {})[tier] = dict(
            mean_top12_margin=float(np.mean(margins)),
            boundary_share_top25=float(np.mean(shares)),
            boundary_base_rate=float(np.mean(bases)))
        print(f"[trunc50 {tier}] mean top1-top2 margin {np.mean(margins):.3f} | "
              f"boundary share of top-25 {np.mean(shares):.3f} (base rate {np.mean(bases):.3f})")
        del m
        torch.cuda.empty_cache()

    # ---------- 4+5. canonical 100k: DC/RMS per edge, effective rank ----------
    d, L = DIMS["100k"]
    m = load_model(ULTRACHAT_CKPT["100k"], d, L)
    dcrms, ranks = [], []
    t = torch.arange(0, 2000, 0.02, device=DEV)  # warped-time grid, covers slow tail
    with torch.no_grad():
        for li, layer in enumerate(m.layers):
            sigma = torch.exp(torch.clamp(layer.log_sigma, layer.log_sigma_min, layer.log_sigma_max))
            rho = layer._rho()
            sig_eff = (sigma * rho)                      # (I, N)
            om_eff = (layer.omega * rho)
            a, b = layer.a, layer.b                      # (I, O, N)
            env = torch.exp(-sig_eff[:, None, :] * t[None, :, None])          # (I, T, N)
            ph = om_eff[:, None, :] * t[None, :, None]
            ker_re = env * torch.cos(ph)                                      # (I, T, N)
            ker_im = env * torch.sin(ph)
            psi = (torch.einsum("itn,ion->iot", ker_re, a)
                   - torch.einsum("itn,ion->iot", ker_im, b))                 # (I, O, T)
            dc = psi.mean(-1).abs()
            rms = psi.pow(2).mean(-1).sqrt().clamp(min=1e-12)
            dcrms.append((dc / rms).flatten().cpu().numpy())
            for k in range(layer.n_states):
                Mk = np.concatenate([a[:, :, k].detach().cpu().numpy(),
                                     b[:, :, k].detach().cpu().numpy()], axis=1)
                sv = np.linalg.svd(Mk, compute_uv=False)
                pv = sv / sv.sum()
                ranks.append(float(np.exp(-(pv * np.log(pv + 1e-12)).sum())))
    dcrms = np.concatenate(dcrms)
    out["canonical_100k"] = dict(
        dc_rms_median=float(np.median(dcrms)),
        dc_rms_frac_below_01=float(np.mean(dcrms < 0.1)),
        eff_rank_per_mode=[round(r, 1) for r in ranks],
        d_model=d, n_modes=6)
    print(f"[100k] |DC|/RMS median {np.median(dcrms):.3f}, frac<0.1 {np.mean(dcrms < 0.1):.3f}")
    print(f"[100k] effective rank per mode (L0 then L1): {[round(r, 1) for r in ranks]} of {d}")

    with open("experiments/figures/factcheck_regen.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved experiments/figures/factcheck_regen.json")


if __name__ == "__main__":
    main()
