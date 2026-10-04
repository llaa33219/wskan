"""Canonical-100k edge statistics: |DC|/RMS per edge and readout-gain
effective rank per mode (fact-check regen, standalone).

Run: PYTHONPATH=.:experiments .venv/bin/python experiments/factcheck_edge_stats.py
"""

from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "experiments")
from W11_excess_structure import load_model

DEV = "cuda"


def main():
    m = load_model("checkpoints/wskan11_ultrachat_100k_3ep_s42/latest.pt", 40, 2)
    dcrms, ranks = [], []
    t = torch.arange(0, 2000, 0.02, device=DEV)
    with torch.no_grad():
        for li, layer in enumerate(m.layers):
            sigma = torch.exp(torch.clamp(layer.log_sigma, layer.log_sigma_min, layer.log_sigma_max))
            rho = layer._rho()
            sig_eff = sigma * rho
            om_eff = layer.omega * rho
            a, b = layer.a, layer.b
            env = torch.exp(-sig_eff[:, None, :] * t[None, :, None])
            ph = om_eff[:, None, :] * t[None, :, None]
            ker_re = env * torch.cos(ph)
            ker_im = env * torch.sin(ph)
            psi = (torch.einsum("itn,ion->iot", ker_re, a)
                   - torch.einsum("itn,ion->iot", ker_im, b))
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
    print(f"|DC|/RMS: median {np.median(dcrms):.4f}, frac<0.1 {np.mean(dcrms < 0.1):.4f}, "
          f"n_edges {len(dcrms)}")
    print(f"effective rank per mode (L0 then L1): {[round(r, 1) for r in ranks]} of 40")


if __name__ == "__main__":
    main()
