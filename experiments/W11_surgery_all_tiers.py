"""Per-tier surgery battery + per-layer boundary causal test.

  A. Surgical interventions at every tier (1k..10m, UltraChat s42):
     rho-flatten, omega-shuffle, feature-shuffle, full-permutation gauge.
  B. Per-layer boundary-Delta causal clamp at 100k (2 layers) vs 1m
     (6 layers): is the clock's causal role concentrated or distributed?

Outputs: figures/w11_surgery_all_tiers.json
Run: PYTHONPATH=.:experiments .venv/bin/python experiments/W11_surgery_all_tiers.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
sys.path.insert(0, "experiments")
from W11_excess_structure import load_model
from W11_round6_probes import shuffle_modes

DEV = "cuda"
BOUNDARY = torch.tensor([32, 10, 46, 33, 63, 44, 59, 58], device=DEV)
LOWER = torch.arange(97, 123, device=DEV)
DIMS = {"1k": (4, 1), "10k": (12, 1), "100k": (40, 2), "1m": (80, 6), "10m": (512, 2)}


def eval_ce(m, eval_ids, n=60000, blk=512):
    losses = []
    with torch.no_grad():
        for s in range(0, n, blk):
            ch = eval_ids[s:s + blk + 1]
            lg = m(ch[:-1].unsqueeze(0))[0]
            losses.append(F.cross_entropy(lg, ch[1:]).item())
    return float(np.mean(losses))


CURRENT: dict = {}


def wi_ce(m, ids, n=60000):
    ces, wis = [], []
    with torch.no_grad():
        for s in range(0, n, 512):
            ch = ids[s:s + 513]
            CURRENT["ids"] = ch[:-1]
            lg = m(ch[:-1].unsqueeze(0))[0]
            ce = F.cross_entropy(lg, ch[1:], reduction="none")
            ces.append(ce)
            wis.append(ch[:-1] == 32)
    ce = torch.cat(ces)
    wi = torch.cat(wis)
    return float(ce[wi].mean())


def clamp_layer_boundary(m, li, ids):
    orig = m.layers[li]._compute_dt

    def patched(x):
        dt = orig(x)
        idx_flat = CURRENT["ids"]
        mask = torch.isin(idx_flat, BOUNDARY)[None, :, None].float()
        letters = torch.isin(idx_flat, LOWER)[None, :, None].float()
        letter_mean = (dt * letters).sum(1, keepdim=True) / letters.sum(1, keepdim=True).clamp(min=1)
        return dt * (1 - mask) + letter_mean * mask
    m.layers[li]._compute_dt = patched
    return orig


def main():
    from experiments.V1_train_tinystories_lm import load_data
    _, ev = load_data(1400000, 500, 42, "ultrachat")
    eval_ids = ev.long().to(DEV)
    out = {"surgery": {}, "perlayer_causal": {}}

    for tier, (d, L) in DIMS.items():
        ck = f"checkpoints/wskan11_ultrachat_{tier}_3ep_s42/latest.pt"
        base = eval_ce(load_model(ck, d, L), eval_ids)
        m = load_model(ck, d, L)
        for layer in m.layers:
            layer.log_rho.data.zero_()
        rho_ce = eval_ce(m, eval_ids)
        m = load_model(ck, d, L)
        shuffle_modes(m, "omega", torch.Generator().manual_seed(0))
        om_ce = eval_ce(m, eval_ids)
        m = load_model(ck, d, L)
        perm_f = torch.randperm(8, generator=torch.Generator().manual_seed(0))
        for layer in m.layers:
            layer.M_B.data = layer.M_B.data[perm_f]
            layer.M_C.data = layer.M_C.data[perm_f]
        ft_ce = eval_ce(m, eval_ids)
        m = load_model(ck, d, L)
        shuffle_modes(m, "all", torch.Generator().manual_seed(0))
        gauge_ce = eval_ce(m, eval_ids)
        out["surgery"][tier] = dict(base=round(base, 4), rho_flat=round(rho_ce - base, 4),
                                    omega_shuffle=round(om_ce - base, 4),
                                    feature_shuffle=round(ft_ce - base, 4),
                                    gauge=round(gauge_ce - base, 10))
        print(f"{tier}: base {base:.4f} | rho {rho_ce - base:+.4f} | omega {om_ce - base:+.4f} | "
              f"feat {ft_ce - base:+.4f} | gauge {gauge_ce - base:+.1e}")

    for tier in ("100k", "1m"):
        d, L = DIMS[tier]
        m = load_model(f"checkpoints/wskan11_ultrachat_{tier}_3ep_s42/latest.pt", d, L)
        base_wi = wi_ce(m, eval_ids)
        rows = []
        for li in range(L):
            orig = clamp_layer_boundary(m, li, eval_ids)
            wi = wi_ce(m, eval_ids)
            m.layers[li]._compute_dt = orig
            rows.append(round(wi - base_wi, 4))
        out["perlayer_causal"][tier] = dict(baseline_wi=round(base_wi, 4), per_layer_delta=rows)
        print(f"{tier}: baseline wi-CE {base_wi:.4f} | per-layer boundary-clamp deltas {rows}")

    with open("experiments/figures/w11_surgery_all_tiers.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved experiments/figures/w11_surgery_all_tiers.json")


if __name__ == "__main__":
    main()
