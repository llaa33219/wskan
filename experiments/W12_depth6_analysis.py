"""W12 depth-confound control analysis (review round 4, 2026-10-04).

The campaign tiers confound capacity with depth: 1m is 6 layers, 10m is 2.
The 100k-tier clock is causally concentrated at L0 while the 6-layer 1m
tier spreads it evenly - so "concentration" might track depth, not size.
The depth6 control (d=19, L=6, 109k params ~ 100k tier) decides: if a
~100k-param 6-layer model spreads the clock's causal role across layers,
concentration is a depth property; if it concentrates at L0, it tracks
capacity/protocol.

Per seed: per-layer space/letter dt ratios; per-layer boundary clamp
(clamp that layer's boundary dt to its letter mean) vs count-matched
letter control, word-initial CE damage.

Outputs: figures/w12_depth6_analysis.json
Run: PYTHONPATH=.:experiments .venv/bin/python experiments/W12_depth6_analysis.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
sys.path.insert(0, "experiments")
from models.V11_WSKAN import WaveletStateKANLMV11
from experiments.V1_train_tinystories_lm import load_data

DEV = "cuda"
BOUNDARY = torch.tensor([32, 10, 46, 33, 63, 44, 59, 58], device=DEV)
LOWER = torch.arange(97, 123, device=DEV)


def load_depth6(seed):
    m = WaveletStateKANLMV11(vocab_size=256, d_model=19, n_layers=6, use_feature_bc=True,
                             wz_diag=False, g_rank=None, bc_rank=19, bf16_scan=False)
    sd = torch.load(f"checkpoints/wskan11abl-depth6_ultrachat_100k_3ep_s{seed}/latest.pt",
                    weights_only=False)
    m.load_state_dict(sd["state_dict"] if "state_dict" in sd else sd)
    return m.eval().to(DEV)


def analyze(seed, ev_ids):
    m = load_depth6(seed)
    ids = ev_ids[:60000].long().to(DEV)

    dt_ratios = []
    dts_per_layer = [[] for _ in range(6)]
    with torch.no_grad():
        for s in range(0, 20000, 512):
            idx = ids[s:s + 512].unsqueeze(0)
            x = m.tok_emb(idx)
            for li, (norm, layer) in enumerate(zip(m.prenorms, m.layers)):
                h = norm(x)
                dts_per_layer[li].append(layer._compute_dt(h)[0].cpu())
                x = x + layer(h, idx)
    for li in range(6):
        dt = torch.cat(dts_per_layer[li], 0).to(DEV)
        sp = dt[ids[:dt.shape[0]] == 32].mean()
        lt = dt[torch.isin(ids[:dt.shape[0]], LOWER)].mean()
        dt_ratios.append(round(float(sp / lt), 3))

    def wi_ce(clamp_layer=None):
        patched = []
        cur_ids = {}
        if clamp_layer is not None:
            layer = m.layers[clamp_layer]
            orig = layer._compute_dt
            def mk(x, orig=orig, cur_ids=cur_ids):
                dt = orig(x)
                cur = cur_ids["ids"]
                mask = torch.isin(cur, BOUNDARY)[None, :, None].float()
                letters = torch.isin(cur, LOWER)[None, :, None].float()
                lm = (dt * letters).sum(1, keepdim=True) / letters.sum(1, keepdim=True).clamp(min=1)
                return dt * (1 - mask) + lm * mask
            layer._compute_dt = mk
            patched.append((layer, orig))
        ces, wis = [], []
        with torch.no_grad():
            for s in range(0, 60000, 512):
                ch = ids[s:s + 513]
                cur_ids["ids"] = ch[:-1].to(DEV)
                lg = m(ch[:-1].unsqueeze(0))[0]
                ce = F.cross_entropy(lg, ch[1:], reduction="none")
                ces.append(ce)
                wis.append(ch[:-1] == 32)
        for layer, orig in patched:
            layer._compute_dt = orig
        ce = torch.cat(ces)
        return float(ce[torch.cat(wis)].mean())

    base = wi_ce()
    per_layer = []
    for li in range(6):
        per_layer.append(round(wi_ce(li) - base, 4))
    return dict(dt_ratio_per_layer=dt_ratios, base_wi=round(base, 4),
                boundary_clamp_damage_per_layer=per_layer)


def main():
    _, ev = load_data(1400000, 500, 42, "ultrachat")
    out = {}
    for s in (42, 7):
        out[f"s{s}"] = analyze(s, ev)
        print(f"s{s}: {out[f's{s}']}", flush=True)
    with open("experiments/figures/w12_depth6_analysis.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved experiments/figures/w12_depth6_analysis.json")


if __name__ == "__main__":
    main()
