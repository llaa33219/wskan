"""W12 per-layer clock battery for control variants (review round 5).

Generalizes W12_depth6_analysis to arbitrary checkpoints: per-layer
space/letter dt ratios + per-layer boundary-clamp word-initial damage.

  scramfeat (d=40, L=2, random byte-partition feature basis): does the
  boundary clock still emerge with a meaningless feature basis?
  wide2 (d=145, L=2, ~960k): does the 2-layer skeleton at 1m capacity
  concentrate the clock at L0 (like 2-layer 100k) or spread it?

Outputs: figures/w12_variant_clocks.json
Run: PYTHONPATH=.:experiments .venv/bin/python experiments/W12_variant_clocks.py
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

MODELS = {
    "scramfeat_s42": ("checkpoints/wskan11abl-scramfeat_ultrachat_100k_3ep_s42/latest.pt", 40, 2, 32),
    "scramfeat_s7": ("checkpoints/wskan11abl-scramfeat_ultrachat_100k_3ep_s7/latest.pt", 40, 2, 32),
    "wide2_s42": ("checkpoints/wskan11abl-wide2_ultrachat_1m_54k_s42/latest.pt", 145, 2, 32),
    "wide2_s7": ("checkpoints/wskan11abl-wide2_ultrachat_1m_54k_s7/latest.pt", 145, 2, 32),
}


def load(ckpt, d, L, bcr):
    if "scramfeat" in ckpt:
        import models.V7_WSKAN as _v7
        seed = int(ckpt.rsplit("_s", 1)[1].split("/")[0])
        _g = np.random.default_rng(seed + 5555)
        _assign = torch.tensor(_g.integers(0, 7, 256), dtype=torch.long)
        def _scrambled(idx):
            oh = F.one_hot(_assign.to(idx.device)[idx], 7).float()
            return torch.cat([oh, torch.ones_like(oh[..., :1])], dim=-1)
        _v7.byte_features = _scrambled
    m = WaveletStateKANLMV11(vocab_size=256, d_model=d, n_layers=L, use_feature_bc=True,
                             wz_diag=False, g_rank=None, bc_rank=bcr, bf16_scan=False)
    sd = torch.load(ckpt, weights_only=False)
    m.load_state_dict(sd["state_dict"] if "state_dict" in sd else sd)
    return m.eval().to(DEV)


def analyze(m, ids):
    n_layers = len(m.layers)
    dt_ratios = []
    dts_per_layer = [[] for _ in range(n_layers)]
    with torch.no_grad():
        for s in range(0, 20000, 512):
            idx = ids[s:s + 512].unsqueeze(0).to(DEV)
            x = m.tok_emb(idx)
            for li, (norm, layer) in enumerate(zip(m.prenorms, m.layers)):
                h = norm(x)
                dts_per_layer[li].append(layer._compute_dt(h)[0].cpu())
                x = x + layer(h, idx)
    for li in range(n_layers):
        dt = torch.cat(dts_per_layer[li], 0)
        sp = dt[ids[:dt.shape[0]] == 32].mean()
        lt = dt[torch.isin(ids[:dt.shape[0]], LOWER.cpu())].mean()
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
                ch = ids[s:s + 513].to(DEV)
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
    per_layer = [round(wi_ce(li) - base, 4) for li in range(n_layers)]
    return dict(dt_ratio_per_layer=dt_ratios, base_wi=round(base, 4),
                boundary_clamp_damage_per_layer=per_layer)


def main():
    _, ev = load_data(1400000, 500, 42, "ultrachat")
    ids = ev[:60000].long()
    out = {}
    for name, (ck, d, L, bcr) in MODELS.items():
        m = load(ck, d, L, bcr)
        out[name] = analyze(m, ids)
        print(f"{name}: {out[name]}", flush=True)
        del m
        torch.cuda.empty_cache()
    with open("experiments/figures/w12_variant_clocks.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved experiments/figures/w12_variant_clocks.json")


if __name__ == "__main__":
    main()
