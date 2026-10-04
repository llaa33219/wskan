"""W12 implicit-boundary clock test, 5 seeds (review response, 2026-10-04).

The W11 implicit-boundary test (space-free text, true boundaries recorded
from the source) was single-seed (s42). This reruns it on all five campaign
seeds. Protocol identical to W11_implicit_boundary.implicit_test; only the
checkpoint varies.

Outputs: figures/w12_implicit_boundary_5seed.json
Run: PYTHONPATH=.:experiments .venv/bin/python experiments/W12_implicit_boundary_5seed.py
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
from experiments.V1_train_tinystories_lm import load_data

DEV = "cuda"
SEEDS = (42, 7, 123, 2024, 31337)


def implicit_test(tr_arr, ckpt):
    src = bytes(tr_arr[:40000])
    sf_bytes, true_bnd = [], []
    out_i = 0
    for c in src:
        if c == 32:
            true_bnd.append(out_i)
        else:
            sf_bytes.append(c)
            out_i += 1
    sf = bytes(sf_bytes)
    true_bnd = np.array([p for p in true_bnd if p < len(sf) - 1])
    m = load_model(ckpt, 40, 2)
    arr = np.array(list(sf))
    letters = ((arr >= 97) & (arr <= 122))
    internal = letters.copy()
    internal[true_bnd] = False

    with torch.no_grad():
        dt_layers = [[] for _ in range(2)]
        for s in range(0, len(sf) - 512, 512):
            idx = torch.tensor(list(sf[s:s + 512]), device=DEV).unsqueeze(0)
            x = m.tok_emb(idx)
            for li, (norm, layer) in enumerate(zip(m.prenorms, m.layers)):
                h = norm(x)
                dt_layers[li].append(torch.clamp(F.softplus(layer.W_dt(h)), max=1.0)[0].cpu().numpy())
                x = x + layer(h, idx)
        dts = [np.concatenate(v, 0) for v in dt_layers]

    BPOS = torch.tensor(true_bnd, device=DEV)

    def wi_ce(clamp_positions=None, li=1):
        ces, wis = [], []
        orig = m.layers[li]._compute_dt
        cur = {}
        if clamp_positions is not None:
            def patched(x):
                dt = orig(x)
                mask = cur["mask"]
                lm = dt.mean(1, keepdim=True)
                return dt * (1 - mask) + lm * mask
            m.layers[li]._compute_dt = patched
        with torch.no_grad():
            for s in range(0, 20000, 512):
                ch = torch.tensor(list(sf[s:s + 513]), device=DEV).unsqueeze(0)
                if clamp_positions is not None:
                    Lx = ch.shape[1] - 1
                    loc = clamp_positions[(clamp_positions >= s) & (clamp_positions < s + Lx)] - s
                    mk = torch.zeros(1, Lx, 1, device=DEV)
                    if len(loc):
                        mk[0, loc, 0] = 1.0
                    cur["mask"] = mk
                lg = m(ch[:, :-1])[0]
                ce = F.cross_entropy(lg, ch[0, 1:], reduction="none")
                ces.append(ce)
                pos = torch.arange(s, s + 512, device=DEV)
                wis.append(torch.isin(pos, BPOS))
        m.layers[li]._compute_dt = orig
        ce = torch.cat(ces)
        return float(ce[torch.cat(wis)].mean())

    ratios = {}
    for li in range(2):
        dt = dts[li]
        b = true_bnd[true_bnd < dt.shape[0]]
        ratios[f"L{li}"] = round(float(dt[b].mean() / dt[internal[:dt.shape[0]]].mean()), 3)
    b0 = wi_ce(li=0)
    clamps = {li: round(wi_ce(BPOS, li=li) - b0, 4) for li in (0, 1)}
    rng = np.random.default_rng(0)
    ctrl = torch.tensor(np.sort(rng.choice(np.where(letters)[0], size=len(true_bnd), replace=False)),
                        device=DEV)
    ctrls = {li: round(wi_ce(ctrl, li=li) - b0, 4) for li in (0, 1)}
    del m
    torch.cuda.empty_cache()
    return dict(dt_ratio=ratios, baseline_wi=round(b0, 4),
                implicit_clamp=clamps, control_clamp=ctrls)


def main():
    tr, _ = load_data(1400000, 500, 42, "ultrachat")
    tr_arr = tr.numpy()
    out = {}
    for s in SEEDS:
        ckpt = f"checkpoints/wskan11_ultrachat_100k_3ep_s{s}/latest.pt"
        out[f"s{s}"] = implicit_test(tr_arr, ckpt)
        print(f"s{s}: {out[f's{s}']}", flush=True)
    l0 = [out[f"s{s}"]["dt_ratio"]["L0"] for s in SEEDS]
    l1 = [out[f"s{s}"]["dt_ratio"]["L1"] for s in SEEDS]
    c1 = [out[f"s{s}"]["implicit_clamp"][1] for s in SEEDS]
    k1 = [out[f"s{s}"]["control_clamp"][1] for s in SEEDS]
    out["summary"] = dict(
        dt_ratio_L0_mean=round(float(np.mean(l0)), 3), dt_ratio_L0_std=round(float(np.std(l0)), 3),
        dt_ratio_L1_mean=round(float(np.mean(l1)), 3), dt_ratio_L1_std=round(float(np.std(l1)), 3),
        clamp_L1_mean=round(float(np.mean(c1)), 4), clamp_L1_std=round(float(np.std(c1)), 4),
        ctrl_L1_mean=round(float(np.mean(k1)), 4), ctrl_L1_std=round(float(np.std(k1)), 4),
        per_seed_L1=list(zip([round(v, 3) for v in l1], [round(v, 4) for v in c1], [round(v, 4) for v in k1])),
    )
    print("summary:", json.dumps(out["summary"], indent=2))
    with open("experiments/figures/w12_implicit_boundary_5seed.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved experiments/figures/w12_implicit_boundary_5seed.json")


if __name__ == "__main__":
    main()
