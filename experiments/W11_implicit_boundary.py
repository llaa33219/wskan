"""Implicit-boundary clock test (round-35): is the clock byte-keyed or
partially linguistic? Space-free text (spaces removed, true boundaries
recorded): measure Delta at implicit boundaries vs letter-internal, and
clamp Delta at implicit boundaries to test word-initial CE damage.

Also: the 5-seed boundary-clamp damage battery at 100k (L0 clamp),
strengthening the headline causal ratio.

Outputs: figures/w11_implicit_boundary.json
Run: PYTHONPATH=.:experiments .venv/bin/python experiments/W11_implicit_boundary.py
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
BOUNDARY = torch.tensor([32, 10, 46, 33, 63, 44, 59, 58], device=DEV)
LOWER = torch.arange(97, 123, device=DEV)


def five_seed_clamp(ids):
    def wi_ce(m, clamp=False):
        orig = None
        if clamp:
            orig = m.layers[0]._compute_dt
            def patched(x):
                dt = orig(x)
                idx_flat = ids[: x.shape[1]]
                mask = torch.isin(idx_flat, BOUNDARY)[None, :, None].float()
                letters = torch.isin(idx_flat, LOWER)[None, :, None].float()
                lm = (dt * letters).sum(1, keepdim=True) / letters.sum(1, keepdim=True).clamp(min=1)
                return dt * (1 - mask) + lm * mask
            m.layers[0]._compute_dt = patched
        ces, wis = [], []
        with torch.no_grad():
            for s in range(0, len(ids) - 512, 512):
                ch = ids[s:s + 513]
                lg = m(ch[:-1].unsqueeze(0))[0]
                ce = F.cross_entropy(lg, ch[1:], reduction="none")
                ces.append(ce)
                wis.append(ch[:-1] == 32)
        if orig is not None:
            m.layers[0]._compute_dt = orig
        ce = torch.cat(ces)
        return float(ce[torch.cat(wis)].mean())
    deltas = []
    for s in (42, 7, 123, 2024, 31337):
        m = load_model(f"checkpoints/wskan11_ultrachat_100k_3ep_s{s}/latest.pt", 40, 2)
        deltas.append(wi_ce(m, clamp=True) - wi_ce(m, clamp=False))
    return deltas


def implicit_test(tr_arr):
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
    m = load_model("checkpoints/wskan11_ultrachat_100k_3ep_s42/latest.pt", 40, 2)
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
        if clamp_positions is not None:
            def patched(x):
                dt = orig(x)
                L = x.shape[1]
                mask = torch.zeros(1, L, 1, device=DEV)
                mask[0, clamp_positions[clamp_positions < L], 0] = 1.0
                lm = dt.mean(1, keepdim=True)
                return dt * (1 - mask) + lm * mask
            m.layers[li]._compute_dt = patched
        with torch.no_grad():
            for s in range(0, 20000, 512):
                ch = torch.tensor(list(sf[s:s + 513]), device=DEV).unsqueeze(0)
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
    return dict(dt_ratio=ratios, baseline_wi=round(b0, 4),
                implicit_clamp=clamps, control_clamp=ctrls)


def main():
    _, ev = load_data(1400000, 500, 42, "ultrachat")
    ids = ev[:60000].long().to(DEV)
    deltas = five_seed_clamp(ids)
    print(f"5-seed boundary-clamp damage: {np.mean(deltas):.4f} +- {np.std(deltas):.4f} "
          f"(per seed: {[round(d, 4) for d in deltas]})")
    tr, _ = load_data(1400000, 500, 42, "ultrachat")
    imp = implicit_test(tr.numpy())
    print(json.dumps(imp, indent=2))
    out = dict(five_seed_clamp=dict(mean=round(float(np.mean(deltas)), 4),
                                    std=round(float(np.std(deltas)), 4),
                                    per_seed=[round(d, 4) for d in deltas]),
               implicit_boundary=imp)
    with open("experiments/figures/w11_implicit_boundary.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved experiments/figures/w11_implicit_boundary.json")


if __name__ == "__main__":
    main()
