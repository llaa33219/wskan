"""Round-8: why does the landscape enter the organized basin so early?

The open question after round 7: the ~250-step emergence of the
organization is measured, not explained. This battery tests three
parameterization-level drivers:

  1. Gradient anatomy at init: per-parameter-group gradient norms at
     step 0 on real data, and per-mode variance of dL/dlog_sigma - is the
     symmetry between modes broken by the data gradient immediately, and
     does that survive a flat-omega init (no frequency prior)?
  2. Delta-init-scale sensitivity (1k): W_dt bias init -> Delta in
     {0.005, 0.05, 0.5}; does emergence timing depend on the initial
     step scale?
  3. Flat-omega emergence (1k): does the organization emerge as fast when
     all modes start perfectly symmetric (omega=0, sigma uniform, rho=1)?

Outputs: figures/w11_basin_origin.json.
Run: PYTHONPATH=.:experiments .venv/bin/python experiments/W11_basin_origin.py
"""

from __future__ import annotations

import json
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
sys.path.insert(0, "experiments")
from W11_excess_structure import make_model, structure_metrics

DEV = "cuda"


def get_batch(src, bs=32, blk=256, gen=None):
    ix = torch.randint(len(src) - blk - 1, (bs,), generator=gen)
    return torch.stack([src[i:i + blk + 1] for i in ix]).long().to(DEV)


def grad_anatomy(m, batch):
    m.zero_grad()
    lg = m(batch[:, :-1])
    loss = F.cross_entropy(lg.reshape(-1, 256), batch[:, 1:].reshape(-1))
    loss.backward()
    groups = {}
    for name, p in m.named_parameters():
        if p.grad is None:
            continue
        parts = name.split(".")
        if parts[0] == "layers":
            k = f"L{parts[1]}.{parts[2]}"
        elif parts[0] == "prenorms":
            k = "prenorms"
        else:
            k = parts[0]
        groups[k] = groups.get(k, 0.0) + float(p.grad.pow(2).sum())
    return groups, float(loss.item())


def per_mode_grad(m):
    """Per-mode gradient energy of log_sigma and log_rho: is mode symmetry
    broken by the data gradient at init?"""
    out = {}
    for li, layer in enumerate(m.layers):
        gs = layer.log_sigma.grad  # (I, N)
        gr = layer.log_rho.grad    # (N,)
        per_mode_sigma = gs.pow(2).sum(0).cpu().numpy()          # (N,)
        out[f"L{li}"] = dict(
            dlog_sigma_per_mode=np.round(per_mode_sigma, 8).tolist(),
            sigma_mode_cv=float(per_mode_sigma.std() / (per_mode_sigma.mean() + 1e-12)),
            dlog_rho_per_mode=np.round(gr.pow(2).cpu().numpy(), 10).tolist(),
        )
    return out


def short_run(tier_d=4, steps=2000, every=250, dt_init=None, flat_omega=False, arr=None,
              train_ids=None, eval_ids=None):
    d, L = (4, 1) if tier_d == 4 else (40, 2)
    torch.manual_seed(42)
    m = make_model(d, L)
    if dt_init is not None:
        for layer in m.layers:
            layer.W_dt.bias.data.fill_(math.log(math.expm1(dt_init)))
    if flat_omega:
        for layer in m.layers:
            layer.omega.data.zero_()
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.0)
    traj = []
    for step in range(1, steps + 1):
        batch = get_batch(train_ids)
        lg = m(batch[:, :-1])
        loss = F.cross_entropy(lg.reshape(-1, 256), batch[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if step % every == 0:
            m.eval()
            with torch.no_grad():
                met = structure_metrics(m, arr)
            m.train()
            l0 = met["layers"][-1]
            traj.append(dict(step=step, eval=round(float(loss.item()), 3),
                             tsladder=round(l0["timescale_ladder"]["r2"], 3),
                             clock=round(l0.get("clock_space_letter", float("nan")), 3),
                             gate=round(l0.get("gate_split_named_share", float("nan")), 3)))
    return traj


def main():
    from experiments.V1_train_tinystories_lm import load_data
    train_ids, _ = load_data(1400000, 500, 42, "tinystories")
    arr = train_ids[:2048].numpy()
    out = {}

    print("=" * 70)
    print("1. gradient anatomy at init (100k config)")
    batch = get_batch(train_ids)
    for tag, fw in (("pi-harmonic omega", False), ("flat omega", True)):
        torch.manual_seed(42)
        m = make_model(40, 2)
        if fw:
            for layer in m.layers:
                layer.omega.data.zero_()
        m.train()
        groups, loss = grad_anatomy(m, batch)
        pm = per_mode_grad(m)
        out[f"grad_{'flat' if fw else 'pi'}"] = dict(loss=loss, groups=groups, per_mode=pm)
        top = sorted(groups.items(), key=lambda kv: -kv[1])[:8]
        print(f"  [{tag}] loss {loss:.3f} | top grad groups: " +
              ", ".join(f"{k} {v:.4g}" for k, v in top))
        for li, row in pm.items():
            print(f"    {li}: dL/dlog_sigma per-mode CV {row['sigma_mode_cv']:.3f}, "
                  f"per-mode energies {row['dlog_sigma_per_mode']}")

    print("=" * 70)
    print("2. Delta-init-scale sensitivity (1k, 2k steps)")
    for c in (0.005, 0.05, 0.5):
        traj = short_run(dt_init=c, arr=arr, train_ids=train_ids, eval_ids=None)
        out[f"dt_init_{c}"] = traj
        print(f"  Delta0={c}: " + " | ".join(
            f"step {r['step']} clock {r['clock']:.2f} tsl {r['tsladder']:+.2f}" for r in traj[:4]))

    print("=" * 70)
    print("3. flat-omega emergence (1k, 2k steps)")
    traj = short_run(flat_omega=True, arr=arr, train_ids=train_ids, eval_ids=None, steps=2000)
    out["flatomega_emergence"] = traj
    print("  flat-omega: " + " | ".join(
        f"step {r['step']} clock {r['clock']:.2f} tsl {r['tsladder']:+.2f}" for r in traj[:4]))

    with open("experiments/figures/w11_basin_origin.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("saved experiments/figures/w11_basin_origin.json")


if __name__ == "__main__":
    main()
