"""W12 learning-curve analysis of the retrain battery (review response, 2026-10-04).

Point-5 experiment, part 1. v2: same-surface evaluation. v1 compared
ablation-harness logs (16-chunk eval surface) against campaign train logs
(a different, larger eval surface) - a cross-surface comparison that
fabricated a gap. This version rebuilds the BASE curve on the ablation eval
surface by evaluating the campaign intermediate checkpoints (25k..108k)
with the identical 16-chunk protocol used by W11_ablation_retrain.py.

Per variant (nofeat, flatomega; seeds 42/7/123; ultrachat 100k, 108k steps):
  - step-matched gap Delta(step) = CE_variant(step) - CE_base(step)
  - data overhead: for each baseline reference point t_b, the first step
    t_v where the variant reaches CE_base(t_b); median ratio t_v / t_b

Outputs: figures/w12_learning_curves.json
Run: PYTHONPATH=.:experiments .venv/bin/python experiments/W12_learning_curve.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
sys.path.insert(0, "experiments")
from W11_excess_structure import load_model
from experiments.V1_train_tinystories_lm import load_data

DEV = "cuda"
SEEDS = (42, 7, 123)
VARIANTS = ("nofeat", "flatomega")
CKPT_STEPS = [25000, 50000, 75000, 100000, 108000]


def base_curve_same_surface(seed):
    _, ev = load_data(1400000, 500, seed, "ultrachat")
    eval_ids = ev.long().to(DEV)
    ev_chunks = [eval_ids[s:s + 513] for s in range(0, 16 * 512, 512)]
    curve = {}
    m = None
    for step in CKPT_STEPS:
        p = f"checkpoints/wskan11_ultrachat_100k_3ep_s{seed}/ckpt_step{step}.pt"
        if not Path(p).exists():
            p = f"checkpoints/wskan11_ultrachat_100k_3ep_s{seed}/latest.pt"
        m = load_model(p, 40, 2)
        with torch.no_grad():
            ce = float(np.mean([F.cross_entropy(m(c[:-1].unsqueeze(0))[0], c[1:]).item()
                                for c in ev_chunks]))
        curve[step] = ce
        print(f"  base s{seed} step {step}: {ce:.4f}", flush=True)
    del m
    torch.cuda.empty_cache()
    return curve


def load_variant(variant, seed):
    r = json.loads(Path(f"checkpoints/wskan11abl-{variant}_ultrachat_100k_3ep_s{seed}/result.json").read_text())
    return {e["step"]: e["eval"] for e in r["log"]}


def at(curve, step):
    k = min(curve, key=lambda k: abs(k - step))
    return curve[k]


def first_reach(curve, target):
    for k in sorted(curve):
        if curve[k] <= target:
            return k
    return None


def main():
    base = {s: base_curve_same_surface(s) for s in SEEDS}
    out = {"ckpt_steps": CKPT_STEPS, "base_curve": {str(s): base[s] for s in SEEDS}, "variants": {}}
    for v in VARIANTS:
        gaps = {t: [] for t in CKPT_STEPS}
        ratios = []
        for s in SEEDS:
            var = load_variant(v, s)
            for t in CKPT_STEPS:
                gaps[t].append(at(var, t) - base[s][t])
            for t_b in CKPT_STEPS[:-1]:
                t_v = first_reach(var, base[s][t_b])
                if t_v is not None:
                    ratios.append(t_v / t_b)
        out["variants"][v] = dict(
            gap_mean={str(t): round(float(np.mean(gaps[t])), 4) for t in CKPT_STEPS},
            gap_std={str(t): round(float(np.std(gaps[t])), 4) for t in CKPT_STEPS},
            data_overhead_median_ratio=round(float(np.median(ratios)), 3) if ratios else None,
            area_nat_ksteps=round(float(np.trapezoid(
                [np.mean(gaps[t]) for t in CKPT_STEPS], CKPT_STEPS)) / 1000, 3),
        )
        print(v, json.dumps(out["variants"][v], indent=2), flush=True)
    with open("experiments/figures/w12_learning_curves.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved experiments/figures/w12_learning_curves.json")


if __name__ == "__main__":
    main()
