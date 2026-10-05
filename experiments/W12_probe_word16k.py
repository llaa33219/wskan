"""W12 word-identity probe at 16k positions (review round 5, 2026-10-04).

The round-1 null-calibrated battery used 4,096 positions; a reviewer asked
whether the held-out ridge word-identity R2 (<=0.02) measures the
representation or probe overfitting at 500 classes / 4k positions. This
reruns the word-identity row at 16,384 positions on the two tiers that
matter (100k canonical, 10m ceiling): in-sample R2 + shuffled null, and
the held-out ridge sweep.

Outputs: figures/w12_probe_word16k.json
Run: PYTHONPATH=.:experiments .venv/bin/python experiments/W12_probe_word16k.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "experiments")
from W11_excess_structure import load_model
from W11_round5_probes import word_labels
from W12_probe_nulls import make_labels, insample_with_null, ridge_sweep

DEV = "cuda"
TIERS = {"100k": (40, 2), "10m": (512, 2)}


def extract_resb(m, arr, chunk=4096):
    outs = []
    with torch.no_grad():
        for s in range(0, len(arr), chunk):
            idx = torch.tensor(arr[s:s + chunk].astype(np.int64), device=DEV).unsqueeze(0)
            x = m.tok_emb(idx)
            for li, (norm, layer) in enumerate(zip(m.prenorms, m.layers)):
                h = norm(x)
                if li == len(m.layers) - 1:
                    outs.append(layer.res_B(h)[0].detach().cpu().numpy())
                    break
                x = x + layer(h, idx)
    return np.concatenate(outs, 0)


def main():
    from experiments.V1_train_tinystories_lm import load_data
    tr, _ = load_data(1400000, 500, 42, "ultrachat")
    arr = tr[:16384].numpy()
    out = {}
    for tier, (d, L) in TIERS.items():
        print(f"== {tier}", flush=True)
        m = load_model(f"checkpoints/wskan11_ultrachat_{tier}_3ep_s42/latest.pt", d, L)
        X = extract_resb(m, arr).reshape(len(arr), -1)
        labels = make_labels(arr)
        y, ncls = labels["word500"]
        r2, null = insample_with_null(X, y, ncls)
        best = ridge_sweep(X, y, ncls)
        out[tier] = dict(n_positions=len(arr), n_classes=ncls,
                         insample_r2=round(r2, 4), shuffled_null=round(null, 4),
                         ridge={k: (round(v, 4) if isinstance(v, float) else v)
                                for k, v in best.items()})
        print(f"   word500@16k: in-sample {r2:.3f} (null {null:.3f}) | "
              f"held-out ridge R2 {best['r2']:.3f} acc {best['acc']:.3f} "
              f"(null {best['null_acc']:.3f}, alpha {best['alpha']})", flush=True)
        del m
        torch.cuda.empty_cache()
    with open("experiments/figures/w12_probe_word16k.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved experiments/figures/w12_probe_word16k.json")


if __name__ == "__main__":
    main()
