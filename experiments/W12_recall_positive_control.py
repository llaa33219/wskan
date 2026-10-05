"""W12 recall-probe positive control (review round 5, 2026-10-04).

The memorize-then-recall probe fails on both WSKAN and matched Mamba-2 -
but a probe that fails on every model proves nothing. Positive control:
an induction (in-context copying) task that models at this scale should
pass if the measurement has power to detect recall-type behavior at all.

  A. induction: K(4 random bytes) + 32 random filler + K[:3] -> predict
     K[3]. Passing = the pipeline can detect copying when it exists.
  B. long-range memorize-recall (the original null): "the code is " + K
     (6 bytes) + 300 bytes of eval text + "the code is " -> teacher-forced
     per-byte accuracy on K.

Models: wskan11 100k s42, mamba2 100k s42 (200 trials each).
Outputs: figures/w12_recall_positive_control.json
Run: PYTHONPATH=.:experiments .venv/bin/python experiments/W12_recall_positive_control.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "experiments")
from W11_excess_structure import load_model
from experiments.V1_train_tinystories_lm import Mamba2ByteLM, load_data

DEV = "cuda"
PRINTABLE = list(range(97, 123))


def wskan_next(m, ctx):
    with torch.no_grad():
        return int(m(torch.tensor(ctx, device=DEV).unsqueeze(0))[0, -1].argmax())


def mamba_next(m, ctx):
    with torch.no_grad():
        return int(m.model(input_ids=torch.tensor(ctx, device=DEV).unsqueeze(0)).logits[0, -1].argmax())


def run(nextfn, ev_arr, n=200, seed=0):
    rng = np.random.default_rng(seed)
    ind_hit, mem_hits = 0, []
    for _ in range(n):
        K4 = [int(rng.choice(PRINTABLE)) for _ in range(4)]
        filler = [int(rng.integers(0, 256)) for _ in range(32)]
        if wskan_next is nextfn or True:
            pred = nextfn(K4 + filler + K4[:3])
        ind_hit += int(pred == K4[3])
        K6 = [int(rng.choice(PRINTABLE)) for _ in range(6)]
        s = int(rng.integers(0, len(ev_arr) - 400))
        story = [int(c) for c in ev_arr[s:s + 300]]
        pre = list(b"the code is ")
        ctx = pre + K6 + story + pre
        hits = 0
        for k in K6:
            p = nextfn(ctx)
            hits += int(p == k)
            ctx.append(k)
        mem_hits.append(hits / 6)
    return dict(induction_acc=round(ind_hit / n, 3),
                memorize_recall_byte_acc=round(float(np.mean(mem_hits)), 3))


def main():
    _, ev = load_data(1400000, 500, 42, "ultrachat")
    ev_arr = ev.numpy()
    out = {}
    mw = load_model("checkpoints/wskan11_ultrachat_100k_3ep_s42/latest.pt", 40, 2)
    out["wskan11_100k"] = run(lambda c: wskan_next(mw, c), ev_arr)
    print("wskan11:", out["wskan11_100k"], flush=True)
    del mw
    torch.cuda.empty_cache()
    mm = Mamba2ByteLM(scale="100k").to(DEV).eval()
    sd = torch.load("checkpoints/mamba2_ultrachat_100k_3ep_s42/latest.pt", weights_only=False)
    mm.load_state_dict(sd["state_dict"] if "state_dict" in sd else sd)
    out["mamba2_100k"] = run(lambda c: mamba_next(mm, c), ev_arr, seed=1)
    print("mamba2:", out["mamba2_100k"], flush=True)
    with open("experiments/figures/w12_recall_positive_control.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved experiments/figures/w12_recall_positive_control.json")


if __name__ == "__main__":
    main()
