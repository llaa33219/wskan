"""Extended training dynamics (20k steps): does the gate-split equilibrium
plateau, and when does the clock emerge at 1k? Extends the 3k-step window
of W11_excess_structure.py part E (round-7 point 2).

Run: .venv/bin/python experiments/W11_dynamics_20k.py --tier {1k|100k}
"""

from __future__ import annotations

import argparse
import json
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
sys.path.insert(0, "experiments")
from W11_excess_structure import make_model, structure_metrics, SIZE


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tier", choices=["1k", "100k"], required=True)
    p.add_argument("--steps", type=int, default=20000)
    args = p.parse_args()

    from experiments.V1_train_tinystories_lm import load_data
    d, L = SIZE[args.tier]
    ds = "tinystories" if args.tier == "1k" else "ultrachat"
    train_ids, eval_ids = load_data(1400000, 500, 42, ds)
    eval_ids = eval_ids.long().cuda()
    torch.manual_seed(42)
    m = make_model(d, L)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    arr = train_ids[:2048].numpy()

    def get_batch(bs=32, blk=256):
        ix = torch.randint(len(train_ids) - blk - 1, (bs,))
        return torch.stack([train_ids[i:i + blk + 1] for i in ix]).long().cuda()

    traj = []
    for step in range(1, args.steps + 1):
        batch = get_batch()
        lg = m(batch[:, :-1])
        loss = F.cross_entropy(lg.reshape(-1, 256), batch[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == 1:
            m.eval()
            with torch.no_grad():
                lg2 = m(eval_ids[:513].unsqueeze(0)[:, :-1])
                ev = F.cross_entropy(lg2.reshape(-1, 256), eval_ids[1:513].unsqueeze(0).reshape(-1)).item()
                met = structure_metrics(m, arr)
            m.train()
            row = dict(step=step, eval=round(float(ev), 4),
                       layers=[{k: v for k, v in lay.items() if k != "antipodal"} | {
                           "antipodal": lay["antipodal"]["mean_min_cos"]} for lay in met["layers"]])
            traj.append(row)
            l0 = met["layers"][-1]
            print(f"step {step:6d} eval {ev:.3f} | tsladder {l0['timescale_ladder']['r2']:+.2f} | "
                  f"gate {l0.get('gate_split_named_share', float('nan')):.3f} | "
                  f"clock {l0.get('clock_space_letter', float('nan')):.2f}", flush=True)

    out = f"experiments/figures/w11_dynamics20k_{args.tier}.json"
    with open(out, "w") as f:
        json.dump(traj, f, indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
