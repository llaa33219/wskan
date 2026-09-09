"""Full causal battery for the word clock: necessity replication + sufficiency.

Conditions (per seed, 64x512 held-out UltraChat blocks, teacher-forced CE):
  baseline     - untouched
  clamp-bnd    - Delta at boundaries := letter-mean        (necessity)
  clamp-let    - Delta at count-matched letters := letter-mean (control)
  inject-tick  - Delta at count-matched letters := boundary-mean (sufficiency:
                 false word boundaries injected mid-word)

Metrics: overall CE; CE right after the intervened positions; CE at true
word-initial positions. Predictions if the clock story is right:
  clamp-bnd: overall up, word-initial up strongly
  inject-tick: overall up, CE after injected ticks jumps toward word-initial
  levels (model treats false ticks as word boundaries)

Run: .venv/bin/python experiments/V6_causal_clock_full.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.V1_train_tinystories_lm import get_batch, load_data
from models.V6_WSKAN import WaveletStateKANLMV6

device = "cuda"
torch.manual_seed(0)

_, eval_ids = load_data(200000, 500, 42, "ultrachat")
B, L = 64, 512
blocks = [get_batch(eval_ids, 1, L, device) for _ in range(B)]
batch = torch.cat(blocks, 0)
b, targets = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()

BOUNDARY = torch.zeros(256, dtype=torch.bool)
for c in b" \n.,!?;:'\"()-":
    BOUNDARY[c] = True
LOWER = torch.tensor([(97 <= c <= 122) for c in range(256)])


def run_seed(seed):
    model = WaveletStateKANLMV6()
    model.load_state_dict(
        torch.load(f"checkpoints/wskan6_ultrachat_100k_s{seed}/latest.pt", weights_only=False)["state_dict"]
    )
    model.eval().cuda()

    orig = {i: layer._compute_dt for i, layer in enumerate(model.layers)}

    def make_patched(li):
        def patched(x):
            dt = orig[li](x)
            cfg = getattr(model.layers[li], "_intervention", None)
            if cfg is None:
                return dt
            kind, mask = cfg
            Lx = x.shape[1]
            m = mask[:, :Lx]
            pos = LOWER.to(device)[b[:, :Lx]]
            if kind == "clamp":
                src = pos
            else:  # inject: replacement value = boundary-mean
                src = BOUNDARY.to(device)[b[:, :Lx]]
            val = (dt * src.unsqueeze(-1)).sum(1) / src.sum(1, keepdim=True).clamp(min=1)
            mf = m.unsqueeze(-1).float()
            dt = dt * (1 - mf) + val.unsqueeze(1) * mf
            return dt
        return patched

    for i, layer in enumerate(model.layers):
        layer._compute_dt = make_patched(i)

    def masks_for(kind):
        boundary_pos = BOUNDARY.to(device)[b]
        letter_pos = LOWER.to(device)[b]
        k = int(boundary_pos.float().sum())
        g = torch.Generator(device="cpu").manual_seed(7)
        scores = torch.rand(b.shape, generator=g).to(device)
        if kind == "clamp-let":
            sc = scores.clone(); sc[~letter_pos] = -1.0
        else:  # inject-tick (letters only - false ticks must be mid-word)
            sc = scores.clone(); sc[~letter_pos] = -1.0
        thresh = sc.flatten().kthvalue(sc.numel() - k + 1).values
        return sc >= thresh

    def evaluate(cfgs):
        for i, layer in enumerate(model.layers):
            layer._intervention = cfgs
        with torch.no_grad():
            logits = model(b)
            ce = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 256), targets.reshape(-1), reduction="none"
            ).view(B, L)
        return ce

    out = {}
    # baseline
    ce = evaluate(None)
    out["baseline"] = (ce.mean().item(),)
    wi = (b == 32)
    out["baseline_wi"] = (ce[wi].mean().item(),)

    for kind in ("clamp-bnd", "clamp-let", "inject-tick"):
        mask = BOUNDARY.to(device)[b] if kind == "clamp-bnd" else masks_for(kind)
        ce = evaluate((kind.split("-")[0], mask))
        after = torch.zeros_like(mask)
        after[:, :-1] = mask[:, 1:]  # CE at positions right after an intervened byte
        out[kind] = (ce.mean().item(), ce[after].mean().item(), ce[wi].mean().item())
    return out


print(f"{'seed':>5} {'baseline':>9} | {'clamp-bnd':>22} | {'clamp-let':>22} | {'inject-tick':>24}")
for seed in (42, 123, 2024):
    r = run_seed(seed)
    cb = r["clamp-bnd"]; cl = r["clamp-let"]; it = r["inject-tick"]
    print(f"{seed:>5} {r['baseline'][0]:>9.4f} | {cb[0]:>8.4f} after:{cb[1]:>7.4f} wi:{cb[2]:>7.4f} "
          f"| {cl[0]:>8.4f} after:{cl[1]:>7.4f} wi:{cl[2]:>7.4f} "
          f"| {it[0]:>8.4f} after:{it[1]:>7.4f} wi:{it[2]:>7.4f}")
print("\nreference: baseline word-initial CE =", f"{r['baseline_wi'][0]:.4f}")
