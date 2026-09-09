"""Replay the deterministic wskan3real/seed-123 NaN with anomaly detection.

Reproduces RNG state at step 50000 (seed -> model init -> 50k get_batch calls
with the same eval cadence), loads the pre-NaN checkpoint, then continues
training under torch.autograd.detect_anomaly until the NaN recurs.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.V1_train_tinystories_lm import load_data, get_batch
from models.V3_WSKAN import WaveletStateKANLMV3

SEED, RESUME_STEP, NAN_STEP = 123, 50000, 54194
device = "cuda"

torch.manual_seed(SEED)
train_ids, _ = load_data(200000, 1000, SEED)
model = WaveletStateKANLMV3(oscillatory=False).to(device)  # consumes init RNG as in the run

# replay RNG consumption of steps 1..50000 (1 train batch/step + eval every 2500)
for step in range(1, RESUME_STEP + 1):
    _ = torch.randint(0, len(train_ids) - 257, (32,))
    if step % 2500 == 0:
        _ = torch.randint(0, 793374 - 257, (32,))

ckpt = torch.load("checkpoints/wskan3real_tinystories_lm_s123/ckpt_step50000.pt", weights_only=False)
model.load_state_dict(ckpt["state_dict"])
print(f"resumed from step {RESUME_STEP}, replaying with anomaly detection...")

opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100000)
for _ in range(RESUME_STEP):
    sched.step()

model.train()
caught = {}

def hook(name):
    def fn(mod, inp, out):
        if name in caught:
            return
        tensors = out if isinstance(out, tuple) else (out,)
        if any(isinstance(t, torch.Tensor) and not torch.isfinite(t).all() for t in tensors):
            caught[name] = [t.detach() for t in inp if isinstance(t, torch.Tensor)]
            print(f"FIRST NON-FINITE OUTPUT: {name} ({mod.__class__.__name__})")
            for i, t in enumerate(caught[name]):
                print(f"  input[{i}] absmax={t.abs().max().item():.4g} shape={tuple(t.shape)}")
    return fn

for n, mod in model.named_modules():
    if not list(mod.children()):
        mod.register_forward_hook(hook(n))

with torch.autograd.detect_anomaly():
    for step in range(RESUME_STEP + 1, NAN_STEP + 2):
        batch = get_batch(train_ids, 32, 256, device)
        loss = model.loss(batch)
        if not torch.isfinite(loss):
            print(f"NaN/Inf loss at step {step}")
            torch.save(batch.cpu(), "/tmp/opencode/nan_batch.pt")
            for n, p in model.named_parameters():
                if not torch.isfinite(p).all():
                    print(f"  non-finite param: {n}")
            break
        opt.zero_grad()
        loss.backward()
        bad = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        if bad:
            print(f"step {step}: non-finite grads in {bad}")
            torch.save(batch.cpu(), "/tmp/opencode/nan_batch.pt")
            break
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0:
            print(f"step {step} loss {loss.item():.4f}")
