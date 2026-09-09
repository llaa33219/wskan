"""Causal test of the word-clock finding (V6_LANGUAGE_LEARNING_ANALYSIS.md).

Question: is the 2-3x elevated Delta at word boundaries causally used, or a
correlational byproduct?

Conditions (teacher-forced CE on held-out UltraChat):
  baseline      - normal model
  boundary      - Delta at space/newline/punct positions replaced by the
                  per-sample, per-channel mean Delta of lowercase positions
  random-ctrl   - same replacement value, applied at a random position set of
                  the same size (controls for "any perturbation hurts")

If boundary >> random-ctrl >= baseline, the *positions* of the fast clock are
causally load-bearing. Also reports CE on bytes immediately after a space
(word-initial bytes) where the effect should concentrate.

Run: .venv/bin/python experiments/V6_causal_clock_test.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.V1_train_tinystories_lm import get_batch, load_data
from models.V6_WSKAN import WaveletStateKANLMV6

torch.manual_seed(0)
device = "cuda"
model = WaveletStateKANLMV6()
model.load_state_dict(torch.load("checkpoints/wskan6_ultrachat_100k_s42/latest.pt", weights_only=False)["state_dict"])
model.eval().cuda()

_, eval_ids = load_data(200000, 500, 42, "ultrachat")
B, L = 64, 512
blocks = [get_batch(eval_ids, 1, L, device) for _ in range(B)]
batch = torch.cat(blocks, 0)  # (B, L+1)
b = batch[:, :-1].contiguous()
targets = batch[:, 1:].contiguous()

BOUNDARY = torch.zeros(256, dtype=torch.bool)
for c in b" \n.,!?;:'\"()-":
    BOUNDARY[c] = True
AFTER_SPACE = torch.zeros(256, dtype=torch.bool)
AFTER_SPACE[32] = True

orig_dt = {}
for li, layer in enumerate(model.layers):
    orig_dt[li] = layer._compute_dt

    def make_patched(li):
        def patched(x):
            dt = orig_dt[li](x)
            repl_mask = getattr(model.layers[li], "_repl_mask", None)
            if repl_mask is not None:
                letter = torch.tensor(
                    [(97 <= c <= 122) or (65 <= c <= 90) for c in range(256)],
                    device=x.device,
                )
                letter_pos = letter[b[:, : x.shape[1]]]  # (B, Lx)
                letter_mean = (
                    dt.permute(0, 2, 1)[:, :, letter_pos[0]].mean(-1).unsqueeze(-1)
                    if False else
                    (dt * letter_pos.unsqueeze(-1)).sum(1) / letter_pos.sum(1, keepdim=True).clamp(min=1)
                )  # (B, i) per-sample letter mean
                m = repl_mask[:, : x.shape[1]].unsqueeze(-1).float()
                dt = dt * (1 - m) + letter_mean.unsqueeze(1) * m
            return dt
        return patched

    model.layers[li]._compute_dt = make_patched(li)


def set_masks(condition):
    with torch.no_grad():
        boundary_pos = BOUNDARY.to(device)[b]
        if condition == "baseline":
            masks = [None] * len(model.layers)
        elif condition == "boundary":
            masks = [boundary_pos] * len(model.layers)
        elif condition == "letter":
            letter_pos = torch.tensor(
                [(97 <= c <= 122) for c in range(256)], device=device
            )[b]
            masks = []
            g = torch.Generator(device="cpu").manual_seed(7)
            for _ in range(len(model.layers)):
                # same count as boundary, drawn from lowercase positions only
                scores = torch.rand(b.shape, generator=g).to(device)
                scores[~letter_pos] = -1.0
                k = int(boundary_pos.float().sum())
                thresh = scores.flatten().kthvalue(scores.numel() - k + 1).values
                masks.append(scores >= thresh)
        else:  # random-ctrl: same count per sample
            masks = []
            g = torch.Generator(device="cpu").manual_seed(7)
            for _ in range(len(model.layers)):
                m = torch.rand(b.shape, generator=g).to(device) < (boundary_pos.float().mean())
                masks.append(m)
    for layer, m in zip(model.layers, masks):
        layer._repl_mask = m


@torch.no_grad()
def evaluate(condition):
    set_masks(condition)
    logits = model(b)
    ce = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 256), targets.reshape(-1), reduction="none"
    ).view(B, L)
    # word-initial: current input byte is a space -> CE of the first byte after it
    wi = (b == 32)
    return ce.mean().item(), ce[wi].mean().item(), ce[~wi].mean().item()


print(f"{'condition':<14} {'overall CE':>10} {'CE@after-space':>15} {'CE@other':>10}")
res = {}
for cond in ("baseline", "boundary", "letter", "random-ctrl"):
    overall, wi, other = evaluate(cond)
    res[cond] = (overall, wi, other)
    print(f"{cond:<14} {overall:>10.4f} {wi:>15.4f} {other:>10.4f}")

for c in ("boundary", "letter", "random-ctrl"):
    d = res[c][0] - res["baseline"][0]
    print(f"delta overall {c:>12}-clamp: {d:+.4f}")
print(f"delta after-space  boundary: {res['boundary'][1]-res['baseline'][1]:+.4f} | letter: {res['letter'][1]-res['baseline'][1]:+.4f} | random: {res['random-ctrl'][1]-res['baseline'][1]:+.4f}")
