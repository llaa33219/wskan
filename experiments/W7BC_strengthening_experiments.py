"""Strengthening experiments for W7BC_HOW_IT_SPEAKS.md.

  E1. Whitespace removal: does Delta still pulse at implicit word boundaries
      when spaces are deleted from the input? (key test of "invented
      segmentation")
  E2. Delta distributions per byte class (quantiles, not just means)
  E3. Decompose the boundary-clamp cost: direct (at clamped positions) vs
      word-initial vs everything else
  E4. Exclusion battery for the unexplained residual-B write variance:
      word identity, prev-word identity, sentence position, bigram, doc position

Output: figures/v7e_strengthening.json + fig_w7bc_exp*.png
Run: .venv/bin/python experiments/W7BC_strengthening_experiments.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.V1_train_tinystories_lm import get_batch, load_data
from models.V7_WSKAN import WaveletStateKANLMV7

device = "cuda"
torch.manual_seed(0)
OUT = Path(__file__).resolve().parent / "figures"
S = {}

model = WaveletStateKANLMV7(use_feature_bc=True, wz_diag=False, g_rank=None, bc_rank=32)
model.load_state_dict(torch.load("checkpoints/wskan7bc_ultrachat_interp_s42/latest.pt",
                                 weights_only=False)["state_dict"])
model.eval().cuda()

train_ids, eval_ids = load_data(200000, 500, 42, "ultrachat")
probe = train_ids[:2048].numpy().astype(np.int64)


def dts_for(text_ids: torch.Tensor):
    idx = text_ids.to(device).unsqueeze(0)
    out = []
    with torch.no_grad():
        x = model.tok_emb(idx)
        for norm, layer in zip(model.prenorms, model.layers):
            h = norm(x)
            out.append(layer._compute_dt(h)[0].cpu())
            x = x + layer(h, idx)
    return out


def ce_of(text_ids: torch.Tensor) -> float:
    idx = text_ids.to(device).unsqueeze(0)
    with torch.no_grad():
        logits = model(idx)
        return F.cross_entropy(logits[0, :-1].reshape(-1, 256),
                               text_ids[1:].reshape(-1).to(device)).item()


# ---------------- E1. whitespace removal ----------------
print("=== E1 ===")
stripped = probe[probe != 32]
# implicit boundary: position i in stripped text corresponds to a word start
# in the original (the byte right before it was a space in the original)
j, bound_mask = 0, np.zeros(len(stripped), dtype=bool)
for i, c in enumerate(probe):
    if c != 32:
        if i > 0 and probe[i - 1] == 32:
            bound_mask[j] = True
        j += 1
dts_a = dts_for(torch.tensor(probe))
dts_b = dts_for(torch.tensor(stripped))
S["E1_ce_with_spaces"] = ce_of(torch.tensor(probe))
S["E1_ce_no_spaces"] = ce_of(torch.tensor(stripped))
internal = ~bound_mask
for li in range(3):
    db = dts_b[li]
    S[f"E1_L{li}_dt_implicit_boundary"] = float(db[bound_mask].mean())
    S[f"E1_L{li}_dt_internal"] = float(db[internal].mean())
    S[f"E1_L{li}_ratio"] = float(db[bound_mask].mean() / db[internal].mean())
print({k: round(v, 4) for k, v in S.items() if k.startswith("E1")})

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for li in range(3):
    axes[li].hist(dts_b[li][internal].numpy(), bins=50, alpha=0.6, density=True, label="word-internal")
    axes[li].hist(dts_b[li][bound_mask].numpy(), bins=50, alpha=0.6, density=True, label="implicit boundary")
    axes[li].set_title(f"L{li}: Delta without spaces in input")
    axes[li].legend(fontsize=7)
fig.suptitle("E1: does the clock still see word boundaries when spaces are removed?")
fig.tight_layout()
fig.savefig(OUT / "fig_w7bc_exp1_nospace.png", dpi=130)
plt.close(fig)

# ---------------- E2. Delta distributions ----------------
CLASSES = {"space": lambda t: t == 32, "newline": lambda t: t == 10,
           "punct": lambda t: np.isin(t, list(b".,!?;:'\"()-")),
           "upper": lambda t: (t >= 65) & (t <= 90),
           "lower": lambda t: (t >= 97) & (t <= 122),
           "digit": lambda t: (t >= 48) & (t <= 57)}
print("=== E2 ===")
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for li in range(3):
    for row, (cname, pred) in enumerate(CLASSES.items()):
        if row > 2:
            continue
    for cname, pred in list(CLASSES.items()):
        m = pred(probe)
        q = np.quantile(dts_a[li][m].numpy(), [0.1, 0.25, 0.5, 0.75, 0.9])
        S.setdefault("E2_dt_quantiles", {}).setdefault(cname, {})[f"L{li}"] = [round(float(v), 3) for v in q]
        S.setdefault("E2_dt_std", {}).setdefault(cname, {})[f"L{li}"] = round(float(dts_a[li][m].std()), 4)
    for cname, pred in CLASSES.items():
        m = pred(probe)
        axes[0][li].hist(dts_a[li][m].numpy(), bins=40, alpha=0.5, density=True, label=cname)
    axes[0][li].legend(fontsize=6); axes[0][li].set_title(f"L{li} Delta histogram by class")
# boxplot per class across layers
for ci, (cname, pred) in enumerate(CLASSES.items()):
    m = pred(probe)
    axes[1][0].boxplot([dts_a[0][m].mean(-1).numpy()], positions=[ci], widths=0.6, showfliers=False)
    axes[1][1].boxplot([dts_a[1][m].mean(-1).numpy()], positions=[ci], widths=0.6, showfliers=False)
    axes[1][2].boxplot([dts_a[2][m].mean(-1).numpy()], positions=[ci], widths=0.6, showfliers=False)
for li in range(3):
    axes[1][li].set_xticks(range(6), list(CLASSES.keys()), rotation=30, fontsize=7)
    axes[1][li].set_title(f"L{li} Delta distributions")
fig.tight_layout()
fig.savefig(OUT / "fig_w7bc_exp2_dt_dist.png", dpi=130)
plt.close(fig)

# ---------------- E3. decompose the clamp cost ----------------
print("=== E3 ===")
B, L = 32, 512
batch = torch.cat([get_batch(eval_ids, 1, L, device) for _ in range(B)], 0)
b, targets = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
BOUNDARY = torch.zeros(256, dtype=torch.bool)
for c in b" \n.,!?;:'\"()-":
    BOUNDARY[c] = True
LOWER = torch.tensor([(97 <= c <= 122) for c in range(256)])
orig = {i: layer._compute_dt for i, layer in enumerate(model.layers)}
for li, layer in enumerate(model.layers):
    def mk(li):
        def p(x):
            dt = orig[li](x)
            m = getattr(model.layers[li], "_ivm", None)
            if m is None:
                return dt
            Lx = x.shape[1]
            mm = m[:, :Lx]
            pos = LOWER.to(device)[b[:, :Lx]]
            val = (dt * pos.unsqueeze(-1)).sum(1) / pos.sum(1, keepdim=True).clamp(min=1)
            mf = mm.unsqueeze(-1).float()
            return dt * (1 - mf) + val.unsqueeze(1) * mf
        return p
    model.layers[li]._compute_dt = mk(li)

@torch.no_grad()
def ce_per_pos(clamped: bool):
    for li, layer in enumerate(model.layers):
        layer._ivm = BOUNDARY.to(device)[b] if clamped else None
    logits = model(b)
    return F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1),
                           reduction="none").view(B, L)

ce_base = ce_per_pos(False)
ce_clamp = ce_per_pos(True)
d = (ce_clamp - ce_base)
at_bnd = BOUNDARY.to(device)[b]
after_bnd = torch.zeros_like(at_bnd)
after_bnd[:, 1:] = at_bnd[:, :-1]
other = ~(at_bnd | after_bnd)
tot = float(d.sum())
S["E3_total_delta_CE"] = tot
S["E3_at_boundary_share"] = float(d[at_bnd].sum() / tot)
S["E3_word_initial_share"] = float(d[after_bnd].sum() / tot)
S["E3_elsewhere_share"] = float(d[other].sum() / tot)
S["E3_per_pos_mean_at_boundary"] = float(d[at_bnd].mean())
S["E3_per_pos_mean_word_initial"] = float(d[after_bnd].mean())
S["E3_per_pos_mean_elsewhere"] = float(d[other].mean())
for layer in model.layers:
    layer._ivm = None
print({k: round(v, 4) for k, v in S.items() if k.startswith("E3")})

# ---------------- E4. exclusion battery for residual-B context ----------------
print("=== E4 ===")
words = []
cur = []
for c in probe:
    if c == 32 or c == 10:
        if cur:
            words.append(bytes(cur))
        cur = []
    else:
        cur.append(c)
if cur:
    words.append(bytes(cur))
top_words = {w for w, _ in Counter(words).most_common(500)}
word_idx = np.zeros(len(probe), dtype=int)   # id of the word the byte belongs to
prev_word_idx = np.zeros(len(probe), dtype=int)
wid = 0
cur_ids = {}
i = 0
last_wid = 0
for c in probe:
    if c == 32 or c == 10:
        last_wid = wid
        wid += 1
    word_idx[i] = wid
    prev_word_idx[i] = max(0, wid - 1)
    i += 1
# map word id -> frequent group id (or -1)
word_bytes = {}
i = 0
cur = []
cw = []
for c in probe:
    if c == 32 or c == 10:
        pass
word_id_group = np.full(len(probe), -1)
freq_ids = {w: k for k, w in enumerate(sorted(top_words))}
# rebuild word text per id
wid = 0
buf = []
j = 0
word_start = 0
for i, c in enumerate(probe):
    if c == 32 or c == 10:
        w = bytes(buf)
        gid = freq_ids.get(w, -1)
        word_id_group[word_start:i + 1] = gid
        buf = []
        word_start = i + 1
        wid += 1
    else:
        buf.append(c)
w = bytes(buf)
word_id_group[word_start:] = freq_ids.get(w, -1)

sent_pos = np.zeros(len(probe))
d = 0
for i, c in enumerate(probe):
    sent_pos[i] = min(d, 60)
    d = 0 if c in b".!?\n" else d + 1

doc_third = np.clip(np.arange(len(probe)) * 3 // len(probe), 0, 2)

def r2_of(R, groups):
    groups = [g for g in groups if g.sum() > 0]
    gidx = np.full(len(probe), -1, dtype=int)
    for gi, m in enumerate(groups):
        gidx[m] = gi
    keep = gidx >= 0
    Rk = R[keep]
    gm = np.stack([R[m].mean(0) for m in groups])
    pred = gm[gidx[keep]]
    return float(np.mean(1 - ((Rk - pred) ** 2).sum(0) / ((Rk - Rk.mean(0)) ** 2).sum(0)))

S["E4"] = {}
hiddens = []
idx_probe = torch.tensor(probe, device=device).unsqueeze(0)
with torch.no_grad():
    x = model.tok_emb(idx_probe)
    for norm, layer in zip(model.prenorms, model.layers):
        h = norm(x)
        hiddens.append(h[0].cpu())
        x = x + layer(h, idx_probe)
for li, layer in enumerate(model.layers):
    H = hiddens[li]
    with torch.no_grad():
        R = layer.res_B(H.cuda()).cpu().numpy().reshape(len(probe), -1)
    res = {
        "word_identity_top500": r2_of(R, [word_id_group == k for k in range(500)]),
        "prev_word_identity_top500": r2_of(R, [np.roll(word_id_group, 1) == k for k in range(500)]),
        "position_in_sentence": r2_of(R, [(sent_pos >= lo) & (sent_pos < hi) for lo, hi in [(0, 5), (5, 15), (15, 40), (40, 61)]]),
        "byte_bigram_prev": r2_of(R, [((probe == c1) & (np.roll(probe, 1) == c2)) for c1 in [ord('t'), ord('e'), ord('a'), ord(' ')] for c2 in [ord('h'), ord('e'), ord('s'), ord(' ')] ]),
        "doc_third": r2_of(R, [doc_third == k for k in range(3)]),
    }
    S["E4"][f"L{li}"] = {k: round(v, 4) for k, v in res.items()}
    print(f"L{li}:", {k: round(v, 3) for k, v in res.items()})

(OUT / "v7e_strengthening.json").write_text(json.dumps(S, indent=2))
print("saved", OUT / "v7e_strengthening.json")
