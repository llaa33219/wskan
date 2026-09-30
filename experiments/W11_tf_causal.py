"""Transformer boundary causal test (round-31): does the attention baseline
rely on boundary bytes for word-initial prediction, as WSKAN and mamba2 do?

Masks attention to boundary bytes and measures word-initial CE damage
against a matched control (masking letter positions instead).
WSKAN: 19x; mamba2: 1.4x; transformer: measured here.

Outputs: figures/w11_tf_causal.json
Run: PYTHONPATH=.:experiments .venv/bin/python experiments/W11_tf_causal.py
"""

from __future__ import annotations

import json
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from experiments.baselines import TinyTransformerLM
from experiments.V1_train_tinystories_lm import load_data

DEV = "cuda"
BOUNDARY = torch.tensor([32, 10, 46, 33, 63, 44, 59, 58], device=DEV)
LOWER = torch.arange(97, 123, device=DEV)


def fwd(m, idx, mask_kind=None):
    B, L = idx.shape
    pos = torch.arange(L, device=idx.device)
    x = m.tok_emb(idx) + m.pos_emb(pos)[None, :, :]
    bytes_ = idx[0]
    is_bnd = torch.isin(bytes_, BOUNDARY)
    is_let = torch.isin(bytes_, LOWER)
    for blk in m.blocks:
        h = blk.ln1(x)
        q, k, v = blk.qkv(h).chunk(3, dim=-1)
        hd = q.shape[-1] // blk.heads
        q = q.view(B, L, blk.heads, hd).transpose(1, 2)
        k = k.view(B, L, blk.heads, hd).transpose(1, 2)
        v = v.view(B, L, blk.heads, hd).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / (hd ** 0.5)
        causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=idx.device))
        scores = scores.masked_fill(~causal, float("-inf"))
        if mask_kind == "boundary":
            scores = scores.masked_fill(is_bnd[None, None, None, :], float("-inf"))
        elif mask_kind == "control":
            scores = scores.masked_fill(is_let[None, None, None, :], float("-inf"))
        a = torch.nan_to_num(F.softmax(scores, dim=-1), 0.0)
        x = x + blk.proj((a @ v).transpose(1, 2).reshape(B, L, -1))
        h2 = blk.ln2(x)
        x = x + blk.fc2(F.gelu(blk.fc1(h2)))
    return m.head(m.norm(x))


def wi_ce(m, ids, mask_kind):
    ces, wis = [], []
    with torch.no_grad():
        for s in range(0, len(ids) - 512, 512):
            ch = ids[s:s + 513].unsqueeze(0)
            lg = fwd(m, ch[:, :-1], mask_kind)[0]
            ce = F.cross_entropy(lg, ch[0, 1:], reduction="none")
            ces.append(ce)
            wis.append(ch[0, :-1] == 32)
    ce = torch.cat(ces)
    wi = torch.cat(wis)
    return float(ce[wi].mean())


def main():
    _, ev = load_data(1400000, 500, 42, "ultrachat")
    ids = ev[:60000].long().to(DEV)
    m = TinyTransformerLM(d_model=40, n_layers=4, block=512).to(DEV).eval()
    sd = torch.load("checkpoints/tf_ultrachat_100k_3ep_s42/latest.pt", weights_only=False)
    m.load_state_dict(sd["state_dict"] if "state_dict" in sd else sd)
    b_wi = wi_ce(m, ids, None)
    m_wi = wi_ce(m, ids, "boundary")
    c_wi = wi_ce(m, ids, "control")
    out = dict(baseline_wi=round(b_wi, 4), boundary_masked_wi=round(m_wi, 4),
               boundary_delta=round(m_wi - b_wi, 4), control_masked_wi=round(c_wi, 4),
               control_delta=round(c_wi - b_wi, 4),
               ratio=round((m_wi - b_wi) / max(c_wi - b_wi, 1e-9), 3))
    print(json.dumps(out, indent=2))
    with open("experiments/figures/w11_tf_causal.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
