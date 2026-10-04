"""Mamba-2 causal boundary battery (round-12): is mamba2's boundary-skip
load-bearing, as WSKAN's boundary-reset is?

WSKAN's clock passed a causal battery (remove boundary Delta pulses ->
word-initial CE degrades 21x more than matched controls). The mamba2
comparison in the monograph was observational (dt and write magnitude drop
at spaces). This script applies the analogous intervention to mamba2:
raise dt at space positions to the letter mean (removing the boundary
dip), and compare word-initial CE damage against a matched control
(same-sized dt change applied at letter positions).

Outputs: figures/w12_mamba2_causal_lossmatched.json
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
DEV = "cuda"
CURRENT_MASK: dict = {}


def softplus_inv(y):
    return torch.log(torch.expm1(y))


CKPT = "checkpoints/mamba2_ultrachat_100k_3ep_s42/ckpt_step75000.pt"  # loss-matched to wskan11 100k (1.1823 vs 1.184)


def main():
    from experiments.V1_train_tinystories_lm import Mamba2ByteLM, load_data
    m = Mamba2ByteLM(scale="100k").to(DEV).eval()
    sd = torch.load(CKPT, weights_only=False)
    m.load_state_dict(sd["state_dict"] if "state_dict" in sd else sd)
    _, ev = load_data(1400000, 500, 42, "ultrachat")
    ids = ev[:200000].long().to(DEV)
    mixers = [ly.mixer for ly in m.model.backbone.layers]

    # pass 1: measure per-layer per-head letter-mean and space-mean dt
    cap: dict = {}
    def meas(mi, mx):
        def hk(mod, inp, outp):
            proj = outp if torch.is_tensor(outp) else outp[0]
            cap.setdefault(mi, []).append(
                F.softplus(proj[..., -mx.num_heads:] + mx.dt_bias)[0].detach())
        return hk
    hooks = [mx.in_proj.register_forward_hook(meas(mi, mx)) for mi, mx in enumerate(mixers)]
    with torch.no_grad():
        m.model(input_ids=ids[:8192].unsqueeze(0))
    for h in hooks:
        h.remove()
    a8 = ids[:8192]
    letter8 = ((a8 >= 65) & (a8 <= 90)) | ((a8 >= 97) & (a8 <= 122))
    space8 = a8 == 32
    means = {mi: (torch.stack(v).mean(0)[letter8].mean(0),
                  torch.stack(v).mean(0)[space8].mean(0)) for mi, v in cap.items()}

    def eval_with(mode: str):
        def surg(mi, mx):
            def hk(mod, inp, outp):
                proj = outp if torch.is_tensor(outp) else outp[0]
                nh = mx.num_heads
                dt_raw = proj[..., -nh:]
                cur = F.softplus(dt_raw + mx.dt_bias)
                lm, sm = means[mi]
                mask = CURRENT_MASK["space"] if mode == "boundary" else CURRENT_MASK["letter"]
                if mode == "boundary":
                    tgt = lm[None, None, :].expand_as(cur)
                else:
                    tgt = (cur + (sm - lm)[None, None, :]).clamp_min(1e-6)
                new_dt = torch.where(mask[None, :, None], tgt, cur)
                out = proj.clone()
                out[..., -nh:] = softplus_inv(new_dt.clamp_min(1e-6)) - mx.dt_bias
                return out
            return hk
        if mode != "none":
            hooks = [mx.in_proj.register_forward_hook(surg(mi, mx)) for mi, mx in enumerate(mixers)]
        wi_losses, let_losses = [], []
        with torch.no_grad():
            for s in range(0, 60000, 512):
                chunk = ids[s:s + 513]
                cur = chunk[:-1]
                CURRENT_MASK["space"] = cur == 32
                CURRENT_MASK["letter"] = ((cur >= 97) & (cur <= 122))
                logits = m.model(input_ids=cur.unsqueeze(0)).logits[0]
                tgt = chunk[1:]
                ce = F.cross_entropy(logits, tgt, reduction="none")
                wi = (cur == 32)
                let = ((cur >= 97) & (cur <= 122))
                wi_losses.append(ce[wi].mean().item())
                let_losses.append(ce[let].mean().item())
        if mode != "none":
            for h in hooks:
                h.remove()
        return float(np.mean(wi_losses)), float(np.mean(let_losses))

    base = eval_with("none")
    bnd = eval_with("boundary")
    ctl = eval_with("control")
    out = dict(
        baseline=dict(word_initial=base[0], letter=base[1]),
        boundary_removed=dict(word_initial=bnd[0], letter=bnd[1],
                              d_wi=round(bnd[0] - base[0], 4), d_let=round(bnd[1] - base[1], 4)),
        control=dict(word_initial=ctl[0], letter=ctl[1],
                     d_wi=round(ctl[0] - base[0], 4), d_let=round(ctl[1] - base[1], 4)),
    )
    print(json.dumps(out, indent=2))
    print(f"boundary-removal word-initial damage {bnd[0] - base[0]:+.4f} "
          f"vs control {ctl[0] - base[0]:+.4f}")
    with open("experiments/figures/w12_mamba2_causal_lossmatched.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
