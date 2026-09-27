"""Round-6 fast measurements (run while ablation retrains train).

  A. Multi-seed, multi-permutation interventions at 100k (seeds 42/7/123,
     3 permutations): omega-shuffle, sigmarho-shuffle, feature-shuffle,
     rho-flatten - mean +/- std instead of single samples.
  B. mamba2 boundary operation: write magnitude (dt * ||B||) and dt at
     spaces vs letters - is the mamba2 boundary op "skip" (small write)
     vs WSKAN's "reset" (large Delta step)?
  C. Antipodal specialists with the ORIGINAL metric (V7BC Debt 3):
     per-channel activation profiles over (byte class x position-in-word)
     bins, z-scored; count mirror-image pairs (profile corr < -0.8) on
     wskan11 100k and the wskan7bc interp checkpoint, init vs trained.

Outputs: figures/w11_round6_probes.json.
Run: .venv/bin/python experiments/W11_round6_probes.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from W11_excess_structure import make_model, load_model, eval_ce
from models.V7_WSKAN import byte_features

DEV = "cuda"
CKPTS = {42: "checkpoints/wskan11_ultrachat_100k_3ep_s42/latest.pt",
         7: "checkpoints/wskan11_ultrachat_100k_3ep_s7/latest.pt",
         123: "checkpoints/wskan11_ultrachat_100k_3ep_s123/latest.pt"}


def shuffle_modes(m, what, g):
    for layer in m.layers:
        N = layer.n_states
        perm = torch.randperm(N, generator=g)
        if what in ("all", "omega"):
            layer.omega.data = layer.omega.data[:, perm]
        if what in ("all", "sigmarho"):
            layer.log_sigma.data = layer.log_sigma.data[:, perm]
            layer.log_rho.data = layer.log_rho.data[perm]
        if what == "all":
            layer.a.data = layer.a.data[:, :, perm]
            layer.b.data = layer.b.data[:, :, perm]
            layer.M_B.data = layer.M_B.data[:, :, perm]
            layer.M_C.data = layer.M_C.data[:, :, perm]
            for res in (layer.res_B, layer.res_C):
                W2 = res[1].weight.data
                r = W2.shape[1]
                W2.copy_(W2.view(layer.in_dim, N, r)[:, perm, :].reshape(layer.in_dim * N, r))
                res[1].bias.data.copy_(res[1].bias.data.view(layer.in_dim, N)[:, perm].reshape(-1))


def part_a(eval_t):
    print("=" * 70)
    print("A. multi-seed multi-permutation interventions (100k)")
    rows = {}
    for seed, ck in CKPTS.items():
        base = load_model(ck, 40, 2)
        base_ce = eval_ce(base, eval_t)
        row = {"baseline": round(base_ce, 4)}
        for what, label in (("omega", "omega_shuffled"), ("sigmarho", "sigmarho_shuffled")):
            deltas = []
            for pseed in range(3):
                m = load_model(ck, 40, 2)
                shuffle_modes(m, what, torch.Generator().manual_seed(pseed))
                deltas.append(eval_ce(m, eval_t) - base_ce)
            row[label] = dict(mean=round(float(np.mean(deltas)), 4),
                              std=round(float(np.std(deltas)), 4))
        m = load_model(ck, 40, 2)
        for layer in m.layers:
            layer.log_rho.data.zero_()
        row["rho_flattened"] = round(eval_ce(m, eval_t) - base_ce, 4)
        deltas = []
        for pseed in range(3):
            m = load_model(ck, 40, 2)
            g = torch.Generator().manual_seed(pseed)
            perm_f = torch.randperm(8, generator=g)
            for layer in m.layers:
                layer.M_B.data = layer.M_B.data[perm_f]
                layer.M_C.data = layer.M_C.data[perm_f]
            deltas.append(eval_ce(m, eval_t) - base_ce)
        row["features_shuffled"] = dict(mean=round(float(np.mean(deltas)), 4),
                                        std=round(float(np.std(deltas)), 4))
        rows[str(seed)] = row
        print(f"  s{seed}: base {row['baseline']:.4f} | omega {row['omega_shuffled']['mean']:+.3f}±{row['omega_shuffled']['std']:.3f} "
              f"| sigmarho {row['sigmarho_shuffled']['mean']:+.3f}±{row['sigmarho_shuffled']['std']:.3f} "
              f"| rho-flat {row['rho_flattened']:+.4f} | features {row['features_shuffled']['mean']:+.3f}±{row['features_shuffled']['std']:.3f}")
    return rows


@torch.no_grad()
def part_b(arr):
    print("=" * 70)
    print("B. mamba2 boundary operation: skip vs reset")
    from experiments.V1_train_tinystories_lm import Mamba2ByteLM
    m = Mamba2ByteLM(scale="100k").to(DEV).eval()
    sd = torch.load("checkpoints/mamba2_ultrachat_100k_3ep_s42/latest.pt", weights_only=False)
    m.load_state_dict(sd["state_dict"] if "state_dict" in sd else sd)
    mixers = [ly.mixer for ly in m.model.backbone.layers]
    mx0 = mixers[0]
    nh = mx0.num_heads
    d_inner = mx0.intermediate_size
    d_state = mx0.ssm_state_size
    n_groups = mx0.n_groups
    conv_dim = mx0.conv_dim
    print(f"    in_proj {mx0.in_proj.out_features} = d_inner {d_inner} + conv_dim {conv_dim} + heads {nh}")
    idx = torch.tensor(arr.astype(np.int64), device=DEV).unsqueeze(0)
    cap = {}

    def make_hook(mi, mx):
        def hook(mod, inp, outp):
            proj = outp if torch.is_tensor(outp) else outp[0]
            dt = F.softplus(proj[..., -nh:] + mx.dt_bias)          # (1, L, nh)
            xbc = proj[..., d_inner:d_inner + conv_dim]            # (1, L, conv_dim)
            B = xbc[..., d_inner:d_inner + n_groups * d_state]     # (1, L, G*state)
            cap[mi] = (dt.detach(), B.detach())
        return hook
    hooks = [mx.in_proj.register_forward_hook(make_hook(mi, mx)) for mi, mx in enumerate(mixers)]
    m.model(input_ids=idx)
    for h in hooks:
        h.remove()
    space = torch.tensor(arr == 32, device=DEV)
    letter = torch.tensor(((arr >= 65) & (arr <= 90)) | ((arr >= 97) & (arr <= 122)), device=DEV)
    out = []
    for mi in sorted(cap):
        dt, B = cap[mi]
        dt, B = dt[0], B[0]                                    # (L, nh), (L, G*state)
        write = dt.mean(-1) * B.norm(dim=-1)                   # ZOH write scale proxy
        row = dict(layer=mi,
                   dt_space_letter=round(float(dt[space].mean() / dt[letter].mean()), 3),
                   write_space_letter=round(float(write[space].mean() / write[letter].mean()), 3))
        out.append(row)
        print(f"  layer {mi}: dt space/letter = {row['dt_space_letter']:.3f} | "
              f"write-mag space/letter = {row['write_space_letter']:.3f}")
    return out


@torch.no_grad()
def antipodal_pairs(m, arr, label):
    """V7BC Debt-3 metric: channel profiles over (byte class x pos-in-word)
    bins of the prenormed layer input activations; mirror-image pairs."""
    idx = torch.tensor(arr.astype(np.int64), device=DEV).unsqueeze(0)
    x = m.tok_emb(idx)
    classes = torch.zeros(len(arr), dtype=torch.long, device=DEV)
    a = torch.tensor(arr.astype(np.int64), device=DEV)
    classes[(a == 32)] = 0
    classes[(a == 10) | ((a != 32) & (a != 10) & ~(((a >= 48) & (a <= 57)) | ((a >= 65) & (a <= 90)) | ((a >= 97) & (a <= 122))))] = 1
    classes[((a >= 65) & (a <= 90))] = 2
    classes[((a >= 97) & (a <= 122))] = 3
    classes[((a >= 48) & (a <= 57))] = 4
    vowel = ((a == 97) | (a == 101) | (a == 105) | (a == 111) | (a == 117)
             | (a == 65) | (a == 69) | (a == 73) | (a == 79) | (a == 85))
    classes[vowel] = 5
    d = torch.zeros(len(arr), dtype=torch.long, device=DEV)
    run = torch.zeros(len(arr), dtype=torch.long, device=DEV)
    c = 0
    for i in range(len(arr)):
        run[i] = c
        c = 0 if arr[i] in (32, 10) else c + 1
    posbin = (run > 0).long()                            # 0 = word-initial, 1 = internal
    feats = classes * 2 + posbin                         # 12 bins
    result = []
    for li, (norm, layer) in enumerate(zip(m.prenorms, m.layers)):
        h = norm(x)[0]                                   # (L, d) activations
        prof = torch.zeros(h.shape[1], 12, device=DEV)
        for b in range(12):
            mask = feats == b
            if mask.any():
                prof[:, b] = h[mask].mean(0)
        prof = (prof - prof.mean(0)) / (prof.std(0) + 1e-9)   # z-score per bin
        C = np.corrcoef(prof.cpu().numpy())
        np.fill_diagonal(C, 1.0)
        n_anti = int(((C < -0.8).sum()) // 2)
        result.append(dict(layer=li, antipodal_pairs=n_anti,
                           min_corr=float(C.min())))
        print(f"  {label} L{li}: antipodal pairs (corr<-0.8): {n_anti}, min corr {C.min():+.2f}")
    return result


def part_c(arr):
    print("=" * 70)
    print("C. antipodal specialists, original profile metric")
    out = {}
    out["wskan11_100k_trained"] = antipodal_pairs(
        load_model("checkpoints/wskan11_ultrachat_100k_3ep_s42/latest.pt", 40, 2), arr, "wskan11-100k")
    out["wskan11_100k_init"] = antipodal_pairs(make_model(40, 2), arr, "wskan11-100k-init")
    out["wskan7bc_trained"] = antipodal_pairs(
        load_model("checkpoints/wskan7bc_ultrachat_interp_s42/latest.pt", 32, 3), arr, "wskan7bc")
    out["wskan7bc_init"] = antipodal_pairs(make_model(32, 3), arr, "wskan7bc-init")
    return out


def main():
    from experiments.V1_train_tinystories_lm import load_data
    tr, ev = load_data(1400000, 500, 42, "ultrachat")
    eval_t = ev[:200000].to(DEV)
    arr = tr[:4096].numpy()
    out = dict(A=part_a(eval_t), B=part_b(arr), C=part_c(arr))
    with open("experiments/figures/w11_round6_probes.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("saved experiments/figures/w11_round6_probes.json")


if __name__ == "__main__":
    main()
