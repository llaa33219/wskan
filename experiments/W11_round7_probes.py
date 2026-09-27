"""Round-7 fast probes (while retrains run).

  A. Antipodal chance baseline: permutation test on the 12-bin profile
     labels (200 permutations) -> null distribution of mirror-pair counts;
     pair counts for 3 trained seeds + init.
  B. Seed-level convergence of the organization (5 campaign seeds, 100k):
     coarse metrics (clock ratio, ladder R2, gate share) mean +/- std, and
     gauge-invariant mode vectors (sorted sigma~, omega~ per layer) across
     seeds - does the coarse structure converge while assignments diverge?
  C. Baseline fixed-chunk CEs for seeds 7/123 (+ wskan11real) for the
     paired retrain deltas.

Outputs: figures/w11_round7_probes.json.
Run: PYTHONPATH=.:experiments .venv/bin/python experiments/W11_round7_probes.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
sys.path.insert(0, "experiments")
from W11_excess_structure import make_model, load_model, geo_fit

DEV = "cuda"
SEEDS = [42, 7, 123, 2024, 31337]


@torch.no_grad()
def channel_activations(m, arr):
    """Per-layer prenormed activations (L, d) + the 12-bin label per position."""
    idx = torch.tensor(arr.astype(np.int64), device=DEV).unsqueeze(0)
    a = torch.tensor(arr.astype(np.int64), device=DEV)
    classes = torch.zeros(len(arr), dtype=torch.long, device=DEV)
    classes[(a == 32)] = 0
    punct = ~((a == 32) | (a == 10) | ((a >= 48) & (a <= 57)) | ((a >= 65) & (a <= 90)) | ((a >= 97) & (a <= 122)))
    classes[(a == 10) | punct] = 1
    classes[((a >= 65) & (a <= 90))] = 2
    classes[((a >= 97) & (a <= 122))] = 3
    classes[((a >= 48) & (a <= 57))] = 4
    vowel = ((a == 97) | (a == 101) | (a == 105) | (a == 111) | (a == 117)
             | (a == 65) | (a == 69) | (a == 73) | (a == 79) | (a == 85))
    classes[vowel] = 5
    run = torch.zeros(len(arr), dtype=torch.long, device=DEV)
    c = 0
    for i in range(len(arr)):
        run[i] = c
        c = 0 if arr[i] in (32, 10) else c + 1
    posbin = (run > 0).long()
    feats = classes * 2 + posbin
    acts = []
    x = m.tok_emb(idx)
    for norm, layer in zip(m.prenorms, m.layers):
        h = norm(x)[0]
        acts.append(h)
        x = x + layer(h, idx)
    return acts, feats


def pair_count(h, feats, perm=None):
    f = feats if perm is None else perm[feats]
    d = h.shape[1]
    p = torch.zeros(d, 12, device=h.device)
    for b in range(12):
        mask = f == b
        if mask.any():
            p[:, b] = h[mask].mean(0)
    p = (p - p.mean(0)) / (p.std(0) + 1e-9)
    C = np.corrcoef(p.cpu().numpy())
    np.fill_diagonal(C, 1.0)
    return int(((C < -0.8).sum()) // 2)


def part_a(arr):
    print("=" * 70)
    print("A. antipodal chance baseline (permutation test, 200 perms)")
    out = {}
    rng = np.random.default_rng(0)
    for tag, m in [("init", make_model(40, 2))] + [
        (f"s{s}", load_model(f"checkpoints/wskan11_ultrachat_100k_3ep_s{s}/latest.pt", 40, 2))
        for s in (42, 7, 123)]:
        acts, feats = channel_activations(m, arr)
        rows = []
        for li, h in enumerate(acts):
            obs = pair_count(h, feats)
            # null: shuffle the position->bin assignment itself (breaks the
            # activation/profile correspondence; bin-label relabeling would
            # leave correlations invariant)
            null = [pair_count(h, feats[torch.randperm(len(feats), device=DEV)]) for _ in range(200)]
            null = np.array(null)
            z = (obs - null.mean()) / (null.std() + 1e-9)
            rows.append(dict(layer=li, observed=obs, null_mean=round(float(null.mean()), 2),
                             null_p95=float(np.percentile(null, 95)), z=round(float(z), 2)))
            print(f"  {tag} L{li}: observed {obs} | null {null.mean():.1f} (p95 {np.percentile(null, 95):.0f}) | z {z:+.1f}")
        out[tag] = rows
    return out


@torch.no_grad()
def part_b(arr):
    print("=" * 70)
    print("B. seed-level convergence (5 campaign seeds, 100k)")
    idx = torch.tensor(arr.astype(np.int64), device=DEV).unsqueeze(0)
    coarse, sig_sorted, om_sorted = [], [], []
    for s in SEEDS:
        m = load_model(f"checkpoints/wskan11_ultrachat_100k_3ep_s{s}/latest.pt", 40, 2)
        row = {"seed": s}
        sv, ov = [], []
        for li, layer in enumerate(m.layers):
            sigma = torch.exp(torch.clamp(layer.log_sigma, layer.log_sigma_min, layer.log_sigma_max))
            rho = layer._rho()
            se = (sigma * rho).median(dim=0).values.cpu().numpy()
            oe = (layer.omega.abs() * rho).median(dim=0).values.cpu().numpy()
            row[f"L{li}_tsladder_r2"] = round(geo_fit(se)["r2"], 3)
            row[f"L{li}_freqladder_r2"] = round(geo_fit(oe)["r2"], 3)
            sv.append(np.sort(se).round(3).tolist())
            ov.append(np.sort(oe).round(3).tolist())
        x = m.tok_emb(idx)
        clocks, gates = [], []
        for norm, layer in zip(m.prenorms, m.layers):
            h = norm(x)
            dt = torch.clamp(F.softplus(layer.W_dt(h)), max=1.0)[0]
            space = torch.tensor(arr == 32, device=DEV)
            letter = torch.tensor(((arr >= 65) & (arr <= 90)) | ((arr >= 97) & (arr <= 122)), device=DEV)
            clocks.append(float(dt[space].mean() / dt[letter].mean()))
            Bn_named = torch.einsum("blf,fin->blin",
                                    __import__("models.V7_WSKAN", fromlist=["byte_features"]).byte_features(idx),
                                    layer.M_B)
            Br = layer.res_B(h).view(1, idx.shape[1], layer.in_dim, layer.n_states)
            gates.append(float(Bn_named.pow(2).mean() / (Bn_named.pow(2).mean() + Br.pow(2).mean())))
            x = x + layer(h, idx)
        row["clock"] = [round(c, 3) for c in clocks]
        row["gate_named_share"] = [round(g, 3) for g in gates]
        coarse.append(row)
        sig_sorted.append(sv)
        om_sorted.append(ov)
        print(f"  s{s}: tsladder R2 {[row[f'L{li}_tsladder_r2'] for li in range(2)]} | "
              f"clock {row['clock']} | gate {row['gate_named_share']}")
    sig_arr = np.array(sig_sorted)   # (seeds, layers, 6)
    om_arr = np.array(om_sorted)
    conv = dict(sorted_sigma_spread_cv=(sig_arr.std(0) / (sig_arr.mean(0) + 1e-9)).round(3).tolist(),
                sorted_omega_spread_cv=(om_arr.std(0) / (om_arr.mean(0) + 1e-9)).round(3).tolist())
    print(f"  sorted sigma~ cross-seed CV per (layer, mode-rank): {conv['sorted_sigma_spread_cv']}")
    print(f"  sorted omega~ cross-seed CV: {conv['sorted_omega_spread_cv']}")
    return dict(per_seed=coarse, convergence=conv)


def part_c():
    print("=" * 70)
    print("C. baseline fixed-chunk CEs (paired-delta denominators)")
    from experiments.V1_train_tinystories_lm import load_data
    _, ev = load_data(1400000, 500, 42, "ultrachat")
    eval_ids = ev.long().to(DEV)
    ev_chunks = [eval_ids[s:s + 513] for s in range(0, 16 * 512, 512)]

    def fixed_ce(m):
        with torch.no_grad():
            return float(np.mean([F.cross_entropy(m(c[:-1].unsqueeze(0))[0], c[1:]).item() for c in ev_chunks]))
    out = {}
    for s in (7, 123):
        out[f"wskan11_s{s}"] = round(fixed_ce(load_model(
            f"checkpoints/wskan11_ultrachat_100k_3ep_s{s}/latest.pt", 40, 2)), 4)
        out[f"wskan11real_s{s}"] = round(fixed_ce(load_model(
            f"checkpoints/wskan11real_ultrachat_100k_3ep_s{s}/latest.pt", 40, 2, oscillatory=False)), 4)
        print(f"  s{s}: wskan11 {out[f'wskan11_s{s}']}, wskan11real {out[f'wskan11real_s{s}']}")
    out["wskan11_s42"] = 1.2907
    out["wskan11real_s42"] = 1.3280
    return out


def main():
    from experiments.V1_train_tinystories_lm import load_data
    tr, _ = load_data(1400000, 500, 42, "ultrachat")
    arr = tr[:4096].numpy()
    out = dict(A=part_a(arr), B=part_b(arr), C=part_c())
    with open("experiments/figures/w11_round7_probes.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("saved experiments/figures/w11_round7_probes.json")


if __name__ == "__main__":
    main()
