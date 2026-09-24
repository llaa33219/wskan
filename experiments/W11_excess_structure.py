"""W11 excess-structure battery: WHY does the organization arise?

The monograph (section 6) records as open: the frequency ladder, the
structure/content gate split, and the antipodal specialists exceed what
next-byte prediction strictly requires. This battery attacks the origin:

  A. Init vs trained (wskan11, 1k/10k/100k): which structures are present
     at initialization (architectural prior) and which are learned.
  B. wskan11real (omega==0 ablation): does a pure-decay model build the
     same timescale ladder / clock / gate split? (oscillation-independent?)
  C. Necessity interventions at 100k canonical: flatten the rho ladder,
     shuffle mode assignment, shuffle feature tables -> eval-CE deltas.
     If destruction hurts, the structure is load-bearing, not excess.
  D. mamba2 100k: does another architecture on the same data develop the
     same organization? (A ladder drift, dt clock, B byte-class split)
  E. Training dynamics (1k, 100k): order of emergence of each structure
     over a from-scratch run.

Outputs: figures/w11_excess_structure.json + printed tables.
Run: .venv/bin/python experiments/W11_excess_structure.py
"""

from __future__ import annotations

import json
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from models.V11_WSKAN import WaveletStateKANLMV11
from models.V7_WSKAN import byte_features

DEV = "cuda"
SIZE = {"1k": (4, 1), "10k": (12, 1), "100k": (40, 2)}
FEAT_ROWS = ["space", "newline", "punct", "upper", "lower", "digit", "vowel", "const"]


def make_model(d: int, L: int, oscillatory: bool = True) -> WaveletStateKANLMV11:
    return WaveletStateKANLMV11(vocab_size=256, d_model=d, n_layers=L, use_feature_bc=True,
                                wz_diag=False, g_rank=None, bc_rank=min(32, d),
                                bf16_scan=False, oscillatory=oscillatory).to(DEV).eval()


def load_model(ckpt: str, d: int, L: int, oscillatory: bool = True) -> WaveletStateKANLMV11:
    m = make_model(d, L, oscillatory)
    sd = torch.load(ckpt, weights_only=False)
    m.load_state_dict(sd["state_dict"] if "state_dict" in sd else sd)
    return m.eval()


def geo_fit(values: np.ndarray) -> dict:
    """Geometric-ladder fit: R^2 of log(v) ~ rank, plus endpoint ratio."""
    v = np.sort(np.abs(values))[::-1]
    v = np.clip(v, 1e-8, None)
    x = np.arange(len(v))
    A = np.stack([x, np.ones_like(x)], 1)
    coef, *_ = np.linalg.lstsq(A, np.log(v), rcond=None)
    pred = A @ coef
    denom = ((np.log(v) - np.log(v).mean()) ** 2).sum()
    ss = 0.0 if denom < 1e-12 else 1 - ((np.log(v) - pred) ** 2).sum() / denom
    return dict(r2=float(ss), ratio=float(v[0] / v[-1]))


@torch.no_grad()
def structure_metrics(m: WaveletStateKANLMV11, arr: np.ndarray | None) -> dict:
    out = {"layers": []}
    idx = torch.tensor(arr.astype(np.int64), device=DEV).unsqueeze(0) if arr is not None else None
    for li, layer in enumerate(m.layers):
        sigma = torch.exp(torch.clamp(layer.log_sigma, layer.log_sigma_min, layer.log_sigma_max))
        rho = layer._rho()
        sig_eff = (sigma * rho).median(dim=0).values.cpu().numpy()     # per mode
        om_eff = (layer.omega.abs() * rho).median(dim=0).values.cpu().numpy()
        row = dict(
            timescale_ladder=geo_fit(sig_eff),
            freq_ladder=geo_fit(om_eff) if float(np.abs(om_eff).max()) > 0 else None,
            rho=sorted(layer._rho().cpu().numpy().round(3).tolist()),
            omega_median_per_mode=sorted(om_eff.round(3).tolist()),
        )
        # antipodal structure of readout gains per output channel
        layer._compose_g()
        a, b = layer.a.cpu().numpy(), layer.b.cpu().numpy()      # (I, O, N)
        O = a.shape[1]
        G = np.stack([np.concatenate([a[:, o, :].ravel(), b[:, o, :].ravel()]) for o in range(O)])
        Gn = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-12)
        C = Gn @ Gn.T
        np.fill_diagonal(C, 1.0)
        mins = C.min(axis=1)
        row["antipodal"] = dict(mean_min_cos=float(mins.mean()),
                                frac_strong=float((mins < -0.6).mean()))
        if idx is not None:
            x = m.tok_emb(idx)
            for pj, (norm, lay) in enumerate(zip(m.prenorms, m.layers)):
                h = norm(x)
                if pj == li:
                    Bn = torch.einsum("blf,fin->blin", byte_features(idx), lay.M_B)
                    Br = lay.res_B(h).view(1, idx.shape[1], lay.in_dim, lay.n_states)
                    v_named = Bn.pow(2).mean().item()
                    v_res = Br.pow(2).mean().item()
                    row["gate_split_named_share"] = v_named / (v_named + v_res + 1e-12)
                    dt = torch.clamp(F.softplus(lay.W_dt(h)), max=1.0)[0]     # (L, I)
                    bytes_ = arr
                    space = torch.tensor(bytes_ == 32, device=DEV)
                    letter = torch.tensor(((bytes_ >= 65) & (bytes_ <= 90)) | ((bytes_ >= 97) & (bytes_ <= 122)),
                                          device=DEV)
                    row["clock_space_letter"] = float(dt[space].mean() / dt[letter].mean())
                x = x + lay(h, idx)
        out["layers"].append(row)
    return out


@torch.no_grad()
def eval_ce(m, eval_t: torch.Tensor, block: int = 512, nchunks: int = 60) -> float:
    losses = []
    for s in range(0, min(len(eval_t) - block - 1, nchunks * block), block):
        chunk = eval_t[s:s + block + 1].long()
        lg = m(chunk[:-1].unsqueeze(0))[0]
        losses.append(F.cross_entropy(lg, chunk[1:]).item())
    return float(np.mean(losses))


def part_a(arr):
    print("=" * 70)
    print("A. init vs trained structure (wskan11)")
    out = {}
    for tier in ("1k", "10k", "100k"):
        d, L = SIZE[tier]
        init_m = make_model(d, L)
        tr_m = load_model(f"checkpoints/wskan11_tinystories_{tier}_3ep_s42/latest.pt", d, L) \
            if tier != "100k" else load_model("checkpoints/wskan11_ultrachat_100k_3ep_s42/latest.pt", d, L)
        out[tier] = dict(init=structure_metrics(init_m, arr), trained=structure_metrics(tr_m, arr))
        for li in range(L):
            i0, t0 = out[tier]["init"]["layers"][li], out[tier]["trained"]["layers"][li]
            print(f"  {tier} L{li}: timescale ladder R2 init {i0['timescale_ladder']['r2']:+.2f} "
                  f"-> trained {t0['timescale_ladder']['r2']:+.2f} (ratio {t0['timescale_ladder']['ratio']:.1f}x) | "
                  f"freq ladder R2 {i0['freq_ladder']['r2'] if i0['freq_ladder'] else float('nan'):+.2f} -> "
                  f"{t0['freq_ladder']['r2'] if t0['freq_ladder'] else float('nan'):+.2f} | "
                  f"gate named share {i0.get('gate_split_named_share', float('nan')):.2f} -> "
                  f"{t0.get('gate_split_named_share', float('nan')):.2f} | "
                  f"clock {i0.get('clock_space_letter', float('nan')):.2f} -> "
                  f"{t0.get('clock_space_letter', float('nan')):.2f} | "
                  f"antipodal mean-min-cos {i0['antipodal']['mean_min_cos']:+.2f} -> "
                  f"{t0['antipodal']['mean_min_cos']:+.2f}")
    # antipodal cross-check on the wskan7bc interp checkpoint (origin of the claim)
    m7 = load_model("checkpoints/wskan7bc_ultrachat_interp_s42/latest.pt", 32, 3)
    met7, met7i = structure_metrics(m7, arr), structure_metrics(make_model(32, 3), arr)
    out["wskan7bc_antipodal"] = dict(
        init=[met7i["layers"][li]["antipodal"] for li in range(3)],
        trained=[met7["layers"][li]["antipodal"] for li in range(3)])
    for li in range(3):
        print(f"  wskan7bc L{li}: antipodal init {met7i['layers'][li]['antipodal']['mean_min_cos']:+.2f} -> "
              f"trained {met7['layers'][li]['antipodal']['mean_min_cos']:+.2f} "
              f"(frac strong {met7['layers'][li]['antipodal']['frac_strong']:.2f})")
    return out


def part_b(arr):
    print("=" * 70)
    print("B. wskan11real (omega==0) at 100k: same organization without oscillation?")
    m = load_model("checkpoints/wskan11real_ultrachat_100k_3ep_s42/latest.pt", 40, 2, oscillatory=False)
    out = structure_metrics(m, arr)
    for li, row in enumerate(out["layers"]):
        print(f"  real L{li}: timescale ladder R2 {row['timescale_ladder']['r2']:+.2f} "
              f"(ratio {row['timescale_ladder']['ratio']:.1f}x) | "
              f"gate named share {row.get('gate_split_named_share', float('nan')):.2f} | "
              f"clock {row.get('clock_space_letter', float('nan')):.2f} | "
              f"antipodal {row['antipodal']['mean_min_cos']:+.2f}")
    return out


def part_c(eval_t):
    print("=" * 70)
    print("C. necessity interventions at 100k canonical (eval CE deltas)")
    d, L = SIZE["100k"]
    base_ce = None
    results = {}

    def fresh():
        return load_model("checkpoints/wskan11_ultrachat_100k_3ep_s42/latest.pt", d, L)

    m = fresh()
    base_ce = eval_ce(m, eval_t)
    results["baseline"] = base_ce
    print(f"  baseline: {base_ce:.4f}")

    m = fresh()
    for layer in m.layers:
        layer.log_rho.data.zero_()
    ce = eval_ce(m, eval_t)
    results["rho_flattened"] = dict(ce=ce, delta=ce - base_ce)
    print(f"  rho->1 (timescale ladder off): {ce:.4f} ({ce - base_ce:+.4f})")

    m = fresh()
    g = torch.Generator().manual_seed(0)

    def shuffle_modes(m, what):
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

    m = fresh()
    shuffle_modes(m, "all")
    ce = eval_ce(m, eval_t)
    results["modes_shuffled_gauge"] = dict(ce=ce, delta=ce - base_ce)
    print(f"  full mode permutation incl. biases (gauge symmetry, expect ~0): {ce:.4f} ({ce - base_ce:+.4f})")

    m = fresh()
    shuffle_modes(m, "omega")
    ce = eval_ce(m, eval_t)
    results["omega_shuffled"] = dict(ce=ce, delta=ce - base_ce)
    print(f"  omega-only shuffled (frequency-mode consistency off): {ce:.4f} ({ce - base_ce:+.4f})")

    m = fresh()
    shuffle_modes(m, "sigmarho")
    ce = eval_ce(m, eval_t)
    results["sigmarho_shuffled"] = dict(ce=ce, delta=ce - base_ce)
    print(f"  sigma+rho-only shuffled (timescale-mode consistency off): {ce:.4f} ({ce - base_ce:+.4f})")

    m = fresh()
    perm_f = torch.tensor([7, 6, 5, 4, 3, 2, 1, 0])
    for layer in m.layers:
        layer.M_B.data = layer.M_B.data[perm_f]
        layer.M_C.data = layer.M_C.data[perm_f]
    ce = eval_ce(m, eval_t)
    results["features_shuffled"] = dict(ce=ce, delta=ce - base_ce)
    print(f"  feature tables row-shuffled (gate split misassigned): {ce:.4f} ({ce - base_ce:+.4f})")
    return results


def part_d(arr):
    print("=" * 70)
    print("D. mamba2 100k: does the same data build the same organization elsewhere?")
    from experiments.V1_train_tinystories_lm import Mamba2ByteLM
    out = {}
    for tag in ("init", "trained"):
        m = Mamba2ByteLM(scale="100k").to(DEV).eval()
        if tag == "trained":
            sd = torch.load("checkpoints/mamba2_ultrachat_100k_3ep_s42/latest.pt", weights_only=False)
            m.load_state_dict(sd["state_dict"] if "state_dict" in sd else sd)
        mixers = [ly.mixer for ly in m.model.backbone.layers]
        ladders = [geo_fit(torch.exp(mx.A_log).detach().cpu().numpy()) for mx in mixers]
        idx = torch.tensor(arr.astype(np.int64), device=DEV).unsqueeze(0)
        dt_means = {}
        hooks = []

        def make_hook(mi, mx):
            def hook(mod, inp, outp):
                proj = outp if torch.is_tensor(outp) else outp[0]
                dt_raw = proj[..., -mx.num_heads:]   # HF layout: [gate, xBC, dt]
                dt_means[mi] = F.softplus(dt_raw + mx.dt_bias).detach()
            return hook
        for mi, mx in enumerate(mixers):
            hooks.append(mx.in_proj.register_forward_hook(make_hook(mi, mx)))
        if tag == "init":
            print(f"    (in_proj out_features: {mixers[0].in_proj.out_features}, "
                  f"num_heads: {mixers[0].num_heads}, A_log shape: {tuple(mixers[0].A_log.shape)})")
        with torch.no_grad():
            m.model(input_ids=idx)
        for h in hooks:
            h.remove()
        space = torch.tensor(arr == 32, device=DEV)
        letter = torch.tensor(((arr >= 65) & (arr <= 90)) | ((arr >= 97) & (arr <= 122)), device=DEV)
        clock = [float(dt_means[mi][0, space].mean() / dt_means[mi][0, letter].mean()) for mi in dt_means]
        out[tag] = dict(A_ladders=ladders, dt_clock_space_letter=clock)
        print(f"  {tag}: A ladder R2 per layer {[round(l['r2'], 2) for l in ladders]} | "
              f"dt clock space/letter per layer {[round(c, 2) for c in clock]}")
    return out


def part_e(tier: str, steps: int, every: int, arr):
    print("=" * 70)
    print(f"E. training dynamics at {tier}: order of emergence ({steps} steps)")
    from experiments.V1_train_tinystories_lm import load_data
    d, L = SIZE[tier]
    ds = "tinystories"
    train_ids, eval_ids = load_data(1400000, 500, 42, ds)
    train_ids, eval_ids = train_ids.long().to(DEV), eval_ids.long().to(DEV)
    torch.manual_seed(42)
    m = make_model(d, L)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.0)
    traj = []

    def get_batch(src, bs=32, blk=256):
        ix = torch.randint(len(src) - blk - 1, (bs,), device=DEV)
        return torch.stack([src[i:i + blk + 1] for i in ix])

    for step in range(1, steps + 1):
        batch = get_batch(train_ids)
        lg = m(batch[:, :-1])
        loss = F.cross_entropy(lg.reshape(-1, 256), batch[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if step % every == 0 or step == 1:
            m.eval()
            with torch.no_grad():
                lg = m(eval_ids[:513].unsqueeze(0)[:, :-1])
                ev = F.cross_entropy(lg.reshape(-1, 256), eval_ids[1:513].unsqueeze(0).reshape(-1)).item()
                met = structure_metrics(m, arr)
            m.train()
            row = dict(step=step, train=float(loss.item()), eval=float(ev), layers=met["layers"])
            traj.append(row)
            l0 = met["layers"][-1]
            print(f"  step {step:5d} eval {ev:.3f} | tsladderR2 {l0['timescale_ladder']['r2']:+.2f} | "
                  f"freqladderR2 {l0['freq_ladder']['r2'] if l0['freq_ladder'] else float('nan'):+.2f} | "
                  f"gate {l0.get('gate_split_named_share', float('nan')):.2f} | "
                  f"clock {l0.get('clock_space_letter', float('nan')):.2f} | "
                  f"antip {l0['antipodal']['mean_min_cos']:+.2f}")
    return traj


def main():
    from experiments.V1_train_tinystories_lm import load_data
    tr, ev = load_data(1400000, 500, 42, "ultrachat")
    arr = tr[:2048].numpy()
    eval_t = ev[:200000].to(DEV)
    out = {}
    out["A_init_vs_trained"] = part_a(arr)
    out["B_wskan11real"] = part_b(arr)
    out["C_interventions"] = part_c(eval_t)
    out["D_mamba2"] = part_d(arr)
    out["E_dynamics_1k"] = part_e("1k", 3000, 250, arr)
    out["E_dynamics_100k"] = part_e("100k", 3000, 250, arr)
    with open("experiments/figures/w11_excess_structure.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("saved experiments/figures/w11_excess_structure.json")


if __name__ == "__main__":
    main()
