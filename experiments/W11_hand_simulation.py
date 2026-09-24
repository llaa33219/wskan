"""W11 hand-simulation: deriving the next-token prediction by hand.

The monograph's stated ceiling (W11_FINAL_INTERPRETATION_REPORT.md section 6):
"we do not derive outputs without execution". This script attacks it:

  A. Exact replication: an itemized, from-scratch fp32 forward pass (serial
     scan, explicit warped-time history sum) reproduces the model's logits
     to fp tolerance at 1k/10k/100k - so the "hand" arithmetic IS the model.
  B. Worked example at 1k (d=4, L=1, N=6): every intermediate printed -
     clock field, per-mode reads, top history terms - for real contexts.
  C. Truncation curve: how many of the biggest history terms a human must
     sum before the prediction stabilizes (1k, 10k).
  D. Term-count scaling table 1k -> 100k: where hand-followability ends.
  E. 100k dictionary-guided prediction: pre-registered rule (unique
     morphological completions) scored against the canonical checkpoint.

Outputs: figures/w11_hand_simulation.json + printed worked examples.
Run: .venv/bin/python experiments/W11_hand_simulation.py
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
FEAT_NAMES = ["space", "newline", "punct", "upper", "lower", "digit", "vowel", "const"]


def load_model(ckpt: str, d: int, L: int) -> WaveletStateKANLMV11:
    m = WaveletStateKANLMV11(vocab_size=256, d_model=d, n_layers=L, use_feature_bc=True,
                             wz_diag=False, g_rank=None, bc_rank=min(32, d), bf16_scan=False)
    sd = torch.load(ckpt, weights_only=False)
    m.load_state_dict(sd["state_dict"] if "state_dict" in sd else sd)
    return m.eval().to(DEV)


@torch.no_grad()
def hand_forward(m: WaveletStateKANLMV11, idx: torch.Tensor) -> dict:
    """Itemized forward pass. Every intermediate the model computes, recomputed
    here step by step (serial scan, explicit warped-time history sums)."""
    Bsz, L = idx.shape
    d = m.d_model
    x = m.tok_emb(idx)
    layers_out = []
    for li, (norm, layer) in enumerate(zip(m.prenorms, m.layers)):
        h = norm(x)
        I, N = layer.in_dim, layer.n_states
        sigma = torch.exp(torch.clamp(layer.log_sigma, layer.log_sigma_min, layer.log_sigma_max))
        rho = layer._rho()
        sig_eff = sigma * rho                        # (I, N)
        om_eff = layer.omega * rho                   # (I, N)
        dt = torch.clamp(F.softplus(layer.W_dt(h)), max=1.0)   # (B, L, I) the clock
        Bn = torch.einsum("blf,fin->blin", byte_features(idx), layer.M_B) \
            + layer.res_B(h).view(Bsz, L, I, N)
        Cn = torch.einsum("blf,fin->blin", byte_features(idx), layer.M_C) \
            + layer.res_C(h).view(Bsz, L, I, N)
        u = Bn * h.unsqueeze(-1) * dt.unsqueeze(-1)  # ZOH write (B, L, I, N)
        # serial scan h_n = a_n h_{n-1} + u_n
        lam = torch.complex(-sig_eff, om_eff)        # (I, N)
        hst = torch.zeros(Bsz, I, N, dtype=torch.complex64, device=idx.device)
        states = []
        for n in range(L):
            a_n = torch.exp(lam * dt[:, n].unsqueeze(-1))
            hst = a_n * hst + u[:, n].to(torch.complex64)
            states.append(hst.clone())
        hstL = states[-1]
        # readout y_o = sum_ik C (h_re a - h_im b), PER POSITION (full hand scan)
        hand_full = []
        y_last = None
        for p in range(L):
            rp = Cn[:, p] * states[p].real
            ip = Cn[:, p] * states[p].imag
            yp = torch.einsum("bik,iok->bo", rp, layer.a) - torch.einsum(
                "bik,iok->bo", ip, layer.b)
            y_last = yp
            hand_full.append(yp * F.silu(layer.W_z(h[:, p])) + h[:, p] @ layer.w_base)
        hand_full = torch.stack(hand_full, dim=1)   # (B, L, O)
        y = y_last                                   # pre-gate readout at final position
        gate = F.silu(layer.W_z(h[:, -1]))
        base = h[:, -1] @ layer.w_base
        yg = y * gate                                # wave path only (gate applied once)
        # full-sequence residual update via the model's fused kernel;
        # verified below to equal the hand scan at every position
        full = layer(h, idx)
        scan_err = (full - hand_full).abs().max().item()
        x = x + full
        # explicit history-sum decomposition of the final state:
        # h_{n,ik} = sum_m exp(lam_ik (T_n - T_m)) u_{m,ik}, T = cumsum(dt)
        T = torch.cumsum(dt[0], dim=0)               # (L, I) warped time
        dT = T[-1].unsqueeze(0) - T                  # (L, I) warped distance to final
        K = torch.exp(lam.unsqueeze(0) * dT.unsqueeze(-1))     # (L, I, N) complex kernel
        terms = K * u[0].to(torch.complex64)         # (L, I, N) per-history-term state
        layers_out.append(dict(h=h, dt=dt[0], Bn=Bn[0], Cn=Cn[0], T=T, dT=dT, K=K,
                               terms=terms, read_gain=(layer.a, layer.b), C_last=Cn[0, -1],
                               y=y[0], gate=gate[0], base=base[0], yg=yg[0],
                               scan_err=scan_err))
    final_raw = x[0, -1]
    normed = m.norm(final_raw.unsqueeze(0)).squeeze(0)
    logits = m.head(normed.unsqueeze(0)).squeeze(0)
    return dict(logits=logits, final_raw=final_raw, layers=layers_out)


@torch.no_grad()
def margin_decomposition(m: WaveletStateKANLMV11, idx: torch.Tensor, hf: dict):
    """Exact additive decomposition of the final logits into named parts whose
    sum equals the model logits (post-norm units, norm constant included)."""
    final_raw = hf["final_raw"].cpu().numpy()
    mean = final_raw.mean()
    std = float(np.sqrt(final_raw.var() + 1e-5))
    gn = m.norm.weight.cpu().numpy()
    W = m.head.weight.detach().cpu().numpy()
    normd = lambda v: (v / std) * gn
    parts = {}
    parts["emb"] = normd(m.tok_emb(idx[0, -1]).cpu().numpy())
    for li, lo in enumerate(hf["layers"]):
        parts[f"L{li}-base"] = normd(lo["base"].cpu().numpy())
        parts[f"L{li}-wave"] = normd(lo["yg"].cpu().numpy())
    parts["norm-const"] = (-mean) / std * gn + m.norm.bias.cpu().numpy()
    logits = sum(parts.values())
    with torch.no_grad():
        ref = m.norm(hf["final_raw"].unsqueeze(0)).squeeze(0).cpu().numpy()
    leak = float(np.abs(logits - ref).max())
    assert leak < 1e-4, f"decomposition leak {leak:.2e}: parts do not sum to norm(final_raw)"
    return parts, W @ logits, std


def per_history_terms(m, hf, li: int, std: float):
    """Per-(position, channel, mode) contributions to the final logit VECTOR,
    post-gate, post-norm. Returns (L, I, N, vocab)."""
    lo = hf["layers"][li]
    gn = m.norm.weight.detach().cpu().numpy()
    W = m.head.weight.detach().cpu().numpy()
    a, b = lo["read_gain"]
    Cn = lo["C_last"].detach().cpu().numpy()        # (I, N)
    gate = lo["gate"].detach().cpu().numpy()        # (O,)
    terms = lo["terms"].detach().cpu().numpy()      # (L, I, N) complex
    re, im = terms.real, terms.imag
    an, bn = a.detach().cpu().numpy(), b.detach().cpu().numpy()   # (I, O, N)
    # y[m,i,k,o] = C[i,k]*(re[m,i,k]*a[i,o,k] - im[m,i,k]*b[i,o,k])
    y_all = (re[:, :, None, :] * Cn[None, :, None, :] * an[None]
             - im[:, :, None, :] * Cn[None, :, None, :] * bn[None])   # (L,I,O,N)
    y_all = y_all.transpose(0, 1, 3, 2)             # (L,I,N,O)
    y_all = y_all * gate[None, None, None, :]
    normed = (y_all / std) * gn[None, None, None, :]
    return np.einsum("liko,vo->likv", normed, W)      # (L, I, N, vocab)


def ctxs_from_eval(dataset: str, n: int, length: int, seed: int = 0):
    from experiments.V1_train_tinystories_lm import load_data
    _, eval_ids = load_data(1400000, 500, 42, dataset)
    arr = eval_ids.numpy().astype(np.int64)
    rng = np.random.default_rng(seed)
    starts = rng.choice(len(arr) - length - 1, size=n, replace=False)
    return [arr[s:s + length + 1] for s in starts]


def main():
    out = {}
    # ---------- A. exact replication ----------
    print("=" * 70, "\nA. exact replication (hand forward vs model forward, fp32)")
    for tier in ("1k", "10k", "100k"):
        d, L = SIZE[tier]
        m = load_model(f"checkpoints/wskan11_tinystories_{tier}_3ep_s42/latest.pt", d, L)
        idx = torch.tensor(list(b"Once upon a time, there was a littl"), device=DEV).unsqueeze(0)
        with torch.no_grad():
            ref = m(idx)[0, -1]
        hf = hand_forward(m, idx)
        err = (ref - hf["logits"]).abs().max().item()
        scan_err = max(lo["scan_err"] for lo in hf["layers"])
        n_terms = sum(lo["terms"].numel() for lo in hf["layers"])
        out[f"replication_{tier}"] = dict(max_logit_err=err, max_scan_err=scan_err,
                                          history_terms=n_terms)
        print(f"  {tier:>4}: max |logit err| = {err:.2e}   max per-position scan err = {scan_err:.2e}   "
              f"history terms = {n_terms}")

    # ---------- model quality (honesty: 1k/10k are weak predictors) ----------
    from experiments.V1_train_tinystories_lm import load_data
    _, eval_ids = load_data(1400000, 500, 42, "tinystories")
    eval_t = eval_ids[:200000].long().to(DEV)
    for tier in ("1k", "10k"):
        d, L = SIZE[tier]
        m = load_model(f"checkpoints/wskan11_tinystories_{tier}_3ep_s42/latest.pt", d, L)
        with torch.no_grad():
            losses, accs = [], []
            for s in range(0, len(eval_t) - 513, 50000):
                chunk = eval_t[s:s + 513]
                lg = m(chunk[:-1].unsqueeze(0))[0]
                losses.append(F.cross_entropy(lg, chunk[1:]).item())
                accs.append((lg.argmax(-1) == chunk[1:]).float().mean().item())
        out[f"quality_{tier}"] = dict(ce=float(np.mean(losses)), acc=float(np.mean(accs)))
        print(f"  {tier:>4} quality: CE {np.mean(losses):.4f}  next-byte acc {np.mean(accs):.3f}")

    # ---------- B. worked example at 1k ----------
    print("=" * 70, "\nB. worked example at 1k (d=4, 1 layer, 6 modes)")
    d, L1 = SIZE["1k"]
    m = load_model("checkpoints/wskan11_tinystories_1k_3ep_s42/latest.pt", d, L1)
    # pick an eval window ending mid-word where the model is correct, and one where wrong
    windows = ctxs_from_eval("tinystories", 40, 24, seed=1)
    good, bad = None, None
    for w in windows:
        ctx, nxt = torch.tensor(w[:-1], device=DEV).unsqueeze(0), int(w[-1])
        with torch.no_grad():
            pred = int(m(ctx)[0, -1].argmax())
        if chr(nxt).isalpha() and chr(int(w[-2])).isalpha():
            if pred == nxt and good is None:
                good = (w, pred, nxt)
            if pred != nxt and bad is None:
                bad = (w, pred, nxt)
    worked = {}
    for tag, item in (("correct", good), ("wrong", bad)):
        if item is None:
            continue
        w, pred, nxt = item
        ctx_bytes = bytes(int(c) for c in w[:-1])
        idx = torch.tensor(list(ctx_bytes), device=DEV).unsqueeze(0)
        hf = hand_forward(m, idx)
        parts, logits, std = margin_decomposition(m, idx, hf)
        order = np.argsort(-logits)[:5]
        lo = hf["layers"][0]
        print(f"\n  [{tag}] context: {ctx_bytes!r}  -> actual next {chr(nxt)!r}")
        print(f"  clock field dt (per byte, 4 channels):")
        for p in range(idx.shape[1]):
            c = ctx_bytes[p]
            ch = chr(c) if 32 <= c < 127 else f"\\x{c:02x}"
            print(f"    {p:2d} {ch!r:>6}: dt = {np.array2string(lo['dt'][p].cpu().numpy(), precision=3, separator=' ')}")
        print(f"  logit parts (top-5 bytes shown; parts sum exactly to logits):")
        W_head = m.head.weight.detach().cpu().numpy()
        for v in order:
            row = "  ".join(f"{k}={float(W_head[v] @ parts[k]):+.2f}"
                            for k in ("emb", "L0-base", "L0-wave", "norm-const"))
            print(f"    {chr(v)!r:>6}: total {logits[v]:+.3f} | {row}")
        # top history terms for the predicted byte
        contrib = per_history_terms(m, hf, 0, std)   # (L, I, N, vocab)
        v0 = int(order[0])
        flat = contrib[..., v0].reshape(-1)
        top = np.argsort(-np.abs(flat))[:10]
        Lc, I, N = contrib.shape[:3]
        print(f"  top-10 history terms for logit[{chr(v0)!r}] (byte-pos, char, ch, mode, warped-dist, contribution):")
        for t in top:
            p, i, k = np.unravel_index(t, (Lc, I, N))
            c = ctx_bytes[p]
            ch = chr(c) if 32 <= c < 127 else f"\\x{c:02x}"
            wd = float(lo["dT"][p, i])
            print(f"    pos {p:2d} {ch!r:>6} ch{i} mode{k} dT={wd:+.3f} -> {flat[t]:+.3f}")
        worked[tag] = dict(context=ctx_bytes.decode("latin1"), actual=chr(nxt), pred=chr(v0),
                           top5={chr(int(v)): round(float(logits[v]), 3) for v in order})
    out["worked_1k"] = worked

    # ---------- C. truncation curve ----------
    print("=" * 70, "\nC. truncation: top-k history terms needed for stable argmax")
    for tier in ("1k", "10k", "100k"):
        d, L = SIZE[tier]
        ck = (f"checkpoints/wskan11_tinystories_{tier}_3ep_s42/latest.pt" if tier != "100k"
              else "checkpoints/wskan11_ultrachat_100k_3ep_s42/latest.pt")
        m = load_model(ck, d, L)
        ds = "tinystories" if tier != "100k" else "ultrachat"
        ks, k0_hits = [], 0
        for w in ctxs_from_eval(ds, 50, 24, seed=7):
            idx = torch.tensor([int(c) for c in w[:-1]], device=DEV).unsqueeze(0)
            hf = hand_forward(m, idx)
            parts, logits, std = margin_decomposition(m, idx, hf)
            v_full = int(np.argmax(logits))
            flats = []
            for li in range(len(hf["layers"])):
                full = per_history_terms(m, hf, li, std)      # (L,I,N,vocab)
                flats.append(full.reshape(-1, full.shape[-1]))
                logits = logits - full.sum(axis=(0, 1, 2))    # strip this layer's history
            flat = np.concatenate(flats, axis=0)              # all layers' history terms
            base_logits = logits
            order_t = np.argsort(-np.abs(flat[:, v_full]))
            k0_hits += int(int(np.argmax(base_logits)) == v_full)
            for k in range(1, len(order_t) + 1):
                approx = base_logits + flat[order_t[:k]].sum(0)
                if int(np.argmax(approx)) == v_full:
                    ks.append(k)
                    break
            else:
                ks.append(len(order_t))
        out[f"truncation_{tier}"] = dict(median_k=float(np.median(ks)), p90_k=float(np.percentile(ks, 90)),
                                         frac_le25=float(np.mean(np.array(ks) <= 25)), n=len(ks),
                                         frac_k0=float(k0_hits / len(ks)),
                                         total_terms=int(flat.shape[0]))
        print(f"  {tier:>4}: median {np.median(ks):.0f}, p90 {np.percentile(ks, 90):.0f} of {flat.shape[0]} terms; "
              f"argmax stable with <=25 terms in {np.mean(np.array(ks) <= 25) * 100:.0f}% of contexts; "
              f"static parts alone (k=0) already correct in {k0_hits / len(ks) * 100:.0f}%")

    # ---------- D. term-count scaling ----------
    print("=" * 70, "\nD. term-count scaling (24-byte context)")
    for tier, (d, L) in SIZE.items():
        terms = 24 * d * 6 * L
        print(f"  {tier:>4}: d={d:3d} L={L}  history terms = {terms:6d}  (+ {256 * d} head products per position)")
        out.setdefault("scaling", {})[tier] = dict(d=d, layers=L, history_terms=terms, head_products=256 * d)

    # ---------- E. 100k dictionary-guided prediction ----------
    print("=" * 70, "\nE. 100k canonical: pre-registered completion rule")
    m = load_model("checkpoints/wskan11_ultrachat_100k_3ep_s42/latest.pt", *SIZE["100k"])
    from experiments.V1_train_tinystories_lm import load_data as ld2
    tr, ev = ld2(1400000, 500, 42, "ultrachat")
    tr_arr = tr.numpy()[:2_000_000]
    ev_arr = ev.numpy()
    # build prefix->next-byte stats over a training sample (the "human dictionary")
    from collections import defaultdict
    stats = defaultdict(lambda: np.zeros(256, dtype=np.int64))
    for s in range(len(tr_arr) - 5):
        stats[bytes(tr_arr[s:s + 4])][tr_arr[s + 4]] += 1
    cases, n_alpha, n_seen, n_pure = [], 0, 0, 0
    rng = np.random.default_rng(3)
    cand = rng.choice(len(ev_arr) - 40, size=8000, replace=False)
    for s in cand:
        ctx = ev_arr[s:s + 24]
        nxt = int(ev_arr[s + 24])
        key = bytes(ctx[-4:])
        if not all(chr(c).isalpha() for c in key):
            continue
        n_alpha += 1
        cnt = stats.get(key)
        if cnt is None or cnt.sum() < 8:
            continue
        n_seen += 1
        top = int(cnt.argmax())
        if cnt[top] / cnt.sum() < 0.75:
            continue
        n_pure += 1
        cases.append((ctx, nxt, top, float(cnt[top] / cnt.sum())))
        if len(cases) >= 20:
            break
    print(f"  filter: {n_alpha} alpha-suffix -> {n_seen} seen >=8x -> {n_pure} pure >=75% -> {len(cases)} cases")
    hits_byte, hits_path, hits_mode = 0, 0, 0
    details = []
    for ctx, nxt, rule_byte, conf in cases:
        idx = torch.tensor([int(c) for c in ctx], device=DEV).unsqueeze(0)
        hf = hand_forward(m, idx)
        parts, logits, std = margin_decomposition(m, idx, hf)
        v_model = int(np.argmax(logits))
        hits_byte += int(v_model == rule_byte)
        # path dominance: L1-wave largest positive contributor to top1-top2 margin?
        v2 = int(np.argsort(-logits)[1])
        W_head_e = m.head.weight.detach().cpu().numpy()
        margins = {k: float(W_head_e[v_model] @ p - W_head_e[v2] @ p) for k, p in parts.items()}
        dom_path = max(margins, key=margins.get)
        hits_path += int(dom_path == "L1-wave")
        # per-mode decomposition of L1-wave for the margin
        lo = hf["layers"][1]
        a, b = lo["read_gain"]
        C_last = lo["C_last"].detach().cpu().numpy(); gate = lo["gate"].detach().cpu().numpy()
        gn = m.norm.weight.detach().cpu().numpy()
        terms = lo["terms"].detach().cpu().numpy()
        re, im = terms.real, terms.imag
        an, bn = a.detach().cpu().numpy(), b.detach().cpu().numpy()
        per_mode = []
        W = m.head.weight.detach().cpu().numpy()
        for k in range(6):
            y_o = (re[:, :, k].sum(0)[:, None] * C_last[:, k][:, None] * an[:, :, k]
                   - im[:, :, k].sum(0)[:, None] * C_last[:, k][:, None] * bn[:, :, k])
            yg = y_o.sum(0) * gate  # sum over input channels, then the output gate
            v = (yg / std) * gn
            per_mode.append(float(W[v_model] @ v - W[v2] @ v))
        top_modes = set(np.argsort(-np.abs(per_mode))[:2].tolist())
        hits_mode += int(top_modes <= {1, 4})
        details.append(dict(ctx=bytes(int(c) for c in ctx[-8:]).decode("latin1"), rule=chr(rule_byte),
                            model=chr(v_model), actual=chr(nxt), dom_path=dom_path,
                            per_mode=[round(x, 2) for x in per_mode]))
    n_e = max(len(cases), 1)
    out["guided_100k"] = dict(n=len(cases), byte_rule_acc=hits_byte / n_e,
                              l1wave_dominant=hits_path / n_e,
                              top2_modes_in_dict=hits_mode / n_e, details=details[:6])
    print(f"  {len(cases)} unique-completion contexts: model==rule byte {hits_byte}/{n_e}, "
          f"L1-wave dominant {hits_path}/{n_e}, top-2 modes subset of dict(modes 1,4) {hits_mode}/{n_e}")
    for dd in details[:6]:
        print(f"    ...{dd['ctx']!r} rule {dd['rule']!r} model {dd['model']!r} actual {dd['actual']!r} "
              f"dom={dd['dom_path']} per-mode {dd['per_mode']}")

    with open("experiments/figures/w11_hand_simulation.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nsaved experiments/figures/w11_hand_simulation.json")


if __name__ == "__main__":
    main()
