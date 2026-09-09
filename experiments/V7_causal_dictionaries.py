"""Push to maximal interpretability: causal dictionaries + context + mode-level attribution.

  J. causal verification of the dictionaries (named tables, channel clusters,
     L2 hub, kernel families) - intervention battery on held-out UltraChat
  K. the unexplained 'context' in residual-B: dialogue-turn state hypothesis
  L. exact per-(channel,mode) attribution of the L2 wavelet decision

Run: .venv/bin/python experiments/V7_causal_dictionaries.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.V1_train_tinystories_lm import get_batch, load_data
from models.V7_WSKAN import WaveletStateKANLMV7
from scipy.cluster.vq import kmeans2

device = "cuda"
torch.manual_seed(0)
OUT = Path(__file__).resolve().parent / "figures"

model = WaveletStateKANLMV7(use_feature_bc=True, wz_diag=False, g_rank=None)
model.load_state_dict(torch.load("checkpoints/wskan7bc_ultrachat_100k_s42/latest.pt", weights_only=False)["state_dict"])
model.eval().cuda()
S = {}

_, eval_ids = load_data(200000, 500, 42, "ultrachat")
B, L = 32, 512
batch = torch.cat([get_batch(eval_ids, 1, L, device) for _ in range(B)], 0)
b, targets = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
IDX = b


def forward_ce(h_zero=None, out_zero=None, family_zero=None, mrow_zero=None):
    """Manual forward with interventions; returns overall CE."""
    with torch.no_grad():
        x = model.tok_emb(b)
        for li, (norm, layer) in enumerate(zip(model.prenorms, model.layers)):
            h = norm(x)
            if h_zero is not None and li in h_zero:
                h[:, :, h_zero[li]] = 0.0
            if mrow_zero is not None and li in mrow_zero:
                keep = layer.M_B[mrow_zero[li]].clone()
                layer.M_B[mrow_zero[li]] = 0.0
                out = layer(h, b)
                layer.M_B[mrow_zero[li]] = keep
            else:
                if family_zero is not None and li in family_zero:
                    a_keep = layer.a.clone(); b_keep = layer.b.clone()
                    for (i, o) in family_zero[li]:
                        layer.a[i, o] = 0.0
                        layer.b[i, o] = 0.0
                    out = layer(h, b)
                    layer.a.copy_(a_keep); layer.b.copy_(b_keep)
                else:
                    out = layer(h, b)
            if out_zero is not None and li in out_zero:
                out[:, :, out_zero[li]] = 0.0
            x = x + out
        logits = model.head(model.norm(x))
        ce = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1))
    return ce.item()


def class_ce_map():
    with torch.no_grad():
        x = model.tok_emb(b)
        for norm, layer in zip(model.prenorms, model.layers):
            h = norm(x)
            x = x + layer(h, b)
        logits = model.head(model.norm(x))
        ce = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1), reduction="none").view(B, L)
    return ce


# =================== J. causal dictionary battery ===================
print("=== J. interventions (overall CE; baseline first) ===")
base = forward_ce()
S["baseline_ce"] = base
print(f"baseline {base:.4f}")

NEWLINE, DIGIT, PUNCT = 1, 5, 2  # feature indices in byte_features
S["named_table_zero"] = {}
for name, fi in [("newline", NEWLINE), ("digit", DIGIT), ("punct", PUNCT)]:
    ce = forward_ce(mrow_zero={li: fi for li in range(3)})
    S["named_table_zero"][name] = ce
    print(f"zero M_B[{name}] rows (all layers): {ce:.4f}  ({ce-base:+.4f})")

# channel clusters -> ablate each L0 cluster and the L2 ch1 hub
text = train_ids_probe = eval_ids[:2048].numpy().astype(np.int64) if False else None
train_ids_full, _ = load_data(200000, 500, 42, "ultrachat")
probe_text = train_ids_full[:2048].numpy().astype(np.int64)
idx_probe = torch.tensor(probe_text, device="cuda").unsqueeze(0)
hiddens = []
with torch.no_grad():
    x = model.tok_emb(idx_probe)
    for norm, layer in zip(model.prenorms, model.layers):
        h = norm(x)
        hiddens.append(h[0].cpu())
        x = x + layer(h, idx_probe)

classes_f = [("space", lambda t: t == 32), ("upper", lambda t: (t >= 65) & (t <= 90)),
             ("lower", lambda t: (t >= 97) & (t <= 122)), ("digit", lambda t: (t >= 48) & (t <= 57)),
             ("punct", lambda t: np.isin(t, list(b".,!?;:'\"()-")))]
pos_word = np.zeros_like(probe_text)
pw = 99
for n, t in enumerate(probe_text):
    pw = 0 if t == 32 else pw + 1
    pos_word[n] = min(pw, 6)

def chan_labels(li):
    H = hiddens[li].numpy()
    prof = [H[pred(probe_text)].mean(0) for _, pred in classes_f] + [H[pos_word == j].mean(0) for j in range(7)]
    P = np.stack(prof)
    Pn = (P - P.mean(1, keepdims=True)) / (P.std(1, keepdims=True) + 1e-9)
    _, lab = kmeans2(Pn.T, 5, minit="++", seed=0)
    return lab

S["channel_cluster_ablate_L0"] = {}
for c in range(5):
    lab = chan_labels(0)
    chs = [int(i) for i in np.where(lab == c)[0]]
    ce = forward_ce(h_zero={0: chs})
    S["channel_cluster_ablate_L0"][f"c{c}_n{len(chs)}"] = ce
    print(f"zero L0 input cluster c{c} (n={len(chs)}): {ce:.4f}  ({ce-base:+.4f})")

ce = forward_ce(out_zero={2: [1]})
S["L2_ch1_hub_out_zero"] = ce
print(f"zero L2 output hub ch1: {ce:.4f}  ({ce-base:+.4f})")

# family ablation on L2 (zero each kernel family's edges)
lags = np.arange(48)
layer2 = model.layers[2]
with torch.no_grad():
    dtm = layer2._compute_dt(hiddens[2].cuda()).mean(0).cpu()
rho = torch.exp(layer2.log_rho.detach().clamp(-3, 3)).cpu()
sigma = torch.exp(layer2.log_sigma.detach().clamp(-6, 8)).cpu()
om = layer2.omega.detach().cpu()
a2, b2 = layer2.a.detach().cpu(), layer2.b.detach().cpu()
lg = torch.tensor(lags, dtype=torch.float32)
env = torch.exp(-(sigma * rho).unsqueeze(-1) * lg * dtm[:, None, None])
ph = (om * rho).unsqueeze(-1) * lg * dtm[:, None, None]
K = (torch.einsum("ion,inl->iol", a2, env * torch.cos(ph))
     + torch.einsum("ion,inl->iol", b2, env * torch.sin(ph)))
Kf = K.reshape(-1, len(lags)).numpy()
Kn = Kf / (np.abs(Kf).max(axis=1, keepdims=True) + 1e-9)
_, klab = kmeans2(Kn, 6, minit="++", seed=0)
edge_io = [np.divmod(e, layer2.out_dim) for e in range(Kf.shape[0])]
S["family_ablate_L2"] = {}
for f in range(6):
    edges = [edge_io[e] for e in np.where(klab == f)[0]]
    ce = forward_ce(family_zero={2: edges})
    S["family_ablate_L2"][f"F{f}_n{len(edges)}"] = ce
    print(f"zero L2 family F{f} (n={len(edges)} edges): {ce:.4f}  ({ce-base:+.4f})")

# =================== K. turn-state hypothesis ===================
print("\n=== K. what explains the residual-B 'context'? ===")
pb = probe_text.astype(np.uint8).tobytes()
role = np.zeros(len(pb), dtype=int)   # 1 = assistant turn, 2 = user turn
dist_turn = np.zeros(len(pb))
marks = []
for marker, val in ((b"Assistant:", 1), (b"User:", 2)):
    start = 0
    while True:
        i = pb.find(marker, start)
        if i < 0:
            break
        marks.append((i + len(marker), val))
        start = i + 1
marks.sort()
cur_role, cur_d = 1, 0
mi = 0
for n in range(len(pb)):
    if mi < len(marks) and n >= marks[mi][0]:
        cur_role, cur_d = marks[mi][1], 0
        mi += 1
    role[n] = cur_role
    dist_turn[n] = min(cur_d, 60)
    cur_d += 1
S["turn_state"] = {}
for li, layer in enumerate(model.layers):
    H = hiddens[li]
    with torch.no_grad():
        R = layer.res_B(H.cuda()).cpu().numpy().reshape(len(probe_text), -1)
    def r2(groups):
        groups = [g for g in groups if g.sum() > 0]
        gidx = np.full(len(probe_text), -1, dtype=int)
        for gi, m in enumerate(groups):
            gidx[m] = gi
        keep = gidx >= 0
        Rk = R[keep]
        gm = np.stack([R[m].mean(0) for m in groups])
        pred = gm[gidx[keep]]
        return float(np.mean(1 - ((Rk - pred) ** 2).sum(0) / ((Rk - Rk.mean(0)) ** 2).sum(0)))
    role_g = [role == 1, role == 2]
    dist_g = [ (dist_turn >= lo) & (dist_turn < hi) for lo, hi in [(0, 5), (5, 15), (15, 40), (40, 61)] ]
    next_cls = [np.roll(pred(probe_text), -1) for _, pred in classes_f]
    res = {"role": r2(role_g), "dist_to_turn": r2(dist_g), "next_byte_class(lookahead)": r2(next_cls)}
    S["turn_state"][f"L{li}"] = res
    print(f"L{li}: {res}")

# =================== L. exact per-(i,k) attribution of the L2 wavelet decision ===================
print("\n=== L. mode-level attribution: 'frien' -> 'd' (L2 wavelet path) ===")
ctx = b"User: Hello! How are you today, my dear frien"
idx = torch.tensor(list(ctx), dtype=torch.long, device="cuda").unsqueeze(0)
with torch.no_grad():
    x = model.tok_emb(idx)
    hs = []
    for norm, layer in zip(model.prenorms, model.layers):
        h = norm(x)
        hs.append(h)
        x = x + layer(h, idx)
    # replicate L2 scan sequentially to obtain final hidden state
    layer = model.layers[2]
    h_in = hs[2]
    sigma = torch.exp(torch.clamp(layer.log_sigma, layer.log_sigma_min, layer.log_sigma_max))
    rho = layer._rho()
    lam = torch.complex(-sigma * rho, layer.omega * rho)
    dt = layer._compute_dt(h_in)
    Bn = layer._compute_B(h_in, idx)
    Cn = layer._compute_C(h_in, idx)
    hst = torch.zeros(1, layer.in_dim, layer.n_states, dtype=torch.complex64, device="cuda")
    for n in range(h_in.shape[1]):
        a_n = torch.exp(lam * dt[:, n].unsqueeze(-1))
        hst = a_n * hst + (Bn[:, n] * dt[:, n].unsqueeze(-1) * h_in[:, n].unsqueeze(-1)).to(torch.complex64)
    C_last = Cn[0, -1]              # (i, N)
    g_re, g_im = layer.a, layer.b   # (i, o, N)
    read_re, read_im = C_last * hst[0].real, C_last * hst[0].imag     # (i, N)
    contrib_ok = torch.einsum("ik,iok->iok", read_re, g_re) - torch.einsum(
        "ik,iok->iok", read_im, g_im)  # (i, o, N)
    gate = F.silu(layer.W_z(h_in[:, -1]))[0]  # (o,)
    head_d = model.head.weight[ord("d")]      # (o,)
    per_mode = (contrib_ok * gate.unsqueeze(0).unsqueeze(-1)).sum(1)  # (o, N) -> sum over o for logit
    logit_modes = (per_mode * head_d.unsqueeze(-1)).sum(0)            # (N,)
    print("per-mode logit contribution for 'd':", [f"{v:+.3f}" for v in logit_modes.tolist()])
    top_k = torch.topk(logit_modes.abs(), 3)
    S["mode_attribution_frien_d"] = {f"mode{k}": float(logit_modes[k]) for k in top_k.indices.tolist()}
    # also which (i,k) pairs dominate
    flat = (contrib_ok * gate.unsqueeze(0).unsqueeze(-1)).sum(1) * head_d.unsqueeze(-1)  # (o,N)... per o
    pair = (contrib_ok * gate.unsqueeze(0).unsqueeze(-1) * head_d.unsqueeze(0).unsqueeze(-1)).sum(1)  # (i, N)
    top_pair = torch.topk(pair.abs().flatten(), 5)
    S["top_ik_pairs"] = {f"i{int(i)},k{int(k)}": float(pair[i, k])
                         for i, k in [np.divmod(t.item(), layer.n_states) for t in top_pair.indices]}
    print("top (i,k) pairs:", S["top_ik_pairs"])

(OUT / "v7d_causal_summary.json").write_text(json.dumps(S, indent=2))
print("\nsummary ->", OUT / "v7d_causal_summary.json")
