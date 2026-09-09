"""Final interpretability battery: the five remaining debts.

  E. residual-B feature semantics   (fig_v7c_resid.png)  - what drives the content path
  F. family-to-circuit connectivity  (fig_v7c_circuit.png) - kernel families x channel clusters x hubs
  G. W_z gate structure              (fig_v7c_wz.png)
  H. embedding geometry              (fig_v7c_emb.png)
  I. exact logit attribution         (fig_v7c_logits.png) - decision margin by path

Run: .venv/bin/python experiments/V7_final_interpret.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.V1_train_tinystories_lm import load_data
from models.V7_WSKAN import WaveletStateKANLMV7, byte_features
from scipy.cluster.vq import kmeans2

OUT = Path(__file__).resolve().parent / "figures"
CKPT = Path(__file__).resolve().parent.parent / "checkpoints/wskan7bc_ultrachat_100k_s42/latest.pt"

model = WaveletStateKANLMV7(use_feature_bc=True, wz_diag=False, g_rank=None)
model.load_state_dict(torch.load(CKPT, weights_only=False)["state_dict"])
model.eval().cuda()
S = {}

train_ids, _ = load_data(200000, 500, 42, "ultrachat")
text = train_ids[:2048].numpy().astype(np.int64)
idx_probe = torch.tensor(text, device="cuda").unsqueeze(0)

hiddens = []
x = model.tok_emb(idx_probe)
with torch.no_grad():
    for norm, layer in zip(model.prenorms, model.layers):
        h = norm(x)
        hiddens.append(h[0].cpu())
        x = x + layer(h, idx_probe)

classes = [("space", lambda t: t == 32), ("upper", lambda t: (t >= 65) & (t <= 90)),
           ("lower", lambda t: (t >= 97) & (t <= 122)), ("digit", lambda t: (t >= 48) & (t <= 57)),
           ("punct", lambda t: np.isin(t, list(b".,!?;:'\"()-")))]
pos_word = np.zeros_like(text)
pw = 99
for n, t in enumerate(text):
    pw = 0 if t == 32 else pw + 1
    pos_word[n] = min(pw, 6)

def chan_profiles(H):
    prof = [H[pred(text)].mean(0) for _, pred in classes] + [H[pos_word == j].mean(0) for j in range(7)]
    P = np.stack(prof)
    return (P - P.mean(1, keepdims=True)) / (P.std(1, keepdims=True) + 1e-9)

# ============ E. residual-B write semantics: variance decomposition ============
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
S["resid_B"] = {}
top_bytes = [b for b in np.argsort(-np.bincount(text, minlength=256))[:32] if b > 0]
for li, layer in enumerate(model.layers):
    H = hiddens[li]
    with torch.no_grad():
        R = layer.res_B(H.cuda()).cpu().numpy().reshape(len(text), -1)  # (L, i*N)
    def r2(masks):
        gidx = np.zeros(len(text), dtype=int)
        for gi, m in enumerate(masks):
            gidx[m] = gi
        gm = np.stack([R[m].mean(0) for m in masks])
        pred = gm[gidx]
        return 1 - ((R - pred) ** 2).sum(0) / ((R - R.mean(0)) ** 2).sum(0)
    cls_groups = [pred(text) for _, pred in classes]
    byte_groups = [text == b for b in top_bytes]
    byte_groups.append(~np.any(byte_groups, axis=0))
    pos_groups = [pos_word == j for j in range(7)]
    prev_cls = [np.roll(cls_groups[j], 1) for j in range(len(classes))]
    res = {
        "byte_class": float(np.mean(r2(cls_groups))),
        "byte_identity_top32": float(np.mean(r2(byte_groups))),
        "position_in_word": float(np.mean(r2(pos_groups))),
        "prev_byte_class": float(np.mean(r2(prev_cls))),
    }
    S["resid_B"][f"L{li}"] = res
    axes[li].bar(res.keys(), res.values(), color="steelblue")
    axes[li].set_title(f"L{li} residual-B write variance explained")
    axes[li].tick_params(axis="x", rotation=30, labelsize=7)
fig.suptitle("What drives the content (residual) write path?")
fig.tight_layout()
fig.savefig(OUT / "fig_v7c_resid.png", dpi=130)
plt.close(fig)

# ============ F. family x channel-cluster circuit ============
lags = np.arange(48)
fig, axes = plt.subplots(3, 2, figsize=(12, 12))
S["circuit"] = {}
for li, layer in enumerate(model.layers):
    rho = torch.exp(layer.log_rho.detach().clamp(-3, 3)).cpu()
    sigma = torch.exp(layer.log_sigma.detach().clamp(-6, 8)).cpu()
    om = layer.omega.detach().cpu()
    a, b = layer.a.detach().cpu(), layer.b.detach().cpu()
    mean_dt = hiddens[li].abs()  # placeholder
    dt_layer = model.layers[li]._compute_dt  # noqa
    with torch.no_grad():
        dtm = dt_layer(hiddens[li].cuda()).mean(0).cpu()
    lg = torch.tensor(lags, dtype=torch.float32)
    env = torch.exp(-(sigma * rho).unsqueeze(-1) * lg * dtm[:, None, None])
    ph = (om * rho).unsqueeze(-1) * lg * dtm[:, None, None]
    K = (torch.einsum("ion,inl->iol", a, env * torch.cos(ph))
         + torch.einsum("ion,inl->iol", b, env * torch.sin(ph)))
    Kf = K.reshape(-1, len(lags)).numpy()
    Kn = Kf / (np.abs(Kf).max(axis=1, keepdims=True) + 1e-9)
    _, klab = kmeans2(Kn, 6, minit="++", seed=0)
    edge_in, edge_out = np.divmod(np.arange(Kf.shape[0]), layer.out_dim)

    Pn = chan_profiles(hiddens[li].numpy())
    _, clab = kmeans2(Pn.T, 5, minit="++", seed=0)

    M_in = np.zeros((6, 5))
    M_out = np.zeros((6, 5))
    for f in range(6):
        sel = klab == f
        for c in range(5):
            M_in[f, c] = np.mean(clab[edge_in[sel]] == c)
            M_out[f, c] = np.mean(clab[edge_out[sel]] == c)
    axes[li][0].imshow(M_in, cmap="viridis", aspect="auto")
    axes[li][0].set_xticks(range(5), [f"ch{c}" for c in range(5)])
    axes[li][0].set_yticks(range(6), [f"F{f}" for f in range(6)])
    axes[li][0].set_title(f"L{li}: family -> input channel cluster")
    axes[li][1].imshow(M_out, cmap="viridis", aspect="auto")
    axes[li][1].set_xticks(range(5), [f"ch{c}" for c in range(5)])
    axes[li][1].set_yticks(range(6), [f"F{f}" for f in range(6)])
    axes[li][1].set_title(f"L{li}: family -> output channel cluster")
    S["circuit"][f"L{li}_sizes"] = np.bincount(klab, minlength=6).tolist()
fig.suptitle("Circuit diagram: kernel family vs channel-cluster connectivity (row-normalized per family)")
fig.tight_layout()
fig.savefig(OUT / "fig_v7c_circuit.png", dpi=130)
plt.close(fig)

# ============ G. W_z gate structure ============
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
S["wz"] = {}
for li, layer in enumerate(model.layers):
    Wz = layer.W_z.weight.detach().cpu().numpy()  # (out, in)
    Pn = chan_profiles(hiddens[li].numpy())
    _, clab = kmeans2(Pn.T, 5, minit="++", seed=0)
    G = np.stack([Wz[clab == c].mean(0) for c in range(5)])  # (out-cluster, in-channel)
    im = axes[li].imshow(G, cmap="RdBu_r", vmin=-np.abs(G).max(), vmax=np.abs(G).max(), aspect="auto")
    axes[li].set_yticks(range(5), [f"out c{c}" for c in range(5)])
    axes[li].set_xlabel("input channel")
    axes[li].set_title(f"L{li}")
    fig.colorbar(im, ax=axes[li])
    open_frac = float((np.abs(Wz) < 0.05).mean())
    S["wz"][f"L{li}_nearzero_frac"] = open_frac
fig.suptitle("W_z: mean gate row per output-channel cluster (red=amplify, blue=suppress)")
fig.tight_layout()
fig.savefig(OUT / "fig_v7c_wz.png", dpi=130)
plt.close(fig)

# ============ H. embedding geometry ============
E = model.tok_emb.weight.detach().cpu().numpy()
Ec = E - E.mean(0)
svals = np.linalg.svd(Ec, compute_uv=False)
proj = Ec @ np.linalg.svd(Ec)[2][:2].T
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
cls_of = np.zeros(256, dtype=int)
for ci, (_, pred) in enumerate(classes):
    cls_of[pred(np.arange(256))] = ci + 1
cols = ["gray", "red", "orange", "green", "purple", "brown"]
for ci in range(6):
    sel = cls_of == ci
    axes[0].scatter(proj[sel, 0], proj[sel, 1], s=8, c=cols[ci],
                    label=["other", "space", "upper", "lower", "digit", "punct"][ci])
axes[0].legend(fontsize=7)
axes[0].set_title("byte embedding PCA (first 2 comps)")
def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
within, between = [], []
rng = np.random.default_rng(0)
for ci in range(1, 6):
    ids = np.where(cls_of == ci)[0]
    if len(ids) > 2:
        for _ in range(50):
            i, j = rng.choice(ids, 2, replace=False)
            within.append(cos(E[i], E[j]))
        i = rng.choice(ids); j = rng.choice(np.where(cls_of != ci)[0])
        between.append(cos(E[i], E[j]))
axes[1].hist(within, bins=40, alpha=0.6, label="same class")
axes[1].hist(between, bins=40, alpha=0.6, label="diff class")
axes[1].legend()
axes[1].set_title("cosine similarity: within vs between class")
S["emb"] = {"within_mean": float(np.mean(within)), "between_mean": float(np.mean(between)),
            "pc1_share": float(svals[0] ** 2 / (svals ** 2).sum()),
            "pc2_share": float(svals[1] ** 2 / (svals ** 2).sum())}
for bt in [ord("e"), ord("t"), ord(" "), ord("."), ord("5"), ord("\n")]:
    d = E @ E[bt] / (np.linalg.norm(E, axis=1) * np.linalg.norm(E[bt]) + 1e-9)
    nn = np.argsort(-d)[:4]
    S["emb"][f"nn_{repr(chr(bt))}"] = [repr(chr(int(k))) for k in nn]
fig.tight_layout()
fig.savefig(OUT / "fig_v7c_emb.png", dpi=130)
plt.close(fig)

# ============ I. exact logit attribution ============
def attribute(context: bytes, label: str):
    idx = torch.tensor(list(context), dtype=torch.long, device="cuda").unsqueeze(0)
    idx = idx[:, -256:]
    contribs = {}
    with torch.no_grad():
        x0 = model.tok_emb(idx[:, -1])          # (1, d) at final position
        h = x0
        W_head = model.head.weight.cpu()
        contribs["token-emb"] = (W_head @ h[0].cpu()).numpy()
        x = model.tok_emb(idx)
        for li, (norm, layer) in enumerate(zip(model.prenorms, model.layers)):
            hn = norm(x)
            base = (hn[0, -1] @ layer.w_base).cpu()          # (d,)
            full = layer(hn[:, -1:, :], idx[:, -1:])
            wav = (full[0, -1].cpu() - base)                  # (d,)
            contribs[f"L{li}-base"] = (W_head @ base).numpy()
            contribs[f"L{li}-wave"] = (W_head @ wav).numpy()
            x = x + layer(hn, idx)
        logits = model(idx)[0, -1].cpu().numpy()
    top2 = np.argsort(-logits)[:2]
    margin = {k: float(v[top2[0]] - v[top2[1]]) for k, v in contribs.items()}
    total = float(logits[top2[0]] - logits[top2[1]])
    print(f"[{label}] pred {chr(top2[0])!r} vs {chr(top2[1])!r} margin {total:.2f}")
    for k, v in sorted(margin.items(), key=lambda kv: -abs(kv[1])):
        print(f"   {k:>10}: {v:+.3f}")
    return margin

fig, ax = plt.subplots(figsize=(10, 5))
all_margins = {}
for i, (ctx, lab) in enumerate([
    (b"The quick brown fox jumps over the lazy dog. The dog barks and th", "th->e"),
    (b"User: Hello! How are you today, my dear frien", "friEN->d"),
    (b"Once upon a time, there was a little girl named Lily.\n\nOne day, Lily went to the park to play with her friend", "friend->s/next"),
]):
    m = attribute(ctx, lab)
    all_margins[lab] = m
    ax.barh([f"{lab}|{k}" for k in m.keys()], list(m.values()))
ax.set_xlabel("contribution to decision margin (logit gap top1 - top2)")
fig.tight_layout()
fig.savefig(OUT / "fig_v7c_logits.png", dpi=130)
plt.close(fig)
S["logit_attribution"] = all_margins

(OUT / "v7c_final_summary.json").write_text(json.dumps(S, indent=2, default=str))
print(json.dumps(S, indent=2, default=str)[:2200])
print("figures in", OUT)
