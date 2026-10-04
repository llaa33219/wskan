"""W12 handcraft: a language model whose every parameter is written by hand
from the measured anatomy - no training.

Recipe sources (all measured, canonical monograph):
  - clock: letters Delta ~ 0.10, boundaries 0.21-0.28 (2-2.6x)
  - ladder: effective sigma geometric across modes; omega rising
  - gate split: named features carry structure, residual carries content
  - word-length rhythm: slow mode accumulates since-last-boundary

Model: 1k tier (d=4, L=1, N=6). Everything set by hand: embedding as a
functional class manifold, W_dt clock weights, mode (sigma, omega), the
feature tables, gains, gates. Goal: structurally fluent pseudo-text
(boundary rhythm, vowel/consonant texture, capitalization after '.'),
measured by eval CE vs the trained 1k model and a unigram baseline.

Run: PYTHONPATH=.:experiments .venv/bin/python experiments/W12_handcraft.py
"""

from __future__ import annotations

import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from models.V11_WSKAN import WaveletStateKANLMV11

DEV = "cuda"


def build_embedding():
    """256x4 hand-set functional manifold with balanced axes:
    ch0 = letter-ness, ch1 = vowel(+)/consonant(-), ch2 = boundary-ness,
    ch3 = sentence-terminal punctuation. Non-printable bytes get a strong
    negative prior on every channel so they never win."""
    E = np.full((256, 4), -2.0, dtype=np.float32)
    for b in range(256):
        c = chr(b)
        if b in (9, 10, 13) or 32 <= b < 127:
            E[b] = 0.0
        if c.isalpha() and b < 128:
            E[b, 0] = 1.0
            E[b, 1] = 0.8 if c.lower() in "aeiou" else -0.8
            E[b, 2] = -0.6
        if c in " \n":
            E[b, 2] = 1.0
            E[b, 0] = -0.8
        if c in ".!?":
            E[b, 2] = 0.8
            E[b, 3] = 1.0
            E[b, 0] = -0.8
    return E


def build_head():
    """Per-byte readout vectors on the same four axes, calibrated so class
    margins are decisive through the final LayerNorm. Uppercase reads ch3
    (driven only by the sentence-terminal memory)."""
    W = np.zeros((256, 4), dtype=np.float32)
    W[:] = -8.0                       # non-printables never win
    freq = " etaoinshrdlucmfwypvbgkqjxz"
    for b in range(256):
        c = chr(b)
        if not (b in (9, 10, 13) or 32 <= b < 127):
            continue
        if c.isalpha():
            tilt = 1.3 - 0.02 * freq.index(c) if c in freq else 0.6
            W[b] = [2.2 * tilt, 1.0 if c.lower() in "aeiou" else -1.0, -1.2, 0.0]
            if c.isupper():
                W[b, 0] *= 0.7
                W[b, 3] = 1.2
        elif c == " ":
            W[b] = [-0.8, 0.0, 6.0, 0.0]
        elif c == "\n":
            W[b] = [-0.8, 0.0, 4.2, 0.3]
        elif c in ".!?":
            W[b] = [-0.8, 0.0, 4.4, 1.4]
        else:
            W[b] = [-0.5, 0.0, 0.4, 0.0]
    return W


def softplus_inv(y):
    return math.log(math.expm1(y))


def handcraft():
    m = WaveletStateKANLMV11(vocab_size=256, d_model=4, n_layers=1, n_states=6,
                             use_feature_bc=True, wz_diag=False, g_rank=None,
                             bc_rank=4, bf16_scan=False).to(DEV)
    with torch.no_grad():
        m.tok_emb.weight.copy_(torch.tensor(build_embedding(), device=DEV))
        for nrm in list(m.prenorms) + [m.norm]:
            nrm.weight.fill_(1.0)
            nrm.bias.zero_()
        # prenorm bias makes h dense-positive so the output gate stays open
        # on every channel (sparse embeddings would let silu choke the wave path)
        m.prenorms[0].bias.fill_(0.5)
        layer = m.layers[0]
        I, N = 4, 6

        # clock: reads ch2 (boundary-ness) -> big dt at boundaries
        layer.W_dt.weight.zero_()
        layer.W_dt.weight[:, 2] = 2.2          # boundary channel drives dt
        base = softplus_inv(0.10)              # letters ~ 0.10
        layer.W_dt.bias.fill_(base)

        # mode ladder: geometric sigmas; omega rising on the fast modes,
        # ZERO on the slow integrators (length/sentence memory must not ring)
        sig = torch.tensor([8.0, 4.0, 2.0, 1.0, 0.5, 0.25], device=DEV)
        layer.log_sigma.data = sig.log().unsqueeze(0).expand(I, N).contiguous()
        om = torch.tensor([0.0, 2.0, 4.0, 8.0, 0.0, 0.0], device=DEV)
        layer.omega.data = om.unsqueeze(0).expand(I, N).contiguous()
        layer.log_rho.data.zero_()

        # B table (features x I x N): [space, newline, punct, upper, lower, digit, vowel, const]
        # measured mechanism: letters ACCUMULATE into slow modes; the clock
        # resets them at boundaries (big Delta -> big decay + big write).
        MB = torch.zeros(8, I, N, device=DEV)
        MB[6, 1, 0] = 1.5      # vowel -> fast mode 0 on ch1
        MB[4, 0, 1] = 0.8      # lowercase letter -> fast mode 1 (letter recency)
        MB[4, 0, 4] = 0.3      # letter-ness (ch0) -> slow mode 4 (word-length accumulate)
        MB[4, 1, 3] = 0.5      # letter -> mode 3 (mid)
        MB[2, 2, 5] = 1.5      # sentence-final punct -> slowest mode 5
        MB[0, 2, 4] = -2.0     # space -> anti-write (reset of word memory)
        MB[0, 2, 2] = -1.5     # space -> fast mode 2 (just-had-boundary damper)
        MB[1, 2, 2] = -1.5     # newline same
        MB[2, 2, 2] = -1.0     # punct same, milder
        MB[7, 0, 3] = 0.3      # const -> mid mode (mild always-on)
        layer.M_B.data = MB
        # C table: read modes at the positions where their content matters
        MC = torch.zeros(8, I, N, device=DEV)
        MC[4, 1, 0] = 1.2      # at letters: read vowel-ness of recent letters
        MC[7, 0, 4] = 2.0      # always read word-length accumulation
        MC[7, 2, 5] = 0.9      # always read sentence-terminal memory
        MC[7, 2, 2] = 1.0      # always read the just-had-boundary damper
        MC[4, 0, 1] = 0.6      # at letters: read letter recency
        MC[0, 0, 1] = -0.8     # right after a space, damp letter-recency
        layer.M_C.data = MC
        # content residuals small
        for res in (layer.res_B, layer.res_C):
            for mod in res:
                mod.weight.data *= 0.05
                if mod.bias is not None:
                    mod.bias.data.zero_()

        # gains a (I,O,N): mode k -> output channel, sign choices hand-set
        a = torch.zeros(I, I, N, device=DEV)
        b = torch.zeros(I, I, N, device=DEV)
        a[1, 1, 0] = -0.7      # recent-vowel evidence -> push next byte consonant-ward
        a[0, 0, 1] = 0.6       # letter recency -> letter-ness
        a[0, 2, 4] = 14.0      # word-length memory (ch0, mode4) -> boundary pressure
        a[2, 2, 2] = 2.0       # just-had-boundary damper -> suppress boundary right after
        a[2, 3, 5] = 1.2       # sentence-terminal memory -> push uppercase
        a[2, 2, 5] = -1.5      # sentence-terminal memory -> suppress immediate re-punct
        layer.a.data = a
        layer.b.data = b

        # output gate: uniform-open (row sums of h are positive by the bias,
        # so silu(sum) > 0 on every channel, every position)
        layer.W_z.weight.data = torch.ones(I, I, device=DEV) * 1.0
        # base skip: small identity (keep current byte class visible)
        layer.w_base.data = torch.eye(I, device=DEV) * 0.15
        layer.w_base.data[:, 2] -= 0.25   # lower the boundary baseline (letters win short words)
        layer.w_base.data[:, 3] -= 0.55   # lowercase by default; sentence memory lifts ch3

        # head: fully hand-set per-byte readout on the four axes
        m.head.weight = torch.nn.Parameter(torch.tensor(build_head(), device=DEV))
        m.head.bias = torch.nn.Parameter(torch.zeros(256, device=DEV))
        garbage = [b for b in range(256) if not (b in (9, 10, 13) or 32 <= b < 127)]
        m.head.bias.data[garbage] = -20.0
    return m.eval()


@torch.no_grad()
def eval_ce(m, ids, n=60000, blk=512):
    losses = []
    for s in range(0, n, blk):
        ch = ids[s:s + blk + 1]
        lg = m(ch[:-1].unsqueeze(0))[0]
        losses.append(F.cross_entropy(lg, ch[1:]).item())
    return float(np.mean(losses))


@torch.no_grad()
def eval_ce_static(m, ids, n=60000, blk=512):
    """Static skeleton only: wave path zeroed (a = b = 0), so the forward is
    embedding + base skip + head - no clock, no scan, no gates."""
    a_backup = m.layers[0].a.data.clone()
    b_backup = m.layers[0].b.data.clone()
    m.layers[0].a.data.zero_()
    m.layers[0].b.data.zero_()
    ce = eval_ce(m, ids, n, blk)
    m.layers[0].a.data.copy_(a_backup)
    m.layers[0].b.data.copy_(b_backup)
    return ce


def unigram_ce(train_ids, eval_ids, n=60000):
    """Unigram baseline: byte frequencies from the training stream."""
    counts = np.bincount(train_ids.numpy(), minlength=256).astype(np.float64)
    probs = (counts + 0.5) / (counts.sum() + 0.5 * 256)
    logp = torch.tensor(np.log(probs), dtype=torch.float32, device=DEV)
    tgt = eval_ids[1:n + 1].long().to(DEV)
    return float(-logp[tgt].mean())


def main():
    from experiments.V1_train_tinystories_lm import load_data
    tr_ts, ev_ts = load_data(1400000, 500, 42, "tinystories")
    m = handcraft()
    n_params = sum(p.numel() for p in m.parameters())
    print(f"hand-crafted model: {n_params} params")
    # debug probe: space-vs-letter margin as a function of word length
    with torch.no_grad():
        for k in range(1, 9):
            ctx = ("a" * k).encode()
            idx = torch.tensor(list(ctx), dtype=torch.long, device=DEV).unsqueeze(0)
            lg = m(idx)[0, -1]
            sp, lt = lg[32].item(), lg[ord('e')].item()
            print(f"  after {'a' * k!r:>10}: logit(space) {sp:+.2f} vs logit(e) {lt:+.2f}")
    print(f"TinyStories eval CE: {eval_ce(m, ev_ts.long().cuda()):.4f}")
    print(f"static skeleton only (wave path zeroed): {eval_ce_static(m, ev_ts.long().cuda()):.4f}")
    print(f"unigram baseline: {unigram_ce(tr_ts, ev_ts):.4f}")
    # texture metrics on a sampled continuation
    torch.manual_seed(0)
    idx = torch.tensor(list(b"the cat"), device=DEV).unsqueeze(0)
    with torch.no_grad():
        for _ in range(2000):
            probs = torch.softmax(m(idx)[0, -1] / 0.8, -1)
            idx = torch.cat([idx, torch.multinomial(probs, 1).view(1, 1)], 1)
    gen = idx[0, 7:].cpu().numpy()
    words = bytes(gen.tolist()).split(b" ")
    wl = [len(w) for w in words if w]
    import numpy as np
    print(f"texture: space rate {(gen == 32).mean():.2f}, lowercase {((gen >= 97) & (gen <= 122)).mean():.2f}, "
          f"mean word length {np.mean(wl):.1f} (TinyStories ~4), printable {np.mean([(32 <= b < 127 or b in (9,10,13)) for b in gen]):.3f}")
    print("t=0.8 sample:")
    print(m.generate(b"One day", max_new=300, temperature=0.8).decode("latin1"))


if __name__ == "__main__":
    main()
