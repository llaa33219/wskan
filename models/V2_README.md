# V2 — Gated (Selective) Wavelet-State-KAN

**Motivation.** The V1 matched-parameter comparison against Mamba-2 on
TinyStories (see `experiments/V1_MAMBA_COMPARISON_REPORT.md`) showed Mamba-2
ahead by 0.74 nats. The diagnosed gap: V1's recurrent edges are **LTI** —
after training, every edge applies a fixed wavelet kernel to every input.
Mamba's advantage is *selectivity*: its B, C, Δ are functions of the current
input, so it can choose what to store and what to forget.

**V2 adds that selectivity while preserving the wavelet identity.**

## Mathematics

Each edge keeps the V1 wavelet-SSM (diagonal $\lambda_k = -\sigma_k + i\omega_k$,
gain $g_k$, scale $s$, all shared with the static wavelet mode). Around the
LTI core, V2 adds H3/Mamba-style multiplicative gates:

$$u_n = x_n \odot \mathrm{SiLU}(W_g x_n) \qquad\text{(input gate)}$$
$$h_n = \bar{A} h_{n-1} + \bar{B} u_n,\qquad \bar A_k = e^{\lambda_k s} \qquad\text{(LTI wavelet-SSM, unchanged)}$$
$$y_n = \big(g \cdot h_n\big) \odot \mathrm{SiLU}(W_z x_n) + W_{\text{base}} x_n \qquad\text{(output gate + direct pass)}$$

What this buys:

- **Content-dependent write/read.** Which bytes enter the state and which
  state components reach the output now depend on the input — the mechanism
  V1 lacked. This is the H3 recipe (fixed SSM + multiplicative gating), which
  is known to close much of the gap to attention on language.
- **The wavelet identity is untouched.** The state dynamics are still LTI, so
  the edge kernel is still the SSM impulse response $K_l = \mathrm{Re}[g\bar B
  \bar A^l] \approx \Delta\,\psi(l\Delta)$, the FFT path still applies
  (verified numerically vs. the sequential reference to 2.4e-7), and the
  stability guarantee $|\bar A_k| < 1$ still holds.
- **Static mode unchanged** — the layer inherits V1's `forward_static`.

## What V2 deliberately does NOT do

Fully per-step input-dependent $\Delta_n, B_n, C_n$ (true Mamba selectivity)
would make the system time-varying: the convolution-kernel identity and the
FFT path would no longer hold, and the "wavelet on the edge" story would
weaken to "SSM on the edge". That trade is the **V3 candidate** (selective
scan / chunked SSD with wavelet-initialized dynamics).

## Parameters (LM config: d=32, 3 layers, N=6, byte vocab 256, tied head)

| Component | Params |
|---|---|
| Token embedding (tied) | 8,192 |
| 3 × (V1 edge layer 28,672 + W_g 1,024 + W_z 1,024) | 92,160 |
| Final LayerNorm | 64 |
| **Total** | **100,416** |

That is +2.9% over the Mamba-2 baseline (97,592) and +6.5% over V1 (94,272):
if V2 beats Mamba-2 the reading is conservative; if it loses, parameters are
not the excuse.

## Stability and bounds

Same clamps as V1 ($\sigma, s \in [e^{-4}, e^{2}]$); verified finite outputs
and gradients at $\log\sigma = \log s = \pm 10$.

## Files

| File | Content |
|---|---|
| `models/V2_WSKAN.py` | `GatedWaveletStateKANLayer`, `WaveletStateKANLMV2` |
| `experiments/V1_train_tinystories_lm.py` | shared training script (`--model wskan2`) |
| `experiments/V2_COMPARISON_100K_REPORT.md` | 100k-step three-way comparison |
