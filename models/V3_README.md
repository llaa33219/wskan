# V3 — Selective Wavelet-State-KAN

**The point of V3: full selectivity without abandoning the wavelet.**
Mamba's Δ (input-dependent step size) is, in wavelet language, a
**content-driven dynamic dilation**. V3 makes that literal.

## Mathematics

Per input channel $i$, a bank of $N$ damped oscillators
$\lambda_{ik} = -\sigma_{ik} + i\omega_{ik}$ (S4D-Lin init:
$\sigma = \tfrac12$, $\omega_k = \pi k$) defines the wavelet skeleton
$\psi_{ik}(t) = \mathrm{Re}[e^{\lambda_{ik} t}]$. Edge $(i,o)$ mixes the modes
with learned complex gains $g_{iok} = a_{iok} + i\,b_{iok}$:

$$\psi_{io}(t) = \sum_k \mathrm{Re}\!\left[g_{iok}\, e^{\lambda_{ik} t}\right].$$

The recurrent pass is a selective scan with all three of Mamba's
input-dependent quantities:

$$\Delta_{n,i} = \min(\mathrm{softplus}(W_\Delta x_n),\ 1),\qquad
B_n = W_B x_n,\qquad C_n = W_C x_n,$$
$$h_{n,ik} = e^{\lambda_{ik}\Delta_{n,i}}\, h_{n-1,ik} + B_{n,ik}\, x_{n,i},$$
$$y_{n,o} = \sum_{i,k} C_{n,ik}\, \mathrm{Re}[g_{iok} h_{n,ik}]
\ \odot\ \mathrm{SiLU}(W_z x_n) \ +\ x_n W_{\text{base}}.$$

**Closed form = content-warped wavelet transform.** With warped time
$T_n = \sum_{j\le n}\Delta_j$,

$$h_{n,ik} = \sum_{m\le n} e^{\lambda_{ik}(T_n - T_m)}\, B_{m,ik} x_{m,i},$$

i.e. the edge applies its wavelet to the input with the time axis warped by
the content itself. Token "distance" is no longer position but accumulated
dilation — this is the exact sense in which Mamba's Δ and wavelet dilation
are the same operation.

## Computation (chunked SSD scan)

Within a chunk of length $C=16$, outputs are computed from the causal decay
matrix $D_{nm} = e^{\lambda(T_n - T_m)}$ ($m \le n$); chunks carry state
sequentially. Two numerical guarantees:

- $T_n - T_m \ge 0$ inside the causal mask and $\mathrm{Re}\,\lambda < 0$
  (via $\sigma = e^{\log\sigma} > 0$), so every exponent has non-positive
  real part — **no overflow is possible**, in contrast to naive
  cumulative-product scans.
- $\sigma \in [e^{-4}, e^2]$ (same underflow guard as V1/V2) and
  $\Delta \le 1$.

The hot path is implemented in real arithmetic as batched `bmm`s (inductor
cannot codegen complex ops); the chunked path matches the sequential loop
reference to 4e-6.

## What V3 keeps and what it gives up (honest list)

- **Keeps:** wavelet skeleton ($\lambda$, $g$) learned and shared with the
  static mode; unconditional stability; the static wavelet edge
  `forward_static` unchanged in spirit (nominal $\psi_{io}$ with per-edge
  dilation/translation).
- **Gives up:** the LTI convolution-kernel identity and the FFT path. The
  kernel exists only for the nominal system ($\Delta$, $B$, $C$ fixed);
  selectivity is inherently time-varying. This is the deliberate price for
  full selectivity, paid knowingly after the V2 result.
- **Granularity note:** state lives per (channel, mode), not per edge —
  per-edge time-varying state is memory-infeasible. Edge functions enter
  through the gain mixing $g_{iok}$.

## Parameters (LM config: d=32, 3 layers, N=6, byte vocab 256, tied head)

101,856 total (+4.3% over Mamba-2's 97,592; includes 3 pre-norm LayerNorms).
Slightly above the nominal 100k budget — documented, not hidden.

## Stability: pre-norm is load-bearing

`WaveletStateKANLMV3` uses pre-norm residual blocks (`x + layer(norm(x))`).
Without inter-layer normalization, the multiplicative output gate makes layer
gain superlinear in residual scale, and the residual stream can blow up
multiplicatively across layers (observed: 55 → 2.5e6 → 5.7e25 → NaN at
~step 54k, deterministic for one seed). Pre-norm removes the feedback path.
See `experiments/V3_MULTISEED_ABLATION_REPORT.md` for the full incident log.

## Files

| File | Content |
|---|---|
| `models/V3_WSKAN.py` | `SelectiveWaveletStateKANLayer`, `WaveletStateKANLMV3` |
| `experiments/V1_train_tinystories_lm.py` | shared trainer (`--model wskan3 --compile`) |
| `experiments/V3_COMPARISON_REPORT.md` | 100k-step four-way comparison |
