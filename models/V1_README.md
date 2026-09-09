# V1 — Wavelet-State-KAN (WSKAN)

**Core idea.** In a standard Wavelet-KAN, each edge function is a fixed mother
wavelet (e.g. Morlet, Mexican hat) with learnable dilation and translation.
In WSKAN, the wavelet on each edge is instead defined as the **impulse response
of a stable linear state-space model (SSM)**. This makes the wavelet
*mathematically state-space-compatible*: the same parameters admit both a
closed-form wavelet evaluation (static mode) and a recurrent SSM scan
(recurrent mode).

## Mathematics

Each edge $(i,j)$ carries $N$ damped oscillator modes — a diagonal
continuous-time SSM

$$h'(t) = A h(t) + B u(t), \qquad A = \mathrm{diag}(\lambda_1,\dots,\lambda_N),
\qquad \lambda_k = -\sigma_k + i\omega_k,\ \sigma_k > 0.$$

Its impulse response is a **damped-oscillation wavelet**

$$\psi(t) = \sum_{k=1}^{N} \mathrm{Re}\!\left[g_k e^{\lambda_k t}\right]
= \sum_{k=1}^{N} e^{-\sigma_k t}\big(a_k \cos \omega_k t + b_k \sin \omega_k t\big),
\qquad g_k = a_k + i b_k \; (= C_k B_k).$$

### Static mode (WavKAN-style edge)

$$\phi(x) = w_{\text{base}}\,\mathrm{SiLU}(x) +
\frac{w_{\text{wav}}}{\sqrt{s}}\;\psi\!\left(\frac{x-\mu}{s}\right),$$

with learnable dilation $s = e^{\log s} > 0$ and translation $\mu$.
The $1/\sqrt{s}$ factor preserves wavelet $L_2$ energy under dilation.

### Recurrent mode (SSM-style edge)

Zero-order-hold discretization with step size $\Delta = s$:

$$h_n = \bar{A} h_{n-1} + \bar{B} x_n, \qquad
\bar{A}_k = e^{\lambda_k s},\quad \bar{B}_k = \frac{\bar{A}_k - 1}{\lambda_k},
\qquad y_n = w_{\text{wav}}\,\mathrm{Re}[g \cdot h_n] + w_{\text{base}} x_n.$$

The **wavelet scale $s$ is the SSM step size** — dilation and SSM timescale
are the same parameter. The translation $\mu$ has no recurrent analogue and is
unused in recurrent mode.

### Acceleration: the recurrent mode is a convolution

The recurrence is linear and time-invariant, so its output equals the causal
convolution of the input with the SSM impulse response (kernel)
$K_l = \mathrm{Re}[g\,\bar{B}\,\bar{A}^l]$:

$$y_n = w_{\text{wav}} \sum_{l=0}^{n} K_l\, x_{n-l} + w_{\text{base}} x_n.$$

`forward_recurrent` therefore evaluates the whole sequence via **FFT in
$O(L \log L)$** — the same trick S4 uses. The naive $O(L)$ Python loop is
kept as `forward_recurrent_loop` for streaming use and as a numerical
reference; the two agree to ~2e-7 (float32) and the FFT path is 56–299x
faster in our measurements (batch 16, L = 200–4000, single RTX 4070).

**Interpretation:** for small $\Delta$, $K_l \approx \Delta\,\psi(l\Delta)$
— the recurrent mode is (approximately) a discrete wavelet transform of the
input with the edge's own wavelet. The correspondence is not exact: the ZOH
factor $\bar{B}$ shifts the kernel and the static-mode DC correction is not
part of the recurrence; measured median deviation is ~1.6% at the default
initialization.

### Stability (guaranteed)

$\sigma_k = e^{\log\sigma_k} > 0$ by parameterization, hence
$|\bar{A}_k| = e^{-\sigma_k s} < 1$. The recurrence is unconditionally stable
for any learned parameters — no eigenvalue can drift outside the unit circle.

### Wavelet admissibility (zero mean)

A mother wavelet must satisfy $\int \psi = 0$. The sine part is odd and
integrates to zero; the cosine part integrates to
$\frac{2\sigma_k}{\sigma_k^2 + \omega_k^2}$. We subtract a closed-form DC
correction using a unit-mass envelope $\bar{g}(t) = \frac{\bar\sigma}{2}
e^{-\bar\sigma |t|}$ of the same family:

$$\tilde\psi(t) = \psi(t) - \Big(\textstyle\sum_k a_k
\frac{2\sigma_k}{\sigma_k^2+\omega_k^2}\Big)\,\bar{g}(t),
\qquad \bar\sigma = \tfrac{1}{N}\textstyle\sum_k \sigma_k.$$

**Honest caveat:** the integral identity is exact for the pure envelope
$e^{-\sigma|t|}$. In code, $|t|$ is smoothed to $\sqrt{t^2+\varepsilon}$
($\varepsilon = 10^{-2}$) to keep $\psi$ differentiable at 0, so the zero-mean
property holds approximately, not exactly. The residual DC is $O(\varepsilon)$.

### Initialization

S4D-Lin style: $\sigma_k = \tfrac12$, $\omega_k = \pi k$ for $k = 1..N$,
which spreads the modes across frequencies; gains $a_k, b_k \sim \mathcal N(0, 0.1^2)$.

## Usage

```python
from models.V1_WSKAN import WaveletStateKAN

model = WaveletStateKAN([1, 8, 1], n_states=8)
y = model(x)            # x: (B, 1)        -> static wavelet edges
y = model(x_seq)        # x: (B, L, 1)     -> recurrent SSM edges
```

The mode is dispatched on input rank: 2D = static, 3D = recurrent.

## Limitations (stated honestly)

- The FFT path needs the full sequence up front (no streaming); use
  `forward_recurrent_loop` for online inference.
- $\mu$ is ignored in recurrent mode, so static and recurrent modes are not
  bit-identical reductions of each other (kernel vs. static wavelet differ
  by ~1.6% at default init, from the ZOH factor and DC correction).
- The two-sided wavelet extends the causal SSM impulse response via the
  smoothed even envelope $e^{-\sigma\sqrt{t^2+\varepsilon}}$; this is a
  design choice, not a canonical SSM object.
- No benchmark results yet. V1 is an architecture drop with a sanity check
  (`experiments/V1_sanity_check.py`) plus a TinyStories LM training report
  (`experiments/V1_TINYSTORIES_REPORT.md`), not an empirical claim of
  superiority.
- **Learnable-scale bounds:** $\sigma$ and $s$ are clamped to
  $[e^{-4}, e^{2}] \approx [0.018, 7.39]$. Without the clamp, $e^{-\sigma s}$
  can underflow to exact 0 in float32 and the complex power $\bar{A}^l$
  yields NaN gradients via $\log 0$ in its backward pass — this happened in
  practice at ~step 32,500 of a long LM run. The clamp restricts the
  learnable wavelet scales to this range.

## Files

| File | Content |
|---|---|
| `models/V1_WSKAN.py` | `WaveletStateKANLayer`, `WaveletStateKAN` |
| `models/V1_LM.py` | `WaveletStateKANLM` — byte-level LM wrapper (embedding + recurrent-mode layers with residuals + LayerNorm + tied head). Residuals/LayerNorm are a pragmatic, documented deviation from pure KAN. |
| `experiments/V1_sanity_check.py` | 1D multi-frequency fitting sanity check |
| `experiments/V1_train_tinystories_lm.py` | TinyStories byte-level LM training (94,272 params) |
| `experiments/V1_TINYSTORIES_REPORT.md` | training report with honest limitations |
