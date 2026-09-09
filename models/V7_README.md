# V7 — Interpretable-by-Construction Wavelet-State-KAN

**Design rule:** every reparameterization must pass the neutrality test
(3-seed parity with V6 on UltraChat/100k) or its cost is documented as an
interpretability/performance tradeoff. Precedent: V6's factorized Δ was
exactly such a move and passed.

## The three reparameterizations

### 1. Feature-factorized B/C (`use_feature_bc`)

$$B_{n,ik} = \underbrace{\sum_f \alpha_f(\text{byte}_n)\, M^{B}_{f,ik}}_{\text{named: fixed features} \times \text{learned table}} + \underbrace{W^{B}_{\text{lr}} x_n}_{\text{rank-32 residual}}$$

Fixed named features α (8): is_space, is_newline, is_punct, is_upper,
is_lower, is_digit, is_vowel, constant. **The named part is interpretable by
lookup**: feature f's write/read pattern into every (channel, mode) is the
row M_f — no probing needed. The residual (rank 32 from the hidden state)
preserves context-dependent gating; it is the only part that still requires
probing, and it is small. C likewise.

### 2. Diagonal output gate (`wz_diag`)

$y \odot \mathrm{SiLU}(w_z \odot x)$ — per-channel gain, readable as 32
scalars per layer instead of a 32×32 matrix.

### 3. Low-rank edge gains (`g_rank`)

$$g_{iok} = \sum_{r=1}^{R} u_{ir}\, v_{or}\, m_{rk} \;\Rightarrow\; \psi_{io}(t) = \sum_r u_{ir} v_{or}\, \phi_r(t)$$

Each layer's 1,024 edge functions become **R=32 named temporal filters φ_r
with per-input/output loadings** — the edge atlas collapses from 6,144
individual functions to 32 filters plus two loading matrices (and the shared
u, v bases are exactly the "input-role / output-role" dictionaries the atlas
was meant to discover). Implemented as non-persistent buffers composed each
forward (autograd flows through u, v, m).

## Parameters

82,194 at the 100k config (d=32, L=3, N=6) — **−19.3% vs V6** (101,874):
the factored forms are intrinsically smaller. If parity holds, V7 is
*smaller and* more analyzable; if it loses, the loss conflates the
constraints with the size cut (documented honestly; a size-matched variant
is a follow-up).

## Verification

- Chunked vs sequential loop (byte-idx padded consistently): 1.2e-7.
- Adversarial σ = e⁸, ρ = e³: outputs/grads finite.
- Init loss 5.546 ≈ ln(256); 500-step real-data smoke trains (eval 1.89).

## Analysis surfaces unlocked (vs V6)

| question | V6 answer requires | V7 answer is |
|---|---|---|
| what does a space write into mode k? | probing | M_B[is_space, :, k] lookup |
| which channel's output is suppressed? | probing | w_z scalars |
| what temporal filters exist? | 6,144-kernel atlas | 32 φ_r plots |
| input/output roles | clustering | u, v loadings |

## Files

| File | Content |
|---|---|
| `models/V7_WSKAN.py` | `InterpretableWaveletStateKANLayer`, `WaveletStateKANLMV7`, `byte_features` |
| trainer flag | `--model wskan7` |
| report | `experiments/V7_ULTRACHAT_REPORT.md` (neutrality test) |
