# V6 — Factorized-Timescale Wavelet-State-KAN

**The synthesis version:** V4's timescale freedom on V3's analyzable
skeleton. Motivated by the interpretability comparison (V3 = analyzable
wavelet, V4 = rich memory structure but diluted identity) and the factorized
dilation argument.

## Design

$$\Delta_{n,i,k} = \Delta^{\text{dyn}}_{n,i} \times \rho_k$$

- $\Delta^{\text{dyn}}_{n,i}$: input-dependent, per channel (V3's content
  warp; $W_\Delta: d \to d$).
- $\rho_k$: static learnable ladder, one per layer shared across channels
  ($N$ params; init $\rho=1$; bounded $[e^{-3}, e^3]$).

$\rho$ is treated as **eigenvalue scaling** ($\tilde\lambda_{ik} = \rho_k
\lambda_{ik}$), so the write $u = B\,\Delta^{\text{dyn}}\,x$ is the exact
first-order ZOH form at channel granularity — V4's ZOH consistency without
per-mode write/timescale coupling.

## What each parent contributes

| From V3 (analyzability) | From V4 (capability) |
|---|---|
| single warped time per channel | freed σ bounds [e⁻⁶, e⁸] |
| edge = genuine wavelet on that axis | multi-resolution ladder ρ_k |
| kernel/spectrum tools apply verbatim (on $\tilde\lambda = \rho\lambda$) | long-memory slow modes: half-life$_k$ = 0.693/(ρ_kσ_k·Δ̄_i) |
| pure content gate B | ZOH-consistent write |

Given up (measured-unused): per-mode content-driven timescale gating.
**Falsifiable claim:** if V6 matches V4 on benchmarks, that gating has no
value at this scale and V6 supersedes V4 as the analyzable choice.

## Parameters

101,874 at the 100k config — V3 (101,856) + 18 (three ρ ladders). Cheapest
version delta in the project.

## Verification

- Chunked vs sequential loop: 3.6e-7.
- Adversarial σ = e⁸, ρ = e³: outputs and grads finite.
- Init loss 5.547 ≈ ln(256).

## Files

| File | Content |
|---|---|
| `models/V6_WSKAN.py` | `FactorizedWaveletStateKANLayer`, `WaveletStateKANLMV6` |
| trainer flag | `--model wskan6` |
