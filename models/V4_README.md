# V4 — Free-Timescale Wavelet-State-KAN

**One theme: free the wavelet timescales.** V4 changes nothing else.

## Motivation (from measured evidence, not taste)

1. **The σ clamp was binding.** In the V3 TinyStories run, 33% of decay modes
   sat exactly at the σ ceiling (e² ≈ 7.39); on UltraChat, 32–43%. The clamp
   was introduced as an underflow guard for the V1/V2 `Abar ** lags` kernel
   path. V3's chunked-SSD form never exponentiates positive arguments (the
   causal decay matrix has Re ≤ 0 everywhere), so the guard's reason no
   longer exists — only its constraint remained.
2. **Δ was per-channel.** V3 gave every token one dilation per channel, so
   all N wavelet modes on a channel shared one timescale. Wavelet theory is
   multi-resolution: modes at different frequencies should be allowed their
   own time warping.
3. **V3 silently dropped the ZOH input gain.** V1/V2 used
   B̄ = (Ā−1)/λ ≈ Δ·B; V3 used u = B·x. V4 restores the Δ factor
   (u = B·Δ·x), matching both the discretization and Mamba's `dt·B`.

## Deltas from V3 (complete list)

| Change | V3 | V4 |
|---|---|---|
| σ bounds | [e⁻⁴, e²] | [e⁻⁶, e⁸] |
| Δ granularity | per (token, channel) | per (token, channel, mode) |
| Δ projection | W_Δ: d→d | low-rank d→8→d·N |
| Write | u = B·x | u = B·Δ·x |

Warped time becomes per-mode: T_{n,ik} = Σ_j Δ_{j,ik}, and

$$h_{n,ik} = \sum_{m\le n} e^{\lambda_{ik}(T_{n,ik}-T_{m,ik})}\, B_{m,ik}\,\Delta_{m,ik}\, x_{m,i}$$

— each wavelet mode performs its own content-warped transform.

## Parameters

- 100k-scale config (d=32, L=3, N=6): **104,640** (+2.7% over V3's 101,856,
  +7.2% over Mamba-2's 97,592 — documented, not hidden).
- 10m-scale config (d=256, L=5, N=6): **9,666,048**.

## Verification

- Chunked scan vs sequential loop reference: max error 4.2e-7.
- Adversarial σ = e⁸ (previously fatal territory): outputs and grads finite.
- Init loss 5.567 ≈ ln(256) (calibrated).

## Results

See `experiments/V4_ULTRACHAT_REPORT.md` (3-seed, 100k steps, UltraChat,
identical protocol to the V3 comparison runs).

## Files

| File | Content |
|---|---|
| `models/V4_WSKAN.py` | `FreeWaveletStateKANLayer`, `WaveletStateKANLMV4` |
| trainer flag | `--model wskan4` |
