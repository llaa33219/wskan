# V5 — Conv-Augmented, Geometric-Ladder Wavelet-State-KAN

**Scope:** two targeted additions over V4, each behind a flag for separate
ablation. Nothing else changed.

## Delta 1: short causal depthwise conv (flag: `use_conv`)

`x <- SiLU(Conv1d_kernel4_depthwise(x))` on the layer input before the
selective scan. Mamba-2 always had this; WSKAN never did — and the 100k-scale
evidence (a third of σ modes pinned at the fast-decay ceiling) says the
wavelet modes were being forced to impersonate a local conv. The conv takes
over local structure so the modes can specialize. Cost: d·5 params/layer
(480 total at d=32) — negligible.

Note: the conv is temporal, so it applies to recurrent mode only; the static
nominal-wavelet mode is unchanged.

## Delta 2: geometric (octave-spirit) ω ladder (flag: `omega_init`)

S4D-Lin spaces mode frequencies linearly (ω_k = πk). Wavelet multiresolution
is octave-structured. V5 spaces them geometrically over the *same* range:
ω_k = π·r^(k−1), r = N^(1/(N−1)) — for N=6: π, 4.50, 6.43, 9.21, 13.17, 18.85.
Same coverage, wavelet-shaped density (more resolution at low frequencies).

This complements the earlier ω≡0 ablation: that tested *whether oscillation
matters* (it does, −0.03 to −0.05 nats); this tests *whether the ladder shape
matters*.

## Parameters

105,120 at the 100k config (d=32, L=3, N=6) — +0.5% over V4 (104,640),
+7.7% over Mamba-2 (97,592). Documented, not hidden.

## Verification

- Chunked vs sequential loop: 1.2e-7 (conv on), 2.4e-7 (conv off).
- Adversarial σ = e⁸: outputs and grads finite.
- Init loss 5.547 ≈ ln(256).

## Results

See `experiments/V5_ULTRACHAT_REPORT.md`.

## Files

| File | Content |
|---|---|
| `models/V5_WSKAN.py` | `ConvWaveletStateKANLayer`, `WaveletStateKANLMV5` |
| trainer flags | `--model wskan5` (both), `wskan5nc` (no conv), `wskan5lin` (linear ladder) |
