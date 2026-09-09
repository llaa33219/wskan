# Memory Comparison: V4 vs Mamba-2 at Matched 100k Scale (UltraChat)

**Date:** 2026-09-07 · **Question:** does Mamba-2 actually use long-range
token memory at the same scale where V4's long-memory modes stay dormant?

## Method

Same probe as `V4_MEMORY_ANALYSIS.md`: per-head effective half-life
0.693/(|A_h|·⟨Δ_h⟩) on a real prompt, from
`checkpoints/mamba2_ultrachat_100k_s42/latest.pt` (HF Mamba-2, 97,592 params;
per-head A = −exp(A_log), input-dependent Δ via in_proj + dt_bias, softplus).

## Mamba-2 per-head half-life (tokens)

| Layer | median | max |
|---|---|---|
| L0 | 0.81 | 128.0 |
| L1 | 0.32 | 180.6 |
| L2 | 2.50 | 17.6 |

V4 reference (per-mode): median 0.83–0.97, p99 3–50, max 38–168.

## Verdict: structurally the same, behaviorally the same

1. **Mamba-2 has the same long-tail structure**: a few heads with 128–180
   token half-life, bulk under ~1 token. V4's 168-token tail is not unique —
   matched-size Mamba-2 grows an equivalent tail.
2. **Mamba-2 also fails the recall probe** (memorize number + name →
   interleaved story → recall): it produces fluent generic text instead of
   the stored facts, just like V3/V4.
3. Therefore the difference between the architectures at this scale is not
   "who can look far" — neither does, beyond latent tails — but **how well
   the near-local computation is shaped**, where the wavelet modes give V3/V4
   their measured edge (−0.03 to −0.05 nats).
4. Implication for scale-up: both architectures carry dormant long-memory
   machinery; whether it activates is a scale/question for 10M+ runs, equally
   open for both.

## Caveats

- Half-life is a per-head/per-mode *capacity* measure; actual information
  routing depends on learned B/C usage, which this probe does not isolate.
- dt_limit in this transformers version defaults to (0, inf) — Mamba-2's Δ
  was unclamped here; our V4 caps Δ ≤ 1. Noted for protocol completeness.
