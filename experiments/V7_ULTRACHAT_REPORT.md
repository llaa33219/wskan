# V7 UltraChat Report: The Price of Interpretability-by-Construction

**Date:** 2026-09-09 · **Protocol:** UltraChat/100k steps/3 seeds, identical
to V6/Mamba-2 runs. **Design rule (pre-registered in V7_README):** each
reparameterization passes a neutrality test or its cost is documented.

## Results (best eval CE)

| model | params | change | mean ± std | paired vs V6 |
|---|---|---|---|---|
| wskan6 (reference) | 101,874 | — | 1.1608 ± 0.0040 | — |
| wskan7bc16 | 96,882 | feature-B/C, rank-16 residual | 1.1954 ± 0.0121 | **+0.0346 (3/3 worse)** |
| wskan7bc | 118,386 | feature-B/C, rank-32 residual | **1.1446 ± 0.0220** | −0.0162 (2/3 better) |
| wskan7z (s42 only) | ~100k | diagonal W_z only | 1.1803 | +0.0211 |
| wskan7g (s42 only) | ~90k | low-rank g only | 1.2368 | **+0.0776** |
| wskan7 (s42→3 seeds) | 82,194 | all three | 1.2473 ± 0.0321 | +0.0865 (3/3 worse) |

## Findings

1. **The cost attribution is clean.** The V7 regression (+0.087) is almost
   entirely the low-rank **g** constraint (+0.078 alone): collapsing 6,144
   edge functions into 32 shared-basis filters is a real expressivity cut —
   the edge-gain structure V6 learns is genuinely high-rank across modes.
   Diagonal W_z costs a little (+0.021); feature-B/C costs nothing.
2. **B/C lookup-interpretability is free at rank 32 / +16% params**
   (wskan7bc: −0.016 vs V6, and beats Mamba-2's 1.1792 despite the framing).
   At strict parameter parity (rank 16, 96.9k ≈ Mamba-2's 97.6k) it costs
   +0.035 nats (PPL 3.19 → 3.30). The interpretability/performance frontier
   for the write/read gates is therefore *quantified*:
   | residual rank | params | ΔCE vs V6 |
   |---|---|---|
   | 32 (full-ish) | +16% | −0.016 (free) |
   | 16 | −5% | +0.035 |
3. **What this buys:** in wskan7bc, "what does a space write into mode k of
   channel i" is the table lookup M_B[is_space, i, k] — no probing. The
   only probing left is the small residual path.

## Verdict

- **Adopted for analysis work:** `wskan7bc` (V6 + feature-factorized B/C at
  rank 32). Zero loss, 36% of the model's largest unexplained block becomes
  lookup-interpretable, and the causal-test machinery (clamp/inject) applies
  identically to the named tables.
- **Rejected:** low-rank g (cost ≫ benefit; the V6 edge population should be
  analyzed post-hoc via SVD/atlas instead), full V7 combo.
- **Not adopted:** diagonal W_z (small cost, 3% of params — not worth it).

## Incident log

- wskan7bc s2024 crashed twice during compile: first the real cause was
  **/tmp full (55 GB of torchinductor cache from the session's many
  compiled runs)** → "No space left on device". Cleaned; runs resumed.
- One config iteration: LM constructor initially missed the bc_rank
  passthrough (instant TypeError, no time lost).

## Reproduction

```bash
.venv/bin/python experiments/V1_train_tinystories_lm.py --model {wskan7bc|wskan7bc16} \
    --dataset ultrachat --block 512 --steps 100000 --seed {42|123|2024} \
    --compile --lr-schedule cosine
```
