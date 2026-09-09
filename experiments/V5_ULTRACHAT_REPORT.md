# V5 UltraChat Report: Both Additions Fail to Help — Architecture Iteration at 100k Is Closed

**Date:** 2026-09-08 · **Status:** 3-seed main + single-seed ablations, same
UltraChat/100k protocol as V3/V4 runs.

## Results (best eval CE, byte-level)

| Model | s42 | s123 | s2024 | mean ± std |
|---|---|---|---|---|
| **wskan5 (conv + geometric ladder)** | 1.1954 | 1.1755 | 1.1808 | 1.1839 ± 0.0103 |
| wskan4 (V5 minus both) | **1.1923** | **1.1366** | **1.1531** | **1.1607 ± 0.0286** |
| wskan3 | 1.1711 | 1.1663 | 1.1510 | 1.1628 ± 0.0105 |
| mamba2 | 1.1645 | 1.1527 | 1.2203 | 1.1792 ± 0.0361 |
| wskan3real | 1.2196 | 1.2208 | 1.1998 | 1.2134 ± 0.0118 |

Paired V5 − V4: +0.0031 / +0.0389 / +0.0277 → **mean +0.0232 (V5 worse, 3/3
seeds).**

Single-seed ablations (s42): wskan5 1.1954 · wskan5nc (no conv) 1.1911 ·
wskan5lin (linear ladder) 1.2017.

## Findings

1. **The short conv does not help WSKAN** (1.1954 vs 1.1911 without — the
   difference is in the wrong direction and within seed noise). The
   "wavelet modes were impersonating a conv" hypothesis is *not* supported:
   giving them a dedicated conv changed nothing. Unlike Mamba, WSKAN's local
   feature extraction through near-delta wavelet modes appears already
   sufficient (or the bottleneck lies elsewhere).
2. **Geometric vs linear ω ladder: no clear effect** (1.1954 vs 1.2017,
   −0.006, single seed, noise-level). The ladder *shape* does not matter on
   this benchmark; only oscillation itself does (per the earlier ω≡0
   ablation).
3. **V5 is a regression vs V4** (3/3 paired). Adding two components that
   individually do nothing apparently cost a little optimization noise —
   a common pattern, and the reason single-variable ablation discipline
   matters.

## Verdict: architecture iteration at 100k is closed

The pre-registered warning from the V4 report ("if 1+2 don't help, stop
iterating at 100k") is now triggered by evidence:

- V3 → V4 (parameterization freedom): neutral.
- V4 → V5 (structural conv + ladder): neutral-to-negative.
- Meanwhile V3's core results stand: parity-or-better vs Mamba-2 and a real,
  replicated wavelet contribution (−0.03/−0.05 nats vs ω≡0).

**WSKAN's architecture iteration is saturated at this scale.** Further gains
must come from the remaining levers: (a) scale — the 10M probe showed memory
and behavior change qualitatively with capacity; (b) a third dataset to
satisfy the project's ≥3-dataset rule; (c) tasks that reward long memory
(explicit retrieval/needle benchmarks) if the dormant long-memory tail is
ever to be exercised.

V3 remains the reference architecture for any future claim (best-verified
numbers); V4 is the more principled parameterization at equal performance;
V5 is not adopted.

## Reproduction

```bash
for m in wskan5 wskan5nc wskan5lin; do
  for s in 42 123 2024; do  # ablations used s42 only
    .venv/bin/python experiments/V1_train_tinystories_lm.py --model $m \
      --dataset ultrachat --block 512 --steps 100000 --seed $s \
      --compile --lr-schedule cosine --ckpt-every 25000
  done
done
```
