# V6 UltraChat Report: The Synthesis Works — V4 Performance, V3 Analyzability

**Date:** 2026-09-08 · **Status:** 3-seed, same UltraChat/100k protocol.
**Pre-registered claim under test:** "if V6 matches V4, per-mode
content-driven timescale gating has no value at this scale, and V6 supersedes
V4 as the analyzable choice."

## Results (best eval CE, byte-level)

| Model | s42 | s123 | s2024 | mean ± std |
|---|---|---|---|---|
| **wskan6 (factorized)** | 1.1592 | 1.1654 | 1.1578 | **1.1608 ± 0.0040** |
| wskan4 | 1.1923 | 1.1366 | 1.1531 | 1.1607 ± 0.0286 |
| wskan3 | 1.1711 | 1.1663 | 1.1510 | 1.1628 ± 0.0105 |
| mamba2 | 1.1645 | 1.1527 | 1.2203 | 1.1792 ± 0.0361 |
| wskan5 (rejected) | 1.1954 | 1.1755 | 1.1808 | 1.1839 ± 0.0103 |

Paired V6 − V4: −0.0331 / +0.0288 / +0.0047 → **mean +0.0001: dead even.**
Paired V6 − V3: mean −0.0020. V6's seed std (0.0040) is the **tightest of
any version** — 7x tighter than V4, 2.6x tighter than V3, 9x tighter than
Mamba-2.

**The pre-registered claim is confirmed:** V4's per-mode content-driven
timescale gating contributes nothing at this scale; the factorized ladder
reproduces V4's performance (and beats V3's mean with far tighter variance).

## The analyzability is real, not just claimed

1. **A clean scale ladder emerged.** The learned ρ (initialized all-1.0)
   settled into a *monotone* ladder in every layer — e.g. L2:
   [0.66, 0.47, 0.44, 0.39, 0.37, 0.35] — a single interpretable
   multi-resolution object per layer (range ~1.9x), exactly the "wavelet
   scale ladder" the factorization was designed to expose.
2. **The long-memory tail survives.** Per-mode half-life (via ρ·σ, on one
   warped time axis per channel): median 0.8–0.9 tokens, p99 3–33,
   **max 9–99 tokens** — comparable to V4's dormant tail (38–168), and
   analyzable with V3's exact toolset (single warped time, no per-mode
   time axes, no write/timescale confound).
3. V3-style edge-wavelet/kernel/spectral analysis applies verbatim on the
   effective modes λ̃ = ρλ (one warped axis per channel).

## Version table (UltraChat 100k, 3 seeds)

| Version | mean best CE | verdict |
|---|---|---|
| V3 | 1.1628 ± 0.0105 | reference architecture |
| V4 | 1.1607 ± 0.0286 | equal perf, richer but less analyzable |
| V5 | 1.1839 ± 0.0103 | rejected (regression) |
| **V6** | **1.1608 ± 0.0040** | **adopted: V4 performance + V3 analyzability + lowest variance** |

## Recommendation

V6 is the project's go-forward architecture: equal-best performance,
tightest seeds, +18 parameters over V3, and the only version where the
wavelet identity, the memory tail, and the scale ladder are all directly
readable from the checkpoint. Future work (scale-up, third dataset,
retrieval benchmarks) should build on V6.

## Reproduction

```bash
.venv/bin/python experiments/V1_train_tinystories_lm.py --model wskan6 \
    --dataset ultrachat --block 512 --steps 100000 --seed {42|123|2024} \
    --compile --lr-schedule cosine --ckpt-every 25000
```
