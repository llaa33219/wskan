# V4 UltraChat Report: Freeing the Timescales Is Neutral at 100k Scale

**Date:** 2026-09-07 · **Status:** 3-seed controlled comparison vs V3/Mamba-2
(same UltraChat protocol: block 512, 100k steps, batch 32, lr 3e-3 cosine).

## Results (best eval CE, byte-level)

| Model | s42 | s123 | s2024 | mean ± std |
|---|---|---|---|---|
| wskan4 (freed σ, per-mode Δ) | 1.1923 | **1.1366** | **1.1531** | **1.1607 ± 0.0286** |
| wskan3 | **1.1711** | 1.1663 | 1.1510 | 1.1628 ± 0.0105 |
| mamba2 | 1.1645 | 1.1527 | 1.2203 | 1.1792 ± 0.0361 |
| wskan3real | 1.2196 | 1.2208 | 1.1998 | 1.2134 ± 0.0118 |

Paired (per-seed) wskan4 − wskan3: +0.0212 / −0.0297 / +0.0021 →
**mean −0.0021: a wash.**

## The interesting part: V4 uses its freedom, but gains nothing from it

- **The freed σ range is heavily used:** 31–48% of modes now sit *beyond* the
  old V3 ceiling (σ up to 16.7, more than 2× the old cap of 7.39). In V3 these
  modes were pinned at the boundary; in V4 they drift past it freely.
- **Per-mode Δ differentiation is real:** median max/min Δ spread across modes
  is 3.3–6.0× — modes genuinely use independent timescales.
- **...and eval loss did not move** (Δ −0.002 nats vs V3, mixed signs across
  seeds). Honest conclusion: at ~100k params on this task, the σ clamp was a
  binding-but-harmless constraint — the capacity the model wanted past the
  ceiling is fast-decay (near-delta) capacity, which buys nothing on loss.
  The "binding constraint" hypothesis is **refuted** as a performance lever
  at this scale.
- The wavelet contribution persists under V4 (V4 − wskan3real: −0.0527 mean,
  3/3 seeds; note wskan4real was not run — cross-architecture reading).
- V4 vs mamba2: mean −0.0185 but mixed signs (+0.028/−0.016/−0.067);
  mamba2 s2024 was its outlier run. Same verdict as V3: mean-edge with
  better/worse seed variance depending on the pair.

## What V4 settles

1. V3's design is **not** limited by its numerical guards — V4 is the proof
   (the guards moved, the loss did not).
2. The 100k-scale performance plateau is therefore a capacity/data-scale
   property, not an artifact of the stability clamps.
3. V4 remains the strictly more principled parameterization (per-mode
   warped-time wavelets, ZOH-consistent writes) at +2.7% params, and is the
   better base for scale-up, where the 10M probe showed slower decays become
   useful. But on present evidence, **V3 and V4 are equivalent at 100k**;
   claims should cite V3's numbers unless a longer/10M run distinguishes them.

## Limitations

- Same protocol limits as before: byte-level, ~100k, UltraChat only for V4
  (TinyStories V4 run not performed).
- wskan4real ablation not run (V4's wavelet claim currently rests on the V3
  ablation plus the V4 vs wskan3real gap).

## Reproduction

```bash
.venv/bin/python experiments/V1_train_tinystories_lm.py --model wskan4 \
    --dataset ultrachat --block 512 --steps 100000 --seed {42|123|2024} \
    --compile --lr-schedule cosine --ckpt-every 25000
```
