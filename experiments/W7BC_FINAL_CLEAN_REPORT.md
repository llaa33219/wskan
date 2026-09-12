# Final Clean Report: All Model Comparisons on the Fresh Interp Tier

**Date:** 2026-09-11 · **Authority:** this report supersedes the scattered
era reports for *numeric* claims. It is computed exclusively from the
`checkpoints/*_interp_*` tier, regenerated 2026-09-10..11 with a single code
revision, single protocol per dataset, three seeds {42, 123, 2024}, cosine
LR, lr 3e-3, grad clip 1.0, 100k steps. Era reports remain for incident
history and interpretability narratives.

## Provenance (what is and isn't this data)

- **All numbers below come from 37 freshly regenerated runs** (mamba2 ×6,
  wskan7bc ×3 from the first clean wave; wskan3/3real/4/6/7/7bc16/7g/7z ×20
  ultrachat + wskan/2/3/3real ×8 tinystories from the second wave). Same
  trainer, same week, no mid-run code edits.
- Per-seed values differ from era reports by up to ~0.03 CE — GPU-model
  nondeterminism, within seed variance. Those earlier numbers remain the
  historical record of their runs; **the interp tier is the canonical
  reference from here on.**
- Protocols: TinyStories block 256, UltraChat block 512, else identical.

## 1. TinyStories (3 seeds, best eval CE)

| model | CE | notes |
|---|---|---|
| wskan (V1) | 1.5001 | LTI wavelet-SSM edges, no selectivity |
| wskan2 (V2) | 0.9856 | + static gates |
| **wskan3 (V3)** | **0.8101 ± 0.0132** | + full selectivity |
| wskan3real (ω≡0) | 0.8526 ± 0.0180 | ablation control |
| mamba2 | 0.8194 ± 0.0107 | reference SSM |

**Version ladder (the core mechanistic finding, replicated):** removing
selectivity costs 0.69 nats (V1→V3); the wavelet structure costs
**0.0425 nats** (V3 vs ω≡0, paired per seed: −0.0350 / −0.0400 / −0.0525,
3/3).

**wskan3 vs mamba2:** mean −0.0093 (wskan3 better) but paired by seed:
−0.0079 / −0.0311 / +0.0112 — **statistically tied** at 100k on this
dataset (consistent with the era conclusion).

## 2. UltraChat (3 seeds, best eval CE)

| model | CE | |
|---|---|---|
| **wskan7bc (V7 adopted)** | **1.1711 ± 0.0274** | best of the WSKAN series |
| wskan4 (V4) | 1.1837 ± 0.0254 | freed σ, per-mode Δ |
| wskan3 (V3) | 1.1882 ± 0.0125 | fully selective |
| mamba2 | 1.1886 ± 0.0302 | reference SSM |
| wskan6 (V6) | 1.1910 ± 0.0108 | factorized timescales |
| wskan7bc16 | 1.2262 ± 0.0122 | param-parity ablation |
| wskan3real (ω≡0) | 1.2443 ± 0.0170 | ablation control |
| wskan7 (full V7) | 1.2767 ± 0.0182 | rejected |
| wskan7g (s42) | 1.2698 | low-rank g ablation |
| wskan7z (s42) | 1.2097 | diagonal W_z ablation |

**Wavelet ablation (the project's headline, replicated):** V3 − ω≡0 =
**−0.0560 nats** (paired: −0.0562 / −0.0605 / −0.0514, 3/3) — larger than
on TinyStories (−0.0425), matching the era finding that the wavelet
advantage grows with text complexity.

**Version relationships (fresh):** wskan7bc < wskan4 < wskan3 ≈ mamba2 <
wskan6 — V4 vs V3 is −0.0045 (still neutral), V6 lands between V3 and V4
this time (+0.0073 vs V4; era showed parity), and **wskan7bc is the best of
the WSKAN series**, as adopted.

**wskan7bc vs mamba2:** mean −0.0176 (wskan7bc better) with mixed pairs
(+0.0053 / +0.0252 / −0.0828) — wskan7bc better on mean and worst-case
std profile; per-seed outcome depends on mamba's s2024 outlier. Honest
reading: parity-to-slight-edge for wskan7bc at this scale.

**V7 ablation ordering replicates:** wskan7 (full) worst, low-rank g next,
then bc16, then full wskan7bc — the interpretability/performance tradeoff
from the V7 report holds on fresh data.

## 3. Cross-dataset summary

| claim | fresh evidence |
|---|---|
| Selectivity is the dominant gap-closer (V1→V3) | 1.5001 → 0.8101 (TS) |
| Wavelet (ω) contribution is real | −0.0425 (TS) & −0.0560 (UC), 3/3 seeds each |
| Wavelet advantage grows with task complexity | −0.0425 → −0.0560 |
| wskan7bc ≥ Mamba-2 at ~100k, on mean with better seed-std | UC: 1.1711±.0274 vs 1.1886±.0302 |
| wskan3 ≈ Mamba-2 (tied) on TinyStories | 0.8101±.0132 vs 0.8194±.0107 |

All claims are 3-seed, paired, and protocol-matched. For the 225-run
cross-family matrix (5 families × 5 sizes × 3 datasets × 3 seeds — separate
budget-matched protocol), see `W7BC_CROSS_FAMILY_REPORT.md`.

## 4. What did NOT change (history, kept honest)

- The interpretability series (word clock, causal batteries, mode
  attribution, dictionaries, production trace) was computed on the earlier
  d32L3 checkpoints; its mechanisms stand, and its numbers are of the same
  magnitude as the interp tier's. Those reports are not renumbered here.
- The incidents that produced this tier (checkpoint overwrite by the
  matrix, /tmp cache exhaustion) are documented in commit history and the
  README checkpoint-tier note.

## Reproduction

```bash
.venv/bin/python experiments/V1_train_tinystories_lm.py --model {m} \
  --scale interp --dataset {tinystories|ultrachat} --seed {42|123|2024} \
  --steps 100000 --lr-schedule cosine --compile
# block 256 (tinystories) / 512 (ultrachat)
```
