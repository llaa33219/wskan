# Why the Organization Arises: the "Excess Structure" Is Not Excess

**Date:** 2026-09-24 · **Subject:** wskan11 (1k/10k/100k, 3-epoch tier), wskan11real, mamba2, wskan7bc-interp
**Script:** `experiments/W11_excess_structure.py` · numbers `figures/w11_excess_structure.json`

The monograph (W11_FINAL_INTERPRETATION_REPORT.md, section 6) recorded as an
open question: the frequency ladder, the structure/content gate split, and
the antipodal specialists *exceed what next-byte prediction strictly
requires* — why does the excess arise? This report answers the question
with five measurements and corrects its premise: **the structure is not
excess at all.**

## 1. First, necessity: the structure is load-bearing (interventions, 100k canonical)

Eval-CE deltas on the canonical 100k checkpoint (UltraChat eval, 60 chunks
of 512 bytes, baseline CE 1.2804), single-permutation samples:

| intervention | ΔCE |
|---|---|
| ρ → 1 (timescale ladder flattened, σ kept) | **+0.125** |
| ω shuffled across modes only (frequency–mode consistency destroyed) | **+2.61** |
| σ+ρ shuffled across modes only (timescale–mode consistency destroyed) | **+1.91** |
| feature-table rows shuffled (byte-feature ↔ table assignment destroyed) | **+5.94** |
| full mode permutation, all mode-indexed tensors (gauge test) | **+0.00000001** |

Two conclusions. (i) Every piece of the "excess" organization is
load-bearing, and most of it is worth far more than the wavelet itself
(the ω≡0 ablation costs ~0.04 nats at this tier — the frequency *mode
consistency* alone is worth 65× that). (ii) The full-permutation gauge
result says exactly what matters: **a mode is a coherent object**. The
value is in the *within-mode* consistency (each mode's frequency matched
to its timescale, readout gains, and gate tables); the cross-mode
ordering is pure gauge. (An earlier iteration of this battery reported
+0.72 for the full permutation; that was a bug — `res_B/res_C` output
biases carry a mode axis and were not permuted. With biases included the
symmetry is exact to 1e-8. Recorded for honesty.)

The premise "exceeds what the task strictly requires" was wrong. The task
requires this organization; naive intuition about requirements was what
exceeded the evidence.

## 2. Prior vs learned (init vs trained, wskan11)

| structure | at init | after training | verdict |
|---|---|---|---|
| timescale ladder (σ·ρ per mode) | flat (R² = 0; σ ≡ 0.5, ρ ≡ 1 by init) | R² 0.64–0.89, ratio 1.3–2.3× | **learned** |
| frequency ladder (median ω per mode) | R² = 0.94 (π·[1..6] harmonic init) | R² 0.90–0.98 | **prior skeleton, preserved/refined** |
| boundary clock (Δ space/letter) | 1.00 (no clock) | 1.95–2.56 | **learned** |
| gate named-share (B variance from named tables) | 0.97–1.00 (residual ≈ 0 at init) | 0.16–0.31 | **learned equilibrium** |
| antipodal gain pairing (mean min-cos) | −0.09…−0.20 | −0.12…−0.22 | weak drift (§6) |

The frequency ladder's *skeleton* is an architectural prior (harmonics of
π); training preserves it (even slightly degrades it at 100k L1: 0.94 →
0.90). Everything else is learned from flat initialization.

## 3. Oscillation-independent (wskan11real, ω ≡ 0, 100k)

The pure-decay ablation builds the **same** organization — timescale
ladder R² 0.85–0.94 (stronger than wskan11's 0.64–0.89), clock 1.52–2.43,
gate named-share 0.17–0.19. The ladder, clock, and gate split are not
about oscillation or phase; they are about **multiresolution coverage of
local statistics**. Oscillation rides on top of this skeleton (and adds
its small consistent bonus, §2 of the monograph).

## 4. Architecture-general (mamba2, 100k, same data)

- **A ladder**: HF init is already log-spaced (R² 0.93, ratio 7× — a
  stronger ladder than WSKAN's, entirely by prior); training preserves it
  (R² 0.88–0.93). Two architectures, two routes to the same object: mamba2
  is *given* the multiresolution ladder, WSKAN *builds* it.
- **Boundary clock**: trained mamba2's per-head dt is strongly
  boundary-responsive — space/letter ratio **0.12–0.65 per layer**, i.e.
  with the *opposite sign convention* from WSKAN's 1.95–2.56. Mamba-2
  ticks *slower* at boundaries (less state update), WSKAN ticks *faster*
  (larger warped distance). The object — boundary-responsive selectivity —
  is general; the convention is architecture-specific. (Init dt ratios
  vary by random seed; the trained inversion is the signal.)

## 5. Emergence order (from-scratch dynamics, 3,000 steps)

100k tier: by step 250 (eval CE 5.45 → 1.55) the timescale organization
and the clock (1.39) are already in place; the gate named-share collapses
from ~1.0 to 0.06–0.08 within 250 steps (the content residual grows
first) and then slowly re-equilibrates (0.16–0.31 at convergence). 1k
tier: the timescale ladder forms as fast, but the boundary clock has
*not* emerged by 3,000 steps (0.92) — the clock is slower to emerge at
the smallest scale (full 1k runs are 20k steps). Step-1 ladder R² values
are metric noise (tiny perturbations of a flat init fit any line);
robust readings start ~step 250.

## 6. The antipodal caveat (honesty)

By the readout-gain cosine metric, antipodal pairing is **weak at every
tier and checkpoint tested** — wskan11 1k/10k/100k (−0.12…−0.22 trained
vs −0.09…−0.20 init), and the wskan7bc interp checkpoint where the claim
originated (−0.13…−0.17, fraction of strong pairs 0.00). The monograph's
§1.6 "antipodal specialists" claim rests on a *different* object
(kernel-family routing structure from the causal-dictionaries battery),
not on gain-space pairing; the two should not be conflated, and §1.6
should be read as a routing-level claim only. Flagged for re-measurement
with the original metric.

## 7. Answer

1. **Not excess — load-bearing.** The organization is worth +0.125 to
   +5.94 nats per piece; the mode is a coherent object (exact gauge).
2. **Origin is decomposed, not mysterious.** Frequency skeleton: given by
   the prior (π-harmonics), preserved. Timescale ladder, clock, gate
   split: learned from flat init, within the first few hundred steps,
   without oscillation, and — for the ladder and boundary-responsive
   gating — across architectures (mamba2: given-and-preserved ladder,
   opposite-sign clock).
3. **Why gradient descent finds it:** multiresolution timescales are the
   first-order structure of byte text — bigram statistics live at one
   scale, trigram/morphological statistics at another (cf. the round-5
   finding that the 10M write path is predominantly trigram-level). Any
   model that writes multi-order local statistics needs multiple
   timescales; boundaries are where the statistics change, so gating keys
   on them. The structure is what the data's correlation orders look like
   inside an SSM-parameterized edge.
4. **What remains open (narrower now):** why the *specific* assignments
   converge as they do (seed-to-seed variance of mode assignment, not
   measured here); the antipodal-metric discrepancy (§6); and whether the
   mamba2 sign convention reflects conv-then-scan ordering or pure
   parameterization taste.

*Limits:* interventions are single-permutation samples on one seed
(assignment-shuffle costs vary by permutation; the gauge result is
exact); the CE eval uses 60 chunks; dynamics are 3,000 of 20,000 steps;
the antipodal metric is gain-space cosine only.
