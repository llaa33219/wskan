# Why the Organization Arises: the "Excess Structure" Is Not Excess

**Date:** 2026-09-24 · **Subject:** wskan11 (1k/10k/100k, 3-epoch tier), wskan11real, mamba2, wskan7bc-interp
**Script:** `experiments/W11_excess_structure.py` · numbers `figures/w11_excess_structure.json`

The monograph (W11_FINAL_INTERPRETATION_REPORT.md, section 6) recorded as an
open question: the frequency ladder, the structure/content gate split, and
the antipodal specialists *exceed what next-byte prediction strictly
requires* — why does the excess arise? This report answers the question
with six measurements and corrects its premise twice: the structure is
load-bearing for the found solution (§1) but — after the round-6 retrain
test (§8) — **not required by the task**; its origin is a prior +
optimization-landscape story, not a necessity story.

## 1. First, necessity: the structure is load-bearing (interventions, 100k canonical)

Eval-CE deltas on the canonical 100k checkpoint (UltraChat eval, 60 chunks
of 512 bytes), **3 seeds × 3 permutations** (round-6 strengthening; the
first version of this table was single-seed, single-permutation):

| intervention | ΔCE (mean ± std over seeds/perms) |
|---|---|
| ρ → 1 (timescale ladder flattened, σ kept) | +0.125 / +0.328 / +0.319 (per seed; deterministic) |
| ω shuffled across modes only (frequency–mode consistency destroyed) | **+2.51 ± 0.10** (seed means 2.48/2.63/2.44, perm-std ~0.5) |
| σ+ρ shuffled across modes only (timescale–mode consistency destroyed) | **+1.77 ± 0.21** (seed means 1.83/1.94/1.53) |
| feature-table rows shuffled (byte-feature ↔ table assignment destroyed) | **+5.51 ± 0.11** (seed means 5.58/5.38/5.59) |
| full mode permutation, all mode-indexed tensors (gauge test) | **+0.00000001** (exact) |

The gauge result says exactly what matters: **a mode is a coherent
object**. The value is in the *within-mode* consistency (each mode's
frequency matched to its timescale, readout gains, and gate tables); the
cross-mode ordering is pure gauge. (An earlier iteration of this battery
reported +0.72 for the full permutation; that was a bug — `res_B/res_C`
output biases carry a mode axis and were not permuted. With biases
included the symmetry is exact to 1e-8. Recorded for honesty.)

*Scope correction (round-6):* surgical interventions measure the value of
the structure **within the found solution**, not whether the *task*
requires it — a retrained structure-less model could in principle recover.
That retraining test is `experiments/W11_ablation_retrain.py` (§8 below);
the surgical numbers should be read as "the found solution's organization
is not decorative", not as "no simpler solution exists".

## 2. Prior vs learned (init vs trained, wskan11)

| structure | at init | after training | verdict |
|---|---|---|---|
| timescale ladder (σ·ρ per mode) | flat (R² = 0; σ ≡ 0.5, ρ ≡ 1 by init) | R² 0.64–0.89, ratio 1.3–2.3× | **learned** |
| frequency ladder (median ω per mode) | R² = 0.94 (π·[1..6] harmonic init) | R² 0.90–0.98 | **prior skeleton, not destroyed** (100k L1: 0.94 → 0.90 — slightly *degraded*, so "refined" would be wrong) |
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
  boundary-responsive — space/letter dt ratio **0.13–0.65 per layer**,
  i.e. with the *opposite sign* from WSKAN's 1.95–2.56. *Round-6
  measurement resolved the "convention vs computation" question:* the
  ZOH write magnitude (dt·‖B‖) at spaces vs letters is 0.15–0.70 in
  mamba2's deep layers — mamba2 **skips** boundaries (the state is barely
  written or decayed there), while WSKAN **resets** at boundaries (large
  Δ step = strong decay + strong write). Same object (boundary-responsive
  gating), genuinely different operations — measured, not hypothesized.
  (Init dt ratios vary by random seed; the trained inversion is the
  signal.)

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

## 6. The antipodal caveat (honesty, round-6 replication)

Two metrics, two answers. By the **readout-gain cosine** metric,
antipodal pairing is weak everywhere — wskan11 1k/10k/100k
(−0.12…−0.22 trained vs −0.09…−0.20 init) and wskan7bc (−0.13…−0.17).
By the **original V7BC profile metric** (channel activation profiles over
12 byte-class × position-in-word bins; mirror pairs = profile corr
< −0.8), trained wskan11-100k has **11–19 mirror pairs per layer vs 6 at
init** (wskan7bc: 6–9 vs 0 at init) — so profile-level antipodality is
real and learned, but (i) a chance-level baseline of such pairs exists at
init, and (ii) the pair counts exceed the "1–3 specialist pairs" of the
original clustering formulation. The defensible claim: *training roughly
doubles-to-triples mirror-image channel profiles above a nonzero chance
baseline*; "antipodal specialists" as a named circuit class is stronger
than the current metric supports.

## 7. Answer (round-6 revised)

1. **Solution-load-bearing ≠ task-required — the round-6 review was
   right, and §8 now separates the two claims by measurement.** Within the
   found solution the organization is load-bearing (surgery: +0.13…+5.9
   nats, 3 seeds × 3 perms; the mode is a coherent object — exact gauge).
   But retraining without the structure recovers almost everything
   (§8: feature tables +0.006, ρ ladder +0.009, single mode +0.041
   param-matched, no frequency +0.037). An earlier version of this section
   said "the task requires this organization; naive intuition about
   requirements was what exceeded the evidence" — **that claim is
   retracted**: the retrain test falsified it. What exceeds the evidence
   was the surgical inference itself.
2. **Origin is decomposed, and it is a prior+landscape story, not a
   task-necessity story.** Frequency skeleton: given by the prior
   (π-harmonics), not destroyed. Timescale ladder, clock, gate split:
   learned from flat init within a few hundred steps, oscillation-free
   (wskan11real), and in analogous form across architectures (mamba2:
   given-and-preserved ladder; boundary gating with the opposite
   *operation* — skip vs reset, §4). Gradient descent finds this
   organization *immediately and reliably* under this parameterization;
   the parameterization makes it the path of least resistance, not the
   task's only solution.
3. **§7.3's earlier argument ("the data's correlation orders require
   multiple timescales") is downgraded from argument to small measured
   effect:** multiresolution state helps (+0.041, param-matched
   single-mode), oscillation helps (+0.037) — both real, both modest,
   neither necessary.
4. **What remains open (narrower):** seed-level convergence of the
   specific assignments; the antipodal formulation gap (§6); why the
   landscape prefers this basin so early (the ~250-step emergence is
   measured, not explained).
5. **The interpretability value framing survives, strengthened:** the
   architecture makes an *analyzable* organization the path of least
   resistance — that, not task necessity, is the reason the structure is
   worth studying.

## 8. The retrain test (round-6): what the task actually requires

Surgery measures the found solution's dependence; the task-requirement
question needs retraining with the structure impossible from the start.
Campaign protocol (UltraChat, 100k tier, 3ep, 108k steps, seed 42;
`experiments/W11_ablation_retrain.py`), fixed 16-chunk eval:

| variant | params | eval CE | Δ vs baseline |
|---|---|---|---|
| wskan11 baseline | 113.6k | 1.2907 | — |
| nofeat (named feature tables removed) | 113.6k | 1.2971 | **+0.006** |
| rhofrozen (ρ ≡ 1, no learnable ladder) | 113.6k | 1.2993 | **+0.009** |
| n1wide (single mode, param-matched) | 109.3k | 1.3321 | **+0.041** |
| wskan11real (ω ≡ 0, from the campaign) | 113.6k | 1.3280 | +0.037 |
| n1 (single mode, *not* param-matched) | 48.0k | 1.4469 | +0.156 (confounded) |

The surgery/retrain gap is the headline: feature tables +5.5 by surgery
vs **+0.006** by retrain (~900×); ρ ladder +0.13–0.33 vs +0.009; mode
consistency +1.8–2.6 vs (mode *count*) +0.041. Two notes. (i) ρ-freeness
is recoverable because the effective timescale is σ·ρ and σ is free per
mode — the ρ ladder is a redundant parameterization; surgery hurts
because σ cannot retake ρ's share in-place, retraining just learns it.
(ii) The n1/n1wide pair shows why param-matching matters: half the
apparent cost of losing the mode bank was lost parameters, not lost
structure.

*Limits:* single seed per variant; one tier (100k), one dataset; the
eval set differs from the campaign's random-batch eval (baselines
re-measured on the same fixed chunks).

*Limits (whole report):* interventions are now 3 seeds × 3 permutations
(§1); CE evals use 60 chunks; dynamics are 3,000 of 108,000 steps; the
antipodal claim is metric-sensitive (§6); retrains are single-seed.
