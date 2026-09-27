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
0.90). *Round-7 sharpening:* the prior is **unnecessary** — a trainable
flat-ω init matches baseline CE (mean Δ −0.002 over 3 seeds, §8) and the
frequency organization is rebuilt from zero (ladder R² 0.83–0.96). The
π-harmonic init chooses the basin, not the loss; whether the learned
frequency placement is load-bearing in either case is the ω≡0 question
(+0.034, §8). Everything else in the table is learned from flat
initialization.

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

## 5. Emergence order (from-scratch dynamics; round-7: extended to 20,000 steps)

*The first version of this section used a 3,000-step window (2.8% of the
campaign length); round-7 extended it to 20,000
(`experiments/W11_dynamics_20k.py`, figures `w11_dynamics20k_*.json`).*

1k tier (TinyStories): timescale organization is immediate (ladder R²
0.85–0.95 throughout, after the noisy first steps); the gate named-share
collapses from 1.00 to ~0.02–0.14 within the first ~250–2,000 steps
(content residual grows first), then **re-equilibrates slowly to a plateau
of 0.34 at ~16–18k steps** (matching the converged campaign value 0.31) —
the re-equilibration does plateau, and the 3k window was mid-transition.
The boundary clock emerges at 2–4k steps (1.53 → 2.31) — later than the
ladder, and later than at 100k. 100k tier (UltraChat, 20k): clock at 2.18
by 2.5k steps, plateau ~2.2; gate share collapses to ~0.08 and creeps to
0.10 at 20k (converged campaign value 0.16–0.20 — the slow
re-equilibration continues past 20k at this tier; the 20k run's cosine
schedule ends at 20k, so its eval CE 1.341 is not step-matched to the
campaign's 108k schedule).

Emergence order, both tiers: (1) timescale organization, immediately;
(2) content-residual growth (named-share collapse); (3) boundary clock
(250 steps at 100k, 2–4k at 1k); (4) slow gate-split re-equilibration to
a plateau.

## 6. The antipodal claim — retracted (round-7 permutation test)

Three metrics, and the honest end point. (i) Readout-gain cosine: weak
everywhere (−0.12…−0.22 trained). (ii) The V7BC profile metric (12-bin
activation profiles, mirror pairs at corr < −0.8) *seemed* to support the
claim (trained 11–19 pairs vs 6 at init) — but the init comparison was
the wrong baseline. (iii) A proper null (round-7: shuffle the
position→bin assignment, 200 permutations) produces **18–53 mirror pairs
by chance** — far *above* the observed trained counts (2–16; z = −1.8 …
−5.9). Pair counts on this metric are anti-correlated with real
structure, not evidence of it. **The "antipodal specialists"
characterization is retracted.** What survives is the channel-level
*causal* evidence (V7D: zeroing the L0 boundary singleton costs +1.08 CE)
— specialist singleton channels exist and are load-bearing; their alleged
mirror-image pairing was a metric artifact.

## 7. Answer (round-6 revised)

1. **Solution-load-bearing ≠ task-required — the round-6 review was
   right, and §8 now separates the two claims by measurement.** Within the
   found solution the organization is load-bearing (surgery: +0.13…+5.9
   nats, 3 seeds × 3 perms; the mode is a coherent object — exact gauge).
   Retraining without the structure (3 seeds, §8): feature tables and the
   ρ ladder are **not distinguishable from zero** (deltas straddle zero);
   mode multiplicity +0.028 ± 0.019 (3/3 positive); oscillation +0.034 ±
   0.009; the π-harmonic prior is unnecessary (−0.002 ± 0.009, and the
   frequency ladder is rebuilt from flat init). An earlier version of this
   section said "the task requires this organization" — **retracted** in
   round 6; the round-7 multi-seed rerun then downgraded the first
   retrain numbers themselves from "+0.006/+0.009" to "indistinguishable
   from zero at this noise level". Both corrections stand.
2. **Origin is decomposed, and it is a prior+landscape story, not a
   task-necessity story.** Frequency skeleton: given by the prior
   (π-harmonics), not destroyed — but the prior is unnecessary (flat-ω
   retrains match baseline and rebuild the ladder, §8). Timescale ladder,
   clock, gate split: learned from flat init within a few hundred steps,
   oscillation-free (wskan11real), and in analogous form across
   architectures (mamba2: given-and-preserved ladder; boundary gating
   with the opposite *operation* — skip vs reset, §4). Gradient descent
   finds this organization *immediately and reliably* under this
   parameterization — "reliably" now measured across 5 seeds (§9): same
   coarse organization, same sorted mode-value distribution (CV 3–15%);
   the parameterization makes it the path of least resistance, not the
   task's only solution.
3. **The earlier "the data's correlation orders require multiple
   timescales" argument is replaced by the measured effect:**
   multiresolution state helps (+0.028 ± 0.019, param-matched
   single-mode, 3 seeds), oscillation helps (+0.034 ± 0.009) — both real,
   both modest, neither necessary.
4. **What remains open (narrower):** why the landscape prefers this basin
   so early (the ~250-step emergence is measured, not explained); the
   seed-level question is closed at the gauge-invariant level (§9); the
   antipodal claim is retracted (§6).
5. **The interpretability value framing survives, strengthened:** the
   architecture makes an *analyzable* organization the path of least
   resistance — that, not task necessity, is the reason the structure is
   worth studying. The performance cost of that choice is the
   architecture-level gap to mamba2, not the organization (§10).

## 8. The retrain test (round-6, extended to 3 seeds in round-7): what the task actually requires

Surgery measures the found solution's dependence; the task-requirement
question needs retraining with the structure impossible from the start.
Campaign protocol (UltraChat, 100k tier, 3ep, 108k steps, seeds 42/7/123;
`experiments/W11_ablation_retrain.py`). **All numbers on 16 fixed eval
chunks (513 bytes each); NOT comparable to the campaign's random-batch
eval in Appendix C of the monograph (baseline 1.2907 here vs 1.184
there, same checkpoint, different eval set).** Paired per-seed deltas
(variant − baseline, same seed, same chunks); baseline CEs 1.2907 /
1.2996 / 1.3082 for seeds 42/7/123 (baseline seed std 0.009):

| variant | ΔCE per seed (42 / 7 / 123) | mean ± std |
|---|---|---|
| nofeat (named feature tables removed) | +0.006 / −0.004 / −0.007 | **−0.002 ± 0.007** |
| rhofrozen (ρ ≡ 1, no learnable ladder) | +0.009 / +0.010 / −0.022 | **−0.001 ± 0.017** |
| n1wide (single mode, param-matched 109k) | +0.041 / +0.036 / +0.007 | **+0.028 ± 0.019** |
| flatomega (ω init 0, trainable; π prior removed) | +0.008 / −0.004 / −0.009 | **−0.002 ± 0.009** |
| wskan11real (ω ≡ 0 frozen; campaign, paired) | +0.037 / +0.041 / +0.025 | **+0.034 ± 0.009** |

Round-7 verdicts, by variant:

- **Feature tables and the ρ ladder: not distinguishable from zero at 3
  seeds** — the deltas straddle zero (sign flips, as the round-7 review
  predicted for the single-seed +0.006/+0.009). The correct statement is
  "no measurable task-requirement at this noise level", not the round-6
  "+0.006".
- **Mode multiplicity: small, consistently positive cost** (+0.028,
  positive in 3/3 seeds) — comparable to oscillation itself (+0.034).
  Real but modest.
- **The π-harmonic frequency prior is unnecessary**: flat-ω trainable
  init matches the baseline (mean −0.002), and the trained flat-ω models
  *rebuild* frequency organization from zero (ladder R² 0.83–0.96,
  narrower ω range 0.9–1.7). The prior chooses the basin, not the loss.
- ρ-freeness is recoverable because the effective timescale is σ·ρ and σ
  is free per mode — a redundant parameterization. The n1/n1wide pair
  shows why param-matching matters: half the apparent cost of losing the
  mode bank was lost parameters, not lost structure.

*Limits:* 3 seeds per variant; one tier (100k), one dataset; fixed-chunk
eval.

## 9. Seed-level convergence (round-7): what "reliably" means

Five campaign seeds, 100k, UltraChat — coarse structure converges
tightly: boundary clock 1.5–2.65 at every layer and seed; gate
named-share 0.136–0.195; timescale-ladder R² 0.56–0.95. And the
gauge-invariant mode content — the *sorted* per-mode (σ̃, ω̃) vectors —
agrees across seeds with cross-seed coefficient of variation 3–15% per
mode-rank. The specific mode *indices* are gauge (§1), so "assignment
convergence" is only defined up to permutation; what converges is the
multiset of mode values. "Path of least resistance" is therefore precise
at this level: **the same coarse organization and the same mode-value
distribution, at every seed** — while the indexing and (per round-6) the
necessity of any single piece do not survive.

## 10. Terminology and the performance tradeoff (round-7)

**Terminology.** "Excess structure" was the reviewer's framing and this
report's working term, but it presupposes a task-minimal structure that
was never shown to exist. The accurate phrase is
**architecturally-induced organization**: prior skeleton (frequency
ladder) plus the optimization landscape's preferred basin (everything
else). The file name stays for continuity; the term should be read with
this meaning.

**The performance tradeoff, stated plainly.** Mamba-2 outperforms wskan11
(100k UltraChat: 1.158 vs 1.184; 10m: 0.716 vs 0.833 — Appendix C of the
monograph) with a *different* organization. So the WSKAN organization is
not a performance optimum, and the interpretability is not free: the cost
is the architecture-level gap (~0.03 nats at 100k, ~0.12 at 10m). What
the retrain test adds: the gap is not caused by the *analyzable
structure itself* (removing the feature tables or the ρ ladder costs
≈ 0) — it is the price of the *parameterization* that makes the
structure analyzable (edge functions as the unit). Whether that price is
worth a ground-truth benchmark substrate is a research-value judgment,
stated as such — not hidden.

**Frequency vs timescale ladders are parameterization-specific objects.**
WSKAN gets a frequency skeleton by prior (π-harmonics) and a timescale
ladder by learning; mamba2 has no frequency object at all (real A) and
gets its timescale ladder from init prior; the transformer baseline gets
frequencies from fixed RoPE and has no decay ladder. What generalizes
across all three is only "a multi-scale bank"; the frequency/timescale
decomposition is WSKAN's coordinate system, not a universal one.

*Limits:* single seed per variant; one tier (100k), one dataset; the
eval set differs from the campaign's random-batch eval (baselines
re-measured on the same fixed chunks).

*Limits (whole report):* interventions are now 3 seeds × 3 permutations
(§1); CE evals use 60 chunks; dynamics are 3,000 of 108,000 steps; the
antipodal claim is metric-sensitive (§6); retrains are single-seed.
