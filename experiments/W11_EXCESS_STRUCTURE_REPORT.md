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
   Retraining without the structure (3–5 seeds, §8): feature tables and the
   ρ ladder are **not distinguishable from zero** (deltas straddle zero);
   mode multiplicity +0.033 ± 0.020 (5/5 positive, p = 0.021);
   oscillation +0.037 ± 0.007 (p = 3e-4); the π-harmonic prior is
   unnecessary (−0.002 ± 0.009, and the frequency ladder is rebuilt from
   flat init). An earlier version of this section said "the task requires
   this organization" — **retracted** in round 6; the round-7 multi-seed
   rerun then downgraded the first retrain numbers themselves from
   "+0.006/+0.009" to "indistinguishable from zero at this noise level";
   round 8 extended n1wide/wskan11real to 5 seeds for significance. All
   corrections stand.
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
   multiresolution state helps (+0.033 ± 0.020, param-matched
   single-mode, 5 seeds, p = 0.021), oscillation helps (+0.037 ± 0.007,
   p = 3e-4) — both real, both modest, neither necessary.
4. **What remains open (narrower):** the early-basin question is now
   measured (§10: data gradients differentiate modes from step 0; the Δ
   init scale gates the clock's emergence; nothing needs the frequency
   prior); the seed-level question is closed at the gauge-invariant level
   (§9); the antipodal claim is retracted (§6).
5. **The interpretability value framing survives, strengthened:** the
   architecture makes an *analyzable* organization the path of least
   resistance — that, not task necessity, is the reason the structure is
   worth studying. The performance cost of that choice is the
   architecture-level gap to mamba2, not the organization (§11).

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

| variant | ΔCE per seed (42 / 7 / 123 / 2024 / 31337) | mean ± std | t (df) | p |
|---|---|---|---|---|
| nofeat (named feature tables removed) | +0.006 / −0.004 / −0.007 | −0.002 ± 0.007 (n=3) | — | n.s. |
| rhofrozen (ρ ≡ 1, no learnable ladder) | +0.009 / +0.010 / −0.022 | −0.001 ± 0.017 (n=3) | — | n.s. |
| n1wide (single mode, param-matched 109k) | +0.041 / +0.036 / +0.007 / +0.022 / +0.060 | **+0.033 ± 0.020 (n=5)** | 3.7 (4) | **0.021** |
| flatomega (ω init 0, trainable; π prior removed) | +0.008 / −0.004 / −0.009 | −0.002 ± 0.009 (n=3) | — | n.s. |
| wskan11real (ω ≡ 0 frozen; campaign, paired) | +0.037 / +0.041 / +0.025 / +0.040 / +0.042 | **+0.037 ± 0.007 (n=5)** | 11.4 (4) | **3e-4** |

Round-7 verdicts, by variant:

- **Feature tables and the ρ ladder: not distinguishable from zero at 3
  seeds** — the deltas straddle zero (sign flips, as the round-7 review
  predicted for the single-seed +0.006/+0.009). The correct statement is
  "no measurable task-requirement at this noise level", not the round-6
  "+0.006".
- **Mode multiplicity: small and now significant** — +0.033 ± 0.020 over
  5 seeds, positive in 5/5, t = 3.7, p = 0.021; the assumption-free sign
  test agrees (5/5 positive: p = 1/32 ≈ 0.031). Oscillation: +0.037 ±
  0.007, p = 3e-4 (5 seeds). Both real, both modest.
- **The π-harmonic frequency prior is unnecessary**: flat-ω trainable
  init matches the baseline (mean −0.002), and the trained flat-ω models
  *rebuild* frequency organization from zero (ladder R² 0.83–0.96,
  narrower ω range 0.9–1.7). The prior chooses the basin, not the loss.
  *This is basin selection in the lottery-ticket sense, measured
  directly:* two inits, two frequency organizations, one loss level —
  the init decides which analyzable basin you get, not whether you
  converge.
- ρ-freeness is recoverable because the effective timescale is σ·ρ and σ
  is free per mode — a redundant parameterization. The n1/n1wide pair
  shows why param-matching matters: half the apparent cost of losing the
  mode bank was lost parameters, not lost structure.

*Second-dataset replication (round-9, TinyStories, 108k steps — note:
fewer epochs than TinyStories' 300k-step campaign schedule, but internally
consistent since baseline and variants share the protocol):*

| variant | ΔCE per seed (42 / 7) | reading |
|---|---|---|
| nofeat | −0.001 / +0.007 | zero again (sign flips) |
| n1wide | +0.045 / +0.064 | positive again, slightly larger than UltraChat |

The headline survives a second dataset: feature tables remain unrequired,
the mode bank's small cost replicates.

*Limits:* 3 seeds per variant (5 for n1wide/wskan11real) on UltraChat,
2 seeds on TinyStories; one tier (100k); fixed-chunk eval.

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

## 10. Why the basin is entered so early (round-8: the last open sub-question, measured)

The ~250-step emergence was measured but unexplained. Three
parameterization-level drivers tested (`experiments/W11_basin_origin.py`,
figures `w11_basin_origin.json`):

1. **Gradient anatomy at init (100k config, real data):** the step-0
   gradient is dominated by the token embedding (~30× the next group),
   then the base skip path; the SSM/gate parameters receive ~1e-7. But
   the per-mode breakdown of dL/dlog_σ is already **non-uniform across
   modes** (per-mode energy CV 0.21–0.49) — and remains so with a flat-ω
   init (CV 0.31–0.33). The data gradient differentiates modes from step
   zero; the frequency prior is not what breaks mode symmetry (the random
   gate tables suffice). The 250-step emergence is then simply the time
   the initially-tiny SSM gradients need to accumulate — there is no
   barrier; the organized basin is the bottom of the local landscape.
2. **Δ-init-scale sensitivity (1k):** the softplus bias init (Δ₀ = 0.05)
   controls emergence *order*: with Δ₀ = 0.005 (long-memory init) the
   clock emerges slower but the ladder faster; with Δ₀ = 0.5 (short-memory
   init) the clock does **not** emerge within 2k steps at all. The initial
   step scale sets how much boundary contrast survives the decay — a
   concrete parameterization property gating the clock.
3. **Flat-ω emergence (1k):** ladder and clock emerge at the same speed
   from a perfectly symmetric mode init (ω = 0, σ uniform, ρ = 1) —
   symmetry breaking is data-driven, prior-free.

*Limits:* the Δ-scale runs are 2k steps at 1k only. (Gradient anatomy was
single-batch in the first version; round-9 re-ran it on 20 batches:
per-mode CV L0 0.427 ± 0.077, L1 0.278 ± 0.050; embedding gradient share
94.7% ± 0.1% — stable.)

## 11. Terminology and the performance tradeoff (round-7)

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

## 12. The benchmark substrate, demonstrated (round-9)

The claimed value "a benchmark on which post-hoc interpretability methods
can be validated against exact answers" was aspirational until now.
`experiments/W11_benchmark_demo.py` scores two standard post-hoc
attribution methods against the exact per-position decision contributions
(the model's own arithmetic, Appendix D of the monograph), on three
canonical 100k decisions:

| context (tail) | decision | saliency (grad×input) Spearman ρ, top-3 | LOO-replacement ρ, top-3 |
|---|---|---|---|
| `…my dear frien` | d vs t | +0.88, 3/3 | +0.34, 2/3 |
| `…annual repor` | t vs a | +0.70, 3/3 | +0.46, 3/3 |
| `…cup of suga` | r vs l | +0.89, 1/3 | +0.40, 1/3 |

Gradient saliency tracks the truth moderately (ρ 0.7–0.9) but misranks
the decisive terms (top-3 overlap as low as 1/3); **leave-one-out
replacement — a standard "causal" attribution — agrees with the truth at
only ρ ≈ 0.34–0.46**. On this substrate the error is quantified, not
suspected. Demo 2 (already in the monograph §5): a linear probe on the
write path concludes "no word identity" (R² ≈ 0.15); the substrate's
nonlinear probe + readout analysis shows the information is present but
linearly inaccessible — a false negative the substrate adjudicates.
Neither demo shows the methods are useless; they show their error is
measurable here and not elsewhere.

*Limits (whole report, round-9 current):* interventions 3 seeds × 3
permutations (§1); retrains 3 seeds per variant (n1wide/wskan11real 5),
replicated on a second dataset at 2 seeds (§8); surgical CE evals 60
chunks, retrain evals 16 fixed chunks (not comparable to Appendix C of
the monograph); dynamics extended to 20,000 steps (§5) — still short of
the 108k campaign; basin-origin: 20-batch gradient anatomy + 2k-step 1k
runs (§10); the antipodal claim is retracted, not merely metric-sensitive
(§6); benchmark demo: 3 decisions, 2 post-hoc methods (§12); one tier
(100k) throughout.
