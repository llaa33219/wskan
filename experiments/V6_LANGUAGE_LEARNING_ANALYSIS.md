# How V6 Learned Language: A Mathematical Dissection of the Trained Model

**Date:** 2026-09-08 · **Subject:** `wskan6_ultrachat_100k_s42` (3-seed-adopted
architecture, UltraChat, 100k steps). Script: `experiments/V6_deep_analysis.py`.
Figures: `experiments/figures/fig_v6_*.png`, numbers: `v6_analysis_summary.json`.
Every claim below is read directly from the checkpoint's edge functions —
the KAN property that makes this analysis possible at all.

## Executive summary

The model learned language as **a content-driven clock over a hierarchical,
overdamped filter bank**: word boundaries advance the internal clock 2–3×
faster than letters (the network discovered word segmentation *through time
warping*), receptive fields grow from delta-like (layer 0) to broad smooth
envelopes spanning tens of tokens (layer 2), mode frequencies concentrate at
word-scale periodicities (~6-token period), and edge usage becomes
increasingly hub-sparse with depth. Two popular hypotheses were tested and
**honestly rejected**: strict constant-Q wavelet self-organization (weak
fit), and locking onto discrete text spectral peaks (the byte spectrum is
broadband; no peak-to-peak match).

## Finding 1 — The clock discovered words (strongest result)

Mean dynamic dilation Δ per byte class, real 2048-byte probe:

| byte class | Δ (L0) | Δ (L1) | Δ (L2) |
|---|---|---|---|
| newline | **0.396** | **0.348** | **0.481** |
| space | **0.380** | **0.284** | **0.469** |
| punctuation | 0.345 | 0.347 | 0.425 |
| uppercase (word-initial) | 0.231 | 0.289 | 0.436 |
| digits | 0.151 | 0.221 | 0.284 |
| lowercase | 0.138 | 0.134 | 0.214 |

Word boundaries (newline/space/punct) advance the warped clock **2–3× faster
than letters**, in every layer; word-initial uppercase sits between. Since
state decay per token is e^{−σρΔ}, a boundary erases ~e^{2–3×} more memory
than a letter: **the effective unit of distance is the word, not the byte.**
The time-warp parameterization absorbed tokenization — the model segmented
words without ever being given one.

Complementary: write strength |B| is highest at letters (1.93) vs boundaries
(1.66–1.72) — content is *stored* at letters, while boundaries mostly
*advance the clock*.

## Finding 2 — A depth hierarchy of temporal receptive fields

Top-|g| edge kernels (effective, per-edge, using learned ρ, σ, ω and mean Δ):

| Layer | kernel shape | span |
|---|---|---|
| 0 | delta-like spikes, ringing only within ~5 tokens | bytes/n-grams |
| 1 | mixed: spikes + broad smooth bumps at lag 10–30 | words/phrases |
| 2 | smooth single-lobe envelopes, tails past 40 tokens | phrase/clause scale |

Fine-to-coarse with depth: early layers extract local byte features; the
last layer integrates over long spans. (This is *read directly from the
learned functions*, not inferred from behavior.)

## Finding 3 — Mode frequencies sit at word scale, but NOT at spectral peaks

Learned mode frequencies: median **0.165 cycles/token (≈6-token period)**,
p90 0.31, max 0.53 — concentrated exactly at word/morpheme periodicity scale.
However, the English byte-stream periodogram (byte-value and
space-indicator) is **broadband in that band**: no sharp peaks to lock onto.
Honest conclusion: the frequency placement matches the *linguistically
relevant scale*, but we cannot claim resonance with discrete text
periodicities — either those periodicities are too diffuse to see in the
periodogram, or the placement reflects optimization dynamics rather than
spectral matching. Claim discipline: scale-consistent, peak-matching
unverified.

## Finding 4 — Constant-Q wavelet self-organization: rejected (quantified)

Log-log fit of effective decay vs frequency across modes:

| Layer | slope | R² | Q median [p10, p90] |
|---|---|---|---|
| 0 | 0.25 | 0.175 | 0.67 [0.33, 1.27] |
| 1 | 0.30 | 0.290 | 0.67 [0.29, 1.18] |
| 2 | 0.26 | 0.237 | 0.57 [0.24, 1.12] |

A strict constant-Q (Gabor) bank would give slope 1 with tight fit. The data
show only a weak σ ~ ω^0.26–0.30 trend with ~5–10× vertical scatter. The
earlier qualitative "σ∝ω law" impression from V3 does **not** survive
quantification — recorded as a correction. Q ≈ 0.6 also means modes are
*overdamped*: the ω-ablation gain (−0.03..−0.05 nats) comes from phase
structure in short kernels, not from sustained ringing.

## Finding 5 — Hub sparsity grows with depth

Edge-strength Gini coefficient: L0 0.297 → L1 0.344 → L2 0.464;
max/median |g|: 4.6 → 7.6 → 28.5. The last layer concentrates function into
a few hub edges (consistent with V3's hub-node finding) — pruning pressure
is measurable and grows with depth.

## Synthesis: the mathematical answer to "how did it learn language"

$$\text{language model} \approx \underbrace{\text{word-clocked}}_{\Delta \text{ warp (Finding 1)}} \; \underbrace{\text{hierarchical overdamped filter bank}}_{\rho\sigma,\rho\omega \text{ (Findings 2–4)}} \; \underbrace{\text{with hub-sparse mixing}}_{|g| \text{ (Finding 5)}}$$

at this scale — register learning without semantics or recall (see
`V6_QUALITATIVE_SAMPLES.txt`, `V4_MAMBA_MEMORY_COMPARISON.md`), consistent
with the dormant long-memory tail.

## Limitations

- One checkpoint (s42), one probe text; single-seed analysis (the findings'
  effect sizes — 2–3× clock ratios, Gini growth — are large relative to
  typical seed variation, but replication is cheap and advisable).
- Byte-class gating is *correlational*. A causal test (clamp Δ to the
  letter-mean at boundaries at inference, measure degradation) is designed
  but not run — future work.
- The ρ ladder analysis used layer-level effective parameters; per-channel
  ladders differ and were not clustered here.
