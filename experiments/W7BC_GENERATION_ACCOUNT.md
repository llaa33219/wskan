# How It Speaks: The Production-Side Account of wskan7bc

**Date:** 2026-09-10 · **Subject:** `wskan7bc_ultrachat_100k_s42` (the
matrix-tier checkpoint, d=40, L=2; identical architecture to the d32L3
interpretation checkpoint — a restored copy is retraining).
Script: `experiments/W7BC_generation_analysis.py` · figure:
`figures/fig_w7bc_generation.png` · numbers: `v7e_generation_summary.json`.

## The generated trace (T=0.3, 160 bytes)

`"Assistant: I saucter a field of the story of the customer productivity
and easy to constructed the stress of the..."` — register-fluent,
content-free, as established. The point here is not the text but the
**mechanism trace underneath it**.

## 1. The clock paces production

Measured Δ *during generation* (per layer, mean over channels):

| class | Δ (L0) | Δ (L1) |
|---|---|---|
| letter | 0.073 | 0.098 |
| space | **0.186** | **0.157** |
| punctuation | **0.198** | **0.253** |

The word clock fires **2–2.6× faster at boundaries during production** —
the same mechanism measured in teacher-forced analysis (§Word clock), now
observed pacing the emission loop itself. The clock heatmap shows L0 firing
in discrete bursts locked to word boundaries, L1 carrying a continuous
carrier modulated by the same rhythm; a punctuation event triggers a
synchronized spike across clock, wavelet paths, and modes.

## 2. What drives decisions during production (exact per-position margin decomposition)

Mean margin contributions by emitted-byte class:

| path | letter emission | space emission |
|---|---|---|
| token-emb (prior) | **−0.22** (opposes!) | −0.04 |
| L0 base / wave | +0.17 / +0.50 | −0.12 / −0.21 |
| L1 base / wave | +0.27 / **+1.52** | +0.27 / **+0.36** |

- **Word-internal letters are produced by the wavelet paths** (L1-wave
  dominates 3× over everything else); the token-embedding prior votes
  *against* the produced letter on average — production is
  context-over-prior, exactly as in the training-side attribution.
- **Space emission is driven by L1** (wave + base), with L0-wave actively
  *opposing* (negative) — the lower layer resists ending the word, the
  deeper layer ends it. Production boundaries are a layer disagreement
  resolved in favor of the deeper layer.

## 3. Mode-level production is distributed at the last layer

Mean |per-mode margin contribution| when emitting a boundary vs a letter:
no mode specializes to boundary emission at L2 (mixed: k2 0.82 / k4 0.97 /
k5 0.73 at boundaries vs k1 0.88 / k2 0.90 / k5 0.93 at letters). Local
canonical events *can* localize to a single mode (e.g. `frien→d` → channel
12·mode 1, per the mode-attribution report), but **boundary emission is a
distributed collective decision** — an honest contrast between local
morphological events (single-mode) and global structural events
(distributed).

## 4. The complete production loop, stated

$$\text{speak} = \text{softmax}\Big(\underbrace{\text{head}}_{\text{readout}} \cdot \underbrace{\big[\text{emb} + \sum_l (\text{base}_l + \text{wave}_l)\big]}_{\text{residual sum of exact, attributable parts}}\Big)$$

driven by a word-clocked filter bank whose Δ pulses at boundaries, where
word-internal emissions are wavelet-path decisions against the embedding
prior, and word endings are L1-over-L0 disagreement resolutions. Every term
on the right-hand side was measured exactly from the checkpoint.

## Caveats

- This run used the **matrix-tier checkpoint (d40L2)** because the
  benchmark overwrote the earlier d32L3 interpretation checkpoint (same
  tag). A restored d32L3 copy is retraining under a protected tag; all
  production findings here are architecture-level, not shape-specific, and
  should replicate (not yet replicated — stated honestly).
- One generation (one prompt, one seed, T=0.3). The trace figure
  generalizes the mechanism; per-event localizations are samples.
- No newlines were generated in this trace (null entries in the class
  table).

## Reproduction

```bash
.venv/bin/python experiments/W7BC_generation_analysis.py
```
