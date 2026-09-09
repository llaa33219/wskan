# Deep Interpretability Report: Closing the Four Debts

**Date:** 2026-09-09 · **Subject:** `wskan7bc_ultrachat_100k_s42` (the adopted
analysis configuration: V6 + feature-factorized B/C at rank 32).
Script: `experiments/V7_deep_interpret.py`. Figures: `fig_v7bc_*.png`,
numbers: `v7bc_deep_summary.json`. Probe: 2,048 bytes of held-in UltraChat.

## Debt 1 (B/C semantics) — resolved for the named half, with a surprise

Reading M_B/M_C directly (no probing — pure table lookup):

- **The named path self-organized to carry STRUCTURAL bytes, not content.**
  Strongest writers/readers: newline, digit, punct. Letters are nearly
  absent from the named path (`lower`, `const` rows are almost black).
  The rank-32 residual (hidden-state) path must carry the alphabetic
  content — a clean division of labor that emerged on its own:
  *discrete, byte-identifiable structure goes through the interpretable
  table; continuous content goes through the residual.*
- **Digit behaves as a structural token** in UltraChat (list/numbering
  formatting), not as letter-like content.
- **Newline's read routing migrates with depth**: read peaks move from
  modes 1–2 (L0) → mode 1 (L1) → modes 3–4 (L2) — the most visible
  depth-reorganization effect in the tables.
- **Space is read more than written** (structural roles differ by side).
- **Null result (honest):** the hypothesis that boundary bytes write
  preferentially into long-memory modes is *refuted* — correlation between
  space's write strength and mode half-life ≈ 0.05 / 0.006 / −0.06 per
  layer. The word clock acts through Δ (decay speed), not through targeted
  writes into slow modes.

## Debt 2 (edge atlas) — family structure established; high rank confirmed

Clustering all 1,024 edge kernels per layer (k-means, 6 clusters):

- **Six recurring families**: step-like kernels, smooth saturating ramps,
  narrow spikes, V-shaped transients, notches, and rare **long-tailed**
  kernels (the two smallest clusters, 16–30 edges — the dormant
  long-memory edges again).
- **No sustained oscillation anywhere** — consistent with Q ≈ 0.6
  (overdamped); the wavelet advantage is phase shape, not ringing.
- **Effective rank of g ≈ 28.8–29.4 of 32 per mode** — quantified
  confirmation that the edge structure is genuinely high-rank (this is why
  V7's rank-32 filter constraint cost +0.078 nats). Family-level
  description, not compression, is the right analysis granularity.

## Debt 3 (channel dictionary) — established: generalists + antipodal specialists

Per layer, channels clustered by their z-scored profiles over byte class ×
position-in-word (12 features):

- **"Many generalists + few specialists"**: every layer has large neutral /
  always-on / always-off clusters plus 1–3 single-channel specialist
  clusters carrying the sharpest structure (boundary detector,
  word-initial detector, uppercase-suffix detector) — and the specialists
  come in **antipodal pairs** (mirror-image profiles).
- **Depth shifts the code from byte class to word position**: L0 contrasts
  live on class columns; L1/L2 contrasts live on position-in-word columns —
  the network literally retunes from *what byte* to *where in the word*.

## Debt 4 (clock drivers) — measured: the clock is distributed

Top-8 driver channels (|W_dt| weighted by activation) carry 33–37% of the
clock drive per layer. Cross-referencing with the channel dictionary: the
drivers are mostly **mainstream generalist channels**, not the specialist
clusters (only L2 has 2/8 drivers in its small cluster). **There is no
dedicated "clock neuron"** — consistent with the causal battery (Δ
information is load-bearing at arbitrary positions, not one module).

## Updated interpretability ledger (vs the previous audit)

| component | before | now |
|---|---|---|
| B/C named half (~18%) | tables existed, unread | **read: structural-byte routing, depth migration** |
| B/C residual (~18%) | probing needed | probing needed (unchanged) |
| g (36%) | population stats only | **family atlas + effective rank quantified** |
| channels / residual stream | never established | **dictionary at cluster level** |
| clock internals | byte-class behavior only | **driver concentration measured (distributed)** |
| W_z (3%), embedding (8%) | unexamined | unexamined |
| composition | open | open |

Remaining open (honest): residual-B/C feature semantics, per-edge roles
inside families, W_z, embedding geometry, and full mechanistic simulation.
The single-cluster-level channel dictionary and driver cross-ref used one
seed/checkpoint; replication is cheap.

## Reproduction

```bash
.venv/bin/python experiments/V7_deep_interpret.py
```
