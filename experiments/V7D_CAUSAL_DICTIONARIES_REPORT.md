# Causal Dictionaries, the Context Test, and Mode-Level Attribution

**Date:** 2026-09-09 · **Subject:** `wskan7bc_ultrachat_100k_s42`.
Script: `experiments/V7_causal_dictionaries.py` · numbers:
`figures/v7d_causal_summary.json`. Intervention CE on 32×512 held-out
blocks (baseline CE 1.2923).

## J. The correlational dictionaries are now causal (intervention battery)

| intervention | ΔCE | reading |
|---|---|---|
| zero M_B[newline] rows (3 layers) | +0.048 | named write tables are **causally load-bearing** (residual path partially compensates) |
| zero M_B[punct] rows | +0.054 | 〃 |
| zero M_B[digit] rows | +0.033 | 〃 |
| zero L0 input cluster c0 (n=11) | +4.24 | every L0 cluster is collectively critical |
| zero L0 input cluster c1 (n=9) | +3.41 | 〃 |
| zero L0 input cluster c2/c3 | +2.59 / +2.62 | 〃 |
| **zero L0 cluster c4 (n=1, boundary specialist)** | **+1.08** | **a single specialist channel is causally mighty** |
| zero L2 output hub ch1 | +0.037 | hub real but modest alone |
| zero L2 family F0/F1/F2/F4 (n≈230 each) | +0.29…+0.36 | the four big families are co-equal workhorses |
| zero L2 family F5 (n=119) | +0.15 | secondary |
| **zero L2 family F3 (n=20, specialist reader)** | **+0.017** | **the rare specialist family is nearly redundant** |

Verdicts: the named tables, the channel dictionary (including the
boundary-specialist singleton), the L2 hub, and the big kernel families all
pass causal tests. Notable asymmetry: *singleton channel specialists are
load-bearing, but the small specialist **family** is not* — specialist
structure lives at channel level, not edge level.

## K. The context test: dialogue-turn hypothesis REFUTED

Candidate explanation for the unexplained residual-B variance
(R² 0.35–0.59 after byte identity + position): dialogue-turn state.

| predictor | R² (L0/L1/L2) |
|---|---|
| role (User vs Assistant turn) | 0.0004 / 0.0004 / 0.0005 |
| distance to turn boundary | 0.0013 / 0.0019 / 0.0025 |
| next-byte class (lookahead control) | 0.026 / 0.050 / 0.052 |

**Refuted.** The contextual write path does not track turn structure at
this scale. The unexplained context is something subtler (candidates:
word-identity/lexical statistics beyond the current byte — untested).
Recorded as an open item, honestly.

## L. Exact mode-level attribution: who decides `frien → d`?

The L2 wavelet path's +13.6 decision contribution (previous report)
decomposes **exactly** (all terms are read from the model state) into
per-mode and per-(channel, mode) pieces:

- per-mode logit contributions: k0 −1.76, **k1 +4.21**, k2 −1.38,
  **k3 −8.72**, k4 −1.32, k5 +0.46 — a fight between modes 1 and 3,
  integrated over output channels.
- top (i,k) pairs: **(i12, k1) +13.79** — a single named oscillator
  (channel 12, mode 1) carries the morphological completion — then
  (i19, k3) +4.91, (i21, k4) +3.75, opposed by (i18, k4) −3.16,
  (i26, k4) −2.58.

This is the deepest composition slice achieved: for a canonical prediction,
the deciding contribution is attributable to *individual named modes* —
exactly, not approximately.

## Where this leaves the "complete interpretation" frontier

Achieved: every component class has now been read *and* (except embedding
and W_z) causally tested or exactly decomposed; canonical predictions
decompose to named modes.

Still open, in order of size: (1) the non-turn "context" in residual-B
(~30–40% of write variance; turn hypothesis refuted, lexical statistics
untested); (2) full per-edge enumeration (now bounded by causal family +
circuit structure; the F3 result shows some structure is redundant);
(3) whole-model forward hand-simulation — still the acknowledged ceiling.

## Reproduction

```bash
.venv/bin/python experiments/V7_causal_dictionaries.py
```
