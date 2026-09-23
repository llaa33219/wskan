# Final Interpretability Battery: The Last Five Debts

**Date:** 2026-09-09 · **Subject:** `wskan7bc_ultrachat_100k_s42`.
Script: `experiments/V7_final_interpret.py` · figures `fig_v7c_*.png`,
numbers `v7c_final_summary.json`.

## E. Residual-B semantics — the content path is byte-identity addressing that contextualizes with depth

Variance of the residual (content) write path explained by each predictor
(mean R² over all 192 write directions per layer):

| layer | byte identity (top-32) | position-in-word | byte class | prev-byte class |
|---|---|---|---|---|
| L0 | **0.977** | 0.124 | 0.114 | 0.027 |
| L1 | **0.655** | 0.226 | 0.171 | 0.083 |
| L2 | **0.410** | 0.163 | 0.100 | 0.062 |

At L0 the "uninterpretable" residual path is **almost pure byte-identity
lookup** (R² 0.98) — a content-addressable write table. With depth,
identity share falls and positional/contextual share rises: writes go from
*what byte* to *where in the word / what context*. Combined with the named
tables (structural bytes), the full write gate now reads as:
**named features (structure) + byte identity (content, L0) + context
(deeper)** — the B/C debt is closed at this granularity.

## F. Circuit diagram — routing migrates across depth

Kernel-family × channel-cluster connectivity (per family, row-normalized):

- **L0**: families F2/F4 are specialists reading ch0+ch3, writing to the
  ch0/ch1 hubs; other families are flat readers.
- **L1**: a uniform routing band — every family reads and writes the
  ch2/ch3 cluster (ch0/ch4 dark). No family specialization; L1 is a
  pass-through router.
- **L2**: a **universal output hub on ch1** — all six families write into
  it; family F3 is the sole specialist reader of ch1.
- No family↔cluster diagonal; activity *migrates* across clusters with
  depth (F2/F4→ch0/1 → ch2/3 band → ch1 hub).

## G. W_z — mostly-open gate with cluster structure

Only 7–9% of gate weights are near zero: the output gate is predominantly
amplifying, not suppressing; its cluster-mean rows (figure) show which
channel clusters amplify which — a modulation layer, not a bottleneck.

## H. Embedding geometry — a clean functional manifold

- Within-class cosine similarity 0.47 vs between-class 0.07; PC1 carries
  52% of variance.
- Nearest neighbors are functionally perfect: `e→[i,a,o]` (vowels),
  `t→[T,c,s]` (dental/sibilant + case), `space→[,, ., :]` (separators),
  `5→[6,3,4]` (adjacent digits), `\n→[;,:,space]` (structural).
  The byte manifold is organized by *linguistic function*, learned, not
  imposed.

## I. Exact logit attribution — decisions emerge from opposing contributions

> **Erratum (2026-09-24, round-4 review):** the table below was computed by
> calling each layer on a **length-1 slice** of its input, which discards the
> scan state — the "wave" contributions are history-free and the numbers are
> therefore incorrect as path attributions (the qualitative finding that
> layers disagree survives; the magnitudes do not). The corrected method —
> full-sequence layer calls with the final LayerNorm applied per summand, so
> the parts sum exactly to the margin — is used in
> `W11_FINAL_INTERPRETATION_REPORT.md` §A.5 (canonical wskan11 checkpoint:
> frien→d margin 8.30; L1-wave +7.39 decisive against L0-wave −5.34).

The residual stream is additive and the head linear, so the decision margin
(top1 − top2 logit) decomposes **exactly** into path contributions:

| case | token-emb | L0 base/wave | L1 base/wave | L2 base/wave |
|---|---|---|---|---|
| `th→e` | −0.78 | **+5.50 / +3.13** | +0.39 / −1.05 | **−7.52 / +1.83** |
| `frien→d` | −0.56 | −0.04 / +0.22 | +0.34 / +0.87 | +4.25 / **+13.59** |
| `friend→next` | +2.15 | −4.32 / −0.54 | −2.19 / −0.62 | **−8.81 / +0.14** |

Findings: (1) layers **disagree** — `th→e` is L0's strong yes, L2-base's
strong no, L2-wave's partial restore: the decision is an integration of
opposing votes, not monotonic accumulation. (2) Morphological completion
(`frien→d`) is carried overwhelmingly by the **L2 wavelet path (+13.6)** —
the edge functions, not the base skip, decide word endings. (3) The token
embedding itself often votes *against* the eventual winner — context
overrules the prior.

## Ledger after this battery

| debt | status |
|---|---|
| residual B/C semantics | **closed** (identity→context gradient; E) |
| family→circuit roles | **closed at family granularity** (F); per-edge roles remain |
| W_z | **read** (open gate, cluster modulation) |
| embedding geometry | **closed** (functional manifold) |
| composition | **first exact slice** (path-level decision attribution; full hand-simulation remains the acknowledged ceiling) |

Remaining (honest): per-edge semantics inside families (6,144 objects — now
bounded by the family + circuit structure), and whole-model hand
simulation. Everything else on the audit list has been read directly from
the checkpoint.

## Reproduction

```bash
.venv/bin/python experiments/V7_final_interpret.py
```
