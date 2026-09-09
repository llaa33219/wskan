# V3 Interpretability Report: The Learned Edge Functions

**Date:** 2026-09-06 · **Subject:** `checkpoints/wskan3_tinystories_lm_s42/latest.pt`
(100k-step, best seed). Script: `experiments/V3_interpret.py`. Figures:
`experiments/figures/`. Summary data: `interpret_summary.json`.

## KAN interpretability is realized

Every edge (i, o) of every layer owns an explicit, plottable function
ψ_io(t) = Σ_k Re[g_iok e^{λ_ik t}] — 1,024 edge functions per layer, no
weight matrices. All findings below are read directly from these functions.

## Findings

### 1. Edge wavelets are genuine, diverse, non-degenerate (fig_edge_wavelets.png)

Top-|g| edges per layer show Morlet/Gabor-like packets: compact support,
2–5 zero crossings, varied amplitude (0.8–2.5), symmetry and oscillation
counts differing per edge. No collapsed (flat/spike) functions. The model
uses the wavelet family as a real function space, not a reparameterized bias.

### 2. A physical law emerged: σ ∝ ω (fig_modes.png)

Learned modes keep the S4D-Lin frequency grid (mean ω drift 0.9–1.6 rad,
growing with depth) but reorganize decay: **low-frequency modes learned slow
decay, high-frequency modes fast decay** (σ rising from ~0.3–2 at low ω to
the 6.5–7.4 ceiling at high ω), consistently across all 3 layers. This is
the classic damped-oscillator structure of physical wavelets — learned, not
imposed.

### 3. Edge sparsity with hub nodes (fig_edge_magnitude.png)

|g| is sparse: few dominant edges per layer, many near-dead edges (L2 most
sparse). Output nodes 17–20 act as hubs across layers (vertical stripes in
L1). p90/median |g| = 1.8–2.2. → KAN-style edge pruning is directly
applicable (future work).

### 4. The honest surprise: effective recurrent kernels are near-delta (fig_impulse.png)

The *nominal* impulse responses decay within ~3 tokens, non-oscillatory.
Measured on a real prompt, the effective state half-life is **~1.1–1.2
tokens** (⟨σ⟩≈5.9, ⟨Δ⟩≈0.1). Two implications, stated plainly:

- At byte-level TinyStories with ~100k params, the task is dominated by very
  local structure, and the model spent its capacity there. The wavelet's
  measured contribution (−0.030 nats, see V3_MULTISEED_ABLATION_REPORT.md)
  therefore comes from the **oscillatory phase shape of a very short kernel**
  (how it weighs the last 1–2 tokens), not from long-range wavelet memory.
- **33% of σ modes sit exactly at the clamp ceiling** (e² ≈ 7.39): the
  σ ∈ [e⁻⁴, e²] bound introduced as an underflow guard is *binding*. The
  model wants even faster decay. Raising the ceiling (or reparameterizing
  decay per-mode with dt) is a documented V4 consideration.

Note the asymmetry: static-mode wavelets (§1) retain rich oscillatory shape
because they are evaluated over a normalized input range, while recurrent
kernels live at token timescale where σΔ is large.

## What this does and does not show

- **Does:** KAN edge interpretability is real for WSKAN — functions, spectra,
  sparsity, and dynamics are all directly readable from the checkpoint.
- **Does not:** the analysis is on one seed/checkpoint and one dataset;
  long-memory behavior may emerge at larger scale or on tasks that need it
  (e.g., long-range retrieval). No pruning or symbolic-regression pass has
  been performed yet.

## Reproduction

```bash
.venv/bin/python experiments/V3_interpret.py
```
