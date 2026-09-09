# V4 Memory Analysis: Long-Memory Modes Appeared, But Stay Dormant at 100k

**Date:** 2026-09-07 · **Subject:** `wskan4_ultrachat_100k_s42/latest.pt` ·
**Question:** did V4's per-mode Δ + freed σ change the near-delta memory seen
in V3 (half-life ~1 token, uniform)?

## Per-mode effective half-life (0.693 / (σ_k · ⟨Δ_k⟩), real prompt)

| Layer | median | p90 | p99 | **max** |
|---|---|---|---|---|
| L0 | 0.95 | 1.29 | **50.1** | **168.4** |
| L1 | 0.97 | 1.18 | 3.0 | 100.1 |
| L2 | 0.83 | 1.21 | 6.7 | 38.3 |

V3 comparison: uniform ~1.1–1.2 tokens, no tail (per-channel Δ forced all
modes on a channel to share one timescale — a long-memory tail was
structurally impossible).

**V4's per-mode Δ created a dedicated slow-channel tail**: a small set of
modes with 40–170-token memory now exists. The *machinery* for long-range
tokens is present for the first time.

## But the model does not route through them (yet)

Readout-weight share by half-life bucket (|g|·⟨Δ⟩·|C| weighted):

| Layer | hl > 10 tok | 3–10 tok | < 3 tok |
|---|---|---|---|
| L0 | 0.12% | 0.78% | 99.09% |
| L1 | 0.05% | 0.46% | 99.48% |
| L2 | 0.79% | 0.38% | 98.83% |

~99% of readout weight stays on <3-token modes. The long-memory modes are
**latent capacity**: present, differentiated, unused at this scale.

## Recall probe

The explicit memorize-then-recall probe (number + name, interleaved story,
then asked) still fails: V4 produces fluent generic dialogue text instead of
the stored facts.

## Honest interpretation

- Capability and usage decoupled: V4 *can* express 100+-token memory (V3
  could not), but 100k params on UltraChat does not *incentivize* it — the
  loss-optimal solution here remains local.
- This is consistent with the 10M probe (σ dropped 6x, half-life 2–5 tokens):
  memory length scales with capacity. The V4 tail is the structure that
  should *activate* first at larger scale — a testable prediction for any
  future 10M+ V4 run.
- Claim discipline: "V4 has long-memory modes" is true structurally; "V4 uses
  long memory" is false at 100k. Reports should keep the two apart.
