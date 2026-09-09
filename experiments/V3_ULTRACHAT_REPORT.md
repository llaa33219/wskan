# V3 UltraChat Report: Does Long Memory Matter? (2-Dataset Evidence)

**Date:** 2026-09-06 · **Status:** 3-seed, second dataset. Motivation: TinyStories'
simple grammar may never require long-range memory; UltraChat-200k multi-turn
QA should. Protocol change vs TinyStories runs: **block 512** (vs 256) so that
longer dependencies can exist within a sample; everything else identical.

## Setup

- Data: `HuggingFaceH4/ultrachat_200k` train_sft, rendered as
  `User: ... / Assistant: ...` turns, first ~180M bytes (parity with the
  TinyStories byte count); eval on 500 test_sft conversations.
- Models/params unchanged: wskan3 101,856 · wskan3real 101,856 · mamba2 97,592.
- 100k steps, batch 32, block 512, AdamW 3e-3 cosine, seeds {42, 123, 2024}.

## Results (best eval CE, byte-level)

| Model | s42 | s123 | s2024 | mean ± std |
|---|---|---|---|---|
| **wskan3 (wavelet)** | **1.1711** | 1.1663 | **1.1510** | **1.1628 ± 0.0105** |
| mamba2 | 1.1645 | **1.1527** | 1.2203 | 1.1792 ± 0.0361 |
| wskan3real (no ω) | 1.2196 | 1.2208 | 1.1998 | 1.2134 ± 0.0118 |

| Paired difference | s42 | s123 | s2024 | mean ± std |
|---|---|---|---|---|
| wskan3 − mamba2 | +0.0066 | +0.0136 | −0.0693 | −0.0164 ± 0.0460 |
| **wskan3 − wskan3real** | **−0.0485** | **−0.0545** | **−0.0488** | **−0.0506 ± 0.0034** |

## Finding 1: the wavelet contribution GROWS with task complexity (hypothesis supported)

The ω-ablation gap widened from **−0.0301 ± 0.0034** (TinyStories) to
**−0.0506 ± 0.0034** (UltraChat) — a 1.7x increase, extremely consistent
across seeds. On richer, longer-range text, oscillatory (wavelet) modes are
worth more. This directly supports the motivation for this experiment.

## Finding 2: but the model still does NOT use long memory (hypothesis partially refuted)

Probing the trained wskan3 (s42) on a multi-turn UltraChat-style prompt:

- ⟨Δ⟩ ≈ 0.11–0.13 per layer (TinyStories: ~0.10) — unchanged.
- **Effective state half-life ≈ 1 token** on all layers (TinyStories: ~1.2).
- σ at clamp ceiling: 32–43% of modes (TinyStories: 33%) — the clamp binds
  *harder* here.
- A direct recall probe (number + name stated, story interleaved, then asked)
  fails: the model rambles generically instead of answering.

Honest conclusion: at ~100k params, **both** on TinyStories and UltraChat,
the learned dynamics are near-local. The wavelet advantage does not come from
longer memory — it comes from better *shaping of a very short kernel*, and
that shaping matters more on complex text. The "long memory" capacity of the
wavelet-SSM remains unused at this scale; whether it activates at larger
scale/params is an open question (V4 candidate: raise the σ ceiling, longer
blocks, explicit retrieval tasks).

## Finding 3: wskan3 vs mamba2 — statistically tied, wskan3 more consistent

Mean favors wskan3 (−0.0164) but the paired differences flip sign across
seeds (mamba2 wins s42/s123 narrowly; loses s2024 by a lot — its s2024 run
was an outlier, best @57.5k then drifted). wskan3's seed std is 3.4x smaller
(0.0105 vs 0.0361). Correct claim: **parity in mean, better seed-robustness**,
on this dataset too.

## Qualitative (wskan3 s42, recall probe generation)

The model produces fluent dialogue-form text that matches the UltraChat
register but fails factual recall — consistent with the 1-token half-life
measurement. Sample stored in the analysis transcript; per-token quality is
visibly ahead of TinyStories-checkpoint outputs.

## Limitations

- Two datasets total; project rule asks ≥3 for firm claims — one more
  dataset (e.g., a code or math corpus) is the next step.
- mamba2 s2024's outlier run inflates its std; a 4th seed would sharpen the
  comparison.
- Block 512 with byte tokens still only spans ~2–4 sentences of dialogue;
  "long memory" at kilobyte scale is untested.

## Reproduction

```bash
.venv/bin/python experiments/V1_train_tinystories_lm.py --model {wskan3|wskan3real|mamba2} \
    --dataset ultrachat --block 512 --steps 100000 --seed {42|123|2024} \
    --compile --lr-schedule cosine --ckpt-every 25000
```
