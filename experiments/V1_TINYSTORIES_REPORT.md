# V1 TinyStories Training Report

**Date:** 2026-09-05 · **Status:** capability demonstration, *not* a benchmark claim

## What was done

A byte-level language model built from V1 Wavelet-State-KAN layers (recurrent /
SSM mode, FFT convolution path) was trained on TinyStories.

### Model (94,272 parameters — under the 100k budget)

| Component | Params | Detail |
|---|---|---|
| Token embedding (tied LM head) | 8,192 | 256 vocab × d=32, init N(0, 0.02²) |
| 3 × WaveletStateKANLayer | 86,016 | d=32→32, N=6 states/edge, 28 params/edge |
| Final LayerNorm | 64 | |
| **Total** | **94,272** | verified programmatically |

### Setup

- **Data:** `roneneldan/TinyStories` — 200,000 train-split stories
  (180,723,994 bytes after UTF-8 encoding; byte-level, no tokenizer), 1,000
  validation-split stories held out for eval (793,374 bytes).
- **Training:** 20,000 steps, batch 32, block 256, AdamW lr 3e-3, grad clip
  1.0, seed 42. Total wall time ≈ 9 min on one RTX 4070.
- **Artifacts:** `checkpoints/V1_tinystories_lm/` — `ckpt_step*.pt` (the saved
  bundle of edge *functions*: λ, g, s, μ, w_base, w_wav per edge),
  `edge_functions_step*.json` (dominant (σ, ω) per edge), `samples_step*.txt`,
  `train_log.csv`, `latest.pt`.

## Results

- Init loss 5.5594 ≈ ln(256) = 5.545 (correctly calibrated uniform init).
- Final: **train 1.607, eval 1.578** (byte-level perplexity ≈ e^1.578 ≈ 4.8).
- Eval tracked train closely throughout (no overfitting at this scale; eval
  occasionally below train because eval batches are a fixed small sample).

Loss trajectory (eval): 5.48 → 1.80 @1k → 1.70 @2k → 1.61 @10k →
1.56 @17k → **1.58 @20k**.

## Qualitative test (required by project policy; numbers alone are not enough)

Prompt `Once upon a time`, temperature 0.8, 200 bytes:

- **Step 5,000:** `"...Sarres, Reling the big the hore rouspontanten tinnese
  there. He was now on tlit man..."` — word boundaries and common short words
  ("the", "He was", "They said") learned; most words are pseudo-English.
- **Step 20,000:** `"...there the on the cime sis. They and sookid thit day u
  toke romply and the broud oo do like. at the tored on the bill ron the came.
  The rabey ha undy and said some the was wan..."` — character statistics,
  spacing, punctuation rhythm, and frequent function words ("the", "and",
  "said", "was") are captured, but **the model does not yet produce coherent
  sentences or story structure**. This is the expected regime for a ~100k
  byte-level model: it has learned English morphology statistics, not
  semantics.

## Honest limitations

- **Single seed (42), single dataset.** AGENTS.md rules for benchmark-grade
  claims (≥3 seeds, ≥3 datasets, competitor baselines) are *not* met; this
  report only demonstrates that the V1 architecture trains stably and learns
  language statistics at the ~100k scale. No superiority claim is made.
- Byte-level modeling was chosen to keep the vocabulary tiny under the
  parameter budget; byte-level samples always look noisier than BPE samples
  at equal loss. Perplexity figures are per **byte**, not per token.
- Residual connections and a final LayerNorm are used around/between KAN
  layers for trainability — a documented deviation from "pure" KAN.
- The model is too small to judge the architecture's language-modeling
  quality; a fair test needs matched-parameter baselines (e.g. MLP mixer,
  Mamba block) at the same budget, which is future work.

## Reproduction

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt datasets
.venv/bin/python experiments/V1_train_tinystories_lm.py \
    --steps 20000 --train-stories 200000 --eval-stories 1000
```
