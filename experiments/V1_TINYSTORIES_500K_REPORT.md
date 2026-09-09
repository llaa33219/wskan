# V1 TinyStories 500k-Step Training Report

**Date:** 2026-09-05 · **Status:** capability demonstration, *not* a benchmark claim.
Supersedes nothing; complements `V1_TINYSTORIES_REPORT.md` (20k-step run).

## Setup

Same model and data as the 20k report: `WaveletStateKANLM` (94,272 params,
byte-level vocab 256, d=32, 3 layers, N=6 states/edge), 200,000 TinyStories
train-split stories (180.7M bytes), 1,000 held-out validation stories.

Differences from the 20k run:

- **500,000 steps** (batch 32, block 256 — ≈ 4.1B bytes seen, ≈ 23 epochs),
  AdamW lr 3e-3 with **cosine decay to 0**, grad clip 1.0, seed 42.
- Wall time 13,763 s (≈ 3 h 50 min) on one RTX 4070, ≈ 26.6 ms/step.
- Checkpoints (edge-function bundles) + generation samples every 10,000 steps
  in `checkpoints/wskan_tinystories_lm_500k/`.

## Numerical-stability incident (important finding)

A first 500k attempt **diverged to NaN at step ~32,500** and later crashed in
cuBLAS (`CUBLAS_STATUS_EXECUTION_FAILED`) once NaN parameters reached the
complex GEMM. Root cause: `log_sigma` drifted large, so
`Abar = exp(-sigma * s)` underflowed to exact 0.0 in float32, and the complex
power `Abar ** lags` produced NaN gradients through `log(0)` in its backward
pass.

**Fix (in `models/V1_WSKAN.py`):** `sigma` and `s` are now clamped to
`[e^-4, e^2] ≈ [0.018, 7.39]`; gradients are zero outside the range, so the
unstable regime is unreachable. A non-finite-loss guard was also added to the
training loop. Verified adversarially: `log_sigma = log_s = ±10` yields finite
outputs and finite gradients. The clamp is a documented restriction of the
learnable wavelet scales. The restarted run passed the previous crash point
without incident and remained finite for all 500k steps.

## Results

- Init 5.559 (≈ uniform ln 256) → 2.5k: 1.72 → 5k: 1.59 → 25k: 1.58 →
  75k: 1.52 → **best eval 1.4996 @ step 375k** → final (lr → 0):
  **train 1.5872, eval 1.5369** (byte-level PPL ≈ 4.65).
- The eval loss plateaued around 1.55–1.63 after ~75k steps; the long tail of
  training plus cosine decay bought only ~0.04 nats over the 20k-step run.
  Interpretation: at 94k parameters the model is **capacity-bound, not
  step-bound** — more steps cannot substitute for more edge functions.
- No overfitting (eval ≈ train throughout; 23 epochs did not degrade eval).

## Qualitative test

Prompt `Once upon a time`, temperature 0.8, 200 bytes:

- **Step 50k:** `"...Ben was day, the big sochaudine on speckied. But shanked
  aid showes things. They leave a day surly..."` — mostly pseudo-words with
  correct spacing; some real function words.
- **Step 200k:** `"...Lily said, 'Thary in't go her a cantidnes... You are
  gave it to spertings..."` — dialogue punctuation, quoted-speech structure,
  more real words ("Lily said", "You are", "gave it").
- **Step 500k:** `"...thinger tha grew. She and Care wanted to her round and
  rewing. The little gundee and she wanded... One day, they wanderrey
  wanter..."` — clear story skeleton phrases ("The little ...", "One day,
  they ..."), character-name-like tokens, paragraph breaks; **still not
  coherent English sentences.**

Honest assessment: 500k steps visibly improved story *scaffolding* (names,
dialogue quotes, "One day" openings, paragraph rhythm) but did not cross into
grammatical coherence. Combined with the loss plateau, this supports the
capacity-bound interpretation above.

## Honest limitations

- Single seed, single dataset, no matched-parameter baseline — AGENTS.md
  benchmark rules (≥3 seeds, ≥3 datasets, competitor comparison) are not met;
  this is a capability/stability demonstration only.
- Byte-level modeling understates sample quality relative to BPE tokenizers;
  perplexity is per byte.
- The best eval (1.4996 @ 375k) vs final (1.5369) gap suggests mild eval-set
  noise; the checkpoint at 375k is technically the best by eval.
- Residual connections + final LayerNorm remain a documented deviation from
  pure KAN.

## Reproduction

```bash
.venv/bin/python experiments/V1_train_tinystories_lm.py \
    --steps 500000 --train-stories 200000 --eval-stories 1000 \
    --lr-schedule cosine --ckpt-every 10000 --eval-every 2500
```
