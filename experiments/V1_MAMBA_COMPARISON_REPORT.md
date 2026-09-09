# V1 vs Mamba-2: Matched-Parameter TinyStories Comparison

**Date:** 2026-09-05 · **Status:** controlled comparison, single seed — see limitations.

## Protocol (identical for both models)

- Byte-level LM (vocab 256, no tokenizer), residual stack, tied embedding/head.
- Data: 200,000 TinyStories train stories (180.7M bytes); eval on 1,000 held-out validation stories.
- 500,000 steps, batch 32, block 256, AdamW lr 3e-3, cosine decay to 0, grad clip 1.0, seed 42.
- Same script, same data order: `experiments/V1_train_tinystories_lm.py --model {wskan|mamba2}`.

| | WSKAN V1 | Mamba-2 (HF `transformers` 5.16.1) |
|---|---|---|
| Params | 94,272 | **97,592 (+3.5%)** |
| Config | d=32, 3 layers, N=6 states/edge | d=64, 3 layers, expand 2, 8 heads, head_dim 16, state 8, conv 4 |
| Step time (eager) | 26.6 ms | 125 ms |
| Step time (compiled) | untested | 20.9 ms (used for the run) |
| **Best eval CE** | 1.4996 @375k | **0.7618 @490k** |
| Best eval PPL (per byte) | 4.48 | **2.14** |
| Final eval CE @500k | 1.5369 | 0.8681 |
| Wall time | 3h49m | 3h13m (compiled) |

## Result

**Mamba-2 wins decisively: −0.74 nats at matched (slightly larger) parameter
count.** The +3.5% parameter advantage cannot explain a gap of this size.
Qualitatively the difference is equally clear (temperature 0.8, 200 bytes):

- **WSKAN @500k:** `"...thinger tha grew. She and Care wanted to her round and
  rewing. The little gundee and she wanded..."` — story scaffolding, but not
  grammatical.
- **Mamba-2 @500k:** `"Once upon a time, there was a giant for a while there.
  ... The old man started to cry and grab the bigger..."` — mostly grammatical
  English with story structure; occasional word-level noise.

## Analysis: why Mamba-2 wins here

1. **WSKAN's recurrent mode is LTI; Mamba is input-dependent.** WSKAN's edge
   kernels K (the wavelets) are fixed after training: every input byte gets
   convolved with the same filter bank and summed. Mamba-2's B, C, and Δ are
   *functions of the current input*, so it can selectively store/forget
   content — the mechanism that matters most for language. This is the core
   architectural gap, and it points to the natural V2 direction: make the
   wavelet-SSM's B/C (or Δ) input-dependent, i.e. a *selective* Wavelet-State
   edge.
2. **No multiplicative gating in WSKAN.** Mamba blocks gate their output
   (z-branch); WSKAN layers are purely additive sums of edge responses.
3. WSKAN's eval plateaued by ~75k steps (capacity-bound), while Mamba-2 kept
   improving past 400k — Mamba uses its ~97k parameters far more effectively
   for this task.

## Incident log (reproducibility honesty)

- **First Mamba run was invalid.** The adapter passed pre-shifted `labels` to
  HF's `Mamba2ForCausalLM`, whose loss shifts labels internally — the model
  was trained to predict *two bytes ahead*. Its eval loss (1.27) looked fine,
  but qualitative samples were incoherent character soup, which exposed the
  bug (this is exactly why the project requires qualitative tests). Fixed by
  computing cross-entropy manually; this report uses only the corrected run.
- WSKAN's first 500k attempt hit the NaN-underflow bug documented in
  `V1_TINYSTORIES_500K_REPORT.md`; fixed via the σ/s clamp before the run
  reported here.

## Limitations

- **Single seed (42), single dataset.** Per project rules, ≥3 seeds and ≥3
  datasets are required before this comparison can be called conclusive.
  Treat the 0.74-nat gap as strong but preliminary evidence.
- Byte-level, ~100k params only; ranking could shift at other scales or with
  BPE tokenization.
- Mamba-2 received +3.5% parameters; a strictly-matched rerun is cheap to add.
- WSKAN ran eager while Mamba-2 ran compiled — irrelevant to loss, but
  throughput numbers are not like-for-like in that row.

## Reproduction

```bash
.venv/bin/python experiments/V1_train_tinystories_lm.py --model wskan  --steps 500000 --lr-schedule cosine --ckpt-every 10000
.venv/bin/python experiments/V1_train_tinystories_lm.py --model mamba2 --compile --steps 500000 --lr-schedule cosine --ckpt-every 10000
```
