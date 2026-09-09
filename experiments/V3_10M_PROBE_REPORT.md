# V3 10M-Parameter Probe Report (UltraChat)

**Date:** 2026-09-07 · **Status:** quick probe, single seed, 10k steps —
explicitly NOT a benchmark verdict. Question: was the near-delta memory at
~100k params a capacity artifact?

## Setup

- ~10M params each: wskan3/wskan3real d=256, L=5, N=6 (9,915,648);
  mamba2 d=512, L=6, state 64, heads 16 (10,055,200).
- UltraChat (~180M bytes), block 512, batch 32, lr 1e-3 cosine, 10,000 steps
  (~164M bytes seen ≈ 0.9 epoch), seed 42.
- Engineering notes: V3 uses per-chunk gradient checkpointing + per-chunk
  torch.compile (789 ms/step, 3.7 GB); mamba2 uses HF gradient checkpointing
  + chunk_size=32 compute tiling (math-identical) to fit memory; mamba2 had a
  brief init transient (loss 8.1 at step 1, recovered by step 500).

## Result 1: scale DOES unlock longer memory — user's hypothesis confirmed

Trained wskan3-10m, probed on a multi-turn prompt:

| Layer | ⟨σ⟩ (100k-scale: ~5.9) | σ at clamp ceiling | half-life (tokens) |
|---|---|---|---|
| L0 | 0.90 | 0% | 4.6 |
| L1 | 1.00 | 0% | 4.9 |
| L2 | 1.11 | 0% | 4.2 |
| L3 | 1.28 | 0% | 3.0 |
| L4 | 1.57 | 0% | 2.2 |

At 100k params, a third of σ modes were pinned at the clamp ceiling with
~1-token half-life. At 10M, **no mode touches the ceiling** and decay is
4–6x slower. The near-delta kernels were indeed a small-capacity artifact.

## Result 2: but explicit recall still fails

The recall probe (memorize number + name, interleaved story, then asked)
still produces fluent non-answering text. Half-life ~5 tokens is longer but
far short of the ~400-byte probe distance, and 10k steps is too few for
semantic recall anyway. Long-range *use* of memory at 10M is open.

## Result 3: undertrained performance ordering (read with care)

| Model | best eval CE | PPL |
|---|---|---|
| mamba2 | **0.8443** @9.5k | 2.33 |
| wskan3real | 0.8733 @9.5k | 2.39 |
| wskan3 | 0.9171 @10k | 2.50 |

At this 10k-step probe the wavelet variant trails — the reverse of the 100k
ranking. Cautions: (a) wskan3 was still improving at the cutoff (best = final
step), mamba2 had already peaked @9.5k; (b) single seed; (c) 10k steps is
~0.9 epoch — far from converged. A plausible mechanism: with fast-decay
short kernels (100k regime) oscillatory phase shaping pays immediately; with
slow-decay long kernels (10M regime) the oscillatory modes have more
optimization surface and may need more steps. **No claim is made either way
until a longer run.**

## Next steps

- 50k-step run at 10M (est. 7–11 h per V3 model on one GPU) for a real
  verdict; mamba2 needs ~4 h (it is faster per step).
- Multi-seed only after the longer-run signal justifies it.

## Reproduction

```bash
.venv/bin/python experiments/V1_train_tinystories_lm.py --model {wskan3|wskan3real|mamba2} \
    --scale 10m --dataset ultrachat --block 512 --steps 10000 --lr 1e-3 \
    --lr-schedule cosine --seed 42 [--compile  # mamba2 only]
```
