# V3 Multi-Seed Ablation Report: The Wavelet Contribution Is Real

**Date:** 2026-09-06 · **Status:** 3-seed controlled comparison (per project
rule: ≥3 seeds). Single dataset (TinyStories) — multi-dataset remains open.

## Protocol

Byte-level LM (vocab 256), 200,000 TinyStories train stories, 1,000 held-out
validation stories, 100,000 steps, batch 32, block 256, AdamW lr 3e-3, cosine
decay, grad clip 1.0, seeds {42, 123, 2024}, same script for all models.

Three models:
- **wskan3**: V3 selective wavelet-SSM (complex modes λ = −σ + iω, oscillatory).
- **wskan3real**: identical but ω frozen at 0 — pure decay modes (Mamba-like
  dynamics). Ablates *only* the wavelet/oscillation; same parameter count.
- **mamba2**: HF Mamba-2 baseline.

All V3-config runs use the pre-norm residual fix (see incident below).

## Results (best eval CE over 100k steps, byte-level; lower is better)

| Model | s42 | s123 | s2024 | mean ± std |
|---|---|---|---|---|
| **wskan3 (wavelet)** | **0.7986** | **0.8231** | **0.8103** | **0.8107 ± 0.0123** |
| mamba2 | 0.8193 | 0.8298 | 0.8137 | 0.8209 ± 0.0082 |
| wskan3real (no ω) | 0.8281 | 0.8502 | 0.8441 | 0.8408 ± 0.0114 |

Paired per-seed differences (same seed, same data order):

| Comparison | s42 | s123 | s2024 | mean ± std | wins |
|---|---|---|---|---|---|
| wskan3 − mamba2 | −0.0207 | −0.0067 | −0.0034 | **−0.0103 ± 0.0092** | 3/3 |
| wskan3 − wskan3real | −0.0295 | −0.0271 | −0.0338 | **−0.0301 ± 0.0034** | 3/3 |

## Conclusions

1. **The wavelet contributes, measurably and consistently.** Removing the
   oscillatory structure (ω=0) while keeping everything else identical costs
   +0.030 nats (PPL 2.25 → 2.32), consistent across all 3 seeds
   (±0.0034). The damped-oscillation (wavelet) function space is doing real
   work that pure exponential decay cannot.
2. **wskan3 beats mamba-2 on all 3 seed pairings**, mean −0.0103 nats.
   Honest reading: a *small, consistent* edge — not a blowout. With one
   dataset and +4.3% parameters (101,856 vs 97,592), the correct claim is
   **"matches or slightly exceeds Mamba-2 at ~100k scale"**, not superiority.
3. **The striking one:** wskan3real (0.8408) is *worse* than mamba2 (0.8209).
   Our selective machinery with real modes does not match Mamba-2 — it is
   precisely the wavelet modes that close the gap and tip V3 ahead. The
   edge-function choice is the differentiator, which is the thesis of this
   project.
4. Qualitatively (V3 s42): `"Once upon a time, there lived a picture in the
   park. One day, she went to the paper..."` — grammatical, story-structured,
   on par with Mamba-2 samples.

## Incident log: the residual-scale blowup (root-caused and fixed)

- 2 of 5 initial V3-config runs died with NaN (wskan3real/s123
  deterministically at step 54,194 — reproduced twice, bit-identical).
- Diagnosis via RNG-replay + forward hooks: the **residual stream blew up
  multiplicatively across layers** (55 → 2.5e6 → 5.7e25 in three layers).
  Mechanism: the output gate `y ⊙ SiLU(W_z x)` makes layer gain superlinear
  in the residual scale — a positive feedback loop that V1/V2's LTI paths
  never triggered but V3's selectivity did.
- **Fix: pre-norm residual blocks** (`x + layer(norm(x))`, the same choice
  Mamba makes internally). The poisoned batch went from 5.7e25 to bounded
  1.9e3 residual scale; all 6 rerun seeds passed every previous NaN point
  cleanly. +192 parameters (3 LayerNorms), included in the count above.
- Side lesson recorded: parameter magnitudes stayed small throughout
  (pmax < 5); weight-based monitoring would never have caught this.
  Activation monitoring would.

## Limitations

- Single dataset (TinyStories); ≥3 datasets required for a firm claim.
- Byte-level, ~100k params; scaling behavior unknown.
- wskan3 carries +4.3% params over mamba2 (the direction of the wskan3real
  result makes this moot for the wavelet-contribution claim, which is
  parameter-matched).
- V1/V2 numbers in earlier reports predate the pre-norm fix; they are
  comparable only within their own reports.

## Reproduction

```bash
for m in wskan3 wskan3real mamba2; do
  for s in 42 123 2024; do
    .venv/bin/python experiments/V1_train_tinystories_lm.py --model $m --seed $s \
      --compile --steps 100000 --lr-schedule cosine --ckpt-every 25000
  done
done
```
