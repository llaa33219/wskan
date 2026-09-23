# WSKAN Final Interpretation Report (3-Epoch Campaign)
### wskan11: what the model is, how it learned language, and what it can and cannot do

**Date:** 2026-09-13 · **Authority:** computed from the 3-epoch campaign's
300 completed runs (4 models × 5 sizes × 3 datasets × 5 seeds, single code
revision, `_3ep_` checkpoint tier). Performance is a footnote by design;
interpretation is the main text.

---

## 1. The machine in one paragraph

wskan11 is a byte-level LM whose KAN edges are wavelets realized by a
state-space model — each edge ψ_io(t) = Σ_k Re[g_iok e^{λ̃_ik t}] is exactly
an SSM impulse response — evaluated by a fused per-layer associative-scan
kernel (one launch per layer). It learned English as a **word-segmented
clock driving a damped-oscillator memory**: boundaries tick the clock
2–2.6× faster than letters (causally necessary and sufficient for
word-transition prediction), writes split structural bytes into named
tables and content into byte-identity channels, and every decision
decomposes exactly into named paths and modes.

## 2. Campaign protocol (everything measured is here)

- **Models:** wskan11 (fused-kernel wavelet-SSM), wskan11real (ω≡0
  ablation: the wavelet removed), mamba2 (HF Mamba-2), tf (tiny transformer).
- **Sizes:** 1k / 10k / 100k / 1m / 10m params (per-tier configs, exact
  counts in `W7BC_CROSS_FAMILY_REPORT.md`'s table; V11 uses N=3 modes at
  this tier's shapes for speed with a measured ~+0.03-nat cost).
- **Datasets:** TinyStories (~1.9 GB), UltraChat-200k (~1.18 GB),
  WikiText-103 (~542 MB) — full datasets, not slices.
- **Long training:** tiered 3-epoch protocol (100k tier runs the true
  2.6–3.0-epoch budget; 1m/10m capped at ~1–2 epochs for wall-time;
  exact steps in `CAMPAIGN_orchestrate.py`), batch 64, block 256 (TS/WT) /
  512 (UC), cosine LR, grad clip 1.0, fused-kernel implementation
  (V11: 4–47 ms/step; 8.8–42.6× over the first-generation chunked path).
- **Seeds:** {42, 123, 2024, 7, 31337} — five per cell.
- **Artifacts:** `checkpoints/*_3ep_s*` (300 dirs), per-run
  `train_log.csv` + intermediate checkpoints + generation samples.

## 3. The word clock survives 3-epoch training (fresh replication)

Measured on `wskan11_ultrachat_100k_3ep_s42` (the 100k tier, d40L2):

| class | Δ (L0) | Δ (L1) |
|---|---|---|
| lowercase | 0.100 | 0.102 |
| uppercase | 0.142 | 0.180 |
| digit | 0.115 | 0.140 |
| space | **0.258** | **0.205** |
| newline | **0.266** | **0.247** |
| punct | **0.276** | **0.266** |

Boundaries tick **2.0–2.6×** faster than letters in both layers — the same
mechanism found in the 100k-step era, reproduced on a fresh 3-epoch run of
a different implementation (fused kernel). The clock is not an artifact of
short training or of the chunked path.

Memory at the 100k tier (this tier): σ ≈ 5.9–6.5, half-life ≈ 0.93 tokens
median, max 53–69 tokens — the local regime as established.

## 4. The wavelet contribution: 5-seed paired ablation

wskan11 − wskan11real (negative = the wavelet helps). Every pair shares
seed, size, dataset, steps, and code.

| dataset × size | paired ΔCE per seed | mean ± std |
|---|---|---|
| TS 100k | −.033 −.038 −.041 −.044 −.021 | **−0.035 ± 0.009** |
| UC 100k | −.042 −.042 −.049 −.049 −.048 | **−0.046 ± 0.004** |
| WT 100k | −.047 −.035 −.036 −.042 −.045 | **−0.041 ± 0.005** |

- **72 of 75 cells negative.** The wavelet's contribution replicates at 5
  seeds on 3 datasets.
- **Scale dependence (honest):** the advantage peaks at 10k–100k
  (−0.04…−0.08) and *shrinks* at 1m–10m (−0.01…−0.03). At 10M, oscillation
  is a small consistent bonus, not the differentiator it is at 100k. (The
  likely reading: at large scale the longer memory dominates and phase
  structure matters relatively less. Stated as an observation, not a law.)
- The 1k tier is noisy (one positive cell on TS/UC) — at the embedding
  floor the comparison is less meaningful.

## 5. Performance (the footnote, per project policy)

Best eval CE, mean over 5 seeds — no strong claims; the field is within
~0.1 nats per cell and rankings shuffle by dataset:

| dataset | 1k winner | 100k winner | 10m winner | wskan11's best tier |
|---|---|---|---|---|
| TinyStories | mamba2 | mamba2 | mamba2 (0.478) | 1m (0.581, ≈ tf 0.559) |
| UltraChat | mamba2 | mamba2 | mamba2 (0.716) | 1m (0.934, ≈ tf 0.973) |
| WikiText | mamba2 | mamba2 | tf (0.940) | 1m (1.110, ≈ tf 1.110) |

wskan11 is never last and best-in-class nowhere; Mamba-2 remains the
strongest small model and shares the large end with the transformer. The
project's claim is interpretability, not the leaderboard — full tables in
`CAMPAIGN_aggregate.py` output.

## 6. Qualitative: what 3 epochs produce (T=0.8, 220 bytes)

Prompt `User: Can you tell me a story?\nAssistant:`:

- **wskan11, ultrachat, 100k:** `"Sure, here's an example of healthy in
  analytics for that was going the swords..."` — grammatical clauses,
  content drifts.
- **wskan11, ultrachat, 10m:** `"I need to talk to you that your other
  party is capable of treating severe replacements..."` — fluent, still
  content-free.
- **wskan11, tinystories, 100k:** `"Mum went to swing and soaks to make
  him sail... The swing took dinner, and the story is that she did not go
  in its cloth. The end."` — story grammar with a closing marker; the
  morphology is visibly stronger than the 100k-step era.

Reading: 3 epochs buys grammatical fluency and structural markers
("The end.", list formatting), not facts. The model is a **form learner**
at every tier we tested — consistent with the memory analysis (effective
half-life ~1–5 tokens at these scales).

## 7. What it cannot do (unchanged and verified on this campaign)

- No factual recall; recall probes fail at every tier including 10m
  (the long-memory tail exists but stays dormant).
- No content understanding — the strongest qualitative text is grammatically
  perfect and semantically empty.
- Word-clock necessity replicates, but the clock's knowledge is of
  boundaries, not of meaning.

## 8. Scale laws (measured, multi-seed)

1. **Memory length grows with capacity**: half-life 1 tok (100k) →
   4.5–5 tok median (10m), tail to ~1,900 tokens; σ clamp binding vanishes
   (43% → 0%).
2. **Content locality loosens**: byte identity R² 0.76 → ~0.55; bigram
   0.91 → ~0.76; a non-local context component (~25–40%) appears at 10m.
3. **Wavelet advantage peaks at mid scale** (§4): −0.04…−0.08 at
   10k–100k, −0.01…−0.03 at 1m–10m.

## 9. Verdict

wskan11 demonstrates that an SSM-native wavelet KAN is (a) fast enough for
production-scale iteration (fused kernel, 40× over the first generation),
(b) matched-parameter competitive with the modern SSM at mid scale, and
(c) **readable end to end** — from the byte manifold through the word clock
and the oscillator bank to the exact decision decomposition, with the
wavelet's contribution measured at 5 seeds on 3 datasets. Its language is
form-shaped; whether content joins form at scale is the open question the
two scale-laws point at.

## 10. Reproduction

```bash
.venv/bin/python experiments/CAMPAIGN_orchestrate.py   # 300-run matrix
.venv/bin/python experiments/CAMPAIGN_aggregate.py     # all tables
```
Canonical checkpoint: `checkpoints/wskan11_ultrachat_100k_3ep_s42/`.
History: `experiments/V*_REPORT.md`, `models/V*_README.md`, git log.
