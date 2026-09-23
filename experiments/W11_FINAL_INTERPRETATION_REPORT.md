# Why Language Emerges in WSKAN
### The definitive mechanism monograph: from bytes to fluent form, every step read from the model's own functions

**Date:** 2026-09-13 · **Subject:** wskan11 (fused-kernel wavelet-SSM),
canonical checkpoint `wskan11_ultrachat_100k_3ep_s42` (3-epoch campaign,
5 seeds). Every number below was measured from checkpoints' own mathematical
objects; the central ones are intervention-proven. **This document contains
no benchmark claims** — leaderboards are in `CAMPAIGN_aggregate.py`'s
output if anyone insists; they are not the point.

---

## 0. The question and the answer

**Why does a pile of damped oscillators with a learned clock learn to write
fluent English?**

Because the architecture's two primitives — *content-warped time* and
*multi-scale damped oscillation* — align with the two deep facts of
language: **words are the unit** (not bytes), and **structure is
multi-scale** (letters → morphemes → words → phrases → register). The model
does not receive these facts. It *discovers* them: it invents a word
boundary clock (causally proven), arranges its oscillators into a
frequency ladder, and routes decisions through channels that specialize
into boundary detectors and position trackers — all measurable in the
checkpoint, none of it programmed.

The rest of this document is the evidence chain, in order, from the input
to the spoken output.

---

## 1. The emergence chain (each link measured, most links causal)

### 1.1 The byte manifold organizes itself by linguistic function

The learned byte embedding (256×d, tied to the head) is a **functional
manifold**: within-class cosine similarity 0.47 vs between-class 0.07; PC1
carries 52% of variance. Nearest neighbors are linguistically exact —
`e → [i, a, o]` (vowels), `t → [T, c, s]` (dental/sibilant + case),
`space → [",", ".", ":"]` (separators), `5 → [6, 3, 4]` (adjacent digits),
`\n → [;, :, space]` (structural). *The geometry of English's byte alphabet
is learned, not imposed.* (§E, `V7_final_interpret.py`)

### 1.2 The model invents tokenization — as a clock

The single most important object: Δ = softplus(W_dt · norm(x)), the
input-driven dilation. Measured on the canonical checkpoint (3-epoch, both
layers):

| byte class | Δ (L0) | Δ (L1) |
|---|---|---|
| lowercase | 0.100 | 0.102 |
| uppercase | 0.142 | 0.180 |
| digit | 0.115 | 0.140 |
| **space** | **0.258** | **0.205** |
| **newline** | **0.266** | **0.247** |
| **punct** | **0.276** | **0.266** |

Word boundaries advance the clock **2–2.6× faster** than letters, in every
layer. Since per-token state decay is e^{−ρσΔ}, a boundary erases
exponentially more of the current word's memory — **the effective unit of
distance is the word, not the byte**. This is tokenizer-free word
segmentation, learned through time warping alone.

The distribution confirms it is not a mean artifact: space's p10 ≈
lowercase's p90 (E2 quantiles). And the whitespace-removal test sharpens
the claim honestly: with spaces deleted, the input layer's boundary pulse
vanishes (ratio 1.01) while deeper layers keep ~16% elevation at implicit
boundaries — the clock is an **input-driven boundary detector**, primarily
byte-keyed at L0, partially space-independent deeper.

### 1.3 The causal proof (necessity AND sufficiency, replicated)

Teacher-forced CE on held-out text, Δ interventions:

| intervention | overall CE | at word-initial |
|---|---|---|
| baseline | 1.321 | 2.82 |
| remove boundary Δ (clamp to letter-mean) | 2.240 | **4.32 (+1.50)** |
| count-matched control (letters) | 1.993 | 2.89 (+0.07) |
| inject boundary-strength ticks mid-word | **3.49** | 3.67 |

Removing the boundary clock costs **21× more than the matched control**
at word transitions (necessity). Injecting false ticks mid-word makes the
model emit a "next word" distribution where a word-internal letter was
required — CE above even true word-initial difficulty (sufficiency: the
tick *causes* boundary behavior). The clamp cost decomposes 33% at
boundaries / 28% word-initial / 46% propagated — the clock is global in
effect, strongest exactly at word transitions. All of this replicates
across seeds and implementations (chunked-era and fused-kernel-era
checkpoints agree).

### 1.4 The oscillator bank organizes time into a ladder

Each channel carries N damped oscillators λ = −σ + iω. What training built:

- **Learned ρ ladders are monotone geometric** (e.g. L2: [0.96, 1.00, 0.89,
  0.85, 0.84, 0.83]) — a multi-resolution timescale ladder emerged
  unsupervised.
- **Mode frequencies concentrate at word scale** (median 0.165
  cycles/token ≈ 6-token period) — the band of morphemes/words, avoiding
  both the DC spike and byte-scale noise.
- **Q ≈ 0.6 (overdamped)**: the modes act as short-kernel *phase shapers*,
  not resonators; the constant-Q dictionary hypothesis was tested and
  honestly rejected (log-log R² ≤ 0.29).
- **Edge gains are genuinely high-rank** (effective rank 28–29/32) — the
  learned wavelet population resists compression, which is why the rank-32
  filter approximation costs +0.078 nats (V7 ablation).

### 1.5 The gates split structure from content — self-organized

The write/read gates come in two parts (V7's feature-factorization made
this readable by lookup):

- **Named tables carry structural bytes**: digit (1.42), newline, punct
  strong; letters near-zero. The interpretable channel self-organized to
  carry discrete formatting; newline's read routing migrates across depth
  (modes 1–2 → 1 → 3–4).
- **The residual path carries content**: byte identity R² = 1.00 (L0,
  trivially) → 0.76 (L1) → 0.50 (L2); position-in-word 0.12 → 0.23 → 0.16.
- **The "context" is local byte statistics** (exclusion battery): bigrams
  R² 0.91/0.84 (> identity at depth); word identity 0.12–0.14; sentence
  position ~0.007; document position ~0.001; dialogue-turn state ≤ 0.003
  (refuted). Nothing word-level or discourse-level was found at 100k.

### 1.6 Circuits: generalists, antipodal specialists, one hub

Channels cluster into many generalists plus 1–3 **antipodal specialist
singletons** per layer (boundary detector, word-initial detector,
uppercase-suffix detector — mirror-image pairs). The singletons are
individually load-bearing: zeroing the L0 boundary singleton alone costs
**+1.08 CE**. Depth retunes the code from byte-class (L0) to
position-in-word (L1/L2). Kernel families (~6 per layer) route through
channel clusters with a depth-migrating topology, converging onto a single
L2 output hub channel that the head reads. Family ablations: the four big
families are co-equal workhorses (+0.29–0.36 each); the small specialist
family is nearly redundant (+0.017) — specialization lives at channel
level, not edge level.

### 1.7 The decision: exact integration, opposing votes

Additive residual + linear head ⇒ every decision margin decomposes
**exactly**. Canonical case `frien→d`: margin 8.86 for 'd' over 'n';
per-mode contributions [+1.94, +3.91, +0.73, **+7.70**, +4.87, −0.35],
channel 0 dominant — morphological completion localizes to named modes.
Word endings resolve as **L1-over-L0 disagreement resolutions**: the
shallow layer resists ending the word (L0-wave negative at boundaries), the
deep layer ends it. The token-embedding prior votes *against* the produced
letter on average — context beats prior.

### 1.8 Production: the clock paces speech

During autoregressive generation the clock pulses at boundaries exactly as
in analysis (letters 0.085–0.128 vs spaces 0.194–0.257). Word-internal
letters are wavelet-path decisions against the negative token prior; word
endings are layer disagreements resolved deep. A punctuation event triggers
a synchronized spike across clock, wavelet paths, and modes — the
production loop is the analysis loop, seen from the inside.

---

## 2. What the wavelet specifically buys (mechanism evidence, 5 seeds)

The ablation (wskan11 vs ω≡0, identical everything) is the cleanest proof
that the oscillatory function space does real work:

- **72 of 75 cells negative** (5 seeds × 5 sizes × 3 datasets): −0.035 /
  −0.046 / −0.041 nats at the 100k tier (TinyStories/UltraChat/WikiText).
- **Scale shape (honest):** the advantage peaks at 10k–100k
  (−0.04…−0.08) and shrinks at 1m–10m (−0.01…−0.03). Oscillation is the
  differentiator at small/mid scale — exactly where memory is short and
  phase-shaped kernels do the lifting.

This is the KAN answer to "why this architecture": the edge function space
(damped oscillations) matches the structure of the signal's local
transitions.

## 3. What 3 epochs buy (qualitative, all tiers)

Grammatical fluency and structural markers ("The end.", list formatting,
dialogue register) — with content still absent. The morphology strengthens
with epochs while semantics never arrive — consistent with the memory
analysis (effective half-life ~1 token at 100k; recall probes fail
everywhere, Mamba-2 included).

## 4. Scale laws (measured, multi-seed)

1. **Memory length grows with capacity**: half-life 1 → 4.5–5 tokens
   median, tail to ~1,900; σ clamp binding 43% → 0%.
2. **Content locality loosens**: byte identity 0.76 → ~0.55; bigrams
   0.91 → ~0.76; a non-local context component (~25–40%) appears at 10M.
3. **Wavelet advantage peaks at mid scale** (§2).

## 5. The honest boundary

- **No facts, no recall, no content** — at every scale probed. The machine
  computes form; form is what there is at this scale.
- The clock knows boundaries, not meaning.
- The dormant long-memory tail (up to ~1,900 tokens at 10M) carries < 1%
  of readout weight — present, unused.
- No forward hand-simulation: we decompose any decision exactly after the
  fact and predict interventions qualitatively; we do not derive outputs
  without execution (the scale-imposed ceiling).

## 6. Why this is the KAN answer

A transformer hides its computation in QK^V·V products; an SSM hides it in
recurrent state. wskan11's computation is **named functions all the way
down**: ψ_io(t) per edge, ρ ladders per layer, Δ per position, named gate
tables, per-mode decision contributions. That is what made every measurement
in this document possible without a single probing model or activation
atlas — the functions are the analysis.

## 7. Reproduction

```bash
.venv/bin/python experiments/W7BC_canonical_probe.py        # §1.2–1.5 (canonical ckpt)
.venv/bin/python experiments/V7_causal_clock_full.py        # §1.3 causal battery
.venv/bin/python experiments/W7BC_generation_analysis.py    # §1.8 production trace
.venv/bin/python experiments/CAMPAIGN_aggregate.py          # §2 ablation + footnote tables
```
Canonical checkpoint: `checkpoints/wskan11_ultrachat_100k_3ep_s42/`.
Interpretation checkpoints (protected): `checkpoints/*_interp_*`.
Numbers: `figures/v7e_canonical_probe.json`, `v7e_10m_analysis.json`,
`v7e_strengthening.json`, campaign CSVs.
