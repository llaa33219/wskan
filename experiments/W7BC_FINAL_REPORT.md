# WSKAN7BC Final Report
### Wavelet-State-KAN with feature-factorized gates: architecture, benchmark, and the complete account of how it learned and speaks language

**Date:** 2026-09-11 · **Status:** definitive consolidated report of the
Wavelet-State-KAN project (V1–V7). All numbers are from the canonical fresh
interp tier (single code revision, single week, 3 seeds) unless labeled
otherwise. Sources linked throughout; reproduction index in §10.

---

## 1. Executive summary

**What was built.** WSKAN7BC is a 118k-parameter byte-level language model
whose KAN edge functions are **damped-oscillator wavelets realized by a
state-space model** — each edge ψ_io(t) = Σ_k Re[g_iok e^{λ̃_ik t}] is, by
construction, the impulse response of a stable SSM, and the same parameters
drive a fully-selective recurrent scan (Mamba-style) through a factorized
timescale (Δ_dyn × ρ ladder).

**What it achieves.** Best of the WSKAN series (UltraChat eval CE
1.1711 ± 0.0274); parity-to-slight-edge over a matched Mamba-2 baseline at
~100k; the only model in the top-3 of all 15 cells of a 5-family × 5-size ×
3-dataset × 3-seed benchmark.

**What was learned about it (the point of the project).** The model learned
language as a **word-clocked damped-oscillator memory**: an input-driven
clock ticks 2–2.3× faster at word boundaries (invented tokenization,
causally proven necessary *and* sufficient), writes are split between
structural bytes (lookup tables) and byte-identity content, convolved by
high-rank wavelet kernel families through circuits that converge on an
output hub, and every decision is exactly decomposable into named paths,
modes, and channels. The account replicates at 10M with 3/3 seeds and
yields two scale-laws: memory length grows with capacity (1 → 5 tokens,
tail to ~1,900), and the content channel loosens beyond pure byte
statistics.

**Honesty footer.** It is a register learner: fluent form, no facts, no
recall, dormant long-memory at 100k. This report documents mechanisms of
*form*, not evidence of content understanding.

---

## 2. Architecture

### 2.1 Lineage (why these pieces exist)

| version | delta | verdict |
|---|---|---|
| V1 | LTI wavelet-SSM edges (ψ = SSM impulse response), FFT path | too weak (1.53 CE TS) |
| V2 | + static input/output gates | closes 76% of the Mamba gap |
| V3 | + full selectivity (Δ,B,C input-dependent), chunked SSD | parity with Mamba-2 |
| V4 | freed σ, per-mode Δ, ZOH write | performance-neutral |
| V5 | short conv + geometric ladder | rejected (regression) |
| V6 | **factorized timescales** Δ_dyn × ρ | V4 performance, V3 analyzability, tightest seeds |
| **V7 (wskan7bc)** | V6 + **feature-factorized B/C** | adopted: interpretable-by-construction at zero loss |

### 2.2 The edge: a wavelet that is a state-space system

Each of the d² edges per layer owns an explicit function:

$$\psi_{io}(t) = \sum_{k=1}^{N} \mathrm{Re}\!\left[g_{iok}\, e^{\tilde\lambda_{ik} t}\right],
\qquad \tilde\lambda_{ik} = \rho_k\,(-\sigma_{ik} + i\omega_{ik}),$$

- λ̃: damped-oscillator eigenvalues (S4D-Lin init, then learned),
- ρ_k: static per-layer scale ladder (learned; monotone ladders emerge),
- g_iok = a_iok + i·b_iok: complex edge gains.

This ψ is *exactly* the impulse response of a diagonal SSM
(h′ = λ̃h + u, y = Re[g·h]), so the same parameters admit two modes:
closed-form **static** evaluation (WavKAN-style edges) and a **recurrent**
selective scan.

### 2.3 The recurrent pass (how text is processed)

Per channel i, mode k, token n:

$$\Delta^{\text{dyn}}_{n,i} = \min(\mathrm{softplus}(W_\Delta \cdot \mathrm{norm}(x)_n),\ 1),
\qquad h_{n,ik} = e^{\tilde\lambda_{ik}\Delta^{\text{dyn}}_{n,i}}\, h_{n-1,ik}
+ B_{n,ik}\,\Delta^{\text{dyn}}_{n,i}\, x_{n,i},$$
$$y_{n,o} = \Big(\sum_{i,k} C_{n,ik}\,\mathrm{Re}[g_{iok} h_{n,ik}]\Big)
\odot \mathrm{SiLU}(W_z x_n) + (x W_{\text{base}})_o.$$

Closed form, with warped time T_{n,i} = Σ_j Δ^dyn_{j,i}:
h_{n,ik} = Σ_{m≤n} e^{λ̃_ik(T_n−T_m)} B_{m,ik} Δ_m x_{m,i} — **each edge
performs a wavelet transform of the input along a content-warped time
axis.**

**Write/read gates (the V7 signature):**
B_{n,ik} = Σ_f α_f(byte_n)·M^B_{f,ik} + W^B_lr x_n,
C likewise, where α_f ∈ {is_space, is_newline, is_punct, is_upper,
is_lower, is_digit, is_vowel, 1} are **fixed named byte features** — the
named part is interpretable by lookup (M^B[is_space, :, k]), the rank-32
residual carries context. The tradeoff was measured: this costs nothing at
rank 32 (+16% params) and +0.035 nats at strict param parity (V7 report).

**Guarantees (by construction).** σ = e^{logσ} > 0 ⇒ |e^{λ̃Δ}| < 1
(unconditional stability); causal decay exponents have Re ≤ 0 (no overflow
possible — two NaN incidents in earlier versions were root-caused and
fixed, see incident history); chunked SSD scan matches the sequential
reference to ~1e-7.

### 2.4 Configurations

| tier | d_model | layers | N | params |
|---|---|---|---|---|
| interp (canonical) | 32 | 3 | 6 | **118,386** |
| 1k | 4 | 1 | 6 | 2,042 |
| 10k | 12 | 1 | 6 | 9,186 |
| 100k | 40 | 2 | 6 | 113,612 |
| 1m | 80 | 6 | 6 | 985,956 |
| 10m | 512 | 2 | 6 | 10,153,996 |

Training (canonical): byte-level LM (vocab 256, tied head), pre-norm
residuals + final LayerNorm, block 256 (TinyStories) / 512 (UltraChat),
batch 32, AdamW lr 3e-3, cosine to 0, grad clip 1.0, 100k steps, seeds
{42, 123, 2024}.

---

## 3. Performance

### 3.1 Canonical interp-tier numbers (fresh, single code/protocol)

**UltraChat (block 512, 100k steps, 3 seeds, best eval CE):**

| model | CE | |
|---|---|---|
| **wskan7bc** | **1.1711 ± 0.0274** | best of the WSKAN series |
| wskan4 | 1.1837 ± 0.0254 | |
| wskan3 | 1.1882 ± 0.0125 | |
| mamba2 | 1.1886 ± 0.0302 | |
| wskan6 | 1.1910 ± 0.0108 | |
| wskan7bc16 | 1.2262 ± 0.0122 | param-parity ablation |
| wskan3real (ω≡0) | 1.2443 ± 0.0170 | wavelet ablation control |
| wskan7 (full) | 1.2767 ± 0.0182 | rejected |

**TinyStories (block 256):** wskan (V1) 1.5001 → wskan2 0.9856 → wskan3
0.8101 ± 0.0132 (tied with mamba2 0.8194 ± 0.0107); wskan3real
0.8526 ± 0.0180.

### 3.2 The headline ablation (3/3 seeds, both datasets)

**The wavelet structure contributes measurably:** V3 − (V3 with ω frozen at
0, identical everything) = **−0.0425 nats** on TinyStories and **−0.0560
nats** on UltraChat, paired per seed, 3/3 — and larger on the harder
dataset. With real decay modes the same selective machinery *loses to
Mamba-2*; the wavelet (oscillatory) modes are exactly what tips the
comparison. This is the project's central empirical claim and it has
survived every regeneration of the data.

### 3.3 Cross-family benchmark (225 runs: 5 families × 5 sizes × 3 datasets × 3 seeds)

wskan7bc vs Mamba-2, tiny transformer, gated dilated conv, LSTM — matched
params per tier (1k/10k/100k/1m/10m), budget-matched steps
(`W7BC_CROSS_FAMILY_REPORT.md`):

- **Mamba-2 sweeps 1k–100k (9/9) and 10m (3/3).**
- **Ranking inverts at 1m: LSTM & wskan7bc lead** (wskan wins WikiText-1m;
  #2 on the other two); Mamba-2 falls to 4th on all three datasets.
- **wskan7bc is the only model in the top-3 of all 15 cells** (12× #2,
  3× #3) — never the best, never off the podium: the most robust profile
  in the field.
- Family profiles: attention weakest small / competitive large; conv
  uniformly mid-weak; LSTM the surprise (best at 1m); Mamba-2
  barbell-shaped.

---

## 4. How it learned language: the mechanism (all intervention-verified)

Canonical checkpoint: `wskan7bc_ultrachat_interp_s42` (d32L3). Everything
below was measured from the checkpoint's own objects; central claims are
intervention-proven.

### 4.1 The word clock (the centerpiece)

Δ per byte class (fresh): **space/newline/punct 0.17–0.26 vs lowercase
0.08–0.12** across all layers — boundaries advance the clock **2–2.3×
faster than letters**; per-position decay e^{−ρσΔ} makes the effective
distance unit the word, not the byte. The distribution confirms it:
space's p10 ≈ letter's p90.

**Causal proof (necessity + sufficiency):**
- Remove boundary Δ: overall CE 1.321 → 2.240; at word-initial predictions
  2.82 → 4.32. Matched-size letter control: +0.67 / +0.07. **Necessity
  ratio ≈ 21× at word transitions.**
- Inject boundary-strength ticks mid-word: CE explodes to 3.49 — false
  ticks *create* word-transition behavior. **Sufficiency.**
- Damage decomposition: 33% at boundaries / 28% word-initial / 46%
  propagated elsewhere — global mechanism, locally strongest at the tick.
- Whitespace-removal test: the L0 pulse vanishes without spaces (ratio
  1.01) while L1/L2 keep ~16% implicit-boundary elevation — the input-layer
  clock is byte-keyed; deeper layers partially reconstruct boundaries from
  word-internal statistics. (Precise statement: input-driven word-boundary
  detector, not a space-free word model.)
- Driven by mainstream channels (no dedicated clock neuron; top-8 drivers
  ≈ 35% of drive).

### 4.2 The memory: damped oscillators on a geometric ladder

Learned monotone ρ ladders per layer (e.g. L2 [0.96, 1.00, 0.89, 0.85,
0.84, 0.83]). Mode frequencies concentrate at word scale (~0.16
cycles/token ≈ 6-token period); Q ≈ 0.6 (overdamped — phase structure, not
ringing). Constant-Q self-organization **rejected** (R² ≤ 0.29). g is
genuinely high-rank (effective rank 28–29/32) — the reason the rank-32
filter compression costs +0.078 nats.

### 4.3 The write/read gates: structural vs content channels

- **Named tables carry structural bytes** (digit 1.42, newline, punct
  strong; letters ~0.2–0.4) — the interpretable channel self-organized to
  carry discrete formatting; newline's read routing migrates with depth.
- **Residual path carries content**: byte identity R² = 1.00 (L0,
  trivial) → 0.76 (L1) → 0.50 (L2); position-in-word 0.12 → 0.23 → 0.16.
- **The "context" is local byte statistics** (exclusion battery): common
  bigrams R² 0.91/0.84 (> identity at depth); word identity 0.12–0.14;
  sentence/doc position ~0.001–0.007; dialogue-turn state ≤ 0.003
  (refuted). Nothing word-level or discourse-level was found.

### 4.4 Circuits and hubs

Channels: many generalists + 1–3 antipodal specialist singletons per layer
(boundary detector, word-initial detector, uppercase-suffix detector); the
singletons are individually load-bearing (zeroing one costs +1.08 CE).
Depth shifts the code from byte-class to position-in-word. Kernel families:
~6 per layer (steps, ramps, spikes, V-transients, notches, rare long-tails);
the four big families are co-equal workhorses (+0.29–0.36 CE each on
ablation); the small specialist family is nearly redundant. Routing
migrates across clusters with depth and converges onto a **single L2 output
hub channel** — the universal write target the head reads.

### 4.5 The decision and production

The final state is an additive sum; the tied head is linear — every
decision decomposes **exactly** into token-emb + per-layer (base + wavelet)
parts. Canonical case `frien→d`: margin 8.86, per-mode contributions
[+1.94, +3.91, +0.73, **+7.70** (k3), +4.87, −0.35], channel 0 dominant —
morphological completion localizes to named modes. Word endings are
**L1-over-L0 disagreement resolutions** (shallow resists, deep decides).
Production traces show the clock pulsing at boundaries during emission
(0.085–0.128 letters vs 0.194–0.257 spaces), and the token-embedding prior
voting *against* produced letters on average — context overrules prior.

---

## 5. Scale analysis: 100k vs 10M (3-seed each)

| dimension | 100k | 10M (3/3 seeds) |
|---|---|---|
| word clock (space/lower Δ) | 2.0–2.3× | 1.7–2.1× (also TS/WT) |
| causal necessity at word-initial | +1.50 vs +0.07 | +1.1–1.55 vs +0.16 |
| σ mean | ~6.0 (33–43% at ceiling) | 0.66–0.83 (**0% at ceiling**) |
| half-life median | ~1.1 tok | 4.5–5.0 tok |
| half-life max | ~100 tok | **~1,900 tok** |
| byte identity (write R², L1) | 0.76 | 0.51–0.58 |
| bigrams (write R²) | 0.91 | 0.71–0.79 |

**Scale-law 1: memory length grows with capacity** — 8× slower decay, no
clamp binding, tail from ~100 to ~1,900 tokens. The near-delta 100k regime
was capacity-imposed.
**Scale-law 2: the content channel loosens** — local byte statistics lose
share; a ~25–40% non-local contextual component emerges at 10M (neither
byte- nor word-level; the next probe target).
Everything else (word clock causality, whitespace test, byte-keyed input
clock, canonical attribution `frien→d` 3/3) replicates.

---

## 6. Comparison with other models (interpretability axis)

| | wskan7bc | mamba2 | transformer | conv | lstm |
|---|---|---|---|---|---|
| named function objects | **edge wavelets, ρ ladders, gate tables** | A_log/dt/B/C matrices | attention maps | conv kernels | gate weights |
| fixed per-unit function | **yes** | no (input-dependent) | maps not functions | yes | no |
| exact decomposition | **path & mode level** | no | partial | partial | no |
| causal batteries demonstrated | **clock/tables/clusters/hub/families** | dt readout only | none here | none here | none here |
| main opacity | residual-B context | in_proj entanglement | MLP blocks | dense mixing | gate entanglement |

The performance-interpretability frontier at this budget is not close:
wskan7bc is the only entry that is both top-3 everywhere and deeply
readable.

---

## 7. What it cannot do (honest boundary)

1. **No facts, no recall** — effective memory ~1 token at 100k; recall
   probes fail identically to matched Mamba-2.
2. **Dormant long-memory at 100k** — slow modes exist (< 1% readout
   weight); they activate with scale (10M: tail ~1,900 tokens) but recall
   probes still fail at 10M.
3. **The "context" is local byte statistics** — nothing word-level found
   at 100k; a non-local component appears at 10M but is unidentified.
4. **No forward hand-simulation** — exact post-hoc decomposition and
   qualitative intervention predictions, not derivation without execution
   (scale-imposed ceiling).
5. **Form, not content** — every mechanism documented is about linguistic
   form; no evidence of semantic understanding at this scale.

---

## 8. Incident history (reproducibility honesty)

NaN underflow in V1's kernel path (σ/s clamp fix); V3's causal-decay
pre-mask overflow (diff clamp fix); residual-scale blowup without pre-norm
(discovered via RNG-replay + forward hooks; pre-norm adopted); Mamba-2
adapter double label-shift (caught by qualitative samples);
torch.compile complex-op fallback (real-form bmm rewrite); checkpoint
overwrite by the benchmark matrix (interp tier created, artifacts
regenerated); /tmp inductor-cache exhaustion (twice; managed); mamba-1m
single-layer config collapse (corrected to 2-layer). All documented in the
era reports and git history.

---

## 9. Verdict

WSKAN7BC demonstrates that a KAN whose edge functions are SSM-realizable
wavelets can (a) match or slightly exceed a matched Mamba-2 at the ~100k
scale while being the most robust model across 15 benchmark cells, and (b)
be **read like a book**: every parameter is a named mathematical object,
the central mechanism (word clock) is causally proven, and the full
pipeline from byte manifold to output hub to exact decision decomposition
is measurable. Its language is form-shaped: a word-segmented clock driving
a geometric-ladder damped-oscillator memory. Whether content joins form at
larger scale is the open question the 10M scale-laws point at.

---

## 10. Reproduction index

| artifact | path |
|---|---|
| architecture | `models/V6_WSKAN.py`, `models/V7_WSKAN.py` (+ `V*_README.md`) |
| canonical checkpoints | `checkpoints/wskan7bc_{ultrachat,tinystories}_interp_s{42,123,2024}/` |
| canonical probe | `experiments/W7BC_canonical_probe.py` |
| strengthening probes | `experiments/W7BC_strengthening_experiments.py`, `W7BC_10m_analysis.py` |
| main interpretation | `experiments/W7BC_HOW_IT_SPEAKS.md` |
| 10M replication | `experiments/W7BC_10M_INTERPRETABILITY_REPORT.md` |
| canonical numbers | `experiments/W7BC_FINAL_CLEAN_REPORT.md` |
| cross-family benchmark | `experiments/W7BC_CROSS_FAMILY_REPORT.md` (+ `BENCH_*.py`) |
| generation/production | `experiments/W7BC_GENERATION_ACCOUNT.md` |
| full history | `experiments/V1_*` … `V7*_REPORT.md`, git log |
