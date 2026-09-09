# How Wavelet-State-KAN Learns Language: The Complete Mathematical Account

**Version 1.0 (final synthesis) · 2026-09-08**
**Subject:** V6 Wavelet-State-KAN, byte-level LM, 101,874 parameters,
UltraChat-200k, 100k training steps, seeds {42, 123, 2024}.
**Sources:** all numbers in this document are drawn from the project's
experiments and reports (cited inline); every claim about the model's
internals is read directly from its learned edge functions — the property
that makes a KAN analyzable at all.

---

## 0. The answer in one paragraph

V6 learned English **by inventing a word-segmented internal clock**: an
input-driven dilation Δ whose fast ticks at word boundaries (2–3× faster
than at letters) drive an overdamped, multi-scale wavelet filter bank whose
receptive fields grow with depth from byte-scale spikes to 40+-token
envelopes, mixed through hub-sparse edges. The tick mechanism is **causally
both necessary and sufficient** for predicting what follows a word — proven
by removal and injection across three independent training runs. What the
model did *not* learn is equally definitive: no facts, no recall — it is a
*register learner*; long-range memory machinery exists in the checkpoint
(tails up to ~100 tokens) but stays dormant at this scale, exactly as it
does in matched-size Mamba-2.

---

## 1. The mathematical object

### 1.1 Edge functions (the KAN substrate)

Every edge (i, o) of every layer carries an explicit, plottable function

$$\psi_{io}(t) \;=\; \sum_{k=1}^{N}\, \mathrm{Re}\!\left[g_{iok}\; e^{\tilde\lambda_{ik} t}\right],
\qquad \tilde\lambda_{ik} = \rho_k\,\lambda_{ik} = -\rho_k\sigma_{ik} + i\,\rho_k\omega_{ik},$$

a sum of damped oscillators (a wavelet family) with learned complex gains
$g_{iok} = a_{iok} + i\,b_{iok}$. There are no weight matrices on edges:
1,024 explicit functions per layer.

### 1.2 The selective scan (how the functions process text)

Per input channel i, mode k, token n:

$$h_{n,ik} = e^{\tilde\lambda_{ik}\,\Delta^{\text{dyn}}_{n,i}}\, h_{n-1,ik}
\;+\; B_{n,ik}\,\Delta^{\text{dyn}}_{n,i}\,x_{n,i},$$
$$y_{n,o} = \sum_{i,k} C_{n,ik}\,\mathrm{Re}[g_{iok} h_{n,ik}]
\;\odot\; \mathrm{SiLU}(W_z x_n) \;+\; x_n W_{\text{base}}.$$

with the **factorized dilation** $\Delta_{n,i,k} = \Delta^{\text{dyn}}_{n,i}\cdot\rho_k$:
the content warp $\Delta^{\text{dyn}}$ is per channel; the multi-resolution
ladder $\rho_k$ is static per layer (learned; init 1). Closed form, with
warped time $T_{n,i} = \sum_j \Delta^{\text{dyn}}_{j,i}$:

$$h_{n,ik} = \sum_{m\le n} e^{\tilde\lambda_{ik}(T_{n,i}-T_{m,i})}\, B_{m,ik}\,\Delta^{\text{dyn}}_{m,i}\, x_{m,i}$$

— **each edge performs a wavelet transform of the input along a
content-warped time axis.** The write $u = B\Delta x$ is the exact
first-order ZOH form (ρ treated as eigenvalue scaling), so the
discretization is consistent with the static-mode wavelet identity.

### 1.3 Numerical guarantees

Causal decay exponents have Re ≤ 0 by construction (no overflow possible);
σ ∈ [e⁻⁶, e⁸], ρ ∈ [e⁻³, e³], Δ ≤ 1; chunked-SSD scan matches the
sequential reference to 3.6e-7; unconditional stability |e^{λ̃Δ}| < 1.

---

## 2. What the architecture is worth (performance context)

### 2.1 The V1→V6 ladder isolates *which* math matters

TinyStories, 100k steps, matched params (best eval CE, 3 seeds):

| step | mechanism added | best eval |
|---|---|---|
| V1 — LTI wavelet-SSM edges only | — | 1.5273 |
| V2 — + static input/output gates | selectivity-lite | 0.9894 |
| V3 — + full selectivity (Δ,B,C input-dependent) | **the wavelet filter bank was never the bottleneck; selectivity was** | 0.8003 |

Contribution decomposition at V3: selectivity ≈ **0.69 nats**; oscillatory
(wavelet) structure ≈ **0.03–0.05 nats** (ablation ω≡0: TinyStories
−0.0301±0.0034, UltraChat −0.0506±0.0034; 3/3 seeds each).

### 2.2 V6 vs Mamba-2 (UltraChat, 100k steps, 3 seeds)

| model | params | best eval CE ± std |
|---|---|---|
| **wskan6** | 101,874 | **1.1608 ± 0.0040** (tightest of all) |
| mamba-2 (HF) | 97,592 | 1.1792 ± 0.0361 |
| wskan3real (ω≡0 control) | 101,856 | 1.2134 ± 0.0118 |

Mean-edge over Mamba-2 with 9× tighter seed variance; the ω≡0 control is
decisively worse — **the wavelet function space is the differentiator**, not
the selective machinery per se (wskan3real loses to Mamba-2 outright).

### 2.3 Honest negative results (what did NOT help)

- V4 (freed σ, per-mode Δ, ZOH): performance-neutral (−0.0021); the σ
  ceiling was *binding but harmless* — capacity past it buys nothing at 100k.
- V5 (short conv + geometric ω init): regression (+0.0232, 3/3). WSKAN's
  near-delta modes were already sufficient local extractors.
- Constant-Q self-organization: **rejected** (log-log slope 0.25–0.30,
  R² ≤ 0.29; corrects an earlier qualitative impression from V3).
- Mode frequencies do *not* lock onto text spectral peaks (the byte
  spectrum is broadband; only scale-consistency can be claimed).

---

## 3. The dissection: how language is encoded in the functions

### 3.1 The word clock (Finding 1 — strongest)

Mean learned Δ per byte class (2048-byte probe, all layers):

| class | Δ (L0/L1/L2) |
|---|---|
| newline | 0.396 / 0.348 / **0.481** |
| space | 0.380 / 0.284 / **0.469** |
| punctuation | 0.345 / 0.347 / 0.425 |
| uppercase (word-initial) | 0.231 / 0.289 / 0.436 |
| lowercase | 0.138 / 0.134 / 0.214 |

Boundaries advance the internal clock **2–3× faster** than letters, in every
layer; state decay per token is $e^{-\rho\sigma\Delta}$, so a boundary erases
exponentially more memory. **The effective unit of distance is the word, not
the byte — the model invented word segmentation through time warping, with
no tokenizer.** Complement: write strength |B| peaks at letters (1.93 vs
1.66–1.72) — content is *stored* at letters, boundaries *advance the clock*.

### 3.2 Causal verification (both directions, 3/3 seeds)

| intervention | overall CE | Δ at word-initial | CE after intervened byte |
|---|---|---|---|
| baseline | 1.22 | — (wi CE = 2.75) | — |
| remove Δ at boundaries (**necessity**) | 2.13–2.36 | **+1.6…+2.7** | — |
| remove Δ at count-matched letters (control) | 1.80 | +0.07…+0.09 | — |
| inject boundary-Δ mid-word (**sufficiency**) | **3.57–3.88** | +0.5…+0.6 | **3.70–3.95** |

Removing boundary Δ cripples exactly word-transition prediction (≥18× the
matched control). Injecting false ticks mid-word *creates* word-transition
behavior: the model predicts a "next word" distribution where a
word-internal letter was required (CE 3.7–4.0, above even true word-initial
difficulty 2.75). The clock is not a correlate — it is the mechanism.

### 3.3 Hierarchical temporal receptive fields (read from kernels)

Top-|g| edges per layer, effective kernels on real text:

| layer | shape | span | role |
|---|---|---|---|
| 0 | delta-like spikes (ringing ≤ ~5 tokens) | bytes/n-grams | local feature extraction |
| 1 | spikes + broad bumps at lag 10–30 | words/phrases | transition scale |
| 2 | smooth single-lobe envelopes, tails > 40 tokens | phrase/clause | integration |

Fine-to-coarse with depth, directly visible in the learned functions.

### 3.4 The frequency bank: word scale, overdamped

Learned mode frequencies: median **0.165 cycles/token (≈ 6-token period)**,
p90 0.31, max 0.53 — concentrated at word/morpheme periodicity scale,
avoiding both the DC spike and byte-scale aliasing band. Q = ω̃/2σ̃ median
≈ 0.6 (overdamped): the measured wavelet advantage (−0.03…−0.05 nats) comes
from **phase structure in short kernels**, not sustained ringing.

### 3.5 Hub-sparse mixing

Edge-strength Gini grows with depth (0.297 → 0.344 → 0.464; max/median |g|
4.6 → 7.6 → 28.5): the last layer concentrates its function into a few hub
edges — measurable pruning pressure.

---

## 4. What the model did NOT learn (equally important)

1. **No facts, no recall.** "Capital of France" → fluent generic text; the
   memorize-then-recall probe fails after interleaved distractors, for V6
   *and* for matched Mamba-2 (`V6_QUALITATIVE_SAMPLES.txt`).
2. **Register, not semantics.** Generations are dialogue-formatted, register
   correct ("Sure, here are some examples of…"), content-free.
3. **Dormant long-memory tail.** V6's slowest modes reach ~99-token
   half-life but carry < 1% of readout weight. Matched Mamba-2 has the same
   dormant tail (max 128–180 tokens) — this is a scale property, not an
   architecture property.
4. **Scale changes the physics.** At 10M params (probe run): σ drops 6×
   (0.9–1.6 vs ~6), half-life grows to 2–5 tokens, the σ ceiling stops
   binding entirely. Memory length scales with capacity; the 100k regime is
   deliberately local.

---

## 5. The complete picture

$$\boxed{\;\text{language}(x) \;=\; \underbrace{\text{word-clocked}}_{\textstyle \Delta^{\text{dyn}}:\; \text{boundaries tick 2–3×}} \bigg(\; \underbrace{\text{overdamped wavelet bank}}_{\textstyle \tilde\lambda=\rho\lambda:\; \text{word-scale, hierarchical}} \;+\; \underbrace{\text{hub-sparse mixing}}_{\textstyle |g|:\; \text{Gini} \uparrow \text{ depth}} \;\bigg)\;}$$

with the clock causally verified (necessity + sufficiency), the frequency
placement scale-consistent, and the semantics absent — a complete, honest
account of what 100k parameters of Wavelet-State-KAN do with language.

**Falsifiable predictions for larger scale** (from the 10M probe + dormant
tails): (i) the readout weight on slow modes will grow with parameter count;
(ii) the word-clock ratio Δ(boundary)/Δ(letter) persists (it is
linguistic, not capacity-driven); (iii) explicit recall remains impossible
until effective half-life × readout weight covers the probe distance.

---

## 6. Reproducibility index

| artifact | path |
|---|---|
| architecture (adopted) | `models/V6_WSKAN.py`, `models/V6_README.md` |
| trainer | `experiments/V1_train_tinystories_lm.py --model wskan6` |
| checkpoints (3 seeds) | `checkpoints/wskan6_ultrachat_100k_s{42,123,2024}/` |
| dissection script + figures | `experiments/V6_deep_analysis.py`, `experiments/figures/` |
| causal batteries | `experiments/V6_causal_clock_test.py`, `V6_causal_clock_full.py` |
| source reports | `V1_TINYSTORIES*`, `V1_MAMBA_COMPARISON*`, `V2_COMPARISON_100K*`, `V3_*`, `V4_*`, `V5_*`, `V6_*` (this directory) |

Known limitations: single-dataset training per model (rule-5's ≥3 datasets
not yet met project-wide); interventions at inference only; per-channel ρ
clustering unexplored; the sufficiency injection used boundary-mean Δ, not
transplanted individual values.
