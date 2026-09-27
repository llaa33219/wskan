# Anatomy of Language Form in WSKAN
### How the model computes the local statistics of fluent text - a mechanistic anatomy, every step read from the model's own functions

**Date:** 2026-09-13 · **Subject:** wskan11 (fused-kernel wavelet-SSM),
canonical checkpoint `wskan11_ultrachat_100k_3ep_s42` (3-epoch campaign,
5 seeds). **Scope (corrected after external review):** this is an anatomy of
*how the model computes form* - boundary timing, oscillator structure,
gates, decisions. It is not a theory of language emergence: what emerged is
the computation of local statistics of fluent text, and that is all the
evidence supports. Performance context is included compactly with units
(Appendix C) - not hidden, not central.

**Position in the field (round-3 meta-question, stated explicitly):** this
is *interpretable-by-design* work - we built a model whose computation is
named functions, then read what it learned. Its value: ground truth for
what computation is learnable, a benchmark substrate for post-hoc methods,
and clean hypothesis tests. Its limit (honest): it does not directly help
interpret production models, and parts of the readability are architectural
artifacts by construction - the document flags those wherever they matter.

---

## 0. The question and the answer

**What did the model actually learn, and how does it compute it?**

Measured answer: it learned to compute the **local statistics of fluent
text** - and the machinery it built for that is more structured than the
task requires. It routes existing boundary bytes through a learned clock
(2-2.6x faster ticks at boundaries), arranges damped oscillators into a
self-organized frequency ladder, splits write gates into structural and
content channels, and makes decisions as exact, attributable integrations.
None of this is programmed; all of it is measurable in the checkpoint.

What it did NOT learn (same evidence): facts, recall, content. The output
is grammatical fluency without meaning. That boundary is part of the
findings, not a footnote.

---

## A. The mathematics of the mechanism

This section is the rigorous form; each equation carries its measured value
in this model. Derivations are exact where marked exact; measured
quantities are labeled.

### A.1 The edge function space (what a KAN edge is here)

Every edge (i, o) owns an explicit scalar function built from N damped
oscillators per input channel:

$$\psi_{io}(t) = \sum_{k=1}^{N} \mathrm{Re}\!\left[g_{iok}\, e^{\tilde\lambda_{ik} t}\right]
= \sum_k e^{-\tilde\sigma_{ik} t}\left(a_{iok}\cos\tilde\omega_{ik} t + b_{iok}\sin\tilde\omega_{ik} t\right),$$

with $\tilde\lambda_{ik} = \rho_k\lambda_{ik} = \rho_k(-\sigma_{ik} + i\omega_{ik})$
(the static ladder ρ_k reparameterizes both decay and frequency). This is
**exactly** the family of impulse responses of finite-dimensional stable
LTI systems — a theorem in both directions: h′ = λ̃h + u, y = Re[g·h] has
impulse response ψ, and every finite sum of damped exponentials is such a
response. So "the edge function" and "the SSM" are the same object. The
admissibility (zero-mean) correction is closed-form:

$$\tilde\psi(t) = \psi(t) - \Big(\textstyle\sum_k a_k \frac{2\sigma_k}{\sigma_k^2+\omega_k^2}\Big)\,\frac{\bar\sigma}{2}e^{-\bar\sigma|t|},
\qquad \bar\sigma = \tfrac1N\textstyle\sum_k\sigma_k,$$

exact for the pure envelope and O(ε) under the smoothed modulus
√(t²+ε) used in code (documented in `models/V1_README.md`).

### A.2 The selective scan and its exact closed form (the content-warped transform)

Per channel i, mode k, the recurrence with input-driven step Δ^dyn_{n,i}
and ZOH-consistent write:

$$h_{n,ik} = e^{\tilde\lambda_{ik}\Delta^{\text{dyn}}_{n,i}}\, h_{n-1,ik} + B_{n,ik}\,\Delta^{\text{dyn}}_{n,i}\, x_{n,i},$$

has the exact solution, with warped time $T_{n,i} = \sum_{j\le n}\Delta^{\text{dyn}}_{j,i}$:

$$h_{n,ik} = \sum_{m\le n} e^{\tilde\lambda_{ik}(T_{n,i}-T_{m,i})}\, B_{m,ik}\,\Delta^{\text{dyn}}_{m,i}\, x_{m,i}.$$

**This is the key equation of the model**: the edge applies its wavelet to
the input with the time axis warped by content. Token distance is not
position distance; it is accumulated dilation. Stability is by
construction: σ > 0 and Δ ≥ 0 ⇒ |e^{λ̃Δ}| = e^{−ρσΔ} < 1, so the state can
never blow up regardless of learned parameters (verified adversarially at
σ = e⁸, ρ = e³).

*(Units matter - a round-2 review tripped here. And the round-3 review's
"100k-centric" point is answered in Appendix B: the same anatomy measured
at all five tiers - the clock is universal, its dimensions scale.)* All dynamical quantities
live in **warped-time units** (ΣΔ), not tokens. Measured on the canonical
checkpoint (marginal medians over modes): L0 σ̃ ≈ 5.7, ω̃ ≈ 9.9; L1 σ̃ ≈ 5.6,
ω̃ ≈ 11.4 rad per warped unit; Δ̄ ≈ 0.11–0.15.

*Round-4 correction:* the pointwise identities (Q = ω/2σ, half-life =
ln2/(σΔ̄), cycles/token = ωΔ̄/2π) hold **per mode, exactly, by construction**,
but **not for medians** — the median of a ratio of correlated quantities is
not the ratio of the medians, and σ̃, ω̃, Δ̄ are correlated across modes.
An earlier version of this paragraph marked the medians as mutually coherent
with ✓; that was wrong, and the marks are removed. What is exact: any single
mode's tuple. Representative example (L1, channel 0, mode 3): σ̃ = 6.107,
ω̃ = 13.169, Δ̄ = 0.153 ⇒ Q = 1.078, half-life = 0.740 tokens,
cycles/token = 0.321 — identity exact. The boundary-tick decay ratio is
e^{σ̃Δ̄(r−1)} ≈ 2.5× per boundary at median values (order-of-magnitude,
not an identity).

### A.3 What the word clock is, mathematically

The measured Δ field (§1.2) has boundary:letter ratio r ≈ 2.0–2.6. In
warped time, a boundary multiplies effective distance by ~(1+r)/2, and the
kernel $e^{\tilde\lambda(T_j - T_i)}$ decays accordingly - boundaries
exponentially gate cross-word influence. **Precise claim (review-corrected):**
the clock is an input-driven boundary-responsive distance modulation,
byte-keyed at the input layer, partially generalized at depth. We did NOT
show that Δ aligns with word boundaries independently of spaces, nor that
no bypass paths exist; what the battery shows is that Δ at boundary
positions is position-specifically load-bearing for word-transition
prediction (21× the matched control). That is the exact scope of the
causal claim.

### A.4 The gates' algebra (structure vs content)

$$B_{n,ik} = \underbrace{\sum_f \alpha_f(\text{byte}_n)\, M^B_{f,ik}}_{\text{structural (lookup-readable)}} + \underbrace{W^B_{\text{lr}}\, x_n}_{\text{content (residual)}},$$

with fixed named features α ∈ {space, newline, punct, upper, lower, digit,
vowel, 1}. Measured split (§1.5): the named rows carry the discrete
formatting bytes (digit 1.42, newline, punct); the residual carries byte
identity (R² 1.00→0.76→0.50 with depth) and local bigram statistics
(R² 0.91/0.84). The same form holds for C (read).

### A.5 The exact decision decomposition

The residual stream is additive and the head is linear (tied embedding),
so for the final position the logit of byte v is

$$\text{logit}_v = W^{(v)}_{\text{head}} \cdot \Big[\, \text{emb}(x_{-1}) + \sum_{l=1}^{L} \big(\underbrace{x_{\text{in},l} W_{\text{base},l}}_{\text{skip}} + \underbrace{\text{wave}_l}_{\text{scan output}}\big)\Big],$$

and every summand is readable. Since the scan output itself is
$y_{n,o} = \sum_{i,k} C_{n,ik}\,\mathrm{Re}[g_{iok} h_{n,ik}]$, the decision
decomposes **exactly** to named (channel, mode) pairs. *Round-4 correction:*
the earlier version of this table quoted pre-norm units that did not sum to
the margin (and, worse, were computed from a history-free length-1 call that
dropped the scan state). Recomputed on the canonical checkpoint with the full
context and the final LayerNorm applied to each summand, the measured
`frien→d` decision (margin **8.30** for 'd' over the runner-up 't') is:

| summand | margin contribution |
|---|---|
| token embedding | +0.83 |
| L0 base (skip) | +5.34 |
| L0 wave (scan) | −5.34 |
| L1 base (skip) | +0.86 |
| L1 wave (scan) | **+7.39** |
| norm constant | −0.79 |
| **sum** | **+8.30** (exact) |

The L1-wave term further decomposes per mode (summed over channels):
[−2.91, +5.64, −0.81, +1.22, **+7.36**, −3.09] — modes 4 and 1 carry the
morphological completion, modes 0 and 5 vote against. No probing, no
approximation: this is the model's own arithmetic.

### A.6 The parallel-scan algebra (implementation, for completeness)

The recurrence is the associative composition
$(a_1, u_1) \circ (a_2, u_2) = (a_1 a_2,\; a_2 u_1 + u_2)$ on complex pairs;
the log-depth Hillis-Steele scan evaluates it in O(L log L) with the prefix
product p_n = Π_{j≤n} a_j updated by p_n ← p_n · p_{n−s} at shift s
(invariant: p is the prefix product *so far* — the comment in
`models/V8_WSKAN.py` marks the line where this was once wrong). Backward is
the same scan over time-reversed inputs with the **conjugated** multiplier:
since the forward Jacobian of h_n w.r.t. h_{n−1} is the complex multiply by
a_n, its adjoint is the multiply by conj(a_n) — so
D_n = G_n + conj(a_{n+1})·D_{n+1}, and
da_re = D_re·h_prev_re + D_im·h_prev_im,
da_im = D_im·h_prev_re − D_re·h_prev_im, du = D
(all verified ≤ 2e-6 against serial-loop autograd). The fused V11 kernel
computes both directions with `tl.associative_scan` over a coalesced
(L, N) tile — one launch per layer.

### A.7 What is measured vs what is derived (honesty map)

**Derived (exact):** the SSM⇔wavelet identity, admissibility correction,
the warped-time closed form, stability bounds, the decision decomposition,
the scan's forward/backward recurrences.
**Measured (checkpoint-read or intervention):** the Δ field values, the ρ
ladders, frequency placement, the causal effect sizes, the R² splits, the
Q statistic, the effective ranks.
**Rejected by measurement (kept for honesty):** constant-Q self-organization,
text-spectral-peak locking, turn-state tracking, the conv substitution,
the clamp hypothesis.

---

## 1. The computation chain (each link measured, most links causal)

### 1.1 The byte manifold organizes itself by linguistic function

The learned byte embedding (256×d, tied to the head) is a **functional
manifold**: within-class cosine similarity 0.47 vs between-class 0.07; PC1
carries 52% of variance. Nearest neighbors are linguistically exact —
`e → [i, a, o]` (vowels), `t → [T, c, s]` (case variant + frequent
consonants),
`space → [",", ".", ":"]` (separators), `5 → [6, 3, 4]` (adjacent digits),
`\n → [;, :, space]` (structural). *The geometry of English's byte alphabet
is learned, not imposed.* (§E, `V7_final_interpret.py`)

### 1.2 The model routes existing boundaries through a learned clock

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
exponentially more of the current word's memory. **Corrected claim (external
review):** the input layer's pulse is largely byte-keyed recognition of the
existing boundary bytes - with spaces deleted it vanishes at L0 (ratio
1.01). What is non-trivial: (a) deeper layers retain ~16% elevation at
implicit boundaries without any space input (partial learned
generalization), and (b) the *function* of this modulation is
position-specifically load-bearing (§1.3), i.e. the model did not merely
detect boundaries - it built its memory gating on them.

The distribution confirms the elevation is not a mean artifact: space's
p10 ≈ lowercase's p90 (E2 quantiles).

### 1.3 Position-specific causal evidence (necessity + sufficiency, with the reviewed caveat)

Teacher-forced CE on held-out text, Δ interventions:

| intervention | overall CE | at word-initial |
|---|---|---|
| baseline | 1.321 | 2.82 |
| remove boundary Δ (clamp to letter-mean) | 2.240 | **4.32 (+1.50)** |
| count-matched control (letters) | 1.993 | 2.89 (+0.07) |
| inject boundary-strength ticks mid-word | **3.49** | 3.67 |

Removing the boundary clock costs **21× more than the count-matched letter
control** at word transitions - this is stronger than "perturbing an
important parameter hurts", because the control isolates the *positions*.
Per-seed word-initial ΔCE (V6-era 3-seed battery):

| seed | boundary-clamp | letter-clamp (control) |
|---|---|---|
| 42 | +1.61 | +0.07 |
| 123 | +2.67 | +0.09 |
| 2024 | +2.41 | +0.08 |

The control effect is small and tight (+0.07..+0.09, seed std ≈ 0.01) while
the boundary effect is +1.5..+2.7 - the ratio is 18-30× with the control
far above zero but far below the effect. The 10M replication shows the same
pattern (+1.1..+1.55 vs +0.15..+0.18).
Injecting false ticks mid-word makes the model emit a "next word"
distribution where a word-internal letter was required (CE above even true
word-initial difficulty). *(Round-2 note on magnitude: the injection raised
Δ to boundary-strength at ~20% of all positions - a large-scale scrambling
of the distance field, not a gentle perturbation. The CE explosion reflects
intervention magnitude; it does not contradict fluency - fluency under an
intact field is the baseline, and the field here was deliberately wrecked
at one in five positions.)* **Reviewed caveat:** this proves the boundary-Δ
field is causally load-bearing for word-transition prediction; it does not
prove that Δ treats boundaries as linguistic objects (vs. as the byte
class), nor that no bypass path exists - both are open. All of this
replicates across seeds and implementations.

**Surgical version (round-3 answer to "interventions too coarse"):** a
*single* boundary-strength tick injected at one mid-word letter position:
CE at the immediately following position jumps 0.98 → **7.55** (the model
confidently emits a word-initial distribution mid-word), at the next-but-one
0.97 → 4.28, while overall CE moves only +0.04. The effect is local,
specific, and surgical - the tick makes a word boundary, one position at a
time.

### 1.4 The oscillator bank organizes time into a ladder - and the data built it

Each channel carries N damped oscillators λ = −σ + iω. What training built:

- **Learned ρ ladders are monotone geometric** (e.g. L2: [0.96, 1.00, 0.89,
  0.85, 0.84, 0.83]).

**Round-3 control (task-induced vs architecture-induced):** ρ init is flat
(ρ = 1 for all modes, `models/V6_WSKAN.py`), so the ladder is learned. But
learned *from what*? A shuffled-byte control (same training, byte order
destroyed within 512-byte windows - word statistics gone, byte histogram
kept) settles at a **non-monotone ρ** ([0.50, 0.45, 0.78, 1.36, 1.15, 0.93])
with mode frequencies 15x lower (0.013 vs 0.196 cycles/token). So both the
ladder and the word-scale concentration are **data-induced**: the
architecture supplies the interpretable coordinates; the data writes the
structure into them. That is the precise division of credit.
- **Mode frequencies concentrate at word scale** (median 0.165
  cycles/token ≈ 6-token period) — the band of morphemes/words, avoiding
  both the DC spike and byte-scale noise.
- **Q ≈ 0.8 on the canonical checkpoint** (0.57-0.67 on the earlier
  interp-era one; both marginally underdamped): the modes oscillate briefly
  (2-5 zero crossings per kernel, measured) but do not resonate; the
  constant-Q dictionary hypothesis was tested and honestly rejected
  (log-log R² ≤ 0.29). *(Review note: these are short-kernel oscillators,
  not "leaky integrators" and not resonators.)*
- **The frequency claim is functional, not just median:** readout-weighted
  mean ω̃ (weighted by |g|·Δ̄ per mode) is 10.4/10.8 rad per warped unit vs
  unweighted median 9.9/11.4 - the modes that actually get read sit at the
  same frequencies as the population median, so "word-scale" is not a
  median artifact (round-2 point 4 answered with measurement).
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

### 1.6 Circuits: generalists, specialist singletons, one hub

*Round-6 retitle (was "antipodal specialists"): the mirror-image pairing
is real but weaker and less exclusive than the original clustering
formulation — see below.* Channels cluster into many generalists plus
1–3 **specialist singletons** per layer (boundary detector, word-initial
detector, uppercase-suffix detector). The singletons are individually
load-bearing: zeroing the L0 boundary singleton alone costs **+1.08 CE**.
Depth retunes the code from byte-class (L0) to position-in-word (L1/L2).
On the "antipodal" pairing specifically (measured in
`W11_EXCESS_STRUCTURE_REPORT.md` §6): profile-level mirror pairs exist
above the init baseline (11–19 vs 6 pairs per layer at 100k, profile corr
< −0.8), so the pairing is learned — but a chance-level baseline exists
and gain-space antipodality is weak; read "antipodal" as "mirror-image
activation profiles", not as an exact circuit class. Kernel families (~6
per layer) route through channel clusters with a depth-migrating
topology, converging onto a single L2 output hub channel that the head
reads. Family ablations: the four big families are co-equal workhorses
(+0.29–0.36 each); the small specialist family is nearly redundant
(+0.017) — specialization lives at channel level, not edge level.

### 1.7 The decision: exact integration, opposing votes

Additive residual + linear head ⇒ every decision margin decomposes
**exactly**. Canonical case `frien→d` (margin 8.30 for 'd' over 't', full
table in §A.5): the L0 base vote (+5.34) and token prior (+0.83) favor 'd',
the L0 wave path opposes it (−5.34), and the **L1 wave path decides it
(+7.39)**, carried by modes 4 (+7.36) and 1 (+5.64) against modes 0 and 5
(−2.91, −3.09). Word endings resolve as **L1-over-L0 disagreement
resolutions**: the shallow layer's wavelet path resists ending the word, the
deep layer ends it. Across the attribution battery the token-embedding prior
votes *against* the produced letter on average — context beats prior.

### 1.8 Production: the clock paces speech

During autoregressive generation the clock pulses at boundaries exactly as
in analysis (letters 0.085–0.128 vs spaces 0.194–0.257). Word-internal
letters are wavelet-path decisions against the negative token prior; word
endings are layer disagreements resolved deep. A punctuation event triggers
a synchronized spike across clock, wavelet paths, and modes — the
production loop is the analysis loop, seen from the inside.

---

## 2. What the wavelet specifically buys: modest, consistent, scale-shrinking

The ablation (wskan11 vs ω≡0, identical everything) measures the
oscillatory function space's contribution. **It is small, and we say so:**

- **72 of 75 cells negative** (5 seeds × 5 sizes × 3 datasets): −0.035 /
  −0.046 / −0.041 nats at the 100k tier (TinyStories/UltraChat/WikiText).
- **Scale shape (honest):** the advantage peaks at 10k–100k
  (−0.04…−0.08) and shrinks at 1m–10m (−0.01…−0.03). Oscillation is the
  differentiator at small/mid scale — exactly where memory is short and
  phase-shaped kernels do the lifting.

**Honest size assessment (review-driven):** −0.03…−0.05 nats at mid scale
is a real but modest effect (a few percent of CE), and it shrinks with
scale - at 10M the oscillation is a small consistent bonus, not the
mechanism. The claim that survives is narrow: at small/mid scale, edge
functions with phase structure beat pure-decay edges, consistently. Nothing
larger is claimed.

**Why the architecture exists, then (round-5 answer):** not for the nats.
The ablation answers only "does the phase structure do anything at all" —
yes, consistently, exactly where memory is short. The architecture's
reason to exist is the analyzable object itself: edge functions as named,
causal, exactly decomposable mechanisms, i.e. everything in this document
outside Appendix C. If the performance effect vanished entirely at every
scale, the interpretability results would stand unchanged. Whether a
narrow-but-real efficiency effect at small scale is *important* is a
judgment we leave to the reader; we claim only that it is measured.

## 3. What 3 epochs buy (qualitative, all tiers)

Grammatical fluency and structural markers ("The end.", list formatting,
dialogue register) — with content still absent. The morphology strengthens
with epochs while semantics never arrive — consistent with the memory
analysis (effective half-life ~1 token at 100k; recall probes fail
everywhere, Mamba-2 included).

## 4. Scale laws (measured, multi-seed)

1. **Memory length grows with capacity**: half-life median 0.53 → 2.45
   tokens, tail 4 → ~3,190 (canonical all-sizes battery, Appendix B; an
   earlier draft quoted 4.5–5/~1,900 from the pre-campaign V3-era probe —
   superseded). σ clamp binding 43% → 0%.
2. **Content locality loosens — and the looseness is identified
   (round-5)**: byte identity 0.76 → ~0.55; bigrams 0.91 → ~0.76; the
   remainder at 10M (~25–40%) was "unidentified" in earlier revisions. A
   sequential family decomposition of the 10M write path (UltraChat,
   `experiments/W11_round5_probes.py`) shows it is predominantly
   **trigram-level statistics**: cumulative explained variance 0.48
   (byte) → 0.65 (bigram) → **0.94 (trigram)**; word identity adds 0.09,
   previous-word/positions ≤ 0.02 each; 5.4% genuinely unidentified
   (single seed, single dataset — scope noted). What grows with scale is
   the *order* of local statistics the model writes, not a new class of
   structure.
3. **Wavelet advantage peaks at mid scale** (§2).

## 5. What the model does NOT represent linearly (a main result, not a boundary)

The exclusion battery is direct quantitative evidence on representational
content, and it is one of this document's main findings:

| predictor of the write path | linear R² |
|---|---|
| byte identity | 1.00 (L0, trivial) → 0.76 → 0.50 |
| common bigrams (prev+current byte) | 0.91 / 0.84 |
| word identity (top-500) | 0.12-0.14 |
| position in sentence | ~0.007 |
| position in document | ~0.001 |
| dialogue-turn state | ≤ 0.003 |

**The model has no *linear* readout of anything above the bigram level**:
no word identity, no sentence position, no document structure, no turn
state is linearly decodable from the write path. *Round-4 correction and
round-5 refinement:* an earlier version of this section said "provably does
not represent anything above the bigram level" — that overclaims, because
the battery used linear probes only. Nonlinear probes (1-hidden-layer MLP,
500-way word-identity, **held-out split**; the 37.8% figure quoted in the
round-4 revision was train-contaminated and is superseded) measure how the
information attenuates along the path the model itself uses:

| signal | linear R² | MLP held-out acc (chance 0.3%) |
|---|---|---|
| residual-B (the write path) | 0.149 | 26.3% |
| y (post-gate wave output — what enters the stream) | 0.155 | 23.4% |
| residual stream (what the head sees) | 0.157 | 21.5% |

Two facts, kept separate. (i) Word-level information **is present** at
every stage, and the model's own gate/readout does **not** filter it out —
it attenuates only mildly (26% → 22%). (ii) At no stage is it **linearly
accessible**, and the model's readout (C) is linear — so the information
is not exposed in any form the model's own linear machinery can read.
Whether downstream nonlinearities (the next layer's gates) exploit it
anyway is possible in principle and not established either way. The honest
title of this section's finding is therefore **"linearly accessible form,
linearly inaccessible content"** — a statement about what the model can
*use*, not about what its signals contain. Sentence/document/turn
variables remain undetected even linearly, and their nonlinear status is
open. The recall statement (§6) is behavioral and unaffected.

## 6. The honest boundary

- **No facts, no recall** — at every scale probed (behavioral
  memorize-then-recall probe: number+name stated, story interleaved, then
  asked; fails identically on matched-size Mamba-2, so this is a
  scale/training-level observation shared across architectures - not an
  architecture-level claim, and whether larger scale fixes it is open).
  **Content** is *not absent* from the signals — §5 shows word-level
  information present throughout, but only in linearly inaccessible form.
  The machine computes form; the only content it provably *uses* is form.
- The clock knows boundaries, not meaning.
- **The organization: load-bearing for the found solution, NOT required
  by the task — both directions measured
  (`W11_EXCESS_STRUCTURE_REPORT.md`).** Surgery on the found solution
  (3 seeds × 3 perms): ladder flatten +0.13–0.33, mode-consistency
  shuffles +1.8–2.6, feature tables +5.5 nats; a full mode permutation is
  an exact gauge (+1e-8). Retraining with the structure impossible
  (campaign protocol): feature tables +0.006, ρ ladder +0.009, single
  mode +0.041 (param-matched), no frequency +0.037. Gradient descent
  finds this organization *immediately and reliably* (π-harmonic prior +
  ~250-step emergence, oscillation-free in wskan11real, boundary gating
  also in mamba2 with the opposite operation — skip vs reset, measured) —
  but the task barely requires any single piece of it. The structure is
  the path of least resistance under this parameterization, which is what
  makes it worth analyzing.
- The dormant long-memory tail (up to ~3,190 tokens at 10M, Appendix B)
  carries < 1% of readout weight — present, unused.
- Forward derivation without execution: **not achieved, at any scale
  (Appendix D)**. What exists is *transparent simulation* — execution
  whose every intermediate is named and printed. At 1k/10k it is complete
  (the whole computation fits in working memory; median 12–26 of ≤ 1,728
  history terms carry the decision); at 100k it is truncated (median 6 of
  11,520; ≤ 25 terms suffice in 74% of contexts). But the gates are
  input-dependent and must be *executed*, not derived — so "the ceiling is
  lifted" was too generous; the precise statement is that the computation
  is fully transparent at small scale and translucent at 100k.

## 7. Why this is (and isn't) a KAN-specific answer

*Corrected after review:* exact residual-stream decomposition is NOT unique
to this architecture - any additive-residual + linear-head model
(transformers included) admits the same path-level decomposition. What is
specific to wskan11 is the **unit of decomposition**: per-edge functions
ψ_io(t) (named time functions) rather than per-head attention maps or MLP
neuron activations, plus the clock field Δ as a first-class object. The
decomposition machinery is shared; the objects decomposed into are not.

*On the value of interpretable-by-design (round-5):* this work does not
directly help interpret production models, and we do not claim otherwise.
What it offers is (a) a **ground truth** of learnable computation — every
mechanism here is measured, not hypothesized; (b) a **benchmark** on which
post-hoc interpretability methods can be validated against exact answers;
(c) **clean hypothesis tests** (clock, gate split, mode roles) of the kind
that are ambiguous in larger models. How much that is worth is the
community's call; what we stand behind is that the measurements are exact
and the claims are scoped to them.

## 8. Reproduction

```bash
.venv/bin/python experiments/W7BC_canonical_probe.py        # §1.2–1.5 (canonical ckpt)
.venv/bin/python experiments/V7_causal_clock_full.py        # §1.3 causal battery
.venv/bin/python experiments/W7BC_generation_analysis.py    # §1.8 production trace
.venv/bin/python experiments/CAMPAIGN_aggregate.py          # §2 ablation + footnote tables
.venv/bin/python experiments/W11_hand_simulation.py         # Appendix D hand-simulation
.venv/bin/python experiments/W11_round5_probes.py           # §4 trigram identification, §5 usefulness ladder
.venv/bin/python experiments/W11_excess_structure.py        # §6 organization-origin battery
.venv/bin/python experiments/W11_ablation_retrain.py --variant {n1|n1wide|nofeat|rhofrozen}  # §6 retrain test
```
Canonical checkpoint: `checkpoints/wskan11_ultrachat_100k_3ep_s42/`.
Interpretation checkpoints (protected): `checkpoints/*_interp_*`.
Numbers: `figures/v7e_canonical_probe.json`, `v7e_10m_analysis.json`,
`v7e_strengthening.json`, campaign CSVs.

## Appendix B. The same anatomy at every scale (5 tiers x 5 seeds)

*(Answers the round-3 review: the anatomy is not a 100k story.)* The full
battery - word clock, memory profile, causal battery, variance
decomposition, canonical attribution - run on the 3-epoch wskan11
checkpoints at every tier (`experiments/W11_all_sizes_analysis.py`,
numbers: `figures/w11_all_sizes.json`). All UltraChat.

| tier | clock ratio (L1, 5 seeds) | half-life med (tok) | half-life max | word-initial effect vs control | frien→d |
|---|---|---|---|---|---|
| 1k | 1.63 ± 0.40 | 0.53 | 4 | +1.70 ± 1.08 vs +0.00 ± 0.00 | 0/5 |
| 10k | 2.41 ± 0.24 | 0.63 | 2* | +4.01 ± 2.02 vs +0.01 ± 0.01 | 2/5 |
| 100k | 1.95 ± 0.27 | 0.92 | 69 | +2.50 ± 1.00 vs +0.13 ± 0.01 | 5/5 |
| 1m | 1.71 ± 0.05 | 1.28 | 92 | +1.12 ± 0.11 vs +0.20 ± 0.01 | 5/5 |
| 10m | 2.48 ± 0.14 | 2.45 | 3190 | +3.16 ± 0.56 vs +0.35 ± 0.01 | 5/5 |

\* the 10k max=2 is a single-layer-tier artifact (no second layer to host
slow modes); flagged, not smoothed over.

**What is scale-invariant:** the word clock (ratio 1.6-2.5 at every tier,
every seed) and its causal role (the word-initial effect exceeds the
matched control by an order of magnitude at every tier; the control is
tightly above zero). **What scales:** memory (median 0.5 -> 2.5 tokens,
tail 4 -> 3190), morphological competence (frien→d succeeds only >= 100k),
and the causal effect's tightness (noisiest at 1k). The anatomy is
universal; its dimensions are not.

**Round-3 answer ("if dimensions change, doesn't the interpretation
change?"):** the *objects* (clock field, mode bank, gate tables, circuit
topology) persist at every scale; the *content* loaded into them changes.
Round-5 measurement sharpened what changes: the 10M write path's
non-bigram share is predominantly **trigram-level** statistics (§4), not
discourse structure - so the scale change is the *order of local
statistics*, not a new class of object. Interpretability transfers at the
object level; the semantic inventory grows by one n-gram order at 10M.
And on "do we need to read all 32 modes": the exact decomposition is
the floor; the ~6-family clustering (Appendix B-era analysis) is the
working resolution; reading all modes is never required.

## Appendix C. Performance context (units declared, no claims)

All numbers are **byte-level cross-entropy** (nats per byte) on held-out
data, 3-epoch campaign, matched parameter tiers, 5 seeds. This is context,
not a claim of superiority - at every tier the field is within ~0.1 nats
and rankings shuffle by dataset.

| tier | wskan11 | wskan11real | mamba2 | tf |
|---|---|---|---|---|
| 100k (TinyStories) | 0.776±.008 | 0.812±.005 | **0.747±.004** | 0.836±.020 |
| 100k (UltraChat) | 1.184±.021 | 1.230±.018 | **1.158±.019** | 1.288±.027 |
| 100k (WikiText) | 1.303±.012 | 1.344±.008 | **1.282±.010** | 1.397±.021 |
| 10m (TinyStories) | 0.546±.005 | 0.558±.004 | **0.478±.007** | 0.487±.009 |
| 10m (UltraChat) | 0.833±.006 | 0.861±.007 | **0.716±.007** | 0.799±.027 |

Reading: Mamba-2 leads at 100k (narrowly) and at 10m (clearly); wskan11 is
competitive but not dominant anywhere; the ω≡0 ablation loses to wskan11 in
72/75 paired cells. Full tables: `CAMPAIGN_aggregate.py`.

## Appendix D. Deriving the next token by hand (the ceiling, attacked)

*This appendix answers a natural question raised against section 6: if the
model is a bundle of named functions, can its next-token prediction be
derived by hand, not merely decomposed after the fact? Script:
`experiments/W11_hand_simulation.py`, numbers:
`figures/w11_hand_simulation.json`.*

**What "by hand" means here (machine-verified hand-simulation):** an
itemized from-scratch forward pass — serial scan, explicit warped-time
history sums, every gate value printed — reproduces the model's logits to
**max error 3.8e-6** at the 1k/10k/100k tiers (per-position scan error
≤ 1.5e-5), and every decomposition below is integrity-checked
(parts must sum to the normed final state within 1e-4, asserted). The
script performs only arithmetic whose every intermediate is displayed; any
single step is checkable with a calculator. At 1k (d = 4, one layer, six
modes) the entire computation fits on a page: 576 history terms for a
24-byte context, 4 clock numbers per byte.

**Worked example (1k, TinyStories, eval stream).** Context
`utiful yellow sunrise.\nO` → the model produces 'n' (a new sentence:
"Once/On..."). The clock field is readable at a glance — spaces tick
Δ ≈ [0.68, 0.65, 0.63, 0.26], '.' ≈ [0.82, 0.72, 0.59, 0.32], letters
lower — and the decision decomposes exactly: embedding −3.46, base +1.66,
**wave +9.26**, norm-const +0.69 (sum 8.15, exact). The wave's top history
term is the **period two bytes back**, read through mode 4 at warped
distance 0.628: +5.37; the newline adds +3.16; the current 'O' contributes
+0.74. The decision is literally "the period–newline pair two steps back,
attenuated by the learned kernel, primes a sentence start, and given 'O'
the completion 'n' (On/Once) wins." A wrong
case is equally legible: after `"Ow! Tha` the model says 'n' (**"Than"**)
where the text has 't' ("That") — a near-miss, driven by the base path
(+9.05); failures at this scale are readable, not mysterious.

**How many terms a hand must sum (truncation curve, 50 contexts each):**

| tier | history terms | static parts alone correct | median terms needed | ≤ 25 terms | p90 |
|---|---|---|---|---|---|
| 1k | 576 | 10% | 12 | 76% | 38 |
| 10k | 1,728 | 14% | 26 | 50% | 135 |
| 100k | 11,520 | 26% | 6 | 74% | 97 |

Two findings. (i) The static parts (embedding + base + norm) almost never
decide alone — **history is the decision**, even at 1k. (ii) Yet the
decision is a low-rank event: the median context needs 6–26 of up to
11,520 history terms, ranked by contribution to the predicted byte, for
the prediction to stabilize — and this holds at 100k, not just at the toy
tiers. A human summing the top terms reproduces the model's argmax in
most contexts; the long tail is real but rarely load-bearing.

**Dictionary-guided prediction at 100k (the honest limit).** A
pre-registered rule — 4-byte suffixes seen ≥ 8 times in training with
≥ 75% next-byte purity (20 eval contexts) — matches the model's output
**20/20**: the model has these completions. But the *mechanism* varies:
the L1 wave path is the dominant margin contributor in only 8/20 (L0
wave/base decide the rest), and the canonical `frien→d` mode dictionary
{modes 1, 4} appears in the top-2 of just 2/20. **The object-level
anatomy transfers; the specific mode assignments are per-decision, not a
global dictionary** — the Appendix B answer, now measured at the decision
level.

**What this closes and what it does not.** *Round-5 terminology
correction:* "lifting the ceiling" was too generous. What exists is
**transparent simulation** — execution whose every intermediate is named
and printed — not derivation. At 1k/10k the transparent simulation is
complete (every number on the page); at 100k it is truncated (median 6
terms, plus the exact post-hoc decomposition of §A.5). **Derivation
without execution is achieved nowhere**: the gates are input-dependent,
so per-position gate values must still be computed (printed, not
predicted), and no claim is made that a human *without* the printed
tables predicts anything.
