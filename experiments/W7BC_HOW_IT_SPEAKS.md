# How wskan7bc Speaks Language: The Complete Mathematical Mechanism

**The project's main document.** Interpretability first; performance is a
footnote. Every number below was measured on the canonical fresh checkpoint
`checkpoints/wskan7bc_ultrachat_interp_s42` (d=32, 3 layers, N=6, byte-level
LM, 118,386 parameters, UltraChat 100k steps) with
`experiments/W7BC_canonical_probe.py` and the analysis scripts in this
directory. Nothing here is inferred from behavior alone: every claim is read
from the model's own mathematical objects, and the central ones are
intervention-verified.

## 0. The answer

wskan7bc speaks by running a **word-clocked damped-oscillator memory** over
a word-segmented timeline it invented: bytes enter through a
class-structured embedding manifold; an input-driven clock Δ ticks 2–2.3×
faster at word boundaries than inside words (causally verified as both
necessary and sufficient for word-transition prediction); each boundary
tick decays the oscillator states toward the next word; byte identities are
*written* into damped-oscillator modes whose timescales span a learned
geometric ladder, *read out* through gates that specialize structure
(newline/digit/punct) to lookup tables and content to a byte-identity
channel; 1,024 explicit wavelet kernels per layer (organized in ~6
high-rank families) convolve the written content in warped time; circuits
route through channel clusters to a final-layer output hub; and the tied
embedding head converts the hub state into the next byte — the decision
being an exact, attributable integration in which the token prior often
loses and word endings are resolved as deep-over-shallow disagreements. It
is a fluent **register learner**: it knows the forms of language, not (at
this scale) the facts.

## 1. The inventory (every parameter is a named object)

| object | shape (per layer) | what it is |
|---|---|---|
| byte embedding E (tied head) | 256×d | class-structured manifold (below) |
| W_dt | d→d | the **word clock**: Δ = softplus(W_dt·norm(x)+b), capped ≤1 |
| M_B, M_C | 8×d×N | **named write/read tables** over fixed byte features |
| res_B, res_C | d→32→d·N | residual write/read path (byte-identity → context) |
| σ, ω | d×N | damped-oscillator decay/frequency banks (λ = −σ+iω) |
| ρ | N | static scale ladder: effective modes λ̃ = ρλ |
| g = a+ib | d×d×N | edge gains: the **1,024 explicit wavelet kernels** |
| W_B→(M_B,res_B), W_C→(M_C,res_C) | above | write/read gates |
| W_z, w_base, w_wav | d×d, d×d, d×d | output gate, skip, per-edge scale |
| prenorms, final norm | d | stabilize the residual (load-bearing, incident history) |

Per-edge function: ψ_io(t) = Σ_k Re[g_iok e^{λ̃_ik t}]. Selective scan:
h_{n,ik} = e^{λ̃_ik Δ_{n,i}} h_{n-1,ik} + B_{n,ik} Δ_{n,i} x_{n,i},
y = (Σ_{i,k} C_{n,ik} Re[g·h]) ⊙ SiLU(W_z x) + x·w_base.
Stability |e^{λ̃Δ}| < 1 and causal-decay non-positivity are guaranteed by
construction (σ = e^{logσ} > 0, Δ ≥ 0).

## 2. Reading: text becomes state on a word-segmented timeline

**Byte manifold.** The embedding is functionally organized (within-class
cosine 0.47 vs between 0.07; PC1 52%): nearest neighbors are vowels→vowels,
digits→adjacent digits, space→separators, newline→structural. Bytes enter
already sorted by linguistic function.

**The word clock (centerpiece).** Measured Δ per byte class (fresh):

| class | Δ (L0/L1/L2) |
|---|---|
| lowercase | 0.084 / 0.104 / 0.123 |
| uppercase, digit | 0.12–0.13 / 0.11–0.18 / 0.11–0.20 |
| space | **0.194 / 0.204 / 0.256** |
| newline | **0.205 / 0.174 / 0.243** |
| punctuation | **0.204 / 0.189 / 0.204** |

Word boundaries advance the internal clock **2–2.3× faster** than letters
in every layer. Since state decay is e^{−ρσΔ}, each boundary erases
exponentially more of the current word's memory: **the effective unit of
distance is the word.** The tokenizer was never given; segmentation was
learned through time warping.

**Causal proof (fresh, 32×512 held-out):**
- Remove boundary Δ: overall CE 1.321 → **2.240 (+0.92)**; at word-initial
  predictions 2.82 → **4.32 (+1.50)**.
- Same-size control at letter positions: +0.67 overall, **+0.07** at
  word-initial — a **21×** necessity ratio.
- Inject boundary-strength ticks mid-word: CE explodes to **3.49** —
  false ticks *create* word-transition behavior (sufficiency).

The clock is driven by mainstream channels (top-8 drivers ≈ 35% of drive,
no dedicated clock neuron) — a distributed mechanism, consistent with its
load-bearing ubiquity.

## 3. The memory: what damped oscillators store

**Timescales.** The ρ ladders learned monotone geometric structure (L2:
[0.96, 1.00, 0.89, 0.85, 0.84, 0.83]); effective mode half-lives span
~1 token (bulk) to a dormant tail of ~40–100 tokens (< 1% readout weight at
this scale). Mode frequencies concentrate at word-scale (~0.16 cycles/token
≈ 6-token period), Q ≈ 0.6 (overdamped: phase structure, not ringing).
The constant-Q dictionary hypothesis was tested and **rejected** (R² ≤ 0.29).

**What is written.** The two-part gate (measured by direct table lookup and
variance decomposition):
- **Named tables (M_B/M_C): structural bytes only.** digit 1.42, newline,
  punct strong; letters ~0.2–0.4 — the lookup-interpretable channel
  self-organized to carry discrete formatting, with newline's read routing
  migrating across depth.
- **Residual path: byte identity → context.** R² of write variance: byte
  identity 0.98 (L0) → 0.66 (L1) → 0.41 (L2); position-in-word rises
  0.12 → 0.23 → 0.16. The content memory addresses by *what byte* early,
  *where/what context* late. The remaining context is not dialogue-turn
  state (hypothesis **refuted**, R² ≤ 0.003).

**The kernels.** g is genuinely high-rank (effective rank 28–29/32 per mode
— the reason the rank-32 filter compression costs +0.078 nats, V7 ablation).
Kernels cluster into ~6 families per layer (steps, ramps, spikes, V-transients,
notches, rare long-tails) — no sustained oscillation anywhere.

## 4. The assembly: circuits and hubs

Channels form a dictionary: **many generalists + 1–3 antipodal specialist
singletons** (boundary detector, word-initial detector, uppercase-suffix
detector) per layer; the specialists are individually load-bearing (zeroing
the L0 boundary singleton costs +1.08 CE). Depth shifts the code from
byte-class (L0) to position-in-word (L1/L2). Kernel families route:
specialist families F2/F4 feed L0 hubs; L1 is a uniform routing band; all
six families converge onto a **single L2 output hub channel** (universal
write target), which the head reads. Family ablations: the four big
families are co-equal workhorses (+0.29–0.36 CE each); the small
specialist family is nearly redundant (+0.017).

## 5. The decision: exact integration, then production

The final state is an additive sum; the tied head is linear — so every
decision margin decomposes **exactly**:

logit_v = W_head[v] · (emb(x_last) + Σ_l (base_l + wave_l))

Canonical case `frien→d` (fresh): margin 8.86 in favor of 'd' over 'n';
per-mode contributions [+1.94, +3.91, +0.73, **+7.70**, +4.87, −0.35] with
channel 0 dominating. Morphological completion localizes to named modes;
boundary emission is distributed (no single boundary mode).

**Production (generated trace, fresh).** During autoregressive emission the
clock pulses at boundaries exactly as in analysis (letters 0.085–0.128 vs
spaces 0.194–0.257): word-internal letters are produced by the wavelet
paths against a *negative* token-embedding prior (context beats prior);
word endings resolve as **L1-over-L0 disagreements** (shallow resists,
deep decides). The generated register is fluent; the content is empty —
consistent with everything above: the machine computes *form*, and at this
scale form is what there is.

## 6. What it cannot do (the honest boundary)

1. **No facts, no recall** — memory half-life ~1 token effective; recall
   probes fail identically to matched Mamba-2.
2. **The long-memory tail is dormant** — slow modes exist (up to ~100
   tokens) but carry < 1% of readout weight at 100k params; the 10M probe
   showed the tail activates with capacity.
3. **~30–40% of the write variance is unexplained context** (not byte
   identity, not position, not turn structure — open).
4. **No forward hand-simulation** — we can decompose any decision after the
   fact and predict intervention effects qualitatively, not derive outputs
   without executing the network (the scale-imposed ceiling, §full account).

## 7. Provenance and reproduction

- Canonical numbers: `figures/v7e_canonical_probe.json` from
  `experiments/W7BC_canonical_probe.py` on `wskan7bc_ultrachat_interp_s42`.
- Deep batteries: `V7_deep_interpret.py`, `V7_final_interpret.py`,
  `V7_causal_dictionaries.py` (+ their reports) — mechanisms replicated on
  this checkpoint (word clock, causal battery, production pacing).
- Performance (footnote): `W7BC_FINAL_CLEAN_REPORT.md` — wskan7bc is the
  best of the WSKAN series and mean-edge over Mamba-2 at ~100k; the
  interpretation above does not depend on winning any benchmark.

## 8. The mechanism in one sentence

**A learned word-segmented clock drives a geometric-ladder damped-oscillator
memory, whose structural-vs-content write gates, high-rank wavelet kernel
families, specialist channels, and a convergent output hub integrate
context into next-byte decisions that are exactly decomposable and — at the
clock — causally proven.**

```bash
.venv/bin/python experiments/W7BC_canonical_probe.py   # reproduces §2–§6
```
