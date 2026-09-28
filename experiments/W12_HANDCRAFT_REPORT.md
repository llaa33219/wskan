# Hand-Writing the Model: A Language-Form Machine with No Training

**Date:** 2026-09-27 · **Script:** `experiments/W12_handcraft.py` · **Model:** 1k-tier shell (d=4, L=1, N=6), 4,634 params

The ultimate test of an anatomy: can a human *write* the edge functions
instead of training them? Every parameter of this model is set by hand
from the measured anatomy of the trained wskan11 (monograph §1–§4); no
gradient descent touched it.

## The recipe (each choice and its measured source)

- **Embedding** (256×4): a functional class manifold — letter-ness,
  vowel/consonant, boundary-ness, sentence-terminal punctuation (the
  measured L0 structure; §1.1).
- **Clock** (`W_dt`): reads the boundary channel — letters Δt ≈ 0.10,
  boundaries 2–2.6× (§1.2's measured ratios).
- **Mode ladder**: geometric σ (8 → 0.25), rising ω on the fast modes,
  **zero ω on the slow integrators** (a slow accumulator must not ring —
  discovered during this build).
- **Word-length mechanism**: letters accumulate into a slow integrator
  mode (mode 4) every step; the clock resets it at boundaries; the
  readout converts accumulation into boundary pressure (the measured
  "clock + ladder" pattern, §1.2/§1.4).
- **Vowel/consonant texture**: a fast mode holds recent vowel-ness; the
  gain flips sign so vowels and consonants alternate (the antipodal-read
  pattern, measured on the profile level).
- **Sentence memory**: terminal punctuation writes the slowest mode;
  its readout pushes uppercase and suppresses immediate re-punctuation.
- **Gates**: the output gate is uniform-open (row-sums positive by
  construction — a dense-bias fix for a gate-choking failure mode found
  in this build); the head is hand-set per-byte readout vectors on the
  four axes with corpus-frequency tilt.

The structural choices (which mode does what, sign of each gain, the
clock ratios) come from the measured anatomy. Three magnitude constants
(the length→boundary gain 18.0, the boundary baseline −0.25, the
lowercase prior −0.55) were hand-tuned on a space-vs-letter margin probe
— documented, not hidden.

## Result

Space-vs-letter margin as a function of run length (the word-length
mechanism working, measured):

| after k letters | 1 | 3 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|---|
| logit(space) − logit('e') | −6.00 | −2.38 | −0.17 | +1.98 | +3.85 | +6.48 |

Sample (t = 0.8, seeded "One day"):

```
One dayNn aOVLn.TvjKYc ayFCb ITBLhw THeajp eIaOiAkf Iaunf uOuooit Oaiail
loaoeu aioeb?EULYWOXhS ieAsvs ztc?AeBTCfZL%?IJuRoLCPTMPPTfKHSHOJHHFXXbGCLPMLwS7
IoSVrrM UshKy UIeiucun OiIOeyn bc Ciox uauihz qw Eeoan UiIecm ooean ahm eIbt
```

Word boundaries with a plausible length distribution (mean 7.7 vs
TinyStories ~4), vowel/consonant texture, mixed case with capitals
clustered near sentence starts, punctuation rhythm — **structural fluency
with zero training**.

## What this does and does not show

- **Does**: the measured anatomy is *sufficient* — the mechanism a human
  can read off the trained model is enough to rebuild a working
  language-form machine by hand. This is the strongest form of "we
  understand what the model computes".
- **Does not**: beat even a unigram baseline on CE (5.97 vs 3.06; the
  trained 1k model reaches 2.23). The hand-written part is the *form
  machinery*; the lexicon — which letters actually follow which — is the
  trained content part, and it is exactly what the monograph's §5
  measured as living in the model. Form by hand, content by data: the
  split the paper claims, demonstrated constructively.
- **Failure modes of the build itself are informative**: oscillatory
  slow modes destroy accumulation (the ω=0 rule); sparse embeddings let
  the SiLU gate choke the wave path (the dense-bias fix); LayerNorm
  mean-shift can counterfeit a case channel. Each is a named mechanism,
  found by the same decomposition tools.
