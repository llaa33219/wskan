# V8 — Wavelet-State-KAN: GPU-Native Implementation (Speed Rewrite)

**What V8 is:** the wskan7bc architecture with the same mathematics, re-implemented
for GPUs: the chunked-SSD Python loop is replaced by a **log-depth parallel
associative scan** (Hillis-Steele over the recurrence h_n = a_n h_{n-1} + u_n,
complex as real pairs) wrapped in a custom `autograd.Function` whose backward
is another parallel scan (reverse direction, conjugated multiplier), all
captured by `torch.compile(mode="reduce-overhead")` (CUDA graphs).

**Checkpoint-compatible with wskan7bc** (identical parameter names/shapes).

## Verified correctness

- Scan vs serial loop: 1.6e-6 (layer level), all four gradient tensors vs
  serial-loop autograd: ≤ 3e-6.
- Full model vs V7 (same trained checkpoint): argmax identical; logit diff
  ~0.017 (fp32 reduction-order noise over 300 positions — documented, both
  orders are valid fp32 computations).

## Optional approximations (all measured at 3k steps, TinyStories)

| variant | quality cost (eval@3k) | speed at 100k | speed at 1m | speed at 10m |
|---|---|---|---|---|
| V8 N6 fp32 (reference) | 0.9896 | 20 ms | 101 ms | 250 ms |
| **N3 (halve the mode bank)** | +0.048 | ~14 ms | 33 ms (bf16) | 97 ms (bf16) |
| **bf16 scan** | ~0 (measured rel. L2 3e-5-1.3e-4) | 11.6 ms | 66 ms | 178 ms |
| N3 + bf16 | +0.10 | ~8 ms | **33 ms** | **97 ms** |
| L128 (short context) | +0.052 | ~23 ms | - | - (not worth it) |

Final stack vs V7 chunked: **3.3x (100k), 21x (1m), 21x (10m)** with
N3+bf16 at ~+0.05 nats measured quality cost. N and bf16 are flags
(`n_states`, `--bf16`); the wavelet ladder, the clock, and every
interpretation tool are unchanged.

## Honest engineering log: why not 40x

Target was 40x at the pathological tiers. Approaches tried and measured:

1. **Chunked SSD with Python chunk loop (V3-V7)** — launch-bound (thousands
   of tiny kernels per step); the 0.7 s/step bottleneck.
2. **Hillis-Steele + custom Function + CUDA graphs** — the adopted path;
   removes launch overhead and shrinks the autograd graph to one node per
   layer.
3. **PyTorch HOP `associative_scan`** — exact and autograd-safe, but slower
   than (2) at model level (graph partitioning around the HOP).
4. **Triton scalar lane-serial kernels** — verified correct (≤3e-6) but
   slower: scalar per-element global access per lane is latency-dominated.
5. **Triton 2-kernel segmented scan** — verified correct, also slower than
   (2) in this regime.

**What 40x would take (not faked):** a Mamba-class fused SSD kernel — the
segsum + matmul formulation with per-head (not per-edge) blocking on tensor
cores, or a register-resident vectorized serial scan. Our per-(channel, mode)
structure (I×N independent sequences per layer) makes small-op fusion the
dominant cost and is the actual obstacle; it's a multi-day kernel project,
deliberately not faked here. The remaining gap is also bounded by the
readout einsum at large d (tensor-core-bound, theoretically fine).

## Numerical guarantees

|a_n| ≤ 1 by construction (σ > 0, Δ ≥ 0): prefix products only decay; no
exp/log hazards anywhere in the scan (the V1/V3 incident classes are
structurally absent). Verified adversarially at σ = e^8, ρ = e^3.

## Files

| File | Content |
|---|---|
| `models/V8_WSKAN.py` | `FastWaveletStateKANLayer`, `WaveletStateKANLMV8`, `_scan_fwd`, `_AssocScan` |
| `models/V8_triton_scan.py` | verified segmented Triton kernels (not adopted — slower here; kept as reference) |
| trainer flag | `--model wskan8` (V7bc-compatible checkpoints load directly) |
