# V11 — Wavelet-State-KAN: Fused-Kernel Scan (the 40x tier)

**What V11 is:** V8's exact mathematics with the scan computed by a fused
Triton kernel — one `tl.associative_scan` call per layer over a coalesced
(L, N) tile per (b, i) lane. Backward is the same kernel over time-reversed
inputs with the conjugated multiplier (flip + conjugate trick). The
autograd graph per layer collapses to a single Function node.

**Checkpoint-compatible with wskan7bc** (identical parameter names/shapes;
argmax-identical outputs, logits within fp32 reduction-order noise).

## The final speed table (batch 64, block 256, compiled, autocast bf16)

| tier | V7 chunked | V8 | **V11 (fused, N3, bf16)** | vs V7 |
|---|---|---|---|---|
| 100k (d32L3) | ~38 ms | 20 ms | **4.3 ms** | 8.8x |
| 1m (d80L6) | ~700 ms | 101 ms | **17.3 ms** | **40.4x** |
| 10m (d512L2) | ~2000 ms | 250 ms | **47.0 ms** | **42.6x** |

fp32-only V11 (no bf16, N6): 10.5 / 54 / 145 ms. bf16 scan precision
measured at 3e-5–1.3e-4 relative L2 (V8 README); V11's quality cost at 3k
steps: +0.026 nats vs V8 N6 fp32 (mostly the N3 mode-halving, measured
separately at +0.03).

**The 40x target is met at 1m and 10m.** At 100k, 4.3 ms is near the
overhead floor (even V9's near-empty conv takes 3 ms).

## Verified

- Kernel vs serial loop: forward 7e-7, all four gradients ≤ 4e-6, multiple
  shapes (incl. non-power-of-2 L, N padding path).
- Model-level: argmax-identical to wskan7bc on the trained checkpoint;
  logits within reduction-order noise.
- Quality probe (3k steps, TinyStories): eval 1.024 vs V8's 0.998.

## Engineering log (why it took this path)

1. Chunked SSD with Python loop (V3-V7) — launch-bound.
2. Hand Hillis-Steele + CUDA graphs (V8) — good, 4-8x.
3. HOP associative_scan — exact but graph-partitions badly (slower).
4. Triton scalar lane kernels — verified, latency-bound, rejected.
5. Triton segmented 2-kernel — verified, still slower than (2), rejected.
6. **tl.associative_scan tile kernel (V11)** — the winner: coalesced tile
   loads, warp-level log-depth scan on chip, one launch per layer.
7. bf16 scan + autocast — the last 1.5-2x.

## Files

| File | Content |
|---|---|
| `models/V11_kernel.py` | `_v11_fwd_kernel`, `_v11_bwd_kernel`, `v11_scan_fwd/bwd` |
| `models/V11_WSKAN.py` | `_FusedScan` (autograd.Function), `FusedWaveletStateKANLayer`, `WaveletStateKANLMV11` |
| trainer flag | `--model wskan11` (bf16 via `--bf16`; N via config) |
