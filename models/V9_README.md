# V9 — Wavelet-Generated FIR Convolution (NOT ADOPTED as the main line)

**What it is:** the edge wavelets evaluated at integer lags into short causal
convolution kernels (K=32), with V2-style static input/output gates. No scan,
no recurrence - one tensor-core conv1d per layer.

**Speed (measured, compiled):** 100k: 3.0 ms/step (13x vs V7, 4x vs V8),
1m: 15.4 ms (46x vs V7), 10m: 85.7 ms (23x vs V7). Params 100,608 at the
100k config (d32L3K32).

**Why not adopted — the approximation cost is measured, not argued:**

| step | V9 (FIR conv) | V2 (LTI, exact FFT kernels) | wskan7bc (selective) |
|---|---|---|---|
| 2,000 | 1.76 | 1.23 | 1.01 |
| 10,000 | 1.51 | ~1.17 | ~0.95 |

V9 trails its own LTI sibling (V2) by ~0.4 nats and the selective line by
more. The truncation itself is harmless (near-delta kernels decay long
before K=32); the cost is the dropped word clock (Delta warping) - which the
causal battery priced at +0.92 nats when removed. The evidence predicted
V9's failure mode and the measurement confirmed it.

**Lesson (recorded):** within this family, the recurrent clock is not
overhead to be approximated away - it is the mechanism. Speed must come from
faster evaluation of the same recurrence (V8), not from replacing it.

**Residual value:** V9 is the project's extreme-speed interpretable tier
(kernels are the wavelets; nothing to probe) - useful for rapid sanity
checks and as the LTI ablation endpoint.

## Reproduction

```bash
# models/V9_WSKAN.py ; trainer flag --model wskan9 (not wired into the matrix)
```
