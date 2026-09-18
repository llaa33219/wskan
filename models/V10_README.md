# V10 — Segmented (Word-Level) Wavelet-SSM (prototype, NOT adopted)

**The idea:** the word clock's measured structure (binary Delta:
boundary vs letter) made exact — segment at boundary bytes, run LTI
wavelet convs inside words (tensor cores), carry the state across words
with the boundary decay (the clock's causal core, preserved exactly).

**Quality (measured, TinyStories 3k steps):** eval 1.17 — between V2
(1.29, LTI) and wskan7bc (1.01, full selectivity), far ahead of V9's FIR
approximation (1.64). The segmented structure demonstrably preserves most
of the clock's contribution.

**Speed (measured, compiled):** ~149 ms/step at the 100k config — SLOWER
than V8 (9-21 ms). The pack/scatter/gather machinery plus complex-pair
carry arithmetic eats the theoretical conv win at these small shapes;
inductor cannot codegen complex ops, and the per-batch segmentation fights
CUDA graphs. A real-form rewrite is sketched but unmerged.

**Status:** prototype. The approach is quality-validated (the word-level
restructuring works as an architecture), speed-blocked at this scale.
Adoption requires either (a) a real-arithmetic rewrite + fused segmentation
kernel, or (b) larger models where the grid overhead amortizes. Not wired
into the trainer.

**Honest ledger entry:** the "binary clock" decomposition is the right
idea; its engineering cost exceeded the budget of this pass.

## Files

| File | Content |
|---|---|
| `models/V10_WSKAN.py` | `SegmentedWaveletLayer`, `WaveletStateKANLMV10` (correct, unoptimized) |
