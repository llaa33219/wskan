# Paper Fact-Check Report: `paper/main.tex`

Date: 2026-10-02 (audit + re-verification + fixes applied the same day).
Auditor: Sisyphus (orchestrated: 4 verification agents + direct code audit +
GPU re-measurement of every disputed number). Scope: every quantitative
claim in the paper vs. the repo's reports (`experiments/*.md`), result files
(`experiments/figures/*.json`), scripts, checkpoints, model code
(`models/`), and the 13 bibliography entries against arXiv. The GitHub URL
was excluded per the author's instruction.

**Status: all confirmed problems fixed in `paper/main.tex` (and two repo
files). This report records what was wrong, the re-measured truth, and the
fix applied.**

Re-measurement artifacts: `experiments/figures/factcheck_regen.json`
(produced by `experiments/factcheck_regen.py` and
`experiments/factcheck_edge_stats.py`, fp32, canonical checkpoints).

## A. Contradicted claims — FIXED in the paper

1. **"Mamba-2 leads in every benchmark cell" (was main.tex:125).**
   Re-running `CAMPAIGN_aggregate.py` on the 300 checkpoints: Mamba-2 leads
   11/15 cells. WSKAN-11 wins the 1m tier on all three datasets (0.5813 vs
   0.6385 TS, 0.9344 vs 1.0103 UC, 1.1101 vs 1.1700 WT); the transformer wins
   10m WikiText (0.9403 vs 0.9481) and 1m TinyStories. FIXED: main-text claim
   reworded; the omitted 10m-WikiText row (wskan11 1.023±.017 / real
   1.045±.019 / mamba2 0.948±.020 / tf **0.940±.015**) added to Appendix C;
   Appendix C intro reworded.

2. **"3-epoch budgets per (tier, dataset)" (Appendix E).** Measured from
   steps x tokens/step / train bytes: 0.26–2.99 epochs; only the 100k tier is
   ~3 epochs (the orchestrator docstring already said so). FIXED: Appendix E
   now states "~3-epoch budgets at the 100k tier (2.6–3.0); capacity-tiered
   caps elsewhere (0.3–2.0), matched across models".

3. **"WikiText-2" (Appendix E).** The loader reads `wikitext-103-raw-v1`.
   FIXED to WikiText-103.

4. **"+0.03–0.12 nats of CE" (intro + Limitations).** True range
   +0.021…+0.289 across tiers; 100k is below the floor, 1k/10k exceed the
   ceiling, 1m is negative (WSKAN wins). FIXED: "+0.02–0.12 at the 100k and
   10m tiers; +0.15–0.29 at 1k–10k" in both places.

## B. Numbers with no source — re-measured, then FIXED or confirmed

5. **frien→d decomposition.** Recomputed in fp32 on the canonical checkpoint
   (margin 8.4392, parts +0.829 / +5.321 / −5.306 / +0.879 / +7.504 /
   −0.789, sum 8.438). The paper's printed "unrounded" parts (+0.832, +5.342,
   …) summed to 8.462 — wrong digits. The per-mode list
   [−2.85, +5.64, −0.81, +1.25, +7.36, −3.09] is CORRECT (re-measured:
   −2.846/+5.638/−0.806/+1.250/+7.361/−3.094). The repo's conflicting 8.30
   (definitive report A.5) is the **bf16** model-forward value; the paper's
   stated protocol is "fp32 in analysis", so 8.44 stands. FIXED: parts
   corrected to the re-measured values; the "0.02 display gap" note replaced
   with the true statement (bf16 forward gives 8.30; all analysis is fp32).

6. **"|DC|/RMS per edge: median 0.12; 36–40% below 0.1".** No artifact
   existed. Re-measured on the canonical checkpoint: window-dependent —
   median 0.14 / 34% below 0.1 over the first 5 warped-time units; median
   0.007 / 100% over 2000 units. The kernels are essentially zero-mean at
   long horizon. FIXED: the paper now reports the windowed measurement with
   the horizon stated.

7. **"Boundary bytes hold 17% of the top-25 terms at every tier — exactly
   their base rate".** No artifact existed; re-measured over the 50
   truncation contexts per tier: 42.6% / 47.1% / 28.0% / 28.9% / 21.0%
   (base rates 24.2% / 24.2% / 19.9% / 19.9% / 19.9%). The claim was false;
   boundary positions are over-represented at small tiers and at base rate
   only at 10m. FIXED: §hand now reports the measured values and the
   corrected interpretation.

8. **"Mean top-1/top-2 margin grows 0.94 → 1.15 → 2.37" (Appendix D
   narrative).** No artifact existed, but re-measurement CONFIRMED it:
   0.943 / 1.149 / 2.369 (1k/10k/100k; 1m 3.571, 10m 3.759). No fix needed.

9. **Detectability bounds "0.011 / 0.028 / 0.015 nats".** No artifact; and
   inconsistent with the report's own per-seed deltas. Recomputed standard
   80%-power two-sided MDEs (n=3, df=2): **0.04 / 0.09 / 0.05 nats** (3–8%
   of CE). FIXED in §decision's audit paragraph and twice in Limitations.

10. **Effective rank "≈29 of 32 per mode".** That was the V7-era d=32 model.
    The canonical wskan11-100k (d=40) measures 34.6–36.7 of 40. FIXED to
    "≈35–37 of 40".

11. **"5.6×–400× across the five tiers".** Recomputed from raw per-seed JSON:
    753× / 383× / 19.3× / 5.7× / 9.0× (1k control ≈ 0). FIXED to
    "5.6×–750×" in all four locations.

## C. Smaller corrections — FIXED

12. Per-seed control list: 4th value +0.1247 printed as +0.13 → **+0.12**.
13. Duplicated sentence "word-ending behavior (sufficiency)." removed.
14. "72 of 75 paired cells" → **73 of 75** (regenerated); the
    "−0.04…−0.08 at 10k–100k" range corrected to "−0.02…−0.08 at 1k–100k".
15. "±0.03 nats" retrain bound → **0.031** (10k n1wide exceeds 0.03).
16. §exclusion text "R² ≈ 0.12–0.15" → **0.10–0.15** (matches its own table).
17. Mamba-2 gating numbers reworded as ratios ("drops to 0.13–0.65× the
    letter mean"; ZOH write "0.15–0.70× in the deep layers" — layer 0 goes
    the other way, 1.4×, now stated).
18. "parameter-matched (Appendix C)" → (Appendix E); Appendix C has no
    parameter counts.
19. "up to 10m parameters" → "≈10m" (10m tier is 10,153,996 params).
20. Reproduction map: `V7_causal_clock_full.py` does not exist → replaced
    with `W11_all_sizes_analysis.py` (5-seed battery) +
    `V6_causal_clock_full.py` (sufficiency, V6-era).
21. Bibliography `kan-lm-audit`: fabricated subtitle removed, authors added
    (Alves, F., Vicente, R. — the arXiv paper is real and the paper's
    specific claims about it verified exact); related-work sentence now
    carries the source's own caveat that the 99.9% fPCA compression is
    basis-imposed, not learned.
22. `W12_HANDCRAFT_REPORT.md`: length→boundary gain 18.0 → **14.0** (the
    value in the script that produced CE 5.45); the probe table was
    mislabeled (it showed logit(space), not the margin) → replaced with the
    true margins (−11.44 … +2.91). `W12_handcraft.py` now actually computes
    the static-skeleton ablation and unigram baseline it cited — re-run:
    full 5.4470 / static-only 7.5337 / unigram 3.0601, matching the report
    and the paper (5.45 / 7.53 / 3.06 / −2.08 nats).
23. Root README Structure section now lists V8–V11 (was stale at V1–V7).

## D. Verified clean (no change)

- **Model code = paper math**: recurrence, warped time, 8 named byte
  features, output equation, tied head, N=6, σ₀=0.5, ρ₀=1, ω₀=π·[1..6],
  Δt₀=0.05 (V3:68-103, V6:49, V7, V8, V11). Triton fused scan correct;
  V11_README's "fwd 7e-7, grads ≤4e-6" covers the paper's "≤4×10⁻⁶".
  (Minor undocumented detail: `_compute_dt` clamps at 1.0.)
- Appendix B table, Appendix C 5 original rows, Appendix D truncation table,
  exclusion probe table (30 values), benchmark demo table (12 values),
  surgery/retrain audit table, emergence-dynamics numbers, implicit-boundary
  test, 5-seed clamp +0.556±0.062, per-tier causal battery, Mamba-2
  (+0.28/+0.08; +0.19/+0.29) and transformer (+2.93/+4.75) numbers, "1.4× vs
  19×" (same metric on both sides: d_wi(intervention)/d_wi(control)).
- 12 of 13 bibliography entries exact (ID, title, authors, year).
- 300/300 campaign checkpoints present; hand-simulation 3.8e-6 replication;
  1e-4 decomposition assert present at every use.

## E. Residual known issues (documented, not blocking)

- Several single-run measurements (implicit-boundary test, both
  cross-architecture causal tests) carry no seed spread; the paper presents
  them beside 5-seed results. Consider marking n=1 explicitly.
- The definitive report (`W11_FINAL_INTERPRETATION_REPORT.md` A.5) still
  shows the bf16 frien→d decomposition (8.30); it predates the fp32 protocol
  and is the historical record — the paper now uses the fp32 values.
- No LaTeX toolchain on this machine; the PDF was not recompiled after the
  edits. `paper/main.pdf` is stale until rebuilt.
- Checkpoints remain `.pt`; convert to `.safetensors` before any Hugging
  Face release.
