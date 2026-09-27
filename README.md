# Wavelet-State-KAN

A Kolmogorov-Arnold Network where each edge wavelet is the impulse response of
a stable state-space model (SSM). One parameterization, two modes: closed-form
wavelet edges (static, WavKAN-style) and recurrent SSM edges (Mamba-style scan).

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

## Structure

```
models/
  V1_WSKAN.py        # V1 architecture (edge = SSM impulse-response wavelet)
  V1_LM.py           # V1 byte-level LM wrapper (~94k params, tied head)
  V1_README.md       # V1 math, admissibility, stability, limitations
  V2_WSKAN.py        # V2 architecture (V1 + input/output gating, selective)
  V2_README.md       # V2 math, design rationale, parameter budget
  V3_WSKAN.py        # V3 architecture (fully selective, chunked SSD scan)
  V3_README.md       # V3 math: content-warped wavelet transform
  V4_WSKAN.py        # V4 architecture (freed sigma, per-mode dt, ZOH write)
  V4_README.md       # V4 math and rationale
  V5_WSKAN.py        # V5 architecture (short conv + geometric ladder; not adopted)
  V5_README.md       # V5 design and ablation flags
  V6_WSKAN.py        # V6 architecture (factorized timescales; ADOPTED go-forward)
  V6_README.md       # V6 design: V4 capability on V3's analyzable skeleton
  V7_WSKAN.py        # V7: interpretable-by-construction variants (feature-B/C adopted)
  V7_README.md       # V7 design and neutrality-test protocol
experiments/
  V6_FINAL_LANGUAGE_ACCOUNT.md   # CAPSTONE: complete mathematical account of how V6 learned language
  V1_sanity_check.py          # 1D fitting sanity check (static mode)
  V1_train_tinystories_lm.py  # shared TinyStories trainer (--model wskan|wskan2|mamba2)
  V1_TINYSTORIES_REPORT.md    # 20k-step training report
  V1_TINYSTORIES_500K_REPORT.md # 500k-step training report + NaN incident fix
  V1_MAMBA_COMPARISON_REPORT.md # 500k matched-param Mamba-2 comparison (Mamba wins)
  V2_COMPARISON_100K_REPORT.md  # 100k three-way: V2 closes 76% of the Mamba gap
  V3_COMPARISON_REPORT.md       # 100k four-way: V3 reaches parity with Mamba-2
  V3_MULTISEED_ABLATION_REPORT.md # 3-seed: wavelet contribution real; V3 >= Mamba-2
  V3_INTERPRETABILITY_REPORT.md   # learned edge functions: shapes, spectra, sparsity
  V3_ULTRACHAT_REPORT.md          # 3-seed UltraChat: wavelet gap grows; memory still local
  V3_10M_PROBE_REPORT.md          # 10M probe: scale unlocks longer memory; undertrained ranking
  V4_ULTRACHAT_REPORT.md          # V4 3-seed: freeing timescales is neutral at 100k
  V4_MEMORY_ANALYSIS.md           # V4 grows dormant long-memory modes (up to 168 tok)
  V4_MAMBA_MEMORY_COMPARISON.md   # Mamba-2 has the same dormant tail; both fail recall
  V5_ULTRACHAT_REPORT.md          # V5: conv+ladder neutral-to-negative; 100k iteration closed
  V6_ULTRACHAT_REPORT.md          # V6: V4 perf + V3 analyzability + lowest variance; adopted
  V6_QUALITATIVE_SAMPLES.txt      # side-by-side generations (V6 vs Mamba-2)
  V6_deep_analysis.py             # mathematical dissection of the trained V6
  V6_LANGUAGE_LEARNING_ANALYSIS.md # how it learned language: word-clocked filter bank
  V6_CAUSAL_CLOCK_REPORT.md       # causal test: boundary-Delta removal hurts word transitions 16x vs control
  V6_CAUSAL_CLOCK_FULL_REPORT.md  # necessity replicated + sufficiency (false ticks create word transitions), 3 seeds
  V7_ULTRACHAT_REPORT.md          # price of interpretability: B/C lookup free at rank32; lowrank g rejected
  V7_deep_interpret.py            # deep battery: tables, atlas, channels, clock drivers
  V7BC_DEEP_INTERPRETABILITY_REPORT.md # four debts closed: structural-byte routing, kernel families, channel dictionary
  V7_final_interpret.py            # final battery: residual path, circuit, W_z, embedding, logit attribution
  V7C_FINAL_INTERPRETABILITY_REPORT.md # last five debts: identity addressing, routing, exact attribution
  V7_causal_dictionaries.py        # causal battery: tables, clusters, hub, families + turn test + mode attribution
  V7D_CAUSAL_DICTIONARIES_REPORT.md # dictionaries now causal; turn hypothesis refuted; frien->d = mode (i12,k1)
  baselines.py                     # tf / gated-conv / lstm baselines
  BENCH_orchestrate.py + BENCH_worker.sh + BENCH_aggregate.py  # 225-run matrix tooling
  W7BC_CROSS_FAMILY_REPORT.md        # 5 families x 5 sizes x 3 datasets x 3 seeds + interpretability axis
  V3_interpret.py                 # edge-function extraction + figures
  W11_all_sizes_analysis.py       # all-sizes anatomy battery (5 tiers x 5 seeds; Appendix B)
  W11_hand_simulation.py          # deriving the next token by hand (Appendix D)
  W11_round5_probes.py            # usefulness ladder (nonlinear probes) + 10M trigram identification
  W11_excess_structure.py         # origin of the organization: interventions, init-vs-trained, dynamics
  W11_ablation_retrain.py         # task-requirement test: retrain with structure impossible
  W11_round6_probes.py            # multi-seed interventions, mamba2 skip-vs-reset, antipodal profile metric
  W11_round7_probes.py            # antipodal permutation test, 5-seed convergence, paired baselines
  W11_dynamics_20k.py             # emergence trajectories extended to 20k steps
  W11_basin_origin.py             # why the organized basin is entered early (gradient anatomy, dt-init scale)
  W11_benchmark_demo.py           # post-hoc attribution methods scored against exact ground truth
  W11_EXCESS_STRUCTURE_REPORT.md  # structure is solution-load-bearing, not task-required; origin decomposed
checkpoints/
  {model}_{dataset}_{size}_s{seed}/   # benchmark matrix tier (225 runs)
  {model}_{dataset}_interp_s{seed}/   # canonical interpretation tier
  {model}_..._interp_500k/            # early long runs (single seed)
```
The `interp` tier holds the checkpoints that interpretation reports are
based on (protected from overwrites). The `*_100k_*` names may be reused by
benchmark reruns; analysis should always read the `interp` tier.
Note: interpretation reports written before this tier existed reference
`*_100k_*` paths — the corresponding canonical artifacts are the same-named
`*_interp_*` ones (numbers reproduced within seed/GPU nondeterminism noise;
per-seed figures in those reports remain the historical record of their
original runs).

Architecture files are versioned as `V<version>_<name>.py` with a matching
`V<version>_README.md`; see each version's README for its math and caveats.

## Quickstart

```bash
.venv/bin/python experiments/V1_sanity_check.py
```

## Key reports (start here)
- **`experiments/W11_FINAL_INTERPRETATION_REPORT.md` — THE definitive report (3-epoch campaign, 5 seeds, full interpretation; performance as footnote)**
- `experiments/W7BC_FINAL_REPORT.md` — wskan7bc consolidated report (matrix era)
- `experiments/W7BC_HOW_IT_SPEAKS.md` — the complete mathematical mechanism (canonical numbers)
- `experiments/W7BC_FINAL_CLEAN_REPORT.md` — canonical comparison numbers (fresh interp tier)
- `experiments/W7BC_10M_INTERPRETABILITY_REPORT.md` — 10M×3-seed replication + scale laws
- `experiments/V6_FINAL_LANGUAGE_ACCOUNT.md` — how the model learned language (capstone)
- `experiments/W7BC_GENERATION_ACCOUNT.md` — how it speaks (production-side)
- `experiments/W7BC_CROSS_FAMILY_REPORT.md` — 225-run cross-family benchmark
- `experiments/V7*_REPORT.md`, `V1_*`, `V2_*`, `V3_*`, `V4_*`, `V5_*` — version history (incident logs + narratives)
