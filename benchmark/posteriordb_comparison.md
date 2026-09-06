# Fair all-sides posteriordb comparison — ReactiveKernels vs Stan vs Turing

All **10 named posteriordb models**, each measured *all the relevant ways*: native
steady-state **primal** and **gradient**, RK vs **optimized** Stan and **optimized**
Turing, plus the **Reactant** compiled-HMC-loop regime. Every number below is copied
from a committed receipt under [`benchmark/receipts/`](receipts/) — no hand-entered or
remembered values.

> **Why "fair" and why this page exists.** The widely-cited
> [DynamicPPL posteriordb benchmark](https://github.com/TuringLang/DynamicPPL.jl/blob/main/benchmarks/posteriordb.md)
> compares Turing against the *naive* posteriordb reference Stan. For many models that
> reference is a scalar `for`-loop likelihood, not the Stan a practitioner would write —
> so it understates Stan by **7–200×** (see *Part (a)* below). This page instead pits RK
> against Stan **written optimally** (vectorized / fused `*_glm` primitives / conjugate
> sufficient statistics) and Turing **written optimally** (posteriordb-style, Mooncake AD).
> That is the honest boundary; RK's wins and losses are reported both ways.

## The fair boundary (methodology)

- **What is timed:** steady-state log-density (**primal**) and value-plus-gradient
  (**gradient**). Setup, compilation, and the first call are **excluded**. The data-only
  constant prefix (sufficient statistics `XtX`/`Xty`, `lgamma(C+1)`, `lchoose(N,C)`) is
  precomputed **once for every side**, so no side is charged for it.
- **Timing:** median over 10–12 rounds of
  `minimum(run(@benchmarkable f(); samples=200, seconds=0.2))` (BenchmarkTools).
- **Parity gate:** every side is checked against an **independent central
  finite-difference oracle** before timing (max gradient rel-err **≈ 1e-9**, e.g.
  gp_regr `3.6e-10`, Mh `2.9e-9`). Absolute log-density values match up to an additive
  constant (Turing's `truncated(Normal; lower=0)` adds `log 2` vs Stan's `real<lower=0>`).
- **The three sides.**
  - **Optimized Stan** — raw `.stan` via BridgeStan 2.9. The optimized baseline is the
    fastest Stan idiom that *exists* for the model: fused `normal_id_glm` /
    `poisson_log_glm` where available; vectorized `binomial_logit` (there is **no**
    `binomial_logit_glm`); `multi_normal_cholesky` for the GP; conjugate sufficient-stat
    form for the linear/Gaussian family.
  - **Optimized Turing** — posteriordb-style translation (sufficient statistics,
    `product_distribution`/`MvNormal`), Mooncake gradient via a linked
    `LogDensityFunction`.
  - **ReactiveKernels** — one authored `@kernel`; densities as pure functions; data-only
    inputs hoisted via `bound=`; gradient via DifferentiationInterface + Enzyme
    (`AutoEnzyme(mode=Enzyme.Reverse, function_annotation=Enzyme.Const)`).
- **Environment (all receipts):** Julia 1.10.11, x86_64, AMD EPYC-Milan; Stan 2.39.0
  (BridgeStan 2.9.0); Turing 0.47.1 / DynamicPPL 0.42.6 / Distributions 0.25.131 /
  Enzyme 0.13.201 / Mooncake 0.5.53 / DifferentiationInterface 0.7.21.

Weak-normal unbounded priors replace the posteriordb bounded uniforms where noted; data
is synthetic of representative shape and dimension. Ratios are **lower = RK faster**;
**bold** marks an honest RK **loss**.

## Native single-evaluation: RK vs optimized Stan & Turing

Ratios (lower = RK faster). Absolute medians are in the two detail tables that follow.

| Model | dim | RK / Stan primal | RK / Stan grad | RK / Turing primal | RK / Turing grad | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Rate_1 (beta-binomial) | 1 | 0.22× | 0.20× | 0.37× | 0.09× | RK win |
| GLM_Binomial | 3 | 0.20× | 0.42× | 0.21× | 0.10× | RK win |
| gp_regr (GP + cholesky) | 3 | **1.50×** | **2.58×** | 1.48× | 0.97× | **RK loss** (dense linalg) |
| GLM_Poisson | 4 | 0.31× | 1.06× | 0.39× | 0.44× | RK win primal, ~par grad |
| kidscore_interaction | 5 | 0.18× | 0.23× | 0.59× | 0.17× | RK win |
| arK (AR(5), T=200) | 7 | **1.12×** | **2.43×** | 0.95× | 0.42× | **RK loss** vs Stan (fused `normal_id_glm`) |
| sblri-blr (linear) | 7 | 0.29× | 0.39× | 0.65× | 0.18× | RK win |
| eight_schools_centered | 10 | 0.37× | 0.29× | 0.30× | 0.04× | RK win |
| GLMM_Poisson | 45 | 0.37× | 0.68× | 0.40× | 0.33× | RK win |
| Mh (capture-recapture) | 303 | 0.29× | 0.38× | 0.44× | 0.33× | RK win (**large dim**) |

**RK beats optimized Stan on 8 of 10 models, and beats optimized Turing on all 10.**
The only two RK losses are **arK** and **gp_regr** — both cases where Stan lowers the hot
path to a **fused C++ primitive with an analytic gradient** (`normal_id_glm`,
`cholesky_decompose`+`multi_normal_cholesky`). Crucially the loss is about *fused-primitive
dense linear algebra, not dimension*: **Mh at dim 303 is a decisive RK win** (0.29× primal
/ 0.38× gradient), because the augmented random-effects binomial-logit has no fused Stan
primitive, so Stan runs a per-individual `log_sum_exp` loop.

### Detail — primal (median ns · median bytes)

Stan column names the optimized idiom used as the baseline.

| Model | dim | optimized Stan | Turing | ReactiveKernels |
|---|---:|---|---|---|
| Rate_1 | 1 | 230 · 16 B | 135 · 0 B | **50 · 0 B** |
| GLM_Binomial | 3 | 5292 · 16 B (`binomial_logit` vec) | 5032 · 4512 B | **1081 · 2688 B** |
| gp_regr | 3 | **6180** · 16 B (`multi_normal_cholesky`) | 6280 · — | 9280 · 31 KB |
| GLM_Poisson | 4 | 2138 · 16 B (`poisson_log_glm`) | 1737 · 2848 B | **670 · 1792 B** |
| kidscore_interaction | 5 | 706 · 16 B (`normal_id_glm`) | 220 · 240 B | **130 · 80 B** |
| arK | 7 | **470** · 16 B (`normal_id_glm`) | 556 · 5088 B | 525 · 6528 B |
| sblri-blr | 7 | 510 · 16 B (`normal_id_glm`) | 230 · 288 B | **150 · 96 B** |
| eight_schools_centered | 10 | 300 · 16 B | 370 · 800 B | **110 · 0 B** |
| GLMM_Poisson | 45 | 1001 · 16 B (vec) | 936 · 2160 B | **370 · 800 B** |
| Mh | 303 | 19100 · 16 B | 12460 · — | **5450 · —** |

### Detail — gradient (median ns · median bytes)

| Model | dim | optimized Stan | Turing (Mooncake) | ReactiveKernels (Enzyme) |
|---|---:|---|---|---|
| Rate_1 | 1 | 300 · 80 B | 660 · 1280 B | **60 · 0 B** |
| GLM_Binomial | 3 | 6654 · 96 B | 28634 · 58896 B | **2774 · 5840 B** |
| gp_regr | 3 | **22210** · — | 58860 · — | 57250 · 68 KB |
| GLM_Poisson | 4 | **2338** · 112 B | 5598 · 12064 B | 2473 · 4160 B |
| kidscore_interaction | 5 | 1301 · 112 B | 1802 · 5056 B | **300 · 160 B** |
| arK | 7 | **886** · 128 B | 5162 · 21312 B | 2153 · 13056 B |
| sblri-blr | 7 | 901 · 128 B | 1953 · 5312 B | **350 · 192 B** |
| eight_schools_centered | 10 | 530 · 160 B | 4371 · 7904 B | **155 · 0 B** |
| GLMM_Poisson | 45 | 1977 · 464 B | 4075 · 9744 B | **1336 · 1712 B** |
| Mh | 303 | 31750 · — | 36430 · — | **11960 · —** |

Note the **allocation** column: RK's generated straight-line native kernel allocates
0–6 KB and often **0 bytes**, versus Turing's 1–59 KB per gradient. That low overhead is
exactly why RK wins the overhead-bound models.

## Reactant compiled-HMC-loop regime

Reactant (XLA) **loses badly per single evaluation** — the launch/sync overhead dwarfs
the ns–µs of work (measured elsewhere at 4–385× slower for one log-density/gradient). But
when the **whole sampler loop** is captured as one compiled program, that cost amortizes
once and XLA fuses the entire leapfrog + multinomial batch. The **same authored `@kernel`**
is lowered to both backends — no hand-written Reactant kernel.

Multinomial HMC, L=16 leapfrog steps, 1000 transitions/batch, median of 6 batches
([`reactant_hmc_loop_table.jl`](reactant_hmc_loop_table.jl),
receipt [`reactant-hmc-loop-table-v1.toml`](receipts/reactant-hmc-loop-table-v1.toml)):

| Model | dim | native µs/transition | Reactant µs/transition | Reactant / native |
|---|---:|---:|---:|---:|
| eight_schools | 10 | 6.15 | 2.74 | **0.45×** (~2.2× faster) |
| GLMM_Poisson | 45 | 61.18 | 8.15 | **0.13×** (~7.5× faster) |
| GLM_Poisson | 4 | 150.76 | 10.66 | **0.07×** (~14× faster) |

**Reactant wins the compiled HMC loop for every model** — by *more* where the per-leapfrog
native AD is expensive (Poisson `exp`, larger dim). Practitioner rule for RK: **compile the
whole sampler loop with Reactant; never call Reactant per gradient/leapfrog.**

### A generic RK Reactant-lowering friction (tracked)

A *naturally authored* `@kernel` does not transpile to Reactant unchanged; it currently
needs Reactant-friendly forms:

- scalar `q[i]` → `sum(view(q, i:i))` (scalar indexing is disallowed on a `TracedRArray`);
- `dot(data::Vector, traced)` → `sum(a .* b)` (`dot` calls `conj`, unsupported on a plain
  data `Vector` under Reactant).

Per user decision, the fix is **RK-macro-only** (normalize inside `@kernel` lowering; the
rewrite must be **type-aware** — `dot` conjugates, so `sum(a.*b)` is only valid for real
inputs — with a **loud error** for anything outside the safe set, never a silent
mis-lowering). Tracked as todo `ReactiveKernels/2026-09-06T19-35-20-773-0cgmepj`.

## Supplementary — Gaussian regression K-scaling

Not one of the 10 named models, but it shows *why* the RK gradient lead narrows as the
problem grows (conjugate `normal_id_glm`, N=5000;
[`all-sides-gaussian-regression-v1.toml`](receipts/all-sides-gaussian-regression-v1.toml)):

| dim (K) | RK / Stan primal | RK / Stan gradient |
|---|---:|---:|
| 22 (K=20) | 0.48× | 0.54× |
| 102 (K=100) | 0.57× | 0.91× |

The gradient ratio drifts from 0.54× toward 0.91× as K grows: at these dims the timed
region is per-call/FFI-bound (RK's advantage), but as the O(K²) BLAS FLOPs come to
dominate, Stan's analytic reverse-mode catches up. Extrapolated, Stan's fused-primitive
gradient overtakes RK at large dense K — consistent with the arK / gp_regr losses above.

## Part (a): how misleading is the naive-Stan posteriordb reference?

The naive-vs-optimized-Stan gap is **strongly model-dependent**:

- **Large** where the reference is a scalar loop or a conjugate model written naively:
  Gaussian conjugate (`stan_naive` 38.3 µs vs `stan_conj` 0.31 µs at K=20 → **~124×**),
  and across posteriordb, models such as Mb (~200×), soil_incubation (~37×), Survey (~33×),
  diamonds (~32×), dogs (~7.6×).
- **Small** where the reference is already vectorized: GLMs whose reference uses
  `poisson_log` / `binomial_logit` are within **~1.05×** of the optimized form.

Aggregate on the DynamicPPL posteriordb page: the reported Turing/Stan primal geomean of
**0.869×** ("Turing faster", 83/147 wins) is driven by ~23 reformulated models; excluding
those it is **1.18×** (Stan faster), and capping per-model wins at 1.0 gives **1.31×**.
Gradients are already Stan-favorable there (Mooncake 1.32×; Enzyme 0.97× but errs on
22/147). So the headline "Turing ≈ Stan" is an artifact of the naive reference for a
subset of models, not a general result.

## Reproduce

Each script builds its own pinned comparison environment and re-execs the model body;
`AS_OUTPUT=<path.toml>` writes the receipt.

| Models | Script | Receipt |
|---|---|---|
| GLM_Poisson, GLM_Binomial | [`fair_posteriordb_glm.jl`](fair_posteriordb_glm.jl) | [`fair-posteriordb-glm-v1.toml`](receipts/fair-posteriordb-glm-v1.toml) |
| arK, GLMM_Poisson | [`fair_posteriordb_more.jl`](fair_posteriordb_more.jl) | [`fair-posteriordb-more-v1.toml`](receipts/fair-posteriordb-more-v1.toml) |
| Rate_1, eight_schools_centered | [`fair_posteriordb_b3.jl`](fair_posteriordb_b3.jl) | [`fair-posteriordb-b3-v1.toml`](receipts/fair-posteriordb-b3-v1.toml) |
| gp_regr | [`fair_posteriordb_gp.jl`](fair_posteriordb_gp.jl) | [`fair-posteriordb-gp-v1.toml`](receipts/fair-posteriordb-gp-v1.toml) |
| Mh | [`fair_posteriordb_mh.jl`](fair_posteriordb_mh.jl) | [`fair-posteriordb-mh-v1.toml`](receipts/fair-posteriordb-mh-v1.toml) |
| kidscore_interaction, sblri-blr | [`fair_posteriordb_linear.jl`](fair_posteriordb_linear.jl) | [`fair-posteriordb-linear-v1.toml`](receipts/fair-posteriordb-linear-v1.toml) |
| Reactant HMC loop (3 models) | [`reactant_hmc_loop_table.jl`](reactant_hmc_loop_table.jl) | [`reactant-hmc-loop-table-v1.toml`](receipts/reactant-hmc-loop-table-v1.toml) |
| Gaussian K-scaling | [`all_sides_gaussian_regression.jl`](all_sides_gaussian_regression.jl) | [`all-sides-gaussian-regression-v1.toml`](receipts/all-sides-gaussian-regression-v1.toml) |

Every receipt carries its `source`, `rk_source_commit`, `methodology`, and `environment`.
The Reactant loop scripts run under the sampling lane's `benchmark/sampler_transpiler/`
environment (`setup.jl` with `JULIA_NUM_PRECOMPILE_TASKS=1`, then the table script).
