# Writing models with `@rkppl`

`ReactiveKernelsPPL` is a small probabilistic-programming layer on top of
ReactiveKernels. It lives in the nested package
`packages/ReactiveKernelsPPL`, and ReactiveKernels core does not depend on it.
A model is an ordinary Julia block of `~` statements and assignments. The layer
turns it into a `@kernel` graph with a packed unconstrained parameter vector,
the transforms and their log-Jacobians, data preprocessing done inside the
graph, and the canonical `likelihood` / `prior` / `log_jacobian` / `posterior`
nodes.

The same language is what BayesianRegressionModels' `RKBRMI` backend emits for
an `@brm` formula. Hand-written and BRM-emitted models go through one pipeline:

```
@rkppl block ─ lower_rkppl ─▶ StructuralPlan ─ bind_data ─▶ bound plan
   ─ build_kernel ─▶ (; spec, layout) ─ prepare_query / prepare_sampler
```

Every example on this page runs at docs-build time, exactly as displayed. The
canonical programs come verbatim from the package's corpus
(`packages/ReactiveKernelsPPL/test/corpus/`), which also pins how each one
lowers.

## A first model

Bind data by keyword when calling the model; that lowers and binds in one step.
`build_kernel` generates the program, and `prepare_query(built, plan, :sampler)`
returns the log-posterior over the packed unconstrained coordinates.

```@eval
Main.ReactiveKernelsDocs.render_rkppl_corpus_example("01_gaussian.jl", :rkppl_first_model; preamble = "using ReactiveKernelsPPL")
```

This is the `@kernel` program `build_kernel` generates for the model above —
the graph that the sampler cut is prepared from:

```@eval
Main.ReactiveKernelsDocs.render_rkppl_kernel_program("01_gaussian.jl")
```

## Every parameter is declared

Declarations are strict: every parameter needs an explicit prior statement. A
name that is not a data column, a definition, or a declared parameter is an
error, so a typo can never become a parameter:

```@eval
Main.ReactiveKernelsDocs.render_rkppl_strict_error()
```

Coefficient vectors are declared the same way, sized by their design matrix:
`b[axes(X, 2)] .~ Normal.(0, 1)`.

## Julia semantics, written out

- Use Distributions.jl constructors (`Normal`, `Exponential`, `Gamma`, …).
- Broadcasting is explicit: `mu = a .+ b .* x`.
- A vector response uses the dotted tilde, `y .~ Normal.(mu, sigma)`. A plain
  `y ~ Normal(...)` on a data vector is rejected.
- A `~` whose left-hand side is a bound data column is an observation; every
  other `~` declares a parameter.
- Single assignment, no `if`, no `target +=`. Loops are written as
  `@plate` cells or `@scan` recurrences (see [Plates](#Plates)).

## Factor levels

`c[levels(g)] .~ Normal.(0, 2)` declares one coefficient per level of `g`, and
`c[g]` gathers them per observation. Levels are full rank, so an intercept plus
a full-cover factor is rejected as unidentified. Offsets are plain data
summands.

```@eval
Main.ReactiveKernelsDocs.render_rkppl_corpus_example("10_levels_prior.jl", :rkppl_levels)
```

## Group-level effects

Varying effects are library submodels. Each body states its priors and returns
an array that the use site reads with ordinary Julia indexing. The default
scale prior is `HalfNormal(1)`; correlated margins use `LKJCholesky(K, 1.0)`.
`K` is a literal, at least two for a correlated entry.

| Statement | Returned value | Observation-level read |
|---|---|---|
| `b ~ varying_coefs(g)` | One coefficient per group | `b[g]`, or slope `x .* b[g]` |
| `b ~ varying_coefs_correlated(g, K)` | A groups × K matrix | `b[g, 1] .+ x .* b[g, 2]` |
| `b ~ varying_coefs_centered(g)` | One directly sampled coefficient per group | `b[g]` |
| `b ~ varying_coefs_centered_correlated(g, K)` | Directly sampled multivariate rows | `b[g, 1] .+ x .* b[g, 2]` |
| `u ~ varying_stratified(g, s)` | One value per observation, with a scale per stratum | `u`, or slope `x .* u` |
| `r ~ varying_stratified_correlated(g, s, K)` | One K-component row per observation, with scales and a correlation factor per stratum | `r[:, 1] .+ x .* r[:, 2]` |

These are the actual library definitions, read from the loaded submodels:

```@eval
Main.ReactiveKernelsDocs.render_rkppl_varying_definitions()
```

To use a different prior, write the body at the use site and change its prior
statement. With the priors unchanged, the library call and the written body
lower to the same plan. Draw names carry the use-site prefix: `b_sd`, `b_z`,
and, for correlated margins, `b_L`. Centered entries expose `b_c` instead of
`b_z`; their coordinates are the coefficients themselves, so their densities
at a packed point differ from the non-centered entries.

```@eval
Main.ReactiveKernelsDocs.render_rkppl_corpus_example("27_varying_slope_lib.jl", :rkppl_varying_slope)
```

The stratified correlated entry returns rows already aligned with the
observations. Its draws contain S×K scales, a K×K×S stack of factors, and J×K
standard-normal coordinates, where S and J count the sorted distinct strata
and groups. Read stratum k's factor as `r_L[:, :, k]`. The returned rows
currently support literal column reads; passing that result whole to a
function or reading it by row is not supported yet.

```@eval
Main.ReactiveKernelsDocs.render_rkppl_corpus_example("65_stratified_lib.jl", :rkppl_stratified)
```

The stratified array cells run natively and compile under Reactant, including
reverse-mode gradients. Centered multivariate row priors currently run
natively; their Reactant density lowering remains unsupported. The older
`varying_draws` / `varying_effect` statements still lower while their callers
migrate to these library bodies.

Multi-membership uses the union of the membership columns as one data-only
definition, `gg = vcat(g1, g2)`. `levels(gg)` then sizes one shared set of
coefficients. Each observation weights its gathers; the definition runs once
at binding:

```@eval
Main.ReactiveKernelsDocs.render_rkppl_corpus_example("62_mm_intercept_lib.jl", :rkppl_membership)
```

## Declared arrays as values

`z[levels(g), 1:K] .~ Normal.(0, 1)` declares a groups × K matrix;
`L ~ LKJCholesky(K, eta)` declares a lower-triangular matrix. Sized vectors
(`sd[1:K] .~ HalfNormal.(1)`) and multivariate rows
(`eachrow(c[levels(g), 1:K]) .~ MvNormalCholesky(zeros(K), F)`) are also plain
Julia values. Their priors are explicit, and their dimensions determine their
packed coordinates.

Array-valued definitions retain known axes: `b = z * (sd .* L)'` is groups ×
K, so `b[g, 1]` gathers one margin per observation. Positional reads such as
`L[2, 1]` and `M[:, 1]` remain ordinary Julia reads. A submodel's returned
array follows the same rule. A data-only definition can size a declared array
through `levels(gg)`, as in the multi-membership example above.

Gathering by a derived column, or from a definition whose axes the layer cannot
derive (such as a module function's result), is not supported yet. Pass a
declared array whole to a model-level Julia function when it needs the full
value; a bare array combined directly with observation data must first be
indexed to the observation axis.

## Design matrices

Bind the matrix once with `hcat` (the `1` is the intercept column), use it only
as `X * b`, and size the coefficients with an axes prior. Scalar prior arguments
are shared across elements; literal vectors give one value per element.

```@eval
Main.ReactiveKernelsDocs.render_rkppl_corpus_example("46_matrix_gaussian.jl", :rkppl_matrix)
```

For spline and approximate GP terms, compute a data-side basis and use a
library submodel whose body states every prior. See [Smooths and HSGPs with
`@rkppl`](rkppl-smooths.md) for tensor, periodic and grouped variants.

## Plates

`@plate for i in R … end` writes a loop whose cells each mean one iteration
of that Julia loop. Scalar arithmetic needs no dots; vector and matrix
intermediates use ordinary Julia broadcasts and matrix products.
Shapes come from named data (the range, a data index column), never from a
separate size argument. A cell holds observations, per-cell latents, per-cell
submodel calls and cell locals. The whole loop lowers at once, exactly like its
broadcast spelling.

```@eval
Main.ReactiveKernelsDocs.render_rkppl_corpus_example("99_plate_32_gaussian.jl", :rkppl_plate)
```

The stratified library body above uses both per-level and per-observation
cells. `L[k] ~ LKJCholesky(K, eta)` inside a plate over `levels(s)` declares
one factor per stratum. An observation cell can then read `sd[s[i], :]`,
`L[s[i]]` and `z[g[i], :]`, compute a matrix-vector product, and name a
scalar output with `r[i] = ...` or a row with `b[i, 1:K] = ...`. Shared arrays
enter the generated RK plates through `Ref`; increasing observations or
strata does not multiply their compiled cell regions.

A per-level row statement, `b[j, :] ~ MvNormalCholesky(zeros(2), F)` inside a
plate over `levels(g)`, is equivalent to
`eachrow(b[levels(g), 1:2]) .~ MvNormalCholesky(zeros(2), F)`.
`b[j, 1:2]` and `MvNormal` also work. Per-level prior arguments are shared;
arguments varying by level and deterministic per-level assignments are not
built yet. In an array cell, an observation must read a named per-index output
(`y[i] ~ Normal(r[i], sigma)`), rather than a cell local directly.

`@scan begin … end` writes a sequential recurrence, such as an AR(1) state
(corpus `38_scan_ar.jl`).

## Submodels

`@rkppl name(args...) = begin … end` defines a reusable block. Using it,
`sigma ~ half_scale(1.0)`, expands it inline under the left-hand side's name, so
it lowers exactly like the hand-inlined program. A submodel whose result is a
response pointer is used as an observation stream: `y ~ stream(x, g)`.
`Base.merge(model, override)` replaces or appends statements by name.

```@eval
Main.ReactiveKernelsDocs.render_rkppl_submodel_example()
```

## Querying and fitting

`prepare_sampler` adds a reverse-mode gradient over the same cut. The
gradient's executable authority is the package test suite, so it is shown here
as source:

```julia
using Enzyme
using DifferentiationInterface: AutoEnzyme

q = prepare_sampler(built, plan, u; backend = AutoEnzyme(; mode = Enzyme.Reverse))
q(u)                                   # log posterior (world-age safe)
g = similar(u)
value, _ = sampler_value_and_gradient!(q, g, u)
prepare_query(built, plan, :likelihood)  # presets: :sampler | :likelihood | :prior | :log_jacobian
constrain(built.layout, u)               # named constrained values
restore_draws(built.layout, U)           # U: layout.total × draws
```

- `:sampler` is the posterior preset; `:posterior` is the generated node's
  name, not a preset.
- Explicit parameter declarations keep their authored names. For
  `a ~ Normal(0, 1); b ~ Normal(0, 2); mu = a .+ b .* x`, the coordinates
  are `a`, `b` and the constrained values are `nt.a`, `nt.b`.
  A level-sized declaration `c[levels(g)]` gives `c.1`, `c.2`, … and `nt.c`.
  Adding another reader preserves those names and the single declared prior.
  Explicit whole-predictor R2D2 and Horseshoe constructs retain their own
  coefficient layouts. Read `coordinate_names(built.layout)` and `constrain`;
  rebuild old prepared models and packed-draw mappings when migrating.
- A kernel returned by `prepare_query` closes over code generated at build time.
  Call it from top level or through `Base.invokelatest`; `SamplerQuery` calls
  already carry that barrier.
- Spline (`s`, `t2`) bases come from host LAPACK eigenvectors, which fix each
  column only up to sign. The layer flips every penalized column so that its
  first significant entry is positive, so spline coordinates mean the same thing
  on every machine.

## Programs emitted by BRM

`RKBRMI(brmi)` lowers an `@brm` formula to this same surface. The emitted
program is ordinary `@rkppl` source: shared submodel definitions such as
`popefs_normal_i_c_r(...)` for an intercept + slope + group effect, plus a main
block. For plain population, group-level and GLM models, evaluating the printed
source by hand reproduces the backend's density exactly. Some BRM features
(ordinal discrimination, per-threshold coefficients, missing-data `mi()`) are
still patched into the lowered plan after lowering, so for those the printed
source is not yet the whole model. See the BRM documentation for the formula
side.

## Debugging

- `kernel_expr(plan, built.layout)` returns the generated `@kernel` program
  shown above.
- `packages/ReactiveKernelsPPL/report/transpile_report.jl --surface model.jl
  --data data.jl` writes a markdown report of the real pipeline: the plan, the
  generated program, and the posterior at a probe point. With
  `--artifact model.jls` it reports on a serialized BRM emission instead.
- `SurfaceLoweringError` means the spelling is not admitted, and its message
  names the fix. `ContractValidationError` comes from the IR or binding layer.
