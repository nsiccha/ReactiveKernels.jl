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

Every example panel on this page runs at docs-build time, exactly as displayed. The
canonical programs come verbatim from the package's corpus
(`packages/ReactiveKernelsPPL/test/corpus/`), which also pins how each one
lowers.

## A first model

Bind inputs by keyword, then observe responses with `model(; x) | (; y)`.
`condition(model(; x); y)` is the same operation. A call keyword naming a
declared variable pins it: `model(; x, sigma = 0.3) | (; y)` removes the
`sigma` declaration and its density. Observing it instead,
`model(; x) | (; y, sigma = 0.3)`, keeps its density as likelihood.
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

## Parameter priors

Scalar parameter priors include `Normal`, `Cauchy`, `Exponential`, `Gamma`,
`LogNormal`, `Beta`, `InverseGamma`, `StudentT`, `Laplace`, `Logistic`,
`Uniform` and `Weibull`. Arguments can read data, sampled parameters and
ordinary definitions; scalar expressions such as `1 + exp(a)` are values too.

`x ~ truncated(D, lo, hi)` uses the normalized Distributions.jl density for
any of these univariate families. Bounds may be literals, model-level data,
sampled values or scalar expressions; use `-Inf` or `Inf` for an open end.
The transform uses the intersection with the base distribution's support.
A parameter-dependent bound also determines `constrain`, `unconstrain` and
`logjac`, independent of declaration order. Array and plate priors accept
shared scalar bounds. Per-element bounds are not built yet.

```julia
a ~ Normal(0, 1)
x ~ truncated(Weibull(2 + exp(a), 1), exp(a), 3 + exp(a))
p ~ Dirichlet(3, exp(a))
y .~ Normal.(x, 1)
```

`Dirichlet(alpha)` takes a concentration vector from data, a sampled array,
a vector literal containing live values, or an array-valued definition.
`Dirichlet(K, a)` takes a literal dimension and a live scalar concentration.
The concentration shape determines the simplex size at binding; it does not
replicate the density body. Concentrations must be positive.

Dynamic Normal, Cauchy and Weibull truncation supports native and Reactant
primal and reverse execution. Gamma, Beta and InverseGamma truncation calls
incomplete gamma or beta functions whose traced methods are still unavailable;
those normalizers support native execution. The existing simplex transform's
Reactant limitation also applies to hierarchical Dirichlet priors.

## Julia semantics, written out

- Use Distributions.jl constructors (`Normal`, `Exponential`, `Gamma`, …).
- Broadcasting is explicit: `mu = a .+ b .* x`.
- A vector response uses the dotted tilde, `y .~ Normal.(mu, sigma)`. A plain
  `y ~ Normal(...)` on a data vector is rejected.
- A `~` whose left-hand side is a bound data column is an observation; every
  other `~` declares a parameter.
- `Ordinal` permits an intercept alongside its thresholds, for both
  cumulative and stopping-ratio responses. Both declarations and their
  priors are translated as written.
- Single assignment, no `if`, no `target +=`. Loops are written as
  `@plate` cells or `@scan` recurrences (see [Plates](#Plates)).

## Factor levels

`c[levels(g)] .~ Normal.(0, 2)` declares one coefficient per level of `g`, and
`c[g]` gathers them per observation. A full-cover factor may appear alongside
an intercept, with the fixed or hierarchical priors written in the model.
RK-PPL translates these declarations without imposing an identifiability or
posterior-propriety test. Offsets are plain data summands.

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
have the same mathematical plan. Library draws use the call's namespace:
`nt.b.sd`, `nt.b.z`, and, for correlated margins, `nt.b.L`. Centered entries
expose `nt.b.c` instead of `nt.b.z`; their coordinates are the coefficients
themselves, so their densities
at a packed point differ from the non-centered entries.

Centered correlated coefficients use ordinary row priors:
`eachrow(B[levels(g), 1:K]) .~ MvNormalCholesky(mu, F)`, with a K-vector
mean `mu` and lower-triangular covariance factor `F`. Read their effects as
`B[g, 1] .+ x .* B[g, 2]`. The legacy `varying_draws` / `varying_effect`
keywords are `eta`, `levels`, and `sd`; centered coefficients are declared
through array priors or the centered library entries above.

```@eval
Main.ReactiveKernelsDocs.render_rkppl_corpus_example("27_varying_slope_lib.jl", :rkppl_varying_slope)
```

The stratified correlated entry returns rows already aligned with the
observations. Its draws contain S×K scales, a K×K×S stack of factors, and J×K
standard-normal coordinates, where S and J count the sorted distinct strata
and groups. Read stratum k's factor as `nt.r.L[:, :, k]`. The returned rows
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

A single index on a multi-axis definition is linear and positional:
`b = (z * sd)'` makes a row, and `b[g]` reads its column-major positions.
A module function's result has no tracked level axes: `b = identity(z)`
followed by `b[g, 1]` uses integer positions too. Binding checks that positional
indices are positive integers and within known sizes. If the result's size
depends on a function call, Julia checks its bounds when the result is read.
Linear indexing of an adjoint row currently supports native execution and
Enzyme gradients; Reactant tracing fails on that wrapper. Linear indexing of
a plain matrix and positional gathers from module-produced arrays compile.
See [backend limitations](constraints.md#acceptance-and-existing-limitations).

Gathering by a derived column is not supported yet. Pass a declared array
whole to a model-level Julia function when it needs the full value; a bare
array combined directly with observation data must first be indexed to the
observation axis.

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
`b[j, 1:2]` and `MvNormal` also work; these multivariate priors use shared
arguments. Scalar priors can vary by level, and a level cell can define
deterministic values:

```@eval
Main.ReactiveKernelsDocs.render_rkppl_corpus_example("101_level_plate_values.jl", :rkppl_level_values)
```

`c[k] ~ Normal(m[k], 1)` also works with a bound Julia array `m`.
Here `m[k]` uses the authored Julia index; a declared `z[levels(g)]` instead
uses its level axis, so noncontiguous and string labels gather the correct
coordinate. Deterministic outputs retain that axis for later reads such as
`d[g]`. Each cell remains one RK plate body as the number of levels grows.
Declared array reads inside a level cell are aligned by label before entering
the plate. With `z[levels(g)[2:end]]`, `z[k]` is zero for the omitted first
level and reads the selected coordinate for every other level. A declaration
on `levels(h)` also works when every loop label occurs on that full axis;
the order and number of its coordinates can differ from `levels(g)`.
Matrix row reads follow the same rule. Raw bound-array reads and lazy
branches keep their authored Julia semantics.
Reactant's remaining limitation for an invalid host-constant index inside
an inactive live branch is recorded in the [core constraints](constraints.md).

In an array cell, an observation must read a named per-index output
(`y[i] ~ Normal(r[i], sigma)`), rather than a cell local directly.

`@scan begin … end` writes a sequential recurrence, such as an AR(1) state
(corpus `38_scan_ar.jl`).

## Submodels

`@rkppl name(args...) = begin … end` defines a reusable block. Using it,
`sigma ~ half_scale(1.0)`, expands it inline under the left-hand side's name, so
it lowers exactly like the hand-inlined program. A submodel whose result is a
response pointer is used as an observation stream: `y ~ stream(x, g)`.
`Base.merge(model, override)` replaces or appends statements by name.
Submodels also accept declared keyword defaults and statement replacements:
`custom = merge(linear_pk_log_f, :(slope ~ Normal(0, 0.5)))` returns a new
library body, which a program uses as `log_F ~ custom(sched; k = 5)`.

Each call owns a lexical namespace. For `z ~ sm(x)`, `z.b` reads the
submodel's local `b`, and bare `z` is the actual returned Julia value in
arithmetic and function arguments. Nested calls compose paths: `z.w.b`.
Locals never create caller bindings, so `a.b_c`, `a_b.c`, and a caller
parameter or data value `a_b_c` coexist.

Property access checks the local namespace first. If the scope has no local
with that name, it reads a property of the returned value. Explicit
`getproperty(z, :b)` always reads the returned value, even when `z.b` names
a local. The caller can read sampled and deterministic locals, including
data-only definitions. Ordinary caller assignments cannot redefine them;
explicit `merge` operations replace their declarations. Loop variables remain local
to their loop. Per-cell calls support `theta[i].b` and the whole local array
`theta.b`; their scalar-body restrictions still apply.

```@eval
Main.ReactiveKernelsDocs.render_rkppl_submodel_example()
```

### Event-axis bioavailability

`log_F ~ linear_pk_log_f(sched; k = 5, c = 1.5)` combines a linear log-dose
effect with an HSGP in schedule operation order. The schedule's operation
fields and basis are data; the body states every prior, including the named
length-scale validity floor. The reference dose defaults to one in the
amount column's units; replace `reference_dose = 1` in the body to change it.
These are the live library statements:

```@eval
Main.ReactiveKernelsDocs.render_rkppl_library_definitions((:linear_pk_log_f,))
```

The synthetic schedule program below shows the call, preparation, and
evaluation. Its draws are `nt.log_F.slope`, `nt.log_F.rho`,
`nt.log_F.sigma`, and `nt.log_F.z`. Replace a prior by merging the library
body, then replace the call statement in the model. The old
`log_F = linear_pk_log_f(...)` form is retired because assignments do not
declare parameters. Rebuild old prepared models and coordinate mappings.

```@eval
Main.ReactiveKernelsDocs.render_rkppl_corpus_example("99_plate_50_grouped_pk_logf.jl", :rkppl_event_lp; preamble = "using ReactiveKernelsPPL")
```

This is the generated kernel for that program:

```@eval
Main.ReactiveKernelsDocs.render_rkppl_kernel_program("99_plate_50_grouped_pk_logf.jl")
```

## Replace, pin, and condition

Each operation returns a new model. The original program and shared submodel
remain reusable. A statement replacement changes the declaration in place;
an indexed replacement must match its complete left-hand side.

For example, `merge(model, :(b[axes(X, 2)] .~ Normal.(0, 2)))` changes an
array prior, and `merge(model, :(z.w.tau ~ HalfCauchy(0.5)))` changes a
nested local's prior.

Scoped replacements use the same dotted paths as ordinary reads. Derive a
reusable submodel variant with `merge(submodel, :(tau ~ HalfNormal(2)))`;
relative paths in a nested variant are expanded separately at each call.

A pin removes the entire declaration and stores its number or array as input
data. It adds no density and no sampled coordinate. A call keyword naming a
declaration has the same meaning.

Scoped keyword pins use `var"z.w.tau" = 0.3`.

Conditioning keeps the sampling statement. Its density at the given
constrained value contributes to `likelihood`; the observed variable has no
sampled coordinate and no transform Jacobian. For example, observing
`tau ~ HalfNormal(1)` at `0.3` contributes
`logpdf(truncated(Normal(), 0, Inf), 0.3)`. A pin of `tau` contributes nothing.

`model(; x) | (; y, tau = 0.3)` and
`condition(model(; x); y, tau = 0.3)` are equivalent.
`condition(plan; tau = 0.6)` rebinds that observation in a new bound plan.

This example executes all four operations and checks the density retained by
conditioning against Distributions.jl:

```@eval
Main.ReactiveKernelsDocs.render_rkppl_rewrites_example()
```

Direct lowering declares observation roles separately from supplied data:
`lower_rkppl(ast, data; conditioned = (:y, :tau))`, then `bind_data(plan, data)`.
Pass a captured model to `lower_rkppl` when it carries scoped merge edits.
Sized array observations must have the declaration's shape. Partial indexed
writes do not replace a complete declaration and fail explicitly.

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
  A submodel parameter `b` in call `z` has coordinate `z.b` (a vector has
  `z.b.1`, `z.b.2`, …) and constrained value `nt.z.b`. Nested calls return
  nested NamedTuples, such as `nt.z.w.b`; `unconstrain` accepts that same
  structure, and `restore_draws` stacks its leaves across draws. These
  containers include sampled locals; deterministic locals, observed slots,
  and submodel return values are read in the model itself.
  Generated draw blocks keep their existing names under the call. If an
  author local claims that name, the generated block gets the first free
  numeric suffix; the author local keeps its name. Read `coordinate_names`
  for the resulting block labels.
  Rebuild prepared layouts and draw mappings when migrating from old
  flattened names such as `z_b`. Unusual identifiers use Julia's `var"…"`
  spelling in coordinate labels, so a literal name `var"z.b"` stays distinct
  from the scoped path `z.b`.
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
