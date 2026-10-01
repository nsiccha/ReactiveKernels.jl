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

`varying_effect(g, [x])` draws a per-group slope (`[1]` for an intercept)
together with its scale. `varying_draws` / `varying_slice` share correlated
margins across several uses.

```@eval
Main.ReactiveKernelsDocs.render_rkppl_corpus_example("27_varying_slope.jl", :rkppl_varying_slope)
```

## Design matrices

Bind the matrix once with `hcat` (the `1` is the intercept column), use it only
as `X * b`, and size the coefficients with an axes prior. Scalar prior arguments
are shared across elements; literal vectors give one value per element.

```@eval
Main.ReactiveKernelsDocs.render_rkppl_corpus_example("46_matrix_gaussian.jl", :rkppl_matrix)
```

## Plates

`@plate for i in R … end` writes a loop whose cells each mean one iteration
of that Julia loop, so every value in a cell is a scalar and needs no dots.
Shapes come from named data (the range, a data index column), never from a
separate size argument. A cell holds observations, per-cell latents, per-cell
submodel calls and cell locals. The whole loop lowers at once, exactly like its
broadcast spelling.

```@eval
Main.ReactiveKernelsDocs.render_rkppl_corpus_example("99_plate_32_gaussian.jl", :rkppl_plate)
```

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
- Coordinates are named after the predictor the layer reconstructs, not after
  your parameter names: `mu.Intercept`, `mu.x`, `mu.g_1`, …. Read
  `coordinate_names(built.layout)` and `constrain`.
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
