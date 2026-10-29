# Changelog

All notable changes to ReactiveKernelsPPL.jl are documented here, newest
first.

Every landing that adds or changes public API (an export, a macro, a keyword,
a method, or documented behaviour a dependant can rely on) bumps the patch
digit in `Project.toml` and adds its entry under a new heading at the top of
this file. An entry that breaks something a dependant relies on says
**Breaking:** and what to change. The minor digit, the breaking one in
[semantic versioning](https://semver.org/) before 1.0, has not moved yet.
A package or app that calls that API states the version as its floor, for
example `[compat] ReactiveKernelsPPL = "0.1.1"`. An environment that pins an
older ReactiveKernelsPPL commit is then refused when it resolves, with
`Unsatisfiable requirements detected for package ReactiveKernelsPPL`, instead
of failing or silently behaving differently at run time. Changes before 0.1.1
are recorded in the git history only.

## 0.1.14

- A compact cell sampler (`prepare_compact_cell_sampler`) reads a selected
  packed coordinate in constant time, through a map from packed position to
  selection slot that the sampler keeps. It was a binary search over the
  selection on every read. The selection is checked once when it changes,
  where every call checked it, so a run of `cell_value_and_gradient!` calls
  over one selection pays one comparison each. Values and gradients are
  unchanged. On BRM's regional panel at 34,626 packed coordinates with a
  2,892-coordinate selection, a 27-row cell's gradient takes ~38 µs instead
  of ~78 µs, a one-row cell's ~2.9 µs instead of ~4.1 µs, and one subject's
  2,795 cell gradients ~14 ms instead of ~21 ms. The sampler holds one
  `Int32` per packed coordinate.

## 0.1.13

- A proven scalar location stays a scalar beside a design matrix, a `@scan`
  or a latent `@plate` too, as it already did in a plan with one ordinary
  observation axis: the observation plate broadcasts it, with no row vector
  and no fused row product (`dot`) of it per evaluation. Densities and
  gradients are unchanged. A plan with a packed `mi(…)` response keeps the
  rows, since it gathers every plate input by job.

## 0.1.12

- A `@plate` output holding one number per index (`c[i] = sum(lam)`) lowers
  as a location, a scale or a composition operand also when its cell reads a
  composed predictor (`f = 0.4 .* mu[main] .+ u[m]`, then
  `lam = exp.(f[rows[i]])`). On 0.1.11 and earlier lowering refused it with
  "predictor c applies `_ppl_plate_column` inside a composition".

## 0.1.11

- Probit and cloglog cumulative-ordinal interior levels are accurate when
  both their cutpoints lie far above `eta`: the mass comes from the
  complementary tails. On 0.1.10 and earlier the log-CDFs rounded it away. A
  probit level between 7 and 8 was off by `6.7e-6`, and a cloglog level
  between 5 and 6 gave `-Inf` where the log mass is `-148.4`. Values,
  gradients and pointwise values below the upper tail are unchanged.

## 0.1.10

- A `@plate` cell may destructure, as Julia does: `(u, v) = f(x[i])` binds
  each element, with `f` run once per cell, whether it reads parameters or
  only data. An extra element is dropped, and a missing one raises
  `BoundsError` at evaluation. On 0.1.9 and earlier lowering refused it with
  `selected plate assignments bind a local or an indexed scalar column`.

## 0.1.9

- A sampling right-hand side whose head is a parametric constructor,
  `T[i] ~ Law{LogNormal}(loc, s)` or `b ~ Law{LogNormal}(…)`, lowers as
  Julia's constructor call, in a joint `@plate` cell latent and at model
  level: its `sampling_geometry` receives the constructor `Law{LogNormal}`.
  On 0.1.8 and earlier lowering refused it with `external RHS … needs a
  module-visible constructor`. The constructor's base and type parameters
  resolve in the model module; one that names no binding there, or a model
  value, is refused, naming it.

## 0.1.8

- A design rebound from a binding without a dotted-cell response that held
  one array per index there (`bind_data(fit, (; t = tnew))`, `y` absent)
  observes one array per index, also when its cells read only computed
  values. `build_kernel(design)` then builds the fit's program, and a query
  of it on the fit binding equals the fit's own build. On 0.1.7 and earlier
  that build observed one value per index, and the fit binding was refused
  as a different program.

## 0.1.7

- A dotted `@plate` cell observing one array per index (`y[i] .~ D.(…)`)
  binds when its loop iterates another array's indices
  (`@plate for i in eachindex(total)`) and some cells are empty, so that the
  response holds fewer entries in all than there are cells. On 0.1.6 and
  earlier binding refused it with `response slice … indexes outside`: the
  loop range was checked against the flattened entries, not the cells.
  Binding now validates every entry the cells observe. Before, it read only
  as many flattened entries as there are cells, so an invalid entry past them
  (a negative Poisson count in the last cell) was not refused.

## 0.1.6

- An index written as integer arithmetic of literals is the position it
  evaluates to, as in Julia: `cs[4 + 1]` reads `cs[5]`, `Z[2 * 1, 1]` reads
  `Z[2, 1]`, and `z[end - (1 + 1)]` reads `z[end - 2]`. A submodel position
  `cs[S + 1]` called with a literal `S` reads that position too. On 0.1.5
  and earlier `lower_rkppl` refused these with `ContractValidationError`.

## 0.1.5

- A value computed with `cexpexp.` (the inverse cloglog link) evaluates where
  a distribution argument reads it: a Gamma mean `Gamma.(k, cexpexp.(eta) ./ k)`,
  an ordinal discrimination `Ordinal.(…, cexpexp.(eta))` and a transformed
  mean. On 0.1.4 and earlier lowering and `build_kernel` succeeded, and the
  first evaluation raised `UndefVarError: _ppl_cexpexp not defined`. Requires
  ReactiveKernels 0.1.9.

## 0.1.4

- A location summand computed from `ifelse.` over scalars is a scalar, as one
  computed from `logistic` is: `x[rows] .+ b .* s` with
  `s = ifelse.(a .> 0, a, b)`, or the same `ifelse.` inline. On 0.1.3 and
  earlier `bind_data` failed with `ContractValidationError:
  [_rkppl_leaf_1_off] term references missing column _rkppl_leaf_1`.
  Requires ReactiveKernels 0.1.8.

## 0.1.3

- A location summand computed from a `logistic` value is a scalar, as one
  computed from `exp` is: `p .+ e .* logistic.(a)` and `p .+ e .* s` with
  `s = logistic.(a)` beside a product definition `e = logistic.(b) .* exp.(a)`,
  `x[rows] .+ b .* logistic.(a)`, `x .+ 2 .* logistic(a)`, `x .+ e`, and a
  data-only `m = logistic(mean(x))` read as `x .+ b .* m`. On 0.1.2 and
  earlier `bind_data` failed with `ContractValidationError:
  [_rkppl_leaf_1_off] term references missing column _rkppl_leaf_1`.
  Requires ReactiveKernels 0.1.7.

## 0.1.2

- `logistic` is a scalar value function, as `exp` is. Undotted over a scalar,
  `logistic(x)` is the inverse logit, the same function `logistic.` broadcasts
  (`exp(-log1pexp(-x))`, with finite derivatives at any argument): in a
  definition (`p = logistic(logit_p)`, `e = logistic(z) .* exp.(r)`), a prior
  argument, a distribution argument (`Binomial.(n, logistic(a))`), a
  data-only definition, a plate cell, and a `@scan` seed or step.
  `admitted_functions()` lists it. Before, lowering refused it as a value
  (``calls `logistic`, which only lowers in `.~` responses, not as a value``),
  and a `@scan` body read it as the logistic distribution kernel
  (``kernel object :location_scale has 0 unbound kernel port(s)``).
- A count response's rate or mean is any value, inline as its named
  definition: `Poisson.(logistic(a) .* x)`, `Poisson.(.*(0.25, logistic(a),
  exp.(b .* x) .* p))`, `Poisson.(e[g])`, `NegativeBinomial2.(exp.(d) .* (u .+
  v), phi)`, a dotted function that is not a link (`Poisson.(abs.(eta) .+
  0.1)`), and a mixture component's rate. This covers `Poisson`,
  `NegativeBinomial2`, `NegativeBinomial`, `HurdlePoisson`,
  `ZeroInflatedPoisson`, `InverseGaussian`, `Exponential`, `BetaBinomial2` and
  `ZeroInflatedBinomial`. Before, those slots took only a name or a dotted
  inverse link (``broadcast the link (`exp.(eta)` …)``), while
  `rate = logistic(a) .* x; Poisson.(rate)` lowered.
- **Breaking:** an undotted inverse link over a vector in a link slot
  (`Bernoulli.(exp(eta))`, `Binomial.(n, normcdf(eta))`) fails at lowering with
  the dotted spelling, as Julia's `MethodError` and the same call in a
  definition do. A Bernoulli or Binomial probability read it as the dotted
  link; write `exp.(eta)`. Over a scalar it is the dotted link's value
  (`Poisson.(exp(a))`).

## 0.1.1

- A `@plate` output whose cells are vectors is one vector per cell at model
  scope, as the Julia loop leaves it: `sum(cells)` adds the cells
  elementwise, `cells[g]` is cell `g`'s vector and `cells[k]` (a data index
  vector) the cells `k`, and a value computed from them is a location like
  any other vector (`obs .~ Normal.(sum(cells), s)`,
  `obs .~ Normal.(sum(cells[k]) .+ cells[3], s)`). Before, a location reading
  `cells[1]` or `sum(cells)` failed `bind_data` with `term references missing
  column _rkppl_leaf_1`, and `sum(cells)` over cells that read one array per
  index raised `MethodError` from `zero(Vector{Float64})`. Native values and
  ordinary Enzyme Reverse gradients.
- A cell that builds an array literal from per-index values
  (`cells[i] = [b * a1[i], b * a2[i]]`, `[x[i]; y[i]]`, a comprehension) runs
  once per index, as a retained RK plate cell. It was evaluated over whole
  columns, which built one array per literal entry (`[b .* a1, b .* a2]`), so
  `cells[k]` was checked against each cell's length instead of the cells.
- Requires ReactiveKernels 0.1.2.
