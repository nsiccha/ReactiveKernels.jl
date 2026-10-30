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

## 0.1.21

- `prepare_partition(built, plan, observations; by, locality = :error)`,
  `PartitionQuery`, `partition_labels`, `prepare_partition_sampler(built, plan,
  observations, u0; by, backend, locality)`, `PartitionSampler` and
  `partition_value_and_gradient!(s, g, u, label)`: one partition of the named
  observations (all cells whose key equals a label, such as one subject's)
  as one compiled pass, with its value and dense gradient. `by` names a
  supplied data column with one entry per cell of each observation's plate
  (one name, or `observation => column` pairs). Requires ReactiveKernels
  0.1.23.
- `prepare_partition_sampler(…; compact = true, partitioned = ())`: with
  `compact` (the default) the gradient is over each partition's footprint,
  the packed coordinates it reads, computed at preparation for every label
  (`ReactiveKernels.partition_footprint(s, label)` lists them; `g` has the
  footprint's length), the active coordinates overlaying a read-only packed
  base as in `prepare_compact_cell_sampler` (native Enzyme Reverse needs
  `Enzyme.set_runtime_activity`); `compact = false` gives a dense gradient
  over the whole packed vector, nonzero only on the footprint (atlas-17, one
  unit: compact 2.6 ms, dense 3.1 ms). `partitioned` names the parameters whose
  rows belong to one partition each, bare when indexed by the key column
  (`z[levels(subject)]`) or as `parameter => labels`; a partition that reads
  another label's rows of them is refused at preparation with a
  `ReactiveKernels.PartitionSeparabilityError`. The sampler prepares the
  footprint kernel once (`ReactiveKernels.partition_footprint_kernel`).
- An LKJ factor's unconstrained block is read as a statement of its own
  (`_ppl_lkj_raw_<name>`), as every other block is: read inside the `tanh.`
  expression, a cell or partition counted the whole packed vector as read, so
  a model with correlated group effects had every coordinate in every
  partition's footprint.
- `prepare_cell_query`, `prepare_cell_sampler` and
  `prepare_compact_cell_sampler` take `locality = :error | :warn | :ignore`
  (`nothing`, the default, reads ReactiveKernels' `cell_locality`
  preference): with `:error` a query is refused with a
  `ReactiveKernels.PartitionLocalityError` naming each model value the cell
  reads at its rows but would compute whole, and the statement that computes
  it.

## 0.1.20

- A constructed endpoint (`object(; …).endpoint(x)`) accepts any method-bearing
  kernel object, including one a downstream front end builds on ReactiveKernels'
  object skeleton. Requires ReactiveKernels 0.1.20.

## 0.1.19

- A categorical's level labels no longer reach the generated program as a
  literal array. The level codes of a gather over a `levels(g)` axis, a
  factor contrast and a level plate's values read the axis's labels as a
  data-only graph value (`_ppl_levels(g)`, `_ppl_unique(g)` or
  `_ppl_values(g)`, indexed by the axis's selection), which preparation
  folds under `bound=`, and `_declared_codes` matches labels through a
  dictionary instead of a nested loop. The labels are still the program's:
  a build's recorded program carries them as quoted data, so `prepare_query`
  refuses a binding with other labels as before (`rkppl-use` §14). On the
  ArbeitsplatzAtlas program, whose two Gemeinde columns hold 43,796 rows over
  10,950 labels each, the two level-code statements compiled for 60 s and
  45 s of the bound-data prefix's first call and matched for 1.2 s of its
  warm run. A `jax_source` module computes the labels in `prepare_data`
  (`_rk_levels`: the sorted distinct values of a plain column).

## 0.1.18

- A cell query (`prepare_cell_query`, compact cell samplers) computes an
  affine predictor with an offset at the cell's rows: RKPPL declares the
  predictor's coefficient vector `::AbstractVector{Float64}`, as its other
  coordinate views are declared. For `mu = a .+ b .* x .+ z[g]` with
  `y .~ Normal.(mu, s)`, one cell allocates 592 B at 100 and at 100,000 rows;
  0.1.17 computed `mu` whole in every cell (1,424 B and 800,576 B).
- With ReactiveKernels 0.1.16, a cell also computes at its rows a covariate
  completed from drawn entries (`observed .+ mask .* vcat(free, grouped)[lookup]`,
  as BRM completes `mi()` covariates) and read through a gather, a location
  RKPPL reads through a typed alias, and a location gathered from a sum of a
  predictor at two row indices. Requires ReactiveKernels 0.1.16.

## 0.1.17

- **Breaking:** ReactiveKernelsPPL has no Reactant path, following
  ReactiveKernels 0.1.15. Generated kernels read packed blocks through
  `Base.view` (they called `ReactiveKernels._tensorized_view`) and carry no
  `within_compile` branches. The fields of a `SamplerQuery` can no longer be
  compiled with Reactant; call the query natively. Requires ReactiveKernels
  0.1.15.

## 0.1.16

- A query preparation (`prepare_query`, `prepare_sampler`, `prepare_query_ad`,
  `prepare_cell_query`, `prepare_cell_sampler`,
  `prepare_compact_cell_sampler`, `prepare_predictive`, `jax_source`) checks
  that its binding generates the build's program once per binding and build,
  and a cell query reads its observation's plate from the program the build
  recorded. On 0.1.15 and earlier every preparation generated the whole
  program again for the check, and a cell query generated it a second time,
  however small its query: on a 223,941-coordinate BRM program (1,984
  statements) each cell query and each compact cell sampler took about 50 s,
  against 0.02 s and 0.10 s now once the cell has compiled. The first
  preparation of a binding still generates the program once. A binding of
  another program is refused on every preparation, as before.

## 0.1.15

- A `@plate` cell that computes an array per index (`b[i, 1:1] = …`) looks a
  label index up on a `levels(…)` axis also when the index is a definition
  computed from data (`g = glab[pos]`, then `z[g[i], :]` over
  `z[levels(g), 1:1]`), as it does for a data column. This includes an index
  passed to a submodel, such as BRM's stratified group effects over a
  grouping computed from data. On 0.1.14 and earlier the cell indexed the
  array with the label itself: a string label raised `ArgumentError: invalid
  index`, and an integer label read the entry at that position, a
  `BoundsError` past the axis and the wrong level otherwise, right only where
  every label equals its position (labels `1:G`). The data-column spelling is
  unchanged.

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
