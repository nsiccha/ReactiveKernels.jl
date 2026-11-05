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

## 0.1.100

- Adaptive `tune_smc(...; grid_start=:independent)` starts each grid pilot at
  its own copy of `init` and runs up to `options.nthreads` pilots concurrently.
  Seeds and completed schedules are independent of worker count; evaluators
  and sampler state are privately owned. The reference pilot remains serial.
  The default `:sequential` strategy retains its preceding-knot warm starts.
- Independent tuning checkpoints retain every active/completed knot and resume
  with a different worker count. Callbacks and atomic checkpoint writes are
  serialized, stops are cooperative, and workers join before return or error.
- `resume_smc(...; nthreads=n)` overrides the saved production worker cap.
  Omission retains it. Numerical clouds, evidence and RNG streams are preserved;
  worker-count history and reduction diagnostics may differ across counts.

## 0.1.99

- Submodel tuple and named-tuple returns preserve whole vector fields,
  including unread returns and fields whose labels match caller bindings.
  A returned field shared between a bare likelihood argument and arithmetic
  keeps its value in either response order. This restores the documented
  Julia return-value semantics without changing the local-first meaning of
  `call.local` or explicit return-property reads with `getproperty`.

## 0.1.98

- SMC checkpoint and carried-coordinate identities normalize caller-owned
  sampling expressions just as built programs do. Helpers resolved in a fresh
  generated model module no longer make adaptive tuning, production or carried
  checkpoints depend on that transient module when deserialized. The sampler
  state and RNG restoration are unchanged. Existing checkpoints still require
  their original Julia/package versions; this does not migrate older files
  that already contain missing generated-module references.

## 0.1.97

- `L ~ LKJCholesky(K, eta)` accepts positive integer dimensions from bound
  scalar data and data-only expressions such as `length(channels)`, including
  reusable submodel arguments. Rebinding the same definition fixes the packed
  coordinate count at `K*(K-1)÷2`, including the constant factor at `K=1`, with
  the existing normalized prior, Jacobian and native Reverse path.

## 0.1.96

- `prepare_query_ad` accepts a vector of authored `FactorSelection`s from one
  catalogue, validates its bound program once, and returns queries in order.
  With `coordinates=:support`, fixed coordinate maps and selected plate rows
  use compact storage. Existing single-selection binding checks remain.
- Per-event `factor_support` borrows a supplied exemplar for read-only use;
  calls without one share a catalogue-owned default. Selection support reuses
  the event footprint kernels. Ownership inventory no longer copies the full
  packed point for every event. Requires ReactiveKernels 0.1.84.

## 0.1.95

- Require ReactiveKernels 0.1.83 for precise cell-query footprints through
  visible scalar generator sums, including GMRF readers with data-only index
  helpers. The authored model and ordinary native Reverse remain unchanged.

## 0.1.94

- A built observation graph can be rebound between empty and nonempty data
  with unchanged parameter layout and data types. Empty observations retain
  their plates and contribute zero likelihood while priors, Jacobians and
  ordinary reverse-mode gradients remain active. This includes heterogeneous
  mixtures, ranged observations and arrays observed inside plate cells.

## 0.1.93

- Export `rkppl_documentation(model_or_plan; path=nothing)` and immutable
  `RKPPLDocumentation` records. Literal Julia docstrings on declarations,
  reusable definitions and component uses retain raw Markdown, authored paths,
  definition provenance and source locations outside the numerical graph.
  Lowered plans instantiate nested and repeated component documentation once
  per authored scope, including per-cell components. Binding, pins and
  unannotated replacements preserve it; annotated replacements update it.
- Documentation queries copy the record vector without resolving components,
  evaluating data or rendering Markdown. Generated kernel source remains the
  actual numerical program; authored documentation is available separately.

## 0.1.92

- Native `MixtureModel` observations compose heterogeneous full-real component
  graphs, including Normal/Cauchy and caller-owned `sampling_distribution_graph`
  mappings with independent argument counts. Components retain their ordinary
  value computations inside the existing weighted log-sum-exp lowering.
  Requires ReactiveKernels 0.1.80 for shared component work inside support guards.

## 0.1.91

- Adaptive `tune_smc` accepts nested Treebars progress, an observational stop
  callback, and atomic latest checkpoints through `checkpoint_dir`, `resume`
  and `overwrite`. Cooperative interruption returns `SMCTuningPartial`, never
  a usable schedule. Resumption restores WarmupHMC's active sampler and both
  random streams while retaining completed reference/grid pilots; target,
  binding/frame, endpoint scores, configuration and versions are checked.
  In-memory partials can also be serialized and passed as `resume=partial`.
  Boundary history and obsolete checkpoint retention are bounded; the active
  sampler payload still contains its retained draws. Stops occur at sampler
  initialization/window boundaries or between pilots, not at hard deadlines.

## 0.1.90

- Native `MixtureModel` also accepts computed shared probability vectors
  through `Ref`, including `Ref([1 - w, w])` for a derived logistic scalar,
  named vectors, and general-K expressions. These values retain their
  authored priors and whole-vector input axes. Existing literal, simplex
  and bounded sampled-complement validation remains. Computed data-only
  vectors are validated at binding; general live vectors are checked for
  finite, nonnegative, normalized probabilities before logs.
- Regression coverage checks normalized density, ordinary Enzyme Reverse,
  pointwise likelihood, full printed-source replay, zero probabilities,
  changed shared data, support, type stability and bounded emitted size.

## 0.1.89

- Export `parameter_metadata(built/layout)` to inspect stored constrained
  parameter ranks and shapes, ragged cell shapes, authored name paths and
  packed coordinate counts without evaluating or preparing the model.

## 0.1.88

- Each bound plan reuses demanded data-only results across binding, layout,
  building and query preparation. Whole-value preprocessing used to run again
  for parameter sizes and each prepared query. New bindings have independent
  stores, unused expressions remain lazy, and `frozen_values` can read an
  existing result. Results live with the binding; bound inputs and results
  remain read-only. Requires ReactiveKernels 0.1.79.

## 0.1.87

- Native `MixtureModel` responses accept an expression-valued probability
  vector per observation, such as `vcat.(1 .- w, w)` for a logistic predictor
  `w`. Inline expressions and named values read the same graph, with the
  authored priors, coordinates and component laws. Each probability vector
  must have one entry per component, be finite and nonnegative, and sum to
  one. Zero-probability components retain finite gradients when logistic
  probabilities saturate. Shared probabilities continue to use `Ref(weights)`,
  with an axis independent of the observations even for scalar component locations.

## 0.1.86

- Host transforms synchronize their lazy bijector caches. Concurrent
  `constrain` calls can share a completed, read-only layout, including on first
  use. Caller-owned geometry and bound functions must also support concurrent
  calls. Draw restoration shapes and selective graph queries are unchanged.

## 0.1.85

- A data-only definition that computes a sparse array stays sparse when bound
  (`L = sparse(F)`). It used to be converted to a dense array, which lost the
  structure a cell reads through: a `SparseMatrixCSC` left factor reads its right
  operand only at the columns its rows store (RK 0.1.77). A dense value keeps
  being bound as before.

## 0.1.84

- `prepare_cell_factors`: a sum whose summands read another sum is no longer a
  reduction. With `dose = sum(a[events[i]] .* exp.(e[events[i]]))` and
  `y[i] .~ Normal.(0.01 * dose .* exp.(eta[obs[i]]), sigma)`, only `dose` is one;
  the density total stays in the outer factor, which reads `dose` from its sum
  vector as a number and computes the total at its own rows. Such a cell used to
  be refused under `locality = :error` (`0.01dose has no row form, and its type is
  not known to be a number`). When that type was known, each term of the density
  total recomputed `dose` and read all of its source, every cell's coordinates,
  with no refusal. Requires RK 0.1.76.
- **Breaking:** `cell_reductions(f)` of such a cell lists one sum fewer, so the sum
  vector passed to the outer factor is shorter and the total has no summand query.
  Follow `cell_reductions(f)`; sums that read no other sum are unchanged.

## 0.1.83

- Strict cell queries accept a response location whose sum has a summand of
  unknown shape, and a declared `LKJCholesky` factor read at rows. `mu = .+(r, f)`
  with `f` a submodel's result is computed at the cell's rows, as is the factor
  in `b = L * z`, instead of being refused as `PartitionLocalityError` (or
  computed whole per cell under `locality = :warn`). Values and derivatives are
  unchanged. A value of neither one entry nor one per row keeps its
  `DimensionMismatch`, and a reader of the whole factor (`sum(L)`) still keeps
  it whole. Requires ReactiveKernels 0.1.74.

## 0.1.82

- Strict cell queries and cell factors read one element of a computed matrix at
  the cell's row, through ReactiveKernels 0.1.73: a producer plate's
  `signal[p] = exp(a[p]) .* paths[p] .+ b[p, 2]` with `b = z * F` reads only row
  `p` of `z`. Its footprint is that row's coordinates and the coordinates of `F`.
- `prepare_cell_factors(...; partitioned)` computes an array-valued producer
  plate's cell at the selected row, so a term's local footprint is that
  trajectory's coordinates. 0.1.79 refused such a cell, and from 0.1.80 it
  prepared with the producer's input read whole in every term.
- **Breaking:** under `locality = :error`, a matrix with no row form read as
  `M[p, 2]` is now refused by name, as `v[p]` of such a value already was
  (ReactiveKernels 0.1.73 says what to change).
- Requires ReactiveKernels 0.1.73.

## 0.1.81

- Every object of a model whose caller-owned law takes an argument with a
  resolved binding (a property read `cells[i].geometry`, an array literal
  `[1.0, 0.4]`) deep-copies, at model level and in an `@plate` cell: the layout,
  the plan, the built program, `density_factors`, `prepare_cell_query` and
  `prepare_cell_factors`. `deepcopy` failed with `deepcopy of Modules not
  supported`, because the generated program rebuilt such a binding as
  `Core.GlobalRef(<module>, :name)`, holding the module as a value. It now reads
  the module through its own binding, so `kernel_expr` and `code_expr` print
  `Core.GlobalRef(Base.Base, :vect)` where they printed
  `Core.GlobalRef(Base, :vect)`. The printed source still replays, and such a
  law's value and gradient are unchanged. A joint cell latent's printed source
  whose law takes such an argument now replays too; it held
  `$(QuoteNode(:(Base.vect)))`, which does not parse back into that source.
- `RKPPLModel`, a binding of one (`model(; x)`) and `RKPPLSubmodel` deep-copy too;
  the copy shares the module the source resolves in.

## 0.1.80

- Strict cell factors now know which values are numbers, as `prepare_cell_query`
  already did: `prepare_cell_factor_query`, `prepare_cell_factor_ad`, compact
  cell derivatives and local footprints prepare from the packed point (`u0`, or
  zeros). A centred pool's `sum(z)` and a readout's scalar arguments no longer
  refuse. Query calls with other argument types cache a specialization, so a
  recording vector is still passed through without copying.
- **Breaking:** without a shared prefix, direct calls to a `CellFactorQuery`'s
  `q.kernel` and its full footprint kernel need the exact prepared types
  (`Vector{Float64}`, `Int` positions, a `Vector{Float64}` of sums). Call `q(...)`
  instead, or `partition_footprint_kernel(q; shared = false)`.
- A value now known to be a number has no rows, so what it reads is read whole by
  each cell and appears in that cell's footprint, where it used to be refused.
  Values and derivatives are right; a caller screening cell width sees the cost.
- An outer factor with no reductions differentiates one active vector. Callers
  keep the `(sums, coordinates)` shape for directions and results. Forward over
  reverse now works on such a factor, and large cells no longer raise
  `EnzymeRuntimeActivityError` on the unused sum operand under reverse over forward.
- Requires ReactiveKernels 0.1.70.

## 0.1.79

- `prepare_partition` and `prepare_partition_sampler` accept a key as its labels,
  per observation: `by = (; y = labels)`, or `y => labels` pairs, with one label
  per cell. The labels need not be data of the model and no statement need read
  them, which is what a model bound through BRM needs: BRM binds only the columns
  a statement reads. They stay out of the program; the query owns a copy, and
  `built` and `plan` are unchanged. A parameter named in `partitioned` under such
  a key gives its levels' labels, `parameter => labels`.
- **Breaking:** a partition key whose length differs from its plate's cell count
  is refused at preparation (RK 0.1.69). A shorter key used to be accepted and
  left the cells past its end out of every partition, so the labels no longer
  summed to the likelihood. Requires RK 0.1.69.

## 0.1.78

- `prepare_partition` and `prepare_partition_sampler` accept the pairs their
  documentation promises for `by`: one `observation => column` pair or a
  collection of such pairs (a vector, a tuple, a generator), besides one column
  name, a named tuple and a dictionary. A vector or tuple of pairs used to be
  read by position and refused as giving no key column, and a single pair raised
  a `MethodError`. Any other `by`, such as a vector of names, raises a
  `ContractValidationError` that names the accepted spellings, as does a `by`
  that gives no key for a named observation; an observation given two different
  columns raises one naming both. Requires ReactiveKernels 0.1.68.

## 0.1.77

- **Breaking:** the learning filter's window boundary keeps correlations. At each
  rollover the members' weighted covariance is summarized by up to
  `max_reflections` leading correlation directions (new
  `initialize_filter_smc` keyword, default 16), fitted in the members'
  marginally standardized frame with `WarmupHMC.fit_gaussian_moments`; the
  remaining directions share the remaining standardized variance equally. The
  next window's prior and ensemble draws use this diagonal-plus-low-rank
  Gaussian instead of the marginal moments. A diagonal boundary treated the
  marginal widths of correlated parameters (a hierarchical intercept and its
  group effects) as independent information, so every window contracted the
  directions the data do not inform, compounding to about 1 % of the exact
  posterior width after twelve windows. `max_reflections=0` restores the
  diagonal boundary and walk.
- Replenishment random walks use the same correlated covariance of the weighted
  members (`2.38^2/d` times it), so moves stay inside the narrow parameter
  combinations the data pin down instead of being rejected there. At most the
  distinct-position ESS minus 2 directions are kept; with none the walk stays
  diagonal.
- `carry_smc(completed; basis=:correlation)` carries the same standardized
  correlation summary (diagonal-plus-low-rank reference, `max_reflections`
  directions). Unlike `basis=:fit`, which learns covariance directions in native
  coordinates and spends them on the widest coordinates, it keeps correlations
  among parameters on very different scales; use it for a learning-filter
  calibration. `fit_diagnostics` reports `directions` and
  `effective_sample_size`.
- History entries add `boundary_directions` and `proposal_directions`;
  `proposal_scale` is the walk's per-coordinate standard deviation.
  Parameter-learning checkpoints use schema 3 and record `max_reflections`.

## 0.1.76

- A joint cell latent's cell reads its packed block through a view of the
  packed vector, so an observation that selects entries of `field[i]` reads
  only those coordinates and a strict cell query's footprint is their
  positions (an empty selection reads none). A transform that couples the
  block, or computes it whole, reads the cell's own block. Joint priors stay
  one factor per index; shape checks, values and ordinary Reverse gradients
  are unchanged, and a cell's work no longer grows with its block's length.
- Strict cell factors (`prepare_cell_factors(...; locality = :error)`) now
  accept such observations: a factor term reads only its row's coordinate,
  with the joint shape check still applied to the whole value. Per-index
  weights and indices may come as data columns or as one named tuple of row
  vectors per index (`r = rows[i]; mu[i] = r.weight1 .* field[i][r.index1]`).
  Requires ReactiveKernels 0.1.67.

## 0.1.75

- Strict cell queries, compact cell derivatives and cell factors compute
  per-position helpers at the cell's rows, including enumerated
  comprehensions and loops that push one row per trajectory. The outer factor
  and the cell query now agree on such helpers. Require ReactiveKernels 0.1.66.

## 0.1.74

- Prepared queries of a model whose caller-owned law takes an authored
  argument with a resolved binding (a property read `holder.inner`, an array
  literal `[1.0]`) can be deep-copied, for example once per chain, and the
  copy evaluates as the original. This needs ReactiveKernels 0.1.65, now the
  compat floor. `model_view(built; bound)` displays such a model's generated
  source instead of failing with `deepcopy of Modules not supported`.

## 0.1.73

- **Breaking:** learning-filter replenishment now resamples the current weighted
  parameter/filter pairs, then moves them with particle-Metropolis random walks
  scaled by `2.38/sqrt(d)` times the members' weighted spread, accepted by the
  window prior ratio times the window likelihood-estimate ratio. It no longer
  discards learned positions to draw a fresh window-prior population, and the
  tuned NUTS kernel now drives only explicit recovery (its `moves` still sets
  proposals per member). Replenishment triggers on the ESS over distinct
  parameter positions, so resampled copies sharing the weight no longer hide a
  collapse. Full spatial collapse raises `SMCDegeneracyError` after mutation;
  invalid Gaussian rollover scales report coordinates and the preserved input
  state. Failed steps consume neither their observation nor the input RNG.
- `tune_filter_smc(state; ...)` independently tunes the current window reference;
  `rejuvenate_smc(state; tuning, moves)` attempts fixed-budget recovery without
  resampling, consuming data or changing weights/evidence. Failed attempts remain
  transactional; retry a failed observation only after recovery succeeds.
- Bounded learning histories distinguish assimilation and recovery, and report
  distinct members, distinct-position ESS, actual movement, weighted coordinate
  spread, the random-walk scale and kernel budgets separately from acceptance
  and member ESS. Parameter-learning checkpoints
  use schema 2 and require matching versions. The marginal-Gaussian/independent
  state boundary approximation is unchanged; diversity does not certify mixing.

## 0.1.72

- `factor_support` of a vector simplex (`s ~ Dirichlet(alpha)`) or ordered
  vector (`c ~ Ordered(D, n)`) prior or Jacobian now names only the entry's
  own packed block plus any sampled hyperparents; the ordering Jacobian names
  all but the first coordinate. Earlier versions returned every packed
  coordinate, because the transform read its block inside a larger
  expression. Values, gradients and ownership are unchanged.

## 0.1.71

- Map resolved TDist, Gumbel, Weibull, InverseGamma, InverseGaussian,
  Rayleigh, Chi, Chisq, FDist and BetaPrime constructors to owned distribution
  graphs. Preserve their arities, defaults and parameterizations, including
  BetaPrime's one-argument equal-shape constructor. Caller-authored restriction
  and normalization remain distinct; custom laws retain the existing fallback.
- Use the same inverse-Gaussian object for built-in responses and constructor
  graphs, including evidence tails and posterior-predictive laws. Require
  DistributionKernels 0.1.3.

## 0.1.70

- Add `prepare_smc_window(...; coordinates=:auto)` for changing authored
  coordinate sets. Compact targets combine the old active marginal, fresh
  owned prior events and new observations, retaining native derivatives.
- Add `smc_window_coordinates` and `smc_recover` for compact particle order
  and lazy correlated recovery. Selective `carry_smc` integrates the inactive
  conditional into one bounded-rank joint Gaussian with loss diagnostics.
  Compact covariance scales also serve as normalized importance references.
- Default `coordinates=:all` retains its strict frame contract. Selective
  windows reject partly fresh atomic events, ownerless densities whose
  structural support reaches fresh coordinates, and unsupported observation
  locality. See `prepare_smc_window` for the supported boundaries. Requires
  RK 0.1.61.

## 0.1.69

- Expose strict local factors for ordinary array sums such as
  `sum(z[rows[k]] + w[rows[k]])`, including transparent forwarding helpers,
  with or without shared-prefix caching. Each term retains both gathered
  operands and the original addition's shape checks. Requires RK 0.1.60.


## 0.1.68

- **Breaking:** SMC now raises `SMCDegeneracyError` instead of returning or
  advancing an exactly collapsed positive-dimensional cloud, even at temperature
  1. The exception preserves the input state and rejected-stage diagnostics.
  Catch it explicitly when recovery is wanted; a single constant coordinate,
  unchanged move or surviving ancestor is not a failure criterion.
- `rejuvenate_smc(state; tuning, moves)` runs a fixed number of independently
  tuned mutation transitions at the current temperature, without resampling or
  changing weights or evidence. The new schedule and RNG continuation survive
  checkpoints. Bounded stage histories distinguish `kind=:temper` and
  `kind=:rejuvenate`; failed attempts never advance the input state. Recovery
  restores an opportunity to explore, not a guarantee of posterior accuracy.

## 0.1.67

- Evaluate selected density events whose cell vectors were folded into bound
  constants. Flat prior events retain their authored coordinate ownership and
  empty read support, including dense and compact native derivatives and
  selections mixed with active factors.

## 0.1.66

- Compose authored Distributions Normal, LogNormal, Gamma, Exponential,
  Cauchy, Laplace, Logistic and DirichletMultinomial constructors into owned
  arithmetic graphs. Resolve constructor identities and defaults, preserving
  Gamma's scale parameterization. Shared normalization and bound-data terms
  can be reused across observations; stored distribution objects and custom
  families retain the numerical callback and caller-owned graph extension.
- Require DistributionKernels 0.1.2 and RK 0.1.59 for these graphs and their
  composed lazy support guards.

## 0.1.65

- Add `with_cell_batch(owner, u) do batch` for value-only fixed-point sweeps
  over cells, summands and outer factors. The callback owns a copied point,
  refreshes shared values once, and rejects expired, mismatched or invalidated
  batches without requiring an AD backend.
- Add `partition_footprint_kernel(q; shared=false)` for structural footprints
  restricted to declared local coordinates, including queries with no shared
  prefix. Value and local-footprint queries also accept compatible active
  `cell_pullback!` contexts. Full footprints and mixed derivatives remain
  available through their existing APIs. Requires RK 0.1.58.

## 0.1.64

- Caller-owned sampling laws may return a density `KernelSpec` from
  `sampling_distribution_graph`, alongside an expression constructing an RK
  object with `.logpdf(value)`. Density graphs compose for whole parameters,
  joint plate-cell latents and observations, independently of
  `sampling_geometry`. Static reconstruction preserves printed source replay
  for local specs and parametric constructors; complete explicit keyword
  boundaries compose in declaration order. Omitted defaults and splats keep
  the ordinary numerical RHS path. Requires RK 0.1.57 for object endpoint
  arguments computed inside a plate cell.

## 0.1.63

- Preserve vector-event broadcasts and grouped input axes in finite-state
  filters. Broadcast kernel-backed `LogDensity` laws and nested truncated
  laws retain their observation semantics, including singleton records.
- Infer current slices and finite carries from source-recorded vector helpers
  using declared input ports, including grouped recurrences and current-clock
  gathers. Drop unused group diagnostics and historical input columns.
  Whole-vector event laws keep their full helper computation. Requires RK
  0.1.56; see `prepare_filter` for supported helper shapes and record examples.

## 0.1.62

- Add `density_factors`, `select_factors`, `factor_coordinates` and
  `factor_support` for authored
  prior and coordinate-Jacobian events. `prepare_query` and `prepare_query_ad`
  accept factor selections; `coordinates` prepares compact derivatives over
  a read-only packed base. Joint laws and coupled transforms remain atomic,
  and structural supports retain hyperparameter dependencies. Owned coordinates
  are available separately from those transitive read supports. Requires
  RK 0.1.55 for independent selected cells of shared upstream plates.

## 0.1.61

- Add `partitioned` to compact cell AD and cell factor preparation, caching
  shared-only whole-array producers while retaining strict local derivatives
  over the partitioned coordinates. Factor preparation also accepts `u0`.
- Add `prepare_cell_pullback`, `cell_pullback!`, `prepare_cell_accumulator`
  and `accumulate_cell!`. Accumulate weighted local gradient, HVP and third
  contributions into shared cotangents, then apply the prefix pullback once
  per pass for exact mixed population derivatives with linear storage.
  Local derivative verbs accept the active pass in place of their base
  vector to avoid per-cell shared-coordinate scans.
  Require RK 0.1.54 for additive native Enzyme differentiation.

## 0.1.60

- Compute interval-censored likelihoods and truncated-prior normalizers
  with shared log-space subtraction. Censored observations use the same
  distribution log tails. Normal intervals in either far tail retain finite
  log densities and ordinary native Reverse gradients; LogNormal and the
  other closed-form log-tail families benefit from the same calculation.
- Preserve log-space tail accumulation for mixture evidence and stable
  Normal tails for probit ordinal observations. Interval endpoints and
  censoring/truncation semantics are unchanged.

## 0.1.59

- Strict cell queries and outer factors accept ordinary helper gathers such
  as `gather(z, indices) = [z[i] for i in indices]`, including wrappers and
  row-wise scalar operations. Values, packed footprints and native compact
  derivatives retain only the selected local coordinates. Requires RK 0.1.53;
  cross-row helper work still faces locality checks.

## 0.1.58

- Preserve ordinary reverse derivatives when mixture components have equal
  log densities, including all-zero initialization of complement-weight
  mixtures. Scalar and dotted `logaddexp` vocabulary calls use RK's existing
  stable primal, which retains the difference at finite ties. Values and
  mixture weights are unchanged.

## 0.1.57

- Add `prepare_smc(make_problem, u0; backend, model_id, binding_id, frame_id)`
  for prepared LogDensityProblems densities with supplied gradients. The
  factory rebuilds independent problem workspaces for pilots, production runs,
  mutation workers, updates and resume. Reuse Gaussian-reference tempering,
  fixed tuning schedules and nonfinite diagnostics; checkpoints retain data
  and explicit identities without serializing the factory or solver state.

## 0.1.56

- Recognize calibrated arrays of any rank and `eachrow`/`eachcol` slice
  declarations in finite-state filters and static parameter selection,
  including scoped submodels. Their priors are neither evaluated nor scored
  again. Missing calibration reports the authored declaration instead of an
  internal `haskey(..., nothing)` error.

## 0.1.55

- Discover gathered sums in guarded NegativeBinomial2 cells, including
  weighted summands, cell-local vectors and nonlinear outer densities over
  multiple sums. Preserve local summand footprints and ordinary native Enzyme
  derivatives. Require RK 0.1.52 for typed identity traversal during factoring.

## 0.1.54

- Add `SMCOptions(nthreads=Threads.nthreads())` for concurrent fixed-kernel
  particle mutation. Workers own sampler/AD scratch and disjoint output
  columns. Serial weighting, resampling and child-seed draws preserve the
  existing one-worker random stream, including for uneven particle counts.
  Stage summaries record the mutation worker count; checkpoint continuation
  retains the requested count. Concurrent Enzyme execution requires its
  concurrent-GC repair, currently available on the dogfood line.

## 0.1.53

- Prepare cell queries with the canonical packed `Vector{Float64}` and `Int`
  position types, preserving strict locality through opaque scalar helpers,
  type-selected collectors and guarded captured-array reads. Uncached queries
  preserve local reads through other Float64 vector types, using cached
  specializations for each packed-vector and position type, including Cartesian
  indices. Require RK 0.1.51 for this lowering.
- **Breaking:** direct calls to a cell query's `q.kernel` require those exact
  argument types. Call `q(u, i)` for other vector and position types; the kernel
  remains available for footprint analysis.

## 0.1.52

- Require RK 0.1.49 for strict cell queries through a visible type-selected
  scalar collector and broadcast offset. Selected values and ordinary Reverse
  gradients keep constant allocation as the plate grows. Vector flattening
  and conditions that depend on runtime values retain their locality limits.

## 0.1.51

- Add optional `partitioned` and `u0` to `prepare_cell_query` and
  `prepare_partition`. A declared shared prefix caches whole group-level
  producers before strict residual lowering. Shared-coordinate changes,
  including in-place edits, refresh owned cache values; local changes reuse
  them. Partition queries retain the sampler's separability check.
- `partition_locality(q)` reports the executed residual. `q.kernel` retains
  the original packed-coordinate footprint, including prefix dependencies;
  call `q(u, index)` for cached evaluation. Undeclared queries retain their
  behavior. Cached queries copy non-dense position vectors at preparation's
  dense boundary and require separate instances for concurrent callers.
- Apply the shared split before strict sampler preparation and preserve full
  and local gradients through the prefix. Each query compares shared
  coordinates in O(shared dimension); refreshing adds the shared computation.
  Cache storage grows with group-level outputs, not row-capable observations.



## 0.1.48

- Add `carry_smc(completed; basis=:fit, max_reflections=16, maxiter=100)`.
  The public WarmupHMC 0.2.10 moment fitter learns bounded correlation
  directions from the weighted completed cloud, including selected static
  coordinates, and retains its owned data-only diagnostics. No old cloud or
  dense covariance is retained. Finite fitting budgets can lose correlations;
  convergence of fitted directions does not certify covariance accuracy.

- Add `carry_smc(completed; basis=portable_scale)` to fit weighted posterior
  moments in a supplied bounded-reflection basis. The learned orientation is
  retained, while proposal magnitudes are replaced by posterior moments; no
  dense covariance or retained cloud is introduced. Explicit `scale` keeps its
  existing copy-as-given meaning. Clarify that a tuning reference is a normalized
  proposal and need not have posterior covariance.
- Document read-only learning-state positions, weights, retained-window length
  and bounded replenishment/acceptance diagnostics for consumers.

- Add selected-static `carry_smc(completed; parameters=...)` and
  `smc_parameters(carried, u)`. Completed fits with latent histories can
  carry only selected parameter moments and portable native geometry;
  scan coordinates, unused bound data and unused definitions are discarded.
  The filter uses that constrained-parameter mapping by default.
- `prepare_filter_smc` accepts an explicit `calibration_model_id` when a
  consumer's verified batch and streaming identity schemes differ. Both
  identities remain checked and checkpointed, and the ordinary filter's
  boundary identity is preserved.

- Add `prepare_filter_smc`, `tune_filter_smc` and `initialize_filter_smc`
  for bounded-window parameter learning over authored finite-state filters.
  A shared carried Gaussian initializes the ensemble. Predictive scores
  reweight its members; low ESS triggers replenishment and fixed-kernel
  particle-Metropolis moves that rebuild the conditional states. Rejections
  keep the previous likelihood estimate and state cloud. Independent pilot
  kernels remain fixed throughout production.
- Window boundaries explicitly approximate the joint posterior by Gaussian
  marginal parameter moments and an independent pooled state distribution.
  This discards parameter correlations and parameter-state dependence, and
  is not exact full-history SMC². An ordinary state-only filter can supply
  the initial pooled boundary without counting its training likelihood twice.
  Records, diagnostics and data-only checkpoints have bounded retention;
  replay validates source, coordinate, calibration and mapping identities.

- Add `carry_smc`, `SMCCarriedReference` and `prepare_smc_window` for
  bounded-window static-parameter updates. The carried normalized Gaussian
  replaces the old posterior; only the new window's likelihood is evaluated.
  Original priors and coordinate Jacobians are not counted again. Summaries
  retain the coordinate/preprocessing identity and no old observations.
  Correlated portable scales, independent pilot tuning, bounded diagnostics
  and exact checkpoint continuation use the existing SMC engine. This is an
  explicit assumed-density approximation, not full-history posterior replay.

## 0.1.47

- Fit `tune_smc(...; pilot=:adaptive, reference=:pilot)` from retained pilot
  draw moments in the learned reflection basis, with
  `reference_inflation=2.0` multiplying fitted standard deviations. The
  reference no longer reuses the pilot's local mutation scale. Cross-covariances
  in that basis are discarded; inflation does not guarantee posterior coverage.
  Each grid pilot still exports its mutation scale and paired step size unchanged.
- Record the reference pilot's native-coordinate `draw_sd` and inflation in
  `schedule.pilot.reference_work`. Bounded stage summaries add `coordinate_sd`
  for the post-mutation cloud and `reference_sd` for the initial Gaussian
  reference (`nothing` on exact posterior-to-posterior updates), without keeping
  historical clouds or allocating a dense covariance.
- **Breaking:** learned-reference runs change numerically. Setting
  `reference_inflation=1` uses uninflated pilot moments, not the old mutation
  scale. Checkpoint replay requires matching package versions and code; rebuild
  schedules and checkpoints when upgrading.

## 0.1.43

- Require ReactiveKernels 0.1.47 for strict factor queries of ragged censored
  observations sharing array-valued predictors with another in-cell
  observation. Factors preserve empty and repeated rows, structural packed
  reads, and ordinary Reverse derivatives through both censored atoms and
  density terms.

## 0.1.42

- Add independent, pre-production pilot tuning with `tune_smc`, a data-only
  `SMCTuningSchedule` and `SMCOptions(tuning=...)`. The fixed kernel chosen for
  each production stage depends only on temperature and the prepared schedule.
  Schedules validate their exact bridge identities and normalized reference;
  cumulative updates require fresh tuning or explicit untuned options.
  `smc_reference` returns an owned reference for initial bridges. Checkpoints
  retain tuning without rerunning pilots. Bounded stage summaries record the
  selected kernel, evaluation count, mean acceptance and particle displacement.
- Support normalized correlated `SMCReference`s and fixed NUTS metrics from
  WarmupHMC's plain bounded-reflection scale data. `pilot=:adaptive` learns a
  portable active scale with its paired step size at each fixed temperature;
  optional `reference=:pilot` learns and freezes a normalized reference before
  the grid pilots and production. Explicit independent RNG/initial positions,
  bounded geometry/recording pools and data-only checkpoints preserve the
  separation from production adaptation. Requires WarmupHMC 0.2.9.

## 0.1.41

- Preserve row-local coordinate supports for retained conditional cell
  predictors such as `a > 0 ? z[group[i],1] : z[group[i],2]`, including named
  gathered columns and ragged index vectors. Footprints union both arms
  when bound indices prove their reads in range; otherwise the conservative
  whole-block support remains. Values, native Reverse and inactive-error
  semantics are unchanged. Requires ReactiveKernels 0.1.44.

## 0.1.40

- Fix ordinary cell queries for a named summand such as
  `terms = exp.(eta[r1[k]])` followed by `sum(terms)` and
  `sum(eta[r2[k]] .^ 2)`. Both gathers contribute their rows to the shared
  predictor window; earlier versions could omit the second gather's rows
  and throw `BoundsError`. Require ReactiveKernels 0.1.43 for this repair.

## 0.1.39

- Preserve strict cell and partition locality through nested data-only calls
  in sampled compositions. Binding exposes eager data subexpressions and
  follows visible kernels' metadata dependencies through named values, so
  their actual shapes reach downstream concatenations, gathers and designs.
  Helpers run once at their required stage; shared plate captures keep
  preparation staging and lazy expressions retain their evaluation scope.

## 0.1.38

- Require ReactiveKernels 0.1.42. A partition sampler with `partitioned`
  reuses shared-only intermediates across labels and local-coordinate
  changes, refreshing them when shared coordinate values change. Cached
  arrays own their storage, and failed refreshes remain invalid until retried
  successfully. Full gradients retain the shared prefix's pullback.
- Add `shared = false` to `partition_value_and_gradient!` and
  `partition_footprint` for local derivatives only, skipping that pullback.
  Compact destinations follow the local footprint; dense destinations keep
  the full packed length with shared entries zero. This requires the existing
  `partitioned` declaration; separability checks remain in force.
- Accept bare Symbol and Pair partition declarations consistently. Prepare
  matching kernels and AD operators for mixed concrete partition-label types.

## 0.1.37

- Report invalid SMC densities, gradients and density ratios with exported
  `SMCNonfiniteError`, retaining copied exact unconstrained positions, values,
  gradients, endpoint and bridge temperature. Error display is bounded and
  mutation failures propagate directly across the worker task. Initial fits,
  cumulative updates and checkpoint evaluation preserve fail-fast behavior
  and input state/RNG ownership. **Breaking:** callers catching `DomainError`
  for these numerical failures should catch `SMCNonfiniteError` instead.

## 0.1.36

- Add optional finite-state filtering through `prepare_filter`,
  `initialize_filter`, `filter_step`, `filter_stream`, `filter_checkpoint`
  and `resume_filter`. Transitions derive from authored scans, submodels and
  kernel providers. Fixed constrained parameter ensembles retain conditional
  state clouds; state-only warm-up avoids reweighting calibrated parameters.
  Authored parameter transitions support explicitly dynamic models.
- Add `filter_parameter_names` and `filter_parameters` to select fixed
  parameters from nested constrained draws without historical latent paths.
  Bound-plan preparation selects required fixed inputs without retaining
  training columns or freezing newly computed covariates. Frozen preprocessing,
  temporal design rows and fixed-vector products, zero-state models,
  selected fixed groups, predictive scores, bounded
  diagnostics, explicit RNG ownership and portable continuation are covered.
  Work and retained size remain bounded in elapsed time for fixed particles,
  group count and finite lag depth. Requires ReactiveKernels 0.1.41.

## 0.1.35

- Preserve natural scalar expressions in strict partition queries through
  ReactiveKernels 0.1.40. Scalar offsets beside a design product retain their
  broadcast shape instead of passing through an opaque row-normalization
  helper. Shared predictors may have a different row count from observations;
  ordinary indexed `exp` and subtraction need no model annotations.

## 0.1.34

- Add optional SequentialMonteCarlo/WarmupHMC integration: `prepare_smc`,
  `initialize_smc`, `smc`, `smc_step`, `update_smc`, `smc_checkpoint` and
  `resume_smc`. Adaptive bridges start from a scored normalized diagonal
  Gaussian reference or a previous posterior for cumulative data. Fixed NUTS
  moves preserve the target; priors and coordinate Jacobians occur once.
  Explicit RNG state, stage checkpoints, bounded diagnostics and ancestry
  use linear-size storage. This first implementation runs particles serially
  and replays complete data histories after parameter changes.

## 0.1.33

- Require ReactiveKernels 0.1.39 for local direct reads in sum-factored
  cells. An outer density combining independent sum inputs with scalar
  predictor reads keeps only those predictors' packed coordinates, including
  when several sums and direct reads are composed from an earlier plate.
  Factor preparation passes its locality policy to the recovered row reads.

## 0.1.32

- Require ReactiveKernels 0.1.38 for visible-kernel metadata dependencies.
  Data-only tuple components feeding completed-covariate gathers now bind
  once before shape analysis. Strict cell queries retain local parameter
  reads through the completed column, design product and gathered effect,
  matching the same model supplied with those arrays directly.

## 0.1.31

- Require ReactiveKernels 0.1.37 to identify visible kernel plate domains.
  Data-only definitions such as `ids = entry_axis(ends)` now bind once before
  shape analysis, so grouped kernel columns with computed domains retain
  strict cell locality through a design matrix and offset. Domain aliases
  and ragged groups are supported; other data-only captures keep their
  existing preparation-time staging.

## 0.1.30

- A partition's plan holds the bound data at the label's rows too (design
  blocks included), and the compact sampler reads a packed block's rows from
  its active vector at slots computed once per label from that label's slot
  map, not through the overlay's per-element slot load and branch; a
  partition reading a packed coordinate outside its footprint is refused by
  name at preparation. Requires ReactiveKernels 0.1.36.
- Grouped compact partitions on Julia 1.10 use ordinary Enzyme Reverse with
  dogfood `a6458e9c` (constant-field repairs after `1280e6ee`). On the BRM
  `4e53855` nested panel (42,768 rows, 12 regions), compact evaluation and
  gradient take 2.55 ms versus 3.47 ms dense and 8.84 ms for BRM's unit path;
  the whole sweep's per-unit share is 1.15 ms (compact 2.22×). These are
  interleaved minima over 15 rounds on gordito, Julia 1.10.12, one Julia
  thread, libEnzyme `782c9d95`; values and gradients pass independent checks.
  The 1.5× target remains unmet. The earlier 2.04× measurement used runtime
  activity and is not comparable ordinary-Reverse acceptance evidence.


## 0.1.29

- Require ReactiveKernels 0.1.35 for graph-visible shape propagation.
  A grouped vector plate followed by an entry gather can feed
  `lx = log.(entries)`, `X = hcat(one, lx)` and `mu = X * beta .+ offset`
  in `prepare_cell_query(...; locality = :error)`. Each cell reads its
  owning group and the shared coefficients. Previously lost shape facts
  inserted a whole-vector normalization and strict preparation failed.

## 0.1.28

- Fix outer `prepare_cell_factor_ad` preparation for exactly one sum.
  Its active sum exemplar is always a vector, matching the documented
  `(sums, direct_active)` directions; singleton kernel output unwrapping
  previously produced a zero-dimensional array and rejected vector HVP
  and third-order directions before differentiation.

## 0.1.27

- `ParameterGeometry` accepts function-shaped `@kernel` endpoints in its
  existing `constrain`, `unconstrain` and `logjac` slots. Host transforms retain
  prepared callables; emitted transforms compose the authored graph at a full
  argument boundary, including explicitly supplied keywords. This exposes
  identity and row-local vector transforms to strict partitions while preserving
  actual coupled dependencies and the caller's joint prior. Scalar cells
  and joint cell latents use the same endpoints. Printed `kernel_expr` source
  reconstructs the geometry through `sampling_geometry`, including parametric
  law constructors. Ordinary callbacks,
  omitted defaults and splatted arguments retain their numerical call path.
  Requires ReactiveKernels 0.1.33.

## 0.1.26

- `prepare_cell_factor_query(f, :outer)` and
  `prepare_cell_factor_query(f, j)` return scalar `CellFactorQuery` objects
  for the outer density and one summand, before compact AD indices are
  known. Call `outer(u, sums, cell)` or `term(u, cell, term_position)`.
  Recording `AbstractVector{Float64}` points pass through without copying;
  the public `kernel` supports RK's structural footprint API for discovering
  each factor's packed-coordinate selection, including retained branches.
  This completes the value/support preparation path for the sum-factored
  derivatives added in 0.1.25.

## 0.1.25

- `prepare_cell_factors(built, plan, observation)`, `cell_reductions` and
  `reduction_rows` discover a cell's sums, including inline expressions,
  local names and earlier plate outputs. `prepare_cell_factor_ad` prepares
  any RK derivative operator for the outer density (`:outer`, active over
  the sum vector and selected direct packed coordinates) or one summand
  (a reduction index, active over selected packed coordinates). Calls use
  the ordinary RK derivative verbs and compact coordinate selections.
  Empty reductions make no summand calls; no-sum cells have an ordinary
  outer density. The pieces support Hessian and third-order chain rules for
  sparse Laplace without a dense block over every coordinate a sum reads.
  Requires ReactiveKernels 0.1.32.

## 0.1.24

- Document and test the corrected upstream `DirichletMultinomial` dependency
  used by the optional Distributions adapter. The repository acceptance setup
  pins Distributions commit `83cec5e3a8a35167d54f19da9dc5afb54f59273e`, whose
  `logpdf` rejects invalid counts and mismatched totals before evaluating the
  density. Native Enzyme gradients on valid observations are unchanged.
  Other environments must select that dependency revision explicitly;
  upgrading RKPPL alone does not repair Distributions 0.25.132. The adapter
  continues to delegate directly to the distribution's public `logpdf`.


## 0.1.23

- `prepare_partition` and `prepare_partition_sampler` (dense and compact)
  prepare the partition with a plan port: each label's plan (ReactiveKernels
  `partition_plan`) is computed once at preparation (`PartitionQuery.plans`,
  `PartitionSampler.plans`) and passed to every evaluation and gradient, so a
  call recomputes no row bookkeeping. On registered Enzyme (0.13.216) the
  gradient of a planned partition needs runtime activity
  (`AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse))`): the
  plan's row vectors meet the active vector in the body's views, which
  Enzyme's static activity analysis refuses (Enzyme snag
  `plan-const-int-r-4671f4a7`). Requires ReactiveKernels 0.1.28.
## 0.1.22

- `prepare_compact_cell_ad(prepare_operator, built, plan, observations, base0,
  indices0, backend, exemplars...; on_error, locality)` and `CompactCellAD`:
  a ReactiveKernels derivative operator (`prepare_ad_hvp`, `prepare_ad_third`,
  `prepare_ad`, `prepare_ad_pullback`, `prepare_ad_pushforward`) over one cell's
  log density, active over a compact selection of packed coordinates, as
  `prepare_compact_cell_sampler` is for the gradient. The ReactiveKernels
  verbs take it with `base, active, indices, cell` last:
  `ad_gradient_and_hvp(q, directions, base, active, indices, cell)` gives the
  cell's `m × m` Hessian block over its `m` selected coordinates from one
  call with the `m` unit directions, and
  `ad_gradient_hvp_and_third(q, (v,), base, active, indices, cell)` the
  third-order contraction `∇(vᵀ H v)` over the selection. Each call costs one
  cell, not the model. Static (default-activity) Enzyme needs Enzyme
  `dogfood` from `21a909f7`; registered Enzyme 0.13.216 raises
  `EnzymeRuntimeActivityError` at the compact overlay's read (Enzyme snag
  `static-reverse-s-f9483e9c`).

## 0.1.21

- `prepare_partition(built, plan, observations; by, locality = :error)`,
  `PartitionQuery`, `partition_labels`, `prepare_partition_sampler(built, plan,
  observations, u0; by, backend, locality)`, `PartitionSampler` and
  `partition_value_and_gradient!(s, g, u, label)`: one partition of the named
  observations (all cells whose key equals a label, such as one subject's)
  as one compiled pass, with its value and dense gradient. `by` names a
  supplied data column with one entry per cell of each observation's plate
  (one name, or `observation => column` pairs). Requires ReactiveKernels
  0.1.24.
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
