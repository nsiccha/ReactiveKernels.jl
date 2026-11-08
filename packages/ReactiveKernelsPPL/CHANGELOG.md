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

## 0.1.127

- Requires ReactiveKernels 0.1.115. A dotted cell latent's cell reads only its
  own block (0.1.126) also when the cell reads a value computed from
  parameters, such as a location `mu = mu_Intercept .+ tau .* z[g]` read as
  `mu[i]`, in one cell or written to a plate output that a second plate
  observes (BRM's emitted program). Before, every index's footprint held every
  index's coordinates of the latent there (snag `dotted-cell-late-3789b8ea`).
  Values and ordinary Reverse gradients are unchanged
  (`test_cell_vector_latent_footprints.jl`).

## 0.1.126

- A dotted cell latent, `x[i] .~ D.(args…)` in a `@plate` cell, is read in
  its cell through a view of its own block of the packed vector, indexed by a
  bound table of unit ranges, as a joint cell latent's block is. A strict
  cell query's footprint, a partition's footprint (`prepare_partition(…; by)`)
  and their compact gradients then hold that index's coordinates only. Before,
  every index read every index's coordinates, so a consumer that assigns
  coordinates to units by footprint (BRM's SVI and NLME views) kept them all
  global, and a compact gradient per label carried the whole latent. Values,
  ordinary Reverse gradients, draws and the emitted program's independence
  of the data's size are unchanged (`test_cell_vector_latent_footprints.jl`).
- Not built yet: a latent whose prior transforms its coordinates (a
  positive or bounded family, such as `LogNormal`) still reads every index's
  coordinates. Its cell views the constrained vector, a plate over the whole
  latent, and the cell windows do not compute that plate at a view's rows; a
  top-level `view(x, r[i])` of such a vector reads all of it the same way.

## 0.1.125

- Fixes wrong Reverse gradients from 0.1.124 for multivariate normal
  densities. The value was right; the gradient lost the density's
  contribution, or returned uninitialized memory for it. Measured cases:
  - slice priors `eachrow(B[…]) .~ MvNormal(mu, Sigma)` and
    `.~ MvNormalCholesky(mu, F)` beside a `@plate`, such as correlated group
    effects with a sampled scale and LKJ factor (the slice prior alone was
    right);
  - `MvNormal` with a sampled covariance, observed in a plate cell or as one
    vector (a Gaussian process, with no plate).

  0.1.124 handed fresh `undef` storage for the slices' solved margins and the
  covariance factor to a helper Julia does not inline, a shape Enzyme
  differentiates wrongly here (Enzyme snag `rkppl-mvnormal-s-1df3d9b3`). The
  helpers now start from zeroed storage. Values are unchanged. The
  non-allocating sampler still reuses those buffers, and its gradients are
  now right too. A dependant that fits such models with 0.1.124 should take
  0.1.125 as its floor.

## 0.1.124

- With MutatingFunctions loaded, through the new weak dependency and
  `ReactiveKernelsPPLMutatingFunctionsExt`, the non-allocating sampler
  (`prepare_sampler(...; nonallocating = true)`) also reuses buffers for:
  - the simplex transform, the vector and its log-Jacobian;
  - the ordered transform;
  - an `MvNormal` covariance's Cholesky factor and the slices' solved margins.

  A warmed density (`q.kernel(u)`) and gradient then allocate nothing for
  Dirichlet parameters, ordered vectors and `MvNormal` likelihoods, as well as
  for the shapes of 0.1.119. Values and gradients are unchanged.
- The `MvNormal` factor zeroes its upper triangle before the factor loop
  rather than inside it. Native Enzyme reverse otherwise put the loop's reads
  on its tape, 704 B per gradient at K = 20.
- One vector's slice rows are a reshaped view rather than a reshaped
  `Vector`, which builds an array object.
- The ReactiveKernels compat floor is 0.1.113.

## 0.1.123

- A compact authored-factor query (`prepare_query_ad(...; coordinates)`) keeps
  much less per query. The queries of one selection structure now call that
  structure's one prepared program, with their own selection (cell
  memberships, row plan, coordinate slots, folded constant) as call arguments.
  Before, each query rebound its own copy of the program: a kernel, plan and
  constants table per query. A dense query (no `coordinates`) still rebinds per
  selection. Values and gradients are unchanged.
- **Breaking** for a custom `prepare_operator` of a compact query: it now receives the selection's
  arguments after `(base, active)` and must pass them on, as `prepare_ad` and
  the other ReactiveKernels operators do. The query's own verbs
  (`ad_value_and_gradient(q, base, active)` and the rest) append them. A direct
  call of `q.prepared` passes `q.context` after `(base, active)`.

## 0.1.122

- Windows of one recurrence, fitted apart, can be joined through their boundary
  state (multiple shooting): `prepare_joined_windows(built, plans, us; boundary
  = :h_out => :h_in, backend)` takes one binding of a program per window, in
  time order, and returns a `JoinedWindows`. `h_in` is a declared parameter the
  window's `@scan` starts from (`h[1] = h_in`), `h_out` an authored value, the
  state at the window's end (`h_out = h[T]`, or a per-index output of a `@plate`
  of recurrences). The joined log density feeds each window's output to the
  next window's input entry with the same `coordinate_identities` (a level label
  or the position), drops a fed entry's prior and Jacobian, and counts every
  coordinate held by an earlier window (a global, a unit's static) once, so
  windows that cover a series one after another have the series' value and
  gradient; gradients flow back through each output. `joined_value_and_gradient!`,
  `joined_point`, `window_points`, `joined_coordinates` and `joined_problem` (a
  `LogDensityProblems` target, `JoinedWindowsProblem`) read and evaluate it. A
  scan's sampled steps, the inputs and the parameters named in `locals` belong
  to their window only. Not built: a plate of stochastic recurrences (the
  density catalogue below refuses its cell scans), boundary inputs with joint
  supports (a simplex) or bounds that read parameters.
- `prepare_query(built, plan, (:a, :b))` evaluates authored values by name (a
  definition, a carried array, a per-index output), the tuple of their values
  (one name gives its value itself), and `prepare_query_ad` takes the same tuple
  for pullbacks and pushforwards.
- `density_factors` catalogues a model with a top-level non-centered `@scan`
  (a deterministic or assigned seed, several carried arrays, innovations): the
  scan's sampled seeds and steps are one atomic prior event owning its packed
  block, as a joint law is. It used to refuse the whole model, and so did
  `prepare_slices` and selective SMC windows built on it.
- A binding whose level labels differ from the build's, with the same packed
  shape, now keeps its own labels in a query's layout (`q.layout` of
  `prepare_sampler`, `prepare_slices`, and so `coordinate_identities` of it).
  It used to report the build's labels, so a unit could be matched to another
  binding's unit at the same position.

## 0.1.121

- A simplex whose `Dirichlet` concentration is bound data, such as
  `s ~ Dirichlet(alpha)` with `alpha` bound beside the data or BRM's monotonic
  effect with its concentration as an input, keeps one program for every
  concentration length. The packed coordinates already read `length(alpha)`;
  the concentration check now reads it too, where it used to write this
  binding's length as a number, so a build for 3 levels served no binding with 4.

## 0.1.120

- A partition query or sampler (`prepare_partition`,
  `prepare_partition_sampler`) keeps less per label. A label's plan held its
  rows and gathered data once for every statement reading them, and a
  shared-prefix sampler keeps two plans per label; on BRM's minibatch SVI
  target for a toy with one station per unit, one label's four plans held 82
  small arrays with 11 distinct contents. Equal arrays of the plans of one
  preparation are now one object (ReactiveKernels 0.1.107). The plans are also
  stored once per label, in a label-indexed `AbstractDict` (`q.plans` and
  `s.plans` read as before), where a `Dict` kept about two inline slots of each.
- A density catalogue (`density_factors`) no longer keeps a copy of its
  binding's data. It kept one to refuse a selection's query after an in-place
  change of the bound data, and now keeps a hash of every element of it
  instead, which refuses the same queries.
- Measured on that SVI target (BRM `2c9500f4`, 128 to 512 units, live heap with
  each object counted once): the partition sampler from 5.6 to 3.1 KB per unit
  (6.4 to 2.7 KB with `field_states = :units`), its value query from 2.8 to 1.8
  KB, the model's own value query from 2.0 to 1.1 KB, and the catalogue's copy,
  2.5 KB per unit, is gone. Values and gradients are unchanged. Per-unit factor
  queries (`prepare_query_ad(...; coordinates = :support)`, about 7 KB per unit
  there) are unchanged.

## 0.1.119

- `prepare_sampler(...; nonallocating = true)` with ReactiveKernels 0.1.106
  allocates nothing per call, after warm-up, in its density kernel
  (`q.kernel(u)`) and its gradient (`sampler_value_and_gradient!`) for a
  design-matrix product, a coefficient vector of scalar parameters, a `@scan`
  trajectory, a ranged response gather (a plate over `eachindex(y)`) and an
  interval-bounded vector. Values and gradients are unchanged. The gather
  `_ppl_range_values` is `@traceable`, so the non-allocating kernel gathers
  into a buffer of its own. Still allocating: the simplex and ordered
  transforms and an `MvNormal` covariance's Cholesky factor. `q(u)` itself
  returns its number through the world-age barrier, which boxes it (16 B).
  The ReactiveKernels compat floor is 0.1.106.

## 0.1.118

- A declared `L ~ LKJCholesky(K, eta)` factor whose dimension `K` reads data
  (a bound integer such as `channels`, `length(channels)`, `size(M, 2)` or a
  definition) is read by the program: its packed range, the `K(K - 1)/2`
  partials count (`_ppl_lkj_count_L`), the transform, Jacobian and prior loops,
  the prior's normalizer, and the positions of every block packed after it.
  Every `K` of one declaration now generates the same `kernel_expr` and one
  `build_kernel` serves every binding, conditioned factors included. Before,
  the program spelled `K`, so each dimension generated its own program and
  shifted the literal positions of the blocks after it. A literal `K` keeps
  the program it always had. With a literal `eta`, a data `K` computes the
  normalizer in the program (`lkj_logconst`'s general formula, which equals its
  `eta == 1` branch to rounding). This removes "an LKJ array's dimension" from
  the 0.1.108 list of paths that stay literal; per-level LKJ stacks keep their
  literal `K`, and LKJ scalar edges and stacks keep their literal coordinates.
- **Breaking:** a raw `ReactiveKernels.prepare(built.spec; have = (:unconstrained,),
  want = :L)` cut of a factor whose `K` reads data now needs that data, as a cut
  of a data-sized array already does: name it in `have` or pass it as `bound`
  (`rkppl-use` section 14). A literal `K` needs nothing new.

## 0.1.117

- **Slices of one bound model.** `prepare_slices(built, plan, observations, u0;
  by, backend)` prepares the model restricted to the cells whose label (as
  `prepare_partition_sampler`'s `by` gives it) is in a chosen set, for example
  one region of a national binding.
  - A slice's density is its cells, plus every authored prior and Jacobian term
    that owns a coordinate they reach, with the coordinates those terms read added
    in turn. Each term counts once, however many labels reach it.
  - `slice_coordinates(slices, labels)` gives the slice's packed positions in
    the whole binding; nothing outside them is read or differentiated, and other
    labels' cells are not evaluated.
  - `slice_value_and_gradient!(slices, g, u, labels; compact = true)` evaluates
    one label or a collection of them. `slice_problem(slices, labels; base)` is a
    first-order `LogDensityProblems` target over the compact coordinates, through
    the new `ReactiveKernelsPPLLogDensityProblemsExt`.
  - The binding is prepared once, and another label set reuses its compiled
    kernels.
  - For units nested in the labels, a label's slice equals the model bound to
    that label's data alone, in value and gradient.
  - A density that owns no coordinate and reads coordinates both inside and
    outside a slice is refused for that slice with a `ContractValidationError`.
- **`coordinate_identities(layout)`.** It names each packed coordinate by its
  authored path and an index holding the bound label of a `levels(g)`,
  `unique(g)` or data-vector axis, or the level at each position of the
  positional twin `1:length(levels(g))` (BRM's group effects); other axes,
  `1:length(levels(g)) - k` included, keep positions. The same coordinate of
  two bindings therefore compares equal where `coordinate_names` gives its
  position. `LayoutTable` keeps a declared array's bound axis labels for this
  in a new last field, `axis_labels`; its existing constructors are unchanged.
  `coordinate_names` is unchanged.
- **A factor selection of many sites prepares a concretely typed program.**
  `prepare_query` and `prepare_query_ad` add a selection's site totals two at a
  time, left to right, instead of in one `+` over all of them. Julia does not
  infer Base's varargs `+` past 32 operands, so a selection of more sites (a
  slice's prior on a real model) left a call dispatched at run time in the
  prepared program, which RK warned about. The sum and its rounding are
  unchanged.

## 0.1.116

- Authored-factor queries (`prepare_query` and `prepare_query_ad` over
  `select_factors` selections) prepare one program per selection structure of a
  catalogue, meaning the same kinds and sites, each read whole or by cells, and
  rebind it for each selection. A query keeps its own entries, row plan,
  coordinate slots and derivative workspace. Neither the query nor the binding
  keeps a copy of the model graph, which each selection used to leave in the
  binding (about two per selection, for the binding's lifetime). On a toy with
  one scalar prior per unit, 1,024 per-unit
  `prepare_query_ad(...; coordinates = :support)` queries kept 4.3 MB instead of
  159 MB and prepared in 7.8 ms per unit instead of 70 ms, with the same values
  and gradients.

## 0.1.115

- An ordered or threshold vector's length may be a bare bound name, as an array
  axis and an `LKJCholesky` dimension already may: `c ~ Ordered(Normal(0, 1), n)`
  and `t[1:n] .~ Normal.(0, 1)` with `n` an integer bound beside the data, read
  by `OrderedLogistic`, cumulative and stopping-ratio `Ordinal` responses. Before,
  lowering refused it with "length must be a nonnegative integer or a data-only
  expression", although `Int(n)` passed. The program reads `n` from the binding,
  so one build serves every count. A fractional, negative, Boolean or sampled
  value fails at binding.

## 0.1.114

- `prepare_partition` and `prepare_partition_sampler` lower their partition
  kernels from the exemplar point (`u0`) and the first label, as
  `prepare_cell_query` already lowers its kernel from `u0`. A value whose
  producer branches on a type, such as BRM's `flatten_cells`
  (`eltype(cells) <: Number ? cells : reduce(vcat, cells)`), now takes its typed
  arm and is computed at each partition's rows. A response that reads such a
  collected plate value whole, `y .~ Normal.(flatten_cells(cells), sd)`, then
  reads only its own partition's coordinates, as the cell query's footprint of
  the same cells already said. Before, the value was computed whole:
  `locality = :error` refused it, and with `partitioned` each partition read
  every unit's coordinates of that value and was refused with
  `PartitionSeparabilityError`.
- **Breaking:** a `PartitionQuery`'s `q.kernel` is typed for a
  `Vector{Float64}` point and the first label's type, as a `CellQuery`'s kernel
  is. `q(u, label)` still takes any point and label type, preparing a kernel for
  a new pair once, and so does
  `ReactiveKernels.partition_footprint(q.kernel, u, label, nothing)` (ReactiveKernels
  0.1.104). A direct `q.kernel(u, label, plan)` call and a held
  `partition_footprint_kernel(q.kernel)` need that point and label type: call
  `q(u, label)` or `partition_footprint` instead. A key column with no label is
  refused by name.
- Requires ReactiveKernels 0.1.104.

## 0.1.113

- An external law's packed range and the shape its `constrain` and `logjac`
  endpoints receive are read from the data, like a declared array's: BRM's
  GMRF field `field[1:support.geometry.ncoordinates] ~ law(…)` reads
  `unconstrained[4:_ppl_last_field]` and `(_ppl_axis1_field,)`, where 0.1.108
  spliced this binding's `unconstrained[4:15], (12,)`. Every
  station and read count of such a field now generates the same `kernel_expr`,
  and one `build_kernel` serves every binding, the empty field included. The
  same holds for a `@plate` of open-RHS latents over data
  (`@plate for i in eachindex(x)`). Snag `gmrf-field-packe-ab48cb09`.
- The endpoints still receive a fresh `Vector`. For a host (non-`@kernel`)
  endpoint, a data-read range copies with `collect(view(…))` rather than
  `unconstrained[a:b]`: when that range is empty at run time, Enzyme's
  `ReverseWithPrimal` returns a wrong value (not gradient) for `getindex` and
  `copy(view(…))` there, so `prepare_sampler` would report a wrong density for
  an empty field (Enzyme snag `enzyme-autodiff-e13097c4`). Composed `@kernel`
  endpoints and plates keep `unconstrained[a:b]`, whose rows cell and partition
  queries follow.
- An extent read through a property of bound data, `z[1:support.geometry.n]`
  or a geometry's packed count `:($(args[1]).geometry.n)`, reads that value.
  0.1.108 took the dotted read for a module path, which names no data, and so
  kept the literal.
- Of the paths 0.1.108 lists as literal, an external law's geometry shape and
  size no longer are, except when the extent reads an observation.

## 0.1.112

- `prepare_sampler(...; nonallocating = true)` keeps one type across the values
  of a bound number, as the ordinary sampler does since 0.1.105. Rebinding one
  build with other numbers reuses the compiled density and gradient instead of
  compiling them again, and each query still owns its buffers. Requires
  ReactiveKernels 0.1.103.

## 0.1.111

- `partitioned = :none` declares that partitions own no parameter, for
  `prepare_partition`, `prepare_partition_sampler`, `prepare_cell_query`,
  `prepare_cell_factors` and `prepare_compact_cell_ad`. Every coordinate is then
  shared, as with an all-`missing` label vector on some parameter, which keeps
  working. The default `partitioned = ()` still declares nothing, so strict
  locality still refuses a value with no row form. A parameter named `none` makes
  the bare symbol ambiguous and is refused; `(:none,)` still names that parameter.
- `partition_batch_value_and_gradient!(s, g, u, labels; weights = nothing)` gives a
  batch of partitions' summed value and its gradient over the whole packed vector,
  running one pullback through the shared values instead of one per label. Sweeping
  400 units of one space-time toy took 8.9 s through `partition_value_and_gradient!`
  and 69 ms this way, with the same values and gradients. A whole-model gradient
  took 23 ms there, so a batch is for a subset of the units. It requires native
  `AutoEnzyme` reverse. `PartitionSampler` gains a type parameter.
- Requires ReactiveKernels 0.1.100, whose refusal names these routes.

## 0.1.110

- One `build_kernel` now also serves a binding with other level labels, or
  another number of them: `prepare_query`, `prepare_sampler` and the cell and
  partition preparations no longer refuse it with "needs its own
  `build_kernel(plan)`". The program reads the labels from the binding, so the
  build was already correct for them; only the check compared their values. Each
  query reads this binding's layout, so a factor term's coefficient names
  (`coordinate_names`, such as `mu.g_b`) are this binding's. A path that still
  holds a literal sized by the labels, such as a legacy coefficient width or a
  response-derived threshold count, generates another program and is still
  refused.
- A selection of levels that runs to the end, `c[levels(g)[2:end]] .~ …`, reads
  its extent from the binding like every other level axis. Before, its size
  stayed one binding's literal: the check that a size reads data took `end` for
  a name, and the bind-time evaluator could not evaluate it. Both now treat
  `end` and `begin` in an index as Julia does.

## 0.1.109

- Lowering one source twice gives one set of generated names (a selected
  argument, an index, a destructured value, a density port), so with
  ReactiveKernels 0.1.99 two builds of one model in a process share their
  types and the second compiles nothing more. A name is a function of the
  source and its position in the lowering, instead of `gensym`. A submodel
  called twice still has its names apart: one lowering counts once over its
  submodels. Requires ReactiveKernels 0.1.99.

## 0.1.108

- Sizes and data-derived numbers stay data in the generated program, never one
  binding's literals. A packed block reads its range from the data the program
  already has: `z[1:n] .~ LogNormal.(mu, sd)` is `view(unconstrained, 1:n)`, the
  next block starts at `n + 1`, `Z[1:r, 1:k]` is `reshape(flat, r, k)`, and
  `s ~ truncated(Normal(0, 1), floor, Inf)` is `floor + exp(u)`. Every extent of
  one declaration and every value of such a bound now generates the same
  `kernel_expr`, and one `build_kernel` serves every binding. No input is added,
  so a cut needs the data it always did, and a model whose positions depend on
  no data generates the program it always did. This covers declared arrays
  (elementwise and slices), `@plate` latents over data, scans (centered and
  non-centered, with their prior bounds), threshold vectors, an ordered
  response's level count, and the positions of coefficient, GLM, external and
  LKJ array blocks. Requires ReactiveKernels 0.1.97.
- The prepared query type is the same whenever the data take the same arms: RK
  still splits a plate's data-bound branch by the arms its lanes take, so an
  empty likelihood, or an ordered response with no interior level, prepares its
  own type from the same program. Other level labels keep the program and its
  type, and still ask for their own `build_kernel` (`_program_identity`).
- What stays literal: any size read from an observation, since it can be absent
  in a design binding (a `@plate` latent over `eachindex(y)`, thresholds counted
  from `levels(y)`); a read that does not evaluate to the size binding packs;
  legacy coefficient and GLM block widths, and so the scalar coordinates packed
  after them; positive vectors and LKJ scalar edges and stacks, whose statements
  are one per coordinate; an LKJ array's dimension; an external law's geometry
  shape and size; a plate latent over a computed value; a transform's kind when
  a bound's value decides it (an infinite data-derived bound); and a number the
  model's author wrote. Density-factor queries keep their own literal row counts
  and windows.

## 0.1.107

- A `partitioned` cell family (`prepare_cell_query`, `prepare_cell_factors`,
  `prepare_compact_cell_ad` and their operators) works when its shared prefix
  reads only some shared coordinates. Since 0.1.102 such a family threw
  `BoundsError` at an unchanged point, or, when the coordinate the prefix reads
  came later in the packed vector, counted every call as a shared change, so
  any raw call ended a live `with_cell_batch` or `cell_pullback!` pass. Values
  were never stale.
- `prepare_compact_cell_ad` with `partitioned` reads a shared coordinate its
  residual reads at rows from the base, as a local partition does. Since 0.1.102
  its derivatives otherwise refused such a coordinate as outside the footprint,
  although a compact selection may hold only partitioned coordinates. A prefix
  that reads every shared coordinate compiles as before. A cell that also reads
  a block at rows can still raise `EnzymeRuntimeActivityError` under static
  Reverse until Enzyme carries the repair of its read of base and active rows
  (Enzyme snag `static-reverse-m-c913effd`).
- With a prefix, a pass and a raw call's comparison now follow the coordinates
  the prefix reads, as its cache does since 0.1.102. Changing a shared
  coordinate the prefix does not read leaves a live pass valid; the pass reads
  its own copied point, so its results are unchanged. A family with no prefix
  still counts every shared coordinate.
## 0.1.106

- `prepare_sampler` has one type across the values of a bound number, such as a
  prior's location or scale, as `prepare_query` already had. Rebinding one build
  with other numbers reuses the compiled gradient instead of compiling it again;
  another row count does too, on the same gather schedule. Requires
  ReactiveKernels 0.1.95. Values and gradients are unchanged.
- A gather with a bound response scale, such as `y .~ Normal.(z[idx], noise)`,
  gives up a gradient speedup it only had while the number was spelled in the
  generated code: `log(noise)` now runs on every row there, as it already did in
  the density and in programs without a gather (ReactiveKernels 0.1.95 has the
  numbers; snag `bound-response-s-9739d33d` tracks lifting it out of the loop).
  A prior's numbers are computed above their loop and cost about 6 percent.

## 0.1.105

- Binding refuses a fixed `nu` or `zi` outside its support on every spelling. A
  `ZeroInflatedPoisson.(rate, z)` or `StudentT.(nu, mu, s)` column, in a dotted
  observation or a `@plate` cell, a computed argument (`z ./ 2`) and a definition
  (`zz = 1.5 .* z`, `zz = z .- 0.1`) built before, and the likelihood was `-Inf`
  at every coordinate; only a scalar was refused. They now fail
  `[y_resp] zi data z must lie in [0, 1]` or
  `[y_resp] nu data nu must be finite positive numerics`, over the rows the
  response observes. An infinite `nu` column, which gave a finite density, is
  refused as a scalar and a literal already were. A value that reads a sampled
  name keeps its lazy `-Inf` branch, and values in support have the same density.
- A scale fixed by an offsets-only definition (`sd = w .- 3`) is refused the same
  way, and a refusal of a computed scale says what was written instead of
  `_rkppl_synth_1`.

## 0.1.104

- Each observation's pointwise density plate declares its cell's density as
  `Float64`, as a grouped observation's totals plate already did. A partition
  of several observations (`prepare_partition`) then knows each observation's
  sum type even when its density comes from a helper or a censoring branch
  that inference cannot type, so with a ReactiveKernels release that guards a
  partition sum without cells (0.1.90 and later) a unit with no cell of such
  an observation neither computes nor reads the work only that observation
  needs. Values are unchanged.

## 0.1.103

- `prepare_partition` and `prepare_partition_sampler` with `partitioned`
  (a cached shared prefix) evaluate a joint vector observation cell, such as
  `y[i] ~ MvNormal([mu[left[i]], mu[right[i]]], C)` in an `@plate for` loop,
  again: every call refused the label's own plan with `this kernel was
  prepared from exemplar argument types …`. Requires ReactiveKernels 0.1.91.

## 0.1.102

- Partition sampler gradients with `shared=false` differentiate a compact
  conditional program over local coordinates. They reuse shared prefix values
  without allocating cotangents for that frontier, and mixed shared/local
  parameter reads materialize only the selected rows. Full gradients retain
  the shared-prefix pullback; compact and dense destinations keep their existing
  values and coordinate order. Requires ReactiveKernels 0.1.89.
- Shared-prefix active storage and cache invalidation follow the coordinates
  actually read by that prefix. Shared parameters read only at selected rows
  remain in the compact residual instead of creating whole-panel cotangents.
  Auxiliary scalar predictor cells use the same compact slot map as the
  observation partition, avoiding a shadow of the full packed point.
  Conditional row reads gather their base rows before mixing them with active
  coordinates, so the mixed call retains only those rows in either AD direction.

## 0.1.101

- Add `prepare_discrete_latents(built, plan)`, `discrete_latent_draws` and
  `discrete_latents_node`: the posteriors of the discrete latents a program
  sums out, at given parameters, and exact draws from them. A whole-event
  hidden Markov observation (`y ~ LogDensity(normal_hmm_logdensity, …)` and
  the other RK hidden Markov laws, observed or conditioned) gives its state
  path's posterior; a plain `MixtureModel` response gives each observation's
  component probabilities. The graph computes the posteriors; drawing happens
  in plain Julia at the query boundary. Other summed-out latents are refused,
  naming them. Requires ReactiveKernelsDistributionKernels 0.1.4.

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
