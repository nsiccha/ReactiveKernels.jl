# Changelog

All notable changes to ReactiveKernels.jl are documented here, newest first.
The nested packages under `packages/` keep their own versions and changelogs.

Every landing that adds or changes public API (an export, a macro, a keyword,
a method, or documented behaviour a dependant can rely on) bumps the patch
digit in `Project.toml` and adds its entry under a new heading at the top of
this file. An entry that breaks something a dependant relies on says
**Breaking:** and what to change. The minor digit, the breaking one in
[semantic versioning](https://semver.org/) before 1.0, has not moved yet.
A package or app that calls that API states the version as its floor, for
example `[compat] ReactiveKernels = "0.1.2"`. An environment that pins an
older ReactiveKernels commit is then refused when it resolves, with
`Unsatisfiable requirements detected for package ReactiveKernels`, instead of
failing or silently behaving differently at run time. Changes before 0.1.2
are recorded in the git history only.

## 0.1.167

- `jax_source` translates explicit `string(...)` calls through the same path
  as string interpolation. Shape guards can build their diagnostic from
  names and dimensions in recorded or provider-supplied helpers, including
  host `prepare_data`; a failing guard retains its message. Symbol literals
  in plain lowered helpers use the same Python representation as recorded
  source.

## 0.1.166

- `jax_source` with `data`: `prepare_data` computes `axes(x)` of a data array
  (the tuple of its ranges) and `Base.Broadcast.broadcast_shape` of such
  tuples, so a data-only shape that a value's axes give, one per plate cell
  say, runs on the host. Both used to raise `JAXEmissionError` there (`axes`
  without a dimension; a function with branches). Device code is unchanged
  (snag `jax-source-of-a-30375424`).

## 0.1.165

- `jax_source`: a plate's cell knows a shared operand that is known while JAX
  traces (a prepared integer, a literal, a shape) as such, as the code
  outside the plate does. `jax.vmap` passes an operand it does not map to
  the cell unchanged, so a prepared integer `H` stays its `StaticInt` there,
  also under `jax.jit`. A cell recipe that reads a parameter beside `H` now
  translates: a range `2:H` (a generator `sum(b * x[t, j] for t in 2:H)`),
  an allocation `fill(b, H)` and a `@traceable` helper's loop to `H` raised
  `JAXEmissionError` (`a range whose length depends on data`, `an array whose
  length depends on data`, `a loop whose trip count depends on data`)
  because the cell took `H` as a traced value. A data-only cell recipe such
  as a scan's `2:H` translated since 0.1.154 by running in `prepare_data`;
  it now stays in the cell, so the module no longer prepares a copy of the
  range. A shape test on such an operand in a cell decides while tracing
  (snag `jax-source-refus-206a8219`).

## 0.1.164

- A plate cell whose value is one broadcast of 32 or more operands, such as
  an n-ary `.+` of many `x_k[i] .* beta[k]` terms, and which a nested plate or
  a vector likelihood reads entry by entry, is now type stable, so ordinary
  Enzyme Reverse differentiates it. The cell's composed broadcast mapped,
  instantiated, typed and read its operand tuple through Base, which stops
  specializing a tuple at 32 elements: preparation warned that the entry call
  is not type stable, and the gradient raised `EnzymeRuntimeActivityError`
  (33 operands) or aborted the process with a GC error (32 operands). It now
  uses the structural expansion a top-level wide broadcast already uses.
  Broadcasts of fewer than 32 operands are unchanged (snag
  `plate-cell-n-ary-3391765b`).

## 0.1.163

- With bound data, a gather that a plate cell or partition reads at its rows
  keeps its row form when a total-only plate downstream of that cell also
  reads it. Bound preparation gives a residual gather that feeds a
  total-only plate a run schedule (`_readonly_gather_runs`). It used to give
  one to every ancestor of that plate, including a gather that reaches it
  only through a plate cell. A cell runs its plate at its own rows, where no
  run schedule applies, and the rewritten gather had no row form. A cell
  prepared with `locality = :error` then raised `PartitionLocalityError`
  ("its producer `_readonly_gather_runs` has no row form"). RKPPL's
  `prepare_cell_factor_ad(…, :outer, …)` raised it for a nested model whose
  cell reads a per-subject gather, `mu[i]` with `mu = a .+ s .* z[subject]`,
  because the exemplar sums it evaluates at preparation are such a plate.
  This was BRM's sparse Laplace, `integrate = (:subject, :unit)`, in snag
  `cell-factor-ad-r-b4e2a0f4`. The run schedule now stops at a plate cell. A
  gather that also reaches the plate another way is read whole there, and
  keeps its runs.

## 0.1.162

- In a partition, a matrix value that several one-cell members ask for at
  their own rows is now one row per requested row, as a number already was
  (0.1.157). This covers a reshape of a parameter vector and its product with
  a matrix, BRM's `b = z * transpose(tau .* L)` group effects. A reshape's row
  at one row is the row itself (`_CellOneRow`), not a reshaped copy. Its
  product with a matrix is a row whose entries are summed where they are
  read, in the order of RK's own product loop. Neither allocates an array or
  its Enzyme shadow. ReactiveKernelsPPL's compact overlay is read at the
  requested rows without copying them.
  - BRM's `unit_cost.jl` PK toy: the one-subject gradient takes about half
    the time (1,248 → 624 B).
  - The joint PK/QT toy allocates about 30% less (6,016 → 4,272 B), with no
    clear change in time.
  - Values and gradients on both toys are bitwise equal to 0.1.157.
  - A product that the previous version left to BLAS for strided operands now
    sums in RK's order instead. On an RKPPL model with `b = z * F` this moved
    values by up to 3 ulp and gradients by up to 7 ulp (relative 9e-16).

## 0.1.161

- A native scan whose trajectory is read only through one element,
  `trajectory[k]`, `last(trajectory)` or `first(trajectory)`, runs the steps
  through `k` and keeps only that output: no trajectory is allocated. It
  extends the slice-only demand `trajectory[lo:hi]`. The value and the
  `BoundsError` of an index outside the trajectory are unchanged; every case
  the slice demand keeps materialized (a WANT or another reader, `history`,
  an effectful step, a non-concrete element type, a non-`Int` index, a
  sequence that is neither a vector nor a range) stays materialized.
- `sum(p)` or `logsumexp(p)` of an authored plate `p` written inside a larger
  expression (`scale^2 * sum(terms)`) now streams the plate's total, as the
  same fold written as its own statement did; the cells are no longer stored.
  The total is the statement form's, bit for bit; against reducing the stored
  cells as a vector, only the floating-point rounding of the reduction can
  differ, as for every streamed plate total. A plate folded by two different
  functions, by a fold statement with a declared type or the returned value,
  or inside a branch or loop keeps its expression as written.
- Together these make DistributionKernels' one-sided LogNormal `var()`
  (`lower_truncated_lognormal`, `upper_truncated_lognormal`) allocate nothing
  per call, where it allocated about 3–7 KB, and cut its time and its Enzyme
  Reverse gradient's by a third to a half (snag `dk-one-sided-log-c73f7aba`).

## 0.1.160

- New: `register_source_provider!(provider)` and `TraceableSource(argument_names,
  body, mod)`, so a package that generates a method (a macro that writes it
  into its caller's module) can hand `jax_source` the method's Julia source.
  `jax_source` asks each registered provider, `provider(f, argument_types)`,
  for a call that has no translation of its own and no `@traceable` source,
  and translates the body it returns as it translates an `@traceable` body:
  `jax.lax.fori_loop` for a `for` loop, `jax.lax.cond` for an `if`,
  `.at[i].set` for `A[i] = v`, and host Python in `prepare_data`. Before,
  such a method stopped `jax_source` with `JAXEmissionError: the function
  ... has loops or branches`. Register the provider once, from the
  `__init__` of the package's extension on ReactiveKernels. Native
  execution, `prepare` and the compiled paths never read the source (snag
  `register-generat-a22e2325`, decision `0os7s7u`).

## 0.1.159

- `jax_source`: a matrix padded along a ragged row, such as a mixture law's
  `cell = (x .- centres) ./ scales` for `x = vcat(x1, x_rest)` with `x_rest`
  one row of per-row features, now reduces in the JAX functions.
  `sum(cell; dims = 1)` (and `prod`, `maximum` and `minimum` over `dims`)
  raised `JAXEmissionError` (`sum over dims of a Matrix{Float64}`); it now
  sums the row's entries only. A whole-array `sum(cell)` or `sum(f, cell)`
  emitted but read the padding as entries, a wrong value whenever rows have
  different lengths, and `maximum(cell)` failed in Python; both now mask the
  padded axis. The translation records which axes of an array the padding
  covers (a broadcast keeps its operands', a transpose swaps them, an outer
  product `x * x'` and `I(n)` cover both), and `vec` of such a matrix puts its
  valid entries first. A reduction or `vec` of a padded array whose padded
  axes it does not record (a matrix product with a padded operand) now raises
  `JAXEmissionError` instead of reading the padding (snag
  `jax-source-sum-c-bcfa9d53`).

## 0.1.158

- Hessian-vector products and third-order derivatives (`prepare_ad_hvp`,
  `prepare_ad_third`, `prepare_ad_accumulator` at orders 1 and 2) work for a
  kernel with many inactive HAVE ports, under both reverse over forward and
  forward over reverse. From about 28 such ports they raised
  `EnzymeRuntimeActivityError` ("Constant memory is stored (or returned) to a
  differentiable variable"). Each inactive port crossed the backend as its own
  `Constant` context, and from 32 contexts DifferentiationInterface and
  Enzyme splat them through a heap tuple; inside a nested derivative that
  tuple stores the constant operands beside the active point. Past 16 plain
  inactive operands, they now cross as one `Constant` tuple, as the hidden
  bound operands already do. An RKPPL compact cell takes each shared value
  it reads as such a port, so BRM's sparse Laplace raised this for a model
  whose cells read 33 of them (snag `varyingsource3-b-82246291`). Kernels with
  at most 16 plain inactive operands differentiate exactly as before.

## 0.1.157

- A plate cell, and a partition with `single = true`, runs an authored plate
  that it reads at its one row as that plate's one cell. The value is held in a
  non-allocating one-element vector, instead of a plate computed over a
  one-row array. A vector it reads as a lane, computed as a number at that
  row, is read as one too.
- In a partition, a value asked for at several such rows (two one-cell
  members of one subject, each reading the subject's predictor at its own
  row) is one number per row asked for, instead of an array over their union.
  Each after the first is computed only when no earlier one has its row, so a
  row asked for twice is still computed once. The program does not depend on
  whether the rows agree. Values and gradients are unchanged: on BRM's
  `unit_cost.jl` toys (24 subjects) they are bitwise equal to the previous
  version for every subject, and the joint PK/QT model's local gradient went
  from 113 to 87 allocations (todo `1yggoad`).
- Fixed: a `vcat` read at one row converts each entry to the `vcat`'s element
  type. An integer literal beside floating-point operands gave a `Union` that
  native Enzyme Reverse refused.

## 0.1.156

- A plate that reads a bound value through an undeclared name for it
  (`t = s`, with `s::Float64` bound) now keeps that value's declared type
  when its preparation is copied and composed. Bound preparation used to
  type the plate's operand `t` as `Number`, so a graph built from the
  prepared recipes held `t::Number` beside `s::Float64`, and `merge` refused
  it with `cannot merge structural aliases for value :t: Number does not match
  Float64`. RKPPL builds its factor catalogue that way, so a selective SMC
  window whose fresh prior reads a data-valued scale through such a name
  (BRM's `log_rate_prior_scale = offset_scale`) failed to prepare (snag
  `selective-fresh-96f7fe96`).

## 0.1.155

- `ReactiveKernels.input_check!(spec, input, f, args...)` checks a HAVE input
  at the entry of every prepared cut of `spec` that takes it and reads it:
  native, each `vectorize` lane, the primal of every derivative operator, a
  non-allocating kernel and a `jax_source` module. The cut calls `f(args...)`,
  where a `Symbol` names a value of the kernel (the input, or a value computed
  from data such as a size) and anything else is passed as written. It runs
  before any recipe when its arguments are inputs or bound data, otherwise as
  soon as the cut has computed them; the cut computes them even when nothing
  it returns needs them, and a split plan keeps them on the side that reads
  the input. The check is not a recipe, so no read analysis (footprints,
  partitions, cells, activity) sees it: it must read the input's shape alone.
  A cut calls it as `ReactiveKernels._input_check(f, args...)`, which Enzyme
  treats as constant, so a gradient never differentiates the check or its
  error path; it prints that way in `code_expr`. A graph derived from the
  kernel keeps the check on the same value: the copies a cut's preparation
  makes, `plate_cell`, `plate_partition`, `compose`, `rename`, a call of the
  kernel inside another and a stateful kernel's child (`InputCheck`,
  `Graph.checks`, `Plan.checks`). Registering a check advances the graph's
  version, so plans and caches made before it are not reused.
  ReactiveKernelsPPL 0.1.153 uses it to refuse a packed point of another
  binding's length. Snag `rk-prepared-cut-0e0b98ed`, decision `05x3m7a`.

## 0.1.154

- `jax_source` with `data`: a recipe of a top-level plate's cell that reads
  only the plate's data and that the JAX functions cannot compute (a
  `@traceable` helper that builds a list with `push!`, sorts, or loops to a
  data length) now runs in `prepare_data`, as a loop over the cells, and the
  plate reads its values as inputs of its own, one per cell (a vector per
  cell is ragged). It used to be translated as JAX code and raised
  `JAXEmissionError` (`the function fpiseq has no JAX translation`, say),
  although the same recipes at the top level ran on the host. A recipe the
  functions can compute stays in the cell, so modules that emitted before are
  unchanged (snag `jax-source-cell-a3ec8540`).

## 0.1.153

- A prepared child called inside another prepared program, such as a lazy
  arm's call of a kernel that owns a plate or a scan, now infers concretely
  however cold the inference cache is. Its call re-entered the prepared-call
  methods (`_prepared_call`, an `_ArrayFunction`'s call,
  `_native_array_body_call` and the generated function's
  `generated_callfunc`), and Julia's inference limited each such recurring
  edge as possible unbounded recursion, so in a cold cache the caller
  inferred `Any`. Native preparation usually left the pieces inferred, but
  Enzyme infers in a cache of its own: DistributionKernels'
  `lower_truncated_lognormal`/`upper_truncated_lognormal` `var()` endpoint
  (a quadrature plate in a lazy arm) inferred `Any` there. An RKPPL sampler
  whose observation plate read that variance failed static
  `AutoEnzyme(mode = Enzyme.Reverse)` with `EnzymeRuntimeActivityError`, and
  one computing it in a plate cell failed with an `ArgumentError` about an
  array of `Any` (snags `native-lognormal-a666a8e1`,
  `native-truncated-9d7d9eda`). These methods now carry a
  `recursion_relation`, as the source-call methods have since nested source
  operations were made to infer (`41a1f4d30`): a call of a different prepared
  program, whose type does not contain its caller's, is well founded. The same program again keeps
  Julia's default limiting. Only the `generated_callfunc` methods of caches
  ReactiveKernels creates (its own, and a package image's) are marked.
  Emitted code is unchanged; the gradients are exact, as before at warm
  caches.

## 0.1.152

- Reverse AD over a tuple of active ports (`active = (:a, :b)`) holds each
  scalar field of a tuple or `NamedTuple` port as `Ref` storage, as it already
  held a scalar port. `ad_accumulator_storage` therefore gives such a field a
  `Ref`, and `ad_accumulate!` adds into it. Before, a `NamedTuple` port with a
  `Float64` field beside an array got immutable storage for that field, which
  an accumulating pass could neither reset nor add into (snag
  `rkppl-zero-cell-a7c54028`). `ad_gradient` and `ad_value_and_gradient`
  still return plain `Float64` fields.
- A pullback of one WANT port whose value is a tuple or `NamedTuple`
  (`ad_pullback`, `prepare_ad_pullback`) differentiates the seeded inner
  product, as for an array port. With reverse `AutoEnzyme` it threw
  `ArgumentError: broadcasting over dictionaries and NamedTuples is reserved`
  from DifferentiationInterface's Enzyme pullback.

## 0.1.151

- `jax_source(spec; data, definitions)` takes a data-only value of the
  program from its definition when the program's recipe of it has no
  translation on the host or in the functions, or when a value `keep` names
  reads it. The value is then an input of `prepare_data`, listed in
  `UNTRANSLATED`, and `prepare_data` computes what the program reads from it
  (its fields, a block's extent, a gather). Before, `g = prep(labels)` with
  `prep` untranslatable and read as `g.n` and `g.pos` raised
  ``ArgumentError: jax_source: `keep` names `_ppl_axis1_z`, which is no
  data-only value of the program``, or, without `keep`, `JAXEmissionError`
  ("`prepare_data` does not compute it on the host either"). A module that
  emitted before is unchanged unless its `UNTRANSLATED` named a definition
  that was no input (below).
- A named-tuple input of `prepare_data` is a `dict` by field name, or its
  values in field order; `prepare_data` reads each field as it reads a data
  input of that type.
- `UNTRANSLATED` lists only definitions `prepare_data` takes as inputs.
  Before, it also listed one that no input replaced, and a definition reading
  it gave the reason "`g` is not defined in module `Main`"; that reason now
  names the definition it reads and why that has no translation.
- `definitions` that no data input reads, directly or through another
  definition, are no longer computed in `prepare_data`.
- Snag `jax-source-refus-8c6ed1ec` (BRM `rkppl_model(:car_icar)` and
  `(:car_bym2)`).

## 0.1.150

- `jax_source`: a function broadcast over a padded vector of vectors
  (`Diagonal.(rows)` of a ragged plate's per-cell vectors) applies to each
  row, as Julia does, mapped over the rows with each row's length; it was
  vectorized over the numbers. `y[g] ~ MvNormal(m, Diagonal(fill(s^2, n)))`
  in a cell now translates.

## 0.1.149

- `jax_source`: a `throw` under a condition that depends on data translates
  as a runtime check, masked by its conditions, instead of refusing the
  module with "a throw that depends on data has no JAX translation". So the
  `mean()` of ReactiveKernelsDistributionKernels' one-sided LogNormal moment
  objects (`lower_truncated_lognormal`, `upper_truncated_lognormal`, and a
  LogNormal's own mean as the lower one at a bound of 0), whose scale guard
  is `scale >= 0 ? scale : throw(DomainError(...))`, now emits (snag
  `jax-source-refus-05f10f2c`). The module raises on exactly the values the
  native program throws on: a ternary or `if` whose other arm is the value is
  the check followed by that arm (no `jax.lax.cond`), `x > 0 || throw(...)`
  is a check of `x > 0`, and a throw in a vmapped cell holds for the lanes
  that do not take its arm. The right side of a traced `&&`/`||` is now
  masked by its left side, so a runtime check there no longer fails for
  lanes that never run it. An exception's message is its first string
  argument (`DomainError(x, "message")` used to raise with `x`); a throw or
  check in a helper called under its caller's data-dependent condition is
  still refused.
- `jax_source`: two-argument `hypot` is `jnp.hypot`, also with mixed argument
  types (`hypot(c + 2, 3)` used to be refused inside Base's `promote`).
- `jax_source`: a range whose ends are data with no static limit (a scan over
  `n:-1:2` with `n` computed from a parameter) is refused with a
  `JAXEmissionError` naming the recipe. It used to emit a `jnp.arange` of a
  traced value, a module that JAX rejected only when it traced it.
- The LogNormal variances (`var()` of the moment objects above) are still
  refused: a plate child spliced under a lazy arm (`_logmills_quadrature`)
  reaches the module without its plan, and the continued fraction's length is
  data.

## 0.1.148

- `jax_source`: `jax.grad` through a `logaddexp` whose two inputs are both
  `-Inf` is finite, as natively. A hidden-Markov law
  (`hmm_log_weights_logdensity` and its kin) whose transition weights have a
  state that no move reaches at some step (a `-Inf` weight; a left-to-right
  chain) had the right value and a gradient that was NaN in every entry
  (snag `jax-grad-of-the-3dac142a`). Both `logaddexp`s, ReactiveKernels'
  tie-exact one (mixture and log-sum-exp folds, the hidden-Markov forward
  pass) and `LogExpFunctions.logaddexp`, were `jnp.logaddexp`, whose
  derivative at a tie of two `-inf`s is NaN; reverse mode multiplies even
  the zero cotangent of such an absorbed term by it. Both are now the
  module's `_rk_logaddexp` helper, the tie-exact form written as JAX
  selections: the same values, non-finite inputs included, the (1/2, 1/2)
  derivative at a finite tie as before, and at an infinite tie, whose value
  is its first argument, (1, 0), as both give natively.
- `jax_source`: likewise `LogExpFunctions.logsumexp` of a traced vector whose
  entries are all `-Inf` (a hand-written forward pass over such a state):
  `jsp.logsumexp`'s derivative there is NaN. It is now the module's
  `_rk_logsumexp` helper, the maximum-shifted log-sum-exp: the same values,
  the softmax derivative at finite entries, and the maximum itself, with the
  derivative 0, when that is not finite, as native Enzyme gives. Host and
  static values, which are never differentiated, keep SciPy's `logsumexp`.

## 0.1.147

- `jax_source` keeps Julia's grouping when an operand after the first is an
  expression of the operator's own Python precedence level. Python's `*`, `/`
  and `@` share one level and associate left, so `a .* ((B ./ c) * w)` used
  to emit `a * B / c @ w`, which is `((a * B) / c) @ w`: a broadcast error
  (`operands could not be broadcast together`) when `B` is not square, a
  wrong value when it is. It now emits `a * (B / c @ w)`. `a + (b - c)` and
  `a * (b * c)` keep their parentheses too, since floating-point `+` and `*`
  are not associative (snag `jax-source-drops-410c82c2`).

## 0.1.146

- `jax_source`: `I(n)` (a `UniformScaling` called with a size) is the n×n
  identity, padded to its static bound when `n` is data; `Diagonal(v)` of a
  vector is the dense diagonal matrix; a matrix literal of numbers
  (`[a b; c d]`, `hvcat`) translates; and a plate whose cell returns a
  square matrix stacks the matrices, each cell's size its extent, as a cell
  returning a vector does.

## 0.1.145

- `jax_source` translates `cumsum(x; dims = d)` as `jnp.cumsum` along the
  static axis `d`. Before, any keyword `cumsum` raised `JAXEmissionError`
  ("keyword arguments to `cumsum` have no JAX translation"), so a recipe
  such as BRM's monotonic-effects table
  `cumsum(vcat(zeros(1, n), simplex_incr); dims = 1)` did not emit (snag
  `jax-source-simpl-249ccdc6`).
- Three constructs in a straight-line plain helper, translated from its
  lowered code, now emit as they do in source:
  - an index's `end` (`M[1:(end - 1), :]`, lowered to `lastindex`) is the
    static dimension size. Before, the module failed in Python with a
    `ConcretizationTypeError` while JAX traced it;
  - `size(x, d)` takes a dimension that another helper returns and that is
    static while JAX traces (`size(U, axis(o))`), past `x`'s rank 1 as for
    a literal dimension. Before, it raised `JAXEmissionError` ("`size` with
    a computed dimension");
  - a dotted literal power (`x .^ 2`) is the elementwise power. Before, the
    emitted module was not valid Python (`jnp.vectorize(...)(<callable>, ...)`).

## 0.1.144

- `jax_source(...; data)`: `prepare_data` translates `sort`, `sort!`,
  `unique`, `unique!`, `searchsortedfirst` and `searchsortedlast` of vectors
  of numbers or strings, in Julia's `isless` order (-0.0 before 0.0, every
  `NaN` last; `unique` keeps the first of each `isequal` class, in order;
  1-based searches). `sort!` and `unique!` rewrite a `push!` list in place;
  `unique!` of a NumPy array, whose length is fixed, has no host translation.
  Before, a data-only helper calling one of them had no host translation, and
  the JAX functions then refused it (snag `jax-host-sort-un-0edfe976`).
- The host's `isequal` tells -0.0 from 0.0, as Julia's does.
- A `push!` list that a helper, `return` or recipe ends with reaches its
  caller as the array it stands for, not as a Python list. `first`, `last`
  and the searches read a list in place: before, each read converted the
  whole list, so a loop testing `last(v)` while it grows `v` was quadratic.
- A data-only recipe without a host translation: the `prepare_data` comment
  that names it gives the reason, and when the JAX functions cannot compute it
  either, their `JAXEmissionError` names the host reason as well. Before, the
  error named only the functions' reason.

## 0.1.143

- `jax_source` keeps a tuple or `NamedTuple` local that a loop or a branch
  carries as a tuple. A loop in a recipe or a `@traceable` helper over such a
  local (`s = i <= n ? f(s) : s`) used to start its `fori_loop` from one array
  of the tuple, so the module failed in Python with `true_fun output is a
  <class 'tuple'> but false_fun output is a pytree leaf` or a carry-structure
  `TypeError`. Each element now enters with its own dtype, and past a ragged
  row's length each element keeps its value (snag
  `jax-source-tuple-6f1e9e4c`).

## 0.1.142

- `jax_source` translates `besselix(nu, x)` and `besseli(nu, x)`
  (SpecialFunctions) of any order, a data value or one the program computes
  included (snag `jax-source-perio-96f952a3`): a periodic Hilbert-space GP
  weights harmonic `j` by `besselix(j, 1 / rho^2)`. Any order but 0 and 1
  raised "`besselix` of an order other than a literal 0 or 1". A literal
  order 0 or 1 is still `jsp.i0e` and its kin. Any other order in the JAX
  functions is the module's `_rk_besselix` helper (JAX 0.12 has no
  general-order function): the power series, or the uniform expansion in
  large order followed by the downward recurrence, within about 4e-13 of
  SciPy relative for orders 0 to 1000; `jax.grad` in `x` is the recurrence
  identity, and a derivative in the order is NaN, as Julia's derivative rule
  gives. `prepare_data` and values known while emitting use SciPy's `ive` and
  `iv`. A negative integer order and, for an integer order, a negative `x`
  follow Julia; a negative non-integer order is NaN in the JAX functions.
- A package can register a scalar function's translation that reads the
  call's arguments, such as a literal order, with a method of
  `ReactiveKernels._jax_elementwise_call(ctx, f, args)`. It is applied to
  scalars and, broadcast, elementwise to arrays.
- A keyword call in a plain helper's lowered code, `Core.kwcall(NamedTuple{
  names}(values), f, args...)`, translates as the same call in source does:
  `vec(sum(lambda; dims = 2))` (the shape of BRM's `hsgp_sqrt_spd`) is
  `jnp.sum(..., keepdims=True)` raveled. It raised "a type applied to a
  computed parameter", which named no keyword call; a keyword call RK has no
  translation for now raises "keyword arguments to `f` have no JAX
  translation", naming the function (BRM's periodic basis helper
  `_brm_hsgp_basis_state(...; anchor, mask)`, snag
  `jax-source-perio-96f952a3`).

## 0.1.141

- A `vectorize` or `prepare_batched` reader with a `NativeScheduling` of two
  or more workers runs its data-only prefix plates again when a plate's cells
  read an array argument. From 0.1.114 each scheduled chunk of such a plate
  read the argument's broadcast preparation as an unbound name and the call
  failed with `UndefVarError: ##plate_prepared#k not defined` (inside a
  `TaskFailedException`), whatever the program. A one-thread process never
  enters the prefix, so it showed only with two or more threads.

## 0.1.140

- New: `plate_partition(…; single = true)` declares that every key holds each
  label exactly once, so a partition is one cell of each named plate (one
  subject's cell of a plate over subjects). The partition then runs that cell
  at one `Int` row, as `plate_cell` does, instead of over a vector of one row.
  A key that holds a label zero or several times throws `ArgumentError` at
  evaluation. Without `single`, partitions are unchanged.
- A plate cell, and a partition with `single = true`, computes the values it
  reads at its one row as numbers where it can, instead of one-element arrays
  with an Enzyme shadow each. A capture the cell reads outside every branch at
  one position per cell asks for that one row as an `Int`; a value of one
  element per row read there only by such reads or by other such values is a
  number: a pointwise expression, a product `A * w` with a vector `w`, a column
  `A[:, k]`, a direct gather `v[I]` or `A[I, k]` and a `vcat` of numbers and
  vectors. Values are equal; a product row `A[r, :] * w` is summed in column
  order, so values and gradients through it can differ from Base's product in
  the last bits. An authored plate computed at rows, and any value a plate
  reads as a lane, keep their one-row arrays. On an RKPPL partition sampler
  over one cell per subject (24 subjects, 10 observations each), one subject's
  local gradient went from 1.22–1.28 µs / 1,232 B to 0.45–0.47 µs / 592 B,
  6.2–6.4 to 2.3–2.4 times its share of a whole-model gradient, with bitwise
  equal values and gradients (todo `1yggoad`).

## 0.1.139

- `jax_source` follows a plain helper's lowered code through module-qualified
  calls (`M.f(x)`, lowered `getproperty(M, :f)`), `getglobal(M, name)` of a
  module and name known while emitting, `Val` tags (`Val(:f)`, `Val{:f}()`),
  Vararg methods and their splats (`f(args...)`, lowered
  `Core._apply_iterate`), and functions or types passed on as arguments.
  StanBlocks `@deffun @juliacompat` functions call each callee as
  `StanBlocks.jcall(Val{:f}(), _JuliaCallScope{M}(), args...)`, so one whose
  body is straight-line now translates to the arithmetic it reaches; before,
  every one raised `JAXEmissionError: the field read jcall on
  ReactiveKernels._JaxCallable{Module}(StanBlocks)` (snag
  `jax-source-stanb-06917e13`: BrunoPMX's `pk_generator`,
  `logistic_dose_response` and `biexponential_rates`). A `@deffun` whose body
  has a loop or an `if` stays a plain helper with loops or branches, which has
  no translation.

## 0.1.138

- `jax_source`: a gather through a ragged integer row is padded as the row is
  (snag `jax-ragged-index-b8420d16`). In a plate cell, `xs[rs[i]]`, `xs[r]` or
  `view(xs, r)` with `R::Vector{Vector{Int}}` lost the row's lengths, so the
  cell's reduction summed the padding entries, silently: the JAX value
  repeated each row's last index up to the longest row (2.093 where native
  gives 1.352), and an empty row added `x[end]`. Its gradient took the same
  extra terms. The gathered value now carries the row's lengths, and every
  reduction of it masks them.
- `sum(f, v)`, `dot(u, v)` and `u' * v` of a padded (ragged) row now mask its
  padding as `sum(v)` does; they summed it, also for a ragged input's own row.
- A row of a ragged value read through `getindex`/`view` in lowered code keeps
  its lengths, as `x[i]` in source does.
- A gather through a ragged row beside another kept dimension (`M[r, :]`)
  raises `JAXEmissionError`: its padding would sit on an inner axis that no
  reduction masks. Its JAX value used to sum the padding.

## 0.1.137

- `jax_source`: a scan whose step outputs a named tuple of numbers and
  numeric vectors (a K-state filter's `(lpd, w, m)`) translates. Each vector
  field stacks as the `(steps, K)` array of its rows, as `jax.lax.scan`
  stacks it; `steps[t].w`, `steps[end].m` and generators over the steps read
  it as native does. It raised "a scan whose step outputs … has no JAX
  translation yet".

## 0.1.136

- `jax_source`'s `prepare_data` writes an element of an array that a list or
  dictionary holds, `rows[i][j] = r` (snag `jax-host-nested-6a304948`); it
  raised `JAXEmissionError: the assignment target (rows[i])[counts[i]] has no
  host translation`. The write is in place: with no copy when the source built
  the list from allocations (`[Vector{Int}(undef, n[k]) for k in …]`, a
  count-then-fill partition), else after one copy of the list and its
  elements, so a source never writes its inputs.
- In `prepare_data`, an `if`, `elseif`, `&&` or `||` whose condition is
  decided while emitting (`eltype(x) <: Integer`, `length(t) > 0` of a tuple)
  translates its taken arm alone, as `? :` and the JAX functions already did.
  Both arms were translated, so an untaken arm with no method for these types
  (`Int128(maximum(x))` of strings) raised a raw `ErrorException: type
  TypeofBottom has no field parameters` from `jax_source`.

## 0.1.135

- `jax_source` translates `clamp` and `convert(T, x)` (snag
  `jax-source-clamp-af26e700`). `clamp` is a straight-line Base function whose
  lowered code converts its operands with `convert(promote_type(...), x)`,
  and that `convert` had no translation, so a recipe calling `clamp` (the
  `hsgp(...)` basis of a BRM model) raised `JAXEmissionError: the function
  convert has no JAX translation`. `convert(T, x)` is now `x` when `x` is a
  `T` already, and `T(x)` for a concrete float or integer `T`. `clamp` keeps
  Julia's definition, two `jnp.where` selections, so at a bound its
  derivative is the one Julia's `ifelse` gives (`jnp.clip` would halve it).
- A type that lowered code or a type-level call passes as a value (`Int` read
  as a global, a `where T` parameter, `promote_type(...)`) is now read as a
  type by the calls that take one. In a plain helper, `floor(Int, x)` emitted
  invalid Python (`jnp.floor(<callable>, x)`) and `ones(T, n)` was refused;
  both now translate.

## 0.1.134

- `jax_source(spec; data, definitions)`: a definition may read a raw data
  value that is no input of the program (a value the program reads only
  through a copy, `_rkppl_value_r => :r`). `prepare_data` takes it as a
  parameter, typed by its value in `data`. ReactiveKernelsPPL 0.1.142 uses
  this so `prepare_data` asks for authored names only.

## 0.1.133

- A scan with two or more outputs (two carried arrays, or a carried array and
  a sampled innovation) reads each trajectory out of its shared recurrence
  through a column plate. That plate's row is now named by the source, like
  the scan's seeds and steps, instead of with `gensym`'s process counter. An
  `@rkppl` declaration holding such a scan, in a `@plate` cell or at top
  level, bound again to the same data or at other sizes now gives an equal
  `kernel_expr`, as ReactiveKernelsPPL 0.1.136 states for what a model call
  generates. Built types, values and gradients are unchanged: two builds of
  one binding already shared their types. Snag `kernel-expr-scan-1a8411f4`.

## 0.1.132

- `jax_source`: the hidden-Markov laws of ReactiveKernelsDistributionKernels
  0.1.5 translate (`normal_hmm_log_weights_logdensity` and its siblings under
  `LogDensity`). The forward pass is a `jax.lax.scan` with `history`, and the
  folds over the states are `jax.lax.fori_loop`s, so one module serves every
  number of steps and states. What it takes, all generic:
  - a shape test decides while tracing. Prepared integers are `StaticInt`s,
    so the functions know them then; `size(x, d)` past `x`'s rank, or of a
    number, is 1; comparisons of integer literals fold, so a conditional on
    one takes its arm alone; `x in t` over a tuple of numbers is Python's
    `in` when both are known, else `jnp.isin`;
  - an index Julia requires to be 1 (past an array's rank, `v[i, 1, 1]`, or of
    a number, `s[1, 1, 1]`) is checked, while tracing when it is known then,
    else as a runtime check, and dropped;
  - `dropdims(x; dims = d)` translates.
- `@_record_source` accepts a leading `@inline`, which the method keeps.
- In `prepare_data`, `stack(rows; dims)` of a plate of vectors is `np.stack`.
  The plate is a list of arrays on the host, so a later `raw_basis[:, b]`
  raised `TypeError` (BRM's HSGP basis with `orthogonal_to`).
- RK's tie-exact `logaddexp` (mixture densities) translates as
  `jnp.logaddexp`, and integer `div` (lowered as `checked_sdiv_int`, as in an
  HSGP mode index) truncates toward zero as in Julia.
- A scan with `history` translates (a vector of leading entries or a number):
  `jax.lax.scan` carries the result vector and writes step `j`'s output at its
  entry; an index of `eachindex(x)` is bounded by the sequence's length, so a
  generator over `1:j-1` runs to that bound, masked. `j ± c` keeps a static
  bound. A ragged column's empty row keeps the column's element type when
  padded (an integer index column stayed float and could not index), and a
  ragged value a function returns in a `dict` comes with `<name>_lengths`.
- Generated module helpers drop the `_rk_` prefix (`_standardized`,
  `_isequal_np`), so a generated helper never takes a fixed helper's name: a
  module defined `_rk_isequal_np` twice and `prepare_data` recursed.

## 0.1.131

- A non-allocating program differentiates under native Enzyme reverse through
  a `vcat` of operands of different array types, such as a vector parameter
  stacked on a matrix parameter, `vcat(permutedims(c), D)`, or `vcat(c', D)`
  (snag `rkppl-vcat-of-pa-decf6725`). The `vcat` step filled its buffer
  through a loop over the operands, which the type-stability check reported
  ("The prepared program's entry call is not type stable", at
  `_vcat_destination_matches`) and native Enzyme reverse refused with
  `IllegalTypeAnalysisException` or `EnzymeNoTypeError`. The step now checks
  and copies its operands one inlined call per operand, and dense matrix
  operands fill a matrix buffer through the native body's `vcat` copy.
  Registered Enzyme (0.13.219) still refuses the step for a fresh `Matrix`
  row beside a reshaped view, as an ordered vector's row is: it cannot
  compile Base's `vcat` of them, the step's result for a buffer of another
  shape, even where that call never runs. The Enzyme dogfood line
  differentiates it.

## 0.1.130

- `jax_source(spec; data, definitions)`: `definitions`, an ordered collection
  of `name => expression`, computes data values in `prepare_data` from the
  other data values, in host NumPy, before the program's data-only prefix.
  `prepare_data` then takes the data those expressions read in place of the
  values they define. With `definitions`, the module also has `DATA` (its
  parameters' dtype, rank and raggedness) and `UNTRANSLATED`: each definition
  without a host translation, or given as a `JAXEmissionError` naming the
  reason, with that reason; such a definition stays a `prepare_data` input.
  ReactiveKernelsPPL 0.1.140 passes a binding's data-only definitions.
- `jax_source(spec; data, keep)`: `prepare_data` also returns the data-only
  values `keep` names (the functions take them too).
- A scan over a range whose end is data with a static bound (a group's
  length in a ragged plate cell, `for t in 2:length(y[g])`) runs to that
  bound. Past the group's length its carry keeps its value and its outputs
  are padding, which its extent masks; `last` of a padded vector reads the
  entry at its length, and `vcat` of values before a padded vector adds their
  lengths to its extent. Such a cell's scan raised `ConcretizationTypeError`
  in Python (a traced `jnp.arange` stop).
- `sum(x; init)` and `prod(x; init)` translate; `sinpi` and `cospi` translate,
  their argument reduced to `[-1, 1]` first. `sum`, `prod`, `maximum` and
  `minimum` with `dims = d` reduce that axis and keep it, as Julia does.
- A scan whose steps output a tuple of numbers translates, as one of named
  numbers already did; both are held as the tuple of their columns. Reading
  such a vector by element (`out[end].level`) reads every column at that
  index: the module read `out[out.shape[0] - 1][0]`, which fails in Python
  with `AttributeError`. A generator over a vector's elements (`sum(o.square
  for o in out)`, also over a plain or padded vector) is `jax.vmap` of the
  element over it, and `merge` of named tuples translates (a scan seeded by
  `merge(init, (b = 0.0,))`).
- Generated Python names drop the counters that differ between builds of one
  program (a gensym's `#<n>`, a value id's long numeric suffix), so one
  declaration emits one module text.
- `JAXSource` has a third field, `prepared`: each value `prepare_data`
  returns, by its program name, to its key (`JAXSource(text, functions)`
  still constructs one). With `definitions`, a definition of a name that is no
  data value is a host local of `prepare_data`, so the names a definition
  reads resolve to the program's definitions before any global function; and
  `DATA` and the `prepare_data` `JAXFunction` name its parameters as Python
  spells them. On the host, `plate(xs...) do … end` over data is a map.
- With `data`, a module's functions take `(unconstrained, **prepared)` and
  read the prepared inputs inside, so their signatures name what the caller
  supplies (user GO `08jy2dh`); positional prepared arguments are no longer
  accepted. A function's WANT may be a nested `NamedTuple`, returned as nested
  `dict`s.
- `jax_source(...; function_docs)`: a `NamedTuple` of function name => text
  gives functions their docstrings; every module function has one and
  annotates its parameters (`ArrayLike`) and result. Module-level helpers
  are private (`_rk_standardized`, was `rk_standardized`).
- `prepare_data` returns each integer scalar it prepares as a `StaticInt`, an
  `int` that `jax.jit` treats as static, so the functions jit with the
  prepared data as arguments (`jax.jit(f)(u, **prepared)`). A block end or a
  scan length was a traced argument there, and slicing by it raised
  `IndexError: Array slice indices must have static start/stop/step`.
- `jax_source` translates a derivative rule's primal cut again (DistributionKernels'
  `loggamma` and `logbeta`, so RKPPL's `NegativeBinomial2`, `BetaBinomial2`,
  `Gamma`-family and other densities that call them). Since 0.1.27 a lowering's
  callables table holds each operation's source record rather than the
  operation, and the emitter refused the record with `JAXEmissionError: a
  source part ... called without its operation`.
- `jax_source` translates the CDF-based evidence of the location-scale
  families: `censored`, `interval_censored` and `truncated` responses and
  truncated priors. Their lazy arms call a prepared endpoint
  (`standardized`) that the arm's source captures; the emitter did not pass a
  branch's captures, did not read the captures a generated body unpacks
  (`##captures#1 is not defined`), and had no translation for a prepared
  child called as a function, which now becomes a module-level function named
  after it (`_rk_standardized`).
- `prepare_data` keeps the write of a `get!` whose value the source discards
  (a statement `get!(d, k, v)`). It dropped the call, so RKPPL's level codes of
  a string or categorical grouping column were all 0 and the JAX log density
  read the wrong coefficients without an error.
- `view`, `Base.maybeview` and `getindex` called with values in a plain
  helper's lowered code translate as a gather, including a `:` (whole-axis)
  index: `reshape(view(v, :), 1, length(v))` (RKPPL's multivariate normal slice
  priors under `LKJCholesky`). A whole-axis slice of a ragged value raises a
  `JAXEmissionError`.

## 0.1.129

- `prepare_ad` over a `NonAllocatingKernel` differentiates a plate whose cells
  return arrays: it raised `MethodError: no method matching
  size(::_NestedPlateCache)`.
- `prepare_ad` over a `NonAllocatingKernel` differentiates a program whose
  exemplar call left a branch arm untaken: a step in that arm allocates, as
  the ordinary program does. It raised `ArgumentError: AD preparation over a
  non-allocating kernel requires every retained cache slot to be seeded`.
- A non-allocating kernel reads the buffer an authored plate or scan output
  keeps through a slot typed by that output's array type, so the program is
  concretely typed (it was a `Ref{Any}`).
- New: `recipe_inventory(::NonAllocatingKernel)`, as for a `PreparedKernel`.
- `ReactiveKernels._nonallocating_ad_default(backend)`,
  `ReactiveKernels._nonallocating_plan_supported(plan)` and
  `ReactiveKernels._runs_batched_plates(kernel)`, internal: whether
  ReactiveKernelsPPL's sampler prepares a non-allocating kernel by default for
  `backend` (the Enzyme extension declares it for reverse-mode `AutoEnzyme`),
  for `plan`, and for an ordinary program that runs a plate as matrix
  products. ReactiveKernelsPPL 0.1.133 reads them.

## 0.1.128

- A scan whose step keeps its types runs every step in its loop. A native scan
  ran its first step before the loop (peeled it) to type a result whose output
  type inference cannot fix, and to give the loop the carry type that step
  returns. When the step's inferred output type is concrete and it returns the
  seed's carry type, the peel did no work, and on Julia 1.10 Enzyme Reverse of
  the peeled loop cost 5–8% more than the same loop without it, for a cause
  Enzyme could not locate. Such a scan now starts its loop at the first index:
  in a plate cell, at top level, in a demand-driven slice, with `history =`, and
  when the scan operation runs outside a generated body. Values are unchanged.
  A scan whose carry changes type at its first step (an `Int` seed summing
  `Float64`s), whose output type is not concrete, or whose fused plate consumer
  sums cells that are not numbers keeps the peel. `code_expr` shows both forms;
  the types select one at compile time, and only it is compiled. On Julia
  1.10.12 with Enzyme dogfood `b5daa2ac`, the Reverse gradient of an ARMA(1,1)
  scan cell's generated function went from 1.05–1.11 to 0.96–1.01 times an
  inlined hand loop, and a 3-state HMM filter cell from 1.08–1.10 to 1.01–1.02
  times (historical benchmark).

## 0.1.127

- A plate whose cells multiply a vector of their own by a matrix every cell
  shares runs as matrix products over tiles of 256 cells, given exemplars
  (`prepare_sampler` passes them). Each cell's number is one entry of a vector
  over the tile and each cell's vector one row of a matrix, every statement
  runs once per tile, and `A * v` is one `V * transpose(A)` per tile, the
  native counterpart of the JAX target's `jax.vmap` of a plate. A
  random-Fourier-feature field of 1,444 cells × 38 channels took 4.4 ms per
  gradient against 31.0 ms for its cell loop (snag
  `plate-of-per-row-d1b4ed43`, decision `0tpji4i`).
- Batched: dotted calls over the cell's numbers and vectors and shared numbers
  and vectors; `A * v`; the products, sums, differences and quotients Julia
  defines between the cell's numbers, vectors and shared numbers; `sum(v)`
  and `sum(f, v)`; `y[i, :]`, `y[i]`, `y[i, j]` at the cell's integer and
  `v[k]` at a shared one; other functions of the cell's numbers. A statement
  without such a form, and any statement reading one, runs cell by cell inside
  each tile, so an `@rkppl` plate whose cells end in their observations'
  densities runs batched through `prepare_sampler`. A plate is batched when one
  statement is a shared matrix times a cell's vector and it holds no nested
  plate, scan, shared call or lane branch; its tiles run for 16 cells or more
  when the call's types are the exemplars' (elements, for the cells'
  arguments), its arguments span its one axis and each vector argument's
  vectors share a length. Anything else runs the cell loop.
- A partition (`plate_partition`), and a plate computed at the rows a cell query
  or partition reads, run those rows batched the same way.
- Values and gradients of a batched plate agree with the cell loop to
  rounding, not bit for bit. `RK_BATCHED_PLATES=0` keeps the cell loop, and
  `JULIA_DEBUG=ReactiveKernels` logs each plate planned batched. The
  non-allocating step, the opt-in plate reverse, one-cell queries, lane
  branches, nested plates and scans, and per-cell matrices keep the cell loop
  for now.
  "Plates of products run as matrix products" in the compiler guide has the
  rules.

## 0.1.126

- A plate cell's slice is read in place when its reader hands another argument
  to BLAS or a foreign call, or returns a fresh array. The read-only proof
  follows the slice through the callee's optimized code, but an authored cell
  call (any `@rkppl` or BRM cell `f(a, A[i, :])`) runs through a callable that
  takes its arguments as one vararg tuple, and the proof followed that whole
  tuple: every array argument counted as the slice, so a callee that only
  indexes its row and mixes its result with another argument
  (`correlation * raw`, a `gemv!` on `correlation`) kept a copy per cell. The
  proof now follows only the slice's element of a tuple the body reads element
  by element; a tuple used any other way is followed whole, as before. And it
  counted any array a call returned as derived from the slice, so a
  vector-valued cell (the result of `correlation * raw`) kept its copy too; a
  call's result is now derived only when the callee may return a value derived
  from the slice. The slice itself is held to the same rules as before: no
  write, store, return or foreign call.
- Preparation logs why a slice keeps its copy. With
  `JULIA_DEBUG=ReactiveKernels`, each refused slice is logged at the debug
  level (group `:readonly_slices`) with the reason, its reader and, when the
  read-only proof refused, each frame that refused and the statement of its
  optimized code it stopped at. See "Slices a cell only reads" in the compiler
  guide.

## 0.1.125

- A program that native Enzyme differentiates only with a repair on Enzyme's
  `dogfood` line can name that repair before its derivative is prepared
  (`_require_enzyme_repairs(backend, repairs, requester, floor)`; the Enzyme
  extension supplies it). On Julia 1.10, with an Enzyme backend (`AutoEnzyme`,
  or a `SecondOrder` holding one), it calls `Enzyme.require_dogfood_repairs`
  where Enzyme reports its repairs (dogfood from `a68104d`), which throws
  `Enzyme.DogfoodRepairMissing` naming each missing one. Where Enzyme cannot
  report them (a registered release, older dogfood) it warns once per requester,
  naming the floor, and lets the derivative be prepared. Other backends and
  Julia versions are not checked.
- `_reads_lazy_view_rows(kernel)` says whether a prepared program reads a lazy
  view (`_cell_lazy_view`) at a cell's rows through `_cell_source_rows`, that is,
  takes a `view` of the lazy array itself at those rows. On Julia 1.10 native
  Enzyme can need dogfood repair `const_slot_reloads` for that shape: in a large
  enough program it raises `EnzymeRuntimeActivityError` without it (snag
  `rk-partition-pat-8d339a5f`). The check finds the shape, not the size.

## 0.1.124

- A plate cell reads a slice in place when every call that reads it provably
  only reads it. A row, column or gather passed to a function (`row = A[i, :]`
  then `f(a, row)`, and any `@rkppl` cell `f(a, A[i, :])`, which RKPPL splits
  so) allocated a copy per cell, and native reverse its shadow, so a row read by
  an opaque function was the largest single item of a large field model's
  gradient. Preparation with `exemplars` (`prepare_sampler` and the partition
  and cell queries pass them) now proves from the callee's code that it writes
  nothing derived from the slice, does not return it, and cannot tell a view
  from a copy, and then emits `view`. On a toy model of that shape a gradient
  takes 0.41 to 0.54 of the time, and its bytes stop growing with the row; a
  partition's unit gradient, the path SVI runs, is read in place too. Values are
  the copy's bit for bit. Gradients can differ in the last bit where Enzyme's
  contracted derivative arithmetic (the dogfood default) fuses a multiply-add
  into the parent's shadow, and are identical under strict arithmetic. A callee
  the proof cannot follow keeps its copy, which costs what it cost: a dotted
  call or BLAS over the slice, `sum` or `prod` of it (Base may fold a view in
  another order), a closure capturing it, a callee with a `Vector` method beside
  its generic one. Scan steps, plates nested in a cell, a slice nested in a
  `@kernel` plate's call and a gather of several rows of a value a cell query
  computes only at its rows also keep their copies. `RK_READONLY_SLICES=0` keeps
  every copy, for comparison. See "Slices a cell only reads" in the compiler
  guide.

## 0.1.123

- `prepare_nonallocating`, and through it
  `prepare_sampler(...; nonallocating = true)`, now runs the stable rewrites
  `prepare` runs: a composition such as `log(exp(u))`, `log.(exp.(x))`,
  `log(1 / (1 + exp(-u)))` or a branch arm's `log(s)` of `s = exp(u)` is
  computed in its stable form, inside its buffered steps too. A non-allocating
  kernel's values and ordinary-Reverse gradients change where `exp` overflows
  or underflows, from `NaN` or an infinity to the finite value `prepare` gives,
  and by rounding elsewhere: they now agree with `prepare` bitwise on the four
  PosteriorDB models that differed.
- New: `prepare_nonallocating` takes `simplify`, as `prepare` does;
  `simplify = false` or the project's `simplify` preference keeps the literal
  program. A `PreparedKernel` argument contributes its plan, not its own
  setting.
- A rewritten dotted call (`log.(exp.(x))`) no longer allocates its inner
  function as a `Ref` on each call, in `prepare` too.

## 0.1.122

- A plate over no rows prepares the program the same plate over rows
  prepares, by default (no `branch_arms` setting needed). The pass that caches
  a plate cell's work over bound data per lane at preparation used to leave a
  plate with an empty bound domain as authored, so a binding with no rows (an
  RKPPL `@plate` whose data column is empty, such as a region without spectra)
  prepared another program, another sampler type and another compile than a
  binding with rows. It now rewrites an empty domain as it rewrites a nonempty
  one. It evaluates nothing: the cached per-cell values are empty, and a value
  every cell would share is a typed placeholder that no cell reads. Such a
  value now also has a placeholder when it is an array (a data column a cell
  reads whole gets an empty one of its type) or a tuple or named tuple of
  values that have one; with `branch_arms = :all`, 0.1.119 left the plate as
  authored for these. Values and gradients are unchanged, and a binding with
  rows prepares exactly the program it prepared before. A shared value of
  another type (a struct holding arrays) still keeps the authored plate.
  A branch whose condition no lane varies, such as the support guard of a
  written scale, is decided for a plate over no rows as for one over rows; a
  condition that throws there keeps the plate. A branch its lanes decide still
  prepares a program per arm pattern unless `branch_arms = :all` (0.1.119).

## 0.1.121

- The opt-in generated summed-plate reverse (`with_plate_reverse`, preference
  `plate_reverse = "on"`) no longer returns a wrong gradient. Before, a plate
  program reading an array that another plate filled in place
  (`_plate_similar_output`, then `buf[index] = value`), or an operand Enzyme
  differentiated but the generated reverse did not, got a zero cotangent with
  no error. On a downstream application SH v5 model 798 of 890 gradient coordinates were
  wrong. Its activity analysis now follows in-place writes, and any call
  whose active operands the generated reverse does not cover takes Enzyme's
  ordinary reverse, so a coverage gap costs speed, never correctness. Plate
  programs with more than 32 operands, and lazy cell-broadcast operands
  (`MixedDuplicated`), no longer fail.
- With the stable rewrites (0.1.117, on by default), plates where a rewrite
  fires take the generated reverse again: each `_rk_*_of` helper call is
  expanded into the method its operands select.
- Hoisted cell invariants are swept once after the loop, and a division by
  an invariant multiplies by its hoisted inverse: about 1.17x faster on a
  Gaussian random-intercept plate.

## 0.1.120

- A derivative rule whose data input is a structure (a record or tuple holding
  arrays) differentiates in a planned partition body of
  `_PARTITION_SEPARATE_MIN` or more statements, with either spelling of that
  input: `geometry = nothing`, or a cotangent port that returns `nothing`. The
  wrapper such a body calls its array statements through took its operands as
  one tuple, which hid the structure's constant activity from Enzyme. Reverse
  then failed with `no method matching _accumulate!(::Record, ::Nothing)` for
  the port spelling and `data-only input requires Const` for the declared one,
  and a body with fewer statements passed. The wrapper now has one positional
  method per operand count, and a statement with more than 64 operands stays
  inline. A child kernel prepared for a lazy arm forwarded its operands the same
  way and failed the same way with a rule in its body; it has the same methods
  (snag `derivativerule-r-0d7e245c`).

## 0.1.119

- New, opt-in: `branch_arms = :all`, a `prepare` keyword and the
  `[ReactiveKernels]` preference `branch_arms = "all"` in a project's
  `LocalPreferences.toml` (default `"taken"`, today's behaviour). With it, a
  plate cell's lazy branch whose condition reads bound data keeps one prepared
  program whichever arms the binding's lanes take. Every arm is a plate, over
  no lanes when none takes it, and a domain of no lanes is split too, so data
  whose lanes all take one arm, or that has none, prepares the program a mixed
  binding prepares: an RKPPL censored response with a bound scale keeps one
  sampler type whether its rows are all observed, all censored, mixed or
  absent. An arm without lanes evaluates nothing: it calls no plate program,
  its shared work stays behind the plate's nonempty guard, and its work over
  bound values, such as `log(s)` of a bound scale, is not run at preparation,
  so an invalid value on an arm no lane takes still never throws. An arm of one
  lane splits a nested branch as arms of more lanes or none do. Reassembling a
  pointwise vector whose lanes all took one arm skips the gather. A gradient
  differentiates a plate program over 64 cells or more through its call, so
  its loop vectorizes, and a smaller one inline. Values and gradients are
  unchanged.
- Its cost, measured on a Linux fleet host against 0.1.114: an RKPPL censored response
  over 10 rows, all observed, takes 48 ns per density and 171 ns per sampler
  gradient (31 and 82 ns by default), and over 200 rows 106 and 613 ns (60 and
  482 ns). Over 2000 rows it costs what the default does, and several Normal
  likelihoods over 2000 rows each take 28–33% less per gradient. Every arm is
  compiled: that model's first gradient over 200 rows took 29.5 s (14.7 s).
- This is 0.1.110's change (reverted in 0.1.111), now opt-in on the user's
  decision; by default nothing changes. A condition that reads bound numbers
  alone still selects its arm at preparation, and a bound singleton beside
  live arguments of unknown length (a plate of one row) still prepares its own
  program. With the generated plate reverse on as well (`with_plate_reverse`),
  an eligible program takes its rule either way.
- Fixed: a plate program no longer treats a named-tuple field's or a keyword's
  name as a variable. Since 0.1.113, a body with two or more plates whose cells
  built `(location = m, scale = s)` could raise ``UndefVarError: `location` not
  defined``, as an RKPPL `:observations` query over a ragged plate did.

## 0.1.118

- A number among the operands of a `vcat` that a plate cell or a cell query
  reads (`padded = vcat(frac, 0)`, `padded[row[i]]`) is that number, whichever
  integer it is. The cell took an `Int` for the position of one of the `vcat`'s
  inputs. A value no input holds, such as `0`, failed preparation with a
  `BoundsError`. A value an input holds, such as `1` in `vcat(frac, 1)`, gave
  **wrong values with no error**: that operand read the input again instead of
  the number, and a read past the `vcat`'s end returned a value instead of
  throwing. A cell's values disagreed with the whole program's from 0.1.16,
  when these forms were added, until now. Any other number type was already
  right, and the whole program is unaffected (snag
  `cell-vcat-litera-04f008d6`).

## 0.1.117

- Preparation computes numerically fragile compositions in their stable form:
  `log(exp(x))` as `x`, `log(logistic(x))` as `loglogistic(x)`, `log(1 + x)` as
  `log1p(x)`, `exp(x) - 1` as `expm1(x)`, `log(1 + exp(x))` as `log1pexp(x)`,
  `log(exp(a) + exp(b))` and `log(sum(exp.(x)))` as log-sum-exps,
  `y / exp(x)` as `y * exp(-x)`, and the rest of the table on the compiler page
  ("Stable rewrites"). The written form is finite only while its intermediate
  is representable, and its ordinary reverse forms `0 * Inf = NaN` at the edges:
  `log(exp(u))` at `u < -745` or `u > 709`. The stable form agrees with it to
  within rounding wherever it is finite, and stays finite, with a finite
  derivative, beyond. Errors, lazy branches and non-`Real` operands keep the
  written program, and result types are unchanged. The rewrite applies across
  statements, inside plate cells and inside a cell's lazy-branch arms. A
  sampled positive scale's `exp` and the law's `log(scale)` now meet as `u`,
  and `bernoulli(logistic(η))` is as exact as `bernoulli(; logit = η)`
  (`-20.000000002061153` at η = 20, y = false, where it gave `-20.00000002003398`,
  and `-40.0` at η = 40, where it gave `-Inf`). Results change in their last
  bits wherever a rule fires.
- `prepare`, `prepare!` and `prepare_ad(::KernelSpec, …)` take `simplify`:
  `false` compiles the literal arithmetic, and `nothing` (the default) takes
  the enclosing preparation's setting, else the project preference
  `[ReactiveKernels] simplify`, else `true`. Cached preparations are kept per
  setting.
- Not yet covered: `prepare_nonallocating`, the reactive compiler, the batched
  lowering and `jax_source`, values read per row from a gathered or zipped
  array, and `@traceable` helper bodies. Opaque functions are never rewritten.

## 0.1.116

- A generated reverse for summed plate programs, opt-in: with the
  `plate_reverse` preference set to `"on"`, or inside
  `with_plate_reverse(f)`, a prepared gradient differentiates each summed plate
  program with a reverse generated from the program's own code instead of the
  backend's reverse over its loop. The output adjoint of a sum reaches every
  cell unchanged, so the reverse is one pass over the cells: each cell runs
  forward and then its own reverse sweep, shared values' adjoints accumulate
  across cells, a batched operand's cotangent is added into its shadow at the
  cell (a gathered operand's shadow scatters it), and invariants hoisted above
  the loop are swept once after it. Partials come from DiffRules; lazy branches
  stay lazy. The primal is the program itself, unchanged. A plate the
  generator does not differentiate keeps the backend's reverse;
  `plate_reverse_report(kernel)` names, for each plate program, whether it took
  the generated reverse and otherwise why not.
- New dependency: DiffRules.

## 0.1.115

- A footprint (`partition_footprint`, a cell query's or a partition's) maps
  the reads of a view held in a typed local whose type inference cannot prove,
  such as `x::AbstractVector{Float64} = view(u, 8:15)` of an abstractly typed
  port in a kernel that owns a plate. Assigning the view converts it to the
  declared type; the footprint decides when it is evaluated whether that is the
  view itself (the view already has the type) or a copy, which reads every
  viewed position. Before, any such view counted as a copy, so every reader of
  it read the whole range: a ReactiveKernelsPPL dotted cell latent read every
  index's block in its cell once the cell also read a value computed from
  parameters (snag `dotted-cell-late-3789b8ea`). A declared type the view does
  not have (`head::Vector{Float64} = view(u, 2:4)`) still reads every viewed
  position. Values and gradients are unchanged (`test_view_footprints.jl`).

## 0.1.114

- A plate program (0.1.112) is now only the plate's cell work: its loop, the
  invariants above it and the preparation of the arguments the cells read.
  The body keeps the plate's setup and calls the program only when the domain
  has cells, so a plate over no rows calls none.
- A plate program is called with each operand as an argument of its own
  (`_PlateCall`) instead of one argument tuple per call. Gradients still
  differentiate the plate loops inline.
- On Julia 1.13 a lowered body holding plate programs compares equal to its
  `deepcopy`: a deep copy keeps each plate program, whose expression nothing
  changes once built. Julia 1.13 compares an expression's leaves that are not
  expressions by identity.

## 0.1.113

- A non-allocating kernel (`prepare_nonallocating`, and through it
  ReactiveKernelsPPL's `prepare_sampler(...; nonallocating = true)`) gives more
  array results a buffer of its own, with the same values:
  - the arrays a `@traceable` helper builds when its body is straight-line:
    assignments (`x = value`, `a, b = x, y`), calls run for their effect (an
    argument check) and a final expression. This includes a helper whose own
    result is a number or a tuple, such as a density;
  - each arm of an authored `if`/`elseif`, decomposed into steps that run only
    when that arm is taken;
  - a call with a registered `MutatingFunctions.apply!!` method whose result
    is a tuple holding an array, such as a transformed vector and its
    log-Jacobian. The slot is stored again only when the result's arrays
    change, since storing the tuple in it boxes it;
  - a broadcast that plate cells only index and that is not composed lazily,
    such as `log.(w)` read as `logw[z]`;
  - the gradient of `A + s * I`, which allocated the matrix and its shadow
    on every call.
- Fixed: in a non-allocating kernel, a `@traceable` helper's recorded source
  replaced a more specific plain method of the same function, such as a
  `Float64` specialization, so the kernel computed the generic method's
  result. The source now stands for a call only when its method is the one
  the call dispatches to. `_traceable_source` returns that method's signature
  as a fourth element.

## 0.1.112

- A prepared body with two or more top-level plates runs each plate's loop as
  a program of its own (`_PlateProgram`), which the body calls where the plate
  stood. Compiled into one function, the loops did not vectorize alike: LLVM
  vectorized the first and left the others scalar. An RKPPL model with Normal
  likelihoods over 2000 rows each, measured on a Linux fleet host (Julia 1.10.12), took
  0.22–0.30, 0.69–0.70, 0.96–1.60 and 1.13–1.21 ns per row and likelihood
  with 1, 2, 4 and 8 likelihoods; it now takes 0.20–0.25 at every count.
  Values, gradients, allocations and compile times are unchanged, and a body
  with one plate keeps its loop inline as before. `code_expr` prints each plate
  program in place as `(function (args...) … end)(args...)`. Ordinary AD
  differentiates the body with its plate loops inline: a reverse sweep
  vectorizes nothing, and registered Enzyme refuses a bound array passed beside
  an active value in a plate program's argument tuple. Only `prepare`d bodies
  do this, and only plates that keep no per-cell tables; a `passes` transform
  sees the plates inline.

## 0.1.111

- Reverts 0.1.110. A data-bound plate branch again prepares only the arms its
  lanes take, so a binding whose lanes all take one arm runs its unsplit plate
  again (an RKPPL censored Normal response with a bound scale and every row
  observed: density back to 0.47 µs from 3–4 µs at 2000 rows), and another
  pattern of taken arms again prepares, and compiles, its own program. The
  one-program behaviour returns once split arms keep that speed.

## 0.1.110

- A plate cell's lazy branch whose condition reads bound data keeps one
  prepared program whichever arms the binding's lanes take. Preparation splits
  such a plate into one plate per arm; it used to drop an arm that no lane
  takes and to leave a plate with no lanes unsplit, so data whose lanes all
  took one arm, or that had none, prepared another program and compiled again:
  an RKPPL censored response with a bound scale, for example, whose rows were
  all observed or all at their censoring bound. Every arm is now a plate, over
  no lanes when none takes it, and a domain of no lanes is split too. Such an
  arm evaluates nothing: its shared work stays behind the plate's nonempty
  guard, and its work over bound values, such as `log(s)` of a bound scale, is
  not run at preparation, so an invalid value on an arm no lane takes still
  never throws. An arm of one lane now splits a nested branch as arms of more
  lanes or none do. Values and gradients are unchanged.
- A binding whose lanes all take one arm now runs the program a mixed binding
  runs: the taken arm reads its live arguments through lane indices beside the
  other arm's empty plate, where it used to be one unsplit plate, and its
  first gradient also compiles the arm no lane takes. For an RKPPL censored
  Normal response with a bound scale and every row observed, measured on
  a Linux fleet host (Julia 1.10.12): density 0.47 → 3.2–4.3 µs and gradient
  4.1 → 6.0–8.2 µs at 2000 rows, and `prepare_sampler` with its first gradient
  in a fresh process 14–15 → 32–33 s at 200 rows. Those are what the same
  model with half its rows censored takes, before and after this change
  (13 µs, 32 µs; 31–36 s).
- A condition that reads bound numbers alone still selects its arm at
  preparation, and a bound singleton beside live arguments of unknown length
  (a plate of one row) still prepares its own program.

## 0.1.109

- A partition sum lowered under an owner guard reads a plate its group fills
  under that guard only after the fill. A group of guarded sums (0.1.90)
  places each statement under the guard of what it reads, and moved a name
  under a guard only when the guarded statement assigned it. A plate's buffer
  is allocated before its fill loop, and that loop writes it in place
  (`buffer[i] = v`), so a buffer allocated unguarded and filled under a guard
  stayed unguarded. Its readers then ran before the guarded block, reading
  rows that were never assigned. This hit the local partition derivative
  (`partition_value_and_gradient!(…; shared = false)`) of a model with a
  unit-owned non-centered completion beside a centered shared one, BRM's
  `mi(a) ~ Normal(mu_a, 1)` with `mi(b) ~ LogNormal(0.5, 0.3)`: its primal
  and derivative raised `UndefRefError` in `_cell_joined_rows` (snag
  `cell-joined-rows-1ab77c3c`). An in-place write (an indexed or property
  write, an updating or broadcast assignment, a mutating call) now binds the
  written name as an assignment does, so its readers follow the guarded fill.

## 0.1.108

- A program that calls a child kernel in a generator term, a loop or a lazy arm
  keeps one prepared type across builds. Such a call embeds objects in the
  compiled body: the child's source operations, spliced into the caller's
  source, and a child prepared when its caller is defined, either called under
  the arm or captured by an operation (an object's in-place endpoint such as
  `prepared standardized`). The body's content id reads every field of what it
  embeds, and some fields named the build: an operation's definition token and
  module, and a prepared child's plan, lowered recipes and value ids. A
  generator that evaluates every build in a fresh module, as
  BayesianRegressionModels does, therefore prepared another sampler type for
  every build of the same program and binding, and compiled it again: an downstream
  archive target's sampler, rebuilt with the same binding or another data
  value. The body now embeds an execution copy of each object, of the same
  type: an operation keeps its record, call identity and throw-stripped twin
  with one token and no module, and a prepared child keeps its compiled body
  and tables with an empty plan and recipes and canonical value ids. An object
  that reaches state a call could write (a `Ref`, a cache cell, a `Core.Box`)
  is embedded as before. The plan and `code_expr` keep the original objects. A
  Julia-owned closure (a generator or comprehension body) that holds such
  objects shares its type across builds the same way.

## 0.1.107

- A caller that keeps one `partition_plan` per label can make equal arrays of
  those plans one object: `_intern_partition_plan(pool, plan)` with a pool
  `_PartitionPlanPool()` per preparation (internal, `src/plate_partition.jl`).
  It replaces each array of plain bits that has the content of an array already
  in the pool by that array, through tuples, named tuples, views and broadcast
  extrusions, and keeps the plan's type and values. A plan's `partition_labels`
  is kept as it is, because the planned body compares a label's rows with its
  partition table's own rows by identity; its arrays become the ones the other
  fields share. ReactiveKernelsPPL 0.1.120 interns every plan set it prepares.

## 0.1.106

- A non-allocating kernel (`prepare_nonallocating`, and through it
  ReactiveKernelsPPL's `prepare_sampler(...; nonallocating = true)`) gives more
  array results a buffer of its own instead of allocating them on every call,
  with the same values:
  - `vcat` of vector literals of numbers beside vectors (`vcat([a], [b], v)`),
    the coefficient vector and scan trajectory ReactiveKernelsPPL writes;
  - the unary minus of an array inside a dotted call (`log1pexp.(-u)`), now
    part of the broadcast instead of a negated copy;
  - a `@traceable` helper whose body is one expression, decomposed over its
    arguments as `jax_source` translates it, and functions and constants a
    generated source names by `GlobalRef`;
  - a square dense matrix plus a `UniformScaling` (`K + sigma * I`).

  Its gradient no longer allocates for `vcat` of dense vectors or `copy` of a
  contiguous view: both copy element by element, where native Enzyme reverse
  recorded Base's loops on its tape on every call.
- On Julia 1.13, a non-allocating kernel holding a plate of kernel cells (a scan
  per column, for example) no longer copies itself to the heap on every call:
  832 B per call for a plate of three scans since 0.1.103. The plate now enters
  each cell's body directly. Julia 1.10 was not affected.

## 0.1.105

- Another binding of one program no longer compiles its partition footprint
  kernel again (`partition_footprint`, `partition_footprint_kernel`, and the
  footprints ReactiveKernelsPPL's `prepare_partition` and
  `prepare_partition_sampler` compute at preparation). The footprint lowering
  computed the positions of a read through bound data from this binding's data
  and wrote them into the generated body as a literal: a block's packed range
  when its start or end is a bound number (RKPPL's positions of every block
  after a data-sized one), and the positions of a gather by a bound index. A
  binding with other sizes then had another body and another compile. In one
  process binding an downstream model to four station networks, that body was the only
  model code each later binding compiled, in a preparation that spent 10 to 17 s
  compiling. The body
  now reads those numbers and indices from the kernel's constants table when the
  footprint runs, so every binding shares one body. Footprints are unchanged.
  A footprint restricted to a selection of positions (ReactiveKernelsPPL's
  `partition_footprint_kernel(q; shared = false)`) keeps its literal positions,
  projected onto the selection once, because its body holds the selection
  itself.

## 0.1.104

- `partition_footprint(kernel, args...)` of a kernel prepared from `exemplars`
  takes arguments of other types too (another label type, a view of the point):
  it returns the footprint of the same plan lowered from those arguments' types,
  preparing one footprint kernel per type. Before, such a call threw the
  kernel's exemplar-type `ArgumentError`. `partition_footprint_kernel(kernel)`
  and the kernel itself stay typed for their exemplars. ReactiveKernelsPPL
  0.1.114 prepares its partition kernels from exemplars and relies on this for
  labels of mixed types.

## 0.1.103

- A non-allocating kernel no longer has its own type per preparation.
  `prepare_nonallocating` compiled the kernel's constants, callables and caches
  into its generated body, with a token per preparation so that two kernels never
  shared a slot. Each preparation was then another type and another compile, and
  a bound number's value was a literal of that body. The body now holds code only,
  and each call passes it one reference to the kernel's tables and caches.
  Preparations of one program share one body and one type across the values of a
  bound number, so ReactiveKernelsPPL's `prepare_sampler(...; nonallocating = true)`
  keeps one type, as the ordinary sampler does since 0.1.95. Each kernel still
  owns its slots, and a `deepcopy` now owns its own; it used to share them with
  the original through the body.
- Another preparation of such a program no longer compiles its primal, which took
  1.1 s per preparation on 12 gather blocks and 8.4 s on 48. The density reads
  bound numbers at run time instead of as literals. Against the old body, in one
  process at one load, a gather or regression with a bound scale took 1.05 to 1.24
  times as long at 120 and 2000 rows, a gather with a sampled scale 1.02 to 1.06
  times, and bound priors, group plates and scans 0.58 to 1.02 times; every case
  stayed faster than the ordinary kernel. A plate
  of array cells, whose body Julia does not inline, costs 117 ns per call against
  67 ns.

## 0.1.102

- A partition sampler or query with a cached shared prefix (`partitioned=`)
  keeps a plate read through a type guard at each partition's rows. One example
  is a per-state field plate whose cells call an opaque helper over parameter
  arrays, read as `eltype(cells) <: Real ? cells : throw(...)` (the guard
  BayesianRegressionModels writes for an offset of a plate output). The split
  had decided the guard from the declared packed type `AbstractVector{Float64}`.
  Over that type the helper infers `Any`, so the guard kept no arm, and the
  plate, the guard and everything after them were lifted into the prefix. Every
  partition gradient then pulled back through all of the plate's cells. The
  split now decides this for a dense `Vector{Float64}` point, the type these
  queries are prepared with. Its prefix and residual keep the branch, so each
  point type they are prepared for proves its own arm. Values and gradients are
  unchanged. A unit's gradient now costs its own rows, not the whole plate.

## 0.1.101

- A read-only gather of a `Vector` by a `Vector{Int}` (`_readonly_gather`, which
  RKPPL's `theta[child[j]]` in a dotted plate cell reaches) no longer runs
  `Base.view`'s alias check, as a gather of a packed slice already did not. The
  result type is unchanged, and the gather still checks bounds. On a ragged
  nested plate over a computed parameter, such as a non-centered PosteriorDB
  Bones with `theta = 36 .* z`, value and gradient took 1.37 to 1.39 times the
  hand-written loop on Julia 1.12 and 1.13, and now take about the same as it.
  Julia 1.10 is unchanged, and so are values and gradients.

## 0.1.100

- `PartitionLocalityError` and the `locality = :warn` warning name the route for
  a value that reads only shared parameters: `partitioned` in the queries of
  ReactiveKernelsPPL 0.1.111, or `partitioned = :none` when partitions own no
  parameter. The refusal also says that `locality = :warn` computes such a value
  again for every cell or partition, instead of offering it as the only way on.

## 0.1.99

- Building one program twice in a process, as a generator does whenever it
  evaluates its definition again, gives the types the first build gave, so the
  second build compiles nothing more. `@kernel` named what the author left
  unnamed (a lifted cell invariant, a computed endpoint binding, a plate's
  computed argument) with `gensym`, whose counter moves on between builds, and
  those names reach types: a split recipe carries its ports in its type, an
  endpoint call its signature. Each build then had new types, and every reader
  of them compiled again. A generated name is now a function of the definition
  and of its position in the expansion, in the shape of a `gensym`. A source
  operation's provenance token is a field of its key instead of a type
  parameter. A bound plate's axes and a child snapshot's values no longer
  follow value-id order, which differs between builds.
- Two definitions of one source still stay two computations, and a binding
  from another program is still refused. Values and gradients are unchanged.

## 0.1.98

- A plate cell whose lazy branch is decided by bound data, such as a response
  density under `noise > 0 ? normal(mu, noise).logpdf(y) : -Inf` with `noise` and
  `y` bound, runs its arm's work over bound values at preparation instead of on
  every lane. `log(noise)` is one value of the binding, read at run time, and
  `log(y)` one value per lane, so a known scale costs no more per call than a
  sampled one. An arm's work over literals that the compiler does not fold, such
  as a written Student-t's `logbeta`, moves the same way. Gradients are unchanged,
  and values agree up to the rounding of a loop that now vectorizes. An invalid
  bound scale still takes the fallback arm without evaluating the other.
- This removes the per-row `log(scale)` that 0.1.95's entry describes. On its 12
  blocks of 2000 rows the gradient falls from 291 to 126 us and the density from
  203 to 57 us, in one process at one load.

## 0.1.97

- A range of scalars (`lo:hi`) inside a larger expression is no longer split
  out as a cached partial-evaluation value: constructing it costs no more
  than the arithmetic already kept inline. `view(A, lo:hi)` with `lo` and
  `hi` bound data therefore keeps its range-view row form, and partition
  footprints, shared-prefix read sets and cell shapes resolve its bound
  endpoints exactly as literal ones.
- `reshape(v, n, k)` with every dimension an input now has the same row form
  as `reshape(v, n, 2)`: a cell reads only the packed positions its own rows
  hold. A bare call of inputs alone used to be computed whole.
  ReactiveKernelsPPL 0.1.108 reads its packed blocks both ways.
- A plate operand inferred as `Union{}`, as the cell of an empty plate can be, no
  longer raises `Tuple field type cannot be Union{}` while typing the plate's
  output: the plate's region validates its axis, as for an unknown operand.

## 0.1.96

- A destructuring outside a plate whose right-hand side holds a data-only part or
  an index or sum boundary, such as `(s, q) = f(exp.(x), a)` under `bound = (; x)`
  or `(P, w) = f(t, modes[1])` in a model passed to `prepare_cell_factors`, no
  longer fails with `ArgumentError: Tuple contains 2 elements, must contain
  exactly 1 element`. It splits as a statement with one name does: the data-only
  part runs once per binding, a sum over gathered rows becomes a cell reduction,
  and the call still runs once per evaluation and binds every name.

## 0.1.95

- A bound number's value no longer changes a prepared derivative's type.
  `prepare_ad` rewrites the generated body when a bound value is a tuple or a
  record holding arrays, and that rewrite spelled every bound number in it: a
  scalar beside that value, and the value's own numeric leaves. Each value was
  then another generated function, so another type and a new compile of the
  whole derivative. Numbers, and pointer-free immutables that hold them (a range,
  a scale), now cross as hidden operands like the arrays do. Values and
  derivatives are unchanged. A leaf that holds pointers but no array (a `Ref`, a
  `String`) stays a literal of the rebuild.
- A program that took that rewrite no longer has those numbers folded into its
  compiled derivative, so its gradient computes what they had folded away. A
  bound response scale then computes `log(scale)` on every row in the gradient,
  as it already did in the primal and in every program without the rewrite. On
  12 blocks of 2000 rows a gradient took 113 us with the numbers in the body and
  271 us without them, at one load in one process. Bound prior numbers, which
  run above their loop, cost about 6 percent. Lifting that work out of the loop
  is tracked by snag `bound-response-s-9739d33d`.

## 0.1.94

- A cell or partition query of a tuple-valued plate whose readers read different
  entries runs each of the plate's statements only at rows where one of those
  entries is read, still once per row. A label that reads one entry neither
  computes nor reads what only another entry needs, so `partition_footprint` and
  `partition_separability` no longer report those coordinates for it. Values are
  unchanged.

## 0.1.93

- `prepare` with `bound=` data no longer fails with a `MethodError` (`Cannot
  convert` a tuple to `Int64`) when a data-only computation under two guards of
  a plate cell throws, such as `sqrt(d)` for a negative `d` in an arm of an
  inlined endpoint. The computation stays under both guards, as authored. The
  failure came from 0.1.88's typed sets for nested-arm speculation.

## 0.1.92

- In a partition over several plates lowered with one cell emitter, work that
  only some of the plates read runs when one of those plates has a cell with
  the label: a whole value shared by two plates that both have no cell for the
  label is neither computed nor read, as one only a single plate reads already
  was (0.1.90).

## 0.1.91

- `partition_plan_kernel(kernel)` and `partition_plan(kernel, …)` lower a
  label's plan from the argument types `kernel` was prepared with (its
  exemplars, `nothing` at the plan port), as `prepare` lowered the kernel. A
  kernel prepared from exemplars types its plan port by that lowering, while
  the plan kernel lowered without those types and could compute a plan with
  other fields, which the kernel then refused (`this kernel was prepared from
  exemplar argument types …`). RKPPL's `prepare_partition` and
  `prepare_partition_sampler` with `partitioned` (a shared prefix) failed this
  way on a joint vector observation cell, such as `y[i] ~ MvNormal(…)` in an
  `@plate for` loop. Kernels prepared without exemplars are unchanged.

## 0.1.90

- A partition sum that shares a computed value with the other sums of its
  partition (so they lower with one cell emitter) and has no cell with the
  label now computes and reads none of its own work: whole values only it
  reads, the rows of values only it reads, and its plate run under its
  nonempty-row guard, and the sum is zero otherwise, as for a partition sum
  lowered alone. Before, an empty sum in such a group still evaluated that
  work, so the label's footprint held coordinates only the sum's other labels
  read (`partition_separability` then refused them, for example RKPPL
  `prepare_partition` over several observations with one observation absent
  from a unit), and invalid values there (a negative `log` argument) failed the
  label. Values the sums share are still computed once, at the union of their
  rows, and row bookkeeping that reads only the label and bound data stays in a
  partition's plan. A partition sum with a separability rule also guards
  whole work that reads only scalars of the ruled port (`u[7]`).

## 0.1.89

- Composed cell broadcasts read direct row-slice operands in place, including
  rows of producer windows, without changing ordinary slice ownership or
  dispatch. Planned partitions retain only the selected rows of bound dense
  operands, preserving their view types and axes, so ordinary reverse calls
  do not allocate shadows of an unselected parent panel. Lazy cell-view
  construction stays visible to its readers across planned array operations.
  Contiguous window gathers check the window span once before their row loop.
  Shared-prefix splitting keeps parameter views in the residual when their
  readers need only selected rows, avoiding whole-block derivative frontiers.

## 0.1.88

- A plate nested in another plate's cell decides a lazy branch whose condition
  reads only a number that its cell received from the outer plate (an element
  of a `Vector{Float64}` or `BitVector` operand, for example) once per outer
  cell, as a top-level plate already does for a shared condition, instead of
  in every nested cell. RKPPL's support guards on a ragged observation group
  (valid cutpoints, positive discrimination) now leave the nested cell loop.
  Values, ordinary derivatives, laziness and empty-domain behaviour are
  unchanged.
- Binding a guarded arm of a nested cell whose speculation excludes nothing no
  longer fails with a `MethodError` (`_partial_split_recipes` and a
  `Set{Union{}}`) on Julia 1.13; RKPPL's group-indexed `Ref` cutpoints with a
  cumulative or stopping-ratio ordinal prepare again.

## 0.1.87

- A kernel prepared from a module that replaced another module at the same
  path (a REPL or test file `include`d again, `include_string`, re-evaluating
  a submodule of a package) runs its own module's functions. Before, it could
  run the code compiled for the replaced module, including a package image's
  program for it, so an edited definition silently kept its old behaviour.
  Kernels of the replaced module keep running that module's code.

## 0.1.86

- Position-batched readers read a slice that is directly an operand of a dotted
  call (`a .* x[k]`, `f.(x[k])`) in place inside the fused broadcast instead of
  copying it into a lane slot first, and a `vcat` of numbers and dense vectors
  in a position residual (`vcat(0.0, v[rows])`) fills a lane slot instead of
  allocating at every cell and position. A lane slice checks its indices once.
  Values, element types and errors are unchanged.

## 0.1.85

- A planned partition's array statement that takes a type or a function
  argument (for example the output allocation `similar(marker, T, axes)`)
  is compiled for that exact type or function behind the statement wrapper,
  instead of dispatching at run time. Preparing such a program no longer
  warns that `_partition_array_call` is not type stable. Values are unchanged.

## 0.1.84

- Fixed selections prepared by frontends can pass precomputed partition rows
  without allocating and scanning a key for every unselected plate cell.
  Partition domain checks and ordinary native evaluation are unchanged.

## 0.1.83

- Plate-cell and producer-window footprints follow visible gathers inside
  scalar generator sums with `init`, including views, data-only index helpers,
  nested sums and filters. Direct scalar captures also request only the needed
  rows of eligible producers, including copying packed-parameter slices.
  Shape-based views retain their storage identity. Index programs keep runtime
  loops, the authored callback and ordinary Reverse; unproved or lazy reads
  retain conservative footprints.

## 0.1.82

- Partial evaluation preserves fusion between live dotted subexpressions while
  retaining cached data-only operands. Position-batched borrowed readers of
  indexed ragged vector cells reuse their cell buffers without allocating a
  temporary vector per cell and position, including with bound data. Source
  that requires lexical captures keeps its existing closure path.

## 0.1.81

- Multi-worker `NativeScheduling` calls also schedule materialized data-only
  prefix plates within their worker budget. Equal cells share one result per
  call across workers; changed inputs are recomputed on later calls. Prefix
  tasks join before position workers begin. Scalar dependencies, reductions,
  scans and one-worker calls keep their native order.

## 0.1.80

- Lazy plate arms retain complete positional single-output child graphs over
  named ports. Their shared recipes can run once within the selected arm,
  including after binding its condition, while empty and untaken arms remain
  unevaluated.

## 0.1.79

- Bound preparation can reuse data-only prefix results in a frontend-owned
  binding store, including intermediate results and split data subexpressions.
  The shared compilation caches retain no binding's values. Ordinary core
  `prepare(...; bound)` calls keep their per-preparation behavior.

## 0.1.78

- `derivative_rule` accepts literal `nothing` entries in its `directions` and
  `cotangents` mappings to declare data-only inputs, including structured data.
  Every input must still be named, and both branches must agree on data-only
  positions. No dummy graph direction or cotangent output is needed. Forward
  cuts take directions only for differentiable inputs; reverse activity masks
  keep the original input positions and reject data-only bits. Enzyme skips
  directions for `Const` data and rejects active data; ChainRules pullbacks
  return `NoTangent()` at data positions.
- Batched reverse accumulation through generated vector rules no longer hits
  an ambiguous method at the end of the argument tuple.

## 0.1.77

- A cell or partition of a product over a bound sparse factor, `L * eta` with `L`
  a `SparseMatrixCSC` bound as data and `eta` a dense vector or matrix, reads
  `eta` only at the columns the rows store. It computes `eta` at those rows and
  its footprint holds them, instead of the whole operand; values equal the
  authored sparse product exactly, and the rows are indexed once, when the data
  is bound, so a cell costs its own stored entries. A dense factor's numeric zeros
  still remove no read. The product must be a statement of its own: inside a larger
  expression it still reads `eta` whole. Needs `SparseArrays` loaded, as a bound
  sparse matrix does.
- `partition_footprint` reads a view or a read nested in a larger expression at
  its literal positions, as in `tanh.(view(u, 2:3:8))` or `exp(u[3])`. It used to
  report the whole port, which hid every other precise read of it: one stacked LKJ
  edge made a stratified group effect's cells report every subject's coordinates.
  A shared-prefix split reads such a recipe the same way.

## 0.1.76

- Internal support for ReactiveKernelsPPL 0.1.84's handling of a sum whose
  summands read another sum (`_cell_factor_nesting`, `_cell_factor_sum_cell!`,
  `_CellFactorSumRead`). No public API changes.

## 0.1.75

- Cells that read one shared plate compute it once. A partition over several
  plates (`plate_partition`), or several `plate_cell`s at one position, that
  read the same plate at their cells (an `@plate for` cell observed twice)
  now compute that plate, and every value computed at rows, once at the union
  of the rows they read, as the whole kernel computes it once per cell.
  Earlier versions copied the shared plate into each reading cell and gave
  each cell its own row computations, so one unit's partition over two
  observations ran the shared cell body twice. A shared plate whose rows
  cannot be computed this way is still copied into each cell, so locality is
  unchanged.

## 0.1.74

- A dotted expression whose operand is a number computed from literals and named
  constants, `sigma .* (rho .* sqrt(2pi)) .^ (1 / 2)`, now has a row form, so a
  strict cell computes it at its rows. A parameter-free call that returns an
  array keeps no row form.
- A package can give its own function a row form with
  `_cell_row_method(::typeof(f)) = (; at, rows, dims)` (the compiler page,
  "Locality: named, not silent"). ReactiveKernelsPPL 0.1.83 requires it.

## 0.1.73

- A cell reads one element of a computed matrix at its row. `b[q, 2]` in a plate
  body, with `q` the element of a batched integer argument and every later index
  a literal position or `:`, now computes `b` at the rows `q` selects when `b`
  has a row form, as `v[q]` and `M[q, :]` already did. So does a top-level
  literal read such as `b[1, 2]`. `b = z * F` then reads only those rows of `z`,
  in strict cell queries and in cell factors, whose flattened bodies used to
  refuse this read.
- A scalar read keeps its row demand when binding data splits its expression.
  Binding hoists `paths[p]` out of `exp(a[p]) .* paths[p]` and leaves `exp(a[p])`
  as one closure, which read `a` whole. A cell factor prepared with `partitioned`
  refused it as a value of unknown type before ReactiveKernelsPPL 0.1.80, and
  computed `a` whole in every term from that release until this one.
- A footprint of a matrix port read as `X[q, :]` or `X[q, 2]` in a cell reports
  the positions of those rows. It used to report the rows' indices as linear
  positions, which missed the later columns.
- **Breaking:** under `locality = :error`, a matrix with no row form read this
  way is now refused by name, as `v[q]` of such a value already was. It used to
  count as read whole legitimately, and every cell computed it whole. Give it a
  row form, compute it from shared coordinates only and declare the others
  `partitioned`, or pass `locality = :warn`.

## 0.1.72

- Equal plate cells also share a value computed by a plate that reads data
  alone, in a position-batched reader's prefix and in `bound=` preparation. A
  design in a plate of its own ran once per cell, kept one copy per cell, and a
  plate reading it hashed the whole design for every cell. It now runs once per
  distinct input and keeps one copy. The plate's cell values are one object per
  equal group, as a moved cell value's already are. A plate the caller receives
  keeps one array per cell, and a plate of Base arithmetic keeps its loop.
- A shared call compares its arguments by identity first, then by value when
  their objects are new, so it hashes each distinct object at most once however
  many cells hold it. A call of a package's function now also shares equal values
  in different objects. Each subject's rows of a data column are different indices,
  so equal subjects hold equal values in fresh objects: such a call ran once per
  subject and now runs once per distinct input. An expression of Base's functions
  over a value equal cells usually hold as one object still compares by identity
  alone.
- A field read `x.name` bound to its own name in a plate cell no longer keys the
  whole record.

## 0.1.71

- A code finding of the type-stability check now names where its statement was
  written: under the statement, one `at name @ file:line` line per frame,
  innermost first, including the functions it was inlined from and the macro
  expansions between them. The warning and `TypeInstabilityError` print the same
  text. Before, an instability in a helper inlined into the prepared program was
  reported as "the prepared program's entry call" with only an SSA index and the
  statement. The warning's once-per-process key now includes the location, so
  the same statement written elsewhere warns again.

## 0.1.70

- Per-position helpers gain row forms for more shapes. A matrix is a position
  argument over its first axis (`x[g, d]`, with `axes(x, 1)` or
  `1:size(x, 1)`); a `zeros`/`ones` result may be read and written at its own
  row any number of times; a prologue may compute values from arguments passed
  whole (`L * reshape(z, k, :)`); and each arm of a branch may write the row, as
  long as every path writes it exactly once. `vec(x)` of a value with rows reads
  only the rows of `x` its positions lie in.
- A helper's row may hand a whole array to a closure it builds or to a helper
  with one applicable method, when that code only indexes it
  (`sum(w[t] * cs[ix[t]] for t in eachindex(ix))`). A row that reduces or
  iterates the array stays without a row form.
- A `view(A, lo:hi)` of an input that one reader reads whole and another at rows
  counts as an input, since it builds no elements. `A[lo:hi]` copies and stays a
  strict-locality violation.
- A cell's outlining also uses the strict split boundaries when the ordinary ones
  hold nothing, so `sd .* (z .- total / n)` has a row form on the cell-query path.

## 0.1.69

- **Breaking:** `plate_partition` refuses a key whose length differs from its
  plate's cell count, with an `ArgumentError` naming the key and both counts. A
  shorter key used to be accepted: the cells past its end belonged to no
  partition, so the labels no longer summed to the plate. A longer key used to
  raise a `BoundsError` only for a label holding a position past the end. A
  partition prepared with its label bound is not checked, because binding the
  label folds the rows the check reads.

## 0.1.68

- `plate_partition`'s `by` accepts the pairs its docstring promises: one
  `plate => key` pair or a collection of such pairs (a vector, a tuple, a
  generator), besides one name, a named tuple and a dictionary. A vector or
  tuple of pairs used to be read by position and refused as giving no key, and
  a single pair raised a `MethodError`. Any other `by`, such as a vector of
  names, raises an `ArgumentError` that names the accepted spellings, as does
  a `by` that gives no key for a named plate; a plate given two different keys
  raises an `ArgumentError` naming both.

## 0.1.67

- Footprints follow reads through views: a view contributes the parent
  positions its readers read, including views whose indices read the parent's
  shape. A typed local that converts a view into a copy still reads every
  viewed position. Inside a plate cell, a shared array read through views and
  argument-returning shape checks is read at the positions the cell's
  selections reach, so a strict cell that selects entries of its own block
  reports those positions only. Gathers inside one compound expression
  (`w .* v[I]`) count as reads of the gathered positions.
- A cell reads a field of existing storage at its rows: a named tuple's
  vector field, or a field of a concrete struct whose `getproperty` is
  Base's `getfield` fallback (`rows[i].weight1`). A computed property keeps
  its whole-value path.
- A package's argument-returning check with a validator
  (`_checked_alias_validator`) passes its argument's rows to a strict cell
  and validates the whole argument once from its type and axes. `view(v, I)`
  of a vector reads a cell window's rows as a gather does.

## 0.1.66

- Compute per-position helpers at a cell's rows. Comprehensions and `map` over
  `eachindex`, `axes(x, 1)`, `1:length`, `enumerate` or `zip`, and loops that
  build one row per position by a single `push!` or `out[g] =`, are proved from
  lowered code. Arrays a row only indexes elsewhere (`field[ix]`) are passed
  whole; cross-row or carried work, including a whole-array reduction in a
  row, keeps the existing locality checks.
- In a strict cell factor, an index into one cell's own value (`row[hour]` of
  `row = air[traj]`) reads that value instead of demanding its rows, so outer
  factors of per-trajectory rows agree with cell queries.

## 0.1.65

- `deepcopy` of a graph, plan, prepared or vectorized kernel (any object
  holding a generated program) now succeeds when a recipe's authored source,
  or a generated program that inlines it, holds a module as a literal value, as
  ReactiveKernelsPPL's rebuilt law arguments do (`Core.GlobalRef(<module>,
  :name)`). The copy owns its expressions and shares the modules in them, as
  it already shared a source operation's module. Previously the copy failed
  with `deepcopy of Modules not supported`.

## 0.1.64

- Apply an elementwise dotted call over an expanding gather by a cached index,
  `f.(x[k])`, before the gather: under `bound=` and in position readers'
  per-cell prefixes, a cell whose index reads every element of `x`'s first
  `m` and is longer than `m` evaluates `f.(view(x, 1:m))[k]`. Values are
  unchanged bit for bit and `f` runs on no element the authored form leaves
  unread; reverse-mode derivatives agree up to floating-point rounding where
  an index repeats. See "Elementwise calls over a data-indexed gather" in
  `docs/src/compiler.md` at revision `4d462ef4170948f2e77e3fa80cda5be5c2b6dc02`,
  before the documentation rewrite removed that page.

## 0.1.63

- Keep a shared-prefix cut's input types when preparing cell factors with
  `partitioned`. A cell's per-index reads of earlier plates (`loc[i]`) are
  then scalar reads, as without the cut, instead of a group sum whose rows are
  the cell's own position; genuinely gathered sums prepare strict summands.
  Factor discovery no longer takes a scalar position as a sum's row labels.

## 0.1.62

- Carry row demand through a selected array-valued producer cell, preserving
  both its outer index and the requested inner rows under strict locality.
  Locality reports and footprint kernels retain a prepared kernel's exemplar
  type evidence and call contract, preserving the ignored plan argument.

## 0.1.61

- Supply compact Gaussian marginal, conditional and replacement algebra for
  selective carried inference. Diagonal plus bounded-rank covariance keeps
  storage linear in dimension and supports ordinary native derivatives.

## 0.1.60

- Evaluate binary ordinary array addition and subtraction at requested rows
  when Base arithmetic over standard numeric arrays is proved, including
  transparent forwarding helpers. Preserve full-array shape checks and
  guard the original operand types before applying the local row form.
- Use typed row forms when discovering a scalar cell's sum factors, so
  gathered operands combined with ordinary addition retain their row labels.

## 0.1.59

- Treat the Boolean `!` operator as a pure endpoint call, rather than a
  mutating method whose name ends in `!`. This permits lazy support guards
  in composed distribution moment graphs.

## 0.1.58

- Support local packed-coordinate footprint projection in RKPPL cell queries.
  Footprint lowering can omit shared frontier ports and intersect constant
  reads before expanding them, avoiding shared-block enumeration per cell.
  Ordinary full structural footprints retain their existing semantics.

## 0.1.57

- Constructed RK object endpoints accept plate-local method arguments, as
  their constructor arguments already do. A call such as
  `normal(mu, sigma).logpdf(transformed)` may read `transformed` computed in
  that cell, retaining its graph, lazy branches and native reverse derivatives.
- Endpoint constructor and method arguments also recognize local assignments
  inside lazy branch blocks, preserving their evaluation within the guard.

## 0.1.56

- Derive finite scalar recurrences from recorded `@traceable` vector helpers
  for streaming consumers. Retain the original arithmetic, lazy branches,
  output conversion and ordinary derivatives without executing the helper
  during preparation or unrolling its data-dependent loop.

## 0.1.55

- Compose shared upstream plates into selected consumers with independent
  index ports. Each consumer evaluates only its selected cells, including
  different row selections, while whole-plate queries retain their boundary.

## 0.1.54

- Add `prepare_ad_accumulator`, `ad_accumulator_storage` and `ad_accumulate!`
  for additive gradients of scalar directional contractions. Directions can
  cover a subset of the active boundary while the result includes all mixed
  derivatives. Native reverse Enzyme writes into supplied cotangent storage
  without clearing or copying large shared operands between contributions.
- Preserve selected shared-prefix cuts when discovering cell sum factors.

## 0.1.53

- Recover row locality through ordinary per-index helper comprehensions and
  straight-line wrappers. Selected cells retain their gathered-coordinate
  reads, including scalar operations and array-valued element helpers, using
  runtime loops and ordinary native derivatives. Analysis executes no helper
  and retains locality checks for cross-row or unproved helper bodies.

## 0.1.52

- Discover cell sums through typed identity assignments that already satisfy
  their declaration, allowing earlier reduction plates to compose with their
  consuming cells. Assignments that can convert the value remain in the graph.
  Preserve a standard plate buffer's known rank when its element type is
  unknown, so chained plates can prove an abstract array declaration redundant.

## 0.1.51

- Carry exact exemplar argument types through selected-cell locality analysis,
  retaining narrower inferred types at abstract ports. Scalar plate collectors
  can then preserve locality through ordinary Julia helper calls.
- Preserve lazy guarded reads of captured arrays while constructing selected
  row windows. Inactive invalid indices request no rows; an active invalid
  read still raises its bounds error in the authored cell body.

## 0.1.50

- Keep repeated gathers from different columns of one source local to the
  requested cells. Gather shape analysis now retains each pattern of surviving
  axes instead of requiring identical trailing indices. Dotted predictors such
  as `B[I, 1] .* x .+ B[I, 2] .* z` preserve locality through packed views,
  reshapes and matrix products, including when data is bound.
- Preserve the row form of `view(A, I)` when binding precomputes an integer
  range into an input. The same integer-index checks as ordinary gathers apply.

## 0.1.49

- Preserve a visible lazy branch's selected shape and row form when concrete
  input types prove its condition. A scalar plate passed through a collector
  such as `eltype(cells) <: Number ? cells : reduce(vcat, cells)` retains
  strict selected-row locality through a following broadcast offset, with
  ordinary native Reverse derivatives. Analysis executes no condition or arm;
  runtime and effectful conditions retain their authored behavior.

## 0.1.48

- Preserve strict row locality when splitting shared partition prefixes.
  Row-capable gathers and broadcasts stay in the residual; opaque shared
  producers become cached inputs. Literal parameter slices remain eligible
  for the shared boundary. Preserve inferred numeric element types and ranks
  at the new boundary, including when a cached view becomes an owned copy.
- Include selected lowered-plan producers in AD activity reachability, so
  outlined active outputs retain their derivatives even when their recipes
  are absent from the source graph.
- Compute array-output pullbacks through the seeded inner product. Returned
  views over read-only array wrappers no longer require a writable output
  shadow; ordinary reverse and forward AD retain the same primal graph.


## 0.1.47

- Preserve field dependencies when factoring a cell that shares array-valued
  tuple outputs between observations. Literal reads of an unconverted tuple
  forward its individual fields; selecting an existing ragged input remains
  an input read. Strict factor queries retain local packed-coordinate reads
  and ordinary Reverse derivatives for censored ragged observations.
- Retain numeric element types through a shared transposed broadcast such as
  `transpose(scale .* L)`, so downstream scalar ratios remain recognizable
  as scalars during strict cell preparation. Matrix transposes still require
  the original whole shared operand and acquire no row-local implementation.

## 0.1.46

- Keep cell capture reads local through a scan's unchanged shared arguments.
  A scan reading `sigma[rows[i]]` inside subject cell `i` now computes
  row-wise sigma producers at that subject's rows, with the same structural
  footprint and ordinary native Reverse derivatives. Bound indices and known
  shapes must prove the reads valid, since an empty scan executes no steps.
  Unknown or invalid proofs retain whole captures; scan carry, history and
  iterated elements are not treated as unchanged cell indices. This repairs
  foreign sigma-effect reads in BRM's grouped AR1 likelihood without changing
  its model (`prepare-a-strict-3790e275`).

## 0.1.45

- Row forms for the grouped field BayesianRegressionModels emits for
  `hsgp(x; by = g)`: a reduction over the axes other than the first
  (`sum(X; dims = 2)`, also `prod`, `maximum`, `minimum`; alone, under `vec`
  or under `dropdims`) whose operand is an input or a broadcast operand, a
  gather `A[I, :]` reading whole rows of its source, and `transpose(w)` /
  `adjoint(w)` / `w'` of a vector as a broadcast operand or alone. A plate cell
  or partition reading `vec(sum(PHI .* scaled_z[g, :]; dims = 2))` with
  `scaled_z = z .* transpose(weights)` now computes the field at its rows
  only, with ordinary native Reverse derivatives and a row footprint, instead
  of refusing it under `locality = :error` (`PartitionLocalityError`, "its
  producer `vec(sum(PHI .* rows; dims = 2))` has no row form"; snag
  `run-brm-s-full-g-2dc0667a`). A `vec` that would interleave rows is refused
  as `:operands` when the operand's rank is known, else checked by the window.

## 0.1.44

- Keep cell footprints precise through retained lazy branches when bound
  index data proves both arms' capture reads in range. Direct matrix reads,
  named gathered columns and shared ragged index vectors use the existing
  row windows; structural supports union both arms while arithmetic stays
  lazy. Unknown or invalid index proofs retain conservative whole-capture
  reads, preserving inactive-error safety and ordinary derivatives.

## 0.1.43

- Preserve all row requests when separate cell expressions read a shared
  predictor through `vec` aliases of index arrays. Named summands followed
  by another gather now compute both row sets, including overlaps, repeated
  and empty rows. Values, ordinary Reverse derivatives and structural
  footprints retain the authored computation without traversing other rows.

## 0.1.42

- Provide the shared-prefix partition split used by ReactiveKernelsPPL.
  Structural reads of declared shared coordinates determine the cut;
  selected producers and multi-output ownership are preserved. Intermediates
  remain live inputs to the residual, allowing their cotangents to flow
  through a separate generic pullback.

## 0.1.41

- Add cold graph projection for incremental execution of authored finite
  `scan` providers. The generated transition calls their original concrete
  step kernels; pointwise work stays pointwise and whole-history operations
  are rejected. ReactiveKernelsPPL uses this to derive streaming transitions
  from ordinary provider graphs.

## 0.1.40

- Strict cells and partitions preserve ordinary scalar indexed expressions,
  including `w = exp.(a[1])` and `shift = a[2] .- a[1]`. Their vector
  producers compute the indexed rows alongside the observation's row demand.
  Shared type analysis retains array rank and recognizes inferred scalar
  shapes; partial evaluation preserves the source companions needed for
  subsequent row lowering. Footprints and native Reverse follow the same
  generated program, without author annotations.

## 0.1.39

- Preserve scalar vector reads when decomposing cell factors. Direct reads
  such as `eta[anchor[k]]` alongside sums retain row demand through the
  predictor, so outer values, coordinate footprints and derivatives do not
  traverse all latent levels. Bracket and `getindex` spellings, including
  literal indices, share the existing cell lowering and bounds checks.

## 0.1.38

- Track visible kernel inputs needed for output metadata as well as plate
  domains. Data-only array operands of concatenations and gathers can bind
  before shape analysis, preserving row-local composition when those arrays
  come from a tuple-valued helper. Shared plate captures retain their usual
  preparation staging; dependency analysis executes no recipes.

## 0.1.37

- Track which visible `KernelSpec` inputs determine authored-plate domains,
  including domains computed by earlier graph recipes. RKPPL can bind their
  data-only definitions before output-shape analysis. Dependency analysis
  executes no recipes, and authored plate loops remain runtime loops.

## 0.1.36

- A plan holds every value computed from the bound data and the label alone:
  the index bookkeeping and label-constant data (a data column's rows, a
  design block's rows and their `hcat`), computed once per label by pure
  operations (sources, plain functions, partition ops; plates, scans and
  cells stay in the body); never a value computed from an unbound port, its
  alias, a type the body computes, or a value the body writes into.
- `plate_partition(…; plan_constant = (:port, …))` names HAVE ports whose
  values are constant per label, which the plan may read; a lazy view of
  inputs serves its rows from its parts through its own
  `_cell_lazy_rows(op, inputs, rows, bind, constant)` method:
  ReactiveKernelsPPL's compact overlay reads its active vector at slots
  planned per label instead of a slot load and a branch per element.
- Planned partitions use ordinary Enzyme Reverse. Enzyme dogfood from
  `1280e6ee` repairs constant plan fields; grouped compact partitions on
  Julia 1.10 also need `a6458e9c` (snag `static-reverse-g-173719b3`).
  Registered 0.13.216 has the upstream activity-analysis defects; runtime
  activity is not an accepted remedy.
- Plan splitting preserves indexed/property writes, writes through aliases,
  mutating calls and repeated assignments in the evaluation body. Empty and
  nonempty plate portions share one plan type rather than exporting the
  final loop state (`nothing` or a `CartesianIndex`). Reusing a
  plan with another label throws `ArgumentError` instead of silently returning
  the original label's value.


## 0.1.35

- Propagate shape facts through a visible `KernelSpec`'s recipes, including
  authored plates, without executing its cells or opaque Julia helpers.
  RKPPL can preserve the row axis of a grouped plate output through a
  broadcast column, concatenated design, matrix product and offset, so a
  strict observation cell computes only its owning group. Numerical
  transfer rules remain in OutputSignatures; partial graph facts do not
  authorize numerical substitutions.

## 0.1.34

- Bound preparation caches shared data-only scalar work from nested plate
  cells, including pure expressions inside lazy branch arms. A speculative
  read that throws or produces a nonfinite floating-point value stays under
  its original loop and branch, so empty groups and inactive invalid inputs
  keep their values and ordinary derivatives. Both plate loops remain
  runtime loops, and array results retain their per-evaluation storage.

## 0.1.33

- Generated programs can compose an authored graph reconstructed by an
  explicitly registered structural constructor. The compiler resolves only
  that constructor's static syntax; ordinary calls remain ordinary calls.
  RKPPL uses this internal integration point to emit replayable caller-owned
  parameter geometry without adding global bindings or opaque runtime graph
  preparation. Nested graph recipes retain their normal locality analysis.

## 0.1.32

- Retain sum and eager-operand source boundaries for factored cell queries,
  separately from the boundaries used by ordinary partial evaluation. The
  cut pieces keep their authored error policy and use the existing graph,
  cell row lowering and derivative operators. Cell result types propagate
  from their scalar bodies when one cell supplies another cell's row index.
  ReactiveKernelsPPL 0.1.25 exposes the outer and summand queries.

## 0.1.31

- Authored source companions infer and execute when ReactiveKernels loads
  from source or a consumer prepares and warms a kernel during package
  precompilation on Julia 1.12+. The precompile wrapper type is defined before
  the source-call generator, and the generator reads a consumer's constant
  expression-cache binding as compiler metadata even when that binding is
  newer than its definition world. Strict type-stability checks remain
  enabled, and the generated numerical body is unchanged.


## 0.1.30

- Strict plate cells and partitions accept vector-valued products with an
  inline right operand, such as `PHI * (weights .* z)`, and multi-factor
  products such as `A * W * z`. When Julia infers a vector result, its full
  shape follows from the left factor's row axis; plate-computed weights keep
  the rank inferred from their batched inputs. Dotted readers can therefore
  consume only the requested rows without computing the whole product or
  evaluating its right operand to discover its shape. This fixes strict
  partition preparation for BRM HSGP predictors.


## 0.1.29

- An authored recipe's operation carries its source as a record instead of a
  closure (stage 3 of the operation-table migration, decision `0dgfkf5`).
  `_KernelSource` holds what runs: the captured values, the source compiled
  once as a runtime-generated function over them (its type carries the
  canonical source's id; the record's type the authored parameter names), or the native
  closure for a source Julia must own (a local function, a generator, an
  exception scope, a boxed capture). The recipe's operation (`_KernelSourceOp`)
  keeps the definition token, the form and the authoring module as values, not
  type parameters, and a prepared kernel's callables table holds the record,
  never the operation, so one program's fresh builds share one kernel type
  without execution copies: `_KernelSourceFunction`, `_ExecutableSourceFunction`,
  `src/execution_sources.jl`, the guarded and latest-world call modes and the
  per-process executable table are gone. A source's captured values sit in the
  constants table at its recipe's position and a capture reads as
  `getfield(__constants__[k], :c)`, so an inlined recipe's record leaves the
  callables table and a captured array crosses native AD as a hidden operand
  like bound data. `prepare_sampler`'s
  latest-world preparation and `_kernel_eval_definition`'s
  `latest_world_native_sources` are accepted and do nothing: a record is
  callable from any world. The record's call splices the companion's body
  itself, so a source embedding other sources (a plate cell calling an
  endpoint) nests one method over distinct record types, which inference
  treats as well founded; values, gradients, plans and recipes are unchanged.
- An exact call of a function or type bound `const` in its module is written
  by name in the generated body (`w = Base.sin(z)`, `n = Base.length(v)`), so
  the statement names exactly what runs and such an operation leaves the
  callables table; a kernel of inline sources and exact calls has an empty
  table. The nonallocating and derivative-rule lowerings, which key on the
  table, keep every call in it as before. A scan's hoisted step statements
  are found by their output locals, whatever their shape. A lazy branch at
  the top level runs its whole source in the body too (`c ? a : b` as an
  `if`); a plate cell's branch keeps its metadata at its position for the
  lane lowerings.
- A plate cell of one value that computes nothing only its sibling values
  read prepares again: 0.1.28 gave the cell a plan port (`plate_partition(…;
  plan)`) and the drop of its unread operands rebuilt the cell without that
  field (`_AuthoredPlateCellDraft` with four arguments), a `MethodError` at
  `prepare` (`test_plate_cell.jl`, RKPPL `test_cell_query.jl`).
## 0.1.28

- `plate_partition(…; plan = port)` / `plan = port => T`: a plan port holding
  the partition's integer bookkeeping (a label's positions, every pushed
  value's rows, gather and capture positions, the shapes it checks), computed
  once per label by `partition_plan(kernel, args...)` from the bound data and
  read by the body instead of recomputed on every call; a derivative carries
  none of it. `partition_plan_kernel(spec; have, want, bound)` prepares the
  kernel that computes a label's plan (its result's type declares the port);
  `partition_plan_kernel(kernel)` returns the plan kernel of a prepared
  partition. An untyped plan port is typed by evaluating the plan on
  `prepare`'s `exemplars`; a plan port that cannot be typed is refused naming
  the remedy. A plan holds index bookkeeping only: never a data column, a
  data value's alias, or a type the body computes. On registered Enzyme a
  planned partition's gradient needs runtime activity (the plan's constant
  row vectors meet the active vector in the body's views).
- A partition's windows carry the positions of a gather index's entries,
  computed apart from the gather (once per label with a plan); row takes are
  a positions step and a view.

## 0.1.27

- A kernel whose names end in a non-ASCII character (`p_α`) prepares again
  when its body is embedded in another lowering, an authored scan's step for
  one. 0.1.26 names an inline statement's locals after the authored names
  (`#p_α#op1`) and renumbers them on embedding, and that renumbering cut the
  name at a byte offset inside `α`: `prepare` threw `StringIndexError`
  (snag `inline-shifted-l-b520877e`).

## 0.1.26

- **Breaking:** the generated function takes two tables instead of the
  operation table `__ops__`: `function (__constants__, __callables__, ports...)`.
  Both have one entry per lowered recipe: `__constants__[k]` is the value of a
  bound constant (`prepare(…; bound)`), read as `x = __constants__[k]`, and
  `__callables__[k]` an operation the body still calls,
  `y = (__callables__[k])(x)`; every other entry is `nothing`. A recipe whose
  source runs in the body references neither: its statement is
  `lhs = let var"#x#opk" = x, …; <source> end` with locals named from the
  recipe's position, a literal statement its literal, an alias its argument,
  and a source with captures reads them from its authored closure in the
  callables table. `lower_with_ops` returns `(ast, constants, callables, recipes)`
  and the compiled function is called as `f(constants, callables, ports...)`;
  `lower` is unchanged. A prepared kernel holds `constants` and `callables`
  instead of `ops`, keeps only the callables its body references (a plate's
  or scan's operation whose loop is inlined is not among them), and a
  rebinding (`bound=` with the same graph and types) replaces the constants
  table alone. The nonallocating program's signature is
  `(__constants__, __callables__, __caches__, __cache_apply__, ports...)`,
  caches indexed like the callables. Values, gradients, plans, recipes,
  `recipe_inventory` and `explain` are unchanged; `code_expr` prints the code
  that runs. Measured on a downstream application model: the log-density
  kernel's type text 2.58 MB with the table, 536 KB in 0.1.23, 494 KB now
  (1130 of its 3660 table positions are empty; the 1747 closures kept are
  the plate cells' operations, which the plate loops still type through
  them), the bound-data prefix's 312 KB → 92 KB → 79 KB; the log density
  and the first native gradient are bitwise the table's. On the K=64
  synthetic program the prepared kernel's type text is 1.03 MB → 255 KB →
  228 KB with 648 of 1617 positions empty, and its density, gradient sum and
  gradient hash are identical.
- The display-only readable rewrite (`_readable_expr`, test-only) is gone:
  the generated code is the readable code (decision `06631v9`'s direction),
  and the tests read `code_expr` directly.

## 0.1.25

- A cell (`plate_cell`, `plate_partition`) of one value of a shared cell no
  longer computes what only the other values read. Values that share cell
  work come from one plate returning a tuple, read as `t[k]` (the cell RKPPL
  emits for an `@plate for` loop computing several observations' arguments);
  a query reading some entries already sliced the shared cell to them, but
  every value the cell captured was still computed, whole, with everything
  upstream of it. A drafted plate now takes only the shared operands its
  planned cell reads, and work nothing else reads leaves the plan. Before,
  `partition_locality` listed such a value as `:shared` (`:never_at_rows`),
  so `locality = :error` did not name it (snag `sibling-only-pla-2b3df69e`).
- `@plate for` with several outputs gives each output a plate over the loop
  statements it reads, exactly the call form written for that output alone:
  it captures, zips and checks only what those statements read. Before, each
  output's plate took every value the whole loop read, so a HAVE/WANT query
  of one output needed its sibling outputs' inputs, and a cell of one output
  computed the values only its siblings read.

## 0.1.24

- `plate_partition(spec, plates; by, index, name, locality)`: the cells of
  one or more authored plates whose key equals the label held by the HAVE
  port `index` (one subject's observations), summed, as ONE compiled pass:
  each plate loops once over the positions holding the label, and every
  value the partition's cells read at their rows is computed once per
  partition at the union of those rows. `by` gives each plate's key (one
  entry per cell, usually bound data); under `bound=` each label's positions
  are computed once per binding. A label no named plate has is refused by
  name; a plate without a cell of the label contributes nothing.
- Locality guarantee: a value a cell or partition reads at its rows but would
  compute whole is now named instead of computed silently. `plate_cell` and
  `plate_partition` take `locality = :error | :warn | :ignore`
  (`PartitionLocalityError` lists each such value with its producer's source,
  the reason and the reader that asked for rows; `:warn` names each once per
  process); `plate_cell` defaults to the project preference `cell_locality`
  (`"warn"` unless set), `plate_partition` to `:error`.
  `partition_locality(kernel)` reports the classification of every value a
  cell reads (`:rows`, `:input`, `:data`, `:scalar`, `:shared`,
  `:whole`).
- Two more producers are computed at a cell's rows: `view(A, lo:hi)` /
  `A[lo:hi]` and `reshape(v, n, k...)` with literal or scalar-input bounds and
  dimensions, the shapes ReactiveKernelsPPL unpacks packed parameter blocks
  with, so a block read at a cell's rows reads its own packed coordinates
  only.
- A cell's row window reads a vector of rows as one gather of its data at
  the rows' positions (the positions computed first); the previous
  row-by-row read made native Enzyme Reverse refuse a partition's gather of
  a product's rows (`EnzymeRuntimeActivityError`). A partition's windows
  hold their rows sorted and unique and a read finds its row by bisection,
  so a partition of r cells reading a value row by row costs O(r log r), not
  O(r²), and a value gathered at the rows of many cells is computed once per
  distinct row; a single cell's windows are unchanged.
- A plate body reading a row of a shared matrix, `M[q, :]` with `q` the
  element of a batched integer argument, captures that row as `v[q]` captures
  an element: a cell or partition reads only its own rows of the matrix.
- Locality names two more cases that were computed whole silently: a
  captured value whose index lane has no known type (`:untyped_index`), and a
  value pushed for one reader and dropped because another reader at rows
  cannot take it in its role (`:unshaped`, naming that reader). A literal
  range or reshape of a matrix (linear positions, not rows) is refused as
  `:operands`.
- `partition_footprint` reads a lane of one row as that row (extruded over
  the partition) instead of throwing, and a lowering nested in the footprint
  lowering (a plate body, an embedded kernel) no longer records or returns
  footprints of its own. `partition_footprint_kernel(kernel)` prepares the
  footprint kernel for a caller to hold; `partition_separability` takes a
  computed `footprint`.

## 0.1.23

- Each authored recipe's source runs in the generated body. A fused recipe
  used to be one entry of the operation table, a separately generated
  function of its source called as `lhs = __ops__[k](x, y)`; the body now
  carries the source itself,
  `lhs = let var"#opk" = __ops__[k], var"#x#opk" = x, …; <source> end`,
  with the recipe's authored closure in the slot for its captures
  (`getfield(__ops__[k], :c)`) and locals named from the slot, so one plan
  lowers to one expression. Literal statements (`shift::Float64 = -1.5`),
  aliases, fused sources with data-only split nodes and operations built
  without a throw-stripped twin inline too, as do a plate's cell recipes
  inside the plate loop and a scan's step statements. A branch, a
  reduction, a raw function, a bound constant, a plate, a scan, an embedded
  kernel, a source reading a module binding the module can reassign (its
  definition-time callee stays pinned), and a source outside the emitter's
  grammar keep their call through the slot. `RK_INLINE_SOURCES=0` keeps
  every call in the table for comparison. The plan, its recipes,
  `recipe_inventory` and `explain` are unchanged; `code_expr` prints the
  code that runs. On a downstream application model the log-density kernel's
  table type shrinks from 2.58 MB to 548 KB of type text and the bound-data
  prefix's from 312 KB to 94 KB; the log density and the K=64 synthetic
  program's density and gradient are bitwise equal to the table's.
- Lowering reads an operation's fields with one method instance for every
  operation type (`_source_parameter_names`, the source module, the
  type-stability audit's recipe type query through `Base._return_type`, the
  plate and scan predicates, partial evaluation's branch reads, the gather
  predicate), and the plate loop's recipe emitter has one closure type for
  every plate (its captures travel through a `Ref{Any}` cell). Measured side by
  side on a downstream application model, its `problem` stage (prepare, the
  audit, `prepare_ad`) compiles 6,691 method instances in 103.5 s instead
  of 7,677 in 111.3 s, and a K=64 synthetic program's prepare takes 34.7 s
  instead of 38.2 s (1,638 instead of 2,219 methods).
- The embedded-kernel splice keeps named-tuple field labels (`(g = …, a = a)`,
  `(; x)`) and keyword labels as syntax; it renamed them with the locals,
  which the generated source function used to hide.
- The `independent` reconvergent plate diamond's empty-sequence native
  Reverse gradient is pinned as an Enzyme boundary on Julia 1.10
  (`test_plate_fusion_reconvergence_ad.jl`, the
  `repro_enzyme_branch_allocation_phi.jl` class, decision `0ajp3qj`'s
  policy); it failed identically before this change.

## 0.1.22

- One cell of a plate (`plate_cell`; RKPPL's `prepare_cell_query` and cell
  samplers) computes at the rows it reads a value it reads at its own position
  through several batched arguments. The outputs of one `@plate for` loop are
  one plate each, over a domain of its own, and compose into the cell that
  reads them, so a value two statements of the loop read at the loop index,
  `log(c[i]) - log(xi[i])` and `(1 - c[i]) * scale`, is read through two
  domains. A plate gathered by a reader likewise reads a value through each
  of its batched arguments (`plate(d1, d2) do i, j; c[i] - c[j] end`).
  Earlier versions computed such a value whole in every cell, with everything
  upstream of it.
- A cell also computes at its rows a dotted value that reads an array at
  literal positions (`p .* e[1] .+ (1 .- p) .* e[2]`) when that array declares
  no type but holds numbers: an alias of a typed value (`e = lr`), a dotted
  value of typed arrays (`e = logistic.(lz) .* exp.(lr)` with
  `lr::AbstractVector{Float64}`, typed by its elements, which Base computes
  from its operands' elements), or an authored plate's value (typed from its
  cell). Earlier versions required the array's own declared type and computed
  such a value whole in every cell. Values are unchanged.

## 0.1.21

- One cell of a plate (`plate_cell`; RKPPL's `prepare_cell_query` and cell
  samplers) computes a column view `view(A, :, k)` at the rows it reads, as it
  computes the column read `A[:, k]`: alone and as an operand of a dotted
  expression, so a product `A = X * W` and its `hcat` design are computed at
  those rows too. RKPPL reads every column `P[:, j]` of a per-observation
  value through such a view, as in the columns of a BRM joint population
  effect for several responses. Earlier versions computed the view whole in
  every cell, and with it the product, the design and everything upstream.
  Values are unchanged.

## 0.1.20

- **Breaking:** the iterative-method compiler, the NUTS/HMC sampler
  examples and their tests, benchmarks and documentation pages are removed.
  They moved to a separate private package that depends on ReactiveKernels.
  Pure kernels, plates, scans, prepared AD, partial evaluation,
  `ReactiveState`, `prepare_reactive` and RKPPL are unchanged. What to
  change:
  - `compile_stateful`, `stateful_compiler_bindings`, `functionalize_stateful`,
    `stateful_snapshot`, `compile_state_transition`, `initial_transition_state`,
    `validated_compiled_transition`, `stateful_control_bounds`, the effect and
    observation ports (`pure_callable_port`, `effect_callable_port`,
    `effect_lowering_port`, `structured_state_port`,
    `total_functional_lowering`, `initial_transition_effects`,
    `transition_with_effects`, `drain_observations!`), `rng_provider`,
    `OrderedRNGReplay`, `prepare_transpiled`, `initial_transpiled_state`,
    `transpiled_endpoint`, `partial` and `PartialFunction` are no longer
    defined here.
  - The nested packages `ReactiveKernelsNUTSExamples`,
    `ReactiveKernelsStreamingStats` and `ReactiveKernelsHMCDiagnostics` are no
    longer part of this repository.
  - `ReactiveKernelsCompatibilityExamples` no longer has `ReactiveHMCExamples`
    or the ReactiveHMC artifacts: `CompatibilityArtifacts.all_artifacts()`
    returns the three ReactiveObjects graphs, and `reactivehmc_artifacts` is
    gone. `ReactiveKernelsKernelExamples` no longer has `HMCExample`.
  - The all-82 posteriordb benchmark measures primal and gradient only; its
    receipts keep their recorded HMC-throughput cells, which the validator
    checks where present and the docs no longer render. The motorcycle HSGP
    HMC driver and its receipts are removed.
  - `@kernel` no longer builds stateful free methods: a methodless body that
    updates a field of its first argument in place, or a `!!` name, raises an
    `ArgumentError`. A method-bearing object keeps its owner snapshot and pure
    endpoint templates only; `copy!!`, `KernelObject`, `KernelView`,
    `kernel_methods`, `kernel_recipe_ast`, `kernel_module`, `kernel_token`, the
    callee and type snapshots and the primitive effect registry are gone.
    Pure objects and their endpoints build and behave as before.
- `ReactiveKernels._compile(ast, context, native_context)` compiles a lowered
  body with its global names resolved in `context`; `compile(ast)` is that
  core with ReactiveKernels as the context, unchanged.
- A front end can build on method-bearing objects: `_KernelObjectSkeleton` is
  the supertype of the object `@kernel` binds, the pure object machinery reads
  an object only through `_kernel_pure_skeleton`, and
  `_kernel_object_definition` returns the parts and the pure-skeleton
  constructor of an object definition.

## 0.1.19

- One cell of a plate (`plate_cell`; RKPPL's `prepare_cell_query` and cell
  samplers) computes a gather straight from an inline `vcat`,
  `completed = vcat(observed, drawn)[order]`, at the rows it reads, as it
  computes the two-statement spelling `stacked = vcat(observed, drawn)` and
  `completed = stacked[order]`. Earlier versions computed that recipe whole in
  every cell, `vcat` included, so every cell read every drawn entry. The cell
  lowering now outlines a call written where a row form reads a named value
  (a gather's source or index, an operand of `vcat`, `hcat` or `*`, or a
  non-dotted operand of a dotted expression, as in
  `observed .+ mask .* vcat(free, grouped)[lookup]`) into a value of its own,
  along the recipe's strict subexpressions, and keeps it when the recipe is
  then computed at rows. Values are unchanged. `prepare_nonallocating` keeps
  such a recipe as written.
- A cell also computes at its rows a read-only gather
  (`ReactiveKernels._readonly_gather(A, I)`, the view RKPPL reads a bare
  gathered `@rkppl` definition through, such as
  `w = vcat(w_observed, y_mis)[w_order]` or `w = s[idx]`), as it computes
  `A[I]`. Earlier versions computed such a definition whole in every cell.

## 0.1.18

- A cell (`plate_cell`; RKPPL's `prepare_cell_query` and cell samplers)
  evaluates when a number it computes is read by two or more of its values
  computed at rows, such as `s = exp.(b)` read by `l1 = s .* x` and
  `l2 = s .+ w`. Since 0.1.16 a value with several readers is computed once,
  at all the rows they read, and each reader takes its own rows from that
  result. A number has no rows to take, so the cell raised
  `MethodError: no method matching _cell_take_rows(::Float64, ::Vector{Int64}, ::Int64)`
  (a rank-0 array failed the same way). Each reader now reads the number
  whole. BRM-emitted NLME programs raised it on a downstream application's
  whole model. Values of other cells are unchanged.

## 0.1.17

- A prepared kernel's call enters its generated body directly, with the
  operation table and one argument per HAVE port written out. It went through
  the `RuntimeGeneratedFunction` call method, which splats its arguments; past
  Julia's 32-element splat limit that splat is a call dispatched at run time.
  A program with 32 or more HAVE ports whose prepared callable is that
  generated function (one with no plate, no scan over its inputs and no
  embedded kernel holding either) therefore dispatched every call at run
  time, and `type_stability = :error` refused it, naming
  `Core._apply_iterate(Base.iterate, RuntimeGeneratedFunctions.generated_callfunc, …)`
  in "The prepared program's entry call". BRM-emitted simulator programs over
  37 and 39 data columns were refused that way. Values are unchanged.

## 0.1.16

- One cell of a plate (`plate_cell`; RKPPL's `prepare_cell_query` and compact
  cell samplers) computes more of the values it reads at its own rows instead
  of whole:
  - `vcat(a, b, ...)` of inputs and number literals, such as a completed
    covariate `observed .+ mask .* vcat(free, grouped)[lookup]` or a padded
    field `vcat(phi, 0.0)`: each operand is computed at the rows of it the
    cell reads, none for an operand it does not read. Matrices stack along
    their first axis, with their other dimensions checked as `vcat` checks
    them.
  - a typed alias `b::T = a` whose declaration converts no elements
    (`AbstractVector`, no declaration). A declaration that converts
    elements (`b::Vector{Float64} = a`) keeps `b` whole.
  - one source gathered by several index vectors in one expression
    (`exp.(t[r1]) .+ exp.(t[r2])`).
  - a shared value that a plate computed at rows reads through an index
    array (`sum(exp.(log_w[rows[k]]))`), as cells already read it.
- Each value computed at rows is computed once per cell, at all the rows its
  readers read. Earlier versions computed it, with everything upstream of it,
  once per set of rows read (a predictor gathered by two index vectors in two
  definitions ran twice).
- Measured on a downstream application's BRM program (223,941 coordinates, BRM
  0.2.63; a Linux fleet host, Julia 1.10.12, one thread, at the zero point): one cell's
  value reads 26,145 packed coordinates instead of 252,447, another 49,215
  instead of 724,502 and two more about 50,000 instead of 713,760; a warm
  value takes 0.36 ms instead of 4.1 ms and 0.50 ms instead of 17.1 ms, a
  warm compact cell gradient 9.9 ms instead of 30.2 ms and 23.0 ms instead
  of 74.1 ms. Preparing a cell costs more: its first compact gradient
  compiles in 133–140 s instead of 54–72 s, its first value in about 21 s
  instead of 8 s. A value an ordinary function computes (BRM's grouped-shares
  values and CAR fields there) is still computed whole in every cell, also as
  a `vcat` operand the cell reads no row of, since the `vcat` needs its length.

## 0.1.15

- **Breaking:** ReactiveKernels has no Reactant path. The Reactant extension
  (`ReactiveKernelsReactantExt`), the tensorized lowering, the
  `ReactiveKernelsReactantODESolvers` package and the `ReactantCore`
  dependency are removed. Kernels, plates, scans, prepared AD and RKPPL run
  natively as before. A program compiled through Reactant stays on 0.1.14.
  What to change:
  - `compile_ad_gradient` and `compile_ad_value_and_gradient` are removed;
    use `ad_gradient` and `ad_value_and_gradient[!]`.
  - `ReactiveKernels.traced` is removed; delete hand-written `traced`
    methods. `@traceable` still defines the native method and records its
    source.
  - `lower_with_ops` no longer takes `tensorized`.
  - `prepare_transpiled` no longer takes `backend` or `backend_options`, and
    `rng_provider(Val(:reactant))` is removed.
  - `jax_source` no longer recognizes `ReactantCore.@trace`; write plain
    `if` and `while`, which it translates the same way.
- A recipe that is exactly a call of a `@traceable` helper (`a = helper(x)`)
  keeps the helper bound when the kernel is defined, as a call of any other
  function does. On 0.1.14 and earlier such a call read the binding when the kernel was
  prepared, so rebinding a captured local helper between definition and
  preparation changed the result.

## 0.1.14

- The type-stability check's code audit (`type_stability_level = :rk` and the
  default `:all`) no longer reports the code that Julia's logging macros
  (`@debug`, `@info`, `@warn`, `@error`, `@logmsg`) write into a function's
  body, the record's message and fields included. That code is Julia's own,
  dispatches at run time by design, and runs only when its log statement is
  reached. A dependency that logs on a cold branch, such as SpecialFunctions'
  `beta_inc` with `@debug` on underflow (reached through a negative-binomial
  cdf), therefore no longer makes `type_stability = :error` refuse the
  program at `:all`. Other run-time dispatch in such a function is still
  reported.

## 0.1.13

- With an Enzyme that has `Enzyme.set_atomic_add` (EnzymeCore 0.8.27, the
  local Enzyme `dogfood` line), every `prepare_ad_*` preparation, and so every
  RKPPL sampler, query and cell sampler gradient, prepares a reverse-mode
  `AutoEnzyme` that leaves that setting to Enzyme (`AutoEnzyme()`,
  `AutoEnzyme(; mode = Enzyme.Reverse)`, the reverse part of a `SecondOrder`)
  with `set_atomic_add(mode, false)`. Its derivative then compiles plain
  shadow updates in a process with several default-pool threads, where Enzyme
  otherwise compiles each one as an atomic update. A prepared operator serves
  one caller at a time, so no other thread updates its shadows. A mode that
  sets the switch either way is kept, and the operator's `backend` field holds
  the backend it was prepared with. Measured on Julia 1.10.12 with 8 default
  threads: an RK `prepare_ad` Gaussian likelihood gradient (434 × 4) took
  1381 ns instead of 4394 ns, and an RKPPL `prepare_sampler` regression
  gradient 659 ns instead of 2114 ns. With registered Enzyme, which lacks the
  switch, nothing changes.

## 0.1.12

- With a reverse-over-forward backend,
  `SecondOrder(AutoEnzyme(; mode = Enzyme.Reverse),
  AutoEnzyme(; mode = Enzyme.Forward))`, `prepare_ad_hvp` and
  `prepare_ad_third` hold memory linear in the number of active coordinates
  `n`. RK prepares the reverse gradient of the directional derivative
  `x -> Df(x)[v]` itself. On 0.1.11 and earlier they held
  DifferentiationInterface's own preparation, which also prepares a
  forward-mode gradient whose Enzyme preparation holds `n` one-hot vectors of
  length `n`: 8n² bytes per Hessian-vector preparation and 16n² per
  third-order one, 187 GiB at 158,343 coordinates.
- `ad_gradient_and_hvp` and `ad_gradient_and_hvp!` take the gradient from the
  outer reverse backend in one sweep. On 0.1.11 and earlier,
  DifferentiationInterface's `gradient_and_hvp` took it from the inner
  forward backend, ⌈n/16⌉ forward passes: 10.9 ms against 31 µs at 4,000
  coordinates.
- A tuple `active` selector works for reverse-over-forward Hessian-vector
  products: each direction and each result is a tuple of components. On
  0.1.11 and earlier it failed in DifferentiationInterface's `pick_batchsize`.

## 0.1.11

- On Julia 1.12 and later, a kernel body may read a name that two `using`s
  export as distinct constant bindings of one stored value (Julia defines
  the name, and plain code in the module reads it). On 0.1.10 and earlier
  `compile`, and with it `prepare`, threw `Constant binding was imported from
  multiple modules` from `Base.binding_module`. Julia 1.10 and 1.11 were
  unaffected.

## 0.1.10

- `prepare_ad_pullback`, `prepare_ad_pushforward`, the one-shot
  `ad_pullback` and `prepare_ad(...; retain)` accept WANT ports that no
  active HAVE port reaches: a port computed from bound data alone (which
  `bound` partial evaluation folds into a constant, such as a gather
  `loc[rows]` or `view(loc, rows)`), from inactive HAVE values alone, or an
  inactive HAVE port itself. Their derivative is zero: a pullback adds
  nothing for them, a pushforward returns a zero tangent, and their values
  come from the same sweep. When every selected port is such a port, the
  pullback is zero and the pushforward tangents are zero without a backend
  call, and a pullback's seed is still checked against the port values'
  shapes at preparation. On 0.1.9 and earlier such a port raised
  `EnzymeRuntimeActivityError` under plain Enzyme Reverse or Forward, alone
  or beside ports that read the point, and a bound view beside them failed
  with `MethodError: Cannot convert an object of type Vector{Float64} to an
  object of type SubArray`.

## 0.1.9

- A generated body that reads a name its module imported under another name
  (`import LogExpFunctions: cexpexp as _ppl_cexpexp`) reads that binding. On
  0.1.8 and earlier it read the owner's binding of the alias name instead:
  `LogExpFunctions._ppl_cexpexp`, which is undefined, so the first evaluation
  raised `UndefVarError`, or, where the owner defines a function under the
  alias name too, that other function.

## 0.1.8

- Requires OutputSignatures 0.1.6, whose rules describe the comparisons `<`,
  `<=`, `>`, `>=`, `==`, `!=` and `ifelse` over machine-real scalars and
  their broadcasts. Value metadata of a call or broadcast of these is
  therefore known: `ifelse.(a .> 0, a, b)` over scalars is a scalar. On 0.1.7
  and earlier such a call had no metadata.

## 0.1.7

- A package can state that its own scalar function computes a function
  OutputSignatures already describes, with a method of
  `ReactiveKernels._metadata_function(f)` (as it adds a JAX translation with
  `_jax_elementwise`): the call and broadcast metadata of `f` are then that
  function's. ReactiveKernelsPPL 0.1.3 states it for `_ppl_logistic`, its
  inverse logit, whose calls had no metadata on 0.1.6 and earlier.

## 0.1.6

- A plate cell that reads a row-wise value at several rows computes it at
  those rows only: `v[r]` with `r` one integer vector per cell, or
  `v[rows[k]]` through a shared index array (a total over a group of rows,
  RKPPL's `groups(g => cells)`). On 0.1.5 and earlier such a cell computed
  `v` and everything it reads for the whole domain.
- Values a cell reads at its rows are computed at those rows for more
  producers: column reads `A[:, k]`, `hcat` of row-aligned operands (a design
  matrix with a computed column), a product whose left factor is one of
  these, and dotted expressions with column reads, products `A * w` and
  literal-position reads `a[1]` among their operands. BRM's stratified group
  effects, scaled group blocks and completed-covariate designs are therefore
  computed at the queried row; on 0.1.5 and earlier each was computed whole in
  every cell query, compact cell gradients included.
- Native Enzyme Reverse through an `hcat` read at several rows of a cell
  fails in Enzyme (snag `native-reverse-t-6d51cae6`); its values are exact.

## 0.1.5

- A rebuild of one program reuses the code compiled for the first build: in
  the same process, when a program generator evaluates each build in a module
  of its own (BRM's lowering modules), and from a package image whose workload
  built the same program. On 0.1.4 and earlier a position reader (`vectorize`,
  `prepare_batched`, a `NativeScheduling` reader) of a rebuilt graph had new
  types, and its first read compiled the whole reader again; so did every
  executable of a program lowered in a fresh module, or holding a closure Julia
  lowers itself (a local function, a generator, an exception scope, a boxed
  cell, the `@trace` arms of a tensorized branch). A native function defined
  anew in each lowering module is still a different function, and code that
  calls it compiles again.
- Preparation's type-stability check, the position-reader constructors and a
  borrowed reader's `copy` compile once per process instead of once per
  prepared program.

## 0.1.4

- A per-row value that native lowering keeps as its arithmetic (a dotted
  broadcast or an `hcat` read only by indexing, such as
  `sigma = exp.(s0 .+ s1 .* x)`), indexed by an integer vector, a range, a
  colon or a mask, computes only the entries it selects. A group cell
  gathering its rows, `sigma[rows[g]]`, therefore computes those rows; it
  computed the whole value at every read, once per group, so a density cost
  a full pass over the rows per group (300 groups of 10 rows: 2.0 ms and
  7.3 MB per call, now 40 µs and 110 KB, beside 37 µs and 91 KB for the
  same rows gathered from a stored vector). Values are unchanged. Where the
  value combined a sampled scalar with a data column, ordinary Enzyme
  Reverse (registered 0.13.216) also refused the gradient with
  `EnzymeRuntimeActivityError`; it now differentiates it.

## 0.1.3

- Preparation checks the code of the prepared program, not only the types of
  its values. `type_stability_level` sets how far: `:values` infers the type of
  every value, as 0.1.2 did; `:rk` also audits the body ReactiveKernels
  generates and the functions of its packages, with every other function a
  leaf whose call must dispatch statically and return a concrete type; `:all`,
  the new default, also audits every function that code calls, a client
  helper's body included, except Julia's own Base, Core and standard libraries.
  A finding names its method and each non-concrete value or call dispatched at
  run time. A helper that dispatches at run time but asserts its result
  (`f(x) = (g(x))::Float64`) now warns, and so does a declared local
  (`mu::Vector{Float64} = f(x)`) over a helper whose result is not inferred.
  `prepare_ad*` audit the callable their backend differentiates.
- Preparation can refuse a type instability instead of warning once per
  process. With `type_stability = :error`, `prepare`, `prepare!` and every
  `prepare_ad*` throw a `TypeInstabilityError` whose `messages` name every
  finding, at every preparation. The default stays `:warn`.
- Both settings are keywords and ReactiveKernels preferences of the same name.
  A project sets them for every preparation it makes, including those that
  packages such as ReactiveKernelsPPL make for it, for example with
  `Preferences.set_preferences!(ReactiveKernels, "type_stability" => "error")`
  (`[ReactiveKernels] type_stability = "error"` in its `LocalPreferences.toml`,
  which `Pkg.test` carries into the test environment). The preferences are read
  at the first preparation of a Julia session; a new session picks up a change
  without recompiling anything. A call's keywords take precedence for that call
  and the preparations nested in it. Cached preparations are kept per setting,
  so a kernel prepared under `:warn` or a lower level is never returned to a
  stricter call. `Preferences` is a new dependency; `PrecompileTools` already
  loaded it.
- The typed-IR audit that held RK's own lowering to type-stable programs in the
  test suite (`test/type_stability_audit.jl`) moved to
  `src/type_stability_audit.jl`; the RK and RKPPL gates use it from there.

## 0.1.2

- A summed plate whose cells hold arrays adds them elementwise, as Julia's
  `sum` does: with `cells = plate(groups) do xs; b .* xs; end`,
  `total = sum(cells)` is `b .* sum(groups)`, and a cell `[b * u, b * v]` or
  a matrix cell `xs * xs'` sums the same way, also when the plate consumes a
  scan or runs by index runs. The fused total starts from the first cell; it
  was seeded with `zero(T)` for the cell type, which raised
  `MethodError: no method matching zero(::Type{Vector{Float64}})` before any
  cell ran. Cells of different lengths raise Julia's `DimensionMismatch`, and
  an empty plate fails as `sum(Vector{Float64}[])` does. Numeric cells keep
  their `zero(T)` start. Native values and ordinary Enzyme Reverse gradients.
