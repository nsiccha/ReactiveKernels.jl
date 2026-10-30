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
- Measured on ArbeitsplatzAtlas's BRM program (223,941 coordinates, BRM
  0.2.63; gordito, Julia 1.10.12, one thread, at the zero point): one cell's
  value reads 26,145 packed coordinates instead of 252,447 (`osm_pub`),
  49,215 instead of 724,502 (`ba_n`) and about 50,000 instead of 713,760
  (`gr_n`, `hh_n`); a warm value takes 0.36 ms instead of 4.1 ms (`osm_pub`)
  and 0.50 ms instead of 17.1 ms (`ba_n`), a warm compact cell gradient 9.9 ms
  instead of 30.2 ms and 23.0 ms instead of 74.1 ms. Preparing a cell costs
  more: its first compact gradient compiles in 133–140 s instead of 54–72 s
  (`ba_n`, `gr_n`), its first value in about 21 s instead of 8 s. A value an
  ordinary function computes (BRM's grouped-shares values and CAR fields
  there) is still computed whole in every cell, also as a `vcat` operand the
  cell reads no row of, since the `vcat` needs its length.

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
