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

## 0.1.389

- Partitioned JAX reads can prepare distinct row coordinates for a unit
  even when the source table is larger than the request. The indexed
  reader reuses point-independent rows in their original observation order.

## 0.1.388

- Regenerated JAX modules skip a guarded matrix exponential and its reverse
  pass when every lane in a `vmap` batch skips the call. Mixed batches keep
  the exponential available for each lane that selects it.

## 0.1.387

Graph generators can reuse compiled assembly instructions while supplying the
operands of the current expansion, including a child graph used in a plate.

## 0.1.386

- Ordinary runtime checks in `jax_source` evaluate the active lanes on the
  device and call Python on failure. Valid batched gradients can run through
  checked loop steps without a Python callback at every iteration.

## 0.1.385

- `jax_source` uses the device integer type for the padding and initial
  values of whole-array `minimum` and `maximum`. Float32 modules can
  reduce prepared integer rows with JAX's 64-bit mode on or off.

## 0.1.384

- Compare `kernel_revision(child)` before reusing a parent that spliced a supplied child. The revision tracks graph edits and public port bindings.

## 0.1.383

- `jax_source` translates `argmax(x)` on numeric vectors to a one-based
  integer index. Ties select the first maximum, including the first NaN.
  Positive zero sorts above negative zero. An empty vector raises when the
  reduction is evaluated.

## 0.1.382

- Exported JAX RK45 solves control local error in the state and its
  sensitivities to the initial state and floating-point RHS inputs.
  Gradients use those forward sensitivities to avoid cancellation near an
  equilibrium. Value-only calls also track them, which can increase their
  cost and step count.
  Regenerate an exported module to pick up the change.

## 0.1.381

- `jax_source` supports `reduce(vcat, cells)` when a cell broadcasts
  independently sized prepared rows. `prepare_data` counts the selected
  entries after singleton expansion, including repeated and empty rows.

## 0.1.380

- A scheduled prefix plate can capture more than 32 operands under
  `type_stability = :error`. Inline and threaded chunks pass those operands
  to the generated cell loop in one tuple. The audit follows the same call.

## 0.1.379

- Native Enzyme preparation checks each opt-in summed-plate rule at the
  exemplar. A skipped plate, an unchecked schedule or a rule Enzyme declines
  uses the ordinary inline derivative. Other plates can keep their checked
  schedules. If the exemplar throws during verification, all plates use
  ordinary differentiation. The visit recorder runs only during preparation.

## 0.1.378

- The generated reverse of a summed plate shares an accumulator when the
  value and a partial derivative sum the same term.
- A prepared rule keeps the exponent range of each bound operand in its
  tape slot. Live operands are checked at evaluation.
- When the range check permits it, a shared reciprocal moves outside the
  row loop, so a crossed Gaussian sweep accumulates its unscaled residual.
- A cell's `ifelse` selects the cotangent of the taken arm directly.
- The generated rule call is inlined into its callable. This restores the
  split reverse through a constant tape slot with registered Enzyme.

## 0.1.377

- A reader that `vectorize` or `prepare_batched` builds with
  `schedule = NativeScheduling(...)` prepares under `type_stability = :error`
  when its program has concretely typed inputs and a data-only prefix plate.
  The code audit (`type_stability_level = :rk` or `:all`) no longer reports
  the launcher that runs a prefix plate's cells across worker tasks. That
  launcher reads the call's worker budget from task-local storage and
  dispatches the cells at run time, once per call of the plate. The audit
  checks the cells it dispatches instead, with the operand types the
  prepared program passes, so at `:all` a client function those cells call
  is still reported. Under the default `type_stability = :warn` the same
  reader no longer logs that warning.
- The table through which a scheduled prefix's cells share a call's result
  is type stable, apart from one read of its untyped slot under its lock,
  which the audit lists as justified. A reduced prefix plate whose cells
  share a call no longer returns that call's table from its plate program,
  whose result type was not concrete.

## 0.1.376

- Indexed JAX events can reuse a data-dependent row lookup across draws.
  An observation grid that reads a single coefficient row for a station
  evaluates that row once and broadcasts it across the observations.
  Dynamic repeated indices use a direct lookup table.

## 0.1.375

- The opt-in generated reverse of summed plates (`with_plate_reverse`, the
  `plate_reverse` preference) takes its one pass on more programs, and where
  it engaged at a loss it no longer does. Measured through the public
  non-allocating sampler on Julia 1.13.1 with Enzyme dogfood, 2,000 rows:
  - A likelihood that a data-bound branch partitions (a count likelihood's
    zero and positive rows) reads its group effects through a composed
    gather. Each part now takes the grouped one pass with one buffer entry
    per group. A grouped Poisson gradient took 0.59 of ordinary Enzyme's time
    and 1.08 of a hand-written gradient's; before, the rule did not engage.
  - Several gathered operands, such as two crossed grouping factors, keep
    ordinary arithmetic under one range check. Before, every cell took the
    separate-exponent quotient: ten times ordinary Enzyme on a crossed
    Gaussian model, now 0.68.
  - Quotient partials use plain arithmetic while their products stay in the
    normal range, with the same bits, and the separate-exponent form outside
    it. A grouped negative-binomial gradient went from 1.09 to 0.80 of
    ordinary Enzyme.
  - A cell that selects with `ifelse` on a predicate over active values is
    differentiated. Before, the plate was refused.
  - The range check behind the shared reciprocal measures the binary
    exponents of the values a sweep reads and charges them against the
    `Float64` range. Before, any value outside 2⁻⁴⁰ to 2⁴⁰ selected the
    separate-exponent sweep. A grouped Gaussian whose response and scale are
    multiplied by 10¹³ or 10⁻¹³ now takes the time it takes at unit scale
    (1.08 and 0.97 µs against 1.00) instead of 38 µs.
  - The program chooses the schedule from operand types and sizes before the
    backend sees the call, and the one-pass reverse no longer depends on
    operands staying unchanged after the primal. Every prepared operator
    (`prepare_ad` and its relatives, over allocating and non-allocating
    kernels) evaluates once at its exemplar. If the backend refused a rule
    call there, the operator is prepared again with the plates inline, as
    without the generated reverse, so no later evaluation pays for a refused
    rule. A plate whose call cannot be checked this way keeps no rule: the
    one-shot forms (`ad_gradient(kernel, backend, ...)`) and a call at the
    operand limit differentiate the plate inline.

## 0.1.374

- Indexed JAX density events can read a value produced by
  `jax.lax.platform_dependent`. When the device selects one implementation
  for the whole batch, `vmap` can batch both the selected rows and their
  gradients. If a branch depends on the individual point, each lane still
  executes only its selected arm, including when a loop carry supplies the
  branch condition.

## 0.1.373

- In a native summed plate, a straight-line cell can use a vector logarithm
  for a sum of Float64 exponentials, including the shifted normalizer of a
  positional categorical likelihood. Preparation applies stable rewrites
  first, so a two-term log-add-exp retains its overflow-safe evaluation.
  The vector logarithm can differ from Base by a last-place rounding step.
  Ordinary reverse mode uses the Base logarithm and its reciprocal partial.

## 0.1.372

- A softplus in a loop the compiler can turn into vector code is computed
  with vector arithmetic. That covers a summed plate whose cell calls
  `rk_log1pexp` or `LogExpFunctions.log1pexp` on a `Float64` beside
  straight-line arithmetic only (`+`, `-`, `*`, `/`, `ifelse`, `max`, `min`,
  `abs`, comparisons, a scalar `exp`), and a broadcast of either that a
  prepared non-allocating program materializes by itself. On a CPU with
  AVX-512 the density of a Bernoulli-logit regression with 3,000 rows,
  `y .~ BernoulliLogit.(logit_p)`, takes about 6 µs where it took about
  38 µs under Julia 1.13 (10 µs from 42 µs under Julia 1.10), and binomial
  and ordered-logistic densities take between a quarter and two fifths of
  their time. Their gradients take about 30% less; the Bernoulli gradient
  takes the same. The swap applies
  where the running process compiles that loop into vectors of at least four
  doubles (AVX2 or AVX-512, where the softplus itself is 1.4 to 6 times
  faster); elsewhere a program runs the code it ran before. A softplus in such a loop can differ from `rk_log1pexp` in its last
  digit, so a density and the value returned with a gradient can differ by
  one unit in the last place per cell. Generated code shows the call as
  `ReactiveKernels._vector_rk_log1pexp(x)` or
  `ReactiveKernels._vector_log1pexp(x)`.
- Everything else runs as before: a plate that keeps its cells; a cell that
  runs a scan, reads an array by index, branches, or calls any other function
  (Base's `log`, say); a softplus fused into a larger broadcast;
  `sum(log1pexp, x)`; a hand-written `log1p(exp(x))`. One call that stays
  scalar keeps the whole loop scalar, and there the vector softplus would
  take twice as long as the one it replaces.
- New dependency: SLEEFPirates 0.6.46, for its `log1p`. It brings 13
  packages and adds about 0.75 s to loading ReactiveKernels. The standard
  library InteractiveUtils is now loaded too, to read the compiled loop's
  vector width.

## 0.1.371

- A summed plate whose cells call `exp` on a number runs as vector code.
  Julia compiles `exp` of a machine float as a call, and that call kept the
  plate's summed loop scalar. A native program now inlines Base's method
  where it calls `exp` on a `Float64` or `Float32`. The density of a Poisson
  regression with 3,000 rows, `y .~ Poisson.(exp.(log_lambda))`, takes about
  4 µs where it took about 13 µs. A total can differ in its last digits, because the
  vector loop adds the cells in another order. Ordinary Enzyme Reverse still
  applies its own rule for `exp`, so gradients are unchanged. Generated code
  shows a statement that is one scalar `exp` as
  `ReactiveKernels._native_scalar_exp(x)` where it showed `Base.exp(x)`.

## 0.1.370

- A program that a package image compiled is reused at run time when the
  package image is part of a system image (PackageCompiler's
  `create_sysimage`), as it already was when the package image is loaded on
  its own. Before, ReactiveKernels registered a package's programs only from
  a package callback, which Julia never runs for a package a system image
  holds, so every such program was compiled again on first use.
  `__init__` now also registers the programs of the packages already loaded
  when it runs.

## 0.1.369

- Indexed JAX density events evaluate events that contain runtime checks.
  Checks that read only data run once, through `_rk_check_event_data`. Checks
  that read the parameters run over the whole event at each selected read,
  including checks under a condition on the parameters.
- Selected rows can read Cholesky factors of a batch of matrices, cumulative
  sums and products at gathered positions, and a concatenation at a traced
  position. They read the entire output of a loop or a scatter.
- A selected read computes each distinct entry it requests once, such as a
  factor shared by many rows, and still never computes an entry it does not
  request. The derivative of a read under a condition keeps only the selected
  rows' residuals, without the data or dense zero cotangents for them. At 149
  stations of the clustered ODL model, one station's value and gradient take
  2.4 ms against 102 ms for the full density. Batches grow linearly.

## 0.1.368

- Preparing a program that calls `rk_ode_rk45_tol` reports no type-stability
  finding in the solver's code at the default `type_stability_level = :all`,
  so `prepare` with `type_stability = :error` no longer refuses it. The
  adapter passes DP5's default step-size controller unresolved, and the
  solver resolves it to the same settings, so every step and result is
  unchanged. Left to its default, OrdinaryDiffEqCore had resolved the
  controller against an untyped placeholder list in `_ode_init`.
  `rk_ode_bdf_tol` is unchanged.

## 0.1.367

- A `float_type = Float32` JAX module solves `rk_ode_rk45_tol` and
  `rk_ode_bdf_tol` in float32 whether JAX's 64-bit mode is on or off, with
  diffrax and in the per-lane CUDA kernel. Before, with 64-bit mode on (as in
  a process that loads a float64 module, such as a ReactiveKernelMethods
  bundle), the diffrax solve's times, state and solution were float64, so a
  lazy branch on the solution failed with `cond branches must have equal
  output types`, and the per-lane kernel's step-size control was float64.

## 0.1.366

- JAX kernels can gather matrix rows with a ragged integer index or a leading
  Boolean mask. Reductions and matrix products use each group's selected rows,
  including repeated indices and empty selections.

## 0.1.365

- Indexed JAX density events can read selected rows of a conditional when
  one arm broadcasts a scalar and the other computes an array. Batched
  values and gradients retain the requested row order and multiplicity.
  Indexed bounded-parameter Jacobians also retain JAX's softplus derivatives.

## 0.1.364

- A JAX ODE module warns once when an active Float32 solve requests a positive
  relative tolerance below Float32 epsilon. The message reports the tolerance
  and how to select Float64 or relax the target. The solver uses the requested
  relative and absolute tolerances.

## 0.1.363

- Scheduled reads use the same compiled worker for small and large batches.
  A single job runs on the calling task, including with `workers=1`.
  **Breaking:** `code_expr(reader)` for a scheduled reader returns a named
  tuple. Use `.worker` to inspect the generated chunk expression and
  `.prefix` for the shared expression (`nothing` when absent).

## 0.1.362

- JAX export supports dense real symmetric eigenpairs and eigenvalues, including
  the RK-owned spectral primitives. `Symmetric(A, :L)` reads the lower triangle
  and `Symmetric(A)` reads the upper triangle.

## 0.1.361

- `jax_source` translates the reduced factors `F.R` and `Matrix(F.Q)` of
  `F = qr(A)` for dense real matrices. A kernel can construct
  `LowerTriangular(A)` or `UpperTriangular(A)`, solve with either factor,
  and compute the Euclidean norm of a real vector.

## 0.1.360

The JAX exporter can trace indexed event values through pointwise operations, reshapes, gathers and reductions. ReactiveKernelsPPL uses this translation to evaluate the rows selected by a runtime partition label.

## 0.1.359

- JAX index expressions such as `row[2:end]` read the logical length of a
  padded numerical row. The same expression works for rows of different lengths.

## 0.1.358

- A bound Normal regression keeps its least-squares statistics in more shapes.
  Binding factors the shared design once with a pivoted QR. When a gathered
  coefficient stands beside shared columns, binding first projects those
  columns and the response off the gathered direction in each group. Each call
  then evaluates the residual sum from the retained statistics instead of from
  the rows. This now applies when the location is affine in the tracked
  coefficients and the scale or precision varies by row through data (`sigma
  .* w`, `sigma ./ sqrt.(p)`), when the coefficients are tuple projections or
  enter through `muladd`, and when one gathered coefficient (`alpha[g]`)
  stands beside scalar slopes or a bound matrix design. For a shared-scale
  regression such as `Normal.(a .+ b .* x, sigma)`, binding compares the
  estimated cost of QR evaluation with the cost of visiting the rows after
  duplicate-row grouping. If the scale guard fails during evaluation, the
  fallback retains that grouping. Grouped intercepts and slopes with a
  centered bound column can use QR from five observations per group when
  the fit is usable.
  Crossed indices, multiple gathered coefficients alongside shared columns,
  and a coefficient that enters nonlinearly or twice keep the authored row
  calculation, and a rebinding whose statistics are not valid falls back to
  it. Priors, Jacobians and input checks are unchanged. `js_source` emits the
  authored rows for such a regression.
- A prepared `vcat` can refill its vector buffer from numbers and dense
  vectors of the buffer's element type, as in `vcat(0.0, alpha)`, when the
  length matches and the inputs do not alias the buffer.
- When an embedded prepared kernel has discarded its source plan, `js_source`
  raises `JSEmissionError` and names the endpoint it could not translate.

## 0.1.357

- A plate cell value that is a lazy branch whose condition the cell's
  concrete operand types decide, such as Stan's scalar and array `+` written
  `typeof(x) <: Number ? x .+ y : x + y` (in the cell, or in a
  function-shaped kernel spliced into it), is lowered as the arm it takes.
  The arm's broadcast then keeps the lowering it has when written alone,
  such as a lazy broadcast its reader consumes, instead of a fresh array in
  every cell. Native Reverse gradients of such a cell allocate what the arm
  alone does: over 100 cells of five entries each, 80 B in one allocation
  instead of 42,560 B in 406 (snag `type-decided-bra-2686ef6f`). A condition
  is decided only when Julia reduces it to a constant for those types, so no
  effect or error is dropped; a condition read from the cell's data stays a
  lazy branch. Values and derivatives are unchanged.
- A short-circuit condition its first operand decides (`typeof(x) <: Number ||
  typeof(y) <: Number`, `typeof(x) <: Integer && …` for a float `x`) now counts
  as decided by concrete types, here and in the selected-row locality of
  0.1.49. Before, the deleted statement Julia leaves before its return kept it
  dynamic.

## 0.1.356

- `prepare_nonallocating` reuses the array a `@traceable` helper creates with
  `zeros` or `ones` (`zeros(Float64, K, K)`, `ones(n)`) and then fills by
  indexed stores. Those stores, and the stores of the `copy(A)` updates of
  0.1.337, may now use scalar locals (`i, j, w = 1, 2, 1.0`, or a temporary
  inside the loop), `if` branches, `+=`-style updates and scalar arithmetic
  such as `sqrt`, `exp`, `log`, `^` and comparisons. A changed size, or an argument that aliases the previous
  result, gets a fresh array. Native Enzyme Reverse through such a helper
  gives the ordinary program's gradient and allocates nothing for it.

## 0.1.355

- On a CUDA device, a `jax_source` module whose ODE right-hand side branches
  (a ternary or `if` in a `@traceable` function, which becomes a
  `jax.lax.cond`) or selects with an integer index compiles again. 0.1.353's
  per-lane kernel picked a branch with an int32 `select_n`, which Pallas'
  Triton lowering refuses, so such a module failed with `Failed to parse
  Triton module: … 'arith.select' op operand #0 must be bool-like`. It now
  selects with bool predicates only, as XLA lowers `select_n`. Pallas'
  interpreter accepts either, so the tests read the kernels' jaxprs for it.

## 0.1.354

- A finding of `ad_derivative_report` (and of the derivative check at
  preparation) in the program ReactiveKernels generates names the statement
  it is in: its `frames` hold `macro expansion` at `rk:<value>:1`, `<value>`
  being the value the statement assigns, the name `ad_cost_profile` reports
  (`(other statements)` for code between statements). Before, Enzyme's report
  dropped those statement scopes, whose line was 0, and such a finding named
  only the generated body (`RuntimeGeneratedFunctions.jl:243` above
  `codegen.jl:8664`), and Enzyme merged the calls of one function in
  different statements into one finding. Each scope now has line 1 (a line
  node without a file after its `push_loc` and after its `pop_loc`); the
  compiled code and its derivative are unchanged. This also corrects native
  debug info, so the frames of profile samples and backtraces too: on Julia
  1.10, code of the generated body itself (a call that is not inlined, a
  buffer Enzyme keeps) after a block's first statement read as that
  statement's, and from Julia 1.12 code between two statements read as the
  earlier one's. Code of a plate's loop outside its cells, such as the
  buffers Enzyme keeps for the loop, is in `(other statements)`, as in
  `ad_cost_profile`.

## 0.1.353

- On a CUDA device, a `jax_source` module integrates each `rk_ode_rk45_tol`
  solve per lane in one Pallas (Triton) kernel, instead of diffrax's
  batched while loop. Every vmapped call (plate cells, a sampler's chains)
  joins one kernel call; each lane steps with its own step sizes, and only
  the 32 lanes of a program wait for each other. The kernel takes diffrax's
  `Dopri5` steps (first step, tableau, PID control, dense output, per-output
  budget, retry of a non-finite trial, failure statuses), so values match the
  CPU module up to rounding. Gradients are forward-mode sensitivities carried
  through the kernel and contracted by a custom VJP: the derivative of the
  same discrete solve. On a density with a five-state solve in each of 24
  plate cells, a batched value and gradient of 64 chains took 45 ms instead
  of 431 ms on a shared RTX 5060 Ti. `rk_ode_bdf_tol` keeps diffrax's
  `Kvaerno5` on CUDA, with forward-mode sensitivities; so does a right-hand
  side the kernel cannot evaluate, with a warning. CPU modules are unchanged.
  `RK_JAX_ODE_KERNEL` (or the module's `RK_ODE_KERNEL`) is `auto`; `off` keeps
  diffrax on CUDA, `interpret` runs the CUDA path on any backend in Pallas'
  interpreter. The ODE section of the JAX guides is now its own page,
  "JAX ODE solves" (`docs/src/jax-ode.md`).

## 0.1.352

- The opt-in generated reverse of summed plates (`with_plate_reverse`, the
  `plate_reverse` preference) now runs on top of the optimized program.
  Enabling it no longer disables summed-plate factoring, bound least-squares
  statistics or product batching, so a prepared density is the same program
  with it on or off. A plate summed from bound statistics stays with the
  backend, and `plate_reverse_report` says so.
- A plate call takes a generated rule only where a one-pass schedule applies
  to its operands: a gather with fewer groups than cells, a joint value and
  slope cut, or an additive nested total. Every other call runs inline, as
  it does with the generated reverse disabled. The two-pass rule is no
  longer used. `plate_reverse_report` still answers `"rule"` when a rule
  exists for the plate; the call decides whether to take it.
- Measured on six natural PosteriorDB ports (public sampler, Julia 1.13.1,
  Enzyme dogfood 0.13.221, one thread, a loaded shared host), value and
  gradient with the generated reverse on divided by off, default settings:
  EarnHeight 0.94, NES1984 1.02, WellsDAAEC 1.03, Mesquite 0.98,
  KidscoreMomiq 1.02, RadonVariableInterceptNCMN 0.81. Before this change the
  first, second and fifth were 17.9, 16.5 and 7.2, because enabling it
  replaced their bound statistics with a per-row rule.

## 0.1.351

- Under Enzyme reverse mode, a derivative rule no longer copies the arrays
  held inside an unchanged input, such as a struct, tuple or view passed as
  `Const` data, when the loaded Enzyme carries the dogfood repair
  `overwritten_reachable_data` (Enzyme.jl dogfood `8a381501` with libEnzyme
  `05ce259c`). That repair makes `EnzymeRules.overwritten` cover the data
  behind the references an argument holds. The adapter reads such an input in
  place when its flag is unset and the rule's result is a bits value or an
  array of numbers whose memory overlaps none of the input's arrays. That
  check covers the arrays of a mutable struct such as a `Dict`, and costs
  one comparison per array, not per element. Before, every array inside a
  container was copied on each reverse call. For example, one weight read
  from a struct of 65,536 numbers took 524,464 B per gradient, where it now
  takes 128 B. A rule's pullback over a BRM GMRF footprint grew from
  105,440 B at 256 states to 1,190,928 B at 4,096 states, and now takes
  34,208 B at both. Without the repair, or in split mode, those arrays are
  still copied. So are they when the caller writes, after the rule's call, to
  memory Enzyme cannot show is apart from the input. A prepared RK program
  that stores the rule's result into its output buffers does this, so a BRM
  GMRF coefficient row read through a prepared query still copies the
  footprint: 15.5 MB per row Jacobian at 4,096 states.

## 0.1.350

- A tuple selector's derivative accepts a fixed-size immutable array port or
  field, such as a StaticArrays `SVector`, beside mutable arrays. It crosses
  the derivative as `Ref` storage like a scalar, so `ad_accumulator_storage`
  gives it a `Ref` that `ad_accumulate!` adds into, and a gradient returns
  the array type unchanged.

## 0.1.349

- A code finding of the type-stability check in a function the prepared
  program calls now lists the calls that reach it, outermost first, under the
  heading "The prepared program reaches it through these calls". Each call shows
  the signature the function is compiled and checked for, the program's value
  whose statement makes the call (``computing `mu` ``), and the argument types the
  call passes where that signature is wider, such as pass-through varargs
  compiled as `Vararg{Any}`. Its frames follow, innermost first. Each signature
  a function is compiled for is now its own finding. Before, such a finding
  named only the function's own frames, so it did not say which statement of the
  program called the function, or with which argument types. The warning's
  once-per-process key includes the calls.

## 0.1.348

- The native loop for non-strided matrix products checks the shared dimension
  directly. This removes a generic multiplication from the compiled derivative
  that made strict `prepare_ad` and `prepare_ad_hvp` refuse a valid row-gather
  product.

## 0.1.347

- With MutatingFunctions loaded, `prepare_into` automatically reuses private
  storage for eligible dense numerical computations, including matrix products
  and nested broadcasts. The writer copies the result into the caller's array.

## 0.1.346

- A program generator can reuse lowered graph construction instructions across
  fresh definitions in one evaluation module. Each evaluation builds its own
  graph from the current definition's recipe sources and helper bindings.

## 0.1.345

- `prepare_into` prepares an ordinary Julia kernel to fill a caller-supplied
  array. One writer can alternate destinations across calls. A supported final
  broadcast writes directly, and other kernels run once and copy their array
  result.

## 0.1.344

- JAX exports retry an adaptive ODE trial that overflows from a finite state.
  A recovered solve returns its values and differentiates through its accepted
  steps. The caller's tolerances and per-output step budget still apply.

## 0.1.343

- Concurrent kernel preparations can read their project preferences safely on
  first use. A preparation's keywords still override the project settings for
  its nested preparations.

## 0.1.342

- The generated reverse of a summed plate preserves finite quotients and
  seeded derivatives when a shared divisor is subnormal or an intermediate
  product would overflow. It checks the input range before reusing a shared
  reciprocal and otherwise evaluates the original division.

## 0.1.341

- A kernel prepared with `prepare_nonallocating` can reuse the factor matrix for
  `cholesky(Symmetric(A))` on strided Float32 or Float64 matrices and for
  `rk_cholesky_lower(A)` on strided real matrices. A change in size or an
  input that aliases the previous factor gets fresh storage. Both operations
  preserve the caller's covariance and use their ordinary triangle conventions.

## 0.1.340

- `jax_source` can flatten plate cells that prepend scalars to a prepared
  ragged row with `vcat`. Each scalar contributes one entry per cell,
  including cells whose ragged row is empty.

## 0.1.339

- For real vectors, `jax_source` now emits `x * transpose(y)` as a matrix
  with one row per entry of `x` and one column per entry of `y`.
  In a ragged plate cell, the scalar inner product `transpose(x) * x`
  sums only the valid entries.

## 0.1.338

- The log rising factorial and its shape derivative keep intermediate products
  finite in Float32 for the supported small-count product calculation.
  JavaScript can emit the shape derivative's digamma fallback through `tanpi`.

- The generated reverse of a summed plate now includes the derivative of a
  shared value first computed inside a lazy branch, including when the
  prepared program introduces an alias for its scalar input before the loop.

## 0.1.337

- `prepare_nonallocating` reuses the result of a dense numerical `copy(A)` in
  a `@traceable` helper that fills the copy with indexed stores and returns it.
  For example, a row replacement written as a loop over `eachindex(v)` keeps
  the original matrix intact even when `v` is a view into it. A changed shape
  or an input that aliases the previous destination gets a fresh result.

## 0.1.336

- `prepare_nonallocating` reuses the vector behind a dense row-vector
  product. Transposed and adjoint vectors multiplied by a dense matrix
  of the same BLAS element type now follow ordinary Julia's rounding.

## 0.1.335

- `ad_cost_profile` accepts `min_samples` and `max_sample_seconds` to spend
  additional sampling time on a value or gradient pass with few observations
  of the calling thread. The default keeps one fixed window per pass.

## 0.1.334

- The opt-in generated reverse of a summed plate can vectorize dense
  derivatives when each cell writes to distinct storage. For a grouped
  derivative, it also checks whether successive group indices address
  consecutive entries in the parameter and gradient buffers. It vectorizes
  verified spans and processes the remaining observations in order.

## 0.1.333

- `jax_source` preserves the dtype of signed and unsigned 8-, 16-, 32- and
  64-bit integers in scalar result casts, typed constructors, and prepared
  arrays. With JAX's 64-bit mode off, 64-bit device dtypes use 32 bits.
  Adding two `Int32` counters now works as the body of a JAX loop with an
  `int32` carry.

## 0.1.332

- A scheduled reader used in a package precompile workload retains its serial
  driver and generated call for the sample's input types. A fresh process can
  reuse them for small batches alongside the worker code for larger batches.

## 0.1.331

- `prepare_nonallocating` reuses buffers for mixed scalar and range indexing, dimensional reductions of strided views, and broadcasts that gather array entries. Rank-changing reshapes can reuse their headers when the prepared kernel owns the source buffer.

## 0.1.330

- Strict cell queries on Julia 1.12 and 1.13 recognize captured
  comprehensions and row-building loops, including helpers that read a
  shared array and compute only the requested trajectory's rows.

## 0.1.329

- Streaming a plate into `logsumexp` skips the first floating-point
  correction from an empty total. Subsequent cells compute a stable
  correction with a nonpositive exponent. This speeds up short support
  reductions inside a larger observation plate.

## 0.1.328

- When every row of a bound plate selects the same branch arm, subsequent
  bindings with the same arm and cell rewrites reuse its prepared program.
  Each binding evaluates its own guard and refreshes its bound values. This
  speeds up preparing a regression sampler with a new vector of response scales.

## 0.1.327

- `stage_reverse` and `reverse_cut` check the number of arguments supplied
  at the call. Enzyme Reverse can now use a valid residual tuple even when
  Julia cannot infer its length.

## 0.1.326

- Reading a nonempty batch with `NativeScheduling` in a package-image
  workload warms the scheduled prefix and position workers even on a
  single-threaded builder. An explicit `workers=1` warms the serial path.

## 0.1.325

- `prepare_nonallocating` also buffers `hcat` (as it did `vcat`),
  `permutedims` of a matrix, and `cumsum` / `cumprod` along a constant
  `dims`: each fills a cache it owns from the second call on, where it
  allocated a fresh array on every call. `permutedims` of a vector keeps
  allocating, because Base returns a reshape that shares the vector's memory.
- `end` and `begin` inside an index, a tuple value such as `(a, b)`, and a
  destructuring assignment `a, b = f(x)` or `z, _ = f(x)` in a `@traceable`
  body no longer send that body to the whole-statement fallback, so the
  buffered steps around them stay buffered.

## 0.1.324

- `jax_source` again names private helpers for prepared kernels after the
  captured endpoint or requested result. For example, a truncated Cauchy
  prior calls `_standardized` and `_standard_logccdf`, and their docstrings
  identify the quantities they compute.

## 0.1.323

- A binding prepared on exemplars (`prepare(graph; have, want, bound,
  exemplars)`, and an authored kernel's `prepare(spec; bound, exemplars)`)
  takes the graph's bound-preparation cache, as a binding without exemplars
  already did. The cache keys an entry on the exemplars' call signature as
  well, so a later binding of the same bound ports and signature runs only its
  data prefix and shares the first binding's compiled residual, typed for the
  same exemplars. Before, every such binding planned, partially evaluated,
  lowered and compiled afresh. Its kernel's values are unchanged.
- `prepare_nonallocating` of a kernel a bound template produced shares the
  non-allocating program of the template's earlier bindings, per exemplar
  signature, passes and settings, and gives each kernel its own copy of the
  unfilled step caches. `prepare_ad` over non-allocating kernels that share
  one step program keeps the differentiated program per active input and
  seeded-cache state: each binding seeds its own caches, concretizes them and
  computes its own folded values. Values and gradients equal a fresh
  preparation's bitwise.
- ReactiveKernelsPPL's `prepare_sampler` prepares on all three, so a sampler
  or SMC target on another binding of one build (cumulative `update_smc`
  data, say) costs that binding's data work: on a three-coefficient Bayesian
  regression (one thread, strato2) a new binding took about 5 ms, against
  170–280 ms before (snag `rkppl-smc-with-d-80655b87`).

## 0.1.322

- A summed plate nested in another plate's cell, such as the positions of a
  whole-event law, runs once per distinct row of its bind-time caches per
  lane of the enclosing plates, weighted by its row count, as a top-level
  summed plate over repeated bound rows does since 0.1.184. A cached row index
  into an array that is the same in every row (a `fill`, an `hcat` of fills,
  numbers and bound columns with equal rows) does not tell positions apart,
  so a species' all-zero sites of `hcat(fill(theta, n), never)` occupancy
  operands are one cell. Values agree with the plate's sum to rounding; no
  exception moves. `simplify = false` keeps every position.
- `size(x, d)` and `length(x)` of a live array built in a plate cell by
  `fill` with bind-time dimensions, or by `hcat`/`[a b]` of such arrays,
  numbers and bind-time arrays (through a check returning it), are bind-time
  values, so the work they feed is cached at binding.

## 0.1.321

- `jax_source`: a plate over a padded vector of tuples of numbers, such as
  the stacked step outputs of a scan inside a plate cell whose length comes
  from the data, maps that vector as a padded group vector: its cells are
  masked by its length. Before, the plate passed that single length to every
  cell as if each tuple had a length of its own, and `jax.vmap` raised
  `ValueError: vmap was requested to map its argument along axis 0, which
  implies that its rank should be at least 1, but is only 0` (RKPPL's
  `group_random_walk` and marginalized `@plate` cell filters). Tuples with an
  array or tuple field keep their per-entry lengths as before.

## 0.1.320

- A compiled reactive state invalidates per element where it can. With an
  index, `touch!(state, x, index)` and `mutate!(f, state, x, index)` say that
  only those elements of the HAVE array `x` changed (an integer or
  `CartesianIndex` of `x`, or a collection of them; checked before `f` runs).
  An authored plate that reads `x` only at its own cell (an iterated argument,
  with the plate's axes) then reruns just those cells, at once, through the
  native kernel of one cell that `plate_cell` lowers, and a `Float64` `sum`
  over the plate is updated by the cells' difference instead of summed again.
  Every other dependent of `x` is invalidated as before. A single-site move of
  `theta[j]` read through `sum(plate(theta, y) do ... end)` therefore costs
  one cell, not one per element.
  - The plate's values are bitwise those of a whole recomputation. The sum's
    running total is compensated (Neumaier) and summed whole again once as
    many cells have changed as the plate has, so it equals a fresh `sum` to
    within rounding, not bitwise.
  - A plate value already returned by `get!` is never overwritten (the next
    change copies the buffer first). A frozen plate or sum stays frozen; a
    value installed by `assign!` or `freeze!` recomputes from its recipe on
    the next change; `copy` and `copyto!` carry the incremental state.
  - Without an index, `touch!` and `mutate!` are unchanged. A plate's
    one-cell kernel is prepared on the first change that names its elements,
    so preparing a state costs what it did; a kernel that cannot be prepared
    warns once and leaves its plate to recompute whole.
  - Measured on a single-site MH move of `theta[j]` that reads the total
    (the Gen-home-turf benchmark's hand-authored arm; gordito, Julia 1.10.12,
    one CPU, host load about 45), median per move at J = 10 / 100 / 1,000 /
    10,000: typed ports 0.50 / 1.45 / 9.8 / 87 µs without the index and
    0.48 / 0.48 / 0.54 / 0.73 µs with it; untyped ports, as in the benchmark,
    1.3 / 2.1 / 10.8 / 131 µs and 1.1 / 1.9 / 2.1 / 1.8 µs. The total after
    20,000 moves is within 2e-16 of a fresh sum. Preparing the state took
    3.3–3.5 s cold and 1.0 s for a second program, as on 0.1.316; the first
    move that names elements prepares the one-cell kernel (about 4 s cold,
    1 s warm).
## 0.1.319

- A derivative rule that retains arrays for its reverse pass (a
  `derivative_rule`, and RK's own rules such as the one behind RKPPL's
  `NormalIDGLM`) no longer makes the check of the derivative native Enzyme
  compiles refuse the gradient under `type_stability = :error` at
  `type_stability_level = :all`. Retention snapshots those arrays with
  concretely typed code instead of Base's `deepcopy`, whose
  `@nospecialize` argument compiled to a boxed `jl_invoke` that the check
  counts as run-time dispatch, even where a program never takes that path
  (snag `rkppl-ark-dogfoo-72035cd9`: an AR(K) regression through
  `NormalIDGLM`). What is retained, borrowed or copied is unchanged.
  Residuals that hold a mutable container other than a dense array of bits
  values (a mutable struct, an array of arrays) still take `deepcopy`, which
  the check reports where Enzyme compiles the rule into the derivative.

## 0.1.318

- `rk_affine_logistic`'s derivative rule computes both sigmoid slopes, σ(u)
  and σ(-u), from one `exp(-abs(u))`. It evaluated `exp(-log1pexp(-u))` and
  `exp(-log1pexp(u))`, two `log1pexp` and two `exp` in every gradient call.
  The primal is unchanged. Both slopes are within 1.1 ulp of a 256-bit oracle
  (up to 6 ulp before), and a tail slope such as σ(-40) stays representable.
  Together with ReactiveKernelsPPL 0.1.221 this halves the gradient of an
  interval-bounded `@scan` (snag `ukdrivers-log1pe-1df47969`; numbers in that
  package's changelog).

## 0.1.317

- A primal-only `rk_log1pexp(x)` (a value without a derivative, such as a
  Bernoulli-logit density's value) skips work in both tails: it returns `x`
  for `x >= 34` without an `exp`, and `exp(x)` for `x < -37` without a
  `log1p`. The values are the same Float64 bits as before. A derivative still
  runs the joint value/partial recipe, unchanged. Logits far from zero, as at
  a sampler's random initial points, made the value pay one `exp` and one
  `log1p` per row where `LogExpFunctions.log1pexp` pays at most one `exp`.
- `ad_cost_profile` counts the time of a program RK compiled for a recipe's
  expression, such as a scalar derivative rule's `exp(-abs(x))`, towards the
  statement that calls the rule. Before, it counted it as `(other statements)`,
  which hid a rule's cost from the statement that called it.

## 0.1.316

- A compiled reactive state (`prepare_reactive`, and the recipes
  `prepare_reactive_nonallocating` does not reuse in place) runs an authored
  plate, a `plate_cell` value and a scan with the loop `prepare` emits for that
  recipe. Before, its getters called those operations' fallback methods, which
  ran each cell through the scalar kernel's own call and typed the result at
  run time: about 130 ns and 7 allocations per plate cell, and 3.6 µs and 74
  allocations for one `plate_cell` read, where `prepare` takes nanoseconds and
  allocates nothing. Values, invalidation, `freeze!` and `assign!` are
  unchanged; with concretely typed ports a cell read after `set!` of its index
  allocates nothing. Preparing the state now also prepares each such recipe,
  which costs about what `prepare` costs to compile its loop.

## 0.1.315

- A summed plate whose cells the compiler sums by contiguous index runs (a
  gathered operand such as RKPPL's `beta[county_idx]` over grouped data) reads
  the same once-per-call reciprocal scale `exp(-u)` as its flat loop. Before,
  only the flat loop read it, and the run-grouped loop, the one that runs on
  such data, computed `exp(-u)` in every cell. A Normal likelihood whose
  location keeps a shared term beside a gathered slope, such as
  `alpha .+ beta[county_idx] .* x`, divides by its scale in every cell, so
  its density cost more than its own gradient. Values and derivatives are
  unchanged: both forms compute `y * exp(-u)`.

## 0.1.314

- With Enzyme's dogfood repair `derivative_report_reachability` (dogfood
  `db46ba80` or later), the derivative check of `type_stability_level = :all`
  and `ad_derivative_report`'s `unstable` also count run-time dispatch and
  boxes in code the derivative runs without differentiating it (Enzyme's
  `sweep = :primal`), since that repair marks error-only code such as error
  message builders `throw_only`. Calls inside Julia's own functions there stay
  exempt, as in the primal check, which treats their bodies as leaves (logging
  machinery included). Without the repair, that code stays left to the primal
  check, as in 0.1.312.

## 0.1.313

- `jax_source` and `js_source` translate a program in Julia's latest world,
  so a function that defines a program's helpers, `@traceable` functions or
  kernels and then translates it gets the same module as a translation from
  the top level, with no `Base.invokelatest` around the call. Before, a
  helper defined after the calling function began read as a function with no
  method: `jax_source` raised `JAXEmissionError` ("the function `f` has no
  applicable Julia method for argument types (Float64)"), or listed a
  data-only definition calling it in `UNTRANSLATED`.

## 0.1.312

- A native-Enzyme gradient runs the derivative Enzyme compiles, not the
  prepared program's own code, and the type-stability check now covers it. At
  `type_stability_level = :all` (the default), every `prepare_ad*` with
  `AutoEnzyme` (or a `SecondOrder` with an `AutoEnzyme` part, so also
  `prepare_sampler` and `rk_logdensity_problem`) evaluates its operator once at
  the exemplar, compiling it as `warm = true` does, and reads Enzyme's report
  of the derivatives that evaluation ran. Calls the derivative dispatches at
  run time, calls Enzyme differentiates at run time, and boxes, on paths that
  return, are `type_stability` findings: warned once per process, or refused
  with `TypeInstabilityError` under `type_stability = :error`. `:rk` and
  `:values` still check the primal only.
- New setting `ad_allocations = :ignore | :warn | :error`, a keyword of every
  `prepare_ad*` and a preference of the same name, scoped like
  `type_stability`: it warns about or refuses (`DerivativeAllocationError`)
  allocations in that derivative, on Julia's heap or native (Enzyme's tape,
  which `@allocated` does not count). The default, `:ignore`, only lists them.
  A setting other than `:ignore` checks the derivative at every level.
- New `ad_derivative_report(f)` and `ad_derivative_report(prepared, args...)`
  return an `ADDerivativeReport`: every derivative Enzyme ran for the code, its
  findings with source lines, and the unstable calls and allocations among them.
- Reading the derivative needs Enzyme's dogfood repair `derivative_report`
  (dogfood 83ff174a or later). With any other Enzyme nothing changes, except
  that a preparation checked with `type_stability = :error` or with
  `ad_allocations` warns once per process that the derivative is not checked.

## 0.1.311

- `ad_cost_profile` now reports the time a program spends managing memory, so
  a gradient whose excess over its value is allocation, freeing or garbage
  collection is named as such instead of being booked on whichever statement
  ran it. Each `ADCostPart` gains `primal_memory_seconds` and
  `gradient_memory_seconds` (time allocating, freeing, zero-filling or
  copying in Julia's runtime or the C library, including native Enzyme's
  `malloc`/`free` of the reverse pass's tape) and `callees` (the functions,
  neither Julia's nor RK's own, that the statement's time is spent in, such
  as a model's helper). Every sample inside a garbage collection now counts
  as `(garbage collection)`, not as the statement whose allocation started
  the collection, which varied between runs and hosts. The display adds a
  memory-management summary line and the columns `memory` and `mostly in`
  (the largest callee, when it holds at least half of the part's gradient
  time).

## 0.1.310

- JAX source scans a padded sequence with `history` (a dose list per ragged
  row of a plate): a step past the row's length runs no body and keeps the
  carry and history, so its reads of padding reach neither a value nor a
  derivative. The result carries the row's length. Before, it was refused.
- A later plate selects one cell of an earlier plate's vector cells by index
  (`units[k]`). Before, it was refused as "a ragged array without its
  lengths".
- A vector slice between ends that are data (`x[(p + 1):(p + q)]`) or that the
  trace knows, read from a padded vector, translates, with Julia's
  `BoundsError` for a nonempty range outside the vector. Padded rows of one
  length stored with different widths broadcast and scan together.
- `prepare_data` passes a vector of records whose vector fields differ in
  length per record with those fields padded and `<name>_lengths`, the tuple
  of their lengths (`None` for a number field). Before, NumPy refused it.
- Fixed: a record a source region returns is read by its field names in the
  caller, and only in the entries that use it (`KeyError`).
- Fixed: NaN reverse-mode gradients, with correct values, when a data-sized
  allocation filled by a loop reached a function with no finite derivative at
  zero (`sqrt`, `log`) through a masked reduction. Every padded array a loop
  carries now leaves it repeating its last valid entry.

## 0.1.309

- Added `ad_cost_profile(value, value_and_gradient; seconds, threshold,
  allocations)` and `ad_cost_profile(prepared::PreparedADKernel, args...)`:
  where a prepared program's value and native gradient spend their time,
  split by the program's statements. It times warm calls of both and, unless
  `threshold` is given and the gradient ÷ value ratio does not exceed it, runs
  each under Julia's sampling profiler and `Profile.Allocs`, attributing every
  sample of the calling thread to the statement it ran. The result, an
  `ADCostProfile` of `ADCostPart`s, lists each statement's value and gradient
  time per call, their ratio, its samples and its Julia heap bytes; the parts
  add up to the whole. Statements are named after the assigned value (`mu`,
  `plate_<name>` for a plate cell's statement); `(other statements)`,
  `(outside generated code)` and `(garbage collection)` hold the rest.
- Every program RK compiles now runs each statement that assigns a value in a
  source scope named after it (`Expr(:meta, :push_loc, Symbol("rk:<name>"))`,
  the lowered form of a macro expansion), which is what lets a profile of the
  fused native Enzyme gradient name the statement each reverse-pass sample ran.
  Scopes are debug information only: values, gradients and warm times are
  unchanged (measured on a Linux fleet host, Julia 1.10.12, Enzyme dogfood with
  libEnzyme `5639cd2f`: an RKPPL Student-t model with a scan, bit-identical
  values and gradients, warm gradient 63.98 against 65.95 µs and 67.33 against
  67.76 µs in two alternating processes, and no change in preparation or
  first-gradient compile time). `code_expr` and every reader of a compiled
  program's source see it without the scopes. Stack traces of an error inside
  a prepared program now name the statement as a `macro expansion` frame in
  file `rk:<name>`.
- New dependency: the `Profile` standard library.

## 0.1.308

- Fixed: with `branch_arms = :all`, a plate cell with several lazy branches
  side by side whose conditions read bound data was split into one plate per
  combination of their arms, 2^k plates for k branches, also for combinations
  no lane can take. A law with a few dozen flag branches never finished
  preparing. Such a cell is now one plate whose loop decides each
  branch per lane: its program grows with the branches, and one program still
  serves every binding. A lone branch, with any branches nested in its arms,
  is still one plate per leaf arm, as before, and `:taken` is unchanged.
  Values and gradients are unchanged. Measured on a Linux fleet host
  (Julia 1.10.12): 24 independent branches prepare in 1.2 s as one plate;
  with 2 or 3 independent branches, density is faster at every size (3 branches
  over 2000 rows: 15.75 → 13.39 µs), gradients over 200 rows or more 5–11%
  slower (17.57 → 18.54 µs) and over 10 rows faster (0.16 → 0.10 µs), and
  the first gradient compiles in 0.2–0.4 s instead of 0.9–1.9 s.

## 0.1.307

- **Fix:** a rebuild of one program reuses the code compiled for the first
  build again when the program holds a lazy branch whose arms are calls, such
  as a censored location-scale observation (`censored(Normal(...); lower =
  ...)`, whose arms run the distribution's endpoints). Since `b78ce392`
  (0.1.206) an ordinary branch carries its arms' graphs, and a body that
  embedded the branch kept them, with the value ids of the build that made
  them. So every rebuild compiled its readers and their derivatives again:
  for BRM's `brm_nlme_model` of one specification built a second time in a
  process, about 40 s of preparation and 120 s of `check_protocol`. A law
  kernel that every build shares, whose child kernel's branch reads the
  observation, compiled again at every preparation, of the same build too:
  preparing re-keyed the shared branch's graphs. The embedded branch now
  leaves its graphs out; lowering still reads them from the plan. Values are
  unchanged.

## 0.1.306

- JAX source translates `vec` and `reshape` of a matrix (or any array of
  rank two or more) of records in device code, such as a plate's records
  over `collect(Iterators.product(a, b))`: `vec` is the vector of those
  records in Julia's column-major order, each field (nested fields too)
  flattened alike, so `map(vec(records)) do r; r.field end` reads it as it
  reads a vector of records, and `reshape` keeps that linear order. Records
  keep their fields' lengths. A padded array's `vec` takes its valid records
  first, as `vec` of a padded numerical array orders its entries; `reshape`
  of it raises `JAXEmissionError`, as for a padded numerical array. Before,
  the module raveled or reshaped the tuple of field arrays and failed in
  Python (`TypeError: ravel requires ndarray or scalar arguments`).

## 0.1.305

- `@traceable` takes `@inline` and `@noinline` on either side of it
  (`@traceable @inline f(x) = ...`, `@inline @traceable f(x) = ...`). The
  method keeps the hint and the recorded source omits it; before, the first
  spelling was refused at definition and `jax_source` refused the second.
- Fixed in JAX source since 0.1.296: `x === nothing`, `x !== nothing` and
  `isnothing(x)` of a data-only value that is `nothing` or a tuple or array
  were refused (`the function isnothing calls itself`). Host preparation
  compares such a value with Python's `is None` again.

## 0.1.304

- A device JAX helper's write to its array argument now also reaches the
  caller from inside a `for` or `while` loop, an `if` arm, or the right side
  of a `&&` or `||`, where the construct does not itself assign that local
  (`for i in eachindex(t); result[i] = step!(cells, weights); end`): the
  construct carries the local, as it carries one it assigns. A loop or
  branch carries each outer local its body passes to a function whose
  `@traceable` source may write that argument, by index or through a
  helper it calls. Before, these calls raised `JAXEmissionError`.
- **Fix:** the right side of a `&&` or `||` whose value is used ran
  unconditionally in JAX when it rebound a local, by an index write or a
  helper's write to its array argument (`t = x > 0 && step!(w, x) > 0`), so
  the module wrote `w` where Julia skips the right side, a different value
  and gradient. It now runs only where Julia's does.
- A write the emitter cannot see before translating the loop or branch (a
  helper called through a name bound inside its body, a provider's source)
  still raises `JAXEmissionError`.

## 0.1.303

- JAX source translates a splatted tuple or named tuple in a tuple literal
  or a call (`(a, rest(...)...)`, `tuple(a, rest...)`, `f(t...)`), in
  `@traceable` source as in plain lowered helpers: it passes the tuple's
  fields, whose count is its type's. A helper that recurses over a record's
  field names and splats each step's result (`NamedTuple{names}(...)` of
  per-field values) therefore emits, unrolled over the names, never over
  data. Splatting a vector, whose length is data, still raises
  `JAXEmissionError`.
- A device JAX function whose helper writes an array argument in place
  (`w[i] += x`, also through a helper it calls) now writes the caller's
  array, as Julia's does: the function returns each written array after its
  value, and the caller rebinds the local it passed, so later reads (another
  helper's, or a sequential update across calls) see the write. Before, the
  write was dropped silently and the emitted module computed a different
  value. Writes before a helper's own `w = copy(w)` reach the caller, later
  ones the copy. A write the caller cannot rebind (inside a loop or branch
  that does not assign that local, to a value another kernel statement
  reads, to an expression other than a local name or a new array, or to an
  argument the helper also rebinds inside a loop, branch or `let`) raises
  `JAXEmissionError`.

## 0.1.302

- JAX source translates device `for` loops, generators and comprehensions
  over a vector's elements, over `enumerate` (one-based positions) and over
  `zip` of vectors (to the shortest), with destructured targets. Each is the
  runtime loop over the positions, so emitted source does not grow with the
  data and `jax.grad` differentiates it; a padded row's elements take lazy
  steps to its length. Loop targets stay the loop's own.

## 0.1.301

- `jax_source` host preparation collects and sorts tuple records without
  coercing their fields, including `issorted`, `by` field projections,
  dictionary generators and broadcast lookups. Local named helpers preserve
  writes to captured collections. `all` and `any` with a predicate stop at
  the first deciding element; numerical array equality preserves shape and
  exact integer comparisons. Host loops and generated JAX bodies remain
  bounded as input lengths change. The guides describe host preparation on
  its own page, `docs/src/jax-host.md`.
- `jax_source` reads a submodel's fixed-shape output (an `LKJCholesky`
  factor's diagonal, since 0.1.291) from the submodel's result. The caller
  named a local of the submodel's body instead, a Python `NameError`.

## 0.1.300

- `js_source` translates one element of a range: `r[i]`, `r[begin]` and
  `r[end]` (so `lastindex(r)`) for `a:b`, `a:s:b` and a vector's index range
  (`eachindex(x)`, `axes(x, 1)`). The element is computed from the range's
  ends, without building the window it selects, and a position outside the
  range is Julia's `BoundsError` with Julia's message. Before, each of these
  raised `JSEmissionError`, and a helper had to slice first (`x[r][1]`), which
  copies the window and also checks its far end.

## 0.1.299

- `jax_source`: a `map` do-block whose elements are numerical arrays of one
  shape (a plate record's vector field, `map(records) do r; r.nested.y end`)
  is a vector of dense rows. `first`, `last`, indexing, `similar`, `eachindex`
  and loops read each row as the array it is. Before, a helper reading such a
  row raised `JAXEmissionError: a ragged array without its lengths`.
- `jax_source`: a `map` do-block over entries with lengths of their own (rows
  of a ragged vector, records whose fields are padded rows) maps each entry
  with its lengths and returns the lengths of padded results beside them.
  Before, a `map` over a ragged vector of vectors computed on the device read
  each row's padding as entries, so a reduction of the mapped values returned
  a wrong value; records with padded fields failed in Python.
- `jax_source` translates `for x in xs` over a vector's entries in device code
  as the loop over `eachindex(xs)` that reads `xs[k]` into `x`. The loop
  variable is local to the loop, as Julia's is.
- `jax_source` translates `similar(A)`, `similar(A, T)` and `similar(A, [T,]
  dims...)` of numerical arrays, `sum` of a vector of dense rows (Julia's `+` of
  the entries, refusing an empty collection as Julia does), and generators
  over dense rows. `sum` of a vector of rows with lengths of their own raises
  a `JAXEmissionError` instead of failing in Python.
- `jax_source`: `eachindex` of a ragged vector of vectors, or of a vector of
  records with padded fields, ranges over its entries. Before, a loop over it
  used the rows' lengths as its count and failed in Python.

## 0.1.298

- `js_source` reads, indexes and iterates a vector of records that a recipe
  or helper builds, such as `[k(d) for d in dose]` where a prepared kernel `k`
  returns a named tuple, or a helper returns a tuple or named tuple: `length`,
  `eachindex`, `x[i]`, `x[indices]`, `x[a:b]`, `@view x[a:b]`, `for`,
  comprehensions and generators with their filters and reductions, `count`,
  `any`, `all`, `isempty`, `filter`, `map`, `findfirst`,
  `sum(f, x)`, broadcasts, `copy`, `collect`, `push!` and `x[i] = r`. Field
  projections such as `[r.a for r in records]` and `getproperty.(records, :a)`
  give vectors of that field's type. These vectors remain internal values:
  `types=` inputs and constants keep the vector types listed in the guide.

## 0.1.297

- `jax_source` converts a value stored into an array (`A[i] = v`, `A[i, :] = row`,
  `A[i] += v`, and a row stored into a vector of vectors) to the array's
  element type, as Julia's `setindex!` does: a `Float64` value stored into a
  `Float32` array is rounded to float32 in the emitted code. JAX's
  `.at[i].set(v)` performed that cast implicitly, with a FutureWarning
  ("scatter inputs have incompatible types") that a later JAX release turns
  into an error. Stores whose value already has the array's dtype are
  unchanged.

## 0.1.296

- JAX source translates nearest-loop `break`, convergence `while` loops,
  acyclic lowered helper branches, `@simd`, static Boolean `Val` arguments,
  numerical array destructuring, chained comparisons and integer parity.
  Data-dependent iteration stays in runtime loops. Unbounded loops need an
  authored rule for reverse differentiation.
- Optional numerical scalars cross device branches with a presence flag;
  `something` and `isnothing` retain their native lazy behavior.
- Real-vector `evalpoly` uses a runtime Horner loop, including checked
  `c[1:n]` prefixes. Padded lengths survive conditional results and tuple
  destructuring; bounded `min`/`max` range ends retain their capacity proofs.
  Matrix allocations can carry one logical extent across padded axes.
- General `derivative_rule` calls emit authored JVP and activity-selected
  staged VJP adapters through JAX's `hijax.HiPrim` API (tested with 0.12.0.1).
  Padded array ports retain their logical lengths and exclude padding from
  cotangents. Runtime checks remain guarded under branches and batching.
- Derivative regions retain their source plans for backend emission. Native
  calls keep concrete execution tables and do not read the retained plans;
  embedded native copies omit that build-time metadata.

## 0.1.295

- `jax_source` translates empty `NamedTuple()` constructors in prepared
  recipes and helper bodies, including kernels with no inputs and nested
  numerical records. Empty records retain the same representation as `(;)`.
  Literal field-name tuples in type parameters also support the typed
  constructor spelling `NamedTuple{()}(())`.
- Numerical record plates retain empty fields and derive their mapped axes
  from the first stored numerical leaf, including when empty fields come first.
- Prepared type-constructor recipes use the existing JAX constructor
  translations, including numeric conversions.

## 0.1.294

- `jax_source(...; data = ())` prepares input-free recipes on the host when
  they have no device translation, including constant Cartesian products.
  Data preparation needs no dummy input; live numerical parameters remain
  function inputs, and device-supported constants still fold in the functions.

## 0.1.293

- JAX emission preserves structural `NamedTuple` field names obtained from
  type parameters or `keys(record)` through local bindings, tuple selection,
  and helper arguments and results, including host data preparation. Record
  values and array lengths remain runtime inputs, with ordinary reverse
  differentiation preserved.

## 0.1.292

- `js_source` lets an authored plate read concrete tuples and named tuples
  produced by an earlier plate, including records containing ragged vectors.
  Field projections retain runtime iteration and the producer's shared work.
- Plates read vector views directly from their parent and broadcast singleton
  views and ranges alongside other inputs, including empty inputs.
- Reshaping a vector or vector view into a matrix keeps shared storage and
  nonnegative runtime dimensions, so hierarchical latent matrices can be emitted too.
- Numeric-vector and vector-view `cumsum` and `cumprod` use cumulative loops.
- Matrix columns support a vector of row indices with checked runtime gathering.
- Borrowed graph gathers retain their parent through plates; ordinary helpers
  accept view arguments while preserving their loops and lazy reads.
- Bound numbers, ranges and numeric tuple/named-tuple fields stay in module data
  storage, including input-check arguments, instead of specializing function bodies.

## 0.1.291

- `@traceable` and `TraceableSource` accept body-level early returns, including
  guards before loops. Recorded source preserves their lazy continuations;
  native methods keep their authored control flow. Loop-crossing returns remain
  unsupported.
- JAX source supports `continue`, short-circuit conditional writes, and internal
  nested-vector buffers with fixed-width numerical rows and `isassigned` state.
  Prepared integer-data bounds size allocations such as
  `isempty(ids) ? 0 : maximum(ids)`, including inside ragged cells. Loops retain
  their bodies across changing data sizes. Unsupported row alias mutations,
  helper argument mutation, and direct public buffer outputs are diagnosed.
- Device loops can read homogeneous numerical tuples at computed integer
  indices, including source `ntuple` lambdas with an authored `Val` arity
  behind an early-return guard. Empty padded buffer cells are guarded before
  evaluation, preserving ordinary gradients of shared arguments; integer
  reductions of cell data request prepared bounds for their allocations.

## 0.1.290

- `jax_source` supports plate cells returning numerical tuples and named
  records with nested records and array fields over multiple mapped axes.
  Subsequent plates retain field lengths and the record array's own padding;
  empty axes skip cell execution. Mapping and ordinary JAX reverse differentiation
  keep bounded bodies as data sizes grow.

## 0.1.289

- `js_source` preserves Julia's `Float64` spelling in `string` and string
  interpolation, including error messages: integral floats retain `.0`,
  signed zero and infinities retain their Julia spelling, and scientific
  notation uses Julia's exponent thresholds. Integer counts stay integers.

## 0.1.288

- `jax_source` prepares Cartesian products of data-bound numerical ranges
  and arrays with runtime host iteration. Collected products retain Julia's
  shape, first-axis-fastest order and empty axes. Homogeneous numerical
  tuple arrays cross to JAX as one array per field with the same shape,
  for compiled plates, indexing and reductions; source bodies do not grow
  with supports.

## 0.1.287

- Accept keyword arguments in `@traceable` methods and translate their calls
  in JAX functions and host data preparation. Required and typed keywords,
  dependent defaults, shorthand calls and nested helpers retain Julia's
  semantics. Keyword values remain runtime helper arguments.
- Preserve `on_error = :ignore` through captured helper calls with keywords.

## 0.1.286

- JAX host preparation translates `findall(vector)`,
  `findall(predicate, vector)` and `Dict(zip(keys, values))`, including
  `eachindex` values and dictionary lookups inside data-only plates.
  Searches return one-based positions, zip remains reusable and stops at its
  shortest input (its zero-argument form remains infinite). Dictionaries
  preserve duplicate-key replacement and Julia's `isequal` key semantics.
  These operations retain runtime loops,
  including for empty inputs, and can supply standalone `prepare_data`
  definitions without Julia-computed intermediate values.

## 0.1.285

- `jax_source` keeps predicates and arithmetic over prepared integers static
  when argument names overlap backend names, such as `isfinite(n)`. Those
  calls use the existing NumPy translation instead of leaving a traced JAX
  value in a Python conditional. Lazy guards and ordinary gradients are
  preserved, including when prepared data are passed to a jitted function.

## 0.1.284

- Add the internal log rising-factorial numerical graph used by distribution
  kernels. Small integer offsets evaluate chunked products and their shape
  partial together; other offsets use the optional SpecialFunctions extension.
  The ordinary derivative adapters come from that one graph, and the product
  loops retain their runtime count.

## 0.1.283

- `jax_source` preserves padded length bounds through integer addition,
  subtraction and multiplication by input-shape expressions. A helper can
  allocate at `(1 + length(parameters)) * length(row)` when `row` is ragged;
  its capacity comes from the input shapes during JAX tracing. Signed
  arithmetic tracks both interval endpoints, and loops retain their runtime
  structure. One emitted module continues to accept different input sizes.
  Zero-capacity buffers skip the last-entry read when padding a loop result.

## 0.1.282

- `js_source` translates nested vectors of numbers and Booleans, including
  ragged and empty groups: literals, inputs, bound data, indexing, iteration,
  comprehensions and plate cells retain their nesting. `vec` preserves the
  vector and `reduce(vcat, rows)` concatenates one level with runtime loops.
  Prepared kernels can consume those vectors inside another kernel.
  `js_run` preserves the nested
  output types and applies its typed-array mode to numeric leaves.

## 0.1.281

- `js_source` translates nested tuple literals and tuple `length`, including
  calls in validation helpers. It also translates `fill!` of numerical and
  Bool vectors with supported element conversions, preserving aliases and
  converting the fill value once even when the destination is empty. The
  destination length remains a runtime value. Reads of reassigned locals
  retain their values, including the temporary reused by keyword sorters.
  These operations let BRM's
  type-stable fixed-domain HSGP basis emit with scalar or per-axis mode counts
  and retain its input checks.

## 0.1.280

- `jax_source`: the cells of a plate whose cells fill their stored rows (an
  earlier plate's vectors or matrices of one fixed shape, such as one 3×3
  propagator per step length) reach a later plate that maps them as dense
  arrays, without lengths, so `size` and `length` of one cell are static there.
  Before, each cell carried its stored size as a padded length, and a helper
  that checked a typed argument's dimension (StanBlocks' `@juliacompat`
  `matrix[3, 3]` check, `size(x, i)` with `i` the helper's argument) raised
  `JAXEmissionError: size of a padded value along a computed dimension`.
- `jax_source`: `size(x, d)` of a padded value (a ragged row in its cell) along
  a dimension `d` that is static while JAX traces, such as a helper's integer
  argument, is the row's length along its padded axis and the stored size
  along another, as for a literal dimension. It raised the same error before.

## 0.1.279

- `jax_source` takes `rejecting`, the names of module functions that answer
  `-inf` where a numerical method cannot produce a value at a valid input
  (natively a `ReactiveKernels.NumericalFailure`), as a log density outside
  the model's support does. An ODE solve's exceeded per-output budget and
  failed solve are such failures. The module emits them as
  `jax.experimental.checkify` checks, which leave every plate cell's `vmap`,
  scan step, branch and loop as values, and evaluates its functions through
  `checkify`: those `rejecting` names return `-inf` for the whole evaluation,
  also under `jax.jit`, `jax.grad` and `jax.vmap`, and the others raise, as
  before, with the same message. Invalid ODE inputs and every other runtime
  check still raise. The module's other runtime checks are then `checkify`
  checks too, so the first failure in program order decides, as natively: a
  check on a failed solve's state (a solve continuing from it) neither raises
  nor hides the failure. A module without such a failure is unchanged.
- Such a module needs a JAX whose `checkify` has `checkify.Error.has_error` and
  differentiates a `custom_vjp` (every diffrax solve) after a check. No JAX
  release has either: jax 0.12.0.1 gives a wrong or failing gradient there
  (`benchmark/repro_jax_checkify_custom_vjp_control_flow.py`). Branch
  `jax-v0.12.0.1-checkify` of `https://github.com/nsiccha/jax` carries both;
  without them, the module refuses to import with an `ImportError` naming it.
- A JAX ODE solve fails where its step size falls below 16 ulps of the latest
  time, or where it steps from a non-finite state or its error estimate is
  not finite, as the native solve stops as `DtLessThanMin` or `Unstable`.
  Before, a finite-time blow-up (`dy/dt = k y²`) rejected steps until
  diffrax's limit of 2³¹ − 1 steps, about 220 s an evaluation, and only then
  failed; it now fails at once. Values and gradients where the solve
  succeeds are unchanged.

## 0.1.278

- `jax_source`: host `prepare_data` translates more ordinary data-only Julia,
  with native-equal values, so a `@traceable` data helper keeps its natural
  spelling:
  - typed comprehensions `T[f(x) for x in xs]`, each element converted as
    Julia's `convert` does (`nothing`, or a non-integral value for an integer
    type, raises), and nested generators `[f(i, j) for j in a for i in b if c]`;
  - `push!` and `append!` onto any local vector (a literal such as `[0.0]`, a
    `vcat` or slice result, an input): it becomes a Python list, a fresh copy,
    so an input is never written; `sizehint!(v, n)` returns `v`;
  - `x[findfirst(p, x)]` and `x[findfirst(p, x)] = v`, raising
    `ArgumentError: invalid index: nothing of type Nothing` as Julia does;
  - `issorted(v)` and a scalar `isless(a, b)` in Julia's order (-0.0 before
    0.0, every NaN last, then `missing`); `isless` of numbers also in the JAX
    functions;
  - `Set`: `Set()`, `Set{T}()`, `Set(v)`, `in`, `push!`, `union!`, `delete!`,
    `length`, `isempty`, with Julia's `isequal` membership (-0.0 and 0.0 two
    keys, every NaN one). A set is not iterated: Julia iterates it in its hash
    order, so that raises `JAXEmissionError`;
  - `(; a, b) = nt` destructuring (also in the JAX functions), `view(x, r)`,
    `@view x[r]` and `@views`;
  - anonymous functions (`d -> d >= 0.0`), called where host code calls them:
    `filter(f, v)`, `filter!(f, v)` (shrinking a list in place), predicates.
    One reading locals of its own code is not passed to another helper.
- `jax_source`: a host helper that writes its argument in place (`rv[k] = x`,
  `push!(rv, x)`, `sort!(rv)`, also through a helper it calls) writes its
  caller's vector, as Julia's does, whether or not the caller reads the
  result: `merge_into!(scratch, a, b)` used to write a copy, so the caller's
  `scratch` was silently unchanged. The caller passes its own local (a copy of
  an input or constant first, a list when grown); a slice or other expression
  there raises `JAXEmissionError`.
- `jax_source`: after a branch or loop, a host local that is a list on one
  path and an array on another is read as an array, never as a Python list
  (`rv .* 2` of a list repeated it).

## 0.1.277

- `ReactiveKernels.jax_source` with `data` partitions a top-level plate by the
  branches of its cell whose condition reads only the plate's data (a
  censored row's `y <= lower`), also when they sit inside an arm of a
  parameter-dependent branch (RKPPL's support guard). `prepare_data` returns,
  for each pattern of those decisions, the 0-based positions of the cells that
  take it (`<value>_lanes_<k>`) and the order that restores the plate
  (`<value>_lane_order`); the functions run one `jax.vmap` per pattern over its
  cells, each translated with the pattern's conditions decided, so a cell
  evaluates only the arms it takes. Before, the plate was one `jax.vmap` whose
  batched `jax.lax.cond` evaluated both arms on every cell: a censored
  Student-t response ran its incomplete-beta continued fraction on every row,
  not only on the censored ones. The patterns follow the cell's source (every
  pattern has its plate, also when no cell takes it), so the module text does
  not depend on the data. Values and gradients are unchanged. A plate with
  more than eight patterns, a mapped argument that is not a vector of numbers,
  or a value that is not a vector of numbers stays one `jax.vmap`.

## 0.1.276

- `js_source` translates arithmetic on `Bool`s as Julia does it, with native
  values and types: a `Bool` operand of `+`, `-`, `*`, `/`, `^`, `min`/`max`,
  a comparison or another function of Julia's own, next to `Float64`s,
  `Int64`s or other `Bool`s, runs Julia's method for those operands,
  translated from its typed code. `(a .> x) .+ (b .> x)` gives `Int`s,
  `w .* (v .> x)` gives `Float64`s, `true * true` stays a `Bool`, and the
  zeros keep their signs: `false * -Inf` is `-0.0` and `false + -0.0` is
  `-0.0`, as natively. `Int(b)`, `Float64(b)`, `float(b)`, `convert`,
  `promote`, `zero`, `one` and `oneunit` take a `Bool` (`0` or `1`). `sum` of a
  `Bool` vector counts its `true`s, and `+`, `-`, `*` and `/` of `Bool`
  vectors translate like those of number vectors. Before, each was refused
  with "`+` with argument types (Bool, Bool) has no JavaScript translation".
- `js_source` reads the `BitVector` of a comparison broadcast (`x .> c`) like
  a `Vector{Bool}`: `sum(x .> c)`, `count(x .> c)`, `any`/`all` of it and
  `w .* (x .> c)` translate, with Julia's types (`Int` from `sum`). A
  `BitMatrix` is still refused.
- `float` of an `Int` is a `Float64` while translating, as natively. It used
  to keep the `Int` type, so a later `^` or `div` could take the integer path.

## 0.1.275

- `js_source` emits a `const` global array the code reads: a vector, or a
  matrix as an array of row arrays, of `Float64`, `Int64` or `Bool` becomes a
  module-level array (`const _rk_const_LEVELS = [50.0, 100.0];`) holding its
  value at emission, emitted once, which the module reads and writes as the
  kernel does the global. A broadcast over one (`scale .* LEVELS`, `2 .* LEVELS`,
  `x .+ GRID`) loops over it with Julia's length rules. Such a broadcast used
  to recurse until a bare `StackOverflowError`, and any other run-time use
  was refused with "pass it as a kernel input"; an array without a JavaScript
  value (a `Vector{String}`) is still refused by name, now also inside a
  broadcast. `jax_source` already emits these arrays as literals.
- A matrix bound at preparation is a frozen module constant of frozen rows,
  as a bound vector is; a run-time use of one was refused.
- A broadcast known while translating inside another (`x .+ (2 .* 3)`) is
  fused like any other; it was refused as having a non-concrete element type.

## 0.1.274

- `js_source` translates predicates over tuples, matrix literals, matrices
  built element by element and matrix slices, with native results and native
  errors:
  - `all(f, t)`, `any(f, t)` and `count(f, t)` of a tuple, and `all(t)`,
    `any(t)`, `count(t)` of a tuple of `Bool`s: `f` is called on each
    element with that element's own type, and `all`/`any` stop at the first
    element that decides them, as Julia does (`all(a -> length(a) == n,
    axes)`, `all(>=(1), rows)`); a function that may return something other
    than a `Bool` is refused by name;
  - matrix literals of numbers: `[a b; c d]`, `[a b]` and `Float64[a b; c d]`,
    in a plain helper (`hvcat`, `hcat`, `typed_hvcat`, `typed_hcat`) and in a
    kernel's recipe (RK's `_native_hvcat` and `_native_hcat`), in Julia's
    element type; a layout `hvcat` rejects raises Julia's error where the
    literal is;
  - `Matrix{T}(undef, m, n)` for `Float64`, `Int64` and `Bool`, and writes
    `m[i, j] = x`, `m[:, j] = v`, `m[i, :] = v`, converting to the element
    type and checking positions and lengths as Julia does;
  - `diag(m)`, and `extrema(v)` of a vector;
  - matrix slices `view(m, :, j)` and `view(m, i, :)`, checked where they
    are made: an element read and `length` read the matrix, and a function
    of the whole slice (`extrema`, `sum`, `minimum`, `maximum`, `collect`, …)
    reads it as it is when it is read; a write through a slice is refused;
  - a tuple of values of one type (pairs of bounds, axis vectors) read at a
    position computed at run time, where only tuples of numbers were;
  - a closure built on every pass of a loop (`all(v -> lo <= v <= hi,
    axes[j])` inside `for j in 1:d`), holding the values it captured on
    that pass.

  So the LKJ density of a Cholesky factor written as a matrix literal
  (`lkj_corr_cholesky(eta).logpdf([1.0 0.0; r sqrt(1 - r^2)])`) emits, and a
  type-stable Hilbert-space GP basis written with these operations emits and
  matches native (snag `js-source-all-pr-b15faf4f`).

## 0.1.273

- `js_source` translates the quantiles of RK's distribution laws:
  `standard_student_t`, `student_t`, `normal`, `lognormal`, `exponential`
  and `gamma`. It also translates the gamma law's `cdf`, `ccdf`, `logcdf`
  and `logccdf`, and direct calls of SpecialFunctions' `beta_inc`,
  `beta_inc_inv`, `gamma_inc` and `gamma_inc_inv`. Node agrees with native
  to about 1e-13 relative or better; far below 1e-15 the vendored code is
  the more accurate side.
- **`beta_inc` and `beta_inc_inv`** come from pinned stdlib-js `betainc`
  and `betaincinv` (Boost's `ibeta`), with Julia's argument checks and
  special cases. SpecialFunctions' code there is not concretely typed, so it
  cannot be translated.
- **`gamma_inc` and `gamma_inc_inv`** translate from SpecialFunctions' own
  Julia code. stdlib-js's `gammainc` is inaccurate above a shape of about
  1000.
- **Emitter coverage for that code:**
  - stepped ranges (`n:-1:1`, `a:s:b`), constant or computed at run time:
    iterated, with `length`, `first`, `last`, `step` and `isempty`;
  - `ntuple(f, Val(n))`, `cumprod` and `cumsum` of a tuple, `map` over a
    tuple, and `minmax`;
  - `findfirst` of a predicate over a tuple, with `something`;
  - `iszero` and `LazyString` exception messages;
  - splatting a tuple computed at run time.

## 0.1.272

- `jax_source`: a `for` loop to a padded row's length (`for t in 1:length(r)`
  over a ragged row in a plate cell, or `eachindex(r)`) runs no body past the
  row's end. It used to run the body at every step up to the padded width and
  select the carry with `jnp.where`, so a body that is not finite at a step
  Julia never runs (`log(length(r) - t + 1)`, say) gave the right value but
  `jax.grad` NaN. Each step is now a lazy branch, as a counted `while` to a
  padded length has been since 0.1.263 (`jax.lax.cond` per step; under
  `jax.vmap` the untaken arm's reads go through `stop_gradient`).
  CONTRIBUTING.md, "Lazy branches".
- After such a loop, and after a counted `while` to a padded length, an array
  padded by that length repeats its last valid entry in its padding, the
  padding every padded value holds. Before, an array written at the loop
  index got the body's values at the padded steps, which were valid only when
  the body was.

## 0.1.271

- `js_source` translates more of the Base a plain helper uses over a window
  of a series, with native results and native errors:
  - the comparison predicates `==(x)`, `!=(x)`, `<(x)`, `<=(x)`, `>(x)`,
    `>=(x)` and `isequal(x)`, wherever a predicate is taken: `count`, `any`,
    `all`, `filter`, a generator's condition, and `findfirst`/`findlast`;
  - `findfirst(p, v)` and `findlast(p, v)` of a vector or a range, and of a
    vector of `Bool`s, giving the position or `nothing`, and `isnothing`;
    a predicate that does not return a `Bool` is refused by name;
  - `sum(f, v)`, `maximum(f, v)` and `minimum(f, v)` of a vector or a range,
    `f` a function, a closure or a predicate (`sum(abs2, x .- y)`,
    `sum(isnan, x)`): a sum is pairwise as Julia's is, and an empty source
    gives Julia's result (`sum(abs2, Float64[])` is `0.0`) or its
    `MethodError`;
  - `x[a:b]`, a checked copy of the window (`BoundsError` out of bounds);
    `collect` of a range or a vector;
  - `merge` of named tuples (`merge(nt, (; a))`), later fields winning, in
    Julia's field order;
  - `abs2` and `isequal` of numbers.

## 0.1.270

- `js_source` translates Julia's irrational constants (`π`, `ℯ`,
  `MathConstants.γ`, and those of any package's `@irrational`, such as
  `IrrationalConstants.log2π`) through Julia's own methods for them, with the
  native values. Arithmetic rounds the constant to the float it promotes to
  (`log(tail * π)` multiplies by `Float64(π)`); a comparison rounds it up or
  down as Julia does, so `x < π` holds for `x == Float64(π)`, and `==` is
  false; `ℯ^x` is `exp(x)` and `sin(π)` is `0.0`. A call of three or more
  operands (`2 * π * r`) folds from the left, as Julia's method does. A
  constant times or over a vector or matrix (`π * v`, `A / ℯ`) is the
  broadcast Julia's method makes. A helper given `π` is specialized on it
  rather than taking it as a parameter. Native execution is unchanged.

## 0.1.269

- `js_source` translates matrix arithmetic on its row-array matrices: `*` of
  matrices, vectors, scalars and the transposes `m'`, `transpose(m)` and `v'`
  (chains such as `A * B * v` and RK's own recipe product included); `+` and
  `-` of matrices; `sum`, `prod`, `maximum` and `minimum` of a matrix, whole
  or with `dims = 1` or `2`; `vec` (a fresh vector in column order, where
  Julia's shares the matrix's memory); `permutedims`; `zeros`, `ones` and
  `fill(x, m, n)`; and broadcasts over matrices, columns and rows with Julia's
  shape rules, in place into a matrix too. `prod` of a vector translates.
  Errors are Julia's (`DimensionMismatch`, `ArgumentError`, `MethodError` for
  an empty `maximum`). `\`, factorizations and indexing a transposed matrix
  stay refused by name.
- `js_source`: a range of length one in a broadcast against a longer vector,
  `(1:1) .+ x`, is broadcast like a vector of length one. It read
  `start + i` before and returned wrong values silently.
- `js_run` decodes a `BitVector` or `BitMatrix` output (a comparison
  broadcast such as `x .> 0`) to that type.

## 0.1.268

- `js_source` emits prepared kernels, bound data included. The new
  `js_source(kernel; types, name)` takes a kernel as `prepare` returns it
  (a flat, tuple or shaped WANT), such as one prepared with
  `prepare(spec; have, want, bound)` or an RKPPL query from `prepare_query`:
  its module's function takes the kernel's remaining HAVE values and returns
  what the kernel returns. `types` defaults to the kernel's exemplar types.
  A recipe of another kernel may call such a kernel too
  (`probability = QUERY(unconstrained)`); the call becomes a module-level
  function translated from the kernel's plan, once per kernel and argument
  types, as `jax_source` translates one. In both, bound numbers are literals
  and bound vectors of `Float64`, `Int64` or `Bool` are frozen module
  constants, emitted once. Before, `js_source(kernel)` had no method, and a
  recipe calling a prepared kernel failed (up to 0.1.263 with "the
  expression `$(Expr(:static_parameter, 4))` as an operand has no
  JavaScript translation").
- A kernel's input checks (`input_check!`, such as ReactiveKernelsPPL's
  packed-length check of `unconstrained`) run at the entry of a `js_source`
  module's function, raising the native exception. Before, every module
  skipped them: a packed point of another length returned values where the
  native kernel raises `ContractValidationError`.
- `js_source` translates `vcat` of vectors and numbers of the result's
  element type (RKPPL's `vcat([e])` coefficients) and `*` of strings (the
  message a check builds). Native execution is unchanged.

## 0.1.267

- `ReactiveKernels.js_source` translates `rk_ode_rk45_tol`. The emitted module
  integrates with the Dormand–Prince 5(4) method of the vendored
  [`@reside-ic/dopri`](https://github.com/mrc-ide/dopri-js) 1.0.1 (MIT,
  pinned and bundled in `src/js_vendor/`, its licence in the bundle), copied
  into a module only when it integrates an ODE, and drives it as the native bridge drives OrdinaryDiffEq's `DP5()`:
  the same argument checks and exception types, one continuous solve on the
  normalized time span, outputs from the dense interpolant, at most
  `maxsteps` accepted steps between consecutive outputs, and a
  `JuliaError` named `ODESolveFailure` with `retcode` and `t` where native
  throws one: `MaxIters` for the budget, `Unstable` where the step size
  vanishes (as native's does after a blow-up or a NaN in the right-hand
  side). The library takes its own step sequence, so values agree with
  native to the solvers' accuracy, not to rounding: within the requested
  tolerance on smooth problems, and as close to the true solution as native
  is where the global error exceeds the tolerance. `rk_ode_bdf_tol` stays
  refused, now naming itself instead of an internal `isempty`.
- `js_source` modules take and return `Matrix{Float64}`, `Matrix{Int64}` and
  `Matrix{Bool}` as arrays of row arrays: `m[i][j]` is Julia's `m[i+1, j+1]`.
  A matrix without rows keeps its column count as a `cols` property of the
  outer array (`Object.assign([], { cols: 3 })` for `zeros(0, 3)`; `[]` is
  0×0). Inside a module `m[i, j]`, `m[:, j]`, `m[i, :]`, `size`, `axes`,
  `length`, `firstindex`, `lastindex` and `copy` translate. `js_run` passes
  and decodes matrices, with `Float64Array` rows under
  `vectors = :float64array`.
- `js_source` translates vector literals, `[a, b]` and `Float64[a, b]`, with
  Julia's promotions.

## 0.1.266

- `js_source` translates lazy iterators over a vector or a range: a
  comprehension with a condition (`[f(x) for x in v if p(x)]`), a generator
  over a filter or over another generator, and `Iterators.filter`, consumed by
  `collect`, `sum`, `maximum` and `minimum` (each also with `init`), `count`,
  `any`, `all` and `isempty`. Each is one loop that calls the functions in
  Julia's order. A reduction folds from the first element (or `init`) as
  Julia reduces an iterator, so a generator's `sum` is Julia's to the last
  bit, and an empty iterator gives Julia's result or its error. `argmin`,
  `argmax`, `findmin` and `findmax` of a vector (Julia's order: the first
  `NaN`, then `-0.0` before `0.0`) and `filter` of a vector translate too, and
  a helper may read a module's constant, such as `Iterators.filter`. Before,
  these raised `JSEmissionError` (`Base.Iterators.Filter … has no JavaScript
  translation`).

## 0.1.265

- `js_source` translates keyword calls of plain helpers, named tuples built
  and read in a recipe or helper, and destructuring. `f(x; a, b)` inlines
  `f`'s keyword sorter: inference decides its checks of the keyword names,
  defaults computed from other arguments are emitted, and the body function
  becomes a JavaScript function of the keyword values. A keyword `f` does not
  take, or a required one left out, throws Julia's `MethodError` or
  `UndefKeywordError` when the call runs, as natively. `(; a, b)` and
  `NamedTuple{names}(tuple)` build an object, and `x.name` reads it.
  `a, b = t` destructures a tuple or named tuple, and a generator over a range
  written in the source (`a, b, c = (zeros(n) for _ in 1:3)`). A call whose
  result inference knows to be a constant, and that has no effect and cannot
  throw, is that constant whatever its arguments: `LogExpFunctions.log1pexp`
  and `logistic` translate through their type-dependent thresholds. A local
  that holds values of several types is read with the type inference knows
  where it is read. Also translated: `Vector{T}(undef, n)` and `similar(x)`
  (filled with zeros), `x[index]` for a vector of indices (`BoundsError` before
  any element is read), and `@view x[a:b]` read by `sum`, `length` and
  indexing. Native execution is unchanged.

## 0.1.264

- `js_source` translates RK's distribution laws, so a browser module computes
  the same law the RK, RKPPL and BRM models compute. Examples:
  `standard_student_t(nu).cdf(z)`, `student_t(nu, location, scale).logpdf(x)`,
  and `normal`, `lognormal`, `exponential` and the `gamma` density from
  ReactiveKernelsDistributionKernels, in a recipe, a plate or a comprehension.
  Before, every law with a special function raised `JSEmissionError`.
- **Special functions:**
  - SpecialFunctions' `erf`, `erfc`, `erfcx`, `gamma`, `loggamma` and
    `logabsgamma` call C libraries or read a `Float64`'s bits. The module
    computes them with pinned stdlib-js implementations that RK carries in
    `src/js_vendor/`, copied into a module only when it uses them.
  - Their Julia special cases and `DomainError`s are kept.
  - Functions built on them, such as `logbeta`, `logerfc`, `log1pexp` and
    `log1mexp`, translate from their own Julia code.
- **Emitter coverage for the code under those laws:**
  - tuples: a read at a position known while translating or computed at run
    time, destructuring, and `first`/`last`;
  - a method's static parameters;
  - `evalpoly` with a coefficient tuple;
  - `promote`, and `oftype` with an irrational constant;
  - `isinteger`, `real`, `copysign`, and `iseven`/`isodd` of a `Float64`;
  - checked math (`sqrt`, `log`) of an `Int64`;
  - RK's guarded native calls.
- **Agreement:** Node agrees with native to about 1e-14 relative or better,
  far tails included.
- **Distributions.jl objects:** a law written with them (`cdf(TDist(nu), x)`)
  is refused with a `JSEmissionError` that names the RK spelling.
- **Not covered yet:** the Gamma CDF, which needs stepped ranges inside
  `gamma_inc`, and matrix laws such as `lkj_corr_cholesky`.

## 0.1.263

- `jax_source` translates a counted `while i < n` loop (or `<=`) whose `n`
  is a padded row's length, such as `length(t)` of a vector computed in a
  plate cell, as a `jax.lax.fori_loop` to the row's static limit whose steps
  are lazy branches (`jax.lax.cond`): a step past `n` runs no body and keeps
  the carry, so nothing the body would compute there reaches the value or the
  gradient. `jax.grad` reverses it. Before, only a static `n` was counted; this
  one emitted a `jax.lax.while_loop`, and `jax.grad` raised `Reverse-mode
  differentiation does not work for lax.while_loop or lax.fori_loop with
  dynamic start/stop values`. ReactiveKernelsPPL's stopping-ratio
  `interval_censored`, `censored` and `truncated` ordinal evidence over
  thresholds computed in a cell (`Ref(cuts)` of a cell local) reaches this
  loop through its stage survival. A `while` whose body assigns a name its
  limit reads is no longer counted, static limits included: it stays a
  `jax.lax.while_loop`, which reads the new limit at every step as Julia
  does. Native execution is unchanged.

## 0.1.262

- `jax_source` translates Statistics' `mean`, `var` and `std` through a new
  package extension, `ReactiveKernelsStatisticsExt` (weak dependency
  Statistics): `mean(x)`, `mean(f, x)`, `mean(x; dims)`, and `var`/`std`
  with `dims`, `corrected` (a literal) and `mean`. Each composes the existing
  `sum`, `length` and `size` translations, so padded rows reduce over their
  valid entries and `prepare_data` (host) calls take NumPy's spelling; an
  empty argument gives NaN, as in Julia. Before, a centred effect
  `s * (x - mean(x))`, such as a BayesianRegressionModels assignment exported
  with `jax_source(rk.model, rk_bound_plan(rk))`, failed with "the function
  `_mean` has loops or branches". Native execution is unchanged.

## 0.1.261

- A plate cell whose statements each read the one before, such as
  `m_k = m_{k-1} .+ b[k] .* x_k[rows[i]]` for k up to 140, with a later reader
  taking the last value entry by entry, now prepares and differentiates
  however long the chain is. Such statements compose into lazy recipes that
  hold each other, so their types used to nest as deeply as the chain was
  long. On a 140-statement cell, Julia's and Enzyme's type inference then
  overflowed the stack (`Internal error: stack overflow in type inference of
  _cell_broadcast_entry`, then `StackOverflowError`). Where it did not, the
  first ordinary Reverse gradient took superlinearly longer: on a toy cell of
  140 statements, 1322 s against 41 s now. A recipe nested more than 32 levels
  deep is now stored at its own statement, in one fresh array whose entries
  are read through its operands' recipes. The statements after it compose over
  that array. Values and gradients are unchanged.

## 0.1.260

- `jax_source` translates a kernel that captures a `NamedTuple` of data and
  reads it whole, for example by passing it to a `@traceable` helper
  (`metadata = (; weights, indices)`; `f(x, metadata)`). The record is the
  Python tuple of its fields, as a named-tuple HAVE already crosses, and a
  field read indexes it by position. Each array field is a typed module
  constant named after the record and the field (`METADATA_WEIGHTS`), keeping
  Julia's shape, axis order and element type (matrices and higher-rank integer
  arrays included); a record holding an array is itself a module constant
  (`METADATA`), and a record of literals stays a static literal (it can size
  an allocation). A field that is a function or a type raises
  `JAXEmissionError` naming the field. Before, any captured named tuple read
  whole raised `JAXEmissionError` (snag emit-an-rk-kerne-6bbd4365).
- Fix (`jax_source`, silent wrong value): a tuple assignment that writes
  arrays by index inside a loop or a branch (`loc[p], scale[p] = l, s`) now
  carries those arrays through `jax.lax.fori_loop` / `jax.lax.cond`, as a
  single `loc[p] = l` already did. Before, only a tuple target's bare names
  were carried, so the writes were dropped and the arrays kept their values
  from before the loop.

## 0.1.259

- `jax_source` translates a reduction of a tuple of numbers:
  `logsumexp((a, b))` (a mixture law's candidate log densities),
  `maximum`, `minimum`, `prod` and `sum` of such a tuple, its entries of
  mixed types too (`sum((x, n))`). The tuple becomes one array of its entries'
  promoted type (a float one for `logsumexp`), since `jax.numpy` reductions
  refuse a Python tuple; the emitted module raised `TypeError: max requires
  ndarray or scalar arguments, got <class 'tuple'>` on its first call before.
  `jax.grad` of a tuple's `logsumexp` is its softmax weights, and 0 when every
  entry is `-Inf`, as an array's is.

## 0.1.258

- `prepare(spec; have, want)` and `prepare!(cache, spec; have, want)` accept a
  shaped WANT, the WANT `js_source` and `jax_source` already accept: a
  `NamedTuple`, or a tuple holding a tuple or a `NamedTuple`, whose leaves are
  port names, nested as deep as wanted. The kernel returns that shape, each
  name replaced by its port's value: `want = (curve = :curve, count = :n)`
  returns `(curve = …, count = …)`, aliases included. A port named twice is
  computed once; bound data, a `PreparationCache` and an authored signature's
  defaults and keywords work as for a flat WANT. Before, such a WANT raised
  `ArgumentError: want entries must be port names or Value objects`; other
  entry points (`plan`, `extract`, `vectorize`, …) still take port names, and
  now say which ones accept a shape.
- `js_run` returns a one-name tuple WANT (`want = (:x,)`) as the bare value, as
  the native kernel of that WANT returns it; it gave a one-element tuple. Every
  WANT form now comes back exactly as `prepare(spec; have, want)` returns it.

## 0.1.257

- `ReactiveKernels.js_run(source, entry, cases; node, vectors, timeout)`
  (prototype, not exported) runs a `js_source` module in Node.js: it calls the
  function `entry` once per case, each a tuple of the HAVE values in `INPUTS`
  order, in one Node process, and returns one `JSOutcome` per case. A
  successful call's `value` has the shape and types the native kernel returns
  (a value, a `Tuple` or a `NamedTuple`); a call that throws gives its error's
  `name` and `message` and whether it was the module's `JuliaError`; every
  outcome reports whether the call changed the arrays it was given. A run that
  fails (no Node.js, a module that does not import, a missing entry, a crash or
  a timeout) throws `JSRunError`. `vectors = :float64array` passes numeric
  vectors as `Float64Array`s. Node.js is `node` (an executable or a `Cmd`),
  by default `ENV["RK_JS_NODE"]`, else `node` on the PATH.
- `js_source` modules also export `RETURNS`, the WANT's shape, and give each
  `OUTPUTS` entry its Julia `type`.

## 0.1.256

- `ReactiveKernels.js_source` translates a range written in a helper or a
  kernel, such as `for _ in 1:60`, `OneTo(3)`, `x .^ (0:4)`,
  `[f(k) for k in 1:5]`, `map(f, 1:4)` or `plate(1:4) do i ... end`. Such a
  range is known while translating, so it needs no kernel input, and its
  bounds are written as number literals. It used to raise `JSEmissionError:
  the value 1:60 of type UnitRange{Int64} is needed at run time`.
- A branch whose condition is known while translating (an `if` on a `const`
  flag, the entry test of a loop over `1:0`, `isempty` of a range a helper is
  called with) leaves out the code it never runs. It used to raise an internal
  `JSEmissionError` ("no label for a jump").

## 0.1.255

- `type_stability_level = :rk` audits the functions of exactly the packages of
  ReactiveKernels' repository (ReactiveKernels, ReactiveKernelsPPL, the other
  packages under `packages/`, and their extensions), identified by package
  UUID, the same test `jax_source` uses since 0.1.253. Before, it recursed into
  every package whose name starts with `ReactiveKernels`, including
  ReactiveKernelMethods' packages (ReactiveKernelsNUTSExamples,
  ReactiveKernelsStreamingStats, …), which are now leaves like any other
  consumer: a call into them must be dispatched statically and return a
  concrete type, and their bodies are audited at `:all`, the default.

## 0.1.254

- `ReactiveKernels.js_source(spec; have, want, types, name)` (prototype, not
  exported) emits a self-contained JavaScript (ES) module that computes a
  kernel's WANT values from its HAVE values in a browser or Node.js, with no
  server. Plain Julia helpers the kernel calls are translated from their typed
  code, branches, early returns and loops included, without `@traceable` or
  another annotation; ordinary recipes, broadcasts, `plate` and `scan`
  translate, the last two as loops over runtime lengths. Values are `Float64`,
  `Int64` and `Bool` scalars and vectors of them; indexing is checked, and
  `log`, `sqrt`, `^` and the like raise Julia's `DomainError`. The module
  exports the kernel function, `INPUTS`, `OUTPUTS` and `JuliaError`, whose
  `name` is the Julia exception type. A construct without a translation (a C
  call, `fma`, a matrix, a scan with `history`) raises `JSEmissionError`
  naming it and its call chain.

## 0.1.253

- `jax_source` treats the functions of every package of ReactiveKernels'
  repository (ReactiveKernels, ReactiveKernelsPPL,
  ReactiveKernelsDistributionKernels and the other packages under
  `packages/`) and of their extensions as its own by package UUID, wherever
  the package is installed. One model now emits one module text whether
  ReactiveKernelsPPL is developed under ReactiveKernels' root or added by URL
  with `repo-subdir` into a package tree of its own. Before, an installed
  ReactiveKernelsPPL's internal helpers, such as `_ppl_cell_latent_block` of
  a dotted ragged cell latent, were emitted as named private Python functions
  ("from its Julia definition") that evaluate every argument, while a
  developed checkout, and ReactiveKernels' own tests, translated them in
  place.

## 0.1.252

- A prepared vector of tuples (or of named numbers) crosses from
  `prepare_data` to a `jax_source` module's functions as the tuple of its
  columns, and the functions convert each column with `jnp.asarray`, as they
  already convert numeric arrays (a float32 module casts each float column).
  A plate cell then indexes it by its traced index whether the prepared data
  are host NumPy arrays (eager calls, `jax.jit` with the data closed over) or
  `jax.jit` arguments. Before, an `@rkppl` model with a dotted ragged cell
  latent (`e[g] .~ Normal.(…)`) raised `TracerArrayConversionError` in
  `log_density`, `log_likelihood`, `constrain` and `jax.grad` unless the
  prepared data were passed to `jax.jit` as arguments, wherever its per-cell
  dimensions were read inside the cell (since 0.1.240, when RKPPL is
  installed in a package tree of its own). A data definition that builds the
  vector with a comprehension (`[(minimum(r), maximum(r)) for r in R]`) was
  returned as a list of rows, which the functions cannot index by a traced
  index in any mode; `prepare_data` now returns its columns. A list of rows
  passed to the functions directly raises a `TypeError` naming the expected
  form.

## 0.1.251

- A failed solve of `rk_ode_rk45_tol` or `rk_ode_bdf_tol` throws the new
  `ReactiveKernels.ODESolveFailure(method, retcode, t)`, a subtype of the new
  abstract `ReactiveKernels.NumericalFailure`, instead of an untyped
  `ErrorException`. It covers what the method cannot do from valid inputs:
  an unsuccessful SciML return code (`retcode`, for example `:Unstable`) and
  `maxsteps` exceeded between two requested outputs (`retcode === :MaxIters`);
  `t` is the time, in the caller's units, the solver had reached. Invalid
  inputs keep throwing `ArgumentError` or `DomainError`, and a wrong RHS
  length `DimensionMismatch`. A density target can now answer `-Inf` for
  exactly this failure, as Stan rejects a proposal whose ODE solve fails,
  while other errors propagate, as ReactiveKernelsPPL's sampler targets now
  do. The message keeps its old text and adds the time:
  `native DP5 failed: Unstable at t = …`.
  **Breaking:** code that caught the failure as `ErrorException` catches
  `ReactiveKernels.ODESolveFailure` (or `ReactiveKernels.NumericalFailure`)
  instead. The gradient's solve throws the same exception. `jax_source`
  modules still raise at a failed solve.

## 0.1.250

- `prepare_ad` over a `NonAllocatingKernel` no longer fails when the program
  throws at the exemplar. Its seeding call (one primal evaluation that gives
  the operations their buffers) used to let the error escape, so an exemplar
  that is a valid type/shape but not a point where the program evaluates (an
  ODE that does not integrate there) failed the whole preparation. It now
  warns, naming the error, and continues: the operations that call did not
  reach allocate on every gradient call, as an untaken branch's do, and values
  and gradients are unchanged. `warm = true` likewise keeps its preparation
  when its gradient evaluation at the exemplar throws, with a warning. The
  check that the objective is a scalar, which calls the kernel at the
  exemplar when its declared type is not a `Number` and inference (not
  consulted for a `NonAllocatingKernel`) does not prove one, falls back to
  inference when that call throws; only an objective inference does not prove a `Number` is then
  refused, naming `name::Float64` as the declaration that avoids the call.
  An `InterruptException` is rethrown throughout. The docstrings say which
  values the exemplar decides.

## 0.1.249

- `jax_source` translates the native ODE bridges `rk_ode_rk45_tol` and
  `rk_ode_bdf_tol`, also in plate cells and through helpers such as
  BayesianRegressionModels' `ode_rk45`: one diffrax solve (`diffrax.Dopri5()`,
  and `diffrax.Kvaerno5()` for the stiff entry) with the caller's tolerances,
  the native per-output step budget and dense outputs, reverse-differentiated
  through the discrete solve so `jax.grad`'s accuracy follows the tolerances.
  Modules that solve an ODE import `diffrax`; others do not. Ragged output
  times, traced tolerances and runtime failures (invalid inputs, an exceeded
  budget, a failed solve) are handled as in the docs' "ODE solves" section.
- Without their solver packages, `rk_ode_rk45_tol` and `rk_ode_bdf_tol`
  throw an error naming the packages to load instead of a `MethodError`; their
  inferred result stays `Matrix{Float64}`, so `jax_source` translates code
  calling them without the Julia solvers loaded.
- JAX source slices a padded array that keeps each padded axis whole
  (`sol[:, j]` of a matrix padded along its rows): the slice is padded along
  those axes, where it was refused before.
- In a plain helper's lowered code, a splat of a tuple the code built
  (`g(f, x, args...)` through `Core._apply_iterate`) passes the tuple's
  elements as they were given, so a function among them stays a function
  instead of becoming a read of an untranslatable Python tuple.

## 0.1.248

- JAX source flattens a plate's cells that return vectors of one length fixed
  by the cell's own expression (`vcat(level[i], level[i] - fall[i])` of
  scalars, a vector literal, such a tuple field): `reduce(vcat, cells)` reads
  the stored rows in order (`jnp.reshape(cells, (-1,))`), with no masked
  gather or runtime check, also inside a helper. These cells used to raise
  `JAXEmissionError: flattening a ragged value whose lengths are not those of
  a prepared ragged input`, so a BRM `jax_source(...; values = (:mean_y,))`
  of such a kept value failed. Cells whose lengths come from neither a
  prepared ragged input nor the cell's own expression are still refused.

## 0.1.247

- The opt-in generated summed-plate reverse (`with_plate_reverse`) now covers a
  plate whose cells the compiler sums by contiguous index runs: a gathered
  operand whose index vector is sorted into runs of mean length at least 8
  (RKPPL's `z[group]` with rows sorted by group). The rule is generated from
  the plate's flat cell loop, and its cells are then scheduled by the same
  runs: the gathered read is taken once per run and the run's cotangent is
  added once. Such a plate used to keep the backend's reverse, reported as
  "an intermediate plate total needs a propagated adjoint". A dense shadow or
  overlapping storage keeps the per-cell loop, and an operand gathered through
  another index vector keeps its per-cell reads and writes inside the runs.
  Values and gradients are unchanged, apart from rounding.
  - On PosteriorDB's RadonVariableInterceptNCMN (919 rows in 85 counties),
    with Julia 1.13.1 and Enzyme dogfood `d37eb448`, the opted-in complete
    gradient's median paired time is 0.87 of default (seven rounds, range
    0.82–1.23; five wins) on a loaded shared host. The variability does not
    establish a consistent speedup. It allocates the one-pass buffer on each
    call (784 B), where default allocates nothing. Earlier dogfood measurements
    gave 0.58–0.67 on this model and 0.37 on a variable-slope Radon port;
    forced two-pass grouping gave 1.11 with zero allocations. Those results
    predate the backend's SIMD adjoint improvement. Schedule selection is
    unchanged.
- The Enzyme extension schedules by runs only where a rule can read the run
  metadata in its reverse pass: on Julia before 1.12, or when Enzyme reports
  dogfood repair `unchanged_custom_rule_roots` (dogfood `c036eb1`). Without
  the repair, the first gradient of such a plate that takes two passes
  segfaulted on Julia 1.13.1. There the plate keeps the backend's reverse and
  `plate_reverse_report` gives "an index-run reverse needs Enzyme dogfood
  repair `unchanged_custom_rule_roots`". The backend-only reproducer is
  `benchmark/repro_enzyme_unchanged_rule_roots.jl`.

## 0.1.246

- JAX source binds a comprehension straight to its target:
  `c = [f(k) for k in 1:K]` emits `c = jax.vmap(lambda k: ...)(jnp.arange(1, K + 1))`
  (guarded for an empty range as before). The element is a `lambda` when it
  is one expression and a nested function named after the target
  (`c_element`) when it has statements, as branch arms are. The range is bound
  only when a mask reads it, and no `element`/`indices`/`values` temporaries
  or `c = values` alias remain. Generators and `map` do-blocks share the rule.
- A spliced law or kernel operation (a `censored(Normal)` density, a
  location-scale family's endpoints) translates in place like authored code:
  its value is an expression instead of a `value_N` binding, so its one-expression
  `jax.lax.cond` arms are `lambda:` or `jnp.where`, its locals keep their
  names (`logcdf_z = jax.lax.cond(...)`), and a `let` result is not bound as
  `let_value`. An argument the body reads more than once is bound under the
  body's name for it. A prepared helper call is bound only when the helper runs
  a runtime check, so the check still runs where Julia's call does.
- A literal the authoring layer materialized under a generated port name
  (`normal(0, s)`'s location) is read as the literal (`amp != 0.0`), not bound
  as `location_binding = 0.0`.
- Values, `jax.grad`, `jax.jit` values and raised errors are unchanged; only
  local names and statement layout inside the generated functions change.

## 0.1.245

- `jax_source` translates `repeat`: `repeat(A, counts...)` (Julia's
  `outer`) and the `inner`/`outer` keywords, in Julia's axis order, counts
  past `A`'s rank adding trailing axes, and Julia's argument checks (no
  negative count; with `inner`, count lists naming every axis) raising with
  Julia's message. The JAX functions need static counts;
  `prepare_data` computes a repeat of data with counts of data, so a
  categorical model's per-term prior scales repeated once per category
  (`repeat([a, b], 1, k)`) emit one module for every category count. Before,
  the call failed with "keyword arguments to `repeat` have no JAX
  translation".

## 0.1.244

- A partition plan (`plate_partition(…; plan)`) holds the row bookkeeping of
  the windows several partition members read, also when those rows follow
  the shape of a value computed from the parameters (a product `z * F` with
  `F = transpose(tau .* L)`). That bookkeeping sits under the guard the
  members share, where a shape was not planned, so every planned call
  recomputed the reshape rows, their union, the range rows and the positions
  as fresh integer vectors and allocated what an unplanned call did. The plan
  now holds such a shape and the arm checks it against the value's shape on
  every call, as at the top level; a name the arm plans that a later member's
  guard reads is bound from the plan as well. The plan of such a partition
  therefore has other fields: a declared plan type (`plan = :port => T`) is
  the type of a plan this version's `partition_plan_kernel` computes.
  On a two-member kernel whose members read one row of that product each, a
  planned call allocates 560 instead of 1056 bytes (one cell per label) and
  592 instead of 832 (rows) on Julia 1.13.1; values and Enzyme gradients are
  unchanged.

## 0.1.243

- `jax_source` translates a derivative rule's functions and a function
  broadcast over an array when they run under a condition that depends on
  data and call a helper with recorded source (`@traceable`, `@_record_source`
  or a supplied source, each a named module-level function since 0.1.233).
  From 0.1.233 that raised "a function's active condition is outside its
  caller's scope": a censored Student-t likelihood's lower tail, for one, runs
  the incomplete-beta fraction rule under the censoring branch. These
  functions are shared by every call site, so they are now translated with no
  caller condition; one that contains a runtime check is still refused when
  called under such a condition.

## 0.1.242

- **Breaking:** `jax_source`'s `doc` keyword (kernel and `Plan` methods) no
  longer replaces the module docstring. The module docstring keeps its
  generated first line, `Generated by ReactiveKernels.jax_source; do not
  edit.`, and `doc`, the author's description of the program, follows it after
  a blank line. `doc` takes a string or a Julia docstring such as
  `@doc TOY_KERNEL`. A caller that passed `doc` to replace the generated line
  now gets both; no caller in this repository or its known consumers passed it.
- Emitted Python docstrings (module, functions, helpers and submodel regions)
  keep authored text as written: backslashes are escaped, so `\theta` is no
  longer read as a tab and `\sigma` no longer an invalid escape sequence, and
  so is a quote that would close the docstring early, such as a final `"`.

## 0.1.241

- A prepared pullback over a `NonAllocatingKernel` no longer gives wrong
  cotangents when a caller passes a value an earlier call returned back in.
  Such values are borrowed: they live in storage the preparation owns, and
  each call overwrites them. A seed, active point or argument that shares
  memory with that storage is now copied before the call. A destination of
  `ad_value_and_pullback!` that shares memory with the seed, the point,
  another argument or that storage receives the cotangent through an owned
  temporary; a returned value that overlaps it comes back as a copy. Disjoint
  calls are unchanged and take no copy.

## 0.1.240

- JAX source emits a plain authored helper (no `@traceable`, defined outside
  Julia and ReactiveKernels) as a named Python function with its method's
  argument names and Julia docstring, as recorded-source helpers already are,
  instead of inlining its lowered code. A broadcast of a named helper
  vectorizes that function itself (`jnp.vectorize(toy_mean)`) rather than a
  private positional wrapper. Helpers whose name starts with `_` and
  ReactiveKernels' own recorded helpers (the stable rewrites' `_rk_div_by_of`)
  stay private (`_div_by_of`). Lowered helper bodies compose single-use
  temporaries into expressions and bind assigned locals by their authored names.
- A named JAX function that returns one of its arguments' lengths unchanged
  (a retained tuple's field, `getfield(pair, 2)`, also `@traceable`) gives its
  caller that argument's own length expression, so a later
  `reduce(vcat, cells)` still finds the prepared flat positions.

## 0.1.239

- JAX source names only what a reader needs. A traced branch between two
  values already bound or literal is `jnp.where` (same value and derivative),
  other `jax.lax.cond` arms that are one expression are `lambda`s, and a
  condition decided while tracing is Python's own `if`, so a shape guard reads
  `if not (n == k): raise ValueError(...)` instead of two nested functions and
  a `value` binding. Consecutive statements needed by the same entries share
  one output-selection block, and checks every entry runs (the packed-length
  check) run once before the entries' returns.

## 0.1.238

- A prepared reverse Enzyme gradient whose active ports are all of the
  kernel's inputs (a tuple selector with no DI context) now writes into the
  caller's matching dense arrays and `Ref`s directly. This applies when Enzyme
  reports dogfood repair `structured_gradient_inplace`; registered Enzyme and
  older dogfood keep the copy path.
- `ad_value_and_gradient!` and `ad_value_and_pullback!` reject a mismatched
  structured destination before writing anything when they copy the
  cotangent: a tuple of another length, a named tuple with other keys, an
  array whose length differs from its cotangent's, or an unsupported
  destination. An array destination longer than its cotangent used to be
  filled only up to the cotangent's length, without error.

## 0.1.237

- JAX source computes `sortperm(v)` and `invperm(p)` of data in the host
  `prepare_data`, as it computes `sort` and `unique`. `sortperm` of a vector of
  numbers or strings orders as Julia's `isless` (-0.0 before 0.0, every NaN
  last) and keeps equal values in ascending position; `invperm` of a vector of
  integers raises `ArgumentError: argument is not a permutation` where Julia
  throws it. Keyword orders (`rev`, `by`, `lt`) still have no translation. In
  the JAX functions (a recipe that also reads a parameter) neither function
  has a translation, and the refusal now names it and says to compute it in a
  definition that reads data alone; `sortperm`'s used to name only a
  `NamedTuple`.

## 0.1.236

- `rk_cholesky_lower` stores each factor entry once, accumulating its sum in a
  scalar, instead of updating the factor in place. Values are unchanged and
  native Enzyme reverse no longer tapes the in-place reads: a per-row 3 x 3
  factor, solve and log-determinant per mixture component, differentiated
  through `prepare_sampler`, takes 465 instead of 527 microseconds per
  gradient. The returned factor's parent storage now has a zero strict upper
  triangle; it previously held `A`'s upper entries.

## 0.1.235

- Bound plates retain a shared scalar guard at runtime when its arms contain
  no work over bound inputs to prepare. The selected loop keeps shared work
  outside its cells; mixed and wholly bound arms retain their existing cached
  preparation. Changing an eligible guard or the observation count reuses the
  same emitted program.
- JAX export supports `all` and `any` with `iszero` or `!iszero` over built-in
  real numeric inputs, including empty and ragged arrays. Padding contributes
  the reduction's identity; other predicates retain the existing refusal.

## 0.1.234

- JAX emission translates `rk_cholesky_lower` with lower-triangle semantics,
  runtime square/positive-pivot checks, ordinary backend derivatives and
  empty or padded leading blocks. Lower-triangular solves, `diag(A)` and
  `diagm(v)` translate on device and in host preparation. Source loops and
  comprehensions over `axes(A, d)` retain the selected dimension instead of
  refusing matrix inputs.
- Prepared matrix-valued plate inputs retain their rank and row extents,
  including empty plates. Ragged packing pads the first axis and checks
  that trailing dimensions agree; raw `pad_ragged` accepts `rank=2` for
  an empty list of matrix cells.

## 0.1.233

- JAX source retains named, documented functions for source-backed helpers and
  shares compatible public entries through one computation body. Static entry
  selection preserves each entry's required work and checks. Helper boundaries
  carry runtime extents, ragged positions and integer bounds; helpers using scan
  history remain nested in the scan's lexical scope. Nested plate cells and
  scan steps take names from their authored values.

## 0.1.232

- The opt-in generated summed-plate reverse (`with_plate_reverse`) runs one
  pass over the cells for a plate with a gathered operand (`view(z, g)` with an
  index vector; RKPPL's `z[group]`), at least 128 cells, and fewer groups than
  cells (the operand's parent shorter than the plate). Its augmented primal
  computes the value and the unit-seed partials together, gathering the
  operand's partials per group into a buffer as long as its parent, and the
  reverse pass only scales them by the seed. Invariant terms of the cell
  density are summed once per plate rather than per cell, and a shared value's
  partial through the gathered read is one product per group. The schedule is
  chosen per call from the operands' sizes; every other plate (fewer cells, no
  gathered operand, only unit-range views such as a sampler's parameter block,
  or a parent at least as long as the plate) keeps two passes at its previous
  cost. A shadow read at more than one site on an execution path keeps per-cell
  accumulation. Gradients are unchanged up to rounding. The one-pass tape is
  one allocation per gradient (the buffer plus the scalar partials). On a
  grouped Gaussian model the rule's gradient goes from about 1.9 to about 1.45
  times a hand-written closed form at 2,000 cells, and to about 1.3 times at
  100,000 cells. The rule remains opt-in.

## 0.1.231

- JAX export keeps each retained tuple-plate field's ragged lengths tied to
  its prepared input. Flattening a projected field now drops padding and
  supports rebinding the emitted module to different row and cell counts.
  Empty tuple-valued plates skip indexed cell reads, as empty numeric plates do.

## 0.1.230

- JAX numerical functions return typed JAX arrays for constant terms and numeric
  leaves of tuple/dict results as well as computed values. Eager constant outputs
  follow `jax.default_device`, so device-placement checks also cover a zero log
  Jacobian. Host preparation and static shape/coordinate metadata remain host values.

## 0.1.229

- Partition plans retain constant row calculations used to size fresh scan
  outputs, including buffers selected by a conditional. Mutating an output no
  longer pulls its marker's row calculations into each gradient call. Aliases
  of existing storage and mutations in conditions stay in ordinary execution.

## 0.1.228

- JAX emission resolves `isa` checks when Julia argument types prove the answer,
  including the checks in lowered `oftype` calls. Plain callees can now use
  `oftype(float(x), NaN)` or `oftype(float(x), Inf)` in guarded branches.
  Partially overlapping abstract types still require an unsupported runtime
  type check; they are not folded to a false result.

## 0.1.227

- Preserve the reciprocal tail of `trigamma` in JAX emission at large positive
  inputs and return zero at positive infinity.
- Correct the JAX derivative of `besseli(n, x)` at `x = 0` for a runtime order
  `n = 0`; its exponential scale now matches the scaled function's sign
  convention at zero.

## 0.1.226

- Add `rk_log1pexp`, an owned scalar softplus callable whose stable value and
  partial share one exponential in a pure numerical graph. Its generated
  derivative adapters compose with ordinary AD and plate reductions.
- Bernoulli logit densities and the whole-vector Bernoulli GLM use the shared
  primitive. The probability-input route retains its lazy boundary handling.

## 0.1.225

- General derivative rules select their primal and active cotangents together
  before splitting forward and reverse stages. The split preserves the selected
  producers and shares their retained work; a primal requiring the unavailable
  reverse seed retains the existing independent staging. Declared recipe costs
  determine selection and do not guarantee the fastest runtime strategy.
- Enzyme's retained derivative values preserve snapshots across caller mutation,
  including views and nested containers that alias an input or returned value.
  Shared references remain shared within the snapshot. Flat scalar/array
  environments can retain direct array inputs without copying when the backend
  establishes that they remain unchanged and no mutable primal escapes.

## 0.1.224

- Bound Gaussian statistics choose between the collapsed sum and authored rows
  with a runtime guard. Changing bound values or row counts across the statistics
  threshold preserves prepared kernel, sampler and AD types when input types
  stay the same. Both paths retain reusable storage, including overlapping
  regions with intermediate plates and matrix products. Input checks keep their
  evaluation order, and declared conversions are preserved. AD preparation can
  initialize a concrete array cache step from an untaken branch without running
  that branch; later calls retain its lazy behavior and ordinary derivatives.

- Generated summed-plate reverse can be enabled when preparing AD for an
  already prepared kernel that uses native direct reads. A single plate now
  retains its boundary for AD and appears in `plate_reverse_report`, while
  ordinary native execution still inlines it. Enabling the rule during kernel
  preparation also preserves native direct reads, including when an unsupported
  derivative falls back to ordinary AD. Direct-read eligibility proves that each
  read uses the current cell index. Factored reductions with intermediate totals
  keep ordinary AD until their adjoints can be propagated; the report names that
  fallback. The generated rule remains opt-in and is not a general speedup.

## 0.1.223

- `jax_source`: a padded (ragged) row's length less a constant bounds what
  it sizes. `n = length(x)` of a row padded to width `L` has the static bound
  `L`, and `n - c` now has `L - c` (it had `L`), so a range or an array of
  length `n - c` pads to the width Julia's has when the row is longest.
  Before, `x[2:n] .- x[1:(n - 1)]` padded the two ranges to `L - 1` and `L`
  entries and failed in Python with `incompatible shapes for broadcasting`.
  Past its stop, a range to such a length repeats the stop, as padding
  repeats a row's last entry, so a gather by it pads with its own last entry
  and the differences' padding is the last valid difference.
- `jax_source` translates `diff(x)` of a vector and `diff(x; dims = d)` of an
  unpadded array along a literal axis. A padded row's differences have one
  valid entry fewer, and their padding repeats the last valid difference.
  Before, `diff` raised `JAXEmissionError` (keyword arguments to `diff` have
  no JAX translation), also in a data-only cell recipe.

## 0.1.222

- A partition of one cell per label (`plate_partition(…; single = true)`) and
  a plate computed at one row of a cell run their one cell once, checked to
  be one cell, instead of a loop over a one-cell domain; plates nested in the
  cell keep their loops. A matrix row product's entry (`_CellRowProduct`) sums
  over a counted range, and `_gather_copy(array, indices)` copies a gather
  with a counted loop (ReactiveKernelsPPL 0.1.171 gathers its compact
  overlay's constant base with it). Under Enzyme Reverse each loop entry
  allocated the caches its reverse pass reads through libc (`malloc`, and
  `realloc` per entry when the trip count is not known on entry), which a
  one-subject call paid per call. BRM's joint NLME toy's one-subject gradient
  made 79 such calls and now makes 22, about its share of the sweep's; the PK
  toy's made 6 and now none. Values and gradients are unchanged, bit for
  bit, on both toys.

## 0.1.221

- `jax_source` translates a plate over `eachrow(M)` or `eachcol(M)`: the
  plate maps the matrix's rows (or its transpose's) with `jax.vmap`, and each
  cell reads one slice, as natively. A categorical-logit response over one
  logit row per observation (ReactiveKernelsPPL 0.1.170) emits this plate.

## 0.1.220

- A guarded partition plan of a wide model is typed and computed while a
  package image is generated. 0.1.212 lifted a comprehension of a compiled
  body into a program of its own only when the locals its element reads, and
  the iteration variable, fit the 64 operands a program call takes; a model
  whose guarded bookkeeping reads more (109 and 175 in one real plan)
  kept the comprehension as a closure, and `prepare` of its partition
  (ReactiveKernelsPPL's `prepare_partition`, BayesianRegressionModels'
  `brm_nlme_model` with a grouping) still failed in a workload with
  `partition plan: the plan port … is untyped and evaluating its plan on the
  exemplars failed: Impossible to correctly handle OpaqueClosure inside
  @generated returned during precompile process.` Such an element now
  receives those locals as one tuple, which its program unpacks; plans, their
  types (also for a label without rows) and values are unchanged.

## 0.1.219

- A non-allocating program (`prepare_nonallocating`, and so ReactiveKernelsPPL's
  default sampler) keeps the tables of indexed work that nested plates reuse
  across an enclosing plate's cells (0.1.203) in buffers it owns, typed from
  the program's static types, and refills them on each call. Before, each
  density call allocated every table, and each gradient call every table and
  its derivative: the PosteriorDB capture model Mt's sampler took 6
  allocations (320 B) per density and 10 (608 B) per gradient, where it took
  2 (64 B) and 2 (96 B) without the tables; it takes 2 and 2 again (snag
  posteriordb-mt-c-01c1a87b). An ordinary prepared kernel still allocates its
  tables on each call.

## 0.1.218

- The opt-in generated summed-plate reverse (`with_plate_reverse`) is faster.
  Values and gradients are unchanged, apart from rounding.
  - A cell loop's invariant factors (the total's adjoint, literal partials,
    partials that read only shared values such as `1/σ`) apply once after the
    loop instead of in every cell.
  - Operands that span the plate are read directly rather than through
    broadcast extrusion.
  - Against 0.1.210's rule, opted-in gradients are 1.35–1.55× faster on a
    random-intercept Normal regression (2,000 and 100,000 rows) and up to
    1.2× faster on a fixed-effects one. They are now 1.08–1.28× faster than the
    default preparation on those models; 0.1.210's rule was 0.8× the default
    on the random-intercept one (Julia 1.10.12 and 1.13.1, Enzyme dogfood).
  - The default preparation is still faster on the PosteriorDB ports measured
    (the rule at 0.3–0.98× the default on Julia 1.13.1), and on small plates.
  - With registered Enzyme 0.13.219, the rule's allocating program
    (`nonallocating = false`) can raise `EnzymeMutabilityException` at an
    `.innerparm` store (reactivekernels-use §7aq); Enzyme dogfood
    differentiates it.

## 0.1.217

- `jax_source`: a `Bool` array index (`BitVector`, `Vector{Bool}`) selects
  the entries where it is true, as Julia's logical indexing does. Before, it
  was shifted as one-based positions (`x[mask - 1]`), so every element read
  the first or the last entry: `table[table[:, 2] .!= 1, 4]` in
  `prepare_data` returned the wrong amounts and length, silently whenever the
  shapes still agreed. The host uses NumPy boolean masks (lists keep their
  selected entries); device code gathers at the selected positions, padded to
  the mask's length with the count as its extent, and a masked write writes
  only the selected entries. A same-shape mask (`A[A .> 0]`) selects in
  column-major order. A single `Bool` index into an array raises
  `JAXEmissionError`, as Julia refuses it.

## 0.1.216

- The stable rewrites compute a symmetric law's standardized log kernel at a
  positive value's exponential from the value's logarithm. A half-Student or
  half-Cauchy prior on `exp(u)` (`student_t(nu, 0, s).logpdf(exp(u))`, an
  RKPPL `restricted.(StudentT.(nu, 0, s), 0, Inf)` prior on a positive
  parameter) is `-((nu + 1) / 2) * log1pexp(2(u - log(s)) - log(nu))`: no
  exponential, branch or logarithm of the scale, finite where `exp(u)`
  overflows, and in a plate its `log(nu)` and `log(s)` run once before the
  cells. A cell reads such a value through a lazy cell array of `exp.(u)`; a
  materialized one keeps its reads. A kernel opts in by declaring
  `_rk_log_abs_form` and its parameter part. A location that is not zero, a
  negative `nu` and non-real operands run the literal kernel.

## 0.1.215

- `jax_source` translates broadcasts of padded (ragged) rows padded by
  different lengths along the same axes, such as two per-row arguments of one
  plate cell broadcast into one matrix (`(vcat(x1, x_rest) .- centres) ./
  sqrt.(scales .^ 2 .+ vcat(s1, s_rest) .^ 2)`). The result is padded along
  those axes by Julia's broadcast length, the rows' common length or the
  other row's where one has length one, so `sum(cell; dims = 1)`, whole
  reductions and `vec` of it mask its padding; before, its padded axes were
  not recorded and those reductions raised; a vector broadcast of two such
  rows read the first row's length, wrong where the other row is the longer
  and the first has length one. Lengths Julia refuses to broadcast
  (`DimensionMismatch`) are not checked, as for a padded row broadcast
  against an unpadded array: a check would be a host callback in every
  evaluation. Rows padding different axes by different lengths (`x .* s'`)
  are still refused.

## 0.1.214

- `jax_source` bounds a range whose end is a prepared-data entry times a
  static integer, plus a static integer: `pk[1:2 * n[i]]` of two interleaved
  values per counted read, `1:(n[r] * 2 + 1)`, `(length(x) - 2 * n[r] + 1):length(x)`
  and `1:K * n[r]` of a prepared integer `K`. Its static number of steps is
  the same function of the array's largest entry (its least for a negative
  multiple, either where the multiple's sign is known only while tracing),
  and ends of one record with one positive multiple (`(2a - 1):(2b)`) take
  that multiple of the longest record's range. Before, a multiple lost the
  bound that `n[i] + c` keeps: an `@rkppl` cell's `v[1:2 * n[i]]` raised `a
  range whose length depends on data has no JAX translation`, and a
  `@kernel`'s `x[1:2 * n[r]]` emitted a slice JAX rejects when it traces.
  A padded row's length times a nonnegative integer literal
  (`1:2 * length(r)`) likewise keeps its static limit, scaled.

## 0.1.213

- Calling a `KernelSpec` on positional values, `spec(args...)`, prepares its
  default boundary once per graph version and keeps that kernel in the graph,
  as plain `prepare(spec; bound)` keeps its bound preparations. Later calls
  look it up, about 2 µs on top of the kernel's own call. Before, every call
  planned and lowered the kernel again: 1.0 ms per call of a five-input
  kernel whose held `prepare(spec)` took 0.36 µs. A change to the graph (such
  as `input_check!`) or another preparation setting (such as `simplify`)
  prepares afresh. A hot loop that holds `prepare(spec)` still skips the
  lookup.

## 0.1.212

- A partition plan whose row bookkeeping sits behind a condition on the
  label (cached by the plan since 0.1.170) is typed and computed while a
  package image is generated, as in a PrecompileTools workload. The plan
  body cached that bookkeeping in a comprehension, which
  `RuntimeGeneratedFunctions` compiles to an opaque closure, and Julia
  refuses to expand one while it generates an image: `prepare` of such a
  partition with an untyped plan port (ReactiveKernelsPPL's
  `prepare_partition`, BayesianRegressionModels' `brm_nlme_model` with a
  grouping) failed with `partition plan: the plan port … is untyped and
  evaluating its plan on the exemplars failed: Impossible to correctly
  handle OpaqueClosure inside @generated returned during precompile
  process.`, and a plan kernel called there failed the same way. `compile`
  now runs a comprehension or generator of a compiled body as a function of
  its own over the body's locals that its element reads, where the element
  assigns none of them; plans, their types (also for a label without rows)
  and values are unchanged.

## 0.1.211

- `jax_source` again translates a padded plate over a scan's tuple step
  outputs, such as an `@rkppl` `@plate` cell holding a stochastic `@scan`
  whose step count varies by cell. Since 0.1.199 (batched runtime checks) the
  plate's valid-cell mask read the step count and shape of the Python tuple of
  columns instead of its first column, so tracing the module raised
  `AttributeError: 'tuple' object has no attribute 'shape'`. The mask now
  spans the plate's mapped axes of the first column, as the empty-plate test
  already did.

## 0.1.210

- A summed plate whose rows are per-cell arrays runs once per distinct row
  too. 0.1.184 grouped rows whose bound operands are primitive numbers; a
  bound operand that holds one dense array of them per cell, such as one
  capture history per individual, now compares its rows by value as well. A
  capture-recapture likelihood over 237 histories of 3 occasions, 8 of them
  distinct, runs 8 cells (PosteriorDB Mt: ordinary Reverse gradient 6.8 to
  1.1 µs). Rows also compare only by the bound vectors the cell reads: a lane
  index whose bound gather preparation already cached still sets the domain
  but no longer makes every row distinct (PosteriorDB Mb, 31 distinct of 318
  histories: 16 to 1.9 µs). A gather still tells rows apart whether or not
  the cell uses its value, so every index is read as before. Values agree with
  the row-by-row sum to rounding, as before.

## 0.1.209

- `jax_source` keeps a data name that Python reserves, or that the module's
  code uses (`class`, `lambda`, `in`, `id`, `type`, `sum`, ...), at the
  module's data interface. `prepare_data`'s keywords and the dict it returns,
  the functions' `**prepared`, `DATA`, `INPUTS`, `PREPARED`, `UNTRANSLATED`
  and `JAXSource.prepared` all name it as the data does, as they already did
  every other identifier; only the Python local holding it is respelled
  (`class_`). No parameter can be named `class`, so `prepare_data` takes such
  inputs through `**data` (its signature ends `**data: ArrayLike`) and raises
  `TypeError` for a missing or unexpected one. Before, the interface used the
  respelled name (`DATA`, `INPUTS`, `PREPARED` and the parameter said `class_`
  while `UNTRANSLATED` said `class`), so `prepare_data(**raw)` with the
  binding's own names failed. A name that is no Python identifier (a
  submodel's `r.g`) is still `r_g`.
  **Breaking:** a caller that passed such an input as `class_=`, or
  positionally, passes it by its own name: `prepare_data(**raw)`.

## 0.1.208

- A bound preparation (`prepare(...; bound)`, `prepare!(cache, ...; bound)`,
  `prepare_ad(...; bound)`) evaluates a summed Gaussian observation plate over
  a bound response from least-squares statistics frozen at binding, when its
  location is linear in live values with bound coefficients: a bound matrix
  times a live vector (`X * beta`), live numbers, or live vectors gathered
  through one bound index (`alpha[g] .+ f .* beta[g]`), each times bound
  coefficients, plus bound offsets. The density and its ordinary reverse then
  cost time in the parameter count (K² for a dense design of K columns, a few
  operations per index value for gathered ones) instead of the row count.
  Values and gradients agree with the per-row plate to rounding, and are more
  accurate when the response is far from zero. A binding whose data do not fit
  (too few rows per parameter, non-finite entries) compiles today's program.
  Pointwise and per-cell queries keep the per-row plate. New keyword
  `bound_statistics` (default `true`) on every preparation entry that takes
  `simplify`, project preference `[ReactiveKernels] bound_statistics`, and
  `with_bound_statistics(f, enabled)` for the preparations `f` makes (a PPL
  sampler's, say): `bound_statistics = false` keeps the per-row plate.

## 0.1.207

- `with_plate_reverse` applies generated summed-plate reverse programs to
  non-allocating kernels after their cache activity and ownership rewrite.
  Reused primal buffers and AD-owned caches retain their existing ownership.
  `plate_reverse_report` accepts non-allocating kernels and reports static
  eligibility; backend operand annotations can still require ordinary AD.
- An authored plate's axis marker no longer prevents the generated reverse
  when the marker is used only by derivative-inert queries. Numerical alias
  reads and writes retain the conservative fallback; marker checks remain.
- The generated reverse adapter keeps single-lane calls statically dispatched
  by using separate local names in its batched-lane closure.

## 0.1.206

- `jax_source`: a branch whose condition is traced (`jax.lax.cond`) returns
  one JAX type from both arms. Julia's arm types are promoted (element by
  element for an `if` that also assigns locals), and an arm returning an
  integer known while tracing, such as a prepared integer (`StaticInt`),
  returns it as an array. Before, `s == 0 && y == m ? 1.0 : s` over a
  prepared integer `s` (a Gaussian's guarded scale in a plate cell) raised
  `cond branch outputs must have the same pytree structure`, and an
  integer arm beside a float arm raised `cond branches must have equal
  output types`.

## 0.1.205

- `@traceable` takes a trailing Vararg argument: `args...`, `args::T...`,
  `args::Vararg{T}` and `args::Vararg{T,N} where {N}`. Before, the
  `Vararg{...}` forms failed to define the method (`TypeError: in TypeVar, in
  upper bound, expected Type, got Vararg{Any, N}`), and `args...` recorded no
  source, so `jax_source` refused a body with loops. The recorded source reads
  the formal as the tuple of the trailing arguments, as the method does, in
  `jax_source`'s functions and `prepare_data` and in non-allocating
  preparation. In `jax_source`, a `where` count such as `N` reads as its
  value.
- `jax_source` translates `sum(f, t)` and `sum(f, t; init)` of a tuple `t`
  (such as those arguments, or a literal `(a, b, c)`), in the functions and
  in `prepare_data`: `f` of each element, added in order from `init`. The
  fold follows the tuple's fixed length. Before, it raised a
  `JAXEmissionError` ("this `sum` call").

## 0.1.204

- `jax_source` takes `float_type`, `Float64` (the default) or `Float32`: the
  dtype of every float value the program does not type `Float32` itself, for
  an accelerator whose float64 is slow. A `Float32` module casts its float
  inputs to `float32`, its `prepare_data` returns `float32` data, its dtype
  metadata says so, and it does not turn on JAX's 64-bit mode; its device
  integers are JAX's default integer (`int32` without that mode). An `fma`,
  which rounds once through a float64 product, still turns it on. It stays
  float32 where another module turned 64-bit mode on: integers a float is
  computed from alone (`log(k)`, `i / j`), NumPy-computed constants, `I(n)`,
  scan carries and the literal arms of traced branches take its float type.
  The `Float64` module of the same program is its reference, and is unchanged
  (snag jax-source-float-5e6b7604).
- A traced branch whose arm returns a static value (a literal, a NumPy
  constant) returns it in the branch's float dtype where that dtype is
  `float32`: `jax.lax.cond` refused a float64 literal arm beside a `float32`
  one, also in a `Float32` program's `Float64` module.

## 0.1.203

- Nested native plates reuse elementary scalar work on an invariant array
  across enclosing cells. For example, a whole-event Bernoulli mixture can
  compute `log(p[j,k])` and `log1p(-p[j,k])` once per matrix entry instead of
  once per observation. The same lowering handles logit expressions and
  invariant subexpressions inside event-dependent arithmetic. Tables and loops
  keep runtime sizes. Domain and storage checks retain the authored path for
  singular live inputs, unsupported arrays and effectful regions; empty and
  ragged domains retain their values, ordinary derivatives and access checks.

## 0.1.202

- `jax_source` generates scalar derivative-rule adapters from the same
  activity-selected mathematical graph as native AD. Joint cuts and repeated
  calls share their emitted helpers; inactive partials are not contracted.
- Add the shared incomplete-beta fraction value/partials graph used by
  distribution tails. First-order native and JAX gradients retain its partials
  instead of a reverse tape of 300 fraction iterations.

## 0.1.201

- `jax_source(...; data, definitions, definitions_module = Main)`: the
  definitions' globals, such as a call head `plate`, resolve in
  `definitions_module`. A binder passes the module it evaluates its
  definitions in, so the module it emits no longer depends on what the caller
  imported. Before, they always resolved in `Main`, so a definition calling a
  function the caller's `Main` did not see stayed a `prepare_data` input
  (`UNTRANSLATED`: "`f` is not defined in module `Main`"), and one `Main`
  defined differently was translated as that function. The default keeps the
  previous behaviour for direct callers.

## 0.1.200

- Summed-plate factoring also recognizes the direct array reads introduced
  in 0.1.194. In 0.1.196 only the broadcast-read loop was factored, so a
  shared-scale Normal plate with matching array axes retained its per-cell
  scale arithmetic, and a plate over one array was never factored. Both
  direct and broadcast-read loops now take their shared factors and terms
  out of the sum, with the same guarded fallback and ordinary reverse.

## 0.1.199

- `jax_source` combines a plate's runtime validity checks across its active
  lanes, including nested plates, before calling the host. Previously JAX
  scheduled one callback per cell, which dominated the hidden-Markov model's
  emission calculation. Inactive and padded lanes are masked, and an
  active invalid value still raises the check's error.
- A history scan whose reads have a provable bounded lag carries a rolling
  buffer and lets `lax.scan` stack its outputs separately. This includes
  previous-row recurrences with a data-sized row width, such as the HMM
  forward pass. This bounds the primal history carry; gradients of shared
  arrays can still carry full-sized buffers. Unknown history accesses keep
  the full buffer; seeds and reads of unwritten entries keep their values.
  Empty history scans skip their body, including its indexed reads.
  Prepared integer bounds survive history-index tracking, so scanned ranges
  can also bound nested loops.

## 0.1.198

- `jax_source` host preparation keeps helper function and type arguments in
  the translator instead of emitting them as invalid Python literals. This
  fixes data-only expressions such as `1 / sqrt(exp(x))`, including when `x`
  comes from a scan. Nested helpers retain their callable identities; closure
  captures and aggregate fields that hold numerical values remain runtime
  arguments. Native arithmetic and data-dependent iteration are unchanged.

## 0.1.197

- Strict plate-cell queries recognize ordinary array addition and subtraction
  through transparent helper functions on Julia 1.13. The row analysis follows
  a constant function binding loaded into a separate lowered-code statement,
  restoring helpers such as `add_arrays(a, b) = a + b`. Full-array shape checks,
  custom dispatch checks, and rejection of mutable bindings or extra helper
  work remain in effect.

## 0.1.196

- A summed plate takes shared work out of its sum (with the stable rewrites,
  `simplify = true`, the default). When a cell's value is `a + c * t`, with
  `a` and `c` reading only values its cells share, the cells sum `t` alone and
  the plate adds `n * a + c * sum(t)` once. A Normal observation plate with
  one scale (`y .~ Normal.(mu, sigma)`) now sums `(y - mu)^2` per cell and
  takes `-0.5 / sigma^2`, `log(sigma)` and `0.5 * log(2pi)` once per call;
  before, each cell divided by `sigma`, tested `sigma == 0` twice and
  subtracted the hoisted `log(sigma)`, and the ordinary reverse accumulated
  `sigma`'s derivative cell by cell, a serial chain that kept the reverse
  loop scalar (snag normal-plate-kee-8322185b). Laplace, Cauchy, logistic and
  Student-t plates with a shared scale take their additive terms out the same
  way; the Laplace scale leaves the sum too. Values and derivatives change in
  their last bits. The plate's own loop still runs, and gives its own values,
  when a law's guard on a shared value holds (`sigma == 0`), when a shared
  factor is zero, subnormal or not finite (a scale below about 1e-154 or above
  about 1e154), or when the factored sum is not finite (a residual above
  about 1e154). Where the density itself is not finite, the derivatives can
  differ: at a scale so small that a Cauchy cell's `z^2` overflows, the
  scale's derivative is NaN instead of the plate's own `-n / sigma`.
  `simplify = false` keeps the plate's own loop.

## 0.1.195

- Native plate lowering retains concrete result types through preceding
  plates and scalar aliases, including exemplar-prepared abstract inputs.
  Lazy branch arms reuse that evidence to move shared work outside the cell
  loop. In a Normal mixture with shared component scales, the outer support
  guard and scale logarithms now run once per nonempty call; the original divisions and
  ordinary reverse derivatives are preserved. Empty and inactive branches
  still skip their guarded work, and bound data retain runtime loops.

## 0.1.194

- A summed plate's cells read each batched array directly when every array
  they read has exactly the plate's axes; the plate checks this once per call
  and otherwise runs its loop as before. A plate over one batched array reads
  it directly always, since that array has the plate's axes. Base's broadcast
  read tests, at every read, whether each axis has length one, and native
  Enzyme's reverse loop kept that test in the address of every shadow load
  and store, so a summed plate's gradient did not vectorize: an `@rkppl`
  regression `mu = X * beta; y .~ Normal.(mu, sigma)` (N = 1192, K = 2) took
  1843–1932 ns per value and gradient against 770–797 ns for the hand loop,
  and now takes 778 ns (gordito, Julia 1.13.1, Enzyme 0.13.219). Values and
  gradients are unchanged, and nothing about shapes is fixed at preparation.
  The direct-read copy compiles the summed loop a second time. A cell loop
  with a nested plate or a scan, one carrying lane slots, cell tables, shared
  call tables or scheduled groups (a multi-dimensional plate), and the opt-in
  generated plate reverse keep their one loop as before (snag
  plate-reverse-lo-31fd6869).

## 0.1.193

- A partition reads a batched lane that views a vector at a range
  (`view(u, lo:hi)`, such as a packed parameter block) at the positions of that
  vector its rows hold, computed apart from the read, so a partition's plan
  computes them once per label. A compact overlay serves them from its active
  vector when its slot map is constant per label. Before, the lane was sliced
  on every call, which allocated a fresh index vector each time, also for a
  partition with no cells in that plate. Values and gradients are unchanged.

## 0.1.192

- Provide shared recurrence-window reduction planning for ReactiveKernelsPPL:
  a sum over literal backward offsets can read scalar lags without allocating
  a slice. Data-derived window widths and trajectory lengths remain runtime
  iteration.
- `jax_source` converts a numeric tuple to an array for `sum`, including
  fixed scan windows, so standalone JAX preserves their values and gradients.

## 0.1.191

- A non-allocating kernel keeps the type of a value it reads back from a step's
  buffer. A value checked by a branch whose other arm throws
  (`x = ok ? y : throw(...)`) took the type `Any` while the program was
  decomposed into buffer-filling steps, and the steps computed from it, such
  as the arms of a recorded helper's `n == 0 ? float.(s) : log.(s)`, kept
  untyped buffers, so their results reached the program as `Any`. Every plate
  cell reading them then dispatched at run time and allocated. Since 0.1.131,
  when DistributionKernels' hidden-Markov helpers record their sources,
  that included every emission cell of an exponential or normal HMM law whose
  scales are sampled: `prepare_sampler`'s default density of a two-stream
  exponential HMM over 416 steps took 60 µs and 6,673 allocations per call,
  against 15 µs and 27 for `nonallocating = false`; it now takes 14 µs and 16.
  A throwing arm no
  longer widens a branch's type, a decomposed value keeps the type the
  program's planning inferred for it when that is narrower, and a step whose
  buffer preparation still could not type (a kernel prepared without
  argument types, say) returns its result as the type its own call infers,
  while still reusing that buffer. Values are unchanged (snag
  default-non-allo-a0c23dd8).

## 0.1.190

- `prepare_nonallocating` reuses destinations for matrix products nested in
  broadcasts consumed by a reduction or plate, including materialized
  broadcast operands such as `A * (x .* s)`. These products now take the same
  cache steps as products assigned to their own statements. The outer
  broadcast keeps its existing lazy or materialized behavior, and destination
  guards still accommodate changing runtime shapes.

## 0.1.189

- `jax_source` translates `findfirst(predicate, vector)` in host data
  preparation, including captured predicates such as `isequal(label)`,
  `Base.Fix1` and `Base.Fix2`. Searches return the first one-based index or
  `nothing` and stop evaluating at the first match. Positional kernels that
  map group labels to their levels now prepare those indices in the emitted
  module. Previously unary `isequal` was incorrectly typed as `Bool`, so
  these searches emitted an unconditional `findfirst threw` error.

## 0.1.188

- The stable rewrites (0.1.117, on unless `simplify = false`) reach the
  preparations that compile their own programs. There,
  `log1pexp(-log(exp(u)))` was `Inf` at `u = -800` and its gradient `NaN` at
  `u = 710` and `800`; each now gives `prepare`'s finite value and
  derivative.
  - `plate(spec; have, want, batched)`. A recipe's source now runs in the
    generated loop, as in `lower`, and recipes that depend on the same batched
    ports recompute under one condition. A value hoisted above the loop stays
    computed once: a per-element statement reads it as it is, so `log(s)` per
    element of a shared `s = exp(u)` is computed as written there, where
    `prepare` reads `u` (computing `exp` per element instead measured 3.5×
    slower on a 10,000-element z-score loop). A composition among the hoisted
    statements, or among loop statements that depend on the same batched
    ports, is rewritten.
  - `vectorize` and `prepare_batched` (position batching). Each part of the
    position driver is rewritten; a per-position statement reads a value of
    the shared prefix as it is, as above.
  - `prepare_reactive` and `prepare_reactive_nonallocating`, for compositions
    inside one recipe's source. A composition across two state values is
    computed as written: each value is a slot that `freeze!` can set, so
    `w = log(v)` reads `v`, never what computed it. A recipe reused in place
    by `prepare_reactive_nonallocating` runs its own operation.
- New: `plate(spec; …)`, `vectorize`, `prepare_batched`, `prepare_reactive`
  and `prepare_reactive_nonallocating` take `simplify`, as `prepare` does;
  `nothing` (the default) reads an enclosing setting or the project
  preference. A `PreparedKernel` given to `vectorize` contributes its plan,
  not its setting. A rebound kernel's kept position drivers are kept per
  setting.
- The `prepare_reactive(graph; have, want)` docstring is attached to that
  method again (it documented an internal helper).

## 0.1.187

- The opt-in generated plate reverse (`with_plate_reverse`) keeps the
  backend's reverse for a plate program whose batched operand is a lazy cell
  array, such as `mu = z .+ w .* x` read by `y .~ Normal.(mu, sigma)`.
  Enzyme passes a lazy broadcast of arrays only as `Duplicated`, with a
  shadow of the same lazy structure, and the generated reverse could not add a
  cell's cotangent into it: preparing such a sampler with the rule on failed
  with `CanonicalIndexError: setindex! not defined for
  ReactiveKernels._CellBroadcast` (PosteriorDB UKDrivers). The program now
  falls back when the sampler is prepared, and `plate_reverse_report` names
  the reason, "batched operand `mu` is a lazy cell array, whose shadow the
  generated reverse does not write". The Enzyme adapter also never treats a
  lazy cell array as a covered batched operand. Gradients are unchanged
  wherever preparation succeeded before; the default preparation is
  unchanged.

## 0.1.186

- `jax_source` retains prepared integer bounds through ragged rows and scan
  elements. Loops inside a scan nested in a plate can use a scanned count,
  an indexed row entry, or a data-only cell computation as their end.
  When a computed integer needs a bound, its existing data-only recipe runs
  in `prepare_data`. Counts remain runtime data and loop bodies stay compact;
  the emitted primal supports ordinary `jax.grad`. Empty scans with numeric
  or numeric-tuple outputs skip the step, including its indexed reads.
  Scanning `eachindex(row)` or `axes(row, d)` keeps the row's valid length,
  so padded steps do not contribute to its results.

## 0.1.185

- `jax_source`: a `for` loop to a padded row's length, such as
  `for t in 1:length(r)` in a ragged plate's cell, leaves its outer arrays as
  Julia's loop does. The loop runs to the longest row's length. Before, its
  steps past the row's end changed an outer array whenever the body wrote it
  anywhere other than the row's own padding, so the value was silently
  wrong. Affected writes were at a fixed position (`y[1] += r[t]`), at the
  loop index into an array not of the row's length, inside a nested loop, or
  by rebinding the array or a tuple that holds one (`y = y .+ r[t]`). Those
  steps now keep each written entry and select a rebound array whole. A write
  at the loop index into an array of the row's length (`out[t] = f(r[t])`)
  still writes its padding, so padded values, and gradients through masked
  reductions over them, are unchanged.

## 0.1.184

- A summed plate whose rows repeat runs once per distinct row. When a plate's
  total is its only use and every operand that varies across its cells is
  bound data or a gather through a bound index (`a[group]`), binding groups
  the equal rows, and the plate evaluates one cell per group, times the
  group's row count, in the forward and the reverse pass. A Bernoulli-logit
  regression over 1179 rows of 10 distinct `(income, vote)` pairs then runs
  10 cells instead of 1179. The total agrees with the row-by-row sum to
  rounding. A plate whose rows repeat less than twice on average, a plate
  over no rows, a plate whose cells are read one by one, and every plate
  under `simplify = false` keep their own loop. A live operand whose
  declaration does not say it is a number counts as shared only when the
  preparation's `exemplars` prove it one, as the lowering infers it.

## 0.1.183

- An input check (`input_check!`) no longer slows a gradient. Enzyme treats
  the check as constant and never inlines it, so a gradient calls it as a
  compiled method. That method was not specialized on the check's arguments,
  because Julia does not specialize a vararg that is only passed on, so every
  gradient made a generic call through the boxed calling convention.
  ReactiveKernelsPPL registers a check on every program. On small
  PosteriorDB ports this made the default sampler gradient 10 to 14 ns
  (1.06x to 1.36x) slower than before the check. The check now specializes:
  on a toy hierarchical model, a checked gradient takes 58.7 ns against
  58.0 ns unchecked, where it took 65.0 against 57.6. Values, gradients and
  refusals are unchanged, and primal calls, which inlined the check already,
  are unaffected (snag `packed-point-inp-ce5df38b`).

## 0.1.182

- The opt-in generated summed-plate reverse (`with_plate_reverse`, or the
  preference `plate_reverse = "on"`) returns the backend's gradient on Julia
  1.13. Julia 1.13's `@simd` is hygienic and binds its range inside a block
  (`let; r = range; …`) where 1.10–1.12 bind it directly (`let r = range`),
  and the generated reverse recognized only the direct form. On 1.13 it found
  no cell loop in any plate program and kept each loop forward-only, so every
  plate that `plate_reverse_report` listed as `"rule"` returned a gradient
  without its cells' contributions, with no error: whole components were zero
  (snag plate-reverse-wr-8f83d36c, todo 1aput86). Julia 1.10–1.12 were not
  affected.
- A plate program holding any other loop that reads an active value now
  keeps the backend's reverse, and `plate_reverse_report` names the reason
  ("a loop over active values that is not a summed cell loop"). Before, the
  generated reverse ran such a loop forward only. On every Julia version
  this dropped the gradient of a plate whose rows come in runs of one group
  (rows sorted by a grouping index, runs of at least 8, summed run by run):
  the plate was listed as `"rule"` and its gradient had a relative error of
  about 1 (PosteriorDB RadonCounty and RadonVariableInterceptSlopeNCMN on
  Julia 1.10.12). A plain cell loop over an active array already took the
  backend's reverse when called; it now does so from preparation, and the
  report says so.

## 0.1.181

- `jax_source`: a loop or generator over no elements, such as
  `for r in 1:length(a)` or `sum(f(a[i]) for i in idx)` with empty data, runs
  no JAX loop or `vmap`: Python decides the static element count while JAX
  traces, and the loop's carry keeps its initial values. Before, JAX traced
  the body anyway and its gathers from the empty array raised `IndexError`.
  A nonempty range over an empty array still raises, as Julia's `BoundsError`.
- A step past the stop of a range whose ends are prepared-data entries (RK
  0.1.179) is a lazy branch (`jax.lax.cond`) that keeps the carry and reads
  nothing; a generator's is a zero the reduction masks. Before, such a step
  ran the body at the clamped stop and selected the old carry, so on an
  empty interval a non-finite derivative there (`log(x[t])` at an
  `x[t] = 0` Julia never reads) made `jax.grad` NaN.

## 0.1.180

- `jax_source`: `size` of a padded value, such as a ragged row in its plate
  cell, is its length along the axes its padding pads, as `length` is, and
  `size(x)` holds it too. Before, it was the padded width, the longest row's
  length, so a helper sized by `size(x, 1)` (a StanBlocks `@juliacompat`
  body's `vector[n]` argument) computed and returned that many entries with
  no length: a row's mean `sum(x) / size(x, 1)` was wrong, and a column
  derived from a shorter row was assembled with padding where Julia raises
  `DimensionMismatch`. A padded value whose padded axes are not known now
  raises for `size` instead.
- A tuple-valued plate cell's field without a length (a field computed from
  shared values, beside a padded one) has none in a later plate. Before,
  that plate read the `None` the cell's lengths hold for it as the field's
  length, so `reduce(hcat, ·)` of the field raised "the columns differ in
  length" although every column filled the width (snag
  jax-tuple-valued-1ead4c4c).
- A plate over a tuple-valued plate's values reads whether it has no cells
  from their first field. It read the values' `shape`, which their Python
  tuple does not have, so such a plate raised `AttributeError` whenever it
  mapped them first.

## 0.1.179

- `jax_source` translates loops, generators, comprehensions and slices over
  ranges whose ends are entries of integer arrays of the prepared data
  (`data`), such as a record's interval `starts[r]:stops[r]`, also as a
  plate cell's `a:b` over `starts` and `stops`, with shifted ends
  (`starts[r] + 1`) and with one static end. Before, the module refused them
  (`a loop whose first index depends on data`, or `whose trip count depends
  on data`). Such a range takes a static number of steps that `prepare_data`
  computes and returns with the data under `_rk_`-prefixed keys: the longest
  record's range when both ends are entries at one position, else the
  largest stop less the least start. Steps past a range's stop repeat its
  last element and leave the loop's values, arrays included, unchanged, so
  values (empty ranges included) are Julia's and `jax.grad` reverses the loop
  as a scan. The work is that bound for every range. A range from or to an
  element of such a range, or of a static range (`for j in i:n`), is bounded
  by that range's ends. A start that is a traced input still raises, now
  naming the prepared-data route.
- A comprehension over a range from a static start past 1 to a padded row's
  length (`[r[t] for t in 2:length(r)]` in a cell) has as many entries as
  its range; it counted from 1, so its sum and length read padding.
- A slice of a padded row up to its length (`r[1:length(r)]` in a cell) is a
  padded gather; it was refused (`slicing a padded group vector has no JAX
  translation`).

## 0.1.178

- `jax_source` translates `reduce(hcat, columns)` for a vector of numeric
  vectors or matrices, such as a plate's per-cell vectors: the matrix keeps
  Julia's orientation (column `k` is `columns[k]`; a matrix contributes its
  columns in turn), on the device and in host `prepare_data`, and
  differentiates as a transpose. Columns of different lengths raise when the
  module runs, as Julia's `DimensionMismatch` does; no columns raise while
  JAX traces, as Julia's empty `reduce` does. `stack` shares the same check,
  which now also rejects ragged rows of one length shorter than their padded
  width instead of returning the padding.
- `reduce(vcat, cells)` and `reduce(hcat, columns)` translate when a plain
  helper calls them, as they do in a kernel's own recipe; before, a helper's
  call fell through to Base's generic `reduce` and failed on `IndexStyle`.

## 0.1.177

- `jax_source` translates `fma` of `Float64` and `Float32` values, scalar and
  broadcast, in JAX and host `prepare_data`: it rounds `x * y + z` once,
  through a module helper (`_rk_fma`, `_rk_fma32`), and ordinary `jax.grad`
  gives its derivatives (`y`, `x`, `1`). JAX has no fused multiply-add, and
  XLA fuses `x * y + z` on CPU under `jax.jit` but not eagerly, so the helper
  forms the product exactly instead. Subnormal values follow the backend:
  XLA's CPU backend flushes them to zero, host NumPy keeps them. Integer `fma`
  and `muladd` are `x * y + z`, as written in source; XLA may fuse `muladd`,
  as Julia may. Before, emission failed with
  ``the function `have_fma` has no JAX translation`` (``muladd_float`` for
  `muladd`).

## 0.1.176

- `jax_source` translates immutable closures created inside plain helpers,
  fixed-argument `Base.afoldl`, terminal forward guards, and scalar `Ref(x)`
  broadcasting. Captured numerical values remain runtime inputs; array Ref
  payloads remain whole objects. The same implementation serves JAX and host
  `prepare_data`, retaining authored size checks and ordinary differentiation.

## 0.1.175

- `jax_source` preserves `Float32` in typed allocations, array literals and
  conversions, loop/history buffers and host `prepare_data` arrays. Prepared
  input metadata reports `float32` for those values. `Float64` keeps `float64`;
  `float` no longer widens `Float32` inputs.

## 0.1.174

- `jax_source(...; keep=...)` plans retained metadata from data alone, even
  when no emitted function uses it or a function chooses an inverse producer
  that reads parameters. Structural aliases share the prepared value. A
  missing host translation reports the underlying operation.
- Empty numeric plates skip their cell bodies during JAX tracing, including
  indexed reads into empty shared arrays. Fully empty ragged rows have zero
  padded width. Source remains independent of bound lengths.
- A helper with no applicable Julia method now raises a `JAXEmissionError`
  naming that call, rather than failing while inspecting the empty return
  type. The public BRM documentation reproductions and remaining capability
  boundaries are recorded under `benchmark/brm_docs_retained`.

## 0.1.173

- `jax_source` preserves the rank of `Array{T}(undef, dims)` when `dims` is a
  tuple literal, local or helper argument, and accepts multidimensional
  allocations with separate dimensions. Tuple shapes retain bounded ragged
  vector lengths. `Array{Int}` allocations keep integer elements in device code
  and host `prepare_data` instead of being treated as `Float64` buffers.

## 0.1.172

- `jax_source` emits valid Python when every input is host-prepared data.
  The generated functions take `(**prepared)` without a leading comma, so
  `prepared = prepare_data(x=...)` followed by `kernel(**prepared)` works
  even when there are no remaining positional inputs.

## 0.1.171

- Partition plans retain label-only row bookkeeping inside an outer conditional,
  including source calls with lexical `let` arguments. Preparation evaluates
  only the selected branch and stores its results; value and gradient calls
  keep the original guard and read those results. Plan types remain independent
  of branch occupancy. Active calculations, nested conditionals and writable
  scratch storage retain their ordinary execution. On a synthetic joint model,
  the one-subject gradient allocates about 28% less; paired subject calls were
  about 14% faster, or 10% after normalizing against each process's sweep.
  Values and ordinary Enzyme gradients matched the previous implementation.

## 0.1.170

- `jax_source` translates `join` and generators of values available while
  JAX traces, including joined dimension-error diagnostics in recorded and
  source-provider helpers. The emitted Python retains iteration, filters,
  Julia's product iteration order and optional final delimiters. The same
  translation supports concrete data values in host `prepare_data`.

## 0.1.169

- `jax_source`: a call of an authored kernel (a `KernelSpec`) with positional
  arguments is translated as Julia runs it, `prepare(spec)(args...)`: as that
  prepared kernel, a module-level function of its plan. A data-only
  definition calling a program-level `@kernel` (BRM's covariate anchors,
  `x_fit_mean = brm_covariate_mean(x_fit)`) is now computed by
  `prepare_data`, in a host function, so the module takes the raw data
  alone. It used to be an input of `prepare_data`, listed in `UNTRANSLATED`
  as "the function `KernelSpec(…)` has no JAX translation … or author it as
  a kernel", although it was one. A kernel with keyword arguments or
  defaults, or one called with no arguments (a graph view), raises a
  `JAXEmissionError` that says so (snag `jax-source-data-ee25db34`).
- `jax_source`: the error for a plain helper with loops or branches names
  `@traceable` as the way to translate it, besides authoring a kernel or a
  `ReactiveKernels._jax_function_call` method.

## 0.1.168

- `jax_source` translates explicit `string(...)` calls through the same path
  as string interpolation. Shape guards can build their diagnostic from
  names and dimensions in recorded or provider-supplied helpers, including
  host `prepare_data`; a failing guard retains its message. Symbol literals
  in plain lowered helpers use the same Python representation as recorded
  source.

## 0.1.167

- `jax_source` computes numerically fragile compositions in their stable
  form, as native preparation does by default (`simplify = false` keeps the
  literal arithmetic in both).
  An authored `log(exp(a))`, `log1p(exp(a))`, `log(exp(a) + exp(b))`,
  `log1p(-logistic(a))` and the rest of the stable rewrite table now translate to the
  stable forms native runs, whether the composition is one expression, spans
  statements (`v = exp(a)` followed by `log(v)`), or reaches into a lazy arm
  or a plate's cell. The module's value
  and `jax.grad` stay finite where `exp(a)` overflows: at a = 800,
  `log(exp(a))` was `Inf` with a `NaN` gradient, where native gives 800.
  The stable helpers translate from their own Julia methods, so both
  backends read one rule catalogue. `loglogistic` and `log1mlogistic`
  translate too (`jax.nn.log_sigmoid`, `-jax.nn.softplus`). One
  composition native rewrites is still translated as written: a law's term
  on a value computed outside the law (`Normal(0, s)`'s `log(s)` of
  `s = exp(a)`).
- `ReactiveKernels.@_record_source` records a method whose argument has no
  name (`f(::typeof(exp), x::Real)`, a method selected by an argument it
  never reads). It used to record nothing for such a method.
- `jax_source` translates a `@traceable` or `@_record_source` body that reads
  its method's `where` parameter (`exp(k * x)` of a `::Val{k}` argument): the
  parameter is the value the call binds. It used to raise ``k is not
  defined``.
- `jax_source` with `data`: a definition that reads its own name refines the
  raw value the caller passes under that name (`y => fill(y)`). That value
  stays a `prepare_data` parameter, taken as the caller gives it, and the
  definitions before the refinement read it. ReactiveKernelsPPL uses it for
  a response whose missing entries the caller passes as `None`.

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
