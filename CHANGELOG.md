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
  an index repeats. See `docs/src/compiler.md`, "Elementwise calls over a
  data-indexed gather".

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
  that runs. Measured on the ArbeitsplatzAtlas model: the log-density
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
  code that runs. On the ArbeitsplatzAtlas model the log-density kernel's
  table type shrinks from 2.58 MB to 548 KB of type text and the bound-data
  prefix's from 312 KB to 94 KB; the log density and the K=64 synthetic
  program's density and gradient are bitwise equal to the table's.
- Lowering reads an operation's fields with one method instance for every
  operation type (`_source_parameter_names`, the source module, the
  type-stability audit's recipe type query through `Base._return_type`, the
  plate and scan predicates, partial evaluation's branch reads, the gather
  predicate), and the plate loop's recipe emitter has one closure type for
  every plate (its captures travel through a `Ref{Any}` cell). Measured side by
  side on the ArbeitsplatzAtlas model, its `problem` stage (prepare, the
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
  whole. BRM-emitted NLME programs raised it on ArbeitsplatzAtlas's whole
  model. Values of other cells are unchanged.

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
