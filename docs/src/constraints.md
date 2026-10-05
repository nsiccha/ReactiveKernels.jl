# Core constraints

These constraints apply to every ReactiveKernels lowering, extension, and
example package. They also apply when data is bound during preparation.
Backend limitations must be reported explicitly; they do not relax these rules.

## Place implementations in their owning layer

Totally general numerical and compiler machinery belongs in ReactiveKernels
proper. General PPL authoring, binding and lowering belongs in ReactiveKernelsPPL.
Reusable statistical model construction and backend emission belongs in BRM;
its statistical implementations serve both backends. PK-specific models and
helpers belong only in downstream RKPPLBench. StanBlocks owns transpilation to
Stan. Existing implementations in the wrong layer must migrate; historical
placement and sharing a repository are not exceptions. Temporary compatibility
must have a concrete migration task and a consumer-adoption condition.

## Preserve data-dependent iteration

**Do not statically unroll a loop whose trip count derives from data.** This
includes runtime inputs, bound arrays, shapes, subject or observation counts,
operation tables, ragged ranges, dose counts, and capacities computed from any
of them. Knowing a length at preparation or trace time does not make it a
structural constant. For example, expanding one statement per bit of the largest
bound dose count is forbidden just as expanding one statement per subject is.

The primary instruction is the user's September 22 decision `1rvu25u`, recorded
under `ReactiveKernels:brm:tgi:reactant:no-unroll`, decision
`2026-09-22T08-49-18-901-1rvu25u`: retain real loops or batched-array structure.
Retain a runtime loop, scan, or equivalent batched array structure. Stateful
iteration needs the appropriate carried state and output storage. Shapes may
specialize an executable; the lowering must not duplicate the body according
to those shapes or values. Native execution retains ordinary iteration.
Fixed structural algebra, such as the scalar entries of an intrinsically
three-compartment operator, is distinct
from data-derived iteration and may be expanded.

A surviving loop alone does not establish that it implements the intended
work. Inspect its reachable body and callees, and any equivalent array work,
alongside selected values, ordinary derivatives and growth across data sizes.
Semantics-preserving movement of pure invariant arithmetic is permitted;
operation counts are evidence to interpret, not an independent user policy.

## Preserve lazy branches

The primary instruction is the user's September 22 reply to the eager-workaround
brief `15coonh`, which expressly allowed safe intended elementwise selection.
Its internal source record is `ReactiveKernels:brm:tgi:reactant:no-unroll`,
message `2026-09-22T08:54:55+02:00`, brief
`2026-09-22T08-50-05-258-15coonh`. The user clarified on October 4
(`ReactiveKernels:snag.reactant-xla-laz-3679d90e:upstream`, message
`2026-10-04T10:13:56+02:00`)
that they had not requested a blanket ban on inactive pure work. The broader
execution-policy wording introduced in `3c7af187` was an agent interpretation,
not an additional user instruction.

**Do not replace required lazy control flow with eager `ifelse`, masked
evaluation of both branches, or equivalent predication to bypass a compiler
failure.** Preserve selected values, ordinary derivatives, observable effects,
exceptions and invalid-access safety. An eager rewrite must not expose an
inactive invalid access, side effect or unsafe arithmetic. Clamping indices or
inventing dummy buffers does not make such a rewrite valid.

Ordinary elementwise selection between already valid values remains a selection;
it does not authorize an unsafe rewrite of arbitrary control flow. Keep required
guards in the authored graph and preserve Julia semantics in primal and
derivative execution. A backend may simplify or speculate pure arithmetic when
it preserves those semantics: a discarded floating-point `NaN` or `Inf` is not
itself a correctness failure if it cannot affect the selected result, gradient
or observable behavior. This rule does not require a particular conditional
instruction or prohibit every computation of an unused pure intermediate.
If a backend cannot preserve the required semantics, report the unsupported
case and isolate the failure instead of adopting an eager workaround.

When a plate cell's branch condition reads bound data only, preparation
evaluates it per lane and splits the plate into one plate per taken arm (see
the [compiler](compiler.md), "Data-bound branches in plate cells"). This is
the lazy semantics made structural: each lane still evaluates only its own
arm, and no backend receives the branch.

The user's October 4 recurrence-prevention request (the same upstream record,
message `2026-10-04T10:26:24+02:00`) and broader audit request (`RKPPLBench`,
message `2026-10-04T10:55:23+02:00`)
are implemented by this review check: before adding or strengthening a constraint,
reviewers must identify the primary user instruction or demonstrate its semantic
or technical necessity, verify the proposed rule's scope, and distinguish
observable semantics from explicitly requested structure and performance or
resource measurements. Copied docs, `AGENTS.md`, repeated summaries, successful
assertions, brief acknowledgements and experiment-publication approval are not
independent authority for an expanded policy. A new execution-policy requirement
needs its own user direction or demonstrated semantic or technical necessity
with a clear scope; a diagnostic finding alone does not supply it.

## Derivative rules come from one mathematical graph, never by hand

**Do not write backend-specific derivative rules as durable code, and never
attach a rule to a function this repository does not own.** The user-ratified
policy (ReactiveKernels:reactant decision `2026-09-14T13-13-44-484-15lmrss`,
page [derivative rules](manual-derivative-rules.md) with
`examples/manual_derivative_rule.jl`) is that a numerical primitive's
derivatives are authored once as an ordinary pure-math `@kernel` graph — the
stable primal plus named partials, or forward and reverse branches as outputs
of one multi-output graph — and that every AD-protocol adapter (ChainRules,
Enzyme, Reactant once its upstream bridge exists) is *generated*
from activity-selected cuts of that graph and attached to a callable this
repository owns. Separately authored JVP/VJP declarations, hand-written
`EnzymeRules`/ChainRules methods, function or runtime-activity annotations,
and finite-difference substitutions are rejected as durable code; the
acceptance corpus differentiates the authored primal with the backend's
ordinary reverse mode and nothing else. A rule on a foreign function such as
`SpecialFunctions.loggamma` is type piracy on top of that: it silently changes
every Enzyme user in the session.

The generator is shipped (`src/derivative_rules.jl`): `scalar_derivative_rule`
turns a scalar graph that authors the primal plus one named partial per input,
and `derivative_rule` a graph that authors a forward branch, a reverse branch,
or both over array or scalar ports, into an RK-owned callable. The Enzyme and
ChainRules adapters (`ext/ReactiveKernelsEnzymeExt.jl`,
`ext/ReactiveKernelsChainRulesCoreExt.jl`)
derive every direction from the activity-selected cuts of that graph. The rules in package source are
ReactiveKernels' `rk_expm`, `rk_symmetric_eigvals` and `rk_symmetric_eigvecs`,
plus DistributionKernels' `loggamma` and `logbeta`. The generic matrix rules
are also imported by existing PPL/distribution consumers. The ODE backsolve
adjoint consumes a caller's `DerivativeRule` right-hand side. Reverse-mode adapters
stage each rule in two cuts whose residuals come from cross-stage liveness, so
a shared intermediate is retained rather than recomputed. Rule cuts already
trace under Reactant as plain mathematics. Emitting them as EnzymeMLIR custom
rules is gated upstream on
[EnzymeAD/Enzyme#2516](https://github.com/EnzymeAD/Enzyme/pull/2516), which
no Reactant release carries yet. ReactiveKernels generates that adapter once a
release carries the mechanism.

A new backend failure of the ordinary path remains a backend limitation:
isolate it with a backend-only reproducer under
`benchmark/`, record it on this page, and, if it aborts the process, skip the
affected acceptance cases by name until either the backend lowers the shape or
a generated rule on an owned callable covers it.

## Acceptance and existing limitations

Reactant 0.2.290 rejects a nested `Float64` broadcast over a packed view
with a `SubArray` reindexing error. The simplex transform reads an ordinary
slice of its already `Float64` packed port instead, with identical values
and ordinary reverse-mode derivatives. The backend-only reproducer is
`benchmark/reactant_subarray_broadcast_reindex.jl`; prior-only and shared
simplex acceptance is in `test_capability_scan_priors_reactant.jl`.

A lowering change must demonstrate that increasing relevant data lengths or
capacities does not replicate loop bodies or control-flow regions. Check the
generated backend structure as well as primal and AD parity with native Julia,
including inactive branches and empty or ragged cases where supported. A small
Julia statement count alone does not establish this: tracing can still expand
a host loop.

Interpret acceptance evidence according to what it establishes:

| Evidence | Required interpretation |
| --- | --- |
| Changed-input values, ordinary derivatives, observable effects, ownership and invalid-access safety | Semantic correctness checks; do not relax them to accommodate a compiler. |
| Retained data-derived iteration and bounded body growth | Separately requested structural invariants; pure-branch speculation does not waive them. |
| Complete inventories and missing HLO conditionals in pure-arithmetic guards | Structural diagnostics; a missing conditional alone is not a correctness failure without a relevant requirement or observable impact. |
| Runtime, allocations and data movement | Workload-specific performance or resource evidence; state the workload, direction, comparison and measurement conditions. |

Compare retained control-flow regions, nonlinear work and indexing across all
tested data sizes, and keep complete optimized operation inventories as
diagnostics. Shape specialization may share constants, simplify scalar
arithmetic or simplify singleton derivative tapes, including an identity
broadcast of one scalar tape index, so small shapes need not
have identical raw inventories. The bounded-inventory criterion `1arw7kp` was
an agent engineering choice following the user's scoped no-preference
delegations `01pwx0r` and `08vkgt9`, not a user-authored specification. Review
that criterion with evidence and its actual scope; neither exact count equality
nor eventual complete-count equality follows from `1rvu25u` alone. Record
growth and identify replicated data-derived bodies, reduction stages and
scalar simplifications separately. Pure branch simplification is subject to
the semantic requirements above, rather than a blanket conditional-instruction
count.

For compiled acceptance inspect both optimized MLIR and the HLO of the actual
default executable. XLA may remove singleton loops, simplify pure branches or
move invariant work after MLIR checks pass. Verify the separate retained
data-derived iteration requirement and operation-growth diagnostics. For
branches, check selected values, ordinary derivatives, effects and safety at
changed valid and invalid inputs; pure speculation with preserved semantics is
not a failing branch check merely because the executable uses a selection.

Default XLA vector reductions can add reduction stages as array lengths grow
even when optimized MLIR retains one vector-reduction expression. This separate
complete-inventory growth diagnostic is isolated by
`benchmark/repro_reactant_vector_reduction_growth.jl`. Reduction-stage growth
alone does not establish replicated source bodies or a performance regression;
inspect the work and measure any relevant resource or runtime impact.

These are required constraints, not a claim that every existing path already
conforms. Every functional stateful method with authored control flow lowers
through the retained control program, and host-drained observational records
travel in its loop carry as structure-of-arrays storage written by one dynamic
slot write per call site (the [compiler](compiler.md) page describes it), so
no stateful path replicates a body per admitted iteration. The
Reactant [scan](scan.md) lowering retains one `while` loop for every
iterated-sequence shape, including bound host sequences.

Grouped PK recurrences expose a subject plate containing retained event scans.
Their fixed-size matrix and named carry intermediates batch as typed leaves,
with their authored wrappers restored inside each cell. Their full Reactant
path still fails in the ordinary StaticArrays matrix exponential: its branch
condition is a traced Boolean. This is a dependency capability boundary,
isolated by `benchmark/repro_reactant_static_matrix_exp.jl` and tracked in
[issue #34](https://github.com/nsiccha/ReactiveKernels.jl/issues/34).
Eager branches, parameter-dependent host propagation, and data-derived
unrolling are not acceptable fixes. See the [scan limitations](scan.md).

A generator reduction inside a traced body, such as a sum over the doses of a
schedule, lowers to one retained loop when it has an `init` and iterates a
data-length iterator (`eachindex(x)`, `axes(x, d)`, `a:b`); the emitted
program is then the same for three doses and for six. A generator over a
helper-produced iterator, and a sum without `init`, are still traced once per
element, and so is a loop whose scope rebinds a name `@trace for` resolves
where it expands; these do not conform to the first rule. Under the retained
loop the reduction index is a traced value: indexing written in the term is
lowered by RK, host tables included, with one index per dimension (`W[i, j]`
at two traced indices is one gather). A helper function the term calls with
the index receives the traced value; its body gets the same lowering when the
helper is defined with `@traceable`, and an ordinary helper's read fails with
`Scalar indexing is disallowed`. A per-type rule (one method per schedule-plan
type) therefore keeps its dispatch and the retained loop. More generally a
helper may carry a `ReactiveKernels.traced` method, a separate tracing
implementation the kernel calls in its place; it must itself satisfy these
constraints.

Standalone reproducers under `benchmark/` (the backend and its AD engine
only, no ReactiveKernels code) isolate the remaining backend limitations —
and lock the one Reactant 0.2.289 lifted:

- A lazy branch inside a batched plate cell compiles and evaluates for every
  lane count, and since Reactant 0.2.289 reverse compilation through it
  lowers too, with compiled gradients matching native Enzyme:
  `repro_reactant_batch_if_reverse.jl` passes exactly and now guards the lift
  as a regression test, and RK issue #13 closed with its retained-loop
  reproducer passing exactly. Before 0.2.289 reverse failed once the batching
  pass realized the plate as a loop (six lanes failed where four lanes,
  unrolled per lane, succeeded). The boundary concerned only branches whose
  condition reads a live value: a condition on bound data is split away
  during preparation and never reaches the backend.
  A plate branch whose condition reads only shared scalar or `Ref` operands
  now lowers to one lazy decision around the batch, using its authored branch
  dependency metadata. Its selected arm retains the original lane domain,
  including constant fallbacks and otherwise unused lane inputs. Ordinary
  MLIR primal and reverse have matching complete operation inventories
  at three, seven and eleven lanes. Default CPU XLA on Reactant 0.2.290 can
  replace these pure shared guards with selections and speculate floating-point
  logarithms/division. The tested selected values and ordinary gradients still
  agree with native Julia, including negative and zero scales. This is a
  compiler-policy observation, not by itself a correctness defect.
  `repro_reactant_pure_lazy_guard.jl` isolates it without ReactiveKernels;
  `test_invariant_plate_branches_reactant.jl` keeps numerical, ownership and
  size-growth acceptance, with executable inventories as diagnostics.
  `probe_reactant_late_branch_barrier.jl` preserves additional executable
  branch regions as an experiment using private compiler APIs. It establishes
  no requirement to adopt that stronger execution policy, create a strict mode
  or change ordinary defaults. A barrier before AD instead fails because its
  backend adjoint is missing. Required branch effects and invalid-access safety
  still need their own acceptance; pure-arithmetic parity does not certify them.
  Lane-dependent conditions still reach the backend's small-batch boundary.
  Default optimized reverse expands those small lazy batches into one branch
  region per lane instead of retaining the batch loop:
  `repro_reactant_lazy_batch_growth.jl` has correct values and gradients at
  two and five lanes, but different operation inventories. The fixed-structure
  requirement remains unmet for that shape; changing optimizer flags is not
  acceptance of the ordinary path.
- Reverse compilation through a retained `while` loop whose exit is data
  dependent (the adaptive ODE solver's `(n < maxiters) & (t < t1)`) fails
  because the loop has no statically known iteration count:
  `repro_reactant_adaptive_while_reverse.jl`. Generic matrix binary power
  shape has the same default reverse failure on Reactant 0.2.290:
  `repro_reactant_matrix_power_reverse.jl` checks its native and compiled primal,
  retained integer loop and lazy branch, and native ordinary Enzyme gradient.
  Compiled reverse remains unsupported for that shape. The solver keeps the
  retained loop; its supported gradient is the backsolve adjoint, whose right-hand
  side is a `DerivativeRule` (the rule constraint above): the augmented
  system's vector-Jacobian products are the rule's authored reverse cut,
  evaluated inside the retained loop, so nothing differentiates anything.
- A live branch with an out-of-bounds read of a host constant in its inactive
  arm fails while Reactant traces that arm, before compilation:
  `repro_reactant_inactive_constant_index.jl`. Native primal and Enzyme AD
  retain the branch. Data-bound conditions split away the inactive arm, and
  live branches around traced undefined arithmetic retain compiled primal
  and AD parity. The invalid host-constant indexing shape remains unsupported;
  its named acceptance case is excluded from compiled parity, with the exact
  `BoundsError` pinned. No index clamping or dummy buffer is introduced.
- Reverse compilation through a guarded diagonal reduction can fail with an
  MLIR dominance error when its matrix also feeds shared response expressions:
  `repro_reactant_shared_matrix_loop_reverse.jl` reproduces this on Reactant
  0.2.290 with plain matrix products, weighted gathers and a retained diagonal
  loop. PPL constructs the diagonal as one graph value, reuses those entries
  in the factor, and reads that value in its guarded prior. The named
  `LKJCholesky(size(M, 1), eta)` shared-response case now passes ordinary
  compiled reverse at K = 2, 4, 8 and 16, including reused coordinates and
  invalid inactive logarithms. Its emitted transform and prior retain loops;
  default optimized primal and reverse have identical operation inventories
  at K = 8 and 16 as observation and group counts grow. The backend still
  expands small data-derived K = 2 and 4, so full fixed-structure acceptance
  remains unmet. `repro_reactant_guarded_diagonal_growth.jl` isolates this
  optimizer boundary with the backend alone. No flag change is acceptance.
  A declared literal two-dimensional factor keeps its one scalar diagonal
  prior equation.
- Native Enzyme reverse mode aborts the process (an LLVM assertion in its
  shadow-allocation caching, reached while it differentiates SpecialFunctions'
  `logabsgamma` port) when lazily evaluated branches around
  `loggamma`/`logbeta` sit in non-inlined functions differentiated together,
  one inside a loop: `repro_enzyme_lgamma_branch.jl`. That is the shape of a
  guarded `logpdf` plus an observation plate. The lazy guards stay; the
  distribution sources call DistributionKernels' own `loggamma`/`logbeta`,
  rules generated from their pure-math graphs (the rule constraint above),
  which makes them primitives for Enzyme, so the failing body is never
  differentiated.
- Native Enzyme 0.13.209 reverse differentiation of
  `LogExpFunctions.logistic` returns `NaN` at `1000.0` and `1000.0f0`,
  despite finite primal values. It also loses representable tail derivatives,
  such as the Float64 derivative at `40.0`. This reproduces without
  ReactiveKernels on Julia 1.10.12 with LogExpFunctions 0.3.29 and 1.0.1:
  `repro_enzyme_logistic_reverse.jl`. The equivalent ordinary primal
  `exp(-log1pexp(-x))` passes native reverse checks against a high-precision
  oracle in both precisions, including `-1000`, `0`, `1000` and saturated
  tails. This is native evidence; it does not establish compiled acceptance.
  [Enzyme issue #3583](https://github.com/EnzymeAD/Enzyme.jl/issues/3583)
  and [PR #3595](https://github.com/EnzymeAD/Enzyme.jl/pull/3595) track the
  dependency boundary. No local rule is attached to the foreign function.
- Native Enzyme reverse mode fails static activity analysis
  (`EnzymeRuntimeActivityError`) when a non-inlined function returns a
  `Float64` array read from constant data, bare or inside a tuple, named
  tuple or struct, although its arguments are all constant. A non-inlined
  identity on a constant named tuple raises nothing and instead accumulates
  the adjoint into the constant data:
  `repro_enzyme_noinline_const_aggregate_return.jl`. A guard whose error
  message interpolates a value is enough to keep a helper from inlining. That
  is the shape of a module function that unwraps a bound schedule column
  inside a parameter-dependent `@rkppl` call. The PPL generator emits a
  data-only call that a parameter-dependent call consumes as its own
  statement, so preparation folds it once and the gradient never
  differentiates it. A bound tuple or named tuple then crosses the
  differentiated call as one operand per array leaf. Helpers inside a
  parameter-dependent function remain the backend's limitation.
- Native Enzyme 0.13.209 reverse mode fails static activity analysis
  (`EnzymeRuntimeActivityError`) when buffers allocated in the two arms of a
  branch meet in one value, the empty arm's buffer is never written with
  active data, and Base `sum` reads the merged value:
  `repro_enzyme_branch_allocation_phi.jl` reproduces it with Enzyme only. One
  allocation after a branch that peels the first step to type it still fails,
  because the optimizer splits that allocation back into the arms. This was
  the shape of an empty authored scan summed by a recipe or plate cell. When
  the step's inferred output type is concrete, the native scan lowering now
  allocates each buffer once, before its emptiness branch; values, element
  types and the retained loop are unchanged, and ordinary reverse passes for
  empty and nonempty sequences. A step without a concrete inferred output type
  keeps per-arm allocation. No activity configuration or derivative rule is
  involved.
- Native Enzyme 0.13.209 on Julia 1.10.12 also rejects an ordinary untyped
  comprehension whose generator captures both an active array and a constant
  floating-point array: `repro_enzyme_generator_const_array_capture.jl`
  reproduces the failure with Enzyme only. Typed Julia IR constructs the mixed
  closure and calls `Base.collect(::Generator)`; the rejected store is the
  constant array pointer into that closure. This is distinct from returning
  data through a non-inlined helper. Merely inlining the reader, replacing
  `eachindex(idx)` with dynamic `1:length(idx)`, or using `map`/`zip` still
  fails. Explicitly typed comprehensions, an explicit generator with a
  call-site-inlined `collect`, and a fresh-buffer runtime loop pass ordinary
  reverse for Float32 and Float64, including empty inputs. A typed
  comprehension performs Julia's ordinary element conversion, so it is only
  an equivalent control when that element type is intended. PPL's ordinary
  and array-bearing prepared-input readers keep their authored primal and
  remain native reverse capability gaps in `test_native_generator_capture.jl`.
  No consumer function, activity configuration or derivative rule is replaced.
  A six-line isolated Enzyme compiler prototype marks
  `Base.collect(::Generator)` for inlining before activity analysis and makes
  the byte-unchanged ordinary and prepared readers pass standard reverse,
  with runtime activity disabled. That prototype is not in released Enzyme
  and still needs broader compiler validation. The repair belongs in the
  backend compiler; related
  [Enzyme issue #2386](https://github.com/EnzymeAD/Enzyme.jl/issues/2386)
  tracks comprehension activity analysis. This evidence and boundary note are
  interim tracking, not completion of that capability.
- Two backend rewrite patterns, `reshape_dynamic_slice` and `reshape_dus`,
  never finish on a reshape that inserts a unit dimension ahead of a dropped
  one: each creates a constant for the inserted dimension, fails a later
  check, and leaves MLIR's greedy rewriting without a fixed point.
  `repro_reactant_reshape_slice_rewrite.jl` runs each pattern alone on at most
  four operations, and `repro_reactant_passthrough_loop.jl` reaches the first
  through the default pipeline; both hold on Reactant 0.2.284 and 0.2.289. A
  slice along a trailing axis at a traced index, copied into an output buffer,
  produces that reshape. Slot columns therefore slice and update in storage
  layout, where the slot axis leads and both patterns fail before creating
  anything: the retained position loop, a recipe-free structured passthrough
  and an embedded plate under `prepare_batched` compile with the default
  optimizer. The loop and the scalar semantics are retained.
- Eigendecomposition of a traced matrix does not lower, at several stacked
  layers: `eigen(Symmetric(A))` dies during tracing in `isdiag` →
  `overloaded_triu(::UpperTriangular)` (Reactant defines it for
  `TracedRArray{T, 2}` only; the same missing method is upstream
  EnzymeAD/Reactant.jl#3369 via symmetric solve), while nonsymmetric
  `eigen(A)` dies branching on a traced `Bool` and `eigvals(Symmetric(A))`
  has no traced method at all: `repro_reactant_eigen_symmetric.jl`. That is
  the shape of the posteriordb `kronecker_gp` example's exact
  Kronecker-eigenspace marginal likelihood: its owned
  `rk_symmetric_eigvals`/`rk_symmetric_eigvecs` margins call
  `eigen(Symmetric(·))` as their primal and fail the same way.
- Arbitrary-order `SpecialFunctions.besselix(order, x)` has no method for
  a traced scalar `x` in Reactant 0.2.289:
  `repro_reactant_besselix_order.jl` isolates the missing method without
  ReactiveKernels. BRM's periodic HSGP spectral weights require this
  function. The producer-only reproducer preserves the dependency failure
  after statistical model tests move to BRM. The ordinary formula stays
  intact, with no foreign-function derivative rule or tracing workaround.
- Evidence normalizers that call `SpecialFunctions.gamma_inc` or `beta_inc`
  have no traced scalar method in Reactant 0.2.290:
  `repro_specialfunctions_evidence.jl` isolates both calls (and `besselix`)
  without ReactiveKernels. This affects compiled Gamma/Beta, Poisson/Binomial,
  negative-binomial and zero-inflated/hurdle evidence; their native values and
  Enzyme gradients work. VonMises evidence reaches the Bessel limitation
  above. The PPL acceptance pins each exact missing-function `MethodError`,
  rather than refusing those families or accepting unrelated exceptions.
- The default slice optimizer aborts reverse compilation of chained strided
  gathers with a mismatched `stablehlo.add` shape:
  `repro_reactant_partition_gather_reverse.jl` reproduces it with Reactant and
  Enzyme only. Mixed clamp arms of LogNormal and Weibull evidence reach this
  shape. Their compiled primal works, and compiled reverse matches native
  gradients at 6, 12 and 24 rows with the explicit `optimize=:only_enzyme`
  pipeline, which retains the trace and reverse pass. The lazy arms and
  observation loops stay intact; this is an optimizer limitation.
- A retained `@trace` loop updates its input tracer handles even when the
  authored inputs are read-only. If a lazy thunk captures those handles, they
  can escape into a child region and tracing fails with an operand-dominance
  error: `repro_reactant_captured_loop_branch.jl borrowed` isolates the failure
  with Reactant and Enzyme only. The `copied` mode passes primal and reverse
  with the same lazy branch and retained loop. Owned evidence-tail helpers
  enter with fresh handles through `_loop_capture_traced`; native values pass
  through unchanged, and no branch or iteration is expanded or predicated.
- Mixed beta-binomial clamp arms support native values and reverse, compiled
  primal, and compiled reverse with RK's explicit `optimize=:no_slice_slice`
  pipeline. That pipeline removes the faulty slice-combination pass and
  count-loop unrolling, preserving retained data-derived loops. Default
  compiled reverse reaches the chained-slice abort above; `only_enzyme`
  instead fails with `WhileOp does not have induction
  variable for cache removal`. `repro_reactant_batched_count_reverse.jl`
  isolates a retained count loop in a batched cell with Reactant and Enzyme
  only; its `only_enzyme` reverse segfaults in `LoopCheckpointing`. The clamp
  reverse acceptance selects the verified pipeline explicitly. Beta-binomial
  truncation and interval evidence retain their loops and pass compiled
  reverse and fixed-operation-count acceptance at 6, 12 and 24 observations.
- Linear indexing of an adjoint vector (`b = (z * sd)'`, then `b[g]`)
  fails during Reactant 0.2.290 tracing: the adjoint converts the integer
  positions to two-dimensional Cartesian indices, which the backend applies
  to its underlying one-dimensional `LinearIndices`:
  `repro_reactant_adjoint_linear_gather.jl`. Ordinary matrix linear indexing
  compiles in the same standalone reproducer. RKPPL keeps ordinary Julia
  indexing; this row gather supports native primal and Enzyme reverse but is
  excluded by name from compiled parity and structure acceptance. Linear
  matrix gathers and positional gathers from module-produced arrays pass
  compiled primal, reverse and fixed-operation-count checks.
- Gathering from an adjoint matrix (`reshape(u, 2, 3)'[g, 1]`) also
  fails during Reactant 0.2.289 tracing: the backend applies the adjoint's
  Cartesian indices to the untransposed ancestor's `LinearIndices`.
  `repro_reactant_adjoint_axis_gather.jl` isolates the failure and a
  passing second-axis gather from the plain reshaped matrix. RKPPL
  preserves the authored adjoint; this shape passes native primal and
  Enzyme reverse but is excluded by name from compiled acceptance.
  Declared second-axis gathers, elementwise array definitions and level
  subsets pass compiled primal, AD and operation-count checks.

- A one-dimensional traced view indexed by `CartesianIndex{1}` fails in
  Reactant 0.2.290 because `Base.reindex` expects an index tuple:
  `repro_reactant_simplex_view.jl`. RKPPL's simplex transform now uses
  ordinary range indexing to materialize its packed-coordinate and
  Jacobian slices before fused broadcasts. Its stick-breaking arithmetic
  is unchanged; hierarchical Dirichlet and monotonic values pass compiled
  primal, reverse and operation-count acceptance.
- Ordinary vector cutpoints also materialize the packed range before the
  cumulative-order validity reduction, avoiding the same traced-view
  reindexing boundary. Their element priors and identity transform are
  unchanged; invalid order takes a lazy `-Inf` branch.
- Reverse compilation with an empty active array leaves `tensor.empty`,
  which Reactant 0.2.290 cannot export to XLA:
  `repro_reactant_empty_gradient.jl` isolates a constant scalar loss and its
  ordinary Enzyme gradient, plus an empty multivariate batch beside a nonempty
  factor, without ReactiveKernels. Native reverse returns the correct empty/zero
  gradients, and compiled primal succeeds. Empty multivariate slice batches
  return zero natively and in compiled primal execution. A zero-coordinate
  RK-PPL sampler therefore supports native AD and compiled values; compiled
  gradient acceptance pins this exact export error. Empty observation and
  parameter domains with a nonempty coordinate pack pass compiled reverse.
  No dummy batch or handwritten derivative substitutes for the empty case.
- An opaque Julia function's ordinary `hcat` of a live scalar and a constant one-entry vector fails
  during Reactant 0.2.290 primal tracing with `Scalar indexing is disallowed`
  in `Base.typed_hcat`: `repro_reactant_scalar_hcat.jl` isolates it without
  ReactiveKernels. Native values and ordinary Enzyme reverse match independent
  algebra; an equivalent scalar-formula control compiles primal and reverse.
  Compiler-visible calls resolve to RK's ordinary concatenation lowering,
  including imported, qualified, aliased and `GlobalRef` bindings. RKPPL's
  one-row `sampled_scalar` matrix case keeps its authored column and passes
  default compiled values and reverse. The opaque function remains the backend
  boundary; no scalar-indexing override or derivative rule replaces its body.
