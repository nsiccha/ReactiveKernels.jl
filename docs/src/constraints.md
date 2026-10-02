# Core constraints

These constraints apply to every ReactiveKernels lowering, extension, and
example package. They also apply when data is bound during preparation.
Backend limitations must be reported explicitly; they do not relax these rules.

## Preserve data-dependent iteration

**Do not statically unroll a loop whose trip count derives from data.** This
includes runtime inputs, bound arrays, shapes, subject or observation counts,
operation tables, ragged ranges, dose counts, and capacities computed from any
of them. Knowing a length at preparation or trace time does not make it a
structural constant. For example, expanding one statement per bit of the largest
bound dose count is forbidden just as expanding one statement per subject is.

Retain a runtime loop, scan, or equivalent backend control-flow operation with
explicit carried state and output buffers. Shapes may specialize an executable;
the lowering must not duplicate the body according to those shapes or values.
Native execution retains ordinary iteration. Fixed structural algebra, such as
the scalar entries of an intrinsically three-compartment operator, is distinct
from data-derived iteration and may be expanded.

## Preserve lazy branches

**Do not replace required lazy control flow with eager `ifelse`, masked
evaluation of both branches, or equivalent predication to bypass a compiler
failure.** An inactive branch must remain inactive, including its indexing,
allocation, mutation, potentially undefined arithmetic, and derivative work.
Clamping indices or inventing dummy buffers does not make such a rewrite valid.

Ordinary elementwise selection between already valid values remains a selection;
it does not authorize evaluating an otherwise inactive computation. Preserve
the authored Julia branch and loop semantics in both primal and derivative
execution. If a backend cannot express them, report the unsupported case and
isolate the backend failure instead of adopting an eager workaround.

When a plate cell's branch condition reads bound data only, preparation
evaluates it per lane and splits the plate into one plate per taken arm (see
the [compiler](compiler.md), "Data-bound branches in plate cells"). This is
the lazy semantics made structural: each lane still evaluates only its own
arm, and no backend receives the branch.

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
DistributionKernels' `loggamma`, `logbeta`, `rk_symmetric_eigvals` and
`rk_symmetric_eigvecs` (the eigen pair re-exported by ReactiveKernelsPPL), plus
ReactiveKernelsPPL's `rk_expm` and its transit two-compartment response rule
(`prepare_transit_twocmt_rule`); the ODE backsolve
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

A lowering change must demonstrate that increasing relevant data lengths or
capacities does not replicate loop bodies or control-flow regions. Check the
generated backend structure as well as primal and AD parity with native Julia,
including inactive branches and empty or ragged cases where supported. A small
Julia statement count alone does not establish this: tracing can still expand
a host loop.

These are required constraints, not a claim that every existing path already
conforms. Every functional stateful method with authored control flow lowers
through the retained control program, and host-drained observational records
travel in its loop carry as structure-of-arrays storage written by one dynamic
slot write per call site (the [compiler](compiler.md) page describes it), so
no stateful path replicates a body per admitted iteration. The
Reactant [scan](scan.md) lowering retains one `while` loop for every
iterated-sequence shape, including bound host sequences.

The experimental rectangular PK path retains its loops and lazy branches but
still fails reverse compilation. Its eager-branch and data-derived unrolling
workarounds are not acceptable fixes. See the [scan limitations](scan.md).

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
- Reverse compilation through a retained `while` loop whose exit is data
  dependent (the adaptive ODE solver's `(n < maxiters) & (t < t1)`) fails
  because the loop has no statically known iteration count:
  `repro_reactant_adaptive_while_reverse.jl`. The solver keeps the retained
  loop; its supported gradient is the backsolve adjoint, whose right-hand
  side is a `DerivativeRule` (the rule constraint above): the augmented
  system's vector-Jacobian products are the rule's authored reverse cut,
  evaluated inside the retained loop, so nothing differentiates anything.
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
  ReactiveKernels. The periodic HSGP library's spectral weights require
  this function, so that effect supports native primal and Enzyme gradients
  but cannot compile with Reactant. Its acceptance test pins this exact
  `MethodError`; other failures remain errors. The ordinary formula stays
  intact, with no foreign-function derivative rule or tracing workaround.
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
  `repro_reactant_simplex_view.jl`. Reactant's broadcast element-type probe
  reaches that index in the existing simplex transform's `Float64.(view)`
  nest. Dirichlet priors, including live concentrations, retain native
  primal and Enzyme reverse support; compiled acceptance pins this exact
  failure until the backend fixes it. The authored transform stays intact.
- Reverse compilation with a zero-length active vector leaves `tensor.empty`,
  which Reactant 0.2.290 cannot export to XLA:
  `repro_reactant_empty_gradient.jl` isolates a constant scalar loss and its
  ordinary Enzyme gradient without ReactiveKernels. Native reverse returns
  the correct empty gradient, and compiled primal succeeds. A zero-coordinate
  RK-PPL sampler therefore supports native AD and compiled values; compiled
  gradient acceptance pins this exact export error. Empty observation and
  parameter domains with a nonempty coordinate pack pass compiled reverse.
