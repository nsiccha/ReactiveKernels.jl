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

## Derivative rules come from one mathematical graph, never by hand

**Do not write backend-specific derivative rules as durable code, and never
attach a rule to a function this repository does not own.** The user-ratified
policy (ReactiveKernels:reactant decision `2026-09-14T13-13-44-484-15lmrss`,
page [derivative rules](manual-derivative-rules.md) with
`examples/manual_derivative_rule.jl`) is that a numerical primitive's
derivatives are authored once as an ordinary pure-math `@kernel` graph — the
stable primal plus named partials, or forward and reverse branches as outputs
of one multi-output graph — and that every AD-protocol adapter (ChainRules,
Mooncake, Enzyme, Reactant once its upstream bridge exists) is *generated*
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
or both over array or scalar ports, into an RK-owned callable. The Enzyme,
ChainRules and Mooncake adapters (`ext/ReactiveKernelsEnzymeExt.jl`,
`ext/ReactiveKernelsChainRulesCoreExt.jl`, `ext/ReactiveKernelsMooncakeExt.jl`)
derive every direction from the activity-selected cuts of that graph. DistributionKernels' `loggamma` and
`logbeta` are the only rules in package source; the ODE backsolve adjoint
consumes a caller's `DerivativeRule` right-hand side. Reverse-mode adapters
stage each rule in two cuts whose residuals come from cross-stage liveness, so
a shared intermediate is retained rather than recomputed. Reactant rule
emission (gated on the upstream EnzymeMLIR custom-rule bridge; rule cuts
already trace as plain mathematics) remains ReactiveKernels:review todo
`2026-09-14T17-10-05-750-0dsqq02`.

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

Three further backend limitations are isolated with standalone reproducers
under `benchmark/` (the backend and its AD engine only, no ReactiveKernels
code):

- A lazy branch inside a batched plate cell compiles and evaluates for every
  lane count, but reverse compilation through it fails once the batching pass
  realizes the plate as a loop (six lanes fail where four lanes, unrolled per
  lane, succeed): `repro_reactant_batch_if_reverse.jl`. Plated support guards
  therefore keep their authored branch and lose Reactant reverse gradients
  above that size until the batched-loop branch lowers upstream; native
  execution and native reverse are unaffected.
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
