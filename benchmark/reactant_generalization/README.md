# General Reactant control/effect lowering

This directory records the extraction path from the accepted, source-locked
adaptive-NUTS Reactant implementation to a reusable compiler backend. The
accepted NUTS source and mathematical oracle remain unchanged. This work must
not weaken the existing fail-closed or independent-oracle gates merely to make
another kernel compile.

## Current boundary

ReactiveKernels already supplies a general straight-line dataflow path:
`PreparedKernel` is static program structure under Reactant, tensorized recipe
bodies trace normally, and `replica` maps an unchanged scalar kernel through
`Reactant.Ops.batch`. The missing general surface is mutable, effectful,
data-dependent, and recursive control.

The accepted NUTS path proves that surface can be represented by fixed-capacity
tensors and one traced `while`, but its backend is deliberately specialized.
The following table is the removal checklist.

| NUTS-specific mechanism | Current evidence | General compiler replacement |
|---|---|---|
| Exact `step!` / `finish!` / `start!` method set and `26/15/15` block counts | `_nr_plan`, `_NUTS_REACTANT_CFG` | `_control_program`: source-derived method ids, entries, normalized blocks, terminators, calls, conditions, effects, and frame positions |
| Program-counter keyed branch meanings | `_nr_branch(::Val{kind}, ::Val{pc})` | Lower each captured branch condition from normalized MethodIR through a typed value-lowering algebra |
| Program-counter keyed call arguments | `_nr_callargs` | Lower `TCall.args` using the program formal-position and stored-frame metadata |
| Program-counter keyed mutations and diagnostics | `_nr_effects` | Lower ordered `_PlaceWrite`, `_PlaceSwap`, `_LocalAssign`, `_ExprStmt`, registered calls, and primitive effects from each block |
| Phase-point/tree/proposal tensor fields | `_nuts_frame_to_tensors`, `_NUTS_PP_VEC`, `_NUTS_PP_SCA` | Derive tensor state slots and ownership paths from the prepared canonical layout and effect closure |
| Handwritten derived-value refresh | `_nr_recompute`, `_nr_set_derived` | Reuse the selected recipe/currentness schedule to emit kill, demand-recompute, and bless operations |
| NUTS copy/swap helpers | `_nr_copyowned`, `_nr_swapprop` | Generic path-indexed copy/swap lowering over compiler-owned state slots |
| Direction/exponential counters | `_nr_draw_dir`, `_nr_draw_exp` | Typed RNG primitive effects lowered to explicit bundle ports plus per-stream counters, preserving source ordering and conditional consumption |
| NUTS-specific capacity arithmetic | `nuts_reactant_state`, `_nr_finish` | A backend capacity policy derived from framed methods, live state, caller bounds, and explicit overflow outputs |
| Diagonal-metric/gradient configuration | `_nr_config` | Typed operation support table plus captured callable/constant ports; unsupported operations reject before tracing |
| NUTS-only traced-loop generator | `compile_nuts_reactant_transition` | A backend-neutral control program consumed by a Reactant tensor-state/effect emitter |

`_control_program` is the first landed extraction. It retains normalized source
effects and call/branch operands rather than only NUTS labels, frames the root
uniformly even when acyclic, and keeps truly acyclic siblings inlined. The
native dispatcher and the Reactant backend can therefore share one topology
authority.

The next admitted executable slice is the public, backend-neutral
`compile_state_transition` seam. It combines an endpoint `KernelSpec` with a
bound free update method, lowers exact `map(copy, tuple)` and tuple/named
destructuring, statically unrolls captured integer `Base.Colon` loops, and
threads selected-recipe currentness through every iteration. The same compiler
path executes generalized leapfrog and implicit midpoint over all six
Gaussian/relativistic Euclidean, Riemannian, and diagonal-SoftAbs endpoint
specs. Compiler code contains no integrator, geometry, field-name, or program-
counter cases. Indexed writes and data-dependent structured control remain the
next admission boundary; the specialized adaptive-NUTS emitter is not yet
removed.

## Acceptance rules for every increment

1. Add a kernel whose semantics cannot be satisfied by the previously admitted
   subset; do not add cosmetic variants.
2. Expectations come from an independent eager/source-semantic implementation,
   never from another emitter consuming the same lowered program.
3. Exact-match control decisions, effect/RNG consumption, ownership, and
   diagnostics. Use an explicit justified tolerance only for floating values.
4. Add adversarial negative tests for every newly admitted IR node, operation,
   alias shape, or control construct.
5. Keep the accepted NUTS fixture byte-identical and rerun its full Reactant
   acceptance after every generalization increment.
6. A structural source change must either compile correctly through the generic
   path or reject for a semantic reason. It must never be accepted because a
   stale program-counter table still happens to match.

The HMC semantics owner selects the multi-kernel corpus and owns the independent
oracle/review contract. Implementation proceeds from the simplest new semantic
capability toward the full NUTS migration; NUTS is the final consumer, not the
template from which generic behavior is guessed.

## HMC-owned semantic freeze

The backend IR and every optimization must retain these source-observable
invariants. They are acceptance requirements, not NUTS implementation hints:

1. Momentum refresh uses the current metric.
2. State restore precedes NUTS tree mutation.
3. A nonfinite Hamiltonian difference maps to negative infinity.
4. Direction and exponential RNG streams consume only on entered source paths
   and in source order.
5. Statistics callbacks run before a divergence early exit.
6. Direction changes preserve forward/backward endpoint aliasing.
7. Proposal swaps preserve ownership, while the final accepted proposal copies
   values into the persistent state.
8. A configured `max_depth = 10` has no smaller product cap.
9. Capacity overflow is explicit and fail-closed.

The ordered admission frontier is:

1. Dual averaging
2. Euclidean phase point
3. Leapfrog
4. Generalized leapfrog
5. Welford accumulation
6. Relativistic kinetic energy (RKE)
7. SoftAbs metric
8. Fixed-step HMC
9. Adaptive NUTS
10. Trajectory statistics

The corpus also requires typed conditional RNG, generic path copy/swap,
captured callable and constant ports, and unsupported-operation rejection. The
HMC lane owns the independent oracle for every admitted stage and all selected
logical upstream algorithm variants; a later stage does not replace the gates
for an earlier one.
