# Mutation profile B corpus translation

This report accompanies parallel fixtures; it does not modify the locked
ReactiveHMC/NUTS/WALNUTS source authorities.

Profile B gives compiler-consumed kernel source one sound coarse rule:

- a causal ordinary-Julia transform returns its value into an explicit LHS;
- a bare causal call is allowed only across an exactly resolved RK
  method/kernel/intrinsic boundary with a composable effect summary;
- `=` has ordinary value/as-if semantics, so storage reuse is permitted only
  when ownership, liveness, shape, and alias proofs make it unobservable;
- `.=` requires identity-preserving mutation of the destination; on structured
  RK state it means the existing strong structural-copy operation rather than
  generic NamedTuple broadcasting.

The separate Reactant fixed-ABI restrictions still apply: no shape growth in a
compiled invocation, typed runtime RNG/effect/callable ports, and bounded
control.

## Rewrites in the current corpus

| Corpus | B rewrite | Status |
| --- | --- | --- |
| Fixed-step HMC | `init.mom = randn!(rng, init.mom)`; expose `init.mom = sqrt(metric) * init.mom`; use `.=` for the two endpoint-buffer copies and accepted-state transfer | Compiles through the generic functional transition, matches the frozen legacy receipt case, and passes the accepted Reactant case |
| NUTS | Replace two bare refresh effects with explicit momentum assignments; replace six public `copy!!` sites with structured `.=`; spell the two terminal authority transfers as `return destination .= source` | Both specialized native compilers agree on values/diagnostics; warmed transition remains zero-allocation; the depth-zero Reactant transition matches the expected state and control result |
| WALNUTS | Replace four `fill!` resets and nine `copy!!` sites with `.=`; expand the one short-circuit copy into an explicit `if` | Source and MethodIR validate; execution remains behind the pre-existing bounded recursive-SCC frontier |
| Dual averaging / Welford | No rewrite: scalar `+=`, dotted recurrence writes, and the exact sibling `step!` call already satisfy B | Unaffected |
| Generalized leapfrog / implicit midpoint | No rewrite: all state mutation is already an explicit dotted place write; local `map(copy, ...)` results are bound explicitly | Unaffected |
| Fixed-capacity trajectory/sample statistics | No rewrite: indexed writes, scalar assignments, capacity/count, and overflow are already explicit | Unaffected; upstream dynamically growing `push!`/`append!` statistics remain outside the fixed-shape profile |

The proposal tuple swap intentionally remains ordinary `=` because it rebinds
which endpoint identities occupy the tuple slots; it is not a structural copy.
Exact sibling calls such as `start!`, `finish!`, `integrate!`, `step_f`, and
`stats_f` remain recognizable mathematical calls. Their RK identity/effect
summary is the authority for leaving them bare, never the `!` spelling.

## Compiler delta exercised by the parallel fixtures

- MethodIR now distinguishes `formal .= source` (a place write) from
  `formal = source` (a local rebind).
- Generic functional/native lowering recognizes a direct structured-state root
  `.=` and routes it through the established structural-copy, invalidation,
  currentness, topology, and predication machinery.
- The two specialized NUTS emitters recognize root/index endpoint `.=` and use
  their established endpoint-copy backend.
- The generated native NUTS emitter supports dotted write-and-return directly:
  perform the ordinary dotted write once, then return the destination for a
  value method or the frame for a state method.
- NUTS momentum refresh accepts the mathematical `chol.L * mom` source only in
  the same concrete dense/triangular/diagonal domain admitted by its alias-safe
  `lmul!` backend. This is bufferization, not source-level mutation semantics.

## Deliberate limits and later simplification

Profile B does not itself solve recursive control. Generic formal/indexed
structured-copy propagation through WALNUTS' direct `start!`/`finish!` SCC must
wait for the bounded call-stack/worklist lowering. It also does not make dynamic
`push!` Reactant-compatible; fixed capacity plus logical length/overflow is the
compiled representation.

The prototype generic SCC control builder does not yet carry the result of a
value-position `_SetReturn` across an inlined helper. The specialized NUTS
emitters implement and test the full dotted write-and-return contract; generic
SCC value-return plumbing remains part of the separate bounded-control lane,
not an acceptance claim of this authoring experiment.

The legacy source forms remain accepted in parallel for comparison. After
review and broader acceptance, the compiler can consider removing fixture-level
reliance on public `copy!!` and the NUTS-only legacy bare refresh spelling. The
generic structural-copy backend and exact RK effect summaries remain useful;
profile B simplifies their discovery, not their implementation.

The specialized NUTS Reactant emitter still assigns effects by frozen program
counter after validating the exact control graph. The B fixture therefore makes
the return authority explicit at the two terminal structured mutations; its
normalized CFG is byte-for-byte the same topology as the locked fixture. A
later generic effect-lowering pass should derive those operations from MethodIR
instead of source-position PCs. That machinery is a real removal candidate,
but changing it is deliberately outside this parallel authoring experiment.

## Focused acceptance

- source/MethodIR corpus: 33/33;
- specialized NUTS native parity, destination identity/single evaluation,
  control topology, and warmed allocation: 9/9;
- fixed-step HMC native receipt parity: 1/1;
- fixed-step HMC Reactant accepted case: 7/7;
- specialized NUTS Reactant depth-zero case: 6/6.

The WALNUTS B source/MethodIR remains accepted but intentionally has no runtime
claim until the separate bounded direct-SCC lane completes.
