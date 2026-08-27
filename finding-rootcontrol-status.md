# Root/control status — durable note during KB 503 outage (2026-08-27 ~09:30)

KB is 503-ing; RK report + primer update pending recovery. Re-send the report below when KB is back.

## Approved/committed this session (branch kb-impl/ReactiveKernels-poc-lowering-b2, base 4c6ed92)
- **eac3507** — prepared-endpoint PHASEPOINT (RK-approved): `compile_prepared_initialization` (full cold
  init, honest chol/logdet once) + `compile_prepared_schedule(pf,OW,SH,leaf_ir::MethodIR)` (public;
  PlanKey-tagged type-level `_SelectedTrace`; warmed post-write recompute exact 0-B/@inferred, chol+@node
  Δ0, pgrad Δ1) + `prepared_transition_trace` (write-kill closure over `kernel_plan_producer_owned`).
- **6085efd** — executable LEAPFROG (RK-approved through this SHA): `compile_leapfrog(pf,OW,SH,leaf_ir)`
  composes the real ccb35d3 3-op `leapfrog!` writes with demand-driven prepared recompute. Scalar/vector
  classified RHS (0-B, no nested Broadcasted); callees rebind-checked via `_kernel_resolve_captured_ref`
  + `kernel_rebound` (no Core.eval); STALE-AT-ENTRY contract (`_lf_ensure!` runtime entry guards: normal
  Δ1/0-B, recovery Δ2/analytic, dirty non-producible source rejects before write); real mask kills/blesses.
- Suites: codegen 108, lowering 104, methodir 247, factory 396.

## Root/control design-B — 6 CAPABILITIES/FIXES PROVEN (scratch gen_compiler.jl, all 0-B/@inferred)
1. Mutual+self recursion (SCC defunctionalized → per-MethodId SoA frames + isbits control stack).
2. SCC-inlining (RK 09:29): only the recursive SCC defunctionalized; acyclic-non-suspending siblings
   INLINED into maximal native blocks (`recursive_mids`/`defunctionalized_mids`).
3. Callee-return continuation (RK 09:43): CFG-level inlining with `ret_pc` — an inlined callee `_Return`
   (incl. branch-local early returns) jumps to the call-site continuation, never the caller return; value
   position binds a native local. Proven: helper early-returns, caller's later write still executes.
4. Sparse decl ordinals (RK 09:49): immutable decl→DENSE-store-index map (`sidx`); the SoA tuple is indexed
   by the dense index, the `_CtrlFrame` control TAG stays the exact `MethodId.decl`. Proven on SCC=[2,4].
5. Defunc-set (RK 09:51): defunctionalized = SCC + suspension-bearing ancestors (fixpoint); a truly-acyclic
   driver that CALLS the SCC is defunctionalized (needs a PC continuation), root need not be in the SCC.
6. Complete structured call-edge walk (RK 09:54): generic reflection walk over every RK IR node's fields +
   tuples/vectors → collects each `_Call`/`_CallExpr` decl; catches calls nested in `_For`/`_Guard`/etc.

## 7-8. Walk refinements PROVEN (RK 09:57): descent constrained to `_MExpr`/`_MStmt` + Tuple/Vector/
## NamedTuple/Pair (NOT registration/skeleton metadata); `Pair.second` traversed (kw actual values, else
## a call inside a kw value is silently missed — proven: `helper!(…; g = ping!(…))` defunctionalizes the
## driver); `_edges` UNIONS every candidate MethodId; `_call_mid` emission REJECTS candidate ambiguity
## (requires exactly one MethodId-exact narrowed candidate).

## DESIGN for the LOOP + cross-suspension locals (RK 09:51:45/09:58 — biggest remaining piece)
`_For(var,iter,body)`: iter `1:m` = `_RegisteredCall(Colon(), [_Lit(lo), _FormalRef(hi)])`. CFG: init(i=lo,
store i + hi) → header H(if i<=hi → body B else → after) → B(body; may suspend) → incr I(i+=1; goto H);
`_Break`→after, `_Continue`→I. The loop var `i` + bound `hi` are cross-suspension locals.
STORE-MACHINERY REFACTOR NEEDED: today the machine LOADS read-only formals at block entry only. Cross-
suspension locals are WRITTEN (i=lo, i+=1) and read across a suspension, so the store must (a) hold typed
WRITABLE local columns (formals ∪ cross-suspension locals; reject ambiguous/Any types), (b) SPILL live
locals to the SoA column BEFORE a TCall suspension, (c) reload at block entry. `emit_val(_LocalRef)` →
the loaded local. RK 09:58: an inlined value-helper return temp live across a PC edge must ALSO go in the
SoA column (never a native local across a PC jump → undef/phi/Core.Box); prove typed/LLVM no-Box, not only
@allocated==0. Then value-helper-in-expression, same-name overload EMISSION, exception/taken-trace,
positional+kw/defaults source-ordered, PC-transition census, explicit builder reject of unsupported IR.

## Older REMAINING list (superseded by the above where overlapping; RK 09:19/09:51:45/09:54)
- `_For`/`_While` BUILD in `build_region!` with break/continue + loop continuations (PC back-edges).
- Cross-suspension LOCAL spilling (RK 09:51:45): a local assigned before an SCC call + read after (rebound
  next iteration) must spill to the frame SoA (or recompute from spilled depth); LocalRef continuation
  mapping; source-ordered rebinding. Currently only positional formals spill (read-anywhere, sound).
- value-helper-in-expression; positional+kw/defaults (source-ordered eval); same-name overloads
  (MethodId-exact); exception/taken-trace invalidation; PC-transition census + no-jl_apply_generic gate;
  EXPLICIT reject of any unsupported IR variant in the BUILDER (not just the walk).
Then: commit the comprehensive adversary; rebase ONCE onto syntax 58e3903 (callable-frame seam APPROVED,
RK 09:45); bind the real root/frame (no synthetic storage survives); then the real nuts_state tree +
`compile_leapfrog` splice at start!(depth==1) under one outer epoch.

## Older core note
File: `<scratchpad>/gen_compiler.jl` (+ `test_gen.jl` runner). A 4-method mutual+self recursion adversary
(ping!/pong! mutual + reset!/driver!) from a Tuple of MethodIRs → per-MethodId maximal blocks + liveness →
concrete per-MethodId SoA stores + ONE isbits control stack `_CtrlFrame(mid=MethodId.decl, fidx, pc)` →
finite while-loop dispatcher. Result: **count==2^n for n=0..6 CORRECT, warmed exact 0-B, @inferred.**
Fixes applied: parked loop bug (a frame at pc==0 now pops the frame + control frame); registered-callee
resolution via `registration.source` rebind-checked (`_kernel_resolve_captured_ref`+`kernel_rebound`, no
Core.eval); liveness/`_reads_formal` recurse through `_RegisteredCall`.

## RK 09:29 PERFORMANCE REFINEMENT (reshapes the design — apply next)
Do NOT defunctionalize every method (current prototype does). Only defunctionalize the recursive SCC
(start!/finish!) + step-loop continuations into the PC machine. INLINE/FLATTEN acyclic captured siblings
(reset!/flip!/flip_neg!/swapproposal!/collectstats! + registered leaf/stats/effect bodies) into maximal
native blocks at preparation. Dispatcher = a generated finite branch over literal MethodId/PC, no dynamic
lookup/call, no per-statement dispatch. Add a PC-transition census per leaf/depth + an emitted-code
no-jl_apply_generic/opaque-edge gate EARLY (throughput dies otherwise even at 0-B).

## nuts_state scope (9 methods)
start!(decl9) → collectstats!/start!(self)/swapproposal!/finish!; finish!(decl8) → start!; step!(decl5) →
reset!/flip!/finish!/swapproposal! (with _For depth loop + _If); flip!(decl6) → flip_neg!. New node types:
_For(var,iter,body), _PlaceSwap(targets), _LocalAssign(lhs,rhs), _If(cond,thenb,elseb), _Guard(op,cond,body).
SCC = {start!, finish!} (mutual) + start! self-recursion. Acyclic = reset!/flip!/flip_neg!/swapproposal!/
collectstats!.

## Next increments (per RK 09:19 + 09:29)
1. SCC/acyclic partition of the call graph → inline acyclic callees into the caller block; defunctionalize
   ONLY the SCC + step-loop continuations.
2. Complete the comprehensive adversary: _For/break, early return, positional+kw/defaults (source-ordered
   eval), same-name overload resolution (MethodId-exact), exception/taken-trace invalidation.
3. Cross-suspension liveness MINIMIZATION (current: sound read-anywhere spill).
4. Splice `compile_leapfrog` leaf+recompute at start!(depth==1) (no residual step_f/leapfrog edge; endpoint
   formal stored as concrete EP; gofwd = two distinct branch arms).
5. Real nuts_state start!/finish!/step! tree.
6. src integration + capacity from max_depth + static SCC/call-path bound (prevalidated, overflow rejects);
   bind only on syntax's callable concrete-frame seam (supersedes 2eb01be). Root exception: ONE outer
   handler dirties taken/touched outputs before rethrow; no nested leaf commit leaves them blessed.

## USER CORRECTIONS to fold on the post-58e3903 rebase (RK 10:20 / 10:22 — queued behind root-control)
1. pot_f NOT removed: the faithful phasepoint has BOTH pot_f and grad_f as ALTERNATIVE pot producers.
   Codegen must consume `plan.producer[pot]` (the SELECTED producer), choose the destination-bound grad
   recipe for NUTS, invoke pgrad once/leaf, invoke the non-selected pot_f ZERO times — no hardcoded
   "no alternative producer" assumption. (My euclidean_ep is the pot_f-free simplification; restore pot_f
   on the rebase, keeping the selected-producer model. Do NOT weaken 0-B/identity/currentness gates.)
2. Leaf result identity is PRIVATE: compile_leapfrog's `return owned` is only a private codegen carrier.
   In the composed tree/root, the inlined leapfrog expression value is DISCARDED unless the authored
   MethodIR explicitly returns it. Public result semantics come SOLELY from nuts!!'s authored return state
   (prove result===state). Add a discriminator: changing the internal leaf carrier cannot determine the
   public return. Do NOT document/export leaf result identity. (Soften the compile_leapfrog docstring's
   "returns the owned object (!!-style identity)" claim on the fold.)
