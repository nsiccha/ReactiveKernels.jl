# NUTS→Reactant lowering — handoff (ReactiveKernels:hmc → ReactiveKernels:reactant)

Verified foundation + proof scripts for lowering the ADAPTIVE NUTS transition (from the BYTE-IDENTICAL
`benchmark/nuts_kernel_authoring_fixture.jl`, `@kernel nuts_state`) to a Reactant-traceable form.
Implementation ownership: `ReactiveKernels:reactant` (decision 19rzxhy). Full spec + all findings:
`GET /agents/ReactiveKernels:hmc/todos/2026-08-29T01-38-17-594-00w87ln?plain` (many detailed comments).

Run the scripts from the repository's shared `packages` project on Julia 1.10;
their source and fixture paths are resolved relative to each script. The native
runtime is the parity oracle, while the Reactant `@compile` remains the gate.

## Scripts + what each VERIFIES (all green)
- `oracle.jl` — native oracle: `compile_nuts_native(pf, nuts_state, refresh_momentum!!, nuts!!, frame).root!(fr, scratch, Xoshiro(seed))`. Decodes the phasepoint `_CanonOwned12` SoA: **f4=pos, f5=mom, f7=pot, f8=dpot/dham_dpos, f10=dkin/dham_dmom, f11=kin, f12=ham** (f1/f2/f3/f6/f9=shared placeholders). Ref outputs (md=4, pos0=[1,2] mom0=[3,4] metric=diag(2) step=0.1): Xoshiro(20260829)→pos=[1.3401842630090366,1.3967476576654783], diag n_steps=7/reached_depth=3/acc=0.9996258996668541/dham=0.0031325053049702234.
- `frame_struct.jl` — full frame/tree/proposal/shared structure dump (the state to tensorize).
- `drawlog2.jl` — EXACT RNG draw order via a recorder defined INSIDE `Random` (so `_refresh_rng_domain!` admits it; a scratch method-override is invisible to the @generated gate's world-age). Order: momentum randn D-vec → per outer depth d: 1 `rand(rng,Bool)` direction, then `d` `randexp` (d-1 subtree-merge + 1 top). Count is DATA-DEPENDENT (md=6/seed=12345→28 exps; md=8/seed=777→2, early stop).
- `bundle.jl` — the WRONG structural pre-gen (kept as a caution: "d exps per depth" fails; exp count is data-dependent).
- `replay.jl` — **CORRECT bundle model + native reference driver, VERIFIED**: bundle=(momentum,dirs,exps) separate typed arrays; `ReplayRNG` (in `Random`) feeds native → `native(ReplayRNG(bundle)) === native(Xoshiro(seed))` on pos+all diag, md 2..8. Both native ref + your traced transition consume the identical bundle by counters → parity by construction. GOTCHA: `Random.randn!(r, x::AbstractArray)` MUST type `x` (untyped is less specific than Random's generic → silently no-op).
- `tensorize.jl` — **state tensorization VERIFIED**: `frame_to_state`/`state_to_frame!` capture the FULL mutable NUTS state; round-trip preserves the transition md 2..8. The exact tensor SoA to trace.
- `cfgdump.jl` — the CFG (`build_method`, src/kernel_control.jl): 3 defunctionalized methods `step!`(mid6/26blk)/`finish!`(mid9/15blk)/`start!`(mid10/15blk, recursive TCall); 7 acyclic methods inlined. Terminators TBranch/TCall/TGoto/TRet; SoA cols ep/depth/spilled-locals.
- `interp_spine.jl` — the control-flow SPINE (while csp>=1 + stack + terminators from the CFG): terminates correctly. Becomes your `@trace while sum(csp)>=1`.
- `soa_ref.jl` — `compile_nuts` (SoA imperative dispatcher, kernel_nuts.jl:536-649) == native: the plain-Julia state-machine REFERENCE to tensorize. Your emitter re-targets its emission: control→traced idioms, state→tensorize.jl repr, RNG→replay.jl bundle.

## Key contract (from reactant's own proofs — mirror your probe6_crux2.jl/probe_vecidx.jl):
length-1 Int-vector csp/kd/ke; `stack[csp]` gather / functional `w[csp.±n]=…` scatter (NEVER scalar, NEVER raw Ops.* — silently WRAPS OOB to a wrong in-bounds elt); `@trace while sum(csp)>=1`; `ifelse.` dispatch; gate ALL frozen carries incl diagnostics; capacities `4*(max_depth+2)` stack / `2^max_depth` exps / `max_depth` dirs / D momentum + saturation flag.
