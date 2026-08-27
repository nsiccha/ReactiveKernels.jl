# Inc3 factory/composition substrate tests. Isolated in a module so fixtures cannot
# shadow package exports in the shared Pkg.test Main.
module TestKernelFactory
using ReactiveKernels, Test
using LinearAlgebra, Random
const RKS = ReactiveKernels

# a PURE helper reading reactive places by reference — admissible ONLY via an explicit exact
# declaration (the public `@rk_pure`), never by body inference.
crit(a, b) = a + b
@rk_pure crit 2
@kernel critowner(x, y) = begin
    go!() = begin
        z = crit(x, y)          # reads self places x,y — admissible via @rk_pure
        Base.fill!(x, z)        # a real owned write so the closure runs
    end
end
# an UNDECLARED helper on a self place stays opaque → the closure rejects by name.
uncrit(a, b) = a + b
@kernel undeclowner(x, y) = begin
    go!() = begin
        z = uncrit(x, y)
        Base.fill!(x, z)
    end
end
# same-NAME cross-module helpers get DISTINCT identity-derived tokens (no collision).
module CritA; f(a, b) = a + b; end
module CritB; f(a, b) = a; end
@rk_pure CritA.f 2
@rk_pure CritB.f 2
# borrow / rng declarations (top-level: method defs cannot live in a testset's local scope)
borrowfn(a) = a
@rk_borrows borrowfn 1
rngfn(r, x) = x
@rk_rng rngfn 2 1
# WRONG-ARITY: crit is declared arity 2; calling it with 1 positional must reject deterministically.
@kernel wrongarity(x) = begin
    go!() = begin
        z = crit(x)
        Base.fill!(x, z)
    end
end
# REBIND adversary: a declared helper captured through a NON-const module slot; rebinding the slot
# to a module without the helper must read as rebound (same detached-identity check as primitives).
module Helpers; hot(a, b) = a + b; end
@rk_pure Helpers.hot 2
HelperMod = Helpers
@kernel hotowner(x, y) = begin
    go!() = begin
        z = HelperMod.hot(x, y)
        Base.fill!(x, z)
    end
end
baremodule NoHot end
# DESCRIPTOR-DRIFT adversary: same callable identity, re-declared with different SEMANTICS.
driftfn(a, b) = a
@rk_pure driftfn 2
@kernel driftowner(x, y) = begin
    go!() = begin
        z = driftfn(x, y)
        Base.fill!(x, z)
    end
end

# Faithful euclidean_phasepoint ENDPOINT (mirrors benchmark/nuts_kernel_authoring_fixture.jl):
# methodless — owned set is the recipe closure seeded by the integrator's subject write-roots.
@kernel phasepoint_ep(pot_f, grad_f, metric, pos, mom) = begin
    pot = pot_f(pos)
    pot, dpot_dpos = grad_f(pos)
    chol_metric = cholesky(metric)
    dkin_dmom = chol_metric \ mom
    kin = 0.5 * (@node(logdet(chol_metric)) + dot(mom, dkin_dmom))
    ham = pot + kin
    dham_dpos = dpot_dpos
    dham_dmom = dkin_dmom
end
@kernel leapfrog_ep!(phasepoint; stepsize) = begin
    @. phasepoint.mom -= stepsize * phasepoint.dham_dpos
    @. phasepoint.pos +=       stepsize * phasepoint.dham_dmom
end
# a small kernel whose LIVE mutable graph we mutate after capturing a plan (recipe-input snapshot).
@kernel mutkernel(x) = begin
    y = x + 1
    z = y * 2
end
# authored input ORDER matters for execution binding: `b - a` has inputs (b, a), NOT sorted.
@kernel orderk(a, b) = begin
    z = b - a
end
# _SelfRef subject-write discriminator: a registered callback that writes ONLY its subject's `x`.
@kernel cbwrites!(o) = begin
    @. o.x = 1.0
end
@kernel cbowner(x, y; cb) = begin
    run!() = cb(__self__)
end
# a STATEFUL external gradient functor (DI/counting): construction must retain it by IDENTITY
# (identity + counter), never deep-copy it.
mutable struct CountingGrad; n::Int; end
(c::CountingGrad)(x) = (c.n += 1; x)
# function barriers so 0-B measurements aren't polluted by a boxed loop-local (the ops themselves
# are 0-B; a loop-scoped variable captured by @allocated is what boxes).
_canonset0b(o, v) = RKS._canon_set!(o, Val(1), v)
_canoncopy0b(o, s) = RKS._canon_copy_slot!(o, s, Val(2))
# in-place gradient callable (acceptance-hook shape): writes dest = 2*pos, RETURNS pot = sum(abs2,pos)
pgrad_ex!(dest, pos) = (dest .= 2 .* pos; sum(abs2, pos))
_applygrad0b(b, o, pos) = RKS._kernel_apply_grad!(b, o, pos)
# a pot_f-FREE gradient endpoint (mirrors the ccb final fixture's euclidean_phasepoint): a SINGLE grad
# recipe produces BOTH the scalar potential `pot` and the buffer gradient `dpot_dpos` from `pos`.
@kernel gradonly_ep(grad_f, pos) = begin
    pot, dpot_dpos = grad_f(pos)
    dham_dpos = dpot_dpos
end
@kernel gradonly_step!(phasepoint; stepsize) = begin
    @. phasepoint.pos += stepsize * phasepoint.dham_dpos
end
# an ISOLATED pot_f-FREE full endpoint BYTE-MATCHING the ccb euclidean_phasepoint (RK 07:35): single grad
# recipe (both pot+dpot owned, no collateral), typed `oftype(pot,0.5)` half, @node(logdet), aliases.
@kernel euclidean_ep(grad_f, metric, pos, mom) = begin
    pot, dpot_dpos = grad_f(pos)
    chol_metric = cholesky(metric)
    dkin_dmom = chol_metric \ mom
    kin = oftype(pot, 0.5) * (@node(logdet(chol_metric)) + dot(mom, dkin_dmom))
    ham = pot + kin
    dham_dpos = dpot_dpos
    dham_dmom = dkin_dmom
end
# an ISOLATED duplicate whose live graph the prepared-factory poison test may mutate WITHOUT leaking
# into the clean gradonly_ep used by the copy test (RK 06:59).
@kernel gradonly_poison_ep(grad_f, pos) = begin
    pot, dpot_dpos = grad_f(pos)
    dham_dpos = dpot_dpos
end
@kernel gradonly_poison_step!(phasepoint; stepsize) = begin
    @. phasepoint.pos += stepsize * phasepoint.dham_dpos
end
# a registered diagnostics callback writing n_steps + acceptance_rate (mirrors nuts_stats!'s write-roots)
@kernel synstats!(state) = begin
    state.n_steps += 1
    state.acceptance_rate = state.acceptance_rate + one(state.acceptance_rate)
    return state
end
# a registered callback writing an UNMAPPABLE (non-diagnostic) root — the binding must REJECT it
@kernel badstats!(state) = begin
    state.n_steps += 1
    state.foo = state.foo + 1
    return state
end
# a step integrator with a TYPED required keyword (stepsize::Float64) — the parser must recognize it
@kernel typedstep!(phasepoint; stepsize::Float64) = begin
    @. phasepoint.pos += stepsize * phasepoint.dham_dpos
end
# a COUNTING in-place gradient — proves 'zero extra pgrad' non-vacuously (count stays 1 across child seeds)
mutable struct CountingPgrad; n::Int; end
(c::CountingPgrad)(dest, pos) = (c.n += 1; dest .= 2 .* pos; sum(abs2, pos))
# Two DISTINCT Mode-2 `!!` skeletons for the sampler identity gate (RK 12:37): each mints a def-unique
# Token. `nuts_root!!` is the sampler's compiled-root owner; `other_root!!` is an UNRELATED skeleton whose
# Token must NOT admit the same sampler's transition (pure-type gate, no name special-case).
@kernel nuts_root!!(state; rng) = begin
    step!(state, rng)
    return state
end
@kernel other_root!!(state; rng) = begin
    step!(state, rng)
    return state
end
# A synthetic compiled `root!(frame, scratch, rng) -> frame` standing in for POC's compiled Mode-2 root
# (POC's real root isn't on the syntax branch): it bumps a diagnostic in place and returns the SAME frame,
# so `result === state`. Concretely-typed scratch (a Tuple), rng THREADED (never stored in scratch).
synroot!(frame, scratch, rng) = (RKS._diag_set!(frame.diag, Val(1), RKS._diag_slot(frame.diag, Val(1)) + 1); frame)
# function barrier for the whole-endpoint mixed scalar+buffer copy 0-B / @inferred gate (RK 07:02)
_copyep0b(d, s) = RKS._canon_copy_endpoint!(d, s)
# a buffer that THROWS during copyto! (post-validation mid-copy failure) — proves the epoch contract
# (mask stays zero/dirty; no stale blessed bit) after an ACTUAL mid-copy throw (RK 07:08).
mutable struct ThrowingBuf <: AbstractVector{Float64}; data::Vector{Float64}; end
Base.size(t::ThrowingBuf) = size(t.data)
Base.getindex(t::ThrowingBuf, i::Int) = t.data[i]
Base.setindex!(t::ThrowingBuf, v, i::Int) = (t.data[i] = v)
Base.copyto!(::ThrowingBuf, ::AbstractArray) = error("mid-copy throw")   # unambiguous, wins on arg1
# a SECOND, byte-identical endpoint DEFINITION — same slot/recipe SHAPE, DIFFERENT canonical
# Value identities — to prove the plan key distinguishes definitions (RK 04:17c).
@kernel phasepoint_ep2(pot_f, grad_f, metric, pos, mom) = begin
    pot = pot_f(pos)
    pot, dpot_dpos = grad_f(pos)
    chol_metric = cholesky(metric)
    dkin_dmom = chol_metric \ mom
    kin = 0.5 * (@node(logdet(chol_metric)) + dot(mom, dkin_dmom))
    ham = pot + kin
    dham_dpos = dpot_dpos
    dham_dmom = dkin_dmom
end

# Single-object owned/shared fixture: method `add!` writes `total`; `combined` derives
# from `total`; `scale` is a constant recipe; `seed` is an unwritten source.
@kernel accum(seed) = begin
    total = seed
    scale = 2.0
    combined = total * scale
    add!(x) = begin
        total += x
    end
end

# a registered free integrator, for callable-field/partial resolution.
@kernel leapfrog!(phasepoint; stepsize) = begin
    @. phasepoint.mom -= stepsize * phasepoint.grad
end

# adversary: a Function with a `func` field but DIFFERENT call semantics — must NOT be
# accepted as a token-preserving binder (it never opts into the binder trait).
struct EvilWrap{F} <: Function
    func::F
end
(e::EvilWrap)(args...; kwargs...) = error("evil call semantics")

# call-induced ownership: `s` is written ONLY through the intrinsic `copy!!(s, y)`, not by
# a direct place-write — local write_roots miss it; the call-writes pass catches it.
@kernel callowner(x, y) = begin
    s = x
    drive!() = copy!!(s, y)
end

# identity-bound primitive: `Base.fill!(buf, 0.0)` writes `buf` (an owner field) through
# the registered RK-core primitive — captured DETACHED at definition with its positional
# write descriptor, keyed by the authored slot, rebind-checked (fix 1).
@kernel filler(buf) = begin
    zero!() = Base.fill!(buf, 0.0)
end

# two-primitive capture: distinct primitives captured in one owner must carry DISTINCT
# registration tokens (else they collide as _RegisteredCall edge / cache keys).
@kernel dualprim(a, b) = begin
    go!() = begin
        Base.fill!(a, 0.0)
        Base.copyto!(b, a)
    end
end

# local-spoof adversary: a method-local `fill!` shadows the primitive → it is a body LOCAL,
# excluded at ref collection, NEVER captured as the registered `Base.fill!` identity.
@kernel spooffill(buf) = begin
    zero!() = begin
        fill! = (a, b) -> a
        fill!(buf, 0.0)
    end
end

# alias-module rebind adversary: `AliasMod.fill!` is captured through a NON-const module
# slot; rebinding `AliasMod` to a module without `fill!` must read as rebound.
AliasMod = Base
@kernel aliasfill(buf) = begin
    zero!() = AliasMod.fill!(buf, 0.0)
end
baremodule NoFill end

# ep-chain owner (faithful nuts_state shape): `fwd`/`bwd` become owned SOLELY through
# `step! → finish!(fwd|bwd) → start!(ep) → step_f(ep)` — never a direct write. `step_f` is
# the integrator hole (resolves to a subject-writer); `pot_f` a read-only callback whose
# only argument `metric` must therefore stay SHARED, not spuriously owned.
@kernel epowner(fwd, bwd, metric; step_f, pot_f) = begin
    start!(ep) = step_f(ep)                       # step_f writes subject → start! writes formal 1
    finish!(ep) = start!(__self__, ep)             # sibling call passes __self__ → writes formal 1
    step!() = begin
        finish!(__self__, fwd)                     # → fwd owned via the chain
        finish!(__self__, bwd)                     # → bwd owned via the chain
    end
    energy!() = pot_f(metric)                      # pot_f read-only → metric NOT owned
end

# opaque-subject adversaries: an unregistered mutator on a self place carries no descriptor.
module Qual
    evil!(x) = x
end
@kernel evilowner(buf) = begin
    go!() = evil!(buf)                       # bare unregistered → :opaque, self subject → REJECT
end
@kernel qevilowner(buf) = begin
    go!() = Qual.evil!(buf)                  # qualified unregistered → :opaque, self subject → REJECT
end

# higher-arity primitive adversary: `copyto!(dest,src,do,so,n)` resolves to the SAME
# registration as the 2-arg form but its arity does not match the descriptor → must REJECT.
@kernel bigcopy(a, b) = begin
    go!() = Base.copyto!(a, 1, b, 1, 3)
end

# FULL real-nuts discriminator (RK block pt 7): every owned endpoint reached by a DISTINCT
# mechanism, shared identities untouched, plus the overload + branch-phi + direct-formal-write
# adversaries. Owned: init (intrinsic dest) · fwd/bwd (ep chain + branch-phi + direct formal
# write) · tree/proposals (primitive dest) · control (direct scalar write). Shared: metric,
# chol (read-only reads only) · pot_f (read-only callback arg).
@kernel nutsowner(init, fwd, bwd, tree, proposals, control, metric, chol; step_f, pot_f) = begin
    start!(ep) = step_f(ep)                        # integrator hole writes its subject
    start!(ep, half) = step_f(ep)                   # OVERLOAD (arity 2) — distinct MethodId
    finish!(ep) = start!(__self__, ep)              # sibling → writes formal 1
    extend!(dir) = begin
        t = dir ? fwd : bwd                          # branch-phi alias → SET {fwd, bwd}
        finish!(__self__, t)                         # → BOTH fwd and bwd owned
    end
    flip_neg!(ep) = begin
        @. ep.mom = -ep.mom                          # DIRECT write THROUGH a formal (pt 3)
    end
    step!() = begin
        copy!!(init, proposals)                      # intrinsic dest → init owned
        Base.fill!(tree, 0.0)                        # primitive dest → tree owned
        Base.fill!(proposals, 0.0)                   # primitive dest → proposals owned
        flip_neg!(__self__, fwd)                     # direct-formal-write sibling → fwd owned
        extend!(__self__, true)                      # phi+chain → fwd, bwd owned
        control = control + 1                        # direct scalar self-write → control owned
    end
    energy!() = pot_f(init) * metric[1] * chol[1]    # read-only callback + pure reads of metric/chol
end

# The FULL authored NUTS chain (euclidean_phasepoint / leapfrog! / refresh_momentum!! / nuts_state / nuts!! /
# nuts_stats!) for the end-to-end ergonomic-core gate — same faithful fixture POC compiles against.
module _ErgFix
    include(joinpath(@__DIR__, "..", "benchmark", "nuts_kernel_authoring_fixture.jl"))
end

@testset "Inc3 factory substrate" begin
    @testset "LOCAL owned seed (not authoritative)" begin
        # the direct-write seed is exactly the mutated field
        @test RKS._kernel_factory_direct_writes(accum) == Set((:total,))
        # local seed = direct writes ∪ recipe outputs derived from them (owned-ONLY; the
        # authoritative closure over call effects + the shared complement come later).
        seed = RKS._kernel_factory_local_owned_seed(accum)
        @test :total in seed && :combined in seed          # combined derives from total
        @test !(:scale in seed) && !(:seed in seed)        # constant / unwritten source
    end

    @testset "concrete no-Ref owner storage (redirect 1)" begin
        s = RKS._OwnerState{:tok}((1, 2.0, [3.0]))
        @test RKS.owner_token(s) === :tok
        # Val-indexed reads → constant getfield on the value tuple
        @test RKS._owner_slot(s, Val(1)) === 1
        @test RKS._owner_slot(s, Val(2)) === 2.0
        # slots are a concrete VALUE tuple — NO Ref/RefValue/cell wrapper on any slot
        @test fieldtype(typeof(s), :slots) <: Tuple
        @test !any(t -> t <: Ref, fieldtypes(typeof(RKS.owner_slots(s))))
        @test fieldtypes(typeof(RKS.owner_slots(s))) == (Int, Float64, Vector{Float64})
        # scalar updates commit by ONE typed tuple replacement (same T)
        RKS._owner_commit!(s, (10, 20.0, [30.0]))
        @test RKS._owner_slot(s, Val(1)) === 10
        # a different-typed tuple does not match the typed commit (layout stability)
        @test_throws MethodError RKS._owner_commit!(s, ("x", 1, 2))
    end

    @testset "callable field / partial resolution (toward gate 4)" begin
        # `partial(leapfrog!; stepsize)` resolves to leapfrog!'s registered Token (binder
        # is token-preserving).
        reg = RKS._kernel_resolve_callable(partial(leapfrog!; stepsize = 0.1))
        @test reg !== nothing && reg.token === RKS.kernel_token(leapfrog!)
        # a bare registered kernel resolves directly
        @test RKS._kernel_resolve_callable(leapfrog!).token === RKS.kernel_token(leapfrog!)
        # an opaque Julia callable does NOT resolve → REJECT
        @test RKS._kernel_resolve_callable(sin) === nothing
        @test_throws ArgumentError RKS._kernel_resolve_callable_or_reject(:step_f, sin)
        # a partial binding a NON-kernel rejects too
        @test_throws ArgumentError RKS._kernel_resolve_callable_or_reject(:step_f, partial(sin; a = 1))
        # ADVERSARY: a Function with a `func` field but no binder-trait opt-in is REJECTED,
        # even though its `func` IS a genuine kernel (explicit-registration boundary).
        @test RKS._kernel_binder_target(EvilWrap(leapfrog!)) === nothing
        @test RKS._kernel_resolve_callable(EvilWrap(leapfrog!)) === nothing
        @test_throws ArgumentError RKS._kernel_resolve_callable_or_reject(:step_f, EvilWrap(leapfrog!))
        # the approved binder DOES opt in
        @test RKS._kernel_binder_target(partial(leapfrog!; stepsize = 0.1)) === leapfrog!
    end

    @testset "call-induced writes (transitive ownership seed)" begin
        # `s` is mutated ONLY through the intrinsic `copy!!(s, y)` — NO local place-write,
        # so `write_roots` misses it, but the call-writes pass catches it as owned.
        @test isempty(RKS._kernel_factory_direct_writes(callowner))
        @test :s in RKS._kernel_factory_call_writes(callowner)
    end

    @testset "owned SEED heuristic (PROVISIONAL, owned-candidate ONLY)" begin
        # direct-subject case only: `copy!!(s, y)` has `s` as a direct first-actual, so the
        # seed catches it. NO shared complement from a seed — `shared` is created only by
        # the authoritative API (call-graph/formal-to-actual fixed point + unresolved reject).
        owned = RKS._kernel_factory_owned_seed(callowner)
        @test owned isa Set{Symbol}
        @test :s in owned
        @test !(:s in RKS._kernel_factory_local_owned_seed(callowner))
    end

    @testset "identity-bound primitive effect registry" begin
        # exact Base.fill! → complete positional descriptor: dest(1)=write, x(2)=read,
        # result aliases positional 1 (so `x = fill!(buf,0)` preserves x===buf downstream).
        pf = RKS._kernel_primitive_effect(Base.fill!)
        @test pf.writes == (1,) && pf.reads == (2,) && pf.result_alias == 1
        @test pf.token === Symbol("__rk_primitive_Base_fill!__")
        # two-primitive collision discriminator: DISTINCT primitives carry DISTINCT tokens
        # (a shared constant token would collide in _RegisteredCall edges / plan caches).
        pc = RKS._kernel_primitive_effect(Base.copyto!)
        @test pc.token === Symbol("__rk_primitive_Base_copyto!__")
        @test pf.token !== pc.token
        # copy!! is NOT a positional primitive — it is the structural :intrinsic
        # (owned-closure copy, result===dest), carried by its own registration.
        @test RKS._kernel_primitive_effect(copy!!) === nothing
        @test RKS.kernel_registration(copy!!).kind === :intrinsic
        # a LOCAL function named `fill!` is NOT the registered identity → reject
        local myfill! = (a, b) -> a
        @test RKS._kernel_primitive_effect(myfill!) === nothing
        # foreign callables (qualified or not) are not registered → reject
        @test RKS._kernel_primitive_effect(sin) === nothing
        @test RKS._kernel_primitive_effect(println) === nothing
    end

    @testset "detached Base.fill! capture + rebind (fix 1)" begin
        caps = RKS.kernel_callee_registrations(filler)
        prims = [c for c in caps if c.registration.kind === :primitive]
        @test length(prims) == 1
        prim = only(prims)
        # captured by EXACT identity, with the detached positional descriptor
        @test prim.target === Base.fill!
        @test prim.registration.primitive_effect isa RKS._PrimitiveEffect
        @test prim.registration.primitive_effect.writes == (1,)
        # the registration token is the primitive's DISTINCT identity (no shared constant)
        @test prim.registration.token === Symbol("__rk_primitive_Base_fill!__")
        # two captured primitives in one owner carry DISTINCT tokens (edge-key collision)
        dcaps = [c for c in RKS.kernel_callee_registrations(dualprim)
                 if c.registration.kind === :primitive]
        @test length(dcaps) == 2
        @test length(unique(c.registration.token for c in dcaps)) == 2
        # keyed by the AUTHORED SLOT — an independently reconstructed ref matches (value key)
        ref2 = RKS._CapturedCalleeRef(GlobalRef(@__MODULE__, :Base), :fill!)
        @test RKS.kernel_callee_registration(filler, ref2).primitive_effect.writes == (1,)
        # the slot re-resolves to Base.fill! → NOT rebound
        @test !RKS.kernel_callee_rebound(filler, prim.ref)

        # local-spoof: a method-local `fill!` is excluded → NO primitive capture at all
        @test isempty([c for c in RKS.kernel_callee_registrations(spooffill)
                       if c.registration.kind === :primitive])

        # alias-module rebind: captured through non-const `AliasMod`; rebinding it to a
        # module without `fill!` reads as rebound (detached slot re-resolution).
        aprim = only(c for c in RKS.kernel_callee_registrations(aliasfill)
                     if c.registration.kind === :primitive)
        @test aprim.target === Base.fill!
        @test !RKS.kernel_callee_rebound(aliasfill, aprim.ref)   # AliasMod === Base still
        @eval AliasMod = NoFill                                  # rebind the module slot
        @test RKS.kernel_callee_rebound(aliasfill, aprim.ref)    # NoFill has no `fill!`
        @eval AliasMod = Base                                    # restore
    end

    @testset "authoritative interprocedural ownership fixed point (step 2/3)" begin
        # step_f resolves to the subject-writer leapfrog!; pot_f to a read-only callback.
        stepreg = RKS.kernel_registration(leapfrog!)
        @test RKS._kernel_reg_writes_subject(stepreg)            # leapfrog! writes (:mom,)
        potreg = RKS._KernelRegistration(:pot_tok, :free_method, :phasepoint,
                                         (), (:pot,), false, nothing)
        @test !RKS._kernel_reg_writes_subject(potreg)            # empty write-roots → reader
        fr = Dict(:step_f => stepreg, :pot_f => potreg)

        owned = RKS._kernel_factory_owned_authoritative(epowner; field_regs = fr)
        # fwd AND bwd owned SOLELY through the ep chain (neither is ever directly written)
        @test :fwd in owned && :bwd in owned
        # the read-only callback's argument is NOT spuriously owned
        @test !(:metric in owned)
        # shared is the COMPLEMENT, created only after the closure — excludes fwd/bwd
        shared = RKS._kernel_factory_shared(epowner; field_regs = fr)
        @test :metric in shared
        @test !(:fwd in shared) && !(:bwd in shared)

        # NON-VACUOUS: with step_f resolved to a READ-ONLY callable, the chain writes nothing,
        # so fwd/bwd are NOT owned — ownership flows solely through the resolved write-roots.
        fr_ro = Dict(:step_f => potreg, :pot_f => potreg)
        owned_ro = RKS._kernel_factory_owned_authoritative(epowner; field_regs = fr_ro)
        @test !(:fwd in owned_ro) && !(:bwd in owned_ro)

        # RK block pt 5: the AUTHORITATIVE API REJECTS an unresolved required callable field —
        # it must never let an undecided step_f be blessed shared. The TEMPLATE API defers.
        @test_throws RKS._KernelFactoryReject RKS._kernel_factory_owned_authoritative(epowner)
        @test_throws RKS._KernelFactoryReject RKS._kernel_factory_shared(epowner)
        @test isempty(RKS._kernel_factory_owned_template(epowner))     # holes → no decided write

        # req 4 REJECTS: opaque unregistered mutator on a self place (bare + qualified), and a
        # method-local `fill!` spoof (opaque at the call site) — qualification/`!` not evidence.
        @test_throws RKS._KernelFactoryReject RKS._kernel_factory_owned_authoritative(evilowner)
        @test_throws RKS._KernelFactoryReject RKS._kernel_factory_owned_authoritative(qevilowner)
        @test_throws RKS._KernelFactoryReject RKS._kernel_factory_owned_authoritative(spooffill)
        # exact Base.fill! is admitted (registered primitive) — filler closes with buf owned
        @test :buf in RKS._kernel_factory_owned_authoritative(filler)
        # higher-arity copyto! (5 positionals) mismatches the arity-2 descriptor → REJECT
        @test_throws RKS._KernelFactoryReject RKS._kernel_factory_owned_authoritative(bigcopy)
    end

    @testset "FULL real-nuts ownership discriminator (step 2/3, pt 7)" begin
        stepreg = RKS.kernel_registration(leapfrog!)
        potreg = RKS._KernelRegistration(:pot_tok, :free_method, :phasepoint,
                                         (), (:pot,), false, nothing)
        fr = Dict(:step_f => stepreg, :pot_f => potreg)
        owned = RKS._kernel_factory_owned_authoritative(nutsowner; field_regs = fr)
        shared = RKS._kernel_factory_shared(nutsowner; field_regs = fr)

        # every writable endpoint owned, each via its distinct mechanism
        for f in (:init, :fwd, :bwd, :tree, :proposals, :control)
            @test f in owned
        end
        # branch-phi: BOTH arms owned (not just the last) — fwd via `dir ? fwd : bwd`
        @test :fwd in owned && :bwd in owned
        # read-only / identity fields are SHARED, never spuriously owned
        for f in (:metric, :chol)
            @test f in shared && !(f in owned)
        end
        # owned and shared partition the ports with no overlap
        @test isempty(intersect(owned, shared))
        @test union(owned, shared) == Set{Symbol}(RKS.kernel_port_names(nutsowner))

        # OVERLOAD keying: start! has two declaration MethodIds (arity 1 and 2); the arity-1
        # call site resolves to the arity-1 candidate — both are subject-writers so fwd/bwd
        # still close, but the summary is keyed by MethodId, not the collapsed name.
        ids = [ir.id for ir in RKS.method_irs(nutsowner) if ir.id.name === Symbol("start!")]
        @test length(ids) == 2
    end

    @testset "phase-point ENDPOINT ownership + path→slot plan (RK copy!! policy, step 4)" begin
        integ = RKS.kernel_registration(leapfrog_ep!)
        @test integ.write_roots == (:mom, :pos) || integ.write_roots == (:pos, :mom)
        owned = RKS._kernel_factory_endpoint_owned(phasepoint_ep, integ)
        shared = RKS._kernel_factory_endpoint_shared(phasepoint_ep, integ)
        # OWNED: integrator-written sources + endpoint-dependent closures
        for f in (:pos, :mom, :pot, :dpot_dpos, :dkin_dmom, :kin, :ham, :dham_dpos, :dham_dmom)
            @test f in owned
        end
        # SHARED: read-only authority + metric-only closures (chol_metric, @node(logdet))
        for f in (:pot_f, :grad_f, :metric, :chol_metric)
            @test f in shared && !(f in owned)
        end
        @test any(occursin("node", String(f)) for f in shared)   # @node(logdet) shared
        @test isempty(intersect(owned, shared))

        # the DEEPLY-IMMUTABLE canonical PLAN poc consumes (never recomputes ownership)
        plan = RKS._kernel_factory_endpoint_plan(phasepoint_ep, integ)
        @test RKS.kernel_plan_slot(plan, :pos).role === :owned
        @test RKS.kernel_plan_slot(plan, :dham_dpos).role === :owned
        @test RKS.kernel_plan_slot(plan, :metric).role === :shared
        # ALIAS COLLAPSE: dham_dpos≡dpot_dpos, dham_dmom≡dkin_dmom → SAME canonical slot
        @test RKS.kernel_plan_slot(plan, :dham_dpos).canon == RKS.kernel_plan_slot(plan, :dpot_dpos).canon
        @test RKS.kernel_plan_slot(plan, :dham_dpos).slot  == RKS.kernel_plan_slot(plan, :dpot_dpos).slot
        @test RKS.kernel_plan_slot(plan, :dham_dmom).slot  == RKS.kernel_plan_slot(plan, :dkin_dmom).slot
        # 9 owned authored names, 2 alias pairs collapse → 7 distinct physical owned slots
        @test RKS.kernel_plan_nowned(plan) == 7
        # alias groups exposed (both labels retained), sorted names
        @test any(g -> Set(g) == Set((:dham_dpos, :dpot_dpos)), RKS.kernel_plan_alias_groups(plan))
        @test any(g -> Set(g) == Set((:dham_dmom, :dkin_dmom)), RKS.kernel_plan_alias_groups(plan))
        # EXACT selected-Plan identity carried (Recipe ids), not a re-derived imitation
        @test !isempty(RKS.kernel_plan_recipes(plan))
        # SELECTED PRODUCER map (canonical Value id → selected Recipe id), immutable + sorted
        prod = RKS.kernel_plan_producer(plan)
        @test prod isa Tuple && !isempty(prod) && all(e -> e isa Tuple{Int,Int}, prod)
        cdp = RKS.kernel_plan_slot(plan, :dham_dpos).canon
        @test any(e -> e[1] == cdp, prod)                 # aliased target has a selected producer
        # POST-CONSTRUCTION entry_current = HAVE ∪ producer-map KEYS (recipe-owned; no collateral)
        ec = RKS.kernel_plan_entry_current(plan)
        gphc = RKS.kernel_graph(phasepoint_ep)
        canonof(n) = RKS.canon_id(gphc, phasepoint_ep.ports[n].id)
        @test cdp in ec && canonof(:pos) in ec
        @test length(ec) == length(unique(ec))            # deduped, stable
        # every canonical HAVE is present directly
        for n in phasepoint_ep.have_names
            @test canonof(n) in ec
        end
        # ec carries ONLY HAVE ∪ producer-map keys — nothing else is blessed
        havecanon = Set(canonof(n) for n in phasepoint_ep.have_names)
        prodkeys = Set(e[1] for e in prod)
        @test all(c -> c in havecanon || c in prodkeys, ec)
        # DEF-UNIQUE key as a VALUE type parameter: fresh reads of ONE definition → SAME key;
        # a same-SHAPE DIFFERENT definition → DIFFERENT key (canonical Value identities differ).
        @test RKS.kernel_plan_key(plan) ===
              RKS.kernel_plan_key(RKS._kernel_factory_endpoint_plan(phasepoint_ep, integ))
        @test RKS.kernel_plan_key(plan) isa Tuple && RKS.kernel_plan_key(plan)[1] === integ.token
        plan2 = RKS._kernel_factory_endpoint_plan(phasepoint_ep2, integ)
        @test RKS.kernel_plan_key(plan) != RKS.kernel_plan_key(plan2)
        # the key is a VALUE type parameter → distinct keys give distinct plan TYPES (for @generated)
        @test typeof(plan) !== typeof(plan2)
        # producer tuple EXACTLY matches the actual selected Plan mapping
        havev = Value[phasepoint_ep.ports[n] for n in phasepoint_ep.have_names]
        wantv = Value[phasepoint_ep.ports[n] for n in RKS._kernel_all_ports(phasepoint_ep)
                      if !(n in phasepoint_ep.have_names)]
        plactual = RKS.plan(RKS.kernel_graph(phasepoint_ep); have = havev, want = wantv)
        @test RKS.kernel_plan_producer(plan) ==
              Tuple(sort!([(cid, r.id) for (cid, r) in plactual.producer]))
        # multi-output producer: `pot` is produced by BOTH pot_f(pos) and grad_f(pos); the producer
        # map selects EXACTLY ONE (recipe-owned), so entry_current follows it, not collateral.
        cpot = RKS.canon_id(RKS.kernel_graph(phasepoint_ep), phasepoint_ep.ports[:pot].id)
        @test count(e -> e[1] == cpot, RKS.kernel_plan_producer(plan)) == 1
        # DEEPLY IMMUTABLE: only Tuples reachable — no Vector/Dict anywhere in the seam
        @test RKS.kernel_plan_slots(plan) isa Tuple
        @test RKS.kernel_plan_alias_groups(plan) isa Tuple
        @test RKS.kernel_plan_recipes(plan) isa Tuple && RKS.kernel_plan_producer(plan) isa Tuple
        @test all(s -> s.path isa Tuple && s.canon isa Int, RKS.kernel_plan_slots(plan))
        # PATH type admits INDEXED owner children (Symbol|Int steps for trees[i])
        @test RKS._PlanSlot((:trees, 1, :mom), 1, :owned, 1).path == (:trees, 1, :mom)
        # every owner field classified exactly once
        @test length(owned) + length(shared) == length(RKS._kernel_all_ports(phasepoint_ep))
        # aliased target's PRODUCER stays canonical: dham_dpos resolves to dpot_dpos's producer
        gph = RKS.kernel_graph(phasepoint_ep)
        @test !isempty(RKS.producers_of(gph, RKS.canon_id(gph, phasepoint_ep.ports[:dham_dpos].id)))
        # the Plan's canonical-Value → (role, field index) map the storage family consumes
        @test RKS.kernel_plan_field(plan, cdp) == (RKS.kernel_plan_slot(plan, :dham_dpos).role,
                                                   RKS.kernel_plan_slot(plan, :dham_dpos).slot)
        cmet = RKS.canon_id(gph, phasepoint_ep.ports[:metric].id)
        @test RKS.kernel_plan_field(plan, cmet)[1] === :shared
        @test RKS.kernel_plan_field(plan, -1) === nothing        # unknown canon → nothing

        # DETACHED recipe-input seam (RK 04:43): each selected recipe's canonical input Value ids,
        # captured as immutable Tuples so poc never rereads the live mutable graph.
        ri = RKS.kernel_plan_recipe_inputs(plan)
        @test ri isa Tuple && !isempty(ri) && all(e -> e isa Tuple{Int,<:Tuple}, ri)
        @test all(e -> e[2] isa Tuple, ri)                       # inputs immutable (not live Vectors)
        @test any(e -> canonof(:pos) in e[2], ri)                # some recipe reads pos's canon

        # a plan over a LIVE MUTABLE graph is a SNAPSHOT: mutating the graph after capture does not
        # change the captured recipe-input seam (a fresh plan would differ; the captured one is unchanged).
        gm = RKS.kernel_graph(mutkernel)
        planm = RKS._kernel_factory_plan(mutkernel, Set((:y, :z)), Set((:x,)))
        snap = RKS.kernel_plan_recipe_inputs(planm)
        RKS.add!(gm; inputs = (mutkernel.ports[:x],), outputs = (mutkernel.ports[:z],),
                 op = identity, cost = 1.0, cse_key = nothing, effectful = false)  # alt producer of z
        @test RKS.kernel_plan_recipe_inputs(planm) === snap      # captured snapshot UNCHANGED

        # AUTHORED INPUT ORDER preserved (RK 05:00 pt1/5): `z = b - a` records inputs (b, a), the
        # exact order the execution applier binds — NOT sorted; a position swap is distinguishable.
        go = RKS.kernel_graph(orderk)
        ca = RKS.canon_id(go, orderk.ports[:a].id); cb = RKS.canon_id(go, orderk.ports[:b].id)
        plano = RKS._kernel_factory_plan(orderk, Set((:z,)), Set((:a, :b)))
        rio = RKS.kernel_plan_recipe_inputs(plano)
        @test length(rio) == 1 && rio[1][2] == (cb, ca)         # authored order (b, a)
        @test rio[1][2] != Tuple(sort([ca, cb]))                # NOT sorted → a swap is distinct
        @test cb in RKS.kernel_plan_key(plano)[6][1][2]         # ordered inputs are in the Key
    end

    @testset "bare-identity canonical alias at expansion (RK 03:57 hardening)" begin
        # SINGLE-DEF identity → alias: both names one canonical Value; both labels retained
        @kernel al_single(a) = begin
            b = a
            c = b + 1
        end
        g = RKS.kernel_graph(al_single)
        @test RKS.canon_id(g, al_single.ports[:b].id) == RKS.canon_id(g, al_single.ports[:a].id)
        @test haskey(al_single.ports, :b) && haskey(al_single.ports, :a)

        # MULTI-DEF output → NOT aliased (`b=a; b=c` are alternative producers, not `a===c`)
        @kernel al_multi(a, c) = begin
            b = a
            b = c
        end
        gm = RKS.kernel_graph(al_multi)
        @test RKS.canon_id(gm, al_multi.ports[:b].id) != RKS.canon_id(gm, al_multi.ports[:a].id)
        @test RKS.canon_id(gm, al_multi.ports[:b].id) != RKS.canon_id(gm, al_multi.ports[:c].id)

        # TYPED MISMATCH → uncertain, keep the ordinary identity recipe (no collapse)
        @kernel al_typed(a::Float64) = begin
            b::Float32 = a
        end
        gt = RKS.kernel_graph(al_typed)
        @test RKS.canon_id(gt, al_typed.ports[:b].id) != RKS.canon_id(gt, al_typed.ports[:a].id)

        # REVERSE/TRANSITIVE → no cycle; canon stable (`d=b=a` all collapse to a's class)
        @kernel al_rev(a) = begin
            b = a
            d = b
        end
        gr = RKS.kernel_graph(al_rev)
        ca = RKS.canon_id(gr, al_rev.ports[:a].id)
        @test RKS.canon_id(gr, al_rev.ports[:b].id) == ca
        @test RKS.canon_id(gr, al_rev.ports[:d].id) == ca

        # `_kernel_alias!` bumps Graph.version on a real alias mutation (like CSE/merge), and the
        # canonical target's PRODUCER index stays correct before/after the alias.
        g = ReactiveKernels.Graph()
        va = ReactiveKernels.value!(g, :a, Float64)
        vb = ReactiveKernels.value!(g, :b, Float64)
        ReactiveKernels.add!(g; inputs = (), outputs = (va,), op = () -> 0.0,
                             cost = 1.0, cse_key = nothing, effectful = false)
        prod_before = length(RKS.producers_of(g, RKS.canon_id(g, va.id)))
        ver0 = g.version
        RKS._kernel_alias!(g, vb, va, identity, 1.0)     # b ≡ a
        @test g.version == ver0 + 1                       # real canonical mutation bumped version
        @test RKS.canon_id(g, vb.id) == RKS.canon_id(g, va.id)
        # a's producer index is unchanged; b now resolves to a's producer via canon
        @test length(RKS.producers_of(g, RKS.canon_id(g, va.id))) == prod_before
        @test length(RKS.producers_of(g, RKS.canon_id(g, vb.id))) == prod_before
        # a NO-OP alias (already same class) does NOT bump the version
        ver1 = g.version
        RKS._kernel_alias!(g, vb, va, identity, 1.0)
        @test g.version == ver1
    end

    @testset "public exact-identity effect descriptors (@rk_* + built-in randn!/lmul!)" begin
        # RK-core built-in RNG/effect primitives for refresh_momentum!!
        rn = RKS._kernel_primitive_effect(Random.randn!)
        @test rn.kind === :rng && rn.order === :ordered && rn.rng_arg == 1
        @test rn.writes == (2,) && rn.reads == (1,) && rn.result_alias == 2
        lm = RKS._kernel_primitive_effect(LinearAlgebra.lmul!)
        @test lm.kind === :effect && lm.writes == (2,) && lm.reads == (1, 2) && lm.result_alias == 2

        # a DECLARED pure helper is captured detached as :declared_effect, IDENTITY-derived token
        caps = RKS.kernel_callee_registrations(critowner)
        dec = only(c for c in caps if c.registration.kind === :declared_effect)
        @test dec.target === crit
        @test dec.registration.primitive_effect.kind === :pure
        @test dec.registration.primitive_effect.token === typeof(crit)
        @test !RKS.kernel_callee_rebound(critowner, dec.ref)          # slot resolves → not rebound
        # the closure ADMITS the declared helper over self places (no opaque reject); x owned via fill!
        @test :x in RKS._kernel_factory_owned_authoritative(critowner)
        # an UNDECLARED helper on a self place REJECTS by exact name
        @test_throws RKS._KernelFactoryReject RKS._kernel_factory_owned_authoritative(undeclowner)

        # same-NAME cross-module helpers → DISTINCT identity-derived tokens (no collision)
        @test RKS._kernel_declared_effect(CritA.f).token !== RKS._kernel_declared_effect(CritB.f).token
        @test RKS._kernel_declared_effect(CritA.f).token === typeof(CritA.f)
        # @rk_borrows marks the result as borrowing ALL actuals; @rk_rng names the RNG position
        @test RKS._kernel_declared_effect(borrowfn).borrows == (1,) &&
              RKS._kernel_declared_effect(borrowfn).kind === :pure
        @test RKS._kernel_declared_effect(rngfn).kind === :rng &&
              RKS._kernel_declared_effect(rngfn).rng_arg == 1

        # WRONG ARITY: crit declared arity 2, called with 1 positional → deterministic reject
        @test_throws RKS._KernelFactoryReject RKS._kernel_factory_owned_authoritative(wrongarity)

        # REBIND: HelperMod.hot captured declared; rebinding HelperMod to a module without `hot`
        # reads as rebound (same detached-identity re-resolution as a primitive/registered callee)
        hc = only(c for c in RKS.kernel_callee_registrations(hotowner)
                  if c.registration.kind === :declared_effect)
        @test hc.target === Helpers.hot
        @test !RKS.kernel_callee_rebound(hotowner, hc.ref)
        @eval HelperMod = NoHot
        @test RKS.kernel_callee_rebound(hotowner, hc.ref)
        @eval HelperMod = Helpers

        # DESCRIPTOR DRIFT (RK 04:29/04:30): re-declaring the SAME identity with different semantics
        # (pure/arity-2 → rng) must read REBOUND via the WHOLE-descriptor comparison, even though
        # `typeof(driftfn)` is unchanged — and WITHOUT mutating the captured snapshot.
        dcap = only(c for c in RKS.kernel_callee_registrations(driftowner)
                    if c.registration.kind === :declared_effect)
        @test !RKS.kernel_rebound(dcap.registration, driftfn)     # descriptor matches
        capsnap = dcap.registration.primitive_effect
        @eval @rk_rng driftfn 2 1                                 # same identity, drifted semantics
        @test RKS.kernel_rebound(dcap.registration, driftfn)      # whole-descriptor drift → rebound
        @test dcap.registration.primitive_effect === capsnap      # captured snapshot UNMUTATED
        @test dcap.registration.primitive_effect.kind === :pure
        @eval @rk_pure driftfn 2                                  # restore

        # constructor/macro VALIDATION: rngpos out of range and duplicate/out-of-range positions reject
        @test_throws ArgumentError RKS._effect_rng(rngfn, 2, 3)   # rngpos 3 ∉ 1:2
        @test_throws ArgumentError RKS._effect_check(2, (0,), "x")   # position 0 out of range
        @test_throws ArgumentError RKS._effect_check(3, (1, 1), "x") # duplicate positions
        @test_throws ArgumentError RKS._effect_check(0, (), "x")     # zero arity rejected (positive only)
        @test_throws ArgumentError RKS._effect_check(-1, (), "x")    # negative arity rejected
        # Random.rand 2-positional ordered-RNG built-in (rand(rng, Bool) in step!)
        rd = RKS._kernel_primitive_effect(Random.rand)
        @test rd.kind === :rng && rd.rng_arg == 1 && rd.writes == () && rd.reads == (1, 2)
    end

    @testset "REAL nuts_state authoritative ownership (corrected ccb35d3 fixture, RK pt5/7, 05:18/26)" begin
        # Eval the CORRECTED benchmark/nuts_kernel_authoring_fixture.jl EXACTLY as authored — its OWN
        # seven @rk_* declarations run (no manual overlay, no method-overwrite), so the gate exercises
        # the authored source. Selective import dodges the exported-name collision.
        RN = Module(:RealNuts)
        Core.eval(RN, :(using ReactiveKernels: @kernel, @node, partial, copy!!,
                                               @rk_pure, @rk_borrows, @rk_rng))
        Core.eval(RN, :(using LinearAlgebra, LogExpFunctions, Random))
        fixpath = normpath(joinpath(@__DIR__, "..", "benchmark", "nuts_kernel_authoring_fixture.jl"))
        stmts = filter(s -> !(s isa LineNumberNode), Meta.parseall(read(fixpath, String)).args)
        for st in stmts
            (st isa Expr && st.head === :using) && continue
            Core.eval(RN, st)
        end
        ns = RN.nuts_state
        # all SEVEN authored declared-helper identities are captured (incl. min1exp)
        for (h, k) in ((:finiteorneginf, :pure), (:min1exp, :pure), (:badd, :pure),
                       (:randbernoullilog, :rng), (:logswapprob, :pure),
                       (:compute_criterion, :pure))
            d = RKS._kernel_declared_effect(getfield(RN, h))
            @test d !== nothing && d.kind === k
        end
        # step_f → leapfrog!, stats_f → the registered nuts_stats! diagnostics callback
        fr = Dict(:step_f => RKS.kernel_registration(RN.leapfrog!),
                  :stats_f => RKS.kernel_registration(RN.nuts_stats!))
        owned = RKS._kernel_factory_owned_authoritative(ns; field_regs = fr)
        shared = RKS._kernel_factory_shared(ns; field_regs = fr)
        # EXACT owned: init/fwd/bwd + trees/proposals + control flags + DIAGNOSTICS. nuts_stats! writes
        # n_steps + acceptance_rate (via stats_f(__self__)); reached_depth is a direct step! write;
        # reset! also writes the diagnostics — so the real fixture confounds the callback path (a
        # dedicated synthetic discriminator below isolates the _SelfRef subject-write mapping).
        @test owned == Set((:init, :fwd, :bwd, :trees, :proposals,
                            :gofwd, :may_sample, :may_continue, :dham, :diverged,
                            :n_steps, :reached_depth, :acceptance_rate))
        @test shared == Set((:step_f, :max_depth, :min_dham, :stats_f))
        @test isempty(intersect(owned, shared))
        @test union(owned, shared) == Set{Symbol}(RKS.kernel_port_names(ns))
        # DIRECT (RK 05:38): the three diagnostics are OWNED (via reset/step + the stats callback
        # closure), while step_f/max_depth/min_dham/stats_f remain SHARED.
        for d in (:n_steps, :reached_depth, :acceptance_rate)
            @test d in owned && !(d in shared)
        end
        for s in (:step_f, :max_depth, :min_dham, :stats_f)
            @test s in shared && !(s in owned)
        end
        # LOAD-BEARING (RK 05:39): nuts_stats! writes EXACTLY {n_steps, acceptance_rate}
        # (order-insensitive), and stats_f(__self__) maps those subject roots onto OWNER fields.
        @test Set(RKS.kernel_registration(RN.nuts_stats!).write_roots) == Set((:n_steps, :acceptance_rate))
        @test :n_steps in owned && :acceptance_rate in owned
        # reached_depth is a DIRECT step! write (NOT the stats callback) — still owned with stats_f
        # a resolved no-effect, whereas the callback-only roots are owned via reset/step diagnostics.
        owned_ns = RKS._kernel_factory_owned_authoritative(ns;
            field_regs = Dict(:step_f => RKS.kernel_registration(RN.leapfrog!), :stats_f => nothing))
        @test :reached_depth in owned_ns
        # euclidean_phasepoint is pot_f-free (grad_f, metric, pos, mom)
        @test :pot_f ∉ RKS._kernel_all_ports(RN.euclidean_phasepoint)
        @test :grad_f in RKS._kernel_all_ports(RN.euclidean_phasepoint)
    end

    @testset "_SelfRef subject-write: callback root → owner field (RK 05:40 discriminator)" begin
        # NON-VACUOUS (the real fixture's reset! confounds the callback path): `x`/`y` are otherwise
        # unwritten ports; a registered callback `cb(__self__)` writes ONLY the subject root `x`.
        cbreg = RKS.kernel_registration(cbwrites!)
        @test Set(cbreg.write_roots) == Set((:x,))                     # callback writes subject.x only
        # with the callback RESOLVED: `cb(__self__)` maps the subject write-root onto owner field x →
        # x OWNED, y stays SHARED.
        owned_cb = RKS._kernel_factory_owned_authoritative(cbowner; field_regs = Dict(:cb => cbreg))
        @test :x in owned_cb && !(:y in owned_cb)
        # with the callback a resolved NO-EFFECT (nothing): nothing writes x/y → BOTH shared.
        owned_none = RKS._kernel_factory_owned_authoritative(cbowner; field_regs = Dict(:cb => nothing))
        @test !(:x in owned_none) && !(:y in owned_none)
    end

    @testset "concrete no-Ref construction: isolation, independent masks, F32/F64 (step 5)" begin
        for T in (Float32, Float64)
            buf = T[1, 2, 3]
            a = RKS._kernel_construct_owned(Val(:ep), (T(0), buf))
            b = RKS._kernel_construct_owned(Val(:ep), (T(0), buf))
            # OWNED buffers isolated: two instances never alias, neither aliases the caller input
            @test RKS._owner_slot(a, Val(2)) !== RKS._owner_slot(b, Val(2))
            @test RKS._owner_slot(a, Val(2)) !== buf
            @test RKS._owner_slot(a, Val(2)) == buf               # value-equal (deepcopy)
            # F32/F64 PRESERVED; NO Ref / widening / Any
            @test eltype(RKS._owner_slot(a, Val(2))) === T && RKS._owner_slot(a, Val(1)) isa T
            @test fieldtypes(typeof(RKS.owner_slots(a))) == (T, Vector{T})
            @test !any(t -> t <: Ref, fieldtypes(typeof(RKS.owner_slots(a))))
            # INDEPENDENT currentness masks per instance (init/fwd/bwd carry their own)
            @test RKS._owner_current(a, Val(1)) && RKS._owner_current(b, Val(1))
            RKS._owner_kill!(a, Val(1))
            @test !RKS._owner_current(a, Val(1)) && RKS._owner_current(b, Val(1))
            RKS._owner_bless!(a, Val(1))
            @test RKS._owner_current(a, Val(1))
            # copy!! currentness transfer: dest inherits src's mask
            RKS._owner_kill!(b, Val(2))
            RKS._owner_copy_current!(a, b)
            @test !RKS._owner_current(a, Val(2))
            # mask ops are ALLOCATION-FREE (0-B) after warmup
            @test (@allocated RKS._owner_kill!(a, Val(1))) == 0
            @test (@allocated RKS._owner_current(a, Val(2))) == 0
        end

        # >WORD_BITS DISCRIMINATOR (RK pt1/04:50): a layout wider than one machine word uses W>1
        # words and NEVER shift-wraps — a slot in word 2 is independent of a slot in word 1.
        WB = RKS._WORD_BITS
        nbig = WB + 6
        big = RKS._kernel_construct_owned(Val(:big), ntuple(i -> Float64(i), nbig))
        @test length(RKS.owner_current_mask(big)) == 2            # W = cld(WB+6, WB) = 2
        @test RKS._owner_current(big, Val(WB + 1)) && RKS._owner_current(big, Val(1))
        RKS._owner_kill!(big, Val(WB + 1))                        # kill a word-2 slot
        @test !RKS._owner_current(big, Val(WB + 1)) && RKS._owner_current(big, Val(1))
        RKS._owner_bless!(big, Val(WB + 1))
        @test RKS._owner_current(big, Val(WB + 1))
        # cross-word kill/bless are ALLOCATION-FREE (0-B) — warm the exact Val specialization first
        RKS._owner_kill!(big, Val(WB + 2)); RKS._owner_bless!(big, Val(WB + 2))
        @test (@allocated RKS._owner_kill!(big, Val(WB + 2))) == 0
        RKS._owner_bless!(big, Val(WB + 3)); RKS._owner_kill!(big, Val(WB + 3))
        @test (@allocated RKS._owner_bless!(big, Val(WB + 3))) == 0

        # NONCONTIGUOUS entry-current mask: EXACTLY slots {1, WB+1} set (arbitrary subset across
        # words), nothing between — proves the mask is built from exact indices, not a low-prefix.
        m = RKS._owner_mask(nbig, [1, WB + 1])
        nc = RKS._OwnerState{:nc}(ntuple(i -> Float64(i), nbig), m)
        @test RKS._owner_current(nc, Val(1)) && RKS._owner_current(nc, Val(WB + 1))
        @test !RKS._owner_current(nc, Val(2)) && !RKS._owner_current(nc, Val(WB)) &&
              !RKS._owner_current(nc, Val(WB + 2))
        # copy!! currentness transfer requires IDENTICAL layout type; a different Token/slots/width
        # rejects (nc and big share a width but differ in Token; a small state differs in all).
        @test_throws MethodError RKS._owner_copy_current!(big, nc)    # different Token
        small = RKS._kernel_construct_owned(Val(:sm), (1.0, [2.0]))
        @test_throws MethodError RKS._owner_copy_current!(small, big) # different slots/width
    end

    @testset "shared authority: identity/mutable split + one-across-endpoints (RK pt4/5, 05:06)" begin
        gradf = CountingGrad(0)                 # EXTERNAL identity authority (stateful functor)
        metric = [1.0, 2.0]                      # PER-SAMPLER mutable authority
        # two independent SAMPLER instances
        s1 = RKS._kernel_construct_shared(Val(:auth), (gradf,), (metric, 3.0))
        s2 = RKS._kernel_construct_shared(Val(:auth), (gradf,), (metric, 3.0))
        # external grad_f is the SAME identity across samplers — never deep-copied (RK 05:06)
        @test RKS._shared_slot(s1, Val(1)) === gradf && RKS._shared_slot(s2, Val(1)) === gradf
        gradf(0)                                 # its counter is shared (identity, not a copy)
        @test RKS._shared_slot(s2, Val(1)).n == 1
        # per-sampler mutable metric/chol/@node are DISTINCT across samplers, and NOT the source
        @test RKS._shared_slot(s1, Val(2)) !== RKS._shared_slot(s2, Val(2))
        @test RKS._shared_slot(s1, Val(2)) !== metric

        # within ONE sampler, init/fwd/bwd owned endpoints share the SAME s1 by identity, isolate owned
        endpoints, sh = RKS._kernel_construct_group(Val(:ep), 3, (0.0, [1.0, 2.0, 3.0]), s1)
        @test endpoints[1] !== endpoints[2] && endpoints[2] !== endpoints[3]
        @test RKS._owner_slot(endpoints[1], Val(2)) !== RKS._owner_slot(endpoints[2], Val(2))
        @test sh === s1
        # a metric mutation is seen through the one shared object; its closure kills once
        RKS.shared_slots(sh)[2][1] = 99.0
        @test RKS._shared_slot(sh, Val(2))[1] == 99.0
        @test RKS._shared_current(sh, Val(2)); RKS._shared_kill!(sh, Val(2))
        @test !RKS._shared_current(sh, Val(2))
    end

    @testset "canonical-port superset storage: owned/shared roles, no duplicate, 0-B (RK 05:11/12)" begin
        # WRITER integrator: canonical slots {1,2} OWNED (scalar + array), {3} SHARED (array authority)
        ow = RKS._CanonOwned3(1.0, [2.0, 3.0], nothing, RKS._owner_mask(3, [1, 2]))
        sw = RKS._CanonShared3(nothing, nothing, [4.0, 5.0], RKS._owner_mask(3, [3]))
        # exactly ONE physical value per canonical id — the OTHER role's field is Nothing
        @test RKS._canon_slot(ow, Val(3)) === nothing && RKS._canon_slot(sw, Val(1)) === nothing
        @test RKS._canon_slot(ow, Val(1)) == 1.0 && RKS._canon_slot(sw, Val(3)) == [4.0, 5.0]
        @test (@inferred RKS._canon_slot(ow, Val(1))) == 1.0        # Val accessor type-stable
        # READ-ONLY integrator: SAME struct family, DIFFERENT selected layout — {} owned, {1,2,3} shared
        orr = RKS._CanonOwned3(nothing, nothing, nothing, RKS._owner_mask(3, Int[]))
        srr = RKS._CanonShared3(6.0, [7.0], [8.0], RKS._owner_mask(3, [1, 2, 3]))
        @test typeof(orr).name === typeof(ow).name                 # same TYPE FAMILY (no runtime emission)
        @test all(RKS._canon_slot(orr, Val(i)) === nothing for i in 1:3)   # nothing owned (read-only)
        @test RKS._canon_slot(srr, Val(1)) == 6.0                  # the physical value is in the shared role
        # current mask marks ONLY selected canonical fields
        @test RKS._canon_current_mask(ow) == RKS._owner_mask(3, [1, 2])
        @test RKS._canon_current_mask(orr) == RKS._owner_mask(3, Int[])
        # 0-B per-slot: scalar set + array copy + array identity, warmed, exact 0 B, mixed F32/F64
        for T in (Float32, Float64)
            o = RKS._CanonOwned3(T(1), T[2, 3], nothing, RKS._owner_mask(3, [1, 2]))
            src = RKS._CanonOwned3(T(9), T[7, 8], nothing, RKS._owner_mask(3, [1, 2]))
            v5 = T(5)                                    # pre-construct (dynamic `T(5)` would box)
            _canonset0b(o, v5); _canoncopy0b(o, src)                                   # warmup
            @test (@allocated _canonset0b(o, v5)) == 0                                 # scalar set 0-B
            buf = RKS._canon_slot(o, Val(2))
            @test (@allocated _canoncopy0b(o, src)) == 0                               # array copy 0-B
            @test RKS._canon_slot(o, Val(2)) === buf                                   # array identity kept
            @test RKS._canon_slot(o, Val(2)) == T[7, 8]                                # values transferred
        end

        # PACKAGE-LOAD ARITY FAMILY (RK 05:23/05:24): construction picks N from the plan at COMPILE
        # time (Val-role dispatch, arity from the Tuple TYPE) — fully type-stable / @inferred; the
        # family is predeclared at load (no runtime emission), a layout beyond max REJECTS.
        o2 = @inferred RKS._canon_construct(Val(:owned), (1.0, [2.0]), RKS._owner_mask(2, [1, 2]))
        @test o2 isa RKS._CanonOwned2 && RKS._canon_slot(o2, Val(1)) == 1.0
        s2 = @inferred RKS._canon_construct(Val(:shared), (nothing, [3.0]), RKS._owner_mask(2, [2]))
        @test s2 isa RKS._CanonShared2 && RKS._canon_slot(s2, Val(1)) === nothing
        @test RKS._canon_owned_type(Val(5)) === RKS._CanonOwned5   # world-age-clean compile-time lookup
        @test RKS._canon_shared_type(Val(1)) === RKS._CanonShared1
        # a layout wider than the predeclared family arity rejects deterministically
        wide = ntuple(i -> Float64(i), RKS._CANON_MAXN + 1)
        @test_throws RKS._KernelFactoryReject RKS._canon_construct(Val(:owned), wide, RKS._owner_mask(length(wide)))

        # SCALAR-vs-BUFFER classification for poc expression emission (RK 05:30): from the concrete
        # field TYPE by Val index (literal fieldtype), NOT names — Int/Bool/Float scalars vs vectors.
        mix = RKS._canon_construct(Val(:owned), (1.0, [2.0, 3.0], 4, true), RKS._owner_mask(4))
        @test RKS._canon_slot_kind(mix, Val(1)) === :scalar && RKS._canon_slot_type(mix, Val(1)) === Float64
        @test RKS._canon_slot_kind(mix, Val(2)) === :buffer && RKS._canon_slot_type(mix, Val(2)) === Vector{Float64}
        @test RKS._canon_slot_kind(mix, Val(3)) === :scalar && RKS._canon_slot_type(mix, Val(3)) === Int
        @test RKS._canon_slot_kind(mix, Val(4)) === :scalar && RKS._canon_slot_type(mix, Val(4)) === Bool
        @test (@inferred RKS._canon_slot_kind(mix, Val(2))) === :buffer   # type-stable classification
    end

    @testset "pgrad! destination-aware applier: atomic pot+grad, per-endpoint dest, 0-B (RK 05:02)" begin
        b = RKS._grad_binding(pgrad_ex!, Val(1), Val(2))     # dest = grad slot 1, pot = slot 2
        @test RKS.grad_binding_callable(b) === pgrad_ex!      # IDENTITY in the prepared handle (not Recipe.op)
        for T in (Float32, Float64)
            o = RKS._CanonOwned2(T[0, 0], T(0), RKS._owner_mask(2, [1, 2]))
            gradbuf = RKS._canon_slot(o, Val(1))
            pos = T[1, 2]
            RKS._kernel_apply_grad!(b, o, pos)               # ONE call seeds grad + pot atomically
            @test RKS._canon_slot(o, Val(1)) === gradbuf     # owned grad dest identity kept (no realloc)
            @test RKS._canon_slot(o, Val(1)) == T[2, 4]      # grad written in place
            @test RKS._canon_slot(o, Val(2)) == T(5)         # pot = sum(abs2,[1,2]) seeded from same call
            # a SECOND endpoint uses the SAME typed callable but its OWN grad destination
            o2 = RKS._CanonOwned2(T[0, 0], T(0), RKS._owner_mask(2, [1, 2]))
            RKS._kernel_apply_grad!(b, o2, T[3, 4])
            @test RKS._canon_slot(o, Val(1)) !== RKS._canon_slot(o2, Val(1))   # distinct dest per endpoint
            # ALLOCATION-FREE (0-B) — one call, no unary wrapper, no scratch (warmed)
            _applygrad0b(b, o, pos)
            @test (@allocated _applygrad0b(b, o, pos)) == 0
        end
    end

    @testset "ABSOLUTE superset slot indexing + endpoint construction (RK 05:48/05:49)" begin
        integ = RKS.kernel_registration(leapfrog_ep!)
        plan = RKS._kernel_factory_endpoint_plan(phasepoint_ep, integ)
        gp = RKS.kernel_graph(phasepoint_ep)
        canonf(n) = RKS.canon_id(gp, phasepoint_ep.ports[n].id)
        canons, roles = RKS.kernel_plan_superset(plan)
        # shared pot_f/grad_f/metric precede owned pos/mom → pos/mom get ABSOLUTE positions ≥ 4,
        # NOT per-role slot 1/2 (the bug RK 05:49 reproduced).
        ps = RKS.kernel_plan_slot(plan, :pos).slot
        ms = RKS.kernel_plan_slot(plan, :mom).slot
        @test canons[ps] == canonf(:pos) && roles[ps] === :owned
        @test canons[ms] == canonf(:mom) && roles[ms] === :owned
        @test RKS.kernel_plan_field(plan, canonf(:pos)) == (:owned, ps)
        @test ps >= 4 && ms >= 4
        @test length(unique(canons)) == length(canons)   # ONE absolute index per canonical Value
        # CONSTRUCT owned + shared objects from a distinct-per-canon value map; exercise _canon_slot.
        vals = Dict{Int,Any}(c => Float64(c) for c in canons)
        ow, sh = RKS._kernel_construct_endpoint(Val(:ep), plan, vals)
        for s in RKS.kernel_plan_slots(plan)
            if s.role === :owned
                @test RKS._canon_slot(ow, Val(s.slot)) == Float64(s.canon)
                @test RKS._canon_slot(sh, Val(s.slot)) === nothing
            else
                @test RKS._canon_slot(sh, Val(s.slot)) == Float64(s.canon)
                @test RKS._canon_slot(ow, Val(s.slot)) === nothing
            end
        end
    end

    @testset "pgrad! throwing/partial → neither output current, retry reruns (RK 05:47)" begin
        badpg!(dest, pos) = (dest[1] = 99.0; error("boom"))    # partial write then throw
        b = RKS._grad_binding(badpg!, Val(1), Val(2))
        o = RKS._CanonOwned2([0.0, 0.0], 0.0, RKS._owner_mask(2, Int[]))   # dest/pot NOT current
        @test !RKS._canon_current(o, Val(1)) && !RKS._canon_current(o, Val(2))
        @test_throws ErrorException RKS._kernel_apply_grad!(b, o, [1.0, 2.0])
        # bless is AFTER the call, which threw → neither output became current
        @test !RKS._canon_current(o, Val(1)) && !RKS._canon_current(o, Val(2))
        # a RETRY with a good pgrad! reruns cleanly and blesses BOTH
        goodpg!(dest, pos) = (dest .= 2 .* pos; sum(pos))
        RKS._kernel_apply_grad!(RKS._grad_binding(goodpg!, Val(1), Val(2)), o, [1.0, 2.0])
        @test RKS._canon_current(o, Val(1)) && RKS._canon_current(o, Val(2))
        @test RKS._canon_slot(o, Val(1)) == [2.0, 4.0]
    end

    @testset "construction seam — ORDERED all-outputs (authored order) + producer-owned subset seam (RK 07:09)" begin
        integ = RKS.kernel_registration(gradonly_step!)
        plan = RKS._kernel_factory_endpoint_plan(gradonly_ep, integ)
        g = RKS.kernel_graph(gradonly_ep)
        canonof(n) = RKS.canon_id(g, gradonly_ep.ports[n].id)
        cpot, cdpot = canonof(:pot), canonof(:dpot_dpos)
        ro = RKS.kernel_plan_recipe_outputs(plan)
        po = RKS.kernel_plan_producer_owned(plan)
        seam = RKS.kernel_plan_recipe_seam(plan)
        @test ro isa Tuple && all(e -> e isa Tuple{Int,<:Tuple}, ro)
        @test [e[1] for e in seam] == collect(RKS.kernel_plan_recipes(plan))    # execution order
        @test all(e -> length(e) == 4, seam)                                    # (rid, in, allout, owned)
        gr = Dict(RKS.kernel_plan_producer(plan))[cpot]
        grout = only(os for (rid, os) in ro if rid == gr)
        grown = only(os for (rid, os) in po if rid == gr)
        # EXACT AUTHORED POSITIONAL ORDER from r.outputs — NOT reconstructed by sorted ids
        grrecipe = only(r for r in g.recipes if r.id == gr)
        @test grout == Tuple(RKS.canon_id(g, v.id) for v in grrecipe.outputs)   # authored positional order
        @test Set(grown) ⊆ Set(grout) && Set(grown) == Set((cpot, cdpot))       # pot_f-free -> both owned
        @test grown == Tuple(c for c in grout if c in Set(grown))               # owned subset keeps order
        # the type-stable binding derives pot/dest from these ORDERED outputs (not sorted ids)
        mkvals() = Dict{Int,Any}(canonof(:pos) => [1.0, 2.0], cpot => 0.0, cdpot => [0.0, 0.0], canonof(:grad_f) => pgrad_ex!)
        owp, _ = RKS._kernel_construct_endpoint(Val(:g), plan, mkvals(), (canonof(:grad_f),))
        b = RKS._grad_binding_from_plan(plan, Val(gr), pgrad_ex!, owp)
        @test b.dest === Val(RKS.kernel_plan_field(plan, cdpot)[2])             # buffer output -> dest
        @test b.pot === Val(RKS.kernel_plan_field(plan, cpot)[2])              # scalar output -> pot
        # DETERMINISTIC synthetic COLLATERAL (RK 07:15, non-vacuous): R1 (id 1) produces x,y; R2 (id 2)
        # EMITS y (collateral, owned by R1) then z (owned). Collateral y is ASSIGNED (recipe_outputs) but
        # NEVER blessed (excluded from producer_owned), and the grad binding on R2 REJECTS it.
        sl = [RKS._PlanSlot((:y,), 102, :owned, 1), RKS._PlanSlot((:z,), 103, :owned, 2)]
        slot_sig = Tuple((s.path, s.canon, s.role, s.slot) for s in sl)
        producer = ((102, 1), (103, 2))                                        # y owned by R1, z by R2
        recipes = (1, 2); recipe_inputs = ((1, ()), (2, (102,)))
        recipe_outputs = ((1, (101, 102)), (2, (102, 103)))                    # R2 emits collateral y then z
        key = (:syn, slot_sig, (), producer, recipes, recipe_inputs, recipe_outputs)
        synp = RKS._KernelPlan(key, Tuple(sl), (), producer, recipes, (102, 103), recipe_inputs, recipe_outputs)
        @test RKS.kernel_plan_recipe_outputs(synp) == recipe_outputs           # ordered ALL outputs
        r2out = only(o for (r, o) in RKS.kernel_plan_recipe_outputs(synp) if r == 2)
        r2own = only(o for (r, o) in RKS.kernel_plan_producer_owned(synp) if r == 2)
        @test r2out == (102, 103) && r2own == (103,)                           # y assigned, only z owned
        @test setdiff(Set(r2out), Set(r2own)) == Set((102,))                   # NON-VACUOUS collateral = y
        syno = RKS._CanonOwned2(Float64[0, 0], 0.0, RKS._owner_mask(2, Int[]))
        @test_throws RKS._KernelFactoryReject RKS._grad_binding_from_plan(synp, Val(2), pgrad_ex!, syno)  # rejects collateral
    end

    @testset "construction seam — SHARED prepared-factory core (de-drift: _prepare_factory ≡ helper; allow_destination gate) (POC 14:55)" begin
        integ = RKS.kernel_registration(leapfrog_ep!)
        plan, ops = RKS._kernel_factory_endpoint_plan(euclidean_ep, integ; with_ops = true)
        pf1 = RKS._prepare_factory(euclidean_ep, integ)                          # public entry (calls the helper)
        pf2 = RKS._prepared_factory_from_plan(integ.token, plan, ops; allow_destination = true)  # helper directly
        # BYTE-IDENTICAL: same concrete type + token + grad recipe + external authorities + handle tuple
        @test typeof(pf1) === typeof(pf2)
        @test RKS.kernel_prepared_token(pf1) === RKS.kernel_prepared_token(pf2) === integ.token
        @test RKS.kernel_prepared_grad_recipe(pf1) === RKS.kernel_prepared_grad_recipe(pf2)
        @test RKS.kernel_prepared_external(pf1) == RKS.kernel_prepared_external(pf2)
        @test length(pf1.handles) == length(pf2.handles) &&
              all(typeof(a) === typeof(b) for (a, b) in zip(pf1.handles, pf2.handles))
        # allow_destination=false REJECTS an endpoint carrying a :destination (external-grad) recipe — the gate
        # POC's free-stateful `_prepare_stateful` relies on (a free kernel must not carry an external grad).
        @test_throws RKS._KernelFactoryReject RKS._prepared_factory_from_plan(integ.token, plan, ops; allow_destination = false)
    end

    @testset "construction seam — PREPARED plan (zero planner/graph per instance), HAVE-only, type-stable grad, mask==entry_current (RK 05:47/06:49/06:55)" begin
        integ = RKS.kernel_registration(gradonly_poison_step!)   # ISOLATED endpoint — poison never leaks
        g = RKS.kernel_graph(gradonly_poison_ep)
        canonof(n) = RKS.canon_id(g, gradonly_poison_ep.ports[n].id)
        cpos, cpot, cdpot, cgf = canonof(:pos), canonof(:pot), canonof(:dpot_dpos), canonof(:grad_f)
        plan0 = RKS._kernel_factory_endpoint_plan(gradonly_poison_ep, integ)
        prod = Dict(RKS.kernel_plan_producer(plan0))
        @test haskey(prod, cpot) && haskey(prod, cdpot) && prod[cpot] == prod[cdpot]  # ONE grad recipe
        gr = prod[cpot]
        # PREPARE ONCE (SELF-DISCOVERING — no manual grad_recipe/external): the single plan(graph) call is
        # here; the captured factory discovers the destination recipe + external identity from source shape.
        pf = RKS._prepare_factory(gradonly_poison_ep, integ)
        @test RKS.kernel_prepared_grad_recipe(pf) == gr                          # discovered the grad recipe
        @test cgf in RKS.kernel_prepared_external(pf)                            # discovered grad_f as external
        key_before = RKS.kernel_plan_key(RKS.kernel_prepared_plan(pf))
        canons, _ = RKS.kernel_plan_superset(RKS.kernel_prepared_plan(pf)); N = length(canons)
        pos_slot = RKS.kernel_plan_field(plan0, cpos)[2]
        pot_slot = RKS.kernel_plan_field(plan0, cpot)[2]
        dpot_slot = RKS.kernel_plan_field(plan0, cdpot)[2]
        # POISON the live (isolated) graph AFTER preparation — an alternate producer of dpot_dpos
        RKS.add!(g; inputs = (gradonly_poison_ep.ports[:pos],), outputs = (gradonly_poison_ep.ports[:dpot_dpos],),
                 op = identity, cost = 1.0, cse_key = nothing, effectful = false)
        # CONSTRUCT two instances from the captured plan — ZERO plan()/graph access; poison is invisible
        mkvals() = Dict{Int,Any}(cpos => [1.0, 2.0], cpot => 0.0, cdpot => [0.0, 0.0], cgf => pgrad_ex!)
        ow1, _ = RKS._construct_prepared(pf, mkvals(), pgrad_ex!, [1.0, 2.0])
        ow2, _ = RKS._construct_prepared(pf, mkvals(), pgrad_ex!, [1.0, 2.0])
        @test RKS.kernel_plan_key(RKS.kernel_prepared_plan(pf)) === key_before   # captured key unchanged
        # HAVE-only start proven via a bare construct (no grad yet): pos current, pot/dpot DIRTY
        owr, _ = RKS._kernel_construct_endpoint(Val(RKS.kernel_prepared_token(pf)),
                                                RKS.kernel_prepared_plan(pf), mkvals(), (cgf,))
        @test RKS._canon_current(owr, Val(pos_slot))
        @test !RKS._canon_current(owr, Val(pot_slot)) && !RKS._canon_current(owr, Val(dpot_slot))
        # TYPE-STABLE binding derived from the detached selected recipe outputs (RK 06:49)
        b = @inferred RKS._grad_binding_from_plan(RKS.kernel_prepared_plan(pf), Val(gr), pgrad_ex!, owr)
        @test isconcretetype(typeof(b)) && b.dest === Val(dpot_slot) && b.pot === Val(pot_slot)
        # both prepared instances: identical layout/values; ONE apply → FINAL mask == entry_current
        ecowned = sort([RKS.kernel_plan_field(plan0, c)[2]
                        for c in RKS.kernel_plan_entry_current(plan0)
                        if RKS.kernel_plan_field(plan0, c)[1] === :owned])
        for ow in (ow1, ow2)
            @test RKS._canon_slot(ow, Val(dpot_slot)) == [2.0, 4.0] && RKS._canon_slot(ow, Val(pot_slot)) == 5.0
            @test RKS._canon_current_mask(ow) == RKS._owner_mask(N, ecowned)     # mask == entry_current
        end
        @test RKS._canon_slot(ow1, Val(dpot_slot)) !== RKS._canon_slot(ow2, Val(dpot_slot))  # isolated buffers
    end

    @testset "construction seam — copy owned children (ONE shared, ZERO extra pgrad, exception-safe, 0-B) (RK 06:53/06:56/06:59)" begin
        integ = RKS.kernel_registration(gradonly_step!)
        g = RKS.kernel_graph(gradonly_ep)
        canonof(n) = RKS.canon_id(g, gradonly_ep.ports[n].id)
        cpos, cpot, cdpot, cgf = canonof(:pos), canonof(:pot), canonof(:dpot_dpos), canonof(:grad_f)
        gr = Dict(RKS.kernel_plan_producer(RKS._kernel_factory_endpoint_plan(gradonly_ep, integ)))[cpot]
        pf = RKS._prepare_factory(gradonly_ep, integ)                            # SELF-DISCOVERING
        @test RKS.kernel_prepared_grad_recipe(pf) == gr && cgf in RKS.kernel_prepared_external(pf)
        plan = RKS.kernel_prepared_plan(pf)
        dpot_slot = RKS.kernel_plan_field(plan, cdpot)[2]
        pot_slot = RKS.kernel_plan_field(plan, cpot)[2]
        tok = RKS.kernel_prepared_token(pf)
        mkvals(pg) = Dict{Int,Any}(cpos => [1.0, 2.0], cpot => 0.0, cdpot => [0.0, 0.0], cgf => pg)
        pg = CountingPgrad(0)
        # the GROUP builds the shared authority ONCE (init's full construct); children are OWNED-ONLY.
        init, sh = RKS._construct_prepared(pf, mkvals(pg), pg, [1.0, 2.0])       # ONE pgrad (count -> 1)
        @test pg.n == 1
        fwd = RKS._kernel_construct_owned_child(Val(tok), plan, mkvals(pg))      # owned-only: NO shared built
        bwd = RKS._kernel_construct_owned_child(Val(tok), plan, mkvals(pg))
        @test pg.n == 1                                                          # constructing children ran NO pgrad
        gbuf = RKS._canon_slot(fwd, Val(dpot_slot))
        @test !RKS._canon_current(fwd, Val(dpot_slot))                          # child dirty pre-seed
        RKS._seed_children!(init, fwd, bwd)
        @test pg.n == 1                                                          # seeding children ran NO pgrad
        for ch in (fwd, bwd)
            @test RKS._canon_current_mask(ch) == RKS._canon_current_mask(init)  # VALIDITY transferred
            @test RKS._canon_slot(ch, Val(dpot_slot)) == [2.0, 4.0] && RKS._canon_slot(ch, Val(pot_slot)) == 5.0
        end
        @test RKS._canon_slot(fwd, Val(dpot_slot)) === gbuf                     # child buffer identity kept
        @test RKS._canon_slot(fwd, Val(dpot_slot)) !== RKS._canon_slot(init, Val(dpot_slot))  # NOT aliased
        # dest === src is a mask+value-preserving no-op
        m = RKS._canon_current_mask(init); v = copy(RKS._canon_slot(init, Val(dpot_slot)))
        RKS._canon_copy_endpoint!(init, init)
        @test RKS._canon_current_mask(init) == m && RKS._canon_slot(init, Val(dpot_slot)) == v
        # whole mixed scalar+buffer copy: BOTH Float32 and Float64 via a function barrier — @inferred,
        # result IDENTITY (returns dest), exact 0-B for each (RK 07:02)
        for T in (Float32, Float64)
            pgT = CountingPgrad(0)
            vvT() = Dict{Int,Any}(cpos => T[1, 2], cpot => zero(T), cdpot => T[0, 0], cgf => pgT)
            it, _ = RKS._construct_prepared(pf, vvT(), pgT, T[1, 2])
            @test pgT.n == 1 && RKS._canon_slot(it, Val(pot_slot)) === T(5)     # F32/F64 scalar preserved
            ch = RKS._kernel_construct_owned_child(Val(tok), plan, vvT())
            RKS._seed_children!(it, ch)                                          # warm the specialization
            @test pgT.n == 1                                                     # child seed ran NO pgrad
            @test (@inferred _copyep0b(ch, it)) === ch                          # result identity, inferred
            @test (@allocated _copyep0b(ch, it)) == 0                           # exact 0-B
        end
        # PRE-VALIDATION reject: a buffer-shape mismatch throws BEFORE any mutation → dest keeps OLD mask
        # AND values (this is the ONLY case values are guaranteed unchanged — RK 07:08)
        bad = RKS._kernel_construct_owned_child(Val(tok), plan,
            Dict{Int,Any}(cpos => [9.0], cpot => 7.0, cdpot => [8.0], cgf => pg))
        premask = RKS._canon_current_mask(bad); badpot = RKS._canon_slot(bad, Val(pot_slot))
        badbuf = copy(RKS._canon_slot(bad, Val(dpot_slot)))
        @test_throws RKS._KernelFactoryReject RKS._canon_copy_endpoint!(bad, init)
        @test RKS._canon_current_mask(bad) == premask                          # mask untouched
        @test RKS._canon_slot(bad, Val(pot_slot)) == badpot                    # values untouched (pre-validated)
        @test RKS._canon_slot(bad, Val(dpot_slot)) == badbuf
        # the copier is CanonOwned-only — shared authority is uncopyable BY TYPE
        @test !hasmethod(RKS._canon_copy_endpoint!, Tuple{RKS._CanonShared, RKS._CanonShared})
    end

    @testset "construction seam — captured executable recipe HANDLES (concrete tuple, modes, provenance) (RK 07:12–07:35)" begin
        pf = RKS._prepare_factory(euclidean_ep, RKS.kernel_registration(leapfrog_ep!))  # isolated pot_f-free
        hs = @inferred RKS.kernel_prepared_handles(pf)                          # field access is inferable
        # a CONCRETE tuple in plan.recipes order — isconcretetype guarantees NO field type is Any anywhere
        @test hs isa Tuple && isconcretetype(typeof(hs)) && length(hs) == 6
        @test all(h -> isconcretetype(typeof(h)) && !any(==(Any), fieldtypes(typeof(h))), hs)
        # (_prepare_factory itself is COLD-ONLY — the discovery loop is not type-stable; the STORED handle
        # tuple it produces is fully concrete, which is what poc indexes literally.)
        modes = [RKS.recipe_handle_mode(h) for h in hs]
        @test count(==(:destination), modes) == 1                              # the unique grad recipe
        @test count(==(:ldiv), modes) == 1                                     # the `\` velocity recipe
        @test count(==(:assign), modes) == 4                                   # cholesky/logdet/kin/+
        # the source ops (grad port-call + fused kin) carry DISTINCT stable definition tokens (RK 07:25)
        srcops = [RKS.recipe_handle_op(h) for h in hs if RKS.recipe_handle_op(h) isa RKS._KernelSourceOp]
        @test length(srcops) == 2
        @test RKS.kernel_sourceop_token(srcops[1]) !== RKS.kernel_sourceop_token(srcops[2])
        @test count(o -> RKS.kernel_sourceop_form(o) === :portcall, srcops) == 1  # exactly the grad recipe
        # the FUSED kin op executes numerically over its ACTUAL ordered inputs (pot, node, mom, dkin) — the
        # authored `oftype(pot,0.5)*(node + dot(mom,dkin))` (RK 07:35)
        kinh = only(h for h in hs if RKS.recipe_handle_op(h) isa RKS._KernelSourceOp &&
                    RKS.kernel_sourceop_form(RKS.recipe_handle_op(h)) === :fused)
        @test length(kinh.inputs) == 4                                         # (pot, node, mom, dkin) ids
        kinres = RKS.recipe_handle_op(kinh)(0.3, 2.0, [1.0, 2.0], [0.5, 0.5])
        @test kinres == oftype(0.3, 0.5) * (2.0 + dot([1.0, 2.0], [0.5, 0.5]))

        # RAW/opaque ops REJECT as prepared handles — a manually-inserted closure or an unregistered name
        weird(x) = x
        @test_throws RKS._KernelFactoryReject RKS._recipe_handle((x) -> x, (1,), (2,), (2,))     # raw closure
        @test_throws RKS._KernelFactoryReject RKS._recipe_handle(weird, (1,), (2,), (2,))        # unregistered name
        # HANDLE-LEVEL collateral: a :portcall source op with two ALL-outputs but ONE owned REJECTS (RK 07:30)
        pcall = RKS._KernelSourceOp(Val(:tok), Val(:portcall), (gf, p) -> gf(p))
        @test_throws RKS._KernelFactoryReject RKS._recipe_handle(pcall, (10, 11), (20, 21), (21,))
        @test RKS.recipe_handle_mode(RKS._recipe_handle(pcall, (10, 11), (20, 21), (20, 21))) === :destination
        @test RKS.recipe_handle_mode(RKS._recipe_handle(pcall, (10,), (20,), (20,))) === :assign  # single-output

        # POISON: on the isolated graph, REPLACE the selected Recipe values (same ids/edges, THROWING op)
        # after preparation, and prove the captured handle op tuple is === unchanged AND executes unchanged
        # (RK 07:35). Compare elementwise with === (no objectid on singletons). Restore in finally.
        gp = RKS.kernel_graph(euclidean_ep)
        capd_ops = Tuple(RKS.recipe_handle_op(h) for h in hs)
        kin_before = RKS.recipe_handle_op(kinh)(0.3, 2.0, [1.0, 2.0], [0.5, 0.5])
        saved = copy(gp.recipes)
        try
            for (i, r) in enumerate(gp.recipes)
                gp.recipes[i] = RKS.Recipe(r.id, r.inputs, r.outputs,
                    (a...) -> error("poisoned Recipe.op"), r.cost, r.cse_key, r.effectful)
            end
            hs2 = RKS.kernel_prepared_handles(pf)
            @test all(RKS.recipe_handle_op(hs2[i]) === capd_ops[i] for i in eachindex(capd_ops))  # === unchanged
            @test RKS.recipe_handle_op(kinh)(0.3, 2.0, [1.0, 2.0], [0.5, 0.5]) == kin_before        # exec unchanged
        finally
            empty!(gp.recipes); append!(gp.recipes, saved)
        end

        # PER-CALLEE recipe-op safe domain at concrete binding (RK 07:30): identity + safe types accept,
        # custom-overload types reject
        dok = RKS.kernel_recipe_op_domain_ok
        L = typeof(cholesky([2.0 0.0; 0.0 2.0]))
        @test dok(cholesky, (Matrix{Float64},)) && !dok(cholesky, (Matrix{String},))
        @test dok(logdet, (L,)) && dok(logdet, (Matrix{Float64},))
        @test dok(\, (L, Vector{Float64})) && !dok(\, (Matrix{String}, Vector{Float64}))
        @test dok(+, (Float64, Float64)) && !dok(+, (String, String))
        # DIAGONAL mass (RK 18:34): a concrete builtin Diagonal{numeric,Vector} is admitted for cholesky input
        # AND as the Cholesky backing, so logdet/ldiv follow; F32/F64; the dense-Matrix path above is unchanged.
        for T in (Float64, Float32)
            D = typeof(Diagonal(T[2, 3]))                                    # Diagonal{T,Vector{T}}
            CD = typeof(cholesky(Diagonal(T[2, 3])))                         # Cholesky{T,Diagonal{T,Vector{T}}}
            @test RKS._kernel_dom_diag(D)                                    # the shared concrete-Diagonal predicate
            @test dok(cholesky, (D,))                                        # cholesky(Diagonal) admitted
            @test dok(logdet, (CD,))                                         # logdet(Diagonal-backed Cholesky)
            @test dok(\, (CD, Vector{T}))                                   # (Diagonal-backed Cholesky) \ mom
            @test RKS._recipe_dom_chol(CD)                                   # Diagonal backing sanctioned
        end
        # NEGATIVES — UniformScaling, custom AbstractVector backing, generic structured all REJECT (no widening)
        @test !dok(cholesky, (typeof(1.0 * I),))                            # UniformScaling{Float64} — not a concrete matrix
        @test !dok(cholesky, (typeof(I),))                                  # UniformScaling{Bool}
        @test !RKS._kernel_dom_diag(typeof(1.0 * I))
        @test !dok(cholesky, (Diagonal{Float64,ThrowingBuf},))             # custom AbstractVector backing (not <: Vector)
        @test !RKS._kernel_dom_diag(Diagonal{Float64,ThrowingBuf})
        @test !dok(cholesky, (Tridiagonal{Float64,Vector{Float64}},))      # generic structured — not Diagonal/Matrix
        @test !dok(cholesky, (Diagonal{String,Vector{String}},))           # non-numeric Diagonal
        # a Cholesky over a Diagonal backed by a CUSTOM AbstractVector (not <: Vector) still REJECTS
        @test !RKS._recipe_dom_chol(Cholesky{Float64,Diagonal{Float64,ThrowingBuf}})
        @test !dok(\, (Cholesky{Float64,Diagonal{Float64,ThrowingBuf}}, Vector{Float64}))
        # a Cholesky over a non-Matrix / non-Diagonal structured backing REJECTS
        @test !RKS._recipe_dom_chol(Cholesky{Float64,LowerTriangular{Float64,Matrix{Float64}}})
    end

    @testset "construction seam — ACTUAL mid-copy throw leaves NO stale blessed bit (epoch contract, not rollback) (RK 07:08)" begin
        # sizes MATCH (validation passes) but the dest buffer THROWS during copyto! — a post-validation
        # mid-copy failure. The epoch contract requires the dest mask to end DIRTY (zero) — no stale
        # current bit — even though values are unspecified/partially touched (no rollback).
        src = RKS._CanonOwned2(ThrowingBuf([1.0, 2.0]), 5.0, RKS._owner_mask(2, [1, 2]))   # both current
        dest = RKS._CanonOwned2(ThrowingBuf([0.0, 0.0]), 0.0, RKS._owner_mask(2, [1, 2]))  # starts current
        @test RKS._canon_current_mask(dest) == RKS._owner_mask(2, [1, 2])
        @test_throws ErrorException RKS._canon_copy_endpoint!(dest, src)         # copyto! throws mid-copy
        @test RKS._canon_current_mask(dest) == RKS._owner_mask(2, Int[])         # DIRTY — no stale blessed bit
    end

    @testset "nuts_state — compiler-owned DIAGNOSTICS storage (concrete field ABI, F32/F64, pending/committed epoch) (RK 08:42/08:47/08:48)" begin
        # F32/F64 type PRESERVATION — acceptance_rate/dham follow init.ham's type (no Float64 promotion)
        d32 = RKS._diagnostics_store(Float32); d64 = RKS._diagnostics_store(Float64)
        @test RKS.diagnostics_ham_type(d32) === Float32 && RKS.diagnostics_ham_type(d64) === Float64
        @test typeof(RKS._diag_slot(d32, Val(3))) === Float32 && typeof(RKS._diag_slot(d64, Val(4))) === Float64
        @test (@inferred RKS._diag_slot(d32, Val(3))) isa Float32          # Val accessor type-stable
        # CONCRETE mutable field ABI (NOT the boxing _OwnerState tuple path) — every scalar op exact 0-B
        # AND @inferred for BOTH F32 and F64 (oftype-parameterized value, typed loop)
        _setn(d) = RKS._diag_set!(d, Val(1), 7)
        _seta(d::RKS._DiagnosticsStore{T}) where {T} = RKS._diag_set!(d, Val(3), oftype(zero(T), 0.5))
        _rst(d) = RKS._diagnostics_reset!(d)
        _cmt(d) = RKS._diagnostics_root_commit!(d)
        for d in (RKS._diagnostics_store(Float32), RKS._diagnostics_store(Float64))
            _setn(d); _seta(d); _rst(d); _cmt(d)                            # warm this precision
            @test (@allocated _setn(d)) == 0 && (@allocated _seta(d)) == 0
            @test (@allocated _rst(d)) == 0 && (@allocated _cmt(d)) == 0
            @test (@inferred _seta(d)) === d
            @test (@inferred RKS._diag_slot(d, Val(3))) isa RKS.diagnostics_ham_type(d)
        end
        # RESET is the authoritative source write (RK 08:48): zeros COMMITTED (exception safety) + marks all
        # four PENDING-PRODUCED (current within the epoch). A dominated stats_f read is valid mid-epoch.
        RKS._diagnostics_reset!(d64)
        @test RKS.diagnostics_committed_mask(d64) == UInt(0)               # nothing committed mid-epoch
        @test all(RKS._diag_produced(d64, Val(i)) for i in 1:4)           # all four produced within-epoch
        @test !any(RKS._diag_committed(d64, Val(i)) for i in 1:4)
        # reset → stats read/update → commit: stats_f reads acceptance_rate (produced, valid) then updates
        @test RKS._diag_produced(d64, Val(3))                              # acceptance_rate readable mid-epoch
        RKS._diag_set!(d64, Val(1), 5); RKS._diag_set!(d64, Val(3), 0.8)   # stats writes n_steps + acceptance
        RKS._diagnostics_root_commit!(d64)                                 # SINGLE root commit
        @test RKS.diagnostics_committed_mask(d64) == UInt(0x0f)            # all four blessed at commit
        @test RKS._diag_slot(d64, Val(1)) == 5 && RKS._diag_slot(d64, Val(3)) == 0.8
        # THROW-before-commit: reset + producer writes but NO root commit → committed stays 0 (nothing
        # falsely current cross-epoch)
        dth = RKS._diagnostics_store(Float64)
        RKS._diagnostics_root_commit!(dth)                                 # a prior epoch committed something
        RKS._diagnostics_reset!(dth); RKS._diag_set!(dth, Val(2), 3)       # new epoch: reset + a write
        # (epoch would throw here, before _diagnostics_root_commit!)
        @test RKS.diagnostics_committed_mask(dth) == UInt(0)              # committed reset-zeroed, not blessed
    end

    @testset "nuts_state — concrete owned tree + full sampler FRAME (one shared authority, isolated endpoints/trees/proposals, F32/F64) (RK 08:55)" begin
        # concrete owned TREE byte-matching the fixture: log_weight (2-vec ham sentinel) + bwd/bwd_fwd (mv) +
        # summed_mom (trajectory), all zeroed buffers; F32/F64 preserved; DISTINCT buffers per tree
        for (T, ham) in ((Float32, -0.5f0), (Float64, -0.5))
            tr = RKS._nuts_tree(T[1, 2, 3], ham)
            @test tr.log_weight == fill(oftype(ham, -Inf), 2) && eltype(tr.log_weight) === T
            @test eltype(tr.bwd.mom) === T && tr.bwd.mom == zeros(T, 3) && tr.summed_mom.fwd == zeros(T, 3)
            trs = RKS._nuts_trees(T[1, 2, 3], ham, 4)
            @test length(trs) == 4 && trs[1].bwd.mom !== trs[2].bwd.mom   # per-tree isolation (distinct)
        end
        # TWO-PHASE full FRAME over the REAL pot_f-free SIX-recipe euclidean_ep (RK 09:00/09:02): phase 1
        # builds init (UNEXECUTED) + one shared + UNSEEDED children; POC runs the full six-handle init on
        # init exactly once (pgrad×1); phase 2 seeds the complete state to fwd/bwd/proposals (no recompute).
        integ = RKS.kernel_registration(leapfrog_ep!)
        pf = RKS._prepare_factory(euclidean_ep, integ)
        plan = RKS.kernel_prepared_plan(pf)
        gE = RKS.kernel_graph(euclidean_ep); cf(n) = RKS.canon_id(gE, euclidean_ep.ports[n].id)
        ownedbufs = [RKS.kernel_plan_field(plan, cf(n))[2] for n in (:pos, :mom, :dpot_dpos, :dkin_dmom)]
        ecmask = RKS._canon_current_mask                                       # alias
        entry_owned = sort([RKS.kernel_plan_field(plan, c)[2] for c in RKS.kernel_plan_entry_current(plan)
                            if RKS.kernel_plan_field(plan, c)[1] === :owned])
        Ncanon = length(first(RKS.kernel_plan_superset(plan)))
        function mkvals(T, pg)
            m = T[2 0; 0 2]; d = Dict{Int,Any}()
            for s in RKS.kernel_plan_slots(plan)
                nm = String(s.path[1])
                d[s.canon] =
                    nm == "grad_f" ? pg : nm == "metric" ? m : nm == "chol_metric" ? cholesky(m) :
                    startswith(nm, "##node") ? zero(T) :
                    nm == "pos" ? T[1, 2] : nm == "mom" ? T[3, 4] :
                    nm in ("dpot_dpos", "dham_dpos", "dkin_dmom", "dham_dmom") ? T[0, 0] :
                    zero(T)                                            # pot/kin/ham scalars
            end
            d
        end
        # frozen config carried into the frame: a raw step binder (validated internally) + no-effect stats
        stepb = RKS.partial(leapfrog_ep!; stepsize = 0.1)
        mkframe(T, pg, md) = RKS._construct_nuts_frame(pf, mkvals(T, pg), md;
                                                       step_f = stepb, stats_f = nothing, min_dham = -1000)
        for T in (Float32, Float64)
            pg = CountingPgrad(0)
            frame = mkframe(T, pg, 10)
            @test RKS.nuts_frame_shared(frame) isa RKS._CanonShared              # ONE shared authority
            @test RKS.nuts_frame_ham_type(frame) === T && RKS.diagnostics_ham_type(frame.diag) === T
            @test eltype(frame.trees[1].log_weight) === T
            @test length(frame.trees) == 11 && length(frame.proposals) == 12     # max_depth+1 / +2
            # frozen config = a VALIDATED prepared callable record (leapfrog! registration + bound stepsize)
            # + the no-effect stats binding; the derived `diverged` recipe starts DIRTY (RK 09:11/09:25)
            sc = RKS.nuts_frame_step(frame)
            @test RKS.prepared_callable_registration(sc) === RKS.kernel_registration(leapfrog_ep!)
            @test RKS.prepared_callable_token(sc) === RKS.kernel_registration(leapfrog_ep!).token
            @test RKS.prepared_callable_kwargs(sc) == (; stepsize = 0.1)
            @test RKS.stats_binding_registration(RKS.nuts_frame_stats(frame)) === nothing  # no-effect variant
            @test RKS.nuts_frame_max_depth(frame) == 10 && RKS.nuts_frame_min_dham(frame) === oftype(zero(T), -1000)
            @test !RKS.nuts_frame_diverged_pending(frame) && !RKS.nuts_frame_diverged_committed(frame)
            # phase-1: children are UNSEEDED (dirty) — NOT yet at entry_current, no pgrad ran
            @test ecmask(frame.fwd) != RKS._owner_mask(Ncanon, entry_owned) && pg.n == 0
            # SEEDING an INCOMPLETE init REJECTS (RK 09:10) — init not yet at entry_current before poc runs
            @test_throws RKS._KernelFactoryReject RKS._seed_nuts_children!(frame)
            # POC executes the FULL six-handle init on init exactly once (the one destination pgrad)
            initfn = RKS.compile_prepared_initialization(pf, typeof(frame.init), typeof(frame.shared))
            initfn(frame.init, frame.shared, RKS.kernel_prepared_handles(pf))
            @test pg.n == 1                                                      # pgrad total == 1
            @test ecmask(frame.init) == RKS._owner_mask(Ncanon, entry_owned)     # init mask == entry_current
            # derived `diverged` epoch validity (RK 09:25): produce→pending; a SUCCESS root commits it;
            # the NEXT root's reset clears BOTH pending AND committed; a produce-without-commit (a later
            # throw) leaves committed=false so the old value is never falsely current
            RKS._nuts_produce_diverged!(frame); @test RKS.nuts_frame_diverged_pending(frame)
            RKS._nuts_derived_root_commit!(frame); @test RKS.nuts_frame_diverged_committed(frame)   # success
            RKS._nuts_frame_reset_control!(frame)                                # next root: reset clears BOTH
            @test !RKS.nuts_frame_diverged_pending(frame) && !RKS.nuts_frame_diverged_committed(frame)
            RKS._nuts_produce_diverged!(frame)                                   # produced but NOT committed (throw)
            @test RKS.nuts_frame_diverged_pending(frame) && !RKS.nuts_frame_diverged_committed(frame)
            # phase-2: seed children — complete values+mask copied, ZERO extra pgrad/chol
            RKS._seed_nuts_children!(frame)
            @test pg.n == 1                                                      # no child recompute
            for ch in (frame.fwd, frame.bwd, frame.proposals[1], frame.proposals[end])
                @test ecmask(ch) == RKS._owner_mask(Ncanon, entry_owned)         # fwd/bwd/proposals mask==entry_current
                for s in ownedbufs                                              # census ALL owned buffer slots
                    @test RKS._canon_slot(ch, Val(s)) !== RKS._canon_slot(frame.init, Val(s))  # isolated
                    @test RKS._canon_slot(ch, Val(s)) == RKS._canon_slot(frame.init, Val(s))   # complete/seeded
                end
            end
            @test frame.proposals[1] !== frame.proposals[2]
            # control true; 0-B control+diagnostics reset
            _rc(fr) = RKS._nuts_frame_reset_control!(fr)
            @test frame.gofwd && frame.may_sample && frame.may_continue
            _rc(frame); @test (@allocated _rc(frame)) == 0
        end
        # An ACTUAL init exception (a THROWING destination-grad) leaves init INCOMPLETE + every child DIRTY,
        # and _seed_nuts_children! REJECTS (never seeds an incomplete init) (RK 09:10)
        badpg(dest, p) = (dest[1] = 99.0; error("grad boom"))
        badfn = RKS.compile_prepared_initialization(pf, typeof(mkframe(Float64, badpg, 3).init),
                                                    typeof(mkframe(Float64, badpg, 3).shared))
        frx2 = RKS._construct_nuts_frame(pf, mkvals(Float64, badpg), 3;
                                         step_f = stepb, stats_f = nothing, min_dham = -1000)
        @test_throws ErrorException badfn(frx2.init, frx2.shared, RKS.kernel_prepared_handles(pf))  # init throws
        @test RKS._canon_current_mask(frx2.init) != RKS._owner_mask(Ncanon, entry_owned)  # init INCOMPLETE
        @test_throws RKS._KernelFactoryReject RKS._seed_nuts_children!(frx2)               # seeding rejects
        @test all(RKS._canon_current_mask(ch) != RKS._owner_mask(Ncanon, entry_owned)
                  for ch in (frx2.fwd, frx2.bwd, frx2.proposals[1]))                       # children stay DIRTY
        # negative / incompatible max_depth REJECTS before any partial mutation (RK 09:10)
        @test_throws RKS._KernelFactoryReject RKS._construct_nuts_frame(pf, mkvals(Float64, CountingPgrad(0)),
            -1; step_f = stepb, stats_f = nothing, min_dham = -1000)
        # step_f Token IDENTITY: a DIFFERENT registered integrator than the endpoint Plan's REJECTS (RK 09:27)
        @test_throws RKS._KernelFactoryReject RKS._construct_nuts_frame(pf, mkvals(Float64, CountingPgrad(0)),
            3; step_f = RKS.partial(gradonly_step!; stepsize = 0.1), stats_f = nothing, min_dham = -1000)
        # a bare leapfrog_ep! (matching Token) MISSING the required `stepsize` kwarg REJECTS at construction
        @test_throws RKS._KernelFactoryReject RKS._construct_nuts_frame(pf, mkvals(Float64, CountingPgrad(0)),
            3; step_f = leapfrog_ep!, stats_f = nothing, min_dham = -1000)
        # TWO independent frames: DISTINCT per-sampler mutable shared authority; external grad kept by identity
        shared_pg = CountingPgrad(0)
        f1 = mkframe(Float64, shared_pg, 3)
        f2 = mkframe(Float64, shared_pg, 3)
        @test RKS.nuts_frame_shared(f1) !== RKS.nuts_frame_shared(f2)          # separate mutable authorities
        @test f1.init !== f2.init && f1.trees[1].bwd.mom !== f2.trees[1].bwd.mom  # fully isolated
    end

    @testset "nuts_state — FINAL callable sampler: concrete typed Handles + token-gated Mode-2 dispatch (RK 12:32/12:37/12:39)" begin
        # A fully-prepared frame (phase-1 build + POC init + phase-2 seed, no manual step) via the ONE-shot
        # author-facing entry `_prepare_nuts_frame` — the runnable substrate POC compiles its root against.
        integ = RKS.kernel_registration(leapfrog_ep!)
        pf = RKS._prepare_factory(euclidean_ep, integ)
        plan = RKS.kernel_prepared_plan(pf)
        stepb = RKS.partial(leapfrog_ep!; stepsize = 0.1)
        function mkvals(T, pg)
            m = T[2 0; 0 2]; d = Dict{Int,Any}()
            for s in RKS.kernel_plan_slots(plan)
                nm = String(s.path[1])
                d[s.canon] =
                    nm == "grad_f" ? pg : nm == "metric" ? m : nm == "chol_metric" ? cholesky(m) :
                    startswith(nm, "##node") ? zero(T) :
                    nm == "pos" ? T[1, 2] : nm == "mom" ? T[3, 4] :
                    nm in ("dpot_dpos", "dham_dpos", "dkin_dmom", "dham_dmom") ? T[0, 0] : zero(T)
            end
            d
        end
        prep(T, pg, md) = RKS._prepare_nuts_frame(pf, mkvals(T, pg), md;
                                                  step_f = stepb, stats_f = nothing, min_dham = -1000)
        _fire(skel, k, rng) = skel(k; rng = rng)                               # function barrier for the kw call
        # OWNER token = the `nuts_state` authoring owner (here, the endpoint plan token stands in); ROOT token
        # = the compiled public `nuts_root!!` Mode-2 skeleton. They are DISTINCT — the positive control that
        # the KernelObject keeps the OWNER token while Handles carry the ROOT token (RK 12:39).
        owner_tok = RKS.kernel_prepared_token(pf)
        root_tok = RKS.kernel_token(nuts_root!!)
        @test owner_tok !== root_tok                                            # owner-token != root-token
        for T in (Float32, Float64)
            frame = prep(T, CountingPgrad(0), 4)
            scratch = (zero(T), 0)                                              # concretely-typed Tuple scratch
            k = RKS.nuts_sampler(Val(owner_tok), Val(root_tok), frame, synroot!, scratch)
            @test RKS.kernel_token(k) === owner_tok                             # KernelObject keeps OWNER token
            @test RKS.nuts_handles_root_token(getfield(k, :handles)) === root_tok  # Handles carry ROOT token
            @test isconcretetype(typeof(k)) && isconcretetype(typeof(getfield(k, :handles)))  # nothing untyped
            @test RKS.nuts_sampler_frame(k) === frame                          # wraps the SAME frame
            # DISPATCH is the Mode-2 skeleton call, gated on RootToken. The transition mutates the frame in
            # place (bumps n_steps) and returns the SAME sampler (result === state).
            rng = Random.MersenneTwister(1)                                    # hoisted: not part of the call cost
            n0 = RKS._diag_slot(frame.diag, Val(1))
            r = nuts_root!!(k; rng = rng)
            @test r === k && RKS.nuts_sampler_frame(r) === frame               # result === state
            @test RKS._diag_slot(frame.diag, Val(1)) == n0 + 1                 # root! actually ran in place
            # type-stable dispatch + 0-B (the frame mutation is the only work; the call itself allocates none)
            @test (@inferred _fire(nuts_root!!, k, rng)) === k
            _fire(nuts_root!!, k, rng)                                          # warm the barrier
            @test (@allocated _fire(nuts_root!!, k, rng)) == 0
        end
        # WRONG root-token REJECTS: an UNRELATED Mode-2 skeleton (`other_root!!`) has NO matching method for a
        # sampler whose Handles carry `nuts_root!!`'s RootToken — pure-type gate, no name special-case.
        frame = prep(Float64, CountingPgrad(0), 3)
        k = RKS.nuts_sampler(Val(owner_tok), Val(root_tok), frame, synroot!, (0.0, 0))
        @test_throws MethodError other_root!!(k; rng = Random.MersenneTwister(1))
        # RAW-SIGNATURE keyword gates (RK 12:39): `rng` is a REQUIRED KEYWORD — a MISSING, POSITIONAL, or
        # EXTRA `rng` finds NO method and is rejected (the captured `(state; rng)` contract reused verbatim).
        @test_throws UndefKeywordError nuts_root!!(k)                            # missing required kw rng
        @test_throws MethodError nuts_root!!(k, Random.MersenneTwister(1))       # positional rng
        @test_throws MethodError nuts_root!!(k; rng = Random.MersenneTwister(1), foo = 1)  # extra kw
        # REPEATED same-signature constructions → IDENTICAL concrete object/root/scratch types (typed/0-B)
        ka = RKS.nuts_sampler(Val(owner_tok), Val(root_tok), prep(Float64, CountingPgrad(0), 3), synroot!, (0.0, 0))
        kb = RKS.nuts_sampler(Val(owner_tok), Val(root_tok), prep(Float64, CountingPgrad(0), 3), synroot!, (0.0, 0))
        @test typeof(ka) === typeof(kb)                                         # identical concrete sampler type
        @test typeof(getfield(ka, :handles)) === typeof(getfield(kb, :handles))
    end

    @testset "nuts_state — registered stats_f scalar-effect binding (write-roots→diag slots; nothing/opaque) (RK 08:42/09:11)" begin
        # the registration is DERIVED INTERNALLY (not a caller-supplied pair) and CARRIED in the binding
        reg = RKS.kernel_registration(synstats!)
        @test Set(reg.write_roots) == Set((:n_steps, :acceptance_rate))
        b = RKS._stats_binding(synstats!)
        @test RKS.stats_binding_registration(b) === reg                       # the captured registration
        @test RKS.stats_binding_source(b) === synstats! && RKS.stats_binding_token(b) === reg.token
        @test RKS.stats_binding_produced(b) == (1, 3)                          # n_steps→1, acceptance_rate→3
        # applying the produced-marking blesses exactly those diagnostic slots pending
        d = RKS._diagnostics_store(Float64); RKS._stats_produced!(d, b)
        @test RKS._diag_produced(d, Val(1)) && RKS._diag_produced(d, Val(3))
        @test !RKS._diag_produced(d, Val(2)) && !RKS._diag_produced(d, Val(4))
        # stats_f = nothing is the EXPLICIT no-effect specialization (collectstats! is a no-op)
        nb = RKS._stats_binding(nothing)
        @test RKS.stats_binding_registration(nb) === nothing && RKS.stats_binding_produced(nb) == ()
        @test RKS._stats_produced!(RKS._diagnostics_store(Float64), nb) isa RKS._DiagnosticsStore  # no-op
        # an OPAQUE / unregistered stats_f REJECTS (a diagnostics callback must be a registered @kernel)
        @test_throws RKS._KernelFactoryReject RKS._stats_binding(sin)
        # an UNMAPPABLE write-root REJECTS (no silent filter — never understate effects)
        @test_throws RKS._KernelFactoryReject RKS._stats_binding(badstats!)
        # the SAME complete binder validation applies to stats_f — a partial wrapper cannot silently reduce
        @test_throws RKS._KernelFactoryReject RKS._stats_binding(RKS.partial(synstats!, 42))   # bound positional
        @test_throws RKS._KernelFactoryReject RKS._stats_binding(RKS.partial(synstats!; bogus = 1))  # extra kw
    end

    @testset "nuts_state — COMPLETE binder-contract validation for prepared callables (RK 09:36)" begin
        pc(v) = RKS._prepare_callable(:step_f, v)
        # a valid partial: registration captured + bound kwargs recorded
        ok = pc(RKS.partial(leapfrog_ep!; stepsize = 0.1))
        @test RKS.prepared_callable_registration(ok) === RKS.kernel_registration(leapfrog_ep!)
        @test RKS.prepared_callable_kwargs(ok) == (; stepsize = 0.1)
        # one-call splice helper (poc seam): (leaf MethodIR, bound kwargs) from the CAPTURED source, no reread
        lf, kw = RKS.prepared_callable_leaf(ok)
        @test lf isa RKS.MethodIR && lf.id.name === :leapfrog_ep! && lf.ok && kw == (; stepsize = 0.1)
        # a TYPED required keyword (stepsize::Float64) is RECOGNIZED via the formal parser (not only Symbol)
        @test RKS.prepared_callable_kwargs(pc(RKS.partial(typedstep!; stepsize = 0.1))) == (; stepsize = 0.1)
        @test_throws RKS._KernelFactoryReject pc(typedstep!)                    # bare: missing typed stepsize
        # missing required kw / extra kw / bound positional all REJECT
        @test_throws RKS._KernelFactoryReject pc(leapfrog_ep!)                  # missing required stepsize
        @test_throws RKS._KernelFactoryReject pc(RKS.partial(leapfrog_ep!; stepsize = 0.1, bogus = 1))  # extra kw
        @test_throws RKS._KernelFactoryReject pc(RKS.partial(leapfrog_ep!, 42; stepsize = 0.1))          # positional (largs)
        @test_throws RKS._KernelFactoryReject pc(RKS.partial(leapfrog_ep!, :, 42; stepsize = 0.1))       # positional (rargs)
        # a NESTED binder (partial of a partial) REJECTS — inner actuals would be silently dropped (RK 09:38).
        # The OUTER binding is otherwise VALID (stepsize accepted, no positional/extra kw), so ONLY the
        # nested-binder guard makes it reject — non-vacuous (RK 09:41)
        @test_throws RKS._KernelFactoryReject pc(RKS.partial(RKS.partial(leapfrog_ep!; stepsize = 0.1); stepsize = 0.2))
        # an opaque callable rejects
        @test_throws Union{RKS._KernelFactoryReject,ArgumentError} pc(sin)
    end

    @testset "nuts_state — MIXED endpoints+vectors+scalars frame: scalar 0-B, buffers direct, F32/F64 (RK 08:47/08:51)" begin
        # a representative concrete top-frame slot group (the _CanonOwnedN field ABI, NOT _OwnerState): a
        # buffer (endpoint mom), a buffer (tree), a scalar Int (n_steps), a scalar float (acceptance_rate).
        # Cover BOTH scalar+buffer precision combos so the F32/F64 claim is literal (typed via oftype).
        _sn(f) = RKS._canon_set!(f, Val(3), 9)                             # scalar n_steps
        _sa(f) = RKS._canon_set!(f, Val(4), oftype(RKS._canon_slot(f, Val(4)), 0.7))  # scalar acceptance
        _cp(f, s) = RKS._canon_copy_slot!(f, s, Val(1))                    # buffer copy — DIRECT array
        for (Tsc, Tbuf) in ((Float32, Float64), (Float64, Float32))
            mom = Tbuf[1, 2]; treebuf = Tsc[3, 4]
            frame = RKS._CanonOwned4(mom, treebuf, 0, zero(Tsc), RKS._owner_mask(4, [1, 2]))
            src = RKS._CanonOwned4(Tbuf[7, 8], Tsc[0, 0], 0, zero(Tsc), RKS._owner_mask(4, Int[]))
            @test typeof(RKS._canon_slot(frame, Val(2))) === Vector{Tsc}   # buffer eltype preserved
            @test typeof(RKS._canon_slot(frame, Val(4))) === Tsc           # scalar precision preserved
            _sn(frame); _sa(frame); _cp(frame, src)                        # warm this precision
            @test (@allocated _sn(frame)) == 0 && (@allocated _sa(frame)) == 0  # scalar setfield! exact 0-B
            @test (@allocated _cp(frame, src)) == 0                         # buffer copy exact 0-B (direct)
            @test (@inferred _cp(frame, src)) === frame                     # buffer copy @inferred
            @test RKS._canon_slot(frame, Val(1)) === mom                    # buffer identity kept (direct)
            @test RKS._canon_slot(frame, Val(3)) == 9 && RKS._canon_slot(frame, Val(4)) == oftype(zero(Tsc), 0.7)
            @test (@inferred RKS._canon_slot(frame, Val(4))) isa Tsc
        end
    end

    @testset "poc binder/plan seam accessors (RK 04:41)" begin
        integ = RKS.kernel_registration(leapfrog_ep!)
        plan = RKS._kernel_factory_endpoint_plan(phasepoint_ep, integ)
        # the integrator/owner Token via a dedicated accessor (no key-tuple destructuring by codegen)
        @test RKS.kernel_plan_token(plan) === integ.token
        # binder-kwargs trait: extended ONLY for PartialFunction, retrieves bound kwargs
        pf = partial(leapfrog!; stepsize = 0.1)
        @test RKS._kernel_binder_kwargs(pf) == (; stepsize = 0.1)
        @test RKS._kernel_binder_target(pf) === leapfrog!
        @test RKS._kernel_binder_kwargs(sin) === nothing               # not a binder
        @test RKS._kernel_binder_kwargs(EvilWrap(leapfrog!)) === nothing   # no duck-typing
        # bound numeric stepsize stays runtime-typed: .1 vs .2 are the same PartialFunction TYPE
        @test typeof(partial(leapfrog!; stepsize = 0.1)) === typeof(partial(leapfrog!; stepsize = 0.2))
    end

    @testset "nuts_state — runnable prepare→compile→attach CORE: end-to-end nuts!! from _build_nuts_sampler (RK 13:18 fork-free half)" begin
        # the WHOLE internal chain in ONE call: pf → _prepare_nuts_frame → POC compile_nuts → nuts_sampler →
        # FINAL callable KernelObject, on which the authored `nuts!!(sampler; rng)` runs end-to-end.
        pf = RKS._prepare_factory(_ErgFix.euclidean_phasepoint, RKS.kernel_registration(_ErgFix.leapfrog!))
        plan = RKS.kernel_prepared_plan(pf)
        function ergvals(T)
            m = T[2 0; 0 2]; d = Dict{Int,Any}()
            for s in RKS.kernel_plan_slots(plan)
                nm = String(s.path[1])
                d[s.canon] = nm == "grad_f" ? ((dst, p) -> (dst .= 2 .* p; sum(abs2, p))) :
                    nm == "metric" ? m : nm == "chol_metric" ? cholesky(m) : startswith(nm, "##node") ? zero(T) :
                    nm == "pos" ? T[1, 2] : nm == "mom" ? T[3, 4] :
                    (nm in ("dpot_dpos", "dham_dpos", "dkin_dmom", "dham_dmom")) ? T[0, 0] : zero(T)
            end
            d
        end
        _run(skel, k, rng) = skel(k; rng = rng)                     # function barrier for the kw dispatch
        for T in (Float64, Float32)
            k = RKS._build_nuts_sampler(pf, ergvals(T), _ErgFix.nuts_state, _ErgFix.refresh_momentum!!, _ErgFix.nuts!!;
                                        step_f = RKS.partial(_ErgFix.leapfrog!; stepsize = T(0.1)),
                                        max_depth = 4, min_dham = -1000, stats_f = _ErgFix.nuts_stats!)
            # FINAL concrete callable KernelObject: OwnerToken = nuts_state, RootToken = nuts!! — distinct, both kept
            @test k isa RKS.KernelObject && isconcretetype(typeof(k)) && isconcretetype(typeof(getfield(k, :handles)))
            @test RKS.kernel_token(k) === RKS.kernel_token(_ErgFix.nuts_state)            # OwnerToken = nuts_state
            @test RKS.nuts_handles_root_token(getfield(k, :handles)) === RKS.kernel_token(_ErgFix.nuts!!)  # RootToken = nuts!!
            @test RKS.kernel_token(_ErgFix.nuts_state) !== RKS.kernel_token(_ErgFix.nuts!!)  # distinct
            frame = RKS.nuts_sampler_frame(k)
            @test RKS.diagnostics_ham_type(frame.diag) === T                              # F32/F64 preserved
            # the authored public transition RUNS end-to-end and mutates in place; result === state
            p0 = copy(RKS._canon_slot(frame.init, RKS.kernel_plan_named_slot_val(plan, Val(:pos))))
            rng = Random.Xoshiro(1)
            r = _ErgFix.nuts!!(k; rng = rng)
            @test r === k                                                                # result === state
            @test RKS._canon_slot(frame.init, RKS.kernel_plan_named_slot_val(plan, Val(:pos))) != p0  # real transition
            @test RKS.diagnostics_committed_mask(frame.diag) == UInt(0x0f)               # one deferred epoch commit
            # type-stable + 0-B public transition (warmed barrier, hoisted rng)
            @test (@inferred _run(_ErgFix.nuts!!, k, rng)) === k
            _run(_ErgFix.nuts!!, k, rng)
            @test (@allocated _run(_ErgFix.nuts!!, k, rng)) == 0
        end
        # WRONG-skeleton token gate still holds through the ergonomic core: leapfrog! (a different Mode-2 token)
        # cannot drive a sampler whose Handles carry nuts!!'s RootToken.
        k = RKS._build_nuts_sampler(pf, ergvals(Float64), _ErgFix.nuts_state, _ErgFix.refresh_momentum!!, _ErgFix.nuts!!;
                                    step_f = RKS.partial(_ErgFix.leapfrog!; stepsize = 0.1),
                                    max_depth = 3, min_dham = -1000, stats_f = _ErgFix.nuts_stats!)
        @test_throws MethodError _ErgFix.leapfrog!(k; rng = Random.Xoshiro(1))
        # repeated same-signature constructions → IDENTICAL concrete sampler type (RK 12:32/13:44 gate).
        # POC closed the stable-type blocker at 6ec4147 (deterministic hot-emitter locals): `compile_nuts`
        # now returns a signature-stable root, so two compiles of the SAME signature share ONE concrete type.
        k2 = RKS._build_nuts_sampler(pf, ergvals(Float64), _ErgFix.nuts_state, _ErgFix.refresh_momentum!!, _ErgFix.nuts!!;
                                     step_f = RKS.partial(_ErgFix.leapfrog!; stepsize = 0.1),
                                     max_depth = 3, min_dham = -1000, stats_f = _ErgFix.nuts_stats!)
        @test typeof(k) === typeof(k2)
    end

    @testset "nuts_state — COLD BOOTSTRAP: source inputs → executed-once endpoint, no canon-id Dict (RK 13:44/user)" begin
        mutable struct _CG; n::Int; end
        (c::_CG)(dst, p) = (c.n += 1; dst .= 2 .* p; sum(abs2, p))
        pf = RKS._prepare_factory(_ErgFix.euclidean_phasepoint, RKS.kernel_registration(_ErgFix.leapfrog!))
        plan = RKS.kernel_prepared_plan(pf)
        posv = RKS.kernel_plan_named_slot_val(plan, Val(:pos))
        momv = RKS.kernel_plan_named_slot_val(plan, Val(:mom))
        dpotv = RKS.kernel_plan_named_slot_val(plan, Val(:dpot_dpos))
        for T in (Float64, Float32)
            cg = _CG(0)
            # the AUTHOR supplies ONLY the phasepoint sources (grad_f, metric, pos, mom) in signature order —
            # NO canon-id Dict, NO pre-allocated derived storage. Keep the caller's own arrays to prove non-aliasing.
            cpos = T[1, 2]; cmom = T[3, 4]; cmetric = T[2 0; 0 2]
            sources = (cg, cmetric, cpos, cmom)
            cvals = RKS._bootstrap_canon_values(plan, pf.handles, sources)
            @test cg.n == 1                                                    # exactly ONE gradient at bootstrap
            # values match the reference computation (superset order: 5 node,6 pot,7 dpot,8 chol,9 dkin,10 kin,11 ham)
            C = cholesky(T[2 0; 0 2]); dk = C \ T[3, 4]
            @test cvals[6] ≈ sum(abs2, T[1, 2]) && cvals[7] ≈ 2 .* T[1, 2]     # pot, dpot_dpos
            @test cvals[8] isa Cholesky && cvals[9] ≈ dk                       # chol_metric, dkin_dmom
            @test cvals[11] ≈ cvals[6] + cvals[10]                            # ham = pot + kin
            @test eltype(cvals[7]) === T && typeof(cvals[11]) === T           # F32/F64 preserved
            frame = RKS._construct_nuts_frame_bootstrapped(pf, cvals, 4;
                step_f = RKS.partial(_ErgFix.leapfrog!; stepsize = T(0.1)), stats_f = _ErgFix.nuts_stats!, min_dham = -1000)
            @test cg.n == 1                                                    # frame build does NOT re-run the gradient
            @test RKS._canon_current_mask(frame.init) == frame.entry_mask     # init COMPLETE (no separate POC init/seed)
            # ISOLATION CONTRACT (RK 14:xx): the caller's pos/mom are NEVER aliased into the sampler — the init
            # holds DEEP COPIES, so a transition mutates the copies, never the caller's arrays.
            @test RKS._canon_slot(frame.init, posv) !== cpos                  # init.pos is a COPY of caller pos
            @test RKS._canon_slot(frame.init, momv) !== cmom                  # init.mom is a COPY of caller mom
            @test RKS._canon_slot(frame.init, posv) == cpos                   # ... value-equal (seeded from source)
            # bootstrap-PRODUCED buffers (dpot_dpos) ARE retained by identity (fresh per bootstrap, no discard)
            @test RKS._canon_slot(frame.init, dpotv) === cvals[7]
            @test RKS._canon_slot(frame.fwd, posv) !== RKS._canon_slot(frame.init, posv)  # child ISOLATED
            @test RKS._canon_slot(frame.fwd, posv) == RKS._canon_slot(frame.init, posv)   # ... but complete/seeded
            @test RKS.diagnostics_ham_type(frame.diag) === T
            # end-to-end: compile + attach + run the authored transition
            Cc = RKS.compile_nuts(pf, _ErgFix.nuts_state, _ErgFix.refresh_momentum!!, _ErgFix.nuts!!, frame)
            k = RKS.nuts_sampler(Val(RKS.kernel_token(_ErgFix.nuts_state)), Val(Cc.RootToken), frame, Cc.root!, Cc.scratch)
            p0 = copy(RKS._canon_slot(frame.init, posv)); cpos0 = copy(cpos); cmom0 = copy(cmom); cmet0 = copy(cmetric)
            for _ in 1:3; _ErgFix.nuts!!(k; rng = Random.Xoshiro(1)); end
            r = _ErgFix.nuts!!(k; rng = Random.Xoshiro(1))
            @test r === k && RKS._canon_slot(frame.init, posv) != p0          # result===state; real transition
            @test cpos == cpos0 && cmom == cmom0 && cmetric == cmet0         # caller sources UNCHANGED across transitions
        end
        # per-instance ISOLATION + same-signature IDENTICAL concrete frame type across bootstraps — TWO samplers
        # built from the SAME caller arrays must both be non-aliased from the caller AND from each other.
        spos = [1.0, 2.0]; smom = [3.0, 4.0]; smetric = [2.0 0; 0 2]
        mkframe() = RKS._construct_nuts_frame_bootstrapped(pf,
            RKS._bootstrap_canon_values(plan, pf.handles, (_CG(0), smetric, spos, smom)), 3;
            step_f = RKS.partial(_ErgFix.leapfrog!; stepsize = 0.1), stats_f = _ErgFix.nuts_stats!, min_dham = -1000)
        fa = mkframe(); fb = mkframe()
        @test typeof(fa) === typeof(fb)                                       # deterministic concrete frame type
        @test RKS._canon_slot(fa.init, posv) !== RKS._canon_slot(fb.init, posv)   # independent per-instance buffers
        @test RKS._canon_slot(fa.init, posv) !== spos && RKS._canon_slot(fb.init, posv) !== spos  # neither aliases caller
        @test RKS.nuts_frame_shared(fa) !== RKS.nuts_frame_shared(fb)         # separate shared authorities
    end
end

end # module TestKernelFactory
