# Inc3 factory/composition substrate tests. Isolated in a module so fixtures cannot
# shadow package exports in the shared Pkg.test Main.
module TestKernelFactory
using ReactiveKernels, Test
using LinearAlgebra
const RKS = ReactiveKernels

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

        # the immutable path→typed-slot PLAN poc consumes (never recomputes ownership)
        plan = RKS._kernel_factory_endpoint_plan(phasepoint_ep, integ)
        @test RKS.kernel_plan_slot(plan, :pos)[1] === :owned
        @test RKS.kernel_plan_slot(plan, :dham_dpos)[1] === :owned
        @test RKS.kernel_plan_slot(plan, :metric)[1] === :shared
        # slot indices are 1-based and contiguous within each tuple
        @test Set(RKS.kernel_plan_slot(plan, f)[2] for f in RKS.kernel_plan_owned(plan)) ==
              Set(1:length(RKS.kernel_plan_owned(plan)))
        # every owner field is classified exactly once (owned XOR shared)
        @test length(owned) + length(shared) == length(RKS._kernel_all_ports(phasepoint_ep))
    end
end

end # module TestKernelFactory
