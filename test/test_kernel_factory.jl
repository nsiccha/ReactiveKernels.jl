# Inc3 factory/composition substrate tests. Isolated in a module so fixtures cannot
# shadow package exports in the shared Pkg.test Main.
module TestKernelFactory
using ReactiveKernels, Test
const RKS = ReactiveKernels

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
end

end # module TestKernelFactory
