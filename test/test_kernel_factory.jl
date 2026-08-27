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
    end
end

end # module TestKernelFactory
