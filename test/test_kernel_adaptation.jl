# Gates for lowering an AUTHORED free stateful @kernel (dual_averaging_state / welford_var) to a runnable
# object — the AUTHORED recurrence, NOT the package @reactive type (HMC acceptance G3/G4). This first slice
# covers the generic prepared-factory + concrete-state construction/initialization (the field-initializer
# recipes run into concrete `init`-typed storage). Method execution (fit!/step!) lands in the next slice.
using Test, LinearAlgebra
using ReactiveKernels
const RK = ReactiveKernels

module _AdaptFix
    include(joinpath(@__DIR__, "..", "benchmark", "nuts_kernel_authoring_fixture.jl"))
end

_slotval(pf, owned, shared, nm) = begin
    PL = RK.kernel_prepared_plan(pf); s = RK.kernel_plan_slot(PL, nm)
    role, slot = RK.kernel_plan_field(PL, s.canon)
    RK._canon_slot(role === :owned ? owned : shared, Val(slot))
end
_slotcur(pf, owned, shared, nm) = begin
    PL = RK.kernel_prepared_plan(pf); s = RK.kernel_plan_slot(PL, nm)
    role, slot = RK.kernel_plan_field(PL, s.canon)
    RK._canon_current(role === :owned ? owned : shared, Val(slot))
end

@testset "kernel_adaptation — welford_var: generic prep + construct + init from a template vector" begin
    wv = _AdaptFix.welford_var
    pf = RK._prepare_stateful(wv)
    # owned = fields mutated ∪ derived-from-mutated; welford: n/mean/var (mutated), template shared (have)
    PL = RK.kernel_prepared_plan(pf)
    roles = Dict(String(s.path[1]) => RK.kernel_plan_field(PL, s.canon)[1] for s in RK.kernel_plan_slots(PL))
    @test roles["n"] === :owned && roles["mean"] === :owned && roles["var"] === :owned
    @test roles["template"] === :shared
    for T in (Float64, Float32)
        owned, shared = RK._construct_stateful(wv, pf, T[0, 0, 0])
        @test _slotval(pf, owned, shared, :n) === zero(T)                 # n = zero(eltype(template)), typed
        @test _slotval(pf, owned, shared, :mean) == T[0, 0, 0]
        @test _slotval(pf, owned, shared, :var) == T[0, 0, 0]
        @test eltype(_slotval(pf, owned, shared, :mean)) === T            # concrete template-derived eltype
        @test all(_slotcur(pf, owned, shared, nm) for nm in (:n, :mean, :var, :template))  # all current post-init
    end
end
