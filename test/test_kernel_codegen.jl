# Increment-3 EXECUTABLE codegen gate — the REAL vertical slice. The compiled leapfrog leaf mutates the
# ACTUAL factory `_OwnerState` slots (no parallel struct), consuming the immutable plan seam
# (producer/recipes/entry-current/slots) + the `partial(leapfrog!;stepsize)` binder. Proves: exactly ONE
# in-place `pgrad!` per leaf (destination-aware, writing the canonical grad slot + committing pot); the
# authored 3-line leapfrog over real F32/F64 buffers; no residual `leapfrog!`/`step_f` call; @inferred +
# warmed exact 0 B; and that the graph/currentness/layout/binder come from the seam, never a re-plan.

using ReactiveKernels
using LinearAlgebra
using Test
const RK = ReactiveKernels

module _CgFix
    include(joinpath(@__DIR__, "..", "benchmark", "nuts_kernel_authoring_fixture.jl"))
end
const _CgLF = _CgFix.leapfrog!
const _CgPP = _CgFix.euclidean_phasepoint         # the phase-point endpoint (a stateless KernelSpec)

# The endpoint seam under leapfrog!, the method IR, and the interim recipe-id → applier binding.
_cg_seam() = RK._kernel_factory_endpoint_plan(_CgPP, RK.kernel_registration(_CgLF))
_cg_ir() = RK.method_irs(_CgLF)[1]
function _cg_appliers(seam)
    prod = Dict(kv[1] => kv[2] for kv in RK.kernel_plan_producer(seam))
    Dict(prod[RK._l_field_id(_CgPP, :dpot_dpos)] => :pgrad,
         prod[RK._l_field_id(_CgPP, :dkin_dmom)] => :velocity)
end

# a real in-place gradient that COUNTS its invocations (0-alloc functor, not a boxed closure): writes the
# caller-owned gradient buffer and returns the potential — the exact `(g, x)->pot` harness shape.
mutable struct _CountGrad{P}; f::P; n::Int; end
(c::_CountGrad)(g, x) = (c.n += 1; c.f(g, x))
_quad!(g, x) = (@. g = x; sum(abs2, x) / 2)       # quadratic potential U = ½‖x‖², ∇U = x

# Build + seed a real endpoint state over T (dim d); make dpot_dpos current at entry (the seam contract).
function _cg_state(seam, ::Type{T}, d, meta) where {T}
    st = RK.make_leaf_owner_state(_CgPP, seam, T, d)
    RK._owner_commit!(st, ntuple(i -> begin
        s = RK._owner_slot(st, Val(i))
        s isa AbstractVector ? (s .= T.(1:d) ./ (i + 1)) : s
    end, meta.nslots))
    _quad!(RK._owner_slot(st, Val(meta.dpotslot)), RK._owner_slot(st, Val(meta.posslot)))  # entry-current grad
    st
end

@testset "codegen — recompute graph built from the SEAM producer/recipes, never plan()" begin
    seam = _cg_seam()
    pg = RK._l_seam_plan_graph(_CgPP, seam)
    @test pg.plan === nothing                                    # NOT a core Plan — seam-derived
    # producer map EXACTLY equals kernel_plan_producer (canonical Value id → selected Recipe id)
    mine = Tuple(sort!([(cid, rid) for (cid, rid) in pg.producer]))
    seam_prod = Tuple((cid, rid) for (cid, rid) in RK.kernel_plan_producer(seam) if !(cid in pg.sources))
    @test mine == seam_prod
    # the multi-output gradient Recipe owns BOTH pot and dpot_dpos (atomic), from the seam
    gradrid = pg.producer[RK._l_field_id(_CgPP, :dpot_dpos)]
    @test RK._l_field_id(_CgPP, :pot) in pg.recipe_owned[gradrid]
    @test length(pg.recipe_owned[gradrid]) >= 2
end

@testset "codegen — entry-currentness is the seam's proven kernel_plan_entry_current" begin
    seam = _cg_seam()
    ec = RK._l_seam_entry_current(seam)
    @test ec == Set(RK.kernel_plan_entry_current(seam))
    # dpot_dpos and pos are current at entry (so the first half-kick recomputes NO gradient)
    @test RK._l_field_id(_CgPP, :dpot_dpos) in ec
    @test RK._l_field_id(_CgPP, :pos) in ec
end

@testset "codegen — the emitted leaf is the 3-line leapfrog with NO residual call, ONE pgrad" begin
    seam = _cg_seam(); ir = _cg_ir()
    fn, meta = RK.compile_leaf(ir, _CgPP, seam, _CgFix.partial(_CgLF; stepsize = 0.1);
                               appliers = _cg_appliers(seam))
    src = string(meta.body)
    @test !occursin("leapfrog", src)                            # no residual integrator call
    @test !occursin("step_f", src)
    @test meta.pgrad_execs == 1                                 # exactly one gradient RECIPE exec per leaf
    @test count(a -> occursin("pgrad!", string(a)), meta.body.args) == 1
    # storage is the factory _OwnerState (Val-slot access + a scalar commit), not a parallel struct
    @test occursin("_owner_slot(state", src)
    @test occursin("_owner_commit!(state", src)
end

@testset "codegen — stepsize is CONSUMED from the partial binder, not a raw argument" begin
    seam = _cg_seam(); ir = _cg_ir()
    # two different bound stepsizes -> two different emitted constants (proves the binder is the source)
    _, m1 = RK.compile_leaf(ir, _CgPP, seam, _CgFix.partial(_CgLF; stepsize = 0.1); appliers = _cg_appliers(seam))
    _, m2 = RK.compile_leaf(ir, _CgPP, seam, _CgFix.partial(_CgLF; stepsize = 0.25); appliers = _cg_appliers(seam))
    @test occursin("0.1", string(m1.body)) && !occursin("0.25", string(m1.body))
    @test occursin("0.25", string(m2.body))
    # a binder that resolves to no registered token is rejected
    @test_throws RK._LLowerReject RK.compile_leaf(ir, _CgPP, seam, (x -> x); appliers = _cg_appliers(seam))
end

@testset "codegen — executes over real F32/F64 slots: parity, ONE pgrad! call, @inferred, warm 0 B" begin
    for T in (Float64, Float32)
        seam = _cg_seam(); ir = _cg_ir(); ε = T(0.1); d = 5
        fn, meta = RK.compile_leaf(ir, _CgPP, seam, _CgFix.partial(_CgLF; stepsize = ε);
                                   appliers = _cg_appliers(seam))
        metric = Matrix{T}(2 * I, d, d); cholf = cholesky(metric)
        st = _cg_state(seam, T, d, meta)
        pos0 = copy(RK._owner_slot(st, Val(meta.posslot)))
        mom0 = copy(RK._owner_slot(st, Val(meta.momslot)))
        g0   = copy(RK._owner_slot(st, Val(meta.dpotslot)))     # ∇U(pos0) = pos0 (quadratic)

        pg = _CountGrad(_quad!, 0)
        fn(st, pg, cholf)
        @test pg.n == 1                                         # EXACTLY one pgrad! per leaf

        # analytic parity for U=½‖x‖² (∇U=x), M=2I: reference leapfrog by hand
        mom_h = mom0 .- (ε / 2) .* g0                           # half kick with entry gradient
        vel   = cholf \ mom_h                                   # velocity = M⁻¹ mom
        pos_r = pos0 .+ ε .* vel                                # drift
        mom_r = mom_h .- (ε / 2) .* pos_r                       # half kick with NEW gradient (∇U(pos_r)=pos_r)
        @test RK._owner_slot(st, Val(meta.posslot)) ≈ pos_r
        @test RK._owner_slot(st, Val(meta.momslot)) ≈ mom_r
        @test RK._owner_slot(st, Val(meta.dpotslot)) ≈ pos_r    # canonical grad slot = ∇U(new pos)
        @test RK._owner_slot(st, Val(meta.potslot)) ≈ sum(abs2, pos_r) / 2   # committed pot

        fn(st, pg, cholf)                                       # warm
        @inferred fn(st, pg, cholf)
        @test (@allocated fn(st, pg, cholf)) == 0              # typed exact 0 B
    end
end

@testset "codegen — an exec whose selected Recipe has no bound applier is REJECTED" begin
    seam = _cg_seam(); ir = _cg_ir()
    @test_throws RK._LLowerReject RK.compile_leaf(ir, _CgPP, seam, _CgFix.partial(_CgLF; stepsize = 0.1);
                                                  appliers = Dict{Int,Symbol}())
end
