# Increment-3 EXECUTABLE codegen gate — the REAL vertical slice. The compiled leapfrog leaf mutates the
# ACTUAL factory `_OwnerState` slots (no parallel struct), consuming ONLY the immutable plan seam
# (producer/recipes/entry-current/slots/key) + a DETACHED recipe-inputs map + the validated
# `partial(leapfrog!;stepsize)` binder — NEVER the live graph. Proves: exactly ONE in-place `pgrad!` per
# leaf (destination-aware, writing the canonical grad slot + committing pot); the authored 3-line leapfrog
# over real F32/F64 buffers; a TYPED RUNTIME stepsize (one specialization runs .1/.25 correctly, no baked
# constant); scheduling stable under post-seam graph mutation; binder Token == seam integrator Token; no
# residual `leapfrog!`/`step_f`; @inferred + warmed exact 0 B.

using ReactiveKernels
using LinearAlgebra
using Test
const RK = ReactiveKernels

module _CgFix
    include(joinpath(@__DIR__, "..", "benchmark", "nuts_kernel_authoring_fixture.jl"))
end
const _CgLF = _CgFix.leapfrog!
const _CgPP = _CgFix.euclidean_phasepoint         # the phase-point endpoint (a stateless KernelSpec)

_cg_seam() = RK._kernel_factory_endpoint_plan(_CgPP, RK.kernel_registration(_CgLF))
_cg_ir() = RK.method_irs(_CgLF)[1]
# DETACHED recipe-inputs (interim snapshot; production consumes `kernel_plan_recipe_inputs(seam)`).
_cg_ri(seam) = RK._seam_recipe_inputs_snapshot(_CgPP, seam)
function _cg_appliers(seam)
    prod = Dict(kv[1] => kv[2] for kv in RK.kernel_plan_producer(seam))
    Dict(prod[RK._seam_canon(seam, :dpot_dpos)] => :pgrad,
         prod[RK._seam_canon(seam, :dkin_dmom)] => :velocity)
end
_cg_compile(seam, ir; step = 0.1) =
    RK.compile_leaf(ir, seam, _CgFix.partial(_CgLF; stepsize = step);
                    recipe_inputs = _cg_ri(seam), appliers = _cg_appliers(seam))

# a real in-place gradient that COUNTS its invocations (0-alloc functor, not a boxed closure): writes the
# caller-owned gradient buffer and returns the potential — the exact `(g, x)->pot` harness shape.
mutable struct _CountGrad{P}; f::P; n::Int; end
(c::_CountGrad)(g, x) = (c.n += 1; c.f(g, x))
_quad!(g, x) = (@. g = x; sum(abs2, x) / 2)       # quadratic potential U = ½‖x‖², ∇U = x

function _cg_state(seam, ::Type{T}, d, meta) where {T}
    st = RK.make_leaf_owner_state(_CgPP, seam, T, d)
    RK._owner_commit!(st, ntuple(i -> begin
        s = RK._owner_slot(st, Val(i))
        s isa AbstractVector ? (s .= T.(1:d) ./ (i + 1)) : s
    end, meta.nslots))
    _quad!(RK._owner_slot(st, Val(meta.dpotslot)), RK._owner_slot(st, Val(meta.posslot)))  # entry-current grad
    st
end

@testset "codegen — recompute graph is PURE(seam, detached recipe-inputs) — never the live graph" begin
    seam = _cg_seam(); ri = _cg_ri(seam)
    pg = RK._l_seam_plan_graph(seam, ri)
    @test pg.plan === nothing                                    # seam-derived, not a core Plan
    mine = Tuple(sort!([(cid, rid) for (cid, rid) in pg.producer]))
    seam_prod = Tuple((cid, rid) for (cid, rid) in RK.kernel_plan_producer(seam) if !(cid in pg.sources))
    @test mine == seam_prod
    gradrid = pg.producer[RK._seam_canon(seam, :dpot_dpos)]
    # ONE destination-bound grad Recipe identity produces pot+dpot ATOMICALLY (RK 04:53): pot's selected
    # producer IS the grad recipe — the slice assumes NO alternative pot producer / no unused pot_f slot.
    @test pg.producer[RK._seam_canon(seam, :pot)] == gradrid
    @test RK._seam_canon(seam, :pot) in pg.recipe_owned[gradrid]     # multi-output grad owns pot too
    @test length(pg.recipe_owned[gradrid]) >= 2
    # a missing detached edge for a selected recipe is REJECTED (never silently read from the graph)
    @test_throws RK._LLowerReject RK._l_seam_plan_graph(seam, Dict{Int,Vector{Int}}())
end

@testset "codegen — ADVERSARY: POISON the selected pos field mapping; body stays seam-derived" begin
    # LOAD-BEARING adversary (RK 04:54): after capturing seam+IR, POISON the authoritative field mapping
    # itself — rebind `_CgPP.ports[:pos]` to a FRESH Value so the LIVE `_l_field_id(:pos)` resolves
    # DIFFERENTLY. The seam-derived compile must be immune (the detached field→canon is the authority),
    # while the SPEC-resolver schedule provably DRIFTS — proving the poison is real, not inert.
    seam = _cg_seam(); ir = _cg_ir(); ri = _cg_ri(seam)
    _, m1 = _cg_compile(seam, ir)
    live_pos_before = RK._l_field_id(_CgPP, :pos)
    seam_pos_before = RK._seam_field_canon(seam)[:pos]
    entry = RK._l_seam_entry_current(seam)
    pg = RK._l_seam_plan_graph(seam, ri)
    writes_detached() = [s.key.id for s in
        RK.lower_leaf_schedule(ir, RK._seam_field_canon(seam), pg; entry_current = entry) if s.kind === :write]
    writes_spec() = [s.key.id for s in
        RK.lower_leaf_schedule(ir, _CgPP, pg; entry_current = entry) if s.kind === :write]
    base_detached = writes_detached()
    @test base_detached == writes_spec()                        # agree BEFORE the poison

    g = RK.kernel_graph(_CgPP)
    orig = _CgPP.ports[:pos]
    try
        _CgPP.ports[:pos] = RK.value!(g, :__pos_poison, Float64) # POISON the selected :pos field mapping
        @test RK._l_field_id(_CgPP, :pos) != live_pos_before    # the LIVE spec path DID drift (real poison)
        @test RK._seam_field_canon(seam)[:pos] == seam_pos_before # the DETACHED seam map did NOT
        @test writes_detached() == base_detached                # detached schedule UNCHANGED under poison
        @test writes_spec() != base_detached                    # the SPEC path drifts → detached is load-bearing
        _, m2 = _cg_compile(seam, ir)                           # full compile off the poisoned spec
        @test string(m2.body) == string(m1.body)               # emitted body identical — seam-derived, not spec
    finally
        _CgPP.ports[:pos] = orig                                # restore the fixture
    end
    @test RK._l_field_id(_CgPP, :pos) == live_pos_before        # fixture restored
end

@testset "codegen — entry-currentness is the seam's proven kernel_plan_entry_current" begin
    seam = _cg_seam()
    ec = RK._l_seam_entry_current(seam)
    @test ec == Set(RK.kernel_plan_entry_current(seam))
    @test RK._seam_canon(seam, :dpot_dpos) in ec && RK._seam_canon(seam, :pos) in ec
end

@testset "codegen — the emitted leaf is the 3-line leapfrog with NO residual call, ONE pgrad" begin
    seam = _cg_seam()
    _, meta = _cg_compile(seam, _cg_ir())
    src = string(meta.body)
    @test !occursin("leapfrog", src) && !occursin("step_f", src)   # no residual integrator call
    @test meta.pgrad_execs == 1                                 # exactly one gradient RECIPE exec per leaf
    @test count(a -> occursin("pgrad!", string(a)), meta.body.args) == 1
    @test occursin("_owner_slot(state", src) && occursin("_owner_commit!(state", src)  # factory _OwnerState
end

@testset "codegen — stepsize is a TYPED RUNTIME parameter (NOT baked); binder Token is validated" begin
    seam = _cg_seam(); ir = _cg_ir()
    _, meta = _cg_compile(seam, ir; step = 0.1)
    src = string(meta.body)
    @test occursin("getfield(stepkw, :stepsize)", src)          # loaded at runtime from the binder NamedTuple
    @test !occursin("0.1", src)                                 # the numeric value is NOT baked into the code
    @test meta.bound_formals == (:stepsize,)
    # binder soundness: a wrong-integrator Token is rejected; an unapproved wrapper is rejected
    other = RK.kernel_registration(_CgFix.euclidean_phasepoint)  # a DIFFERENT registered token (not leapfrog!)
    if other !== nothing
        @test_throws RK._LLowerReject RK.resolve_step_binding(_CgFix.partial(_CgFix.euclidean_phasepoint), seam)
    end
    @test_throws RK._LLowerReject RK.resolve_step_binding((x -> x), seam)        # arbitrary callable refused
    @test_throws RK._LLowerReject RK.resolve_step_binding(_CgLF, seam)           # bare kernel (not a binder)
end

@testset "codegen — ONE specialization runs same-typed .1/.25 stepsizes CORRECTLY (adaptation-safe)" begin
    T = Float64; d = 5; seam = _cg_seam()
    fn, meta = _cg_compile(seam, _cg_ir(); step = 0.1)          # compiled ONCE
    metric = Matrix{T}(2 * I, d, d); cholf = cholesky(metric)
    # analytic leapfrog reference for U=½‖x‖² (∇U=x), M=2I
    ref(pos0, mom0, g0, ε) = begin
        mh = mom0 .- (ε/2) .* g0; vel = cholf \ mh; pr = pos0 .+ ε .* vel; mr = mh .- (ε/2) .* pr; (pr, mr)
    end
    for ε in (0.1, 0.25)                                        # SAME NamedTuple type, different Float64 value
        st = _cg_state(seam, T, d, meta)
        pos0 = copy(RK._owner_slot(st, Val(meta.posslot))); mom0 = copy(RK._owner_slot(st, Val(meta.momslot)))
        g0 = copy(RK._owner_slot(st, Val(meta.dpotslot)))
        pg = _CountGrad(_quad!, 0)
        fn(st, pg, cholf, (; stepsize = ε))                    # runtime stepsize threaded — no recompile
        @test pg.n == 1
        pr, mr = ref(pos0, mom0, g0, ε)
        @test RK._owner_slot(st, Val(meta.posslot)) ≈ pr       # the CORRECT ε executed (not a baked 0.1)
        @test RK._owner_slot(st, Val(meta.momslot)) ≈ mr
    end
end

@testset "codegen — executes over real F32/F64 slots: parity, ONE pgrad!, @inferred, warm 0 B" begin
    for T in (Float64, Float32)
        seam = _cg_seam(); ε = T(0.1); d = 5
        fn, meta = _cg_compile(seam, _cg_ir(); step = ε)
        metric = Matrix{T}(2 * I, d, d); cholf = cholesky(metric)
        st = _cg_state(seam, T, d, meta)
        pos0 = copy(RK._owner_slot(st, Val(meta.posslot)))
        mom0 = copy(RK._owner_slot(st, Val(meta.momslot)))
        g0   = copy(RK._owner_slot(st, Val(meta.dpotslot)))     # ∇U(pos0) = pos0 (quadratic)
        kw = (; stepsize = ε)

        pg = _CountGrad(_quad!, 0)
        fn(st, pg, cholf, kw)
        @test pg.n == 1                                         # EXACTLY one pgrad! per leaf

        mom_h = mom0 .- (ε / 2) .* g0
        vel   = cholf \ mom_h
        pos_r = pos0 .+ ε .* vel
        mom_r = mom_h .- (ε / 2) .* pos_r                       # half kick with NEW gradient (∇U(pos_r)=pos_r)
        @test RK._owner_slot(st, Val(meta.posslot)) ≈ pos_r
        @test RK._owner_slot(st, Val(meta.momslot)) ≈ mom_r
        @test RK._owner_slot(st, Val(meta.dpotslot)) ≈ pos_r    # canonical grad slot = ∇U(new pos)
        @test RK._owner_slot(st, Val(meta.potslot)) ≈ sum(abs2, pos_r) / 2   # committed pot

        fn(st, pg, cholf, kw)                                   # warm
        @inferred fn(st, pg, cholf, kw)
        @test (@allocated fn(st, pg, cholf, kw)) == 0          # typed exact 0 B
    end
end

@testset "codegen — authored op identity is PRESERVED as a canonical GlobalRef (never bare spelling)" begin
    # RK 05:00 #3: a bare-symbol callee could bind a different (shadowed) function. The emitted ops must be
    # exact-identity GlobalRefs, and an unqualified authored op is rejected.
    @test RK._exec_callee(GlobalRef(Base, :-)) === GlobalRef(Base, :-)
    @test RK._exec_callee(:(Base.:-)) === GlobalRef(Base, :-)            # qualified Expr → canonical GlobalRef
    @test_throws RK._LLowerReject RK._exec_callee(:-)                    # bare symbol: ambiguous identity
    # the fused broadcast is materialize!/broadcasted (identity-preserving AND 0-alloc), not @__dot__ spelling
    seam = _cg_seam(); _, meta = _cg_compile(seam, _cg_ir())
    src = string(meta.body)
    @test occursin("materialize!", src) && occursin("broadcasted", src)
end

module _CgTypedFix
    include(joinpath(@__DIR__, "..", "benchmark", "nuts_kernel_authoring_fixture.jl"))
    using ReactiveKernels
    # the ccb35d3 typed-half leapfrog form: `oftype(stepsize, 0.5)` (F32-safe, no Float64 promotion)
    @kernel leapfrog_typed!(phasepoint; stepsize) = begin
        @. phasepoint.mom -= oftype(stepsize, 0.5) * stepsize * phasepoint.dham_dpos
        @. phasepoint.pos +=                        stepsize * phasepoint.dham_dmom
        @. phasepoint.mom -= oftype(stepsize, 0.5) * stepsize * phasepoint.dham_dpos
    end
end

@testset "codegen — typed-half (oftype) leapfrog: scalar op is a PLAIN call, 0 B, F32-safe" begin
    LFT = _CgTypedFix.leapfrog_typed!
    seam = RK._kernel_factory_endpoint_plan(_CgPP, RK.kernel_registration(LFT))
    ir = RK.method_irs(LFT)[1]
    prod = Dict(kv[1] => kv[2] for kv in RK.kernel_plan_producer(seam))
    ap = Dict(prod[RK._seam_canon(seam, :dpot_dpos)] => :pgrad, prod[RK._seam_canon(seam, :dkin_dmom)] => :velocity)
    ri = RK._seam_recipe_inputs_snapshot(_CgPP, seam)
    _, meta = RK.compile_leaf(ir, seam, _CgTypedFix.partial(LFT; stepsize = 0.1); recipe_inputs = ri, appliers = ap)
    src = string(meta.body)
    # MIXED expression `oftype(stepsize, 0.5) * stepsize * dham_dpos` (RK 05:26): the typed SCALAR
    # coefficient is a PLAIN call (evaluated once), the VECTOR multiply is FUSED (broadcasted).
    @test occursin("oftype(", src)
    @test !occursin("broadcasted(Main._CgTypedFix.oftype", src) && !occursin("broadcasted(Base.oftype", src)
    @test occursin("broadcasted(Main._CgTypedFix.:*", src)              # the vector op IS fused
    # scalar coefficient appears once per half-kick RHS (2 half-kicks) — evaluated once each, not per-element
    @test count(_ -> true, eachmatch(r"oftype\(", src)) == 2
    barrier(fn, st, pg, cholf, kw) = @allocated fn(st, pg, cholf, kw)   # fn typed → true 0-B measurement
    for T in (Float64, Float32)
        fn, m = RK.compile_leaf(ir, seam, _CgTypedFix.partial(LFT; stepsize = T(0.1)); recipe_inputs = ri, appliers = ap)
        d = 4; metric = Matrix{T}(2 * I, d, d); cholf = cholesky(metric); st = RK.make_leaf_owner_state(_CgPP, seam, T, d)
        RK._owner_commit!(st, ntuple(i -> begin s = RK._owner_slot(st, Val(i))
            s isa AbstractVector ? (s .= T.(1:d) ./ (i + 1)) : s end, m.nslots))
        _quad!(RK._owner_slot(st, Val(m.dpotslot)), RK._owner_slot(st, Val(m.posslot)))
        pg = _CountGrad(_quad!, 0); kw = (; stepsize = T(0.1))
        barrier(fn, st, pg, cholf, kw); barrier(fn, st, pg, cholf, kw)
        @test eltype(RK._owner_slot(st, Val(m.posslot))) === T           # no Float64 promotion (F32-safe)
        @test barrier(fn, st, pg, cholf, kw) == 0                        # typed half is 0 alloc
    end
end

@testset "codegen — an exec whose selected Recipe has no bound applier is REJECTED" begin
    seam = _cg_seam()
    @test_throws RK._LLowerReject RK.compile_leaf(_cg_ir(), seam,
        _CgFix.partial(_CgLF; stepsize = 0.1); recipe_inputs = _cg_ri(seam), appliers = Dict{Int,Symbol}())
end
