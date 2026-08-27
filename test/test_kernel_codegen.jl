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
using Random
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

module _CgStatsFix
    using ReactiveKernels, LogExpFunctions
    smooth(prev, new, w) = (1 - w) * prev + w * new
    min1exp(x) = x >= 0 ? one(x) : exp(x)
    @rk_pure smooth 3                                   # source-only boundary: helpers must be DECLARED,
    @rk_pure min1exp 1                                  # not normalized as opaque through the emitter
    # nuts_stats!-shaped: scalar subject writes (n_steps, acceptance_rate) + a nested indexed write
    @kernel probe_stats!(s; accepted) = begin
        s.n_steps += 1
        s.acceptance_rate = smooth(s.acceptance_rate, min1exp(accepted), 0.1)
        s.trees[1].log_weight[1] = s.dham
    end
end

@testset "codegen — opaque exact-pure Base calls (oftype) REJECT pending the pure-primitive category" begin
    # RK 05:54: MethodIR over the final ccb kernels shows the opaque exact-pure calls are
    # oftype/one/zero/isnothing/logaddexp/eachcol. The source-only guard correctly REJECTS an opaque
    # unregistered call rather than normalizing it from spelling. Once syntax lands the definition-time
    # exact pure-primitive category these capture as identity-bound `_RegisteredCall :primitive`, the
    # emitter lowers them, and the 0-B typed-half execution + a no-opaque-edge assertion return.
    LFT = _CgTypedFix.leapfrog_typed!
    seam = RK._kernel_factory_endpoint_plan(_CgPP, RK.kernel_registration(LFT))
    ir = RK.method_irs(LFT)[1]
    prod = Dict(kv[1] => kv[2] for kv in RK.kernel_plan_producer(seam))
    ap = Dict(prod[RK._seam_canon(seam, :dpot_dpos)] => :pgrad, prod[RK._seam_canon(seam, :dkin_dmom)] => :velocity)
    ri = RK._seam_recipe_inputs_snapshot(_CgPP, seam)
    @test_throws RK._LLowerReject RK.compile_leaf(ir, seam, _CgTypedFix.partial(LFT; stepsize = 0.1);
                                                  recipe_inputs = ri, appliers = ap)
end

@testset "codegen — ORDINARY array calls are NOT broadcast (broadcast context is authority)" begin
    # RK 05:31: slot kind partitions scalars only INSIDE an authored broadcast node. A structural index
    # and a DECLARED registered call with array/slot arguments must stay plain, even in a broadcast context.
    seam = _cg_seam()
    ctx() = RK._EmitCtx(RK._seam_field_canon(seam), RK._seam_slot_of_canon(seam),
                        RK._seam_scalar_canons(seam), (:stepsize,), Int[], Symbol[])
    mom = RK._SelfField((:mom,))
    # getindex(mom, 1) via a structural _Index place → a plain getindex, never broadcast
    e_get = RK._emit_place_ref(RK._Index(mom, (RK._Lit(1),)), ctx())
    @test e_get.head === :call && !occursin("broadcasted", string(e_get)) && occursin("getindex", string(e_get))
    # a DECLARED registered helper call is a plain call — never broadcast, even in a broadcast context
    accw = only(w for w in RK._exec_place_writes(RK.method_irs(_CgStatsFix.probe_stats!)[1])
                if w.owner == (:acceptance_rate,))
    c2 = RK._EmitCtx(Dict(:acceptance_rate => 2), Dict(2 => 2), Set{Int}([2]), (:accepted,), Int[], Symbol[])
    @test !occursin("broadcasted", string(RK._exec_rhs(accw.rhs, c2, true)))    # registered call: plain even bcast=true
    @test occursin("smooth", string(RK._exec_rhs(accw.rhs, c2, false)))
    # an arithmetic operator over a slot IS fused iff inside a broadcast context
    mul = RK._OpCall(GlobalRef(Base, :*), (RK._Lit(0.5), mom), (), false, :operator_candidate)
    @test occursin("broadcasted", string(RK._exec_rhs(mul, ctx(), true)))       # bcast=true → fused
    @test !occursin("broadcasted", string(RK._exec_rhs(mul, ctx(), false)))     # bcast=false → plain
end

# a SYNTHETIC concrete Val-ABI state (TEST-ONLY; syntax's _CanonOwnedN replaces it on the rebase): per-slot
# mutable fields, `_owner_slot`/`_owner_set!` by Val index → constant getfield/setfield (0-B).
mutable struct _SynthStats
    n_steps::Int
    acceptance_rate::Float64
    dham::Float64
    trees::Vector{NamedTuple{(:log_weight,),Tuple{Vector{Float64}}}}
end
ReactiveKernels._owner_slot(s::_SynthStats, ::Val{I}) where {I} = getfield(s, I)
ReactiveKernels._owner_set!(s::_SynthStats, ::Val{I}, v) where {I} = (setfield!(s, I, v); s)

@testset "codegen — EXECUTABLE nuts_stats! scalar + indexed writes update real state at 0 B" begin
    ir = RK.method_irs(_CgStatsFix.probe_stats!)[1]
    @test ir.ok                                                # source-only boundary satisfied (no opaque edge)
    # the helper calls resolve to DECLARED registrations (not opaque), with exact captured identity
    accw = only(w for w in RK._exec_place_writes(ir) if w.owner == (:acceptance_rate,))
    @test accw.rhs isa RK._RegisteredCall                      # smooth is a declared call, not _OpCall :opaque
    @test getfield(accw.rhs.registration, :source) === _CgStatsFix.smooth
    @test accw.rhs.registration.kind === :declared_effect
    writes = RK._exec_place_writes(ir)
    @test length(writes) == 3
    # synthetic layout: n_steps→slot1, acceptance_rate→2, dham→3 (SCALARS), trees→4 (buffer)
    fieldcanon = Dict(:n_steps => 1, :acceptance_rate => 2, :dham => 3, :trees => 4)
    slotof     = Dict(1 => 1, 2 => 2, 3 => 3, 4 => 4)
    scalars    = Set([1, 2, 3])                              # trees (4) is a buffer, not scalar
    stmts = Any[]; formals = Symbol[]
    for pw in writes
        ctx = RK._EmitCtx(fieldcanon, slotof, scalars, (:accepted,), Int[], Symbol[])
        RK._emit_place_write!(stmts, pw, ctx, formals)
    end
    src = join(string.(stmts), "\n")
    @test occursin("_owner_set!(state, Val(1)", src)           # n_steps scalar strong write
    @test occursin("_owner_set!(state, Val(2)", src)           # acceptance_rate scalar strong write
    @test occursin("setindex!", src) && occursin("log_weight", src)  # nested indexed write, NOT a local assign
    @test !occursin("broadcasted", src)                        # ordinary calls, never broadcast
    loads = Any[:($(RK._kw_local(f)) = getfield(stepkw, $(QuoteNode(f)))) for f in formals]
    fn = RK.compile(:((state, stepkw) -> $(Expr(:block, loads..., stmts..., :(return state)))))
    st = _SynthStats(5, 0.4, -1.25, [(; log_weight = [-Inf, -Inf])])
    fn(st, (; accepted = 1.0))
    @test st.n_steps == 6                                      # n_steps += 1 updated REAL state
    @test st.acceptance_rate ≈ 0.9 * 0.4 + 0.1 * 1.0          # smooth(0.4, 1.0, 0.1)
    @test st.trees[1].log_weight[1] == -1.25                  # trees[1].log_weight[1] = dham
    barrier(f, s, kw) = @allocated f(s, kw)
    barrier(fn, st, (; accepted = 1.0))
    @test barrier(fn, st, (; accepted = 1.0)) == 0            # exact 0 B
end

module _CgUnregFix
    using ReactiveKernels
    unregged(a, b) = a + b                              # NOT @rk_* declared — an opaque helper
    @kernel bad_stats!(s; x) = begin
        s.acceptance_rate = unregged(s.acceptance_rate, x)
    end
end

@testset "codegen — an UNREGISTERED helper is REJECTED at the source-only boundary (not normalized)" begin
    ir = RK.method_irs(_CgUnregFix.bad_stats!)[1]
    writes = RK._exec_place_writes(ir)
    accw = only(w for w in writes if w.owner == (:acceptance_rate,))
    @test accw.rhs isa RK._OpCall && accw.rhs.hint === :opaque   # opaque, not a _RegisteredCall
    ctx = RK._EmitCtx(Dict(:acceptance_rate => 2), Dict(2 => 2), Set{Int}([2]), (:x,), Int[], Symbol[])
    # the emitter REFUSES to normalize the opaque unregistered helper through spelling
    @test_throws RK._LLowerReject RK._emit_place_write!(Any[], accw, ctx, Symbol[])
end

module _CgRefreshFix
    using ReactiveKernels, LinearAlgebra, Random
    # refresh_momentum!!-shaped: ordered primitive effects; rng is a runtime keyword; mom OWNED, chol SHARED
    @kernel refresh!!(phasepoint; rng) = begin
        Random.randn!(rng, phasepoint.mom)
        LinearAlgebra.lmul!(phasepoint.chol_metric.L, phasepoint.mom)
        return phasepoint                                  # EXPLICIT return (final source form)
    end
    # renamed runtime-rng formal — ordering must still come from descriptor.rng_arg, not the name
    @kernel refresh_renamed!!(phasepoint; generator) = begin
        Random.randn!(generator, phasepoint.mom)
        LinearAlgebra.lmul!(phasepoint.chol_metric.L, phasepoint.mom)
        return phasepoint
    end
    # an UNEXPECTED extra effect statement (a third primitive) — must reject (exactly two ordered effects)
    @kernel refresh_extra!!(phasepoint; rng) = begin
        Random.randn!(rng, phasepoint.mom)
        LinearAlgebra.lmul!(phasepoint.chol_metric.L, phasepoint.mom)
        Random.randn!(rng, phasepoint.mom)
        return phasepoint
    end
end

# SYNTHETIC state with DISTINCT ABSOLUTE-SUPERSET indices: shared chol=1, owned mom=2 (role-local both=1
# would mask the absolute-index ABI bug syntax just fixed). Plus a currentness bitmask.
mutable struct _SynthRefresh{T}
    chol::Cholesky{T,Matrix{T}}                            # SHARED, absolute slot 1
    mom::Vector{T}                                         # OWNED, absolute slot 2
    current::UInt64                                        # currentness mask
end
ReactiveKernels._shared_slot(s::_SynthRefresh, ::Val{1}) = getfield(s, :chol)
ReactiveKernels._owner_slot(s::_SynthRefresh, ::Val{2}) = getfield(s, :mom)
# currentness bits are ABSOLUTE STORAGE slot indices (Val-encoded tuples): slot I → bit (I-1). @generated
# so the packing (here UInt64; syntax's is NTuple{W,UInt}) is a compile-time constant → one typed store.
@generated function ReactiveKernels._owner_kill_current!(s::_SynthRefresh, ::Val{K}) where {K}
    kill = reduce(|, (UInt64(1) << (i - 1) for i in K); init = UInt64(0))
    :(setfield!(s, :current, getfield(s, :current) & ~$kill); s)
end
@generated function ReactiveKernels._owner_bless_current!(s::_SynthRefresh, ::Val{B}) where {B}
    bless = reduce(|, (UInt64(1) << (i - 1) for i in B); init = UInt64(0))
    :(setfield!(s, :current, getfield(s, :current) | $bless); s)
end

@testset "codegen — EXECUTABLE effect root (refresh shape): generic sequence, role-aware, currentness, F32/F64 0 B" begin
    # ABSOLUTE superset slots: chol_metric(20)=1 shared, mom(10)=2 owned, kinetic dkin(11)=3/kin(12)=4 owned,
    # dpot(8)=5 owned (position-gradient, NOT a mom dependent — stays current). killmap: mom kills kinetic.
    fieldcanon = Dict(:mom => 10, :chol_metric => 20)
    slotof     = Dict(10 => 2, 20 => 1, 11 => 3, 12 => 4, 8 => 5)
    roleof     = Dict(10 => :owned, 20 => :shared)
    killmap    = Dict(10 => [11, 12])                      # mom (10) → kinetic descendants (dkin, kin)
    ir = RK.method_irs(_CgRefreshFix.refresh!!)[1]
    fn, meta = RK.compile_effect_root(ir, fieldcanon, slotof, Set{Int}(), roleof, killmap)
    @test length(meta.calls) == 2 && meta.owned_slots == [2]       # 2 ordered effects; mom (abs 2) only owned write
    @test meta.bless_slots == (2,) && meta.kill_slots == (3, 4)     # bless mom slot, kill kinetic slots (absolute)
    # ACCEPTANCE DISCRIMINATOR (NOT compiler authorization): the refresh body IS exactly randn! then lmul!
    effs = [st for st in ir.body if st isa RK._ExprStmt]
    @test getfield(effs[1].expr.registration, :source) === Random.randn!
    @test getfield(effs[2].expr.registration, :source) === LinearAlgebra.lmul!
    src = join(string.(meta.calls), " ")
    @test findfirst("randn!", src)[1] < findfirst("lmul!", src)[1]  # descriptor/source order preserved
    @test occursin("__s2", src) && meta.owned_slots == [2]         # OWNED mom bound to a local at abs slot 2
    @test occursin("_shared_slot(state, Val(1))", src)             # SHARED chol via _shared_slot at abs 1
    @test occursin("_kernel_property", src) && occursin(":L", src)  # chol_metric.L via core exact-type accessor
    # renamed rng formal → SAME emission (ordering from descriptor.rng_arg, not the name)
    ir2 = RK.method_irs(_CgRefreshFix.refresh_renamed!!)[1]
    _, meta2 = RK.compile_effect_root(ir2, fieldcanon, slotof, Set{Int}(), roleof, killmap; rng = :generator)
    @test replace(join(string.(meta.calls), " "), "rng" => "R") ==
          replace(join(string.(meta2.calls), " "), "generator" => "R")
    # NEGATIVE: an unlowered extra statement is fine (generic sequence) but a missing role rejects; a
    # kill canon with no absolute slot rejects
    @test_throws RK._LLowerReject RK.compile_effect_root(ir, fieldcanon, slotof, Set{Int}(),
        Dict(10 => :owned), killmap)                              # chol role missing -> reject
    @test_throws RK._LLowerReject RK.compile_effect_root(ir, fieldcanon, Dict(10 => 2, 20 => 1),
        Set{Int}(), roleof, killmap)                             # kill canon 11/12 has no slot -> reject
    # REJECT bless/kill OVERLAP: a written root that is its own dependent (mom → [mom]) is contradictory
    @test_throws RK._LLowerReject RK.compile_effect_root(ir, fieldcanon, slotof, Set{Int}(),
        roleof, Dict(10 => [10]))
    # EXECUTE over real state, F32 and F64: mom mutated in place, currentness updated, @inferred, 0 B
    barrier(f, s, r) = @allocated f(s, r)
    for T in (Float64, Float32)
        d = 4; M = Matrix{T}(2 * I, d, d)
        st = _SynthRefresh{T}(cholesky(M), zeros(T, d), typemax(UInt64))   # start all-current
        rng = Random.MersenneTwister(1)
        fn(st, rng)
        @test any(st.mom .!= 0) && eltype(st.mom) === T                    # randn!→lmul! filled+scaled mom
        @test (st.current >> 1) & 1 == 1                                   # mom (slot 2, bit 1) BLESSED current
        @test (st.current >> 2) & 1 == 0 && (st.current >> 3) & 1 == 0     # kinetic (slots 3,4) KILLED
        @test (st.current >> 4) & 1 == 1                                   # dpot (slot 5) stays CURRENT
        @inferred fn(st, rng)
        barrier(fn, st, rng); barrier(fn, st, rng)
        @test barrier(fn, st, rng) == 0                                    # exact 0 B
    end
    # EXCEPTION SOUNDNESS (RK 06:13): a mid-sequence throw (lmul! with a mismatched chol) leaves mom AND
    # kinetic DIRTY (killed before each effect, never blessed); a retry with a valid chol recomputes+blesses.
    let T = Float64
        st = _SynthRefresh{T}(cholesky(Matrix{T}(2 * I, 3, 3)), zeros(T, 4), typemax(UInt64))  # 3×3 chol vs len-4 mom
        @test_throws DimensionMismatch fn(st, Random.MersenneTwister(2))
        @test (st.current >> 1) & 1 == 0                                   # mom DIRTY after the throw (not blessed)
        @test (st.current >> 2) & 1 == 0 && (st.current >> 3) & 1 == 0     # kinetic DIRTY
        st.chol = cholesky(Matrix{T}(2 * I, 4, 4))                         # retry with a valid factor
        fn(st, Random.MersenneTwister(2))
        @test (st.current >> 1) & 1 == 1                                   # mom re-BLESSED on the successful retry
    end
end

module _CgVecFix
    using ReactiveKernels
    vadd(a::Number, b::Number) = a + b                     # a declared PURE scalar helper (broadcasts elementwise)
    @rk_pure vadd 2
    @kernel probe_vec!(phasepoint;) = begin
        @. phasepoint.mom = vadd(phasepoint.mom, phasepoint.dham_dmom)   # declared helper over VECTOR slot args
    end
end

@testset "codegen — vector vs scalar registered calls: fuse over vectors, plain when scalar/non-dotted" begin
    # RK 06:02: registered-call vector-ness derives recursively from slot types, not `x.broadcast` alone.
    seam = _cg_seam()
    ctx() = RK._EmitCtx(RK._seam_field_canon(seam), RK._seam_slot_of_canon(seam),
                        RK._seam_scalar_canons(seam), (), Int[], Symbol[])
    # a DECLARED helper over VECTOR slot args, in a broadcast context → FUSED (broadcasted)
    vcall = only(w.rhs for w in RK._exec_place_writes(RK.method_irs(_CgVecFix.probe_vec!)[1])
                 if w.owner == (:mom,))
    @test occursin("broadcasted", string(RK._exec_rhs(vcall, ctx(), true)))    # vector args + bcast → fused
    @test !occursin("broadcasted", string(RK._exec_rhs(vcall, ctx(), false)))  # non-dotted → plain registered call
    # a DECLARED helper over SCALAR args stays a PLAIN call even in a broadcast context (no re-alloc)
    c2 = RK._EmitCtx(Dict(:acceptance_rate => 2), Dict(2 => 2), Set{Int}([2]), (:accepted,), Int[], Symbol[])
    scall = only(w.rhs for w in RK._exec_place_writes(RK.method_irs(_CgStatsFix.probe_stats!)[1])
                 if w.owner == (:acceptance_rate,))
    @test !occursin("broadcasted", string(RK._exec_rhs(scall, c2, true)))      # scalar registered call under @. → plain
end

mutable struct _SynthCopy{T}
    pos::Vector{T}                                          # owned buffer slot 1
    mom::Vector{T}                                          # owned buffer slot 2
    pot::T                                                  # owned scalar slot 3
    current::UInt64
end
ReactiveKernels._owner_copy_slot!(d::_SynthCopy, s::_SynthCopy, ::Val{1}) = (copyto!(d.pos, s.pos); d)
ReactiveKernels._owner_copy_slot!(d::_SynthCopy, s::_SynthCopy, ::Val{2}) = (copyto!(d.mom, s.mom); d)
ReactiveKernels._owner_copy_slot!(d::_SynthCopy, s::_SynthCopy, ::Val{3}) = (setfield!(d, :pot, getfield(s, :pot)); d)
@generated function ReactiveKernels._owner_copy_current!(d::_SynthCopy, s::_SynthCopy, ::Val{S}) where {S}
    m = reduce(|, (UInt64(1) << (i - 1) for i in S); init = UInt64(0))
    :(setfield!(d, :current, (getfield(d, :current) & ~$m) | (getfield(s, :current) & $m)); d)
end

@testset "codegen — EXECUTABLE copy!! strong-update: full owned closure, dest identity, currentness, 0 B" begin
    fn, meta = RK.compile_copy([1, 2, 3])
    @test meta.slots == (1, 2, 3)
    dest = _SynthCopy{Float64}(zeros(2), zeros(2), 0.0, UInt64(0))     # all-dirty
    src  = _SynthCopy{Float64}([1.0, 2.0], [3.0, 4.0], 9.0, typemax(UInt64))  # all-current
    p = dest.pos                                                       # capture dest's buffer identity
    r = fn(dest, src)
    @test r === dest                                                  # result === dest (identity preserved)
    @test dest.pos == src.pos && dest.mom == src.mom && dest.pot == src.pot   # COMPLETE owned closure transferred
    @test dest.pos === p                                             # copyto! in place — buffer NOT rebound
    @test dest.current == 0b111                                       # owned-slot (1,2,3) currentness transferred
    barrier(f, a, b) = @allocated f(a, b); barrier(fn, dest, src)
    @test barrier(fn, dest, src) == 0                                # exact 0 B
    @inferred fn(dest, src)
    @test_throws RK._LLowerReject RK.compile_copy([1, 2, 2])          # duplicate (un-collapsed alias) rejects
    @test_throws RK._LLowerReject RK.compile_copy(Int[])             # empty closure rejects
end

module _CgTransFix
    using ReactiveKernels, LinearAlgebra, Random
    # a MIXED transition body: registered effects (refresh) THEN a subject place-write (a step counter)
    @kernel transition!!(s; rng) = begin
        Random.randn!(rng, s.mom)
        LinearAlgebra.lmul!(s.chol_metric.L, s.mom)
        s.n_steps += 1
        return s
    end
    @kernel refresh_only!!(s; rng) = begin              # ONLY registered effects, no later place-write
        Random.randn!(rng, s.mom)
        LinearAlgebra.lmul!(s.chol_metric.L, s.mom)
        return s
    end
    @kernel double_step!!(s; rng) = begin               # multi-write scalar — source-order (fresh reads)
        s.n_steps += 1
        s.n_steps += 1
        return s
    end
end

mutable struct _SynthTrans{T}
    chol::Cholesky{T,Matrix{T}}                            # SHARED slot 1
    mom::Vector{T}                                         # OWNED slot 2
    n_steps::Int                                          # OWNED scalar slot 3
    current::UInt64
end
ReactiveKernels._shared_slot(s::_SynthTrans, ::Val{1}) = getfield(s, :chol)
ReactiveKernels._owner_slot(s::_SynthTrans, ::Val{2}) = getfield(s, :mom)
ReactiveKernels._owner_slot(s::_SynthTrans, ::Val{3}) = getfield(s, :n_steps)
ReactiveKernels._owner_set!(s::_SynthTrans, ::Val{3}, v) = (setfield!(s, :n_steps, v); s)
@generated function ReactiveKernels._owner_kill_current!(s::_SynthTrans, ::Val{K}) where {K}
    m = reduce(|, (UInt64(1) << (i - 1) for i in K); init = UInt64(0)); :(setfield!(s, :current, getfield(s, :current) & ~$m); s)
end
@generated function ReactiveKernels._owner_bless_current!(s::_SynthTrans, ::Val{B}) where {B}
    m = reduce(|, (UInt64(1) << (i - 1) for i in B); init = UInt64(0)); :(setfield!(s, :current, getfield(s, :current) | $m); s)
end

@testset "codegen — OUTER EPOCH: mixed effects+write transition, single deferred bless, exception soundness" begin
    fieldcanon = Dict(:mom => 10, :chol_metric => 20, :n_steps => 30)
    slotof     = Dict(10 => 2, 20 => 1, 30 => 3)
    roleof     = Dict(10 => :owned, 20 => :shared, 30 => :owned)
    scalars    = Set([30])                                 # n_steps is a scalar owned slot
    killmap    = Dict{Int,Vector{Int}}()                   # (no derived dependents in this demonstrator)
    ir = RK.method_irs(_CgTransFix.transition!!)[1]
    fn, meta = RK.compile_transition_root(ir, fieldcanon, slotof, scalars, roleof, killmap)
    @test Set(meta.written) == Set([10, 30])              # mom + n_steps are the written roots
    @test meta.bless_slots == (2, 3)                      # ONE deferred bless of both written slots
    # SUCCESS: both mom (slot 2) and n_steps (slot 3) blessed only after the WHOLE body (deferred commit)
    let T = Float64
        st = _SynthTrans{T}(cholesky(Matrix{T}(2 * I, 4, 4)), zeros(T, 4), 0, UInt64(0))
        fn(st, Random.MersenneTwister(1), (;))
        @test st.n_steps == 1 && any(st.mom .!= 0)
        @test (st.current >> 1) & 1 == 1 && (st.current >> 2) & 1 == 1   # mom(slot2)+n_steps(slot3) BLESSED
        barrier(f, a, b, c) = @allocated f(a, b, c); barrier(fn, st, Random.MersenneTwister(1), (;))
        @test barrier(fn, st, Random.MersenneTwister(1), (;)) == 0       # 0 B
    end
    # EXCEPTION: lmul! throws (mismatched chol) BEFORE the n_steps write + the single commit → mom is DIRTY
    # (killed pre-effect, never blessed by a nested early commit); n_steps untouched.
    let T = Float64
        st = _SynthTrans{T}(cholesky(Matrix{T}(2 * I, 3, 3)), zeros(T, 4), 0, typemax(UInt64))
        @test_throws DimensionMismatch fn(st, Random.MersenneTwister(2), (;))
        @test (st.current >> 1) & 1 == 0                                 # mom NOT blessed (deferred commit unreached)
        @test st.n_steps == 0                                           # the write after lmul! never ran
    end
    # EFFECTS-ONLY transition (no later place write) also executes F32/F64 0 B
    let iro = RK.method_irs(_CgTransFix.refresh_only!!)[1]
        fno, mo = RK.compile_transition_root(iro, fieldcanon, slotof, scalars, roleof, killmap)
        @test Set(mo.written) == Set([10]) && mo.bless_slots == (2,)
        barrier(f, a, b, c) = @allocated f(a, b, c)
        for T in (Float64, Float32)
            st = _SynthTrans{T}(cholesky(Matrix{T}(2 * I, 4, 4)), zeros(T, 4), 0, UInt64(0))
            fno(st, Random.MersenneTwister(1), (;)); barrier(fno, st, Random.MersenneTwister(1), (;))
            @test barrier(fno, st, Random.MersenneTwister(1), (;)) == 0
        end
    end
    # MULTI-WRITE SCALAR: two n_steps += 1 must read FRESH each time (source-order) → 2, not 1 (no stale local)
    let ird = RK.method_irs(_CgTransFix.double_step!!)[1]
        fnd, _ = RK.compile_transition_root(ird, fieldcanon, slotof, scalars, roleof, killmap)
        st = _SynthTrans{Float64}(cholesky(Matrix{Float64}(2 * I, 4, 4)), zeros(4), 0, UInt64(0))
        fnd(st, Random.MersenneTwister(1), (;))
        @test st.n_steps == 2                                          # fresh reads: 0→1→2, not a stale 0→1
    end
end

struct _CgEvilProp end
Base.getproperty(::_CgEvilProp, ::Symbol) = error("side-effecting getproperty must NOT run in hot lowering")

@testset "codegen — property accessor is SANCTIONED-ONLY (custom getproperty rejects, never runs)" begin
    # RK 06:22: only Cholesky.L and NamedTuple fields are sanctioned; a custom `getproperty` is refused
    # (and never invoked — the reject fires before it).
    @test RK._kernel_property(cholesky(Matrix{Float64}(2 * I, 3, 3)), Val(:L)) isa LowerTriangular
    @test RK._kernel_property((; log_weight = [1.0, 2.0]), Val(:log_weight)) == [1.0, 2.0]  # NamedTuple field
    @test_throws RK._LLowerReject RK._kernel_property(_CgEvilProp(), Val(:x))                # rejected, not run
end

@testset "codegen — representation census (WIP): refresh_momentum!! + nuts_stats! exact identities, non-opaque" begin
    # RK 05:36/05:47: the final ccb source adds refresh_momentum!! + nuts_stats!; assert their exact Mode-2
    # identities are non-opaque (fold into test_kernel_methodir's 8-kernel loop on the rebase).
    # refresh: registered Random.randn!/LinearAlgebra.lmul! ORDERED primitive effects; mom write + chol read.
    rir = RK.method_irs(_CgRefreshFix.refresh!!)[1]
    @test rir.ok && :opaque_call ∉ rir.effects                     # non-opaque
    reffs = [st.expr for st in rir.body if st isa RK._ExprStmt]
    @test length(reffs) == 2 && all(c -> c isa RK._RegisteredCall && c.registration.kind === :primitive, reffs)
    @test getfield(reffs[1].registration, :source) === Random.randn!         # ordered: randn! first
    @test getfield(reffs[2].registration, :source) === LinearAlgebra.lmul!   # then lmul!
    @test getfield(reffs[1].registration, :primitive_effect).rng_arg === 1   # runtime rng ordering
    @test getfield(reffs[1].registration, :primitive_effect).writes == (2,)  # mom-only write closure
    # nuts_stats: scalar subject writes n_steps/acceptance_rate; declared smooth/min1exp (NOT opaque).
    sir = RK.method_irs(_CgStatsFix.probe_stats!)[1]
    @test sir.ok && :opaque_call ∉ sir.effects
    owners = [w.owner[1] for w in RK._exec_place_writes(sir)]
    @test :n_steps in owners && :acceptance_rate in owners
    accw = only(w for w in RK._exec_place_writes(sir) if w.owner == (:acceptance_rate,))
    @test accw.rhs isa RK._RegisteredCall && accw.rhs.registration.kind === :declared_effect  # smooth declared
    @test getfield(accw.rhs.args[2].registration, :source) === _CgStatsFix.min1exp             # nested min1exp declared
end

@testset "codegen — an exec whose selected Recipe has no bound applier is REJECTED" begin
    seam = _cg_seam()
    @test_throws RK._LLowerReject RK.compile_leaf(_cg_ir(), seam,
        _CgFix.partial(_CgLF; stepsize = 0.1); recipe_inputs = _cg_ri(seam), appliers = Dict{Int,Symbol}())
end
