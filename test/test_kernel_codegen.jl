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

@testset "codegen — an exec whose selected Recipe has no bound applier is REJECTED" begin
    seam = _cg_seam()
    @test_throws RK._LLowerReject RK.compile_leaf(_cg_ir(), seam,
        _CgFix.partial(_CgLF; stepsize = 0.1); recipe_inputs = _cg_ri(seam), appliers = Dict{Int,Symbol}())
end
