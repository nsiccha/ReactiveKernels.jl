# Gates for the executable public NUTS root (kernel_nuts.jl): the real captured nuts_state + refresh_momentum!!
# + nuts!! compile to a runnable, type-stable, exception-safe, RNG-independent public root that mutates a
# concrete _NutsFrame in place under one outer epoch. Uses the faithful ccb authoring fixture as the driver.
using Test, LinearAlgebra, Random
using ReactiveKernels
const RK = ReactiveKernels

module _NutsFix
    include(joinpath(@__DIR__, "..", "benchmark", "nuts_kernel_authoring_fixture.jl"))
end

# --- construction boilerplate (a prepared frame from the fixture's euclidean_phasepoint + leapfrog!) --------
_nuts_pf() = RK._prepare_factory(_NutsFix.euclidean_phasepoint, RK.kernel_registration(_NutsFix.leapfrog!))
function _nuts_mkvals(pf, T)
    PL = RK.kernel_prepared_plan(pf); m = T[2 0; 0 2]; d = Dict{Int,Any}()
    for sl in RK.kernel_plan_slots(PL)
        nm = String(sl.path[1])
        d[sl.canon] = nm == "grad_f" ? ((dst, p) -> (dst .= 2 .* p; sum(abs2, p))) :
            nm == "metric" ? m : nm == "chol_metric" ? cholesky(m) : startswith(nm, "##node") ? zero(T) :
            nm == "pos" ? T[1, 2] : nm == "mom" ? T[3, 4] :
            (nm in ("dpot_dpos", "dham_dpos", "dkin_dmom", "dham_dmom")) ? T[0, 0] : zero(T)
    end
    d
end
function _nuts_frame(pf, T, md)
    frame = RK._construct_nuts_frame(pf, _nuts_mkvals(pf, T), md;
                                     step_f = RK.partial(_NutsFix.leapfrog!; stepsize = T(0.1)), stats_f = nothing, min_dham = -1000)
    RK.compile_prepared_initialization(pf, typeof(frame.init), typeof(frame.shared))(frame.init, frame.shared, RK.kernel_prepared_handles(pf))
    RK._seed_nuts_children!(frame)
    frame
end
_slot(pf, ep, f) = RK._canon_slot(ep, RK.kernel_plan_named_slot_val(RK.kernel_prepared_plan(pf), Val(f)))

@testset "kernel_nuts — executable public root: real transition, result===state, epoch commit" begin
    pf = _nuts_pf(); frame = _nuts_frame(pf, Float64, 3)
    C = RK.compile_nuts(pf, _NutsFix.nuts_state, _NutsFix.refresh_momentum!!, _NutsFix.nuts!!, frame)
    @test C.RootToken === RK.kernel_token(_NutsFix.nuts!!)
    p0 = copy(_slot(pf, frame.init, :pos)); m0 = copy(_slot(pf, frame.init, :mom))
    r = C.root!(frame, C.scratch, Random.Xoshiro(1))
    @test r === frame                                          # authored return contract (result===state)
    @test _slot(pf, frame.init, :pos) != p0                   # full transition mutated init (terminal copy!! ran)
    @test _slot(pf, frame.init, :mom) != m0                   # refresh_momentum!! ran
    @test RK.diagnostics_committed_mask(frame.diag) == UInt(0x0f)   # single deferred epoch commit blessed all 4
    @test RK._diag_slot(frame.diag, Val(4)) != 0.0            # dham is a real (fresh-momentum) energy error, not stale 0
end

# typed-batch allocation gate (RK): a fully-typed Val{N} while loop, warmed, so the LOOP path (not just a
# single call) is measured — this is what an HMC sampler drives. The public root must be EXACTLY 0-B here.
@inline function _nuts_batch(root, fr, sc, rng, ::Val{N}) where {N}
    i = 0; @inbounds while i < N; root(fr, sc, rng); i += 1; end; nothing
end
# measured with a LITERAL Val{N} (a runtime Val(N) would itself be type-unstable and allocate)
_nuts_batch0b(root, fr, sc, rng, ::Val{N}) where {N} =
    (_nuts_batch(root, fr, sc, rng, Val(N)); GC.gc(); @allocated _nuts_batch(root, fr, sc, rng, Val(N)))

@testset "kernel_nuts — public root @inferred + EXACT 0-B IN A LOOP (typed batch n=64/128/256, both rng, F32+F64)" begin
    for T in (Float64, Float32)
        pf = _nuts_pf(); frame = _nuts_frame(pf, T, 5)
        C = RK.compile_nuts(pf, _NutsFix.nuts_state, _NutsFix.refresh_momentum!!, _NutsFix.nuts!!, frame)
        rt = Base.return_types(C.root!, (typeof(frame), typeof(C.scratch), typeof(Random.Xoshiro(1))))
        @test length(rt) == 1 && isconcretetype(rt[1])       # concrete _NutsFrame return
        @test RK.diagnostics_ham_type(frame.diag) === T       # diagnostics carry the frame's ham type (F32/F64)
        @inferred C.root!(frame, C.scratch, Random.Xoshiro(1))
        for rg in (Random.Xoshiro(91), Random.MersenneTwister(2))
            @test _nuts_batch0b(C.root!, frame, C.scratch, rg, Val(64)) == 0
            @test _nuts_batch0b(C.root!, frame, C.scratch, rg, Val(128)) == 0
            @test _nuts_batch0b(C.root!, frame, C.scratch, rg, Val(256)) == 0
        end
    end
end

@testset "kernel_nuts — RNG-INDEPENDENT: one sampler accepts two rng types, scratch/root types unchanged" begin
    pf = _nuts_pf(); frame = _nuts_frame(pf, Float64, 3)
    C = RK.compile_nuts(pf, _NutsFix.nuts_state, _NutsFix.refresh_momentum!!, _NutsFix.nuts!!, frame)
    # rng is a threaded RuntimeArg (never spilled), so the SAME root!/scratch run under two distinct rng types
    C.root!(frame, C.scratch, Random.Xoshiro(1))
    C.root!(frame, C.scratch, Random.MersenneTwister(2))
    @test true                                                # both concrete rng types accepted, one scratch object
end

@testset "kernel_nuts — refresh math on a DENSE SPD metric matches chol.L*z + 0-B (uplo-correct, non-diagonal)" begin
    # a DENSE (non-diagonal) SPD metric: a wrong unconditional LowerTriangular(factors) / wrong-uplo shortcut
    # would give a different momentum than the reference `chol.L * z`; a diagonal metric would not catch it.
    pf = _nuts_pf(); PL = RK.kernel_prepared_plan(pf); M = [2.0 0.5; 0.5 3.0]
    d = _nuts_mkvals(pf, Float64)
    for sl in RK.kernel_plan_slots(PL)
        nm = String(sl.path[1]); nm == "metric" && (d[sl.canon] = M); nm == "chol_metric" && (d[sl.canon] = cholesky(M))
    end
    frame = RK._construct_nuts_frame(pf, d, 3; step_f = RK.partial(_NutsFix.leapfrog!; stepsize = 0.1), stats_f = nothing, min_dham = -1000)
    RK.compile_prepared_initialization(pf, typeof(frame.init), typeof(frame.shared))(frame.init, frame.shared, RK.kernel_prepared_handles(pf))
    RK._seed_nuts_children!(frame)
    C = RK.compile_nuts(pf, _NutsFix.nuts_state, _NutsFix.refresh_momentum!!, _NutsFix.nuts!!, frame)
    z = randn(Random.Xoshiro(7), 2); expected = cholesky(M).L * z    # reference: randn! then chol.L*z
    C.refresh(frame.init, frame.shared, C.cfg.handles, Random.Xoshiro(7))   # same seed
    @test _slot(pf, frame.init, :mom) ≈ expected                    # matches the reference on a DENSE metric
    zerob(rf, ep, sh, h, rg) = (rf(ep, sh, h, rg); GC.gc(); @allocated rf(ep, sh, h, rg))
    @test zerob(C.refresh, frame.init, frame.shared, C.cfg.handles, Random.Xoshiro(7)) == 0   # refresh exact 0-B
end

@testset "kernel_nuts — refresh exception safety: chol shape-throw leaves mom+dependents DIRTY (retry repairs)" begin
    pf = _nuts_pf(); frame = _nuts_frame(pf, Float64, 3)
    C = RK.compile_nuts(pf, _NutsFix.nuts_state, _NutsFix.refresh_momentum!!, _NutsFix.nuts!!, frame)
    PL = RK.kernel_prepared_plan(pf)
    momslot = RK.kernel_plan_named_slot_val(PL, Val(:mom)); hamslot = RK.kernel_plan_named_slot_val(PL, Val(:ham))
    RK._canon_set!(frame.shared, RK.kernel_plan_named_slot_val(PL, Val(:chol_metric)), cholesky([2.0 0 0; 0 2 0; 0 0 2]))
    @test_throws Exception C.refresh(frame.init, frame.shared, C.cfg.handles, Random.Xoshiro(3))
    @test !RK._canon_current(frame.init, momslot)             # mom never falsely current after a mid-write throw
    @test !RK._canon_current(frame.init, hamslot)             # dependent kinetic/ham stay dirty
end

# --- altered-root adversaries: a DRIFTED nuts!! MethodIR must be REJECTED (source authority, not hardcoded) --
@testset "kernel_nuts — altered-root adversaries: drifted nuts!! is rejected (IR-derived, not hardcoded)" begin
    good = RK.method_irs(_NutsFix.nuts!!)[1]
    tok = RK.kernel_token(_NutsFix.refresh_momentum!!)
    b = collect(good.body)
    # reconstruct the root IR from the real template, varying ONLY the body (immutable — rebuild all fields)
    mkroot(body) = RK.MethodIR(good.id, good.self, good.formals, Tuple(body), good.control, good.effects,
                               good.deps, good.kind, good.ok, good.reason, good.signature)
    # POSITIVE: the real authored nuts!! validates
    @test RK._derive_public_root_ops(good, :step!, :rng, tok) == [:refresh, :step]
    # wrong refresh Token (statement 1 not the captured refresh)
    @test_throws Exception RK._derive_public_root_ops(good, :step!, :rng, Symbol("##not_refresh##"))
    # wrong rng keyword/formal (validate against a different runtime-arg name)
    @test_throws Exception RK._derive_public_root_ops(good, :step!, :notrng, tok)
    # wrong subject method name
    @test_throws Exception RK._derive_public_root_ops(good, :notstep, :rng, tok)
    # EXTRA statement (duplicate step) — length != 3
    @test_throws Exception RK._derive_public_root_ops(mkroot(vcat(b, [b[2]])), :step!, :rng, tok)
    # REORDERED (step before refresh)
    @test_throws Exception RK._derive_public_root_ops(mkroot([b[2], b[1], b[3]]), :step!, :rng, tok)
    # POST-RETURN statement (return no longer last)
    @test_throws Exception RK._derive_public_root_ops(mkroot([b[1], b[3], b[2]]), :step!, :rng, tok)
end

@testset "kernel_nuts — END-TO-END callable nuts!!(sampler; rng): Mode-2 dispatch, loop 0-B, token/rng rejects" begin
    OwnerToken = RK.kernel_token(_NutsFix.nuts_state)
    for T in (Float64, Float32)
        pf = _nuts_pf()
        frame = RK._prepare_nuts_frame(pf, _nuts_mkvals(pf, T), 5;
                                       step_f = RK.partial(_NutsFix.leapfrog!; stepsize = T(0.1)), stats_f = nothing, min_dham = -1000)
        C = RK.compile_nuts(pf, _NutsFix.nuts_state, _NutsFix.refresh_momentum!!, _NutsFix.nuts!!, frame)
        # OwnerToken (nuts_state subject-methods) and RootToken (the nuts!! Mode-2 skeleton) are DISTINCT tokens,
        # both preserved into the concrete sampler type — the dispatch gate keys on RootToken, OwnerToken is free.
        @test OwnerToken !== C.RootToken
        sampler = RK.nuts_sampler(Val(OwnerToken), Val(C.RootToken), frame, C.root!, C.scratch)
        @test RK.kernel_token(sampler) === OwnerToken                 # owner token preserved on the KernelObject
        # THE authored public call — the Mode-2 nuts!! skeleton IS the callable
        r = _NutsFix.nuts!!(sampler; rng = Random.Xoshiro(1))
        @test RK.nuts_sampler_frame(r) === frame                      # result === state
        @test RK._canon_slot(frame.init, RK.kernel_plan_named_slot_val(RK.kernel_prepared_plan(pf), Val(:mom))) isa Vector{T}
        # EXACT 0-B in a loop THROUGH THE PUBLIC PATH (hoisted rng), both rng types
        @inline function pub(skel, samp, rg, ::Val{N}) where {N}
            i = 0; @inbounds while i < N; skel(samp; rng = rg); i += 1; end; nothing
        end
        pub0b(skel, samp, rg, ::Val{N}) where {N} = (pub(skel, samp, rg, Val(N)); GC.gc(); @allocated pub(skel, samp, rg, Val(N)))
        for rg in (Random.Xoshiro(91), Random.MersenneTwister(2))
            @test pub0b(_NutsFix.nuts!!, sampler, rg, Val(64)) == 0
            @test pub0b(_NutsFix.nuts!!, sampler, rg, Val(128)) == 0
        end
        @inferred _NutsFix.nuts!!(sampler; rng = Random.Xoshiro(1))
        # rng is a REQUIRED KEYWORD — missing / positional rng is rejected
        @test_throws Exception _NutsFix.nuts!!(sampler)
        @test_throws Exception _NutsFix.nuts!!(sampler, Random.Xoshiro(1))
        # a sampler whose handle RootToken differs from the nuts!! skeleton token finds NO Mode-2 method
        wrong = RK.nuts_sampler(Val(OwnerToken), Val(:not_the_root_token), frame, C.root!, C.scratch)
        @test_throws MethodError _NutsFix.nuts!!(wrong; rng = Random.Xoshiro(1))
    end
end

@testset "kernel_nuts — EFFECTFUL stats_f: nuts_stats! writes inlined on the frame (n_steps advances), 0-B" begin
    pf = _nuts_pf()
    # effectful stats: stats_f = nuts_stats! (writes n_steps + acceptance_rate per leaf)
    frame = RK._construct_nuts_frame(pf, _nuts_mkvals(pf, Float64), 4;
                                     step_f = RK.partial(_NutsFix.leapfrog!; stepsize = 0.1), stats_f = _NutsFix.nuts_stats!, min_dham = -1000)
    RK.compile_prepared_initialization(pf, typeof(frame.init), typeof(frame.shared))(frame.init, frame.shared, RK.kernel_prepared_handles(pf))
    RK._seed_nuts_children!(frame)
    @test RK.stats_binding_registration(RK.nuts_frame_stats(frame)) !== nothing   # effectful binding
    C = RK.compile_nuts(pf, _NutsFix.nuts_state, _NutsFix.refresh_momentum!!, _NutsFix.nuts!!, frame)
    C.root!(frame, C.scratch, Random.Xoshiro(1))
    @test RK._diag_slot(frame.diag, Val(1)) > 0                # n_steps advanced (stats_f ran per leaf)
    @test isfinite(RK._diag_slot(frame.diag, Val(3)))         # acceptance_rate written
    # effectful stats stays exact 0-B in a loop
    for rg in (Random.Xoshiro(91), Random.MersenneTwister(2))
        @test _nuts_batch0b(C.root!, frame, C.scratch, rg, Val(64)) == 0
    end
end

@testset "kernel_nuts — effectful stats validator is TOTAL: an extra effect cannot disappear" begin
    good = RK.method_irs(_NutsFix.nuts_stats!)[1]; b = collect(good.body)
    mk(body) = RK.MethodIR(good.id, good.self, good.formals, Tuple(body), good.control, good.effects,
                           good.deps, good.kind, good.ok, good.reason, good.signature)
    @test RK._validate_stats_body(good, (1, 3)) === nothing              # POSITIVE: real nuts_stats! validates
    # an EXTRA non-PlaceWrite statement (a stand-in effect) before the return: a filtered census would DROP it;
    # the total validator REJECTS it
    @test_throws Exception RK._validate_stats_body(mk([b[1], b[2], RK._ExprStmt(RK._SelfRef()), b[3]]), (1, 3))
    # a POST-RETURN statement rejects
    @test_throws Exception RK._validate_stats_body(mk([b[1], b[3], b[2]]), (1, 3))
    # produced-slot mismatch (declared vs written) rejects
    @test_throws Exception RK._validate_stats_body(good, (1,))
end

@testset "kernel_nuts — STABLE type identity: two same-input compiles give identical root/fn/scratch/sampler types" begin
    OwnerToken = RK.kernel_token(_NutsFix.nuts_state)
    pf = _nuts_pf()
    C1 = RK.compile_nuts(pf, _NutsFix.nuts_state, _NutsFix.refresh_momentum!!, _NutsFix.nuts!!, _nuts_frame(pf, Float64, 5))
    C2 = RK.compile_nuts(pf, _NutsFix.nuts_state, _NutsFix.refresh_momentum!!, _NutsFix.nuts!!, _nuts_frame(pf, Float64, 5))
    @test typeof(C1.root!) === typeof(C2.root!)               # deterministic emission -> identical RGF/root type
    @test typeof(C1.fn) === typeof(C2.fn)
    @test typeof(C1.scratch) === typeof(C2.scratch)
    @test typeof(C1.cfg) === typeof(C2.cfg)                   # leaf + ensures RGF types stable too
    @test C1.RootToken === C2.RootToken
    # so two independently-built samplers share ONE concrete KernelObject type (stable prepared identity)
    f1 = _nuts_frame(pf, Float64, 5); f2 = _nuts_frame(pf, Float64, 5)
    Ca = RK.compile_nuts(pf, _NutsFix.nuts_state, _NutsFix.refresh_momentum!!, _NutsFix.nuts!!, f1)
    Cb = RK.compile_nuts(pf, _NutsFix.nuts_state, _NutsFix.refresh_momentum!!, _NutsFix.nuts!!, f2)
    s1 = RK.nuts_sampler(Val(OwnerToken), Val(Ca.RootToken), f1, Ca.root!, Ca.scratch)
    s2 = RK.nuts_sampler(Val(OwnerToken), Val(Cb.RootToken), f2, Cb.root!, Cb.scratch)
    @test typeof(s1) === typeof(s2)                           # RK's repeated same-signature type-identity gate
    @test s1 !== s2 && RK.nuts_sampler_frame(s1) !== RK.nuts_sampler_frame(s2)   # but values/buffers isolated
end

@testset "kernel_nuts — refresh concrete-domain gate: valid factor/rng pass, custom RNG rejects before exec" begin
    pf = _nuts_pf(); frame = _nuts_frame(pf, Float64, 3)
    C = RK.compile_nuts(pf, _NutsFix.nuts_state, _NutsFix.refresh_momentum!!, _NutsFix.nuts!!, frame)
    H = RK.kernel_prepared_handles(pf)
    # Xoshiro + MersenneTwister pass the per-rng randn! domain (already exercised for loop-0-B elsewhere)
    C.refresh(frame.init, frame.shared, H, Random.Xoshiro(1))
    C.refresh(frame.init, frame.shared, H, Random.MersenneTwister(2))
    @test true
    # a custom RNG is REJECTED by the per-rng randn! domain guard BEFORE any kill/write — values AND masks stay
    struct _CustomRNG <: Random.AbstractRNG end
    momslot = RK.kernel_plan_named_slot_val(RK.kernel_prepared_plan(pf), Val(:mom))
    mom_before = copy(RK._canon_slot(frame.init, momslot)); cur_before = RK._canon_current(frame.init, momslot)
    @test_throws ArgumentError C.refresh(frame.init, frame.shared, H, _CustomRNG())
    @test RK._canon_slot(frame.init, momslot) == mom_before        # mom value untouched
    @test RK._canon_current(frame.init, momslot) == cur_before     # mom currentness mask untouched (rejected before kill)
end

mutable struct _CntGrad; n::Int; end
(g::_CntGrad)(dst, p) = (g.n += 1; dst .= 2 .* p; sum(abs2, p))
struct _CustomRNG2 <: Random.AbstractRNG end

# top-level typed batch (an inner closure boxes and would itself allocate — HMC review)
@inline function _refresh_batch(rf, ep, sh, h, r, ::Val{N}) where {N}
    i = 0; @inbounds while i < N; rf(ep, sh, h, r); i += 1; end; nothing
end
_refresh_batch0b(rf, ep, sh, h, r, ::Val{N}) where {N} =
    (_refresh_batch(rf, ep, sh, h, r, Val(N)); GC.gc(); @allocated _refresh_batch(rf, ep, sh, h, r, Val(N)))
# type-aware content snapshot: arrays/Cholesky by copied CONTENTS, scalars by value (egal, NaN-safe),
# everything else (functor, nothing) by object identity — a `deepcopy;==` snapshot is unsound for the
# external mutable grad functor and Cholesky (HMC review).
_rsnap(v) = v isa LinearAlgebra.Cholesky ? copy(v.factors) : v isa AbstractArray ? copy(v) : v isa Number ? v : Base.objectid(v)
_rsame(v, s) = v isa LinearAlgebra.Cholesky ? v.factors == s : v isa AbstractArray ? v == s : v isa Number ? isequal(v, s) : Base.objectid(v) === s

@testset "kernel_nuts — refresh rng-domain reject leaves ALL owned/shared values+masks + grad counter UNCHANGED" begin
    pf = _nuts_pf(); PL = RK.kernel_prepared_plan(pf); pg = _CntGrad(0); m = Float64[2 0.5; 0.5 3]
    d = Dict{Int,Any}()
    for sl in RK.kernel_plan_slots(PL); nm = String(sl.path[1])
        d[sl.canon] = nm=="grad_f" ? pg : nm=="metric" ? m : nm=="chol_metric" ? cholesky(m) : startswith(nm,"##node") ? 0.0 :
            nm=="pos" ? [1.0,2.0] : nm=="mom" ? [3.0,4.0] : (nm in ("dpot_dpos","dham_dpos","dkin_dmom","dham_dmom")) ? [0.0,0.0] : 0.0
    end
    frame = RK._construct_nuts_frame(pf, d, 3; step_f=RK.partial(_NutsFix.leapfrog!;stepsize=0.1), stats_f=nothing, min_dham=-1000)
    RK.compile_prepared_initialization(pf, typeof(frame.init), typeof(frame.shared))(frame.init, frame.shared, RK.kernel_prepared_handles(pf))
    RK._seed_nuts_children!(frame)
    C = RK.compile_nuts(pf, _NutsFix.nuts_state, _NutsFix.refresh_momentum!!, _NutsFix.nuts!!, frame); H = RK.kernel_prepared_handles(pf)
    # admitted RNGs pass + stay EXACT 0-B in a typed loop (the generated guard folds away)
    for rg in (Random.Xoshiro(91), Random.MersenneTwister(2))
        @test _refresh_batch0b(C.refresh, frame.init, frame.shared, H, rg, Val(64)) == 0
        @test _refresh_batch0b(C.refresh, frame.init, frame.shared, H, rg, Val(128)) == 0
        @test _refresh_batch0b(C.refresh, frame.init, frame.shared, H, rg, Val(256)) == 0
    end
    # snapshot EVERY canon slot's value (contents) + currentness, owned (init) AND shared, plus the grad counter
    slots = unique([(RK.kernel_plan_field(PL, sl.canon)[1], RK.kernel_plan_field(PL, sl.canon)[2]) for sl in RK.kernel_plan_slots(PL)])
    obj(role) = role === :owned ? frame.init : frame.shared
    vals = Dict((r,s) => _rsnap(RK._canon_slot(obj(r), Val(s))) for (r,s) in slots)
    curs = Dict((r,s) => RK._canon_current(obj(r), Val(s)) for (r,s) in slots); grad0 = pg.n
    @test_throws ArgumentError C.refresh(frame.init, frame.shared, H, _CustomRNG2())   # rejected BEFORE any kill
    @test all(_rsame(RK._canon_slot(obj(r), Val(s)), vals[(r,s)]) for (r,s) in slots)  # every owned+shared VALUE unchanged
    @test all(RK._canon_current(obj(r), Val(s)) === curs[(r,s)] for (r,s) in slots)    # every owned+shared currentness MASK unchanged
    @test pg.n == grad0                                                                # grad counter unchanged (no producer ran)
end
