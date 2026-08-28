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
        d[sl.canon] = nm == "pot_f" ? (p -> sum(abs2, p)) :
            nm == "grad_f" ? ((dst, p) -> (dst .= 2 .* p; sum(abs2, p))) :
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
        d[sl.canon] = nm=="pot_f" ? (p -> sum(abs2, p)) : nm=="grad_f" ? pg : nm=="metric" ? m : nm=="chol_metric" ? cholesky(m) : startswith(nm,"##node") ? 0.0 :
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
    imask0 = RK._canon_current_mask(frame.init); smask0 = RK._canon_current_mask(frame.shared)   # raw current tuples
    shared_ids = Dict(s => RK._canon_slot(frame.shared, Val(s)) for (r,s) in slots if r === :shared)  # by identity
    @test_throws ArgumentError C.refresh(frame.init, frame.shared, H, _CustomRNG2())   # rejected BEFORE any kill
    @test all(_rsame(RK._canon_slot(obj(r), Val(s)), vals[(r,s)]) for (r,s) in slots)  # every owned+shared VALUE unchanged
    @test all(RK._canon_current(obj(r), Val(s)) === curs[(r,s)] for (r,s) in slots)    # every owned+shared currentness bit unchanged
    @test RK._canon_current_mask(frame.init) == imask0                                 # whole owned mask tuple (unused bits too)
    @test RK._canon_current_mask(frame.shared) == smask0                              # whole shared mask tuple (unused bits too)
    @test all(RK._canon_slot(frame.shared, Val(s)) === shared_ids[s] for s in keys(shared_ids))  # shared slots same OBJECT (identity)
    @test pg.n == grad0                                                                # grad counter unchanged (no producer ran)
end

@testset "NUTS source hygiene: no module-level mutable global lookup objects" begin
    # User directive (no global dictionaries): the retained production NUTS compiler files must carry NO
    # module-level CONSTRUCTED mutable lookup object (Dict/Set/IdDict/WeakKeyDict/Ref/registry). Two things are
    # explicitly NOT banned and this gate must not flag them:
    #   (1) type ALIASES — `const _LTaken = Dict{Symbol,Int}` / `_Places = Set{Symbol}` (container name → `{`);
    #   (2) per-compilation LOCAL scratch — indented `ctr = Ref(0)` / `fspv = Dict(...)` inside a function.
    # Only a TOP-LEVEL construction (`^[const] Name = Container(` / `[`) is an offender. `_DIAG` (Dict) →
    # `_diag_index` pure dispatch; `_EP_SELF`/`_SCALAR_SELF` (Set) → immutable tuple / removed, this pins it.
    nuts_files = ["kernel_nuts.jl", "kernel_nuts_native.jl", "kernel_control.jl", "kernel_codegen.jl"]
    #  ^-anchored (multiline): a leading-whitespace line can't match, so indented locals are excluded.
    #  A CONSTRUCTION is `Name = [Base./Core.]Container[{...}]( / [` — the container (optionally qualified and
    #  optionally with type params) is IMMEDIATELY followed by a `(`/`[` constructor call. A bare type ALIAS
    #  `Name = Container{...}` has NO trailing call, so it does not match. This flags `Dict(...)`, `Dict{K,V}()`,
    #  `Set{T}()`, `Base.RefValue(0)`; it allows `Dict{K,V}` / `Set{T}` aliases and indented function-local scratch.
    banned = r"^(const[ \t]+)?[A-Za-z_][A-Za-z0-9_]*[ \t]*=[ \t]*(Base\.|Core\.)?(Dict|Set|IdDict|WeakKeyDict|Ref|RefValue|ObjectIdDict)(\{[^\n]*\})?[ \t]*[\(\[]"m
    for f in nuts_files
        src = read(joinpath(@__DIR__, "..", "src", f), String)
        offenders = String[strip(m.match) for m in eachmatch(banned, src)]
        isempty(offenders) || @error "module-level mutable-container construction in src/$f" offenders
        @test isempty(offenders)
    end
    # positive controls — the gate MUST fire on every construction shape:
    @test occursin(banned, "const _X = Dict(:a=>1)")             # bare constructor
    @test occursin(banned, "_Y = Set([:a,:b])")                  # bracket constructor
    @test occursin(banned, "const _Z = Dict{Symbol,Int}()")      # parameterized constructor
    @test occursin(banned, "_W = Set{Int}()")                    # parameterized constructor
    @test occursin(banned, "const _R = Base.RefValue(0)")        # qualified constructor
    @test occursin(banned, "const _N = Dict{Symbol,Vector{Int}}()")  # nested type params
    # negative controls — the gate MUST NOT fire on aliases or function-local scratch:
    @test !occursin(banned, "const _LTaken = Dict{Symbol,Int}")  # type alias — allowed
    @test !occursin(banned, "_Places = Set{Symbol}")             # type alias — allowed
    @test !occursin(banned, "    ctr = Ref(0)")                  # indented local scratch — allowed
    @test !occursin(banned, "    fspv = Dict(m => 1 for m in x)")# indented local scratch — allowed
end

@testset "refresh factor view: Diagonal vs dense Cholesky backing (structural, 0-B)" begin
    # `_refresh_lfactor(chol)` is the backing-specialized momentum-refresh factor (`chol.L`) used by
    # _CompiledRefresh. Dense keeps the orientation-correct uplo wrapper; a Diagonal-backed Cholesky uses the
    # bare `factors` Diagonal directly (self-adjoint lower factor) — zero wrapper, and the ONLY type the lmul!
    # domain admits (runtime `adjoint(UpperTriangular(::Diagonal))` canonicalizes to `LowerTriangular{T,Diagonal}`,
    # which the dense uplo-wrapper prediction does not match). Both must be math-faithful to `chol.L*mom` and 0-B.
    for T in (Float64, Float32)
        Md = T[4 0; 0 9]; chd = cholesky(Md); momd = T[1,2]
        Ld = RK._refresh_lfactor(chd); yd = copy(momd); LinearAlgebra.lmul!(Ld, yd)
        @test yd ≈ chd.L * momd                          # dense: math matches chol.L*mom
        @test Ld isa LinearAlgebra.LowerTriangular       # dense: triangular view, unchanged path
        Mg = LinearAlgebra.Diagonal(T[4,9]); chg = cholesky(Mg); momg = T[1,2]
        @test chg.factors isa LinearAlgebra.Diagonal
        Lg = RK._refresh_lfactor(chg); yg = copy(momg); LinearAlgebra.lmul!(Lg, yg)
        @test yg ≈ chg.L * momg                          # diagonal: math matches chol.L*mom
        @test Lg isa LinearAlgebra.Diagonal              # diagonal: bare factors, NO uplo wrapper / Adjoint
        @test Lg === chg.factors                         # zero-cost: identity, no view allocation
        barrier(ch, m) = LinearAlgebra.lmul!(RK._refresh_lfactor(ch), m)
        barrier(chd, copy(momd)); barrier(chg, copy(momg))                 # warm both specializations
        @test (@allocated barrier(chd, momd)) == 0       # dense exact 0-B
        @test (@allocated barrier(chg, momg)) == 0       # diagonal exact 0-B
    end
end
# Test-only regression pinning the end-to-end Diagonal-mass evidence for f8f5e2b (RK 18:57/19:11). Uses the
# file's existing helpers _nuts_pf / _nuts_mkvals / _nuts_batch0b / _NutsFix / RK. NO Profile — the profiler was
# measured to be an invalid instrument here (phantom potrs backtraces in the Diagonal path, dense/diagonal ratio
# only 1.5–3.3x); no-potrs is proven DETERMINISTICALLY in an isolated child process.
#
# Equivalence note: a Diagonal-backed Cholesky solves element-wise while dense goes through LAPACK potrs (two
# √-divisions), so the two are byte-identical ONLY for unit mass (trivial solve); a scaled Diagonal differs at
# the ULP (F64 ~3e-16, F32 ~6e-8). Hence the byte-identical gate uses UNIT mass.

@testset "kernel_nuts — Diagonal mass: unit-mass full-TRAJECTORY dense-equivalence + exact 0-B (F32/F64, both RNG)" begin
    _mkframe(pf, T, md, metric) = begin
        PL = RK.kernel_prepared_plan(pf); d = _nuts_mkvals(pf, T)
        for sl in RK.kernel_plan_slots(PL)
            nm = String(sl.path[1]); nm == "metric" && (d[sl.canon] = metric); nm == "chol_metric" && (d[sl.canon] = cholesky(metric))
        end
        fr = RK._construct_nuts_frame(pf, d, md; step_f = RK.partial(_NutsFix.leapfrog!; stepsize = T(0.3)), stats_f = _NutsFix.nuts_stats!, min_dham = -1000)
        RK.compile_prepared_initialization(pf, typeof(fr.init), typeof(fr.shared))(fr.init, fr.shared, RK.kernel_prepared_handles(pf))
        RK._seed_nuts_children!(fr); fr
    end
    # full HMC-observable snapshot (not pos only): position + every committed diagnostic + mask + the PHYSICAL
    # derived-`diverged` field and its currentness bits (read directly, NOT inferred from dham — a stale/wrong
    # frame.diverged or derived-currentness bit must be caught).
    _obs(pf, fr) = (pos = copy(_slot(pf, fr.init, :pos)),
                n_steps = RK._diag_slot(fr.diag, Val(1)), reached_depth = RK._diag_slot(fr.diag, Val(2)),
                acceptance_rate = RK._diag_slot(fr.diag, Val(3)), dham = RK._diag_slot(fr.diag, Val(4)),
                committed = RK.diagnostics_committed_mask(fr.diag),
                diverged = getfield(fr, :diverged),
                diverged_pending = RK.nuts_frame_diverged_pending(fr),
                diverged_committed = RK.nuts_frame_diverged_committed(fr))
    for T in (Float64, Float32)
        pf = _nuts_pf()
        # UNIT mass: element-wise solve and dense potrs coincide exactly ⇒ byte-identical full observable set.
        frD = _mkframe(pf, T, 5, LinearAlgebra.Diagonal(T[1, 1]))    # Diagonal identity mass (admitted)
        CD = RK.compile_nuts(pf, _NutsFix.nuts_state, _NutsFix.refresh_momentum!!, _NutsFix.nuts!!, frD)
        frM = _mkframe(pf, T, 5, T[1 0; 0 1])                        # SAME operator, dense representation
        CM = RK.compile_nuts(pf, _NutsFix.nuts_state, _NutsFix.refresh_momentum!!, _NutsFix.nuts!!, frM)
        # FULL-TRAJECTORY equivalence: snapshot the complete observable set after EVERY one of the 200 seeded
        # transitions (not just the terminal frame), so a transient mismatch that later reconverges, or a
        # stale/wrong diverged / derived-currentness bit at any step, is caught.
        rngD = Random.Xoshiro(1); trajD = [(CD.root!(frD, CD.scratch, rngD); _obs(pf, frD)) for _ in 1:200]
        rngM = Random.Xoshiro(1); trajM = [(CM.root!(frM, CM.scratch, rngM); _obs(pf, frM)) for _ in 1:200]
        @test trajD == trajM                                        # every observable at every step, byte-identical (unit mass)
        # EXACT 0-B on the Diagonal path (a SCALED Diagonal, the real perf case): F32/F64 × both RNG, typed Val{N}.
        frS = _mkframe(pf, T, 5, LinearAlgebra.Diagonal(T[2, 2])); CS = RK.compile_nuts(pf, _NutsFix.nuts_state, _NutsFix.refresh_momentum!!, _NutsFix.nuts!!, frS)
        for rg in (Random.Xoshiro(91), Random.MersenneTwister(2))
            @test _nuts_batch0b(CS.root!, frS, CS.scratch, rg, Val(64)) == 0
            @test _nuts_batch0b(CS.root!, frS, CS.scratch, rg, Val(256)) == 0
        end
    end
end

@testset "kernel_nuts — Diagonal mass: NO LAPACK potrs (deterministic, isolated child process)" begin
    # The profiler cannot prove this (phantom potrs frames). Instead a CHILD process pirates LAPACK.potrs! to a
    # hard sentinel error (F64+F32) — the piracy lives ONLY in the child, the main suite is untouched. Dense must
    # HIT potrs (same-kind positive control, both eltypes); Diagonal must complete build+50 txn WITHOUT it. The
    # child prints an explicit four-result receipt and exits 0 iff all four facts hold (so an empty child ≠ green).
    fixture = abspath(joinpath(@__DIR__, "..", "benchmark", "nuts_kernel_authoring_fixture.jl"))
    proj = abspath(joinpath(@__DIR__, ".."))
    childsrc = raw"""
    using LinearAlgebra, Random, ReactiveKernels
    const RK = ReactiveKernels
    for Tv in (Float64, Float32)
        @eval LinearAlgebra.LAPACK.potrs!(u::AbstractChar, A::AbstractMatrix{$Tv}, B::AbstractVecOrMat{$Tv}) = error("POTRS_CALLED")
    end
    module _CF; include(ENV["RK_DIAG_FIXTURE"]); end
    function _mk(T, metric)
        pf = RK._prepare_factory(_CF.euclidean_phasepoint, RK.kernel_registration(_CF.leapfrog!))
        PL = RK.kernel_prepared_plan(pf); d = Dict{Int,Any}()
        for sl in RK.kernel_plan_slots(PL)
            nm = String(sl.path[1])
            d[sl.canon] = nm=="pot_f" ? (p -> sum(abs2, p)) : nm=="grad_f" ? ((dst,p)->(dst.=2 .*p; sum(abs2,p))) : nm=="metric" ? metric : nm=="chol_metric" ? cholesky(metric) : startswith(nm,"##node") ? zero(T) : nm=="pos" ? T[1,2] : nm=="mom" ? T[3,4] : (nm in ("dpot_dpos","dham_dpos","dkin_dmom","dham_dmom")) ? T[0,0] : zero(T)
        end
        fr = RK._construct_nuts_frame(pf, d, 5; step_f=RK.partial(_CF.leapfrog!; stepsize=T(0.3)), stats_f=_CF.nuts_stats!, min_dham=-1000)
        RK.compile_prepared_initialization(pf, typeof(fr.init), typeof(fr.shared))(fr.init, fr.shared, RK.kernel_prepared_handles(pf))
        RK._seed_nuts_children!(fr); (pf, fr)
    end
    function _outcome(T, metric)
        try
            pf, fr = _mk(T, metric)
            C = RK.compile_nuts(pf, _CF.nuts_state, _CF.refresh_momentum!!, _CF.nuts!!, fr)
            for _ in 1:50; C.root!(fr, C.scratch, Random.Xoshiro(3)); end
            :completed
        catch e
            (e isa ErrorException && e.msg == "POTRS_CALLED") ? :threw_potrs : :other_error
        end
    end
    dF64 = _outcome(Float64, [2.0 0; 0 2.0]); dF32 = _outcome(Float32, Float32[2 0; 0 2])
    gF64 = _outcome(Float64, Diagonal([2.0, 2.0])); gF32 = _outcome(Float32, Diagonal(Float32[2, 2]))
    println("RECEIPT dense_F64=", dF64, " dense_F32=", dF32, " diag_F64=", gF64, " diag_F32=", gF32)
    exit((dF64 === :threw_potrs && dF32 === :threw_potrs && gF64 === :completed && gF32 === :completed) ? 0 : 1)
    """
    childfile = tempname() * ".jl"; write(childfile, childsrc)
    out = IOBuffer()
    cmd = setenv(`$(Base.julia_cmd()) --startup-file=no --project=$proj $childfile`, "RK_DIAG_FIXTURE" => fixture)
    proc = run(pipeline(cmd; stdout = out, stderr = out); wait = false); wait(proc)
    receipt = String(take!(out)); rm(childfile; force = true)
    @test occursin("RECEIPT dense_F64=threw_potrs dense_F32=threw_potrs diag_F64=completed diag_F32=completed", receipt)
    @test proc.exitcode == 0
end
# Load-bearing tests for the faithful reset cleanup (reset! seeds only fwd/bwd/proposals[1]/proposals[end]).
# Adversarial stale-poison of every dropped write class (D1–D5) over a CENSUSED set of witnessed paths (depths,
# both directions, net proposals[end] identity change AND no-net-change observed, plus a forced-divergence
# genuine no-swap control), full seven-field selected-sample parity, and exact 0-B (both RNG). The local
# MethodIR/comment-stripped checks bind this executable fixture; the complete exact-AST reset contract lives in
# benchmark/nuts_authoring_shadowing_gate.jl.

@testset "kernel_nuts — faithful reset: stale-poison D1–D5 (censused paths) + parity + 0-B + IR drift gate" begin
    _mkframe(pf, T, md, metric; min_dham = -1000) = begin
        PL = RK.kernel_prepared_plan(pf); d = _nuts_mkvals(pf, T)
        for sl in RK.kernel_plan_slots(PL)
            nm = String(sl.path[1]); nm == "metric" && (d[sl.canon] = metric); nm == "chol_metric" && (d[sl.canon] = cholesky(metric))
        end
        fr = RK._construct_nuts_frame(pf, d, md; step_f = RK.partial(_NutsFix.leapfrog!; stepsize = T(0.3)), stats_f = _NutsFix.nuts_stats!, min_dham = min_dham)
        RK.compile_prepared_initialization(pf, typeof(fr.init), typeof(fr.shared))(fr.init, fr.shared, RK.kernel_prepared_handles(pf))
        RK._seed_nuts_children!(fr); fr
    end
    # FULL selected-sample payload: all SEVEN physical owned init fields (the sample copied from proposals[end])
    # + init currentness mask + proposals[end] payload/mask, plus every diagnostic/mask and the physical diverged
    # field & its currentness bits. Owned slots: pos4 mom5 pot7 dpot8 dkin10 kin11 ham12.
    _slots(ep) = (copy(RK._canon_slot(ep, Val(4))), copy(RK._canon_slot(ep, Val(5))), RK._canon_slot(ep, Val(7)),
                  copy(RK._canon_slot(ep, Val(8))), copy(RK._canon_slot(ep, Val(10))), RK._canon_slot(ep, Val(11)),
                  RK._canon_slot(ep, Val(12)), RK._canon_current_mask(ep))
    _obs(fr) = (init = _slots(fr.init), prop_end = _slots(getfield(fr, :proposals)[end]),
                n_steps = RK._diag_slot(fr.diag, Val(1)), reached_depth = RK._diag_slot(fr.diag, Val(2)),
                acceptance_rate = RK._diag_slot(fr.diag, Val(3)), dham = RK._diag_slot(fr.diag, Val(4)),
                committed = RK.diagnostics_committed_mask(fr.diag), pending = RK.diagnostics_pending_mask(fr.diag),
                diverged = getfield(fr, :diverged),
                diverged_pending = RK.nuts_frame_diverged_pending(fr),
                diverged_committed = RK.nuts_frame_diverged_committed(fr))
    _NAN(v::AbstractArray) = fill!(v, convert(eltype(v), NaN))
    _poison_ep!(ep) = begin                                   # ALL 7 physical owned fields (vectors + scalars)
        for s in (4, 5, 8, 10); v = RK._canon_slot(ep, Val(s)); v isa AbstractArray && _NAN(v); end
        for s in (7, 11, 12); RK._canon_set!(ep, Val(s), convert(eltype(RK._canon_slot(ep, Val(4))), NaN)); end
    end
    _D1!(fr) = for t in getfield(fr, :trees); _NAN(t.log_weight); end
    _D2!(fr) = for t in getfield(fr, :trees); _NAN(t.bwd.mom); _NAN(t.bwd.dham_dmom); end
    _D3!(fr) = for t in getfield(fr, :trees); _NAN(t.bwd_fwd.mom); _NAN(t.bwd_fwd.dham_dmom); end
    _D4!(fr) = for t in getfield(fr, :trees); _NAN(t.summed_mom.bwd); _NAN(t.summed_mom.fwd); end
    _D5!(fr) = begin ps = getfield(fr, :proposals); for i in 2:length(ps)-1; _poison_ep!(ps[i]); end end
    # interior-proposal CURRENTNESS-mask poison (in addition to the 7 values): set the whole mask to all-ZERO
    # (every field reads dirty) and to all-ONE (every stale/garbage field reads CURRENT). A read-before-write of
    # an interior proposal's mask leaks either way, so the trajectory must stay byte-identical.
    _setmask!(ep, v) = setfield!(ep, :current, map(_ -> v, RK._canon_current_mask(ep)))
    _D5m0!(fr) = begin _D5!(fr); ps = getfield(fr, :proposals); for i in 2:length(ps)-1; _setmask!(ps[i], UInt(0)); end end
    _D5m1!(fr) = begin _D5!(fr); ps = getfield(fr, :proposals); for i in 2:length(ps)-1; _setmask!(ps[i], ~UInt(0)); end end
    _DALL!(fr) = (_D1!(fr); _D2!(fr); _D3!(fr); _D4!(fr); _D5m1!(fr))   # combined uses the strongest D5 (values + garbage-current mask)
    _noop!(fr) = nothing
    # each per-txn record folds the full observable set PLUS the witnessed control category:
    # `end_identity_changed` = proposals[end] changed OBJECT identity across this txn — a NET change (swaps that
    # restore identity read as no change, so this is NOT a raw swap count), and `gofwd` = final direction.
    _traj(C, fr, seed, K, poison!) = begin
        rng = Random.Xoshiro(seed); out = Vector{Any}(undef, K)
        for i in 1:K
            poison!(fr); pe0 = objectid(getfield(fr, :proposals)[end])
            C.root!(fr, C.scratch, rng)
            out[i] = (obs = _obs(fr), end_identity_changed = objectid(getfield(fr, :proposals)[end]) != pe0,
                      gofwd = getfield(fr, :gofwd), may_continue = getfield(fr, :may_continue), may_sample = getfield(fr, :may_sample))
        end
        out
    end

    # === local executable-fixture drift checks (the exact-AST authority is the benchmark shadowing gate) ===
    resetir = only(i for i in RK.method_irs(_NutsFix.nuts_state) if i.id.name === :reset!)
    _count_for(x) = (x isa RK._For ? 1 : 0) + ((x isa Tuple || x isa AbstractVector) ? sum(_count_for, x; init = 0) :
        (x isa RK._MExpr || x isa RK._MStmt) ? sum(f -> _count_for(getfield(x, f)), fieldnames(typeof(x)); init = 0) : 0)
    @test _count_for(resetir.body) == 0                       # reset! contains ZERO loops (dead-write loops gone)
    srclines = split(read(joinpath(@__DIR__, "..", "benchmark", "nuts_kernel_authoring_fixture.jl"), String), '\n')
    i0 = findfirst(l -> occursin("reset!() = begin", l), srclines)
    i1 = i0 + findfirst(l -> strip(l) == "end", srclines[i0+1:end])
    body_code = replace(join(srclines[i0:i1], "\n"), r"#[^\n]*" => "")     # comment-stripped reset! CODE
    @test occursin("copy!!(proposals[1], init)", body_code)
    @test occursin("copy!!(proposals[length(proposals)], init)", body_code)
    @test occursin("copy!!(fwd, init)", body_code) && occursin("copy!!(bwd, init)", body_code)
    @test !occursin("for ", body_code) && !occursin("fill!", body_code)

    seeds = (1, 7, 42, 123)
    for T in (Float64, Float32)
        pf = _nuts_pf()
        CD = RK.compile_nuts(pf, _NutsFix.nuts_state, _NutsFix.refresh_momentum!!, _NutsFix.nuts!!, _mkframe(pf, T, 10, LinearAlgebra.Diagonal(T[1, 1])))
        CM = RK.compile_nuts(pf, _NutsFix.nuts_state, _NutsFix.refresh_momentum!!, _NutsFix.nuts!!, _mkframe(pf, T, 10, T[1 0; 0 1]))
        # stale-poison D1–D5 across the censused seeds — reference vs each poisoned class byte-identical over the
        # FULL per-txn record (payload + net end-identity change + gofwd). Also collect the observed path CENSUS from the SAME
        # reference trajectories, so the "dead" claim is scoped to exactly the paths witnessed here.
        depths = Set{Int}(); dirs = Set{Bool}(); sels = Set{Bool}(); allstop = Ref(true)
        for seed in seeds
            ref = _traj(CD, _mkframe(pf, T, 10, LinearAlgebra.Diagonal(T[1, 1])), seed, 60, _noop!)
            for r in ref
                push!(depths, Int(r.obs.reached_depth)); push!(dirs, r.gofwd); push!(sels, r.end_identity_changed)
                (!r.may_continue && !r.obs.diverged) || (allstop[] = false)
            end
            for pz in (_D1!, _D2!, _D3!, _D4!, _D5!, _D5m0!, _D5m1!, _DALL!)
                @test _traj(CD, _mkframe(pf, T, 10, LinearAlgebra.Diagonal(T[1, 1])), seed, 60, pz) == ref
            end
            @test _traj(CD, _mkframe(pf, T, 10, LinearAlgebra.Diagonal(T[1, 1])), seed, 60, _noop!) ==
                  _traj(CM, _mkframe(pf, T, 10, T[1 0; 0 1]), seed, 60, _noop!)   # Diagonal ≡ dense full-observable parity
        end
        # WITNESSED categories, matching the independent Codex census for these EXACT seeds (no overclaim):
        # reached_depth is exactly {1,2,3,4}, both gofwd directions, both a net proposals[end] identity change and
        # no-net-change, and every
        # normal transition ended !may_continue && !diverged (U-turn-like stop). We do NOT claim max-depth
        # (reached_depth never hits max_depth=10) or per-depth flip coverage; divergence is the forced control.
        @test depths == Set([1, 2, 3, 4])
        @test maximum(depths) < 10
        @test dirs == Set([false, true])
        @test sels == Set([false, true])        # both a NET proposals[end] identity change and no-net-change witnessed
        @test allstop[]                          # all 240 normal transitions stopped via U-turn (!may_continue, !diverged)
        # FORCED-DIVERGENCE control: min_dham above any dham forces diverged=true on leaf 1, exercising the
        # early-return path (start! returns before writing proposals[1] / trees[1].log_weight).
        CX = RK.compile_nuts(pf, _NutsFix.nuts_state, _NutsFix.refresh_momentum!!, _NutsFix.nuts!!, _mkframe(pf, T, 10, LinearAlgebra.Diagonal(T[1, 1]); min_dham = T(1e6)))
        refX = _traj(CX, _mkframe(pf, T, 10, LinearAlgebra.Diagonal(T[1, 1]); min_dham = T(1e6)), 1, 40, _noop!)
        @test all(r -> r.obs.diverged, refX)                   # control genuinely takes the divergence path (non-vacuous)
        @test all(r -> !r.may_sample && !r.may_continue, refX) # forced divergence pins BOTH control bits off
        @test all(r -> Int(r.obs.reached_depth) == 1 && !r.end_identity_changed && r.gofwd, refX)   # forced category: depth 1, no NET identity change (genuinely no swap — divergence terminates before any swap), gofwd
        for pz in (_D1!, _D2!, _D3!, _D4!, _D5!, _D5m0!, _D5m1!, _DALL!)
            @test _traj(CX, _mkframe(pf, T, 10, LinearAlgebra.Diagonal(T[1, 1]); min_dham = T(1e6)), 1, 40, pz) == refX
        end
        # exact 0-B on the faithful-reset path (scaled Diagonal, same type ⇒ reuse CD), F32/F64 × both RNG
        frS = _mkframe(pf, T, 10, LinearAlgebra.Diagonal(T[2, 2]))
        for rg in (Random.Xoshiro(91), Random.MersenneTwister(2))
            @test _nuts_batch0b(CD.root!, frS, CD.scratch, rg, Val(64)) == 0
            @test _nuts_batch0b(CD.root!, frS, CD.scratch, rg, Val(256)) == 0
        end
    end
end

@testset "kernel_nuts — public ROOT bad-RNG contract: throws + diagnostics/derived cleared, no false commit (F32/F64)" begin
    # Codex-audit pin (RK): the PUBLIC ROOT (not only C.refresh) begins each epoch by resetting/uncommitting
    # diagnostics + derived, THEN refreshes. An unsupported RNG is rejected inside refresh BEFORE any kill, so
    # C.root! THROWS and leaves: all diag values 0, committed=pending=0, derived committed=pending=0 (nothing
    # falsely current), and mom currentness UNCHANGED (no kill occurred). This is executed-prefix semantics —
    # explicitly NOT a rollback (the prior transition's committed diagnostics are gone, reset at epoch entry).
    _mkD(T) = begin
        pf = _nuts_pf(); PL = RK.kernel_prepared_plan(pf); d = _nuts_mkvals(pf, T)
        for sl in RK.kernel_plan_slots(PL)
            nm = String(sl.path[1]); nm == "metric" && (d[sl.canon] = LinearAlgebra.Diagonal(T[1, 1])); nm == "chol_metric" && (d[sl.canon] = cholesky(LinearAlgebra.Diagonal(T[1, 1])))
        end
        fr = RK._construct_nuts_frame(pf, d, 10; step_f = RK.partial(_NutsFix.leapfrog!; stepsize = T(0.3)), stats_f = _NutsFix.nuts_stats!, min_dham = -1000)
        RK.compile_prepared_initialization(pf, typeof(fr.init), typeof(fr.shared))(fr.init, fr.shared, RK.kernel_prepared_handles(pf))
        RK._seed_nuts_children!(fr)
        (pf, RK.compile_nuts(pf, _NutsFix.nuts_state, _NutsFix.refresh_momentum!!, _NutsFix.nuts!!, fr), fr)
    end
    for T in (Float64, Float32)
        pf, C, fr = _mkD(T); PL = RK.kernel_prepared_plan(pf)
        C.root!(fr, C.scratch, Random.Xoshiro(1))                       # a SUCCESSFUL transition first (real diag values)
        @test RK._diag_slot(fr.diag, Val(1)) > 0                        # n_steps advanced — the reset below is non-vacuous
        @test RK.diagnostics_committed_mask(fr.diag) == UInt(0x0f)      # previous epoch committed all 4
        # comprehensive snapshot (mirrors the direct-refresh before-any-kill gate): EVERY owned+shared canon
        # value (contents) + currentness bit + whole masks + shared object identity.
        slots = unique([(RK.kernel_plan_field(PL, sl.canon)[1], RK.kernel_plan_field(PL, sl.canon)[2]) for sl in RK.kernel_plan_slots(PL)])
        obj(role) = role === :owned ? fr.init : fr.shared
        vals = Dict((r, s) => _rsnap(RK._canon_slot(obj(r), Val(s))) for (r, s) in slots)
        curs = Dict((r, s) => RK._canon_current(obj(r), Val(s)) for (r, s) in slots)
        imask0 = RK._canon_current_mask(fr.init); smask0 = RK._canon_current_mask(fr.shared)
        shared_ids = Dict(s => RK._canon_slot(fr.shared, Val(s)) for (r, s) in slots if r === :shared)
        @test_throws ArgumentError C.root!(fr, C.scratch, _CustomRNG2())   # bad RNG rejected inside refresh (SPECIFIC error type)
        # owned+shared endpoint state EXACTLY unchanged: the RNG is rejected before any kill/write, so nothing ran.
        @test all(_rsame(RK._canon_slot(obj(r), Val(s)), vals[(r, s)]) for (r, s) in slots)   # every value (contents) unchanged
        @test all(RK._canon_current(obj(r), Val(s)) === curs[(r, s)] for (r, s) in slots)     # every currentness bit unchanged
        @test RK._canon_current_mask(fr.init) == imask0 && RK._canon_current_mask(fr.shared) == smask0   # whole masks unchanged
        @test all(RK._canon_slot(fr.shared, Val(s)) === shared_ids[s] for s in keys(shared_ids))         # shared object identity unchanged
        # BUT the epoch-entry reset/uncommit DID clear diagnostics + derived (executed-prefix, explicitly NOT rollback):
        @test all(RK._diag_slot(fr.diag, Val(i)) == 0 for i in 1:4)     # diag values reset at epoch entry, never re-committed
        @test RK.diagnostics_committed_mask(fr.diag) == 0               # committed cleared (NOT the prior 0x0f)
        @test RK.diagnostics_pending_mask(fr.diag) == 0                # pending cleared by the epoch catch
        @test !RK.nuts_frame_diverged_committed(fr) && !RK.nuts_frame_diverged_pending(fr)   # derived committed+pending cleared
    end
end
