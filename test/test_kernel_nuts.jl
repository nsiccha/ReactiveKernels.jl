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
