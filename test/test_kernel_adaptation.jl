# Gates for lowering an AUTHORED free stateful @kernel (dual_averaging_state / welford_var) to a runnable
# object — the AUTHORED recurrence, NOT the package @reactive type (HMC acceptance G3/G4). This first slice
# covers the SOUND base (RK review of 0745ab0): AUTHORITATIVE ownership (never a local-seed setdiff), the
# ACCEPTED cold bootstrap for EXECUTE-ONCE domain-checked construction (no second initializer executor), and
# the AUTHORITATIVE signature binder (unknown/extra kwargs rejected). Method execution (fit!/step!) is next.
using Test, LinearAlgebra
using ReactiveKernels
const RK = ReactiveKernels

module _AdaptFix
    include(joinpath(@__DIR__, "..", "benchmark", "nuts_kernel_authoring_fixture.jl"))
end

_sval(pf, ow, sh, nm) = begin
    PL = RK.kernel_prepared_plan(pf); s = RK.kernel_plan_slot(PL, nm)
    role, slot = RK.kernel_plan_field(PL, s.canon)
    RK._canon_slot(role === :owned ? ow : sh, Val(slot))
end
_scur(pf, ow, sh, nm) = begin
    PL = RK.kernel_prepared_plan(pf); s = RK.kernel_plan_slot(PL, nm)
    role, slot = RK.kernel_plan_field(PL, s.canon)
    RK._canon_current(role === :owned ? ow : sh, Val(slot))
end
_roles(pf) = begin
    PL = RK.kernel_prepared_plan(pf)
    Dict(String(s.path[1]) => RK.kernel_plan_field(PL, s.canon)[1] for s in RK.kernel_plan_slots(PL))
end

@testset "kernel_adaptation — authoritative ownership (not a local-seed setdiff)" begin
    # DA: mutated ∪ derived-from-mutated owned; the constant `mu` (from init only) + params stay shared.
    rDA = _roles(RK._prepare_stateful(_AdaptFix.dual_averaging_state))
    @test all(rDA[n] === :owned for n in ("m", "H", "log_current", "log_final", "current", "final"))
    @test all(rDA[n] === :shared for n in ("init", "mu", "target", "regularization_scale", "relaxation_exponent", "offset"))
    # welford: n/mean/var owned (written by step!, incl. the matrix overload's __self__ sibling write which a
    # local seed would miss); template shared. The authoritative closure resolves interprocedural sibling writes.
    rWV = _roles(RK._prepare_stateful(_AdaptFix.welford_var))
    @test rWV["n"] === :owned && rWV["mean"] === :owned && rWV["var"] === :owned && rWV["template"] === :shared
end

@testset "kernel_adaptation — dual_averaging_state: single-pass construct matches the authored initializers" begin
    da = _AdaptFix.dual_averaging_state
    pf = RK._prepare_stateful(da)
    for (T, tgt) in ((Float64, 0.8), (Float32, 0.8f0))
        ow, sh = RK._construct_stateful(da, pf, T(tgt))
        mu_ref = log(T(10)) + log(T(tgt))
        @test _sval(pf, ow, sh, :m) === one(T)                      # m = one(init), typed
        @test _sval(pf, ow, sh, :H) === zero(T)                     # H = zero(init)
        @test _sval(pf, ow, sh, :mu) ≈ mu_ref                       # mu = log(10) + log(init)
        @test _sval(pf, ow, sh, :log_current) ≈ mu_ref              # log_current = mu - sqrt(m)/reg*H  (H=0)
        @test _sval(pf, ow, sh, :log_final) === zero(T)
        @test _sval(pf, ow, sh, :current) ≈ exp(mu_ref)             # current = exp(log_current)  (exp admitted)
        @test _sval(pf, ow, sh, :final) === one(T)                  # final = exp(log_final) = exp(0) = 1
        @test typeof(_sval(pf, ow, sh, :current)) === T             # concrete init-derived type
        @test all(_scur(pf, ow, sh, n) for n in (:m, :H, :mu, :log_current, :log_final, :current, :final))  # all current
    end
    # keyword override flows through the authoritative binder; unknown / extra reject (never silently ignored)
    ow2, sh2 = RK._construct_stateful(da, pf, 0.8; target = 0.9, offset = 20.0)
    @test _sval(pf, ow2, sh2, :target) == 0.9 && _sval(pf, ow2, sh2, :offset) == 20.0
    @test_throws ArgumentError RK._construct_stateful(da, pf, 0.8; bogus = 1)   # unknown keyword
    @test_throws MethodError RK._construct_stateful(da, pf, 0.8, 0.9)           # extra positional
    # @inferred + repeated same-signature CONCRETE-TYPE IDENTITY (deterministic layout, no boxing)
    @inferred RK._construct_stateful(da, pf, 0.8)
    a = RK._construct_stateful(da, pf, 0.8); b = RK._construct_stateful(da, pf, 0.8)
    @test typeof(a) === typeof(b) && isconcretetype(typeof(a[1])) && isconcretetype(typeof(a[2]))
end

@testset "kernel_adaptation — welford_var: single-pass construct + EXECUTE-ONCE (buffer-identity witness)" begin
    wv = _AdaptFix.welford_var
    pf = RK._prepare_stateful(wv)
    for T in (Float64, Float32)
        ow, sh = RK._construct_stateful(wv, pf, T[0, 0, 0])
        @test _sval(pf, ow, sh, :n) === zero(T)
        @test _sval(pf, ow, sh, :mean) == T[0, 0, 0] && _sval(pf, ow, sh, :var) == T[0, 0, 0]
        @test eltype(_sval(pf, ow, sh, :mean)) === T
        @test _sval(pf, ow, sh, :mean) !== _sval(pf, ow, sh, :var)             # distinct fresh buffers
        @test all(_scur(pf, ow, sh, n) for n in (:n, :mean, :var, :template))
    end
    @inferred RK._construct_stateful(wv, pf, [0.0, 0.0, 0.0])
    a = RK._construct_stateful(wv, pf, [0.0, 0.0, 0.0]); b = RK._construct_stateful(wv, pf, [0.0, 0.0, 0.0])
    @test typeof(a) === typeof(b)
    @test _sval(pf, a[1], a[2], :mean) !== _sval(pf, b[1], b[2], :mean)        # per-instance isolation
    # EXECUTE-ONCE proof: the construct path is the ACCEPTED cold bootstrap (whose per-recipe execute-once is
    # counter-proven NUTS-side by cg.n==1) — NO second initializer executor. Witness it directly: a produced
    # field buffer is the SINGLE bootstrap's output BY IDENTITY (a second executor would replace it with a
    # different buffer). A per-recipe counter on these PURE initializers is precluded by the (correct) domain
    # gate (custom/impure scalar types are rejected), so identity-retention is the equivalent execute-once proof.
    plan = RK.kernel_prepared_plan(pf); H = RK.kernel_prepared_handles(pf)
    src = RK._stateful_sources(wv, pf, ([1.0, 2.0, 3.0],), NamedTuple())
    cvals = RK._bootstrap_canon_values(plan, H, src)                            # ONE bootstrap execution
    ow, sh = RK._construct_endpoint_from_values(plan, H, cvals)
    canons, _ = RK._plan_superset_from_key(RK.kernel_plan_key(plan))
    mslot = RK.kernel_plan_field(plan, RK.kernel_plan_slot(plan, :mean).canon)[2]
    vslot = RK.kernel_plan_field(plan, RK.kernel_plan_slot(plan, :var).canon)[2]
    mi = findfirst(==(RK.kernel_plan_slot(plan, :mean).canon), canons)
    vi = findfirst(==(RK.kernel_plan_slot(plan, :var).canon), canons)
    @test RK._canon_slot(ow, Val(mslot)) === cvals[mi]                          # produced buffer IS the bootstrap output
    @test RK._canon_slot(ow, Val(vslot)) === cvals[vi]
end
