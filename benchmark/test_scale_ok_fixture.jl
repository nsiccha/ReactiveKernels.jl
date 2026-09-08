# Negative fixture for the PRODUCTION scale-aware native parity floor. Includes the SAME shared
# helper the measurement body uses (all80_parity.jl) — NO duplicate definition (performance review).
# Builds ACTUAL per-point Float64 densities and computes the residual SPREAD from them (not an
# analytic `stab = slope*0.5` assignment), then asserts a genuine q-dependent offset ABOVE the
# policy floor FAILS even when a huge common logdensity constant inflates the magnitude.
#
# WORDING (performance review 2026-09-08): the C·|value|·eps tolerance is a chosen POLICY floor.
# Below it lie TWO distinct regimes — (a) < ~1 ULP of the values: INTRINSICALLY UNRESOLVABLE from
# Float64; (b) 1 ULP … C·ULP: REPRESENTABLE but ACCEPTED (the accepted-policy SENSITIVITY LIMIT,
# a deliberate C-wide headroom). A 5e-3 residual at mag 1e13 is ~2.6 ULP — representable, merely
# below the tolerance we chose; NOT a Float64 inevitability.

include(joinpath(@__DIR__, "all80_parity.jl"))
using Test

# Per-point Stan/RK densities differing by a CONSTANT offset plus a q-dependent slope (the injected
# defect). The reference density has slope `ref_slope` in q, so its analytic q-derivative IS
# `ref_slope` and RK's is `ref_slope - slope`. Returns the ACTUAL Float64 residual spread (stab),
# the magnitude, and BOTH analytic derivatives — so the gradient control is DERIVED from the same
# construction (no fictional denominator; performance review 2026-09-08).
function stab_mag(qs; base, offset, slope, ref_slope = 3.0)
    stan = [base + ref_slope * q for q in qs]                       # reference density (dq = ref_slope)
    rk   = [base + ref_slope * q - offset - slope * q for q in qs]  # RK: constant offset + MISSING slope*q
    resid = stan .- rk                                             # = offset + slope*q, in Float64
    off = sum(resid) / length(resid)
    (maximum(abs, resid .- off), max(maximum(abs, stan), maximum(abs, rk)), ref_slope, ref_slope - slope)
end

@testset "production scale_ok: faithful passes, genuine offset FAILS even at huge magnitude" begin
    qs = [-0.5, -0.25, 0.0, 0.25, 0.5]

    # (1) FAITHFUL: constant offset only (slope 0) ⇒ spread is pure Float64 roundoff at earn_height
    #     scale ⇒ PASSES (spread ≪ tol). Real arithmetic, not an assigned stab.
    s1, m1 = stab_mag(qs; base = 6.66e11, offset = 2.0, slope = 0.0)
    @test s1 < parity_tol(m1)
    @test scale_ok(s1, m1)

    # (2) GENUINE q-dependent offset ABOVE the floor at moderate-huge magnitude ⇒ FAILS.
    s2, m2 = stab_mag(qs; base = 1e11, offset = 2.0, slope = 1e-2)   # spread ≈ 5e-3
    @test s2 > parity_tol(m2)                                         # ~5e-3 > ~1.42e-3
    @test !scale_ok(s2, m2)

    # (3) A HUGE common logdensity constant (mag ~1e13) does NOT launder an offset that stays ABOVE
    #     the inflated floor — the case performance required. Real per-point Float64 arithmetic.
    s3, m3 = stab_mag(qs; base = 1e13, offset = 5.0, slope = 1.0)     # spread ≈ 0.5, mag ~1e13
    @test m3 > 9e12
    @test s3 > parity_tol(m3)                                         # ~0.5 > ~0.142
    @test !scale_ok(s3, m3)

    # (4) ACCEPTED-POLICY SENSITIVITY LIMIT (NOT "intrinsically unresolvable"): a 5e-3 spread at mag
    #     1e13 is ~2.6 ULP — REPRESENTABLE, but below the C=64 tolerance we CHOSE, so it PASSES.
    #     Deliberate policy headroom, not a Float64 inevitability; only a sub-1-ULP difference is
    #     intrinsically unresolvable.
    @test 5e-3 > eps(1e13)                                           # ~2.6 ULP — representable
    @test scale_ok(5e-3, 1e13)                                       # accepted by policy, not "unresolvable"

    # (5) NON-FINITE delta or magnitude must FAIL — an Inf value never inflates the tolerance
    #     (wells_dist's saturating -Inf ⇒ NaN stab / Inf mag stays a real failure).
    @test !scale_ok(1.0, Inf)
    @test !scale_ok(NaN, 1e3)
    @test !scale_ok(Inf, 1e3)

    # (6) INDEPENDENT gradient parity catches a q-dependent discrepancy BELOW primal resolution.
    #     COHERENT pair: reference derivative = ref_slope, RK derivative = ref_slope-slope, BOTH from
    #     the SAME construction (no fictional denominator). Pick ref_slope small enough that the
    #     relerr (slope/ref_slope) > 2e-3 while stab (~slope*0.5) stays below the floor.
    s6, m6, ref_d, rk_d = stab_mag(qs; base = 1e11, offset = 0.0, slope = 2e-3, ref_slope = 0.5)
    @test scale_ok(s6, m6)                                          # primal (stab ~1e-3 < tol) cannot see it
    grad_relerr = abs(rk_d - ref_d) / abs(ref_d)                    # = 2e-3/0.5 = 4e-3, from the construction
    @test grad_relerr > 2e-3                                        # independent gradient check DOES catch it
end
println("SCALE_OK_FIXTURE_OK")
