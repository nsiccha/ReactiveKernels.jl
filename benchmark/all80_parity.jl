# Scale-aware native parity floor — SHARED by the measurement body (all80_posteriordb_body.jl)
# and its pure fixture (test_scale_ok_fixture.jl), so the fixture exercises the PRODUCTION helper
# rather than a duplicate definition (performance review 2026-09-08). Plain include (top-level
# defs), so `scale_ok`/`parity_tol`/the constants land directly in the includer's scope.
#
# The offset-residual SPREAD (stab) and the declared-offset error of a FAITHFUL model are bounded
# by the ROUNDOFF RESOLUTION of the large-magnitude log-density values being differenced,
# ~ C·|value|·eps — NOT a flat 1e-6 (unsatisfiable at |logdensity|~1e11, e.g. earn_height) and
# NOT a flat relative fraction of |logdensity| (a 1e-6 rtol at 5e11 would admit ~5e5 absolute
# error — it would MASK a genuine nonconstant offset). `mag` is the magnitude of the ACTUAL values
# differenced (a DISTINCT mag for the RK side vs the Turing side — never one RK-derived mag for
# both). NON-FINITE delta OR magnitude FAILS (an Inf value must never inflate the tolerance).
#
# C=64 is a DOCUMENTED NUMERICAL POLICY (float-noise accumulation headroom), NOT a proven error
# bound. Two DISTINCT regimes sit below the tolerance C·|value|·eps and must not be conflated:
#   (a) < ~1 ULP of the values (< |value|·eps): INTRINSICALLY UNRESOLVABLE from Float64 log-densities.
#   (b) between ~1 ULP and C·|value|·eps: REPRESENTABLE and detectable in principle, but ACCEPTED by
#       this policy — the accepted-policy SENSITIVITY LIMIT, a deliberate C-wide headroom for
#       legitimate roundoff accumulation. NOT a Float64 inevitability. (e.g. at |value|~1e13, 1 ULP
#       ≈ 1.95e-3, so a 5e-3 residual is ~2.6 ULP — representable, merely below the C=64 tolerance.)
# Evidence 2026-09-08: faithful earn_height sits at 4.87·(|logd|·eps) — inside the C=64 headroom;
# genuine offsets sit ≥1.06e10 (wells) or exactly 0 (dogs_hier), ~8 orders above the headroom. The
# independent multi-point GRADIENT parity (a separate < 2e-3 relerr gate) is retained and catches
# q-dependent discrepancies below primal resolution.
const STAB_ATOL = 1e-6
const STAB_ULP_C = 64
parity_tol(mag; atol = STAB_ATOL) = atol + STAB_ULP_C * max(mag, 1.0) * eps(Float64)
scale_ok(delta, mag; atol = STAB_ATOL) = isfinite(delta) && isfinite(mag) &&
    delta < parity_tol(mag; atol = atol)

# Per-side support-boundary classification (Fix E / option 1, performance contract 2026-09-08). At an
# OUT-OF-SUPPORT probe, each side must reject with -Inf. Stan is the REFERENCE and MUST reject — a
# finite Stan is a wrong boundary/registry bug and ALWAYS throws. A per-side RK or Turing failure is
# owned by THAT side: under `discover` it returns the side's diagnostic (row kept, correctness=false);
# under strict it throws (fail closed). A FINITE Turing where RK+Stan are -Inf is NON-EQUIVALENT
# support — NOT a valid reference success (no RK/Turing ratio, no matched-workload/HMC claim). Returns
# (rk_boundary_diag, turing_support_ok, turing_support_diag) with `nothing` where that side is correct.
# (GLMM: upstream Turing declares beta2~Uniform(-10,20) — the Stan TRANSFORM bound — instead of the
#  uniform(-10,10) PRIOR, so it is finite at beta2≈19.8; offset log(30/20)=0.405465.)
function classify_boundary(vb_r, vb_s, vb_t; discover::Bool)
    vb_s == -Inf || error("boundary probe FAIL: reference Stan not -Inf out of support (stan=$vb_s) — wrong boundary/registry")
    rk_boundary_diag = vb_r == -Inf ? nothing :
        "primal_rk: RK did NOT reject the out-of-support point (vb_r=$vb_r, vb_s=$vb_s, vb_t=$vb_t; reference Stan=-Inf required) — genuine RK support defect"
    (rk_boundary_diag === nothing || discover) || error("RK boundary FAIL: $rk_boundary_diag")
    turing_support_ok = vb_t == -Inf
    turing_support_diag = turing_support_ok ? nothing :
        "turing_support: upstream Turing did NOT reject the out-of-support point " *
        "(vb_r=$vb_r, vb_s=$vb_s, vb_t=$vb_t; reference Stan=-Inf required) — NON-EQUIVALENT support; " *
        "RK/Turing comparison INVALID for this model (no ratio, no HMC-success)"
    (turing_support_ok || discover) || error("Turing boundary FAIL: $turing_support_diag")
    (rk_boundary_diag, turing_support_ok, turing_support_diag)
end
