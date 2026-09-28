# Closed-form twocmt + Gamma-transit unit response (`src/transit_twocmt.jl`).
#
# The value oracle is INDEPENDENT of the implementation: adaptive-Simpson
# quadrature of the convolution integral `∫_0^t h11(t-u)*f(u) du` with the
# disposition `h11` from a dense matrix exponential (`LinearAlgebra.exp`,
# no eigen-combination) and the Gamma PDF from Distributions.jl (not the
# transit series). Gradients cross-check native Enzyme against central
# finite differences, the `test_expm_rule.jl` pattern.
#
# Tolerance logic (oracle-limited, not implementation-limited): adaptive
# Simpson over up to 672 h of a peak-O(0.1) integrand holds ~1e-14
# absolute, so the comparison is `max(3e-9*|want|, 3e-14)` — tight where
# the oracle is strong, honest where values sink below its floor.
# Per-case grids keep every oracle point meaningful (deep-tail points
# below the floor get smoke bounds instead), and handoff-straddling pairs
# pin each regime boundary from both sides through the branch-free oracle.
using DifferentiationInterface: AutoEnzyme, gradient
using Distributions: Gamma, pdf
import Enzyme
using LinearAlgebra: exp
using ReactiveKernelsPPL
using Test

# Adaptive Simpson quadrature (self-contained oracle integrator).
function _simpson(f, a, b)
    m = 0.5 * (a + b)
    return (b - a) / 6.0 * (f(a) + 4.0 * f(m) + f(b))
end

function _adaptive_simpson(f, a, b; atol = 1e-14, rtol = 1e-12, maxdepth = 24)
    whole = _simpson(f, a, b)
    function rec(a, fa, m, fm, b, fb, whole, depth)
        lm = 0.5 * (a + m)
        rm = 0.5 * (m + b)
        flm = f(lm)
        frm = f(rm)
        left = (m - a) / 6.0 * (fa + 4.0 * flm + fm)
        right = (b - m) / 6.0 * (fm + 4.0 * frm + fb)
        if depth <= 0 || abs(left + right - whole) <= max(atol, rtol * abs(left + right))
            return left + right
        end
        return rec(a, fa, lm, flm, m, fm, left, depth - 1) +
               rec(m, fm, rm, frm, b, fb, right, depth - 1)
    end
    m = 0.5 * (a + b)
    return rec(a, f(a), m, f(m), b, f(b), whole, maxdepth)
end

# Independent oracle: dense-exp disposition convolved with the Gamma PDF,
# paneled so a narrow input peak on a long tail cannot be missed: a single
# adaptive Simpson over [0, 672] accepts a wrong ~0 when its coarse samples
# straddle the mass (measured), and converges ~1e-11 off on 300 h tails.
function _oracle_twocmt_unit(t, k10, k12, k21, rate, shape)
    t == 0.0 && return 0.0
    A = [-k10 - k12 k21; k12 -k21]
    d = Gamma(shape, 1.0 / rate)
    h11(τ) = (exp(A * τ))[1, 1]
    f(u) = h11(t - u) * pdf(d, u)
    width = min(8 * shape / rate, 8 * sqrt(shape) / rate, t)
    acc = 0.0
    a = 0.0
    while a < t
        b = min(a + width, t)
        acc += _adaptive_simpson(f, a, b)
        a = b
    end
    return acc
end

_transit_fd_gradient(f, x; h = 1e-6) = begin
    g = similar(x, Float64)
    for i in eachindex(x)
        xp = copy(x)
        xm = copy(x)
        xp[i] += h
        xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2h)
    end
    g
end

# (params..., t-grid). Handoff pairs: P1 {5.3, 5.5} straddles w = -1 and
# {323, 325} straddles w = -60 on the slow mode, {614, 616} straddles
# w = +40 on the fast mode; P2 {265, 268} and P8 {265, 268} straddle
# w = +40; P4/P5 {60, 63} straddle w = +40; P7 t = 200 lands the P-series
# at x = 57 near its (60, 8) corner.
const _TRANSIT_CASES = (
    (k10 = 0.08, k12 = 0.15, k21 = 0.05, rate = 0.20, shape = 1.20,
        ts = (0.0, 0.5, 3.0, 5.3, 5.5, 24.0, 100.0, 323.0, 325.0, 336.0, 614.0, 616.0, 672.0)),
    (k10 = 0.50, k12 = 0.30, k21 = 0.40, rate = 0.05, shape = 1.05,
        ts = (0.0, 0.5, 3.0, 24.0, 100.0, 200.0, 265.0, 268.0, 336.0)),
    (k10 = 1.00, k12 = 1e-9, k21 = 1.0 + 1e-9, rate = 0.50, shape = 1.50,
        ts = (0.0, 0.5, 3.0, 10.0, 24.0)),
    (k10 = 2.00, k12 = 1e-6, k21 = 1.00, rate = 0.35, shape = 1.37,
        ts = (0.0, 1.0, 5.0, 24.0, 60.0, 63.0)),
    (k10 = 1.00, k12 = 1e-6, k21 = 2.00, rate = 0.35, shape = 1.37,
        ts = (0.0, 1.0, 5.0, 24.0, 60.0, 63.0)),
    (k10 = 0.08, k12 = 0.15, k21 = 0.05, rate = 0.20, shape = 1.00,
        ts = (0.0, 0.5, 3.0, 24.0, 100.0, 336.0, 672.0)),
    (k10 = 0.08, k12 = 0.15, k21 = 0.05, rate = 0.30, shape = 8.00,
        ts = (0.0, 3.0, 24.0, 100.0, 200.0, 336.0, 672.0)),
    (k10 = 0.50, k12 = 0.30, k21 = 0.40, rate = 0.05, shape = 8.00,
        ts = (0.0, 24.0, 100.0, 265.0, 268.0, 336.0)),
    (k10 = 1.00, k12 = 1e-6, k21 = 1.0 + 1e-6 - 1e-9, rate = 0.50, shape = 1.50,
        ts = (0.0, 3.0, 24.0)),
    (k10 = 1.00, k12 = 1e-6, k21 = 1.0 + 1e-6 + 1e-9, rate = 0.50, shape = 1.50,
        ts = (0.0, 3.0, 24.0)),
)

@testset "transit twocmt closed form vs quadrature oracle" begin
    for c in _TRANSIT_CASES
        for t in c.ts
            got = transit_twocmt_unit(t, c.k10, c.k12, c.k21, c.rate, c.shape)
            want = _oracle_twocmt_unit(t, c.k10, c.k12, c.k21, c.rate, c.shape)
            @test isfinite(got)
            @test got >= 0.0
            @test abs(got - want) <= max(3e-9 * abs(want), 3e-14)
        end
    end
    # t = 0 is exactly zero (empty integral), not merely small.
    for c in _TRANSIT_CASES
        @test transit_twocmt_unit(0.0, c.k10, c.k12, c.k21, c.rate, c.shape) == 0.0
    end
    # Deep Watson tail below the oracle's floor: boundedness smoke test
    # (true value ~4e-16; a broken Watson overflows or returns garbage).
    deep = transit_twocmt_unit(672.0, 0.50, 0.30, 0.40, 0.05, 1.05)
    @test isfinite(deep) && 0.0 <= deep < 1e-13
end

@testset "transit twocmt Enzyme gradient vs finite differences" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    # Interior points spanning all three regimes across the two modes,
    # probed in log coordinates (the production shape: LPs are logs, and
    # a linear findiff step would leave the positive domain at k12 = 1e-9).
    points = (
        (t = 24.0, p = [0.08, 0.15, 0.05, 0.20, 1.20]),
        (t = 100.0, p = [0.50, 0.30, 0.40, 0.05, 1.05]),
        (t = 3.0, p = [1.00, 1e-9, 1.0 + 1e-9, 0.50, 1.50]),
    )
    for (t, p) in points
        f = lq -> transit_twocmt_unit(t, exp(lq[1]), exp(lq[2]), exp(lq[3]),
            exp(lq[4]), exp(lq[5]))
        lp = log.(p)
        g = gradient(f, backend, lp)
        gfd = _transit_fd_gradient(f, lp)
        @test maximum(abs.(g .- gfd)) < 1e-6
    end
end

@testset "transit twocmt vector response matches scalar" begin
    ts = [0.0, 1.0, 5.0, 24.0, 200.0, 672.0]
    got = transit_twocmt_unit_response(ts, 0.08, 0.15, 0.05, 0.20, 1.20)
    want = [transit_twocmt_unit(t, 0.08, 0.15, 0.05, 0.20, 1.20) for t in ts]
    @test got == want
    @test all(>=(0.0), got)
end
