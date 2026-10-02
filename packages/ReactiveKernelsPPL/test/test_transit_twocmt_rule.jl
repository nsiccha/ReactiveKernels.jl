# Run alongside test_transit_twocmt.jl; its quadrature oracle and case corpus
# are independent of the mathematical partials authored by this rule.
using ReactiveKernels: stage_primal, stage_reverse, stage_residuals
using DifferentiationInterface: Constant, AutoEnzyme, gradient
using LinearAlgebra: dot
import Enzyme
using ReactiveKernelsPPL
using Test

_transit_plain_weighted(lp, ts, weights, rtol, terms) = dot(weights,
    transit_twocmt_unit_response(ts, exp(lp[1]), exp(lp[2]), exp(lp[3]),
        exp(lp[4]), exp(lp[5]); series_rtol = rtol, watson_terms = terms))
_transit_rule_weighted(lp, ts, weights) =
    dot(weights, transit_twocmt_rule(ts, exp.(lp)))
const _TRANSIT_LOOSE_RULE = prepare_transit_twocmt_rule(; series_rtol = 1e-6,
    watson_terms = 4)
_transit_loose_weighted(lp, ts, weights) =
    dot(weights, _TRANSIT_LOOSE_RULE(ts, exp.(lp)))

function _transit_rule_primal_bytes(ts, p)
    transit_twocmt_rule(ts, p)
    @allocated transit_twocmt_rule(ts, p)
end
function _transit_direct_primal_bytes(ts, p)
    transit_twocmt_unit_response(ts, p[1], p[2], p[3], p[4], p[5])
    @allocated transit_twocmt_unit_response(ts, p[1], p[2], p[3], p[4], p[5])
end

@testset "transit primal selects values without derivative arrays" begin
    p = [0.08, 0.15, 0.05, 0.2, 1.2]
    for n in (7, 1024)
        ts = collect(range(0., 168.; length = n))
        values = transit_twocmt_rule(ts, p)
        joint, _ = stage_primal(transit_twocmt_rule, Val(2), ts, p)
        @test values == joint
        # A primal should allocate only its returned response, not the
        # n-by-5 parameter Jacobian and n-vector of lag partials.
        bytes = _transit_rule_primal_bytes(ts, p)
        direct_bytes = _transit_direct_primal_bytes(ts, p)
        @test bytes <= direct_bytes + 256
    end
end

@testset "transit generated reverse rule" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    for c in _TRANSIT_CASES
        p = [c.k10, c.k12, c.k21, c.rate, c.shape]
        ts = collect(c.ts)
        weights = [cos(i) for i in eachindex(ts)]
        lp = log.(p)
        contexts = (Constant(ts), Constant(weights), Constant(1e-15), Constant(8))
        plain = gradient(_transit_plain_weighted, backend, lp, contexts...)
        ruled = gradient(_transit_rule_weighted, backend, lp,
            Constant(ts), Constant(weights))
        fd = _transit_fd_gradient(q -> _transit_rule_weighted(q, ts, weights), lp)
        @test isapprox(ruled, plain; rtol = 2e-7, atol = 2e-8)
        @test isapprox(ruled, fd; rtol = 3e-6, atol = 3e-8)
        @test isapprox(transit_twocmt_rule(ts, p),
            transit_twocmt_unit_response(ts, p...); rtol = 2e-14, atol = 1e-15)
        y, tape = stage_primal(transit_twocmt_rule, Val(2), ts, p)
        (pbar,) = stage_reverse(transit_twocmt_rule, Val(2), tape, weights)
        @test isapprox(pbar .* p, ruled; rtol = 2e-14, atol = 1e-14)
        @test :jac in stage_residuals(transit_twocmt_rule, Val(2))
        # Parameter derivatives of the zero-length integral remain finite.
        @test all(isfinite, ruled)
        @test first(y) == 0.0
    end
    p = [0.08, 0.15, 0.05, 0.2, 1.2]
    @test isempty(transit_twocmt_rule(Float64[], p))
    # refused: rule takes 5 rate parameters, got 4 (length mismatch)
    @test_throws DimensionMismatch transit_twocmt_rule([1.0], p[1:4])
    # refused: series tolerance must be positive (mathematically invalid)
    @test_throws ArgumentError prepare_transit_twocmt_rule(; series_rtol = 0.0)
    # refused: Watson series needs >=1 term (mathematically invalid)
    @test_throws ArgumentError prepare_transit_twocmt_rule(; watson_terms = 0)
    # refused: series tolerance must be positive (mathematically invalid)
    @test_throws ArgumentError transit_twocmt_unit_response([1.0], p...;
        series_rtol = 0.0)
    # refused: Watson series needs >=1 term (mathematically invalid)
    @test_throws ArgumentError transit_twocmt_unit_response([1.0], p...;
        watson_terms = 0)
    # Nonuniform output seeds and active lag inputs exercise the second cut.
    ts = [0.5, 5.3, 5.5, 24.0, 100.0, 614.0, 616.0]
    weights = [cos(i) for i in eachindex(ts)]
    y, tape = stage_primal(transit_twocmt_rule, Val(1), ts, p)
    (tsbar,) = stage_reverse(transit_twocmt_rule, Val(1), tape, weights)
    fdts = _transit_fd_gradient(q -> dot(weights, transit_twocmt_rule(q, p)), ts)
    @test isapprox(tsbar, fdts; rtol = 3e-6, atol = 3e-9)
end

@testset "transit explicit accuracy controls" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    for c in _TRANSIT_CASES[1:2]
        p = [c.k10, c.k12, c.k21, c.rate, c.shape]
        ts = collect(c.ts)
        weights = [cos(i) for i in eachindex(ts)]
        lp = log.(p)
        contexts = (Constant(ts), Constant(weights), Constant(1e-6), Constant(4))
        plain = gradient(_transit_plain_weighted, backend, lp, contexts...)
        ruled = gradient(_transit_loose_weighted, backend, lp,
            Constant(ts), Constant(weights))
        @test isapprox(ruled, plain; rtol = 2e-7, atol = 2e-8)
        high = transit_twocmt_unit_response(ts, p...)
        loose = _TRANSIT_LOOSE_RULE(ts, p)
        @test maximum(abs.(loose .- high)) < 1e-6
    end
end
