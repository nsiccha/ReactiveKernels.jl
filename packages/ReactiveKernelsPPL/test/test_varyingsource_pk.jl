# The convolution oracle comes from test_transit_twocmt.jl: a dense matrix
# exponential plus Gamma-PDF quadrature, independent of the transit rule.
# GP evaluation uses a separate basis/matrix expression. Finite differences
# validate native reverse mode; they are never part of the implementation.
using DifferentiationInterface: AutoEnzyme, Constant, gradient
import Enzyme
using LinearAlgebra: dot
using ReactiveKernelsPPL
using Test

const _VS_PK_TEST_CELL = prepare_varyingsource_pk(; watson_terms = 24)

function _vs_pk_test_data(times, dose_times, amounts, treatment)
    lags = sort(unique(vcat(0.0,
        [max(0.0, t - d) for d in dose_times for t in times])))
    indices = [searchsortedfirst(lags, max(0.0, t - d))
        for d in dose_times for t in times]
    dosing_indices = [searchsortedfirst(times, d) for d in dose_times]
    return (; n_times = length(times), dose = amounts, treatment,
        lags, indices, dosing_indices)
end

function _vs_pk_args(q, d)
    m = length(d.dose)
    weights = reshape(q[(9 + 3m):end], 2, 3)
    effect = varyingsource_effectiveness(weights, q[7], q[8])
    return (d.n_times, d.dose, d.treatment, d.lags, d.indices,
        d.dosing_indices, q[9:(8 + m)], q[(9 + m):(8 + 2m)],
        q[(9 + 2m):(8 + 3m)], effect, q[1:6]...)
end

function _vs_pk_objective(q, d, weights)
    m = length(d.dose)
    effect = varyingsource_effectiveness(reshape(q[(9 + 3m):end], 2, 3), q[7], q[8])
    # Pass the constant design and active parameters directly. Returning them
    # together in a fresh pointer-containing tuple obscures static activity.
    concentration = _VS_PK_TEST_CELL(d.n_times, d.dose, d.treatment, d.lags,
        d.indices, d.dosing_indices, q[9:(8 + m)], q[(9 + m):(8 + 2m)],
        q[(9 + 2m):(8 + 3m)], effect, q[1], q[2], q[3], q[4], q[5], q[6])
    return dot(weights, concentration)
end

function _vs_pk_bound_gp_objective(q, d, gp_weights)
    m = length(d.dose)
    effect = varyingsource_effectiveness(gp_weights, q[7], q[8])
    return sum(_VS_PK_TEST_CELL(d.n_times, d.dose, d.treatment, d.lags,
        d.indices, d.dosing_indices, q[9:(8 + m)], q[(9 + m):(8 + 2m)],
        q[(9 + 2m):(8 + 3m)], effect, q[1], q[2], q[3], q[4], q[5], q[6]))
end

function _vs_gp_oracle(weights, dose_slope, conc_slope, dose, conc)
    basis(x, n) = sin.((pi / 3) .* (x + 1.5) .* collect(1:n))
    surface(x, y) = dose_slope * x + conc_slope * y +
        dot(basis(x, size(weights, 1)), weights * basis(y, size(weights, 2))) / 1.5
    x = 2 * (log(dose) - log(10000.0)) / (log(200000.0) - log(10000.0)) - 1
    y = 2 * log1p(clamp(conc, 0.0, 2000.0)) / log1p(2000.0) - 1
    return dose * exp(surface(x, y) - surface(-1.0, -1.0))
end

function _vs_pk_oracle(q, d)
    m = length(d.dose)
    m == 0 && return zeros(d.n_times)
    weights = reshape(q[(9 + 3m):end], 2, 3)
    # Source columns are the first max(treatment) dose columns, as deployed.
    units = [begin
        rate = exp(q[5] + q[8 + j])
        mode = exp(q[6] + q[8 + m + j])
        [_oracle_twocmt_unit(t, exp(q[2]), exp(q[3]), exp(q[4]),
            rate, 1 + rate * mode) for t in d.lags]
    end for j in 1:maximum(d.treatment)]
    concentration = zeros(d.n_times)
    for i in 1:m
        effective = _vs_gp_oracle(weights, q[7], q[8], d.dose[i],
            concentration[d.dosing_indices[i]])
        factor = effective * exp(q[8 + 2m + i] - q[1])
        concentration .+= factor .* units[d.treatment[i]][
            d.indices[((i - 1) * d.n_times + 1):(i * d.n_times)]]
    end
    return concentration
end

function _vs_pk_point(m)
    return vcat(log.([100.0, 0.08, 0.15, 0.05, 0.2, 1.0]), [0.1, -0.15],
        [0.15 * (i - 1) for i in 1:m], [-0.1 * (i - 1) for i in 1:m],
        [-0.2 * (i - 1) for i in 1:m], vec([0.02 -0.03 0.01; -0.01 0.04 -0.02]))
end

@testset "varyingsource normalized dose/concentration GP" begin
    weights = [0.02 -0.03 0.01; -0.01 0.04 -0.02]
    e = varyingsource_effectiveness(weights, 0.1, -0.15)
    @test varyingsource_effective_dose(10000.0, 0.0, e) == 10000.0
    for dose in (5000.0, 10000.0, 80000.0, 300000.0), conc in (-2.0, 0.0, 10.0, 2000.0, 4000.0)
        @test isapprox(varyingsource_effective_dose(dose, conc, e),
            _vs_gp_oracle(weights, 0.1, -0.15, dose, conc); rtol = 3e-15)
    end
    @test_throws ArgumentError varyingsource_effective_dose(0.0, 0.0, e)
    @test_throws ArgumentError varyingsource_effectiveness(zeros(0, 3), 0.1, 0.2)
end

@testset "varyingsource PK superposition and reverse gradients" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    cases = (
        _vs_pk_test_data([0.0, 1.0, 4.0, 8.0, 12.0, 24.0, 28.0],
            [0.0, 4.0, 8.0], [10000.0, 20000.0, 40000.0], [1, 1, 2]),
        _vs_pk_test_data([0.0, 1.0, 4.0, 12.0, 24.0],
            [0.0, 0.0, 12.0], [10000.0, 30000.0, 20000.0], [1, 2, 1]),
    )
    for d in cases
        q = _vs_pk_point(length(d.dose))
        weights = cos.(collect(1:d.n_times))
        args = _vs_pk_args(q, d)
        got = _VS_PK_TEST_CELL(args...)
        @test isapprox(got, _vs_pk_oracle(q, d); rtol = 5e-9, atol = 1e-8)
        @test first(got) == 0.0
        @test all(isfinite, got)
        @test all(>=(0.0), got)
        g = gradient(_vs_pk_objective, backend, q, Constant(d), Constant(weights))
        fd = _transit_fd_gradient(p -> _vs_pk_objective(p, d, weights), q)
        @test isapprox(g, fd; rtol = 4e-6, atol = 2e-7)
        @test all(isfinite, g)
        # The third source column is unused despite the third dose being used.
        @test g[11] == 0.0
        @test g[14] == 0.0
        @test abs(g[17]) > 0.0
        @test abs(g[8]) > 0.0  # feedback through concentration at later doses
        qbound = q[1:(8 + 3length(d.dose))]
        gpbound = reshape(q[(length(qbound) + 1):end], 2, 3)
        gb = gradient(_vs_pk_bound_gp_objective, backend, qbound,
            Constant(d), Constant(gpbound))
        fdb = _transit_fd_gradient(p -> _vs_pk_bound_gp_objective(p, d, gpbound), qbound)
        @test gb ≈ fdb rtol = 4e-6 atol = 2e-7
        @test isapprox(varyingsource_pk_concentration(args...), got;
            rtol = 5e-9, atol = 1e-8)
        obs_indices = [d.n_times, 2, 2]
        @test varyingsource_pk_locs(obs_indices, args...) ==
            varyingsource_pk_concentration(args...)[obs_indices]
        bad = (d.n_times, args[2:4]..., d.indices[1:end-1], args[6:end]...)
        @test_throws DimensionMismatch _VS_PK_TEST_CELL(bad...)
        badmap = (d.n_times, d.dose, [0, 1, 2], args[4:end]...)
        @test_throws ArgumentError _VS_PK_TEST_CELL(badmap...)
    end
end

_vs_empty_objective(q, n) = sum(_VS_PK_TEST_CELL(n, Float64[], Int[], [NaN],
    Int[], Int[], Float64[], Float64[], Float64[], nothing, q[1:6]...))

@testset "varyingsource no-dose arm is lazy" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    q = fill(NaN, 6)
    @test _vs_empty_objective(q, 4) == 0.0
    @test isempty(_VS_PK_TEST_CELL(0, Float64[], Int[], Float64[],
        Int[], Int[], Float64[], Float64[], Float64[], nothing, q...))
    @test gradient(_vs_empty_objective, backend, q, Constant(4)) == zeros(6)
end
