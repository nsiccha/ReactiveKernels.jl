using DifferentiationInterface: AutoEnzyme, Constant, gradient
import Enzyme
using LinearAlgebra: dot, Diagonal
using ReactiveKernelsPPL
using Test

# Independent tensor spectral factorization, using the exp-quad spectrum.
_vs_spectral(omega, rho) = sqrt(sqrt(2pi) * rho) * exp(-0.25 * (omega * rho)^2)
function _vs_weights_oracle(q, k)
    dose, conc, sd = exp.(q[1:3])
    frequencies = (pi / 3) .* collect(1:k)
    return sd .* (Diagonal(_vs_spectral.(frequencies, dose)) *
        reshape(q[4:end], k, k) * Diagonal(_vs_spectral.(frequencies, conc)))
end

function _vs_placebo_oracle(times, weights, rho, sd, lo, hi)
    isempty(times) && return Float64[]
    x = 2 .* (clamp.(times, lo, hi) .- lo) ./ (hi - lo) .- 1
    omega = (pi / 3) .* collect(1:length(weights))
    basis = sin.((x .+ 1.5) * omega') ./ sqrt(1.5)
    return basis * (sd .* _vs_spectral.(omega, rho) .* weights)
end

function _vs_pd_oracle(idxs, dts, concentration, log_placebo, logs)
    isempty(idxs) && return Float64[]
    baseline, kout, theta1, theta2 = exp.(logs)
    states = [baseline]
    for (dt, c, placebo) in zip(dts, concentration, exp.(log_placebo))
        loss = kout * placebo * (1 + c / (theta1 * c + theta2))
        # An augmented linear-system exponential, independent of the recurrence.
        operator = [-loss baseline * kout; 0.0 0.0]
        push!(states, first(exp(operator * dt) * [last(states), 1.0]))
    end
    return states[idxs]
end

_vs_weight_objective(q, k) = sum(varyingsource_gp_weights(q[4:end], exp(q[1]),
    exp(q[2]), exp(q[3])))
_vs_placebo_objective(q, times) = sum(varyingsource_log_placebo(times,
    q[3:end], exp(q[1]), exp(q[2]), 0.0, 24.0))
_vs_pd_objective(q, d) = dot(d.weights, varyingsource_pd_locs(d.idxs,
    d.dts, q[5:7], q[8:10], q[1], q[2], q[3], q[4]))

@testset "varyingsource full GP innovation scaling and placebo" begin
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    for k in (2, 8)
        q = vcat(log.([0.8, 1.1, 0.15]), sin.(collect(1:k^2)) ./ 5)
        @test varyingsource_gp_weights(q[4:end], exp(q[1]), exp(q[2]), exp(q[3])) ≈
            _vs_weights_oracle(q, k) rtol=4e-15
        @test gradient(_vs_weight_objective, backend, q, Constant(k)) ≈
            _transit_fd_gradient(p -> _vs_weight_objective(p, k), q) rtol=2e-6 atol=2e-9
    end
    q = vcat(log.([0.9, 0.3]), [0.2, -0.15, 0.05])
    times = [-2.0, 0.0, 3.0, 24.0, 30.0, 3.0]
    @test varyingsource_log_placebo(times, q[3:end], exp(q[1]), exp(q[2]), 0., 24.) ≈
        _vs_placebo_oracle(times, q[3:end], exp(q[1]), exp(q[2]), 0., 24.) rtol=4e-15
    @test gradient(_vs_placebo_objective, backend, q, Constant(times)) ≈
        _transit_fd_gradient(p -> _vs_placebo_objective(p, times), q) rtol=2e-6 atol=2e-9
    @test isempty(varyingsource_log_placebo(Float64[], Float64[], NaN, NaN, NaN, NaN))
    empty_placebo(q) = sum(varyingsource_log_placebo(Float64[], q[3:end], q[1], q[2], NaN, NaN))
    @test gradient(empty_placebo, backend, fill(NaN, 5)) == zeros(5)
    @test_throws ArgumentError varyingsource_gp_weights(ones(3), 1., 1., 1.)
    @test_throws ArgumentError varyingsource_gp_weights(ones(4), 0., 1., 1.)
end

@testset "varyingsource piecewise-exact PD and native reverse" begin
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    q = vcat(log.([120., 0.03, 0.8, 8.]), [0., 4., 1.], [0.1, -0.2, 0.05])
    for dts in ([2., 6., 6.], [0., 1e-8, 20.])
        d = (; idxs=[1, 2, 4], dts, weights=[0.4, -0.5, 0.7])
        @test varyingsource_pd_locs(d.idxs, dts, q[5:7], q[8:10], q[1:4]...) ≈
            _vs_pd_oracle(d.idxs, dts, q[5:7], q[8:10], q[1:4]) rtol=4e-14
        @test gradient(_vs_pd_objective, backend, q, Constant(d)) ≈
            _transit_fd_gradient(p -> _vs_pd_objective(p, d), q) rtol=3e-6 atol=2e-7
    end
    @test varyingsource_pd_locs([1], Float64[], Float64[], Float64[],
        log(120.), NaN, NaN, NaN) == [exp(log(120.))]
    @test isempty(varyingsource_pd_locs(Int[], [NaN], Float64[], Float64[], NaN, NaN, NaN, NaN))
    empty_pd(q) = sum(varyingsource_pd_locs(Int[], [NaN], Float64[], Float64[], q...))
    @test gradient(empty_pd, backend, fill(NaN, 4)) == zeros(4)
    @test_throws ArgumentError varyingsource_pd_locs([1, 1], [1.], [0.], [0.], q[1:4]...)
    @test_throws ArgumentError varyingsource_pd_locs([3], [1.], [0.], [0.], q[1:4]...)
    @test_throws DimensionMismatch varyingsource_pd_locs([2], [1.], Float64[], [0.], q[1:4]...)
end

const _VS_PKPD_TEST_CELL = prepare_varyingsource_pkpd(; watson_terms=24)

function _vs_pkpd_data(; dosed=true, pk_only=false)
    pk = _vs_pk_test_data([0., 1., 4., 5., 6., 8., 11., 12.],
        dosed ? [0., 4., 8.] : Float64[], dosed ? [10000., 20000., 40000.] : Float64[],
        dosed ? [1, 1, 2] : Int[])
    return (; pk, assay=pk_only ? [1, 1] : [3, 1, 2, 1, 2, 3], pk_idxs=[8, 5],
        pd2_idxs=pk_only ? Int[] : [1, 4], pd3_idxs=pk_only ? Int[] : [2, 4],
        dts=pk_only ? Float64[] : [2., 6., 6.],
        centers=pk_only ? Int[] : [2, 4, 7],
        placebo_times=pk_only ? Float64[] : [1., 5., 11., 1., 5., 11.])
end

function _vs_pkpd_point()
    return vcat(log.([100., .08, .15, .05, 120., .03, .8, 8., 80., 1.2, 12., .2, 1.]),
        [0., .2, .15, 0., -.1, .05, -.1, -.2, 0.], [.1, -.15],
        log.([.9, 1.1, .15]), [.02, -.03, -.01, .04],
        log.([.85, .3, 1.05, .2]), [.1, -.1, .05, .1, 0., -.08])
end

function _vs_pkpd_locations(q, d)
    effect = if isempty(d.pk.dose)
        nothing
    else
        varyingsource_effectiveness(varyingsource_gp_weights(q[28:31],
            exp(q[25]), exp(q[26]), exp(q[27])), q[23], q[24])
    end
    placebo = zeros(Float64, length(d.placebo_times))
    extra = zeros(Float64, length(d.placebo_times))
    if !isempty(d.placebo_times)
        primary = varyingsource_log_placebo(d.placebo_times, q[36:38],
            exp(q[32]), exp(q[33]), 0., 24.)
        secondary = varyingsource_log_placebo(d.placebo_times, q[39:41],
            exp(q[34]), exp(q[35]), 0., 24.)
        for i in eachindex(placebo)
            placebo[i] = primary[i]
            extra[i] = secondary[i]
        end
    end
    m = length(d.pk.dose)
    return _VS_PKPD_TEST_CELL(d.pk.n_times, d.assay, d.pk.dose, d.pk.treatment,
        d.pk.lags, d.pk.indices, d.pk.dosing_indices, d.pk_idxs,
        d.pd2_idxs, d.dts, d.centers, d.pd3_idxs, d.dts, d.centers,
        q[14:(13+m)], q[17:(16+m)], q[20:(19+m)], placebo, extra,
        effect, q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10],
        q[11], q[12], q[13])
end

_vs_pkpd_objective(q, d) = dot(cos.(collect(1:length(d.assay))), _vs_pkpd_locations(q, d))

function _vs_pkpd_oracle(q, d)
    weights = _vs_weights_oracle(vcat(q[25:27], q[28:31]), 2)
    concentration = zeros(d.pk.n_times)
    if !isempty(d.pk.dose)
        units = [begin
            rate = exp(q[12] + q[13 + j])
            mode = exp(q[13] + q[16 + j])
            [_oracle_twocmt_unit(t, exp(q[2]), exp(q[3]), exp(q[4]),
                rate, 1 + rate * mode) for t in d.pk.lags]
        end for j in 1:maximum(d.pk.treatment)]
        for i in eachindex(d.pk.dose)
            factor = _vs_gp_oracle(weights, q[23], q[24], d.pk.dose[i],
                concentration[d.pk.dosing_indices[i]]) * exp(q[19 + i] - q[1])
            r = ((i - 1) * d.pk.n_times + 1):(i * d.pk.n_times)
            concentration .+= factor .* units[d.pk.treatment[i]][d.pk.indices[r]]
        end
    end
    placebo = _vs_placebo_oracle(d.placebo_times, q[36:38], exp(q[32]), exp(q[33]), 0., 24.)
    extra = _vs_placebo_oracle(d.placebo_times, q[39:41], exp(q[34]), exp(q[35]), 0., 24.)
    dts2, dts3 = get(d, :pd2_dts, d.dts), get(d, :pd3_dts, d.dts)
    centers2, centers3 = get(d, :pd2_centers, d.centers), get(d, :pd3_centers, d.centers)
    p2 = length(dts2)
    pbmc = _vs_pd_oracle(d.pd2_idxs, dts2, concentration[centers2],
        placebo[1:p2], q[5:8])
    csf = _vs_pd_oracle(d.pd3_idxs, dts3, concentration[centers3],
        placebo[(p2+1):end] .+ extra[(p2+1):end], [q[9], q[6], q[10], q[11]])
    streams = (concentration[d.pk_idxs], pbmc, csf)
    return [streams[a][count(==(a), d.assay[1:i])] for (i, a) in enumerate(d.assay)]
end

@testset "varyingsource composed PK/PBMC/CSF reverse and lazy arms" begin
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    q = _vs_pkpd_point()
    d = _vs_pkpd_data()
    values = _vs_pkpd_locations(q, d)
    @test length(values) == 6
    @test all(isfinite, values)
    @test values[3] == exp(q[5])
    @test values ≈ _vs_pkpd_oracle(q, d) rtol=5e-9 atol=2e-8
    @test gradient(_vs_pkpd_objective, backend, q, Constant(d)) ≈
        _transit_fd_gradient(p -> _vs_pkpd_objective(p, d), q) rtol=4e-6 atol=3e-7
    dpd = _vs_pkpd_data(; dosed=false)
    qpd = copy(q)
    qpd[vcat(1:4, 12:31)] .= NaN
    pdvalues = _vs_pkpd_locations(qpd, dpd)
    @test pdvalues[[2, 4]] == zeros(2)
    @test all(isfinite, pdvalues)
    gpd = gradient(_vs_pkpd_objective, backend, qpd, Constant(dpd))
    @test gpd[vcat(1:4, 12:31)] == zeros(24)
    @test all(isfinite, gpd)
    dpk = _vs_pkpd_data(; pk_only=true)
    qpk = copy(q)
    qpk[vcat(5:11, 32:41)] .= NaN
    @test all(isfinite, _vs_pkpd_locations(qpk, dpk))
    gpk = gradient(_vs_pkpd_objective, backend, qpk, Constant(dpk))
    @test gpk[vcat(5:11, 32:41)] == zeros(17)
    @test all(isfinite, gpk)
end
