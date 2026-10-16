# `prepare_sampler(...; nonallocating = true)`: the sampler density and its
# reverse gradient prepared over `prepare_nonallocating`. The query owns the
# buffers its operations write into, so an evaluation reuses them instead of
# allocating observation-length intermediates. Values and gradients equal the
# ordinary sampler's. Runs only through `test/run_nonallocating_integration.jl`,
# not the default suite, because it needs the unregistered MutatingFunctions
# weak dependency.
using ReactiveKernels
using ReactiveKernelsPPL
using MutatingFunctions
using DifferentiationInterface: AutoEnzyme
import Enzyme
using Test

const NA_SAMPLER_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

# A Gaussian regression whose location is a matrix-vector product: the
# ordinary kernel allocates the product on every evaluation.
_na_regression = @rkppl begin
    alpha ~ Normal(0, 10)
    beta[axes(X, 2)] .~ Normal.(0, 10)
    sigma ~ HalfCauchy(2.5)
    mu = alpha .+ X * beta
    y .~ Normal.(mu, sigma)
end

# A one-step error recurrence through `@scan`.
_na_recurrence = @rkppl begin
    mu ~ Normal(0, 10)
    phi ~ Normal(0, 2)
    theta ~ Normal(0, 2)
    sigma ~ HalfCauchy(2.5)
    @scan begin
        nu[1] = mu + phi * mu
        err[1] = y[1] - nu[1]
        for t in 2:T
            nu[t] = mu + phi * y[t - 1] + theta * err[t - 1]
            err[t] = y[t] - nu[t]
        end
    end
    y .~ Normal.(nu, sigma)
end

# Per-group latents in a plate.
_na_groups = @rkppl begin
    mu ~ Normal(0, 5)
    tau ~ HalfCauchy(5)
    @plate for j in eachindex(y)
        theta[j] ~ Normal(mu, tau)
        y[j] ~ Normal(theta[j], sigma[j])
    end
end

const _NA_X = [sin(0.7i + j) for i in 1:60, j in 1:3]
const _NA_Y = [0.5 + 0.8 * _NA_X[i, 1] - 0.3 * _NA_X[i, 2] + 0.2 * cos(1.3i) for i in 1:60]
const _NA_SERIES = [sin(0.3t) + 0.1 * cos(2.1t) for t in 1:80]

_na_sampler_cases() = (
    ("matrix-vector regression", _na_regression(; X = _NA_X) | (; y = _NA_Y)),
    ("scan recurrence", _na_recurrence(; T = length(_NA_SERIES)) | (; y = _NA_SERIES)),
    ("group plate", _na_groups(; sigma = [1.5, 0.8, 2.0, 1.1, 0.9]) |
        (; y = [0.4, -1.2, 2.3, 0.1, 0.7])),
)

_na_density_bytes(q, u) = (q(u); q(u); @allocated q(u))

# Julia 1.10 under `--check-bounds=yes` (this runner's flag): Enzyme's static
# activity analysis rejects the non-allocating reverse gradient of the scan
# recurrence. Default bounds checking and Julia 1.13 differentiate it. Tracked
# by ReactiveKernels todo 2026-10-16T02-45-16-840-00lejdv.
_na_scan_gradient_gap(label) = label == "scan recurrence" &&
    VERSION < v"1.11" && Base.JLOptions().check_bounds == 1

@testset "non-allocating sampler equals the ordinary sampler" begin
    for (label, plan) in _na_sampler_cases()
        @testset "$label" begin
            built = build_kernel(plan)
            n = built.layout.total
            us = [[0.1 * sin(k + s) for k in 1:n] for s in 0:2]
            if _na_scan_gradient_gap(label)
                @test_broken (prepare_sampler(built, plan, us[1]; backend = NA_SAMPLER_BACKEND,
                    warm = true, nonallocating = true); true)
                continue
            end
            q = prepare_sampler(built, plan, us[1]; backend = NA_SAMPLER_BACKEND, warm = true)
            qna = prepare_sampler(built, plan, us[1]; backend = NA_SAMPLER_BACKEND,
                warm = true, nonallocating = true)
            @test qna.kernel isa NonAllocatingKernel
            # Several points in turn: the reused buffers carry nothing from
            # one evaluation to the next.
            for u in us
                g, gna = zeros(n), zeros(n)
                value, _ = sampler_value_and_gradient!(q, g, u)
                value_na, returned = sampler_value_and_gradient!(qna, gna, u)
                @test returned === gna
                @test value_na ≈ value rtol = 1e-12
                @test gna ≈ g rtol = 1e-10 atol = 1e-12
                @test qna(u) ≈ q(u) rtol = 1e-12
                @test qna(u) ≈ value_na rtol = 1e-12
            end
            @test qna(Float32.(us[2])) == qna(Vector{Float64}(Float32.(us[2])))
        end
    end
end

@testset "non-allocating sampler density reuses its buffers" begin
    plan = first(_na_sampler_cases())[2]
    built = build_kernel(plan)
    u = [0.1 * sin(k) for k in 1:built.layout.total]
    q = prepare_sampler(built, plan, u; backend = NA_SAMPLER_BACKEND)
    qna = prepare_sampler(built, plan, u; backend = NA_SAMPLER_BACKEND, nonallocating = true)
    ordinary = Base.invokelatest(_na_density_bytes, q, u)
    reused = Base.invokelatest(_na_density_bytes, qna, u)
    # The ordinary density allocates the product (one 60-element vector; the
    # location and likelihood plates fuse into the total); the non-allocating
    # one reuses its buffer and keeps no observation-length vector.
    @test ordinary >= 60 * sizeof(Float64)
    @test reused < ordinary
    @test reused < 60 * sizeof(Float64)
end

@testset "non-allocating sampler options" begin
    plan = first(_na_sampler_cases())[2]
    built = build_kernel(plan)
    u = [0.1 * sin(k) for k in 1:built.layout.total]
    # Eligibility for native forward batching excludes non-allocating kernels:
    # the preparation stays the ordinary reverse one and agrees.
    qf = prepare_sampler(built, plan, u; backend = AutoEnzyme(),
        nonallocating = true, native_forward_batch = true)
    q = prepare_sampler(built, plan, u; backend = NA_SAMPLER_BACKEND)
    g, gf = zeros(length(u)), zeros(length(u))
    @test first(sampler_value_and_gradient!(qf, gf, u)) ≈
        first(sampler_value_and_gradient!(q, g, u)) rtol = 1e-12
    @test gf ≈ g rtol = 1e-10 atol = 1e-12
    # Retained outputs over a non-allocating kernel are not built in
    # ReactiveKernels yet (capability gap, not a refused shape).
    @test_broken (prepare_sampler(built, plan, u; backend = NA_SAMPLER_BACKEND,
        nonallocating = true, retain = (:pointwise,)); true)
end
