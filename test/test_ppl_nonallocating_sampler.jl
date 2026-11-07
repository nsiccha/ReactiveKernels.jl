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

@testset "non-allocating sampler equals the ordinary sampler" begin
    for (label, plan) in _na_sampler_cases()
        @testset "$label" begin
            built = build_kernel(plan)
            n = built.layout.total
            us = [[0.1 * sin(k + s) for k in 1:n] for s in 0:2]
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

@testset "a bound number's value keeps the non-allocating sampler's type" begin
    # The user's direction on numeric values (`ODLPlaus:precompile` decision
    # `2026-11-06T01-09-46-212-0ikeidi`): another value is a runtime input, so
    # it reuses the compiled density and gradient. `z[idx]` gives the gather its
    # index-run metadata, a tuple of arrays and numbers.
    source = @rkppl begin
        z[1:z_length] .~ LogNormal.(mu, sd)
        y .~ Normal.(z[idx], noise)
    end
    function density(u, d)
        lp = 0.0
        for ui in u
            lp += -(ui - d.mu)^2 / (2d.sd^2) - log(d.sd) - log(2pi) / 2
        end
        for (j, i) in enumerate(d.idx)
            lp += -(d.y[j] - exp(u[i]))^2 / (2d.noise^2) - log(d.noise) - log(2pi) / 2
        end
        lp
    end
    function density_gradient(u, d)
        g = [-(ui - d.mu) / d.sd^2 for ui in u]
        for (j, i) in enumerate(d.idx)
            g[i] += (d.y[j] - exp(u[i])) * exp(u[i]) / d.noise^2
        end
        g
    end
    # Both schedules of the gather: flat, and grouped from a mean run of 8. The
    # schedule is the one finite class a binding may change a body by, so each
    # keeps one type across values and row counts on its own.
    for idx_of in (n -> repeat(1:3, n), n -> repeat(1:3, inner = 8n))
        d(mu, sd, noise, n) = begin
            idx = idx_of(n)
            (; y = collect(range(0.1, 0.9; length = length(idx))), idx, z_length = 3,
               mu, sd, noise)
        end
        A = d(0.1, 0.5, 0.2, 1)
        fit = source(; A.idx, A.z_length, A.mu, A.sd, A.noise) | (; A.y)
        built = build_kernel(fit)
        u = [0.1, -0.2, 0.3]
        bindings = (A, d(0.7, 1.3, 0.4, 1), d(-0.4, 0.9, 0.3, 2), A)
        queries = map(b -> prepare_sampler(built, bind_data(fit, b), u;
            backend = NA_SAMPLER_BACKEND, nonallocating = true), bindings)
        @test allequal(typeof.(queries))
        # Each query reads its own values from the one compiled body, and they
        # stay its own when the others run between its calls.
        for _ in 1:2, (b, q) in zip(bindings, queries)
            g = similar(u)
            value, _ = sampler_value_and_gradient!(q, g, u)
            @test value ≈ density(u, b)
            @test q(u) ≈ density(u, b)
            @test g ≈ density_gradient(u, b)
        end
        # Queries of one type never share a slot, neither the density's nor the
        # gradient's.
        slots(k) = filter(slot -> slot isa Base.RefValue, collect(k.caches))
        @test all(queries[i].kernel.caches !== queries[j].kernel.caches
                  for i in eachindex(queries), j in eachindex(queries) if i != j)
        @test all(a !== b for i in eachindex(queries), j in eachindex(queries) if i != j
                  for a in slots(queries[i].kernel), b in slots(queries[j].kernel))
    end
end

@testset "non-allocating programs keep one type across bound numbers" begin
    # Bound prior and response scales in the other shapes of this file.
    regression = @rkppl begin
        alpha ~ Normal(0, prior_scale)
        beta[axes(X, 2)] .~ Normal.(0, prior_scale)
        mu = alpha .+ X * beta
        y .~ Normal.(mu, sigma)
    end
    recurrence = @rkppl begin
        mu ~ Normal(0, prior_scale)
        phi ~ Normal(0, 2)
        sigma ~ HalfCauchy(2.5)
        @scan begin
            nu[1] = mu + phi * mu
            for t in 2:T
                nu[t] = mu + phi * y[t - 1]
            end
        end
        y .~ Normal.(nu, sigma)
    end
    groups = @rkppl begin
        mu ~ Normal(0, prior_scale)
        tau ~ HalfCauchy(5)
        @plate for j in eachindex(y)
            theta[j] ~ Normal(mu, tau)
            y[j] ~ Normal(theta[j], sigma[j])
        end
    end
    cases = (
        (regression, (; X = _NA_X), (; y = _NA_Y),
         (; prior_scale = 10.0, sigma = 0.7), (; prior_scale = 3.0, sigma = 1.4)),
        (recurrence, (; T = length(_NA_SERIES)), (; y = _NA_SERIES),
         (; prior_scale = 10.0), (; prior_scale = 3.0)),
        (groups, (; sigma = [1.5, 0.8, 2.0, 1.1, 0.9]), (; y = [0.4, -1.2, 2.3, 0.1, 0.7]),
         (; prior_scale = 5.0), (; prior_scale = 2.0)),
    )
    for (source, fixed, observed, A, B) in cases
        fit = source(; fixed..., A...) | observed
        built = build_kernel(fit)
        u = [0.1 * sin(k) for k in 1:built.layout.total]
        bound(v) = bind_data(fit, (; fixed..., observed..., v...))
        qA = prepare_sampler(built, bound(A), u; backend = NA_SAMPLER_BACKEND,
            nonallocating = true)
        qB = prepare_sampler(built, bound(B), u; backend = NA_SAMPLER_BACKEND,
            nonallocating = true)
        @test typeof(qA) === typeof(qB)
        ordinary = prepare_sampler(built, bound(B), u; backend = NA_SAMPLER_BACKEND)
        g, gna = similar(u), similar(u)
        value, _ = sampler_value_and_gradient!(ordinary, g, u)
        value_na, _ = sampler_value_and_gradient!(qB, gna, u)
        @test value_na ≈ value rtol = 1e-12
        @test gna ≈ g rtol = 1e-10 atol = 1e-12
        @test qB(u) ≈ value rtol = 1e-12
        # The two values are a real difference, and qA keeps its own.
        gA = similar(u)
        @test first(sampler_value_and_gradient!(qA, gA, u)) ≉ value
    end
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
