module GibbsSliceTests
using Test, ReactiveKernels, ReactiveKernelsPPL, Random, LinearAlgebra, Statistics
using StatsFuns: normcdf
const PPL = ReactiveKernelsPPL

# Monte Carlo standard error of a chain average by batch means.
function batch_mcse(x; batches=50)
    m = length(x) ÷ batches
    means = [mean(view(x, (b - 1) * m + 1:b * m)) for b in 1:batches]
    return std(means) / sqrt(batches)
end

# |estimate - truth| within `z` standard errors of the chain average of `x`.
within(x, truth; z=5) = abs(mean(x) - truth) < z * batch_mcse(x)

# A 1-D chain of the bare transition, counting density evaluations.
function chain1(f, x0, n; w, p, seed)
    rng = Xoshiro(seed)
    calls = Ref(0)
    g = z -> (calls[] += 1; f(z))
    xs = Vector{Float64}(undef, n)
    x, reported = Float64(x0), 0
    for i in 1:n
        x, used = PPL._slice_doubling(rng, g, x, f(x), w, p)
        reported += used
        xs[i] = x
    end
    return xs, reported, calls[]
end

@testset "coordinate updates are exported" begin
    for name in (:prepare_coordinate_conditional, :CoordinateConditional, :coordinate_state,
            :CoordinateState, :coordinate_logdensity, :set_coordinate!, :refresh_products!,
            :coordinate_slice_step!, :coordinate_slice_sweep!)
        @test Base.isexported(ReactiveKernelsPPL, name)
    end
end

@testset "slice transition leaves one-dimensional targets invariant" begin
    normal(z) = -0.5z^2
    xs, reported, calls = chain1(normal, 0.0, 100_000; w=10.0, p=20, seed=1)
    @test reported == calls
    @test within(xs, 0.0)
    @test within(xs .^ 2, 1.0)
    @test within(xs .< 1.0, normcdf(1.0))

    # A width far below the scale: the interval is doubled many times, and
    # the acceptance test for doubled intervals is exercised.
    wide(z) = -0.5(z / 100)^2
    xs, reported, calls = chain1(wide, 3.0, 100_000; w=1.0, p=20, seed=2)
    @test reported == calls
    @test within(xs ./ 100, 0.0)
    @test within((xs ./ 100) .^ 2, 1.0)

    # Without doubling the interval stays at width `w` around the point.
    xs, = chain1(normal, 0.5, 100_000; w=4.0, p=0, seed=3)
    @test within(xs .^ 2, 1.0)
    @test within(xs .< -0.5, normcdf(-0.5))

    # Two separated modes of unequal mass, a narrow width: a doubled interval
    # can contain both modes, and only the acceptance test keeps the chain
    # reversible there (Neal 2003, section 4.2).
    lse(a, b) = max(a, b) + log1p(exp(-abs(a - b)))
    bimodal(z) = lse(log(0.3) - 0.5((z + 1.5) / 0.6)^2, log(0.7) - 0.5((z - 1.5) / 0.6)^2)
    xs, = chain1(bimodal, 0.0, 200_000; w=0.5, p=20, seed=4)
    @test within(xs .> 0.0, 0.3 * normcdf(-1.5 / 0.6) + 0.7 * normcdf(1.5 / 0.6))
    @test within(xs, -0.3 * 1.5 + 0.7 * 1.5)

    # Non-finite values lie outside the slice: a half-line target.
    halfline(z) = z > 0 ? -z : (z == 0 ? 0.0 : NaN)
    xs, = chain1(halfline, 1.0, 100_000; w=10.0, p=20, seed=5)
    @test all(>(0), xs)
    @test within(xs, 1.0)
end

const GAUSSIAN = @rkppl begin
    beta[axes(X, 2)] .~ Normal.(0, s0)
    y .~ Normal.(X * beta, sigma)
end
const LOGISTIC = @rkppl begin
    beta[axes(X, 2)] .~ Normal.(0, sigma)
    y .~ Bernoulli.(logistic.(X * beta))
end

function sweeps!(rng, state, n; refresh_every=500, kwargs...)
    d = length(state.position)
    draws = Matrix{Float64}(undef, d, n)
    drift, evaluations = 0.0, 0
    for i in 1:n
        evaluations += PPL.coordinate_slice_sweep!(rng, state; kwargs...)
        draws[:, i] = state.position
        i % refresh_every == 0 && (drift = max(drift, PPL.refresh_products!(state)))
    end
    return (; draws, drift, evaluations)
end

@testset "coordinate slice sweeps sample a Gaussian linear model exactly" begin
    rng = Xoshiro(20261117)
    n, d = 25, 3
    # Correlated columns, so the conditionals differ from the marginals.
    X = randn(rng, n, d) * [1.0 0.6 0.0; 0.0 0.8 0.5; 0.0 0.0 0.9]
    s0, sigma = 2.0, 0.7
    y = X * [0.8, -0.5, 0.3] .+ sigma .* randn(rng, n)
    precision = Symmetric(X' * X ./ sigma^2 + I ./ s0^2)
    covariance = inv(precision)
    mu = covariance * (X' * y ./ sigma^2)

    plan = GAUSSIAN(; X, s0, sigma) | (; y)
    problem = PPL.prepare_coordinate_conditional(build_kernel(plan), plan, zeros(d))
    @test length(problem.products) == 1
    state = PPL.coordinate_state(problem, zeros(d))
    burn = sweeps!(rng, state, 200)
    run = sweeps!(rng, state, 30_000)
    draws = run.draws
    for j in 1:d
        @test within(draws[j, :], mu[j])
        @test within((draws[j, :] .- mu[j]) .^ 2, covariance[j, j])
    end
    @test within((draws[1, :] .- mu[1]) .* (draws[2, :] .- mu[2]), covariance[1, 2])
    @test run.evaluations > 30_000 * d
    # Column updates drift by rounding only, and the products follow the point.
    @test run.drift < 1e-10
    @test only(state.products) ≈ X * state.position

    # A fixed seed reproduces the chain; the problem keeps no sampler state.
    a = PPL.coordinate_state(problem, mu)
    b = PPL.coordinate_state(problem, mu)
    ra = sweeps!(Xoshiro(7), a, 50)
    rb = sweeps!(Xoshiro(7), b, 50)
    @test ra.draws == rb.draws && ra.evaluations == rb.evaluations

    # Per-coordinate widths, and refusals.
    c = PPL.coordinate_state(problem, mu)
    @test PPL.coordinate_slice_sweep!(rng, c; width=[0.5, 1.0, 2.0]) >= 2d
    @test_throws ArgumentError PPL.coordinate_slice_sweep!(rng, c; width=0.0)
    @test_throws ArgumentError PPL.coordinate_slice_sweep!(rng, c; width=Inf)
    @test_throws ArgumentError PPL.coordinate_slice_sweep!(rng, c; max_doublings=-1)
    @test_throws DimensionMismatch PPL.coordinate_slice_sweep!(rng, c; width=[1.0, 1.0])
    @test_throws BoundsError PPL.coordinate_slice_step!(rng, c, 4)
    kept = copy(c.position)
    @test PPL.coordinate_slice_step!(rng, c, 2; width=1.0) >= 4
    @test c.position[[1, 3]] == kept[[1, 3]]
end

@testset "coordinate slice sweeps match quadrature on a tiny logistic posterior" begin
    rng = Xoshiro(41)
    n = 15
    X = randn(rng, n, 2)
    y = Int.(rand(rng, n) .< 1 ./ (1 .+ exp.(-X * [1.0, -0.7])))
    sigma = 2.0
    logpost(b) = sum(@. y * $(X * b) - log1p(exp($(X * b)))) - 0.5 * sum(abs2, b) / sigma^2

    # Quadrature on a grid wide enough for the posterior's mass.
    grid = range(-8, 8; length=801)
    weights = [logpost([b1, b2]) for b1 in grid, b2 in grid]
    weights = exp.(weights .- maximum(weights))
    weights ./= sum(weights)
    m1 = sum(weights .* grid)
    m2 = sum(weights .* grid')
    v1 = sum(weights .* (grid .- m1) .^ 2)
    v2 = sum(weights .* (grid' .- m2) .^ 2)
    c12 = sum(weights .* (grid .- m1) .* (grid' .- m2))

    plan = LOGISTIC(; X, sigma) | (; y)
    problem = PPL.prepare_coordinate_conditional(build_kernel(plan), plan, zeros(2))
    state = PPL.coordinate_state(problem, zeros(2))
    sweeps!(rng, state, 200)
    run = sweeps!(rng, state, 40_000)
    b1, b2 = run.draws[1, :], run.draws[2, :]
    @test within(b1, m1)
    @test within(b2, m2)
    @test within((b1 .- m1) .^ 2, v1)
    @test within((b2 .- m2) .^ 2, v2)
    @test within((b1 .- m1) .* (b2 .- m2), c12)
    @test run.drift < 1e-10
end
end
