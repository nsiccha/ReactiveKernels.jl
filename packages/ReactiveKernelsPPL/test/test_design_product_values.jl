using DifferentiationInterface, Distributions, Enzyme, ReactiveKernels, ReactiveKernelsPPL, Test

# Likelihoods located at a design product (`X * beta`, with or without an
# intercept) are per-observation statements. ReactiveKernels and RKPPL derive
# their computation from that statement; the fused GLM statements RKPPL once
# accepted are refused.

const _GV_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

_gv_rows(n) = collect(range(-1.0, 1.0; length = n))
_gv_X(n) = hcat(_gv_rows(n), cos.(1:n), 0.3 .* sin.(2 .* (1:n)))
_gv_y(n) = _gv_X(n) * [0.4, -0.2, 0.1] .+ 0.3 .+ 0.05 .* sin.(3 .* (1:n))
_gv_counts(n) = [mod(3i, 4) for i in 1:n]
_gv_flags(n) = [isodd(i ÷ 2) ? 1 : 0 for i in 1:n]
_gv_u(built) = [0.15sin(i + 0.3) for i in 1:built.layout.total]

function _gv_fd(f, u; h = 1e-6)
    return [(f(u .+ h .* (eachindex(u) .== i)) - f(u .- h .* (eachindex(u) .== i))) / 2h
        for i in eachindex(u)]
end

# Likelihood against the independent oracle `lik(constrained values)`, the
# native Enzyme gradient against central differences, and the caller's data
# left unchanged.
function _gv_check(plan, data, lik)
    saved = deepcopy(data)
    built = build_kernel(plan)
    u = _gv_u(built)
    likelihood = prepare_query(built, plan, :likelihood)
    @test Base.invokelatest(likelihood, u) ≈ lik(constrain(built.layout, u)) rtol = 1e-10
    q = prepare_sampler(built, plan, u; backend = _GV_BACKEND)
    g = similar(u)
    value, _ = sampler_value_and_gradient!(q, g, u)
    @test value ≈ q(u)
    @test g ≈ _gv_fd(v -> q(v), u) rtol = 1e-5 atol = 1e-6
    @test data == saved
    return built
end

@testset "zero-centered Normal vector prior scales before squaring" begin
    m = @rkppl begin
        beta[1:2] .~ Normal.(0, 1e150)
    end
    plan = bind_data(m())
    built = build_kernel(plan)
    u = [1e160, 0.0]
    expected = sum(logpdf.(Normal(0, 1e150), u))
    prior = prepare_query(built, plan, :prior)
    @test isfinite(Base.invokelatest(prior, u))
    @test Base.invokelatest(prior, u) ≈ expected rtol = 1e-12
    q = prepare_sampler(built, plan, u; backend = _GV_BACKEND)
    g = similar(u)
    value, _ = sampler_value_and_gradient!(q, g, u)
    @test value ≈ expected rtol = 1e-12
    @test g ≈ [-1e-140, 0.0] rtol = 1e-10
end

@testset "per-observation likelihoods over bound data matrices" begin
    n = 7
    X, y, counts, flags = _gv_X(n), _gv_y(n), _gv_counts(n), _gv_flags(n)
    m = @rkppl begin
        beta[axes(X, 2)] .~ Normal.(0, 10)
        sigma ~ Exponential(1)
        y .~ Normal.(X * beta, sigma)
    end
    built = _gv_check(m(; X) | (; y), (; X, y),
        nt -> sum(logpdf.(Normal.(X * nt.beta, nt.sigma), y)))
    @test coordinate_names(built.layout) == [:sigma, Symbol("beta.1"),
        Symbol("beta.2"), Symbol("beta.3")]

    m = @rkppl begin
        alpha ~ Normal(0, 10)
        beta[axes(lags, 2)] .~ Normal.(0, 10)
        sigma ~ Exponential(1)
        y .~ Normal.(alpha .+ lags * beta, sigma)
    end
    _gv_check(m(; lags = X) | (; y), (; X, y),
        nt -> sum(logpdf.(Normal.(nt.alpha .+ X * nt.beta, nt.sigma), y)))

    m = @rkppl begin
        alpha ~ Normal(0, 10)
        beta1 ~ Normal(0, 1)
        beta2 ~ Normal(0, 1)
        beta3 ~ Normal(0, 1)
        b = [beta1, beta2, beta3]
        counts .~ Poisson.(exp.(alpha .+ X * b))
    end
    built = _gv_check(m(; X) | (; counts), (; X, counts),
        nt -> sum(logpdf.(Poisson.(exp.(nt.alpha .+ X * [nt.beta1, nt.beta2, nt.beta3])),
            counts)))
    @test coordinate_names(built.layout) == [:alpha, :beta1, :beta2, :beta3]

    m = @rkppl begin
        beta[axes(X, 2)] .~ Normal.(0, 10)
        flags .~ BernoulliLogit.(X * beta)
    end
    _gv_check(m(; X) | (; flags), (; X, flags),
        nt -> sum(logpdf.(Bernoulli.(1 ./ (1 .+ exp.(.-(X * nt.beta)))), flags)))
end

# `y ~ NormalIDGLM(X, alpha, beta, sigma)`, `y ~ BernoulliLogitGLM(X, alpha,
# beta)` and `y ~ PoissonLogGLM(X, alpha, beta)` are refused, in call or dotted
# form and inside an evidence wrapper. The requirement is the user's decision
# `ReactiveKernels:posteriordb-perf/decisions/2026-11-20T15-12-50-852-02qiqax`
# (remove the fused GLM objects; a model states its per-observation
# likelihood). The message names the statement to write from the caller's own
# arguments.
@testset "fused GLM statements are refused" begin
    refusal(ex) = try
        lower_rkppl(ex, (:X, :y); conditioned = (:y,))
        nothing
    catch err
        sprint(showerror, err)
    end
    msg = refusal(:(begin
        beta[axes(X, 2)] .~ Normal.(0, 1)
        sigma ~ HalfNormal(1)
        y ~ NormalIDGLM(X, 0, beta, sigma)
    end))
    # refused: a removed fused GLM statement (decision 02qiqax)
    @test msg !== nothing && occursin("`NormalIDGLM` is not an RKPPL statement", msg)
    @test occursin("`y .~ Normal.(X * beta, sigma)`", msg)
    msg = refusal(:(begin
        alpha ~ Normal(0, 1)
        beta[axes(X, 2)] .~ Normal.(0, 1)
        y ~ BernoulliLogitGLM(X, alpha, beta)
    end))
    # refused: a removed fused GLM statement (decision 02qiqax)
    @test msg !== nothing && occursin("`y .~ BernoulliLogit.(alpha .+ X * beta)`", msg)
    msg = refusal(:(begin
        alpha ~ Normal(0, 1)
        beta[axes(X, 2)] .~ Normal.(0, 1)
        y .~ PoissonLogGLM.(X, alpha, beta)
    end))
    # refused: a removed fused GLM statement, dotted (decision 02qiqax)
    @test msg !== nothing && occursin("`y .~ Poisson.(exp.(alpha .+ X * beta))`", msg)
    msg = refusal(:(begin
        beta[axes(X, 2)] .~ Normal.(0, 1)
        sigma ~ HalfNormal(1)
        y ~ censored(NormalIDGLM(X, 0.0, beta, sigma), -1, 1)
    end))
    # refused: a removed fused GLM statement inside an evidence wrapper (decision 02qiqax)
    @test msg !== nothing && occursin("inside the dotted `censored.(...)` wrapper", msg)
end

# The per-observation spelling of the same regression uses ordinary AD:
# a Gaussian response located at `M * v` (inline or named) or
# `alpha .+ M * v` (bound data matrix, declared coefficient vector,
# sampled intercept; one shared scale) sums its cells in the observation
# plate, as every other Normal location does. A non-positive scale keeps
# the plate cell's `-Inf`.
@testset "per-observation Normal over a design product uses ordinary AD" begin
    n = 9
    X, y = _gv_X(n), _gv_y(n)
    g = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    product = @rkppl begin
        beta[axes(X, 2)] .~ Normal.(0, 10)
        sigma ~ Exponential(1)
        y .~ Normal.(X * beta, sigma)
    end
    named_product = @rkppl begin
        beta[axes(X, 2)] .~ Normal.(0, 10)
        sigma ~ Exponential(1)
        mu = X * beta
        y .~ Normal.(mu, sigma)
    end
    intercept = @rkppl begin
        alpha ~ Normal(0, 10)
        beta[axes(X, 2)] .~ Normal.(0, 10)
        sigma ~ Exponential(1)
        mu = alpha .+ X * beta
        y .~ Normal.(mu, sigma)
    end
    fixed_scale = @rkppl begin
        beta[axes(X, 2)] .~ Normal.(0, 10)
        y .~ Normal.(X * beta, s)
    end
    flat = @rkppl begin
        beta[axes(X, 2)] .~ Flat()
        sigma ~ restricted(Flat(), 0, Inf)
        mu = X * beta
        y .~ Normal.(mu, sigma)
    end
    cases = (
        (product(; X) | (; y), (; X, y), nt -> X * nt.beta, nt -> nt.sigma),
        (named_product(; X) | (; y), (; X, y), nt -> X * nt.beta, nt -> nt.sigma),
        (intercept(; X) | (; y), (; X, y), nt -> nt.alpha .+ X * nt.beta, nt -> nt.sigma),
        (fixed_scale(; X, s = 2) | (; y), (; X, y), nt -> X * nt.beta, nt -> 2.0),
        (flat(; X) | (; y), (; X, y), nt -> X * nt.beta, nt -> nt.sigma),
    )
    for (plan, data, mean, scale) in cases
        oracle = nt -> logpdf.(Normal.(mean(nt), scale(nt)), y)
        built = _gv_check(plan, data, nt -> sum(oracle(nt)))
        u = _gv_u(built)
        nt = constrain(built.layout, u)
        @test Base.invokelatest(prepare_query(built, plan, :pointwise), u).y ≈ oracle(nt)
        # The per-observation queries still read the plate.
        obs = Base.invokelatest(prepare_query(built, plan, :observations), u).y
        @test obs.location ≈ mean(nt) && all(obs.scale .≈ scale(nt))
        cell = prepare_cell_query(built, plan, :y)
        @test Base.invokelatest(cell, u, 4) ≈ oracle(nt)[4]
    end

    # A real-support scale keeps the plate cell's guard: -Inf at sigma <= 0.
    signed = @rkppl begin
        beta[axes(X, 2)] .~ Normal.(0, 10)
        sigma ~ Normal(0, 1)
        y .~ Normal.(X * beta, sigma)
    end
    plan = signed(; X) | (; y)
    built = build_kernel(plan)
    likelihood = prepare_query(built, plan, :likelihood)
    u = _gv_u(built)
    nt = constrain(built.layout, u)
    k = findfirst(==(:sigma), coordinate_names(built.layout))
    u[k] = 0.7
    @test Base.invokelatest(likelihood, u) ≈
        sum(logpdf.(Normal.(X * constrain(built.layout, u).beta, 0.7), y))
    u[k] = -0.7
    @test Base.invokelatest(likelihood, u) == -Inf

    # The gradient allocates the design product and its shadow, nothing
    # observation-sized beyond them: the plate reads the location in place.
    # A materialized residual (`y .- mu`, then `dot`) cost two more vectors
    # (snag rkppl-natural-ga-1c464809). Bounded, not exact: allocation
    # identity is not portable across platforms.
    big = 1000
    Xb = hcat(ones(big), sin.(1:big))
    yb = Xb * [0.3, 0.2] .+ cos.(1:big)
    for plan in (flat(; X = Xb) | (; y = yb), intercept(; X = Xb) | (; y = yb))
        built = build_kernel(plan)
        u = _gv_u(built)
        q = prepare_sampler(built, plan, u; backend = _GV_BACKEND)
        gb = similar(u)
        sampler_value_and_gradient!(q, gb, u)
        @test @allocated(sampler_value_and_gradient!(q, gb, u)) < 3 * sizeof(Float64) * big
    end

    # A location that is not a data-matrix product keeps the plate.
    grouped = @rkppl begin
        z[levels(g)] .~ Normal.(0, 1)
        sigma ~ Exponential(1)
        y .~ Normal.(z[g], sigma)
    end
    plan = grouped(; g) | (; y)
    _gv_check(plan, (; g, y), nt -> sum(logpdf.(Normal.(nt.z[g], nt.sigma), y)))
end
