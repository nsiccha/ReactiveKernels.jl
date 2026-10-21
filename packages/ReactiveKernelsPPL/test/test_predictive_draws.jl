using Distributions, Random, ReactiveKernels, ReactiveKernelsPPL, Statistics, Test

# Posterior-predictive draws (snag predictive-rng-f-f51c6561; user pick on
# design decision `05ameuz`: law arguments in the graph, `rand(rng, law)` with
# Distributions.jl at the query boundary). The graph computes each
# observation's law; the extension builds it as a Distributions.jl law. Each
# law is checked against the model's own density: its `logpdf` at the
# observed value is the response's `:pointwise` entry, at more than one
# parameter point. Draws are then checked to come from those laws.

_pd_x = [0.1, 0.5, 1.0, 2.0]
_pd_n = [3, 3, 4, 5]
_pd_lloq = 0.5
_pd_u(built, shift) = shift .+ 0.1 .* (1:built.layout.total) .- 0.05
_pd_addprop(m, a, p) = sqrt(a^2 + (m * p)^2)
_pd_nl(v, mu) = -0.5 * (v - mu)^2

# Two censored Normal laws with the same parameters, to rounding.
_pd_same(l, mu, sd, lower) = l isa Distributions.Censored && l.lower == lower && l.upper === nothing &&
    l.uncensored.μ ≈ mu && l.uncensored.σ ≈ sd

# Every entry's law has the model's density at the observed value.
function _pd_laws_match(model, data, y; points = (0.0, 0.13))
    fit = model(; data...) | (; y)
    built = build_kernel(fit)
    q = prepare_predictive(built, fit)
    pointwise = prepare_query(built, fit, :pointwise)
    return all(points) do shift
        u = _pd_u(built, shift)
        laws = predictive_laws(q, u).y
        pw = Base.invokelatest(pointwise, u).y
        size(laws) == size(y) && all(isapprox(logpdf(laws[I], y[I]), pw[I];
            rtol = 1e-10, atol = 1e-12) for I in CartesianIndices(y))
    end
end

function _pd_draws(model, data, y, rng)
    fit = model(; data...) | (; y)
    built = build_kernel(fit)
    return prepare_predictive(built, fit)(_pd_u(built, 0.0), rng)
end

@testset "predictive draws: each entry's law is the model's own" begin
    x, n, lloq = _pd_x, _pd_n, _pd_lloq
    yc = [1.2, 0.5, 2.5, 4.0]
    yk = [0, 2, 1, 3]
    cases = [
        (@rkppl(begin a ~ Normal(0, 1); s ~ Exponential(1); y .~ Normal.(a .+ x, s) end), (; x), yc),
        (@rkppl(begin a ~ Normal(0, 1); s ~ Exponential(1); y .~ LogNormal.(a .+ x, s) end), (; x), yc),
        (@rkppl(begin a ~ Normal(0, 1); s ~ Exponential(1); y .~ StudentT.(4.0, a .+ x, s) end), (; x), yc),
        (@rkppl(begin a ~ Normal(0, 1); y .~ Poisson.(exp.(a .+ x)) end), (; x), yk),
        (@rkppl(begin a ~ Normal(0, 1); y .~ Bernoulli.(logistic.(a .+ x)) end), (; x), [0, 1, 1, 0]),
        (@rkppl(begin a ~ Normal(0, 1); y .~ Bernoulli.(normcdf.(a .+ x)) end), (; x), [0, 1, 1, 0]),
        (@rkppl(begin a ~ Normal(0, 1); y .~ Binomial.(n, logistic.(a .+ x)) end), (; x, n), yk),
        (@rkppl(begin p ~ Beta(2, 2); y .~ Binomial.(n, p) end), (; n), yk),
        (@rkppl(begin a ~ Normal(0, 1); phi ~ Exponential(1)
            y .~ NegativeBinomial2.(exp.(a .+ x), phi) end), (; x), yk),
        (@rkppl(begin r ~ Exponential(1); p ~ Beta(2, 2); y .~ NegativeBinomial.(r, p) end),
            NamedTuple(), yk),
        (@rkppl(begin a ~ Normal(0, 1); k ~ Exponential(1)
            y .~ Gamma.(k, exp.(a .+ x) ./ k) end), (; x), yc),
        (@rkppl(begin k ~ Exponential(1); t ~ Exponential(1); y .~ Gamma.(k, t .* (1 .+ x)) end),
            (; x), yc),
        (@rkppl(begin a ~ Normal(0, 1); k ~ Exponential(1); y .~ Weibull.(k, exp.(a .+ x)) end),
            (; x), yc),
        (@rkppl(begin a ~ Normal(0, 1); y .~ Exponential.(exp.(a .+ x)) end), (; x), yc),
        (@rkppl(begin a ~ Normal(0, 1); z ~ Beta(2, 2)
            y .~ ZeroInflatedPoisson.(exp.(a .+ x), z) end), (; x), yk),
        (@rkppl(begin a ~ Normal(0, 1); z ~ Beta(2, 2)
            y .~ ZeroInflatedBinomial.(n, logistic.(a .+ x), z) end), (; x, n), yk),
        (@rkppl(begin m ~ Beta(2, 2); phi ~ Exponential(1); y .~ BetaBinomial2.(n, m, phi) end),
            (; n), yk),
        # Evidence: the clamp law and the renormalized law, continuous and discrete.
        (@rkppl(begin a ~ Normal(0, 1); s ~ Exponential(1)
            y .~ censored.(Normal.(a .+ x, s), lloq, Inf) end), (; x, lloq), yc),
        (@rkppl(begin a ~ Normal(0, 1); s ~ Exponential(1)
            y .~ censored.(Normal.(a .+ x, s), -Inf, 3.0) end), (; x), [1.2, 0.5, 2.5, 3.0]),
        (@rkppl(begin a ~ Normal(0, 1); s ~ Exponential(1)
            y .~ truncated.(Normal.(a .+ x, s), 0.0, 5.0) end), (; x), yc),
        (@rkppl(begin a ~ Normal(0, 1); y .~ censored.(Poisson.(exp.(a .+ x)), 0, 2) end),
            (; x), [0, 2, 1, 2]),
        (@rkppl(begin a ~ Normal(0, 1); k ~ Exponential(1)
            y .~ truncated.(Gamma.(k, exp.(a .+ x) ./ k), 0.1, 10.0) end), (; x), yc),
        # The reporter's shape: an additive-proportional scale reading its location.
        (@rkppl(begin a ~ Normal(0, 1); b ~ Normal(0, 1); add ~ Exponential(1); prop ~ Exponential(1)
            mu = exp.(a .+ b .* x); sd = hypot.(add, mu .* prop)
            y .~ censored.(Normal.(mu, sd), lloq, Inf) end), (; x, lloq), yc),
        # A shared location: each entry's law, broadcast over the response.
        (@rkppl(begin mu ~ Normal(0, 1); s ~ Exponential(1); y .~ Normal.(mu, s) end),
            NamedTuple(), yc),
    ]
    for (model, data, y) in cases
        @test _pd_laws_match(model, data, y)
    end
    # A matrix response keeps its shape.
    X = [0.0 1.0; 2.0 3.0; 4.0 5.0]
    @test _pd_laws_match(@rkppl(begin a ~ Normal(0, 1); s ~ Exponential(1)
        y .~ Normal.(a .+ X, s) end), (; X), X .+ 0.1)
end

@testset "predictive draws: draws come from those laws" begin
    x, lloq = _pd_x, _pd_lloq
    y = [1.2, 0.5, 2.5, 4.0]
    m = @rkppl begin
        a ~ Normal(0, 1); b ~ Normal(0, 1); add ~ Exponential(1); prop ~ Exponential(1)
        mu = exp.(a .+ b .* x); sd = hypot.(add, mu .* prop)
        y .~ censored.(Normal.(mu, sd), lloq, Inf)
    end
    fit = m(; x, lloq) | (; y)
    built = build_kernel(fit)
    q = prepare_predictive(built, fit)
    u = _pd_u(built, 0.0)
    laws = predictive_laws(q, u).y
    # Draws are `rand(rng, law)` in entry order, so a seed reproduces them.
    rng = Xoshiro(11)
    expected = [rand(rng, laws[i]) for i in eachindex(laws)]
    @test q(u, Xoshiro(11)).y == expected
    @test q(u, Xoshiro(11)).y == q(u, Xoshiro(11)).y
    @test q(u, Xoshiro(11)).y != q(u, Xoshiro(12)).y
    out = q(u, Xoshiro(3))
    fill!(out.y, NaN)
    @test predictive_draws!(out, q, u, Xoshiro(11)) === out
    @test out.y == expected
    # The clamp law's atom at the lower limit has the base law's mass below it.
    rng = Xoshiro(5)
    draws = reduce(hcat, (q(u, rng).y for _ in 1:20_000))
    for i in eachindex(laws)
        p = cdf(laws[i].uncensored, lloq)
        @test all(>=(lloq), draws[i, :])
        @test abs(mean(draws[i, :] .== lloq) - p) < 5 * sqrt(p * (1 - p) / 20_000)
    end
    # Changed parameters give other laws, and other draws.
    u2 = _pd_u(built, 0.3)
    @test predictive_laws(q, u2).y[1] != laws[1]
    # Truncation keeps draws inside the bounds; counts keep their mean.
    mt = @rkppl begin
        a ~ Normal(0, 1); s ~ Exponential(1)
        y .~ truncated.(Normal.(a .+ x, s), 0.0, 5.0)
    end
    fitt = mt(; x) | (; y)
    builtt = build_kernel(fitt)
    qt, ut = prepare_predictive(builtt, fitt), _pd_u(builtt, 0.0)
    rng = Xoshiro(4)
    dt = reduce(hcat, (qt(ut, rng).y for _ in 1:2000))
    @test all(d -> 0.0 <= d <= 5.0, dt)
    mp = @rkppl begin a ~ Normal(0, 1); y .~ Poisson.(exp.(a .+ x)) end
    fitp = mp(; x) | (; y = [0, 2, 1, 3])
    builtp = build_kernel(fitp)
    qp = prepare_predictive(builtp, fitp)
    up = _pd_u(builtp, 0.0)
    rng = Xoshiro(9)
    dp = reduce(hcat, (qp(up, rng).y for _ in 1:20_000))
    @test eltype(dp) <: Integer
    for i in 1:4
        rate = exp(up[1] + x[i])
        @test abs(mean(dp[i, :]) - rate) < 5 * sqrt(rate / 20_000)
    end
end

@testset "predictive draws: prediction designs without the response" begin
    x, lloq = _pd_x, _pd_lloq
    # A flat response: draws take the design's rows.
    m = @rkppl begin
        a ~ Normal(0, 1); b ~ Normal(0, 1); add ~ Exponential(1); prop ~ Exponential(1)
        mu = exp.(a .+ b .* x); sd = hypot.(add, mu .* prop)
        y .~ censored.(Normal.(mu, sd), lloq, Inf)
    end
    fit = m(; x, lloq) | (; y = [1.2, 0.5, 2.5, 4.0])
    built = build_kernel(fit)
    u = _pd_u(built, 0.0)
    xnew = [0.0, 3.0, 5.0]
    design = bind_data(fit, (; x = xnew, lloq))
    qd = prepare_predictive(built, design)
    laws = predictive_laws(qd, u).y
    @test size(laws) == (3,)
    for i in 1:3
        mu = exp(u[1] + u[2] * xnew[i])
        @test _pd_same(laws[i], mu, hypot(exp(u[3]), mu * exp(u[4])), lloq)
    end
    @test length(qd(u, Xoshiro(1)).y) == 3
    # A scalar cell over data, its scale from a helper function.
    mc = @rkppl begin
        a ~ Normal(0, 1); b ~ Normal(0, 1); add ~ Exponential(1); prop ~ Exponential(1)
        @plate for i in eachindex(x)
            loc = exp(a + b * x[i])
            y[i] ~ censored(Normal(loc, _pd_addprop(loc, add, prop)), lloq, Inf)
        end
    end
    fitc = mc(; x, lloq) | (; y = [1.2, 0.5, 2.5, 4.0])
    builtc = build_kernel(fitc)
    @test _pd_laws_match(mc, (; x, lloq), [1.2, 0.5, 2.5, 4.0])
    qc = prepare_predictive(builtc, bind_data(fitc, (; x = xnew, lloq)))
    lc = predictive_laws(qc, u).y
    @test size(lc) == (3,)
    mu2 = exp(u[1] + 3u[2])
    @test _pd_same(lc[2], mu2, _pd_addprop(mu2, exp(u[3]), exp(u[4])), lloq)
    # One array per index, an empty index included: each index's draws take
    # that index's operands' shape.
    mg = @rkppl begin
        a ~ Normal(0, 1); b ~ Normal(0, 1); add ~ Exponential(1); prop ~ Exponential(1)
        @plate for i in eachindex(x)
            mi = exp.(a .+ b .* x[i])
            si = hypot.(add, mi .* prop)
            y[i] .~ censored.(Normal.(mi, si), lloq, Inf)
        end
    end
    xg = [[0.1, 0.5], [1.0, 2.0, 3.0], Float64[]]
    yg = [[1.2, 0.5], [2.5, 4.0, 0.5], Float64[]]
    fitg = mg(; x = xg, lloq) | (; y = yg)
    builtg = build_kernel(fitg)
    qg = prepare_predictive(builtg, fitg)
    lg = predictive_laws(qg, u).y
    pwg = Base.invokelatest(prepare_query(builtg, fitg, :pointwise), u).y
    @test length.(lg) == length.(yg)
    @test all(all(logpdf(lg[g][j], yg[g][j]) ≈ pwg[g][j] for j in eachindex(yg[g]))
        for g in eachindex(yg))
    xd = [[0.0], Float64[], [3.0, 4.0, 5.0, 6.0]]
    designg = bind_data(fitg, (; x = xd, lloq))
    draws = prepare_predictive(builtg, designg)(u, Xoshiro(2)).y
    @test length.(draws) == length.(xd)
    @test all(d -> all(>=(lloq), d), draws)
    # The same plumbing lifts `:observations` on such a design.
    obs = Base.invokelatest(prepare_query(builtg, designg, :observations), u).y
    @test length.(obs.location) == length.(xd)
    @test obs.location[3] ≈ exp.(u[1] .+ u[2] .* xd[3])
    # The density still needs the response.
    @test_throws ContractValidationError prepare_query(builtg, designg, :pointwise)
end

@testset "predictive draws: sampler, layout and data unchanged" begin
    x = copy(_pd_x)
    y = [1.2, 0.5, 2.5, 4.0]
    xs, ys = copy(x), copy(y)
    m = @rkppl begin a ~ Normal(0, 1); s ~ Exponential(1); y .~ Normal.(a .+ x, s) end
    fit = m(; x) | (; y)
    built = build_kernel(fit)
    @test coordinate_names(built.layout) == [:a, :s]
    sampler = prepare_query(built, fit, :sampler)
    q = prepare_predictive(built, fit)
    for u in ([0.1, -0.2], [-0.4, 0.3])
        a, s = u[1], exp(u[2])
        oracle = logpdf(Normal(0, 1), a) + logpdf(Exponential(1), s) + u[2] +
            sum(logpdf.(Normal.(a .+ x, s), y))
        @test Base.invokelatest(sampler, u) ≈ oracle
        q(u, Xoshiro(1))
        @test Base.invokelatest(sampler, u) ≈ oracle
    end
    @test x == xs && y == ys
    # A number bound to a response is one observation, as in `:pointwise`.
    fitn = m(; x = 0.4) | (; y = 0.3)
    builtn = build_kernel(fitn)
    qn = prepare_predictive(builtn, fitn)
    @test predictive_laws(qn, [0.1, -0.2]).y == [Normal(0.5, exp(-0.2))]
    @test qn([0.1, -0.2], Xoshiro(1)).y isa Vector{Float64}
    @test length(Base.invokelatest(prepare_query(builtn, fitn, :pointwise), [0.1, -0.2]).y) == 1
    # An empty response draws nothing.
    mr = @rkppl begin a ~ Normal(0, 1); s ~ Exponential(1); y .~ Normal.(a .+ x, s) end
    fite = mr(; x = Float64[]) | (; y = Float64[])
    builte = build_kernel(fite)
    @test prepare_predictive(builte, fite)([0.1, -0.2], Xoshiro(1)).y == Float64[]
end

@testset "predictive draws: printed source replays" begin
    x, lloq = _pd_x, _pd_lloq
    m = @rkppl begin
        a ~ Normal(0, 1); s ~ Exponential(1)
        y .~ censored.(Normal.(a .+ x, s), lloq, Inf)
    end
    fit = m(; x, lloq) | (; y = [1.2, 0.5, 2.5, 4.0])
    built = build_kernel(fit)
    source = kernel_expr(fit, built.layout)
    replayed = ReactiveKernelsPPL._eval_kernel_def(source)
    q = prepare_predictive((; spec = replayed, layout = built.layout), fit)
    u = [0.1, -0.2]
    @test q(u, Xoshiro(2)).y == prepare_predictive(built, fit)(u, Xoshiro(2)).y
end

@testset "predictive draws: what is refused, and what is not built yet" begin
    x = _pd_x
    y = [1.2, 0.5, 2.5, 4.0]
    w = [1.0, 2.0, 1.0, 0.5]
    # refused: a weighted likelihood and an interval event define no single
    # draw (design decision `05ameuz`).
    for (model, data, yy) in (
            (@rkppl(begin a ~ Normal(0, 1); s ~ Exponential(1)
                y .~ weighted.(Normal.(a .+ x, s), w) end), (; x, w), y),
            (@rkppl(begin a ~ Normal(0, 1); s ~ Exponential(1)
                y .~ interval_censored.(Normal.(a .+ x, s), y .+ 1.0) end), (; x), y))
        fit = model(; data...) | (; y = yy)
        built = build_kernel(fit)
        @test_throws ContractValidationError prepare_predictive(built, fit)
    end
    # Not built yet (capability gaps; follow-ups of the landing decision):
    # laws that are not one RK distribution object's density, caller-owned
    # sampling laws, and scalar `~` observations.
    yk = [0, 2, 1, 3]
    gaps = [
        (@rkppl(begin a ~ Normal(0, 1); b ~ Exponential(1); y .~ Beta.(exp.(a .+ x), b) end),
            (; x), [0.2, 0.5, 0.7, 0.9]),
        (@rkppl(begin a ~ Normal(0, 1); z ~ Beta(2, 2); y .~ HurdlePoisson.(exp.(a .+ x), z) end),
            (; x), yk),
        (@rkppl(begin s ~ Exponential(1)
            y .~ MixtureModel.(vcat.(Normal.(x, s), Normal.(0, s)), Ref([0.4, 0.6])) end), (; x), y),
        (@rkppl(begin a ~ Normal(0, 1); y .~ LogDensity.(_pd_nl, a) end), NamedTuple(), y),
        (@rkppl(begin a ~ Normal(0, 1); y ~ Normal(a, 1.0) end), NamedTuple(), 0.3),
    ]
    for (model, data, yy) in gaps
        @test_broken _pd_draws(model, data, yy, Xoshiro(1)) isa NamedTuple
    end
end
