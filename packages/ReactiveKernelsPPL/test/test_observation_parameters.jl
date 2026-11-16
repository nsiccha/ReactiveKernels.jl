using DifferentiationInterface, Distributions, Enzyme, ReactiveKernels, ReactiveKernelsPPL, Test

const _OP_MODEL = @rkppl begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    s ~ Exponential(1)
    @plate for i in eachindex(x)
        y[i] ~ BernoulliLogit(a + b * x[i])
        t[i] ~ Exponential(s * exp(b * x[i]))
        z[i] ~ Normal(a + b * x[i], s)
    end
end

_op_close(a::NamedTuple, b::NamedTuple) = keys(a) == keys(b) && all(_op_close.(values(a), values(b)))
_op_close(a::AbstractArray, b::AbstractArray) = size(a) == size(b) && all(_op_close.(a, b))
_op_close(a::Number, b::Number) = isapprox(a, b; rtol = 1e-11, atol = 1e-12)

@testset "named observation parameters: mixed in-cell laws" begin
    x = [-1.0, 0.2, 1.4]
    fit = _OP_MODEL(; x) | (; y = Bool[0, 1, 1], t = [0.4, 1.0, 2.0], z = [0.1, 0.3, 0.8])
    built = build_kernel(fit)
    u = unconstrain(built.layout, (; a = 0.3, b = -0.4, s = 1.2))
    q = prepare_query(built, fit, :observations)
    obs = Base.invokelatest(q, u)
    eta = 0.3 .- 0.4 .* x
    p = 1 ./ (1 .+ exp.(-eta))
    @test keys(obs) == (:y, :t, :z)
    @test obs.y.probability ≈ p
    @test obs.y.logit ≈ eta
    @test obs.t.scale ≈ 1.2 .* exp.(-0.4 .* x)
    @test obs.z.location ≈ eta
    @test obs.z.scale ≈ 1.2
    @test obs.y.probability isa Vector{Float64}
    pw = Base.invokelatest(prepare_query(built, fit, :pointwise), u)
    @test pw.y ≈ logpdf.(Bernoulli.(p), Bool[0, 1, 1])
    @test pw.t ≈ logpdf.(Exponential.(obs.t.scale), [0.4, 1.0, 2.0])

    direction = zeros(length(u))
    direction[findfirst(==(:b), coordinate_names(built.layout))] = 1.0
    ad = prepare_query_ad(prepare_ad_pushforward, built, fit, :observations,
        AutoEnzyme(; mode = Enzyme.Forward), (direction,), u)
    value, (tangent,) = ad_value_and_pushforward(ad, (direction,), u)
    @test _op_close(value, obs)
    @test tangent.y.probability ≈ p .* (1 .- p) .* x
    @test tangent.y.logit ≈ x
    @test tangent.t.scale ≈ obs.t.scale .* x

    seed = (y = (probability = ones(length(x)), logit = zeros(length(x))),
        t = (scale = ones(length(x)),), z = (location = zeros(length(x)), scale = 0.0))
    reverse = prepare_query_ad(prepare_ad_pullback, built, fit, :observations,
        AutoEnzyme(; mode = Enzyme.Reverse), seed, u)
    reverse_value, gradient = ad_value_and_pullback(reverse, seed, u)
    expected = zeros(length(u))
    for (name, value) in ((:a, sum(p .* (1 .- p))),
            (:b, sum((p .* (1 .- p) .+ obs.t.scale) .* x)), (:s, sum(obs.t.scale)))
        expected[findfirst(==(name), coordinate_names(built.layout))] = value
    end
    @test _op_close(reverse_value, obs)
    @test gradient ≈ expected

    # New design values and lengths use the same scientific declaration;
    # response data are unnecessary for this conditional-parameter view.
    for xnew in ([-2.0, 0.0, 0.4, 1.1, 3.0], Float64[])
        design = bind_data(fit, (; x = xnew))
        fresh = prepare_query(built, design, :observations)
        got = Base.invokelatest(fresh, u)
        @test got.y.probability ≈ 1 ./ (1 .+ exp.(-(0.3 .- 0.4 .* xnew)))
        @test got.t.scale ≈ 1.2 .* exp.(-0.4 .* xnew)
        structure(query) = [(e.kind, e.depth) for e in recipe_inventory(query) if e.kind !== :ordinary]
        @test structure(fresh) == structure(q)
        emitted = ReactiveKernels.js_source(fresh; types = (; unconstrained = Vector{Float64}), name = :design)
        js = only(ReactiveKernels.js_run(emitted, :design, [(u,)]))
        @test js.error === nothing
        js.error === nothing && @test _op_close(js.value, got)
    end
    # Public workflow selector and ordinary printed generated source.
    @test workflow_wants(:observations) === :_ppl_observations
    replay = Core.eval(@__MODULE__, :(@kernel $(Meta.parse(string(kernel_expr(fit, built.layout))))))
    replay_query = prepare_query((; spec = replay, layout = built.layout), fit, :observations)
    @test _op_close(Base.invokelatest(replay_query, u), obs)
    source = ReactiveKernels.js_source(q; types = (; unconstrained = Vector{Float64}), name = :observations)
    cases = [(u,), (u .+ 0.2,), (u .- 0.3,)]
    for (args, got) in zip(cases, ReactiveKernels.js_run(source, :observations, cases))
        @test got.error === nothing
        got.error === nothing && @test _op_close(got.value, Base.invokelatest(q, args...))
    end
end

@testset "named observation parameters: flat, shared, ranged and ragged" begin
    shared = @rkppl begin
        p ~ Beta(2, 2); s ~ Exponential(1)
        y .~ Bernoulli.(p)
        t .~ Exponential.(s)
    end
    fit = shared() | (; y = Bool[1, 0, 1], t = [0.2, 0.4, 0.8])
    built = build_kernel(fit)
    u = unconstrain(built.layout, (; p = 0.4, s = 0.7))
    query = prepare_query(built, fit, :observations)
    obs = Base.invokelatest(query, u)
    @test obs.y.probability ≈ 0.4
    @test obs.y.logit ≈ log(0.4 / 0.6)
    @test obs.t.scale ≈ 0.7
    seed = (y = (probability = 1.0, logit = 1.0), t = (scale = 1.0,))
    reverse = prepare_query_ad(prepare_ad_pullback, built, fit, :observations,
        AutoEnzyme(; mode = Enzyme.Reverse), seed, u)
    gradient = ad_pullback(reverse, seed, u)
    @test gradient[findfirst(==(:p), coordinate_names(built.layout))] ≈ 1.24
    @test gradient[findfirst(==(:s), coordinate_names(built.layout))] ≈ 0.7
    source = ReactiveKernels.js_source(query; types = (; unconstrained = Vector{Float64}), name = :shared)
    js = only(ReactiveKernels.js_run(source, :shared, [(u,)]))
    @test js.error === nothing
    js.error === nothing && @test _op_close(js.value, obs)

    for law in (:(Bernoulli.(logistic.(a .+ x))), :(Bernoulli.(normcdf.(a .+ x))),
            :(Bernoulli.(1 .- exp.(-exp.(a .+ x)))))
        plan = lower_rkppl(quote a ~ Normal(0, 1); y .~ $law end,
            (:x, :y); conditioned = (:y,))
        bound = bind_data(plan, Dict(:x => [-1.0, 0.0, 2.0], :y => Bool[0, 1, 1]))
        b = build_kernel(bound)
        o = Base.invokelatest(prepare_query(b, bound, :observations), [0.2])
        pw = Base.invokelatest(prepare_query(b, bound, :pointwise), [0.2])
        @test pw.y ≈ logpdf.(Bernoulli.(o.y.probability), Bool[0, 1, 1])
        @test o.y.logit ≈ log.(o.y.probability) .- log1p.(-o.y.probability)
    end

    ranged = @rkppl begin
        a ~ Normal(0, 1)
        @plate for i in 1:3
            y[i] ~ BernoulliLogit(a + x[i])
            t[i] ~ Exponential(exp(a + x[i]))
        end
    end
    rf = ranged(; x = [-0.5, 0.1, 0.4]) |
        (; y = Bool[1, 0, 1], t = ones(3))
    rb = build_kernel(rf)
    ro = Base.invokelatest(prepare_query(rb, rf, :observations), [0.2])
    @test ro.y.logit ≈ [-0.3, 0.3, 0.6]
    @test ro.t.scale ≈ exp.([-0.3, 0.3, 0.6])

    ragged = @rkppl begin
        a ~ Normal(0, 1); s ~ Exponential(1)
        @plate for i in eachindex(x)
            y[i] .~ BernoulliLogit.(a .+ x[i])
            t[i] .~ Exponential.(s .* exp.(x[i]))
        end
    end
    for x in ([[-0.3, 0.5], Float64[], [1.2]], Vector{Float64}[])
        f = ragged(; x) | (; y = [zeros(Int, length(v)) for v in x], t = [ones(length(v)) for v in x])
        b = build_kernel(f)
        v = unconstrain(b.layout, (; a = 0.2, s = 0.8))
        o = Base.invokelatest(prepare_query(b, f, :observations), v)
        @test _op_close(o.y.logit, [0.2 .+ v for v in x])
        @test _op_close(o.y.probability, [1 ./ (1 .+ exp.(-(0.2 .+ v))) for v in x])
        @test _op_close(o.t.scale, [0.8 .* exp.(v) for v in x])
        design = bind_data(f, (; x))
        query = prepare_query(b, design, :observations)
        @test _op_close(Base.invokelatest(query, v), o)
        source = ReactiveKernels.js_source(query; types = (; unconstrained = Vector{Float64}), name = :ragged)
        js = only(ReactiveKernels.js_run(source, :ragged, [(v,)]))
        @test js.error === nothing
        js.error === nothing && @test _op_close(js.value, o)
    end

    shared_cells = @rkppl begin
        a ~ Normal(0, 1); s ~ Exponential(1)
        @plate for i in eachindex(y)
            y[i] .~ BernoulliLogit.(a)
            t[i] .~ Exponential.(s)
        end
    end
    sf = shared_cells() | (; y = [Bool[0, 1], Bool[]], t = [[0.4, 0.5], Float64[]])
    sb = build_kernel(sf)
    so = Base.invokelatest(prepare_query(sb, sf, :observations),
        unconstrain(sb.layout, (; a = 0.2, s = 0.8)))
    @test so.y.logit ≈ 0.2
    @test so.y.probability ≈ 1 / (1 + exp(-0.2))
    @test so.t.scale ≈ 0.8
end

@testset "named observation parameters: other owned law graphs" begin
    cases = [
        (:(Poisson.(exp.(a .+ x))), (:rate,), (exp.(0.2 .+ [0.1, 0.4]),)),
        (:(Binomial.(3, logistic.(a .+ x))), (:trials, :probability, :logit),
            (3, 1 ./ (1 .+ exp.(-[0.3, 0.6])), [0.3, 0.6])),
        (:(Gamma.(2.0, exp.(a .+ x))), (:shape, :scale), (2.0, exp.([0.3, 0.6]))),
        (:(Gamma.(exp.(a .+ x), 2.0)), (:shape, :scale), (exp.([0.3, 0.6]), 2.0)),
        (:(Weibull.(2.0, exp.(a .+ x))), (:shape, :scale), (2.0, exp.([0.3, 0.6]))),
        (:(NegativeBinomial2.(exp.(a .+ x), 2.0)), (:mean, :dispersion), (exp.([0.3, 0.6]), 2.0)),
        (:(NegativeBinomial.(exp.(a .+ x), 0.6)), (:shape, :probability), (exp.([0.3, 0.6]), 0.6)),
        (:(InverseGaussian.(exp.(a .+ x), 2.0)), (:mean, :shape), (exp.([0.3, 0.6]), 2.0)),
        (:(Beta.(logistic.(a .+ x) .* 2.0, (1 .- logistic.(a .+ x)) .* 2.0)),
            (:alpha, :beta), (2 ./ (1 .+ exp.(-[0.3, 0.6])), 2 ./ (1 .+ exp.([0.3, 0.6])))),
        (:(ZeroInflatedPoisson.(exp.(a .+ x), 0.2)), (:rate, :zero_probability),
            (exp.([0.3, 0.6]), 0.2)),
        (:(ZeroInflatedBinomial.(3, logistic.(a .+ x), 0.2)),
            (:trials, :probability, :zero_probability), (3, 1 ./ (1 .+ exp.(-[0.3, 0.6])), 0.2)),
        (:(BetaBinomial2.(3, logistic.(a .+ x), 2.0)), (:trials, :alpha, :beta),
            (3, 2 ./ (1 .+ exp.(-[0.3, 0.6])), 2 ./ (1 .+ exp.([0.3, 0.6])))),
        (:(censored.(Exponential.(exp.(a .+ x)), 0.5, Inf)), (:scale,), (exp.([0.3, 0.6]),)),
        (:(truncated.(Exponential.(exp.(a .+ x)), 0.5, 3.0)), (:scale,), (exp.([0.3, 0.6]),)),
        (:(weighted.(Exponential.(exp.(a .+ x)), weights)), (:scale,), (exp.([0.3, 0.6]),)),
    ]
    for (law, fields, expected) in cases
        plan = lower_rkppl(quote a ~ Normal(0, 1); y .~ $law end, (:x, :y, :weights); conditioned = (:y,))
        response = fields == (:alpha, :beta) ? [0.3, 0.7] : [1, 2]
        bound = bind_data(plan, Dict(:x => [0.1, 0.4], :y => response, :weights => [0.5, 0.8]))
        built = build_kernel(bound)
        query = prepare_query(built, bound, :observations)
        obs = Base.invokelatest(query, [0.2])
        @test keys(obs.y) == fields
        @test all(_op_close.(values(obs.y), expected))
        source = ReactiveKernels.js_source(query;
            types = (; unconstrained = Vector{Float64}), name = :parameters)
        js = only(ReactiveKernels.js_run(source, :parameters, [([0.2],)]))
        @test js.error === nothing
        js.error === nothing && @test _op_close(js.value, obs)
    end
end
