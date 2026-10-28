using Distributions
using DifferentiationInterface
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Test

# One `@plate` (decision 1cmodra, prong `plates`): a cell means one
# iteration of the Julia `for` loop it is written as. These fixtures use
# scalar observations; cell values may be arrays. Shapes come from named data (the range, a data index
# column), never from dims keys. Uses `_canon`
# from test_corpus.jl (included right after it). Data and models are
# synthetic; densities are checked against Distributions.jl.

_pl_canon(plan) = sprint(_canon, plan)
_pl_q(built, bound, preset, u) =
    Base.invokelatest(prepare_query(built, bound, preset), u)

function _pl_bind(ast, data, cols; dims = Dict{Symbol,Int}())
    plan = lower_rkppl(ast, data; conditioned = data)
    bound = bind_data(plan, Dict{Symbol,AbstractVector}(cols); dims)
    return bound, build_kernel(bound)
end

_pl_plate(R, cells...) = Expr(:macrocall, Symbol("@plate"), LineNumberNode(1),
    Expr(:for, Expr(:(=), :i, R), Expr(:block, cells...)))

@testset "plate cells mean one loop iteration" begin
    data = (:y, :x)
    head = (:(a ~ Normal(0, 1)), :(b ~ Normal(0, 2)), :(s ~ Exponential(1)))
    top = lower_rkppl(Expr(:block, head..., :(mu = a .+ b .* x),
        :(y .~ Normal.(mu, s))), data; conditioned = data)
    cols = Dict(:y => [0.2, 1.1, 0.7], :x => [-0.5, 0.3, 1.2])
    function check(plan; truncated = false, scale = 1)
        bound = bind_data(plan, cols)
        built = build_kernel(bound)
        u = unconstrain(built.layout, (; a = 0.4, b = -0.2, s = 1.1))
        oracle = v -> begin
            q = constrain(built.layout, v)
            distributions = Normal.(q.a .+ q.b .* cols[:x], scale * q.s)
            truncated && (distributions = Distributions.truncated.(distributions, 0, 10))
            logpdf(Normal(), q.a) + logpdf(Normal(0, 2), q.b) +
                logpdf(Exponential(), q.s) + log(q.s) +
                sum(logpdf.(distributions, cols[:y]))
        end
        _check_model_math(built, bound, u, oracle)
    end
    check(top)
    # A scalar observation object and its broadcast spelling are the same
    # statement in a loop body (broadcasting over scalars returns the
    # scalar); both are the vectorized observation.
    for obj in (:(Normal(mu[i], s)), :(Normal.(mu[i], s)))
        got = lower_rkppl(Expr(:block, head..., :(mu = a .+ b .* x),
            _pl_plate(:(eachindex(y)), :(y[i] ~ $obj))), data; conditioned = data)
        check(got)
    end
    # Scalar arithmetic in a cell local: `a + b * x[i]` is the cell value;
    # the dotted spelling has the same density and ordinary reverse.
    for rhs in (:(a + b * x[i]), :(a .+ b .* x[i]))
        got = lower_rkppl(Expr(:block, head...,
            _pl_plate(:(eachindex(y)), :(mu = $rhs), :(y[i] ~ Normal(mu, s)))),
            data; conditioned = data)
        check(got)
    end
    # Response wrappers draw once per index too.
    wraptop = lower_rkppl(Expr(:block, head..., :(mu = a .+ b .* x),
        :(y .~ truncated.(Normal.(mu, s), 0, 10))), data; conditioned = data)
    wrap = lower_rkppl(Expr(:block, head..., :(mu = a .+ b .* x),
        _pl_plate(:(eachindex(y)), :(y[i] ~ truncated(Normal(mu[i], s), 0, 10)))),
        data; conditioned = data)
    check(wraptop; truncated = true)
    check(wrap; truncated = true)
    # Operators over scalar-only operands stay scalar (`2 * s` is the same
    # value in every iteration): the cell local is a scalar definition.
    sc = lower_rkppl(Expr(:block, head..., :(mu = a .+ b .* x),
        _pl_plate(:(eachindex(y)), :(sd = 2 * s), :(y[i] ~ Normal(mu[i], sd)))),
        data; conditioned = data)
    sctop = lower_rkppl(Expr(:block, head..., :(mu = a .+ b .* x),
        :(sd = 2 * s), :(y .~ Normal.(mu, sd))), data; conditioned = data)
    check(sctop; scale = 2)
    check(sc; scale = 2)
end

@testset "plate gathers through a data index column" begin
    data = (:y, :g)
    head = (:(c[levels(g)] .~ Normal.(0, 2)), :(s ~ Exponential(1)))
    top = lower_rkppl(Expr(:block, head..., :(y .~ Normal.(c[g], s))), data; conditioned = data)
    got = lower_rkppl(Expr(:block, head...,
        _pl_plate(:(eachindex(y)), :(y[i] ~ Normal(c[g[i]], s)))), data; conditioned = data)
    # capability: a data-index alias is an ordinary value (P8 1cmodra; todo `15lq8iu`).
    aliased = lower_rkppl(Expr(:block, head..., :(h = g),
        _pl_plate(:(eachindex(y)), :(y[i] ~ Normal(c[h[i]], s)))), data; conditioned = data)
    # A retained plate and a vectorized gather may have different plans;
    # both, including the index alias, must preserve the authored density.
    cols = Dict(:y => [-0.1, 0.8, 0.2], :g => [1, 2, 1])
    for plan in (top, got, aliased)
        bound = bind_data(plan, cols)
        built = build_kernel(bound)
        u = unconstrain(built.layout, (; s = 1.1, c = [-0.2, 0.4]))
        oracle = v -> begin
            q = constrain(built.layout, v)
            sum(logpdf.(Normal(0, 2), q.c)) + logpdf(Exponential(), q.s) +
                log(q.s) + sum(logpdf.(Normal.(q.c[cols[:g]], q.s), cols[:y]))
        end
        _check_model_math(built, bound, u, oracle)
    end
    # Other cross-index reads stay refused.
    # refused: reads `g[i - 1]`, out of bounds at i = 1 in the Julia loop (P3)
    @test_throws "cross-index reads" lower_rkppl(Expr(:block, head...,
        _pl_plate(:(eachindex(y)), :(y[i] ~ Normal(c[g[i - 1]], s)))), data; conditioned = data)
end

@testset "responses observe their own rows" begin
    # A statement broadcasts over the columns it reads (standard Julia), so
    # two responses may observe different rows; n_obs is their total.
    data = (:k1, :n1, :k2, :n2)
    head = (:(theta1 ~ Beta(1.0, 1.0)), :(theta2 ~ Beta(1.0, 1.0)))
    plates = Expr(:block, head...,
        _pl_plate(:(eachindex(k1)), :(k1[i] ~ Binomial(n1[i], theta1))),
        _pl_plate(:(eachindex(k2)), :(k2[i] ~ Binomial(n2[i], theta2))))
    twin = Expr(:block, head..., :(k1 .~ Binomial.(n1, theta1)),
        :(k2 .~ Binomial.(n2, theta2)))
    cols = Dict{Symbol,AbstractVector}(:k1 => [3, 1, 4], :n1 => [5, 5, 6],
        :k2 => [0, 2, 2, 1, 3], :n2 => [4, 4, 3, 2, 5])
    bound, built = _pl_bind(plates, data, cols)
    @test bound.n_obs == 8
    u = [0.3, -0.4]
    nt = constrain(built.layout, u)
    @test _pl_q(built, bound, :likelihood, u) ≈
        sum(logpdf.(Binomial.(cols[:n1], nt.theta1), cols[:k1])) +
        sum(logpdf.(Binomial.(cols[:n2], nt.theta2), cols[:k2])) rtol = 1e-12
    for ast in (plates, twin)
        bnd, model = _pl_bind(ast, data, cols)
        oracle = v -> begin
            q = constrain(model.layout, v)
            sum(logpdf.(Binomial.(cols[:n1], q.theta1), cols[:k1])) +
                sum(logpdf.(Binomial.(cols[:n2], q.theta2), cols[:k2])) +
                logpdf(Beta(1, 1), q.theta1) + logpdf(Beta(1, 1), q.theta2) +
                log(q.theta1 * (1 - q.theta1)) + log(q.theta2 * (1 - q.theta2))
        end
        _check_model_math(model, bnd, u, oracle)
    end
    # refused: explicit indexing must cover the authored response range
    # (standard-Julia semantics, language principle 3).
    bad = merge(cols, Dict{Symbol,AbstractVector}(:n2 => [4, 4, 3]))
    @test_throws "indexed operand n2 does not cover the authored response range" _pl_bind(plates, data,
        bad)
    # refused: responses reading a common column observe one axis, so a
    # shared covariate cannot serve rows of two lengths (principle 3).
    shared = Expr(:block, :(b ~ Normal(0, 1)), :(s ~ Exponential(1)),
        :(y1 .~ Normal.(b .* x, s)), :(y2 .~ Normal.(b .* x, s)))
    @test_throws "column length 4 ≠ the 3 rows of y2" bind_data(
        lower_rkppl(shared, (:y1, :y2, :x); conditioned = (:y1, :y2, :x)), Dict{Symbol,AbstractVector}(
            :y1 => [0.1, 0.2, 0.3, 0.4], :y2 => [0.5, 0.6, 0.7],
            :x => [1.0, 2.0, 3.0, 4.0]))
    # A latent plate owns its authored axis beside an unrelated response.
    lat = Expr(:block, :(tau ~ Exponential(1)), :(s ~ Exponential(1)),
        :(m ~ Normal(0, 1)),
        _pl_plate(:(eachindex(y1)), :(theta[i] ~ Normal(0, tau)),
            :(y1[i] ~ Normal(theta[i], s))),
        _pl_plate(:(eachindex(y2)), :(y2[i] ~ Normal(m * x2[i], s))))
    latplan = lower_rkppl(lat, (:y1, :y2, :x2); conditioned = (:y1, :y2, :x2))
    cols = Dict{Symbol,AbstractVector}(:y1 => [0.1, 0.2, 0.3],
        :y2 => [0.4, 0.5], :x2 => [1.0, 2.0])
    bound = bind_data(latplan, cols)
    built = build_kernel(bound)
    u = collect(range(-0.3, 0.4; length = built.layout.total))
    nt = constrain(built.layout, u)
    @test bound.n_obs == 5
    @test length(nt.theta) == 3
    @test _pl_q(built, bound, :likelihood, u) ≈
        sum(logpdf.(Normal.(nt.theta, nt.s), cols[:y1])) +
        sum(logpdf.(Normal.(nt.m .* cols[:x2], nt.s), cols[:y2]))
end


@testset "a latent plate's size is its range" begin
    # `eachindex(v)` iterates `v`: the latent has one cell per entry of `v`.
    # An independent iterator has two entries beside four observations.
    cols = Dict{Symbol,AbstractVector}(:x => [0.2, 0.4], :y => [0.1, -0.3, 0.2, 0.5])
    baseline = quote
        a ~ Normal(0, 1)
        y .~ Normal.(a, 1)
    end
    ast = Expr(:block, baseline.args...,
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(1),
            Expr(:for, :(i = eachindex(x)), Expr(:block,
                :(eta[i] ~ Normal(0, 1))))))
    data = (:x, :y)
    plan = lower_rkppl(ast, data; conditioned = data)
    @test only(plan.plate_parameters).range == :(eachindex(x))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    u = [0.1cos(i) for i in 1:built.layout.total]
    q = constrain(built.layout, u)
    @test length(q.eta) == length(cols[:x]) == 2
    @test bound.n_obs == length(cols[:y]) == 4
    basebound, basebuilt = _pl_bind(baseline, (:y,), Dict(:y => cols[:y]))
    @test built.layout.total == basebuilt.layout.total + 2
    baseu = unconstrain(basebuilt.layout, (; a = q.a))
    @test _pl_q(built, bound, :likelihood, u) ≈ _pl_q(basebuilt, basebound, :likelihood, baseu)
    @test _pl_q(built, bound, :prior, u) ≈ _pl_q(basebuilt, basebound, :prior, baseu) +
        sum(logpdf.(Normal(), q.eta))
    _check_gradient(built.spec, bound, u)
    # Over the observation axis it binds with one cell per row.
    ast = Expr(:block, :(tau ~ Exponential(1)), :(s ~ Exponential(1)),
        _pl_plate(:(eachindex(y)), :(theta[i] ~ Normal(0, tau)),
            :(y[i] ~ Normal(theta[i], s))))
    plan = lower_rkppl(ast, (:y,); conditioned = (:y,))
    @test only(plan.plate_parameters).range == :(eachindex(y))
    ok = bind_data(plan, Dict{Symbol,AbstractVector}(:y => [0.1, 0.4, -0.2]))
    @test assign_layout(ok).total == 2 + 3
end


@testset "plans without a kernel refuse dims keys" begin
    plan = lower_rkppl(Expr(:block, :(a ~ Normal(0, 1)),
        :(b ~ Normal(0, 1)), :(s ~ Exponential(1)),
        _pl_plate(:(eachindex(y)), :(y[i] ~ Normal(a + b * x[i], s)))),
        (:y, :x); conditioned = (:y, :x))
    # refused: dims keys on a plan with no kernel (wrong data)
    @test_throws "not consumed" bind_data(plan,
        Dict{Symbol,AbstractVector}(:y => [0.1, 0.2], :x => [1.0, 2.0]);
        dims = Dict{Symbol,Int}(:nsub => 2, :whatever_typo => 3))
end
