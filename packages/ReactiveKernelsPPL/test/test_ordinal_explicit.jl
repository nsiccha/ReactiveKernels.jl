using DifferentiationInterface
using Distributions
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Test

# Explicit ordinal cutpoints: `c ~ Ordered(Normal(m, s), n)` declares an
# ordered vector (a plain value), `y .~ OrderedLogistic.(eta, Ref(c))` and
# `y .~ Ordinal.(structure, link, eta, Ref(c))` take the cutpoints as an
# argument, stopping-ratio thresholds are a sized `t[1:n] .~ Normal.(m, s)`
# vector.
# Every threshold vector and its prior are declared in the model body.
# The ordered-logistic oracle is Turing.jl's `OrderedLogistic(η, c)`
# definition, P(y = k) = F(c[k] − η) − F(c[k−1] − η), evaluated through
# Distributions.jl CDFs (Distributions.jl itself has no `OrderedLogistic`).
# Helpers `_canon` (test_corpus.jl), `_query` / `_check_gradient`
# (test_generator.jl) come from the earlier includes.

_oe_cols(; y = [1, 2, 3, 2, 1, 3, 3, 2]) = Dict{Symbol,AbstractVector}(
    :y => y, :x => [0.5, -1.0, 1.5, 0.0, -0.5, 1.0, 0.3, -0.2],
    :z => [0.1, 0.2, -0.3, 0.5, 0.0, 1.0, -1.0, 0.4],
    :g => ["b", "a", "c", "a", "b", "d", "a", "b"])
_oe_point(n) = [0.37 * sin(1.3 * i) - 0.2 for i in 1:n]
_oe_lower(ex, data = (:y, :x)) = lower_rkppl(ex, data; mod = @__MODULE__, conditioned = data)
_oe_canon(ex, data = (:y, :x)) = sprint(_canon, _oe_lower(ex, data))

# log P(y = k) for a cumulative model with link CDF `F` (k = 1..length(c)+1).
function _oe_cumulative_lp(F, eta, c, k)
    K = length(c) + 1
    hi = k == K ? 1.0 : F(c[k] - eta)
    lo = k == 1 ? 0.0 : F(c[k-1] - eta)
    return log(hi - lo)
end

# log P(y = k) for a stopping-ratio model: stop at stage k with F(t[k] − η)
# after surviving every earlier stage with 1 − F(t[j] − η).
function _oe_stopping_lp(F, eta, t, k)
    K = length(t) + 1
    lp = sum((log(1 - F(t[j] - eta)) for j in 1:k-1); init = 0.0)
    return k == K ? lp : lp + log(F(t[k] - eta))
end

_oe_logistic(z) = cdf(Logistic(), z)
_oe_probit(z) = cdf(Normal(), z)
_oe_cloglog(z) = 1 - exp(-exp(z))

@testset "explicit cutpoints: declaration order and extent" begin
    stated = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        eta = a .+ b .* x
    end
    declaration = :(y_cutpoints ~ Ordered(Normal(0, 1), length(levels(y)) - 1))
    response = :(y .~ OrderedLogistic.(eta, Ref(y_cutpoints)))
    ordered = Expr(:block, stated.args..., declaration, response)
    @test _oe_canon(ordered) ==
        _oe_canon(Expr(:block, stated.args..., response, declaration))
    cols = _oe_cols()
    bound(ex) = bind_data(_oe_lower(ex), cols)
    symbolic = build_kernel(bound(ordered))
    concrete_ast = Expr(:block, stated.args...,
        :(y_cutpoints ~ Ordered(Normal(0, 1), 2)), response)
    concrete = build_kernel(bound(concrete_ast))
    @test coordinate_names(symbolic.layout) == coordinate_names(concrete.layout)
    u = _oe_point(symbolic.layout.total)
    @test _query(symbolic.spec, bound(ordered), :posterior, u) ≈
        _query(concrete.spec, bound(concrete_ast), :posterior, u)
    stopping = quote
        b ~ Normal(0, 1)
        y_thresholds[1:length(levels(y)) - 1] .~ Normal.(0, 1)
        eta = b .* x
        y .~ Ordinal.(StoppingRatio(), LogitLink(), eta, Ref(y_thresholds))
    end
    stopping_literal = quote
        b ~ Normal(0, 1)
        y_thresholds[1:2] .~ Normal.(0, 1)
        eta = b .* x
        y .~ Ordinal.(StoppingRatio(), LogitLink(), eta, Ref(y_thresholds))
    end
    dynamic = build_kernel(bound(stopping))
    fixed = build_kernel(bound(stopping_literal))
    @test coordinate_names(dynamic.layout) == coordinate_names(fixed.layout)
    u = _oe_point(dynamic.layout.total)
    @test _query(dynamic.spec, bound(stopping), :posterior, u) ≈
        _query(fixed.spec, bound(stopping_literal), :posterior, u)
end

@testset "ordinal responses require their authored threshold priors" begin
    # Refused: USER 1cmodra (vectors) and 0d5a67r remove implied priors.
    @test_throws "requires explicit cutpoints" _oe_lower(quote
        b ~ Normal(0, 1)
        y .~ OrderedLogistic.(b .* x)
    end)
    for structure in (:Cumulative, :StoppingRatio)
        @test_throws "requires explicit thresholds" _oe_lower(quote
            b ~ Normal(0, 1)
            y .~ Ordinal.($structure(), LogitLink(), b .* x)
        end)
    end
end

@testset "explicit cutpoints: OrderedLogistic density" begin
    plan = _oe_lower(quote
        b ~ Normal(0, 2)
        c ~ Ordered(Normal(0.5, 2), 2)
        eta = b .* x
        y .~ OrderedLogistic.(eta, Ref(c))
    end)
    v = only(plan.vector_parameters)
    @test (v.name, v.family, v.size) == (:c, :ordered_normal, 2)
    @test v.args == (arg1 = 0.5, arg2 = 2.0)
    cols = _oe_cols()
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    u = _oe_point(built.layout.total)
    nt = constrain(built.layout, u)
    c, b = nt.c, nt.b
    @test c isa Vector{Float64} && issorted(c)
    lik = sum(_oe_cumulative_lp(_oe_logistic, b * x, c, y)
        for (x, y) in zip(cols[:x], cols[:y]))
    @test _query(built.spec, bound, :likelihood, u) ≈ lik
    @test _query(built.spec, bound, :prior, u) ≈
        logpdf(Normal(0, 2), b) + sum(logpdf.(Normal(0.5, 2), c))
    # The ordered transform's Jacobian: log(c[2] − c[1]).
    @test _query(built.spec, bound, :log_jacobian, u) ≈ log(c[2] - c[1])
    _check_gradient(built.spec, bound, u)
    # Three declared cutpoints state four categories even when the top
    # category is absent from this sample.
    long = _oe_lower(quote
        b ~ Normal(0, 2)
        c ~ Ordered(Normal(0, 1), 3)
        y .~ OrderedLogistic.(b .* x, Ref(c))
    end)
    long_bound = bind_data(long, cols)
    @test only(long_bound.responses).n_levels == 4
    @test only(long_bound.vector_parameters).size == 3
end

@testset "explicit cutpoints: Ordinal cumulative and stopping densities" begin
    cols = _oe_cols()
    for (link, F) in ((:ProbitLink, _oe_probit), (:CloglogLink, _oe_cloglog))
        plan = _oe_lower(quote
            b ~ Normal(0, 2)
            c ~ Ordered(Normal(0, 1), length(levels(y)) - 1)
            eta = b .* x
            y .~ Ordinal.(Cumulative(), $(Expr(:call, link)), eta, Ref(c))
        end)
        bound = bind_data(plan, cols)
        built = build_kernel(bound)
        u = _oe_point(built.layout.total)
        nt = constrain(built.layout, u)
        c, b = nt.c, nt.b
        @test _query(built.spec, bound, :likelihood, u) ≈
            sum(_oe_cumulative_lp(F, b * x, c, y)
                for (x, y) in zip(cols[:x], cols[:y]))
        @test _query(built.spec, bound, :prior, u) ≈
            logpdf(Normal(0, 2), b) + sum(logpdf.(Normal(), c))
    end
    plan = _oe_lower(quote
        b ~ Normal(0, 2)
        t[1:length(levels(y)) - 1] .~ Normal.(-0.5, 1.5)
        eta = b .* x
        y .~ Ordinal.(StoppingRatio(), LogitLink(), eta, Ref(t))
    end)
    v = only(plan.vector_parameters)
    @test (v.name, v.family, v.size) == (:t, :vector_normal, nothing)
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    u = _oe_point(built.layout.total)
    nt = constrain(built.layout, u)
    t, b = nt.t, nt.b
    @test length(t) == 2
    @test _query(built.spec, bound, :likelihood, u) ≈
        sum(_oe_stopping_lp(_oe_logistic, b * x, t, y)
            for (x, y) in zip(cols[:x], cols[:y]))
    @test _query(built.spec, bound, :prior, u) ≈
        logpdf(Normal(0, 2), b) + sum(logpdf.(Normal(-0.5, 1.5), t))
    @test _query(built.spec, bound, :log_jacobian, u) ≈ 0.0
    _check_gradient(built.spec, bound, u)
end

@testset "Ordered vectors are plain values" begin
    cols = _oe_cols()
    # Free-standing, read by position.
    plan = _oe_lower(quote
        a ~ Normal(0, 1)
        k ~ Ordered(Normal(0.5, 2), 3)
        s = exp(k[3] - k[1])
        mu = a .+ k[2] .* x
        z .~ Normal.(mu, s)
    end, (:x, :z))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    u = _oe_point(built.layout.total)
    nt = constrain(built.layout, u)
    k, a = nt.k, nt.a
    @test issorted(k)
    @test _query(built.spec, bound, :posterior, u) ≈
        logpdf(Normal(0, 1), a) + sum(logpdf.(Normal(0.5, 2), k)) +
        log(k[2] - k[1]) + log(k[3] - k[2]) +
        sum(logpdf.(Normal.(a .+ k[2] .* cols[:x], exp(k[3] - k[1])),
            cols[:z]))
    _check_gradient(built.spec, bound, u)
    # Cutpoints read elsewhere in the model.
    plan = _oe_lower(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        c ~ Ordered(Normal(0, 1), 2)
        y .~ OrderedLogistic.(b .* x, Ref(c))
        mu = a .+ c[1] .* x
        z .~ Normal.(mu, 1.0)
    end, (:y, :x, :z))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    u = _oe_point(built.layout.total)
    nt = constrain(built.layout, u)
    c, b, a = nt.c, nt.b, nt.a
    @test _query(built.spec, bound, :posterior, u) ≈
        logpdf(Normal(), a) + logpdf(Normal(), b) + sum(logpdf.(Normal(), c)) +
        log(c[2] - c[1]) +
        sum(_oe_cumulative_lp(_oe_logistic, b * x, c, y)
            for (x, y) in zip(cols[:x], cols[:y])) +
        sum(logpdf.(Normal.(a .+ c[1] .* cols[:x], 1.0), cols[:z]))
end

@testset "array axis 1:length(levels(g)) - k" begin
    cols = _oe_cols()
    plan = _oe_lower(quote
        a ~ Normal(0, 1)
        sigma ~ Exponential(1)
        w[1:length(levels(g)) - 1] .~ Normal.(0, 1)
        mu = a .+ w[3] .* x
        z .~ Normal.(mu, sigma)
    end, (:z, :x, :g))
    @test only(plan.array_parameters).dims == Any[:(length(levels(g)) - 1)]
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    u = _oe_point(built.layout.total)
    nt = constrain(built.layout, u)
    w = nt.w
    @test length(w) == 3  # g has 4 distinct values
    @test _query(built.spec, bound, :posterior, u) ≈
        logpdf(Normal(), nt.a) + logpdf(Exponential(1), nt.sigma) +
        log(nt.sigma) + sum(logpdf.(Normal(), w)) +
        sum(logpdf.(Normal.(nt.a .+ w[3] .* cols[:x], nt.sigma),
            cols[:z]))
    # `k` equal to the level count leaves an empty axis: w[1] has no element.
    empty = _oe_lower(quote
        a ~ Normal(0, 1)
        w[1:length(levels(g)) - 4] .~ Normal.(0, 1)
        mu = a .+ w[1] .* x
        z .~ Normal.(mu, 1.0)
    end, (:z, :x, :g))
    # refused: w[1] indexes an empty bound axis (Julia BoundsError, P3)
    @test_throws ContractValidationError bind_data(empty, cols)
end

@testset "explicit cutpoints: refusals" begin
    data = (:y, :x, :y2)
    refused(ex) = try
        _oe_lower(ex, data)
        ""
    catch e
        e isa SurfaceLoweringError || rethrow()
        sprint(showerror, e)
    end
    # Cutpoints are one shared vector: standard broadcasting needs `Ref`.
    # refused: remaining entries violate broadcast argument shape, strict declarations or constructor arity (P3/P6, 05oe96l)
    @test occursin("write `Ref(c)`", refused(quote
        b ~ Normal(0, 1)
        c ~ Ordered(Normal(0, 1), 2)
        y .~ OrderedLogistic.(b .* x, c)
    end))
    # refused: remaining entries violate broadcast argument shape, strict declarations or constructor arity (P3/P6, 05oe96l)
    @test occursin("must be a one-axis vector declared", refused(quote
        b ~ Normal(0, 1)
        y .~ OrderedLogistic.(b .* x, Ref(c))
    end))
    # capability: cutpoints with an ordinary vector prior; unordered points have -Inf density (10gzbm9 support-links; todo `1qlbn5b`).
    @test (_oe_lower(quote
        b ~ Normal(0, 1)
        c[1:2] .~ Normal.(0, 1)
        y .~ OrderedLogistic.(b .* x, Ref(c))
    end, data); true)
    # capability: an ordered prior is also a valid prior on stopping-ratio thresholds (P3/P8; todo `1qlbn5b`)
    @test (_oe_lower(quote
        b ~ Normal(0, 1)
        c ~ Ordered(Normal(0, 1), 2)
        y .~ Ordinal.(StoppingRatio(), LogitLink(), b .* x, Ref(c))
    end, data); true)
    # refused: remaining entries violate broadcast argument shape, strict declarations or constructor arity (P3/P6, 05oe96l)
    @test occursin("one-axis vector", refused(quote
        b ~ Normal(0, 1)
        c[1:2, 1:2] .~ Normal.(0, 1)
        y .~ Ordinal.(StoppingRatio(), LogitLink(), b .* x, Ref(c))
    end))
    shared = _oe_lower(quote
        b ~ Normal(0, 1)
        c ~ Ordered(Normal(0, 1), 2)
        y .~ OrderedLogistic.(b .* x, Ref(c))
        y2 .~ OrderedLogistic.(b .* x, Ref(c))
    end, data)
    @test length(shared.vector_parameters) == 1
    @test all(r -> r.thresholds === :c, shared.responses)
    # Extents are expressions over actual bound data; independent numerical
    # and invalid-support controls are in test_prior_observation_audit.jl.
    for n in (:(length(levels(x)) - 1), :(length(levels(y)) - 2),
            :(length(unique(y)) - 1))
        # capability: ordinary cutpoint/vector priors, sizes and reuse (P8 1cmodra; 10gzbm9 shared-slots/level-coverage; todo `1qlbn5b`)
        @test (_oe_lower(quote
            b ~ Normal(0, 1)
            c ~ Ordered(Normal(0, 1), $n)
            y .~ OrderedLogistic.(b .* x, Ref(c))
        end, data); true)
    end
    # capability: ordinary cutpoint/vector priors, sizes and reuse (P8 1cmodra; 10gzbm9 shared-slots/level-coverage; todo `1qlbn5b`)
    @test (_oe_lower(quote
        a ~ Normal(0, 1)
        c ~ Ordered(Normal(0, 1), length(levels(y)) - 1)
        mu = a .+ c[1] .* x
        y2 .~ Normal.(mu, 1.0)
    end, data); true)
    # Nothing reads or consumes it: unused, like an unused simplex.
    # capability: an unused declared parameter is a prior-only draw (10gzbm9 degenerate) (todo `1qlbn5b`)
    @test (_oe_lower(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        k ~ Ordered(Normal(0, 1), 2)
        mu = a .+ b .* x
        y2 .~ Normal.(mu, 1.0)
    end, data); true)
    # Element prior: a literal Normal.
    for d in (:(Cauchy(0, 1)), :(Normal(0, s)), :(Normal(0, -1)))
        program = quote
            b ~ Normal(0, 1)
            s ~ HalfNormal(1)
            c ~ Ordered($d, 2)
            y .~ OrderedLogistic.(b .* x, Ref(c))
        end
        if d == :(Normal(0, -1))
            # refused: Normal's scale must be positive (distribution domain, P3).
            @test occursin("element prior", refused(program))
        else
            # capability: non-Normal and sampled-argument ordered priors (P8 1cmodra; todo `0fkd9yk`).
            @test (_oe_lower(program, data); true)
        end
    end
    # capability: ordinary cutpoint/vector priors, sizes and reuse (P8 1cmodra; 10gzbm9 shared-slots/level-coverage; todo `1qlbn5b`)
    @test (_oe_lower(quote
        b ~ Normal(0, 1)
        t[1:2] .~ Cauchy.(0, 1)
        y .~ Ordinal.(StoppingRatio(), LogitLink(), b .* x, Ref(t))
    end, data); true)
    # refused: remaining entries violate broadcast argument shape, strict declarations or constructor arity (P3/P6, 05oe96l)
    @test occursin("takes the element distribution and the length",
        refused(quote
            b ~ Normal(0, 1)
            c ~ Ordered(Normal(0, 1))
            y .~ OrderedLogistic.(b .* x, Ref(c))
        end))
    # refused: the historical x length cannot broadcast with two cutpoints.
    # A matched numerical control retains their prior and ordered transform.
    vector_plan=_oe_lower(quote
        c ~ Ordered(Normal(0, 1), 2)
        mu = c .* x
        y2 .~ Normal.(mu, 1.0)
    end, data)
    vector_data=Dict{Symbol,Any}(_oe_cols())
    vector_data[:y2]=zeros(length(vector_data[:x]))
    vector_bound=bind_data(vector_plan,vector_data)
    vector_built=build_kernel(vector_bound)
    @test_throws DimensionMismatch _query(vector_built.spec,vector_bound,
        :posterior,zeros(vector_built.layout.total))
end
