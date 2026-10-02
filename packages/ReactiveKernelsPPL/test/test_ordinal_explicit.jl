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
# vector, and the shipped `y ~ ordered_logistic(eta)` states the default.
# With the old default values each explicit spelling lowers to the plan the
# implicit form (`OrderedLogistic.(eta)`, minted `y_cutpoints`) lowers to.
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
_oe_lower(ex, data = (:y, :x)) = lower_rkppl(ex, data; mod = @__MODULE__)
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

@testset "explicit cutpoints: twins of the implicit form" begin
    implicit = _oe_canon(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        eta = a .+ b .* x
        y .~ OrderedLogistic.(eta)
    end)
    stated = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        eta = a .+ b .* x
    end
    @test _oe_canon(Expr(:block, stated.args...,
        :(y_cutpoints ~ Ordered(Normal(0, 1), length(levels(y)) - 1)),
        :(y .~ OrderedLogistic.(eta, Ref(y_cutpoints))))) == implicit
    # The shipped stream submodel states the same two statements.
    @test _oe_canon(Expr(:block, stated.args...,
        :(y ~ ordered_logistic(eta)))) == implicit
    # Cumulative Ordinal: an Ordered vector, data-sized.
    @test _oe_canon(quote
        b ~ Normal(0, 1)
        y_thresholds ~ Ordered(Normal(0, 1), length(levels(y)) - 1)
        eta = b .* x
        y .~ Ordinal.(Cumulative(), ProbitLink(), eta, Ref(y_thresholds))
    end) == _oe_canon(quote
        b ~ Normal(0, 1)
        eta = b .* x
        y .~ Ordinal.(Cumulative(), ProbitLink(), eta)
    end)
    # Stopping ratio: unconstrained sized thresholds.
    stopping_implicit = quote
        b ~ Normal(0, 1)
        eta = b .* x
        y .~ Ordinal.(StoppingRatio(), LogitLink(), eta)
    end
    @test _oe_canon(quote
        b ~ Normal(0, 1)
        y_thresholds[1:length(levels(y)) - 1] .~ Normal.(0, 1)
        eta = b .* x
        y .~ Ordinal.(StoppingRatio(), LogitLink(), eta, Ref(y_thresholds))
    end) == _oe_canon(stopping_implicit)
    # A literal length is concrete before bind and the same plan after.
    cols = _oe_cols()
    bound(ex) = sprint(_canon, bind_data(_oe_lower(ex), cols))
    @test bound(Expr(:block, stated.args...,
        :(y_cutpoints ~ Ordered(Normal(0, 1), 2)),
        :(y .~ OrderedLogistic.(eta, Ref(y_cutpoints))))) ==
        bound(quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            eta = a .+ b .* x
            y .~ OrderedLogistic.(eta)
        end)
    @test bound(quote
        b ~ Normal(0, 1)
        y_thresholds[1:2] .~ Normal.(0, 1)
        eta = b .* x
        y .~ Ordinal.(StoppingRatio(), LogitLink(), eta, Ref(y_thresholds))
    end) == bound(stopping_implicit)
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
    # A literal length that disagrees with the bound levels fails at bind.
    long = _oe_lower(quote
        b ~ Normal(0, 2)
        c ~ Ordered(Normal(0, 1), 3)
        y .~ OrderedLogistic.(b .* x, Ref(c))
    end)
    @test_throws ContractValidationError bind_data(long, cols)
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
    # `k` beyond the level count leaves an empty axis: bind refuses it.
    empty = _oe_lower(quote
        a ~ Normal(0, 1)
        w[1:length(levels(g)) - 4] .~ Normal.(0, 1)
        mu = a .+ w[1] .* x
        z .~ Normal.(mu, 1.0)
    end, (:z, :x, :g))
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
    @test occursin("write `Ref(c)`", refused(quote
        c ~ Ordered(Normal(0, 1), 2)
        y .~ OrderedLogistic.(b .* x, c)
    end))
    @test occursin("must be an ordered vector declared", refused(quote
        y .~ OrderedLogistic.(b .* x, Ref(c))
    end))
    @test occursin("must be an ordered vector declared", refused(quote
        c[1:2] .~ Normal.(0, 1)
        y .~ OrderedLogistic.(b .* x, Ref(c))
    end))
    @test occursin("stopping-ratio thresholds are unconstrained", refused(quote
        c ~ Ordered(Normal(0, 1), 2)
        y .~ Ordinal.(StoppingRatio(), LogitLink(), b .* x, Ref(c))
    end))
    @test occursin("one-axis vector", refused(quote
        c[1:2, 1:2] .~ Normal.(0, 1)
        y .~ Ordinal.(StoppingRatio(), LogitLink(), b .* x, Ref(c))
    end))
    @test occursin("one response per cutpoint vector", refused(quote
        c ~ Ordered(Normal(0, 1), 2)
        y .~ OrderedLogistic.(b .* x, Ref(c))
        y2 .~ OrderedLogistic.(b .* x, Ref(c))
    end))
    # The data-sized length names the response the cutpoints serve.
    for n in (:(length(levels(x)) - 1), :(length(levels(y)) - 2),
            :(length(unique(y)) - 1))
        @test occursin("length(levels(y)) - 1", refused(quote
            b ~ Normal(0, 1)
            c ~ Ordered(Normal(0, 1), $n)
            y .~ OrderedLogistic.(b .* x, Ref(c))
        end))
    end
    @test occursin("serves no ordinal response", refused(quote
        a ~ Normal(0, 1)
        c ~ Ordered(Normal(0, 1), length(levels(y)) - 1)
        mu = a .+ c[1] .* x
        y2 .~ Normal.(mu, 1.0)
    end))
    # Nothing reads or consumes it: unused, like an unused simplex.
    @test_throws ContractValidationError _oe_lower(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        k ~ Ordered(Normal(0, 1), 2)
        mu = a .+ b .* x
        y2 .~ Normal.(mu, 1.0)
    end, data)
    # Element prior: a literal Normal.
    for d in (:(Cauchy(0, 1)), :(Normal(0, s)), :(Normal(0, -1)))
        @test occursin("element prior", refused(quote
            b ~ Normal(0, 1)
            s ~ HalfNormal(1)
            c ~ Ordered($d, 2)
            y .~ OrderedLogistic.(b .* x, Ref(c))
        end))
    end
    @test occursin("element prior", refused(quote
        b ~ Normal(0, 1)
        t[1:2] .~ Cauchy.(0, 1)
        y .~ Ordinal.(StoppingRatio(), LogitLink(), b .* x, Ref(t))
    end))
    @test occursin("takes the element distribution and the length",
        refused(quote
            b ~ Normal(0, 1)
            c ~ Ordered(Normal(0, 1))
            y .~ OrderedLogistic.(b .* x, Ref(c))
        end))
    # An ordered vector is a value, never a predictor coefficient.
    @test !isempty(refused(quote
        c ~ Ordered(Normal(0, 1), 2)
        mu = c .* x
        y2 .~ Normal.(mu, 1.0)
    end))
end
