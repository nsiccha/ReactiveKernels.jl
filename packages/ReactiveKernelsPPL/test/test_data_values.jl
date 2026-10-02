using Distributions
using ReactiveKernels
using ReactiveKernelsPPL
using Test
import DifferentiationInterface
import Enzyme

# Data are values of any shape (user decision 0dejlw1: "data, but also
# parameters. anything can be anything"). A number has no observation axis,
# so it reads as a model-level value — exactly as the definition it stands
# for (`s = 0.7`) — at every entry point: the model call, `@rkppl data`,
# `merge` pins and `lower_rkppl(ast, data)` + `bind_data`. References are
# independent Distributions.jl computations over the constrained draws.
# Self-contained: no helpers from earlier includes.

module DataValuesModels
total(A) = sum(A)
end
const _DV = DataValuesModels

const _DV_X = [0.5, -1.0, 1.5, 0.0, -0.5]
const _DV_Y = [0.3, -0.8, 1.9, 0.2, -0.1]
const _DV_U = [0.2, -0.4]

_dv_built(bound) = build_kernel(bound)
_dv_value(built, bound, preset, u) =
    Base.invokelatest(prepare_query(built, bound, preset), u)
# The likelihood of `y .~ Normal.(a .+ b .* x, sd)` at `u`, against its
# Distributions oracle.
function _dv_check_regression(bound, sd)
    built = _dv_built(bound)
    th = ReactiveKernelsPPL.constrain(built.layout, _DV_U)
    a, b = th.a, th.b
    @test coordinate_names(built.layout) == [:a, :b]
    @test _dv_value(built, bound, :likelihood, _DV_U) ≈
        sum(logpdf.(Normal.(a .+ b .* _DV_X, sd), _DV_Y))
end

_dv_regression() = quote
    a ~ Normal(0, 5)
    b ~ Normal(0, 2)
    mu = a .+ b .* x
    y .~ Normal.(mu, s)
end

function _dv_findiff(f, u; h = cbrt(eps(Float64)))
    g = similar(u, Float64)
    for i in eachindex(u)
        up = copy(u); up[i] += h
        dn = copy(u); dn[i] -= h
        g[i] = (f(up) - f(dn)) / (2h)
    end
    return g
end

@testset "data values: a number reads as a model-level value at every entry point" begin
    m = RKPPLModel(_dv_regression(), _DV)
    # The model call.
    _dv_check_regression(m(; y = _DV_Y, x = _DV_X, s = 0.7), 0.7)
    # `@rkppl data`.
    y, x = _DV_Y, _DV_X
    bound = @rkppl (; y, x, s = 0.7) begin
        a ~ Normal(0, 5)
        b ~ Normal(0, 2)
        mu = a .+ b .* x
        y .~ Normal.(mu, s)
    end
    _dv_check_regression(bound, 0.7)
    # `lower_rkppl(ast, data)` reads each value's shape; the plan then binds
    # any number for `s` without lowering again.
    plan = lower_rkppl(_dv_regression(),
        Dict{Symbol,Any}(:y => y, :x => x, :s => 0.7))
    for sd in (0.7, 1.1)
        _dv_check_regression(bind_data(plan,
            Dict{Symbol,ColumnData}(:y => y, :x => x, :s => sd)), sd)
    end
end

@testset "data values: a pinned number is the definition it replaces" begin
    base = RKPPLModel(quote
        a ~ Normal(0, 5)
        b ~ Normal(0, 2)
        s ~ Exponential(1.0)
        mu = a .+ b .* x
        y .~ Normal.(mu, s)
    end, _DV)
    pinned = Base.merge(base, (; s = 0.7))
    _dv_check_regression(pinned(; y = _DV_Y, x = _DV_X), 0.7)
    # An explicit call keyword wins over the pinned value.
    _dv_check_regression(pinned(; y = _DV_Y, x = _DV_X, s = 1.3), 1.3)
    # The same density as the definition spelling of the pin.
    defined = Base.merge(base, :(s = 0.7))(; y = _DV_Y, x = _DV_X)
    bp = pinned(; y = _DV_Y, x = _DV_X)
    @test _dv_value(_dv_built(bp), bp, :likelihood, _DV_U) ≈
        _dv_value(_dv_built(defined), defined, :likelihood, _DV_U)
end

@testset "data values: numbers compose as Julia values" begin
    twice = RKPPLModel(quote
        a ~ Normal(0, 5)
        b ~ Normal(0, 2)
        t = 2 * s
        mu = a .+ b .* x
        y .~ Normal.(mu, t)
    end, _DV)
    _dv_check_regression(twice(; y = _DV_Y, x = _DV_X, s = 0.35), 0.7)
    root = RKPPLModel(quote
        a ~ Normal(0, 5)
        b ~ Normal(0, 2)
        t = sqrt(s)
        mu = a .+ b .* x
        y .~ Normal.(mu, t)
    end, _DV)
    _dv_check_regression(root(; y = _DV_Y, x = _DV_X, s = 0.49), 0.7)
end

@testset "data values: a number is a prior argument, as its definition is" begin
    # A coefficient prior takes a name (the computed-coefficient lane), so a
    # number bound as data scales it exactly as the definition `tau = 2.0`.
    coef = quote
        a ~ Normal(0, 5)
        b ~ Normal(0, tau)
        mu = a .+ b .* x
        y .~ Normal.(mu, 1.0)
    end
    plan = lower_rkppl(coef,
        Dict{Symbol,Any}(:y => _DV_Y, :x => _DV_X, :tau => 2.0))
    for tau in (2.0, 0.5)
        bound = bind_data(plan,
            Dict{Symbol,ColumnData}(:y => _DV_Y, :x => _DV_X, :tau => tau))
        built = _dv_built(bound)
        a, b = ReactiveKernelsPPL.constrain(built.layout, _DV_U).mu
        @test _dv_value(built, bound, :prior, _DV_U) ≈
            logpdf(Normal(0, 5), a) + logpdf(Normal(0, tau), b)
    end
    defined = RKPPLModel(Expr(:block, :(tau = 2.0), coef.args...), _DV)(;
        y = _DV_Y, x = _DV_X)
    bound = RKPPLModel(coef, _DV)(; y = _DV_Y, x = _DV_X, tau = 2.0)
    @test _dv_value(_dv_built(bound), bound, :prior, _DV_U) ≈
        _dv_value(_dv_built(defined), defined, :prior, _DV_U)
    # A scalar parameter's prior argument.
    m = RKPPLModel(quote
        a ~ Normal(0, 5)
        b ~ Normal(0, 2)
        sigma ~ Exponential(rate)
        mu = a .+ b .* x
        y .~ Normal.(mu, sigma)
    end, _DV)
    bound = m(; y = _DV_Y, x = _DV_X, rate = 1.5)
    built = _dv_built(bound)
    u = [0.2, -0.4, 0.1]
    th = ReactiveKernelsPPL.constrain(built.layout, u)
    a, b = th.mu
    @test _dv_value(built, bound, :prior, u) ≈ logpdf(Normal(0, 5), a) +
        logpdf(Normal(0, 2), b) + logpdf(Exponential(1.5), th.sigma)
end

@testset "data values: a number bound to a response is one observation" begin
    m = RKPPLModel(quote
        a ~ Normal(0, 5)
        b ~ Normal(0, 2)
        mu = a .+ b .* x
        y .~ Normal.(mu, 1.0)
    end, _DV)
    bound = m(; y = 0.3, x = [0.5])
    built = _dv_built(bound)
    @test bound.n_obs == 1
    @test _dv_value(built, bound, :likelihood, _DV_U) ≈
        logpdf(Normal(0.2 - 0.4 * 0.5, 1.0), 0.3)
end

@testset "data values: an array of any dimension reads whole" begin
    m = RKPPLModel(quote
        a ~ Normal(0, 5)
        b ~ Normal(0, 2)
        w = total(A)
        mu = a .+ b .* x
        y .~ Normal.(mu, w)
    end, _DV)
    _dv_check_regression(m(; y = _DV_Y, x = _DV_X, A = fill(0.035, 2, 2, 5)),
        0.7)
end

@testset "data values: gradient through a number (Enzyme vs FD)" begin
    bound = RKPPLModel(_dv_regression(), _DV)(; y = _DV_Y, x = _DV_X, s = 0.7)
    built = _dv_built(bound)
    u = [0.2, -0.4]
    q = prepare_sampler(built, bound, u; backend =
        DifferentiationInterface.AutoEnzyme(; mode = Enzyme.Reverse))
    kern = prepare_query(built, bound, :sampler)
    v, g = Base.invokelatest(ReactiveKernels.ad_value_and_gradient!, q.ad,
        similar(u), u)
    @test v ≈ Base.invokelatest(kern, u)
    @test isapprox(g, _dv_findiff(w -> Base.invokelatest(kern, w), u);
        rtol = 1e-5, atol = 1e-7)
end

@testset "data values: a names-only plan keeps per-observation reads" begin
    # refused: lowering from names alone reads `s` per observation before
    # its value exists, so the bound plan cannot take a number there;
    # lowering with the values is the spelling that reads its shape
    # (decision 0dejlw1). The message names that spelling.
    plan = lower_rkppl(_dv_regression(), (:y, :x, :s))
    err = try
        bind_data(plan, Dict{Symbol,ColumnData}(:y => _DV_Y, :x => _DV_X,
            :s => 0.7))
        nothing
    catch e
        e
    end
    @test err isa ContractValidationError
    @test occursin("lower_rkppl(ast, data)", err.message)
end
