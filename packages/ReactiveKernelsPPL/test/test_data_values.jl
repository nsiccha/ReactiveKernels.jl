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
    _dv_check_regression((m(; x = _DV_X, s = 0.7) | (; y = _DV_Y)), 0.7)
    # `@rkppl data`.
    y, x = _DV_Y, _DV_X
    bound = (@rkppl (; x, s = 0.7) begin
        a ~ Normal(0, 5)
        b ~ Normal(0, 2)
        mu = a .+ b .* x
        y .~ Normal.(mu, s)
    end) | (; y)
    _dv_check_regression(bound, 0.7)
    # `lower_rkppl(ast, data)` reads each value's shape; the plan then binds
    # any number for `s` without lowering again.
    for data in ((; y, x, s = 0.7), Dict{Symbol,Any}(:y => y, :x => x, :s => 0.7))
        plan = lower_rkppl(_dv_regression(), data; conditioned = (:y,))
        for sd in (0.7, 1.1)
            for values in ((; y, x, s = sd), Dict{Symbol,ColumnData}(:y => y, :x => x, :s => sd))
                _dv_check_regression(bind_data(plan, values), sd)
            end
        end
    end
end

@testset "data values: NamedTuple binding shares validation and preserves inputs" begin
    data = (; x = _DV_X, y = _DV_Y, s = 0.7, A = fill(0.125, 2, 2, 2))
    saved = deepcopy(data)
    plan = lower_rkppl(quote
        a ~ Normal(0, 5)
        b ~ Normal(0, 2)
        width = s * total(A)
        mu = a .+ b .* x
        y .~ Normal.(mu, width)
    end, data; conditioned = (:y,), mod = _DV)
    bound = bind_data(plan, data; roles = Dict(:x => :data))
    control = bind_data(plan, Dict{Symbol,Any}(pairs(data)); roles = Dict(:x => :data))
    @test bound.columns == control.columns
    @test bound.roles == control.roles
    @test bound.roles[:x] == :data
    @test bound.n_obs == control.n_obs == length(data.y)
    @test bound.columns[:x] === data.x
    @test bound.columns[:A] === data.A
    @test isempty(plan.columns)
    @test isequal(data, saved)
    built, control_built = build_kernel(bound), build_kernel(control)
    @test kernel_expr(bound, built.layout) == kernel_expr(control, control_built.layout)
    for preset in (:prior, :likelihood, :log_jacobian, :sampler)
        @test _dv_value(built, bound, preset, _DV_U) ==
            _dv_value(control_built, control, preset, _DV_U)
    end
    rebound = bind_data(bound, (; data..., s = 1.1))
    _dv_check_regression(rebound, 1.1)
    @test bound.columns[:s] == 0.7
    @test rebound.columns[:s] == 1.1
    @test isequal(data, saved)

    # refused: role overrides must name supplied data and admitted roles.
    @test_throws ContractValidationError bind_data(plan, data; roles = Dict(:absent => :data))
    @test_throws ContractValidationError bind_data(plan, data; roles = Dict(:x => :unknown))
    # refused: every supplied plate dimension must be consumed by the plan.
    @test_throws ContractValidationError bind_data(plan, data; dims = Dict(:unused => 2))

    observed = lower_rkppl(quote
        a ~ Normal(0, 1)
        tau ~ Exponential(1)
        y .~ Normal.(a .+ x, tau)
    end, (; x = _DV_X, y = _DV_Y); conditioned = (:y,))
    observed_data = (; x = _DV_X, y = _DV_Y, tau = 0.9)
    conditioned = bind_data(observed, observed_data; conditioned = (:y, :tau))
    conditioned_control = bind_data(observed, Dict{Symbol,Any}(pairs(observed_data));
        conditioned = (:y, :tau))
    @test conditioned.conditioned == Set((:tau,))
    @test coordinate_names(build_kernel(conditioned).layout) == [:a]
    @test conditioned.columns == conditioned_control.columns
    @test isempty(observed.conditioned)
    @test observed_data.tau == 0.9

    # Prior-only models need no data mapping or observation axis.
    prior = lower_rkppl(quote
        a ~ Normal(0, 1)
    end, (;))
    empty_bound = bind_data(prior, (;))
    @test isempty(empty_bound.columns)
    @test coordinate_names(build_kernel(empty_bound).layout) == [:a]
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
    _dv_check_regression((pinned(; x = _DV_X) | (; y = _DV_Y)), 0.7)
    # An explicit call keyword wins over the pinned value.
    _dv_check_regression((pinned(; x = _DV_X, s = 1.3) | (; y = _DV_Y)), 1.3)
    # The same density as the definition spelling of the pin.
    defined = Base.merge(base, :(s = 0.7))(; x = _DV_X) | (; y = _DV_Y)
    bp = (pinned(; x = _DV_X) | (; y = _DV_Y))
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
    _dv_check_regression((twice(; x = _DV_X, s = 0.35) | (; y = _DV_Y)), 0.7)
    root = RKPPLModel(quote
        a ~ Normal(0, 5)
        b ~ Normal(0, 2)
        t = sqrt(s)
        mu = a .+ b .* x
        y .~ Normal.(mu, t)
    end, _DV)
    _dv_check_regression((root(; x = _DV_X, s = 0.49) | (; y = _DV_Y)), 0.7)
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
        Dict{Symbol,Any}(:y => _DV_Y, :x => _DV_X, :tau => 2.0); conditioned = Dict{Symbol,Any}(:y => _DV_Y, :x => _DV_X, :tau => 2.0))
    for tau in (2.0, 0.5)
        bound = bind_data(plan,
            Dict{Symbol,ColumnData}(:y => _DV_Y, :x => _DV_X, :tau => tau))
        built = _dv_built(bound)
        nt = ReactiveKernelsPPL.constrain(built.layout, _DV_U)
        a, b = nt.a, nt.b
        @test _dv_value(built, bound, :prior, _DV_U) ≈
            logpdf(Normal(0, 5), a) + logpdf(Normal(0, tau), b)
    end
    defined = RKPPLModel(Expr(:block, :(tau = 2.0), coef.args...), _DV)(;
        x = _DV_X) | (; y = _DV_Y)
    bound = RKPPLModel(coef, _DV)(; x = _DV_X, tau = 2.0) | (; y = _DV_Y)
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
    bound = (m(; x = _DV_X, rate = 1.5) | (; y = _DV_Y))
    built = _dv_built(bound)
    u = [0.2, -0.4, 0.1]
    th = ReactiveKernelsPPL.constrain(built.layout, u)
    a, b = th.a, th.b
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
    bound = (m(; x = [0.5]) | (; y = 0.3))
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
    _dv_check_regression((m(; x = _DV_X, A = fill(0.035, 2, 2, 5)) | (; y = _DV_Y)),
        0.7)
end

@testset "data values: gradient through a number (Enzyme vs FD)" begin
    data = (; x = _DV_X, y = _DV_Y, s = 0.7)
    plan = lower_rkppl(_dv_regression(), data; conditioned = (:y,), mod = _DV)
    bound = bind_data(plan, data)
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
    plan = lower_rkppl(_dv_regression(), (:y, :x, :s); conditioned = (:y, :x, :s))
    err = try
        bind_data(plan, Dict{Symbol,ColumnData}(:y => _DV_Y, :x => _DV_X,
            :s => 0.7))
        nothing
    catch e
        e
    end
    # capability: bind scalar data to a names-only plan at the same model door (P10a 0dejlw1; todo `1qlbn5b`).
    @test_broken (err === nothing || throw(err))

end
