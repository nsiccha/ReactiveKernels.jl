using Distributions
using ReactiveKernels
using ReactiveKernelsPPL
using Test
using Statistics: var
import DifferentiationInterface
import Enzyme

# Functions as values (decision 1cmodra, prong `functions`): an `=`
# definition may call any function visible in the model's module. A
# data-only call is evaluated once by `bind_data` and bound as data; a
# parameter-dependent call runs in the generated kernel under generic AD
# (an RK-owned derivative rule, when the function is one, is picked up by
# Enzyme with no extra spelling). References below are independent
# Distributions.jl computations over the constrained draws, never the
# emitted forms. Self-contained: no helpers from earlier includes.

module FunctionsAsValuesModels
using Statistics: var
const CALLS = Ref(0)
# A user helper (plain Julia, defined in the model's module).
affine_helper(x) = 2x + 1
# A data-only helper that counts its evaluations (bind-once check).
function column_variances(X)
    CALLS[] += 1
    return var.(eachcol(X))
end
shifted(v; by = 1.0) = v .+ by
# An RK-owned derivative rule (the optional registered rule): softplus with
# its authored partial; Enzyme uses the generated rule.
import ReactiveKernels
ReactiveKernels.@kernel softplus_graph(x::Float64) = begin
    y::Float64 = log1p(exp(x))
    dy_dx::Float64 = 1 / (1 + exp(-x))
    return y, dy_dx
end
const softplus_rule = ReactiveKernels.scalar_derivative_rule(softplus_graph;
    primal = :y, partials = (x = :dy_dx,), name = :softplus_rule)
# Whole-value helpers for data off the observation axis: an identity, and
# a generic decayed-event response — observation i (group g, time t) sums
# the earlier events (group eg, time et, amount ea) of its group.
as_vector(x) = collect(x)
function decayed_events(g, t, eg, et, ea, w, scale, k)
    T = promote_type(eltype(scale), typeof(k))
    out = Vector{T}(undef, length(t))
    for i in eachindex(t)
        acc = zero(T)
        for j in eachindex(eg)
            if eg[j] == g[i] && et[j] <= t[i]
                acc += ea[j] * exp(-exp(k) * (t[i] - et[j]))
            end
        end
        out[i] = acc * scale[g[i]] + sum(w) / length(w)
    end
    return out
end
end
const _FV = FunctionsAsValuesModels

# A model module with no Statistics import (dotted vocabulary check).
module FunctionsAsValuesBare
shifted(v; by = 1.0) = v .+ by
end

const _FV_BACKEND = DifferentiationInterface.AutoEnzyme(;
    mode = Enzyme.Reverse)

function _fv_build(ast, cols; mod::Module = _FV)
    plan = lower_rkppl(ast, Tuple(keys(cols)); mod)
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    return plan, bound, built
end

function _fv_value(built, bound, preset, u)
    kern = prepare_query(built, bound, preset)
    return Base.invokelatest(kern, u)
end

function _fv_findiff(f, u; h = cbrt(eps(Float64)))
    g = similar(u, Float64)
    for i in eachindex(u)
        up = copy(u); up[i] += h
        dn = copy(u); dn[i] -= h
        g[i] = (f(up) - f(dn)) / (2h)
    end
    return g
end

_fv_cols() = Dict{Symbol,AbstractVector}(
    :y => [0.5, 1.0, 1.5, 2.0, 0.8, 1.2, 1.9],
    :c => [1, 2, 3, 2, 1, 3, 3],
    :x => [0.5, -1.0, 1.5, 0.0, -0.5, 1.0, 0.25],
)

# Cumulative-simplex contrast through ordinary Julia functions: the
# monotonic-effect mathematics with no built-in construct.
_fv_cum_model() = quote
    zeta ~ Dirichlet([1.0, 2.0])
    a ~ Normal(0, 5)
    b ~ Normal(0, 2)
    sigma ~ Exponential(1.0)
    cum = cumsum(vcat(0.0, zeta))
    m = cum[c]
    mu = a .+ b .* m
    y .~ Normal.(mu, sigma)
end

@testset "functions as values: lowering" begin
    plan = lower_rkppl(_fv_cum_model(), (:y, :c))
    # `cumsum`/`vcat` resolve in the model module (Main here) as GlobalRefs.
    # `cum` and `m` feed only the predictor, so they inline into one
    # extracted observation column: the gather of the cumulative simplex.
    col = only(plan.derived)
    @test col.expr == Expr(:ref, Expr(:call, GlobalRef(Main, :cumsum),
        Expr(:call, GlobalRef(Main, :vcat), 0.0, :zeta)), :c)
    # The simplex is a free-standing vector parameter.
    @test only(plan.vector_parameters).name === :zeta

    # A user-defined helper resolves in the module the program names.
    p2 = lower_rkppl(quote
            m0 ~ Normal(0, 1)
            bx ~ Normal(0, 1)
            s ~ Exponential(1.0)
            t = affine_helper(s)
            mu = m0 .+ bx .* x
            y .~ Normal.(mu, t)
        end, (:y, :x); mod = _FV)
    t = only(a for a in p2.assignments if a.name === :t)
    @test t.expr.args[1] == GlobalRef(_FV, :affine_helper)

    # An undefined function fails at lowering, naming it and the module.
    err = try
        lower_rkppl(quote
                m0 ~ Normal(0, 1)
                s ~ Exponential(1.0)
                t = no_such_helper(s)
                y .~ Normal.(m0, t)
            end, (:y,); mod = _FV)
        nothing
    catch e
        e
    end
    @test err isa SurfaceLoweringError
    @test occursin("no_such_helper", err.message)
    @test occursin("not defined", err.message)

    # A parameter-dependent undotted call over an observation column has no
    # knowable shape: refused with the broadcast spelling.
    err = try
        lower_rkppl(quote
                s ~ Exponential(1.0)
                t = shifted(x; by = s)
                y .~ Normal.(t, 1.0)
            end, (:y, :x); mod = _FV)
        nothing
    catch e
        e
    end
    @test err isa SurfaceLoweringError
    @test occursin("shifted", err.message)
    @test occursin("broadcast", err.message)

    # Calling a model value is not a function call.
    @test_throws SurfaceLoweringError lower_rkppl(quote
            m0 ~ Normal(0, 1)
            s ~ Exponential(1.0)
            t = s(1.0)
            y .~ Normal.(m0, t)
        end, (:y,); mod = _FV)
end

@testset "functions as values: cumsum gather density matches Distributions" begin
    cols = _fv_cols()
    plan, bound, built = _fv_build(_fv_cum_model(), cols)
    lay = built.layout
    for u in ([0.1, -0.3, 0.4, 0.2], [-1.0, 0.7, -0.2, 0.9])
        length(u) == lay.total || error("layout total $(lay.total)")
        th = ReactiveKernelsPPL.constrain(lay, u)
        zeta = th.zeta
        @test length(zeta) == 2 && sum(zeta) ≈ 1.0
        @test coordinate_names(lay) == [:a, :b, :sigma, Symbol("zeta.1")]
        a, b = th.a, th.b
        cum = [0.0; cumsum(zeta)]
        mu = [a + b * cum[ci] for ci in cols[:c]]
        ll = sum(logpdf(Normal(mu[i], th.sigma), cols[:y][i])
            for i in eachindex(mu))
        lp = logpdf(Normal(0, 5), a) + logpdf(Normal(0, 2), b) +
            logpdf(Exponential(1.0), th.sigma) +
            logpdf(Dirichlet([1.0, 2.0]), zeta)
        @test _fv_value(built, bound, :likelihood, u) ≈ ll
        @test _fv_value(built, bound, :prior, u) ≈ lp
    end
end

@testset "functions as values: naming a subexpression never changes legality" begin
    cols = _fv_cols()
    named = _fv_build(_fv_cum_model(), cols)
    inline = _fv_build(quote
            zeta ~ Dirichlet([1.0, 2.0])
            a ~ Normal(0, 5)
            b ~ Normal(0, 2)
            sigma ~ Exponential(1.0)
            mu = a .+ b .* cumsum(vcat(0.0, zeta))[c]
            y .~ Normal.(mu, sigma)
        end, cols)
    summand = _fv_build(quote
            zeta ~ Dirichlet([1.0, 2.0])
            a ~ Normal(0, 5)
            sigma ~ Exponential(1.0)
            cum = cumsum(vcat(0.0, zeta))
            mu = a .+ cum[c]
            y .~ Normal.(mu, sigma)
        end, cols)
    named_summand = _fv_build(quote
            zeta ~ Dirichlet([1.0, 2.0])
            a ~ Normal(0, 5)
            sigma ~ Exponential(1.0)
            cum = cumsum(vcat(0.0, zeta))
            m = cum[c]
            mu = a .+ m
            y .~ Normal.(mu, sigma)
        end, cols)
    u = [0.1, -0.3, 0.4, 0.2]
    @test _fv_value(inline[3], inline[2], :sampler, u) ≈
        _fv_value(named[3], named[2], :sampler, u)
    v = u[1:3]
    @test _fv_value(summand[3], summand[2], :sampler, v) ≈
        _fv_value(named_summand[3], named_summand[2], :sampler, v)
    # Elementwise module calls inline in a predictor extract like their
    # named form, evaluated once at bind (data-only).
    p1, b1, k1 = _fv_build(quote
            a ~ Normal(0, 5)
            b ~ Normal(0, 2)
            sigma ~ Exponential(1.0)
            mu = a .+ b .* affine_helper.(x)
            y .~ Normal.(mu, sigma)
        end, cols)
    p2, b2, k2 = _fv_build(quote
            a ~ Normal(0, 5)
            b ~ Normal(0, 2)
            sigma ~ Exponential(1.0)
            ax = affine_helper.(x)
            mu = a .+ b .* ax
            y .~ Normal.(mu, sigma)
        end, cols)
    w = [0.2, -0.1, 0.3]
    @test _fv_value(k1, b1, :sampler, w) ≈ _fv_value(k2, b2, :sampler, w)
end

@testset "functions as values: parameter-dependent gradient (Enzyme vs FD)" begin
    cols = _fv_cols()
    ast = quote
        zeta ~ Dirichlet([1.0, 2.0])
        a ~ Normal(0, 5)
        b ~ Normal(0, 2)
        s ~ Exponential(1.0)
        cum = cumsum(vcat(0.0, zeta))
        m = cum[c]
        mu = a .+ b .* m
        sigma = affine_helper(s)
        y .~ Normal.(mu, sigma)
    end
    _, bound, built = _fv_build(ast, cols)
    u = [0.2, -0.4, 0.3, 0.1]
    q = prepare_sampler(built, bound, u; backend = _FV_BACKEND)
    kern = prepare_query(built, bound, :sampler)
    v, g = Base.invokelatest(ReactiveKernels.ad_value_and_gradient!, q.ad,
        similar(u), u)
    @test v ≈ Base.invokelatest(kern, u)
    @test all(isfinite, g)
    @test isapprox(g, _fv_findiff(w -> Base.invokelatest(kern, w), u);
        rtol = 1e-5, atol = 1e-7)
end

@testset "functions as values: registered derivative rule" begin
    cols = _fv_cols()
    ast = quote
        a ~ Normal(0, 5)
        b ~ Normal(0, 2)
        s_raw ~ Normal(0, 1)
        sigma = softplus_rule(s_raw)
        mu = a .+ b .* x
        y .~ Normal.(mu, sigma)
    end
    _, bound, built = _fv_build(ast, cols)
    u = [0.3, -0.2, 0.4]
    th = ReactiveKernelsPPL.constrain(built.layout, u)
    sig = log1p(exp(th.s_raw))
    a, b = th.mu
    ll = sum(logpdf(Normal(a + b * cols[:x][i], sig), cols[:y][i])
        for i in eachindex(cols[:y]))
    @test _fv_value(built, bound, :likelihood, u) ≈ ll
    q = prepare_sampler(built, bound, u; backend = _FV_BACKEND)
    kern = prepare_query(built, bound, :sampler)
    _, g = Base.invokelatest(ReactiveKernels.ad_value_and_gradient!, q.ad,
        similar(u), u)
    @test isapprox(g, _fv_findiff(w -> Base.invokelatest(kern, w), u);
        rtol = 1e-5, atol = 1e-7)
end

@testset "functions as values: data-only calls bind once" begin
    n = 7
    X = hcat(collect(1.0:n), [0.3, -1.2, 0.8, 2.0, -0.4, 1.1, 0.0])
    cols = Dict{Symbol,ColumnData}(:y => [0.2, 1.0, -0.5, 1.4, 0.3, 0.9,
        -0.1], :X => X, :x => collect(range(-1.0, 1.0; length = n)))
    ast = quote
        vx = column_variances(X)
        k = sum(vx)
        ax = affine_helper.(x)
        b ~ Normal(0, 1)
        sigma ~ Exponential(1.0)
        s_eff = sigma * k
        mu = b .* ax
        y .~ Normal.(mu, s_eff)
    end
    _FV.CALLS[] = 0
    plan = lower_rkppl(ast, (:y, :X, :x); mod = _FV)
    @test _FV.CALLS[] == 0                     # lowering is data-free
    bound = bind_data(plan, cols)
    @test _FV.CALLS[] == 1                     # evaluated once, at bind
    @test bound.columns[:vx] == [var(X[:, 1]), var(X[:, 2])]
    @test bound.columns[:ax] == 2 .* cols[:x] .+ 1
    @test bound.n_obs == n
    built = build_kernel(bound)
    u = [0.4, -0.2]
    th = ReactiveKernelsPPL.constrain(built.layout, u)
    k = var(X[:, 1]) + var(X[:, 2])
    b = only(th.mu)
    ll = sum(logpdf(Normal(b * (2 * cols[:x][i] + 1), th.sigma * k),
        cols[:y][i]) for i in 1:n)
    @test _fv_value(built, bound, :likelihood, u) ≈ ll
    q = prepare_sampler(built, bound, u; backend = _FV_BACKEND)
    for w in ([0.1, 0.3], [-0.7, 1.2])
        Base.invokelatest(ReactiveKernels.ad_value_and_gradient!, q.ad,
            similar(w), w)
    end
    @test _FV.CALLS[] == 1                     # never recomputed in-graph
    # A caller column under a computed name is refused, not shadowed.
    cols2 = copy(cols)
    cols2[:vx] = [1.0, 2.0]
    @test_throws ContractValidationError bind_data(plan, cols2)
end

@testset "functions as values: dotted vocabulary and keywords" begin
    # `var.` broadcasts the built-in vocabulary's `var` without importing
    # Statistics into the model module; keyword arguments pass through.
    n = 5
    X = hcat(collect(1.0:n), [2.0, -1.0, 0.5, 0.0, 1.5])
    cols = Dict{Symbol,ColumnData}(:y => [0.1, 0.4, -0.3, 0.8, 0.2], :X => X,
        :x => [0.5, -1.0, 0.25, 1.5, 0.0])
    ast = quote
        vx = var.(eachcol(X))
        w = shifted(vx; by = 0.5)
        sq = map(abs2, vx)
        nrm = Base.sqrt(sum(sq))
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        sigma ~ Exponential(1.0)
        s_eff = sigma * sum(w) / nrm
        mu = a .+ b .* x
        y .~ Normal.(mu, s_eff)
    end
    plan = lower_rkppl(ast, (:y, :X, :x); mod = FunctionsAsValuesBare)
    @test !isdefined(FunctionsAsValuesBare, :var)
    bound = bind_data(plan, cols)
    vx = [var(X[:, j]) for j in 1:2]
    @test bound.columns[:vx] == vx
    @test bound.columns[:w] == vx .+ 0.5
    @test bound.columns[:sq] == abs2.(vx)        # function value argument
    @test bound.columns[:nrm] ≈ sqrt(sum(abs2, vx))  # qualified call head
    built = build_kernel(bound)
    u = [0.3, -0.4, 0.2]
    th = ReactiveKernelsPPL.constrain(built.layout, u)
    a, b = th.mu
    sd = th.sigma * sum(vx .+ 0.5) / sqrt(sum(abs2, vx))
    @test _fv_value(built, bound, :likelihood, u) ≈
        sum(logpdf(Normal(a + b * cols[:x][i], sd), cols[:y][i]) for i in 1:n)
end

@testset "functions as values: model-level data inputs" begin
    # A column read only as a module-call argument is a whole value: no
    # observation axis, any length, and never the n_obs anchor.
    cols = Dict{Symbol,ColumnData}(:y => [0.1, 0.4, -0.2, 0.3, 0.0, 0.5],
        :g => [1, 2, 3, 1, 2, 3], :gx => [0.5, -1.0, 2.0])
    ast = quote
        sigma ~ Exponential(1.0)
        b ~ Normal(0, 1)
        gx_m = as_vector(gx)
        v = b .* gx_m
        y .~ Normal.(v[g], sigma)
    end
    plan = lower_rkppl(ast, (:y, :g, :gx); mod = _FV)
    for order in (collect(cols), reverse(collect(cols)))
        @test bind_data(plan, Dict{Symbol,ColumnData}(order)).n_obs == 6
    end
    bound = bind_data(plan, cols)
    @test bound.columns[:gx] == [0.5, -1.0, 2.0]
    built = build_kernel(bound)
    for u in ([0.3, -0.2], [-1.1, 0.7])
        th = ReactiveKernelsPPL.constrain(built.layout, u)
        ll = sum(logpdf(Normal(th.b * cols[:gx][cols[:g][i]], th.sigma),
            cols[:y][i]) for i in 1:6)
        @test _fv_value(built, bound, :likelihood, u) ≈ ll
    end
    # Any per-observation read keeps the column observation-aligned, so
    # its length is checked as before.
    mixed = lower_rkppl(quote
            sigma ~ Exponential(1.0)
            b ~ Normal(0, 1)
            gx_m = as_vector(gx)
            v = b .* gx_m
            mu = v[g] .+ gx
            y .~ Normal.(mu, sigma)
        end, (:y, :g, :gx); mod = _FV)
    @test_throws ContractValidationError bind_data(mixed, cols)
    # So does a name any other plan slot holds (here: response weights).
    weighted_plan = lower_rkppl(quote
            sigma ~ Exponential(1.0)
            b ~ Normal(0, 1)
            gx_m = as_vector(gx)
            v = b .* gx_m
            y .~ weighted.(Normal.(v[g], sigma), gx)
        end, (:y, :g, :gx); mod = _FV)
    @test_throws ContractValidationError bind_data(weighted_plan, cols)
end

@testset "functions as values: data on several axes (Enzyme vs FD)" begin
    # Observations (8), events (5), groups (3) and weights (4): every
    # column off the observation axis reaches the kernel through module
    # calls; the observation index `oi` gathers the model-level reads.
    cols = Dict{Symbol,ColumnData}(
        :y => [0.2, 0.9, 1.4, 0.3, 0.7, 1.1, 0.5, 0.95],
        :oi => collect(1:8), :g => [1, 1, 1, 2, 2, 3, 3, 3],
        :t => [0.5, 1.0, 2.0, 0.5, 1.5, 0.25, 1.0, 3.0],
        :eg => [1, 1, 2, 3, 3], :et => [0.0, 1.0, 0.0, 0.0, 0.5],
        :ea => [1.0, 0.5, 2.0, 1.0, 1.5],
        :gx => [0.2, -0.4, 1.0], :w => [0.1, 0.2, -0.1, 0.4])
    ast = quote
        sigma ~ Exponential(1.0)
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        k ~ Normal(0, 1)
        gm = as_vector(g)
        tm = as_vector(t)
        egm = as_vector(eg)
        etm = as_vector(et)
        eam = as_vector(ea)
        wm = as_vector(w)
        scale = a .+ b .* as_vector(gx)
        reads = decayed_events(gm, tm, egm, etm, eam, wm, scale, k)
        y .~ Normal.(reads[oi], sigma)
    end
    _, bound, built = _fv_build(ast, cols)
    @test bound.n_obs == 8
    kern = prepare_query(built, bound, :sampler)
    for u in ([0.3, -0.2, 0.5, 0.1], [-1.1, 0.7, -0.3, -0.4])
        th = ReactiveKernelsPPL.constrain(built.layout, u)
        reads = _FV.decayed_events(cols[:g], cols[:t], cols[:eg], cols[:et],
            cols[:ea], cols[:w], th.a .+ th.b .* cols[:gx], th.k)
        ll = sum(logpdf(Normal(reads[i], th.sigma), cols[:y][i]) for i in 1:8)
        @test _fv_value(built, bound, :likelihood, u) ≈ ll
        q = prepare_sampler(built, bound, u; backend = _FV_BACKEND)
        v, g = Base.invokelatest(ReactiveKernels.ad_value_and_gradient!,
            q.ad, similar(u), u)
        @test v ≈ Base.invokelatest(kern, u)
        @test isapprox(g, _fv_findiff(w -> Base.invokelatest(kern, w), u);
            rtol = 1e-5, atol = 1e-7)
    end
end
