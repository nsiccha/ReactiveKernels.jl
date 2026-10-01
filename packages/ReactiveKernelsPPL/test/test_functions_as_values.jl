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
as_vector(x) = collect(x)
scaled(v, k) = v .* k
l2norm(v) = sqrt(sum(abs2, v))
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

@testset "functions as values: a gather index is integer data" begin
    cols = _fv_cols()
    gathered(idx, extra...) = quote
        zeta ~ Dirichlet([1.0, 2.0])
        a ~ Normal(0, 5)
        b ~ Normal(0, 2)
        sigma ~ Exponential(1.0)
        $(extra...)
        cum = cumsum(vcat(0.0, zeta))
        m = cum[$idx]
        mu = a .+ m
        y .~ Normal.(mu, sigma)
    end
    # A per-cell latent is a value, never an index (contract, at lowering).
    @test_throws ContractValidationError lower_rkppl(gathered(:x_true,
            Expr(:macrocall, Symbol("@plate"), LineNumberNode(1),
                Expr(:for, Expr(:(=), :i, :(eachindex(x))), Expr(:block,
                    :(x_true[i] ~ Normal(0, 1)),
                    :(x[i] ~ Normal.(x_true[i], 0.5)))))),
        Tuple(keys(cols)); mod = _FV)
    # So is a definition reading a parameter, however it is named.
    @test_throws ContractValidationError lower_rkppl(
        gathered(:k, :(k = c .* b)), Tuple(keys(cols)); mod = _FV)
    # A raw index column must hold integers (bind).
    real_idx = lower_rkppl(gathered(:x), Tuple(keys(cols)); mod = _FV)
    @test_throws ContractValidationError bind_data(real_idx, cols)
    # An integer data column gathers (the corpus 97 shape).
    @test bind_data(lower_rkppl(gathered(:c), Tuple(keys(cols)); mod = _FV),
        cols) isa StructuralPlan
end

@testset "functions as values: dotted built-in over an observation column" begin
    # `tanh` is a scalar built-in outside the dotted elementwise vocabulary;
    # dotted over a data column it broadcasts the built-in itself.
    cols = _fv_cols()
    plan, bound, built = _fv_build(quote
            a ~ Normal(0, 5)
            sigma ~ Exponential(1.0)
            w = tanh.(x)
            mu = a .+ w
            y .~ Normal.(mu, sigma)
        end, cols)
    for u in ([0.3, -0.4], [-1.1, 0.6])
        th = ReactiveKernelsPPL.constrain(built.layout, u)
        a = only(th.mu)
        @test _fv_value(built, bound, :likelihood, u) ≈
            sum(logpdf(Normal(a + tanh(cols[:x][i]), th.sigma), cols[:y][i])
                for i in eachindex(cols[:y]))
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

@testset "functions as values: a definition reached twice (diamond)" begin
    # A definition read by two definitions that both inline into one
    # location is a DAG, not a cycle. The definition walks once unwound
    # their path with `pop!` on a `Set`, which drops an arbitrary element,
    # so legality depended on the names' hashes. Each name below failed
    # under that unwinding in at least one of the two shapes tested here.
    names = (:scale, :base, :q, :tmp, :core, :mid, :alpha2)
    y = [0.1, 0.4, -0.2, 0.3, 0.0, 0.5]
    gx = [0.3, -0.1, 0.2, 0.7, -0.4, 0.05]
    g = [1, 2, 3, 1, 2, 3]
    cols = Dict{Symbol,ColumnData}(:y => y, :gx => gx, :g => g)
    u = [0.2, -0.3, 0.5]
    for s in names
        r1, r2 = Symbol(s, :_1), Symbol(s, :_2)
        _, bound, built = _fv_build(quote
                sigma ~ Exponential(1.0)
                a ~ Normal(0, 1)
                k ~ Normal(0, 1)
                $s = a .+ as_vector(gx)
                $r1 = scaled($s, k)
                $r2 = scaled($s, 2 * k)
                y .~ Normal.($r1[g] .+ $r2[g], sigma)
            end, cols)
        th = ReactiveKernelsPPL.constrain(built.layout, u)
        sc = th.a .+ gx
        ll = sum(logpdf(Normal(3 * th.k * sc[g[i]], th.sigma), y[i])
            for i in eachindex(y))
        @test _fv_value(built, bound, :likelihood, u) ≈ ll
    end
    # A real cycle through the shared definition still fails as one.
    err = try
        lower_rkppl(quote
                sigma ~ Exponential(1.0)
                a ~ Normal(0, 1)
                k ~ Normal(0, 1)
                scale = a .+ as_vector(gx) .+ sum(r2)
                r1 = scaled(scale, k)
                r2 = scaled(scale, 2 * k)
                y .~ Normal.(r1[g] .+ r2[g], sigma)
            end, (:y, :gx, :g); mod = _FV)
        nothing
    catch e
        e
    end
    @test err isa SurfaceLoweringError
    @test occursin("cyclic definition", err.message)
    # Data-only definitions reached twice stay data-only, so a module call
    # over their observation-aligned result binds once instead of being
    # refused as parameter-dependent.
    x = [0.3, -0.1, 0.2, 0.5, 0.1, 0.0]
    cols2 = Dict{Symbol,ColumnData}(:y => y, :x => x)
    nrm = sqrt(sum(abs2, (x .+ 1.0) .* 2.0 .+ (x .+ 1.0 .+ 1.0)))
    for s in names
        u1, u2, v = Symbol(s, :_1), Symbol(s, :_2), Symbol(s, :_v)
        plan = lower_rkppl(quote
                sigma ~ Exponential(1.0)
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                $s = x .+ 1.0
                $u1 = $s .* 2.0
                $u2 = $s .+ 1.0
                $v = $u1 .+ $u2
                w = l2norm($v)
                mu = a .+ b .* (x ./ w)
                y .~ Normal.(mu, sigma)
            end, (:y, :x); mod = _FV)
        bound = bind_data(plan, cols2)
        built = build_kernel(bound)
        th = ReactiveKernelsPPL.constrain(built.layout, u)
        a, b = th.mu
        ll = sum(logpdf(Normal(a + b * x[i] / nrm, th.sigma), y[i])
            for i in eachindex(y))
        @test _fv_value(built, bound, :likelihood, u) ≈ ll
    end
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
