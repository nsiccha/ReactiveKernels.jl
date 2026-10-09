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
squared(v) = v .^ 2
decay_weights(v, s) = s .* exp.(-0.25 .* v)
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
# A helper with forty positional scalar arguments after one vector, summed
# pairwise (no splat in its own body): a call to it carries more than the 32
# arguments Julia's inliner forwards statically.
const WIDE_NAMES = [Symbol(:b, i) for i in 1:40]
@eval wide_reads(v, $(WIDE_NAMES...)) =
    fill(sum(v) + $(foldl((a, b) -> :($a + $b), WIDE_NAMES)), 3)
# A one-element schedule column unwrapped by a guard whose message
# interpolates (counted: the data-only part runs once, outside the
# gradient), and a parameter-dependent read of the schedule.
const UNWRAPS = Ref(0)
function take_schedule(col)
    UNWRAPS[] += 1
    length(col) == 1 ||
        throw(ArgumentError("need one schedule, got $(length(col))"))
    return only(col)
end
schedule_reads(sched, b) = b .* sched.a .+ sched.k
schedule_weights(w::AbstractVector) = w
schedule_weights(w::AbstractMatrix) = w[:, 1]
weighted_reads(sched, w, b) = schedule_weights(w) .* sched.a .+ sched.k .+ b
weighted_reads(sched::Tuple, w, b) =
    schedule_weights(w) .* sched[1] .+ sched[2] .+ b
function bind_scale(x)
    UNWRAPS[] += 1
    return 2.0
end
function bind_matrix(x)
    UNWRAPS[] += 1
    return reshape(copy(x), 1, length(x))
end
matrix_reads(M, w, b) = w .* vec(M) .+ b
const POOLS = Ref(0)
function group_pool(g, h)
    POOLS[] += 1
    return vcat(g, h)
end
function group_index(g, gg)
    lv = sort(unique(gg))
    return Int[findfirst(isequal(v), lv) for v in g]
end
read_rows(values, rows) = values[rows]
# A function-shaped kernel value and a gather helper, read by several
# response locations (shared model-level value check).
ReactiveKernels.@kernel exp_value(x_) = begin
    x = exp.(x_)
end
column_gather(draws, index, margin) = draws[index, margin]
# A call returning a tuple whose elements have different shapes, live in
# its second argument, and a data-only one.
pair_reads(x, b) = (cumsum(b .* x), sum(b .* x))
data_pair(x) = (sum(x), 2 .* x)
end
const _FV = FunctionsAsValuesModels

# A model module with no Statistics import (dotted vocabulary check).
module FunctionsAsValuesBare
shifted(v; by = 1.0) = v .+ by
end

const _FV_BACKEND = DifferentiationInterface.AutoEnzyme(;
    mode = Enzyme.Reverse)

function _fv_build(ast, cols; mod::Module = _FV)
    plan = lower_rkppl(ast, Tuple(keys(cols)); mod, conditioned = Tuple(keys(cols)))
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

# A group definition used by both a declared axis and an inlined reader.
# Keep the original vcat spelling as well as a transitive helper/alias chain.
function _fv_group_axis_case(n; chained = false)
    ast = quote
        gg = vcat(g, h)
        sd[1:1] .~ HalfNormal.(1)
        z[levels(gg), 1:1] .~ Normal.(0, 1)
        C = z .* sd[1]
        i = group_index(g, gg)
        j = group_index(h, gg)
        mu = C[i, 1] .+ C[j, 1]
        reads = read_rows(mu, rows)
        y .~ Normal.(reads, 1)
    end
    if chained
        idx = findfirst(ex -> ex isa Expr && ex.head === :(=) && ex.args[1] === :gg,
            ast.args)
        splice!(ast.args, idx:idx,
            [:(pool = group_pool(g, h)), :(gg = identity(pool))])
    end
    g = collect(1:n)
    cols = Dict{Symbol,Any}(:g => g, :h => circshift(g, 1),
        :rows => reverse(g), :y => [0.2 * cos(i) for i in g])
    # Only y is conditioned, matching the reported public API call.
    plan = lower_rkppl(ast, Tuple(keys(cols)); mod = _FV, conditioned = (:y,))
    bound = bind_data(plan, cols)
    return plan, bound, build_kernel(bound), cols
end

function _fv_group_axis_reference(built, cols, u)
    th = ReactiveKernelsPPL.constrain(built.layout, u)
    c = vec(th.z) .* only(th.sd)
    mu = c[cols[:g]] .+ c[cols[:h]]
    return sum(logpdf.(Normal.(mu[cols[:rows]], 1), cols[:y]); init = 0.0) +
        logpdf(truncated(Normal(), 0, Inf), only(th.sd)) +
        sum(logpdf.(Normal(), th.z); init = 0.0) +
        u[1] # log Jacobian of the single positive scale (sd = exp(u[1]))
end

@testset "functions as values: absorbed declaration dependencies" begin
    for n in (0, 3, 7), chained in (false, true)
        _FV.POOLS[] = 0
        plan, bound, built, cols = _fv_group_axis_case(n; chained)
        @test any(a -> a.name === :gg, (plan.assignments..., plan.derived...))
        @test bound.columns[:gg] == vcat(cols[:g], cols[:h])
        @test built.layout.total == n + 1
        @test bound.n_obs == n
        @test _FV.POOLS[] == Int(chained)
        original = deepcopy(cols)
        u = [0.15 * sin(i) for i in 1:built.layout.total]
        q = prepare_sampler(built, bound, u; backend = _FV_BACKEND)
        for w in (u, u .+ 0.1)
            value, grad = sampler_value_and_gradient!(q, similar(w), w)
            reference(v) = _fv_group_axis_reference(built, cols, v)
            @test value ≈ reference(w)
            @test grad ≈ _fv_findiff(reference, w) rtol = 1e-5 atol = 1e-7
        end
        @test _FV.POOLS[] == Int(chained) # no second call during preparation/AD
        @test cols == original
        # refused: data cannot shadow the model's computed definition.
        @test_throws ContractValidationError bind_data(plan, merge(cols, Dict(:gg => [1])))
    end
    # Prior slots retain the same transitive dependencies as dimension slots.
    for declaration in (:(b ~ Normal(0, k)), :(b[1:3] .~ Normal.(0, k)))
        ast = quote
            k = bind_scale(x)
            $declaration
            reads = shifted(b .* k; by = 0.2)
            y .~ Normal.(reads, 1)
        end
        cols = Dict{Symbol,Any}(:x => [1.0, 2.0, 3.0], :y => [0.1, -0.2, 0.3])
        _FV.UNWRAPS[] = 0
        _, bound, built = _fv_build(ast, cols)
        @test bound.columns[:k] == 2.0
        @test _FV.UNWRAPS[] == 1
        u = fill(0.1, built.layout.total)
        q = prepare_sampler(built, bound, u; backend = _FV_BACKEND)
        reference(w) = begin
            th = ReactiveKernelsPPL.constrain(built.layout, w)
            b = th.b isa Number ? fill(th.b, 3) : th.b
            prior = th.b isa Number ? logpdf(Normal(0, 2), th.b) :
                sum(logpdf.(Normal(0, 2), th.b))
            sum(logpdf.(Normal.(2 .* b .+ 0.2, 1), cols[:y])) + prior
        end
        value, grad = sampler_value_and_gradient!(q, similar(u), u)
        @test value ≈ reference(u)
        @test grad ≈ _fv_findiff(reference, u) rtol = 1e-5 atol = 1e-7
        @test _FV.UNWRAPS[] == 1
    end
end

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
    plan = lower_rkppl(_fv_cum_model(), (:y, :c); conditioned = (:y, :c))
    # `cumsum`/`vcat` resolve in the model module (Main here) as GlobalRefs.
    # The cumulative simplex stays a named graph value; the extracted
    # observation column gathers from it.
    col = only(plan.derived)
    @test col.expr == :(cum[c])
    cumulative = only(a for a in plan.assignments if a.name === :cum)
    @test cumulative.expr == Expr(:call, GlobalRef(Main, :cumsum),
        Expr(:call, GlobalRef(Main, :vcat), 0.0, :zeta))
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
        end, (:y, :x); mod = _FV, conditioned = (:y, :x))
    t = only(a for a in p2.assignments if a.name === :t)
    @test t.expr.args[1] == GlobalRef(_FV, :affine_helper)

    # refused: the called function is absent from the model module (P6, 05oe96l).
    err = try
        lower_rkppl(quote
                m0 ~ Normal(0, 1)
                s ~ Exponential(1.0)
                t = no_such_helper(s)
                y .~ Normal.(m0, t)
            end, (:y,); mod = _FV, conditioned = (:y,))
        nothing
    catch e
        e
    end
    # refused: the first case calls an undeclared function; the later case has a cyclic graph (P6/P3, 05oe96l).
    @test err isa SurfaceLoweringError
    @test occursin("no_such_helper", err.message)
    @test occursin("not defined", err.message)

    # An undotted call takes the column whole; its first result element
    # can supply a scalar scale beside another observation read.
    err = try
        lower_rkppl(quote
                s ~ Exponential(1.0)
                m0 ~ Normal(0, 1)
                t = shifted(x; by = s)
                mu = m0 .+ x
                y .~ Normal.(mu, t[1])
            end, (:y, :x); mod = _FV, conditioned = (:y, :x))
        nothing
    catch e
        e
    end
    # capability: valid ordinary value composition (P8 1cmodra; todo `15lq8iu`).
    @test (err === nothing || throw(err))

    # An observation-length result can also locate the response directly.
    err = try
        lower_rkppl(quote
                s ~ Exponential(1.0)
                t = shifted(x; by = s)
                y .~ Normal.(t, 1.0)
            end, (:y, :x); mod = _FV, conditioned = (:y, :x))
        nothing
    catch e
        e
    end
    # capability: valid ordinary value composition (P8 1cmodra; todo `15lq8iu`).
    @test (err === nothing || throw(err))

    # Calling a model value is not a function call.
    # refused: calling the scalar parameter value `s(1.0)` is a Julia MethodError (P3)
    @test_throws SurfaceLoweringError lower_rkppl(quote
            m0 ~ Normal(0, 1)
            s ~ Exponential(1.0)
            t = s(1.0)
            y .~ Normal.(m0, t)
        end, (:y,); mod = _FV, conditioned = (:y,))
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
    # refused: a gather index must be integer data; the per-cell latent x_true is a parameter (user GO 0fzormv)
    @test_throws ContractValidationError lower_rkppl(gathered(:x_true,
            Expr(:macrocall, Symbol("@plate"), LineNumberNode(1),
                Expr(:for, Expr(:(=), :i, :(eachindex(x))), Expr(:block,
                    :(x_true[i] ~ Normal(0, 1)),
                    :(x[i] ~ Normal.(x_true[i], 0.5)))))),
        Tuple(keys(cols)); mod = _FV, conditioned = Tuple(keys(cols)))
    # So is a definition reading a parameter, however it is named.
    # refused: a gather index must be integer data; k = c .* b reads parameter b (user GO 0fzormv)
    @test_throws ContractValidationError lower_rkppl(
        gathered(:k, :(k = c .* b)), Tuple(keys(cols)); mod = _FV, conditioned = Tuple(keys(cols)))
    # A raw index column must hold integers (bind).
    real_idx = lower_rkppl(gathered(:x), Tuple(keys(cols)); mod = _FV, conditioned = Tuple(keys(cols)))
    # refused: gather index column x holds non-integer reals (non-integer index, user GO 0fzormv)
    @test_throws ContractValidationError bind_data(real_idx, cols)
    # An integer data column gathers (the corpus 97 shape).
    @test bind_data(lower_rkppl(gathered(:c), Tuple(keys(cols)); mod = _FV, conditioned = Tuple(keys(cols))),
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
        a = th.a
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
            end, (:y, :gx, :g); mod = _FV, conditioned = (:y, :gx, :g))
        nothing
    catch e
        e
    end
    # refused: the first case calls an undeclared function; the later case has a cyclic graph (P6/P3, 05oe96l).
    @test err isa SurfaceLoweringError
    # refused: a static value graph must be acyclic (single assignment, P3).
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
            end, (:y, :x); mod = _FV, conditioned = (:y, :x))
        bound = bind_data(plan, cols2)
        built = build_kernel(bound)
        th = ReactiveKernelsPPL.constrain(built.layout, u)
        a, b = th.a, th.b
        ll = sum(logpdf(Normal(a + b * x[i] / nrm, th.sigma), y[i])
            for i in eachindex(y))
        @test _fv_value(built, bound, :likelihood, u) ≈ ll
    end
end

# Calls to the module function `name` in a generated or prepared program.
function _fv_calls(ex, name::Symbol)
    ex isa Expr || return 0
    n = sum(a -> _fv_calls(a, name), ex.args; init = 0)
    if ex.head in (:call, :.) && !isempty(ex.args)
        f = ex.args[1]
        f isa QuoteNode && (f = f.value)
        fname = f isa GlobalRef ? f.name : f isa Function ? nameof(f) : f
        fname === name && (n += 1)
    end
    return n
end

@testset "functions as values: a shared model-level value is evaluated once" begin
    # An intercept plus a gathered column is a model-level value. Read
    # through a kernel or dotted value by two response locations, each
    # location used to inline its own copy of the whole chain, so the gather
    # and the value ran once per reader (snag `rkppl-scalar-int-acf7b8cf`).
    # The value is now one named definition that both locations read.
    g = [1, 1, 2, 2, 3, 3, 3]
    x = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0, 0.25]
    c = [1.1, 0.7, 1.3, 2.0, 0.9, 0.95, 1.4]
    y = [0.5, 1.0, 1.5, 2.0, 0.8, 1.2, 1.9]
    cols = Dict{Symbol,ColumnData}(:g => g, :x => x, :c => c, :y => y)
    u = [0.2, -0.3, 0.4, 0.1, -0.2, 0.3]
    for (value, kernels) in ((:(exp_value(x_)), 1), (:(exp.(x_)), 0))
        ast = quote
            s ~ Exponential(1.0)
            b ~ Normal(0, 1)
            a ~ Normal(0, 1)
            z[levels(g), 1:1] .~ Normal.(0, 1)
            r = column_gather(z, g, 1)
            x_ = a .+ r
            v = $value
            c .~ LogNormal.(log.(v), s)
            y .~ Normal.(b .* v .+ x, s)
        end
        _, bound, built = _fv_build(ast, cols)
        src = kernel_expr(bound, assign_layout(bound))
        @test _fv_calls(src, :column_gather) == 1
        @test _fv_calls(src, :exp_value) == kernels
        th = ReactiveKernelsPPL.constrain(built.layout, u)
        vv = exp.(th.a .+ vec(th.z)[g])
        ll = sum(logpdf.(LogNormal.(log.(vv), th.s), c)) +
            sum(logpdf.(Normal.(th.b .* vv .+ x, th.s), y))
        @test _fv_value(built, bound, :likelihood, u) ≈ ll
        kern = prepare_query(built, bound, :sampler)
        q = prepare_sampler(built, bound, u; backend = _FV_BACKEND)
        # The prepared program evaluates the gather and the value once too.
        prepared = sprint(print, model_view(built; bound, query = q).prepared)
        @test count("column_gather(", prepared) == 1
        @test count("exp.(", prepared) == 1
        val, grad = Base.invokelatest(ReactiveKernels.ad_value_and_gradient!,
            q.ad, similar(u), u)
        @test val ≈ Base.invokelatest(kern, u)
        @test isapprox(grad, _fv_findiff(w -> Base.invokelatest(kern, w), u);
            rtol = 1e-5, atol = 1e-7)
    end
end

# Value reads of the name `sym` in the body of a generated `@kernel`
# program; a bound row count (`_observation_rows(...)`) reads shapes only.
function _fv_reads(ex, sym::Symbol)
    ex === sym && return 1
    ex isa Expr || return 0
    if ex.head === :call && !isempty(ex.args)
        f = ex.args[1]
        name = f isa GlobalRef ? f.name :
            Meta.isexpr(f, :.) && f.args[end] isa QuoteNode ? f.args[end].value : f
        name === :_observation_rows && return 0
    end
    return sum(a -> _fv_reads(a, sym), ex.args; init = 0)
end

@testset "functions as values: a shared per-observation value is evaluated once" begin
    # A per-observation definition read by several locations is evaluated
    # once and read by name, as in Julia. Each location used to inline its
    # own copy of the definition (or the definition also stayed an affine
    # predictor beside an inlined copy), so it ran once per reader.
    g = [1, 2, 1, 3, 2]
    x = [0.1, 0.5, -0.3, 1.2, 0.7]
    c = [1.1, 0.4, 0.9, 2.0, 1.3]
    y = [0.2, 0.9, 0.1, 1.5, 0.8]
    cols = Dict{Symbol,ColumnData}(:g => g, :x => x, :c => c, :y => y)
    cases = (
        # A log-scale location and a scaled location read one value.
        (quote
            a ~ Normal(0, 1); b ~ Normal(0, 1); s ~ Exponential(1.0)
            v = exp.(a .+ b .* x)
            c .~ LogNormal.(log.(v), s)
            y .~ Normal.(2 .* v, s)
        end, src -> _fv_calls(src, :exp) == 1, function (th)
            v = exp.(th.a .+ th.b .* x)
            sum(logpdf.(LogNormal.(log.(v), th.s), c)) +
                sum(logpdf.(Normal.(2 .* v, th.s), y))
        end),
        # An affine predictor read by name and inlined into another
        # definition: computed once, so the data column is read once.
        (quote
            a ~ Normal(0, 1); b ~ Normal(0, 1); s ~ Exponential(1.0)
            eta = a .+ b .* x
            theta = 0.8 .+ abs.(eta)
            y .~ Normal.(theta, s)
            c .~ Normal.(0.8 .+ abs.(eta), s)
        end, src -> _fv_reads(src.args[2], :x) == 1, function (th)
            theta = 0.8 .+ abs.(th.a .+ th.b .* x)
            sum(logpdf.(Normal.(theta, th.s), y)) +
                sum(logpdf.(Normal.(theta, th.s), c))
        end),
        # A factor-coefficient alias read by two predictors: one gather.
        (quote
            a ~ Normal(0, 1); b ~ Normal(0, 1); s ~ Exponential(1.0)
            z[levels(g)] .~ Normal.(0, 1)
            zg = z[g]
            mu1 = a .+ zg
            mu2 = b .* zg
            y .~ Normal.(mu1, s)
            c .~ Normal.(mu2, s)
        end, src -> _fv_reads(src.args[2], :g) == 1, function (th)
            zg = th.z[g]
            sum(logpdf.(Normal.(th.a .+ zg, th.s), y)) +
                sum(logpdf.(Normal.(th.b .* zg, th.s), c))
        end),
        # A location inlines the value while the scale reads its name.
        (quote
            a ~ Normal(0, 1); b ~ Normal(0, 1)
            v = exp.(a .+ b .* x)
            y .~ Normal.(log.(v), v)
        end, src -> _fv_calls(src, :exp) == 1, function (th)
            v = exp.(th.a .+ th.b .* x)
            sum(logpdf.(Normal.(log.(v), v), y))
        end),
        # A composition read by two composed locations.
        (quote
            a ~ Normal(0, 1); b ~ Normal(0, 1); s ~ Exponential(1.0)
            eta = a .+ b .* x
            w = exp.(eta) .* x
            y .~ Normal.(2 .* w, s)
            c .~ Normal.(3 .* w, s)
        end, src -> _fv_calls(src, :exp) == 1, function (th)
            w = exp.(th.a .+ th.b .* x) .* x
            sum(logpdf.(Normal.(2 .* w, th.s), y)) +
                sum(logpdf.(Normal.(3 .* w, th.s), c))
        end),
    )
    for (ast, once, loglik) in cases
        _, bound, built = _fv_build(ast, cols)
        src = kernel_expr(bound, assign_layout(bound))
        @test once(src)
        u = collect(range(-0.3, 0.4; length = built.layout.total))
        th = ReactiveKernelsPPL.constrain(built.layout, u)
        @test _fv_value(built, bound, :likelihood, u) ≈ loglik(th)
        kern = prepare_query(built, bound, :sampler)
        q = prepare_sampler(built, bound, u; backend = _FV_BACKEND)
        val, grad = Base.invokelatest(ReactiveKernels.ad_value_and_gradient!,
            q.ad, similar(u), u)
        @test val ≈ Base.invokelatest(kern, u)
        @test isapprox(grad, _fv_findiff(w -> Base.invokelatest(kern, w), u);
            rtol = 1e-5, atol = 1e-7)
    end
end

# Whether the body of a generated `@kernel` program assigns `sym`.
_fv_assigns(ex, sym::Symbol) = ex isa Expr && (
    (ex.head === :(=) && (ex.args[1] === sym ||
        Meta.isexpr(ex.args[1], :(::)) && ex.args[1].args[1] === sym)) ||
    any(a -> _fv_assigns(a, sym), ex.args))

@testset "functions as values: a definition read by one location stays a named value" begin
    # As in Julia, a computed definition is one named value even when a
    # single location reads it; the location no longer folds it into its
    # own expression (user decision `0fbe312`). Densities are unchanged.
    g = [1, 2, 1, 3, 2]
    x = [0.1, 0.5, -0.3, 1.2, 0.7]
    y = [0.2, 0.9, 0.1, 1.5, 0.8]
    rows = [5, 4, 3, 2, 1]
    cols = Dict{Symbol,ColumnData}(:g => g, :x => x, :y => y, :rows => rows)
    cases = (
        # A model-level scalar.
        (quote
            a ~ Normal(0, 1); s ~ Exponential(1.0)
            t = exp(a) + 1
            y .~ Normal.(t .* x, s)
        end, (:t,), th -> sum(logpdf.(Normal.((exp(th.a) + 1) .* x, th.s), y))),
        # An affine part beside a factor term.
        (quote
            a ~ Normal(0, 1); b ~ Normal(0, 1); s ~ Exponential(1.0)
            z[levels(g)] .~ Normal.(0, 1)
            eta = a .+ b .* x
            mu = eta .+ z[g]
            y .~ Normal.(mu, s)
        end, (:eta,), th -> sum(logpdf.(Normal.(th.a .+ th.b .* x .+ th.z[g], th.s), y))),
        # A chain of compositions.
        (quote
            a ~ Normal(0, 1); b ~ Normal(0, 1); s ~ Exponential(1.0)
            eta = a .+ b .* x
            w = exp.(eta) .* x
            y .~ Normal.(2 .* w, s)
        end, (:eta, :w), th -> sum(logpdf.(Normal.(2 .* exp.(th.a .+ th.b .* x) .* x, th.s), y))),
        # A gather whose index is a data-only definition, read whole by a
        # module call: the index binds as data and `v` is one value.
        (quote
            s ~ Exponential(1.0)
            z[1:3, 1:1] .~ Normal.(0, 1)
            i = group_index(g, g)
            v = z[i, 1]
            reads = read_rows(v, rows)
            y .~ Normal.(reads, s)
        end, (:v,), th -> sum(logpdf.(Normal.(vec(th.z)[g][rows], th.s), y))),
    )
    for (ast, names, loglik) in cases
        _, bound, built = _fv_build(ast, cols)
        src = kernel_expr(bound, assign_layout(bound))
        @test all(nm -> _fv_assigns(src.args[2], nm), names)
        u = collect(range(-0.3, 0.4; length = built.layout.total))
        th = ReactiveKernelsPPL.constrain(built.layout, u)
        @test _fv_value(built, bound, :likelihood, u) ≈ loglik(th)
        kern = prepare_query(built, bound, :sampler)
        q = prepare_sampler(built, bound, u; backend = _FV_BACKEND)
        val, grad = Base.invokelatest(ReactiveKernels.ad_value_and_gradient!,
            q.ad, similar(u), u)
        @test val ≈ Base.invokelatest(kern, u)
        @test isapprox(grad, _fv_findiff(w -> Base.invokelatest(kern, w), u);
            rtol = 1e-5, atol = 1e-7)
    end
end

@testset "functions as values: a positional read of a call's result keeps its shape" begin
    # `pr[1]` of `pr = f(x, b)` is whatever `f` returns at that position —
    # here a per-observation vector or a scalar total — so it is a
    # model-level value of unknown shape, never a proven scalar offset.
    # Named, inline and in-response spellings lower alike (snag
    # `inline-gather-br-28d41671`: every one failed `bind_data` with
    # "term references missing column _rkppl_leaf_1").
    cols = _fv_cols()
    x = cols[:x]
    y = cols[:y]
    path(b) = cumsum(b .* x)
    total(b) = sum(b .* x)
    cases = (
        (quote
            b ~ Normal(0, 1); s ~ Exponential(1.0)
            pr = pair_reads(x, b)
            loc = pr[1]
            y .~ Normal.(loc, s)
        end, th -> sum(logpdf.(Normal.(path(th.b), th.s), y))),
        (quote
            b ~ Normal(0, 1); s ~ Exponential(1.0)
            loc = pair_reads(x, b)[1]
            y .~ Normal.(loc, s)
        end, th -> sum(logpdf.(Normal.(path(th.b), th.s), y))),
        (quote
            b ~ Normal(0, 1); s ~ Exponential(1.0)
            y .~ Normal.(pair_reads(x, b)[1], s)
        end, th -> sum(logpdf.(Normal.(path(th.b), th.s), y))),
        (quote
            b ~ Normal(0, 1); s ~ Exponential(1.0)
            pr = pair_reads(x, b)
            m = pr[2]
            y .~ Normal.(m, s)
        end, th -> sum(logpdf.(Normal.(total(th.b), th.s), y))),
        (quote
            a ~ Normal(0, 1); b ~ Normal(0, 1)
            pr = pair_reads(x, b)
            mu = a .+ pr[1]
            y .~ Normal.(mu, exp(pr[2]))
        end, th -> sum(logpdf.(Normal.(th.a .+ path(th.b), exp(total(th.b))), y))),
        (quote
            a ~ Normal(0, 1); s ~ Exponential(1.0)
            d = data_pair(x)
            mu = a .+ d[2]
            y .~ Normal.(mu, s)
        end, th -> sum(logpdf.(Normal.(th.a .+ 2 .* x, th.s), y))),
        (quote
            a ~ Normal(0, 1); s ~ Exponential(1.0)
            d = data_pair(x)
            mu = d[1] .* a
            y .~ Normal.(mu, s)
        end, th -> sum(logpdf.(Normal.(sum(x) * th.a, th.s), y))),
    )
    for (ast, loglik) in cases
        original = deepcopy(cols)
        _, bound, built = _fv_build(ast, cols)
        u = collect(range(-0.3, 0.4; length = built.layout.total))
        th = ReactiveKernelsPPL.constrain(built.layout, u)
        @test _fv_value(built, bound, :likelihood, u) ≈ loglik(th)
        kern = prepare_query(built, bound, :sampler)
        q = prepare_sampler(built, bound, u; backend = _FV_BACKEND)
        val, grad = Base.invokelatest(ReactiveKernels.ad_value_and_gradient!,
            q.ad, similar(u), u)
        @test val ≈ Base.invokelatest(kern, u)
        @test isapprox(grad, _fv_findiff(w -> Base.invokelatest(kern, w), u);
            rtol = 1e-5, atol = 1e-7)
        @test cols == original
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
    a, b = th.a, th.b
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
    plan = lower_rkppl(ast, (:y, :X, :x); mod = _FV, conditioned = (:y, :X, :x))
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
    b = th.b
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
    # refused: caller column vx collides with the computed definition vx (single assignment; names must not collide with data names)
    @test_throws ContractValidationError bind_data(plan, cols2)
end

@testset "functions as values: data-only part of a parameter-dependent call" begin
    # A data-only definition a parameter-dependent call consumes inlines
    # into that call at lowering. The generator emits it as its own
    # statement, so preparation evaluates it once and the gradient never
    # differentiates it — named or inline. Here it is a guard with an
    # interpolated message unwrapping a multi-field schedule (snag
    # `interpolated-err-41772e14`).
    sched = (; a = [1.0, 2.0], k = [1, 2], f = [0.5, 0.25])
    snapshot = deepcopy(sched)
    cols = Dict{Symbol,AbstractVector}(:y => [0.1, 0.4, -0.2, 0.3],
        :oi => [1, 2, 2, 1], :s => [sched])
    named = quote
        sigma ~ Exponential(1.0)
        b ~ Normal(0, 1)
        sc = take_schedule(s)
        reads = schedule_reads(sc, b)
        y .~ Normal.(reads[oi], sigma)
    end
    inline = quote
        sigma ~ Exponential(1.0)
        b ~ Normal(0, 1)
        reads = schedule_reads(take_schedule(s), b)
        y .~ Normal.(reads[oi], sigma)
    end
    u = [0.2, -0.3]
    for ast in (named, inline)
        plan, bound, built = _fv_build(ast, cols)
        th = ReactiveKernelsPPL.constrain(built.layout, u)
        r = th.b .* sched.a .+ sched.k
        ll = sum(logpdf(Normal(r[cols[:oi][i]], th.sigma), cols[:y][i])
            for i in 1:4)
        @test _fv_value(built, bound, :likelihood, u) ≈ ll
        _FV.UNWRAPS[] = 0
        q = prepare_sampler(built, bound, u; backend = _FV_BACKEND)
        @test _FV.UNWRAPS[] == 1               # folded once by preparation
        g = similar(u)
        for w in (u, [0.7, 0.1])
            Base.invokelatest(ReactiveKernels.ad_value_and_gradient!, q.ad,
                g, w)
        end
        @test _FV.UNWRAPS[] == 1               # never recomputed in-graph
        @test isapprox(g,
            _fv_findiff(w -> Base.invokelatest(q.kernel, w), [0.7, 0.1]);
            rtol = 1e-5, atol = 1e-7)
        @test only(cols[:s]) == snapshot       # bound data untouched
    end
end

@testset "functions as values: prepared records beside declared arrays" begin
    # The same data-only call must fold at preparation when a declared
    # array makes lowering retain its named assignment. Records are leaf
    # function inputs, with no observation axis or numeric shape guard.
    for n in (3, 7), kind in (:namedtuple, :tuple),
            weights in (:data, :vector, :matrix), spelling in (:named, :alias, :inline)
        a = collect(range(0.5, 1.5; length = n))
        k = collect(1:n)
        sched = kind === :namedtuple ? (; a, k) : (a, k)
        oi = [n, 1, n, 2]
        cols = Dict{Symbol,ColumnData}(:s => [sched], :oi => oi,
            :y => [0.2, -0.1, 0.3, 0.5])
        snapshot = deepcopy(cols)
        ast = quote
            b ~ Normal(0, 1)
            y .~ Normal.(reads[oi], 1)
        end
        declarations = weights === :vector ? :(w[1:$n] .~ Normal.(0, 1)) :
            weights === :matrix ? :(w[1:$n, 1:1] .~ Normal.(0, 1)) : nothing
        declarations === nothing ? (cols[:w] = fill(0.2, n)) :
            pushfirst!(ast.args, declarations)
        definitions = spelling === :inline ? [:(reads = weighted_reads(take_schedule(s), w, b))] :
            spelling === :alias ? [:(sc = take_schedule(s)), :(sc2 = identity(sc)),
                :(reads = weighted_reads(sc2, w, b))] :
            [:(sc = take_schedule(s)), :(reads = weighted_reads(sc, w, b))]
        # Place definitions before the response (declaration order does
        # not affect data staging or dependency order).
        splice!(ast.args, length(ast.args):length(ast.args),
            vcat(definitions, ast.args[end:end]))
        _FV.UNWRAPS[] = 0
        plan, bound, built = _fv_build(ast, cols)
        @test _FV.UNWRAPS[] == 0
        @test !haskey(bound.columns, :sc)
        @test bound.n_obs == length(oi)
        u = [0.1 * i - 0.2 for i in 1:built.layout.total]
        q = prepare_sampler(built, bound, u; backend = _FV_BACKEND)
        @test _FV.UNWRAPS[] == 1
        function reference(z)
            th = ReactiveKernelsPPL.constrain(built.layout, z)
            w = weights === :data ? cols[:w] : vec(th.w)
            r = w .* a .+ k .+ th.b
            ll = sum(logpdf(Normal(r[i], 1), y) for (i, y) in zip(oi, cols[:y]))
            lp = logpdf(Normal(), th.b)
            weights === :data || (lp += sum(logpdf.(Normal(), w)))
            return ll + lp
        end
        for z in (u, reverse(u))
            g = similar(z)
            value, _ = Base.invokelatest(ReactiveKernels.ad_value_and_gradient!, q.ad, g, z)
            @test value ≈ reference(z)
            @test isapprox(g, _fv_findiff(reference, z); rtol = 1e-5, atol = 1e-7)
        end
        @test _FV.UNWRAPS[] == 1
        @test cols[:y] == snapshot[:y]
        @test cols[:oi] == snapshot[:oi]
        @test a == (kind === :tuple ? snapshot[:s][1][1] : snapshot[:s][1].a)
        @test k == (kind === :tuple ? snapshot[:s][1][2] : snapshot[:s][1].k)
        # refused: supplied data cannot shadow a computed assignment.
        if any(a -> a.name === :sc, plan.assignments)
            shadowed = copy(cols)
            shadowed[:sc] = [1.0]
            @test_throws ContractValidationError bind_data(plan, shadowed)
        end
    end
end

@testset "functions as values: shared bind and preparation consumers" begin
    cols = Dict{Symbol,ColumnData}(:x => [0.5, 1.0, 1.5], :oi => [3, 1, 3, 2],
        :y => [0.2, -0.1, 0.3, 0.5])
    prior = quote
        k = bind_scale(x)
        b ~ Normal(0, k)
        w[1:3] .~ Normal.(0, 1)
        reads = shifted(scaled(w, k); by = b)
        y .~ Normal.(reads[oi], 1)
    end
    dimension = quote
        M = bind_matrix(x)
        b ~ Normal(0, 1)
        w[axes(M, 2)] .~ Normal.(0, 1)
        reads = matrix_reads(M, w, b)
        y .~ Normal.(reads[oi], 1)
    end
    for (ast, name) in ((prior, :k), (dimension, :M))
        _FV.UNWRAPS[] = 0
        _, bound, built = _fv_build(ast, cols)
        @test _FV.UNWRAPS[] == 1
        @test haskey(bound.columns, name)
        u = [0.1 * i - 0.2 for i in 1:built.layout.total]
        q = prepare_sampler(built, bound, u; backend = _FV_BACKEND)
        g = similar(u)
        value, _ = Base.invokelatest(ReactiveKernels.ad_value_and_gradient!, q.ad, g, u)
        @test isfinite(value)
        @test isapprox(g, _fv_findiff(z -> Base.invokelatest(q.kernel, z), u);
            rtol = 1e-5, atol = 1e-7)
        @test _FV.UNWRAPS[] == 1
        @test cols[:x] == [0.5, 1.0, 1.5]
    end
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
    plan = lower_rkppl(ast, (:y, :X, :x); mod = FunctionsAsValuesBare, conditioned = (:y, :X, :x))
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
    a, b = th.a, th.b
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
    plan = lower_rkppl(ast, (:y, :g, :gx); mod = _FV, conditioned = (:y, :g, :gx))
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
    # A gather takes the gathered column whole, too: the per-group
    # covariate needs no call.
    gplan = lower_rkppl(quote
            sigma ~ Exponential(1.0)
            b ~ Normal(0, 1)
            y .~ Normal.(b .* gx[g], sigma)
        end, (:y, :g, :gx); mod = _FV, conditioned = (:y, :g, :gx))
    gbound = bind_data(gplan, cols)
    @test gbound.n_obs == 6
    gbuilt = build_kernel(gbound)
    for u in ([0.3, -0.2], [-1.1, 0.7])
        th = ReactiveKernelsPPL.constrain(gbuilt.layout, u)
        b = th.b
        ll = sum(logpdf(Normal(b * cols[:gx][cols[:g][i]], th.sigma),
            cols[:y][i]) for i in 1:6)
        @test _fv_value(gbuilt, gbound, :likelihood, u) ≈ ll
    end
    # Whole-ness follows the value, not the spelling: a definition used
    # only whole (gathered, or passed to calls) reads its columns whole,
    # named or inlined.
    for body in (quote
                sigma ~ Exponential(1.0)
                b ~ Normal(0, 1)
                v = b .* gx
                y .~ Normal.(v[g], sigma)
            end, quote
                sigma ~ Exponential(1.0)
                b ~ Normal(0, 1)
                y .~ Normal.((b .* gx)[g], sigma)
            end)
        vbound = bind_data(lower_rkppl(body, (:y, :g, :gx); mod = _FV, conditioned = (:y, :g, :gx)), cols)
        vbuilt = build_kernel(vbound)
        for u in ([0.3, -0.2], [-1.1, 0.7])
            th = ReactiveKernelsPPL.constrain(vbuilt.layout, u)
            ll = sum(logpdf(Normal(th.b * cols[:gx][cols[:g][i]], th.sigma),
                cols[:y][i]) for i in 1:6)
            @test _fv_value(vbuilt, vbound, :likelihood, u) ≈ ll
        end
    end
    xcols = Dict{Symbol,ColumnData}(:y => cols[:y], :gx => cols[:gx],
        :x => [0.3, -0.1, 0.2, 0.8, -0.5, 0.1])
    kept = lower_rkppl(quote
            b ~ Normal(0, 1)
            sigma ~ Exponential(1.0)
            m0 ~ Normal(0, 1)
            sc = b .* gx
            s_eff = sigma * exp(first(sc))
            y .~ Normal.(m0 .+ x, s_eff)
        end, (:y, :x, :gx); mod = _FV, conditioned = (:y, :x, :gx))
    @test any(a -> a.name === :sc, kept.assignments) # named model-level value
    kbound = bind_data(kept, xcols)
    @test kbound.n_obs == 6
    kbuilt = build_kernel(kbound)
    for u in ([0.3, -0.2, 0.1], [-1.1, 0.7, 0.4])
        th = ReactiveKernelsPPL.constrain(kbuilt.layout, u)
        s_eff = th.sigma * exp(th.b * xcols[:gx][1])
        ll = sum(logpdf(Normal(th.m0 + xcols[:x][i], s_eff),
            xcols[:y][i]) for i in 1:6)
        @test _fv_value(kbuilt, kbound, :likelihood, u) ≈ ll
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
        end, (:y, :g, :gx); mod = _FV, conditioned = (:y, :g, :gx))
    # refused: gx is read per observation (`v[g] .+ gx`), so it is observation-aligned; its length 3 against 6 observations is a length mismatch (wrong data; a Julia DimensionMismatch, P3)
    @test_throws ContractValidationError bind_data(mixed, cols)
    # So does a name any other plan slot holds (here: response weights).
    weighted_plan = lower_rkppl(quote
            sigma ~ Exponential(1.0)
            b ~ Normal(0, 1)
            gx_m = as_vector(gx)
            v = b .* gx_m
            y .~ weighted.(Normal.(v[g], sigma), gx)
        end, (:y, :g, :gx); mod = _FV, conditioned = (:y, :g, :gx))
    # refused: weights gx (length 3) do not match the 6 observations (wrong data: length mismatch; a broadcast DimensionMismatch, P3)
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
    # The parameter-dependent call takes the columns directly, too: each is
    # read only as a whole value, so lowering admits the call.
    _, dbound, dbuilt = _fv_build(quote
            sigma ~ Exponential(1.0)
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            k ~ Normal(0, 1)
            reads = decayed_events(g, t, eg, et, ea, w, a .+ b .* gx, k)
            y .~ Normal.(reads[oi], sigma)
        end, cols)
    @test dbound.n_obs == 8
    dkern = prepare_query(dbuilt, dbound, :sampler)
    # A definition passed to several calls is read whole as well.
    @test bind_data(lower_rkppl(quote
            sigma ~ Exponential(1.0)
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            k ~ Normal(0, 1)
            scale = a .+ b .* gx
            r1 = decayed_events(g, t, eg, et, ea, w, scale, k)
            r2 = decayed_events(g, t, eg, et, ea, w, scale, 2 * k)
            y .~ Normal.(r1[oi], sigma)
            y2 .~ Normal.(r2[oi], sigma)
        end, (Tuple(keys(cols))..., :y2); mod = _FV, conditioned = (Tuple(keys(cols))..., :y2)),
        merge(cols, Dict{Symbol,ColumnData}(:y2 => reverse(cols[:y])))).n_obs == 8
    # The same column can pass whole and also supply observation weights.
    # capability: calls share observation data with weights (todo `15lq8iu`).
    @test (lower_rkppl(quote
            sigma ~ Exponential(1.0)
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            k ~ Normal(0, 1)
            scale = a .+ b .* as_vector(gx)
            reads = decayed_events(g, t, eg, et, ea, w, scale, k)
            y .~ weighted.(Normal.(reads[oi], sigma), t)
        end, Tuple(keys(cols)); mod = _FV, conditioned = Tuple(keys(cols))); true)
    # A gather by a per-observation index is observation-aligned, even of
    # a whole column.
    # capability: a call consumes a gather and its result is gathered again
    # using ordinary Julia values (todo `15lq8iu`).
    @test (lower_rkppl(quote
            s ~ Exponential(1.0)
            m0 ~ Normal(0, 1)
            c ~ Normal(0, 1)
            t2 = shifted(gx[g]; by = s)
            mu = m0 .+ c .* g .+ t2[oi]
            y .~ Normal.(mu, 1.0)
        end, Tuple(keys(cols)); mod = _FV, conditioned = Tuple(keys(cols))); true)
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
        @test Base.invokelatest(dkern, u) ≈ v
        dq = prepare_sampler(dbuilt, dbound, u; backend = _FV_BACKEND)
        _, dg = Base.invokelatest(ReactiveKernels.ad_value_and_gradient!,
            dq.ad, similar(u), u)
        @test dg ≈ g
    end
end

@testset "functions as values: elementwise math over level-sized arrays" begin
    # A declaration's whole-value shape does not depend on whether it
    # could instead be used as a predictor coefficient. Both declarations
    # below have the same values in sorted level order. The helper's
    # result is an ordinary vector, gathered by integer positions.
    for (K, n) in ((2, 7), (5, 13))
        cols = Dict(:y => [0.3 * sin(i) for i in 1:n],
            :g => [mod1(i, K) for i in 1:n])
        builds = map((:(levels(g)), :(1:length(levels(g))))) do axis
            _fv_build(quote
                s ~ HalfNormal(1)
                z[$axis] .~ Normal.(0, 1)
                v = exp.(s .* z)
                w = scaled(v, 2.0)
                a ~ Normal(0, 1)
                mu = a .+ w[g]
                y .~ Normal.(mu, 1.0)
            end, cols)
        end
        for (_, bound, built) in builds
            @test only(bound.array_parameters).name === :z
            @test built.layout.total == K + 2
            u = [0.2 * cos(i) for i in 1:built.layout.total]
            th = constrain(built.layout, u)
            want = sum(logpdf.(Normal.(th.a .+
                2 .* exp.(th.s .* th.z)[cols[:g]], 1.0), cols[:y]))
            @test _fv_value(built, bound, :likelihood, u) ≈ want
            q = prepare_sampler(built, bound, u; backend = _FV_BACKEND)
            kern = prepare_query(built, bound, :sampler)
            val, grad = sampler_value_and_gradient!(q, similar(u), u)
            @test val ≈ Base.invokelatest(kern, u)
            @test isapprox(grad,
                _fv_findiff(v -> Base.invokelatest(kern, v), u);
                rtol = 1e-5, atol = 1e-7)
        end
        u = [0.2 * cos(i) for i in 1:K + 2]
        @test _fv_value(builds[1][3], builds[1][2], :sampler, u) ≈
            _fv_value(builds[2][3], builds[2][2], :sampler, u)
        # Naming the elementwise intermediate keeps its shape and density.
        _, ibound, ibuilt = _fv_build(quote
            s ~ HalfNormal(1)
            z[levels(g)] .~ Normal.(0, 1)
            w = scaled(exp.(s .* z), 2.0)
            a ~ Normal(0, 1)
            mu = a .+ w[g]
            y .~ Normal.(mu, 1.0)
        end, cols)
        @test _fv_value(ibuilt, ibound, :sampler, u) ≈
            _fv_value(builds[1][3], builds[1][2], :sampler, u)
    end
end

@testset "functions as values: whole design-sized arrays beside coefficients" begin
    # `axes(B, 2)` over an hcat design has the same coefficient-capable
    # role as `levels(g)`. Whole z is an array while B * b stays affine.
    cols = _fv_cols()
    cols[:x2] = [0.25 * cos(i) for i in eachindex(cols[:y])]
    cols[:c] = [mod1(i, 2) for i in eachindex(cols[:y])]
    _, bound, built = _fv_build(quote
        B = hcat(x, x2)
        s ~ HalfNormal(1)
        z[axes(B, 2)] .~ Normal.(0, 1)
        b[axes(B, 2)] .~ Normal.(0, 1)
        v = exp.(s .* z)
        w = scaled(v, 2.0)
        a ~ Normal(0, 1)
        mu = a .+ B * b .+ w[c]
        y .~ Normal.(mu, 1.0)
    end, cols)
    @test Set(p.name for p in bound.array_parameters) == Set((:z, :b))
    @test any(t -> t.kind === ReactiveKernelsPPL.MatrixTerm,
        only(bound.predictors).terms)
    u = [0.2 * cos(i) for i in 1:built.layout.total]
    th = constrain(built.layout, u)
    # The affine layout packs the intercept followed by the two slopes.
    want = sum(logpdf.(Normal.(th.a .+
        hcat(cols[:x], cols[:x2]) * th.b .+
        2 .* exp.(th.s .* th.z)[cols[:c]], 1.0), cols[:y]))
    @test _fv_value(built, bound, :likelihood, u) ≈ want
end

@testset "functions as values: built-in broadcasts over module values" begin
    # Built-in math over a module value keeps its model-level shape, even
    # when the result then meets a declared array. Naming an intermediate
    # or moving the same leaf math to a helper does not change the density.
    shapes = Tuple{Int,Int}[]
    for (n, K) in ((7, 2), (13, 5))
        cols = Dict{Symbol,ColumnData}(
            :y => [0.3 * sin(i) for i in 1:n],
            :B => [cos(i + j) for i in 1:n, j in 1:K],
            :lam => [0.2 * j for j in 1:K])
        bodies = (quote
                L = squared(lam)
                S = sd .* exp.(-0.25 .* L) .* w
            end, quote
                L = squared(lam)
                E = exp.(-0.25 .* L)
                S = sd .* E .* w
            end, quote
                S = sd .* exp.(-0.25 .* squared(lam)) .* w
            end, quote
                L = squared(lam)
                S = decay_weights(L, sd) .* w
            end, quote
                L = squared(lam)
                S = sd .* Base.exp.(-0.25 .* L) .* w
            end, quote
                L = squared(lam)
                S = sd .* sqrt.(exp.(-0.5 .* L)) .* w
            end)
        builds = map(bodies) do body
            _fv_build(Expr(:block,
                :(w[axes(B, 2)] .~ Normal.(0, 1)),
                :(sd ~ HalfNormal(1)), body.args...,
                :(a ~ Normal(0, 1)), :(mu = a .+ B * S),
                :(y .~ Normal.(mu, 1.0))), cols)
        end
        for (_, bound, built) in builds
            @test bound.n_obs == n
            @test built.layout.total == K + 2
            u = [0.2 * cos(i) for i in 1:built.layout.total]
            th = constrain(built.layout, u)
            mu = th.a .+ cols[:B] *
                (th.sd .* exp.(-0.25 .* cols[:lam].^2) .* th.w)
            want = sum(logpdf.(Normal.(mu, 1.0), cols[:y]))
            @test _fv_value(built, bound, :likelihood, u) ≈ want
            @test _fv_value(built, bound, :sampler, u) ≈
                _fv_value(builds[1][3], builds[1][2], :sampler, u)
        end
        _, bound, built = first(builds)
        push!(shapes, (length(built.spec.graph.values),
            length(built.spec.graph.recipes)))
        u = [0.2 * cos(i) for i in 1:built.layout.total]
        kern = prepare_query(built, bound, :sampler)
        q = prepare_sampler(built, bound, u; backend = _FV_BACKEND)
        value, grad = sampler_value_and_gradient!(q, similar(u), u)
        @test value ≈ Base.invokelatest(kern, u)
        @test isapprox(grad,
            _fv_findiff(v -> Base.invokelatest(kern, v), u);
            rtol = 1e-5, atol = 1e-7)
    end
    @test shapes[1] == shapes[2]

    # Model values may also be gathered before broadcasting. The index
    # supplies the observation axis; it must survive the shared classifier.
    cols = Dict{Symbol,ColumnData}(:y => [0.1, -0.2, 0.3, 0.4],
        :g => [1, 2, 1, 2], :lam => [0.2, 0.5])
    _, bound, built = _fv_build(quote
        L = squared(lam)
        a ~ Normal(0, 1)
        mu = a .+ exp.(-0.25 .* L[g])
        y .~ Normal.(mu, 1.0)
    end, cols)
    u = [0.3]
    th = constrain(built.layout, u)
    want = sum(logpdf.(Normal.(th.a .+
        exp.(-0.25 .* cols[:lam][cols[:g]].^2), 1.0), cols[:y]))
    @test _fv_value(built, bound, :likelihood, u) ≈ want
end

@testset "functions as values: a call wider than 32 arguments (Enzyme vs FD)" begin
    # A parameter-dependent call with a data vector plus forty scalar
    # parameters reaches the kernel as one fused op of 41+ inputs. Its
    # reverse gradient used to fail with `EnzymeRuntimeActivityError` once
    # the op had 32 inputs or more (snag `rkppl-module-cal-79cad594`): the
    # op's entry call splatted its arguments, and past 32 that splat stays a
    # dynamic call holding the constant data next to active parameters.
    cols = Dict{Symbol,ColumnData}(
        :age => [0.0, 0.1, 0.2], :oi => [1, 2, 3, 3],
        :y => [0.1, 0.2, 0.3, 0.4])
    names = _FV.WIDE_NAMES
    priors = [:($b ~ Normal(0.0, 1.0)) for b in names]
    for argument in (:age, :(c .* age), :(a .+ c .* age))
        ast = Expr(:block,
            :(sigma ~ Exponential(1.0)), :(a ~ Normal(0.0, 1.0)),
            :(c ~ Normal(0.0, 1.0)), priors...,
            :(reads = wide_reads($argument, $(names...))),
            :(y .~ Normal.(reads[oi], sigma)))
        _, bound, built = _fv_build(ast, cols)
        kern = prepare_query(built, bound, :sampler)
        n = length(coordinate_names(built.layout))
        @test n == 43
        u = [0.05 * (-1)^i * i / n for i in 1:n]
        q = prepare_sampler(built, bound, u; backend = _FV_BACKEND)
        v, g = Base.invokelatest(ReactiveKernels.ad_value_and_gradient!,
            q.ad, similar(u), u)
        @test v ≈ Base.invokelatest(kern, u)
        @test isapprox(g, _fv_findiff(w -> Base.invokelatest(kern, w), u);
            rtol = 1e-5, atol = 1e-7)
    end
end

# `f(x, b)[k]` reads element k of the call's result, exactly as the named
# `r = f(x, b); r[k]` does; the element has the function's own shape, so a
# vector element is one value per observation and a number broadcasts.
# Read once, inline or named, live or data-only (snag rkppl-indexed-mu-0f0ed295).
@testset "functions as values: a positional read of a call's result" begin
    cols = _fv_cols()
    x, y = cols[:x], cols[:y]
    x0, y0 = copy(x), copy(y)
    cases = (
        (quote
            b ~ Normal(0, 2)
            sigma ~ Exponential(1.0)
            mu = pair_reads(x, b)[1]
            y .~ Normal.(mu, sigma)
        end, quote
            b ~ Normal(0, 2)
            sigma ~ Exponential(1.0)
            r = pair_reads(x, b)
            mu = r[1]
            y .~ Normal.(mu, sigma)
        end, th -> th.b .* cumsum(x)),
        (quote
            a ~ Normal(0, 5)
            b ~ Normal(0, 2)
            sigma ~ Exponential(1.0)
            mu = a .* x .+ pair_reads(x, b)[2]
            y .~ Normal.(mu, sigma)
        end, quote
            a ~ Normal(0, 5)
            b ~ Normal(0, 2)
            sigma ~ Exponential(1.0)
            r = pair_reads(x, b)
            mu = a .* x .+ r[2]
            y .~ Normal.(mu, sigma)
        end, th -> th.a .* x .+ th.b * sum(x)),
        (quote
            a ~ Normal(0, 5)
            sigma ~ Exponential(1.0)
            mu = a .+ data_pair(x)[2]
            y .~ Normal.(mu, sigma)
        end, quote
            a ~ Normal(0, 5)
            sigma ~ Exponential(1.0)
            d = data_pair(x)
            mu = a .+ d[2]
            y .~ Normal.(mu, sigma)
        end, th -> th.a .+ 2 .* x),
    )
    for (inline_ast, named_ast, location) in cases
        inline = _fv_build(inline_ast, cols)
        named = _fv_build(named_ast, cols)
        layout = inline[3].layout
        @test coordinate_names(layout) == coordinate_names(named[3].layout)
        n = length(coordinate_names(layout))
        u = collect(range(-0.4, 0.3; length = n))
        th = ReactiveKernelsPPL.constrain(layout, u)
        # sigma = exp(u_sigma): its log Jacobian is u_sigma.
        prior = (haskey(th, :a) ? logpdf(Normal(0, 5), th.a) : 0.0) +
            (haskey(th, :b) ? logpdf(Normal(0, 2), th.b) : 0.0) +
            logpdf(Exponential(1.0), th.sigma) +
            u[findfirst(==(:sigma), coordinate_names(layout))]
        expected = prior + sum(logpdf.(Normal.(location(th), th.sigma), y))
        @test _fv_value(inline[3], inline[2], :sampler, u) ≈ expected rtol = 1e-12
        @test _fv_value(named[3], named[2], :sampler, u) ≈ expected rtol = 1e-12
        q = prepare_sampler(inline[3], inline[2], u; backend = _FV_BACKEND)
        kern = prepare_query(inline[3], inline[2], :sampler)
        _, g = Base.invokelatest(ReactiveKernels.ad_value_and_gradient!, q.ad,
            similar(u), u)
        @test isapprox(g, _fv_findiff(w -> Base.invokelatest(kern, w), u);
            rtol = 1e-5, atol = 1e-7)
        @test u == collect(range(-0.4, 0.3; length = n))
    end
    @test x == x0 && y == y0
end
