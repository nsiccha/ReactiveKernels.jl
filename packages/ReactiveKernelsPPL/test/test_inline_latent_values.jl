using Test, ReactiveKernelsPPL
import Distributions as D

# An inline distribution argument that reads a per-cell latent lowers as
# its named twin (`w = 0.8 .+ abs.(th)`, then `Weibull.(1.4, w)`): naming a
# subexpression never changes legality or the density (rkppl-use §2/§9).
# Weibull and Gamma scales and the first Beta shape read their argument as
# a value; Normal-family locations compose it with an affine predictor.
# Helpers `_check_gradient` come from test_generator.jl.

const _ILV_DATA = Dict(
    :x => [0.1, 0.5, -0.3, 1.2, 0.7],
    :y => [0.2, 0.9, 0.1, 1.5, 0.8],
    :c => [1.1, 0.4, 0.9, 2.0, 1.3],
    :p => [0.2, 0.4, 0.5, 0.7, 0.9],
)
const _ILV_POINT = (s = 0.8, a = 0.3, b = -0.4,
    th = [0.5, -0.7, 0.2, 1.1, -0.3])

function _ilv_model(stmts...)
    ex = quote
        s ~ Exponential(1)
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        eta = a .+ b .* x
        @plate for i in eachindex(y)
            th[i] ~ Normal(0, 1)
            y[i] ~ Normal(th[i] + x[i], s)
        end
        v = 2 .* th
    end
    append!(ex.args, stmts)
    return Base.remove_linenums!(ex)
end

# Independent density at constrained `q`, before the response under test.
function _ilv_base_logpdf(q, data)
    return D.logpdf(D.Exponential(1), q.s) + D.logpdf(D.Normal(), q.a) +
        D.logpdf(D.Normal(), q.b) + sum(D.logpdf.(D.Normal(), q.th)) +
        sum(D.logpdf.(D.Normal.(q.th .+ data[:x], q.s), data[:y]))
end

# (label, response column, response for an argument w, its density, value)
const _ILV_SLOTS = [
    ("Weibull scale", :c, w -> :(c .~ Weibull.(1.4, $w)),
        (v, d) -> sum(D.logpdf.(D.Weibull.(1.4, v), d[:c]))),
    ("Gamma scale", :c, w -> :(c .~ Gamma.(2.0, $w)),
        (v, d) -> sum(D.logpdf.(D.Gamma.(2.0, v), d[:c]))),
    ("Beta first shape", :p, w -> :(p .~ Beta.($w, 1.5)),
        (v, d) -> sum(D.logpdf.(D.Beta.(v, 1.5), d[:p]))),
    ("Beta second shape", :p, w -> :(p .~ Beta.(1.5, $w)),
        (v, d) -> sum(D.logpdf.(D.Beta.(1.5, v), d[:p]))),
    ("Normal location", :c, w -> :(c .~ Normal.($w, s)),
        (v, d, q) -> sum(D.logpdf.(D.Normal.(v, q.s), d[:c]))),
]

# (expression, its value at q). Every value is positive at `_ILV_POINT`.
const _ILV_VALUES = [
    (:(abs.(th)), (q, d) -> abs.(q.th)),
    (:(0.8 .+ abs.(th)), (q, d) -> 0.8 .+ abs.(q.th)),
    (:(abs.(v) .+ 0.3), (q, d) -> abs.(2 .* q.th) .+ 0.3),
    (:(exp.(b .* th) .* (1 .+ x .^ 2)),
        (q, d) -> exp.(q.b .* q.th) .* (1 .+ d[:x] .^ 2)),
    (:(exp.(eta) .* abs.(th) .+ 0.2),
        (q, d) -> exp.(q.a .+ q.b .* d[:x]) .* abs.(q.th) .+ 0.2),
    (:(exp.(eta) .+ abs.(th)),
        (q, d) -> exp.(q.a .+ q.b .* d[:x]) .+ abs.(q.th)),
]

_ilv_resp_logpdf(f, v, d, q) =
    applicable(f, v, d, q) ? f(v, d, q) : f(v, d)

function _ilv_build(ex, col)
    data = deepcopy(_ILV_DATA)
    before = deepcopy(data)
    names = (:x, :y, col)
    bound = bind_data(lower_rkppl(ex, names; mod = @__MODULE__,
        conditioned = (:y, col)), Dict(k => data[k] for k in names))
    built = build_kernel(bound)
    @test data == before
    return bound, built
end

@testset "inline latent values lower as their named twins" begin
    for (slot, col, response, resp_logpdf) in _ILV_SLOTS,
            (value, value_at) in _ILV_VALUES
        @testset "$slot: $value" begin
            q = _ILV_POINT
            oracle = _ilv_base_logpdf(q, _ILV_DATA) +
                _ilv_resp_logpdf(resp_logpdf, value_at(q, _ILV_DATA),
                    _ILV_DATA, q)
            results = map((_ilv_model(response(value)),
                    _ilv_model(:(w = $value), response(:w)))) do ex
                bound, built = _ilv_build(ex, col)
                u = unconstrain(built.layout, q)
                kernel = prepare_query(built, bound, :sampler)
                lp = Base.invokelatest(kernel, u)
                @test lp ≈ oracle + logjac(built.layout, u) atol = 1e-11 rtol = 1e-11
                _check_gradient(built.spec, bound, u)
                (; lp, names = coordinate_names(built.layout))
            end
            inline, named = results
            @test inline.lp ≈ named.lp atol = 1e-12 rtol = 1e-12
            @test inline.names == named.names
        end
    end
end

@testset "inline latent value beside another reader of its predictor" begin
    # `eta` is also the whole location of `z`; the inline value still reads
    # its value.
    data = merge(_ILV_DATA, Dict(:z => [0.3, 0.1, 0.6, -0.2, 0.4]))
    ex = _ilv_model(:(z .~ Normal.(eta, s)),
        :(c .~ Weibull.(1.4, exp.(eta) .* abs.(th) .+ 0.2)))
    names = (:x, :y, :z, :c)
    bound = bind_data(lower_rkppl(ex, names; mod = @__MODULE__,
        conditioned = (:y, :z, :c)), Dict(k => data[k] for k in names))
    built = build_kernel(bound)
    q = _ILV_POINT
    mu = q.a .+ q.b .* data[:x]
    oracle = _ilv_base_logpdf(q, data) +
        sum(D.logpdf.(D.Normal.(mu, q.s), data[:z])) +
        sum(D.logpdf.(D.Weibull.(1.4, exp.(mu) .* abs.(q.th) .+ 0.2), data[:c]))
    u = unconstrain(built.layout, q)
    kernel = prepare_query(built, bound, :sampler)
    @test Base.invokelatest(kernel, u) ≈ oracle + logjac(built.layout, u) atol = 1e-11 rtol = 1e-11
    _check_gradient(built.spec, bound, u)
end

@testset "whole latent reads stay composition scalars" begin
    # `sum(th)` reads the latent whole, so the composition over `eta`
    # keeps its plan; only elementwise latent reads take the named-twin route.
    ex = _ilv_model(:(c .~ Weibull.(1.4, exp.(eta) .* sum(th))))
    bound, built = _ilv_build(ex, :c)
    @test any(p -> any(t -> t.kind === ComposedTerm, p.terms), bound.predictors)
    q = _ILV_POINT
    oracle = _ilv_base_logpdf(q, _ILV_DATA) + sum(D.logpdf.(D.Weibull.(1.4,
        exp.(q.a .+ q.b .* _ILV_DATA[:x]) .* sum(q.th)), _ILV_DATA[:c]))
    u = unconstrain(built.layout, q)
    kernel = prepare_query(built, bound, :sampler)
    @test Base.invokelatest(kernel, u) ≈ oracle + logjac(built.layout, u) atol = 1e-11 rtol = 1e-11
    _check_gradient(built.spec, bound, u)
end
