using DifferentiationInterface
using Distributions
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Test

# Gathers whose index columns the model otherwise reads only whole: the
# location is passed whole to an ordinary function, so the grouping data
# are model-level inputs. Naming a subexpression never changes legality
# or the density (rkppl-use §2): an inline `(b[g, 1] .+ b[h, 1]) ./ 2`
# lowers exactly like `rg = b[g, 1]; rh = b[h, 1]; r = (rg .+ rh) ./ 2`.
module WholeValueGatherModels
using ReactiveKernelsPPL
read_rows(value, rows) = value[rows]
@rkppl group_values(labels) = begin
    tau[1:1] .~ Normal.(1, 0.1)
    z[levels(labels), 1:1] .~ Normal.(0, 1)
    return z .* tau[1]
end
@rkppl matrix_values(X, ncoef, loc, scale) = begin
    beta[1:ncoef] .~ Normal.(loc, scale)
    return X * beta
end
end

_wvg_rows(axis) = axis === :observations ? :(length(y)) : :(length(g))

# Multi-membership over a submodel-returned `levels(vcat(g, h)) × 1` matrix.
function _wvg_program(spelling, axis)
    r = spelling === :inline ? (:(r = (b[g, 1] .+ b[h, 1]) ./ 2),) :
        (:(rg = b[g, 1]), :(rh = b[h, 1]), :(r = (rg .+ rh) ./ 2))
    return quote
        gg = vcat(g, h)
        b ~ group_values(gg)
        $(r...)
        X = hcat(ones($(_wvg_rows(axis))))
        p ~ matrix_values(X, 1, 0.0, 1.0)
        mu = p .+ r
        reads = read_rows(mu, rows)
        y .~ Normal.(reads, 1)
    end
end

# A positional declared array gathered by an index read only whole.
function _wvg_positional(spelling)
    w = spelling === :inline ? (:(w = c[k] .+ 0.5),) :
        (:(ck = c[k]), :(w = ck .+ 0.5))
    return quote
        c[1:3] .~ Normal.(0, 1)
        a[1:1] .~ Normal.(0, 1)
        X = hcat(ones(length(y)))
        p = X * a
        $(w...)
        mu = p .+ w
        reads = read_rows(mu, rows)
        y .~ Normal.(reads, 1)
    end
end

function _wvg_data(axis)
    y = [0.1, -0.2, 0.3, 0.7, 0.4, -0.5]
    axis === :observations && return Dict{Symbol,Any}(
        :g => ["a", "a", "b", "c", "c", "b"], :h => ["c", "b", "a", "a", "b", "c"],
        :rows => collect(1:6), :y => y)
    # Group data off the observation axis: three rows read six times.
    return Dict{Symbol,Any}(:g => ["a", "b", "c"], :h => ["c", "a", "b"],
        :rows => [1, 2, 3, 3, 1, 2], :y => y)
end

_wvg_positional_data() = Dict{Symbol,Any}(:k => [3, 1, 2, 2, 3, 1],
    :rows => [6, 5, 4, 3, 2, 1], :y => [0.1, -0.2, 0.3, 0.7, 0.4, -0.5])

function _wvg_build(ast, data)
    plan = lower_rkppl(ast, data; conditioned = (:y,),
        mod = WholeValueGatherModels)
    bound = bind_data(plan, data)
    return (; data, bound, built = build_kernel(bound))
end

# Independent Julia indexing and Distributions densities. Every
# coordinate is unconstrained, so the posterior has no Jacobian.
function _wvg_oracle(fx, u, ::Val{:membership})
    nt = constrain(fx.built.layout, u)
    g, h, y = fx.data[:g], fx.data[:h], fx.data[:y]
    lv = sort(unique(vcat(g, h)))
    b = nt.b.z .* nt.b.tau[1]
    r = (b[indexin(g, lv), 1] .+ b[indexin(h, lv), 1]) ./ 2
    mu = ones(length(r), 1) * nt.p.beta .+ r
    likelihood = sum(logpdf.(Normal.(mu[fx.data[:rows]], 1), y))
    prior = sum(logpdf.(Normal(1, 0.1), nt.b.tau)) +
        sum(logpdf.(Normal(), nt.b.z)) + sum(logpdf.(Normal(), nt.p.beta))
    return (; likelihood, prior, posterior = likelihood + prior)
end

function _wvg_oracle(fx, u, ::Val{:positional})
    nt = constrain(fx.built.layout, u)
    y = fx.data[:y]
    mu = ones(length(y), 1) * nt.a .+ (nt.c[fx.data[:k]] .+ 0.5)
    likelihood = sum(logpdf.(Normal.(mu[fx.data[:rows]], 1), y))
    prior = sum(logpdf.(Normal(), nt.c)) + sum(logpdf.(Normal(), nt.a))
    return (; likelihood, prior, posterior = likelihood + prior)
end

function _wvg_findiff(f, u; h = 1e-6)
    return map(eachindex(u)) do i
        up, down = copy(u), copy(u)
        up[i] += h
        down[i] -= h
        (f(up) - f(down)) / (2h)
    end
end

function _wvg_native(fx, u, kind)
    q = prepare_sampler(fx.built, fx.bound, u;
        backend = AutoEnzyme(; mode = Enzyme.Reverse))
    value, grad = sampler_value_and_gradient!(q, similar(u), u)
    oracle = _wvg_oracle(fx, u, kind)
    @test value ≈ oracle.posterior rtol = 1e-12
    @test grad ≈ _wvg_findiff(w -> _wvg_oracle(fx, w, kind).posterior, u) rtol = 1e-5 atol = 1e-7
    kernel = prepare_query(fx.built, fx.bound, :likelihood)
    @test Base.invokelatest(kernel, u) ≈ oracle.likelihood rtol = 1e-12
    return (; value, grad)
end

_wvg_derived(fx) = Set(d.name for d in fx.bound.derived)

@testset "whole-value gathers: naming a gather keeps its shape" begin
    @testset "multi-membership over a submodel value, $axis" for axis in
            (:observations, :groups)
        data = _wvg_data(axis)
        before = deepcopy(data)
        fxs = Dict(s => _wvg_build(_wvg_program(s, axis), data)
            for s in (:inline, :named))
        # The inline sum is an observation-aligned definition, as its
        # named members are; neither is a model-level assignment.
        @test :r in _wvg_derived(fxs[:inline])
        @test :r in _wvg_derived(fxs[:named])
        @test coordinate_names(fxs[:inline].built.layout) ==
            coordinate_names(fxs[:named].built.layout)
        u = [0.3 * sin(1.2i) for i in 1:fxs[:inline].built.layout.total]
        inline = Base.invokelatest(_wvg_native, fxs[:inline], u, Val(:membership))
        named = Base.invokelatest(_wvg_native, fxs[:named], u, Val(:membership))
        @test inline.value == named.value
        @test inline.grad == named.grad
        @test data == before
    end
    @testset "positional declared array" begin
        data = _wvg_positional_data()
        fxs = Dict(s => _wvg_build(_wvg_positional(s), data)
            for s in (:inline, :named))
        @test :w in _wvg_derived(fxs[:inline])
        u = [0.3 * sin(1.2i) for i in 1:fxs[:inline].built.layout.total]
        inline = Base.invokelatest(_wvg_native, fxs[:inline], u, Val(:positional))
        named = Base.invokelatest(_wvg_native, fxs[:named], u, Val(:positional))
        @test inline.value == named.value
        @test inline.grad == named.grad
    end
end
