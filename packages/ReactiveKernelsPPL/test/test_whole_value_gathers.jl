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
# The index is computed by a module call, so it is model-level.
positions(x, lv) = Int[findfirst(==(v), lv) for v in x]
const vcat_ = vcat
# An intrinsic CAR field gathered per area: `phi_full` is per observation
# (a data gather), and `effects[area]` gathers it by a model-level index.
@rkppl field_effects(node1, node2, struct_pos, area, S) = begin
    sigma ~ restricted(Normal(0.0, 1.0), 0.0, Inf)
    phi[1:S] ~ ICAR(node1, node2)
    cs = cumsum(vcat(0.0, phi))
    phi_sum = cs[5] - cs[1]               # BRM's literal spelling for S = 4
    phi_sum ~ Normal(0.0, 0.004)
    phi_full = (vcat(phi, 0.0))[struct_pos]
    effects = sigma .* phi_full
    return effects[area]
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

# A gather of a per-observation value by a model-level index (`effects[ar]`
# with `ar` a module call's result) reads the gathered value whole, as the
# same gather inside a module call does (`vcat(effects[ar])`): the named,
# inline and positional spellings lower, as does a model-scope `vcat` of
# such values that a cell reads (the shape BRM emits for two `car(...)`
# terms, `u = vcat(u_a, u_b)`).
function _wvg_field(spelling)
    u = spelling === :model_scope ? (:(u = vcat(u_a, u_b)),) :
        spelling === :renamed ? (:(u = vcat_(u_a, u_b)),) : ()
    read = spelling === :in_cell ? :(vcat(u_a, u_b)[idx[i]]) : :(u[idx[i]])
    return quote
        area_a = positions(zone, [1, 2, 3, 4])
        f_a ~ field_effects(node1, node2, struct_pos, area_a, 4)
        u_a = f_a
        area_b = positions(zone, [1, 2, 3, 4])
        f_b ~ field_effects(node1, node2, struct_pos, area_b, 4)
        u_b = f_b
        sigma ~ Exponential(1.0)
        $(u...)
        @plate for i in eachindex(y)
            y[i] .~ Normal.($read, sigma)
        end
    end
end

_wvg_field_data() = Dict{Symbol,Any}(:zone => [2, 4, 1, 3], :node1 => [1, 2, 3],
    :node2 => [2, 3, 4], :struct_pos => [1, 2, 3, 4],
    :idx => [[1, 6], [3, 8, 2], [5, 4, 7]],
    :y => [[0.2, -0.1], [0.5, 0.0, 0.3], [-0.4, 0.1, 0.6]])

function _wvg_top(spelling)
    body = spelling === :named ? (:(effects = s .* phi_full), :(ua = effects[ar])) :
        spelling === :inline ? (:(ua = (s .* phi_full)[ar]),) :
        (:(effects = s .* phi_full), :(t = effects[2]), :(ua = s .* phi .+ t))
    return quote
        s ~ Exponential(1.0)
        phi[1:4] .~ Normal.(0, 1)
        ar = positions(zone, [1, 2, 3, 4])
        phi_full = (vcat(phi, 0.0))[struct_pos]
        $(body...)
        u = vcat(ua, ua)
        sigma ~ Exponential(1.0)
        @plate for i in eachindex(y)
            y[i] .~ Normal.(u[idx[i]], sigma)
        end
    end
end

_wvg_cells(u, fx, sigma) = sum(sum(logpdf.(Normal.(u[ix], sigma), yi))
    for (ix, yi) in zip(fx.data[:idx], fx.data[:y]))

# Independent ICAR quadratic form, original Normal density of the
# restricted scale, and the log Jacobians of the three `exp` transforms.
function _wvg_oracle(fx, u, ::Val{:field})
    nt = constrain(fx.built.layout, u)
    d = fx.data
    lv = [1, 2, 3, 4]
    field(f) = f.sigma .* vcat(f.phi, 0.0)[d[:struct_pos]][indexin(d[:zone], lv)]
    fprior(f) = logpdf(Normal(), f.sigma) - 0.5 * sum(abs2,
        f.phi[d[:node1]] .- f.phi[d[:node2]]) + logpdf(Normal(0, 0.004), sum(f.phi))
    values = vcat(field(nt.f_a), field(nt.f_b))
    likelihood = _wvg_cells(values, fx, nt.sigma)
    prior = fprior(nt.f_a) + fprior(nt.f_b) + logpdf(Exponential(1), nt.sigma)
    jac = log(nt.f_a.sigma) + log(nt.f_b.sigma) + log(nt.sigma)
    return (; likelihood, prior, posterior = likelihood + prior + jac)
end

_wvg_top_values(nt, pf, d, ::Val{:top}) = (nt.s .* pf)[indexin(d[:zone], [1, 2, 3, 4])]
_wvg_top_values(nt, pf, d, ::Val{:positional_read}) = nt.s .* nt.phi .+ nt.s * pf[2]

function _wvg_oracle(fx, u, kind::Union{Val{:top},Val{:positional_read}})
    nt = constrain(fx.built.layout, u)
    d = fx.data
    ua = _wvg_top_values(nt, vcat(nt.phi, 0.0)[d[:struct_pos]], d, kind)
    values = vcat(ua, ua)
    likelihood = _wvg_cells(values, fx, nt.sigma)
    prior = logpdf(Exponential(1), nt.s) + sum(logpdf.(Normal(), nt.phi)) +
        logpdf(Exponential(1), nt.sigma)
    return (; likelihood, prior,
        posterior = likelihood + prior + log(nt.s) + log(nt.sigma))
end

@testset "whole-value gathers: a model-level index gathers a column whole" begin
    @testset "two CAR fields concatenated at model scope" begin
        data = _wvg_field_data()
        before = deepcopy(data)
        fxs = Dict(s => _wvg_build(_wvg_field(s), data)
            for s in (:model_scope, :renamed, :in_cell))
        names = coordinate_names(fxs[:in_cell].built.layout)
        @test all(fx -> coordinate_names(fx.built.layout) == names, values(fxs))
        for k in 1:2
            u = [0.3 * sin(1.2i + k) for i in 1:length(names)]
            r = Dict(s => Base.invokelatest(_wvg_native, fx, u, Val(:field))
                for (s, fx) in fxs)
            @test r[:model_scope].value == r[:in_cell].value == r[:renamed].value
            @test r[:model_scope].grad == r[:in_cell].grad == r[:renamed].grad
        end
        @test data == before
        # The group plate stays one retained plate as groups grow.
        more = merge(data, Dict{Symbol,Any}(:idx => repeat(data[:idx], 4),
            :y => repeat(data[:y], 4)))
        grown = _wvg_build(_wvg_field(:model_scope), more)
        structure(fx) = [(e.kind, e.depth) for e in recipe_inventory(fx.built.spec)
            if e.kind !== :ordinary]
        @test (:plate, 0) in structure(fxs[:model_scope])
        @test structure(grown) == structure(fxs[:model_scope])
        u = [0.3 * sin(1.2i) for i in 1:length(names)]
        Base.invokelatest(_wvg_native, grown, u, Val(:field))
    end
    @testset "$spelling top-level gather" for spelling in (:named, :inline, :positional)
        data = _wvg_field_data()
        fx = _wvg_build(_wvg_top(spelling), data)
        u = [0.3 * sin(1.2i) for i in 1:fx.built.layout.total]
        Base.invokelatest(_wvg_native, fx, u, Val(spelling === :positional ?
            :positional_read : :top))
    end
end
