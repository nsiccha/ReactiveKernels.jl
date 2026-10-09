module LevelPositionGatherTests
using ReactiveKernels, ReactiveKernelsPPL, Distributions, DifferentiationInterface, Enzyme, Test
using LinearAlgebra: Cholesky, LowerTriangular
import DataAPI

# A grouped block sized by the data and read at positions the model computes
# (`i = level_positions(g, g)`) is declared on a positional axis of the same
# extent, `z[1:length(levels(g)), 1:K]`. Its gathers are plain Julia
# indexing, from the declaration, a definition or a submodel's return
# (rkppl-use §3). A `levels(g)` axis instead looks each index value up among
# its labels, so positions are not labels there and binding names the
# positional declaration (snag rkppl-positional-72eb5cc6).

const BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

level_positions(labels, source) =
    (lv = DataAPI.levels(source); Int[findfirst(isequal(v), lv) for v in labels])
column_gather(draws, index, margin) = draws[index, margin]

@rkppl effects(g) = begin
    tau[1:1] .~ LogNormal.(0, 1)
    z[1:length(levels(g)), 1:1] .~ Normal.(0, 1)
    return z .* tau[1]
end

@rkppl correlated_effects(g, K) = begin
    tau[1:K] .~ Exponential.(1)
    L ~ LKJCholesky(K, 2.0)
    z[1:length(levels(g)), 1:K] .~ Normal.(0, 1)
    return z * transpose(tau .* L)
end

@rkppl labeled_effects(g) = begin
    tau[1:1] .~ LogNormal.(0, 1)
    z[levels(g), 1:1] .~ Normal.(0, 1)
    return z .* tau[1]
end

# `read` is the gathered group term of `mu`, spelled as a model would.
function program(kind, read)
    block = kind === :k2 ? :(b ~ correlated_effects(g, 2)) :
        kind === :mm ? :(b ~ effects(gg)) :
        kind === :labeled ? :(b ~ labeled_effects(g)) : :(b ~ effects(g))
    ast = quote
        a ~ Normal(0, 1)
        s ~ Exponential(1)
        $(kind === :mm ? :(gg = vcat(g, h)) : nothing)
        $block
        i = $(kind === :mm ? :(level_positions(g, gg)) : :(level_positions(g, g)))
        $(kind === :mm ? :(j = level_positions(h, gg)) : nothing)
        r = $read
        mu = a .+ r
        y .~ Normal.(mu, s)
    end
    filter!(!isnothing, ast.args)
    return ast
end

const READS = (
    k1 = (named = :(b[i, 1]), inline = :(b[level_positions(g, g), 1]),
        opaque = :(column_gather(b, i, 1))),
    k2 = (named = :(b[i, 1] .+ b[i, 2] .* x),
        inline = :(b[level_positions(g, g), 1] .+ b[level_positions(g, g), 2] .* x),
        opaque = :(column_gather(b, i, 1) .+ column_gather(b, i, 2) .* x)),
    mm = (named = :(w .* b[i, 1] .+ (1 .- w) .* b[j, 1]),
        inline = :(w .* b[level_positions(g, gg), 1] .+ (1 .- w) .* b[level_positions(h, gg), 1]),
        opaque = :(w .* column_gather(b, i, 1) .+ (1 .- w) .* column_gather(b, j, 1))),
)

# String labels out of sorted order and integer labels other than 1:G:
# a label lookup could not reproduce the positional oracle on either.
function data(labels, G, n)
    lv = labels === :strings ? ["s$(mod1(7k, 10))" for k in 1:G] : [100k + 7 for k in G:-1:1]
    # `g` meets every label; `h` may meet only some of the pooled `vcat(g, h)`.
    g = [lv[mod1(t, G)] for t in 1:n]
    h = [lv[mod1(2t + 1, G)] for t in 1:n]
    return (; g, h, x = [sin(1.3t) for t in 1:n], w = [0.2 + 0.6 * abs(cos(t)) for t in 1:n],
        y = [0.4 * sin(2.1t) + 0.5 for t in 1:n])
end

function build(kind, read, d)
    names = kind === :mm ? (:g, :h, :w, :y) : kind === :k2 ? (:g, :x, :y) : (:g, :y)
    cols = NamedTuple{names}(Tuple(getproperty(d, nm) for nm in names))
    plan = lower_rkppl(program(kind, read), names; mod = @__MODULE__, conditioned = (:y,))
    bound = bind_data(plan, cols)
    return (; plan, bound, built = build_kernel(bound))
end

# Independent Julia indexing and Distributions densities.
function oracle(kind, d, built, u)
    p = constrain(built.layout, u)
    pos(labels, source) = indexin(labels, sort(unique(source)))
    z = p.b.z
    if kind === :k2
        b = z * transpose(p.b.tau .* p.b.L)
        i = pos(d.g, d.g)
        r = b[i, 1] .+ b[i, 2] .* d.x
        block = sum(logpdf.(Exponential(1), p.b.tau)) +
            logpdf(LKJCholesky(2, 2.0), Cholesky(LowerTriangular(p.b.L)))
    else
        b = z .* only(p.b.tau)
        block = logpdf(LogNormal(0, 1), only(p.b.tau))
        if kind === :mm
            gg = vcat(d.g, d.h)
            r = d.w .* b[pos(d.g, gg), 1] .+ (1 .- d.w) .* b[pos(d.h, gg), 1]
        else
            r = b[pos(d.g, d.g), 1]
        end
    end
    likelihood = sum(logpdf.(Normal.(p.a .+ r, p.s), d.y))
    prior = logpdf(Normal(0, 1), p.a) + logpdf(Exponential(1), p.s) + block +
        sum(logpdf.(Normal(0, 1), z))
    return likelihood + prior + logjac(built.layout, u)
end

function findiff(f, u; h = 1e-6)
    return map(eachindex(u)) do k
        up, down = copy(u), copy(u)
        up[k] += h
        down[k] -= h
        (f(up) - f(down)) / (2h)
    end
end

function evaluate(fx, u)
    q = prepare_sampler(fx.built, fx.bound, u; backend = BACKEND)
    value, grad = sampler_value_and_gradient!(q, similar(u), u)
    pointwise = Base.invokelatest(prepare_query(fx.built, fx.bound, :pointwise), u)
    return (; value, grad = copy(grad), pointwise)
end

@testset "level position gathers: positional axis, plain Julia indexing" begin
    @testset "$kind $labels" for kind in (:k1, :k2, :mm), labels in (:strings, :integers)
        structures = Int[]
        for (G, n) in ((3, 7), (6, 20))
            d = data(labels, G, n)
            before = deepcopy(d)
            fx = build(kind, READS[kind].named, d)
            u = [0.3 * sin(1.7k) - 0.1 for k in 1:fx.built.layout.total]
            r = evaluate(fx, u)
            @test r.value ≈ oracle(kind, d, fx.built, u) rtol = 1e-12
            @test r.grad ≈ findiff(w -> oracle(kind, d, fx.built, w), u) rtol = 1e-5 atol = 1e-7
            # The opaque helper `column_gather` reads the same positions;
            # inline index calls lower as their named definitions.
            for spelling in (:inline, :opaque)
                tw = build(kind, getproperty(READS[kind], spelling), d)
                @test coordinate_names(tw.built.layout) == coordinate_names(fx.built.layout)
                t = evaluate(tw, u)
                @test isequal(t.value, r.value)
                @test isequal(t.grad, r.grad)
                @test isequal(t.pointwise, r.pointwise)
            end
            @test d == before
            push!(structures, length(fx.built.spec.graph.recipes))
        end
        # One graph as levels and observations grow.
        @test structures[1] == structures[2]
    end
end

@testset "level position gathers: one plan, bindings with other groups" begin
    plan = build(:k1, READS.k1.named, data(:strings, 3, 7)).plan
    for (labels, G, n) in ((:strings, 5, 11), (:integers, 4, 9))
        d = data(labels, G, n)
        bound = bind_data(plan, (; g = d.g, y = d.y))
        built = build_kernel(bound)
        u = [0.2 * cos(1.1k) for k in 1:built.layout.total]
        @test built.layout.total == 3 + G
        @test Base.invokelatest(prepare_query(built, bound, :sampler), u) ≈
            oracle(:k1, d, built, u) rtol = 1e-12
    end
end

@testset "level position gathers: the count axis takes the level sources of levels(g)" begin
    # `1:length(levels(gg))` over a definition computed from data at bind
    # (`gg = vcat(g, h)`, the multi-membership pool) is covered above.
    # refused: an extent counted over sampled values is not fixed by the
    # data (array extents are data-determined, user 2026-10-05, rkppl-use §3).
    err = try
        lower_rkppl(quote
            a ~ Normal(0, 1)
            m = a .+ x
            z[1:length(levels(m)), 1:1] .~ Normal.(0, 1)
            y .~ Normal.(a .+ z[1, 1], 1)
        end, (:x, :y); mod = @__MODULE__, conditioned = (:y,))
        nothing
    catch e
        e
    end
    @test err isa SurfaceLoweringError
    @test occursin("axis `levels(m)`: m must be data", sprint(showerror, err))
end

@testset "level position gathers: a levels axis gathers labels" begin
    d = data(:strings, 3, 7)
    # Same coordinates as the positional declaration; the label gather
    # reads the same values when its index holds the labels themselves.
    labeled = build(:labeled, :(column_gather(b, i, 1)), d)
    positional = build(:k1, READS.k1.named, d)
    @test coordinate_names(labeled.built.layout) == coordinate_names(positional.built.layout)
    by_label = build(:labeled, :(b[g, 1]), d)
    u = [0.25 * sin(k) for k in 1:positional.built.layout.total]
    @test isequal(evaluate(by_label, u).value, evaluate(positional, u).value)
    # refused: a `levels(g)` axis looks index values up among its labels
    # (rkppl-use §3), and the positions 1:3 are not labels of `g`; the
    # message names the positional declaration of the same extent.
    plan = lower_rkppl(program(:labeled, :(b[i, 1])), (:g, :y);
        mod = @__MODULE__, conditioned = (:y,))
    err = try
        bind_data(plan, (; g = d.g, y = d.y)); nothing
    catch e
        e
    end
    @test err isa ContractValidationError
    msg = sprint(showerror, err)
    @test occursin("not a label on that axis", msg)
    @test occursin("declare that axis `1:length(levels(g))`", msg)
end

end
