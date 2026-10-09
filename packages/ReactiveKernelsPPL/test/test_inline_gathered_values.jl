module InlineGatheredValueTests
using ReactiveKernels, ReactiveKernelsPPL, Distributions, DifferentiationInterface, Enzyme, Test
using LinearAlgebra: Cholesky, LowerTriangular

# A gather from an inline array value means what the gather from its named
# definition means: naming a subexpression never changes legality or the
# density (rkppl-use §2). `(z .* tau)[1, g]` reads as `r = z .* tau;
# r[1, g]`, including the level lookup on the `levels(g)` axis `r`
# inherits from `z`.

const BACKEND = AutoEnzyme(; mode=Enzyme.Reverse)

# String labels out of sorted order: positions on `levels(g)` never
# coincide with the labels, so a positional read could not pass the oracle.
const g = ["b", "c", "b", "a", "c", "a"]
const gi = [3, 1, 2, 2, 3, 1]
const x = [0.3, -0.5, 1.2, 0.8, -0.1, 0.4]
const y = [0.1, -0.4, 1.1, 0.6, -0.2, 0.5]

passthrough(v) = v

const PRIORS = (
    :(a ~ Normal(0, 1)),
    :(s ~ Exponential(1)),
    :(tau[1:2] .~ Exponential.(1.0)),
    :(z[1:2, levels(g)] .~ Normal.(0, 1)),
    :(w[levels(g)] .~ Normal.(0, 1)),
    :(Z[levels(g), 1:2] .~ Normal.(0, 1)),
    :(L ~ LKJCholesky(2, 2.0)),
    :(v[unique(gi)] .~ Normal.(0, 1)),
)

prior(p) = logpdf(Normal(0, 1), p.a) + logpdf(Exponential(1), p.s) +
    sum(logpdf.(Exponential(1.0), p.tau)) + sum(logpdf.(Normal(0, 1), p.z)) +
    sum(logpdf.(Normal(0, 1), p.w)) + sum(logpdf.(Normal(0, 1), p.Z)) +
    logpdf(LKJCholesky(2, 2.0), Cholesky(LowerTriangular(p.L))) +
    sum(logpdf.(Normal(0, 1), p.v))

# Independent level lookup: the position of each label in `levels(g)`.
code(d) = indexin(d.g, sort(unique(d.g)))

# Each shape: its inline statements, the named twin, and the location an
# independent oracle computes from the constrained values. The named twin
# spells its definition `_rkppl_gathered_1`, the name the inline value
# takes, so both spellings must generate the same program.
const SHAPES = (
    two_axis = (
        (:(loc = a .+ (z .* tau)[1, g]), :(y .~ Normal.(loc, 1.0))),
        (:(_rkppl_gathered_1 = z .* tau),
            :(loc = a .+ _rkppl_gathered_1[1, g]), :(y .~ Normal.(loc, 1.0))),
        (p, d) -> p.a .+ (p.z .* p.tau)[1, code(d)]),
    one_axis = (
        (:(loc = a .+ (w .* s)[g]), :(y .~ Normal.(loc, 1.0))),
        (:(_rkppl_gathered_1 = w .* s),
            :(loc = a .+ _rkppl_gathered_1[g]), :(y .~ Normal.(loc, 1.0))),
        (p, d) -> p.a .+ (p.w .* p.s)[code(d)]),
    # Levels on the first axis of a matrix product with an adjoint factor.
    matmul = (
        (:(loc = a .+ (Z * (tau .* L)')[g, 1]), :(y .~ Normal.(loc, 1.0))),
        (:(_rkppl_gathered_1 = Z * (tau .* L)'),
            :(loc = a .+ _rkppl_gathered_1[g, 1]), :(y .~ Normal.(loc, 1.0))),
        (p, d) -> p.a .+ (p.Z * (p.tau .* p.L)')[code(d), 1]),
    # First-occurrence order: label 3 sits at position 1 of `unique(gi)`, so
    # a positional read would silently take another level's value.
    unique_axis = (
        (:(loc = a .+ (v .* s)[gi]), :(y .~ Normal.(loc, 1.0))),
        (:(_rkppl_gathered_1 = v .* s),
            :(loc = a .+ _rkppl_gathered_1[gi]), :(y .~ Normal.(loc, 1.0))),
        (p, d) -> p.a .+ (p.v .* p.s)[indexin(d.gi, unique(d.gi))]),
    adjoint = (
        (:(loc = a .+ (z' .* tau')[g, 2]), :(y .~ Normal.(loc, 1.0))),
        (:(_rkppl_gathered_1 = z' .* tau'),
            :(loc = a .+ _rkppl_gathered_1[g, 2]), :(y .~ Normal.(loc, 1.0))),
        (p, d) -> p.a .+ (p.z' .* p.tau')[code(d), 2]),
    in_response = (
        (:(y .~ Normal.(a .+ (z .* tau)[2, g], 1.0)),),
        (:(_rkppl_gathered_1 = z .* tau),
            :(y .~ Normal.(a .+ _rkppl_gathered_1[2, g], 1.0))),
        (p, d) -> p.a .+ (p.z .* p.tau)[2, code(d)]),
    # A module call's result has no tracked axes: positional, named or not.
    opaque_call = (
        (:(loc = a .+ passthrough(z)[1, gi] .* x), :(y .~ Normal.(loc, 1.0))),
        (:(_rkppl_gathered_1 = passthrough(z)),
            :(loc = a .+ _rkppl_gathered_1[1, gi] .* x), :(y .~ Normal.(loc, 1.0))),
        (p, d) -> p.a .+ p.z[1, d.gi] .* d.x),
)

model(stmts) = Expr(:block, PRIORS..., stmts...)

lowered(ast, data) = lower_rkppl(ast, Tuple(keys(data));
    mod=@__MODULE__, conditioned=(:y,))

program(bound) = string(Base.remove_linenums!(deepcopy(
    kernel_expr(bound, assign_layout(bound)))))

function differences(f, u)
    h = cbrt(eps(Float64))
    [begin
        up, down = copy(u), copy(u)
        up[i] += h
        down[i] -= h
        (f(up) - f(down)) / (2h)
    end for i in eachindex(u)]
end

# Posterior value and ordinary Enzyme reverse against an independent
# Distributions density; returns them for the inline/named comparison.
function check(bound, data, location)
    built = build_kernel(bound)
    sampler = prepare_sampler(built, bound, zeros(built.layout.total); backend=BACKEND)
    target(u) = begin
        p = constrain(built.layout, u)
        prior(p) + sum(logpdf.(Normal.(location(p, data), 1.0), data.y)) +
            logjac(built.layout, u)
    end
    u = [0.3sin(1.3i) for i in 1:built.layout.total]
    saved = copy(u)
    value, grad = sampler_value_and_gradient!(sampler, similar(u), u)
    @test value ≈ target(u) rtol=1e-12
    @test grad ≈ differences(target, u) rtol=1e-5 atol=1e-7
    @test u == saved
    return (; value, grad, built)
end

structure(spec) = [(e.kind, e.depth) for e in recipe_inventory(spec)
    if e.kind !== :ordinary]

@testset "inline gathered values lower as their named definitions" begin
    data = (; g, gi, x, y)
    original = deepcopy(data)
    @testset "$name" for (name, (inline, named, location)) in pairs(SHAPES)
        bi = bind_data(lowered(model(inline), data), data)
        bn = bind_data(lowered(model(named), data), data)
        @test program(bi) == program(bn)
        ri = Base.invokelatest(check, bi, data, location)
        rn = Base.invokelatest(check, bn, data, location)
        @test ri.value == rn.value
        @test ri.grad == rn.grad
        @test coordinate_names(ri.built.layout) == coordinate_names(rn.built.layout)
    end
    @test data == original
end

@testset "an inline gathered value keeps its structure as rows grow" begin
    inline, _, location = SHAPES.two_axis
    plan = lowered(model(inline), (; g, gi, x, y))
    specs = map((1, 3)) do k
        data = (; g = repeat(g, k), gi = repeat(gi, k), x = repeat(x, k),
            y = repeat(y, k))
        bound = bind_data(plan, data)
        Base.invokelatest(check, bound, data, location).built.spec
    end
    @test structure(specs[1]) == structure(specs[2])
end

@testset "data-only gathered values stay as written" begin
    # A data-only value folds with its gather at binding; it takes no name.
    data = (; g, gi, x, y)
    ast = model((:(loc = a .+ (x .* 2)[gi]), :(y .~ Normal.(loc, 1.0))))
    bound = bind_data(lowered(ast, data), data)
    @test !occursin("_rkppl_gathered_", program(bound))
    Base.invokelatest(check, bound, data, (p, d) -> p.a .+ (d.x .* 2)[d.gi])
end
end
