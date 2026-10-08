module IndexEndpointTests
using ReactiveKernels, ReactiveKernelsPPL, Distributions, DifferentiationInterface, Enzyme, LinearAlgebra, Test

# `end` and `begin` inside an index keep Julia's meaning: `s[end]` reads
# `s[lastindex(s)]` and `L[end, 1]` reads `L[lastindex(L, 1), 1]`, in
# definitions, location summands and plate cells, for every declared array
# kind (rkppl-use §2 and §9, "Indexing keeps Base's meaning").

const BACKEND = AutoEnzyme(; mode=Enzyme.Reverse)

const x = [0.3, -0.5, 1.2, 0.8, -0.1]
const y = [0.1, -0.2, 0.9, 0.4, 0.0]
const X = [0.3 1.0; -0.5 0.2; 1.2 -0.4; 0.8 0.6; -0.1 0.3]

function differences(f, u)
    h = cbrt(eps(Float64))
    [begin
        up, down = copy(u), copy(u)
        up[i] += h
        down[i] -= h
        (f(up)-f(down))/(2h)
    end for i in eachindex(u)]
end

lowered(ast, data) = lower_rkppl(ast, Tuple(keys(data)); mod=@__MODULE__,
    conditioned=(:y,))

# Posterior value, ordinary Enzyme reverse and the printed-source replay
# against `oracle(p, data)`, an independent Distributions density of the
# constrained values `p`.
function check(plan, data, oracle)
    original = deepcopy(data)
    bound = bind_data(plan, data)
    built = build_kernel(bound)
    sampler = prepare_sampler(built, bound, zeros(built.layout.total); backend=BACKEND)
    replayed = ReactiveKernelsPPL._eval_kernel_def(kernel_expr(bound, built.layout))
    replay = prepare_query((; spec=replayed, layout=built.layout), bound, :sampler)
    target(u) = oracle(constrain(built.layout, u), data) + logjac(built.layout, u)
    for shift in (0.0, -0.4)
        u = [0.3sin(i) + shift for i in 1:built.layout.total]
        saved = copy(u)
        value, grad = sampler_value_and_gradient!(sampler, similar(u), u)
        @test value ≈ target(u)
        @test Base.invokelatest(replay, u) ≈ target(u)
        @test grad ≈ differences(target, u) rtol=1e-5 atol=1e-7
        @test u == saved
    end
    @test data == original
    return built
end

lkj_logpdf(L, K, eta) = logpdf(LKJCholesky(K, eta), Cholesky(LowerTriangular(L)))

@testset "end and begin index a simplex in definitions and locations" begin
    ast = quote
        s ~ Dirichlet([1.0, 2.0, 1.5])
        w[axes(X, 2)] .~ Normal.(0, 1)
        t = s[end]
        loc = X * w .+ 0.8 .* s[begin] .- s[end - 1]
        y .~ Normal.(loc, 0.5 + t)
    end
    data = (; X, y)
    built = check(lowered(ast, data), data, (p, d) ->
        logpdf(Dirichlet([1.0, 2.0, 1.5]), p.s) + sum(logpdf.(Normal(0, 1), p.w)) +
        sum(logpdf.(Normal.(d.X * p.w .+ 0.8 * p.s[1] .- p.s[2], 0.5 + p.s[3]), d.y)))
    @test coordinate_names(built.layout) == [Symbol("s.1"), Symbol("s.2"),
        Symbol("w.1"), Symbol("w.2")]
end

@testset "end indexes ordered, sized and LKJ arrays" begin
    ast = quote
        c ~ Ordered(Normal(0, 1), 3)
        z[1:4] .~ Normal.(0, 1)
        L ~ LKJCholesky(3, 2.0)
        b ~ Normal(0, 1)
        sc = exp(c[end] - c[begin]) + L[end, end] + z[end - 1]^2
        loc = b .* x .+ z[end] .+ L[end, 1]
        y .~ Normal.(loc, sc)
    end
    data = (; x, y)
    check(lowered(ast, data), data, (p, d) ->
        sum(logpdf.(Normal(0, 1), p.c)) + sum(logpdf.(Normal(0, 1), p.z)) +
        lkj_logpdf(p.L, 3, 2.0) + logpdf(Normal(0, 1), p.b) +
        sum(logpdf.(Normal.(p.b .* d.x .+ p.z[4] .+ p.L[3, 1],
            exp(p.c[3] - p.c[1]) + p.L[3, 3] + p.z[3]^2), d.y)))
end

@testset "end follows data-sized axes on every binding" begin
    ast = quote
        z[levels(g)] .~ Normal.(0, 1)
        c ~ Ordered(Normal(0, 1), length(levels(g)) - 1)
        b ~ Normal(0, 1)
        sc = exp(c[begin])
        loc = b .* x .+ z[end] .+ c[end]
        y .~ Normal.(loc, sc)
    end
    oracle(p, d) = sum(logpdf.(Normal(0, 1), p.z)) + sum(logpdf.(Normal(0, 1), p.c)) +
        logpdf(Normal(0, 1), p.b) +
        sum(logpdf.(Normal.(p.b .* d.x .+ p.z[end] .+ p.c[end], exp(p.c[1])), d.y))
    plan = lowered(ast, (; g=nothing, x, y))
    for g in ([1, 2, 2, 3, 1], [1, 2, 2, 3, 4])
        K = length(unique(g))
        built = check(plan, (; g, x, y), oracle)
        @test length(constrain(built.layout, zeros(built.layout.total)).z) == K
        @test length(constrain(built.layout, zeros(built.layout.total)).c) == K - 1
    end
end

@testset "end on scalars, data, array definitions and cells" begin
    # A scalar has one position (`a[end]` is `a`); data and array-valued
    # definitions keep their own extents.
    ast = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        z[1:3] .~ Normal.(0, 1)
        v = 2 .* z
        sc = exp(a[begin]) + x[end]^2
        loc = b .* x .+ a[end] .+ v[end]
        y .~ Normal.(loc, sc)
    end
    data = (; x, y)
    check(lowered(ast, data), data, (p, d) ->
        logpdf(Normal(0, 1), p.a) + logpdf(Normal(0, 1), p.b) +
        sum(logpdf.(Normal(0, 1), p.z)) +
        sum(logpdf.(Normal.(p.b .* d.x .+ p.a .+ 2 * p.z[3],
            exp(p.a) + d.x[end]^2), d.y)))
    cells = quote
        z[1:3] .~ Normal.(0, 1)
        b ~ Normal(0, 1)
        @plate for i in eachindex(y)
            y[i] ~ Normal(b * x[i] + z[end], 1)
        end
    end
    check(lowered(cells, data), data, (p, d) ->
        sum(logpdf.(Normal(0, 1), p.z)) + logpdf(Normal(0, 1), p.b) +
        sum(logpdf.(Normal.(p.b .* d.x .+ p.z[3], 1), d.y)))
end

@testset "end in gathers, slices and array definitions" begin
    ast = quote
        Z[levels(g), 1:2] .~ Normal.(0, 1)
        sd[1:2] .~ Exponential.(1)
        L ~ LKJCholesky(2, 2.0)
        b ~ Normal(0, 1)
        M = (sd .* L)'
        sc = exp(sum(L[end, :]) + M[end, end] - Z[end])
        loc = b .* x .+ Z[g, end]
        y .~ Normal.(loc, sc)
    end
    g = [1, 2, 2, 3, 1]
    data = (; g, x, y)
    check(lowered(ast, data), data, (p, d) ->
        sum(logpdf.(Normal(0, 1), p.Z)) + sum(logpdf.(Exponential(1), p.sd)) +
        lkj_logpdf(p.L, 2, 2.0) + logpdf(Normal(0, 1), p.b) +
        sum(logpdf.(Normal.(p.b .* d.x .+ p.Z[d.g, 2],
            exp(sum(p.L[2, :]) + (p.sd .* p.L)'[2, 2] - p.Z[3, 2])), d.y)))
end

@testset "levels selections keep their declaration meaning" begin
    ast = quote
        z[levels(g)[2:end]] .~ Normal.(0, 1)
        b ~ Normal(0, 1)
        y .~ Normal.(b .* x .+ z[g], 1)
    end
    g = [1, 2, 2, 3, 1]
    data = (; g, x, y)
    built = check(lowered(ast, data), data, (p, d) ->
        sum(logpdf.(Normal(0, 1), p.z)) + logpdf(Normal(0, 1), p.b) +
        sum(logpdf.(Normal.(p.b .* d.x .+ [0.0; p.z][d.g], 1), d.y)))
    @test coordinate_names(built.layout) == [:b, Symbol("z.1"), Symbol("z.2")]
end

@testset "an endpoint past the array fails binding" begin
    for read in (:(t = exp(z[end + 1])), :(t = exp(z[begin - 1])))
        ast = quote
            z[1:3] .~ Normal.(0, 1)
            $read
            y .~ Normal.(x, t)
        end
        # refused: Julia raises BoundsError for z[4] and z[0] of a
        # 3-vector (rkppl-use §2, "Indexing keeps Base's meaning").
        @test_throws ReactiveKernelsPPL.ContractValidationError bind_data(
            lowered(ast, (; x, y)), (; x, y))
    end
end
end
