module PPLNamedMatrixProductTests
using ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme, Test
using Distributions: Normal, logpdf

# A design-matrix product read inside a composition keeps its inline meaning
# when it is named first or returned from a submodel: naming a subexpression
# never changes legality or the density (snag `rkppl-named-desi-15f63734`).
module Models
using ReactiveKernels, ReactiveKernelsPPL
plain(v) = v .+ 0.0
@kernel graph(v) = begin
    w = v .+ 0.0
    return w
end
@rkppl population(X) = begin
    beta[axes(X, 2)] .~ Normal.(0, 1)
    return X * beta
end
@rkppl population_named(X) = begin
    beta[axes(X, 2)] .~ Normal.(0, 1)
    value = X * beta
    return value
end
end

const BACKEND = AutoEnzyme(; mode=Enzyme.Reverse)

const SPELLINGS = [
    (:inline, :b, quote
        b[axes(X, 2)] .~ Normal.(0, 1)
        mu = graph(X * b)
    end),
    (:named, :b, quote
        b[axes(X, 2)] .~ Normal.(0, 1)
        p = X * b
        mu = graph(p)
    end),
    (:named_chain, :b, quote
        b[axes(X, 2)] .~ Normal.(0, 1)
        q = X * b
        p = q
        mu = plain(p)
    end),
    (:submodel_function, Symbol("p.beta"), quote
        p ~ population(X)
        mu = plain(p)
    end),
    (:submodel_kernel, Symbol("p.beta"), quote
        p ~ population(X)
        mu = graph(p)
    end),
    (:submodel_named_local, Symbol("p.beta"), quote
        p ~ population_named(X)
        mu = graph(p)
    end),
]

function build_case(body, n)
    x = collect(range(-1.0; step=2.5 / max(n - 1, 1), length=n))
    y = 0.2 .+ 0.6 .* x .+ 0.1 .* sin.(1:n)
    ast = quote
        X = hcat(ones(length(x)), x)
        $(body.args...)
        y .~ Normal.(mu, 1.0)
    end
    data = (; x, y)
    bound = bind_data(lower_rkppl(ast, data; conditioned=(:y,), mod=Models), data)
    return bound, build_kernel(bound), data
end

reference(data, u) = sum(logpdf.(Normal(), u)) +
    sum(logpdf.(Normal.(u[1] .+ u[2] .* data.x, 1.0), data.y))
gradient(data, u) = -u .+ [sum(data.y .- u[1] .- u[2] .* data.x),
    sum((data.y .- u[1] .- u[2] .* data.x) .* data.x)]

@testset "named and submodel-returned matrix products read inside calls" begin
    for (label, name, body) in SPELLINGS
        @testset "$label" begin
            for n in (0, 1, 7)
                bound, built, data = build_case(body, n)
                original = deepcopy(data)
                @test coordinate_names(built.layout) ==
                    [Symbol(name, ".1"), Symbol(name, ".2")]
                post = prepare_query(built, bound, :sampler)
                sampler = prepare_sampler(built, bound, [0.3, -0.7]; backend=BACKEND)
                for u in ([0.3, -0.7], [-0.2, 0.4])
                    original_u = copy(u)
                    @test Base.invokelatest(post, u) ≈ reference(data, u)
                    g = similar(u)
                    value, _ = sampler_value_and_gradient!(sampler, g, u)
                    @test value ≈ reference(data, u)
                    @test g ≈ gradient(data, u)
                    @test u == original_u
                    @test data == original
                end
            end
        end
    end
end

@testset "a named product keeps affine and composed readers" begin
    u = [0.3, -0.7]
    x = collect(range(-1.0, 1.5; length=6))
    y = 0.2 .+ 0.6 .* x
    y2 = 0.1 .- 0.4 .* x
    ast = quote
        X = hcat(ones(length(x)), x)
        b[axes(X, 2)] .~ Normal.(0, 1)
        p = X * b
        mu = graph(p)
        y .~ Normal.(mu, 1.0)
        y2 .~ Normal.(p, 0.5)
    end
    data = (; x, y, y2)
    bound = bind_data(lower_rkppl(ast, data; conditioned=(:y, :y2),
        mod=Models), data)
    built = build_kernel(bound)
    eta = u[1] .+ u[2] .* x
    oracle(w) = sum(logpdf.(Normal(), w)) +
        sum(logpdf.(Normal.(w[1] .+ w[2] .* x, 1.0), y)) +
        sum(logpdf.(Normal.(w[1] .+ w[2] .* x, 0.5), y2))
    @test coordinate_names(built.layout) == [Symbol("b.1"), Symbol("b.2")]
    sampler = prepare_sampler(built, bound, u; backend=BACKEND)
    g = similar(u)
    value, _ = sampler_value_and_gradient!(sampler, g, u)
    @test value ≈ oracle(u)
    residual = y .- eta .+ (y2 .- eta) ./ 0.5^2
    @test g ≈ -u .+ [sum(residual), sum(residual .* x)]
    @test Base.invokelatest(prepare_query(built, bound, :pointwise), u).y2 ≈
        logpdf.(Normal.(eta, 0.5), y2)
end

end
