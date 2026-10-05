module PPLNamedMatrixAdditionTests
using ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme, Test
using Distributions: Normal, logpdf

module Models
using ReactiveKernels, ReactiveKernelsPPL
@rkppl population(X) = begin
    beta[axes(X, 2)] .~ Normal.(0, 1)
    return X * beta
end
@kernel wave(x, a) = begin
    return a .* sin.(x)
end
@rkppl smooth(x) = begin
    a ~ Normal(0, 1)
    return wave(x, a)
end
end

const BACKEND = AutoEnzyme(; mode=Enzyme.Reverse)
const SPELLINGS = [
    (:inline, :beta, quote
        beta[axes(X, 2)] .~ Normal.(0, 1)
        s ~ smooth(x)
        mu = X * beta .+ s
    end),
    (:named, :beta, quote
        beta[axes(X, 2)] .~ Normal.(0, 1)
        p = X * beta
        s ~ smooth(x)
        mu = p .+ s
    end),
    (:submodel, Symbol("p.beta"), quote
        p ~ population(X)
        s ~ smooth(x)
        mu = p .+ s
    end),
]

function build_case(body, n)
    x = collect(range(-0.6; step=1.5 / max(n - 1, 1), length=n))
    data = (; x, y=0.2 .+ 0.4 .* x .+ 0.1 .* sin.(x))
    ast = quote
        X = hcat(ones(length(x)), x)
        $(body.args...)
        y .~ Normal.(mu, 1)
    end
    bound = bind_data(lower_rkppl(ast, data; conditioned=(:y,), mod=Models), data)
    return bound, build_kernel(bound), data
end

function reference(names, beta, data, u)
    q = Dict(zip(names, u))
    mu = q[Symbol(beta, ".1")] .+ q[Symbol(beta, ".2")] .* data.x .+
        q[Symbol("s.a")] .* sin.(data.x)
    return sum(logpdf.(Normal(), u)) + sum(logpdf.(Normal.(mu, 1), data.y))
end

function gradient(names, beta, data, u)
    q = Dict(zip(names, u))
    residual = data.y .- q[Symbol(beta, ".1")] .-
        q[Symbol(beta, ".2")] .* data.x .- q[Symbol("s.a")] .* sin.(data.x)
    return [-u[i] + sum(residual .* (name == Symbol(beta, ".1") ?
        ones(length(data.x)) : name == Symbol(beta, ".2") ? data.x : sin.(data.x)))
        for (i, name) in enumerate(names)]
end

@testset "matrix products add to self-contained graph components" begin
    for (label, beta, body) in SPELLINGS, n in (0, 1, 7)
        @testset "$label / $n" begin
            bound, built, data = build_case(body, n)
            names = coordinate_names(built.layout)
            @test Set(names) == Set([Symbol(beta, ".1"), Symbol(beta, ".2"), Symbol("s.a")])
            original = deepcopy(data)
            u = [0.1 * cos(i) for i in eachindex(names)]
            sampler = prepare_sampler(built, bound, u; backend=BACKEND)
            for w in (u, -u)
                saved = copy(w)
                value, grad = sampler_value_and_gradient!(sampler, similar(w), w)
                @test value ≈ reference(names, beta, data, w) rtol=1e-12
                @test grad ≈ gradient(names, beta, data, w) rtol=1e-10 atol=1e-12
                @test w == saved
                @test data == original
            end
        end
    end
end

end
