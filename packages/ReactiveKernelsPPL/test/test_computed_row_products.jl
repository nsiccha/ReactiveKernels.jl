module ComputedRowProductTests
using Test, ReactiveKernels, ReactiveKernelsPPL
using Distributions: Normal, Exponential, logpdf
using LogExpFunctions: logsumexp
import DifferentiationInterface as DI
import Enzyme

# A per-observation value the graph computes (`hidden = tanh.(a1' .+ x * A)`)
# is the matrix of a product as in Julia: `hidden * B` contracts its columns
# and keeps its rows, named or inline, exactly as the append-column spelling
# `hcat(ones(n), hidden) * vcat(b1', B)` does. An adjoint `a1'` of a declared
# vector is its 1×J row (snag `rkppl-adjoint-ro-fcdc0b6a`: the adjoint failed
# with "unsupported expression head '" and a bare `hidden * B` with
# "hidden is a derived column").

colmean(A) = sum(A; dims = 1) ./ size(A, 1)

const BACKEND = DI.AutoEnzyme(; mode = Enzyme.Reverse)

function build(ast, data, observed)
    bound = bind_data(lower_rkppl(ast, data; mod = @__MODULE__,
        conditioned = observed), data)
    return bound, build_kernel(bound)
end

points(n) = ([0.6 * sin(1.3k + 0.2) for k in 1:n],
    [0.5 * cos(0.7k - 0.4) for k in 1:n])

# Sampler value against an independent log density of the constrained
# values (priors, likelihood and log-Jacobian); its Enzyme Reverse gradient
# against central differences of that oracle.
function check(ast, data, observed, oracle)
    bound, built = build(ast, data, observed)
    n = built.layout.total
    sampler = prepare_sampler(built, bound, first(points(n)); backend = BACKEND)
    total(u) = oracle(constrain(built.layout, u), u)
    results = map(points(n)) do u
        gradient = similar(u)
        value, _ = sampler_value_and_gradient!(sampler, gradient, u)
        @test value ≈ total(u) rtol = 1e-12
        step = 1e-6
        for i in eachindex(u)
            plus, minus = copy(u), copy(u)
            plus[i] += step
            minus[i] -= step
            @test gradient[i] ≈ (total(plus) - total(minus)) / (2step) atol = 1e-6 rtol = 1e-6
        end
        (value, gradient)
    end
    return built, results
end

const N, M, J, K = 7, 4, 3, 4
const x = [0.4 * sin(2.1i + 0.7j) for i in 1:N, j in 1:M]
const y = [mod1(3i + 1, K) for i in 1:N]
const yc = [0.8 * cos(1.7i) for i in 1:N]

row(v) = reshape(collect(v), 1, :)
standard_normal(vs...) = sum(sum(logpdf.(Normal(), v)) for v in vs)
categorical_logit(v) = sum(vcat(0.0, v[i, :])[y[i]] - logsumexp(vcat(0.0, v[i, :]))
    for i in axes(v, 1))

@testset "a computed per-observation matrix is the matrix of a product" begin
    priors(row_declarations) = row_declarations ? quote
        A[axes(x, 2), 1:J] .~ Normal.(0, 1)
        a1[1:1, 1:J] .~ Normal.(0, 1)
        B[1:J, 1:(K - 1)] .~ Normal.(0, 1)
        b1[1:1, 1:(K - 1)] .~ Normal.(0, 1)
    end : quote
        A[axes(x, 2), 1:J] .~ Normal.(0, 1)
        a1[1:J] .~ Normal.(0, 1)
        B[1:J, 1:(K - 1)] .~ Normal.(0, 1)
        b1[1:(K - 1)] .~ Normal.(0, 1)
    end
    response = :(y .~ CategoricalLogit.(v[:, 1], v[:, 2], v[:, 3]))
    oracle(c, _) = standard_normal(c.A, c.a1, c.B, c.b1) +
        categorical_logit(row(c.b1) .+ tanh.(row(c.a1) .+ x * c.A) * c.B)
    spellings = (
        "adjoint rows, named layer" => (false, quote
            hidden = tanh.(a1' .+ x * A)
            v = b1' .+ hidden * B
        end),
        "adjoint rows, inline layer" => (false, quote
            v = b1' .+ tanh.(a1' .+ x * A) * B
        end),
        "declared 1 × J rows" => (true, quote
            hidden = tanh.(a1 .+ x * A)
            v = b1 .+ hidden * B
        end),
        "append-column products" => (false, quote
            hidden = tanh.(hcat(ones(size(x, 1)), x) * vcat(a1', A))
            v = hcat(ones(size(x, 1)), hidden) * vcat(b1', B)
        end),
    )
    data = (; x, J, K, y)
    results = Dict{String,Any}()
    for (name, (rows, layer)) in spellings
        @testset "$name" begin
            ast = Expr(:block, priors(rows).args..., layer.args..., response)
            built, r = check(ast, data, (:y,), oracle)
            @test built.layout.total == M * J + J + J * (K - 1) + (K - 1)
            results[name] = r
        end
    end
    # Every spelling is one model: the same coordinates, values and gradients.
    reference = results["append-column products"]
    for name in first.(spellings)
        for (r, ref) in zip(results[name], reference)
            @test r[1] ≈ ref[1] rtol = 1e-13
            @test r[2] ≈ ref[2] rtol = 1e-12 atol = 1e-13
        end
    end
end

@testset "a computed layer times a coefficient vector is a location" begin
    oracle(c, u) = standard_normal(c.W, c.c, c.w) + logpdf(Normal(), c.a) +
        logpdf(Exponential(1), c.s) + log(c.s) +
        sum(logpdf.(Normal.(c.a .+ tanh.(row(c.c) .+ x * c.W) * c.w, c.s), yc))
    prior = quote
        W[axes(x, 2), 1:J] .~ Normal.(0, 1)
        c[1:J] .~ Normal.(0, 1)
        w[1:J] .~ Normal.(0, 1)
        a ~ Normal(0, 1)
        s ~ Exponential(1)
    end
    for (name, layer) in (
        "named" => quote
            h = tanh.(c' .+ x * W)
            mu = a .+ h * w
        end,
        "inline" => quote
            mu = a .+ tanh.(c' .+ x * W) * w
        end,
    )
        @testset "$name" begin
            ast = Expr(:block, prior.args..., layer.args...,
                :(yc .~ Normal.(mu, s)))
            check(ast, (; x, J, yc), (:yc,), oracle)
        end
    end
end

@testset "a data-only elementwise matrix is the matrix of a product" begin
    # Centering a design matrix (`X = x .- colmean(x)`) builds X once at
    # binding; `X * b` is its matrix-vector product.
    ast = quote
        a ~ Normal(0, 1)
        b[axes(x, 2)] .~ Normal.(0, 1)
        s ~ Exponential(1)
        c = colmean(x)
        X = x .- c
        mu = a .+ X * b
        yc .~ Normal.(mu, s)
    end
    oracle(c, u) = logpdf(Normal(), c.a) + standard_normal(c.b) +
        logpdf(Exponential(1), c.s) + log(c.s) +
        sum(logpdf.(Normal.(c.a .+ (x .- colmean(x)) * c.b, c.s), yc))
    check(ast, (; x, yc), (:yc,), oracle)
end

end # module
