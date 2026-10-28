module ReductionCallTests
using ReactiveKernels, ReactiveKernelsPPL, Distributions, DifferentiationInterface, Enzyme, Test
using LogExpFunctions: logsumexp
using Statistics: mean, std, var

# A built-in reduction keeps its whole-column reading in its one-argument
# form (`sum(x)`). Every other call is Julia's function, as the qualified
# `Base.sum(x; dims = 2)` is: keywords (`dims`, `init`, `corrected`) and a
# function argument (`sum(abs2, W)`) keep their Julia meaning, with the
# result's shape following `dims` (rkppl-use §2, "How a call resolves").

const BACKEND = AutoEnzyme(; mode=Enzyme.Reverse)

const u = [0.4 -1.1 0.3; 1.2 0.2 -0.7; -0.5 0.9 0.1; 0.0 -0.3 1.4;
    0.8 0.6 -0.2; -1.3 0.1 0.5]
const x = [0.3, -0.5, 1.2, 0.8, -0.1, 0.6]
const y = [0.1, -0.2, 0.9, 0.4, 0.0, 0.7]
const DATA = (; u, x, y)

function differences(f, p)
    h = cbrt(eps(Float64))
    [begin
        up, down = copy(p), copy(p)
        up[i] += h
        down[i] -= h
        (f(up)-f(down))/(2h)
    end for i in eachindex(p)]
end

lowered(ast, data) = lower_rkppl(ast, Tuple(keys(data)); mod=@__MODULE__,
    conditioned=(:y,))

# Posterior value, ordinary Enzyme reverse and the printed-source replay
# against `oracle(p)`, an independent Distributions density of the
# constrained values `p`.
function check(plan, oracle; data = DATA)
    original = deepcopy(data)
    bound = bind_data(plan, data)
    built = build_kernel(bound)
    sampler = prepare_sampler(built, bound, zeros(built.layout.total); backend=BACKEND)
    replayed = ReactiveKernelsPPL._eval_kernel_def(kernel_expr(bound, built.layout))
    replay = prepare_query((; spec=replayed, layout=built.layout), bound, :sampler)
    target(p) = oracle(constrain(built.layout, p)) + logjac(built.layout, p)
    for shift in (0.0, -0.4)
        p = [0.3sin(i) + shift for i in 1:built.layout.total]
        saved = copy(p)
        value, grad = sampler_value_and_gradient!(sampler, similar(p), p)
        @test value ≈ target(p)
        @test Base.invokelatest(replay, p) ≈ target(p)
        @test grad ≈ differences(target, p) rtol=1e-5 atol=1e-7
        @test p == saved
    end
    @test data == original
    return built
end

function softmax_model(definitions...)
    Expr(:block, :(W[axes(u, 2), 1:2] .~ Normal.(0, 5)), :(U = u * W),
        definitions..., :(s ~ Exponential(1)), :(y .~ Normal.(P[:, 1], s)))
end
softmax_rows(U) = exp.(U .- logsumexp(U; dims = 2))
softmax_oracle(p) = sum(logpdf.(Normal(0, 5), p.W)) +
    logpdf(Exponential(1), p.s) +
    sum(logpdf.(Normal.(softmax_rows(u * p.W)[:, 1], p.s), y))

@rkppl row_softmax(U) = begin
    P = exp.(U .- log.(sum(exp.(U); dims = 2)))
    return P
end

@testset "row-normalizing a parameter-dependent matrix with `dims`" begin
    for definitions in (
        (:(P = exp.(U .- log.(sum(exp.(U); dims = 2)))),),
        (:(P = exp.(U) ./ sum(exp.(U), dims = 2)),),
        (:(S = sum(exp.(U); dims = 2)), :(P = exp.(U) ./ S)),
        (:(m = maximum(U; dims = 2)), :(E = exp.(U .- m)),
            :(P = E ./ sum(E; dims = 2))),
        (:(E = exp.(U)), :(P = E ./ (2 .* mean(E; dims = 2)))),
    )
        ast = softmax_model(definitions...)
        check(lowered(ast, DATA), softmax_oracle)
        check(lower_rkppl(ast, DATA; mod=@__MODULE__, conditioned=(:y,)),
            softmax_oracle)
    end
    # The bare, qualified and route-around spellings are the same program.
    spellings = (
        :(P = exp.(U .- log.(sum(exp.(U); dims = 2)))),
        :(P = exp.(U .- log.(Base.sum(exp.(U); dims = 2)))),
        :(P = exp.(U) ./ sum.(eachrow(exp.(U)))),
    )
    values = map(spellings) do definition
        bound = bind_data(lowered(softmax_model(definition), DATA), DATA)
        built = build_kernel(bound)
        q = prepare_query(built, bound, :sampler)
        Base.invokelatest(q, [0.2cos(i) for i in 1:built.layout.total])
    end
    @test values[1] ≈ values[2]
    @test values[1] ≈ values[3]
    # The generated program calls Julia's `sum` with the keyword.
    bound = bind_data(lowered(softmax_model(first(spellings)), DATA), DATA)
    built = build_kernel(bound)
    @test occursin("Base.sum(exp.(U); dims = 2)",
        string(kernel_expr(bound, built.layout)))
end

@testset "`dims` keeps the reduced axis; other keywords and function arguments" begin
    scalar(f) = p -> sum(logpdf.(Normal(0, 5), p.W)) +
        logpdf(Exponential(1), p.s) +
        sum(logpdf.(Normal.(f(p.W) .+ 0 .* y, p.s), y))
    for (definition, f) in (
        (:(k = sum(W; dims = 1)[2]), W -> sum(W; dims = 1)[2]),
        (:(k = sum(W; dims = 1)[1, 2]), W -> sum(W; dims = 1)[1, 2]),
        (:(k = sum(abs2, W)), W -> sum(abs2, W)),
        (:(k = sum(abs2, W; dims = 2)[3]), W -> sum(abs2, W; dims = 2)[3]),
        (:(k = std(W; dims = 1, corrected = false)[1] + var(W; dims = 2)[3]),
            W -> std(W; dims = 1, corrected = false)[1] + var(W; dims = 2)[3]),
        (:(k = sum(W; dims = (1, 2))[1, 1]), W -> sum(W)),
    )
        ast = Expr(:block, :(W[axes(u, 2), 1:2] .~ Normal.(0, 5)), definition,
            :(s ~ Exponential(1)), :(y .~ Normal.(k .+ 0 .* y, s)))
        check(lowered(ast, DATA), scalar(f))
    end
    ast = quote
        a ~ Normal(0, 1)
        z[1:3] .~ Normal.(0, 1)
        k = a + sum(z; init = 0.0)
        s ~ Exponential(1)
        y .~ Normal.(k, s)
    end
    check(lowered(ast, DATA), p -> logpdf(Normal(0, 1), p.a) +
        sum(logpdf.(Normal(0, 1), p.z)) + logpdf(Exponential(1), p.s) +
        sum(logpdf.(Normal.(p.a + sum(p.z; init = 0.0), p.s), y)))
end

@testset "data-only reductions, prior arguments and submodel bodies" begin
    c = mean(u; dims = 1)
    ast = quote
        a ~ Normal(0, 1)
        b[axes(u, 2)] .~ Normal.(0, 1)
        c = mean(u; dims = 1)
        mu = a .+ u * b .- (c * b)[1]
        s ~ Exponential(1)
        y .~ Normal.(mu, s)
    end
    check(lowered(ast, DATA), p -> logpdf(Normal(0, 1), p.a) +
        sum(logpdf.(Normal(0, 1), p.b)) + logpdf(Exponential(1), p.s) +
        sum(logpdf.(Normal.(p.a .+ (u .- c) * p.b, p.s), y)))
    ast = quote
        b ~ Normal(0, std(x; corrected = false))
        s ~ Exponential(1)
        y .~ Normal.(b .* x, s)
    end
    check(lowered(ast, DATA), p ->
        logpdf(Normal(0, std(x; corrected = false)), p.b) +
        logpdf(Exponential(1), p.s) + sum(logpdf.(Normal.(p.b .* x, p.s), y)))
    ast = quote
        W[axes(u, 2), 1:2] .~ Normal.(0, 5)
        U = u * W
        P ~ row_softmax(U)
        s ~ Exponential(1)
        y .~ Normal.(P[:, 1], s)
    end
    check(lowered(ast, DATA), softmax_oracle)
end

end
