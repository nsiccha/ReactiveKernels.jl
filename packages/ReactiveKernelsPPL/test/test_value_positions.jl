module ValuePositionTests
using ReactiveKernels, ReactiveKernelsPPL, Distributions, DifferentiationInterface, Enzyme, Test

# A number bound to a data name is one position, as in Julia: called with
# `T = 3`, `h[T]` reads one element of a scan state and `z[N]` one element
# of a sized vector, in a broadcast location, a named definition and a
# plate cell alike, as the literal `h[3]` does (rkppl-use §3 "By
# position"). Integer arithmetic of bound numbers, literals and endpoints is
# the position it evaluates to, a definition of one reads the same, and
# binding checks the value as Julia indexing would.

const BACKEND = AutoEnzyme(; mode=Enzyme.Reverse)
const Y = [0.3, -0.8, 1.1, 0.2]

function differences(f, u)
    h = cbrt(eps(Float64))
    [begin
        up, down = copy(u), copy(u)
        up[i] += h
        down[i] -= h
        (f(up)-f(down))/(2h)
    end for i in eachindex(u)]
end

# Posterior value, ordinary Enzyme reverse and the printed-source replay
# against `oracle(p)`, an independent Distributions density of the
# constrained values `p`.
function check(bound, oracle)
    built = build_kernel(bound)
    sampler = prepare_sampler(built, bound, zeros(built.layout.total); backend=BACKEND)
    replayed = ReactiveKernelsPPL._eval_kernel_def(kernel_expr(bound, built.layout))
    replay = prepare_query((; spec=replayed, layout=built.layout), bound, :sampler)
    target(u) = oracle(constrain(built.layout, u)) + logjac(built.layout, u)
    for shift in (0.0, -0.4)
        u = [0.3sin(i) + shift for i in 1:built.layout.total]
        value, grad = sampler_value_and_gradient!(sampler, similar(u), u)
        @test value ≈ target(u)
        @test Base.invokelatest(replay, u) ≈ target(u)
        @test grad ≈ differences(target, u) rtol=1e-5 atol=1e-7
    end
    return built
end

ar_prior(h) = logpdf(Normal(0, 1), h[1]) +
    sum(logpdf(Normal(0.5 * h[t - 1], 1), h[t]) for t in 2:length(h); init = 0.0)
observed(loc) = sum(logpdf.(Normal.(loc, 1), Y))

const AR = @rkppl begin
    mu ~ Normal(0, 1)
    @scan begin
        h[1] ~ Normal(0, 1)
        for t in 2:T
            h[t] ~ Normal(0.5 * h[t - 1], 1)
        end
    end
    y .~ Normal.(mu .+ h[T], 1)
end

const AR_NAMED = @rkppl begin
    mu ~ Normal(0, 1)
    @scan begin
        h[1] ~ Normal(0, 1)
        for t in 2:T
            h[t] ~ Normal(0.5 * h[t - 1], 1)
        end
    end
    hT = h[T]
    y .~ Normal.(mu .+ hT, 1)
end

const AR_SHIFTED = @rkppl begin
    mu ~ Normal(0, 1)
    @scan begin
        h[1] ~ Normal(0, 1)
        for t in 2:T
            h[t] ~ Normal(0.5 * h[t - 1], 1)
        end
    end
    k = T - 1
    y .~ Normal.(mu .+ h[T - 1] .- 0.5 .* h[k], 1)
end

@testset "a scan state read at a bound number" begin
    for T in (3, 5)
        check(AR(; T) | (; y = Y),
            p -> logpdf(Normal(0, 1), p.mu) + ar_prior(p.h) + observed(p.mu .+ p.h[T]))
        check(AR_NAMED(; T) | (; y = Y),
            p -> logpdf(Normal(0, 1), p.mu) + ar_prior(p.h) + observed(p.mu .+ p.h[T]))
        check(AR_SHIFTED(; T) | (; y = Y),
            p -> logpdf(Normal(0, 1), p.mu) + ar_prior(p.h) +
                observed(p.mu .+ 0.5 .* p.h[T - 1]))
    end
end

const SIZED = @rkppl begin
    mu ~ Normal(0, 1)
    z[1:N] .~ Normal.(0, 1)
    y .~ Normal.(mu .+ z[N], 1)
end

const POSITIONED = @rkppl begin
    mu ~ Normal(0, 1)
    z[1:4] .~ Normal.(0, 1)
    s = exp(z[K] / 4)
    y .~ Normal.(mu .+ z[K] .- z[end - K + 1], s)
end

@testset "a sized vector read at a bound number" begin
    for N in (2, 3)
        check(SIZED(; N) | (; y = Y), p -> logpdf(Normal(0, 1), p.mu) +
            sum(logpdf.(Normal(0, 1), p.z)) + observed(p.mu .+ p.z[N]))
    end
    oracle(K) = p -> logpdf(Normal(0, 1), p.mu) + sum(logpdf.(Normal(0, 1), p.z)) +
        sum(logpdf.(Normal.(p.mu .+ p.z[K] .- p.z[5 - K], exp(p.z[K] / 4)), Y))
    for K in (1, 3)
        check(POSITIONED(; K) | (; y = Y), oracle(K))
    end
end

const CELL = @rkppl begin
    mu ~ Normal(0, 1)
    z[1:4] .~ Normal.(0, 1)
    @plate for i in eachindex(y)
        y[i] ~ Normal(mu + z[N], 1)
    end
end

const MATRIX = @rkppl begin
    mu ~ Normal(0, 1)
    B[1:3, 1:2] .~ Normal.(0, 1)
    s = sum(B[N, :])
    y .~ Normal.(mu .+ B[N, 2] .+ s, 1)
end

@testset "a plate cell and a matrix read at a bound number" begin
    prior(p) = logpdf(Normal(0, 1), p.mu)
    check(CELL(; N = 3) | (; y = Y), p -> prior(p) +
        sum(logpdf.(Normal(0, 1), p.z)) + observed(p.mu .+ p.z[3]))
    check(MATRIX(; N = 2) | (; y = Y), p -> prior(p) +
        sum(logpdf.(Normal(0, 1), p.B)) + observed(p.mu .+ p.B[2, 2] .+ sum(p.B[2, :])))
end

@testset "a plan lowered once reads each bound position" begin
    ast = quote
        mu ~ Normal(0, 1)
        z[1:4] .~ Normal.(0, 1)
        y .~ Normal.(mu .+ z[K], 1)
    end
    plan = lower_rkppl(ast, (; y = Y, K = 1); mod = @__MODULE__, conditioned = (:y,))
    built = build_kernel(bind_data(plan, (; y = Y, K = 1)))
    u = [0.3sin(i) for i in 1:built.layout.total]
    for K in 1:4
        bound = bind_data(plan, (; y = Y, K))
        p = constrain(built.layout, u)
        @test Base.invokelatest(prepare_query(built, bound, :sampler), u) ≈
            logpdf(Normal(0, 1), p.mu) + sum(logpdf.(Normal(0, 1), p.z)) +
            observed(p.mu .+ p.z[K]) + logjac(built.layout, u)
    end
end

@testset "binding checks a bound position as Julia indexing would" begin
    ast = quote
        mu ~ Normal(0, 1)
        z[1:3] .~ Normal.(0, 1)
        y .~ Normal.(mu .+ z[K], 1)
    end
    lowered(K) = lower_rkppl(ast, (; y = Y, K); mod = @__MODULE__, conditioned = (:y,))
    err = try bind_data(lowered(4), (; y = Y, K = 4)); nothing catch e; e end
    @test err !== nothing && occursin("out of bounds", sprint(showerror, err))
    err = try bind_data(lowered(1.5), (; y = Y, K = 1.5)); nothing catch e; e end
    @test err !== nothing && occursin("must be an integer", sprint(showerror, err))
end

@testset "a plan lowered from names alone names the remedy" begin
    ast = quote
        mu ~ Normal(0, 1)
        z[1:3] .~ Normal.(0, 1)
        y .~ Normal.(mu .+ z[K], 1)
    end
    plan = lower_rkppl(ast, (:y, :K); mod = @__MODULE__, conditioned = (:y,))
    err = try bind_data(plan, (; y = Y, K = 2)); nothing catch e; e end
    @test err !== nothing && occursin("lowered from data names alone", sprint(showerror, err))
end

end # module
