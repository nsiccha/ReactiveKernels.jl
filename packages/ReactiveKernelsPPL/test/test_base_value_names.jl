module BaseValueNameTests
using ReactiveKernels, ReactiveKernelsPPL, Distributions, DifferentiationInterface, Enzyme, Test

# `nothing` and `missing` keep Julia's meaning, Base's values, as keyword
# values and positional arguments of module calls, in data-only and
# parameter-dependent definitions (rkppl-use §2, "Keyword arguments pass
# through"; functions keep standard Julia semantics).

const BACKEND = AutoEnzyme(; mode=Enzyme.Reverse)

const x = [0.3, -0.5, 1.2, 0.8, -0.1]
const y = [0.1, -0.2, 0.9, 0.4, 0.0]

shifted(v; by = nothing) = by === nothing ? v : v .+ by
shifted_at(v, by) = by === nothing ? v : v .+ by
filled(v; by = missing) = by === missing ? v : v .+ by
filled_at(v, by) = by === missing ? v : v .+ by

const CALLS = Ref(0)
counted(v; by = nothing) = (CALLS[] += 1; shifted(v; by))

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

data_only_oracle(p, d) = logpdf(Normal(0, 1), p.b) +
    sum(logpdf.(Normal.(p.b .* d.x, 1), d.y))
dependent_oracle(p, d) = data_only_oracle(p, d)

# Each value name: data-only and parameter-dependent calls, as a keyword
# value and as a positional argument.
const CALLS_BY_NAME = (
    (:nothing, :shifted, :shifted_at),
    (:missing, :filled, :filled_at),
)

@testset "$name as a keyword value and a positional argument" for
        (name, kw, at) in CALLS_BY_NAME
    data = (; x, y)
    for call in (:($kw(x; by = $name)), :($at(x, $name)))
        ast = quote
            b ~ Normal(0, 1)
            w = $call
            y .~ Normal.(b .* w, 1)
        end
        check(lowered(ast, data), data, data_only_oracle)
    end
    for call in (:($kw(b .* x; by = $name)), :($at(b .* x, $name)))
        ast = quote
            b ~ Normal(0, 1)
            w = $call
            y .~ Normal.(w, 1)
        end
        check(lowered(ast, data), data, dependent_oracle)
    end
end

@testset "interpolated nothing and missing values read the same" begin
    # An AST emitter may interpolate the values themselves.
    data = (; x, y)
    for ast in (
            quote b ~ Normal(0, 1); w = shifted(b .* x; by = $nothing); y .~ Normal.(w, 1) end,
            quote b ~ Normal(0, 1); w = filled_at(b .* x, $missing); y .~ Normal.(w, 1) end)
        check(lowered(ast, data), data, dependent_oracle)
    end
end

@testset "a data-only call with nothing still runs once, at binding" begin
    ast = quote
        b ~ Normal(0, 1)
        w = counted(x; by = nothing)
        y .~ Normal.(b .* w, 1)
    end
    data = (; x, y)
    plan = lowered(ast, data)
    CALLS[] = 0
    bound = bind_data(plan, data)
    @test CALLS[] == 1
    built = build_kernel(bound)
    sampler = prepare_sampler(built, bound, zeros(built.layout.total); backend=BACKEND)
    u = [0.4]
    value, _ = sampler_value_and_gradient!(sampler, similar(u), u)
    @test value ≈ data_only_oracle(constrain(built.layout, u), data)
    @test CALLS[] == 1
end

@testset "nothing in prior arguments, scalar definitions and plate cells" begin
    data = (; x, y)
    ast = quote
        s ~ Exponential(something(nothing, 2.0))
        b ~ Normal(0, s)
        c = coalesce(missing, something(nothing, b))
        y .~ Normal.(c .* x, 1)
    end
    check(lowered(ast, data), data, (p, d) ->
        logpdf(Exponential(2.0), p.s) + logpdf(Normal(0, p.s), p.b) +
        sum(logpdf.(Normal.(p.b .* d.x, 1), d.y)))
    cells = quote
        b ~ Normal(0, 1)
        @plate for i in eachindex(y)
            y[i] ~ Normal(shifted_at(b * x[i], nothing), 1)
        end
    end
    check(lowered(cells, data), data, dependent_oracle)
end

@rkppl scaled_by(v; by = nothing) = begin
    a ~ Normal(0, 1)
    return shifted(a .* v; by = by)
end

@testset "a submodel keyword default of nothing" begin
    data = (; x, y)
    ast = quote
        m ~ scaled_by(x)
        y .~ Normal.(m, 1)
    end
    check(lowered(ast, data), data, (p, d) -> logpdf(Normal(0, 1), p.m.a) +
        sum(logpdf.(Normal.(p.m.a .* d.x, 1), d.y)))
end

@testset "a model name shadows nothing and missing" begin
    data = (; x, y)
    ast = quote
        b ~ Normal(0, 1)
        missing = 0.5
        w = filled_at(b .* x, missing)
        y .~ Normal.(w, 1)
    end
    check(lowered(ast, data), data, (p, d) -> logpdf(Normal(0, 1), p.b) +
        sum(logpdf.(Normal.(p.b .* d.x .+ 0.5, 1), d.y)))
end

# Base's named numbers (`Inf`, `NaN`, `pi`, `ℯ`, the typed `Inf32`, …) are
# how Julia writes those numbers: each reads as the number, like `0.5`, in
# every position that reads a value (rkppl-use §2). Snag
# `rkppl-assignment-88ffa090`: `w = ifelse.(flag .> 0.5, -Inf, 0.0)` failed
# binding with "references unknown name Inf".

const flag = [0.0, 1.0, 0.0, 0.0, 1.0]

# exp(-Inf) is zero: flagged rows read `b .* x`, the others `b .* x .+ 1`.
masked_oracle(p, d) = logpdf(Normal(0, 1), p.b) +
    sum(logpdf.(Normal.(p.b .* d.x .+ (d.flag .<= 0.5), 1), d.y))

@testset "Inf in data-only and parameter-dependent definitions" begin
    data = (; x, flag, y)
    for ast in (
            # The reporter's spelling: a data-only structural zero.
            quote
                b ~ Normal(0, 1)
                w = ifelse.(flag .> 0.5, -Inf, 0.0)
                y .~ Normal.(b .* x .+ exp.(w), 1)
            end,
            # Parameter-dependent: the infinity meets a live value.
            quote
                b ~ Normal(0, 1)
                w = ifelse.(flag .> 0.5, -Inf, 0.0 .* b)
                y .~ Normal.(b .* x .+ exp.(w), 1)
            end,
            # A scalar definition and a call argument.
            quote
                b ~ Normal(0, 1)
                c = max(b, -Inf)
                w = min.(log.(1 .- flag), Inf)
                y .~ Normal.(c .* x .+ exp.(w), 1)
            end,
            # Inline in the response.
            quote
                b ~ Normal(0, 1)
                y .~ Normal.(b .* x .+ exp.(ifelse.(flag .> 0.5, -Inf, 0.0)), 1)
            end)
        check(lowered(ast, data), data, masked_oracle)
    end
end

@testset "Inf in a plate cell" begin
    data = (; x, flag, y)
    for ast in (
            quote
                b ~ Normal(0, 1)
                @plate for i in eachindex(y)
                    w = flag[i] > 0.5 ? -Inf : 0.0
                    y[i] ~ Normal(b * x[i] + exp(w), 1)
                end
            end,
            quote
                b ~ Normal(0, 1)
                @plate for i in eachindex(y)
                    y[i] ~ Normal(b * x[i] + exp(ifelse(flag[i] > 0.5, -Inf, 0.0)), 1)
                end
            end)
        check(lowered(ast, data), data, masked_oracle)
    end
end

@testset "NaN, pi, ℯ and typed infinities read as Base's numbers" begin
    data = (; x, y)
    ast = quote
        b ~ Normal(0, 1)
        c = isnan(NaN) & (Float64(-Inf32) == -Inf) & isinf(Inf16) ?
            2pi * log(ℯ) : 0.0
        y .~ Normal.(b .* x .+ c, 1)
    end
    check(lowered(ast, data), data, (p, d) -> logpdf(Normal(0, 1), p.b) +
        sum(logpdf.(Normal.(p.b .* d.x .+ 2pi, 1), d.y)))
    # A bare prior argument and a bound.
    ast = quote
        b ~ Normal(0, pi)
        s ~ truncated(Normal(0, ℯ), -Inf, Inf)
        y .~ Normal.(b .* x .+ s, 1)
    end
    check(lowered(ast, data), data, (p, d) -> logpdf(Normal(0, pi), p.b) +
        logpdf(Normal(0, ℯ), p.s) + sum(logpdf.(Normal.(p.b .* d.x .+ p.s, 1), d.y)))
end

@testset "a model name shadows Base's numbers" begin
    data = (; x, y)
    ast = quote
        b ~ Normal(0, 1)
        NaN = 0.5
        w = b .* x .+ NaN
        y .~ Normal.(w, 1)
    end
    check(lowered(ast, data), data, (p, d) -> logpdf(Normal(0, 1), p.b) +
        sum(logpdf.(Normal.(p.b .* d.x .+ 0.5, 1), d.y)))
end

# A module's own binding of the name shadows Base's, as in Julia: there it
# is the module's numeric constant, an input name that data binds (§2).
module OwnNaN
    const NaN = 0.5
end

@testset "a model module's own binding shadows Base's number" begin
    ast = quote
        b ~ Normal(0, 1)
        w = b .* x .+ NaN
        y .~ Normal.(w, 1)
    end
    # refused: Julia reads `NaN` as OwnNaN's own constant, which data must bind
    @test_throws ReactiveKernelsPPL.SurfaceLoweringError lower_rkppl(ast,
        (:x, :y); mod = OwnNaN, conditioned = (:y,))
    check(lower_rkppl(ast, (:x, :y, :NaN); mod = OwnNaN, conditioned = (:y,)),
        (; x, y, NaN = 0.5), (p, d) -> logpdf(Normal(0, 1), p.b) +
            sum(logpdf.(Normal.(p.b .* d.x .+ 0.5, 1), d.y)))
end
end
