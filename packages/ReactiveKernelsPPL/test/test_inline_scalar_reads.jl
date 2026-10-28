module InlineScalarReadTests
using ReactiveKernels, ReactiveKernelsPPL, Distributions, DifferentiationInterface, Enzyme, Test

# A scalar subexpression that reads a column wholly (`1 + sum(x)^2`,
# `sum(th)` of a plate latent) inside a composed location is one scalar
# value, exactly as its named definition `m = 1 + sum(x)^2` is: naming a
# subexpression never changes legality or the density (rkppl-use §2, §9).

const BACKEND = AutoEnzyme(; mode=Enzyme.Reverse)

const x = [0.3, -0.5, 1.2, 0.8, -0.1]
const y = [0.1, -0.2, 0.9, 0.4, 0.0]
const c = [0.5, 1.1, 2.0, 1.4, 0.7]

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
    conditioned=(:y, :c))

# Posterior value, ordinary Enzyme reverse and the printed-source replay
# against `oracle(p, data)`, an independent Distributions density of the
# constrained values `p`; returns the posterior at the first point.
function check(plan, data, oracle)
    original = deepcopy(data)
    bound = bind_data(plan, data)
    built = build_kernel(bound)
    sampler = prepare_sampler(built, bound, zeros(built.layout.total); backend=BACKEND)
    replayed = ReactiveKernelsPPL._eval_kernel_def(kernel_expr(bound, built.layout))
    replay = prepare_query((; spec=replayed, layout=built.layout), bound, :sampler)
    target(u) = oracle(constrain(built.layout, u), data) + logjac(built.layout, u)
    first_value = nothing
    for shift in (0.0, -0.4)
        u = [0.3sin(i) + shift for i in 1:built.layout.total]
        saved = copy(u)
        value, grad = sampler_value_and_gradient!(sampler, similar(u), u)
        first_value === nothing && (first_value = value)
        @test value ≈ target(u)
        @test Base.invokelatest(replay, u) ≈ target(u)
        @test grad ≈ differences(target, u) rtol=1e-5 atol=1e-7
        @test u == saved
    end
    @test data == original
    return first_value
end

const PRELUDE = quote
    s ~ Exponential(1)
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    eta = a .+ b .* x
    @plate for i in eachindex(y)
        th[i] ~ Normal(0, 1)
        y[i] ~ Normal(th[i] + x[i], s)
    end
end

model(statements...) = Expr(:block, PRELUDE.args..., statements...)

prelude_density(p, d) = logpdf(Exponential(1), p.s) +
    logpdf(Normal(0, 1), p.a) + logpdf(Normal(0, 1), p.b) +
    sum(logpdf.(Normal(0, 1), p.th)) +
    sum(logpdf.(Normal.(p.th .+ d.x, p.s), d.y))

# Each scalar factor as a function of the constrained values and data,
# with its inline spelling.
const FACTORS = (
    ("1 + sum(x)^2", :(1 + sum(x)^2), (p, d) -> 1 + sum(d.x)^2),
    ("1 + sum(th)^2", :(1 + sum(th)^2), (p, d) -> 1 + sum(p.th)^2),
    ("sum(x)^2", :(sum(x)^2), (p, d) -> sum(d.x)^2),
    ("(sum(th) + 3) / 2", :((sum(th) + 3) / 2), (p, d) -> (sum(p.th) + 3) / 2),
)

@testset "inline $label matches its named twin" for (label, factor, value) in FACTORS
    data = (; x, y, c)
    oracle(p, d) = prelude_density(p, d) + sum(logpdf.(Normal.(
        exp.(p.a .+ p.b .* d.x) .* value(p, d), p.s), d.c))
    inline = check(lowered(model(
        :(c .~ Normal.(exp.(eta) .* $factor, s))), data), data, oracle)
    named = check(lowered(model(:(m = $factor),
        :(c .~ Normal.(exp.(eta) .* m, s))), data), data, oracle)
    @test inline ≈ named
end
end
