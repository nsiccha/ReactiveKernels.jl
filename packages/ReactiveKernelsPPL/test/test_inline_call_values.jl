module InlineCallValueTests
using ReactiveKernels, ReactiveKernelsPPL, Distributions, DifferentiationInterface, Enzyme, Test

# A module call written inline among an observation's distribution
# arguments means what its named definition means: naming a
# subexpression never changes legality or the density (rkppl-use §2, §9).
# Data-only and parameter-dependent calls, inside sums, under links, in
# mixture components and in a submodel body defined in another module.

const BACKEND = AutoEnzyme(; mode=Enzyme.Reverse)

const x = [0.3, -0.5, 1.2, 0.8, -0.1]
const y = [0.1, -0.2, 0.9, 0.4, 0.0]
const k = [1, 0, 3, 2, 1]
const g = [1, 2, 3, 2, 1]
const gx = [0.5, -1.0, 2.0]   # one value per group, not per observation
const yc = [1, 3, 2, 2, 1]

shiftp(v, by) = v .+ by
twice_plus(b) = 2b + 0.1
per_group(b, v) = b .* v

const CALLS = Ref(0)
counted(v; by = 0.5) = (CALLS[] += 1; v .+ by)

module Elsewhere
using ReactiveKernelsPPL
# Visible only here: the submodel body resolves it in this module.
offset_by(v) = v .+ 0.25
@rkppl observe_offset(xx, yy) = begin
    b ~ Normal(0, 1)
    yy .~ Normal.(offset_by(b .* xx), 1)
    return b
end
end

function differences(f, u)
    h = cbrt(eps(Float64))
    [begin
        up, down = copy(u), copy(u)
        up[i] += h
        down[i] -= h
        (f(up)-f(down))/(2h)
    end for i in eachindex(u)]
end

lowered(ast, data; conditioned=(:y,)) = lower_rkppl(ast, Tuple(keys(data));
    mod=@__MODULE__, conditioned)

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

program(bound, layout) =
    string(Base.remove_linenums!(deepcopy(kernel_expr(bound, layout))))

@testset "an inline call lowers exactly like its named definition" begin
    # The inline call takes the name of the argument it fills, `y_location`;
    # spelled with that name, the named program is the same generated program.
    data = (; x, y)
    inline = quote
        b ~ Normal(0, 1)
        y .~ Normal.(shiftp(b .* x, 0.3), 1)
    end
    named = quote
        b ~ Normal(0, 1)
        y_location = shiftp(b .* x, 0.3)
        y .~ Normal.(y_location, 1)
    end
    bi = bind_data(lowered(inline, data), data)
    bn = bind_data(lowered(named, data), data)
    @test program(bi, assign_layout(bi)) == program(bn, assign_layout(bn))
    check(lowered(inline, data), data, (p, d) -> logpdf(Normal(0, 1), p.b) +
        sum(logpdf.(Normal.(p.b .* d.x .+ 0.3, 1), d.y)))
end

@testset "parameter-dependent calls in locations, sums, links and scalars" begin
    data = (; x, y)
    check(lowered(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        y .~ Normal.(a .+ shiftp(b .* x, 0.3), 1)
    end, data), data, (p, d) -> logpdf(Normal(0, 1), p.a) +
        logpdf(Normal(0, 1), p.b) +
        sum(logpdf.(Normal.(p.a .+ p.b .* d.x .+ 0.3, 1), d.y)))
    check(lowered(quote
        b ~ Normal(0, 1)
        y .~ Normal.(twice_plus(b), 1)
    end, data), data, (p, d) -> logpdf(Normal(0, 1), p.b) +
        sum(logpdf.(Normal(2p.b + 0.1, 1), d.y)))
    counts = (; x, k)
    check(lowered(quote
        b ~ Normal(0, 1)
        k .~ Poisson.(exp.(shiftp(b .* x, -0.2)))
    end, counts; conditioned=(:k,)), counts, (p, d) ->
        logpdf(Normal(0, 1), p.b) +
        sum(logpdf.(Poisson.(exp.(p.b .* d.x .- 0.2)), d.k)))
    # The same inline call read twice keeps one value for both reads.
    check(lowered(quote
        b ~ Normal(0, 1)
        s ~ Exponential(1)
        y .~ Normal.(shiftp(b .* x, 0.3), exp.(shiftp(b .* x, 0.3)) .+ s)
    end, data), data, (p, d) -> logpdf(Normal(0, 1), p.b) +
        logpdf(Exponential(1), p.s) +
        sum(logpdf.(Normal.(p.b .* d.x .+ 0.3,
            exp.(p.b .* d.x .+ 0.3) .+ p.s), d.y)))
end

@testset "a data-only inline call runs once, at binding" begin
    data = (; x, y)
    for (ast, oracle) in (
            (quote
                s ~ Exponential(1)
                y .~ Normal.(counted(x), s)
            end, (p, d) -> logpdf(Exponential(1), p.s) +
                sum(logpdf.(Normal.(d.x .+ 0.5, p.s), d.y))),
            (quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                y .~ Normal.(a .+ b .* counted(x; by = 0.25), 1)
            end, (p, d) -> logpdf(Normal(0, 1), p.a) +
                logpdf(Normal(0, 1), p.b) +
                sum(logpdf.(Normal.(p.a .+ p.b .* (d.x .+ 0.25), 1), d.y))))
        plan = lowered(ast, data)
        CALLS[] = 0
        bound = bind_data(plan, data)
        @test CALLS[] == 1
        built = build_kernel(bound)
        sampler = prepare_sampler(built, bound, zeros(built.layout.total);
            backend=BACKEND)
        u = [0.2 + 0.1i for i in 1:built.layout.total]
        value, _ = sampler_value_and_gradient!(sampler, similar(u), u)
        @test value ≈ oracle(constrain(built.layout, u), data) +
            logjac(built.layout, u)
        @test CALLS[] == 1
        check(plan, data, oracle)
    end
end

@testset "whole-value data passed to an inline call keeps its own length" begin
    # `gx` has one value per group; only the gather `[g]` meets the rows.
    data = (; g, gx, y)
    check(lowered(quote
        b ~ Normal(0, 1)
        y .~ Normal.(per_group(b, gx)[g], 1)
    end, data), data, (p, d) -> logpdf(Normal(0, 1), p.b) +
        sum(logpdf.(Normal.(p.b .* d.gx[d.g], 1), d.y)))
end

@testset "mixture components and categorical logits" begin
    data = (; x, y)
    check(lowered(quote
        b ~ Normal(0, 1)
        y .~ MixtureModel.(vcat.(Normal.(shiftp(b .* x, 0.0), 1),
            Normal.(0, 2)), Ref([0.3, 0.7]))
    end, data), data, (p, d) -> logpdf(Normal(0, 1), p.b) +
        sum(logpdf.(MixtureModel.(vcat.(Normal.(p.b .* d.x, 1),
            Normal.(0, 2)), Ref([0.3, 0.7])), d.y)))
    cats = (; x, yc)
    check(lowered(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        yc .~ CategoricalLogit.(shiftp(b .* x, 0.0), a)
    end, cats; conditioned=(:yc,)), cats, (p, d) ->
        logpdf(Normal(0, 1), p.a) + logpdf(Normal(0, 1), p.b) +
        sum(begin
            eta = [0.0, p.b * d.x[i], p.a]
            eta[d.yc[i]] - log(sum(exp.(eta)))
        end for i in eachindex(d.yc)))
end

@testset "ordinal locations" begin
    sigmoid(z) = 1 / (1 + exp(-z))
    ords = (; x, yc)
    check(lowered(quote
        c ~ Ordered(Normal(0, 1), 2)
        b ~ Normal(0, 1)
        yc .~ OrderedLogistic.(shiftp(b .* x, 0.2), Ref(c))
    end, ords; conditioned=(:yc,)), ords, (p, d) ->
        sum(logpdf.(Normal(0, 1), p.c)) + logpdf(Normal(0, 1), p.b) +
        sum(begin
            eta = p.b * d.x[i] + 0.2
            below(k) = k == 0 ? 0.0 : k == 3 ? 1.0 : sigmoid(p.c[k] - eta)
            log(below(d.yc[i]) - below(d.yc[i] - 1))
        end for i in eachindex(d.yc)))
end

@testset "a submodel body's inline call resolves in its own module" begin
    @test !isdefined(@__MODULE__, :offset_by)
    data = (; x, y)
    check(lowered(quote
        m ~ Elsewhere.observe_offset(x, y)
    end, data), data, (p, d) -> logpdf(Normal(0, 1), p.m.b) +
        sum(logpdf.(Normal.(p.m.b .* d.x .+ 0.25, 1), d.y)))
end

module Unbound end

@testset "an undefined function fails inline as it does named" begin
    # refused: `nowhere_defined` has no binding in the model module, and
    # naming the call does not change that (rkppl-use §2 "How a call
    # resolves": lowering fails naming an undefined function).
    data = (; x, y)
    for ast in (
            quote b ~ Normal(0, 1); y .~ Normal.(nowhere_defined(b .* x), 1) end,
            quote b ~ Normal(0, 1); w = nowhere_defined(b .* x); y .~ Normal.(w, 1) end)
        err = try
            lower_rkppl(ast, (:x, :y); mod=Unbound, conditioned=(:y,))
            nothing
        catch e
            e
        end
        @test err isa ReactiveKernelsPPL.SurfaceLoweringError
        msg = sprint(showerror, err)
        @test occursin("calls `nowhere_defined`, which is not defined " *
            "in module `Unbound`", msg)
        @test !occursin("slice-1", msg)
    end
end
end
