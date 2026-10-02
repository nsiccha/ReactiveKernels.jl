using CategoricalArrays
using DataAPI
using DifferentiationInterface: AutoEnzyme
using Distributions: Normal, Bernoulli, Binomial, Poisson, TDist, censored,
    truncated, cdf, logpdf
using Enzyme
using ReactiveKernelsPPL
using Test

# Independent Julia/Distributions oracles, including the gradient of the
# generated likelihood. Every probe also checks caller-owned data stays intact.
function _semantics_check(ast, data, oracle; names = nothing)
    original = deepcopy(data)
    bound = bind_data(lower_rkppl(ast, data), data)
    built = build_kernel(bound)
    names === nothing || @test coordinate_names(built.layout) == names
    u = collect(range(-0.2; step = 0.15, length = built.layout.total))
    q = prepare_query(built, bound, :likelihood)
    reference(v) = oracle(constrain(built.layout, v))
    @test Base.invokelatest(q, u) ≈ reference(u) rtol = 1e-12 atol = 1e-12
    sampler = prepare_sampler(built, bound, u;
        backend = AutoEnzyme(; mode = Enzyme.Reverse))
    grad = similar(u)
    value, _ = sampler_value_and_gradient!(sampler, grad, u)
    posterior_query = prepare_query(built, bound, :sampler)
    posterior(v) = Base.invokelatest(posterior_query, v)
    h = 1e-5
    fd = [begin
        hi, lo = copy(u), copy(u)
        hi[i] += h
        lo[i] -= h
        (posterior(hi) - posterior(lo)) / (2h)
    end for i in eachindex(u)]
    @test value ≈ posterior(u) rtol = 1e-12 atol = 1e-12
    @test grad ≈ fd rtol = 2e-5 atol = 2e-7
    @test data == original
    return (; bound, built, u)
end

@testset "Julia matrix intercept columns" begin
    x1, x2 = [-1.0, 0.0, 1.0], [0.5, -0.5, 0.2]
    data = Dict{Symbol,Any}(:x1 => x1, :x2 => x2, :y => [0.2, 0.4, -0.1])
    ast = quote
        b[axes(X, 2)] .~ Normal.(0, 1)
        X = hcat(ones(length(x1)), x1, x2)
        mu = X * b
        y .~ Normal.(mu, 1)
    end
    _semantics_check(ast, data, q -> sum(logpdf.(
        Normal.(hcat(ones(length(x1)), x1, x2) * q.b, 1), data[:y])))
    # The intercept can instead be a scalar outside the matrix product.
    outside = quote
        a ~ Normal(0, 1)
        b[axes(X, 2)] .~ Normal.(0, 1)
        X = hcat(x1, x2)
        mu = a .+ X * b
        y .~ Normal.(mu, 1)
    end
    _semantics_check(outside, data, q -> sum(logpdf.(
        Normal.(q.a .+ hcat(x1, x2) * q.b, 1), data[:y])))
    # refused: scalar hcat intercepts are Julia dimension errors (0tz0qfu,
    # matrix prong). The error gives both approved migration choices.
    bad = quote
        b[axes(X, 2)] .~ Normal.(0, 1)
        X = hcat(1, x1, x2)
        mu = X * b
        y .~ Normal.(mu, 1)
    end
    err = try lower_rkppl(bad, data); nothing catch e; e end
    @test err isa SurfaceLoweringError
    @test occursin("ones(length(x1))", sprint(showerror, err))
    @test occursin("mu = a .+ X * b", sprint(showerror, err))
end

@testset "Julia inverse-link function spellings" begin
    x = [-1.0, 0.0, 0.5, 1.0]
    for (link, inverse) in ((:normcdf, x -> cdf(Normal(), x)),
            (:cexpexp, x -> -expm1(-exp(x))))
        for (family, y) in ((:Bernoulli, [false, true, true, false]),
                (:Binomial, [0, 1, 2, 1]))
            prob = Expr(:., link, Expr(:tuple, :eta))
            rhs = family === :Bernoulli ? Expr(:., family, Expr(:tuple, prob)) :
                Expr(:., family, Expr(:tuple, :n, prob))
            ast = quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                eta = a .+ b .* x
                y .~ $rhs
            end
            data = Dict{Symbol,Any}(:x => x, :y => y, :n => fill(3, length(x)))
            _semantics_check(ast, data, q -> sum(
                logpdf(family === :Bernoulli ? Bernoulli(inverse(q.a + q.b * xi)) :
                    Binomial(3, inverse(q.a + q.b * xi)), yi)
                for (xi, yi) in zip(x, y)))
        end
    end
    for (old, replacement, explanation) in ((:probit, "normcdf", "inverse normal CDF"),
            (:cloglog, "cexpexp", "not its inverse")), family in (:Bernoulli, :Binomial)
        prob = Expr(:., old, Expr(:tuple, :eta))
        rhs = family === :Bernoulli ? Expr(:., family, Expr(:tuple, prob)) :
            Expr(:., family, Expr(:tuple, :n, prob))
        # refused: inverse links must have honest Julia names (0tz0qfu,
        # links prong); these old names are not aliases for the new functions.
        err = try
            lower_rkppl(quote
                a ~ Normal(0, 1)
                eta = a .+ 0 .* x
                y .~ $rhs
            end, (:x, :y, :n))
            nothing
        catch e
            e
        end
        @test err isa SurfaceLoweringError
        @test occursin(replacement, sprint(showerror, err))
        @test occursin(explanation, sprint(showerror, err))
    end
end

@testset "evidence rejects impossible response data" begin
    x = [-1.0, 0.0, 1.0]
    for (family, response) in ((:(Normal.(mu, 1)), [1.0, 2.0, 4.0]),
            (:(StudentT.(3.0, mu, 1)), [1.0, 2.0, 4.0]),
            (:(Poisson.(exp.(mu))), [1, 2, 4]))
        for wrap in (:truncated, :censored)
            distribution = Expr(:., wrap, Expr(:tuple, family, :lo, :hi))
            ast = quote
                a ~ Normal(0, 1)
                mu = a .+ 0 .* x
                y .~ $distribution
            end
            data = Dict{Symbol,Any}(:x => x, :y => response,
                :lo => [1, 1, 1], :hi => [4, 4, 4])
            bound = bind_data(lower_rkppl(ast, data), data)
            @test isbound(bound) # both endpoints are admitted
            invalid = copy(response)
            invalid[1], invalid[3] = 0, 5
            bad_data = merge(data, Dict(:y => invalid))
            # refused: impossible evidence is rejected at bind (0tz0qfu,
            # evidence prong), rather than scored as clamped/in-range data.
            err = try
                bind_data(lower_rkppl(ast, bad_data), bad_data)
                nothing
            catch e
                e
            end
            @test err isa ContractValidationError
            message = sprint(showerror, err)
            @test occursin("response y", message)
            @test occursin("rows 1, 3", message)
            # One-sided bounds check only their own side.
            for (lo, hi, bad_y) in ((:lo, Inf, [0, 2, 4]),
                    (-Inf, :hi, [1, 2, 5]))
                dist = Expr(:., wrap, Expr(:tuple, family, lo, hi))
                one = quote
                    a ~ Normal(0, 1)
                    mu = a .+ 0 .* x
                    y .~ $dist
                end
                values = merge(data, Dict(:y => bad_y))
                @test_throws ContractValidationError bind_data(
                    lower_rkppl(one, values), values)
            end
        end
    end
    # In-range Gaussian evidence still agrees with Distributions, including
    # the mass at either censored endpoint and truncation normalization.
    for (wrap, ctor) in ((:censored, censored), (:truncated, truncated))
        ast = quote
            a ~ Normal(0, 1)
            mu = a .+ 0 .* x
            y .~ $(Expr(:., wrap, Expr(:tuple, :(Normal.(mu, 1)), 1, 4)))
        end
        data = Dict{Symbol,Any}(:x => x, :y => [1.0, 2.0, 4.0])
        _semantics_check(ast, data, q ->
            sum(logpdf.(ctor(Normal(q.a, 1), 1, 4), data[:y])))
    end
end

@testset "Julia factor pool order and unobserved levels" begin
    pool = ["low", "mid", "high", "extra"]
    g = categorical(["low", "mid", "high", "low"]; levels = pool)
    data = Dict{Symbol,Any}(:g => g, :y => [0.2, -0.1, 0.4, 0.3])
    ast = quote
        a ~ Normal(0, 1)
        c[levels(g)[2:end]] .~ Normal.(0, 1)
        mu = a .+ c[g]
        y .~ Normal.(mu, 1)
    end
    result = _semantics_check(ast, data, q -> sum(
        logpdf(Normal(q.a + (gi == "low" ? 0.0 :
            q.c[findfirst(==(gi), pool) - 1]), 1), yi)
        for (gi, yi) in zip(g, data[:y]));
        names = [:a, Symbol("c.1"), Symbol("c.2"), Symbol("c.3")])
    @test result.built.layout.total == 4 # the unobserved extra has a prior
    @test only(result.bound.levelmaps).values == pool[2:end]
    @test DataAPI.levels(g) == pool
    @test ReactiveKernelsPPL._grouping_levels([3, 1, 3, 2]) == [1, 2, 3]
    @test ReactiveKernelsPPL._grouping_levels(Symbol[:z, :a]) == [:a, :z]
    @test ReactiveKernelsPPL._grouping_levels(g[1:0]) == pool
    # A full cover and a literal subset both select from the same pool.
    for (axis, selected) in ((:(levels(g)), pool),
            (:(levels(g)[[1, 4]]), pool[[1, 4]]))
        program = quote
            c[$axis] .~ Normal.(0, 1)
            mu = c[g]
            y .~ Normal.(mu, 1)
        end
        bound = bind_data(lower_rkppl(program, data), data)
        @test only(bound.levelmaps).values == selected
    end
end
