using CategoricalArrays
using DataAPI
using DifferentiationInterface: AutoEnzyme
using Distributions: Normal, logpdf
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
    u = collect(range(-0.2, 0.4; length = built.layout.total))
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
