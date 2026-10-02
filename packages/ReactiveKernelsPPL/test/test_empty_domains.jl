using DifferentiationInterface: AutoEnzyme
using Distributions: Normal, Exponential, Uniform, logpdf
using Enzyme
using ReactiveKernelsPPL
using Test

_empty_domain_scale(z) = exp(sum(z))

# Independent densities use the stated distributions, including the empty
# product identity. The same cases exercise actual sampler reverse mode.
function _empty_domain_check(ast, data, oracle; dims = nothing)
    original = deepcopy(data)
    raw = lower_rkppl(ast, data; conditioned = filter(k -> k in (:y, :y2), keys(data)))
    plan = bind_data(raw, Dict{Symbol,Any}(pairs(data)))
    built = build_kernel(plan)
    u = [0.13 + 0.07i for i in 1:built.layout.total]
    ref(v) = oracle(constrain(built.layout, v))
    expected = ref(u)
    for (preset, value) in ((:likelihood, expected.ll), (:prior, expected.pr),
            (:log_jacobian, expected.jac), (:sampler, sum(values(expected))))
        q = prepare_query(built, plan, preset)
        @test Base.invokelatest(q, u) ≈ value atol = 1e-12
    end
    sampler = prepare_sampler(built, plan, u;
        backend = AutoEnzyme(; mode = Enzyme.Reverse))
    grad = similar(u)
    value, _ = sampler_value_and_gradient!(sampler, grad, u)
    @test value ≈ sum(values(expected)) atol = 1e-12
    fd = map(eachindex(u)) do i
        hi, lo = copy(u), copy(u)
        hi[i] += 1e-5
        lo[i] -= 1e-5
        (sum(values(ref(hi))) - sum(values(ref(lo)))) / 2e-5
    end
    @test grad ≈ fd rtol = 2e-5 atol = 1e-7
    nt = constrain(built.layout, u)
    @test unconstrain(built.layout, nt) ≈ u
    if dims !== nothing
        @test size(nt.z) == dims
        @test eltype(nt.z) === Float64
    end
    @test data == original
    @test all(k -> raw.columns[k] == original[k], keys(raw.columns))
    return (; plan, built, u, sampler, grad, oracle)
end

function _empty_observation_fixture(n)
    ast = quote
        a ~ Normal(0, 1)
        sigma ~ Exponential(1)
        y .~ Normal.(a, sigma)
    end
    data = (; y = [0.2sin(i) for i in 1:n])
    oracle(q) = (; ll = sum(logpdf.(Normal(q.a, q.sigma), data.y)),
        pr = logpdf(Normal(), q.a) + logpdf(Exponential(), q.sigma),
        jac = log(q.sigma))
    return (; ast, data, oracle)
end

@testset "empty observations retain proper scalar priors" begin
    for n in (0, 1, 3, 9)
        f = _empty_observation_fixture(n)
        r = _empty_domain_check(f.ast, f.data, f.oracle)
        @test r.plan.n_obs == n
        @test r.built.layout.total == 2
        if n == 0
            @test r.grad ≈ [-r.u[1], 1 - exp(r.u[2])]
        end
    end
    # Different observation axes already admit an empty member. Preserve it
    # when both axes become empty, too.
    for n in (0, 3)
        data = (; y = Float64[], y2 = fill(0.2, n))
        ast = quote
            a ~ Normal(0, 1)
            y .~ Normal.(a, 1)
            y2 .~ Normal.(a, 1)
        end
        r = _empty_domain_check(ast, data, q ->
            (; ll = sum(logpdf.(Normal(q.a, 1), data.y2)),
                pr = logpdf(Normal(), q.a), jac = 0.0))
        @test r.plan.n_obs == n
    end
end

@testset "empty observation predictors retain their declared priors" begin
    for n in (0, 3)
        data = (; y = fill(0.2, n), x = fill(-0.3, n))
        ast = quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 2)
            mu = a .+ b .* x
            y .~ Normal.(mu, 1)
        end
        _empty_domain_check(ast, data, q ->
            (; ll = sum(logpdf.(Normal(q.a - 0.3q.b, 1), data.y)),
                pr = logpdf(Normal(), q.a) + logpdf(Normal(0, 2), q.b), jac = 0.0))
        data = (; y = fill(0.2, n), g = collect(1:n))
        ast = quote
            a ~ Normal(0, 1)
            z[levels(g)] .~ Normal.(0, 1)
            mu = a .+ z[g]
            y .~ Normal.(mu, 1)
        end
        _empty_domain_check(ast, data, q ->
            (; ll = sum(logpdf.(Normal.(q.a .+ q.z, 1), data.y)),
                pr = logpdf(Normal(), q.a) + sum(logpdf.(Normal(), q.z)), jac = 0.0);
            dims = (n,))
    end
end

function _empty_array_fixture(k, n; prior = :(Normal.(0, 1)))
    ast = quote
        z[1:$k] .~ $prior
        a ~ Normal(0, 1)
        s = _empty_domain_scale(z)
        y .~ Normal.(a, s)
    end
    data = (; y = fill(0.2, n))
    oracle(q) = (; ll = sum(logpdf.(Normal(q.a, exp(sum(q.z))), data.y)),
        pr = logpdf(Normal(), q.a) + sum(logpdf.(Normal(), q.z)), jac = 0.0)
    return (; ast, data, oracle)
end

@testset "empty elementwise parameter domains" begin
    for k in (0, 1, 3, 9), n in (0, 3)
        f = _empty_array_fixture(k, n)
        r = _empty_domain_check(f.ast, f.data, f.oracle; dims = (k,))
        @test r.built.layout.total == k + 1
        @test length(coordinate_names(r.built.layout)) == k + 1
    end
    for (prior, distribution) in ((:(Exponential.(1)), Exponential()),
            (:(Uniform.(-1, 2)), Uniform(-1, 2)), (:(Normal.([], 1)), Normal()))
        f = _empty_array_fixture(0, 3; prior)
        oracle(q) = (; ll = sum(logpdf.(Normal(q.a, exp(sum(q.z))), f.data.y)),
            pr = logpdf(Normal(), q.a) + sum(logpdf.(distribution, q.z)), jac = 0.0)
        r = _empty_domain_check(f.ast, f.data, oracle; dims = (0,))
        @test r.built.layout.total == 1
    end
    # Every parameter may be empty: the density is a constant and its gradient
    # has no entries. No invented coordinate or prior is needed.
    data = (; y = Float64[])
    ast = quote
        z[1:0] .~ Normal.(0, 1)
        s = _empty_domain_scale(z)
        y .~ Normal.(0, s)
    end
    r = _empty_domain_check(ast, data, q -> (; ll = 0.0, pr = 0.0, jac = 0.0);
        dims = (0,))
    @test r.built.layout.total == 0
    @test isempty(r.grad)
    draws = restore_draws(r.built.layout, zeros(0, 2))
    @test size(draws.z) == (0, 2)
end

@testset "empty matrix and data-sized parameter domains" begin
    for dims in ((0, 2), (2, 0), (0, 0))
        ast = quote
            z[1:$(dims[1]), 1:$(dims[2])] .~ Normal.(0, 1)
            a ~ Normal(0, 1)
            s = _empty_domain_scale(z)
            y .~ Normal.(a, s)
        end
        data = (; y = [0.2, -0.1, 0.4])
        r = _empty_domain_check(ast, data, q ->
            (; ll = sum(logpdf.(Normal(q.a, 1), data.y)),
                pr = logpdf(Normal(), q.a), jac = 0.0); dims)
        @test r.built.layout.total == 1
    end
    for axis in (:(levels(g)), :(1:length(levels(g))),
            :(1:length(levels(g)) - 1))
        g = axis == :(1:length(levels(g)) - 1) ? [1] : Int[]
        data = (; y = fill(0.2, length(g)), g)
        ast = quote
            z[$axis] .~ Normal.(0, 1)
            a ~ Normal(0, 1)
            s = _empty_domain_scale(z)
            y .~ Normal.(a, s)
        end
        r = _empty_domain_check(ast, data, q ->
            (; ll = sum(logpdf.(Normal(q.a, 1), data.y)),
                pr = logpdf(Normal(), q.a), jac = 0.0); dims = (0,))
        @test r.built.layout.total == 1
    end
    data = (; y = Float64[], B = zeros(0, 2))
    ast = quote
        z[axes(B, 1), axes(B, 2)] .~ Normal.(0, 1)
        a ~ Normal(0, 1)
        s = _empty_domain_scale(z)
        y .~ Normal.(a, s)
    end
    r = _empty_domain_check(ast, data, q ->
        (; ll = 0.0, pr = logpdf(Normal(), q.a), jac = 0.0); dims = (0, 2))
    @test r.built.layout.total == 1
end

@testset "empty-domain malformed controls" begin
    data = (; y = [0.2])
    for axis in (:(1:-1), :(0:0), :(1:true))
        ast = quote
            z[$axis] .~ Normal.(0, 1)
            a ~ Normal(0, 1)
            y .~ Normal.(a, 1)
        end
        # refused: a declaration size is a nonnegative integer and is 1-based
        # (docs/src/constraints.md preserves the authored Julia indexing).
        @test_throws SurfaceLoweringError lower_rkppl(ast, data; conditioned = data)
    end
    for index in (0, 1, -1)
        ast = quote
            z[1:0] .~ Normal.(0, 1)
            a ~ Normal(0, 1)
            mu = a .+ z[$index]
            y .~ Normal.(mu, 1)
        end
        # refused: no position exists in an empty array (Julia bounds).
        @test_throws ContractValidationError bind_data(
            lower_rkppl(ast, data; conditioned = data), Dict{Symbol,Any}(pairs(data)))
    end
    ast = quote
        z[1:0] .~ Normal.([0.0], 1)
        a ~ Normal(0, 1)
        y .~ Normal.(a, 1)
    end
    # refused: a per-element prior vector has a different length from its array.
    @test_throws ContractValidationError bind_data(
        lower_rkppl(ast, data; conditioned = data), Dict{Symbol,Any}(pairs(data)))
    ast = quote
        z[1:length(levels(g)) - 2] .~ Normal.(0, 1)
        a ~ Normal(0, 1)
        y .~ Normal.(a, 1)
    end
    bad = Dict{Symbol,Any}(:y => [0.2], :g => [1])
    # refused: the declared count is negative on the bound data.
    @test_throws ContractValidationError bind_data(lower_rkppl(ast, bad; conditioned = (:y,)), bad)
end
