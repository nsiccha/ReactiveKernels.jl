using DifferentiationInterface, Distributions, Enzyme, ReactiveKernels, ReactiveKernelsPPL, Test

function _latent_axis_fixture(n; iterator=:eachindex, matrix=false, legacy=false)
    domain = matrix ? reshape(collect(1:2n), n, 2) : collect(1:n)
    extent = iterator === :eachindex ? length(domain) : size(domain, 2)
    data = (; observed=[3.2, 5.1], domain, y=[1.1, 1.3, 0.9])
    range = iterator === :eachindex ? :(eachindex(domain)) : :(axes(domain, 2))
    ast = quote
        a ~ Normal(0, 1)
        @plate for j in $range
            missing_value[j] ~ LogNormal(1.4, 0.4)
        end
        observed .~ LogNormal.(1.4, 0.4)
        y .~ Normal.(a, 0.8)
    end
    plan = lower_rkppl(ast, data; conditioned=(:observed, :y))
    if legacy
        plan = ReactiveKernelsPPL._with(plan; plate_parameters=[
            ReactiveKernelsPPL._with(p; range=:domain) for p in plan.plate_parameters])
    end
    bound = bind_data(plan, data)
    built = build_kernel(bound)
    u = unconstrain(built.layout, (; a=0.2, missing_value=fill(4.0, extent)))
    oracle = v -> begin
        p = constrain(built.layout, v)
        logpdf(Normal(), p.a) +
            sum(logpdf.(LogNormal(1.4, 0.4), p.missing_value); init=0.0) +
            sum(log, p.missing_value; init=0.0) +
            sum(logpdf.(LogNormal(1.4, 0.4), data.observed)) +
            sum(logpdf.(Normal(p.a, 0.8), data.y))
    end
    sampler = prepare_sampler(built, bound, u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
    return (; data, bound, built, u, sampler, oracle, extent)
end

@testset "latent plates retain an independent authored data axis" begin
    for (n, matrix, iterator, legacy) in (
        (0, false, :eachindex, false), (1, false, :eachindex, false),
        (4, false, :eachindex, false), (9, false, :eachindex, false),
        (3, true, :eachindex, false), (5, true, :axes, false),
        (3, false, :eachindex, true))
        fx = _latent_axis_fixture(n; matrix, iterator, legacy)
        @test length(constrain(fx.built.layout, fx.u).missing_value) == fx.extent
        @test fx.bound.n_obs == 5
        value, gradient = sampler_value_and_gradient!(fx.sampler, similar(fx.u), fx.u)
        @test value ≈ fx.oracle(fx.u)
        h = cbrt(eps(Float64))
        reference = map(eachindex(fx.u)) do i
            hi, lo = copy(fx.u), copy(fx.u)
            hi[i] += h; lo[i] -= h
            (fx.oracle(hi)-fx.oracle(lo))/(2h)
        end
        @test gradient ≈ reference rtol=5e-6 atol=1e-7
        pointwise = Base.invokelatest(prepare_query(fx.built, fx.bound, :pointwise), fx.u)
        @test length(pointwise.observed) == 2
        @test length(pointwise.y) == 3
        @test fx.data.domain == (matrix ? reshape(collect(1:2n), n, 2) : collect(1:n))
    end
end

@testset "completed covariates agree with the declared-array spelling" begin
    data = (; age_obs=[3.2, 5.1], Jmis_age=[2], age_order=[1, 3, 2],
        y=[1.1, 1.3, 0.9])
    tail = quote
        age = vcat(age_obs, age_mis)[age_order]
        age_obs .~ LogNormal.(1.4, 0.4)
        y .~ Normal.(age, 0.8)
    end
    plate = quote
        @plate for j in eachindex(Jmis_age)
            age_mis[j] ~ LogNormal(1.4, 0.4)
        end
    end
    array = quote age_mis[axes(Jmis_age, 1)] .~ LogNormal.(1.4, 0.4) end
    for head in (plate, array)
        ast = Expr(:block, head.args..., tail.args...)
        bound = bind_data(lower_rkppl(ast, data; conditioned=(:age_obs, :y)), data)
        built = build_kernel(bound)
        @test coordinate_names(built.layout) == [Symbol("age_mis.1")]
        @test bound.n_obs == 5
        @test Base.invokelatest(prepare_query(built, bound, :sampler), [0.0]) ≈
            -28.64912208654744
    end
end

@testset "a plate range source can still carry an observation axis" begin
    ast = quote
        @plate for j in eachindex(x)
            theta[j] ~ Normal(0, 1)
        end
        y1 .~ Normal.(x, 1)
        y2 .~ Normal.(0, 1)
    end
    data = (; x=[0.2, 0.4, 0.6], y1=[0.1, 0.3, 0.5], y2=[0.7, 0.9])
    bound = bind_data(lower_rkppl(ast, data; conditioned=(:y1, :y2)), data)
    built = build_kernel(bound)
    @test built.layout.total == 3
    @test bound.n_obs == 5
    bad = merge(data, (; y1=[0.1, 0.3, 0.5, 0.8]))
    # refused: x is read as three observation values beside four y1 values;
    # the independent latent range cannot waive Julia dimension compatibility
    # (standing @rkppl language principle 3).
    @test_throws "column length 3 ≠ the 4 rows of y1" bind_data(
        lower_rkppl(ast, bad; conditioned=(:y1, :y2)), bad)
end
