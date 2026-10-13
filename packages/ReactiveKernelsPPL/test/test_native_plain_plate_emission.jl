using Test, ReactiveKernelsPPL, ReactiveKernels, DifferentiationInterface, Enzyme

@testset "bound in-cell observations keep embedded operation tables aligned" begin
    # The docs corpus prepares this plate with its observation data bound.
    # Its shared Normal support branch must reserve the same operation slots
    # in the native and tensorized companions.
    model = @rkppl begin
        sigma ~ Exponential(1.0)
        b0 ~ Normal(0.0, 1.0)
        @plate for i in eachindex(obs)
            mu = (b0 * dose[i]) * t[i]
            obs[i] ~ Normal(mu, sigma)
        end
    end
    t = [0.5, 1.0, 2.0, 0.5, 1.0, 2.0]
    dose = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    obs = [0.5, 1.2, 2.1, 0.8, 1.5, 2.5]
    plan = model(; t, dose) | (; obs)
    built = build_kernel(plan)
    u = collect(range(-0.5, 0.5; length = built.layout.total))
    kernel = prepare_query(built, plan, :sampler)
    @test isfinite(kernel(u))
    with_evidence = prepare_query(built, plan, :sampler; exemplars = (u,))
    @test with_evidence(u) ≈ kernel(u)
end

@testset "native scalar intercept over a vector value" begin
    m = @rkppl begin
        alpha ~ Normal(0, 10)
        mu = alpha .+ x
        y .~ Normal.(mu, 1)
    end
    for n in (0, 1, 7, 100)
        x = cos.(1:n)
        y = sin.(1:n)
        plan = m(; x) | (; y)
        built = build_kernel(plan)
        position = [0.3]
        query = prepare_sampler(built, plan, position;
            backend=AutoEnzyme(;mode=Enzyme.Reverse))
        gradient = similar(position)
        value, _ = sampler_value_and_gradient!(query, gradient, position)
        expected = -0.5*(0.3/10)^2-log(10)-0.5log(2pi) +
            sum((-0.5*(y[i]-x[i]-0.3)^2-0.5log(2pi) for i in eachindex(y));
                init=0.0)
        @test value ≈ expected rtol=1e-12
        @test gradient[1] ≈ sum(y .- x .- 0.3)-0.3/100 rtol=1e-12
        @test !occursin("_ppl_design_mu", sprint(show, kernel_expr(plan,built.layout)))
    end
end

@testset "native vector predictor composes with its likelihood plate" begin
    m = @rkppl begin
        alpha ~ Normal(0, 10)
        beta[axes(X, 2)] .~ Normal.(0, 10)
        mu = alpha .+ X * beta
        y .~ Normal.(mu, 1)
    end
    for n in (0, 1, 7, 100)
        X = hcat(sin.(1:n), cos.(1:n))
        y = cos.(1:n) .- sin.(1:n)
        plan = m(; X) | (; y)
        built = build_kernel(plan)
        alpha, beta = 0.3, [0.1, -0.2]
        position = unconstrain(built.layout, (; alpha, beta))
        query = prepare_sampler(built, plan, position;
            backend=AutoEnzyme(;mode=Enzyme.Reverse))
        gradient = similar(position)
        value, _ = sampler_value_and_gradient!(query, gradient, position)
        residual = y .- alpha .- X * beta
        expected = -0.5*(alpha/10)^2-log(10)-0.5log(2pi) +
            sum(-0.5*(b/10)^2-log(10)-0.5log(2pi) for b in beta) +
            sum((-0.5*r^2-0.5log(2pi) for r in residual); init=0.0)
        @test value ≈ expected rtol=1e-12
        expected_gradient = [sum(residual)-alpha/100;
            X' * residual .- beta ./ 100]
        @test gradient ≈ expected_gradient rtol=1e-12
    end
end

@testset "ordinary vector prior keeps scale-before-square behavior" begin
    m = @rkppl begin
        beta[1:2] .~ Normal.(0, 1e150)
    end
    plan = bind_data(m())
    built = build_kernel(plan)
    position = [1e160, 0.0]
    query = prepare_sampler(built, plan, position;
        backend=AutoEnzyme(;mode=Enzyme.Reverse))
    gradient = similar(position)
    value, _ = sampler_value_and_gradient!(query,gradient,position)
    expected = sum(-0.5*(v/1e150)^2-log(1e150)-0.5log(2pi) for v in position)
    @test isfinite(value)
    @test value ≈ expected rtol=1e-12
    @test gradient ≈ [-1e-140, 0.0] rtol=1e-10
end
