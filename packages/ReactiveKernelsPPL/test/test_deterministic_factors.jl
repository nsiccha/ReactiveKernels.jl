using Test, ReactiveKernelsPPL, ReactiveKernels, DifferentiationInterface, Enzyme

@testset "a deterministic scalar may carry a density factor" begin
    model = @rkppl begin
        intercept ~ Normal(0, 1)
        beta_free[1:3] .~ Normal.(0, 3)
        beta_last = -sum(beta_free)
        beta_last ~ Normal(0, 3)
        beta = vcat(beta_free, beta_last)
        eta = intercept .- beta[idx]
        y .~ BernoulliLogit.(eta)
    end
    lognormal(x, sigma) = -0.5(x/sigma)^2 - log(sigma) - 0.5log(2pi)
    logbernoulli(y, eta) = y == 1 ? -log1p(exp(-eta)) : -log1p(exp(eta))
    for idx in (Int[], [4], [2, 4, 1, 4, 3, 2])
        y = Int[isodd(i) for i in eachindex(idx)]
        saved = deepcopy((idx, y))
        plan = model(; idx) | (; y)
        built = build_kernel(plan)
        @test length(plan.factors) == 1
        @test only(plan.factors).name === :beta_last
        @test !("beta_last" in string.(coordinate_names(built.layout)))
        source = sprint(show, Base.remove_linenums!(deepcopy(kernel_expr(plan, built.layout))))
        @test occursin("_ppl_factor_beta_last", source)
        isempty(idx) || @test occursin("ReactiveKernels._readonly_gather(beta, idx)", source)
        intercept, beta_free = 0.3, [0.2, -0.4, 0.1]
        u = collect(Float64, unconstrain(built.layout, (; intercept, beta_free)))
        beta_last = -sum(beta_free)
        eta = intercept .- vcat(beta_free, beta_last)[idx]
        expected_prior = lognormal(intercept, 1) +
            sum(x -> lognormal(x, 3), beta_free; init=0.0) +
            lognormal(beta_last, 3)
        expected = expected_prior + sum(logbernoulli.(y, eta); init=0.0)
        q = prepare_sampler(built, plan, u;
            backend=AutoEnzyme(; mode=Enzyme.Reverse))
        gradient = similar(u)
        value, _ = sampler_value_and_gradient!(q, gradient, u)
        @test value ≈ expected rtol=1e-12
        @test prepare_query(built, plan, :prior)(u) ≈ expected_prior rtol=1e-12
        for k in eachindex(u)
            up, down = copy(u), copy(u)
            up[k] += 1e-6
            down[k] -= 1e-6
            @test gradient[k] ≈ (q(up) - q(down))/2e-6 atol=2e-6
        end
        @test (idx, y) == saved
    end

    double_factor = @rkppl begin
        a ~ Normal(0, 1)
        last = -a
        last ~ Normal(0, 1)
        last ~ Normal(0, 2)
        y .~ Normal.(a, 1)
    end
    @test_throws SurfaceLoweringError double_factor() | (; y=[0.2])

    forward_factor = @rkppl begin
        last ~ Normal(0, 1)
        a ~ Normal(0, 1)
        last = -a
        y .~ Normal.(a, 1)
    end
    @test only((forward_factor() | (; y=[0.2])).factors).name === :last

    vector_factor = @rkppl begin
        a ~ Normal(0, 1)
        values = a .+ x
        values ~ Normal(0, 1)
        y .~ Normal.(values, 1)
    end
    @test_throws SurfaceLoweringError vector_factor(; x=[0.1]) | (; y=[0.2])

    computed_arg = @rkppl begin
        a ~ Normal(0, 1)
        last = -a
        last ~ Normal(0, 2 * scale)
        y .~ Normal.(a, 1)
    end
    @test (computed_arg(; scale=1.5) | (; y=[0.2])).factors[1].args.arg2 isa Symbol
end
