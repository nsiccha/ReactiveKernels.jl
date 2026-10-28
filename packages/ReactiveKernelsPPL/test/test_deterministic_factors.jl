using Test, ReactiveKernelsPPL, ReactiveKernels, DifferentiationInterface, Enzyme

module DeterministicFactorPreprocessing
const calls = Ref(0)
counted_log(raw) = (calls[] += 1; log.(raw))
end

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

@testset "factor preprocessing freezes in RK independently of conditioning" begin
    normal(x, mu) = -0.5*(x-mu)^2 - 0.5log(2pi)
    for definition in (
            [:(z = sum(counted_log(raw)))],
            [:(logged = counted_log(raw)), :(z = sum(logged))]),
            observing in (false, true)
        data = (; raw=exp.([-0.2, 0.5, 0.1, -0.3, 0.4]), y=[0.1, -0.4])
        ast = Expr(:block, :(mu ~ Normal(0, 1)), definition...,
            :(z ~ Normal(mu, 1)), :(y .~ Normal.(mu, 1)))
        DeterministicFactorPreprocessing.calls[] = 0
        plan = bind_data(lower_rkppl(ast, data;
            mod=DeterministicFactorPreprocessing,
            conditioned=observing ? (:z, :y) : (:y,)), data)
        built = build_kernel(plan)
        @test plan.n_obs == length(data.y)
        @test DeterministicFactorPreprocessing.calls[] == 0
        @test !haskey(plan.columns, :z)
        @test occursin("counted_log", string(kernel_expr(plan, built.layout)))
        q = prepare_sampler(built, plan, [0.25];
            backend=AutoEnzyme(; mode=Enzyme.Reverse))
        @test DeterministicFactorPreprocessing.calls[] == 1
        @test !occursin("counted_log", string(code_expr(q.kernel)))
        for mu in (0.25, -0.7)
            g = zeros(1)
            z = sum(log.(data.raw))
            lp, _ = sampler_value_and_gradient!(q, g, [mu])
            @test lp ≈ normal(mu, 0) + normal(z, mu) + sum(normal.(data.y, mu))
            @test g ≈ [-mu + z-mu + sum(data.y .- mu)]
        end
        @test DeterministicFactorPreprocessing.calls[] == 1
    end

    # A shared structural consumer still needs the helper during binding.
    # Preparation uses that same value rather than calling the helper twice.
    data = (; raw=exp.([-0.2, 0.5, 0.1]), y=[0.1, -0.4])
    ast = :(begin
        mu ~ Normal(0, 1)
        logged = counted_log(raw)
        beta[axes(logged, 1)] .~ Normal.(0, 1)
        z = sum(logged)
        z ~ Normal(mu, 1)
        y .~ Normal.(mu, 1)
    end)
    DeterministicFactorPreprocessing.calls[] = 0
    plan = bind_data(lower_rkppl(ast, data; mod=DeterministicFactorPreprocessing,
        conditioned=(:z, :y)), data)
    @test DeterministicFactorPreprocessing.calls[] == 1
    @test plan.columns[:logged] ≈ log.(data.raw)
    built = build_kernel(plan)
    u = unconstrain(built.layout, (; mu=0.25, beta=zeros(3)))
    q = prepare_sampler(built, plan, u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
    @test q(u) ≈ normal(0.25, 0) + 3normal(0, 0) +
        normal(sum(log.(data.raw)), 0.25) + sum(normal.(data.y, 0.25))
    @test DeterministicFactorPreprocessing.calls[] == 1
end

@testset "explicit conditioning selects a definition's density role" begin
    normal(x, mu, sigma) = -0.5*((x-mu)/sigma)^2 - log(sigma) - 0.5log(2pi)
    definitions = (
        [:(z = sum(log.(raw)))],
        [:(logged = log.(raw)), :(z = sum(logged))],
    )
    for n in (0, 1, 7), definition in definitions
        raw = exp.(collect(range(-0.2, 0.4; length=max(n, 2)))[1:n])
        data = (; raw)
        saved = deepcopy(data)
        z = sum(log.(raw))
        ast = Expr(:block, :(mu ~ Normal(0, 1)), definition..., :(z ~ Normal(mu, 3)))
        unbound = lower_rkppl(ast, data)
        free = bind_data(unbound, data)
        observed = bind_data(unbound, data; conditioned=(:z,))
        twin = bind_data(lower_rkppl(:(begin
            mu ~ Normal(0, 1); z ~ Normal(mu, 3)
        end), (; z); conditioned=(:z,)), (; z))
        @test isempty(unbound.conditioned)
        for (bound, observing) in ((free, false), (observed, true), (twin, true))
            built = build_kernel(bound)
            @test coordinate_names(built.layout) == [:mu]
            prior = prepare_query(built, bound, :prior)
            likelihood = prepare_query(built, bound, :likelihood)
            pointwise = prepare_query(built, bound, :pointwise)
            jacobian = prepare_query(built, bound, :log_jacobian)
            q = prepare_sampler(built, bound, [0.25];
                backend=AutoEnzyme(; mode=Enzyme.Reverse))
            for mu in (0.25, -0.7)
                u, g = [mu], zeros(1)
                lp, _ = sampler_value_and_gradient!(q, g, u)
                lp0, lpz = normal(mu, 0, 1), normal(z, mu, 3)
                @test lp ≈ lp0 + lpz
                @test g ≈ [-mu + (z-mu)/9]
                @test Base.invokelatest(prior, u) ≈ lp0 + (observing ? 0 : lpz)
                @test Base.invokelatest(likelihood, u) ≈ (observing ? lpz : 0)
                points = Base.invokelatest(pointwise, u)
                @test keys(points) == (observing ? (:z,) : ())
                observing && @test points.z ≈ lpz
                @test Base.invokelatest(jacobian, u) == 0
            end
            if bound !== twin
                # The authored preprocessing is in the graph in either role;
                # preparation removes it from repeated density/Reverse work.
                @test !haskey(bound.columns, :z)
                emitted = string(Base.remove_linenums!(deepcopy(kernel_expr(bound, built.layout))))
                prepared = string(Base.remove_linenums!(deepcopy(code_expr(q.kernel))))
                @test occursin("sum", emitted)
                @test occursin("raw", emitted)
                @test !occursin("sum", prepared)
                @test !occursin("raw", prepared)
            end
        end
        @test data == saved
    end

    # Names-only, values, captured model and a forward definition select the
    # same role. No second input is needed for the value computed in-graph.
    raw = [0.8, 1.3, 2.1]
    ast = :(begin z ~ Normal(mu, 3); mu ~ Normal(0, 1); z = sum(log.(raw)) end)
    model = ReactiveKernelsPPL.RKPPLModel(ast, @__MODULE__)
    for source in (ast, model)
        inputs = source === ast ? ((:raw,), (; raw)) : ((; raw),)
        for data in inputs
            plan = bind_data(lower_rkppl(source, data; conditioned=(:z,)), (; raw))
            built = build_kernel(plan)
            @test Base.invokelatest(prepare_query(built, plan, :likelihood), [0.25]) ≈
                normal(sum(log.(raw)), 0.25, 3)
        end
    end

    # A different family and preprocessing graph, with a live scale, use the
    # same factor emission and ordinary Reverse rule set.
    for n in (2, 9)
        raw = collect(range(0.3, 1.7; length=n))
        ast = :(begin
            mu ~ Normal(0, 1)
            center = sum(raw) / length(raw)
            deviations = raw .- center
            variance = sum(deviations .* deviations) / length(raw)
            z = (raw[1] - center)^2 / variance
            z ~ Exponential(exp(mu))
        end)
        plan = bind_data(lower_rkppl(ast, (; raw); conditioned=(:z,)), (; raw))
        built = build_kernel(plan)
        mu = -0.3
        u, g = [mu], zeros(1)
        z = (raw[1] - sum(raw)/n)^2 / (sum(abs2, raw .- sum(raw)/n)/n)
        q = prepare_sampler(built, plan, u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
        value, _ = sampler_value_and_gradient!(q, g, u)
        @test value ≈ normal(mu, 0, 1) - mu - z*exp(-mu)
        @test g ≈ [-mu - 1 + z*exp(-mu)]
        @test Base.invokelatest(prepare_query(built, plan, :pointwise), u).z ≈
            -mu - z*exp(-mu)
        @test coordinate_names(built.layout) == [:mu]
    end

    # Choosing the likelihood role preserves live dependencies too; it does
    # not freeze a definition that still reads a free parameter.
    ast = :(begin mu ~ Normal(0, 1); z = 2 * mu; z ~ Normal(0, 1) end)
    plan = bind_data(lower_rkppl(ast, (); conditioned=(:z,)), (;))
    built = build_kernel(plan)
    q = prepare_sampler(built, plan, [0.3]; backend=AutoEnzyme(; mode=Enzyme.Reverse))
    g = zeros(1)
    value, _ = sampler_value_and_gradient!(q, g, [0.3])
    @test value ≈ normal(0.3, 0, 1) + normal(0.6, 0, 1)
    @test g ≈ [-1.5]
    @test Base.invokelatest(prepare_query(built, plan, :likelihood), [0.3]) ≈ normal(0.6, 0, 1)
end
