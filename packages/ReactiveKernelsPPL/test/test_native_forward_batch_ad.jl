using Test, ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme

@testset "small pure summed plates use one native forward batch" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    bernoulli = @rkppl begin
        a ~ Normal(0, 2)
        b ~ Normal(0, 3)
        eta = a .+ b .* x
        y .~ BernoulliLogit.(eta)
    end
    poisson = @rkppl begin
        a ~ Normal(0, 2)
        b ~ Normal(0, 3)
        eta = a .+ b .* x
        y .~ Poisson.(exp.(eta))
    end
    normal = @rkppl begin
        a ~ Normal(0, 2)
        y .~ Normal.(a .+ x, 1)
    end
    cases = (
        (bernoulli, (; x = [-2.0, -0.5, 0.3, 1.7]),
         (; y = Bool[false, true, true, false]), true),
        (poisson, (; x = [-1.0, 0.2, 0.7, 1.5]), (; y = [0, 1, 2, 4]), true),
        (normal, (; x = [-1.0, 0.2, 0.7]),
         (; y = [-0.8, 0.3, 1.1]), true),
        (normal, (; x = Float64[]), (; y = Float64[]), false),
    )
    for (case_index, (model, data, observations, uses_forward)) in
            enumerate(cases)
        plan = model(; data...) | observations
        built = build_kernel(plan)
        u = [-0.35 + 0.15 * (i - 1) for i in 1:built.layout.total]
        fast = prepare_sampler(built, plan, u; backend)
        ordinary = prepare_sampler(built, plan, u; backend,
                                   native_forward_batch = false)
        @test (fast.ad isa PreparedADKernel) == !uses_forward
        @test ordinary.ad isa PreparedADKernel
        basis = uses_forward ? deepcopy(fast.ad.basis) : nothing
        if case_index == 1
            direct = Base.invokelatest(prepare_ad, fast.ad.reverse.kernel,
                backend, u; active = :unconstrained,
                native_forward_batch = true)
            @test !(direct isa PreparedADKernel)
            @test Base.invokelatest(ad_value_and_gradient, direct, u) ==
                Base.invokelatest(ad_value_and_gradient, fast.ad, u)
        end
        for point in (u, u .+ 0.6, u .- 1.1)
            before = copy(point)
            gfast, gordinary = similar(point), similar(point)
            value, _ = sampler_value_and_gradient!(fast, gfast, point)
            reference, _ = sampler_value_and_gradient!(ordinary, gordinary, point)
            @test value ≈ reference rtol = 1e-12
            @test gfast ≈ gordinary rtol = 1e-11 atol = 1e-12
            @test point == before
            if uses_forward
                @test fast.ad.basis == basis
            end
            @test Base.invokelatest(ad_gradient, fast.ad, point) ≈
                gfast rtol = 1e-11
            @test last(Base.invokelatest(ad_value_and_gradient,
                                        fast.ad, point)) ≈ gfast rtol = 1e-11
            for coordinate in eachindex(point)
                step = 1e-5
                plus, minus = copy(point), copy(point)
                plus[coordinate] += step
                minus[coordinate] -= step
                finite = (fast(plus) - fast(minus)) / (2step)
                @test gfast[coordinate] ≈ finite rtol = 1e-6 atol = 1e-7
            end
        end
    end
end

@testset "wide and retained samplers keep ordinary reverse" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    model = @rkppl begin
        beta[1:5] .~ Normal.(0, 1)
        y .~ Normal.(X * beta, 1)
    end
    X = reshape(sin.(1:15), 3, 5)
    plan = model(; X) | (; y = [0.4, -0.2, 0.8])
    built = build_kernel(plan)
    u = zeros(built.layout.total)
    wide = prepare_sampler(built, plan, u; backend)
    @test wide.ad isa PreparedADKernel
    gradient = similar(u)
    @test isfinite(first(sampler_value_and_gradient!(wide, gradient, u)))
    @test all(isfinite, gradient)

    narrow = @rkppl begin
        a ~ Normal(0, 1)
        y .~ Normal.(a .+ x, 1)
    end
    small = narrow(; x = [0.2, 0.6]) | (; y = [0.3, 0.8])
    built_small = build_kernel(small)
    retained = prepare_sampler(built_small, small, [0.1]; backend,
                               retain = (:pointwise,))
    @test retained.ad isa PreparedADKernel
    value, _, kept = sampler_value_gradient_and_retained!(
        retained, zeros(1), [0.1])
    @test isfinite(value)
    @test length(kept.pointwise.y) == 2
end
