using Test, ReactiveKernelsPPL, ReactiveKernels, DifferentiationInterface, Enzyme

@testset "Bernoulli logit retains representable whole-vector tails" begin
    model = @rkppl begin
        beta[1:1] .~ Flat.()
        eta = beta[1] .* x
        y .~ BernoulliLogit.(eta)
    end
    for observed in (0, 1)
        plan = model(; x=[1]) | (; y=[observed])
        built = build_kernel(plan)
        for forward_batch in (false, true)
            q = prepare_sampler(built, plan, [0.0];
                backend=AutoEnzyme(; mode=Enzyme.Reverse),
                native_forward_batch=forward_batch)
            gradient = zeros(1)
            for eta in (-1000.0, -100.0, -50.0, -40.0, -20.0, 0.0,
                        20.0, 40.0, 50.0, 100.0, 1000.0)
                expected, derivative = setprecision(BigFloat, 256) do
                    z = BigFloat(eta)
                    observed == 1 ? (-log1p(exp(-z)), inv(1 + exp(z))) :
                        (-log1p(exp(z)), -inv(1 + exp(-z)))
                end
                value, _ = sampler_value_and_gradient!(q, gradient, [eta])
                @test value ≈ Float64(expected) rtol=2e-13 atol=0.0
                @test gradient[1] ≈ Float64(derivative) rtol=2e-13 atol=0.0
            end
        end
    end
end

@testset "integer arithmetic offsets retain rows, broadcasting and rebinding" begin
    model = @rkppl begin
        beta[1:3] .~ Flat.()
        eta = beta[1] .+ beta[2] .* x .+ beta[3] .* z
        y .~ BernoulliLogit.(eta)
    end
    u = [0.2, -0.35, 0.1]
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    cases = [(Int64, n, singleton) for n in (0, 1, 7, 19) for singleton in (false, true)]
    append!(cases, [(T, 7, false) for T in (Int8, Int32, UInt8)])
    for (T, n, singleton) in cases
        x = singleton ? T[2] : T[mod(i, 4) for i in 1:n]
        z = T[mod(i, 3) for i in 1:n]
        y = Int[mod(i, 2) for i in 1:n]
        saved = deepcopy((x, z, y, u))
        plan = model(; x, z) | (; y)
        built = build_kernel(plan)
        eta = u[1] .+ u[2] .* x .+ u[3] .* z
        expected = [yi == 1 ? -log1p(exp(-e)) : -log1p(exp(e)) for (yi,e) in zip(y,eta)]
        slope = [yi == 1 ? inv(1 + exp(e)) : -inv(1 + exp(-e)) for (yi,e) in zip(y,eta)]
        expected_gradient = [sum(slope), sum(slope .* x), sum(slope .* z)]
        for forward_batch in (false, true)
            q = prepare_sampler(built, plan, u; backend,
                native_forward_batch=forward_batch)
            gradient = similar(u)
            value, _ = sampler_value_and_gradient!(q, gradient, u)
            @test value ≈ sum(expected) rtol=1e-12 atol=1e-12
            @test gradient ≈ expected_gradient rtol=1e-12 atol=1e-12
        end
        pointwise = prepare_query(built, plan, :pointwise)(u)
        @test pointwise.y ≈ expected rtol=1e-12 atol=1e-12
        @test (x, z, y, u) == saved
    end

    original = model(; x=[1,2,3], z=[2,1,0]) | (; y=[0,1,0])
    rebound = model(; x=[2,0,1,3,2], z=[1,1,2,0,3]) | (; y=[1,0,1,0,1])
    built = build_kernel(original)
    query = prepare_query(built, rebound, :sampler)
    fresh = prepare_query(build_kernel(rebound), rebound, :sampler)
    @test query(u) ≈ fresh(u) rtol=1e-12 atol=1e-12
end
