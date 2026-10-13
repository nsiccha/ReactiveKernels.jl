using Test, ReactiveKernelsPPL, ReactiveKernels, DifferentiationInterface, Enzyme

@testset "native scalar intercept over a vector value" begin
    m = @rkppl begin
        alpha ~ Normal(0, 10)
        mu = alpha .+ x
        y .~ Normal.(mu, 1)
    end
    for n in (1, 7, 100)
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
            sum(-0.5*(y[i]-x[i]-0.3)^2-0.5log(2pi) for i in eachindex(y))
        @test value ≈ expected rtol=1e-12
        @test gradient[1] ≈ sum(y .- x .- 0.3)-0.3/100 rtol=1e-12
        @test !occursin("_ppl_design_mu", sprint(show, kernel_expr(plan,built.layout)))
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
