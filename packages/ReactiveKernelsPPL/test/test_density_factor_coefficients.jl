using Test, ReactiveKernels, ReactiveKernelsPPL

@testset "legacy coefficient factors retain authored owners and flat events" begin
    original = _layout_plan()
    priors = [PopulationPrior(:mu, :Intercept, :uniform, -1.0, 2.0),
        PopulationPrior(:mu, :x, :flat, 0.0, 1.0),
        PopulationPrior(:mu, :g, :uniform, 0.0, 4.0)]
    plan = StructuralPlan(original.responses, original.predictors, priors,
        original.parameters, original.assignments, original.columns, original.n_obs;
        levelmaps = original.levelmaps)
    built = build_kernel(plan)
    u = [0.2sin(i) for i in 1:built.layout.total]
    catalogue = density_factors(built, plan)
    @test length(select_factors(catalogue; kind = :prior, site = Symbol("mu.Intercept"))) == 1
    @test length(select_factors(catalogue; kind = :prior, site = Symbol("mu.g"))) == 2
    flat = select_factors(catalogue; kind = :prior, site = Symbol("mu.x"))
    @test length(flat) == 1
    @test prepare_query(built, plan, flat)(u) == 0.0
    @test isempty(factor_support(flat; u0 = u))
    @test factor_coordinates(flat) == [2]
    jac = select_factors(catalogue; kind = :log_jacobian, site = :mu)
    @test length(jac) == 3
    @test factor_support(jac; u0 = u) == [1, 3, 4]
    for (entry, coordinate) in enumerate((1, 3, 4))
        event = only(select_factors(catalogue; kind = :log_jacobian, site = :mu, entries = entry))
        @test factor_support(catalogue, event; u0 = u) == [coordinate]
        @test factor_coordinates(catalogue, event) == [coordinate]
    end
    @test length(collect(catalogue)) == length(unique(collect(catalogue)))
    @test prepare_query(built, plan, select_factors(catalogue; kind = :prior))(u) ≈
        prepare_query(built, plan, :prior)(u)
    @test prepare_query(built, plan, select_factors(catalogue; kind = :log_jacobian))(u) ≈
        prepare_query(built, plan, :log_jacobian)(u)
end
