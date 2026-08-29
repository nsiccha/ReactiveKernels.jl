using ReactiveKernels
using LogExpFunctions: log1pexp
using Test

include(joinpath(@__DIR__, "..", "examples", "bijectors.jl"))
using .BijectorKernelExample

_bijector_recipe_outputs(selected) = Set(
    output.name for recipe in selected.recipes for output in recipe.outputs
)

function _bijector_allocated(kernel::K, args::Vararg{Float64,N}) where {K,N}
    kernel(args...)
    @allocated kernel(args...)
end

@testset "bijectors as demand-planned RK kernels" begin
    @testset "dedicated executable docs page" begin
        root = joinpath(@__DIR__, "..")
        page = read(joinpath(root, "docs", "src", "bijectors.md"), String)
        make = read(joinpath(root, "docs", "make.jl"), String)
        checks = read(joinpath(root, "docs", "check_rendered.jl"), String)
        index = read(joinpath(root, "docs", "src", "index.md"), String)

        @test occursin(
            "\"Bijectors and constrained parameters\" => \"bijectors.md\"",
            make,
        )
        @test occursin("BIJECTOR_DOCS_SOURCE", page)
        @test occursin("execute_example", page)
        @test occursin("\"bijectors.md\" => 1", checks)
        @test occursin("[Bijectors and constrained parameters](bijectors.md)", index)
        @test occursin("@kernel positive_bijector", BIJECTOR_KERNEL_SOURCE)
        @test occursin("@kernel unit_interval_bijector", BIJECTOR_KERNEL_SOURCE)
        @test occursin("@kernel fused_bijector_model", BIJECTOR_KERNEL_SOURCE)
    end

    @testset "positive support" begin
        constrained_plan = plan(positive_bijector; want = :constrained)
        jacobian_plan = plan(positive_bijector; want = :log_jacobian)
        joint_plan = plan(
            positive_bijector; want = (:constrained, :log_jacobian),
        )

        @test _bijector_recipe_outputs(constrained_plan) == Set((:constrained,))
        @test isempty(jacobian_plan.recipes)
        @test _bijector_recipe_outputs(joint_plan) == Set((:constrained,))

        x = 0.7
        constrained = prepare(constrained_plan)
        log_jacobian = prepare(jacobian_plan)
        joint = prepare(joint_plan)
        @test @inferred(constrained(x)) == exp(x)
        @test @inferred(log_jacobian(x)) == x
        @test @inferred(joint(x)) == (exp(x), x)
        @test _bijector_allocated(constrained, x) == 0
        @test _bijector_allocated(log_jacobian, x) == 0
        @test _bijector_allocated(joint, x) == 0
    end

    @testset "stable unit interval" begin
        constrained_plan = plan(unit_interval_bijector; want = :constrained)
        jacobian_plan = plan(unit_interval_bijector; want = :log_jacobian)
        joint_plan = plan(
            unit_interval_bijector; want = (:constrained, :log_jacobian),
        )

        constrained_outputs = Set((:magnitude, :tail, :constrained))
        jacobian_outputs = Set((
            :magnitude,
            :tail,
            :log_normalizer,
            :log_constrained,
            :log_complement,
            :log_jacobian,
        ))
        @test _bijector_recipe_outputs(constrained_plan) == constrained_outputs
        @test _bijector_recipe_outputs(jacobian_plan) == jacobian_outputs
        @test _bijector_recipe_outputs(joint_plan) ==
              union(constrained_outputs, jacobian_outputs)

        constrained = prepare(constrained_plan)
        log_jacobian = prepare(jacobian_plan)
        joint = prepare(joint_plan)
        for x in (-1000.0, -0.7, 0.0, 0.7, 1000.0)
            expected_constrained = x >= 0 ?
                inv(1 + exp(-x)) : exp(x) / (1 + exp(x))
            expected_log_jacobian = -log1pexp(-x) - log1pexp(x)
            @test @inferred(constrained(x)) == expected_constrained
            @test @inferred(log_jacobian(x)) == expected_log_jacobian
            @test @inferred(joint(x)) ==
                  (expected_constrained, expected_log_jacobian)
            @test isfinite(log_jacobian(x))
        end

        x = 0.7
        @test _bijector_allocated(constrained, x) == 0
        @test _bijector_allocated(log_jacobian, x) == 0
        @test _bijector_allocated(joint, x) == 0
    end

    @testset "nested transforms form one prunable graph" begin
        parameters_plan = plan(fused_bijector_model; want = :parameters)
        jacobian_plan = plan(fused_bijector_model; want = :log_jacobian)
        joint_plan = plan(
            fused_bijector_model; want = (:parameters, :log_jacobian),
        )

        @test length(parameters_plan.recipes) == 5
        @test length(jacobian_plan.recipes) == 7
        @test length(joint_plan.recipes) == 10
        for selected in (parameters_plan, jacobian_plan, joint_plan)
            emitted = sprint(show, code_expr(selected))
            @test !occursin("positive_bijector", emitted)
            @test !occursin("unit_interval_bijector", emitted)
        end

        parameters = prepare(parameters_plan)
        log_jacobian = prepare(jacobian_plan)
        joint = prepare(joint_plan)
        inputs = (0.7, -0.4)
        expected_parameters = (
            scale = exp(inputs[1]),
            probability = inv(1 + exp(-inputs[2])),
        )
        expected_log_jacobian = inputs[1] - log1pexp(-inputs[2]) -
                                log1pexp(inputs[2])
        @test @inferred(parameters(inputs...)) == expected_parameters
        @test @inferred(log_jacobian(inputs...)) ≈ expected_log_jacobian
        observed_joint = @inferred joint(inputs...)
        @test observed_joint[1] == expected_parameters
        @test observed_joint[2] ≈ expected_log_jacobian
        @test _bijector_allocated(parameters, inputs...) == 0
        @test _bijector_allocated(log_jacobian, inputs...) == 0
        @test _bijector_allocated(joint, inputs...) == 0
    end
end
