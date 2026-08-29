using ReactiveKernels
using LogExpFunctions: log1pexp
using Test

include(joinpath(@__DIR__, "..", "examples", "bijectors.jl"))
using .BijectorKernelExample

_bijector_recipe_outputs(selected) = Set(
    output.name for recipe in selected.recipes for output in recipe.outputs
)

function _bijector_allocated(kernel::K, x::Float64) where {K}
    kernel(x)
    @allocated kernel(x)
end

@testset "bijectors as demand-planned RK kernels" begin
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
end
