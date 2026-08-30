using DifferentiationInterface
import Enzyme
using LogExpFunctions: log1pexp
using ReactiveKernelsKernelExamples.BijectorKernelExample

const BIJECTOR_ENZYME_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

struct FusedBijectorObjective{K}
    kernel::K
end

(objective::FusedBijectorObjective)(q) = objective.kernel(q[1], q[2])

_fused_bijector_reference(q) =
    q[1] - log1pexp(-q[2]) - log1pexp(q[2])

@testset "bijector wants support plain DI + Enzyme reverse mode" begin
    cases = (
        (positive_bijector, 0.7, :constrained, exp),
        (positive_bijector, 0.7, :log_jacobian, identity),
        (
            unit_interval_bijector,
            -0.7,
            :constrained,
            x -> inv(1 + exp(-x)),
        ),
        (
            unit_interval_bijector,
            -0.7,
            :log_jacobian,
            x -> -log1pexp(-x) - log1pexp(x),
        ),
    )
    for (spec, x, want, reference) in cases
        kernel = prepare(spec; want)
        @test kernel(x) ≈ reference(x)
        observed = DifferentiationInterface.gradient(
            kernel, BIJECTOR_ENZYME_BACKEND, x,
        )
        expected = DifferentiationInterface.gradient(
            reference, BIJECTOR_ENZYME_BACKEND, x,
        )
        @test observed ≈ expected
    end

    fused = FusedBijectorObjective(
        prepare(fused_bijector_model; want = :log_jacobian),
    )
    q = [0.7, -0.4]
    @test fused(q) ≈ _fused_bijector_reference(q)
    observed = DifferentiationInterface.gradient(
        fused, BIJECTOR_ENZYME_BACKEND, q,
    )
    expected = DifferentiationInterface.gradient(
        _fused_bijector_reference, BIJECTOR_ENZYME_BACKEND, q,
    )
    @test all(isfinite, observed)
    @test observed ≈ expected
end
