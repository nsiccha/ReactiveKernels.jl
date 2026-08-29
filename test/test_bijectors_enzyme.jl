using DifferentiationInterface
import Enzyme
using LogExpFunctions: log1pexp

const BIJECTOR_ENZYME_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

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
end
