module SamplerNumericalFailureTests
using Test, ReactiveKernels, ReactiveKernelsPPL, Enzyme
using DifferentiationInterface: AutoEnzyme
import LogDensityProblems

# A point where a numerical method cannot evaluate the model, such as an ODE
# its solver cannot integrate there (`ReactiveKernels.NumericalFailure`), is
# outside the support of RKPPL's density targets: they answer -Inf, as Stan
# rejects a proposal whose ODE solve fails, and every other exception still
# propagates (snag rk-ode-solve-fai-3c664466). WarmupHMC and Pathfinder take
# -Inf as a divergent leaf or a rejected line-search point.

const BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

# Stands in for an ODE solve: the solver's typed failure above `limit`, and an
# ordinary error, which is not a numerical failure, below `-limit`.
function _nf_location(m, limit)
    m > limit && throw(ReactiveKernels.ODESolveFailure("test solver", :Unstable, m))
    m < -limit && throw(ArgumentError("not a numerical failure"))
    return 2m
end

const MODEL = @rkppl begin
    mu ~ Normal(0, 1)
    y .~ Normal.(_nf_location(mu, limit), 1.0)
end

const Y = [0.3, -0.1, 0.5, 0.2]

@testset "density targets answer -Inf for a numerical failure only" begin
    plan = MODEL(; limit = 1.5) | (; y = Y)
    built = build_kernel(plan)
    @test built.layout.total == 1
    good, failing, buggy = [0.4], [2.0], [-2.0]
    for nonallocating in (:auto, false)
        q = prepare_sampler(built, plan, good; backend = BACKEND, nonallocating)
        g = zeros(1)
        expected = sampler_value_and_gradient!(q, g, good)
        expected = (expected[1], copy(expected[2]))
        @test isfinite(expected[1]) && all(isfinite, expected[2])
        @test q(good) == expected[1]

        @test q(failing) == -Inf
        value, returned = sampler_value_and_gradient!(q, g, failing)
        @test value == -Inf
        @test returned === g && all(isnan, g)
        @test LogDensityProblems.logdensity(q, failing) == -Inf
        value, gradient = LogDensityProblems.logdensity_and_gradient(q, failing)
        @test value == -Inf && length(gradient) == 1

        # Anything else is not outside the support: it propagates.
        @test_throws ArgumentError q(buggy)
        @test_throws ArgumentError sampler_value_and_gradient!(q, g, buggy)
        @test_throws ArgumentError LogDensityProblems.logdensity_and_gradient(q, buggy)
        # The query evaluates the next point as before.
        @test sampler_value_and_gradient!(q, g, good) == expected
        @test q(good) == expected[1]
    end
    # The retained presets have no value there: that call throws the failure.
    q = prepare_sampler(built, plan, good; backend = BACKEND, retain = (:pointwise,))
    @test_throws ReactiveKernels.ODESolveFailure sampler_value_gradient_and_retained!(q, zeros(1), failing)
    @test first(sampler_value_gradient_and_retained!(q, zeros(1), good)) == q(good)
end

end
