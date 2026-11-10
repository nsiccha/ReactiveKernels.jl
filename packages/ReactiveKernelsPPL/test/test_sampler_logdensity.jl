module SamplerLogDensityTests
using Test, ReactiveKernels, ReactiveKernelsPPL, Enzyme
using DifferentiationInterface: AutoEnzyme
import LogDensityProblems

# The full-model sampler query (`prepare_sampler`) is a first-order
# LogDensityProblems target over every packed coordinate, the target WarmupHMC,
# Pathfinder and optimizers take (snag rkppl-sampler-qu-d9a47607).

const BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

# Group levels around a sampled location and scale, Normal responses.
const MODEL = @rkppl begin
    mu ~ Normal(0, 1)
    tau ~ Exponential(1)
    sigma ~ Exponential(1)
    z[levels(g)] .~ Normal.(0, 1)
    y .~ Normal.(mu .+ tau .* z[g], sigma)
end

const G = repeat(1:4, inner = 3)
const Y = [0.3, 0.1, 0.5, -0.2, 0.0, -0.4, 1.1, 0.9, 1.3, 0.2, 0.4, 0.1]

@testset "the full-model sampler query is a LogDensityProblems target" begin
    plan = MODEL(; g = G) | (; y = Y)
    built = build_kernel(plan)
    n = built.layout.total
    u = [0.1 * sin(k) for k in 1:n]
    # The default non-allocating sampler and the ordinary one.
    for nonallocating in (:auto, false)
        q = prepare_sampler(built, plan, u; backend = BACKEND, nonallocating)
        @test LogDensityProblems.dimension(q) == n == 7
        @test LogDensityProblems.capabilities(typeof(q)) ==
            LogDensityProblems.LogDensityOrder{1}()
        direct = zeros(n)
        value_direct, _ = sampler_value_and_gradient!(q, direct, u)
        value, gradient = LogDensityProblems.logdensity_and_gradient(q, u)
        @test value == value_direct
        @test gradient == direct
        @test LogDensityProblems.logdensity(q, u) == q(u)
        @test LogDensityProblems.logdensity(q, u) ≈ value rtol = 1e-12
        # Each call returns a fresh gradient: a sampler keeps the gradients it
        # is handed in its trajectories, so a later call must not overwrite them.
        value2, gradient2 = LogDensityProblems.logdensity_and_gradient(q, u .+ 0.2)
        @test gradient2 !== gradient
        @test gradient == direct
        @test gradient2 == last(sampler_value_and_gradient!(q, zeros(n), u .+ 0.2))
        @test value2 ≈ q(u .+ 0.2) rtol = 1e-12
        # Any real vector is converted, as the query's own calls convert it.
        @test LogDensityProblems.logdensity(q, Float32.(u)) == q(Float32.(u))
        @test first(LogDensityProblems.logdensity_and_gradient(q, Float32.(u))) ≈
            q(Float32.(u)) rtol = 1e-12
        # refused: the target's dimension is the layout's coordinate count
        # (LogDensityProblems' `dimension` contract), so a point of another
        # length is a `DimensionMismatch`, never a read past the point.
        @test_throws DimensionMismatch LogDensityProblems.logdensity(q, zeros(n + 1))
        @test_throws DimensionMismatch LogDensityProblems.logdensity_and_gradient(q, zeros(n - 1))
    end
end

@testset "a deep copy of the sampler query is an independent target" begin
    # `cooperative_warmup_mcmc` and `completion_warmup_mcmc` deep-copy one
    # shared target per chain.
    plan = MODEL(; g = G) | (; y = Y)
    built = build_kernel(plan)
    n = built.layout.total
    u = [0.1 * cos(k) for k in 1:n]
    q = prepare_sampler(built, plan, u; backend = BACKEND)
    copied = deepcopy(q)
    value, gradient = LogDensityProblems.logdensity_and_gradient(q, u)
    value_c, gradient_c = LogDensityProblems.logdensity_and_gradient(copied, u)
    @test value_c == value
    @test gradient_c == gradient
    # Evaluating the copy elsewhere leaves the original's answer unchanged.
    LogDensityProblems.logdensity_and_gradient(copied, u .- 0.3)
    @test LogDensityProblems.logdensity_and_gradient(q, u) == (value, gradient)
end

end
