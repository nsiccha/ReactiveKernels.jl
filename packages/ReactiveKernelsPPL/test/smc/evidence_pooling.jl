module SMCEvidencePoolingTests
# Pooling independent SMC clouds (`pool_smc`); snag tempered-smc-log-c57394d6.
# The prepared Gaussian has exact log evidence `logmass`, so the pooled
# estimate is checked against the truth.
using ReactiveKernelsPPL, LogDensityProblems, Random, Statistics, Test
include("prepared_fixture.jl")
using .PreparedSMCFixture
const F = PreparedSMCFixture
const E = Base.get_extension(ReactiveKernelsPPL, :ReactiveKernelsPPLSMCExt)

@testset "pool_smc: log-mean-exp evidence, cloud weights and pooled particles" begin
    factory = F.GaussianFactory()
    t = F.target(factory)
    clouds = [smc(t; rng=Xoshiro(s), particles=300, options=F.options()) for s in 1:6]
    before = deepcopy(F.snapshot.(clouds))
    pooled = pool_smc(clouds)
    @test F.snapshot.(clouds) == before
    logz = [c.log_normalizer for c in clouds]
    @test pooled.log_normalizer ≈ log(mean(exp.(logz))) rtol=1e-12
    @test pooled.cloud_weights ≈ exp.(logz) ./ sum(exp.(logz)) rtol=1e-12
    @test pooled.effective_clouds ≈ 1 / sum(abs2, pooled.cloud_weights) rtol=1e-12
    @test 1 <= pooled.effective_clouds <= 6
    @test size(pooled.positions) == (2, 1800) && length(pooled.weights) == 1800
    @test pooled.positions[:, 301:600] == clouds[2].positions
    @test pooled.weights[301:600] ≈ pooled.cloud_weights[2] .* clouds[2].weights
    @test sum(pooled.weights) ≈ 1
    @test pooled.log_normalizer ≈ factory.logmass atol=0.08
    @test pooled.positions * pooled.weights ≈ factory.location atol=0.06
    # Clouds that disagree: the pool rests on the higher one.
    shifted = E.SMCState(clouds[1].source, clouds[1].target, clouds[1].positions,
        clouds[1].weights, 1.0, clouds[1].log_normalizer - 10, clouds[1].rng,
        clouds[1].options, clouds[1].ancestors, clouds[1].lineage, clouds[1].history,
        clouds[1].update, clouds[1].stage)
    skewed = pool_smc([shifted, clouds[2]])
    @test skewed.log_normalizer ≈ clouds[2].log_normalizer + log((1 + exp(-10 +
        clouds[1].log_normalizer - clouds[2].log_normalizer)) / 2) rtol=1e-12
    @test skewed.effective_clouds < 1.001

    # At temperature 1 different references estimate the same evidence.
    wide = smc(t; rng=Xoshiro(8), particles=300, options=F.options(),
        reference=SMCReference(zeros(2), fill(2.0, 2)))
    @test pool_smc([clouds[1], wide]).effective_clouds > 1
    @test_throws ArgumentError pool_smc(eltype(clouds)[])
    partial = smc(t; rng=Xoshiro(9), particles=300, options=F.options(), stages=1)
    @test_throws ArgumentError pool_smc([clouds[1], partial])
    other = smc(F.target(F.GaussianFactory(location=[0.1, 0.2]); binding_id="data-v2");
        rng=Xoshiro(10), particles=300, options=F.options())
    @test_throws ArgumentError pool_smc([clouds[1], other])
    partial_wide = smc(t; rng=Xoshiro(9), particles=300, options=F.options(), stages=1,
        reference=SMCReference(zeros(2), fill(2.0, 2)))
    mixed = E.SMCState(partial_wide.source, partial.target, partial.positions,
        partial.weights, partial.temperature, partial.log_normalizer, partial.rng,
        partial.options, partial.ancestors, partial.lineage, partial.history,
        partial.update, partial.stage)
    @test_throws ArgumentError pool_smc([partial, mixed])
    @test pool_smc([partial, partial]).log_normalizer ≈ partial.log_normalizer rtol=1e-12
end

end
