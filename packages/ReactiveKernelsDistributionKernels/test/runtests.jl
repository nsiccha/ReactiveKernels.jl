using Distributions: Cauchy, Laplace, Normal, cdf, logpdf, quantile
using ReactiveKernels: KernelObjectSpec, KernelSpec, extract, plan, prepare
using ReactiveKernelsDistributionKernels: DistributionKernelSources
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    LOCATION_SCALE_SOURCE,
    standard_normal, standard_cauchy, standard_laplace, location_scale,
    normal, cauchy, laplace,
    NORMAL_LOGDENSITY, CAUCHY_LOGDENSITY, LAPLACE_LOGDENSITY
using Test

@testset "distribution kernel foundation" begin
    @test all(object -> object isa KernelObjectSpec, (normal, cauchy, laplace))
    @test all(template -> !isnothing(template),
        (standard_normal, standard_cauchy, standard_laplace, location_scale))
    @test all(spec -> spec isa KernelSpec,
        (NORMAL_LOGDENSITY, CAUCHY_LOGDENSITY, LAPLACE_LOGDENSITY))
    @test !isdefined(DistributionKernelSources, :Distributions)
    @test occursin("@kernel location_scale", LOCATION_SCALE_SOURCE)
    @test occursin("@kernel normal = location_scale(standard_normal)",
        LOCATION_SCALE_SOURCE)

    x, location, scale = 0.4, -0.2, 1.3
    p = 0.73
    for (object, reference) in (
            (normal, Normal(location, scale)),
            (cauchy, Cauchy(location, scale)),
            (laplace, Laplace(location, scale)))
        @test prepare(object.logpdf)(location, scale, x) ≈ logpdf(reference, x)
        @test prepare(object.cdf)(location, scale, x) ≈ cdf(reference, x)
        @test prepare(object.quantile)(location, scale, p) ≈ quantile(reference, p)
    end

    @testset "transparent cuts and shared work" begin
        outputs_of(p) = [only(recipe.outputs).name for recipe in p.recipes]
        scale_plan = plan(normal.logpdf;
            have = (:x, :location, :scale), want = :logpdf)
        logscale_plan = plan(normal.logpdf;
            have = (:x, :location, :log_scale), want = :logpdf)
        both_plan = plan(normal.logpdf;
            have = (:x, :location, :scale, :log_scale), want = :logpdf)

        @test :log_scale in outputs_of(scale_plan)
        @test !(:scale in outputs_of(scale_plan))
        @test outputs_of(scale_plan) ==
              [:log_scale, :standardized, Symbol("standard.logpdf"), :logpdf]
        @test :scale in outputs_of(logscale_plan)
        @test !(:log_scale in outputs_of(logscale_plan))
        @test !(:scale in outputs_of(both_plan))
        @test !(:log_scale in outputs_of(both_plan))

        expected = logpdf(Normal(location, scale), x)
        @test prepare(scale_plan)(x, location, scale) ≈ expected
        @test prepare(logscale_plan)(x, location, log(scale)) ≈ expected
        @test prepare(both_plan)(x, location, scale, log(scale)) ≈ expected

        joint = extract(normal;
            have = (:x, :location, :scale), want = (:logpdf, :cdf))
        @test count(recipe -> only(recipe.outputs).name === :standardized,
                    plan(joint).recipes) == 1
        @test all(isapprox.(prepare(joint)(x, location, scale),
            (logpdf(Normal(location, scale), x),
             cdf(Normal(location, scale), x))))

        standard_term = extract(normal;
            have = (:x, :location, :scale),
            want = Symbol("standard.logpdf"))
        @test haskey(normal, Symbol("standard.logpdf"))
        @test prepare(standard_term)(x, location, scale) ≈
              logpdf(Normal(), (x - location) / scale)

        scale_from_log = extract(normal;
            have = (:log_scale,), want = :scale)
        log_from_scale = extract(normal;
            have = (:scale,), want = :log_scale)
        @test prepare(scale_from_log)(log(scale)) ≈ scale
        @test prepare(log_from_scale)(scale) ≈ log(scale)

        quantile_plan = plan(normal.quantile)
        @test :x in outputs_of(quantile_plan)
        @test !(:standardized in outputs_of(quantile_plan))
    end
end
