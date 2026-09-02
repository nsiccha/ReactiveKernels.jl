using Distributions: Bernoulli, Cauchy, Exponential, Geometric, Laplace,
    LogNormal, MvNormal, Normal, Uniform, cdf, logpdf, quantile
using LinearAlgebra: Symmetric, cholesky
using ReactiveKernels: @kernel, KernelObjectSpec, KernelSpec, code_expr, extract,
    plan, plate, prepare
using ReactiveKernelsDistributionKernels: DistributionKernelSources
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    LOCATION_SCALE_SOURCE,
    standard_normal, standard_cauchy, standard_laplace, location_scale,
    BERNOULLI_KERNEL_SOURCE, LOGNORMAL_KERNEL_SOURCE,
    EXPONENTIAL_KERNEL_SOURCE, GEOMETRIC_KERNEL_SOURCE, UNIFORM_KERNEL_SOURCE,
    MVNORMAL_KERNEL_SOURCE, AR1_KERNEL_SOURCE,
    CATEGORICAL_LOGIT_KERNEL_SOURCE, CATEGORICAL_LOGIT_REF_KERNEL_SOURCE,
    normal, cauchy, laplace, bernoulli, lognormal,
    exponential, geometric, uniform, mvnormal, ar1,
    categorical_logit, categorical_logit_ref,
    NORMAL_LOGDENSITY, CAUCHY_LOGDENSITY, LAPLACE_LOGDENSITY
using Test

@kernel _public_normal_plate_total(
        x::Vector{Float64}, location::Vector{Float64},
        scale::Vector{Float64}) = begin
    pointwise = plate(x, location, scale) do xi, li, si
        normal(li, si).logpdf(xi)
    end
    return sum(pointwise)
end

function _public_normal_plate_allocated(kernel, x, location, scale)
    kernel(x, location, scale)
    @allocated kernel(x, location, scale)
end

@testset "distribution kernel foundation" begin
    @test all(object -> hasproperty(object, :logpdf),
        (normal, cauchy, laplace, bernoulli, lognormal,
         exponential, geometric, uniform, mvnormal, ar1,
         categorical_logit, categorical_logit_ref))
    @test all(object -> object isa KernelObjectSpec, (normal, cauchy, laplace))
    @test all(template -> !isnothing(template),
        (standard_normal, standard_cauchy, standard_laplace, location_scale))
    @test all(spec -> spec isa KernelSpec,
        (NORMAL_LOGDENSITY, CAUCHY_LOGDENSITY, LAPLACE_LOGDENSITY))
    @test !isdefined(DistributionKernelSources, :Distributions)
    @test occursin("@kernel location_scale", LOCATION_SCALE_SOURCE)
    @test occursin("@kernel normal = location_scale(standard_normal)",
        LOCATION_SCALE_SOURCE)
    @test all(source -> !occursin("@recipe", source),
        (LOCATION_SCALE_SOURCE, BERNOULLI_KERNEL_SOURCE,
         LOGNORMAL_KERNEL_SOURCE, EXPONENTIAL_KERNEL_SOURCE,
         GEOMETRIC_KERNEL_SOURCE, UNIFORM_KERNEL_SOURCE,
         MVNORMAL_KERNEL_SOURCE, AR1_KERNEL_SOURCE,
         CATEGORICAL_LOGIT_KERNEL_SOURCE,
         CATEGORICAL_LOGIT_REF_KERNEL_SOURCE))

    @testset "public location-scale plate is allocation-free" begin
        xs = [-1.2, -0.1, 0.7, 1.8]
        locations = [0.1, 0.2, 0.4, 0.5]
        scales = [0.8, 1.0, 1.3, 1.5]
        total = prepare(_public_normal_plate_total)
        reference = sum(logpdf.(Normal.(locations, scales), xs))

        @test total(xs, locations, scales) ≈ reference
        @test !occursin("similar", string(code_expr(total)))
        @test _public_normal_plate_allocated(
            total, xs, locations, scales) == 0
    end

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

    @testset "all public families are method-bearing objects" begin
        bernoulli_p = 0.37
        bernoulli_logit = log(bernoulli_p) - log1p(-bernoulli_p)
        bernoulli_reference = Bernoulli(bernoulli_p)
        @test prepare(bernoulli.logpdf;
            have = (:observed, :logit), want = :logpdf)(true, bernoulli_logit) ≈
            logpdf(bernoulli_reference, true)
        @test prepare(bernoulli.cdf;
            have = (:observed, :p), want = :cdf)(false, bernoulli_p) ≈
            cdf(bernoulli_reference, false)
        @test prepare(bernoulli.quantile;
            have = (:q, :p), want = :quantile)(0.8, bernoulli_p) ==
            quantile(bernoulli_reference, 0.8)

        lognormal_reference = LogNormal(location, scale)
        positive_x = 1.4
        @test prepare(lognormal.logpdf;
            have = (:x, :location, :log_scale), want = :logpdf)(
                positive_x, location, log(scale)) ≈
            logpdf(lognormal_reference, positive_x)
        @test prepare(lognormal.cdf)(location, scale, positive_x) ≈
              cdf(lognormal_reference, positive_x)
        @test prepare(lognormal.quantile)(location, scale, p) ≈
              quantile(lognormal_reference, p)

        exponential_reference = Exponential(scale)
        @test prepare(exponential.logpdf;
            have = (:x, :log_scale), want = :logpdf)(positive_x, log(scale)) ≈
            logpdf(exponential_reference, positive_x)
        @test prepare(exponential.cdf)(scale, positive_x) ≈
              cdf(exponential_reference, positive_x)
        @test prepare(exponential.quantile)(scale, p) ≈
              quantile(exponential_reference, p)

        geometric_p = 0.6
        geometric_reference = Geometric(geometric_p)
        @test prepare(geometric.logpdf;
            have = (:observed, :p), want = :logpdf)(3, geometric_p) ≈
            logpdf(geometric_reference, 3)
        @test prepare(geometric.cdf)(geometric_p, 3) ≈
              cdf(geometric_reference, 3)
        @test prepare(geometric.quantile)(geometric_p, 0.8) ==
              quantile(geometric_reference, 0.8)

        lower, upper = -1.0, 2.0
        uniform_reference = Uniform(lower, upper)
        @test prepare(uniform.logpdf)(lower, upper, x) ≈
              logpdf(uniform_reference, x)
        @test prepare(uniform.cdf)(lower, upper, x) ≈ cdf(uniform_reference, x)
        @test prepare(uniform.quantile)(lower, upper, p) ≈
              quantile(uniform_reference, p)

        μ = [-0.2, 0.3, 0.5]
        observation = [0.4, -1.1, 0.7]
        chol = [1.2 0.0 0.0; 0.25 0.8 0.0; -0.1 0.35 1.1]
        covariance = chol * chol'
        precision = inv(covariance)
        precision_chol = Matrix(cholesky(Symmetric(precision)).L)
        mvnormal_reference = MvNormal(μ, covariance)
        for (have, parameter) in (
                ((:x, :μ, :covariance), covariance),
                ((:x, :μ, :chol), chol),
                ((:x, :μ, :precision), precision),
                ((:x, :μ, :precision_chol), precision_chol))
            @test prepare(mvnormal.logpdf; have, want = :logpdf)(
                observation, μ, parameter) ≈ logpdf(mvnormal_reference, observation)
        end

        @test hasproperty(ar1, :logpdf)
        @test occursin("@kernel ar1", AR1_KERNEL_SOURCE)

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
