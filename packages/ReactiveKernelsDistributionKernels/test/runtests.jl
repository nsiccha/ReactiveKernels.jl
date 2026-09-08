using Distributions: Bernoulli, Cauchy, Dirichlet, Exponential, Geometric,
    InverseGamma, Laplace, LKJCholesky, Logistic, LogNormal, MvNormal, Normal,
    TDist, Uniform, cdf, logpdf, quantile
using LinearAlgebra: Cholesky, LowerTriangular, Symmetric, cholesky
using LogExpFunctions: log1pexp
using ReactiveKernels: @kernel, KernelObjectSpec, KernelSpec, code_expr, explain,
    extract, plan, plate, prepare
using ReactiveKernelsDistributionKernels: DistributionKernelSources
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    LOCATION_SCALE_SOURCE,
    standard_normal, standard_cauchy, standard_laplace, standard_student_t,
    standard_logistic, location_scale, student_t,
    BERNOULLI_KERNEL_SOURCE, LOGNORMAL_KERNEL_SOURCE,
    EXPONENTIAL_KERNEL_SOURCE, GEOMETRIC_KERNEL_SOURCE, UNIFORM_KERNEL_SOURCE,
    MVNORMAL_KERNEL_SOURCE, AR1_KERNEL_SOURCE,
    CATEGORICAL_LOGIT_KERNEL_SOURCE, CATEGORICAL_LOGIT_REF_KERNEL_SOURCE,
    INVERSE_GAMMA_KERNEL_SOURCE, DIRICHLET_KERNEL_SOURCE,
    LKJ_CORR_CHOLESKY_KERNEL_SOURCE,
    normal, cauchy, laplace, logistic, bernoulli, lognormal,
    exponential, geometric, uniform, mvnormal, ar1,
    categorical_logit, categorical_logit_ref,
    inverse_gamma, dirichlet, lkj_corr_cholesky,
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
        (normal, cauchy, laplace, logistic, bernoulli, lognormal,
         exponential, geometric, uniform, mvnormal, ar1,
         categorical_logit, categorical_logit_ref, lkj_corr_cholesky))
    @test all(object -> object isa KernelObjectSpec,
        (normal, cauchy, laplace, logistic))
    @test all(template -> !isnothing(template),
        (standard_normal, standard_cauchy, standard_laplace, standard_logistic,
         location_scale))
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
            (laplace, Laplace(location, scale)),
            (logistic, Logistic(location, scale)))
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

@testset "student_t location-scale family" begin
    @test hasproperty(standard_student_t, :logpdf)
    @test hasproperty(student_t, :logpdf)
    @test occursin("@kernel standard_student_t(nu", LOCATION_SCALE_SOURCE)
    @test occursin("@kernel student_t(nu", LOCATION_SCALE_SOURCE)

    # Standardized t against Distributions.TDist across degrees of freedom.
    for nu in (1.0, 2.5, 3.0, 8.0, 30.0)
        reference = TDist(nu)
        for z in (-2.3, -0.4, 0.0, 0.7, 1.9)
            @test prepare(standard_student_t.logpdf;
                have = (:z, :nu), want = :logpdf)(z, nu) ≈ logpdf(reference, z)
            @test prepare(standard_student_t.cdf;
                have = (:z, :nu), want = :cdf)(z, nu) ≈ cdf(reference, z)
        end
        for p in (0.05, 0.27, 0.5, 0.73, 0.95)
            @test prepare(standard_student_t.quantile;
                have = (:p, :nu), want = :quantile)(p, nu) ≈ quantile(reference, p)
        end
    end

    # df=1 recovers the standard Cauchy.
    @test prepare(standard_student_t.logpdf; have = (:z, :nu), want = :logpdf)(
        0.7, 1.0) ≈ prepare(standard_cauchy.logpdf)(0.7)

    # Location-scale t: nu is lifted alongside location/scale, and scale and
    # log_scale are equivalent HAVE routes for the one graph.
    for (nu, location, scale) in ((3.0, 8.0, 10.0), (2.5, -1.0, 0.7), (30.0, 0.2, 2.0))
        for x in (-3.0, 0.4, 8.0, 12.0)
            reference = logpdf(TDist(nu), (x - location) / scale) - log(scale)
            @test prepare(student_t.logpdf;
                have = (:x, :location, :scale, :nu), want = :logpdf)(
                    x, location, scale, nu) ≈ reference
            @test prepare(student_t.logpdf;
                have = (:x, :location, :log_scale, :nu), want = :logpdf)(
                    x, location, log(scale), nu) ≈ reference
            @test prepare(student_t.cdf;
                have = (:x, :location, :scale, :nu), want = :cdf)(
                    x, location, scale, nu) ≈ cdf(TDist(nu), (x - location) / scale)
        end
        for p in (0.1, 0.4, 0.6, 0.9)
            @test prepare(student_t.quantile;
                have = (:p, :location, :scale, :nu), want = :quantile)(
                    p, location, scale, nu) ≈ location + scale * quantile(TDist(nu), p)
        end
    end

    # brms half-t truncation constant: student_t_lccdf(0 | nu, 0, scale) = log(0.5)
    # by symmetry, so cdf(0) = 0.5 for any nu/scale.
    @test prepare(student_t.cdf; have = (:x, :location, :scale, :nu), want = :cdf)(
        0.0, 0.0, 10.0, 3.0) ≈ 0.5

    # Lifts through the generic plate path; the compiled kernel is
    # Distributions.jl-free.
    plated = plate(student_t.logpdf;
        have = (:x, :location, :scale, :nu), want = :logpdf, batched = (:x,))
    observations = [0.4, 8.0, 12.0, -2.0]
    @test plated(observations, 0.0, 10.0, 3.0) ≈
        sum(logpdf(TDist(3.0), xi / 10.0) - log(10.0) for xi in observations)
    @test !occursin("Distributions", string(code_expr(plated)))
end

@testset "inverse_gamma and dirichlet conjugate-prior objects" begin
    @test hasproperty(inverse_gamma, :logpdf)
    @test hasproperty(dirichlet, :logpdf)
    @test occursin("@kernel inverse_gamma(", INVERSE_GAMMA_KERNEL_SOURCE)
    @test occursin("@kernel dirichlet(", DIRICHLET_KERNEL_SOURCE)
    @test all(source -> !occursin("@recipe", source),
        (INVERSE_GAMMA_KERNEL_SOURCE, DIRICHLET_KERNEL_SOURCE))

    @testset "Inverse-Gamma-Normal variance prior" begin
        shape, scale, x, p = 3.0, 2.0, 1.4, 0.6
        reference = InverseGamma(shape, scale)
        # Scale and log-scale are authoritative HAVE routes for one graph.
        @test prepare(inverse_gamma.logpdf;
            have = (:x, :shape, :scale), want = :logpdf)(x, shape, scale) ≈
            logpdf(reference, x)
        @test prepare(inverse_gamma.logpdf;
            have = (:x, :shape, :log_scale), want = :logpdf)(
                x, shape, log(scale)) ≈ logpdf(reference, x)
        @test prepare(inverse_gamma.cdf;
            have = (:x, :shape, :scale), want = :cdf)(x, shape, scale) ≈
            cdf(reference, x)
        @test prepare(inverse_gamma.quantile;
            have = (:p, :shape, :scale), want = :quantile)(p, shape, scale) ≈
            quantile(reference, p)
        # Domain guard: non-positive support is -Inf without control flow.
        @test prepare(inverse_gamma.logpdf;
            have = (:x, :shape, :scale), want = :logpdf)(-1.0, shape, scale) ==
            -Inf
        # Lifts through the generic plate path; the compiled kernel is
        # Distributions.jl-free.
        plated = plate(inverse_gamma.logpdf;
            have = (:x, :shape, :log_scale), want = :logpdf, batched = (:x,))
        observations = [0.6, 1.4, 2.2, 3.1]
        @test plated(observations, shape, log(scale)) ≈
            sum(logpdf(reference, xi) for xi in observations)
        @test !occursin("Distributions", string(code_expr(plated)))
    end

    @testset "Dirichlet-Categorical/Multinomial simplex prior" begin
        alpha = [2.0, 3.0, 1.5]
        x = [0.2, 0.5, 0.3]
        reference = Dirichlet(alpha)
        # One whole-vector graph (like mvnormal), not a scalar plate.
        kernel = prepare(dirichlet.logpdf; have = (:x, :alpha), want = :logpdf)
        @test kernel(x, alpha) ≈ logpdf(reference, x)
        # Off the positive orthant the density guard returns -Inf.
        @test kernel([-0.1, 0.6, 0.5], alpha) == -Inf
        @test occursin("@kernel dirichlet", DIRICHLET_KERNEL_SOURCE)
        @test !occursin("Distributions", string(code_expr(kernel)))
    end
end

@testset "LKJ correlation-Cholesky prior" begin
    # One whole-matrix graph (like mvnormal/dirichlet): the observation is a
    # single K×K lower-triangular correlation Cholesky factor `L`. Density parity
    # is checked against Distributions.LKJCholesky (== Stan's lkj_corr_cholesky).
    kernel = prepare(lkj_corr_cholesky.logpdf; have = (:L, :eta), want = :logpdf)
    # Valid correlation Cholesky factors (unit-norm rows) at K = 2, 3, 4.
    Ls = (
        [1.0 0.0; 0.6 0.8],
        [1.0 0.0 0.0; 0.6 0.8 0.0; 0.3 0.4 0.8660254037844386],
        [1.0 0.0 0.0 0.0;
         0.6 0.8 0.0 0.0;
         0.3 0.4 0.8660254037844386 0.0;
         0.2 0.3 0.4 0.8426149773176359],
    )
    for L in Ls, eta in (0.5, 1.0, 2.0, 4.0)
        reference = LKJCholesky(size(L, 1), eta, :L)
        @test kernel(L, eta) ≈ logpdf(reference, Cholesky(LowerTriangular(L)))
    end
    # dogs_nonhierarchical's exact prior: L_logit_ab ~ lkj_corr_cholesky(2), K = 2.
    let L = [1.0 0.0; 0.6 0.8]
        @test kernel(L, 2.0) ≈ logpdf(LKJCholesky(2, 2.0, :L),
                                      Cholesky(LowerTriangular(L)))
    end
    @test occursin("@kernel lkj_corr_cholesky", LKJ_CORR_CHOLESKY_KERNEL_SOURCE)
    @test !occursin("Distributions", string(code_expr(kernel)))
end

# The bernoulli object carries a per-HAVE `logpdf` route (a direct-p HAVE is not
# routed through the singular `logit = log(p) - log1p(-p)`); the stable logit-HAVE
# log-sum-exp form and the `logp`/`log1mp` object ports are retained. Gradient
# finiteness at the boundary is exercised under AD in `test/test_ad.jl` and
# `test/test_ad_reactant.jl`; this covers the primal, ports, and route shape.
@testset "bernoulli direct-p boundary and logit-route stability" begin
    logp_at(have, args...) = prepare(bernoulli.logpdf; have, want = :logpdf)(args...)

    @testset "boundary and impossible-event primal" begin
        # The probability-1 event, when observed, has logpdf log(1) = 0.
        @test logp_at((:observed, :p), true, 1.0) == 0.0
        @test logp_at((:observed, :p), false, 0.0) == 0.0
        # Impossible events stay -Inf.
        @test logp_at((:observed, :p), false, 1.0) == -Inf
        @test logp_at((:observed, :p), true, 0.0) == -Inf
        # Interior direct-p values match the reference exactly-enough.
        for p in (0.1, 0.37, 0.6, 0.92), observed in (true, false)
            @test logp_at((:observed, :p), observed, p) ≈ logpdf(Bernoulli(p), observed)
        end
    end

    @testset "logit HAVE stays the stable log-sum-exp form (saturating tails)" begin
        for logit in (-30.0, -20.0, -1.3, 0.0, 0.7, 20.0, 30.0), observed in (true, false)
            reference = observed ? -log1pexp(-logit) : -log1pexp(logit)
            # Bit-identical: the logit route is unchanged by the direct-p fix.
            @test logp_at((:observed, :logit), observed, logit) === reference
        end
    end

    @testset "logp / log1mp remain extractable object ports" begin
        @test prepare(extract(bernoulli; have = (:logit,), want = :logp))(0.8) ≈
            -log1pexp(-0.8)
        @test prepare(extract(bernoulli; have = (:logit,), want = :log1mp))(0.8) ≈
            -log1pexp(0.8)
        joint = prepare(extract(bernoulli; have = (:logit,), want = (:logp, :log1mp)))
        @test collect(joint(0.8)) ≈ [-log1pexp(-0.8), -log1pexp(0.8)]
        # A direct-p HAVE can still reach logp (routes p -> logit -> logp).
        @test prepare(extract(bernoulli; have = (:p,), want = :logp))(0.6) ≈ log(0.6)
    end

    @testset "each HAVE selects its own fused logpdf recipe (no cross round-trip)" begin
        selected(plan_str) = split(plan_str, "Alternatives not selected:")[1]
        logit_plan = selected(
            explain(plan(bernoulli.logpdf; have = (:observed, :logit), want = :logpdf)))
        p_plan = selected(
            explain(plan(bernoulli.logpdf; have = (:observed, :p), want = :logpdf)))
        # logit HAVE selects the (observed, logit) recipe and never the direct-p one.
        @test occursin("(observed, logit)", logit_plan)
        @test !occursin("(observed, p)", logit_plan)
        # p HAVE selects the (observed, p) recipe and never forms `logit`.
        @test occursin("(observed, p)", p_plan)
        @test !occursin("(observed, logit)", p_plan)
    end
end
