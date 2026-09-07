using ReactiveKernelsPPLExamples.LogmesquiteLogvasExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

# Graph-independent reference oracle for posteriordb logmesquite_logvas: a
# log-scale Gaussian regression of bush weight on log canopy volume, area, shape,
# log total height, log density, and group; implicit improper-flat priors on the
# seven coefficients and on σ > 0 (only its exp/log Jacobian log|dσ/du| = u enters).
function _logmesquite_logvas_reference(q, log_weight, diam1, diam2, canopy_height,
                                       total_height, density, group)
    nlp(v, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((v - m) / s)^2
    b1, b2, b3, b4, b5, b6, b7, u_sigma = q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8]
    sigma = exp(u_sigma)
    log_jacobian = u_sigma
    mu = b1 .+ b2 .* log.(diam1 .* diam2 .* canopy_height) .+
         b3 .* log.(diam1 .* diam2) .+ b4 .* log.(diam1 ./ diam2) .+
         b5 .* log.(total_height) .+ b6 .* log.(density) .+ b7 .* group
    pointwise = [nlp(lw, m, sigma) for (lw, m) in zip(log_weight, mu)]
    likelihood = sum(pointwise)
    (; log_jacobian, likelihood, posterior = likelihood + log_jacobian)
end

@testset "PPL graph — logmesquite_logvas (posteriordb)" begin
    artifact = evaluate_logmesquite_logvas_source()
    @test artifact.source == strip(LOGMESQUITE_LOGVAS_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = [5.0, 0.8, 0.1, 0.1, 0.1, -0.1, 0.2, log(0.4)]
    reference = _logmesquite_logvas_reference(q, LOGVAS_LOG_WEIGHT, LOGVAS_DIAM1,
        LOGVAS_DIAM2, LOGVAS_CANOPY_HEIGHT, LOGVAS_TOTAL_HEIGHT, LOGVAS_DENSITY, LOGVAS_GROUP)

    @testset "authored on the current baseline surface" begin
        @test occursin("b4 * log(d1 / d2)", LOGMESQUITE_LOGVAS_SOURCE)
        @test occursin("b6 * log(den)", LOGMESQUITE_LOGVAS_SOURCE)
        @test !occursin("struct ", LOGMESQUITE_LOGVAS_SOURCE)
        @test artifact.normal_object === normal
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = prepare(model;
            have = (:unconstrained, :log_weight, :diam1, :diam2, :canopy_height,
                    :total_height, :density, :group),
            want = (:log_jacobian, :likelihood, :posterior))
        log_jacobian, likelihood, posterior =
            p(q, LOGVAS_LOG_WEIGHT, LOGVAS_DIAM1, LOGVAS_DIAM2, LOGVAS_CANOPY_HEIGHT,
              LOGVAS_TOTAL_HEIGHT, LOGVAS_DENSITY, LOGVAS_GROUP)
        @test log_jacobian ≈ reference.log_jacobian
        @test likelihood ≈ reference.likelihood
        @test posterior ≈ reference.posterior
    end

    @testset "one authored plate exposes a buffer-free total" begin
        have = (:unconstrained, :log_weight, :diam1, :diam2, :canopy_height,
                :total_height, :density, :group)
        likelihood_kernel = prepare(model; have, want = :likelihood)
        @test !occursin("similar", string(code_expr(likelihood_kernel)))
    end
end
