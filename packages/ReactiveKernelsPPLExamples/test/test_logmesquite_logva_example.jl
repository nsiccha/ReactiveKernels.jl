using ReactiveKernelsPPLExamples.LogmesquiteLogvaExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

# Graph-independent reference oracle for posteriordb logmesquite_logva: a
# log-scale Gaussian regression of bush weight on log canopy volume, log canopy
# area, and group, with implicit improper-flat priors on the four coefficients
# and on σ > 0 (only its exp/log transform Jacobian log|dσ/du| = u enters).
function _logmesquite_logva_reference(q, log_weight, diam1, diam2, canopy_height, group)
    nlp(v, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((v - m) / s)^2
    b1, b2, b3, b4, u_sigma = q[1], q[2], q[3], q[4], q[5]
    sigma = exp(u_sigma)
    log_jacobian = u_sigma
    mu = b1 .+ b2 .* log.(diam1 .* diam2 .* canopy_height) .+
         b3 .* log.(diam1 .* diam2) .+ b4 .* group
    pointwise = [nlp(lw, m, sigma) for (lw, m) in zip(log_weight, mu)]
    likelihood = sum(pointwise)
    (; parameters = (; beta1 = b1, beta2 = b2, beta3 = b3, beta4 = b4, sigma),
       log_jacobian, mu, pointwise, likelihood,
       posterior = likelihood + log_jacobian, weight_fitted = exp.(mu))
end

@testset "PPL graph — logmesquite_logva (posteriordb)" begin
    artifact = evaluate_logmesquite_logva_source()
    @test artifact.source == strip(LOGMESQUITE_LOGVA_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = [5.0, 0.8, 0.2, 0.2, log(0.4)]
    reference = _logmesquite_logva_reference(q, LOGVA_LOG_WEIGHT, LOGVA_DIAM1,
                                             LOGVA_DIAM2, LOGVA_CANOPY_HEIGHT, LOGVA_GROUP)

    @testset "authored on the current baseline surface" begin
        @test occursin("b3 * log(d1 * d2)", LOGMESQUITE_LOGVA_SOURCE)
        @test occursin(".logpdf(lw)", LOGMESQUITE_LOGVA_SOURCE)
        @test !occursin("struct ", LOGMESQUITE_LOGVA_SOURCE)
        @test artifact.normal_object === normal
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = prepare(model;
            have = (:unconstrained, :log_weight, :diam1, :diam2, :canopy_height, :group),
            want = (:log_jacobian, :likelihood, :posterior))
        log_jacobian, likelihood, posterior =
            p(q, LOGVA_LOG_WEIGHT, LOGVA_DIAM1, LOGVA_DIAM2, LOGVA_CANOPY_HEIGHT, LOGVA_GROUP)
        @test log_jacobian ≈ reference.log_jacobian
        @test likelihood ≈ reference.likelihood
        @test posterior ≈ reference.posterior
    end

    @testset "one authored plate exposes a buffer-free total" begin
        have = (:unconstrained, :log_weight, :diam1, :diam2, :canopy_height, :group)
        pointwise_kernel = prepare(model; have, want = :pointwise)
        likelihood_kernel = prepare(model; have, want = :likelihood)
        pw = pointwise_kernel(q, LOGVA_LOG_WEIGHT, LOGVA_DIAM1, LOGVA_DIAM2, LOGVA_CANOPY_HEIGHT, LOGVA_GROUP)
        @test likelihood_kernel(q, LOGVA_LOG_WEIGHT, LOGVA_DIAM1, LOGVA_DIAM2, LOGVA_CANOPY_HEIGHT, LOGVA_GROUP) ≈ sum(pw)
        @test !occursin("similar", string(code_expr(likelihood_kernel)))
    end
end
