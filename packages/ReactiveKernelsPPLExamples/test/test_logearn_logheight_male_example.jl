using ReactiveKernelsPPLExamples.LogearnLogheightMaleExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

function _logearn_logheight_male_reference(q, earn, height, male)
    nlp(v, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((v - m) / s)^2
    b1, b2, b3, u = q[1], q[2], q[3], q[4]
    sigma = exp(u); log_jacobian = u
    mu = b1 .+ b2 .* log.(height) .+ b3 .* male
    likelihood = sum(nlp(log(e), m, sigma) for (e, m) in zip(earn, mu))
    (; log_jacobian, likelihood, posterior = likelihood + log_jacobian)
end

@testset "PPL graph — logearn_logheight_male (posteriordb)" begin
    artifact = evaluate_logearn_logheight_male_source()
    @test artifact.source == strip(LOGEARN_LOGHEIGHT_MALE_SOURCE, '\n')
    @test artifact.output == Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = [6.0, 0.5, 0.2, log(0.9)]
    reference = _logearn_logheight_male_reference(q, LELHM_EARN, LELHM_HEIGHT, LELHM_MALE)
    @test artifact.normal_object === normal
    @test occursin("b2 * log(h)", LOGEARN_LOGHEIGHT_MALE_SOURCE)

    p = prepare(model; have = (:unconstrained, :height, :male, :earn),
                want = (:log_jacobian, :likelihood, :posterior))
    log_jacobian, likelihood, posterior = p(q, LELHM_HEIGHT, LELHM_MALE, LELHM_EARN)
    @test log_jacobian ≈ reference.log_jacobian
    @test likelihood ≈ reference.likelihood
    @test posterior ≈ reference.posterior

    likelihood_kernel = prepare(model;
        have = (:unconstrained, :height, :male, :earn), want = :likelihood)
    @test !occursin("similar", string(code_expr(likelihood_kernel)))
end
