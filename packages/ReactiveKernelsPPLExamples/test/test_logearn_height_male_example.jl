using ReactiveKernelsPPLExamples.LogearnHeightMaleExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

function _logearn_height_male_reference(q, earn, height, male)
    nlp(v, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((v - m) / s)^2
    b1, b2, b3, u = q[1], q[2], q[3], q[4]
    sigma = exp(u); log_jacobian = u
    mu = b1 .+ b2 .* height .+ b3 .* male
    likelihood = sum(nlp(log(e), m, sigma) for (e, m) in zip(earn, mu))
    (; log_jacobian, likelihood, posterior = likelihood + log_jacobian)
end

@testset "PPL graph — logearn_height_male (posteriordb)" begin
    artifact = evaluate_logearn_height_male_source()
    @test artifact.source == strip(LOGEARN_HEIGHT_MALE_SOURCE, '\n')
    @test artifact.output == Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = [6.0, 0.05, 0.2, log(0.9)]
    reference = _logearn_height_male_reference(q, LEHM_EARN, LEHM_HEIGHT, LEHM_MALE)
    @test artifact.normal_object === normal
    @test !occursin("struct ", LOGEARN_HEIGHT_MALE_SOURCE)

    p = prepare(model; have = (:unconstrained, :height, :male, :earn),
                want = (:log_jacobian, :likelihood, :posterior))
    log_jacobian, likelihood, posterior = p(q, LEHM_HEIGHT, LEHM_MALE, LEHM_EARN)
    @test log_jacobian ≈ reference.log_jacobian
    @test likelihood ≈ reference.likelihood
    @test posterior ≈ reference.posterior

    likelihood_kernel = prepare(model;
        have = (:unconstrained, :height, :male, :earn), want = :likelihood)
    @test !occursin("similar", string(code_expr(likelihood_kernel)))
end
