using ReactiveKernelsPPLExamples.LogearnInteractionZExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal
using Statistics: mean, std

function _logearn_interaction_z_reference(q, earn, height, male)
    nlp(v, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((v - m) / s)^2
    b1, b2, b3, b4, u = q[1], q[2], q[3], q[4], q[5]
    sigma = exp(u); log_jacobian = u
    zh = (height .- mean(height)) ./ std(height)   # std = sample sd (N-1), matches Stan sd()
    mu = b1 .+ b2 .* zh .+ b3 .* male .+ b4 .* (zh .* male)
    likelihood = sum(nlp(log(e), m, sigma) for (e, m) in zip(earn, mu))
    (; log_jacobian, likelihood, posterior = likelihood + log_jacobian)
end

@testset "PPL graph — logearn_interaction_z (posteriordb)" begin
    artifact = evaluate_logearn_interaction_z_source()
    @test artifact.source == strip(LOGEARN_INTERACTION_Z_SOURCE, '\n')
    @test artifact.output == Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = [6.0, 0.1, 0.2, 0.05, log(0.9)]
    reference = _logearn_interaction_z_reference(q, LEIZ_EARN, LEIZ_HEIGHT, LEIZ_MALE)
    @test artifact.normal_object === normal
    @test occursin("sd_height", LOGEARN_INTERACTION_Z_SOURCE)

    p = prepare(model; have = (:unconstrained, :height, :male, :earn),
                want = (:log_jacobian, :likelihood, :posterior))
    log_jacobian, likelihood, posterior = p(q, LEIZ_HEIGHT, LEIZ_MALE, LEIZ_EARN)
    @test log_jacobian ≈ reference.log_jacobian
    @test likelihood ≈ reference.likelihood
    @test posterior ≈ reference.posterior
    # (The sample-sd reduction materializes one squared-deviation buffer, so this
    # model is not buffer-free — unlike the non-standardized earnings variants.)
end
