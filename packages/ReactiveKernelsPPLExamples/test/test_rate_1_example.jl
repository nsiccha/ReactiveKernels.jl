using ReactiveKernelsPPLExamples.Rate1Example
using ReactiveKernelsDistributionKernels.DistributionKernelSources: beta, binomial
using LogExpFunctions: logistic, log1pexp

function _rate1_reference(q, n, k)
    theta = logistic(q[1])
    jac = -log1pexp(-q[1]) - log1pexp(q[1])
    like = log(float(Base.binomial(n, k))) + k * log(theta) + (n - k) * log1p(-theta)
    (; parameters = (; theta), prior = 0.0, likelihood = like,
       log_jacobian = jac, posterior = like + jac)
end

@testset "PPL graph — Rate_1 (posteriordb inferring a rate)" begin
    artifact = evaluate_rate_1_source()
    @test artifact.source == strip(RATE_1_SOURCE, '\n')
    @test artifact.output == Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = [0.1]
    reference = _rate1_reference(q, RATE1_N, RATE1_K)
    @test occursin("binomial(n, theta).logpdf(k)", RATE_1_SOURCE)
    @test artifact.beta_object === beta
    # Pruned want (no :parameters) — exercises subset pruning of the top-level binomial.
    prior, likelihood, posterior = prepare(model;
        have = (:unconstrained, :n, :k),
        want = (:prior, :likelihood, :posterior))(q, RATE1_N, RATE1_K)
    @test likelihood ≈ reference.likelihood
    @test posterior ≈ reference.posterior
end
