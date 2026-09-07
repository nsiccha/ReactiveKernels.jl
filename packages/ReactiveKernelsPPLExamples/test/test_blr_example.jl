using ReactiveKernelsPPLExamples.BLRExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

# Graph-independent oracle for Bayesian linear regression y ~ Normal(X*beta, sigma).
function _blr_reference(q, X, y)
    D = length(q) - 1
    beta = q[1:D]
    log_sigma = q[D + 1]
    sigma = exp(log_sigma)
    nlp(x, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - m) / s)^2
    prior = sum(nlp(b, 0, 10) for b in beta) + nlp(sigma, 0, 10)
    eta = X * beta
    likelihood = sum(nlp(y[i], eta[i], sigma) for i in eachindex(y))
    (; parameters = (; beta, sigma), prior, likelihood, eta, log_jacobian = log_sigma,
       posterior = prior + likelihood + log_sigma)
end

@testset "PPL graph — blr (posteriordb Bayesian linear regression)" begin
    artifact = evaluate_blr_source()
    @test artifact.source == strip(BLR_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = [0.4, -0.2, 0.15, log(0.6)]
    reference = _blr_reference(q, BLR_X, BLR_Y)

    @testset "authored on the current baseline surface" begin
        @test occursin("eta = predictors * beta", BLR_SOURCE)
        @test occursin("normal(0.0, 10.0).logpdf", BLR_SOURCE)
        @test !occursin("struct ", BLR_SOURCE)
        @test artifact.normal_object === normal
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.predictors, model.responses),
                 want = (model.prior, model.likelihood, model.posterior))
        prior, likelihood, posterior = prepare(p)(q, BLR_X, BLR_Y)
        @test prior ≈ reference.prior
        @test likelihood ≈ reference.likelihood
        @test posterior ≈ reference.posterior
    end

    @testset "linear predictor eta from a constrained HAVE" begin
        p = plan(model.graph; have = (model.parameters, model.predictors), want = (model.eta,))
        @test prepare(p)(reference.parameters, BLR_X) ≈ reference.eta
    end
end
