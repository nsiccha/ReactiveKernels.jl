using ReactiveKernelsPPLExamples.ARKExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, cauchy

# Graph-independent oracle for arK (AR(K) as a Gaussian lag-matrix regression).
function _ark_reference(q, ylag, yt)
    K = size(ylag, 2)
    alpha = q[1]
    beta = q[2:K + 1]
    log_sigma = q[K + 2]
    sigma = exp(log_sigma)
    nlp(x, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - m) / s)^2
    clp(x, loc, sc) = -log(pi) - log(sc) - log1p(((x - loc) / sc)^2)
    prior = nlp(alpha, 0, 10) + sum(nlp(b, 0, 10) for b in beta) + clp(sigma, 0, 2.5)
    eta = alpha .+ ylag * beta
    likelihood = sum(nlp(yt[i], eta[i], sigma) for i in eachindex(yt))
    (; parameters = (; alpha, beta, sigma), prior, likelihood, eta,
       log_jacobian = log_sigma, posterior = prior + likelihood + log_sigma)
end

@testset "PPL graph — arK (posteriordb AR(K))" begin
    artifact = evaluate_ark_source()
    @test artifact.source == strip(ARK_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = [0.3, 0.5, -0.2, 0.15, 0.1, -0.05, log(0.4)]
    reference = _ark_reference(q, ARK_YLAG, ARK_YT)

    @testset "authored on the current baseline surface" begin
        @test occursin("lagged = ylag * beta", ARK_SOURCE)
        @test occursin("cauchy(0.0, 2.5).logpdf(sigma)", ARK_SOURCE)
        @test !occursin("struct ", ARK_SOURCE)
        @test artifact.cauchy_object === cauchy
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.ylag, model.yt),
                 want = (model.prior, model.likelihood, model.posterior))
        prior, likelihood, posterior = prepare(p)(q, ARK_YLAG, ARK_YT)
        @test prior ≈ reference.prior
        @test likelihood ≈ reference.likelihood
        @test posterior ≈ reference.posterior
    end

    @testset "lag mean from a constrained HAVE" begin
        p = plan(model.graph; have = (model.parameters, model.ylag), want = (model.lagged,))
        @test prepare(p)(reference.parameters, ARK_YLAG) ≈ ARK_YLAG * reference.parameters.beta
    end
end
