using ReactiveKernelsPPLExamples.RadonPooledExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

# Graph-independent oracle for the complete-pooling Gaussian regression.
function _radon_pooled_reference(q, floor_measure, log_radon)
    nlp(x, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - m) / s)^2
    alpha = q[1]
    beta = q[2]
    log_sigma_y = q[3]
    sigma_y = exp(log_sigma_y)
    log_jacobian = log_sigma_y
    prior = nlp(alpha, 0.0, 10.0) + nlp(beta, 0.0, 10.0) + nlp(sigma_y, 0.0, 1.0)
    mu = alpha .+ beta .* floor_measure
    likelihood = sum(nlp(y, m, sigma_y) for (y, m) in zip(log_radon, mu))
    (; parameters = (; alpha, beta, sigma_y), prior, log_jacobian, likelihood, mu,
       posterior = prior + likelihood + log_jacobian)
end

@testset "PPL graph — radon_pooled (posteriordb)" begin
    artifact = evaluate_radon_pooled_source()
    @test artifact.source == strip(RADON_POOLED_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = [0.7, -0.5, log(0.8)]
    reference = _radon_pooled_reference(q, RADON_POOLED_FLOOR, RADON_POOLED_LOG)

    @testset "authored on the current baseline surface" begin
        @test occursin("normal(a + b * f, s).logpdf(y)", RADON_POOLED_SOURCE)
        @test occursin("normal(0.0, 10.0).logpdf(alpha)", RADON_POOLED_SOURCE)
        @test occursin("mu = plate(", RADON_POOLED_SOURCE)
        @test !occursin("struct ", RADON_POOLED_SOURCE)
        @test artifact.normal_object === normal
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.floor_measure, model.log_radon),
                 want = (model.prior, model.log_jacobian, model.likelihood,
                         model.posterior))
        prior, log_jacobian, likelihood, posterior =
            prepare(p)(q, RADON_POOLED_FLOOR, RADON_POOLED_LOG)
        @test prior ≈ reference.prior
        @test log_jacobian ≈ reference.log_jacobian
        @test likelihood ≈ reference.likelihood
        @test posterior ≈ reference.posterior
    end

    @testset "generated quantity mu from a constrained HAVE" begin
        p = plan(model.graph;
                 have = (model.parameters, model.floor_measure), want = (model.mu,))
        mu = prepare(p)(reference.parameters, RADON_POOLED_FLOOR)
        @test mu ≈ reference.mu
    end

    @testset "one authored plate exposes a buffer-free total" begin
        ll_kernel = prepare(model;
            have = (:unconstrained, :floor_measure, :log_radon), want = :likelihood)
        @test !occursin("similar", string(code_expr(ll_kernel)))
    end
end
