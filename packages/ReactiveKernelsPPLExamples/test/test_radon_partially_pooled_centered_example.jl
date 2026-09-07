using ReactiveKernelsPPLExamples.RadonPartiallyPooledCenteredExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

# Graph-independent oracle for the centered partial-pooling model.
function _radon_pp_centered_reference(q, county_idx, log_radon)
    nlp(x, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - m) / s)^2
    J = length(q) - 3
    alpha = q[1:J]
    mu_alpha = q[J + 1]
    sigma_alpha = exp(q[J + 2])
    sigma_y = exp(q[J + 3])
    log_jacobian = q[J + 2] + q[J + 3]
    fixed_prior = nlp(mu_alpha, 0.0, 10.0) + nlp(sigma_alpha, 0.0, 1.0) +
                  nlp(sigma_y, 0.0, 1.0)
    alpha_prior = sum(nlp(a, mu_alpha, sigma_alpha) for a in alpha)
    prior = fixed_prior + alpha_prior
    mu = alpha[county_idx]
    likelihood = sum(nlp(y, m, sigma_y) for (y, m) in zip(log_radon, mu))
    (; parameters = (; alpha, mu_alpha, sigma_alpha, sigma_y), prior, log_jacobian,
       likelihood, mu, posterior = prior + likelihood + log_jacobian)
end

@testset "PPL graph — radon_partially_pooled_centered (posteriordb)" begin
    artifact = evaluate_radon_partially_pooled_centered_source()
    @test artifact.source ==
          strip(RADON_PARTIALLY_POOLED_CENTERED_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = vcat(0.1 .* collect(1:8), [0.5, log(0.6), log(0.7)])
    reference = _radon_pp_centered_reference(q, RADON_PP_COUNTY, RADON_PP_LOG)

    @testset "authored on the current baseline surface" begin
        @test occursin("mu = alpha[county_idx]", RADON_PARTIALLY_POOLED_CENTERED_SOURCE)
        @test occursin("normal(m, s).logpdf(a)", RADON_PARTIALLY_POOLED_CENTERED_SOURCE)
        @test occursin("bound = (; county_idx)", RADON_PARTIALLY_POOLED_CENTERED_SOURCE)
        @test !occursin("struct ", RADON_PARTIALLY_POOLED_CENTERED_SOURCE)
        @test artifact.normal_object === normal
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.county_idx, model.log_radon),
                 want = (model.prior, model.log_jacobian, model.likelihood,
                         model.posterior))
        prior, log_jacobian, likelihood, posterior =
            prepare(p)(q, RADON_PP_COUNTY, RADON_PP_LOG)
        @test prior ≈ reference.prior
        @test log_jacobian ≈ reference.log_jacobian
        @test likelihood ≈ reference.likelihood
        @test posterior ≈ reference.posterior
    end

    @testset "hierarchical integer-array gather mu from a constrained HAVE" begin
        p = plan(model.graph;
                 have = (model.parameters, model.county_idx), want = (model.mu,))
        mu = prepare(p)(reference.parameters, RADON_PP_COUNTY)
        @test mu ≈ reference.mu
    end
end
