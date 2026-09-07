using ReactiveKernelsPPLExamples.RadonHierarchicalInterceptCenteredExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

# Graph-independent oracle for the centered varying-intercept + county-level
# uranium contextual predictor model.
function _radon_hic_reference(q, county_idx, log_uppm, floor_measure, log_radon)
    nlp(x, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - m) / s)^2
    J = length(q) - 5
    alpha = q[1:J]
    beta = q[J + 1:J + 2]
    mu_alpha = q[J + 3]
    sigma_alpha = exp(q[J + 4])
    sigma_y = exp(q[J + 5])
    log_jacobian = q[J + 4] + q[J + 5]
    beta1 = beta[1]
    beta2 = beta[2]
    fixed_prior = nlp(mu_alpha, 0.0, 10.0) + nlp(sigma_alpha, 0.0, 1.0) +
                  nlp(sigma_y, 0.0, 1.0) + sum(nlp(b, 0.0, 10.0) for b in beta)
    alpha_prior = sum(nlp(a, mu_alpha, sigma_alpha) for a in alpha)
    prior = fixed_prior + alpha_prior
    mu = alpha[county_idx] .+ log_uppm .* beta1 .+ floor_measure .* beta2
    likelihood = sum(nlp(y, m, sigma_y) for (y, m) in zip(log_radon, mu))
    (; parameters = (; alpha, beta, mu_alpha, sigma_alpha, sigma_y), prior,
       log_jacobian, likelihood, mu,
       posterior = prior + likelihood + log_jacobian)
end

@testset "PPL graph — radon_hierarchical_intercept_centered (posteriordb)" begin
    artifact = evaluate_radon_hierarchical_intercept_centered_source()
    @test artifact.source ==
          strip(RADON_HIERARCHICAL_INTERCEPT_CENTERED_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = vcat(0.1 .* collect(1:8) .- 0.3, [0.2, -0.5],
             [0.5, log(0.55), log(0.65)])
    reference = _radon_hic_reference(q, RADON_HIC_COUNTY, RADON_HIC_UPPM,
                                     RADON_HIC_FLOOR, RADON_HIC_LOG)

    @testset "authored on the current baseline surface" begin
        @test occursin("normal(ac + u * b1 + f * b2, s).logpdf(y)",
                       RADON_HIERARCHICAL_INTERCEPT_CENTERED_SOURCE)
        @test occursin("alpha_county = alpha[county_idx]",
                       RADON_HIERARCHICAL_INTERCEPT_CENTERED_SOURCE)
        @test occursin("mu = alpha_county .+ log_uppm .* beta1 .+ floor_measure .* beta2",
                       RADON_HIERARCHICAL_INTERCEPT_CENTERED_SOURCE)
        @test occursin("beta1::Float64 = sum(view(beta, 1:1))",
                       RADON_HIERARCHICAL_INTERCEPT_CENTERED_SOURCE)
        @test occursin("normal(m, s).logpdf(a)",
                       RADON_HIERARCHICAL_INTERCEPT_CENTERED_SOURCE)
        @test occursin("normal(0.0, 10.0).logpdf(b)",
                       RADON_HIERARCHICAL_INTERCEPT_CENTERED_SOURCE)
        @test occursin("bound = (; county_idx)",
                       RADON_HIERARCHICAL_INTERCEPT_CENTERED_SOURCE)
        @test !occursin("struct ", RADON_HIERARCHICAL_INTERCEPT_CENTERED_SOURCE)
        @test artifact.normal_object === normal
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.county_idx, model.log_uppm,
                         model.floor_measure, model.log_radon),
                 want = (model.prior, model.log_jacobian, model.likelihood,
                         model.posterior))
        prior, log_jacobian, likelihood, posterior =
            prepare(p)(q, RADON_HIC_COUNTY, RADON_HIC_UPPM, RADON_HIC_FLOOR,
                       RADON_HIC_LOG)
        @test prior ≈ reference.prior
        @test log_jacobian ≈ reference.log_jacobian
        @test likelihood ≈ reference.likelihood
        @test posterior ≈ reference.posterior
    end

    @testset "contextual + combined mean mu from a constrained HAVE" begin
        p = plan(model.graph;
                 have = (model.parameters, model.county_idx, model.log_uppm,
                         model.floor_measure),
                 want = (model.mu,))
        mu = prepare(p)(reference.parameters, RADON_HIC_COUNTY, RADON_HIC_UPPM,
                        RADON_HIC_FLOOR)
        @test mu ≈ reference.mu
    end
end
