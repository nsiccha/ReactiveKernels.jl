using ReactiveKernelsPPLExamples.RadonVariableSlopeCenteredExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

# Graph-independent oracle for the shared-intercept + centered varying-slope model.
function _radon_vsc_reference(q, county_idx, floor_measure, log_radon)
    nlp(x, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - m) / s)^2
    J = length(q) - 4
    alpha = q[1]
    beta = q[2:J + 1]
    mu_beta = q[J + 2]
    sigma_beta = exp(q[J + 3])
    sigma_y = exp(q[J + 4])
    log_jacobian = q[J + 3] + q[J + 4]
    fixed_prior = nlp(alpha, 0.0, 10.0) + nlp(mu_beta, 0.0, 10.0) +
                  nlp(sigma_beta, 0.0, 1.0) + nlp(sigma_y, 0.0, 1.0)
    beta_prior = sum(nlp(b, mu_beta, sigma_beta) for b in beta)
    prior = fixed_prior + beta_prior
    mu = alpha .+ floor_measure .* beta[county_idx]
    likelihood = sum(nlp(y, m, sigma_y) for (y, m) in zip(log_radon, mu))
    (; parameters = (; alpha, beta, mu_beta, sigma_beta, sigma_y), prior,
       log_jacobian, likelihood, mu,
       posterior = prior + likelihood + log_jacobian)
end

@testset "PPL graph — radon_variable_slope_centered (posteriordb)" begin
    artifact = evaluate_radon_variable_slope_centered_source()
    @test artifact.source == strip(RADON_VARIABLE_SLOPE_CENTERED_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = vcat([0.5], 0.1 .* collect(1:8), [-0.5, log(0.6), log(0.7)])
    reference = _radon_vsc_reference(q, RADON_VSC_COUNTY, RADON_VSC_FLOOR,
                                     RADON_VSC_LOG)

    @testset "authored on the current baseline surface" begin
        @test occursin("normal(a + f * bc, s).logpdf(y)",
                       RADON_VARIABLE_SLOPE_CENTERED_SOURCE)
        @test occursin("beta_county = beta[county_idx]",
                       RADON_VARIABLE_SLOPE_CENTERED_SOURCE)
        @test occursin("normal(m, s).logpdf(b)",
                       RADON_VARIABLE_SLOPE_CENTERED_SOURCE)
        @test occursin("bound = (; county_idx)",
                       RADON_VARIABLE_SLOPE_CENTERED_SOURCE)
        @test !occursin("struct ", RADON_VARIABLE_SLOPE_CENTERED_SOURCE)
        @test artifact.normal_object === normal
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.county_idx,
                         model.floor_measure, model.log_radon),
                 want = (model.prior, model.log_jacobian, model.likelihood,
                         model.posterior))
        prior, log_jacobian, likelihood, posterior =
            prepare(p)(q, RADON_VSC_COUNTY, RADON_VSC_FLOOR, RADON_VSC_LOG)
        @test prior ≈ reference.prior
        @test log_jacobian ≈ reference.log_jacobian
        @test likelihood ≈ reference.likelihood
        @test posterior ≈ reference.posterior
    end

    @testset "gathered + combined mean mu from a constrained HAVE" begin
        p = plan(model.graph;
                 have = (model.parameters, model.county_idx, model.floor_measure),
                 want = (model.mu,))
        mu = prepare(p)(reference.parameters, RADON_VSC_COUNTY, RADON_VSC_FLOOR)
        @test mu ≈ reference.mu
    end
end
