using ReactiveKernelsPPLExamples.RadonVariableInterceptSlopeCenteredExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

# Graph-independent oracle for the centered varying-intercept + varying-slope
# (independent effects) model.
function _radon_visc_reference(q, county_idx, floor_measure, log_radon)
    nlp(x, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - m) / s)^2
    J = div(length(q) - 5, 2)
    sigma_y = exp(q[1])
    sigma_alpha = exp(q[2])
    sigma_beta = exp(q[3])
    alpha = q[4:J + 3]
    beta = q[J + 4:2J + 3]
    mu_alpha = q[2J + 4]
    mu_beta = q[2J + 5]
    log_jacobian = q[1] + q[2] + q[3]
    fixed_prior = nlp(mu_alpha, 0.0, 10.0) + nlp(mu_beta, 0.0, 10.0) +
                  nlp(sigma_y, 0.0, 1.0) + nlp(sigma_alpha, 0.0, 1.0) +
                  nlp(sigma_beta, 0.0, 1.0)
    alpha_prior = sum(nlp(a, mu_alpha, sigma_alpha) for a in alpha)
    beta_prior = sum(nlp(b, mu_beta, sigma_beta) for b in beta)
    prior = fixed_prior + alpha_prior + beta_prior
    mu = alpha[county_idx] .+ floor_measure .* beta[county_idx]
    likelihood = sum(nlp(y, m, sigma_y) for (y, m) in zip(log_radon, mu))
    (; parameters = (; alpha, beta, mu_alpha, mu_beta, sigma_alpha, sigma_beta,
                     sigma_y), prior, log_jacobian, likelihood, mu,
       posterior = prior + likelihood + log_jacobian)
end

@testset "PPL graph — radon_variable_intercept_slope_centered (posteriordb)" begin
    artifact = evaluate_radon_variable_intercept_slope_centered_source()
    @test artifact.source ==
          strip(RADON_VARIABLE_INTERCEPT_SLOPE_CENTERED_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = vcat([log(0.65), log(0.55), log(0.5)],
             0.1 .* collect(1:8) .- 0.2,
             0.05 .* collect(1:8),
             [0.5, -0.3])
    reference = _radon_visc_reference(q, RADON_VISC_COUNTY, RADON_VISC_FLOOR,
                                      RADON_VISC_LOG)

    @testset "authored on the current baseline surface" begin
        @test occursin("normal(ac + f * bc, s).logpdf(y)",
                       RADON_VARIABLE_INTERCEPT_SLOPE_CENTERED_SOURCE)
        @test occursin("alpha_county = alpha[county_idx]",
                       RADON_VARIABLE_INTERCEPT_SLOPE_CENTERED_SOURCE)
        @test occursin("beta_county = beta[county_idx]",
                       RADON_VARIABLE_INTERCEPT_SLOPE_CENTERED_SOURCE)
        @test occursin("normal(m, s).logpdf(a)",
                       RADON_VARIABLE_INTERCEPT_SLOPE_CENTERED_SOURCE)
        @test occursin("normal(m, s).logpdf(b)",
                       RADON_VARIABLE_INTERCEPT_SLOPE_CENTERED_SOURCE)
        @test occursin("bound = (; county_idx)",
                       RADON_VARIABLE_INTERCEPT_SLOPE_CENTERED_SOURCE)
        @test !occursin("struct ", RADON_VARIABLE_INTERCEPT_SLOPE_CENTERED_SOURCE)
        @test artifact.normal_object === normal
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.county_idx,
                         model.floor_measure, model.log_radon),
                 want = (model.prior, model.log_jacobian, model.likelihood,
                         model.posterior))
        prior, log_jacobian, likelihood, posterior =
            prepare(p)(q, RADON_VISC_COUNTY, RADON_VISC_FLOOR, RADON_VISC_LOG)
        @test prior ≈ reference.prior
        @test log_jacobian ≈ reference.log_jacobian
        @test likelihood ≈ reference.likelihood
        @test posterior ≈ reference.posterior
    end

    @testset "gathered + combined mean mu from a constrained HAVE" begin
        p = plan(model.graph;
                 have = (model.parameters, model.county_idx, model.floor_measure),
                 want = (model.mu,))
        mu = prepare(p)(reference.parameters, RADON_VISC_COUNTY, RADON_VISC_FLOOR)
        @test mu ≈ reference.mu
    end
end
