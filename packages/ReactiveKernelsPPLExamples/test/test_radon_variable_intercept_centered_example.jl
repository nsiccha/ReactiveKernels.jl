using ReactiveKernelsPPLExamples.RadonVariableInterceptCenteredExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

# Graph-independent oracle for the varying-intercept + shared-slope model.
function _radon_vi_centered_reference(q, county_idx, floor_measure, log_radon)
    nlp(x, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - m) / s)^2
    J = length(q) - 4
    alpha = q[1:J]
    beta = q[J + 1]
    mu_alpha = q[J + 2]
    sigma_alpha = exp(q[J + 3])
    sigma_y = exp(q[J + 4])
    log_jacobian = q[J + 3] + q[J + 4]
    fixed_prior = nlp(mu_alpha, 0.0, 10.0) + nlp(beta, 0.0, 10.0) +
                  nlp(sigma_alpha, 0.0, 1.0) + nlp(sigma_y, 0.0, 1.0)
    alpha_prior = sum(nlp(a, mu_alpha, sigma_alpha) for a in alpha)
    prior = fixed_prior + alpha_prior
    mu = alpha[county_idx] .+ beta .* floor_measure
    likelihood = sum(nlp(y, m, sigma_y) for (y, m) in zip(log_radon, mu))
    (; parameters = (; alpha, beta, mu_alpha, sigma_alpha, sigma_y), prior,
       log_jacobian, likelihood, mu,
       posterior = prior + likelihood + log_jacobian)
end

@testset "PPL graph — radon_variable_intercept_centered (posteriordb)" begin
    artifact = evaluate_radon_variable_intercept_centered_source()
    @test artifact.source ==
          strip(RADON_VARIABLE_INTERCEPT_CENTERED_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = vcat(0.1 .* collect(1:8), [-0.5, 0.5, log(0.6), log(0.7)])
    reference = _radon_vi_centered_reference(q, RADON_VI_COUNTY, RADON_VI_FLOOR,
                                             RADON_VI_LOG)

    @testset "authored on the current baseline surface" begin
        @test occursin("normal(ac + b * f, s).logpdf(y)",
                       RADON_VARIABLE_INTERCEPT_CENTERED_SOURCE)
        @test occursin("alpha_county = alpha[county_idx]",
                       RADON_VARIABLE_INTERCEPT_CENTERED_SOURCE)
        @test occursin("normal(m, s).logpdf(a)",
                       RADON_VARIABLE_INTERCEPT_CENTERED_SOURCE)
        @test occursin("bound = (; county_idx)",
                       RADON_VARIABLE_INTERCEPT_CENTERED_SOURCE)
        @test !occursin("struct ", RADON_VARIABLE_INTERCEPT_CENTERED_SOURCE)
        @test artifact.normal_object === normal
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.county_idx,
                         model.floor_measure, model.log_radon),
                 want = (model.prior, model.log_jacobian, model.likelihood,
                         model.posterior))
        prior, log_jacobian, likelihood, posterior =
            prepare(p)(q, RADON_VI_COUNTY, RADON_VI_FLOOR, RADON_VI_LOG)
        @test prior ≈ reference.prior
        @test log_jacobian ≈ reference.log_jacobian
        @test likelihood ≈ reference.likelihood
        @test posterior ≈ reference.posterior
    end

    @testset "gathered + combined mean mu from a constrained HAVE" begin
        p = plan(model.graph;
                 have = (model.parameters, model.county_idx, model.floor_measure),
                 want = (model.mu,))
        mu = prepare(p)(reference.parameters, RADON_VI_COUNTY, RADON_VI_FLOOR)
        @test mu ≈ reference.mu
    end
end
