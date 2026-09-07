using ReactiveKernelsPPLExamples.RadonVariableInterceptSlopeNoncenteredExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

# Graph-independent oracle for the non-centered varying-intercept + varying-slope
# (independent effects) model.
function _radon_visn_reference(q, county_idx, floor_measure, log_radon)
    nlp(x, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - m) / s)^2
    J = div(length(q) - 5, 2)
    sigma_y = exp(q[1])
    sigma_alpha = exp(q[2])
    sigma_beta = exp(q[3])
    alpha_raw = q[4:J + 3]
    beta_raw = q[J + 4:2J + 3]
    mu_alpha = q[2J + 4]
    mu_beta = q[2J + 5]
    log_jacobian = q[1] + q[2] + q[3]
    prior = nlp(mu_alpha, 0.0, 10.0) + nlp(mu_beta, 0.0, 10.0) +
            nlp(sigma_y, 0.0, 1.0) + nlp(sigma_alpha, 0.0, 1.0) +
            nlp(sigma_beta, 0.0, 1.0) +
            sum(nlp(ar, 0.0, 1.0) for ar in alpha_raw) +
            sum(nlp(br, 0.0, 1.0) for br in beta_raw)
    alpha = mu_alpha .+ sigma_alpha .* alpha_raw
    beta = mu_beta .+ sigma_beta .* beta_raw
    mu = alpha[county_idx] .+ floor_measure .* beta[county_idx]
    likelihood = sum(nlp(y, m, sigma_y) for (y, m) in zip(log_radon, mu))
    (; parameters = (; alpha_raw, beta_raw, mu_alpha, mu_beta, sigma_alpha,
                     sigma_beta, sigma_y), prior, log_jacobian, likelihood,
       alpha, beta, mu, posterior = prior + likelihood + log_jacobian)
end

@testset "PPL graph — radon_variable_intercept_slope_noncentered (posteriordb)" begin
    artifact = evaluate_radon_variable_intercept_slope_noncentered_source()
    @test artifact.source ==
          strip(RADON_VARIABLE_INTERCEPT_SLOPE_NONCENTERED_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = vcat([log(0.65), log(0.55), log(0.5)],
             0.1 .* collect(1:8) .- 0.4,
             0.05 .* collect(1:8) .- 0.2,
             [0.5, -0.3])
    reference = _radon_visn_reference(q, RADON_VISN_COUNTY, RADON_VISN_FLOOR,
                                      RADON_VISN_LOG)

    @testset "authored on the current baseline surface" begin
        @test occursin("m + s * ar",
                       RADON_VARIABLE_INTERCEPT_SLOPE_NONCENTERED_SOURCE)
        @test occursin("m + s * br",
                       RADON_VARIABLE_INTERCEPT_SLOPE_NONCENTERED_SOURCE)
        @test occursin("alpha_county = alpha[county_idx]",
                       RADON_VARIABLE_INTERCEPT_SLOPE_NONCENTERED_SOURCE)
        @test occursin("beta_county = beta[county_idx]",
                       RADON_VARIABLE_INTERCEPT_SLOPE_NONCENTERED_SOURCE)
        @test occursin("normal(ac + f * bc, s).logpdf(y)",
                       RADON_VARIABLE_INTERCEPT_SLOPE_NONCENTERED_SOURCE)
        @test occursin("normal(0.0, 1.0).logpdf(ar)",
                       RADON_VARIABLE_INTERCEPT_SLOPE_NONCENTERED_SOURCE)
        @test occursin("normal(0.0, 1.0).logpdf(br)",
                       RADON_VARIABLE_INTERCEPT_SLOPE_NONCENTERED_SOURCE)
        @test occursin("bound = (; county_idx)",
                       RADON_VARIABLE_INTERCEPT_SLOPE_NONCENTERED_SOURCE)
        @test !occursin("struct ",
                        RADON_VARIABLE_INTERCEPT_SLOPE_NONCENTERED_SOURCE)
        @test artifact.normal_object === normal
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.county_idx,
                         model.floor_measure, model.log_radon),
                 want = (model.prior, model.log_jacobian, model.likelihood,
                         model.posterior))
        prior, log_jacobian, likelihood, posterior =
            prepare(p)(q, RADON_VISN_COUNTY, RADON_VISN_FLOOR, RADON_VISN_LOG)
        @test prior ≈ reference.prior
        @test log_jacobian ≈ reference.log_jacobian
        @test likelihood ≈ reference.likelihood
        @test posterior ≈ reference.posterior
    end

    @testset "non-centered transformed alpha/beta + gathered mean mu from constrained HAVEs" begin
        pab = plan(model.graph; have = (model.parameters,),
                   want = (model.alpha, model.beta))
        alpha, beta = prepare(pab)(reference.parameters)
        @test alpha ≈ reference.alpha
        @test beta ≈ reference.beta

        pm = plan(model.graph;
                  have = (model.parameters, model.county_idx, model.floor_measure),
                  want = (model.mu,))
        mu = prepare(pm)(reference.parameters, RADON_VISN_COUNTY, RADON_VISN_FLOOR)
        @test mu ≈ reference.mu
    end
end
