using ReactiveKernelsPPLExamples.RadonVariableSlopeNoncenteredExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

# Graph-independent oracle for the shared-intercept + non-centered varying-slope model.
function _radon_vsn_reference(q, county_idx, floor_measure, log_radon)
    nlp(x, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - m) / s)^2
    J = length(q) - 4
    alpha = q[1]
    beta_raw = q[2:J + 1]
    mu_beta = q[J + 2]
    sigma_beta = exp(q[J + 3])
    sigma_y = exp(q[J + 4])
    log_jacobian = q[J + 3] + q[J + 4]
    prior = nlp(alpha, 0.0, 10.0) + nlp(mu_beta, 0.0, 10.0) +
            nlp(sigma_beta, 0.0, 1.0) + nlp(sigma_y, 0.0, 1.0) +
            sum(nlp(br, 0.0, 1.0) for br in beta_raw)
    beta = mu_beta .+ sigma_beta .* beta_raw
    mu = alpha .+ floor_measure .* beta[county_idx]
    likelihood = sum(nlp(y, m, sigma_y) for (y, m) in zip(log_radon, mu))
    (; parameters = (; alpha, beta_raw, mu_beta, sigma_beta, sigma_y), prior,
       log_jacobian, likelihood, beta, mu,
       posterior = prior + likelihood + log_jacobian)
end

@testset "PPL graph — radon_variable_slope_noncentered (posteriordb)" begin
    artifact = evaluate_radon_variable_slope_noncentered_source()
    @test artifact.source == strip(RADON_VARIABLE_SLOPE_NONCENTERED_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = vcat([0.5], 0.1 .* collect(1:8) .- 0.4, [-0.5, log(0.6), log(0.7)])
    reference = _radon_vsn_reference(q, RADON_VSN_COUNTY, RADON_VSN_FLOOR,
                                     RADON_VSN_LOG)

    @testset "authored on the current baseline surface" begin
        @test occursin("m + s * br", RADON_VARIABLE_SLOPE_NONCENTERED_SOURCE)
        @test occursin("beta_county = beta[county_idx]",
                       RADON_VARIABLE_SLOPE_NONCENTERED_SOURCE)
        @test occursin("normal(a + f * bc, s).logpdf(y)",
                       RADON_VARIABLE_SLOPE_NONCENTERED_SOURCE)
        @test occursin("normal(0.0, 1.0).logpdf(br)",
                       RADON_VARIABLE_SLOPE_NONCENTERED_SOURCE)
        @test occursin("bound = (; county_idx)",
                       RADON_VARIABLE_SLOPE_NONCENTERED_SOURCE)
        @test !occursin("struct ", RADON_VARIABLE_SLOPE_NONCENTERED_SOURCE)
        @test artifact.normal_object === normal
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.county_idx,
                         model.floor_measure, model.log_radon),
                 want = (model.prior, model.log_jacobian, model.likelihood,
                         model.posterior))
        prior, log_jacobian, likelihood, posterior =
            prepare(p)(q, RADON_VSN_COUNTY, RADON_VSN_FLOOR, RADON_VSN_LOG)
        @test prior ≈ reference.prior
        @test log_jacobian ≈ reference.log_jacobian
        @test likelihood ≈ reference.likelihood
        @test posterior ≈ reference.posterior
    end

    @testset "non-centered transformed beta + gathered mean mu from constrained HAVEs" begin
        pb = plan(model.graph; have = (model.parameters,), want = (model.beta,))
        beta = prepare(pb)(reference.parameters)
        @test beta ≈ reference.beta

        pm = plan(model.graph;
                  have = (model.parameters, model.county_idx, model.floor_measure),
                  want = (model.mu,))
        mu = prepare(pm)(reference.parameters, RADON_VSN_COUNTY, RADON_VSN_FLOOR)
        @test mu ≈ reference.mu
    end
end
