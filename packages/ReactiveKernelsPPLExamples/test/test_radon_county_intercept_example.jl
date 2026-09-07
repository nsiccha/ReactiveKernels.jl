using ReactiveKernelsPPLExamples.RadonCountyInterceptExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

# Graph-independent oracle for the fixed-prior varying-intercept + shared-slope model.
function _radon_county_intercept_reference(q, county_idx, floor_measure, log_radon)
    nlp(x, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - m) / s)^2
    J = length(q) - 2
    alpha = q[1:J]
    beta = q[J + 1]
    sigma_y = exp(q[J + 2])
    log_jacobian = q[J + 2]
    prior = nlp(beta, 0.0, 10.0) + nlp(sigma_y, 0.0, 1.0) +
            sum(nlp(a, 0.0, 10.0) for a in alpha)
    mu = alpha[county_idx] .+ beta .* floor_measure
    likelihood = sum(nlp(y, m, sigma_y) for (y, m) in zip(log_radon, mu))
    (; parameters = (; alpha, beta, sigma_y), prior, log_jacobian, likelihood, mu,
       posterior = prior + likelihood + log_jacobian)
end

@testset "PPL graph — radon_county_intercept (posteriordb)" begin
    artifact = evaluate_radon_county_intercept_source()
    @test artifact.source == strip(RADON_COUNTY_INTERCEPT_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = vcat(0.1 .* collect(1:8), [-0.5, log(0.7)])
    reference = _radon_county_intercept_reference(q, RADON_CI_COUNTY,
                                                  RADON_CI_FLOOR, RADON_CI_LOG)

    @testset "authored on the current baseline surface" begin
        @test occursin("normal(ac + b * f, s).logpdf(y)",
                       RADON_COUNTY_INTERCEPT_SOURCE)
        @test occursin("alpha_county = alpha[county_idx]",
                       RADON_COUNTY_INTERCEPT_SOURCE)
        @test occursin("normal(0.0, 10.0).logpdf(a)",
                       RADON_COUNTY_INTERCEPT_SOURCE)
        @test occursin("bound = (; county_idx)", RADON_COUNTY_INTERCEPT_SOURCE)
        @test !occursin("struct ", RADON_COUNTY_INTERCEPT_SOURCE)
        @test artifact.normal_object === normal
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.county_idx,
                         model.floor_measure, model.log_radon),
                 want = (model.prior, model.log_jacobian, model.likelihood,
                         model.posterior))
        prior, log_jacobian, likelihood, posterior =
            prepare(p)(q, RADON_CI_COUNTY, RADON_CI_FLOOR, RADON_CI_LOG)
        @test prior ≈ reference.prior
        @test log_jacobian ≈ reference.log_jacobian
        @test likelihood ≈ reference.likelihood
        @test posterior ≈ reference.posterior
    end

    @testset "gathered + combined mean mu from a constrained HAVE" begin
        p = plan(model.graph;
                 have = (model.parameters, model.county_idx, model.floor_measure),
                 want = (model.mu,))
        mu = prepare(p)(reference.parameters, RADON_CI_COUNTY, RADON_CI_FLOOR)
        @test mu ≈ reference.mu
    end
end
