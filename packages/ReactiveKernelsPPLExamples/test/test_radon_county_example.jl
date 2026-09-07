using ReactiveKernelsPPLExamples.RadonCountyExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal
using LogExpFunctions: logistic, log1pexp

# Graph-independent oracle for the centered per-county intercept (no floor) model.
function _radon_county_reference(q, county_idx, log_radon)
    nlp(x, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - m) / s)^2
    J = length(q) - 3
    a = q[1:J]
    mu_a = q[J + 1]
    u_sa = q[J + 2]
    u_sy = q[J + 3]
    sigma_a = 100.0 * logistic(u_sa)
    sigma_y = 100.0 * logistic(u_sy)
    jac_sa = log(100.0) - log1pexp(-u_sa) - log1pexp(u_sa)
    jac_sy = log(100.0) - log1pexp(-u_sy) - log1pexp(u_sy)
    log_jacobian = jac_sa + jac_sy
    prior = nlp(mu_a, 0.0, 1.0) + sum(nlp(aj, mu_a, sigma_a) for aj in a)
    mu = a[county_idx]
    likelihood = sum(nlp(y, m, sigma_y) for (y, m) in zip(log_radon, mu))
    (; parameters = (; a, mu_a, sigma_a, sigma_y), prior, log_jacobian,
       likelihood, mu, posterior = prior + likelihood + log_jacobian)
end

@testset "PPL graph — radon_county (posteriordb)" begin
    artifact = evaluate_radon_county_source()
    @test artifact.source == strip(RADON_COUNTY_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = vcat(0.1 .* collect(1:8), [0.5, -3.0, -3.2])
    reference = _radon_county_reference(q, RADON_COUNTY_IDX, RADON_COUNTY_LOG)

    @testset "authored on the current baseline surface" begin
        @test occursin("100.0 * logistic(u_sigma_a)", RADON_COUNTY_SOURCE)
        @test occursin("mu = a[county_idx]", RADON_COUNTY_SOURCE)
        @test occursin("normal(m, s).logpdf(aj)", RADON_COUNTY_SOURCE)
        @test occursin("bound = (; county_idx)", RADON_COUNTY_SOURCE)
        @test !occursin("struct ", RADON_COUNTY_SOURCE)
        @test artifact.normal_object === normal
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.county_idx, model.log_radon),
                 want = (model.prior, model.log_jacobian, model.likelihood,
                         model.posterior))
        prior, log_jacobian, likelihood, posterior =
            prepare(p)(q, RADON_COUNTY_IDX, RADON_COUNTY_LOG)
        @test prior ≈ reference.prior
        @test log_jacobian ≈ reference.log_jacobian
        @test likelihood ≈ reference.likelihood
        @test posterior ≈ reference.posterior
    end

    @testset "gathered mean mu from a constrained HAVE" begin
        p = plan(model.graph;
                 have = (model.parameters, model.county_idx),
                 want = (model.mu,))
        mu = prepare(p)(reference.parameters, RADON_COUNTY_IDX)
        @test mu ≈ reference.mu
    end
end
