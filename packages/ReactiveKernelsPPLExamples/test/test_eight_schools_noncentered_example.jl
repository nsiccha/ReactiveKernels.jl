using ReactiveKernelsPPLExamples.EightSchoolsNoncenteredExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, cauchy

# Graph-independent oracle for the non-centered 8-schools posterior.
function _es_nc_reference(q, y, sigma)
    J = length(q) - 2
    tt = q[1:J]
    mu = q[J + 1]
    log_tau = q[J + 2]
    tau = exp(log_tau)
    nlp(x, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - m) / s)^2
    clp(x, loc, sc) = -log(pi) - log(sc) - log1p(((x - loc) / sc)^2)
    prior = sum(nlp(t, 0, 1) for t in tt) + nlp(mu, 0, 5) + clp(tau, 0, 5)
    theta = tt .* tau .+ mu
    likelihood = sum(nlp(y[j], theta[j], sigma[j]) for j in 1:J)
    (; parameters = (; theta_trans = tt, mu, tau), prior, likelihood, theta,
       log_jacobian = log_tau, posterior = prior + likelihood + log_tau)
end

@testset "PPL graph — eight_schools_noncentered (posteriordb)" begin
    artifact = evaluate_eight_schools_noncentered_source()
    @test artifact.source == strip(EIGHT_SCHOOLS_NONCENTERED_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = [0.1, -0.2, 0.3, 0.0, 0.15, -0.1, 0.2, 0.05, 1.0, log(4.0)]
    reference = _es_nc_reference(q, ES_NC_Y, ES_NC_SIGMA)

    @testset "authored on the current baseline surface" begin
        @test occursin("tt * t + m", EIGHT_SCHOOLS_NONCENTERED_SOURCE)
        @test occursin("cauchy(0.0, 5.0).logpdf(tau)", EIGHT_SCHOOLS_NONCENTERED_SOURCE)
        @test occursin("theta = plate(", EIGHT_SCHOOLS_NONCENTERED_SOURCE)
        @test !occursin("struct ", EIGHT_SCHOOLS_NONCENTERED_SOURCE)
        @test artifact.cauchy_object === cauchy
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.observations, model.observation_scales),
                 want = (model.prior, model.log_jacobian, model.likelihood,
                         model.posterior))
        prior, log_jacobian, likelihood, posterior =
            prepare(p)(q, ES_NC_Y, ES_NC_SIGMA)
        @test prior ≈ reference.prior
        @test log_jacobian ≈ reference.log_jacobian
        @test likelihood ≈ reference.likelihood
        @test posterior ≈ reference.posterior
    end

    @testset "transformed parameter theta from a constrained HAVE" begin
        p = plan(model.graph; have = (model.parameters,), want = (model.theta,))
        theta = prepare(p)(reference.parameters)
        @test theta ≈ reference.theta
    end

    @testset "one authored plate exposes a buffer-free total" begin
        pw_kernel = prepare(model;
            have = (:unconstrained, :observations, :observation_scales), want = :pointwise)
        ll_kernel = prepare(model;
            have = (:unconstrained, :observations, :observation_scales), want = :likelihood)
        pw = pw_kernel(q, ES_NC_Y, ES_NC_SIGMA)
        @test ll_kernel(q, ES_NC_Y, ES_NC_SIGMA) ≈ sum(pw)
        @test !occursin("similar", string(code_expr(ll_kernel)))
    end
end
