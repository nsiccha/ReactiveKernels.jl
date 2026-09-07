using ReactiveKernelsPPLExamples.LowDimGaussMixCollapseExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, beta
using LogExpFunctions: log1pexp, logaddexp
using SpecialFunctions: logbeta

# Graph-independent reference oracle: the marginalized two-component normal
# mixture with FREE means/scales. Priors mu~N(0,2), sigma~N(0,2) (plain
# normal_lpdf, NO log2 truncation constant — matches Stan), theta~Beta(5,5).
function _c_reference(q, y)
    mu1, mu2, u_s1, u_s2, u_theta = q[1], q[2], q[3], q[4], q[5]
    sigma1 = exp(u_s1); sigma2 = exp(u_s2)
    theta = 1 / (1 + exp(-u_theta))
    log_theta = -log1pexp(-u_theta)
    log1m_theta = -log1pexp(u_theta)
    log_jacobian = u_s1 + u_s2 + log_theta + log1m_theta
    nld(x, loc, sc) = -0.5 * log(2π) - log(sc) - 0.5 * ((x - loc) / sc)^2
    prior = nld(mu1, 0.0, 2.0) + nld(mu2, 0.0, 2.0) +
            nld(sigma1, 0.0, 2.0) + nld(sigma2, 0.0, 2.0) +
            (4 * log(theta) + 4 * log1p(-theta) - logbeta(5.0, 5.0))
    likelihood = 0.0
    for yj in y
        likelihood += logaddexp(log_theta + nld(yj, mu1, sigma1),
                                log1m_theta + nld(yj, mu2, sigma2))
    end
    (; prior, log_jacobian, likelihood, posterior = prior + likelihood + log_jacobian)
end

@testset "PPL graph — low_dim_gauss_mix_collapse (posteriordb, free means)" begin
    artifact = evaluate_low_dim_gauss_mix_collapse_source()
    @test artifact.source == strip(LOW_DIM_GAUSS_MIX_COLLAPSE_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = [-1.2, 1.3, log(0.9), log(1.1), 0.2]
    reference = _c_reference(q, LOW_DIM_GAUSS_MIX_COLLAPSE_Y)

    @testset "authored on the current baseline surface" begin
        @test occursin("beta(5.0, 5.0).logpdf", LOW_DIM_GAUSS_MIX_COLLAPSE_SOURCE)
        @test occursin("logaddexp(", LOW_DIM_GAUSS_MIX_COLLAPSE_SOURCE)
        @test occursin("pointwise = plate(", LOW_DIM_GAUSS_MIX_COLLAPSE_SOURCE)
        @test !occursin("log_mix", LOW_DIM_GAUSS_MIX_COLLAPSE_SOURCE)
        @test !occursin("struct ", LOW_DIM_GAUSS_MIX_COLLAPSE_SOURCE)
        @test artifact.normal_object === normal
        @test artifact.beta_object === beta
    end

    @testset "constrain-only prunes the density work" begin
        p = plan(model.graph; have = (model.unconstrained,), want = (model.parameters,))
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.prior.id) in produced)
        parameters = prepare(p)(q)
        @test parameters isa NamedTuple
        @test parameters.sigma1 > 0.0 && parameters.sigma2 > 0.0
        @test 0.0 < parameters.theta < 1.0
    end

    @testset "marginalized posterior vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.y),
                 want = (model.prior, model.log_jacobian, model.pointwise,
                         model.likelihood, model.posterior))
        prior, log_jacobian, pointwise, likelihood, posterior =
            prepare(p)(q, LOW_DIM_GAUSS_MIX_COLLAPSE_Y)
        @test all(isfinite, pointwise)
        @test prior ≈ reference.prior
        @test likelihood ≈ reference.likelihood
        @test likelihood ≈ sum(pointwise)
        @test log_jacobian ≈ reference.log_jacobian
        @test posterior ≈ reference.posterior
    end

    @testset "one authored plate exposes a buffer-free total" begin
        pointwise_kernel = prepare(model;
            have = (:unconstrained, :y), want = :pointwise)
        likelihood_kernel = prepare(model;
            have = (:unconstrained, :y), want = :likelihood)
        pw = pointwise_kernel(q, LOW_DIM_GAUSS_MIX_COLLAPSE_Y)
        @test likelihood_kernel(q, LOW_DIM_GAUSS_MIX_COLLAPSE_Y) ≈ sum(pw)
        @test occursin("similar", string(code_expr(pointwise_kernel)))
        @test !occursin("similar", string(code_expr(likelihood_kernel)))
    end

    @testset "out-of-support θ is diagnosed as -Inf (Beta(5,5))" begin
        theta_prior_kernel = prepare(model; have = (:parameters,), want = :theta_prior)
        @test theta_prior_kernel(
            (; mu1 = -1.0, mu2 = 1.0, sigma1 = 1.0, sigma2 = 1.0, theta = 1.5)) == -Inf
    end
end
