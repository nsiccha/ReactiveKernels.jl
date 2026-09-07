using ReactiveKernelsPPLExamples.RatsModelExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

# Graph-independent oracle for posteriordb rats_model: a hierarchical linear
# growth curve with per-rat intercept alpha_j / slope beta_j, population priors
# mu_alpha, mu_beta ~ Normal(0, 100), three FLAT scales (exp transform,
# logJ = sum of the three log-scales, NO density term), the hierarchical priors
# alpha_j ~ Normal(mu_alpha, sigma_alpha), beta_j ~ Normal(mu_beta, sigma_beta),
# and the gathered mean alpha[rat] + beta[rat]*(x - xbar).
function _rats_model_reference(q, rat, x, y, xbar)
    nlp(v, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((v - m) / s)^2
    N = div(length(q) - 5, 2)
    alpha = q[1:N]
    beta = q[N + 1:2N]
    mu_alpha = q[2N + 1]
    mu_beta = q[2N + 2]
    sigma_y = exp(q[2N + 3])
    sigma_alpha = exp(q[2N + 4])
    sigma_beta = exp(q[2N + 5])
    log_jacobian = q[2N + 3] + q[2N + 4] + q[2N + 5]
    fixed_prior = nlp(mu_alpha, 0.0, 100.0) + nlp(mu_beta, 0.0, 100.0)
    alpha_prior = sum(nlp(a, mu_alpha, sigma_alpha) for a in alpha)
    beta_prior = sum(nlp(b, mu_beta, sigma_beta) for b in beta)
    prior = fixed_prior + alpha_prior + beta_prior
    mu = alpha[rat] .+ beta[rat] .* (x .- xbar)
    likelihood = sum(nlp(yy, m, sigma_y) for (yy, m) in zip(y, mu))
    (; parameters = (; alpha, beta, mu_alpha, mu_beta, sigma_y, sigma_alpha,
                      sigma_beta),
       prior, log_jacobian, likelihood, mu,
       alpha0 = mu_alpha - xbar * mu_beta,
       posterior = prior + likelihood + log_jacobian)
end

@testset "PPL graph — rats_model (posteriordb)" begin
    artifact = evaluate_rats_model_source()
    @test artifact.source == strip(RATS_MODEL_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = vcat(240.0 .+ collect(1.0:8.0), 6.0 .+ 0.1 .* collect(1.0:8.0),
             [240.0, 6.0, log(6.0), log(10.0), log(0.5)])
    reference = _rats_model_reference(q, RATS_RAT, RATS_X, RATS_Y, RATS_XBAR)

    @testset "authored on the current baseline surface" begin
        @test occursin("normal(a + b * xc, s).logpdf(yy)", RATS_MODEL_SOURCE)
        @test occursin("x_centered = x .- xbar", RATS_MODEL_SOURCE)
        @test occursin("alpha_rat = alpha[rat]", RATS_MODEL_SOURCE)
        @test occursin("beta_rat = beta[rat]", RATS_MODEL_SOURCE)
        @test occursin("normal(m, s).logpdf(a)", RATS_MODEL_SOURCE)
        @test occursin("normal(m, s).logpdf(b)", RATS_MODEL_SOURCE)
        @test occursin("bound = (; rat)", RATS_MODEL_SOURCE)
        @test !occursin("struct ", RATS_MODEL_SOURCE)
        @test artifact.normal_object === normal
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.rat, model.x, model.y,
                         model.xbar),
                 want = (model.prior, model.log_jacobian, model.likelihood,
                         model.posterior))
        prior, log_jacobian, likelihood, posterior =
            prepare(p)(q, RATS_RAT, RATS_X, RATS_Y, RATS_XBAR)
        @test prior ≈ reference.prior
        @test log_jacobian ≈ reference.log_jacobian
        @test likelihood ≈ reference.likelihood
        @test posterior ≈ reference.posterior
    end

    @testset "gathered + combined mean mu from a constrained HAVE" begin
        p = plan(model.graph;
                 have = (model.parameters, model.rat, model.x, model.xbar),
                 want = (model.mu,))
        mu = prepare(p)(reference.parameters, RATS_RAT, RATS_X, RATS_XBAR)
        @test mu ≈ reference.mu
    end

    @testset "generated quantity alpha0 from a constrained HAVE" begin
        p = plan(model.graph;
                 have = (model.parameters, model.xbar), want = (model.alpha0,))
        alpha0 = prepare(p)(reference.parameters, RATS_XBAR)
        @test alpha0 ≈ reference.alpha0
    end
end
