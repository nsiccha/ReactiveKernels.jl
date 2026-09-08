using ReactiveKernelsPPLExamples.GLMM1ModelExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, poisson
using LogExpFunctions: logistic, log1pexp
using SpecialFunctions: loggamma

# Graph-independent oracle for the per-site Poisson-log GLMM. obssite gathers the
# site effect; the year index provably drops out of rep_matrix(alpha', nyear).
function _glmm1_reference(q, obs, obssite)
    nsite = length(q) - 2
    tf(u, L, U) = L + (U - L) * logistic(u)
    jc(u, L, U) = log(U - L) - log1pexp(-u) - log1pexp(u)
    nlp(x, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - m) / s)^2
    alpha = q[1:nsite]
    mu_alpha = q[nsite + 1]
    sd_alpha = tf(q[nsite + 2], 0.0, 5.0)
    log_jacobian = jc(q[nsite + 2], 0.0, 5.0)
    prior = sum(nlp(a, mu_alpha, sd_alpha) for a in alpha) + nlp(mu_alpha, 0.0, 10.0)
    likelihood = sum(c * alpha[s] - exp(alpha[s]) - loggamma(c + 1.0)
                     for (c, s) in zip(obs, obssite))
    (; parameters = (; alpha, mu_alpha, sd_alpha), prior, log_jacobian, likelihood,
       lambda_site = exp.(alpha), posterior = prior + likelihood + log_jacobian)
end

@testset "PPL graph — GLMM1 (posteriordb)" begin
    artifact = evaluate_glmm1_model_source()
    @test artifact.source == strip(GLMM1_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    nsite = GLMM1_NSITE
    q = vcat(0.2 .* sin.(1:nsite), [0.3, 0.4])
    reference = _glmm1_reference(q, GLMM1_OBS, GLMM1_OBSSITE)

    @testset "authored on the current baseline surface" begin
        @test occursin("poisson(; log_rate = lr)", GLMM1_SOURCE)
        @test occursin("normal(m, s).logpdf(a)", GLMM1_SOURCE)
        @test occursin("alpha[obssite]", GLMM1_SOURCE)
        @test !occursin("struct ", GLMM1_SOURCE)
        @test artifact.poisson_object === poisson
        @test artifact.normal_object === normal
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.obs, model.obssite),
                 want = (model.prior, model.log_jacobian, model.likelihood,
                         model.posterior))
        prior, log_jacobian, likelihood, posterior =
            prepare(p)(q, GLMM1_OBS, GLMM1_OBSSITE)
        @test prior ≈ reference.prior
        @test log_jacobian ≈ reference.log_jacobian
        @test likelihood ≈ reference.likelihood
        @test posterior ≈ reference.posterior
    end

    @testset "build+prepare+execute in one ordinary function, repeat-use stable" begin
        function once(q)
            graph = build_glmm1_model_graph()
            k = prepare(graph; have = (:unconstrained, :obs, :obssite),
                        want = :posterior, bound = (; obs = GLMM1_OBS, obssite = GLMM1_OBSSITE))
            k(q)
        end
        v1 = once(q)
        @test v1 ≈ reference.posterior
        k = prepare(build_glmm1_model_graph();
            have = (:unconstrained, :obs, :obssite), want = :posterior,
            bound = (; obs = GLMM1_OBS, obssite = GLMM1_OBSSITE))
        @test k(q) == k(q)          # repeat-use identical
        @test k(q) ≈ reference.posterior
    end

    @testset "data-generic: alternate small valid dimensions" begin
        # A synthetic 3-site / 4-observation dataset proves the graph adapts to
        # the bound data rather than hard-coding the real dimensions.
        obs = [0, 3, 1, 2]; obssite = [1, 2, 3, 2]
        qs = [0.1, -0.2, 0.5, 0.0, 0.3]     # alpha[1..3], mu, u_sd
        k = prepare(build_glmm1_model_graph();
            have = (:unconstrained, :obs, :obssite), want = :posterior,
            bound = (; obs, obssite))
        @test k(qs) ≈ _glmm1_reference(qs, obs, obssite).posterior
    end

    @testset "generated quantity lambda_site from a constrained HAVE" begin
        p = plan(model.graph; have = (model.parameters,), want = (model.lambda_site,))
        lambda = prepare(p)(reference.parameters)
        @test lambda ≈ reference.lambda_site
    end

    @testset "one authored plate exposes a buffer-free total" begin
        ll_kernel = prepare(model;
            have = (:unconstrained, :obs, :obssite), want = :likelihood,
            bound = (; obs = GLMM1_OBS, obssite = GLMM1_OBSSITE))
        @test !occursin("similar", string(code_expr(ll_kernel)))
    end
end
