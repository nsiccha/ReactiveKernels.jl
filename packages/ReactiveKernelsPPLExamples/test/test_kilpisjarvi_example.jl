using ReactiveKernelsPPLExamples.KilpisjarviExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

# Graph-independent reference oracle for posteriordb kilpisjarvi: a Gaussian
# linear regression with data-supplied adjustable Normal priors on the intercept
# and slope, an exp/log transform on σ (Jacobian log|dσ/du| = u), and a
# prediction ypred = α + β·xpred.
function _kilpisjarvi_reference(q, x, y, xpred, pmualpha, psalpha, pmubeta, psbeta)
    nlp(v, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((v - m) / s)^2
    alpha, beta, u_sigma = q[1], q[2], q[3]
    sigma = exp(u_sigma)
    log_jacobian = u_sigma
    log_prior = nlp(alpha, pmualpha, psalpha) + nlp(beta, pmubeta, psbeta)
    mu = alpha .+ beta .* x
    pointwise = [nlp(yi, m, sigma) for (yi, m) in zip(y, mu)]
    likelihood = sum(pointwise)
    (; parameters = (; alpha, beta, sigma), log_prior, log_jacobian, mu, pointwise,
       likelihood, posterior = log_prior + likelihood + log_jacobian,
       ypred = alpha + beta * xpred)
end

@testset "PPL graph — kilpisjarvi (posteriordb)" begin
    artifact = evaluate_kilpisjarvi_source()
    @test artifact.source == strip(KILPISJARVI_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = [9.3, 0.0, log(1.0)]
    reference = _kilpisjarvi_reference(q, KILPISJARVI_X, KILPISJARVI_Y,
                                       KILPISJARVI_XPRED, KILPISJARVI_PMUALPHA,
                                       KILPISJARVI_PSALPHA, KILPISJARVI_PMUBETA,
                                       KILPISJARVI_PSBETA)

    @testset "authored on the current baseline surface" begin
        @test occursin("normal(a + b * xi, s).logpdf(yi)", KILPISJARVI_SOURCE)
        @test occursin("normal(pmualpha, psalpha).logpdf", KILPISJARVI_SOURCE)
        @test occursin("normal(pmubeta, psbeta).logpdf", KILPISJARVI_SOURCE)
        @test occursin("mu = plate(", KILPISJARVI_SOURCE)
        @test !occursin("struct ", KILPISJARVI_SOURCE)
        @test artifact.normal_object === normal

        raw_generated = code_expr(artifact.kernel)
        readable = sprint(
            Base.show_unquoted,
            ReactiveKernels._readable_expr(raw_generated, artifact.kernel);
            context = :limit => false,
        )
        @test !occursin(r"__ops__\[\d+\]", readable)
        @test !occursin(r"\boperation\(", readable)
    end

    @testset "constrain-only prunes the likelihood work" begin
        p = plan(model.graph; have = (model.unconstrained,), want = (model.parameters,))
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.likelihood.id) in produced)
        parameters = prepare(p)(q)
        @test parameters isa NamedTuple
        @test parameters.alpha ≈ reference.parameters.alpha
        @test parameters.beta ≈ reference.parameters.beta
        @test parameters.sigma ≈ reference.parameters.sigma
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.x, model.y, model.xpred,
                         model.pmualpha, model.psalpha, model.pmubeta, model.psbeta),
                 want = (model.log_prior, model.log_jacobian, model.pointwise,
                         model.likelihood, model.posterior))
        log_prior, log_jacobian, pointwise, likelihood, posterior =
            prepare(p)(q, KILPISJARVI_X, KILPISJARVI_Y, KILPISJARVI_XPRED,
                       KILPISJARVI_PMUALPHA, KILPISJARVI_PSALPHA,
                       KILPISJARVI_PMUBETA, KILPISJARVI_PSBETA)
        @test all(isfinite, pointwise)
        @test log_prior ≈ reference.log_prior
        @test log_jacobian ≈ reference.log_jacobian
        @test likelihood ≈ reference.likelihood
        @test likelihood ≈ sum(pointwise)
        @test posterior ≈ reference.posterior
    end

    @testset "generated quantity ypred from a constrained HAVE prunes the density" begin
        p = plan(model.graph;
                 have = (model.parameters, model.xpred), want = (model.ypred,))
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.likelihood.id) in produced)
        ypred = prepare(p)(reference.parameters, KILPISJARVI_XPRED)
        @test ypred ≈ reference.ypred
    end

    @testset "one authored plate exposes a buffer-free total" begin
        pointwise_kernel = prepare(model;
            have = (:unconstrained, :x, :y, :xpred, :pmualpha, :psalpha,
                    :pmubeta, :psbeta), want = :pointwise)
        likelihood_kernel = prepare(model;
            have = (:unconstrained, :x, :y, :xpred, :pmualpha, :psalpha,
                    :pmubeta, :psbeta), want = :likelihood)
        pw = pointwise_kernel(q, KILPISJARVI_X, KILPISJARVI_Y, KILPISJARVI_XPRED,
                              KILPISJARVI_PMUALPHA, KILPISJARVI_PSALPHA,
                              KILPISJARVI_PMUBETA, KILPISJARVI_PSBETA)
        @test likelihood_kernel(q, KILPISJARVI_X, KILPISJARVI_Y, KILPISJARVI_XPRED,
                                KILPISJARVI_PMUALPHA, KILPISJARVI_PSALPHA,
                                KILPISJARVI_PMUBETA, KILPISJARVI_PSBETA) ≈ sum(pw)
        @test occursin("similar", string(code_expr(pointwise_kernel)))
        @test !occursin("similar", string(code_expr(likelihood_kernel)))
    end
end
