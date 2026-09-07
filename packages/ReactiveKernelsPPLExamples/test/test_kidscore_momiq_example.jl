using ReactiveKernelsPPLExamples.KidscoreMomiqExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, cauchy

# Graph-independent reference oracle for the posteriordb kidiq-kidscore_momiq
# model: unconstrained `beta` (identity, no Jacobian), `sigma = exp(log_sigma)`
# with `lb_constrain` Jacobian log_sigma, a Normal likelihood with fitted mean
# μ = β₁ + β₂·mom_iq, a half-Cauchy(0, 2.5) prior on sigma, and the new-point
# predicted mean β₁ + β₂·mom_iq_new. Densities are the fully-normalized standard
# forms (matching ReactiveKernels' `normal` and `cauchy` endpoints).
function _kidscore_momiq_reference(q, kid_score, mom_iq, mom_iq_new)
    beta1 = q[1]
    beta2 = q[2]
    log_sigma = q[3]
    sigma = exp(log_sigma)
    log_jacobian = log_sigma
    linpred = beta1 .+ beta2 .* mom_iq
    pointwise = [-0.5 * log(2π) - log(sigma) - 0.5 * ((y - m) / sigma)^2
                 for (y, m) in zip(kid_score, linpred)]
    likelihood = sum(pointwise)
    log_prior = -log(π) - log(2.5) - log1p((sigma / 2.5)^2)
    (; parameters = (; beta1, beta2, sigma), log_jacobian, linpred,
       pointwise, likelihood, log_prior,
       posterior = log_prior + likelihood + log_jacobian,
       kid_score_pred = beta1 + beta2 * mom_iq_new)
end

@testset "PPL graph — kidscore_momiq (posteriordb)" begin
    artifact = evaluate_kidscore_momiq_source()
    @test artifact.source == strip(KIDSCORE_MOMIQ_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = [25.0, 0.6, log(15.0)]
    reference =
        _kidscore_momiq_reference(q, MOMIQ_KID_SCORE, MOMIQ_MOM_IQ, MOMIQ_MOM_IQ_NEW)

    @testset "authored on the current baseline surface" begin
        @test occursin("normal(", KIDSCORE_MOMIQ_SOURCE)
        @test occursin(".logpdf(score)", KIDSCORE_MOMIQ_SOURCE)
        @test occursin("cauchy(0.0, 2.5)", KIDSCORE_MOMIQ_SOURCE)
        @test occursin("linpred = plate(", KIDSCORE_MOMIQ_SOURCE)
        @test occursin("pointwise = plate(", KIDSCORE_MOMIQ_SOURCE)
        @test !occursin("struct ", KIDSCORE_MOMIQ_SOURCE)
        @test artifact.normal_object === normal
        @test artifact.cauchy_object === cauchy

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
        @test parameters.beta1 ≈ reference.parameters.beta1
        @test parameters.beta2 ≈ reference.parameters.beta2
        @test parameters.sigma ≈ reference.parameters.sigma
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.kid_score, model.mom_iq),
                 want = (model.log_prior, model.log_jacobian, model.pointwise,
                         model.likelihood, model.posterior))
        log_prior, log_jacobian, pointwise, likelihood, posterior =
            prepare(p)(q, MOMIQ_KID_SCORE, MOMIQ_MOM_IQ)
        @test all(isfinite, pointwise)
        @test log_prior ≈ reference.log_prior
        @test log_jacobian ≈ reference.log_jacobian
        @test likelihood ≈ reference.likelihood
        @test likelihood ≈ sum(pointwise)
        @test posterior ≈ reference.posterior
    end

    @testset "generated quantity prediction from a constrained HAVE prunes the density" begin
        p = plan(model.graph;
                 have = (model.parameters, model.mom_iq_new),
                 want = (model.kid_score_pred,))
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.likelihood.id) in produced)
        kid_score_pred = prepare(p)(reference.parameters, MOMIQ_MOM_IQ_NEW)
        @test kid_score_pred ≈ reference.kid_score_pred
    end

    @testset "one authored plate exposes a buffer-free total" begin
        pointwise_kernel = prepare(model;
            have = (:unconstrained, :kid_score, :mom_iq), want = :pointwise)
        likelihood_kernel = prepare(model;
            have = (:unconstrained, :kid_score, :mom_iq), want = :likelihood)
        pw = pointwise_kernel(q, MOMIQ_KID_SCORE, MOMIQ_MOM_IQ)
        @test likelihood_kernel(q, MOMIQ_KID_SCORE, MOMIQ_MOM_IQ) ≈ sum(pw)
        @test occursin("similar", string(code_expr(pointwise_kernel)))
        @test !occursin("similar", string(code_expr(likelihood_kernel)))
    end
end
