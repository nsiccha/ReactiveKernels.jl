using ReactiveKernelsPPLExamples.KidscoreMomWorkExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

# Graph-independent reference oracle for the posteriordb
# kidiq_with_mom_work-kidscore_mom_work model: unconstrained `beta` (identity, no
# Jacobian), `sigma = exp(log_sigma)` with `lb_constrain` Jacobian log_sigma, a
# Normal likelihood with fitted mean μ = β₁ + β₂·work2 + β₃·work3 + β₄·work4, NO
# prior term (the .stan has no `~` statements — flat improper priors, so
# log_prior is a dropped constant ≡ 0), and the new-point predicted mean. The
# density is the fully-normalized standard Normal form (matching ReactiveKernels'
# `normal` endpoint).
function _kidscore_mom_work_reference(q, kid_score, work2, work3, work4,
                                      work2_new, work3_new, work4_new)
    beta1 = q[1]; beta2 = q[2]; beta3 = q[3]; beta4 = q[4]
    log_sigma = q[5]
    sigma = exp(log_sigma)
    log_jacobian = log_sigma
    linpred = beta1 .+ beta2 .* work2 .+ beta3 .* work3 .+ beta4 .* work4
    pointwise = [-0.5 * log(2π) - log(sigma) - 0.5 * ((y - m) / sigma)^2
                 for (y, m) in zip(kid_score, linpred)]
    likelihood = sum(pointwise)
    log_prior = 0.0
    (; parameters = (; beta1, beta2, beta3, beta4, sigma), log_jacobian, linpred,
       pointwise, likelihood, log_prior,
       posterior = log_prior + likelihood + log_jacobian,
       kid_score_pred = beta1 + beta2 * work2_new + beta3 * work3_new +
                        beta4 * work4_new)
end

@testset "PPL graph — kidscore_mom_work (posteriordb)" begin
    artifact = evaluate_kidscore_mom_work_source()
    @test artifact.source == strip(KIDSCORE_MOM_WORK_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = [85.0, 5.0, -5.0, 3.0, log(18.0)]
    reference = _kidscore_mom_work_reference(q, MOM_WORK_KID_SCORE,
        MOM_WORK_WORK2, MOM_WORK_WORK3, MOM_WORK_WORK4,
        MOM_WORK_WORK2_NEW, MOM_WORK_WORK3_NEW, MOM_WORK_WORK4_NEW)

    @testset "authored on the current baseline surface" begin
        @test occursin("normal(", KIDSCORE_MOM_WORK_SOURCE)
        @test occursin(".logpdf(score)", KIDSCORE_MOM_WORK_SOURCE)
        @test occursin("linpred = plate(", KIDSCORE_MOM_WORK_SOURCE)
        @test occursin("pointwise = plate(", KIDSCORE_MOM_WORK_SOURCE)
        @test !occursin("struct ", KIDSCORE_MOM_WORK_SOURCE)
        # This .stan has NO priors (flat) — unlike kidscore_momhs there is no
        # Cauchy on sigma; log_prior is identically the dropped constant 0.
        @test occursin("log_prior::Float64 = 0.0", KIDSCORE_MOM_WORK_SOURCE)
        @test !occursin("cauchy", KIDSCORE_MOM_WORK_SOURCE)
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
        @test parameters.beta1 ≈ reference.parameters.beta1
        @test parameters.beta2 ≈ reference.parameters.beta2
        @test parameters.beta4 ≈ reference.parameters.beta4
        @test parameters.sigma ≈ reference.parameters.sigma
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.kid_score, model.work2,
                         model.work3, model.work4),
                 want = (model.log_prior, model.log_jacobian, model.pointwise,
                         model.likelihood, model.posterior))
        log_prior, log_jacobian, pointwise, likelihood, posterior =
            prepare(p)(q, MOM_WORK_KID_SCORE, MOM_WORK_WORK2, MOM_WORK_WORK3,
                       MOM_WORK_WORK4)
        @test all(isfinite, pointwise)
        @test log_prior == 0.0
        @test log_prior ≈ reference.log_prior
        @test log_jacobian ≈ reference.log_jacobian
        @test likelihood ≈ reference.likelihood
        @test likelihood ≈ sum(pointwise)
        @test posterior ≈ reference.posterior
    end

    @testset "generated quantity prediction from a constrained HAVE prunes the density" begin
        p = plan(model.graph;
                 have = (model.parameters, model.work2_new, model.work3_new,
                         model.work4_new),
                 want = (model.kid_score_pred,))
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.likelihood.id) in produced)
        kid_score_pred = prepare(p)(reference.parameters, MOM_WORK_WORK2_NEW,
                                    MOM_WORK_WORK3_NEW, MOM_WORK_WORK4_NEW)
        @test kid_score_pred ≈ reference.kid_score_pred
    end

    @testset "one authored plate exposes a buffer-free total" begin
        pointwise_kernel = prepare(model;
            have = (:unconstrained, :kid_score, :work2, :work3, :work4),
            want = :pointwise)
        likelihood_kernel = prepare(model;
            have = (:unconstrained, :kid_score, :work2, :work3, :work4),
            want = :likelihood)
        pw = pointwise_kernel(q, MOM_WORK_KID_SCORE, MOM_WORK_WORK2,
                              MOM_WORK_WORK3, MOM_WORK_WORK4)
        @test likelihood_kernel(q, MOM_WORK_KID_SCORE, MOM_WORK_WORK2,
                                MOM_WORK_WORK3, MOM_WORK_WORK4) ≈ sum(pw)
        @test occursin("similar", string(code_expr(pointwise_kernel)))
        @test !occursin("similar", string(code_expr(likelihood_kernel)))
    end
end
