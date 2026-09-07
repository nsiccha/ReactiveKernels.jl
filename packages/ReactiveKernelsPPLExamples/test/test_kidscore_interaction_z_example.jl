using ReactiveKernelsPPLExamples.KidscoreInteractionZExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

# Graph-independent reference oracle for the posteriordb kidiq-kidscore_interaction_z
# model: unconstrained `beta` (identity, no Jacobian), `sigma = exp(log_sigma)`
# with `lb_constrain` Jacobian log_sigma, a Normal likelihood whose fitted mean
# μ = β₁ + β₂·z_mom_hs + β₃·z_mom_iq + β₄·(z_mom_hs·z_mom_iq) uses the STANDARDIZED
# predictors z = (x - mean(x)) / (2·sd(x)) with SAMPLE sd (divisor N-1; Stan's
# transformed data), flat improper priors (log_prior = 0), and the new-point
# predicted mean (standardized by the SAME training-data mean and sd). The mean
# and sample sd are computed over the observed rows, exactly matching Stan's
# `mean()` / `sd()` on the same rows. Densities are the fully-normalized standard
# forms (matching ReactiveKernels' `normal` endpoint).
function _sample_sd(x)
    m = sum(x) / length(x)
    sqrt(sum((xi - m) * (xi - m) for xi in x) / (length(x) - 1))
end
function _kidscore_interaction_z_reference(q, kid_score, mom_hs, mom_iq,
                                           mom_hs_new, mom_iq_new)
    beta1 = q[1]
    beta2 = q[2]
    beta3 = q[3]
    beta4 = q[4]
    log_sigma = q[5]
    sigma = exp(log_sigma)
    log_jacobian = log_sigma
    mean_hs = sum(mom_hs) / length(mom_hs)
    mean_iq = sum(mom_iq) / length(mom_iq)
    sd_hs = _sample_sd(mom_hs)
    sd_iq = _sample_sd(mom_iq)
    z_mom_hs = (mom_hs .- mean_hs) ./ (2.0 * sd_hs)
    z_mom_iq = (mom_iq .- mean_iq) ./ (2.0 * sd_iq)
    inter = z_mom_hs .* z_mom_iq
    linpred = beta1 .+ beta2 .* z_mom_hs .+ beta3 .* z_mom_iq .+ beta4 .* inter
    pointwise = [-0.5 * log(2π) - log(sigma) - 0.5 * ((y - m) / sigma)^2
                 for (y, m) in zip(kid_score, linpred)]
    likelihood = sum(pointwise)
    log_prior = 0.0
    z_hs_new = (mom_hs_new - mean_hs) / (2.0 * sd_hs)
    z_iq_new = (mom_iq_new - mean_iq) / (2.0 * sd_iq)
    (; parameters = (; beta1, beta2, beta3, beta4, sigma), log_jacobian, linpred,
       pointwise, likelihood, log_prior,
       posterior = log_prior + likelihood + log_jacobian,
       kid_score_pred = beta1 + beta2 * z_hs_new + beta3 * z_iq_new +
                        beta4 * (z_hs_new * z_iq_new))
end

@testset "PPL graph — kidscore_interaction_z (posteriordb)" begin
    artifact = evaluate_kidscore_interaction_z_source()
    @test artifact.source == strip(KIDSCORE_INTERACTION_Z_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = [25.0, -5.0, 0.5, 0.05, log(15.0)]
    reference = _kidscore_interaction_z_reference(q, INTERACTION_Z_KID_SCORE,
        INTERACTION_Z_MOM_HS, INTERACTION_Z_MOM_IQ,
        INTERACTION_Z_MOM_HS_NEW, INTERACTION_Z_MOM_IQ_NEW)

    @testset "authored on the current baseline surface" begin
        @test occursin("normal(", KIDSCORE_INTERACTION_Z_SOURCE)
        @test occursin(".logpdf(score)", KIDSCORE_INTERACTION_Z_SOURCE)
        @test occursin("log_prior::Float64 = 0.0", KIDSCORE_INTERACTION_Z_SOURCE)
        @test occursin("sum(mom_hs) / length(mom_hs)", KIDSCORE_INTERACTION_Z_SOURCE)
        @test occursin("sqrt(sum(sq_dev_hs) / (length(mom_hs) - 1))",
                       KIDSCORE_INTERACTION_Z_SOURCE)
        @test occursin("z_mom_hs = plate(", KIDSCORE_INTERACTION_Z_SOURCE)
        @test occursin("inter = plate(", KIDSCORE_INTERACTION_Z_SOURCE)
        @test occursin("linpred = plate(", KIDSCORE_INTERACTION_Z_SOURCE)
        @test occursin("pointwise = plate(", KIDSCORE_INTERACTION_Z_SOURCE)
        @test occursin("2.0 * shs", KIDSCORE_INTERACTION_Z_SOURCE)
        @test !occursin("cauchy", KIDSCORE_INTERACTION_Z_SOURCE)
        @test !occursin("struct ", KIDSCORE_INTERACTION_Z_SOURCE)
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
        @test parameters.beta3 ≈ reference.parameters.beta3
        @test parameters.beta4 ≈ reference.parameters.beta4
        @test parameters.sigma ≈ reference.parameters.sigma
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.kid_score, model.mom_hs,
                         model.mom_iq),
                 want = (model.log_prior, model.log_jacobian, model.pointwise,
                         model.likelihood, model.posterior))
        log_prior, log_jacobian, pointwise, likelihood, posterior =
            prepare(p)(q, INTERACTION_Z_KID_SCORE, INTERACTION_Z_MOM_HS,
                       INTERACTION_Z_MOM_IQ)
        @test all(isfinite, pointwise)
        @test log_prior ≈ reference.log_prior
        @test log_jacobian ≈ reference.log_jacobian
        @test likelihood ≈ reference.likelihood
        @test likelihood ≈ sum(pointwise)
        @test posterior ≈ reference.posterior
    end

    @testset "generated quantity prediction from a constrained HAVE prunes the density" begin
        p = plan(model.graph;
                 have = (model.parameters, model.mom_hs, model.mom_iq,
                         model.mom_hs_new, model.mom_iq_new),
                 want = (model.kid_score_pred,))
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.likelihood.id) in produced)
        kid_score_pred = prepare(p)(reference.parameters,
            INTERACTION_Z_MOM_HS, INTERACTION_Z_MOM_IQ,
            INTERACTION_Z_MOM_HS_NEW, INTERACTION_Z_MOM_IQ_NEW)
        @test kid_score_pred ≈ reference.kid_score_pred
    end

    @testset "the pointwise likelihood plate fuses into the total" begin
        pointwise_kernel = prepare(model;
            have = (:unconstrained, :kid_score, :mom_hs, :mom_iq), want = :pointwise)
        likelihood_kernel = prepare(model;
            have = (:unconstrained, :kid_score, :mom_hs, :mom_iq), want = :likelihood)
        pw = pointwise_kernel(q, INTERACTION_Z_KID_SCORE, INTERACTION_Z_MOM_HS,
                              INTERACTION_Z_MOM_IQ)
        @test likelihood_kernel(q, INTERACTION_Z_KID_SCORE, INTERACTION_Z_MOM_HS,
                                INTERACTION_Z_MOM_IQ) ≈ sum(pw)
        @test occursin("similar", string(code_expr(pointwise_kernel)))
        # Unlike the `_c` / `_c2` variants, the `_z` likelihood is NOT fully
        # buffer-free: the Gelman 2-sd standardization needs the SAMPLE sd, a
        # separate reduction pass over the squared deviations (sum(sq_dev_*))
        # whose intermediate vector materializes. The terminal pointwise reduction
        # still fuses (likelihood ≈ sum(pw) above); the residual buffer is the sd
        # pre-reduction, an inherent two-pass property of the standardized model.
        @test occursin("similar", string(code_expr(likelihood_kernel)))
    end
end
