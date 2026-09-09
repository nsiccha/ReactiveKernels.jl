using ReactiveKernelsPPLExamples.SesameOnePredAExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

# Graph-independent reference oracle for posteriordb sesame_one_pred_a: a Gaussian
# regression of the 0/1 `watched` indicator on the 0/1 `encouraged` assignment
# with flat priors on beta and an improper-flat sigma > 0 (exp transform,
# logJ = log_sigma).
function _sesame_one_pred_a_reference(q, encouraged, watched)
    nlp(v, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((v - m) / s)^2
    b1, b2, u = q[1], q[2], q[3]
    sigma = exp(u)
    log_jacobian = u
    mu = b1 .+ b2 .* encouraged
    likelihood = sum(nlp(w, m, sigma) for (w, m) in zip(watched, mu))
    (; parameters = (; beta1 = b1, beta2 = b2, sigma), log_jacobian, likelihood,
       mu, posterior = likelihood + log_jacobian)
end

@testset "PPL graph — sesame_one_pred_a (posteriordb)" begin
    artifact = evaluate_sesame_one_pred_a_source()
    @test artifact.source == strip(SESAME_ONE_PRED_A_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = [0.5, 0.1, log(0.4)]
    reference = _sesame_one_pred_a_reference(q, SESAME_ENCOURAGED, SESAME_WATCHED)

    @test artifact.normal_object === normal
    @test !occursin("struct ", SESAME_ONE_PRED_A_SOURCE)
    @test occursin("mu = plate(encouraged, beta1, beta2)", SESAME_ONE_PRED_A_SOURCE)
    @test occursin("b1 + b2 * e", SESAME_ONE_PRED_A_SOURCE)
        @test occursin("normal(m, s).logpdf(w)", SESAME_ONE_PRED_A_SOURCE)
    @test occursin("mu = plate(", SESAME_ONE_PRED_A_SOURCE)

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
        p = prepare(model; have = (:unconstrained, :encouraged, :watched),
                    want = (:log_jacobian, :likelihood, :posterior))
        log_jacobian, likelihood, posterior =
            p(q, SESAME_ENCOURAGED, SESAME_WATCHED)
        @test log_jacobian ≈ reference.log_jacobian
        @test likelihood ≈ reference.likelihood
        @test posterior ≈ reference.posterior
    end

    @testset "one authored plate exposes a buffer-free total" begin
        likelihood_kernel = prepare(model;
            have = (:unconstrained, :encouraged, :watched), want = :likelihood)
        @test !occursin("similar", string(code_expr(likelihood_kernel)))
    end
end
