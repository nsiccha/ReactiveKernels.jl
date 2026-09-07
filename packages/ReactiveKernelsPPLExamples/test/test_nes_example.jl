using ReactiveKernelsPPLExamples.NESExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

# Graph-independent reference oracle for the posteriordb `nes` Gaussian
# regression y ~ Normal(X*beta, sigma): unconstrained `beta` (identity, no
# Jacobian), `sigma = exp(log_sigma)` with `lb_constrain` Jacobian log_sigma, a
# Normal likelihood over the matrix-vector linear predictor eta = X*beta, and NO
# prior term (the .stan has no `~` statements — flat improper priors, so
# log_prior is a dropped constant ≡ 0).
function _nes_reference(q, X, y)
    D = length(q) - 1
    beta = q[1:D]
    log_sigma = q[D + 1]
    sigma = exp(log_sigma)
    nlp(x, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - m) / s)^2
    eta = X * beta
    likelihood = sum(nlp(y[i], eta[i], sigma) for i in eachindex(y))
    log_prior = 0.0
    (; parameters = (; beta, sigma), log_prior, likelihood, eta,
       log_jacobian = log_sigma,
       posterior = log_prior + likelihood + log_sigma)
end

@testset "PPL graph — nes (posteriordb ARM Ch.4 party-id regression)" begin
    artifact = evaluate_nes_source()
    @test artifact.source == strip(NES_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = [0.3, -0.9, -0.4, 0.2, 0.5, 0.8, -0.15, -0.3, 0.1, log(2.0)]
    reference = _nes_reference(q, NES_X, NES_PARTYID7)

    @testset "authored on the current baseline surface" begin
        @test occursin("eta = predictors * beta", NES_SOURCE)
        @test occursin("normal(e, s).logpdf(y)", NES_SOURCE)
        @test occursin("pointwise = plate(", NES_SOURCE)
        @test !occursin("struct ", NES_SOURCE)
        # Flat improper priors — no proper prior term appears.
        @test occursin("log_prior::Float64 = 0.0", NES_SOURCE)
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
        @test collect(parameters.beta) ≈ collect(reference.parameters.beta)
        @test parameters.sigma ≈ reference.parameters.sigma
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.predictors, model.responses),
                 want = (model.log_prior, model.log_jacobian, model.pointwise,
                         model.likelihood, model.posterior))
        log_prior, log_jacobian, pointwise, likelihood, posterior =
            prepare(p)(q, NES_X, NES_PARTYID7)
        @test all(isfinite, pointwise)
        @test log_prior == 0.0
        @test log_jacobian ≈ reference.log_jacobian
        @test likelihood ≈ reference.likelihood
        @test likelihood ≈ sum(pointwise)
        @test posterior ≈ reference.posterior
    end

    @testset "linear predictor eta from a constrained HAVE" begin
        p = plan(model.graph; have = (model.parameters, model.predictors),
                 want = (model.eta,))
        @test prepare(p)(reference.parameters, NES_X) ≈ reference.eta
    end
end
