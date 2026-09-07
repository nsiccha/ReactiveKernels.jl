using ReactiveKernelsPPLExamples.DogsLogExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: bernoulli, uniform
using LogExpFunctions: logistic, log1pexp

# Graph-independent reference oracle for the posteriordb dogs_log model:
# unconstrained parameters (identity transform, log Jacobian zero) with EXPLICIT
# uniform priors beta1 ~ uniform(-100, 0), beta2 ~ uniform(0, 100) — each
# contributes -log(100) inside its support and -Inf outside (the model-block
# statement a gradient-only check cannot see) — a Bernoulli likelihood
# y ~ Bernoulli(inv_logit(beta1*n_avoid + beta2*n_shock)), and the shock
# probabilities p = inv_logit(eta).
function _dogs_log_reference(q, n_avoid, n_shock, y)
    beta1, beta2 = q[1], q[2]
    unif(x, lo, hi) = (lo <= x <= hi) ? -log(hi - lo) : -Inf
    log_prior = unif(beta1, -100.0, 0.0) + unif(beta2, 0.0, 100.0)
    eta = beta1 .* n_avoid .+ beta2 .* n_shock
    bern(v, e) = v ? -log1pexp(-e) : -log1pexp(e)
    pointwise = [bern(y[i], eta[i]) for i in eachindex(y)]
    likelihood = sum(pointwise)
    (; parameters = (; beta1, beta2), log_prior, log_jacobian = 0.0, eta,
       pointwise, likelihood, posterior = log_prior + likelihood,
       p = logistic.(eta))
end

@testset "PPL graph — dogs_log (posteriordb)" begin
    artifact = evaluate_dogs_log_source()
    @test artifact.source == strip(DOGS_LOG_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = [-0.2, 0.1]
    reference = _dogs_log_reference(q, DOGS_LOG_N_AVOID, DOGS_LOG_N_SHOCK,
                                    DOGS_LOG_Y_FLAT)

    @testset "authored on the current baseline surface" begin
        @test occursin("bernoulli(logistic(", DOGS_LOG_SOURCE)
        @test occursin(".logpdf(yi)", DOGS_LOG_SOURCE)
        @test occursin("uniform(-100.0, 0.0).logpdf(beta1)", DOGS_LOG_SOURCE)
        @test occursin("uniform(0.0, 100.0).logpdf(beta2)", DOGS_LOG_SOURCE)
        @test occursin("eta = plate(", DOGS_LOG_SOURCE)
        @test occursin("pointwise = plate(", DOGS_LOG_SOURCE)
        @test !occursin("struct ", DOGS_LOG_SOURCE)
        @test artifact.bernoulli_object === bernoulli
        @test artifact.uniform_object === uniform

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
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.n_avoid, model.n_shock, model.y),
                 want = (model.log_prior, model.pointwise, model.likelihood,
                         model.posterior))
        log_prior, pointwise, likelihood, posterior =
            prepare(p)(q, DOGS_LOG_N_AVOID, DOGS_LOG_N_SHOCK, DOGS_LOG_Y_FLAT)
        @test all(isfinite, pointwise)
        @test log_prior ≈ reference.log_prior          # includes the -2*log(100) constant
        @test likelihood ≈ reference.likelihood
        @test likelihood ≈ sum(pointwise)
        @test posterior ≈ reference.posterior
    end

    @testset "uniform-prior support restriction: beta outside its interval is -Inf" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.n_avoid, model.n_shock, model.y),
                 want = (model.log_prior, model.posterior))
        # beta1 = 0.5 is above the uniform(-100, 0) support upper bound.
        lp_hi, post_hi = prepare(p)([0.5, 0.1], DOGS_LOG_N_AVOID, DOGS_LOG_N_SHOCK,
                                    DOGS_LOG_Y_FLAT)
        @test lp_hi == -Inf
        @test post_hi == -Inf
        # beta2 = -0.1 is below the uniform(0, 100) support lower bound.
        lp_lo, post_lo = prepare(p)([-0.5, -0.1], DOGS_LOG_N_AVOID, DOGS_LOG_N_SHOCK,
                                    DOGS_LOG_Y_FLAT)
        @test lp_lo == -Inf
        @test post_lo == -Inf
    end

    @testset "generated quantity p from a constrained HAVE prunes the density" begin
        p = plan(model.graph;
                 have = (model.parameters, model.n_avoid, model.n_shock),
                 want = (model.p,))
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.likelihood.id) in produced)
        probs = prepare(p)(reference.parameters, DOGS_LOG_N_AVOID, DOGS_LOG_N_SHOCK)
        @test probs ≈ reference.p
    end

    @testset "one authored plate exposes a buffer-free total" begin
        pointwise_kernel = prepare(model;
            have = (:unconstrained, :n_avoid, :n_shock, :y), want = :pointwise)
        likelihood_kernel = prepare(model;
            have = (:unconstrained, :n_avoid, :n_shock, :y), want = :likelihood)
        pw = pointwise_kernel(q, DOGS_LOG_N_AVOID, DOGS_LOG_N_SHOCK, DOGS_LOG_Y_FLAT)
        @test likelihood_kernel(q, DOGS_LOG_N_AVOID, DOGS_LOG_N_SHOCK, DOGS_LOG_Y_FLAT) ≈
              sum(pw)
        @test occursin("similar", string(code_expr(pointwise_kernel)))
        @test !occursin("similar", string(code_expr(likelihood_kernel)))
    end
end
