using ReactiveKernelsPPLExamples.DogsExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, bernoulli
using LogExpFunctions: logistic, log1pexp

# Graph-independent reference oracle for the posteriordb dogs model: unconstrained
# parameters (identity transform, log Jacobian zero) with a PROPER
# beta ~ normal(0, 100) prior on each of the three components — a term a gradient
# check DOES see and value parity confirms carries no dropped constant — a
# Bernoulli-logit likelihood y ~ Bernoulli(inv_logit(beta1 + beta2*n_avoid +
# beta3*n_shock)), and the shock probabilities p = inv_logit(eta).
function _dogs_reference(q, n_avoid, n_shock, y)
    beta1, beta2, beta3 = q[1], q[2], q[3]
    normal_lp(x, mu, sig) = -0.5 * log(2π) - log(sig) - 0.5 * ((x - mu) / sig)^2
    log_prior = normal_lp(beta1, 0.0, 100.0) + normal_lp(beta2, 0.0, 100.0) +
                normal_lp(beta3, 0.0, 100.0)
    eta = beta1 .+ beta2 .* n_avoid .+ beta3 .* n_shock
    bern(v, e) = v ? -log1pexp(-e) : -log1pexp(e)
    pointwise = [bern(y[i], eta[i]) for i in eachindex(y)]
    likelihood = sum(pointwise)
    (; parameters = (; beta1, beta2, beta3), log_prior, log_jacobian = 0.0, eta,
       pointwise, likelihood, posterior = log_prior + likelihood,
       p = logistic.(eta))
end

@testset "PPL graph — dogs (posteriordb)" begin
    artifact = evaluate_dogs_source()
    @test artifact.source == strip(DOGS_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = [-0.2, 0.1, -0.05]
    reference = _dogs_reference(q, DOGS_N_AVOID, DOGS_N_SHOCK, DOGS_Y_FLAT)

    @testset "authored on the current baseline surface" begin
        @test occursin("bernoulli(logistic(", DOGS_SOURCE)
        @test occursin(".logpdf(yi)", DOGS_SOURCE)
        @test occursin("normal(0.0, 100.0).logpdf(beta1)", DOGS_SOURCE)
        @test occursin("normal(0.0, 100.0).logpdf(beta2)", DOGS_SOURCE)
        @test occursin("normal(0.0, 100.0).logpdf(beta3)", DOGS_SOURCE)
        @test occursin("eta = plate(", DOGS_SOURCE)
        @test occursin("pointwise = plate(", DOGS_SOURCE)
        @test !occursin("struct ", DOGS_SOURCE)
        @test artifact.normal_object === normal
        @test artifact.bernoulli_object === bernoulli

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
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.n_avoid, model.n_shock, model.y),
                 want = (model.log_prior, model.pointwise, model.likelihood,
                         model.posterior))
        log_prior, pointwise, likelihood, posterior =
            prepare(p)(q, DOGS_N_AVOID, DOGS_N_SHOCK, DOGS_Y_FLAT)
        @test all(isfinite, pointwise)
        @test log_prior ≈ reference.log_prior          # proper normal prior — no dropped constant
        @test likelihood ≈ reference.likelihood
        @test likelihood ≈ sum(pointwise)
        @test posterior ≈ reference.posterior
    end

    @testset "generated quantity p from a constrained HAVE prunes the density" begin
        p = plan(model.graph;
                 have = (model.parameters, model.n_avoid, model.n_shock),
                 want = (model.p,))
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.likelihood.id) in produced)
        probs = prepare(p)(reference.parameters, DOGS_N_AVOID, DOGS_N_SHOCK)
        @test probs ≈ reference.p
    end

    @testset "one authored plate exposes a buffer-free total" begin
        pointwise_kernel = prepare(model;
            have = (:unconstrained, :n_avoid, :n_shock, :y), want = :pointwise)
        likelihood_kernel = prepare(model;
            have = (:unconstrained, :n_avoid, :n_shock, :y), want = :likelihood)
        pw = pointwise_kernel(q, DOGS_N_AVOID, DOGS_N_SHOCK, DOGS_Y_FLAT)
        @test likelihood_kernel(q, DOGS_N_AVOID, DOGS_N_SHOCK, DOGS_Y_FLAT) ≈
              sum(pw)
        @test occursin("similar", string(code_expr(pointwise_kernel)))
        @test !occursin("similar", string(code_expr(likelihood_kernel)))
    end
end
