using ReactiveKernelsPPLExamples.DogsHierarchicalExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: bernoulli
using LogExpFunctions: logistic, log1pexp

# Graph-independent reference oracle for the posteriordb dogs_hierarchical model:
# a, b ∈ [0,1] via the scaled-logit (width-1 lub) transform with its exact
# change-of-variables Jacobian, an IMPLICIT uniform prior over [0,1]² (density 1,
# so log_prior is exactly 0 — no dropped constant), the multiplicative shock
# probabilities p = a^prev_shock · b^prev_avoid, and a Bernoulli(p) likelihood.
function _dogs_hier_reference(q, prev_avoid, prev_shock, y)
    u_a, u_b = q[1], q[2]
    a = logistic(u_a)
    b = logistic(u_b)
    jac = (-log1pexp(-u_a) - log1pexp(u_a)) + (-log1pexp(-u_b) - log1pexp(u_b))
    la = log(a); lb = log(b)
    p = [exp(prev_shock[i] * la + prev_avoid[i] * lb) for i in eachindex(y)]
    bern(v, pr) = v ? log(pr) : log1p(-pr)
    pointwise = [bern(y[i], p[i]) for i in eachindex(y)]
    likelihood = sum(pointwise)
    (; parameters = (; a, b), log_prior = 0.0, log_jacobian = jac, p,
       pointwise, likelihood, posterior = likelihood + jac)
end

@testset "PPL graph — dogs_hierarchical (posteriordb)" begin
    artifact = evaluate_dogs_hierarchical_source()
    @test artifact.source == strip(DOGS_HIER_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = [0.3, -0.4]
    reference = _dogs_hier_reference(q, DOGS_HIER_PREV_AVOID, DOGS_HIER_PREV_SHOCK,
                                     DOGS_HIER_Y_FLAT)

    @testset "authored on the current baseline surface" begin
        @test occursin("bernoulli(exp(", DOGS_HIER_SOURCE)
        @test occursin(".logpdf(yi)", DOGS_HIER_SOURCE)
        @test occursin("a::Float64 = logistic(u_a)", DOGS_HIER_SOURCE)
        @test occursin("b::Float64 = logistic(u_b)", DOGS_HIER_SOURCE)
        @test occursin("log_prior::Float64 = 0.0", DOGS_HIER_SOURCE)
        @test occursin("p = plate(", DOGS_HIER_SOURCE)
        @test occursin("pointwise = plate(", DOGS_HIER_SOURCE)
        @test !occursin("struct ", DOGS_HIER_SOURCE)
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
        @test parameters.a ≈ reference.parameters.a
        @test parameters.b ≈ reference.parameters.b
        @test 0 < parameters.a < 1
        @test 0 < parameters.b < 1
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.prev_avoid, model.prev_shock, model.y),
                 want = (model.log_jacobian, model.pointwise, model.likelihood,
                         model.posterior))
        log_jacobian, pointwise, likelihood, posterior =
            prepare(p)(q, DOGS_HIER_PREV_AVOID, DOGS_HIER_PREV_SHOCK, DOGS_HIER_Y_FLAT)
        @test all(isfinite, pointwise)
        @test log_jacobian ≈ reference.log_jacobian
        @test likelihood ≈ reference.likelihood
        @test likelihood ≈ sum(pointwise)
        @test posterior ≈ reference.posterior
    end

    @testset "no hard support boundary: extreme q stays finite in (0,1)" begin
        # The logistic transform's range equals the declared/prior support (0,1),
        # so there is no -Inf boundary — every real q is interior.
        p = plan(model.graph;
                 have = (model.unconstrained, model.prev_avoid, model.prev_shock, model.y),
                 want = (model.parameters, model.posterior))
        for qv in ([9.0, -9.0], [-9.0, 9.0])
            params, posterior = prepare(p)(qv, DOGS_HIER_PREV_AVOID,
                                           DOGS_HIER_PREV_SHOCK, DOGS_HIER_Y_FLAT)
            @test 0 < params.a < 1
            @test 0 < params.b < 1
            @test isfinite(posterior)
        end
    end

    @testset "generated quantity p from a constrained HAVE prunes the density" begin
        p = plan(model.graph;
                 have = (model.parameters, model.prev_avoid, model.prev_shock),
                 want = (model.p,))
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.likelihood.id) in produced)
        probs = prepare(p)(reference.parameters, DOGS_HIER_PREV_AVOID, DOGS_HIER_PREV_SHOCK)
        @test probs ≈ reference.p
    end

    @testset "one authored plate exposes a buffer-free total" begin
        pointwise_kernel = prepare(model;
            have = (:unconstrained, :prev_avoid, :prev_shock, :y), want = :pointwise)
        likelihood_kernel = prepare(model;
            have = (:unconstrained, :prev_avoid, :prev_shock, :y), want = :likelihood)
        pw = pointwise_kernel(q, DOGS_HIER_PREV_AVOID, DOGS_HIER_PREV_SHOCK, DOGS_HIER_Y_FLAT)
        @test likelihood_kernel(q, DOGS_HIER_PREV_AVOID, DOGS_HIER_PREV_SHOCK,
                                DOGS_HIER_Y_FLAT) ≈ sum(pw)
        @test occursin("similar", string(code_expr(pointwise_kernel)))
        @test !occursin("similar", string(code_expr(likelihood_kernel)))
    end
end
