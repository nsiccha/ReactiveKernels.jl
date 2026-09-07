using ReactiveKernelsPPLExamples.PilotsExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal
using LogExpFunctions: logistic, log1pexp

# Graph-independent reference oracle for posteriordb pilots: two-way crossed
# random-effects Gaussian model. mu_a/mu_b ~ Normal(0,1) (identity transform);
# the three sigmas ∈ [0,100] use scaled-logit interval transforms and carry an
# implicit-uniform (flat, zero) prior; a_j ~ Normal(10*mu_a, sigma_a),
# b_k ~ Normal(10*mu_b, sigma_b); y_hat = a[group] + b[scenario].
function _pilots_reference(q, group_id, scenario_id, y)
    nlp(x, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - m) / s)^2
    jc(u) = log(100.0) - log1pexp(-u) - log1pexp(u)
    ng = 5; ns = 8
    a = q[1:ng]
    b = q[ng + 1:ng + ns]
    mu_a = q[ng + ns + 1]
    mu_b = q[ng + ns + 2]
    u_sa = q[ng + ns + 3]; u_sb = q[ng + ns + 4]; u_sy = q[ng + ns + 5]
    sigma_a = 100.0 * logistic(u_sa)
    sigma_b = 100.0 * logistic(u_sb)
    sigma_y = 100.0 * logistic(u_sy)
    log_jacobian = jc(u_sa) + jc(u_sb) + jc(u_sy)
    prior = nlp(mu_a, 0, 1) + nlp(mu_b, 0, 1) +
            sum(nlp(aa, 10 * mu_a, sigma_a) for aa in a) +
            sum(nlp(bb, 10 * mu_b, sigma_b) for bb in b)
    y_hat = [a[group_id[i]] + b[scenario_id[i]] for i in eachindex(y)]
    likelihood = sum(nlp(y[i], y_hat[i], sigma_y) for i in eachindex(y))
    (; parameters = (; a, b, mu_a, mu_b, sigma_a, sigma_b, sigma_y), log_jacobian,
       prior, likelihood, y_hat, posterior = prior + likelihood + log_jacobian)
end

@testset "PPL graph — pilots (posteriordb)" begin
    artifact = evaluate_pilots_source()
    @test artifact.source == strip(PILOTS_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = vcat(0.1 .* collect(1:5), -0.05 .* collect(1:8), [0.1, -0.1, 0.0, 0.0, 0.0])
    reference = _pilots_reference(q, PILOTS_GROUP_ID, PILOTS_SCENARIO_ID, PILOTS_Y)

    @testset "authored on the current baseline surface" begin
        @test occursin("a[group_id]", PILOTS_SOURCE)
        @test occursin("b[scenario_id]", PILOTS_SOURCE)
        @test occursin("normal(m, s).logpdf(aa)", PILOTS_SOURCE)
        @test occursin("y_hat = plate(", PILOTS_SOURCE)
        @test occursin("bound = (; group_id, scenario_id)", PILOTS_SOURCE)
        @test !occursin("struct ", PILOTS_SOURCE)
        @test artifact.normal_object === normal
    end

    @testset "constrain-only prunes the likelihood work" begin
        p = plan(model.graph; have = (model.unconstrained,), want = (model.parameters,))
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.likelihood.id) in produced)
        parameters = prepare(p)(q)
        @test parameters isa NamedTuple
        @test parameters.a ≈ reference.parameters.a
        @test parameters.sigma_a ≈ reference.parameters.sigma_a
        @test parameters.sigma_y ≈ reference.parameters.sigma_y
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.group_id, model.scenario_id, model.y),
                 want = (model.log_jacobian, model.prior, model.likelihood,
                         model.posterior))
        log_jacobian, prior, likelihood, posterior =
            prepare(p)(q, PILOTS_GROUP_ID, PILOTS_SCENARIO_ID, PILOTS_Y)
        @test log_jacobian ≈ reference.log_jacobian
        @test prior ≈ reference.prior
        @test likelihood ≈ reference.likelihood
        @test posterior ≈ reference.posterior
    end

    @testset "crossed integer-array gather y_hat from a constrained HAVE" begin
        p = plan(model.graph;
                 have = (model.parameters, model.group_id, model.scenario_id),
                 want = (model.y_hat,))
        y_hat = prepare(p)(reference.parameters, PILOTS_GROUP_ID, PILOTS_SCENARIO_ID)
        @test y_hat ≈ reference.y_hat
    end
end
