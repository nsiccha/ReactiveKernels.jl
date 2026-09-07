using ReactiveKernelsPPLExamples.Election88FullExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, bernoulli
using LogExpFunctions: logistic, log1pexp

# Graph-independent reference oracle for the posteriordb election88_full model:
# five varying-intercept Normal priors, a fixed-effect Normal prior, five
# scaled-logit interval transforms (sigma = 100·logistic(u)) with their
# `lub_constrain` Jacobian, and a bernoulli_logit likelihood over the linear
# predictor with the five group intercepts gathered by their integer indices.
function _e88_reference(q, age, edu, age_edu, state, region, black, female,
                        vprev, y, na, nb, nc, nd, ne)
    nlp(x, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - m) / s)^2
    tf(u) = 100.0 * logistic(u)
    jc(u) = log(100.0) - log1pexp(-u) - log1pexp(u)
    ob = na; oc = ob + nb; od = oc + nc; oe = od + nd; obeta = oe + ne
    osig = obeta + 5
    a = q[1:ob]; b = q[ob + 1:oc]; c = q[oc + 1:od]
    d = q[od + 1:oe]; e = q[oe + 1:obeta]
    beta = q[obeta + 1:obeta + 5]
    us = q[osig + 1:osig + 5]
    sig = tf.(us)
    log_jacobian = sum(jc.(us))
    log_prior = sum(nlp(x, 0.0, sig[1]) for x in a) +
                sum(nlp(x, 0.0, sig[2]) for x in b) +
                sum(nlp(x, 0.0, sig[3]) for x in c) +
                sum(nlp(x, 0.0, sig[4]) for x in d) +
                sum(nlp(x, 0.0, sig[5]) for x in e) +
                sum(nlp(x, 0.0, 100.0) for x in beta)
    mu_a = a[age]; mu_b = b[edu]; mu_c = c[age_edu]
    mu_d = d[state]; mu_e = e[region]
    y_hat = [beta[1] + beta[2] * black[i] + beta[3] * female[i] +
             beta[5] * female[i] * black[i] + beta[4] * vprev[i] +
             mu_a[i] + mu_b[i] + mu_c[i] + mu_d[i] + mu_e[i]
             for i in eachindex(y)]
    pointwise = [yi ? -log1pexp(-yh) : -log1pexp(yh) for (yi, yh) in zip(y, y_hat)]
    likelihood = sum(pointwise)
    (; parameters = (; a, b, c, d, e, beta1 = beta[1], beta2 = beta[2],
                     beta3 = beta[3], beta4 = beta[4], beta5 = beta[5],
                     sigma_a = sig[1], sigma_b = sig[2], sigma_c = sig[3],
                     sigma_d = sig[4], sigma_e = sig[5]),
     log_prior, log_jacobian, y_hat, pointwise, likelihood,
     p = logistic.(y_hat), posterior = log_prior + likelihood + log_jacobian)
end

@testset "PPL graph — election88_full (posteriordb)" begin
    artifact = evaluate_election88_full_source()
    @test artifact.source == strip(ELECTION88_FULL_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = artifact.inputs.q
    reference = _e88_reference(q, E88_AGE, E88_EDU, E88_AGE_EDU, E88_STATE,
        E88_REGION, E88_BLACK, E88_FEMALE, E88_VPREV, E88_Y,
        E88_N_AGE, E88_N_EDU, E88_N_AGE_EDU, E88_N_STATE, E88_N_REGION)

    @testset "authored on the current baseline surface" begin
        @test occursin("mu_a = a[age]", ELECTION88_FULL_SOURCE)
        @test occursin("mu_e = e[region_full]", ELECTION88_FULL_SOURCE)
        @test occursin("100.0 * logistic(u_sigma_a)", ELECTION88_FULL_SOURCE)
        @test occursin("normal(0.0, s).logpdf(x)", ELECTION88_FULL_SOURCE)
        @test occursin("bernoulli(logistic(", ELECTION88_FULL_SOURCE)
        @test occursin("y_hat = plate(", ELECTION88_FULL_SOURCE)
        @test occursin("bound = (; age, edu, age_edu, state, region_full",
                       ELECTION88_FULL_SOURCE)
        @test !occursin("struct ", ELECTION88_FULL_SOURCE)
        @test artifact.normal_object === normal
        @test artifact.bernoulli_object === bernoulli
        # (No `__ops__`/`operation(` readable-cleanliness assertions here: like
        # the radon reference, this model has BOUND ports — the five index
        # gathers and the group sizes — which legitimately render as nullary
        # `operation()` placeholders in the readable generated code.)
    end

    @testset "constrain-only prunes the likelihood work" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.n_age, model.n_edu,
                         model.n_age_edu, model.n_state, model.n_region_full),
                 want = (model.parameters,))
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.likelihood.id) in produced)
        parameters = prepare(p)(q, E88_N_AGE, E88_N_EDU, E88_N_AGE_EDU,
                                E88_N_STATE, E88_N_REGION)
        @test parameters isa NamedTuple
        @test parameters.a ≈ reference.parameters.a
        @test parameters.beta1 ≈ reference.parameters.beta1
        @test parameters.beta5 ≈ reference.parameters.beta5
        @test parameters.sigma_a ≈ reference.parameters.sigma_a
        @test parameters.sigma_e ≈ reference.parameters.sigma_e
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.age, model.edu,
                         model.age_edu, model.state, model.region_full,
                         model.black, model.female, model.v_prev_full, model.y,
                         model.n_age, model.n_edu, model.n_age_edu,
                         model.n_state, model.n_region_full),
                 want = (model.log_prior, model.log_jacobian, model.pointwise,
                         model.likelihood, model.posterior))
        log_prior, log_jacobian, pointwise, likelihood, posterior =
            prepare(p)(q, E88_AGE, E88_EDU, E88_AGE_EDU, E88_STATE, E88_REGION,
                       E88_BLACK, E88_FEMALE, E88_VPREV, E88_Y,
                       E88_N_AGE, E88_N_EDU, E88_N_AGE_EDU, E88_N_STATE,
                       E88_N_REGION)
        @test all(isfinite, pointwise)
        @test log_prior ≈ reference.log_prior
        @test log_jacobian ≈ reference.log_jacobian
        @test likelihood ≈ reference.likelihood
        @test likelihood ≈ sum(pointwise)
        @test posterior ≈ reference.posterior
    end

    @testset "gathered linear predictor y_hat and GQ p from a constrained HAVE" begin
        p = plan(model.graph;
                 have = (model.parameters, model.age, model.edu, model.age_edu,
                         model.state, model.region_full, model.black,
                         model.female, model.v_prev_full),
                 want = (model.y_hat, model.p))
        y_hat, prob = prepare(p)(reference.parameters, E88_AGE, E88_EDU,
            E88_AGE_EDU, E88_STATE, E88_REGION, E88_BLACK, E88_FEMALE, E88_VPREV)
        @test y_hat ≈ reference.y_hat
        @test prob ≈ reference.p
    end

    @testset "the likelihood total fuses buffer-free through the gathers" begin
        have = (:unconstrained, :age, :edu, :age_edu, :state, :region_full,
                :black, :female, :v_prev_full, :y,
                :n_age, :n_edu, :n_age_edu, :n_state, :n_region_full)
        bound = (; age = E88_AGE, edu = E88_EDU, age_edu = E88_AGE_EDU,
                 state = E88_STATE, region_full = E88_REGION,
                 n_age = E88_N_AGE, n_edu = E88_N_EDU, n_age_edu = E88_N_AGE_EDU,
                 n_state = E88_N_STATE, n_region_full = E88_N_REGION)
        pointwise_kernel = prepare(model; have, want = :pointwise, bound)
        likelihood_kernel = prepare(model; have, want = :likelihood, bound)
        pw = pointwise_kernel(q, E88_BLACK, E88_FEMALE, E88_VPREV, E88_Y)
        @test likelihood_kernel(q, E88_BLACK, E88_FEMALE, E88_VPREV, E88_Y) ≈ sum(pw)
        @test occursin("similar", string(code_expr(pointwise_kernel)))
        # Even with the five index gathers, the total-only query fuses the whole
        # traversal and materializes no intermediate length-N vector.
        @test !occursin("similar", string(code_expr(likelihood_kernel)))
    end
end
