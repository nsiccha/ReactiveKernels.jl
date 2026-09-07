using ReactiveKernelsPPLExamples.MtExample
using LogExpFunctions: logistic, log1pexp, logaddexp

# Graph-independent oracle for Mt (capture-recapture, time-varying detection,
# data-augmentation log_sum_exp marginalization). The per-individual detection
# log-likelihood is the matrix-vector product Y·u_p (since logit(p[j]) = u_p[j])
# plus the all-zero-history constant.
function _mt_reference(q, Y, s, T)
    om = logistic(q[1]); up = q[2:T + 1]
    jac = (-log1pexp(-q[1]) - log1pexp(q[1])) +
          sum(-log1pexp(-u) - log1pexp(u) for u in up)
    log1mp = [-log1pexp(u) for u in up]
    bern0 = sum(log1mp)
    bern = Y * up .+ bern0
    lo = log(om); l1 = log1p(-om)
    like = 0.0
    for i in eachindex(s)
        like += s[i] > 0 ? lo + bern[i] : logaddexp(lo + bern[i], l1)
    end
    pr = exp(bern0)
    omega_nd = (om * pr) / (om * pr + (1 - om))
    p = [logistic(u) for u in up]
    (; parameters = (; omega = om, p), log_jacobian = jac, likelihood = like,
       posterior = like + jac, omega_nd, pr)
end

@testset "PPL graph — Mt (posteriordb capture-recapture, time-varying p)" begin
    artifact = evaluate_mt_source()
    @test artifact.source == strip(MT_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = vcat(0.2, [-0.1, 0.15, 0.05])
    reference = _mt_reference(q, MT_Y, MT_S, MT_T)

    @testset "authored on the current baseline surface" begin
        @test occursin("logaddexp(", MT_SOURCE)
        @test occursin("ifelse(si > 0", MT_SOURCE)
        @test occursin("Y * u_p", MT_SOURCE)
        @test occursin("pointwise = plate(", MT_SOURCE)
        @test !occursin("struct ", MT_SOURCE)
    end

    @testset "parameters node is exposed and matches the oracle" begin
        # Constrain-only pruning (have=unconstrained, want=just parameters) is not
        # planned when a parameter is a plate-produced vector: the forward plate
        # producer and the `p = parameters.p` inverse edge form a cycle the planner
        # cannot break (identical behaviour in the reference MhExample graph, whose
        # test omits this query for the same reason). The forward density path,
        # which also produces the likelihood/posterior, plans cleanly.
        p = prepare(model;
            have = (:unconstrained, :Y, :s, :T, :M),
            want = (:parameters, :posterior),
            bound = (; Y = MT_Y, s = MT_S, T = MT_T, M = MT_M))
        parameters, posterior = p(q)
        @test parameters isa NamedTuple
        @test parameters.omega ≈ reference.parameters.omega
        @test collect(parameters.p) ≈ reference.parameters.p
        @test posterior ≈ reference.posterior
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = prepare(model;
            have = (:unconstrained, :Y, :s, :T, :M),
            want = (:log_jacobian, :pointwise, :likelihood, :posterior, :omega_nd),
            bound = (; Y = MT_Y, s = MT_S, T = MT_T, M = MT_M))
        log_jacobian, pointwise, likelihood, posterior, omega_nd = p(q)
        @test all(isfinite, pointwise)
        @test log_jacobian ≈ reference.log_jacobian
        @test likelihood ≈ reference.likelihood
        @test likelihood ≈ sum(pointwise)
        @test posterior ≈ reference.posterior
        @test omega_nd ≈ reference.omega_nd
    end

    @testset "log_sum_exp marginalization is exercised (both branches present)" begin
        @test any(>(0), MT_S)   # observed individuals
        @test any(==(0), MT_S)  # augmented (never-detected) individuals
    end

    @testset "detection likelihood is the bound-matrix contraction Y * u_p" begin
        # The time-varying detection log-likelihood is a matrix-vector product,
        # so the per-individual `pointwise` vector materializes a buffer while the
        # summed likelihood reduces to a scalar total.
        pointwise_kernel = prepare(model;
            have = (:unconstrained, :Y, :s, :T, :M), want = :pointwise,
            bound = (; Y = MT_Y, s = MT_S, T = MT_T, M = MT_M))
        likelihood_kernel = prepare(model;
            have = (:unconstrained, :Y, :s, :T, :M), want = :likelihood,
            bound = (; Y = MT_Y, s = MT_S, T = MT_T, M = MT_M))
        pw = pointwise_kernel(q)
        @test likelihood_kernel(q) ≈ sum(pw)
        @test occursin("similar", string(code_expr(pointwise_kernel)))
    end
end
