using ReactiveKernelsPPLExamples.M0Example
using LogExpFunctions: logistic, log1pexp, logaddexp

# Graph-independent oracle for M0 (capture-recapture, single detection prob,
# data-augmentation log_sum_exp marginalization).
function _m0_reference(q, s, lchoose, T)
    om = logistic(q[1]); p = logistic(q[2])
    jac = (-log1pexp(-q[1]) - log1pexp(q[1])) +
          (-log1pexp(-q[2]) - log1pexp(q[2]))
    logp = -log1pexp(-q[2]); log1mp = -log1pexp(q[2])
    lo = log(om); l1 = log1p(-om)
    like = 0.0
    for i in eachindex(s)
        if s[i] > 0
            like += lo + lchoose[i] + s[i] * logp + (T - s[i]) * log1mp
        else
            like += logaddexp(lo + T * log1mp, l1)
        end
    end
    p_never = exp(T * log1mp)
    omega_nd = (om * p_never) / (om * p_never + (1 - om))
    (; parameters = (; omega = om, p), log_jacobian = jac, likelihood = like,
       posterior = like + jac, omega_nd)
end

@testset "PPL graph — M0 (posteriordb capture-recapture, single p)" begin
    artifact = evaluate_m0_source()
    @test artifact.source == strip(M0_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = [0.2, -0.15]
    reference = _m0_reference(q, M0_S, M0_LCHOOSE, M0_T)

    @testset "authored on the current baseline surface" begin
        @test occursin("logaddexp(", M0_SOURCE)
        @test occursin("ifelse(si > 0", M0_SOURCE)
        @test occursin("logistic(u_omega)", M0_SOURCE)
        @test occursin("pointwise = plate(", M0_SOURCE)
        @test !occursin("struct ", M0_SOURCE)
    end

    @testset "constrain-only prunes the likelihood work" begin
        p = plan(model.graph; have = (model.unconstrained,), want = (model.parameters,))
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.likelihood.id) in produced)
        parameters = prepare(p)(q)
        @test parameters isa NamedTuple
        @test parameters.omega ≈ reference.parameters.omega
        @test parameters.p ≈ reference.parameters.p
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = prepare(model;
            have = (:unconstrained, :s, :lchoose, :T, :M),
            want = (:log_jacobian, :pointwise, :likelihood, :posterior, :omega_nd),
            bound = (; s = M0_S, lchoose = M0_LCHOOSE, T = M0_T, M = M0_M))
        log_jacobian, pointwise, likelihood, posterior, omega_nd = p(q)
        @test all(isfinite, pointwise)
        @test log_jacobian ≈ reference.log_jacobian
        @test likelihood ≈ reference.likelihood
        @test likelihood ≈ sum(pointwise)
        @test posterior ≈ reference.posterior
        @test omega_nd ≈ reference.omega_nd
    end

    @testset "log_sum_exp marginalization is exercised (both branches present)" begin
        @test any(>(0), M0_S)   # observed individuals
        @test any(==(0), M0_S)  # augmented (never-detected) individuals
    end

    @testset "one authored plate exposes a buffer-free total" begin
        pointwise_kernel = prepare(model;
            have = (:unconstrained, :s, :lchoose, :T, :M), want = :pointwise,
            bound = (; s = M0_S, lchoose = M0_LCHOOSE, T = M0_T, M = M0_M))
        likelihood_kernel = prepare(model;
            have = (:unconstrained, :s, :lchoose, :T, :M), want = :likelihood,
            bound = (; s = M0_S, lchoose = M0_LCHOOSE, T = M0_T, M = M0_M))
        pw = pointwise_kernel(q)
        @test likelihood_kernel(q) ≈ sum(pw)
        @test occursin("similar", string(code_expr(pointwise_kernel)))
        @test !occursin("similar", string(code_expr(likelihood_kernel)))
    end
end
