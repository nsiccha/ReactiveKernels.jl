using ReactiveKernelsPPLExamples.MbExample
using LogExpFunctions: logistic, log1pexp, logaddexp

# Graph-independent oracle for Mb (capture-recapture, behavioural/trap response,
# data-augmentation log_sum_exp marginalization). The per-individual detection
# log-likelihood collapses to four data counts:
#   bern[i] = a·log(p) + b·log(1-p) + e·log(c) + f·log(1-c).
function _mb_reference(q, a, b, e, f, s, T)
    om = logistic(q[1]); p = logistic(q[2]); c = logistic(q[3])
    jac = (-log1pexp(-q[1]) - log1pexp(q[1])) +
          (-log1pexp(-q[2]) - log1pexp(q[2])) +
          (-log1pexp(-q[3]) - log1pexp(q[3]))
    logp = -log1pexp(-q[2]); log1mp = -log1pexp(q[2])
    logc = -log1pexp(-q[3]); log1mc = -log1pexp(q[3])
    lo = log(om); l1 = log1p(-om)
    like = 0.0
    for i in eachindex(s)
        bern = a[i] * logp + b[i] * log1mp + e[i] * logc + f[i] * log1mc
        like += s[i] > 0 ? lo + bern : logaddexp(lo + bern, l1)
    end
    p_never = exp(T * log1mp)
    omega_nd = (om * p_never) / (om * p_never + (1 - om))
    trap_response = c - p
    (; parameters = (; omega = om, p, c), log_jacobian = jac, likelihood = like,
       posterior = like + jac, omega_nd, trap_response)
end

@testset "PPL graph — Mb (posteriordb capture-recapture, behavioural response)" begin
    artifact = evaluate_mb_source()
    @test artifact.source == strip(MB_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = [0.2, -0.1, 0.25]
    reference = _mb_reference(q, MB_A, MB_B, MB_E, MB_F, MB_S, MB_T)

    @testset "authored on the current baseline surface" begin
        @test occursin("logaddexp(", MB_SOURCE)
        @test occursin("ifelse(si > 0", MB_SOURCE)
        @test occursin("logistic(u_omega)", MB_SOURCE)
        @test occursin("bern = plate(", MB_SOURCE)
        @test occursin("trap_response", MB_SOURCE)
        @test !occursin("struct ", MB_SOURCE)
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
        @test parameters.c ≈ reference.parameters.c
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = prepare(model;
            have = (:unconstrained, :a, :b, :e, :f, :s, :T, :M),
            want = (:log_jacobian, :pointwise, :likelihood, :posterior,
                    :omega_nd, :trap_response),
            bound = (; a = MB_A, b = MB_B, e = MB_E, f = MB_F,
                       s = MB_S, T = MB_T, M = MB_M))
        log_jacobian, pointwise, likelihood, posterior, omega_nd, trap_response = p(q)
        @test all(isfinite, pointwise)
        @test log_jacobian ≈ reference.log_jacobian
        @test likelihood ≈ reference.likelihood
        @test likelihood ≈ sum(pointwise)
        @test posterior ≈ reference.posterior
        @test omega_nd ≈ reference.omega_nd
        @test trap_response ≈ reference.trap_response
    end

    @testset "log_sum_exp marginalization is exercised (both branches present)" begin
        @test any(>(0), MB_S)   # observed individuals
        @test any(==(0), MB_S)  # augmented (never-detected) individuals
        # Every individual's four behavioural counts sum to the occasion count.
        @test all(MB_A .+ MB_B .+ MB_E .+ MB_F .== MB_T)
    end

    @testset "behavioural counts feed a per-individual bern node then a summed total" begin
        pointwise_kernel = prepare(model;
            have = (:unconstrained, :a, :b, :e, :f, :s, :T, :M), want = :pointwise,
            bound = (; a = MB_A, b = MB_B, e = MB_E, f = MB_F,
                       s = MB_S, T = MB_T, M = MB_M))
        likelihood_kernel = prepare(model;
            have = (:unconstrained, :a, :b, :e, :f, :s, :T, :M), want = :likelihood,
            bound = (; a = MB_A, b = MB_B, e = MB_E, f = MB_F,
                       s = MB_S, T = MB_T, M = MB_M))
        pw = pointwise_kernel(q)
        @test likelihood_kernel(q) ≈ sum(pw)
        @test occursin("similar", string(code_expr(pointwise_kernel)))
    end
end
