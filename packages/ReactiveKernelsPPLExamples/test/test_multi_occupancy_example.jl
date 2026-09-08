using ReactiveKernelsPPLExamples.MultiOccupancyExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: cauchy, beta
using LogExpFunctions: logistic, log1pexp, logaddexp

# Graph-independent oracle for the Dorazio-Royle marginalized occupancy model,
# computed from the raw n×J detection matrix X (lchoose via exact Base.binomial).
function _occ_reference(q, X, n, J, K)
    S = (length(q) - 6) ÷ 2
    alpha = q[1]; beta_ = q[2]
    Omega = logistic(q[3]); rho = -1 + 2 * logistic(q[4])
    s1 = exp(q[5]); s2 = exp(q[6])
    log_jacobian = (-log1pexp(-q[3]) - log1pexp(q[3])) +
                   (log(2.0) - log1pexp(-q[4]) - log1pexp(q[4])) + q[5] + q[6]
    uv1 = q[7:6 + S]; uv2 = q[7 + S:6 + 2S]
    cauchy_lp(x) = -log(π) - log(2.5) - log1p((x / 2.5)^2)
    # Beta(2,2): lbeta(2,2) = log(Γ(2)Γ(2)/Γ(4)) = log(1/6) = -log(6), so
    # beta_lpdf(x|2,2) = log(x) + log(1-x) + log(6).
    beta22_lp(x) = log(x) + log1p(-x) + log(6.0)
    prior = cauchy_lp(alpha) + cauchy_lp(beta_) + cauchy_lp(s1) + cauchy_lp(s2) +
            beta22_lp((rho + 1) / 2) + beta22_lp(Omega)
    for i in 1:S
        z1 = uv1[i] / s1; z2 = uv2[i] / s2; om = 1 - rho^2
        prior += -log(2π) - log(s1) - log(s2) - 0.5 * log(om) -
                 0.5 * (z1^2 - 2 * rho * z1 * z2 + z2^2) / om
    end
    lpsi = uv1 .+ alpha; lth = uv2 .+ beta_
    logO = log(Omega); log1mO = log1p(-Omega)
    ll = n * logO
    for i in 1:n
        lp1 = -log1pexp(-lpsi[i]); lps = -log1pexp(-lth[i]); lpf = -log1pexp(lth[i])
        lp_un = logaddexp(lp1 + K * lpf, -log1pexp(lpsi[i]))
        for j in 1:J
            x = X[i, j]
            ll += x > 0 ?
                lp1 + log(Base.binomial(K, x)) + x * lps + (K - x) * lpf : lp_un
        end
    end
    for i in (n + 1):S
        lp1 = -log1pexp(-lpsi[i]); lpf = -log1pexp(lth[i])
        lp_un = logaddexp(lp1 + K * lpf, -log1pexp(lpsi[i]))
        ll += logaddexp(log1mO, logO + J * lp_un)
    end
    (; prior, log_jacobian, likelihood = ll, posterior = prior + ll + log_jacobian)
end

@testset "PPL graph — multi_occupancy (posteriordb)" begin
    artifact = evaluate_multi_occupancy_source()
    @test artifact.source == strip(MULTI_OCC_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    S = MULTI_OCC_S
    n, J, K = MULTI_OCC_N, MULTI_OCC_J, MULTI_OCC_K
    X = MULTI_OCC_X
    q = vcat([0.2, -0.1, 0.3, 0.15, -0.2, 0.1], 0.25 .* sin.(1:S), 0.25 .* cos.(1:S))
    reference = _occ_reference(q, X, n, J, K)

    _bound() = (; X = MULTI_OCC_X, n = MULTI_OCC_N, J = MULTI_OCC_J, K = MULTI_OCC_K)

    @testset "authored in-graph over ONLY the raw counts (normalizer in-graph)" begin
        @test occursin("Xflat = vec(X)", MULTI_OCC_SOURCE)              # flat counts in-graph
        @test occursin("spec = repeat(1:n, J)", MULTI_OCC_SOURCE)       # species coord in-graph
        @test occursin("binomial(; n = kk, logit = lt).logpdf(x)", MULTI_OCC_SOURCE)
        @test occursin("logaddexp(", MULTI_OCC_SOURCE)
        @test occursin("cauchy(0.0, 2.5).logpdf(alpha)", MULTI_OCC_SOURCE)
        @test !occursin("lchoose", MULTI_OCC_SOURCE)                   # no precomputed normalizer
        @test !occursin("struct ", MULTI_OCC_SOURCE)
        @test artifact.cauchy_object === cauchy
        @test artifact.beta_object === beta
    end

    @testset "acceptance entry is ONLY the rebuildable raw matrix + dims" begin
        rebuilt = multi_occupancy_inputs(
            ReactiveKernelsPPLExamples._posteriordb_data("butterfly-multi_occupancy"))
        @test propertynames(rebuilt) == (:X, :n, :J, :K)   # no flat/spec precompute
        @test rebuilt.X == MULTI_OCC_X
        @test (rebuilt.n, rebuilt.J, rebuilt.K) == (n, J, K)
        @test size(MULTI_OCC_X) == (n, J)
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.X, model.n, model.J, model.K),
                 want = (model.prior, model.log_jacobian, model.likelihood, model.posterior))
        prior, log_jacobian, likelihood, posterior =
            prepare(p)(q, MULTI_OCC_X, n, J, K)
        @test prior ≈ reference.prior
        @test log_jacobian ≈ reference.log_jacobian
        @test likelihood ≈ reference.likelihood
        @test posterior ≈ reference.posterior
    end

    @testset "build+prepare+execute in one ordinary function, repeat-use stable" begin
        function once(q)
            k = prepare(build_multi_occupancy_graph();
                have = (:unconstrained, :X, :n, :J, :K),
                want = :posterior, bound = _bound())
            k(q)
        end
        @test once(q) ≈ reference.posterior
        k = prepare(build_multi_occupancy_graph();
            have = (:unconstrained, :X, :n, :J, :K),
            want = :posterior, bound = _bound())
        @test k(q) == k(q)
    end

    @testset "data-generic: alternate small community (n=2, J=2, K=5, S=3)" begin
        Xsmall = [1 0; 0 2]
        d = Dict("X" => [[1, 0], [0, 2]], "n" => 2, "J" => 2, "K" => 5)
        inp = multi_occupancy_inputs(d)
        qs = vcat([0.1, -0.2, 0.2, 0.1, -0.1, 0.15], 0.2 .* [1.0, -1.0, 0.5],
                  0.1 .* [-0.5, 1.0, 0.5])
        k = prepare(build_multi_occupancy_graph();
            have = (:unconstrained, :X, :n, :J, :K),
            want = :posterior, bound = inp)
        @test k(qs) ≈ _occ_reference(qs, Xsmall, 2, 2, 5).posterior
    end
end
