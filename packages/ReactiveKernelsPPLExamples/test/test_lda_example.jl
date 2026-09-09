using ReactiveKernelsPPLExamples
using ReactiveKernelsPPLExamples.LDAExample
using ReactiveKernelsPPLExamples.LDAExample:
    LDA_DOC, LDA_W, LDA_ALPHA, LDA_BETA, LDA_M, build_lda_graph,
    LDA_SOURCE, evaluate_lda_source, lda_fixture
using ReactiveKernelsDistributionKernels.DistributionKernelSources: dirichlet
using LogExpFunctions: logsumexp
using SpecialFunctions: loggamma
using DifferentiationInterface
import Enzyme
using Random

# ---- Independent reference oracle: Stan-2.39 inverse-ILR simplex transform +
# marginalized LDA density, reimplemented from scratch (matches BridgeStan to
# machine precision; this is the graph-independent truth the RK kernel is tested
# against). ----
function _stz(y::AbstractVector{T}) where {T}
    N = length(y); z = zeros(T, N + 1)
    N == 0 && return z
    sw = zero(T)
    for i in N:-1:1
        n = float(i)
        wv = y[i] / sqrt(n * (n + 1))
        sw += wv
        z[i] += sw
        z[i + 1] -= wv * n
    end
    z
end
function _simplex_lj(y)
    z = _stz(y); lse = logsumexp(z)
    logx = z .- lse
    exp.(logx), logx, sum(logx) + 0.5 * log(length(z))
end
function _lda_reference(q, doc, w, alpha, beta, M)
    K = length(alpha); V = length(beta); N = length(w)
    nt = K - 1; np = V - 1; off = 0
    lnZt = loggamma(sum(alpha)) - sum(loggamma, alpha)
    lnZp = loggamma(sum(beta)) - sum(loggamma, beta)
    logJ = 0.0; prior = 0.0
    LOG_THETA = Matrix{Float64}(undef, K, M)
    for m in 1:M
        x, logx, lj = _simplex_lj(q[off+1:off+nt]); off += nt
        LOG_THETA[:, m] = logx; logJ += lj
        prior += lnZt + sum((alpha .- 1) .* logx)
    end
    LOG_PHI = Matrix{Float64}(undef, V, K)
    for k in 1:K
        x, logx, lj = _simplex_lj(q[off+1:off+np]); off += np
        LOG_PHI[:, k] = logx; logJ += lj
        prior += lnZp + sum((beta .- 1) .* logx)
    end
    lik = 0.0
    for n in 1:N
        g = [LOG_THETA[k, doc[n]] + LOG_PHI[w[n], k] for k in 1:K]
        lik += logsumexp(g)
    end
    (; prior, likelihood = lik, log_jacobian = logJ, posterior = prior + lik + logJ)
end

const _LDA_HAVE = (:unconstrained, :doc, :w, :alpha, :beta, :M)

# build + prepare + execute inside ONE ordinary function (world-age smoke)
function _lda_posterior_once(q, doc, w, alpha, beta, M)
    g = build_lda_graph()
    prepare(g; have = _LDA_HAVE, want = :posterior,
        bound = (; doc, w, alpha, beta, M))(q)
end

@testset "PPL graph — LDA (posteriordb ldaK2/ldaK5, topic marginalized)" begin
    @testset "source-authority artifact" begin
        artifact = evaluate_lda_source()
        @test artifact.source == strip(LDA_SOURCE, '\n')
        @test artifact.model isa KernelSpec
        @test artifact.output == Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
        @test artifact.dirichlet_object === dirichlet
        # authored on the intended surface: log-space transform, in-graph basis
        @test occursin("logsumexp(", LDA_SOURCE)
        @test occursin("dirichlet(a).logpdf", LDA_SOURCE)
        @test occursin("sum_to_zero_constrain", LDA_SOURCE)     # documented in-graph
        @test !occursin("struct ", LDA_SOURCE)
    end

    @testset "fresh graph per build; repeat-use stable" begin
        @test build_lda_graph() !== build_lda_graph()
        fx = lda_fixture()
        K = length(fx.alpha); V = length(fx.beta)
        dim = fx.M * (K - 1) + K * (V - 1)
        q = 0.1 .* sin.(collect(1.0:dim))
        kb = prepare(build_lda_graph(); have = _LDA_HAVE, want = :posterior,
            bound = (; fx.doc, fx.w, fx.alpha, fx.beta, fx.M))
        @test kb(q) == kb(q)                                    # deterministic repeat-use
        @test _lda_posterior_once(q, fx.doc, fx.w, fx.alpha, fx.beta, fx.M) ≈ kb(q)
    end

    @testset "three_men1 (K=2): graph nodes vs independent oracle" begin
        doc, w, alpha, beta, M = LDA_DOC, LDA_W, LDA_ALPHA, LDA_BETA, LDA_M
        K = length(alpha); V = length(beta)
        dim = M * (K - 1) + K * (V - 1)
        kb = prepare(build_lda_graph(); have = _LDA_HAVE,
            want = (:prior, :likelihood, :log_jacobian, :posterior),
            bound = (; doc, w, alpha, beta, M))
        rng = Xoshiro(11)
        for _ in 1:3
            q = 0.4 .* randn(rng, dim)                          # nontrivial constrained pts
            pr, lk, lj, po = kb(q)
            ref = _lda_reference(q, doc, w, alpha, beta, M)
            @test pr ≈ ref.prior
            @test lk ≈ ref.likelihood
            @test lj ≈ ref.log_jacobian
            @test po ≈ ref.posterior
            @test isfinite(po)
        end
    end

    @testset "pointwise plate exposes a buffer-free total" begin
        doc, w, alpha, beta, M = LDA_DOC, LDA_W, LDA_ALPHA, LDA_BETA, LDA_M
        K = length(alpha); V = length(beta)
        q = 0.2 .* randn(Xoshiro(3), M * (K - 1) + K * (V - 1))
        pw = prepare(build_lda_graph(); have = _LDA_HAVE, want = :pointwise,
            bound = (; doc, w, alpha, beta, M))(q)
        lk = prepare(build_lda_graph(); have = _LDA_HAVE, want = :likelihood,
            bound = (; doc, w, alpha, beta, M))(q)
        @test length(pw) == length(w)
        @test lk ≈ sum(pw)
    end

    @testset "alternate small dims (data-generic K=3): graph vs oracle" begin
        doc = [1, 1, 2, 2, 3, 3, 1, 2]
        w   = [1, 2, 3, 4, 1, 2, 3, 4]
        M = 3; alpha = [0.7, 1.3, 0.9]; beta = [0.5, 1.0, 1.5, 0.8]
        K = length(alpha); V = length(beta)
        dim = M * (K - 1) + K * (V - 1)
        kb = prepare(build_lda_graph(); have = _LDA_HAVE,
            want = (:prior, :likelihood, :log_jacobian, :posterior),
            bound = (; doc, w, alpha, beta, M))
        for q in ([0.3 .* sin.(1.0:dim)...], [(-0.5 .* cos.(1.0:dim))...])
            pr, lk, lj, po = kb(q)
            ref = _lda_reference(q, doc, w, alpha, beta, M)
            @test pr ≈ ref.prior
            @test lk ≈ ref.likelihood
            @test lj ≈ ref.log_jacobian
            @test po ≈ ref.posterior
        end
    end

    @testset "gradient: RK plain-Enzyme vs central FD of the oracle" begin
        # Independent gradient check (no BridgeStan). Central FD of the graph-free
        # oracle on a handful of coords must match the RK reverse-mode gradient.
        doc, w, alpha, beta, M = LDA_DOC, LDA_W, LDA_ALPHA, LDA_BETA, LDA_M
        K = length(alpha); V = length(beta)
        dim = M * (K - 1) + K * (V - 1)
        AE = AutoEnzyme(mode = Enzyme.Reverse, function_annotation = Enzyme.Const)
        kb = prepare(build_lda_graph(); have = _LDA_HAVE, want = :posterior,
            bound = (; doc, w, alpha, beta, M))
        q = 0.3 .* randn(Xoshiro(7), dim)
        prep = prepare_ad(kb, AE, q; active = :unconstrained)
        g = ReactiveKernels.ad_value_and_gradient!(prep, similar(q), q)[2]
        @test all(isfinite, g)
        coords = (1, 2, 7, M * (K - 1) + 1, dim)                # theta + phi coords
        h = 1e-6
        for i in coords
            qp = copy(q); qp[i] += h; qm = copy(q); qm[i] -= h
            fd = (_lda_reference(qp, doc, w, alpha, beta, M).posterior -
                  _lda_reference(qm, doc, w, alpha, beta, M).posterior) / (2h)
            @test isapprox(g[i], fd; rtol = 1e-4, atol = 1e-3)
        end
    end
end
