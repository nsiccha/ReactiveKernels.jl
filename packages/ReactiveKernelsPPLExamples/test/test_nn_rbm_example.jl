using ReactiveKernelsPPLExamples
using ReactiveKernelsPPLExamples.NNRBMExample
using ReactiveKernelsPPLExamples.NNRBMExample:
    NN_RBM_X, NN_RBM_Y, NN_RBM_K, NN_RBM_J, build_nn_rbm_graph,
    NN_RBM_SOURCE, evaluate_nn_rbm_source, nn_rbm_fixture
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    normal, inverse_gamma, categorical_logit
using LogExpFunctions: logsumexp
using SpecialFunctions: loggamma
using DifferentiationInterface
using LinearAlgebra
import Enzyme
using Random

# ---- Independent reference oracle: neural-RBM softmax density reimplemented from
# scratch (matches BridgeStan to machine precision). ----
_normal_lpdf(x, mu, sd) = -0.5 * log(2π) - log(sd) - 0.5 * ((x - mu) / sd)^2
_invgamma_lpdf(x, a, s) = a * log(s) - loggamma(a) - (a + 1) * log(x) - s / x
function _rbm_reference(q, x, y, K, J)
    N, M = size(x)
    nu_a = 0.5; s2_0_a = (0.05 / M^(1 / nu_a))^2
    nu_b = 0.5; s2_0_b = (0.05 / J^(1 / nu_b))^2
    u_s2a = q[1]; u_s2b = q[2]
    sigma2_alpha = exp(u_s2a); sigma2_beta = exp(u_s2b)
    off = 2
    alpha = reshape(q[off+1:off+M*J], M, J); off += M * J
    beta = reshape(q[off+1:off+J*(K-1)], J, K - 1); off += J * (K - 1)
    alpha1 = q[off+1:off+J]; off += J
    beta1 = q[off+1:off+K-1]; off += K - 1
    @assert off == length(q)
    logJ = u_s2a + u_s2b
    pre = x * alpha .+ alpha1'                 # N × J
    H = tanh.(pre)
    HB = H * beta .+ beta1'                     # N × (K-1)
    sd_a = sqrt(sigma2_alpha); sd_b = sqrt(sigma2_beta)
    prior = sum(_normal_lpdf.(alpha, 0.0, sd_a)) + sum(_normal_lpdf.(beta, 0.0, sd_b)) +
            sum(_normal_lpdf.(alpha1, 0.0, 1.0)) + sum(_normal_lpdf.(beta1, 0.0, 1.0)) +
            _invgamma_lpdf(sigma2_alpha, nu_a / 2, nu_a * s2_0_a / 2) +
            _invgamma_lpdf(sigma2_beta, nu_b / 2, nu_b * s2_0_b / 2)
    lik = 0.0
    for n in 1:N
        v = vcat(1.0, HB[n, :])                 # K logits; class-1 reference logit = 1
        lik += v[y[n]] - logsumexp(v)
    end
    (; prior, likelihood = lik, log_jacobian = logJ, posterior = prior + lik + logJ)
end

const _RBM_HAVE = (:unconstrained, :x, :y, :K, :J)
_rbm_dim(M, K, J) = 2 + M * J + J * (K - 1) + J + (K - 1)

function _rbm_posterior_once(q, x, y, K, J)
    g = build_nn_rbm_graph()
    prepare(g; have = _RBM_HAVE, want = :posterior, bound = (; x, y, K, J))(q)
end

@testset "PPL graph — nn_rbm1b (posteriordb neural-network softmax classifier)" begin
    @testset "first-use: build→prepare→execute in one function runs no demo tail" begin
        # The model_only template makes build_nn_rbm_graph()+prepare()+execute safe
        # INSIDE one ordinary function (no fresh Core.eval in the caller → no
        # world-age hazard). This runs BEFORE any full evaluate_nn_rbm_source()
        # below, so the demo-tail sentinel is UNCHANGED by the first-use path (it
        # is 0 in a fresh import; the delta being 0 is the structural claim).
        before = ReactiveKernelsPPLExamples._DEMO_TAIL_EXECUTIONS[]
        fx = nn_rbm_fixture()
        dim = _rbm_dim(size(fx.x, 2), fx.K, fx.J)
        q = 0.05 .* collect(1.0:dim) .- 0.1
        v = _rbm_posterior_once(q, fx.x, fx.y, fx.K, fx.J)
        @test isfinite(v)
        @test ReactiveKernelsPPLExamples._DEMO_TAIL_EXECUTIONS[] == before
    end

    @testset "source-authority artifact" begin
        artifact = evaluate_nn_rbm_source()
        @test artifact.source == strip(NN_RBM_SOURCE, '\n')
        @test artifact.model isa KernelSpec
        @test artifact.output == Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
        @test artifact.categorical_logit_object === categorical_logit
        @test artifact.inverse_gamma_object === inverse_gamma
        @test occursin("categorical_logit(lc).logpdf", NN_RBM_SOURCE)
        @test occursin("tanh.(pre)", NN_RBM_SOURCE)
        @test occursin("inverse_gamma(", NN_RBM_SOURCE)
        @test occursin("permutedims(HB)", NN_RBM_SOURCE)        # Enzyme-safe vcat
        @test !occursin("struct ", NN_RBM_SOURCE)
    end

    @testset "fresh graph per build; repeat-use stable" begin
        @test build_nn_rbm_graph() !== build_nn_rbm_graph()
        fx = nn_rbm_fixture()
        dim = _rbm_dim(size(fx.x, 2), fx.K, fx.J)
        q = 0.05 .* collect(1.0:dim) .- 0.1
        kb = prepare(build_nn_rbm_graph(); have = _RBM_HAVE, want = :posterior,
            bound = (; fx.x, fx.y, fx.K, fx.J))
        @test kb(q) == kb(q)
        @test _rbm_posterior_once(q, fx.x, fx.y, fx.K, fx.J) ≈ kb(q)
    end

    @testset "mnist_100 (J=10): graph nodes vs independent oracle" begin
        x, y, K, J = NN_RBM_X, NN_RBM_Y, NN_RBM_K, NN_RBM_J
        dim = _rbm_dim(size(x, 2), K, J)
        kb = prepare(build_nn_rbm_graph(); have = _RBM_HAVE,
            want = (:prior, :likelihood, :log_jacobian, :posterior),
            bound = (; x, y, K, J))
        rng = Xoshiro(21)
        for _ in 1:3
            q = 0.2 .* randn(rng, dim)
            pr, lk, lj, po = kb(q)
            ref = _rbm_reference(q, x, y, K, J)
            @test pr ≈ ref.prior
            @test lk ≈ ref.likelihood
            @test lj ≈ ref.log_jacobian
            @test po ≈ ref.posterior
            @test isfinite(po)
        end
    end

    @testset "pointwise/total parity (summed likelihood == sum of pointwise)" begin
        x, y, K, J = NN_RBM_X, NN_RBM_Y, NN_RBM_K, NN_RBM_J
        q = 0.1 .* randn(Xoshiro(5), _rbm_dim(size(x, 2), K, J))
        pw = prepare(build_nn_rbm_graph(); have = _RBM_HAVE, want = :pointwise,
            bound = (; x, y, K, J))(q)
        lk = prepare(build_nn_rbm_graph(); have = _RBM_HAVE, want = :likelihood,
            bound = (; x, y, K, J))(q)
        @test length(pw) == length(y)
        @test lk ≈ sum(pw)
    end

    @testset "alternate small dims (data-generic K=3, J=2): graph vs oracle" begin
        rng = Xoshiro(99)
        N = 5; M = 3; K = 3; J = 2
        x = randn(rng, N, M); y = [1, 2, 3, 1, 2]
        dim = _rbm_dim(M, K, J)
        kb = prepare(build_nn_rbm_graph(); have = _RBM_HAVE,
            want = (:prior, :likelihood, :log_jacobian, :posterior),
            bound = (; x, y, K, J))
        for q in ([0.3 .* sin.(1.0:dim)...], [(-0.4 .* cos.(1.0:dim))...])
            pr, lk, lj, po = kb(q)
            ref = _rbm_reference(q, x, y, K, J)
            @test pr ≈ ref.prior
            @test lk ≈ ref.likelihood
            @test lj ≈ ref.log_jacobian
            @test po ≈ ref.posterior
        end
    end

    @testset "gradient: RK plain-Enzyme vs central FD of the oracle" begin
        x, y, K, J = NN_RBM_X, NN_RBM_Y, NN_RBM_K, NN_RBM_J
        dim = _rbm_dim(size(x, 2), K, J)
        AE = AutoEnzyme(mode = Enzyme.Reverse, function_annotation = Enzyme.Const)
        kb = prepare(build_nn_rbm_graph(); have = _RBM_HAVE, want = :posterior,
            bound = (; x, y, K, J))
        q = 0.15 .* randn(Xoshiro(13), dim)
        prep = prepare_ad(kb, AE, q; active = :unconstrained)
        g = ReactiveKernels.ad_value_and_gradient!(prep, similar(q), q)[2]
        @test all(isfinite, g)
        coords = (1, 2, 3, 2 + size(x, 2) * J + 1, dim)         # scales, alpha, beta, beta1
        h = 1e-6
        for i in coords
            qp = copy(q); qp[i] += h; qm = copy(q); qm[i] -= h
            fd = (_rbm_reference(qp, x, y, K, J).posterior -
                  _rbm_reference(qm, x, y, K, J).posterior) / (2h)
            @test isapprox(g[i], fd; rtol = 1e-4, atol = 1e-3)
        end
    end
end
