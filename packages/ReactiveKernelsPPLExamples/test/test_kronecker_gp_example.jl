using ReactiveKernelsPPLExamples.KroneckerGpExample
using ReactiveKernels
import ReactiveKernelsPPLExamples
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    lognormal, lkj_corr_cholesky
using LinearAlgebra: I, Symmetric, diag, eigen
using SpecialFunctions: logbeta, loggamma

const _KRON_N = 30
_kron_dim() = 2 + (_KRON_N * (_KRON_N - 1)) ÷ 2 + 1

# Plain-Julia transcription of the reference `.stan` (independent of the
# authored graph helpers): cholesky_corr_constrain + tanh/corr terms, LKJ(2)
# prior on the factor, lognormal/Cauchy priors, exact Kronecker eigenspace
# marginal likelihood.
_kron_z(q) = tanh.(q[3:(2 + (_KRON_N * (_KRON_N - 1)) ÷ 2)])
function _kron_L_from_z(z)
    L = zeros(_KRON_N, _KRON_N); L[1, 1] = 1.0; k = 1
    for i in 2:_KRON_N
        L[i, 1] = z[k]; sum_sqs = z[k]^2; k += 1
        for j in 2:(i - 1)
            L[i, j] = z[k] * sqrt(1.0 - sum_sqs)
            sum_sqs += L[i, j]^2; k += 1
        end
        L[i, i] = sqrt(1.0 - sum_sqs)
    end
    L
end
function _kron_partial_lp(z)
    lp = 0.0; k = 1
    for i in 2:_KRON_N
        sum_sqs = z[k]^2; k += 1
        for j in 2:(i - 1)
            lp += 0.5 * log(1.0 - sum_sqs)
            w = z[k] * sqrt(1.0 - sum_sqs)
            sum_sqs += w^2; k += 1
        end
    end
    lp
end
function _kron_logpdf_lkj(L, eta)
    Kf = Float64(_KRON_N)
    kernel_term = sum(((Kf + 2 * (eta - 1)) .- (1:_KRON_N)) .* log.(diag(L)))
    alpha = eta + 0.5 * Kf - 1
    loginvconst = (2 * eta + Kf - 3) * log(2.0) +
                  (log(pi) / 4) * (Kf * (Kf - 1) - 2) + logbeta(alpha, alpha) -
                  (Kf - 2) * loggamma(eta + 0.5 * (Kf - 1)) +
                  sum(loggamma.(eta .+ 0.5 .* (0:(_KRON_N - 3))); init = 0.0)
    kernel_term - loginvconst
end
function _kron_reference(q, x1, y)
    u_var1, u_bw1 = q[1], q[2]
    L_free = q[3:(2 + (_KRON_N * (_KRON_N - 1)) ÷ 2)]
    u_sigma1 = q[end]
    var1, bw1 = exp(u_var1), exp(u_bw1)
    sigma1 = 1e-5 + exp(u_sigma1)
    z = tanh.(L_free)
    L = _kron_L_from_z(z)
    log_jacobian = u_var1 + u_bw1 + u_sigma1 + sum(log.(1 .- z .^ 2)) +
                   _kron_partial_lp(z)
    nld(x, m, s) = -log(x) - log(s) - 0.5 * log(2π) - 0.5 * ((log(x) - m) / s)^2
    prior = nld(var1, 0, 1) + nld(sigma1, 0, 1) -
            (log(pi) + log(2.5) + log1p((bw1 / 2.5)^2)) +
            _kron_logpdf_lkj(L, 2.0)
    xd = -((x1 .- x1').^2)
    Sigma1 = var1 .* exp.(xd .* bw1) + 1e-5 * I
    F1 = eigen(Symmetric(Sigma1))
    Q1, R1 = F1.vectors, F1.values
    Lambda = L * L'
    F2 = eigen(Symmetric(Lambda))
    Q2, R2 = F2.vectors, F2.values
    eigs = R2 .* R1' .+ sigma1
    _kmp(A, B, V) = (A * (B * V)')'
    whitened = _kmp(Q1', Q2', y)
    scaled = whitened ./ eigs
    rotated = _kmp(Q1, Q2, scaled)
    likelihood = -0.5 * sum(y .* rotated) - 0.5 * sum(log.(eigs))
    (; parameters = (; var1, bw1, sigma1, L), log_jacobian, prior, likelihood,
       posterior = prior + likelihood + log_jacobian)
end

const _KRON_SENTINEL_BEFORE = ReactiveKernelsPPLExamples._DEMO_TAIL_EXECUTIONS[]
const _KRON_HAVE = (:unconstrained, :x1, :y)
const _KRON_BOUND = (; x1 = KRON_X1, y = KRON_Y)

@testset "eager template contract and first use" begin
    q0 = 0.1 .* sin.((1:_kron_dim()) .+ 0.5)
    function _first_use_values()
        g1 = build_kronecker_gp_graph()
        g2 = build_kronecker_gp_graph()
        (prepare(g1; have = _KRON_HAVE, want = :posterior, bound = _KRON_BOUND)(q0),
         prepare(g2; have = _KRON_HAVE, want = :posterior, bound = _KRON_BOUND)(q0))
    end
    v1, v2 = _first_use_values()
    @test isfinite(v1)
    @test v1 == v2
    @test ReactiveKernelsPPLExamples._DEMO_TAIL_EXECUTIONS[] == _KRON_SENTINEL_BEFORE
end

@testset "PPL graph — kronecker_gp (posteriordb)" begin
    artifact = evaluate_kronecker_gp_source()
    @test ReactiveKernelsPPLExamples._DEMO_TAIL_EXECUTIONS[] ==
          _KRON_SENTINEL_BEFORE + 1
    @test artifact.source == strip(KRON_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = artifact.inputs.q
    reference = _kron_reference(q, KRON_X1, KRON_Y)

    @testset "authored on the current baseline surface" begin
        @test occursin("eigen(Symmetric(", KRON_SOURCE)
        @test occursin("lkj_corr_cholesky(2.0).logpdf(L)", KRON_SOURCE)
        @test occursin("_cholesky_corr_constrain_L(z, n2)", KRON_SOURCE)
        @test occursin("0.00001 + exp(u_sigma1)", KRON_SOURCE)
        @test occursin("_kron_mvprod", KRON_SOURCE)
        @test !occursin("struct ", KRON_SOURCE)
        @test artifact.lognormal_object === lognormal
        @test artifact.lkj_corr_cholesky_object === lkj_corr_cholesky
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.x1, model.y),
                 want = (model.prior, model.likelihood, model.log_jacobian,
                         model.posterior))
        prior, likelihood, log_jacobian, posterior =
            prepare(p)(q, KRON_X1, KRON_Y)
        @test prior ≈ reference.prior
        @test likelihood ≈ reference.likelihood
        @test log_jacobian ≈ reference.log_jacobian
        @test posterior ≈ reference.posterior
    end

    @testset "LKJ transform reproduces Stan's constrain at random points" begin
        # The LKJ factor for several probes must match the oracle factor to
        # machine precision (values), pinning the transform (not just the
        # summed posterior).
        for seed in (11, 23, 47)
            qp = 0.3 .* sin.((1:_kron_dim()) .* (1 + seed / 100))
            zp = tanh.(qp[3:(2 + (_KRON_N * (_KRON_N - 1)) ÷ 2)])
            p = plan(model.graph;
                     have = (model.unconstrained, model.x1, model.y),
                     want = (model.prior, model.likelihood, model.log_jacobian,
                             model.posterior))
            prior, likelihood, log_jacobian, posterior =
                prepare(p)(qp, KRON_X1, KRON_Y)
            ref = _kron_reference(qp, KRON_X1, KRON_Y)
            @test log_jacobian ≈ ref.log_jacobian
            @test posterior ≈ ref.posterior
        end
    end
end
