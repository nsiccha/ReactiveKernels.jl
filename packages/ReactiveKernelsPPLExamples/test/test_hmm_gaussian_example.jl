using ReactiveKernels
using ReactiveKernelsPPLExamples.HmmGaussianExample
using DifferentiationInterface
using LogExpFunctions: logsumexp
import Enzyme

const _HMMG_AE = AutoEnzyme(mode = Enzyme.Reverse, function_annotation = Enzyme.Const)

_hmmg_normal(x, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - m) / s)^2

function _hmmg_simplex(free)
    N = length(free); K = N + 1
    z = zeros(K); sw = 0.0
    for i in N:-1:1
        n = float(i); w = free[i] / sqrt(n * (n + 1))
        sw += w; z[i] += sw; z[i + 1] -= w * n
    end
    lse = logsumexp(z); zz = exp.(z .- lse)
    (zz, sum(log.(zz)) + 0.5 * log(K))
end
function _hmmg_ordered(u)          # ordered: mu[1]=u[1], mu[k]=mu[k-1]+exp(u[k])
    K = length(u); mu = similar(u)
    mu[1] = u[1]
    for k in 2:K
        mu[k] = mu[k - 1] + exp(u[k])
    end
    (mu, sum(u[2:K]))
end

function _hmmg_reference(q, y, K)
    pi1, jpi = _hmmg_simplex(q[1:(K - 1)])
    Arows = Vector{Vector{Float64}}(); jA = 0.0
    for i in 1:K
        lo = K + (i - 1) * (K - 1)
        Ai, ji = _hmmg_simplex(q[lo:(lo + K - 2)])
        push!(Arows, Ai); jA += ji
    end
    mu, jmu = _hmmg_ordered(q[(K * K):(K * K + K - 1)])
    u_sigma = q[(K * K + K):(K * K + 2K - 1)]
    sigma = exp.(u_sigma); jsig = sum(u_sigma)
    jac = jpi + jA + jmu + jsig
    Amat = permutedims(hcat(Arows...))          # Amat[i,j] = A[i][j]
    logA = log.(Amat)
    N = length(y)
    emit(t) = [_hmmg_normal(y[t], mu[k], sigma[k]) for k in 1:K]
    gamma = log.(pi1) .+ sum(emit(1))           # t=1 quirk: full emission sum
    for t in 2:N
        e = emit(t); newg = similar(gamma)
        for j in 1:K
            acc = [gamma[i] + logA[i, j] for i in 1:K]
            newg[j] = logsumexp(acc) + e[j]
        end
        gamma = newg
    end
    lik = logsumexp(gamma)
    (; pi1, A = Amat, mu, sigma, log_jacobian = jac, likelihood = lik,
       posterior = lik + jac)
end

function _hmmg_fd(f, q; h = 1e-6)
    g = similar(q)
    for i in eachindex(q)
        qp = copy(q); qp[i] += h; qm = copy(q); qm[i] -= h
        g[i] = (f(qp) - f(qm)) / (2h)
    end
    g
end

@testset "PPL graph — hmm_gaussian" begin
    artifact = evaluate_hmm_gaussian_source()
    @test artifact.source == strip(HMM_GAUSSIAN_SOURCE, '\n')
    @test artifact.output == Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    y = HMM_GAUSSIAN_Y
    K = HMM_GAUSSIAN_K
    q = 0.2 .* collect(range(-1.0, 1.0; length = K * K + 2K - 1))

    @testset "authored on the current baseline surface" begin
        @test occursin("reshape(A_free", HMM_GAUSSIAN_SOURCE)
        @test occursin("scan(y[2:T]", HMM_GAUSSIAN_SOURCE)
        @test occursin("bound = (; K))", HMM_GAUSSIAN_SOURCE)
        @test occursin("logpi1 .+ sum(emit1)", HMM_GAUSSIAN_SOURCE)   # t=1 quirk
        @test !occursin("struct ", HMM_GAUSSIAN_SOURCE)
    end

    @testset "density decomposition vs the independent reference oracle" begin
        p = prepare(model; have = (:unconstrained, :y, :K),
            want = (:parameters, :log_jacobian, :likelihood, :posterior),
            bound = (; y, K))
        parameters, log_jacobian, likelihood, posterior = p(q)
        ref = _hmmg_reference(q, y, K)
        @test parameters.pi1 ≈ ref.pi1
        @test parameters.A ≈ ref.A
        @test parameters.mu ≈ ref.mu
        @test parameters.sigma ≈ ref.sigma
        @test issorted(parameters.mu)                       # ordered
        @test all(isapprox.(sum(parameters.A; dims = 2), 1.0; atol = 1e-12))
        @test log_jacobian ≈ ref.log_jacobian
        @test likelihood ≈ ref.likelihood
        @test posterior ≈ ref.posterior
    end

    @testset "native primal + plain-Enzyme gradient vs finite differences" begin
        pk = prepare(model; have = (:unconstrained, :y, :K), want = :posterior,
            bound = (; y, K))
        @test pk(q) ≈ _hmmg_reference(q, y, K).posterior
        prep = prepare_ad(pk, _HMMG_AE, q; active = :unconstrained)
        g = ReactiveKernels.ad_value_and_gradient!(prep, similar(q), q)[2]
        @test all(isfinite, g)
        gfd = _hmmg_fd(qq -> _hmmg_reference(qq, y, K).posterior, q)
        @test g ≈ gfd rtol = 1e-5
        @test pk(q) == pk(q)
    end
end
