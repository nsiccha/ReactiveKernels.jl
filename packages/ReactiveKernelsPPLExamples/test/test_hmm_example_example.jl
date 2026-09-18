using ReactiveKernels
using ReactiveKernelsPPLExamples.HmmExampleExample
using DifferentiationInterface
using LogExpFunctions: logsumexp
import Enzyme

const _HMME_AE = AutoEnzyme(mode = Enzyme.Reverse, function_annotation = Enzyme.Const)

_hmme_normal(x, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - m) / s)^2

# Independent Stan-2.39 constraint oracles (NOT the graph's B-matrix form): the
# online sum_to_zero_constrain, softmax, and positive_ordered_constrain.
function _hmme_simplex(free)
    N = length(free); K = N + 1
    z = zeros(K)
    sw = 0.0
    for i in N:-1:1
        n = float(i)
        w = free[i] / sqrt(n * (n + 1))
        sw += w
        z[i] += sw
        z[i + 1] -= w * n
    end
    lse = logsumexp(z)
    zz = exp.(z .- lse)
    (zz, sum(log.(zz)) + 0.5 * log(K))
end
function _hmme_posord(u)
    K = length(u); mu = similar(u)
    mu[1] = exp(u[1])
    for k in 2:K
        mu[k] = mu[k - 1] + exp(u[k])
    end
    (mu, sum(u))
end

function _hmme_reference(q, y, K)
    theta1, j1 = _hmme_simplex(q[1:(K - 1)])
    theta2, j2 = _hmme_simplex(q[K:(2K - 2)])
    mu, jm = _hmme_posord(q[(2K - 1):(3K - 2)])
    jac = j1 + j2 + jm
    prior = _hmme_normal(mu[1], 3.0, 1.0) + _hmme_normal(mu[2], 10.0, 1.0)
    theta = permutedims(hcat(theta1, theta2))          # theta[j,k]
    N = length(y)
    gamma = [_hmme_normal(y[1], mu[k], 1.0) for k in 1:K]
    for t in 2:N
        newg = similar(gamma)
        for k in 1:K
            acc = [gamma[j] + log(theta[j, k]) for j in 1:K]
            newg[k] = logsumexp(acc) + _hmme_normal(y[t], mu[k], 1.0)
        end
        gamma = newg
    end
    lik = logsumexp(gamma)
    (; theta1, theta2, mu, prior, log_jacobian = jac, likelihood = lik,
       posterior = prior + lik + jac)
end

function _hmme_fd(f, q; h = 1e-6)
    g = similar(q)
    for i in eachindex(q)
        qp = copy(q); qp[i] += h
        qm = copy(q); qm[i] -= h
        g[i] = (f(qp) - f(qm)) / (2h)
    end
    g
end

@testset "PPL graph — hmm_example" begin
    artifact = evaluate_hmm_example_source()
    @test artifact.source == strip(HMM_EXAMPLE_SOURCE, '\n')
    @test artifact.output == Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    y = HMM_EXAMPLE_Y
    K = HMM_EXAMPLE_K
    q = [0.4, -0.3, log(3.0), log(7.0)]

    @testset "authored on the current baseline surface" begin
        @test occursin("scan(y[2:T]", HMM_EXAMPLE_SOURCE)
        @test occursin("bound = (; K))", HMM_EXAMPLE_SOURCE)
        @test occursin("mapslices(logsumexp", HMM_EXAMPLE_SOURCE)
        @test occursin("normal(3.0, 1.0).logpdf", HMM_EXAMPLE_SOURCE)
        @test !occursin("struct ", HMM_EXAMPLE_SOURCE)
    end

    @testset "density decomposition vs the independent reference oracle" begin
        p = prepare(model; have = (:unconstrained, :y, :K),
            want = (:parameters, :prior, :log_jacobian, :likelihood, :posterior),
            bound = (; y, K))
        parameters, prior, log_jacobian, likelihood, posterior = p(q)
        ref = _hmme_reference(q, y, K)
        @test parameters.theta1 ≈ ref.theta1
        @test parameters.theta2 ≈ ref.theta2
        @test parameters.mu ≈ ref.mu
        @test isapprox(sum(parameters.theta1), 1.0; atol = 1e-12)
        @test issorted(parameters.mu)
        @test prior ≈ ref.prior
        @test log_jacobian ≈ ref.log_jacobian
        @test likelihood ≈ ref.likelihood
        @test posterior ≈ ref.posterior
    end

    @testset "native primal + plain-Enzyme gradient vs finite differences" begin
        pk = prepare(model; have = (:unconstrained, :y, :K), want = :posterior,
            bound = (; y, K))
        @test pk(q) ≈ _hmme_reference(q, y, K).posterior
        prep = prepare_ad(pk, _HMME_AE, q; active = :unconstrained)
        g = ReactiveKernels.ad_value_and_gradient!(prep, similar(q), q)[2]
        @test all(isfinite, g)
        gfd = _hmme_fd(qq -> _hmme_reference(qq, y, K).posterior, q)
        @test g ≈ gfd rtol = 1e-5
        @test pk(q) == pk(q)
    end
end
