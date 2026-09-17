using ReactiveKernels
using ReactiveKernelsPPLExamples.IohmmRegExample
using DifferentiationInterface
using LogExpFunctions: logsumexp
import Enzyme

const _IOHMM_AE = AutoEnzyme(mode = Enzyme.Reverse, function_annotation = Enzyme.Const)

function _iohmm_simplex(free)
    N = length(free); K = N + 1
    z = zeros(K); sw = 0.0
    for i in N:-1:1
        n = float(i); w = free[i] / sqrt(n * (n + 1))
        sw += w; z[i] += sw; z[i + 1] -= w * n
    end
    lse = logsumexp(z); zz = exp.(z .- lse)
    (zz, sum(log.(zz)) + 0.5 * log(K))
end

# Independent oracle using the LITERAL Stan forward accumulator (double loop with
# the transition indexed by the previous state i), NOT the graph's scan/eachrow.
function _iohmm_reference(q, y, u, K)
    M = size(u, 2)
    pi1, jpi = _iohmm_simplex(q[1:(K - 1)])
    w_flat = q[K:(K - 1 + K * M)]
    b_flat = q[(K + K * M):(K - 1 + 2K * M)]
    u_sigma = q[(K + 2K * M):(2K * M + 2K - 1)]
    sigma = exp.(u_sigma); jsig = sum(u_sigma)
    jac = jpi + jsig
    cw = -0.5 * log(2π) - log(5.0); cs = -0.5 * log(2π) - log(3.0)
    prior = sum(cw .- 0.5 .* (w_flat ./ 5.0) .^ 2) +
            sum(cw .- 0.5 .* (b_flat ./ 5.0) .^ 2) +
            sum(cs .- 0.5 .* (sigma ./ 3.0) .^ 2)
    W = reshape(w_flat, M, K); Breg = reshape(b_flat, M, K)
    T = length(y)
    unA = u * W
    rowlse = [logsumexp(unA[t, :]) for t in 1:T]
    logA = unA .- rowlse
    means = u * Breg
    logoblik = [(-0.5 * log(2π) - log(sigma[j]) - 0.5 * ((y[t] - means[t, j]) / sigma[j])^2)
                for t in 1:T, j in 1:K]
    gamma = log.(pi1) .+ logoblik[1, :]
    for t in 2:T
        newg = similar(gamma)
        for j in 1:K
            acc = [gamma[i] + logA[t, i] + logoblik[t, j] for i in 1:K]
            newg[j] = logsumexp(acc)
        end
        gamma = newg
    end
    lik = logsumexp(gamma)
    (; pi1, sigma, prior, log_jacobian = jac, likelihood = lik,
       posterior = prior + lik + jac)
end

function _iohmm_fd(f, q; h = 1e-6)
    g = similar(q)
    for i in eachindex(q)
        qp = copy(q); qp[i] += h; qm = copy(q); qm[i] -= h
        g[i] = (f(qp) - f(qm)) / (2h)
    end
    g
end

@testset "PPL graph — iohmm_reg" begin
    artifact = evaluate_iohmm_reg_source()
    @test artifact.source == strip(IOHMM_REG_SOURCE, '\n')
    @test artifact.output == Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    y = IOHMM_REG_Y
    u = IOHMM_REG_U
    K = IOHMM_REG_K
    dim = size(u, 2) * K * 2 + 2K - 1
    q = 0.15 .* collect(range(-1.0, 1.0; length = dim))

    @testset "authored on the current baseline surface" begin
        @test occursin("scan(eachrow(R)", IOHMM_REG_SOURCE)
        @test occursin("softmax", IOHMM_REG_SOURCE) || occursin("logsumexp", IOHMM_REG_SOURCE)
        @test occursin("u * W", IOHMM_REG_SOURCE)             # input-dependent transition design
        @test !occursin("struct ", IOHMM_REG_SOURCE)
    end

    @testset "density decomposition vs the independent reference oracle (literal Stan forward)" begin
        p = prepare(model; have = (:unconstrained, :y, :u, :K),
            want = (:pi1, :sigma, :prior, :log_jacobian, :likelihood, :posterior),
            bound = (; y, u, K))
        pi1, sigma, prior, log_jacobian, likelihood, posterior = p(q)
        ref = _iohmm_reference(q, y, u, K)
        @test pi1 ≈ ref.pi1
        @test sigma ≈ ref.sigma
        @test isapprox(sum(pi1), 1.0; atol = 1e-12)
        @test all(>(0.0), sigma)
        @test prior ≈ ref.prior
        @test log_jacobian ≈ ref.log_jacobian
        @test likelihood ≈ ref.likelihood
        @test posterior ≈ ref.posterior
    end

    @testset "native primal + plain-Enzyme gradient vs finite differences" begin
        pk = prepare(model; have = (:unconstrained, :y, :u, :K), want = :posterior,
            bound = (; y, u, K))
        @test pk(q) ≈ _iohmm_reference(q, y, u, K).posterior
        prep = prepare_ad(pk, _IOHMM_AE, q; active = :unconstrained)
        g = ReactiveKernels.ad_value_and_gradient!(prep, similar(q), q)[2]
        @test all(isfinite, g)
        gfd = _iohmm_fd(qq -> _iohmm_reference(qq, y, u, K).posterior, q)
        @test g ≈ gfd rtol = 1e-5
        @test pk(q) == pk(q)
    end
end
