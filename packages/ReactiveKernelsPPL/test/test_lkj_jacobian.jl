# LKJ Jacobian contract: lkj_corr_cholesky_logpdf (Stan-verbatim, a density
# w.r.t. the correlation-matrix volume element) + lkj_chol_logjac must equal
# (eta-1)*logdet(Omega) + log|d vech(Omega)/du| up to a u-independent const.
# Regression test for the (i-1-j) vs (i-j) Gram-exponent bug: the old
# hyperspherical (sphere-volume) exponent made K=2 theta uniform (arcsine rho)
# instead of sine-weighted (uniform rho at eta=1); posterior parity vs SBBRMI
# failed on L.2.1/L.2.2 with KS_D 0.11/0.21 (study 2026-09-19).
using LinearAlgebra: det
using Test

_npacked(K) = K * (K - 1) ÷ 2

function _vech_Omega(u, K)
    L = lkj_chol_constrain(u, K)
    Om = L * L'
    return [Om[i, j] for i in 2:K for j in 1:i-1]
end

function _logabsdet_vech_jac(u, K; h = cbrt(eps(Float64)))
    p = length(u)
    p == 0 && return 0.0
    J = Matrix{Float64}(undef, p, p)
    for j in 1:p
        up = copy(u); up[j] += h
        dn = copy(u); dn[j] -= h
        J[:, j] = (_vech_Omega(up, K) .- _vech_Omega(dn, K)) ./ (2h)
    end
    return log(abs(det(J)))
end

@testset "lkj jacobian matches the Omega volume element" begin
    for K in (2, 3), eta in (1.0, 2.0)
        base = collect(range(-1.2, 1.2; length = max(_npacked(K), 2)))[1:_npacked(K)]
        us = [clamp.(base .+ 0.37 * (k - 2) .+ 0.11 * collect(1:_npacked(K)) .* (k - 1),
                -3.0, 3.0) for k in 1:6]
        res = map(us) do u
            L = lkj_chol_constrain(u, K)
            lkj_corr_cholesky_logpdf(L, eta) + lkj_chol_logjac(u, K) -
                (eta - 1) * log(det(L * L')) - _logabsdet_vech_jac(u, K)
        end
        @test maximum(res) - minimum(res) < 1e-7
    end
end

@testset "lkj jacobian K=2 closed form" begin
    # theta = pi*sigmoid(t); the (i-j) = 1 exponent keeps one log-sin term.
    t = 0.7
    s = 1.0 / (1.0 + exp(-t))
    @test lkj_chol_logjac([t], 2) ≈
        log(sin(pi * s)) + log(pi) + log(s) + log1p(-s)
end
