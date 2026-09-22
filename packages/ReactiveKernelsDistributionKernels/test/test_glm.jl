# Fused GLM objects: values vs independent oracles, endpoint consistency,
# and analytic-adjoint correctness vs finite differences (generic Enzyme
# through the object kernels — no hand-written rules, 2026-09-21 policy).
using ADTypes
using DifferentiationInterface
using Distributions: Bernoulli, Binomial, Categorical, NegativeBinomial, Normal, Poisson, logpdf
using Enzyme
using LinearAlgebra: dot
using LogExpFunctions: log1pexp, softmax
import LogExpFunctions
using SpecialFunctions: loggamma
using ReactiveKernels: @kernel, code_expr, extract, prepare,
    prepare_ad, ad_value_and_gradient!
using ReactiveKernelsDistributionKernels: DistributionKernelSources
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    BERNOULLI_LOGIT_GLM_KERNEL_SOURCE, bernoulli_logit_glm,
    POISSON_LOG_GLM_KERNEL_SOURCE, poisson_log_glm,
    NORMAL_ID_GLM_KERNEL_SOURCE, normal_id_glm,
    BINOMIAL_LOGIT_GLM_KERNEL_SOURCE, binomial_logit_glm,
    NEG_BINOMIAL_2_LOG_GLM_KERNEL_SOURCE, neg_binomial_2_log_glm,
    CATEGORICAL_LOGIT_GLM_KERNEL_SOURCE, categorical_logit_glm,
    ORDERED_LOGISTIC_GLM_KERNEL_SOURCE, ordered_logistic_glm
using Test

function _glm_fd_grad(f, q)
    g = similar(q, Float64)
    h = 1e-7
    for i in eachindex(q)
        qp = copy(q); qp[i] += h
        qm = copy(q); qm[i] -= h
        g[i] = (f(qp) - f(qm)) / (2h)
    end
    return g
end

@testset "bernoulli_logit_glm fused object" begin
    @test occursin("@kernel bernoulli_logit_glm(", BERNOULLI_LOGIT_GLM_KERNEL_SOURCE)
    @test all(hasproperty(bernoulli_logit_glm, ep) for ep in (:logpdf, :pointwise, :score))

    X = [1.0 -1.5; 1.0 -0.5; 1.0 0.5; 1.0 1.5; 1.0 0.0; 1.0 2.0]
    beta = [0.2, 0.6]
    eta = X * beta
    y = [0, 1, 1, 0, 1, 0]
    ref_cells = [logpdf(Bernoulli(LogExpFunctions.logistic(e)), yi == 1) for (e, yi) in zip(eta, y)]
    ref_total = sum(ref_cells)
    ref_score = [(yi - LogExpFunctions.logistic(e)) for (e, yi) in zip(eta, y)]

    @testset "endpoints match the independent oracle" begin
        @test prepare(bernoulli_logit_glm.logpdf;
            have = (:y, :X, :beta), want = :logpdf)(y, X, beta) ≈ ref_total
        @test prepare(bernoulli_logit_glm.pointwise;
            have = (:y, :X, :beta), want = :pointwise)(y, X, beta) ≈ ref_cells
        @test prepare(bernoulli_logit_glm.score;
            have = (:y, :X, :beta), want = :score)(y, X, beta) ≈ ref_score
        pw = prepare(bernoulli_logit_glm.pointwise;
            have = (:y, :X, :beta), want = :pointwise)(y, X, beta)
        ll = prepare(bernoulli_logit_glm.logpdf;
            have = (:y, :X, :beta), want = :logpdf)(y, X, beta)
        @test sum(pw) ≈ ll
    end

    @testset "eta / mu / working weights extract as owner nodes" begin
        @test prepare(extract(bernoulli_logit_glm;
            have = (:X, :beta), want = :eta))(X, beta) ≈ eta
        mu = LogExpFunctions.logistic.(eta)
        @test prepare(extract(bernoulli_logit_glm;
            have = (:X, :beta), want = :mu))(X, beta) ≈ mu
        @test prepare(extract(bernoulli_logit_glm;
            have = (:X, :beta), want = :working_weights))(X, beta) ≈ mu .* (1 .- mu)
    end

    @testset "Bool/Float responses convert at the host boundary" begin
        # The endpoint takes Vector{Int} (nested splicing requires exact port
        # types); conversion happens host-side, outside the kernel.
        k = prepare(bernoulli_logit_glm.logpdf; have = (:y, :X, :beta), want = :logpdf)
        @test k(Int.(y .== 1), X, beta) ≈ ref_total
        @test k(Int.(Float64.(y)), X, beta) ≈ ref_total
    end

    @testset "saturating tails stay finite and match the cutoff form" begin
        # Stan's fused form (cutoff-20 Taylor tails), written independently.
        stan_cell(yθ) = (em = exp(-yθ);
            yθ > 20 ? -em : yθ < -20 ? yθ : -log1p(em))
        for et in (-30.0, -21.0, -20.0, 20.0, 21.0, 30.0), yi in (0, 1)
            yθ = (2 * yi - 1) * et
            @test prepare(bernoulli_logit_glm.pointwise;
                have = (:y, :X, :beta), want = :pointwise)(
                    [yi], [1.0 et], [0.0, 1.0])[1] ≈ stan_cell(yθ)
        end
    end

    @testset "score tails match Stan's cutoff theta" begin
        # The score endpoint implements the cutoff branches directly, so
        # this is the branches' exact-form proof (the FD check below cannot
        # resolve 1e-9-scale tail contributions).
        stan_theta(s, yθ) = (em = exp(-yθ);
            s * (yθ > 20 ? em : yθ < -20 ? 1.0 : em / (em + 1.0)))
        ks = prepare(bernoulli_logit_glm.score;
            have = (:y, :X, :beta), want = :score)
        for et in (-30.0, -21.0, -20.0, -5.0, 0.0, 5.0, 20.0, 21.0, 30.0),
                yi in (0, 1)
            yθ = (2 * yi - 1) * et
            got = ks([yi], [1.0 et], [0.0, 1.0])[1]
            @test got ≈ stan_theta(2 * yi - 1, yθ)
        end
    end

    @testset "lowered code is self-contained (no helper escapes)" begin
        k = prepare(bernoulli_logit_glm.logpdf; have = (:y, :X, :beta), want = :logpdf)
        sk = string(code_expr(k))
        @test !occursin("_glm_", sk)
        @test !occursin("Distributions", sk)
    end

    @testset "analytic adjoint matches finite differences" begin
        # The object splices with X/beta as implicit named dependencies; the
        # whole logpdf is one fused kernel node (see the ops test above), so
        # this FD check validates the generic-Enzyme adjoint end to end.
        @kernel _bern_glm_model(
                beta::Vector{Float64}, X::Matrix{Float64},
                y::Vector{Int}) = begin
            ll::Float64 = bernoulli_logit_glm.logpdf(y)
            prior::Float64 = -0.5 * dot(beta, beta)
            posterior::Float64 = prior + ll
            return posterior
        end
        q = [0.25, -0.4]
        oracle(qq) = (et = X * qq;
            -0.5 * dot(qq, qq) +
            sum(logpdf(Bernoulli(LogExpFunctions.logistic(e)), yi == 1)
                for (e, yi) in zip(et, y)))
        kb = prepare(_bern_glm_model;
            have = (:beta, :X, :y), want = :posterior, bound = (; X = X, y = y))
        @test kb(q) ≈ oracle(q)
        prep = prepare_ad(kb,
            AutoEnzyme(mode = Enzyme.Reverse, function_annotation = Enzyme.Const),
            q; active = :beta)
        v, g = ad_value_and_gradient!(prep, similar(q), q)
        gref = _glm_fd_grad(oracle, q)
        @test g ≈ gref rtol = 1e-6
        @test v ≈ kb(q) rtol = 1e-12
        # Saturating tails exercise the kernel's cutoff branches too. FD runs
        # against the kernel primal itself here, NOT the Distributions oracle:
        # the oracle's probability form (`Bernoulli(LogExpFunctions.logistic(e))`) suffers
        # 1-p cancellation at saturating eta, and FD amplifies that ~1e-7
        # noise to O(1) — the cutoff form under test has no such noise.
        qtail = [3.0, -12.0]
        _, gt = ad_value_and_gradient!(prep, similar(qtail), qtail)
        @test gt ≈ _glm_fd_grad(kb, qtail) rtol = 1e-5
    end
end

@testset "poisson_log_glm fused object" begin
    @test occursin("@kernel poisson_log_glm(", POISSON_LOG_GLM_KERNEL_SOURCE)
    @test all(hasproperty(poisson_log_glm, ep) for ep in (:logpdf, :pointwise, :score))

    X = [1.0 -1.5; 1.0 -0.5; 1.0 0.5; 1.0 1.5; 1.0 0.0; 1.0 2.0]
    beta = [0.5, 0.3]
    eta = X * beta
    y = [1, 2, 3, 1, 0, 4]
    ref_cells = [logpdf(Poisson(exp(e)), yi) for (e, yi) in zip(eta, y)]
    ref_total = sum(ref_cells)
    ref_score = [yi - exp(e) for (e, yi) in zip(eta, y)]

    @testset "endpoints match the independent oracle" begin
        @test prepare(poisson_log_glm.logpdf;
            have = (:y, :X, :beta), want = :logpdf)(y, X, beta) ≈ ref_total
        @test prepare(poisson_log_glm.pointwise;
            have = (:y, :X, :beta), want = :pointwise)(y, X, beta) ≈ ref_cells
        @test prepare(poisson_log_glm.score;
            have = (:y, :X, :beta), want = :score)(y, X, beta) ≈ ref_score
        pw = prepare(poisson_log_glm.pointwise;
            have = (:y, :X, :beta), want = :pointwise)(y, X, beta)
        ll = prepare(poisson_log_glm.logpdf;
            have = (:y, :X, :beta), want = :logpdf)(y, X, beta)
        @test sum(pw) ≈ ll
    end

    @testset "eta / mu / working weights extract as owner nodes" begin
        @test prepare(extract(poisson_log_glm;
            have = (:X, :beta), want = :eta))(X, beta) ≈ eta
        mu = exp.(eta)
        @test prepare(extract(poisson_log_glm;
            have = (:X, :beta), want = :mu))(X, beta) ≈ mu
        @test prepare(extract(poisson_log_glm;
            have = (:X, :beta), want = :working_weights))(X, beta) ≈ mu
    end

    @testset "tails and zero counts" begin
        kp = prepare(poisson_log_glm.pointwise;
            have = (:y, :X, :beta), want = :pointwise)
        for et in (-30.0, -10.0, -1.0, 0.0, 1.0, 5.0, 10.0, 20.0),
                yi in (0, 1, 5)
            @test kp([yi], [1.0 et], [0.0, 1.0])[1] ≈
                yi * et - exp(et) - loggamma(yi + 1)
        end
    end

    @testset "normalizer folds out of bound queries" begin
        ku = prepare(poisson_log_glm.logpdf;
            have = (:y, :X, :beta), want = :logpdf, bound = (; X = X))
        sku = string(code_expr(ku))
        @test occursin("logpdf__cterm", sku) && occursin(")(y)", sku)
        kb = prepare(poisson_log_glm.logpdf;
            have = (:y, :X, :beta), want = :logpdf, bound = (; X = X, y = y))
        skb = string(code_expr(kb))
        @test occursin("logpdf__cterm", skb) && !occursin(")(y)", skb)
        @test !occursin("Distributions", skb)
    end

    @testset "lowered code is self-contained (no helper escapes)" begin
        k = prepare(poisson_log_glm.logpdf; have = (:y, :X, :beta), want = :logpdf)
        sk = string(code_expr(k))
        @test !occursin("_glm_", sk)
        @test !occursin("Distributions", sk)
    end

    @testset "analytic adjoint matches finite differences" begin
        @kernel _pois_glm_model(
                beta::Vector{Float64}, X::Matrix{Float64},
                y::Vector{Int}) = begin
            ll::Float64 = poisson_log_glm.logpdf(y)
            prior::Float64 = -0.5 * dot(beta, beta)
            posterior::Float64 = prior + ll
            return posterior
        end
        q = [0.4, -0.2]
        oracle(qq) = (et = X * qq;
            -0.5 * dot(qq, qq) +
            sum(logpdf(Poisson(exp(e)), yi) for (e, yi) in zip(et, y)))
        kb = prepare(_pois_glm_model;
            have = (:beta, :X, :y), want = :posterior, bound = (; X = X, y = y))
        @test kb(q) ≈ oracle(q)
        prep = prepare_ad(kb,
            AutoEnzyme(mode = Enzyme.Reverse, function_annotation = Enzyme.Const),
            q; active = :beta)
        v, g = ad_value_and_gradient!(prep, similar(q), q)
        @test g ≈ _glm_fd_grad(oracle, q) rtol = 1e-6
        @test v ≈ kb(q) rtol = 1e-12
        qtail = [2.0, 3.0]
        _, gt = ad_value_and_gradient!(prep, similar(qtail), qtail)
        @test gt ≈ _glm_fd_grad(kb, qtail) rtol = 1e-5
    end
end

@testset "normal_id_glm fused object" begin
    @test occursin("@kernel normal_id_glm(", NORMAL_ID_GLM_KERNEL_SOURCE)
    @test all(hasproperty(normal_id_glm, ep) for ep in (:logpdf, :pointwise, :score))

    X = [1.0 -1.5; 1.0 -0.5; 1.0 0.5; 1.0 1.5; 1.0 0.0; 1.0 2.0]
    beta = [0.5, -0.3]
    eta = X * beta
    sigma = 1.5
    y = eta .+ [0.2, -0.3, 0.1, 0.4, -0.2, 0.0]
    ref_cells = [logpdf(Normal(e, sigma), yi) for (e, yi) in zip(eta, y)]
    ref_total = sum(ref_cells)
    ref_score = [(yi - e) / sigma^2 for (e, yi) in zip(eta, y)]

    @testset "endpoints match the independent oracle" begin
        @test prepare(normal_id_glm.logpdf;
            have = (:y, :X, :beta, :sigma), want = :logpdf)(
            y, X, beta, sigma) ≈ ref_total
        @test prepare(normal_id_glm.pointwise;
            have = (:y, :X, :beta, :sigma), want = :pointwise)(
            y, X, beta, sigma) ≈ ref_cells
        @test prepare(normal_id_glm.score;
            have = (:y, :X, :beta, :sigma), want = :score)(
            y, X, beta, sigma) ≈ ref_score
        pw = prepare(normal_id_glm.pointwise;
            have = (:y, :X, :beta, :sigma), want = :pointwise)(
            y, X, beta, sigma)
        ll = prepare(normal_id_glm.logpdf;
            have = (:y, :X, :beta, :sigma), want = :logpdf)(
            y, X, beta, sigma)
        @test sum(pw) ≈ ll
    end

    @testset "eta / mu extract as owner nodes" begin
        @test prepare(extract(normal_id_glm;
            have = (:X, :beta), want = :eta))(X, beta) ≈ eta
        @test prepare(extract(normal_id_glm;
            have = (:X, :beta), want = :mu))(X, beta) ≈ eta
    end

    @testset "tight sigma and far residuals stay finite" begin
        kp = prepare(normal_id_glm.pointwise;
            have = (:y, :X, :beta, :sigma), want = :pointwise)
        for (sg, et, yi) in ((0.1, 0.0, 0.0), (0.1, 0.0, 0.5),
                (1.5, 20.0, 0.0), (5.0, -15.0, 2.0))
            r = yi - et
            @test kp([yi], [1.0 et], [0.0, 1.0], sg)[1] ≈
                -0.5 * r^2 / sg^2 - log(sg) - 0.5 * log(2π)
        end
    end

    @testset "lowered code is self-contained (no helper escapes)" begin
        k = prepare(normal_id_glm.logpdf;
            have = (:y, :X, :beta, :sigma), want = :logpdf)
        sk = string(code_expr(k))
        @test !occursin("_glm_", sk)
        @test !occursin("Distributions", sk)
    end

    @testset "analytic adjoint matches finite differences" begin
        @kernel _norm_glm_model(
                beta::Vector{Float64}, X::Matrix{Float64},
                y::Vector{Float64}, sigma::Float64) = begin
            ll::Float64 = normal_id_glm.logpdf(y)
            prior::Float64 = -0.5 * dot(beta, beta)
            posterior::Float64 = prior + ll
            return posterior
        end
        q = [0.4, -0.2]
        oracle(qq) = (et = X * qq;
            -0.5 * dot(qq, qq) +
            sum(logpdf(Normal(e, sigma), yi) for (e, yi) in zip(et, y)))
        kb = prepare(_norm_glm_model;
            have = (:beta, :X, :y, :sigma), want = :posterior,
            bound = (; X = X, y = y, sigma = sigma))
        @test kb(q) ≈ oracle(q)
        prep = prepare_ad(kb,
            AutoEnzyme(mode = Enzyme.Reverse, function_annotation = Enzyme.Const),
            q; active = :beta)
        v, g = ad_value_and_gradient!(prep, similar(q), q)
        @test g ≈ _glm_fd_grad(oracle, q) rtol = 1e-6
        @test v ≈ kb(q) rtol = 1e-12
    end

end

@testset "binomial_logit_glm fused object" begin
    @test occursin("@kernel binomial_logit_glm(", BINOMIAL_LOGIT_GLM_KERNEL_SOURCE)
    @test all(hasproperty(binomial_logit_glm, ep) for ep in (:logpdf, :pointwise, :score))

    X = [1.0 -1.5; 1.0 -0.5; 1.0 0.5; 1.0 1.5; 1.0 0.0; 1.0 2.0]
    beta = [0.5, 0.3]
    eta = X * beta
    N = [10, 8, 12, 10, 5, 20]
    y = [4, 3, 7, 10, 0, 15]
    ref_cells = [logpdf(Binomial(Ni, LogExpFunctions.logistic(e)), yi)
        for (e, Ni, yi) in zip(eta, N, y)]
    ref_total = sum(ref_cells)
    ref_score = [yi - Ni * LogExpFunctions.logistic(e) for (e, Ni, yi) in zip(eta, N, y)]

    @testset "endpoints match the independent oracle" begin
        @test prepare(binomial_logit_glm.logpdf;
            have = (:y, :X, :beta, :N), want = :logpdf)(y, X, beta, N) ≈ ref_total
        @test prepare(binomial_logit_glm.pointwise;
            have = (:y, :X, :beta, :N), want = :pointwise)(y, X, beta, N) ≈ ref_cells
        @test prepare(binomial_logit_glm.score;
            have = (:y, :X, :beta, :N), want = :score)(y, X, beta, N) ≈ ref_score
        pw = prepare(binomial_logit_glm.pointwise;
            have = (:y, :X, :beta, :N), want = :pointwise)(y, X, beta, N)
        ll = prepare(binomial_logit_glm.logpdf;
            have = (:y, :X, :beta, :N), want = :logpdf)(y, X, beta, N)
        @test sum(pw) ≈ ll
    end

    @testset "eta / mu / working weights extract as owner nodes" begin
        @test prepare(extract(binomial_logit_glm;
            have = (:X, :beta, :N), want = :eta))(X, beta, N) ≈ eta
        p = LogExpFunctions.logistic.(eta)
        @test prepare(extract(binomial_logit_glm;
            have = (:X, :beta, :N), want = :mu))(X, beta, N) ≈ N .* p
        @test prepare(extract(binomial_logit_glm;
            have = (:X, :beta, :N), want = :working_weights))(X, beta, N) ≈
            N .* p .* (1 .- p)
    end

    @testset "tails need no cutoff (log1pexp form is exact everywhere)" begin
        kp = prepare(binomial_logit_glm.pointwise;
            have = (:y, :X, :beta, :N), want = :pointwise)
        lchoose(n, k) = loggamma(n + 1) - loggamma(k + 1) - loggamma(n - k + 1)
        for et in (-100.0, -30.0, -10.0, -1.0, 0.0, 1.0, 10.0, 30.0, 100.0),
                (Ni, yi) in ((9, 0), (9, 1), (9, 8), (9, 9))
            @test kp([yi], [1.0 et], [0.0, 1.0], [Ni])[1] ≈
                yi * et - Ni * log1pexp(et) + lchoose(Ni, yi)
        end
    end

    @testset "normalizer folds out of bound queries" begin
        ku = prepare(binomial_logit_glm.logpdf;
            have = (:y, :X, :beta, :N), want = :logpdf, bound = (; X = X))
        sku = string(code_expr(ku))
        @test occursin("logpdf__lg", sku) && occursin(")(N, y)", sku)
        kb = prepare(binomial_logit_glm.logpdf;
            have = (:y, :X, :beta, :N), want = :logpdf,
            bound = (; X = X, y = y, N = N))
        skb = string(code_expr(kb))
        @test occursin("logpdf__lg", skb) && !occursin(")(N, y)", skb)
        @test !occursin("Distributions", skb)
    end

    @testset "lowered code is self-contained (no helper escapes)" begin
        k = prepare(binomial_logit_glm.logpdf;
            have = (:y, :X, :beta, :N), want = :logpdf)
        sk = string(code_expr(k))
        @test !occursin("_glm_", sk)
        @test !occursin("Distributions", sk)
    end

    @testset "analytic adjoint matches finite differences" begin
        @kernel _binom_glm_model(
                beta::Vector{Float64}, X::Matrix{Float64},
                y::Vector{Int}, N::Vector{Int}) = begin
            ll::Float64 = binomial_logit_glm.logpdf(y)
            prior::Float64 = -0.5 * dot(beta, beta)
            posterior::Float64 = prior + ll
            return posterior
        end
        q = [0.4, -0.2]
        oracle(qq) = (et = X * qq;
            -0.5 * dot(qq, qq) +
            sum(logpdf(Binomial(Ni, LogExpFunctions.logistic(e)), yi)
                for (e, Ni, yi) in zip(et, N, y)))
        kb = prepare(_binom_glm_model;
            have = (:beta, :X, :y, :N), want = :posterior,
            bound = (; X = X, y = y, N = N))
        @test kb(q) ≈ oracle(q)
        prep = prepare_ad(kb,
            AutoEnzyme(mode = Enzyme.Reverse, function_annotation = Enzyme.Const),
            q; active = :beta)
        v, g = ad_value_and_gradient!(prep, similar(q), q)
        @test g ≈ _glm_fd_grad(oracle, q) rtol = 1e-6
        @test v ≈ kb(q) rtol = 1e-12
        qtail = [2.0, 3.0]
        _, gt = ad_value_and_gradient!(prep, similar(qtail), qtail)
        @test gt ≈ _glm_fd_grad(kb, qtail) rtol = 1e-5
    end
end

@testset "neg_binomial_2_log_glm fused object" begin
    @test occursin("@kernel neg_binomial_2_log_glm(", NEG_BINOMIAL_2_LOG_GLM_KERNEL_SOURCE)
    @test all(hasproperty(neg_binomial_2_log_glm, ep) for ep in (:logpdf, :pointwise, :score))

    X = [1.0 -1.5; 1.0 -0.5; 1.0 0.5; 1.0 1.5; 1.0 0.0; 1.0 2.0]
    beta = [0.5, 0.3]
    eta = X * beta
    mu = exp.(eta)
    phi0 = 2.0
    y = [1, 0, 3, 2, 1, 5]
    ref_cells = [logpdf(NegativeBinomial(phi0, phi0 / (m + phi0)), yi)
        for (m, yi) in zip(mu, y)]
    ref_total = sum(ref_cells)
    # Direct (non-reciprocal) theta spelling: independently validates the
    # reciprocal spelling the score endpoint runs.
    ref_score = [yi - (yi + phi0) * m / (m + phi0) for (m, yi) in zip(mu, y)]

    @testset "endpoints match the independent oracle" begin
        @test prepare(neg_binomial_2_log_glm.logpdf;
            have = (:y, :X, :beta, :phi), want = :logpdf)(y, X, beta, phi0) ≈ ref_total
        @test prepare(neg_binomial_2_log_glm.pointwise;
            have = (:y, :X, :beta, :phi), want = :pointwise)(y, X, beta, phi0) ≈ ref_cells
        @test prepare(neg_binomial_2_log_glm.score;
            have = (:y, :X, :beta, :phi), want = :score)(y, X, beta, phi0) ≈ ref_score
        pw = prepare(neg_binomial_2_log_glm.pointwise;
            have = (:y, :X, :beta, :phi), want = :pointwise)(y, X, beta, phi0)
        ll = prepare(neg_binomial_2_log_glm.logpdf;
            have = (:y, :X, :beta, :phi), want = :logpdf)(y, X, beta, phi0)
        @test sum(pw) ≈ ll
    end

    @testset "eta / mu / working weights extract as owner nodes" begin
        @test prepare(extract(neg_binomial_2_log_glm;
            have = (:X, :beta, :phi), want = :eta))(X, beta, phi0) ≈ eta
        @test prepare(extract(neg_binomial_2_log_glm;
            have = (:X, :beta, :phi), want = :mu))(X, beta, phi0) ≈ mu
        @test prepare(extract(neg_binomial_2_log_glm;
            have = (:X, :beta, :phi), want = :working_weights))(X, beta, phi0) ≈
            (mu .* phi0) ./ (mu .+ phi0)
    end

    @testset "tails need no cutoff (reciprocal forms stay finite)" begin
        kp = prepare(neg_binomial_2_log_glm.pointwise;
            have = (:y, :X, :beta, :phi), want = :pointwise)
        s = log(phi0)
        for et in (-100.0, -30.0, -10.0, -1.0, 0.0, 1.0, 10.0, 30.0, 100.0),
                yi in (0, 1, 5)
            er = exp(et)
            @test kp([yi], [1.0 et], [0.0, 1.0], phi0)[1] ≈
                yi * (et - s) - (yi + phi0) * log1p(er / phi0) +
                (loggamma(yi + phi0) - loggamma(yi + 1) - loggamma(phi0))
        end
        ks = prepare(neg_binomial_2_log_glm.score;
            have = (:y, :X, :beta, :phi), want = :score)
        for et in (-100.0, 100.0), yi in (0, 5)
            er = exp(et)
            @test ks([yi], [1.0 et], [0.0, 1.0], phi0)[1] ≈
                yi - (yi + phi0) / (phi0 / er + 1)
        end
        for et in (-2.0, 0.0, 2.0), yi in (0, 1, 5)
            er = exp(et)
            @test ks([yi], [1.0 et], [0.0, 1.0], phi0)[1] ≈
                yi - (yi + phi0) * er / (er + phi0)
        end
    end

    @testset "normalizer folds out of bound queries" begin
        ku = prepare(neg_binomial_2_log_glm.logpdf;
            have = (:y, :X, :beta, :phi), want = :logpdf,
            bound = (; X = X, y = y))
        sku = string(code_expr(ku))
        # phi free: lgp stays a live call on runtime (y, phi).
        @test occursin("logpdf__lgp", sku) && occursin(")(y, phi)", sku)
        kb = prepare(neg_binomial_2_log_glm.logpdf;
            have = (:y, :X, :beta, :phi), want = :logpdf,
            bound = (; X = X, y = y, phi = phi0))
        skb = string(code_expr(kb))
        # phi bound: lgp folds to a nullary thunk (no per-call lgamma).
        @test occursin("logpdf__lgp", skb) && !occursin(")(y, phi)", skb)
        @test !occursin("Distributions", skb)
    end

    @testset "lowered code is self-contained (no helper escapes)" begin
        k = prepare(neg_binomial_2_log_glm.logpdf;
            have = (:y, :X, :beta, :phi), want = :logpdf)
        sk = string(code_expr(k))
        @test !occursin("_glm_", sk)
        @test !occursin("Distributions", sk)
    end

    @testset "analytic adjoint matches finite differences" begin
        @kernel _nb2_glm_model(
                beta::Vector{Float64}, X::Matrix{Float64},
                y::Vector{Int}, phi::Float64) = begin
            ll::Float64 = neg_binomial_2_log_glm.logpdf(y)
            prior::Float64 = -0.5 * dot(beta, beta)
            posterior::Float64 = prior + ll
            return posterior
        end
        q = [0.4, -0.2]
        oracle(qq) = (et = X * qq;
            -0.5 * dot(qq, qq) +
            sum(logpdf(NegativeBinomial(phi0, phi0 / (exp(e) + phi0)), yi)
                for (e, yi) in zip(et, y)))
        kb = prepare(_nb2_glm_model;
            have = (:beta, :X, :y, :phi), want = :posterior,
            bound = (; X = X, y = y, phi = phi0))
        @test kb(q) ≈ oracle(q)
        prep = prepare_ad(kb,
            AutoEnzyme(mode = Enzyme.Reverse, function_annotation = Enzyme.Const),
            q; active = :beta)
        v, g = ad_value_and_gradient!(prep, similar(q), q)
        @test g ≈ _glm_fd_grad(oracle, q) rtol = 1e-6
        @test v ≈ kb(q) rtol = 1e-12
        qtail = [2.0, 3.0]
        _, gt = ad_value_and_gradient!(prep, similar(qtail), qtail)
        @test gt ≈ _glm_fd_grad(kb, qtail) rtol = 1e-5
    end

    @testset "sampled phi end to end (unpacked beta+phi active)" begin
        # The whole model pinned end to end: one generic-Enzyme reverse over
        # the unpacked beta+phi graph must match the true gradient.
        @kernel _nb2_glm_modelQ(
                q::Vector{Float64}, X::Matrix{Float64}, y::Vector{Int}) = begin
            beta::Vector{Float64} = q[1:2]
            phi::Float64 = q[3]
            ll::Float64 = neg_binomial_2_log_glm.logpdf(y)
            prior::Float64 = -0.5 * dot(q, q)
            posterior::Float64 = prior + ll
            return posterior
        end
        q0 = [0.4, -0.2, 2.0]
        oracleQ(qq) = (et = X * qq[1:2]; p = qq[3]; s = log(p);
            -0.5 * dot(qq, qq) +
            sum(yi * (e - s) - (yi + p) * log1p(exp(e) / p) +
                (loggamma(yi + p) - loggamma(yi + 1) - loggamma(p))
                for (e, yi) in zip(et, y)))
        kb = prepare(_nb2_glm_modelQ;
            have = (:q, :X, :y), want = :posterior, bound = (; X = X, y = y))
        @test kb(q0) ≈ oracleQ(q0)
        prep = prepare_ad(kb,
            AutoEnzyme(mode = Enzyme.Reverse, function_annotation = Enzyme.Const),
            q0; active = :q)
        v, g = ad_value_and_gradient!(prep, similar(q0), q0)
        @test g ≈ _glm_fd_grad(oracleQ, q0) rtol = 1e-6
        @test v ≈ kb(q0) rtol = 1e-12
    end
end

@testset "categorical_logit_glm fused object" begin
    @test occursin("@kernel categorical_logit_glm(", CATEGORICAL_LOGIT_GLM_KERNEL_SOURCE)
    @test all(hasproperty(categorical_logit_glm, ep) for ep in (:logpdf, :pointwise, :score))

    X = [1.0 -1.5; 1.0 -0.5; 1.0 0.5; 1.0 1.5; 1.0 0.0; 1.0 2.0]
    B0 = [0.5 0.0 -0.4; 0.3 -0.2 0.1]
    a0 = [0.1, -0.1, 0.2]
    E0 = X * B0 .+ a0'
    (n, K, C) = (size(X, 1), size(X, 2), length(a0))
    y = [1, 3, 2, 1, 2, 3]
    sms = [softmax(E0[i, :]) for i in 1:n]
    ref_cells = [logpdf(Categorical(sms[i]), y[i]) for i in 1:n]
    ref_total = sum(ref_cells)
    ref_score = [Float64(c == y[i]) - sms[i][c] for i in 1:n, c in 1:C]
    catcell(e, yi) = (m = maximum(e); e[yi] - m - log(sum(exp, e .- m)))

    @testset "endpoints match the independent oracle" begin
        @test prepare(categorical_logit_glm.logpdf;
            have = (:y, :X, :beta, :alpha), want = :logpdf)(y, X, B0, a0) ≈ ref_total
        @test prepare(categorical_logit_glm.pointwise;
            have = (:y, :X, :beta, :alpha), want = :pointwise)(y, X, B0, a0) ≈ ref_cells
        @test prepare(categorical_logit_glm.score;
            have = (:y, :X, :beta, :alpha), want = :score)(y, X, B0, a0) ≈ ref_score
        pw = prepare(categorical_logit_glm.pointwise;
            have = (:y, :X, :beta, :alpha), want = :pointwise)(y, X, B0, a0)
        ll = prepare(categorical_logit_glm.logpdf;
            have = (:y, :X, :beta, :alpha), want = :logpdf)(y, X, B0, a0)
        @test sum(pw) ≈ ll
    end

    @testset "eta / mu extract as owner nodes" begin
        @test prepare(extract(categorical_logit_glm;
            have = (:X, :beta, :alpha), want = :eta))(X, B0, a0) ≈ E0
        @test prepare(extract(categorical_logit_glm;
            have = (:X, :beta, :alpha), want = :mu))(X, B0, a0) ≈
            [sms[i][c] for i in 1:n, c in 1:C]
    end

    @testset "extreme spreads stay finite (stable rowwise softmax)" begin
        kp = prepare(categorical_logit_glm.pointwise;
            have = (:y, :X, :beta, :alpha), want = :pointwise)
        X0 = zeros(1, 2)
        Bt = zeros(2, 3)
        for et in (10.0, 30.0, 100.0, 300.0), yi in (1, 2, 3)
            e = [et, 0.0, -et]
            @test kp([yi], X0, Bt, e)[1] ≈ catcell(e, yi)
        end
    end

    @testset "lowered code is self-contained (no helper escapes)" begin
        k = prepare(categorical_logit_glm.logpdf;
            have = (:y, :X, :beta, :alpha), want = :logpdf)
        sk = string(code_expr(k))
        @test !occursin("_glm_", sk)
        @test !occursin("Distributions", sk)
    end

    @testset "analytic adjoint matches finite differences (beta active)" begin
        @kernel _cat_glm_modelB(
                beta::Matrix{Float64}, alpha::Vector{Float64},
                X::Matrix{Float64}, y::Vector{Int}) = begin
            ll::Float64 = categorical_logit_glm.logpdf(y)
            prior::Float64 = -0.5 * (sum(abs2, beta) + sum(abs2, alpha))
            posterior::Float64 = prior + ll
            return posterior
        end
        oracleB(qv) = (B = reshape(qv, K, C); E = X * B .+ a0';
            -0.5 * (sum(abs2, B) + sum(abs2, a0)) +
            sum(catcell(E[i, :], y[i]) for i in 1:n))
        kb = prepare(_cat_glm_modelB;
            have = (:beta, :alpha, :X, :y), want = :posterior,
            bound = (; X = X, y = y, alpha = a0))
        @test kb(B0) ≈ oracleB(vec(B0))
        prep = prepare_ad(kb,
            AutoEnzyme(mode = Enzyme.Reverse, function_annotation = Enzyme.Const),
            B0; active = :beta)
        v, g = ad_value_and_gradient!(prep, similar(B0), B0)
        @test vec(g) ≈ _glm_fd_grad(oracleB, vec(B0)) rtol = 1e-6
        @test v ≈ kb(B0) rtol = 1e-12
        Btail = B0 .+ 0.5
        _, gt = ad_value_and_gradient!(prep, similar(Btail), Btail)
        kbwrap(qv) = kb(reshape(qv, K, C))
        @test vec(gt) ≈ _glm_fd_grad(kbwrap, vec(Btail)) rtol = 1e-5
    end

    @testset "analytic adjoint matches finite differences (alpha active)" begin
        @kernel _cat_glm_modelA(
                beta::Matrix{Float64}, alpha::Vector{Float64},
                X::Matrix{Float64}, y::Vector{Int}) = begin
            ll::Float64 = categorical_logit_glm.logpdf(y)
            prior::Float64 = -0.5 * (sum(abs2, beta) + sum(abs2, alpha))
            posterior::Float64 = prior + ll
            return posterior
        end
        oracleA(qa) = (E = X * B0 .+ qa';
            -0.5 * (sum(abs2, B0) + sum(abs2, qa)) +
            sum(catcell(E[i, :], y[i]) for i in 1:n))
        kb = prepare(_cat_glm_modelA;
            have = (:beta, :alpha, :X, :y), want = :posterior,
            bound = (; X = X, y = y, beta = B0))
        @test kb(a0) ≈ oracleA(a0)
        prep = prepare_ad(kb,
            AutoEnzyme(mode = Enzyme.Reverse, function_annotation = Enzyme.Const),
            a0; active = :alpha)
        v, g = ad_value_and_gradient!(prep, similar(a0), a0)
        @test g ≈ _glm_fd_grad(oracleA, a0) rtol = 1e-6
        @test v ≈ kb(a0) rtol = 1e-12
        atail = a0 .+ 0.5
        _, gt = ad_value_and_gradient!(prep, similar(atail), atail)
        @test gt ≈ _glm_fd_grad(kb, atail) rtol = 1e-5
    end

    @testset "analytic adjoint matches finite differences (both active)" begin
        # Test dims K=2, C=3 (literals; the bench twin uses 3x3).
        @kernel _cat_glm_modelQ(
                q::Vector{Float64}, X::Matrix{Float64}, y::Vector{Int}) = begin
            alpha::Vector{Float64} = q[1:3]
            betav::Vector{Float64} = q[4:9]
            beta::Matrix{Float64} = reshape(betav, 2, 3)
            ll::Float64 = categorical_logit_glm.logpdf(y)
            prior::Float64 = -0.5 * dot(q, q) / 25.0 - 9 * (0.5 * log(2π) + log(5.0))
            posterior::Float64 = prior + ll
            return posterior
        end
        q0 = vcat(a0, vec(B0))
        oracleQ(qq) = (a = @view qq[1:C]; B = reshape(@view(qq[C + 1:C + K * C]), K, C);
            E = X * B .+ a';
            -0.5 * dot(qq, qq) / 25.0 - length(qq) * (0.5 * log(2π) + log(5.0)) +
            sum(catcell(E[i, :], y[i]) for i in 1:n))
        kb = prepare(_cat_glm_modelQ;
            have = (:q, :X, :y), want = :posterior, bound = (; X = X, y = y))
        @test kb(q0) ≈ oracleQ(q0)
        prep = prepare_ad(kb,
            AutoEnzyme(mode = Enzyme.Reverse, function_annotation = Enzyme.Const),
            q0; active = :q)
        v, g = ad_value_and_gradient!(prep, similar(q0), q0)
        @test g ≈ _glm_fd_grad(oracleQ, q0) rtol = 1e-6
        @test v ≈ kb(q0) rtol = 1e-12
    end

end

@testset "ordered_logistic_glm fused object" begin
    @test occursin("@kernel ordered_logistic_glm(", ORDERED_LOGISTIC_GLM_KERNEL_SOURCE)
    @test all(hasproperty(ordered_logistic_glm, ep) for ep in (:logpdf, :pointwise, :score))

    X = [1.0 -1.5; 1.0 -0.5; 1.0 0.5; 1.0 1.5; 1.0 0.0; 1.0 2.0]
    beta = [0.5, 0.3]
    eta = X * beta
    c0 = [-0.5, 0.5, 1.5]
    C = length(c0) + 1
    y = [1, 2, 4, 3, 1, 2]
    # Manual oracle (no Distributions univariate for the ordered model;
    # the formula itself is cross-checked vs Stan in the head-to-head).
    ordab(e, yi, cuts) = (cp = [-Inf; cuts; Inf];
        (LogExpFunctions.logistic(cp[yi] - e), LogExpFunctions.logistic(cp[yi + 1] - e)))
    ordcell(e, yi, cuts) = (ab = ordab(e, yi, cuts); log(ab[2] - ab[1]))
    # Direct (non-collapsed) theta spelling: independently validates the
    # a + b - 1 collapse the score endpoint runs.
    orddirect(e, yi, cuts) = (ab = ordab(e, yi, cuts);
        (ab[1] * (1 - ab[1]) - ab[2] * (1 - ab[2])) / (ab[2] - ab[1]))
    ref_cells = [ordcell(e, yi, c0) for (e, yi) in zip(eta, y)]
    ref_total = sum(ref_cells)
    ref_score = [orddirect(e, yi, c0) for (e, yi) in zip(eta, y)]

    @testset "endpoints match the independent oracle" begin
        @test prepare(ordered_logistic_glm.logpdf;
            have = (:y, :X, :beta, :cuts), want = :logpdf)(y, X, beta, c0) ≈ ref_total
        @test prepare(ordered_logistic_glm.pointwise;
            have = (:y, :X, :beta, :cuts), want = :pointwise)(y, X, beta, c0) ≈ ref_cells
        @test prepare(ordered_logistic_glm.score;
            have = (:y, :X, :beta, :cuts), want = :score)(y, X, beta, c0) ≈ ref_score
        pw = prepare(ordered_logistic_glm.pointwise;
            have = (:y, :X, :beta, :cuts), want = :pointwise)(y, X, beta, c0)
        ll = prepare(ordered_logistic_glm.logpdf;
            have = (:y, :X, :beta, :cuts), want = :logpdf)(y, X, beta, c0)
        @test sum(pw) ≈ ll
    end

    @testset "eta / mu / working weights extract as owner nodes" begin
        @test prepare(extract(ordered_logistic_glm;
            have = (:X, :beta, :cuts), want = :eta))(X, beta, c0) ≈ eta
        S(e) = [LogExpFunctions.logistic(e - c) for c in c0]
        E1(e) = 1 + sum(S(e))
        E2(e) = 1 + sum((2k + 1) * s for (k, s) in enumerate(S(e)))
        @test prepare(extract(ordered_logistic_glm;
            have = (:X, :beta, :cuts), want = :mu))(X, beta, c0) ≈ E1.(eta)
        @test prepare(extract(ordered_logistic_glm;
            have = (:X, :beta, :cuts), want = :working_weights))(X, beta, c0) ≈
            [((dm = sum(s * (1 - s) for s in S(e)); V = E2(e) - E1(e)^2;
                V > 0 ? dm^2 / V : 0.0)) for e in eta]
    end

    @testset "tails need no cutoff (collapse is exact everywhere)" begin
        kp = prepare(ordered_logistic_glm.pointwise;
            have = (:y, :X, :beta, :cuts), want = :pointwise)
        ks = prepare(ordered_logistic_glm.score;
            have = (:y, :X, :beta, :cuts), want = :score)
        for et in (-100.0, -30.0, -10.0, -1.0, 0.0, 1.0, 10.0, 30.0, 100.0),
                yi in (1, 2, 3, 4)
            ab = ordab(et, yi, c0)
            @test kp([yi], [1.0 et], [0.0, 1.0], c0)[1] ≈ ordcell(et, yi, c0)
            @test ks([yi], [1.0 et], [0.0, 1.0], c0)[1] ≈ ab[1] + ab[2] - 1
        end
        for et in (-2.0, 0.0, 2.0), yi in (1, 2, 3, 4)
            # atol: near-zero thetas agree to oracle rounding (the direct
            # form carries 1ulp noise where the collapse is exact).
            @test ks([yi], [1.0 et], [0.0, 1.0], c0)[1] ≈ orddirect(et, yi, c0) atol = 1e-12
        end
    end

    @testset "saturated extremes: value -Inf, score ideal-exact" begin
        kp = prepare(ordered_logistic_glm.pointwise;
            have = (:y, :X, :beta, :cuts), want = :pointwise)
        ks = prepare(ordered_logistic_glm.score;
            have = (:y, :X, :beta, :cuts), want = :score)
        vhi = [kp([yi], [1.0 800.0], [0.0, 1.0], c0)[1] for yi in 1:C]
        shi = [ks([yi], [1.0 800.0], [0.0, 1.0], c0)[1] for yi in 1:C]
        @test vhi[C] ≈ 0.0
        @test all(v < 0 && isinf(v) for v in vhi[1:C-1])
        @test shi ≈ [fill(-1.0, C - 1); 0.0]
        vlo = [kp([yi], [1.0 -800.0], [0.0, 1.0], c0)[1] for yi in 1:C]
        slo = [ks([yi], [1.0 -800.0], [0.0, 1.0], c0)[1] for yi in 1:C]
        @test vlo[1] ≈ 0.0
        @test all(v < 0 && isinf(v) for v in vlo[2:C])
        @test slo ≈ [0.0; fill(1.0, C - 1)]
    end

    @testset "lowered code is self-contained (no helper escapes)" begin
        k = prepare(ordered_logistic_glm.logpdf;
            have = (:y, :X, :beta, :cuts), want = :logpdf)
        sk = string(code_expr(k))
        @test !occursin("_glm_", sk)
        @test !occursin("Distributions", sk)
    end

    @testset "pointwise cells are lazy: no index clamps, no floor, no 0/0" begin
        # Each observation is one `plate` cell whose edge/interior arms are
        # authored branches; the lowered program carries no clamped gather
        # (`y .^ 0`, `y .- 1 .+ (y .== 1)`), no `1e-300` floor, and the
        # working-weight quotient never forms 0/0 on a zero-variance row.
        kp = prepare(ordered_logistic_glm.pointwise;
            have = (:y, :X, :beta, :cuts), want = :pointwise)
        sk = string(code_expr(kp))
        @test !occursin("1e-300", sk)
        @test !occursin(".^ 0", sk)
        @test !occursin("ifelse", sk)
        ww = prepare(extract(ordered_logistic_glm;
            have = (:X, :beta, :cuts), want = :working_weights))
        @test ww([1.0 800.0], [0.0, 1.0], c0) == [0.0]
        @test ww([1.0 -800.0], [0.0, 1.0], c0) == [0.0]
    end

    @testset "reverse gradient follows the analytic score where FD is noisy" begin
        # dl/dbeta = X' * score(y). At the tail points the interior
        # differences F(hi) - F(lo) cancel to ~1e-9, so finite differences
        # of the primal are unusable, and the reverse of log(F(hi) - F(lo))
        # itself carries the cancellation's ~1e-6 relative error (the score
        # endpoint's collapsed a + b - 1 form does not). The lazy cell's
        # reverse stays finite and within that accuracy: no inactive arm
        # contributes 0 * Inf partials.
        @kernel _ord_glm_modelS(
                beta::Vector{Float64}, X::Matrix{Float64},
                y::Vector{Int}, cuts::Vector{Float64}) = begin
            ll::Float64 = ordered_logistic_glm.logpdf(y)
            return ll
        end
        kb = prepare(_ord_glm_modelS;
            have = (:beta, :X, :y, :cuts), want = :ll,
            bound = (; X = X, y = y, cuts = c0))
        ks = prepare(ordered_logistic_glm.score;
            have = (:y, :X, :beta, :cuts), want = :score)
        prep = prepare_ad(kb,
            AutoEnzyme(mode = Enzyme.Reverse, function_annotation = Enzyme.Const),
            beta; active = :beta)
        for q in ([0.4, -0.2], [3.0, -12.0], [20.0, 5.0])
            v, g = ad_value_and_gradient!(prep, similar(q), q)
            @test isfinite(v)
            @test all(isfinite, g)
            @test g ≈ X' * ks(y, X, q, c0) rtol = 1e-4
        end
    end

    @testset "analytic adjoint matches finite differences (beta active)" begin
        @kernel _ord_glm_modelB(
                beta::Vector{Float64}, X::Matrix{Float64},
                y::Vector{Int}, cuts::Vector{Float64}) = begin
            ll::Float64 = ordered_logistic_glm.logpdf(y)
            prior::Float64 = -0.5 * dot(beta, beta)
            posterior::Float64 = prior + ll
            return posterior
        end
        q = [0.4, -0.2]
        oracleB(qq) = (et = X * qq;
            -0.5 * dot(qq, qq) + sum(ordcell(e, yi, c0) for (e, yi) in zip(et, y)))
        kb = prepare(_ord_glm_modelB;
            have = (:beta, :X, :y, :cuts), want = :posterior,
            bound = (; X = X, y = y, cuts = c0))
        @test kb(q) ≈ oracleB(q)
        prep = prepare_ad(kb,
            AutoEnzyme(mode = Enzyme.Reverse, function_annotation = Enzyme.Const),
            q; active = :beta)
        v, g = ad_value_and_gradient!(prep, similar(q), q)
        @test g ≈ _glm_fd_grad(oracleB, q) rtol = 1e-6
        @test v ≈ kb(q) rtol = 1e-12
        qtail = [2.0, 3.0]
        _, gt = ad_value_and_gradient!(prep, similar(qtail), qtail)
        @test gt ≈ _glm_fd_grad(kb, qtail) rtol = 1e-5
    end

    @testset "analytic adjoint matches finite differences (cuts active)" begin
        @kernel _ord_glm_modelC(
                beta::Vector{Float64}, X::Matrix{Float64},
                y::Vector{Int}, cuts::Vector{Float64}) = begin
            ll::Float64 = ordered_logistic_glm.logpdf(y)
            prior::Float64 = -0.5 * dot(beta, beta)
            posterior::Float64 = prior + ll
            return posterior
        end
        oracleC(qc) = (et = X * beta;
            -0.5 * dot(beta, beta) + sum(ordcell(e, yi, qc) for (e, yi) in zip(et, y)))
        kb = prepare(_ord_glm_modelC;
            have = (:beta, :X, :y, :cuts), want = :posterior,
            bound = (; X = X, y = y, beta = beta))
        @test kb(c0) ≈ oracleC(c0)
        prep = prepare_ad(kb,
            AutoEnzyme(mode = Enzyme.Reverse, function_annotation = Enzyme.Const),
            c0; active = :cuts)
        v, g = ad_value_and_gradient!(prep, similar(c0), c0)
        @test g ≈ _glm_fd_grad(oracleC, c0) rtol = 1e-6
        @test v ≈ kb(c0) rtol = 1e-12
        ctail = c0 .+ 0.5
        _, gt = ad_value_and_gradient!(prep, similar(ctail), ctail)
        @test gt ≈ _glm_fd_grad(kb, ctail) rtol = 1e-5
    end

end
