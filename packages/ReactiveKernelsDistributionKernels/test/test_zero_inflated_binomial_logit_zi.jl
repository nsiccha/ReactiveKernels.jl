using LogExpFunctions: log1pexp
using ReactiveKernels: prepare
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    zero_inflated_binomial_logit_zi
using Test

@testset "ZIB source zero-inflation logit" begin
    lp = prepare(zero_inflated_binomial_logit_zi.logpdf;
        have = (:observed, :n, :logit, :zi_logit), want = :logpdf)
    lk = prepare(zero_inflated_binomial_logit_zi.logkernel;
        have = (:observed, :n, :logit, :zi_logit), want = :logkernel)
    ln = prepare(zero_inflated_binomial_logit_zi.lognormalizer;
        have = (:observed, :n), want = :lognormalizer)
    for (y, logit_p, zi_logit) in ((1, 0.0, 40.0), (0, 1000.0, -40.0))
        logzi = -log1pexp(-zi_logit)
        log1mzi = -log1pexp(zi_logit)
        bin = y == 0 ? -3log1pexp(logit_p) :
            log(3.0) - log1pexp(-logit_p) - 2log1pexp(logit_p)
        expected = y == 0 ?
            max(logzi, log1mzi + bin) +
                log1p(exp(-abs(logzi - (log1mzi + bin)))) :
            log1mzi + bin
        @test isfinite(lp(y, 3, logit_p, zi_logit))
        @test lp(y, 3, logit_p, zi_logit) ≈ expected
        @test lp(y, 3, logit_p, zi_logit) ≈
            lk(y, 3, logit_p, zi_logit) + ln(y, 3)
    end
    lp_prob = prepare(zero_inflated_binomial_logit_zi.logpdf;
        have = (:observed, :n, :p, :zi_logit), want = :logpdf)
    @test lp_prob(0, 3, 0.0, 0.0) ≈ 0.0
    @test lp_prob(3, 3, 1.0, 0.0) ≈ -log(2.0)
    @test lp_prob(1, 3, -0.1, 0.0) == -Inf
    @test lp_prob(1, 3, 1.1, 0.0) == -Inf
    @test lp_prob(0, 3, NaN, 0.0) == -Inf
    @test lp_prob(4, 3, 0.5, 0.0) == -Inf
end
