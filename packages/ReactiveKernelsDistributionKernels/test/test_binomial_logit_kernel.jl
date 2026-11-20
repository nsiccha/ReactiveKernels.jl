module BinomialLogitKernelTests
using Test, ReactiveKernels, ReactiveKernelsDistributionKernels
using DifferentiationInterface: AutoEnzyme
using ReactiveKernelsDistributionKernels.DistributionKernelSources: binomial
using SpecialFunctions: loggamma
import Enzyme

# The kernel `observed * log(p) + (n - observed) * log(1 - p)` at a logit, in
# extended precision so neither tail loses digits. A zero count contributes
# nothing, also where its log-probability is infinite.
function _reference_kernel(observed, n, logit)
    setprecision(BigFloat, 512) do
        b = BigFloat(logit)
        logp = logit == Inf ? big(0.0) : logit == -Inf ? big(-Inf) : -log1p(exp(-b))
        log1mp = logit == Inf ? big(-Inf) : logit == -Inf ? big(0.0) : -log1p(exp(b))
        Float64((observed == 0 ? big(0.0) : observed * logp) +
                (observed == n ? big(0.0) : (n - observed) * log1mp))
    end
end
_agree(a, b; rtol) = isequal(a, b) || isapprox(a, b; rtol)

@testset "binomial logit kernel reads one softplus and keeps both tails" begin
    kernel = prepare(binomial.logkernel; have = (:observed, :n, :logit), want = :logkernel)
    logpdf = prepare(binomial.logpdf; have = (:observed, :n, :logit), want = :logpdf)
    logits = (-Inf, -1000.0, -80.0, -37.5, -3.0, -1.0, -1e-9, -0.0, 0.0, 1e-9, 0.7, 1.0,
        5.0, 36.0, 80.0, 1000.0, Inf)
    for n in (0, 1, 3, 8, 40), observed in unique((0, 1, n ÷ 2, max(n - 1, 0), n)), logit in logits
        observed <= n || continue
        expected = _reference_kernel(observed, n, logit)
        @test _agree(kernel(observed, n, logit), expected; rtol = 4eps())
        coefficient = 0 < observed < n ?
            loggamma(n + 1.0) - loggamma(observed + 1.0) - loggamma(n - observed + 1.0) : 0.0
        @test _agree(logpdf(observed, n, logit), expected + coefficient; rtol = 1e-14)
    end
    # Outside the support, and at a NaN logit.
    @test kernel(4, 3, 0.2) == -Inf
    @test kernel(-1, 3, 0.2) == -Inf
    @test all(isnan, (kernel(0, 3, NaN), kernel(1, 3, NaN), kernel(3, 3, NaN)))
    @test kernel(0, 0, NaN) == 0.0

    # The score is `observed - n * p` on both sides of zero, where the kernel
    # selects its weight, and in both tails.
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    kernel_ad = prepare_ad(kernel, backend, 1, 3, 0.0; active = :logit)
    for n in (1, 3, 8), observed in unique((0, 1, n)),
            logit in (-80.0, -3.0, -1e-9, -0.0, 0.0, 1e-9, 0.7, 36.0, 80.0)
        score = observed - n / (1 + exp(-logit))
        @test only(ad_gradient(kernel_ad, observed, n, logit)) ≈ score atol = 1e-12 rtol = 1e-12
    end
end
end
