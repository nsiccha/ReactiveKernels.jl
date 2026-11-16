using ReactiveKernels, ReactiveKernelsDistributionKernels, Test
using ReactiveKernelsDistributionKernels.DistributionKernelSources: beta_binomial
using DifferentiationInterface: AutoEnzyme, gradient
import Enzyme, SpecialFunctions

@testset "beta-binomial product and logbeta paths" begin
    bb = prepare(beta_binomial.logpdf;
        have = (:observed, :n, :alpha, :beta), want = :logpdf)
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    for (y, n) in ((0, 0), (0, 31), (17, 32), (33, 33), (29, 64),
                  (30, 65), (900, 1500)), (a, b) in ((0.01, 0.03), (2.0, 5.0), (20.0, 4.0))
        g = gradient(v -> bb(y, n, v[1], v[2]), backend, [a, b])
        # High-precision partials of the original beta-function identity.
        psi(x) = SpecialFunctions.digamma(big(x))
        shared = psi(a + b) - SpecialFunctions.digamma(big(a) + big(b) + n)
        expected = Float64[SpecialFunctions.digamma(big(a) + y) - psi(a) + shared,
            SpecialFunctions.digamma(big(b) + n - y) - psi(b) + shared]
        @test g ≈ expected rtol = 1e-10 atol = 1e-10
    end
    # Large shapes retain logbeta: a difference of loggamma calls would lose
    # precision. Compare to the logbeta form with the same grouping.
    for (a, b) in ((2.0^30, 2.0), (3e9, 2e9), (1e12, 2e12)), (y, n) in ((0, 10), (3, 10), (64, 64))
        expected = SpecialFunctions.loggamma(n + 1.0) - SpecialFunctions.loggamma(y + 1.0) -
            SpecialFunctions.loggamma(n - y + 1.0) +
            (SpecialFunctions.logbeta(y + a, n - y + b) - SpecialFunctions.logbeta(a, b))
        @test bb(y, n, a, b) == expected
    end
    for (y, n) in ((-1, 10), (11, 10), (-1, 100), (101, 100))
        @test bb(y, n, -1.0, -2.0) == -Inf
        @test gradient(v -> bb(y, n, v[1], v[2]), backend, [2.0, 5.0]) == [0.0, 0.0]
    end
    cells = plate(beta_binomial.logpdf;
        have = (:observed, :n, :alpha, :beta), want = :logpdf,
        batched = (:observed, :n))
    @test cells(Int[], Int[], 2.0, 5.0) == 0.0
    ns = [0, 31, 32, 33, 64, 65, 1500]
    ys = [0, 0, 16, 33, 29, 65, 900]
    saved = (copy(ns), copy(ys))
    @test cells(ys, ns, 2.0, 5.0) ≈ sum(bb(y, n, 2.0, 5.0) for (y, n) in zip(ys, ns))
    @test (ns, ys) == saved
end
