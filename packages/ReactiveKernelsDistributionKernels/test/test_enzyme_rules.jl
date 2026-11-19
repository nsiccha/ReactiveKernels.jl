# DistributionKernels' `loggamma` / `logbeta` are derivative rules generated
# from one pure-math graph each (`ReactiveKernels.scalar_derivative_rule`):
# the callable IS the graph's primal cut, and the Enzyme adapter in
# ReactiveKernels' own extension reads the partials from the graph's cuts.
# The shape that aborts Enzyme on SpecialFunctions' bodies
# (benchmark/repro_enzyme_lgamma_branch.jl) therefore differentiates, with no
# hand-written rule and no rule on a function this package does not own.
using DifferentiationInterface: AutoEnzyme, derivative, gradient
import Enzyme
using ReactiveKernels
using ReactiveKernels: derivative_cut, prepare, prepare_ad, ad_gradient
using ReactiveKernelsDistributionKernels
using ReactiveKernelsDistributionKernels.DistributionKernelSources: beta, binomial
using SpecialFunctions: digamma
import SpecialFunctions
using Test

# Qualified on purpose: the surrounding runner binds SpecialFunctions'
# `loggamma` in `Main`, and the point of this file is the package's own one.
const _DKS = ReactiveKernelsDistributionKernels.DistributionKernelSources

@testset "Normal and Poisson tail cuts" begin
    for z in (-40.0, -3.0, 0.0, 2.0, 12.0)
        value, slope = derivative_cut(_DKS._standard_normal_logcdf, (true,), z)
        x = big(z)
        probability = SpecialFunctions.erfc(-x / sqrt(big(2))) / 2
        @test value == _DKS._normal_logcdf(z)
        @test slope ≈ Float64(exp(-x^2 / 2) / (sqrt(2 * big(π)) * probability)) rtol = 2e-14
    end
    for (rate, observed) in ((0.0, 0), (0.0, 4), (0.7, 3.9), (20.0, 2),
                             (3.0, big"1.25"), (1e6, 1_000_000),
                             (1e6 + 1000, 1_000_000)), upper in (false, true)
        value, slope, dcutoff = derivative_cut(_DKS._poisson_tail,
            (true, true, false), rate, observed, upper)
        tails = SpecialFunctions.gamma_inc(floor(observed) + 1.0, rate)
        k = big(floor(observed))
        mass = k == 0 ? exp(-big(rate)) :
            exp(k * log(big(rate)) - rate - SpecialFunctions.loggamma(k + 1))
        @test value == Float64(upper ? first(tails) : last(tails))
        @test slope ≈ Float64(upper ? mass : -mass) rtol = 2e-13
        @test iszero(dcutoff)
    end
end

@testset "loggamma / logbeta are rules generated from their graphs" begin
    @test Base.get_extension(ReactiveKernels, :ReactiveKernelsEnzymeExt) !== nothing
    @test Base.get_extension(
        ReactiveKernelsDistributionKernels,
        :ReactiveKernelsDistributionKernelsEnzymeExt) === nothing
    @test _DKS.loggamma isa ScalarDerivativeRule
    @test _DKS.logbeta isa ScalarDerivativeRule
    @test _DKS.loggamma_graph isa KernelSpec
    @test _DKS.logbeta_graph isa KernelSpec
    @test _DKS.loggamma(2.5) == SpecialFunctions.loggamma(2.5)
    @test _DKS.logbeta(1.5, 3.0) == SpecialFunctions.logbeta(1.5, 3.0)
    @test derivative_cut(_DKS.loggamma, (true,), 2.5) ==
        (SpecialFunctions.loggamma(2.5), digamma(2.5))
    @test derivative_cut(_DKS.logbeta, (true, true), 1.5, 3.0) ==
        (SpecialFunctions.logbeta(1.5, 3.0),
         digamma(1.5) - digamma(4.5), digamma(3.0) - digamma(4.5))
    @test derivative_cut(_DKS.logbeta, (false, true), 1.5, 3.0) ==
        (SpecialFunctions.logbeta(1.5, 3.0), digamma(3.0) - digamma(4.5))

    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    @test gradient(_DKS.loggamma, backend, 2.5) ≈ digamma(2.5)
    @test gradient(x -> _DKS.logbeta(x, 3.0), backend, 1.5) ≈ digamma(1.5) - digamma(4.5)
    @test gradient(x -> _DKS.logbeta(3.0, x), backend, 1.5) ≈ digamma(1.5) - digamma(4.5)
    @test gradient(x -> _DKS.logbeta(x, 2x), backend, 1.5) ≈
        (digamma(1.5) - digamma(4.5)) + 2 * (digamma(3.0) - digamma(4.5))
    @test gradient(x -> x > 0 ? _DKS.loggamma(x) : -Inf, backend, 2.5) ≈ digamma(2.5)
    @test gradient(x -> _DKS.loggamma(3.0) + x, backend, 2.5) ≈ 1.0   # inactive argument
    @test gradient(v -> sum(_DKS.loggamma.(v .+ 1)), backend, [1.0, 2.0, 3.0]) ≈
        digamma.([2.0, 3.0, 4.0])

    # The reproducer's shape: a guarded beta prior plus a plate of guarded
    # binomial cells, differentiated together through the prepared kernels.
    prior = prepare(beta.logpdf; have = (:a, :b, :x), want = :logpdf)
    cell = prepare(binomial.logpdf; have = (:n, :p, :observed), want = :logpdf)
    @noinline prior_op(p) = prior(2.0, 2.0, p)
    @noinline cell_op(n, p, k) = cell(n, p, k)
    ks = [3, 5, 2]; ns = [10, 10, 10]
    density(p) = prior_op(p) + sum(cell_op(ns[i], p, ks[i]) for i in eachindex(ks))
    p = 0.3
    expected = (1.0 / p - 1.0 / (1 - p)) + sum(ks) / p - sum(ns .- ks) / (1 - p)
    @test gradient(density, backend, p) ≈ expected
    prepared = prepare_ad(prior, backend, 2.0, 2.0, p; active = :x)
    @test ad_gradient(prepared, 2.0, 2.0, p) ≈ 1.0 / p - 1.0 / (1 - p)
end

# `logrising(a, m) = loggamma(a + m) - loggamma(a)`, the log rising factorial.
# Whole small counts take the product path (value and `d/da` from one pass),
# every other input the loggamma/digamma difference; BigFloat is the oracle.
_logrising_ref(a, m) = Float64(SpecialFunctions.loggamma(big(a) + m) -
                               SpecialFunctions.loggamma(big(a)))
_logrising_dref(a, m) = Float64(digamma(big(a) + m) - digamma(big(a)))

@testset "logrising: one graph, product path and gamma-difference path" begin
    @test _DKS.logrising isa ScalarDerivativeRule
    @test _DKS.logrising_graph isa KernelSpec
    for a in (1e-8, 0.05, 0.7, 3.25, 41.0, 2.5e6, 1e9, 3e9, 1e12),
            m in (0, 1, 2, 5, 31, 32, 33, 64, 65, 400)
        value, slope = derivative_cut(_DKS.logrising, (true, false), a, m)
        @test value == _DKS.logrising(a, m)
        if m <= 64 && a <= 2.0^30
            @test value ≈ _logrising_ref(a, m) rtol = 1e-14 atol = 1e-14
            @test slope ≈ _logrising_dref(a, m) rtol = 1e-13
        else
            # The gamma difference, exactly (with its cancellation at large a).
            @test value == SpecialFunctions.loggamma(a + m) - SpecialFunctions.loggamma(a)
            @test slope == digamma(a + m) - digamma(a)
        end
    end
    # A Float64 count, a fractional count, and the count's own partial.
    @test _DKS.logrising(2.5, 3.0) ≈ _logrising_ref(2.5, 3) rtol = 1e-15
    @test _DKS.logrising(2.5, 3.5) ≈ Float64(SpecialFunctions.loggamma(big(6.0)) -
        SpecialFunctions.loggamma(big(2.5))) rtol = 1e-14
    value, da, dm = derivative_cut(_DKS.logrising, (true, true), 2.5, 3.0)
    @test (value, da) == derivative_cut(_DKS.logrising, (true, false), 2.5, 3.0)
    @test dm == digamma(5.5)
    # m = 0 is exactly zero, also at a shape the product path refuses.
    @test _DKS.logrising(4.0, 0) == 0.0
    @test derivative_cut(_DKS.logrising, (true, false), 3e9, 0) == (0.0, 0.0)

    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    f(a) = _DKS.logrising(a, 17) + 2 * _DKS.logrising(a * a, 5)
    expected = _logrising_dref(1.3, 17) + 4 * 1.3 * _logrising_dref(1.69, 5)
    @test gradient(f, backend, 1.3) ≈ expected rtol = 1e-13
    @test derivative(f, AutoEnzyme(; mode = Enzyme.Forward), 1.3) ≈ expected rtol = 1e-13
    @test gradient(a -> a > 0 ? _DKS.logrising(a, 200) : -Inf, backend, 2.5) ≈
        _logrising_dref(2.5, 200) rtol = 1e-12
end
