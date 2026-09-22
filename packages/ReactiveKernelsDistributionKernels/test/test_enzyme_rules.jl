# The Enzyme extension gives `loggamma` / `logabsgamma` Julia-level reverse
# rules.  Without them, Enzyme's C-level `lgamma_r` handling aborts the
# process (an LLVM assertion, not a Julia error) on the shape every guarded
# `logpdf` plus an observation plate produces: two lazy branches around gamma
# calls in non-inlined functions differentiated together
# (`benchmark/repro_enzyme_lgamma_branch.jl`).
using DifferentiationInterface: AutoEnzyme, gradient
import Enzyme
using ReactiveKernels: extract, prepare, prepare_ad, ad_gradient
using ReactiveKernelsDistributionKernels.DistributionKernelSources: beta, binomial
using SpecialFunctions: digamma, loggamma, logbeta
using Test

@testset "Enzyme reverse rules for loggamma / logabsgamma" begin
    @test Base.get_extension(
        ReactiveKernelsDistributionKernels,
        :ReactiveKernelsDistributionKernelsEnzymeExt) !== nothing
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    @test gradient(loggamma, backend, 2.5) ≈ digamma(2.5)
    @test gradient(x -> logbeta(x, 3.0), backend, 1.5) ≈ digamma(1.5) - digamma(4.5)
    @test gradient(x -> x > 0 ? loggamma(x) : -Inf, backend, 2.5) ≈ digamma(2.5)
    @test gradient(x -> loggamma(3.0) + x, backend, 2.5) ≈ 1.0   # inactive argument

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
