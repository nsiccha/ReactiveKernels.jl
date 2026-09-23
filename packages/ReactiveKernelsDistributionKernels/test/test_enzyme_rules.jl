# INTERIM rules (see ../ext/ReactiveKernelsDistributionKernelsEnzymeExt.jl): the
# package's own `loggamma`/`logbeta` entry points carry reverse rules, so the
# shape that aborts Enzyme on SpecialFunctions' bodies
# (benchmark/repro_enzyme_lgamma_branch.jl) differentiates. They are replaced
# by generated adapters once the derivative-rule generator lands.
using DifferentiationInterface: AutoEnzyme, gradient
import Enzyme
using ReactiveKernels: prepare, prepare_ad, ad_gradient
using ReactiveKernelsDistributionKernels
using ReactiveKernelsDistributionKernels.DistributionKernelSources: beta, binomial
using SpecialFunctions: digamma
import SpecialFunctions
using Test

# Qualified on purpose: the surrounding runner already binds SpecialFunctions'
# `loggamma` in `Main`, and the point of this file is the package's own one.
const _DKS = ReactiveKernelsDistributionKernels.DistributionKernelSources

@testset "interim Enzyme reverse rules on the owned loggamma / logbeta" begin
    @test Base.get_extension(
        ReactiveKernelsDistributionKernels,
        :ReactiveKernelsDistributionKernelsEnzymeExt) !== nothing
    # The rules attach to this package's entry points, never to SpecialFunctions'.
    @test _DKS.loggamma !== SpecialFunctions.loggamma
    @test _DKS.logbeta !== SpecialFunctions.logbeta
    @test parentmodule(_DKS.loggamma) === _DKS
    @test _DKS.loggamma(2.5) == SpecialFunctions.loggamma(2.5)
    @test _DKS.logbeta(1.5, 3.0) == SpecialFunctions.logbeta(1.5, 3.0)

    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    @test gradient(_DKS.loggamma, backend, 2.5) ≈ digamma(2.5)
    @test gradient(x -> _DKS.logbeta(x, 3.0), backend, 1.5) ≈ digamma(1.5) - digamma(4.5)
    @test gradient(x -> _DKS.logbeta(3.0, x), backend, 1.5) ≈ digamma(1.5) - digamma(4.5)
    @test gradient(x -> _DKS.logbeta(x, 2x), backend, 1.5) ≈
        (digamma(1.5) - digamma(4.5)) + 2 * (digamma(3.0) - digamma(4.5))
    @test gradient(x -> x > 0 ? _DKS.loggamma(x) : -Inf, backend, 2.5) ≈ digamma(2.5)
    @test gradient(x -> _DKS.loggamma(3.0) + x, backend, 2.5) ≈ 1.0   # inactive argument

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
