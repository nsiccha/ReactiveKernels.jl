# DistributionKernels' `loggamma` / `logbeta` are derivative rules generated
# from one pure-math graph each (`ReactiveKernels.scalar_derivative_rule`):
# the callable IS the graph's primal cut, and the Enzyme adapter in
# ReactiveKernels' own extension reads the partials from the graph's cuts.
# The shape that aborts Enzyme on SpecialFunctions' bodies
# (benchmark/repro_enzyme_lgamma_branch.jl) therefore differentiates, with no
# hand-written rule and no rule on a function this package does not own.
using DifferentiationInterface: AutoEnzyme, gradient
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
