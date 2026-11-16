# Focused paired comparison of the beta-binomial likelihood. Run in an
# environment with ReactiveKernelsDistributionKernels, Enzyme, Chairmarks and
# SpecialFunctions. Repeat with registered Enzyme and the dogfood source.
# This is a synthetic likelihood benchmark, not the RKPPLBench/Wren model.
using ReactiveKernels, ReactiveKernelsDistributionKernels
using ReactiveKernelsDistributionKernels.DistributionKernelSources: beta_binomial,
    loggamma, logbeta
using Enzyme, Chairmarks, Random, LinearAlgebra, Printf, Pkg

@kernel legacy_beta_binomial(n::Int, alpha::Float64, beta::Float64) = begin
    logpdf(observed::Int)::Float64 =
        (observed >= 0) & (observed <= n) ?
        (loggamma(n + 1.0) - loggamma(observed + 1.0) -
         loggamma(n - observed + 1.0) +
         logbeta(observed + alpha, n - observed + beta) - logbeta(alpha, beta)) : -Inf
end

function measure_likelihood(spec, trials, observations, X, u)
    started = time_ns()
    cells = plate(spec; have = (:observed, :n, :alpha, :beta), want = :logpdf,
        batched = (:observed, :n, :alpha, :beta))
    function density(u)
        eta = X * view(u, 2:length(u))
        phi = exp(u[1])
        mu = @. 1 / (1 + exp(-eta))
        cells(observations, trials, mu .* phi, (1 .- mu) .* phi)
    end
    value = density(u)
    first_gradient = only(Enzyme.gradient(Reverse, density, u))
    first_seconds = (time_ns() - started) / 1e9
    primal = @b $density($u) seconds = 1
    reverse = @b Enzyme.gradient(Reverse, $density, $u) seconds = 1
    return (; value, gradient = first_gradient, first_seconds,
        primal_seconds = primal.time, gradient_seconds = reverse.time)
end

function main()
    println("Julia ", VERSION, " ", Base.GIT_VERSION_INFO.commit_short)
    info = Pkg.dependencies()[Base.PkgId(Enzyme).uuid]
    println("Enzyme ", info.version, " source=", info.source)
    rng = Xoshiro(20261116)
    X = randn(rng, 1770, 30) ./ sqrt(30)
    u = [log(7.0); 0.1 .* randn(rng, 30)]
    for counts in (10:60, 65:150, 700:1500)
        trials = rand(rng, counts, size(X, 1))
        observations = [rand(rng, 0:n) for n in trials]
        old = measure_likelihood(legacy_beta_binomial.logpdf, trials, observations, X, u)
        new = measure_likelihood(beta_binomial.logpdf, trials, observations, X, u)
        @assert isapprox(new.value, old.value; rtol = 1e-12)
        @assert isapprox(new.gradient, old.gradient; rtol = 1e-10, atol = 1e-10)
        @printf("counts=%s value_error=%.3g gradient_error=%.3g\n", string(counts),
            abs(new.value - old.value), maximum(abs.(new.gradient - old.gradient)))
        for (label, r) in (("legacy", old), ("product", new))
            @printf("  %s first=%.3fs density=%.3fus gradient=%.3fus\n", label,
                r.first_seconds, 1e6 * r.primal_seconds, 1e6 * r.gradient_seconds)
        end
    end
end

main()
