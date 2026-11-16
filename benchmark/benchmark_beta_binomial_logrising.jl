# Focused paired comparison of the beta-binomial likelihood. Run in an
# environment with ReactiveKernelsDistributionKernels, Enzyme, Chairmarks and
# SpecialFunctions. Repeat with registered Enzyme and the dogfood source.
# This is a synthetic likelihood benchmark, not the RKPPLBench/Wren model.
# First-call times include preparation and share process/compiler caches;
# compare the warmed density and gradient timings for runtime performance.
using ReactiveKernels, ReactiveKernelsDistributionKernels
using ReactiveKernelsDistributionKernels.DistributionKernelSources: beta_binomial,
    loggamma, logbeta
using Enzyme, Chairmarks, Random, LinearAlgebra, Printf, Pkg, Libdl
using DifferentiationInterface: AutoEnzyme

@kernel legacy_beta_binomial(n::Int, alpha::Float64, beta::Float64) = begin
    logpdf(observed::Int)::Float64 =
        (observed >= 0) & (observed <= n) ?
        (loggamma(n + 1.0) - loggamma(observed + 1.0) -
         loggamma(n - observed + 1.0) +
         logbeta(observed + alpha, n - observed + beta) - logbeta(alpha, beta)) : -Inf
end

# Exercise the prepared RK model interface used by consumers, including bound
# observations and design matrix. The direct closure/BLAS reproducer remains
# in repro_enzyme_beta_binomial_captured_blas.jl (Enzyme snag 09891a70).
for (name, law) in ((:legacy_likelihood, :legacy_beta_binomial),
                    (:product_likelihood, :beta_binomial))
    @eval @kernel $name(u::Vector{Float64}, X::Matrix{Float64},
                       trials::Vector{Int}, observations::Vector{Int}) = begin
        eta = X * view(u, 2:length(u))
        phi = exp(u[1])
        terms = plate(eachindex(trials)) do i
            mu = 1 / (1 + exp(-eta[i]))
            $law(trials[i], mu * phi, (1 - mu) * phi).logpdf(observations[i])
        end
        total = sum(terms)
        return total
    end
end

function measure_likelihood(spec, trials, observations, X, u)
    started = time_ns()
    density = prepare(spec; have = (:u, :X, :trials, :observations), want = :total,
        bound = (; trials, observations, X))
    value = density(u)
    ad = prepare_ad(density, AutoEnzyme(; mode = Reverse), u; active = :u)
    first_gradient = ad_gradient(ad, u)
    first_seconds = (time_ns() - started) / 1e9
    primal = @b $density($u) seconds = 1
    reverse = @b ad_gradient($ad, $u) seconds = 1
    return (; value, gradient = first_gradient, first_seconds,
        primal_seconds = primal.time, gradient_seconds = reverse.time)
end

function main()
    println("Julia ", VERSION, " ", Base.GIT_VERSION_INFO.commit_short)
    info = Pkg.dependencies()[Base.PkgId(Enzyme).uuid]
    println("Enzyme ", info.version, " source=", info.source)
    println("Enzyme libraries ", filter(p -> occursin("Enzyme", p), Libdl.dllist()))
    flush(stdout)
    rng = Xoshiro(20261116)
    X = randn(rng, 1770, 30) ./ sqrt(30)
    u = [log(7.0); 0.1 .* randn(rng, 30)]
    for counts in (10:60, 65:150, 700:1500)
        trials = rand(rng, counts, size(X, 1))
        observations = [rand(rng, 0:n) for n in trials]
        old = measure_likelihood(legacy_likelihood, trials, observations, X, u)
        new = measure_likelihood(product_likelihood, trials, observations, X, u)
        @assert isapprox(new.value, old.value; rtol = 1e-12)
        @assert isapprox(new.gradient, old.gradient; rtol = 1e-10, atol = 1e-10)
        @printf("counts=%s value_error=%.3g gradient_error=%.3g\n", string(counts),
            abs(new.value - old.value), maximum(abs.(new.gradient - old.gradient)))
        for (label, r) in (("legacy", old), ("product", new))
            @printf("  %s first=%.3fs density=%.3fus gradient=%.3fus\n", label,
                r.first_seconds, 1e6 * r.primal_seconds, 1e6 * r.gradient_seconds)
        end
        flush(stdout)
    end
end

main()
