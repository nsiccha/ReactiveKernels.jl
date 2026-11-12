# Emit the reported two-state HMM and native value/ordinary-AD references.
# Use a consumer environment with RK, RKPPL, DistributionKernels, DI and
# Enzyme; pass an output directory. Then run jax_hmm_forward.py on hmm_4.py.
using ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface
import Enzyme
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal_hmm_log_weights_logdensity
const TOY_HMM = @rkppl begin
    mu_a ~ Normal(0, prior_scale)
    mu_b ~ Normal(0, prior_scale)
    sigma ~ Exponential(sigma_prior)
    stay ~ Beta(stay_a, stay_b)
    ones_row = fill(1.0, N)
    locations = reshape(permutedims(hcat(mu_a .* ones_row, mu_b .* ones_row)), 2, 1, N)
    scales = reshape(permutedims(hcat(sigma .* ones_row, sigma .* ones_row)), 2, 1, N)
    stay_row = fill(1.0, N - 1)
    log_transition = reshape(permutedims(hcat(log(stay) .* stay_row, log(1 - stay) .* stay_row,
                                              log(1 - stay) .* stay_row, log(stay) .* stay_row)), 2, 2, N - 1)
    log_initial = [log(0.5), log(0.5)]
    y ~ LogDensity(normal_hmm_log_weights_logdensity, locations, scales, log_transition, log_initial)
end
length(ARGS) == 1 || error("usage: julia jax_hmm_forward.jl OUTPUT_DIRECTORY")
destination = abspath(only(ARGS))
mkpath(destination)
open(joinpath(destination, "native.tsv"), "w") do io
    println(io, "N\tvalue\tg1\tg2\tg3\tg4")
    for N in (4, 100, 400, 1600)
        y = sin.(collect(1:N))
        plan = TOY_HMM(; N, prior_scale=2.0, sigma_prior=1.0, stay_a=9.0, stay_b=1.0) | (; y)
        built = build_kernel(plan)
        src = jax_source(built, plan)
        write(joinpath(destination, "hmm_$(N).py"), src)
        query = prepare_query(built, plan, :sampler)
        u = [-0.3, 0.7, 0.1, 1.2]
        value = query(u)
        ad = prepare_ad(query, AutoEnzyme(; mode = Enzyme.Reverse), u; active = :unconstrained)
        gradient = ad_gradient(ad, u)
        println(io, join([N; value; gradient], '\t'))
        flush(io)
        println((; N, value, gradient, source_bytes=ncodeunits(src.text)))
        flush(stdout)
    end
end
