module EndpointHMCAuthoring
using ReactiveKernels, Random, LinearAlgebra

# Minimal fixed-length endpoint HMC: one final Metropolis decision, with no
# per-step divergence observation or early return. Captured leapfrog and
# phasepoint authorities are shared with the multinomial benchmark.
@kernel hmc_state(init; n_steps=4, step_f=nothing, stats_f=nothing) = begin
    fwd = deepcopy(init)
    step!(rng) = begin
        init.mom = Random.randn!(rng, init.mom)
        init.mom = sqrt(fwd.metric) * init.mom
        fwd.pos .= init.pos
        fwd.mom .= init.mom
        initial_ham = init.ham
        for _ in 1:n_steps
            step_f(fwd)
        end
        if -Random.randexp(rng) < initial_ham - fwd.ham
            init .= fwd
        end
    end
end
end
