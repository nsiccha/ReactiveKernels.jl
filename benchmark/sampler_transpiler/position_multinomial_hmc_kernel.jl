module PositionMultinomialHMCAuthoring
using ReactiveKernels, Random, LinearAlgebra, LogExpFunctions

# Only the selected position is retained between transitions. Momentum is
# refreshed on the next transition; this kernel does not return the selected
# joint phasepoint. RK preserves reusable position-dependent caches.
@kernel multinomial_hmc_state(init; n_steps=4, step_f=nothing, stepsize=0.03) = begin
    anchor = deepcopy(init)
    work = deepcopy(init)
    step!(rng) = begin
        init.mom = Random.randn!(rng, init.mom)
        init.mom = sqrt(anchor.metric) * init.mom
        anchor.pos .= init.pos
        anchor.mom .= init.mom
        work.pos .= init.pos
        work.mom .= init.mom
        initial_ham = init.ham
        log_weight = zero(initial_ham)
        n_fwd = Random.rand(rng, 0:n_steps)
        for i in 1:n_steps
            if i == n_fwd + 1
                work .= anchor
            end
            direction = i <= n_fwd ? stepsize : -stepsize
            step_f(work; stepsize=direction)
            raw_weight = initial_ham - work.ham
            weight = (raw_weight - raw_weight == zero(raw_weight)) ? raw_weight : oftype(raw_weight, -Inf)
            log_weight = logaddexp(log_weight, weight)
            if -Random.randexp(rng) < weight - log_weight
                init.pos .= work.pos
            end
        end
    end
end
end
