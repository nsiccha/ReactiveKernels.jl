module MultinomialHMCAuthoring

using ReactiveKernels, Random, LinearAlgebra, LogExpFunctions

# A mathematical source consumer of the generic compiler. The initial point
# lies uniformly among the L+1 trajectory positions. Streaming weighted
# selection retains a single candidate rather than storing every phasepoint.
@kernel multinomial_hmc_state(init; n_steps=4, step_fwd=nothing, step_bwd=nothing) = begin
    fwd = deepcopy(init)
    bwd = deepcopy(init)
    step!(rng) = begin
        init.mom = Random.randn!(rng, init.mom)
        init.mom = sqrt(fwd.metric) * init.mom
        fwd.pos .= init.pos
        fwd.mom .= init.mom
        bwd.pos .= init.pos
        bwd.mom .= init.mom
        initial_ham = init.ham
        log_weight = zero(initial_ham)
        n_fwd = Random.rand(rng, 0:n_steps)
        for i in 1:n_steps
            if i <= n_fwd
                step_fwd(fwd)
                raw_weight = initial_ham - fwd.ham
                weight = (raw_weight - raw_weight == zero(raw_weight)) ? raw_weight : oftype(raw_weight, -Inf)
                log_weight = logaddexp(log_weight, weight)
                if -Random.randexp(rng) < weight - log_weight
                    init .= fwd
                end
            else
                step_bwd(bwd)
                raw_weight = initial_ham - bwd.ham
                weight = (raw_weight - raw_weight == zero(raw_weight)) ? raw_weight : oftype(raw_weight, -Inf)
                log_weight = logaddexp(log_weight, weight)
                if -Random.randexp(rng) < weight - log_weight
                    init .= bwd
                end
            end
        end
    end
end

end
