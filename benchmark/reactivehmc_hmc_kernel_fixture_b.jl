# Parallel mutation-profile-B translation of the source-faithful
# ReactiveKernels HMC fixture. The locked fixture remains byte-identical in
# reactivehmc_hmc_kernel_fixture.jl. Ordinary `=` retains value/as-if semantics
# and permits only unobservable backend storage reuse; `.=` preserves
# destination identity. Bare causal calls are confined to exact captured RK
# methods/callable ports.
#
# Source-faithful ReactiveKernels translation of ReactiveHMC.jl src/hmc.jl at
# ca9ea4ca41924bb0e1fadc01c717e1333916aba6.
#
# Authoring-boundary changes are limited to the generic seams already accepted
# by the NUTS fixture: runtime `rng` is threaded into step!, `rcopy!` becomes
# an identity-preserving structured `.=` write, `randbernoullilog` is captured
# as a nested method, and numeric defaults derive from `init.ham`. The sampler's
# mathematical/control order is otherwise unchanged.
module ReactiveHMCHMCBMutationAuthoringFixture

using LinearAlgebra
using Random
using ReactiveKernels

@kernel hmc_state(
    init;
    n_steps=1,
    min_dham=oftype(init.ham, -1000),
    step_f=nothing,
    stats_f=nothing,
) = begin
    gofwd = true
    fwd = deepcopy(init)
    dham = zero(init.ham)
    diverged = !(dham >= min_dham)
    randbernoullilog(rng, logprob) = logprob > zero(logprob) ? true : -Random.randexp(rng) < logprob
    step!(rng) = begin
        init.mom = Random.randn!(rng, init.mom)
        init.mom = sqrt(fwd.metric) * init.mom
        fwd.pos .= init.pos
        fwd.mom .= init.mom
        for _ in 1:n_steps
            step_f(fwd)
            raw_dham = init.ham - fwd.ham
            dham = (raw_dham - raw_dham == zero(raw_dham)) ? raw_dham : -(one(raw_dham) / zero(raw_dham))
            stats_f(__self__)
            diverged && return
        end
        if randbernoullilog(__self__, rng, dham)
            init .= fwd
        end
    end
end

end # module ReactiveHMCHMCBMutationAuthoringFixture
