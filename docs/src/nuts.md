# Graph-backed NUTS sampling

`ReactiveKernels` includes the multinomial NUTS transition and warmup utilities
ported from ReactiveHMC.jl. The transition follows Hoffman and Gelman's
[Algorithm 3](https://jmlr.org/papers/v15/hoffman14a.html); Hamiltonian fields
come from a compiled reactive program, so mutating `pos`, `mom`, or `metric`
invalidates and lazily recomputes only their downstream recipes.

The complete runnable source is
[`examples/nuts.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/examples/nuts.jl).
Its model and Hamiltonian are ordinary declarative recipes. The panel below is
one build-executed artifact: **Raw input** is the exact public `@kernel` source
and query, **Generated kernel** is the actual prepared Hamiltonian subkernel,
and **Compute DAG** is the live colored `visualize(hamiltonian_kernel.plan)`
component. Choose **Compare all** to inspect the three views side by side while
preserving the DAG's fit, zoom, pan, and node-inspection state. The generated
pane is **not** the NUTS transition implementation: it supplies the Hamiltonian
and derivatives consumed by that ordinary Julia implementation.

```@eval
Main.ReactiveKernelsDocs.execute_example(@__MODULE__, raw"""
using LinearAlgebra

potential(position) = sum(abs2, position) / 2
potential_gradient(position) = (potential(position), copy(position))

model = @kernel begin
    pos::Vector{Float64}
    mom::Vector{Float64}
    metric::Matrix{Float64}
    chol_metric::Cholesky{Float64,Matrix{Float64}} = cholesky(metric)
    pot::Float64 = potential(pos)
    (pot, dpot_dpos::Vector{Float64}) = potential_gradient(pos)
    (kin::Float64, dham_dmom::Vector{Float64}) = begin
        velocity = chol_metric \ mom
        (0.5 * (logdet(chol_metric) + dot(mom, velocity)), velocity)
    end
    ham::Float64 = pot + kin
    dham_dpos::Vector{Float64} = dpot_dpos
    return (pot, dpot_dpos, chol_metric, kin, ham, dham_dpos, dham_dmom)
end

dimension = 4
inputs = (;
    pos = zeros(dimension),
    mom = zeros(dimension),
    metric = Matrix{Float64}(I, dimension, dimension),
)
hamiltonian_kernel = prepare(model;
    have = (:pos, :mom, :metric),
    want = (:ham, :dham_dpos, :dham_dmom),
)
output = hamiltonian_kernel(Tuple(inputs)...)

docs_example = (;
    name = :nuts_hamiltonian,
    origin = "declarative NUTS Hamiltonian (build executed)",
    inputs,
    kernel = hamiltonian_kernel,
    output,
)
""")
```

## What is generated vs. what implements NUTS

The three-view artifact above compiles only the model-specific numerical layer:
`ham`, `dham_dpos`, and `dham_dmom`. The multinomial tree-building and stopping
logic from Algorithm 3 lives in
[`src/hmc.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/6520e04d24616b1601c649f6ff27dc6a4ac83004/src/hmc.jl):

- [`NUTSState` and `nuts_state`](https://github.com/nsiccha/ReactiveKernels.jl/blob/6520e04d24616b1601c649f6ff27dc6a4ac83004/src/hmc.jl#L424-L495)
  own the endpoints, partial trees, proposals, direction, stopping flags, and
  transition diagnostics.
- [`_start_tree!` and `_finish_tree!`](https://github.com/nsiccha/ReactiveKernels.jl/blob/6520e04d24616b1601c649f6ff27dc6a4ac83004/src/hmc.jl#L562-L653)
  recursively integrate, combine multinomial weights, detect divergence, and
  apply the generalized endpoint-momentum U-turn criterion.
- [`step!`, `refresh_momentum!`, and `sample!`](https://github.com/nsiccha/ReactiveKernels.jl/blob/6520e04d24616b1601c649f6ff27dc6a4ac83004/src/hmc.jl#L655-L735)
  form the transition boundary; [`warmup!`](https://github.com/nsiccha/ReactiveKernels.jl/blob/6520e04d24616b1601c649f6ff27dc6a4ac83004/src/hmc.jl#L934-L1050)
  adds initial step-size search, dual averaging, and Stan-style metric windows.

The real transition entry points are concise. `sample!` refreshes momentum and
delegates to `step!`; `step!` grows successive tree depths through
`_finish_tree!`/`_start_tree!`, performs multinomial proposal selection, stops
on divergence or a U-turn, and installs the selected proposal:

```julia
function sample!(state::NUTSState)
    state.stats_f isa TrajectoryStats && reset!(state.stats_f, state.init)
    refresh_momentum!(state)
    step!(state)
end

function step!(state::NUTSState)
    _reset_transition!(state)
    backward = _backward(state)
    @. backward.mom *= -1
    state.trees[1].log_weight[1] = 0

    for depth in 1:state.max_depth
        rand(state.rng, Bool) && _flip!(state, depth)
        _finish_tree!(state, depth)
        state.depth = depth
        state.may_sample || break
        if _rand_bernoulli_log(
                state.rng,
                state.trees[depth].log_weight[1] -
                    state.trees[depth].log_weight[2],
            )
            _swap_proposal!(state, depth)
        end
        state.may_continue || break
    end
    copyto!(state.init, state.proposals[end])
    diagnostics(state)
end
```

The public workflow wires the generated Hamiltonian provider into that Julia
NUTS implementation, then runs warmup and sampling:

```@example nuts
using Random

point = euclidean_phasepoint(model, inputs)
sampler = nuts_state(point;
    rng = Xoshiro(20260825),
    step_f = partial(leapfrog!; stepsize = 0.35),
    max_depth = 7)
warmup = warmup!(sampler, 50)
chain = sample!(sampler, 100)

count(diagnostic -> diagnostic.diverged, chain.diagnostics)
```

`warmup!` performs initial step-size search, dual averaging, and windowed
diagonal metric adaptation. `sample!` returns samples and per-transition
diagnostics, including acceptance, tree depth, leapfrog count, energy error,
and divergence status.

For a reproducible comparison under identical four-chain settings, run
`julia --startup-file=no benchmark/nuts_comparison.jl`. That script creates a
temporary environment and pins AdvancedHMC and DynamicHMC outside the package's
dependencies. Treat its setup time, sampling time, gradient efficiency,
divergences, ESS, and R-hat as separate measurements; it is not evidence of
blanket sampler superiority.
