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
preserving the DAG's fit, zoom, pan, and node-inspection state.

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

The same compiled reactive program then drives warmup and sampling:

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
