# Graph-backed NUTS sampling

`ReactiveKernels` includes the multinomial NUTS transition and warmup utilities
ported from ReactiveHMC.jl. The transition follows Hoffman and Gelman's
[Algorithm 3](https://jmlr.org/papers/v15/hoffman14a.html); Hamiltonian fields
come from a compiled reactive program, so mutating `pos`, `mom`, or `metric`
invalidates and lazily recomputes only their downstream recipes.

The complete runnable source is
[`examples/nuts.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/examples/nuts.jl).
Its model and Hamiltonian are ordinary declarative recipes:

```@example nuts
using LinearAlgebra, Random, ReactiveKernels

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
nothing # hide
```

The same public spec exposes the selected plan, generated Hamiltonian getter,
and colored compute DAG before sampling:

```@example nuts
dimension = 4
point = euclidean_phasepoint(model, (
    pos = zeros(dimension),
    mom = zeros(dimension),
    metric = Matrix{Float64}(I, dimension, dimension),
))
selected_plan = explain(plan(point))
generated_hamiltonian = code_expr(point, :ham)
dag = visualize(plan(point))

sampler = nuts_state(point;
    rng = Xoshiro(20260825),
    step_f = partial(leapfrog!; stepsize = 0.35),
    max_depth = 7)
warmup = warmup!(sampler, 50)
chain = sample!(sampler, 100)

(selected_plan, generated_hamiltonian, dag,
 count(diagnostic -> diagnostic.diverged, chain.diagnostics))
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
