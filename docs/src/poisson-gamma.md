# Declarative PPL kernel: Poisson-Gamma

`ReactiveKernels` has no built-in probabilistic-programming semantics. Like the
[eight-schools example](eight-schools.md), this one assembles those semantics
manually from ordinary pure Julia recipes, while leaving the graph planner
responsible only for selecting the computation required by a particular
`have`/`want` query.

The complete runnable source is
[`examples/poisson_gamma.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/examples/poisson_gamma.jl).
It implements a shared-rate count model

```math
\begin{aligned}
\lambda &\sim \operatorname{Gamma}(2, 1), \\
y_i &\sim \operatorname{Poisson}(\lambda),
\end{aligned}
```

where several event counts share one Poisson rate `λ` (with the Gamma prior
parameterized by shape `2` and rate `1`).

The single unconstrained coordinate is `log_rate`. The support transform is
`λ = exp(log_rate)`, so the optional log absolute Jacobian determinant is
`log_rate`.

```text
log_rate ──► λ = exp(log_rate) ──► constrained parameters
   │               │
   │               ├─► log prior
   │               ├─► pointwise log likelihood ─► log likelihood
   │               └─► expected count (generated quantity)
   └─ log_rate ──► log Jacobian

log prior + log Jacobian + log likelihood ──► unconstrained log density
```

These remain separate named ports: pointwise terms are a first-class port, so
returning them together with the scalar density shares the likelihood
computation rather than repeating it.

The panel below shows three views of this model: **Raw input** (the source),
**Generated kernel** (`code_expr(density_kernel)`), and **Compute DAG**
(`visualize(density_plan)`).

```@eval
Main.ReactiveKernelsDocs.execute_ppl_example(
    @__MODULE__, :PoissonGammaExample, :POISSON_GAMMA_SOURCE;
    setup = Main.ReactiveKernelsDocs.setup_poisson_gamma!,
)
```

Asking only for constrained parameters selects just the exponential transform
and assembly recipes; the Jacobian and every density recipe disappear:

```julia
constrain_kernel = prepare(model;
    have = :log_rate,
    want = :parameters)
parameters = constrain_kernel(log_rate)
```

Generated quantities can start at an already-constrained boundary. Here the
expected number of events over a future window is a deterministic function of
the rate, so planning removes the transform, Jacobian, prior, likelihood, and
total-density recipes:

```julia
generated_kernel = prepare(model;
    have = (:parameters, :exposure),
    want = :expected)

expected = generated_kernel(parameters, 4.0)
```

Run the walkthrough from the repository root:

```sh
julia --project=. examples/poisson_gamma.jl
```
