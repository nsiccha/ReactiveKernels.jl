# Manual PPL graph: eight schools

`ReactiveKernels` has no built-in probabilistic-programming semantics. This
example shows how to assemble those semantics manually from ordinary pure Julia
recipes, while leaving the graph planner responsible only for selecting the
computation required by a particular `have`/`want` query.

The complete runnable source is
[`examples/eight_schools.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/examples/eight_schools.jl).
It implements the centered model

```math
\begin{aligned}
\mu &\sim \operatorname{Normal}(0, 5), \\
\tau &\sim \operatorname{HalfCauchy}(0, 5), \\
\theta_j &\sim \operatorname{Normal}(\mu, \tau), \\
y_j &\sim \operatorname{Normal}(\theta_j, \sigma_j).
\end{aligned}
```

The unconstrained vector is
`(μ, log_τ, θ₁, …, θ₈)`. Only `τ` needs a support transform, so
`τ = exp(log_τ)` and the optional log absolute Jacobian determinant is
`log_τ`.

```text
unconstrained
  ├─ split ──► μ, log_τ, θ ──► τ = exp(log_τ) ──► constrained parameters
  │                                      │
  │                                      ├─► log prior
  │                                      ├─► pointwise log likelihood ─► log likelihood
  │                                      └─► new-group prediction
  └─ log_τ ──► log Jacobian

log prior + log Jacobian + log likelihood ──► unconstrained log density
```

The important part is that these remain separate graph values. Asking only for
the constrained parameters omits the Jacobian and every density recipe:

```julia
model = build_eight_schools_graph()

k = prepare(model.graph;
    have = (model.unconstrained,),
    want = (model.parameters,))

parameters = k(q)
```

The log-density query requests the full decomposition. Pointwise terms are a
first-class node, so returning them together with the scalar density shares the
likelihood computation rather than repeating it:

```julia
k = prepare(model.graph;
    have = (model.unconstrained, model.observations,
            model.observation_scales),
    want = (model.prior, model.log_jacobian, model.pointwise,
            model.likelihood, model.density))

prior, logjac, pointwise, likelihood, density =
    k(q, observations, observation_scales)

@assert likelihood ≈ sum(pointwise)
@assert density ≈ prior + logjac + likelihood
```

The numeric graph ports are typed at a `Real` boundary, while constrained
parameters and predictions retain their concrete scalar type. The prepared
kernel therefore specializes on ordinary `Float64` inputs and also accepts the
dual-number tuples created by forward-mode AD:

```julia
using ForwardDiff

logdensity(qv) = k(Tuple(qv), observations, observation_scales)
gradient = ForwardDiff.gradient(logdensity, collect(q))
```

Generated quantities can start at an already-constrained boundary. In this
query, planning removes the unconstrained transform, Jacobian, prior, likelihood
reduction, and total-density recipes:

```julia
k = prepare(model.graph;
    have = (model.parameters, model.observations, model.observation_scales,
            model.new_group_scale, model.prediction_innovations),
    want = (model.pointwise, model.new_group))

pointwise, prediction = k(parameters, observations, observation_scales,
                          σ_new, (ϵ_θ, ϵ_y))
```

Prediction takes standard-normal innovations as inputs rather than drawing
random numbers inside a recipe. That preserves the graph's purity: a caller can
draw fresh innovations, replay fixed ones, or batch them using the same prepared
kernel without hiding an effect from the planner.

Run the walkthrough from the repository root:

```sh
julia --project=. examples/eight_schools.jl
```
