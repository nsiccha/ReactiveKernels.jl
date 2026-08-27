# Declarative PPL kernel: eight schools

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

The unconstrained vector is `(μ, log_τ, θ₁, …, θ₈)`. Only `τ` needs a support
transform, so `τ = exp(log_τ)` and the optional log absolute Jacobian determinant
is `log_τ`.

The constrained `parameters` port has **two producers** — this is RK's *multiple
paths to the same port*. One assignment builds the constrained parameters alone;
a second builds them together with the log Jacobian, sharing the one transform.
The planner selects between them per query: a constrain-only query takes the
parameters-only path (the Jacobian is never computed), while any query that needs
the unconstrained density takes the joint path. No macro is involved — two
ordinary assignments to the same name declare the two producers.

```text
                              ┌─ parameters                    (chosen: constrain-only)
unconstrained ─ split ─ τ ────┤
                              └─ (parameters, log_jacobian)     (chosen: density)

parameters ──► log prior
          ──► pointwise log likelihood ──► log likelihood
          ──► new-group prediction

log prior + log Jacobian + log likelihood ──► unconstrained log density
```

The block below is the authored model, executed verbatim while the documentation
is built. Every recipe body is written out inline — the transform, the prior, the
pointwise likelihood — so nothing is hidden behind a helper call; what you read is
exactly what the planner lowers. Pointwise terms are a first-class port too, so
returning them with the scalar density shares the likelihood computation rather
than repeating it.

The panel below is one coherent, build-executed artifact—not three separately
maintained snippets. **Raw input** is the exact source that builds and runs the
query, **Generated kernel** is `code_expr(density_kernel)` from that execution,
and **Compute DAG** is the live colored `visualize(density_plan)` component.
Choose **Compare all** to inspect the same source, subkernel, and selected plan
side by side without resetting the interactive DAG.

```@eval
Main.ReactiveKernelsDocs.execute_example(@__MODULE__, raw"""
@kernel model(unconstrained::Vector{Float64},
              observations::Vector{Float64},
              observation_scales::Vector{Float64},
              new_group_scale::Float64,
              prediction_innovations::Vector{Float64}) = begin
    # Split the unconstrained vector into (μ, log_τ, θ).
    μ::Float64 = unconstrained[1]
    log_τ::Float64 = unconstrained[2]
    θ::Vector{Float64} = unconstrained[3:end]

    # Support transform for the scale: τ = exp(log_τ).
    τ::Float64 = exp(log_τ)

    # Two producers for the SAME `parameters` port — RK's "multiple paths to one
    # port". The first yields the constrained parameters alone; the second yields
    # them together with the log Jacobian log|dτ/dlog_τ| = log_τ, sharing the
    # transform. The planner picks the first for a constrain-only query and the
    # second whenever the Jacobian — hence the unconstrained density — is wanted.
    parameters::EightSchoolsParameters = EightSchoolsParameters(μ, τ, θ)
    (parameters::EightSchoolsParameters, log_jacobian::Float64) =
        (EightSchoolsParameters(μ, τ, θ), log_τ)

    # log prior:  μ ~ Normal(0, 5),  τ ~ HalfCauchy(0, 5),  θⱼ ~ Normal(μ, τ).
    prior::Float64 =
        (-0.5 * log(2π) - log(5.0) - 0.5 * (μ / 5.0)^2) +
        (log(2) - log(π) - log(5.0) - log1p((τ / 5.0)^2)) +
        sum(-0.5 * log(2π) - log(τ) - 0.5 * ((θ[j] - μ) / τ)^2 for j in 1:8)

    # Pointwise log likelihood:  yⱼ ~ Normal(θⱼ, σⱼ).
    pointwise::Vector{Float64} = [
        -0.5 * log(2π) - log(observation_scales[j]) -
            0.5 * ((observations[j] - θ[j]) / observation_scales[j])^2
        for j in 1:8
    ]
    likelihood::Float64 = sum(pointwise)

    # Unconstrained-space log density.
    density::Float64 = prior + log_jacobian + likelihood

    # Deterministic new-group prediction from standard-normal innovations.
    θ_new::Float64 = μ + τ * prediction_innovations[1]
    y_new::Float64 = θ_new + new_group_scale * prediction_innovations[2]
    new_group::NewGroupPrediction = NewGroupPrediction(θ_new, y_new)

    return density
end

q = [1.5, log(2.0), (0.25 .* (1:8))...]
observations = EIGHT_SCHOOLS_Y
observation_scales = EIGHT_SCHOOLS_SIGMA

density_kernel = prepare(model;
    have = (:unconstrained, :observations, :observation_scales),
    want = (:prior, :log_jacobian, :pointwise, :likelihood, :density))

output = density_kernel(q, observations, observation_scales)
prior, logjac, pointwise, likelihood, density = output
@assert likelihood ≈ sum(pointwise)
@assert density ≈ prior + logjac + likelihood

docs_example = (;
    name = :eight_schools_density,
    origin = "compact @kernel model (build executed)",
    inputs = (; q, observations, observation_scales),
    kernel = density_kernel,
    output,
)
"""; setup = Main.ReactiveKernelsDocs.setup_eight_schools!)
```

Asking only for constrained parameters selects the parameters-only producer, so
the split, transform, and assembly run while the joint (Jacobian-bearing)
producer and every density recipe disappear:

```julia
constrain_kernel = prepare(model;
    have = :unconstrained,
    want = :parameters)
parameters = constrain_kernel(q)
```

Generated quantities can start at an already-constrained boundary. In this
query, planning removes the unconstrained transform, Jacobian, prior, likelihood
reduction, and total-density recipes:

```julia
generated_kernel = prepare(model;
    have = (:parameters, :observations, :observation_scales,
            :new_group_scale, :prediction_innovations),
    want = (:pointwise, :new_group))

pointwise, prediction = generated_kernel(
    parameters, observations, observation_scales, 12.0, [0.25, -1.0])

prediction
```

Prediction takes standard-normal innovations as inputs rather than drawing
random numbers inside a recipe. That preserves the graph's purity: a caller can
draw fresh innovations, replay fixed ones, or batch them using the same prepared
kernel without hiding an effect from the planner.

Run the walkthrough from the repository root:

```sh
julia --project=. examples/eight_schools.jl
```
