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

The important part is that these remain separate named ports. The compact block
below is the authored model—not repository-loading plumbing—and is executed
verbatim while the documentation is built. Pointwise terms are a first-class
port, so returning them together with the scalar density shares the likelihood
computation rather than repeating it.

The panel below is one coherent, build-executed artifact—not three separately
maintained snippets. **Raw input** is the exact source that builds and runs the
query, **Generated kernel** is `code_expr(density_kernel)` from that execution,
and **Compute DAG** is the live colored `visualize(density_plan)` component.
Choose **Compare all** to inspect the same source, subkernel, and selected plan
side by side without resetting the interactive DAG.

```@eval
Main.ReactiveKernelsDocs.execute_example(@__MODULE__, raw"""
@kernel model(unconstrained::UnconstrainedParameters,
              observations::SchoolVector,
              observation_scales::SchoolVector,
              new_group_scale::Real,
              prediction_innovations::PredictionInnovations) = begin
    (μ::Real, log_τ::Real, θ::SchoolVector) =
        EightSchoolsExample.split_unconstrained(unconstrained)
    τ::Real = EightSchoolsExample.positive_scale(log_τ)
    parameters::EightSchoolsParameters =
        EightSchoolsExample.assemble_parameters(μ, τ, θ)
    log_jacobian::Real =
        EightSchoolsExample.log_abs_det_jacobian(log_τ)

    prior::Real = EightSchoolsExample.log_prior(parameters)
    pointwise::SchoolVector = EightSchoolsExample.pointwise_log_likelihood(
        parameters, observations, observation_scales,
    )
    likelihood::Real = EightSchoolsExample.sum_log_likelihood(pointwise)
    density::Real = EightSchoolsExample.total_log_density(
        prior, log_jacobian, likelihood,
    )
    new_group::NewGroupPrediction = EightSchoolsExample.predict_new_group(
        parameters, new_group_scale, prediction_innovations,
    )
    return density
end

q = (1.5, log(2.0), ntuple(i -> 0.25 * i, 8)...)
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

Asking only for constrained parameters selects just the split, positive-scale,
and assembly recipes; the Jacobian and every density recipe disappear:

```julia
constrain_kernel = prepare(model;
    have = :unconstrained,
    want = :parameters)
parameters = constrain_kernel(q)
```

The numeric graph ports are typed at a `Real` boundary, while constrained
parameters and predictions retain their concrete scalar type. The prepared
kernel therefore specializes on ordinary `Float64` inputs and also differentiates
cleanly through reverse-mode AD (`DifferentiationInterface` with the Enzyme
backend):

```julia
using DifferentiationInterface
import Enzyme

# Reverse-mode Enzyme with runtime activity enabled: the prepared kernel closes
# over constant model data (the observations and their scales) that Enzyme's
# static activity analysis cannot prove non-differentiable. The closure is
# annotated `Const` because only the numeric input is differentiated.
enzyme_backend = AutoEnzyme(;
    mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
    function_annotation = Enzyme.Const,
)

density_only = prepare(model;
    have = (:unconstrained, :observations, :observation_scales),
    want = :density)
logdensity(qv) = density_only(Tuple(qv), observations, observation_scales)
gradient = DifferentiationInterface.gradient(logdensity, enzyme_backend, collect(q))
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
    parameters, observations, observation_scales, 12.0, (0.25, -1.0))

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
