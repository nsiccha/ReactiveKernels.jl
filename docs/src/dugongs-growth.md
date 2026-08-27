# Declarative PPL kernel: dugongs (nonlinear growth)

This example ports the `dugongs` model from
[posteriordb](https://github.com/stan-dev/posteriordb) (posterior
`dugongs_data-dugongs_model`) into the same declarative-`@kernel` style as the
[eight-schools example](eight-schools.md). Unlike the GLM-shaped examples, the
mean is a **nonlinear** function of the parameters — an asymptotic growth curve.

The complete runnable source is
[`examples/dugongs_growth.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/examples/dugongs_growth.jl).
It relates the length `Yᵢ` of 27 dugongs to their age `xᵢ`:

```math
\begin{aligned}
Y_i &\sim \operatorname{Normal}(\alpha - \beta \lambda^{x_i},\; \sigma), \quad \sigma = 1/\sqrt{\tau}, \\
\alpha, \beta &\sim \operatorname{Normal}(0, 1000), \\
\lambda &\sim \operatorname{Uniform}(0.5, 1), \\
\tau &\sim \operatorname{Gamma}(10^{-4}, 10^{-4}).
\end{aligned}
```

The unconstrained vector is `(α, β, u_λ, log_τ)`. Two coordinates need a support
transform: `λ = 0.5 + 0.5·logistic(u_λ)` lands in `(0.5, 1)`, and `τ = exp(log_τ)`
is positive. The optional log absolute Jacobian determinant collects both
contributions. (As in Stan's `~` statements, the `Gamma` prior's
variate-independent normalizing constant is dropped.)

```text
unconstrained
  ├─ split ──► α, β, u_λ, log_τ ──► λ, σ ──► constrained parameters
  │                                  │
  │                                  ├─► log prior
  │                                  ├─► pointwise log likelihood ─► log likelihood
  │                                  └─► expected length at a new age
  └─ (u_λ, log_τ) ──► log Jacobian

log prior + log Jacobian + log likelihood ──► unconstrained log density
```

The panel below is one coherent, build-executed artifact. **Raw input** is the
exact source that builds and runs the query, **Generated kernel** is
`code_expr(density_kernel)` from that execution, and **Compute DAG** is the live
colored `visualize(density_plan)` component.

```@eval
Main.ReactiveKernelsDocs.execute_example(@__MODULE__, raw"""
@kernel model(unconstrained::UnconstrainedParameters,
              ages::RealVector,
              lengths::RealVector,
              new_age::Real) = begin
    (α::Real, β::Real, u_λ::Real, log_τ::Real) =
        DugongsGrowthExample.split_unconstrained(unconstrained)
    λ::Real = DugongsGrowthExample.bounded_lambda(u_λ)
    σ::Real = DugongsGrowthExample.sd_from_log_precision(log_τ)
    parameters::DugongsParameters =
        DugongsGrowthExample.assemble_parameters(α, β, λ, σ)
    log_jacobian::Real = DugongsGrowthExample.log_abs_det_jacobian(u_λ, log_τ)

    prior::Real = DugongsGrowthExample.log_prior(parameters)
    pointwise::RealVector = DugongsGrowthExample.pointwise_log_likelihood(
        parameters, ages, lengths,
    )
    likelihood::Real = DugongsGrowthExample.sum_log_likelihood(pointwise)
    density::Real = DugongsGrowthExample.total_log_density(
        prior, log_jacobian, likelihood,
    )
    predicted::Real = DugongsGrowthExample.predicted_length(parameters, new_age)
    return density
end

q = (2.7, 1.0, 1.7, log(300.0))
ages = DUGONGS_AGE
lengths = DUGONGS_LENGTH

density_kernel = prepare(model;
    have = (:unconstrained, :ages, :lengths),
    want = (:prior, :log_jacobian, :pointwise, :likelihood, :density))

output = density_kernel(q, ages, lengths)
prior, logjac, pointwise, likelihood, density = output
@assert likelihood ≈ sum(pointwise)
@assert density ≈ prior + logjac + likelihood

docs_example = (;
    name = :dugongs_density,
    origin = "compact @kernel model (build executed) — posteriordb dugongs",
    inputs = (; q, ages, lengths),
    kernel = density_kernel,
    output,
)
"""; setup = Main.ReactiveKernelsDocs.setup_dugongs!)
```

Asking only for constrained parameters selects just the transforms and assembly;
the Jacobian and every density recipe disappear:

```julia
constrain_kernel = prepare(model;
    have = :unconstrained,
    want = :parameters)
parameters = constrain_kernel(q)
```

Generated quantities can start at an already-constrained boundary. Here the
expected length at a new age is a deterministic function of the parameters, so
planning removes the transforms, Jacobian, prior, likelihood, and total-density
recipes:

```julia
generated_kernel = prepare(model;
    have = (:parameters, :new_age),
    want = :predicted)

predicted = generated_kernel(parameters, 20.0)
```

Run the walkthrough from the repository root:

```sh
julia --project=. examples/dugongs_growth.jl
```
