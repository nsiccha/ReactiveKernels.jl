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

The likelihood port has two equivalent producers. Asking for pointwise terms
selects the vector plus `sum`, while a density-only plan selects a fused scalar
loop. The latter avoids an active temporary vector and is exercised with plain
reverse-mode Enzyme through DifferentiationInterface (data passed as
`Constant`s), without runtime activity or a function annotation.

The panel below shows three views of this model: **Raw input** (the source), a
readable **Generated kernel** derived from the executed kernel and selected
plan, and the **Compute DAG** (`visualize(density_plan)`). The exact compiled AST
remains available as `code_expr(density_kernel)`.

```@eval
Main.ReactiveKernelsDocs.execute_ppl_example(
    @__MODULE__, :DugongsGrowthExample, :DUGONGS_SOURCE;
    setup = Main.ReactiveKernelsDocs.setup_dugongs!,
)
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
