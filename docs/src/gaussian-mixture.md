# Declarative PPL kernel: Gaussian mixture (marginalization)

This example ports the `low_dim_gauss_mix` model from
[posteriordb](https://github.com/stan-dev/posteriordb) (posterior
`low_dim_gauss_mix-low_dim_gauss_mix`) into the same declarative-`@kernel` style
as the [eight-schools example](eight-schools.md). Its distinctive structure is
**marginalization**: each observation's discrete component label is integrated
out analytically, exactly as Stan does with `log_mix` (a numerically stable
two-term `log_sum_exp`). No discrete parameter ever appears in the graph.

The complete runnable source is
[`examples/gaussian_mixture.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/examples/gaussian_mixture.jl).
It models observations `yₙ` as a two-component mixture:

```math
\begin{aligned}
p(y_n) &= \theta\, \operatorname{Normal}(y_n \mid \mu_1, \sigma_1) + (1-\theta)\, \operatorname{Normal}(y_n \mid \mu_2, \sigma_2), \\
\mu_1, \mu_2 &\sim \operatorname{Normal}(0, 2), \quad \sigma_1, \sigma_2 \sim \operatorname{HalfNormal}(2), \quad \theta \sim \operatorname{Beta}(5, 5).
\end{aligned}
```

The unconstrained vector is `(μ₁, δ, log_σ₁, log_σ₂, logit_θ)`. The means are kept
**ordered** — `μ₂ = μ₁ + exp(δ)` — to break the mixture's label-switching
symmetry (Stan's `ordered[2]`); each `σ` uses `exp`, and `θ` uses `logistic`. The
optional log absolute Jacobian determinant collects all four transform
contributions. (As in Stan's `~`, the `Beta` prior's variate-independent constant
is dropped.)

```text
unconstrained ──► split ──► μ₁, δ, log_σ₁, log_σ₂, logit_θ
                              │
                              ├─► ordered means μ₁,μ₂ ; σ₁,σ₂ ; θ ──► constrained parameters
                              │            │
                              │            ├─► log prior
observations ─────────────────┴───────────►│ pointwise log_mix ─► log likelihood
                                           └─► component responsibility (generated)
(δ, log_σ₁, log_σ₂, θ) ──► log Jacobian

log prior + log Jacobian + log likelihood ──► unconstrained log density
```

The panel below shows three views of this model: **Raw input** (the source),
**Generated kernel** (`code_expr(density_kernel)`), and **Compute DAG**
(`visualize(density_plan)`).

```@eval
Main.ReactiveKernelsDocs.execute_ppl_example(
    @__MODULE__, :GaussianMixtureExample, :GAUSSIAN_MIXTURE_SOURCE;
    setup = Main.ReactiveKernelsDocs.setup_gaussian_mixture!,
)
```

Asking only for constrained parameters selects just the ordered-means transform,
the two scale transforms, the weight transform, and assembly; the Jacobian and
every density recipe disappear:

```julia
constrain_kernel = prepare(model;
    have = :unconstrained,
    want = :parameters)
parameters = constrain_kernel(q)
```

Generated quantities can start at an already-constrained boundary. Here the
posterior responsibility of component 1 for a new observation — the soft
assignment the marginalization sums over — is a deterministic function of the
parameters, so planning removes the transforms, Jacobian, prior, likelihood, and
total-density recipes:

```julia
generated_kernel = prepare(model;
    have = (:parameters, :new_point),
    want = :responsibility)

responsibility = generated_kernel(parameters, 2.5)
```

Run the walkthrough from the repository root:

```sh
julia --project=. examples/gaussian_mixture.jl
```
