# Declarative PPL kernel: beta-binomial

`ReactiveKernels` has no built-in probabilistic-programming semantics. Like the
[eight-schools example](eight-schools.md), this one assembles those semantics
manually from ordinary pure Julia recipes, while leaving the graph planner
responsible only for selecting the computation required by a particular
`have`/`want` query.

The complete runnable source is
[`examples/beta_binomial.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/examples/beta_binomial.jl).
It implements the classic shared-rate coin-flip model

```math
\begin{aligned}
p &\sim \operatorname{Beta}(2, 2), \\
k_i &\sim \operatorname{Binomial}(n_i, p),
\end{aligned}
```

where each of several experiments contributes `kᵢ` successes in `nᵢ` trials that
share one success rate `p`.

The single unconstrained coordinate is `logit_rate`. The support transform is
`rate = logistic(logit_rate)`, so the optional log absolute Jacobian determinant
is `log(rate) + log(1 - rate)`.

```text
logit_rate ──► rate = logistic(logit_rate) ──► constrained parameters
   │                     │
   │                     ├─► log prior
   │                     ├─► pointwise log likelihood ─► log likelihood
   │                     └─► expected successes (generated quantity)
   └─ rate ──► log Jacobian

log prior + log Jacobian + log likelihood ──► unconstrained log density
```

The important part is that these remain separate named ports. The compact block
below is the authored model, executed verbatim while the documentation is built.
Pointwise terms are a first-class port, so returning them together with the
scalar density shares the likelihood computation rather than repeating it.

The panel below is one coherent, build-executed artifact. **Raw input** is the
exact source that builds and runs the query, **Generated kernel** is
`code_expr(density_kernel)` from that execution, and **Compute DAG** is the live
colored `visualize(density_plan)` component.

```@eval
Main.ReactiveKernelsDocs.execute_example(@__MODULE__, raw"""
@kernel model(logit_rate::Real,
              trials::CountVector,
              successes::CountVector,
              new_trials::Int) = begin
    rate::Real = BetaBinomialExample.logistic(logit_rate)
    parameters::BetaBinomialParameters =
        BetaBinomialExample.assemble_parameters(rate)
    log_jacobian::Real = BetaBinomialExample.log_abs_det_jacobian(rate)

    prior::Real = BetaBinomialExample.log_prior(parameters)
    pointwise::NTuple{5,Real} = BetaBinomialExample.pointwise_log_likelihood(
        parameters, trials, successes,
    )
    likelihood::Real = BetaBinomialExample.sum_log_likelihood(pointwise)
    density::Real = BetaBinomialExample.total_log_density(
        prior, log_jacobian, likelihood,
    )
    expected::Real = BetaBinomialExample.expected_successes(parameters, new_trials)
    return density
end

logit_rate = 0.2
trials = BETA_BINOMIAL_TRIALS
successes = BETA_BINOMIAL_SUCCESSES

density_kernel = prepare(model;
    have = (:logit_rate, :trials, :successes),
    want = (:prior, :log_jacobian, :pointwise, :likelihood, :density))

output = density_kernel(logit_rate, trials, successes)
prior, logjac, pointwise, likelihood, density = output
@assert likelihood ≈ sum(pointwise)
@assert density ≈ prior + logjac + likelihood

docs_example = (;
    name = :beta_binomial_density,
    origin = "compact @kernel model (build executed)",
    inputs = (; logit_rate, trials, successes),
    kernel = density_kernel,
    output,
)
"""; setup = Main.ReactiveKernelsDocs.setup_beta_binomial!)
```

Asking only for constrained parameters selects just the logistic transform and
assembly recipes; the Jacobian and every density recipe disappear:

```julia
constrain_kernel = prepare(model;
    have = :logit_rate,
    want = :parameters)
parameters = constrain_kernel(logit_rate)
```

Generated quantities can start at an already-constrained boundary. Here the
expected success count in a new experiment is a deterministic function of the
rate, so planning removes the transform, Jacobian, prior, likelihood, and
total-density recipes:

```julia
generated_kernel = prepare(model;
    have = (:parameters, :new_trials),
    want = :expected)

expected = generated_kernel(parameters, 20)
```

Run the walkthrough from the repository root:

```sh
julia --project=. examples/beta_binomial.jl
```
