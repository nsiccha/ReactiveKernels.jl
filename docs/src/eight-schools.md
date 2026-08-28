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
          ──► log likelihood  (plated: vectorized over the schools)
          ──► new-group prediction

log prior + log Jacobian + log likelihood ──► unconstrained log density
```

The block below is the authored model, executed verbatim while the documentation
is built. Its ordinary pure recipe functions are defined in the complete source
linked above; the model source shown here is the single authority used by those
tests and by this build. The likelihood is authored ONCE as a scalar per-school
`@kernel` and `plate`d into a vectorized log density: it iterates the batched
ports and sums the scalar result in one fused pass, materializing no
per-observation vector (and hoisting any shared work above the loop).

The panel below is one coherent, build-executed artifact—not three separately
maintained snippets. **Raw input** is the exact source that builds and runs the
query, **Generated kernel** is `code_expr(density_kernel)` from that execution,
and **Compute DAG** is the live colored `visualize(density_plan)` component.
Choose **Compare all** to inspect the same source, subkernel, and selected plan
side by side without resetting the interactive DAG.

```@eval
Main.ReactiveKernelsDocs.execute_ppl_example(
    @__MODULE__, :EightSchoolsExample, :EIGHT_SCHOOLS_SOURCE;
    setup = Main.ReactiveKernelsDocs.setup_eight_schools!,
)
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
query, planning removes the unconstrained transform, Jacobian, prior, likelihood,
and total-density recipes, leaving only the prediction:

```julia
generated_kernel = prepare(model;
    have = (:parameters, :new_group_scale, :prediction_innovations),
    want = :new_group)

prediction = generated_kernel(parameters, 12.0, [0.25, -1.0])
```

Prediction takes standard-normal innovations as inputs rather than drawing
random numbers inside a recipe. That preserves the graph's purity: a caller can
draw fresh innovations, replay fixed ones, or batch them using the same prepared
kernel without hiding an effect from the planner.

Run the walkthrough from the repository root:

```sh
julia --project=. examples/eight_schools.jl
```
