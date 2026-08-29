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
is built, and it is the single source authority used by both the tests and this
build. Every operation is written out **inline** — the split is plain indexing,
the transform is `exp(log_τ)`, and the prior and per-school likelihood are the
distribution math spelled out directly — so nothing is hidden behind an opaque
helper call: what you read in **Raw input** is exactly what runs. Constrained
parameters and the new-group prediction are plain `NamedTuple`s, not custom
types.

The panel below shows three views of this model: **Raw input** (the source), a
readable **Generated kernel** derived from the executed kernel and selected
plan, and the **Compute DAG** (`visualize(density_plan)`). The exact compiled AST
remains available as `code_expr(density_kernel)`. **Compare all** shows the
views side by side.

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
