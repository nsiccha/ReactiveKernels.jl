# Declarative PPL kernel: eight schools

`ReactiveKernels` has no built-in probabilistic-programming semantics. This
example assembles the model from reusable declarative distribution kernels,
ordinary transforms, and `plate`. PPL evaluation policies then reduce to named graph-output selection:
the equivalent of Wren's `Params`, `LogPrior`, and `LogLikelihood` accumulators
is one `want = (:parameters, :prior, :likelihood)` boundary.

The complete runnable source is
[`packages/ReactiveKernelsPPLExamples/src/eight_schools.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsPPLExamples/src/eight_schools.jl).
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

The one model graph exposes the packed `unconstrained` input, named latent ports
`μ`/`log_τ`/`θ`, constrained `parameters`, `log_jacobian`, `prior`,
`likelihood`, `pointwise`, and `posterior`. Preparing a different HAVE/WANT
boundary changes only the generated backward slice; it does not select a second
model implementation.

The constrained `parameters` port has **two producers** — this is RK's *multiple
paths to the same port*. One assignment builds the constrained parameters alone;
a second builds them together with the log Jacobian, sharing the one transform.
The planner selects between them per query: a constrain-only query takes the
parameters-only path (the Jacobian is never computed), while any query that needs
the unconstrained posterior takes the joint path. No macro is involved — two
ordinary assignments to the same name declare the two producers.

```text
                              ┌─ parameters                    (chosen: constrain-only)
unconstrained ─ split ─ τ ────┤
                              └─ (parameters, log_jacobian)     (chosen: posterior)

parameters ──► reusable Normal/Cauchy kernels ──► log prior
          ──► reusable Normal kernel ─ plate ─┬─► pointwise log likelihoods
                                              └─► fused log-likelihood sum
          ──► new-group prediction

log prior + log Jacobian + log likelihood ──► unconstrained log posterior
```

The block below is the authored model, executed verbatim while the documentation
is built, and it is the single source authority used by both the tests and this
build. Every model-level operation is visible there: the split is plain
indexing, the transform is `exp(log_τ)`, and the likelihood and prior reuse the
Normal/Cauchy `KernelSpec`s from the distributions example. The PPL source does
not re-author either density formula; `plate` derives its pointwise and reducing
forms from the shared Normal graph.
Constrained parameters and the new-group prediction are plain `NamedTuple`s, not
custom types.

The reducing plate emits a scalar accumulator loop directly: it does not first
build a pointwise vector and therefore needs neither an output buffer nor an
allocation. Asking for `pointwise` selects the collecting plate instead, where
the one allocated vector is the requested result itself. These are static
prepared boundaries over the same scalar recipe, and both remain traceable by
Reactant.

The three Wren-style outputs are likewise not three RK-specific accumulator
types. They are a static tuple of graph nodes:

```julia
requested_nodes = (:parameters, :prior, :likelihood)
evaluation_kernel = prepare(model;
    have = (:μ, :log_τ, :θ, :observations, :observation_scales),
    want = requested_nodes)

parameters, prior, likelihood =
    evaluation_kernel(μ, log_τ, θ, observations, observation_scales)
```

Because selection happens during planning, the generated kernel contains only
the backward slice needed for those nodes. In particular, it uses the fused
likelihood reduction and prunes the pointwise collector, Jacobian, posterior, and
prediction branches. An eventual PPL-to-RK transpiler only needs to retain the
mapping from PPL results to these named ports; RK handles extraction and Reactant
sees the resulting mathematical `PreparedKernel`. The packed `unconstrained`
vector and the named latent ports are alternate HAVE boundaries over that same
graph, not alternate implementations of the mathematics.

The unconstrained posterior uses that same RK-generated transform, prior, and two
reducing `plate` kernels. Preparation splices both generated loops into one flat
posterior kernel; there is no separately handwritten fused evaluator or nested
`PreparedKernel`. The shared RK/DI preparation API marks observations and
scales constant so only the unconstrained parameters receive adjoints, without
runtime activity. The same plate-authored source retains tensorized lowering
for Reactant.

The panel below shows three views of this model: **Raw input** (the source), a
readable **Generated kernel** derived from the executed kernel and selected
plan, and the **Compute DAG** (`visualize(evaluation_kernel.plan)`). The exact
compiled AST remains available as `code_expr(evaluation_kernel)`. **Compare all** shows the
views side by side.

```@eval
Main.ReactiveKernelsDocs.execute_ppl_example(
    @__MODULE__, :EightSchoolsExample, :EIGHT_SCHOOLS_SOURCE;
    setup = Main.ReactiveKernelsDocs.setup_eight_schools!,
)
```

Asking only for constrained parameters selects the parameters-only producer, so
the split, transform, and assembly run while the joint (Jacobian-bearing)
producer and every posterior recipe disappear:

```julia
constrain_kernel = prepare(model;
    have = :unconstrained,
    want = :parameters)
parameters = constrain_kernel(q)
```

Generated quantities can start at an already-constrained boundary. In this
query, planning removes the unconstrained transform, Jacobian, prior, likelihood,
and posterior recipes, leaving only the prediction:

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
julia --project=packages -e 'using ReactiveKernelsPPLExamples; ReactiveKernelsPPLExamples.EightSchoolsExample.demo()'
```
