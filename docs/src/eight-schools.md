# Declarative PPL kernel: eight schools

`ReactiveKernels` has no built-in probabilistic-programming semantics. This
example shows how to assemble those semantics manually from ordinary pure Julia
recipes. PPL evaluation policies then reduce to named graph-output selection:
the equivalent of Wren's `Params`, `LogPrior`, and `LogLikelihood` accumulators
is one `want = (:parameters, :prior, :likelihood)` boundary.

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
          ──► scalar school likelihood ─ plate ─┬─► pointwise log likelihoods
                                                └─► fused log-likelihood sum
          ──► new-group prediction

log prior + log Jacobian + log likelihood ──► unconstrained log density
```

The block below is the authored model, executed verbatim while the documentation
is built, and it is the single source authority used by both the tests and this
build. Every mathematical operation is visible there: the split is plain
indexing, the transform is `exp(log_τ)`, and the scalar per-school likelihood is
authored once before `plate` derives its pointwise and reducing forms.
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
likelihood reduction and prunes the pointwise collector, Jacobian, density, and
prediction branches. An eventual PPL-to-RK transpiler only needs to retain the
mapping from PPL results to these named ports; RK handles extraction and Reactant
sees the resulting mathematical `PreparedKernel`. The packed `unconstrained`
vector remains an alternate boundary for native HMC, but the PPL/Reactant path
uses named latent ports and therefore never scalar-indexes a traced device array.

The unconstrained density boundary has a second, source-visible fused producer.
Planning selects that scalar loop for a density-only query, so reverse AD
neither materializes an active pointwise vector nor captures the nested plate
kernels in its operation table. The prepared density therefore differentiates
through DifferentiationInterface with plain
`AutoEnzyme(mode = Enzyme.Reverse)`, passing observations and scales as DI
`Constant`s—no Enzyme runtime-activity mode or function annotation is required.
Named-latent and pointwise queries keep the plate route above for Reactant.

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
