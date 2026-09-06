"""
    PPLWorkflow

Thin, **opt-in** helpers that turn the *implied* PPL lowering vocabulary shared
by the declarative-PPL examples into a **named contract**. Nothing here changes
core ReactiveKernels or obliges a non-PPL kernel to adopt it — the helpers are a
convenience layer over the ordinary `prepare` / `plan` HAVE/WANT surface, plus a
documented list of the standard node names and workflow cuts.

Resolution of decision
[`2026-09-06T13-25-41-598-037dt6k`](/agents/ReactiveKernels:ppl:benchmark/decisions/2026-09-06T13-25-41-598-037dt6k)
(user: "convention + thin opt-in helpers in PPL/distributions packages").

# The convention this names

A declarative-PPL `@kernel` is written as one authored graph whose Bayesian
workflow is exposed as **named nodes** and cut with `have` / `want`. The
recurring vocabulary across the examples is:

## HAVE boundaries (two producers of the parameters)
- **packed-unconstrained** — a single `unconstrained::Vector` split into the
  latent components; the native-HMC boundary.
- **named-latent** — the components (`μ`, `τ`, `θ`, …) or the constrained
  `parameters` object supplied directly; the Reactant-friendly boundary.
Both feed the *same* prior/likelihood graph; HAVE authority prunes whichever
edges the query does not need (e.g. supplying both `τ` and `log_τ` makes both
authoritative, so neither is recomputed — no `log(exp(x))` round trip).

## Standard nodes ([`PPL_NODES`](@ref))
| node | meaning |
|---|---|
| `parameters` | constrained parameters (a plain NamedTuple / struct) |
| `log_jacobian` | log\\|d(constrained)/d(unconstrained)\\| for the transform |
| `prior` | log prior in the model space |
| `pointwise` | per-observation log-likelihood terms (the authored `plate`) |
| `likelihood` | the buffer-free summed log-likelihood (same `plate`, fused) |
| `unconstrained_prior` | `prior + log_jacobian` (sampler-space prior) |
| `constrained_logdensity` | `prior + likelihood` (no transform work) |
| `posterior` | `prior + likelihood + log_jacobian` (unconstrained-space target) |

Two authored producers of `parameters` (constrain-only vs joint-with-Jacobian),
half-fold priors from base distribution objects, the plate pointwise-vs-total
duality, generated-quantity pruning (unrequested predictions never lower), and
data-only-prefix hoisting (pass the data-only inputs through `bound=` so their
subgraph is a hoisted constant) are all expressed through the ordinary surface;
the presets below just name the cuts.

## Workflow cuts ([`WORKFLOW_WANTS`](@ref) / [`workflow_wants`](@ref))
| preset | `want` | use |
|---|---|---|
| `:sampler` | `:posterior` | unconstrained-space log density for HMC |
| `:constrained` | `:constrained_logdensity` | density with no transform work |
| `:prior` | `:unconstrained_prior` | sampler-space prior only |
| `:likelihood` | `:likelihood` | fused total log-likelihood |
| `:pointwise` | `:pointwise` | per-observation terms (LOO/WAIC) |
| `:wren` | `(:parameters, :prior, :likelihood)` | the transpiler accumulator triple |

# Example

```julia
using ReactiveKernelsPPLExamples: EightSchoolsExample, PPLWorkflow
model = EightSchoolsExample.build_eight_schools_graph()

# Sampler-space density over the packed-unconstrained boundary:
density = PPLWorkflow.prepare_workflow(model, :sampler;
    have = (:unconstrained, :observations, :observation_scales))

# Same, with the data hoisted as a constant prefix:
bound_density = PPLWorkflow.prepare_workflow(model, :sampler;
    have = (:unconstrained, :observations, :observation_scales),
    bound = (; observations, observation_scales))
```
"""
module PPLWorkflow

using ReactiveKernels: prepare, plan

export PPL_NODES, WORKFLOW_WANTS, workflow_wants, prepare_workflow, plan_workflow

"""
    PPL_NODES

The canonical node names a declarative-PPL `@kernel` exposes, as a NamedTuple
mapping role → node `Symbol`. Author each example against these names so a graph
consumer (or a later transpiler) reads one contract rather than a per-example
convention. See the [`PPLWorkflow`](@ref) module docstring for each node's
meaning.
"""
const PPL_NODES = (
    parameters = :parameters,
    log_jacobian = :log_jacobian,
    prior = :prior,
    pointwise = :pointwise,
    likelihood = :likelihood,
    unconstrained_prior = :unconstrained_prior,
    constrained_logdensity = :constrained_logdensity,
    posterior = :posterior,
)

"""
    WORKFLOW_WANTS

Preset `want` selections for the recurring Bayesian-workflow cuts, as a
NamedTuple mapping preset name → the `want` argument (a bare `Symbol` for a
single node, a `Tuple` for a multi-node accumulator). Prefer
[`workflow_wants`](@ref), which validates the preset name.
"""
const WORKFLOW_WANTS = (
    sampler = :posterior,
    constrained = :constrained_logdensity,
    prior = :unconstrained_prior,
    likelihood = :likelihood,
    pointwise = :pointwise,
    wren = (:parameters, :prior, :likelihood),
)

"""
    workflow_wants(preset::Symbol)

Return the `want` selection for a named workflow `preset` (see
[`WORKFLOW_WANTS`](@ref)). Throws `ArgumentError` naming the valid presets when
`preset` is unknown, so a typo fails loudly rather than silently planning the
wrong cut.
"""
function workflow_wants(preset::Symbol)
    haskey(WORKFLOW_WANTS, preset) || throw(ArgumentError(
        "unknown PPL workflow preset $(repr(preset)); choose one of $(keys(WORKFLOW_WANTS))"))
    WORKFLOW_WANTS[preset]
end

"""
    prepare_workflow(model, preset::Symbol; have, bound = NamedTuple())

Prepare `model` for a named workflow `preset` over the supplied `have` boundary.
Thin wrapper over `ReactiveKernels.prepare(model; have, want, bound)` with
`want = workflow_wants(preset)`. Pass the data-only inputs through `bound=` to
hoist their subgraph as a constant prefix.
"""
prepare_workflow(model, preset::Symbol; have, bound = NamedTuple()) =
    prepare(model; have, want = workflow_wants(preset), bound)

"""
    plan_workflow(model, preset::Symbol; have)

Plan (without preparing) `model` for a named workflow `preset` over `have`. Thin
wrapper over `ReactiveKernels.plan(model; have, want)` with
`want = workflow_wants(preset)`; useful with `ReactiveKernels.explain` to
inspect the pruned graph a preset selects.
"""
plan_workflow(model, preset::Symbol; have) =
    plan(model; have, want = workflow_wants(preset))

end # module PPLWorkflow
