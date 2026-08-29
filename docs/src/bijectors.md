# Bijectors and constrained parameters

```@eval
Main.ReactiveKernelsDocs.render_result_assets()
```

A bijector is a particularly useful have→want graph: from one unconstrained
value it can expose the constrained parameter, the log absolute Jacobian
determinant, or both. ReactiveKernels does not need a separate transform runtime
or an accumulator hierarchy for those cases. Each useful result is a named
graph port, and `prepare(...; want = ...)` selects the required backward slice
before the model runs.

The complete runnable source is
[`examples/bijectors.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/examples/bijectors.jl).
Its `BIJECTOR_KERNEL_SOURCE` constant is the single source authority executed by
the example, tests, and the panel on this page.

## One definition, three demand boundaries

For positive support, the transform and its log Jacobian are

```math
x = \exp(u), \qquad \log\left|\frac{dx}{du}\right| = u.
```

Asking only for `:constrained` emits `exp`. Asking only for `:log_jacobian`
returns the unconstrained input through an identity alias and emits no recipe.
Requesting both emits `exp` once.

The real-to-unit-interval transform uses a stable logistic decomposition. It
computes `tail = exp(-abs(u))`, derives the constrained value without overflow,
and evaluates

```math
\log\left|\frac{dx}{du}\right|
= -\operatorname{log1pexp}(-u) - \operatorname{log1pexp}(u).
```

The log Jacobian therefore stays finite for every finite input even when the
floating-point constrained value rounds to zero or one. Its constrained-only
plan stops before the logarithmic nodes; its Jacobian-only plan omits the
constrained-value recipe; the joint plan shares the magnitude and exponential
tail.

## Reuse inside one fused model graph

`fused_bijector_model` calls both reusable `KernelSpec`s directly:

```julia
@kernel fused_bijector_model(log_scale::Float64,
                             logit_probability::Float64) = begin
    (scale::Float64, scale_log_jacobian::Float64) =
        positive_bijector(log_scale)
    (probability::Float64, probability_log_jacobian::Float64) =
        unit_interval_bijector(logit_probability)
    parameters = (; scale, probability)
    log_jacobian::Float64 =
        scale_log_jacobian + probability_log_jacobian
    return (parameters, log_jacobian)
end
```

Those calls are graph-construction operations, not runtime dispatch. RK clones
each child with fresh value identities, binds its typed boundary to the parent
ports, and inserts its recipes into the parent. Planning then sees one graph:

- `want = :parameters` selects 5 recipes and omits every Jacobian-only node;
- `want = :log_jacobian` selects 7 recipes and omits both constrained-value
  consumers; and
- requesting both selects 10 recipes, sharing the unit-interval tail work.

The build-executed panel below evaluates the exact source authority, asserts
those three plan sizes, checks that the generated joint kernel names neither
child kernel, and renders the selected fused DAG.

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.BijectorKernelExample.BIJECTOR_DOCS_SOURCE,
)
```

## Composition contract

Reusable child kernels currently have a deliberately static splice boundary:

- the callee resolves to a global stateless `KernelSpec`;
- arguments are existing parent graph ports and are passed positionally;
- the child HAVE and WANT boundaries are supplied and destructured exactly;
- corresponding parent and child ports have exact declared types; and
- recipe cost/CSE metadata belongs inside the reusable child, not at the call
  site.

Assign a larger argument expression to its own parent recipe before passing it
to a child. Supply optional positional child inputs explicitly. Keyword-boundary
specs and dynamic callees are rejected while the outer graph is constructed,
so no nested planner, dynamic dispatch, or hidden transform branch reaches the
prepared hot path.

## Differentiation and allocation behavior

Every prepared boundary is an ordinary straight-line Julia callable. The test
suite exercises the constrained and Jacobian wants independently with
DifferentiationInterface and plain Enzyme reverse mode, including the fused
model Jacobian. The standalone and fused prepared kernels are inferred and
allocate zero bytes in steady state after warm-up.

This page covers transform construction and fusion. See
[Eight schools](eight-schools.md) for these named outputs inside a complete PPL
model, [Compiler capability and limits](compiler.md) for the full authoring
contract, and [Non-allocating kernels](nonallocating.md) when requested outputs
contain reusable arrays.
