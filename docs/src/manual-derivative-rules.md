# Manual derivative rule graphs (design example)

One pure ReactiveKernels graph can own the primal, forward-direction, and
reverse-direction mathematics of a small numerical function. Different
HAVE→WANT cuts then remove every branch that a particular rule invocation does
not need.

!!! note "Executable design example, not a shipped adapter generator"
    This page executes current public `@kernel` and `prepare` behavior. The
    backend adapters discussed below are the approved design target; RK does
    not yet generate ChainRules, Mooncake, Enzyme, or Reactant rule methods.

The authoring boundary contains ordinary arrays and formulas. It does not
contain backend tangent types, activity annotations, thunks, accumulation
conventions, or registration declarations.

## Current capability and required RK features

Two parts of this page work in RK today: one graph can contain explicitly
authored primal/JVP/VJP formulas, and `prepare` can prune it when the caller
manually supplies the corresponding HAVE and WANT ports. Everything that turns
that graph into a registered custom AD rule is new work.

Even when the VJP formula is already present, an RK rule generator still needs
to add:

- backend-neutral roles that map graph ports to primal arguments, directions,
  output covectors, input covectors, and retainable residuals;
- activity-driven selection of the appropriate graph cut;
- two-stage reverse planning that computes and retains a residual environment
  before the output covector exists;
- a generated pullback or equivalent backend residual ABI; and
- ChainRules, Mooncake, Enzyme, and—once its upstream bridge exists—Reactant
  registration and tangent-conversion code.

The distinction between two meanings of “pure mathematical formulation” is
important. This example manually supplies both the JVP and VJP mathematics in a
pure graph. If the input were only a primal formula, generating its VJP would
also require reverse-mode AD or program transposition. RK has neither feature,
and this design does not propose reimplementing them.

The existing `prepare_ad_pullback` API is a different boundary: it asks a
`DifferentiationInterface` backend to differentiate a selected primal kernel
and evaluates the VJP for a supplied seed. It does not consume this manual rule
graph, generate a reusable `rrule`-style closure, or register a custom rule.

## One graph, three selected cuts

The panels below read the exact graph from
[`examples/manual_derivative_rule.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/examples/manual_derivative_rule.jl),
execute each prepared cut during the documentation build, and show its generated
Julia and selected compute DAG.

```@eval
Main.ReactiveKernelsDocs.render_manual_derivative_rule_cuts()
```

The primal cut has only `A` and `x`; the forward cut adds `A_dot` and `x_dot`;
the reverse cut instead adds `y_bar`. Consequently the generated forward
kernel has no output-covector path, while the generated reverse kernel has no
input-direction path.

The example writes both the JVP and VJP mathematics in the same source graph.
It does not derive one from the other. An adjoint-pairing test keeps the two
branches consistent without requiring RK to implement a general AD or program
transposition system.

## Value plus a generated-style pullback

A reverse-rule protocol receives the output covector only after the primal has
returned. The backend adapter must therefore stage the one graph: run the
primal cut, return the value and a pullback object, then run the VJP cut when
that object receives `y_bar`.

The example manually owns the following backend-neutral oracle for the desired
staging shape. RK does not generate this code today; a future adapter generator
would replace it with the target backend's exact closure or residual ABI.

```@eval
Main.ReactiveKernelsDocs.render_manual_derivative_pullback_source()
```

This build-executed receipt checks both the full and `x`-only pullbacks:

```@example manual_derivative_rules
result = Main.ManualDerivativeRuleExample.run()

@assert result.y == result.y_forward == result.y_reverse == result.y_x_only
@assert result.pullback_A_bar == result.A_bar
@assert result.pullback_x_bar == result.x_bar == result.x_only_bar
@assert result.adjoint_pair[1] ≈ result.adjoint_pair[2]
@assert result.recipe_ids == (
    primal = (1,),
    forward = (1, 2, 3, 4),
    reverse = (5, 6),
    x_reverse = (6,),
)

(;
    result.recipe_ids,
    result.captured_fields,
    result.adjoint_pair,
)
```

With both inputs active, the reference pullback retains the VJP handle, `A`,
and `x`. The `x`-only cut needs `A` but not `x`, so its pullback is smaller and
its plan contains only the `x_bar` recipe. An actual generator needs a new
cross-stage liveness/residualization pass to discover this capture set and to
retain a compact stable residual instead of an original input when the graph
provides one.

## Backend boundary

Generated ChainRules, Mooncake, and Enzyme adapters would translate the selected
numeric inputs and outputs into their respective rule protocols. Reactant rule
emission additionally depends on the upstream EnzymeMLIR custom-rule bridge;
see [Automatic differentiation through Reactant](reactant-ad.md). Until those
generators exist, this page is executable evidence for graph pruning plus a
manually authored staging oracle—not a claim that registering `matvec_rule`
changes any AD backend or that RK can currently synthesize the pullback.
