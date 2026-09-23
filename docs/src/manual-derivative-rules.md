# Derivative rules from one pure-math graph

A small numerical function's derivatives are authored once, as an ordinary
pure `@kernel` graph that computes the primal *and* its derivative branches.
ReactiveKernels generates everything else from that graph: an RK-owned
callable, one HAVE→WANT cut per activity pattern, and the AD-protocol
adapters. The authoring boundary contains ordinary arrays and formulas. It
does not contain backend tangent types, activity annotations, thunks,
accumulation conventions, or registration declarations, and no rule attaches
to a function this repository does not own (the rule constraint on
[Core constraints](constraints.md)).

Two generators cover two shapes of graph:

- [`scalar_derivative_rule`](@ref) — scalar inputs and a scalar result; the
  graph authors the primal plus one partial derivative per input.
- [`derivative_rule`](@ref) — array or scalar inputs and results; the graph
  authors a forward branch (one direction per input → the tangent), a reverse
  branch (the output covector → one cotangent per input), or both.

The docstrings of both generators, their callables and cut accessors are on
the [API reference](api.md) page.

## Generated scalar rules

A scalar numerical primitive is authored once as a graph whose WANT ports are
its value and one partial derivative per input:

```julia
@kernel loggamma_graph(x::Float64) = begin
    y::Float64 = SpecialFunctions.loggamma(x)
    dy_dx::Float64 = SpecialFunctions.digamma(x)
    return y, dy_dx
end
const loggamma = scalar_derivative_rule(
    loggamma_graph; primal = :y, partials = (x = :dy_dx,), name = :loggamma)
```

`loggamma(x)` runs the primal cut (`want = :y`; the partial is pruned), and
its signature is generic, so the same cut traces under Reactant. The activity
pattern of a differentiated call selects the cut
`want = (:y, partials of the active inputs...)` through
[`derivative_cut`](@ref), and the scalar chain rule combines those partials
with the backend's covector (reverse) or directions (forward, including batch
width). `ReactiveKernelsDistributionKernels` uses exactly this for `loggamma`
and `logbeta`.

## Generated vector rules

The matrix–vector product below authors both branches in one graph. The
exact source is read from
[`examples/manual_derivative_rule.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/examples/manual_derivative_rule.jl):

```@eval
Main.ReactiveKernelsDocs.render_manual_derivative_rule_source()
```

[`derivative_rule`](@ref) names the graph's ports into roles and validates
them against its HAVE/WANT boundary. Inputs are the HAVE ports that are
neither directions nor the covector, in HAVE order (`A`, `x`). The returned
`matvec` is an ordinary callable owned by this repository:

- `matvec(A, x)` runs the primal cut;
- [`forward_cut`](@ref)`(matvec, A, x, A_dot, x_dot)` runs the JVP branch
  and returns `(y, y_dot)`; an inactive input takes a zero direction;
- [`reverse_cut`](@ref)`(matvec, Val(mask), A, x, y_bar)` runs the VJP branch
  for one activity mask (bit `i` set when input `i` is active) and returns
  the cotangents of the active inputs only.

## One graph, selected cuts

The panels below prepare the same HAVE→WANT boundaries explicitly, execute
each cut during the documentation build, and show its generated Julia and
selected compute DAG.

```@eval
Main.ReactiveKernelsDocs.render_manual_derivative_rule_cuts()
```

The primal cut has only `A` and `x`; the forward cut adds `A_dot` and `x_dot`;
the reverse cut instead adds `y_bar`. Consequently the forward kernel has no
output-covector path, while the reverse kernel has no input-direction path.

The graph writes both the JVP and the VJP mathematics. It does not derive one
from the other: RK has neither reverse-mode AD nor program transposition, and
this design does not propose reimplementing them. Generating a VJP from a
primal-only graph is a different, much larger feature. An adjoint-pairing
check keeps the two authored branches consistent instead.

## Two-stage reverse staging

A reverse protocol receives the output covector only after the primal has
returned, so every generated reverse adapter stages the one graph: it runs
the primal cut, retains exactly the inputs that the selected reverse cut
reads — [`reverse_residuals`](@ref)`(matvec, Val(mask))` — and runs
`reverse_cut` when the covector arrives. For `matvec`, `A_bar` reads only
`x` and `x_bar` only `A`, so an `x`-only pullback retains `A` alone and an
input the cut does not read may be passed as `nothing`.

This build-executed receipt checks the cuts, the residual sets and the
adjoint pairing:

```@example manual_derivative_rules
result = Main.ManualDerivativeRuleExample.run()

@assert result.y == result.y_forward == result.prepared.y
@assert result.x_bar == result.x_only_bar
@assert result.adjoint_pair[1] ≈ result.adjoint_pair[2]
@assert result.residuals == (
    A_only = (false, true),
    x_only = (true, false),
    both = (true, true),
)
@assert result.recipe_ids == (
    primal = (1,),
    forward = (1, 2, 3, 4),
    reverse = (5, 6),
    x_reverse = (6,),
)

(;
    result.residuals,
    result.recipe_ids,
    result.adjoint_pair,
)
```

Retaining a compact stable intermediate of the primal stage instead of an
original input (cross-stage liveness) is not generated yet: every residual is
an input of the rule.

## Generated adapters

Each adapter is generic over the rule and is loaded with its AD package:

- **Enzyme** (`ext/ReactiveKernelsEnzymeExt.jl`). Reverse mode runs the
  primal cut in the augmented primal, hands Enzyme a zero shadow for an
  array result, retains the residual inputs (copied when Enzyme reports the
  argument may be overwritten before the reverse pass), then runs the cut of
  the call's activity pattern and accumulates the cotangents into the
  argument shadows; `Active`, `Duplicated` and batched arguments are
  supported. Forward mode runs the forward cut with Enzyme's directions,
  once per batch lane. `DifferentiationInterface` with `AutoEnzyme` reaches
  the same rules.
- **ChainRules** (`ext/ReactiveKernelsChainRulesCoreExt.jl`, loaded with
  `ChainRulesCore`). `rrule` returns the primal and a concretely typed
  pullback that holds the all-active cut's residuals; `frule` runs the
  forward cut, with a zero direction for a `ZeroTangent`.

Not generated yet: a Mooncake adapter. Under Reactant a rule's callable and
cuts trace into the compiled program as plain graph mathematics, but no custom
rule is emitted into EnzymeMLIR — that waits on the upstream custom-rule
bridge (see [Automatic differentiation through Reactant](reactant-ad.md)) —
so Enzyme under Reactant differentiates the traced primal cut.

The first vector consumer is the backsolve adjoint of
`ReactiveKernelsReactantODESolvers`: its right-hand side is a
`DerivativeRule`, and the augmented adjoint system evaluates the rule's
reverse cut once per stage, so nothing differentiates the adaptive loop or
the right-hand side.

## A different boundary: `prepare_ad_pullback`

The existing `prepare_ad_pullback` API asks a `DifferentiationInterface`
backend to differentiate a selected primal kernel and evaluates the VJP for a
supplied seed. It does not consume a rule graph or register a custom rule; the
two surfaces are independent.
