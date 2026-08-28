```@raw html
---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: "ReactiveKernels.jl"
  text: "have → want kernel planning"
  tagline: Lower a dataflow graph and a have/want query into a specialized, straight-line Julia kernel — no traversal, no scheduling, no interpretation on the hot path.
  actions:
    - theme: brand
      text: API Reference
      link: /api
    - theme: alt
      text: View on GitHub
      link: https://github.com/nsiccha/ReactiveKernels.jl

features:
  - title: Plan once, run millions of times
    details: Preparation prunes to exactly the computations needed to turn have into want, selects among alternative producers, and does graph-level CSE — then emits ordinary Julia via RuntimeGeneratedFunctions.
  - title: The graph is compile-time, not a scheduler
    details: The prepared kernel is straight-line code. Reactive invalidation and cache bookkeeping live outside the hot kernel, never inside it.
  - title: Optional cache-filling lowering
    details: prepare_nonallocating applies a final MutatingFunctions-backed AST rewrite, keeping mutation and allocation behavior independent of graph semantics.
  - title: Deliberately narrow
    details: No dynamic scheduling in a prepared hot kernel, no AD, no PPL semantics, no symbolic algebra. It generates excellent plain Julia kernels for pure dataflow graphs and rejects unproved effects in captured state-machine source.
---
```

## Overview

`ReactiveKernels.jl` is a **have → want** computational-graph kernel layer. You
describe a computation once as a transparent graph of named values and the
operations connecting them; you declare what you already **have** and what you
**want**; and the planner prepares a specialized Julia function that produces the
latter from the former.

```julia
using ReactiveKernels

f(x, y) = x + y
h(a) = abs2(a)

@kernel model(f, h, x, y) = begin
    a = f(x, y)
    out = h(a)
end

k = prepare(model)
k(f, h, 1.0, 2.0)    # straight-line: no graph, no planner on this path

@kernel affine(x, factor = 2; offset = factor - 1) = begin
    out = factor * x + offset
end

prepare(affine)(3; offset = 4)  # 10
```

The definition has ordinary function syntax, as in ReactiveObjects.jl, and its
signature is the default input boundary. Functions are ordinary input ports:
`f` and `h` execute only when the prepared kernel is called. Type annotations
are optional metadata; omitting them does not prevent the generated function
from specializing on concrete runtime values. Every named value is exposed as
a port (`model.f`, `model.x`, `model.a`, `model.out`). With no explicit
`return`, derived sinks such as `out` are the default output boundary; a
`return` may select a different default without hiding anything. Declarations,
intermediates, and alternative producers may be forward-referenced; tuple
assignment is one multi-output recipe, and
`@recipe (cost = ..., cse_key = ...)` exposes the existing planner metadata
without dropping to the low-level builder.

Trailing positional defaults and fixed keyword arguments follow ordinary Julia
call syntax. Defaults are evaluated only when omitted, may refer to earlier
arguments, and remain exposed input ports alongside required arguments. A
keyword without a default is required. The prepared call adapter resolves the
signature before entering the same generated positional kernel.

A recipe right-hand side is ordinary Julia. Each recipe compiles into an opaque
operation closed over its free ports, so control-flow and scoping forms —
`try`/`catch`/`finally`, `let`, comprehensions, and `do` blocks — run as plain
Julia inside that operation. Free-port detection flows through them, so a port
used only inside a `try`, `let`, or comprehension is still discovered as a
dependency, and a `catch` variable that happens to share a port's name is
renamed so it never shadows the port. This is safe precisely because the recipe
is opaque: there is no reactive invalidation to track through the branch. It is
the deliberate contrast with an invalidation-tracked reactive method body, where
deferred or exception-conditional execution would defeat field-level
invalidation and these forms are therefore rejected.

This methodless form authors a graph, not a per-kernel object type. Compiled
reactive state supports source mutation through `set!`, `mutate!`, and `touch!`.
Method-bearing `@kernel` definitions enter a separate captured-source compiler
with an implicit receiver and an exact registered-effect boundary; they are not
ordinary stateless recipes. The [compiler capability and limits](compiler.md)
page gives the full planner, lowering, state, control-flow, NUTS, and rejection
contracts.

## Extend by named ports

```julia
@kernel diagnostics(out) = begin
    squared = abs2(out)
end

extended = merge(model, diagnostics)
prepare(extended)(f, h, 1.0, 2.0)  # same inputs/output as `model`
prepare(extended; want = (:out, :squared))(f, h, 1.0, 2.0)
```

`merge` builds a fresh graph, unifies same-name/same-type ports, and preserves
the base boundary by default. That makes an extended kernel a drop-in for the
base while extra requested outputs opt into extra work. The explicit
`boundary = :fragment` option replaces both defaults when that is intentional.

## See the selected DAG

```julia
p = plan(model)
visualize(p)                         # interactive HTML; SVG rich-display fallback
visualize(p; alternatives = true)   # include unselected candidate recipes
save_visualization("plan.html", p)  # interactive HTML with offline SVG fallback
save_visualization("plan.svg", p)   # self-contained, no renderer dependency
save_visualization("plan.dot", p)   # portable Graphviz interchange
```

Values and recipes are separate nodes, so multi-input and multi-output recipes
remain explicit. The preferred HTML surface uses Cytoscape.js with ELK for
layered layout, fit, pan, zoom, focus, and node inspection. The public docs
bundle those libraries locally; standalone HTML retains the dependency-free SVG
fallback. DOT export is available when a downstream tool needs Graphviz-quality
publication layout.

A second mode is **incremental / demand-driven** execution: previously
materialized values join the effective `have` set, changed source values
invalidate only the cached results whose provenance actually depended on them,
and the same planner prepares and runs just the missing computation — all cache
bookkeeping staying outside the generated kernel.

> **Status:** early development — the public API is still being shaped.

See the [non-allocating workflow](nonallocating.md) for persistent array-cache
lowering, or the [API Reference](api.md) for the exported surface.
