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
    details: No dynamic scheduling, no AD, no PPL semantics, no symbolic algebra. It generates excellent plain Julia kernels for any pure dataflow graph.
---
```

## Overview

`ReactiveKernels.jl` is a **have → want** computational-graph kernel layer. You
describe a computation once as a transparent graph of named, typed values and the
operations connecting them; you declare what you already **have** and what you
**want**; and the planner prepares a specialized Julia function that produces the
latter from the former.

```julia
using ReactiveKernels

g = @kernel begin
    x::Float64
    y::Float64
    a::Float64 = f(x, y)
    out::Float64 = h(a)
    return out
end

k = prepare(g)
k(1.0, 2.0)          # straight-line: no graph, no planner on this path
```

The block is declarative and zero-execution: recipe right-hand sides become
stored operations and do not run until a prepared kernel is called. Every port
needs one type annotation somewhere in the block. Declarations, intermediates,
and alternative producers may be forward-referenced; tuple assignment is one
multi-output recipe, and `@recipe (cost = ..., cse_key = ...)` exposes the
existing planner metadata without dropping to the low-level builder.

## Extend by named ports

```julia
diagnostics = @kernel begin
    out::Float64
    squared::Float64 = abs2(out)
    return squared
end

extended = merge(g, diagnostics)
prepare(extended)(1.0, 2.0)                  # same inputs/output as `g`
prepare(extended; want = (:out, :squared))(1.0, 2.0)
```

`merge` builds a fresh graph, unifies same-name/same-type ports, and preserves
the base boundary by default. That makes an extended kernel a drop-in for the
base while extra requested outputs opt into extra work. The explicit
`boundary = :fragment` option replaces both defaults when that is intentional.

## See the selected DAG

```julia
p = plan(g)
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
