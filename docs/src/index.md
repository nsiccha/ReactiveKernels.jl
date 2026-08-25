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

x   = value(:x, Float64)
y   = value(:y, Float64)
a   = value(:a, Float64)
out = value(:out, Float64)

g = Graph()
add!(g, (x, y) => a,  f)
add!(g,  a     => out, h)

k = prepare(g; have = (x, y), want = (out,))
k(1.0, 2.0)          # straight-line: no graph, no planner on this path
```

## See the selected DAG

```julia
p = plan(g; have = (x, y), want = (out,))
visualize(p)                         # SVG in notebooks and rich Julia displays
visualize(p; alternatives = true)   # include unselected candidate recipes
save_visualization("plan.svg", p)   # self-contained, no renderer dependency
save_visualization("plan.dot", p)   # portable Graphviz interchange
```

Values and recipes are separate nodes, so multi-input and multi-output recipes
remain explicit. The built-in SVG renderer is dependency-free; DOT export is
available when a downstream tool needs Graphviz-quality publication layout.

A second mode is **incremental / demand-driven** execution: previously
materialized values join the effective `have` set, changed source values
invalidate only the cached results whose provenance actually depended on them,
and the same planner prepares and runs just the missing computation — all cache
bookkeeping staying outside the generated kernel.

> **Status:** early development — the public API is still being shaped. The
> examples above track the design and will follow the implementation as it lands.

See the [API Reference](/api) for the exported surface.
