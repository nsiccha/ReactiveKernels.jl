# DAG visualization

ReactiveKernels separates the computation graph from its runtime kernel, so
visualization is also a planning-time operation. It never participates in a
prepared kernel's hot path.

## Quick use

```julia
p = plan(g; have = (x, y), want = (out,))

p                                      # SVG in a rich Julia display
visualize(p)                           # the configurable display object
visualize(p; alternatives = true)     # add unselected candidates
visualize(p; orientation = :vertical) # top-to-bottom instead of left-to-right

save_visualization("plan.svg", p)
save_visualization("plan.dot", p)
dot_source(p)
```

`Graph` supports the same calls and shows every registered recipe. A `Plan`
shows only selected recipes by default, because that is usually the answer to
“what will run?”. With `alternatives=true`, backward-reachable candidates that
lost the cost selection remain visible as muted dashed nodes.

## Reading a diagram

The diagram is bipartite: values are ellipses, recipes are boxes, and every
edge follows `value → recipe → value`. This keeps multi-input and multi-output
recipes explicit rather than approximating them with ambiguous direct edges.

| Appearance | Meaning |
|---|---|
| Green value | `HAVE` boundary; supplied by the caller |
| Orange value | `WANT` boundary; returned by the kernel |
| Grey recipe | Registered recipe on a full `Graph` |
| Blue recipe | Selected computation |
| Dashed grey recipe | Reachable alternative not selected |
| Dashed red recipe | Effectful recipe, visible on a full `Graph` but never selectable |

Labels always retain the full value name, type, identity, operation name, cost,
and recipe identity. Long text wraps across lines rather than being clipped.
Structural-CSE aliases share one value node and list all alias names.

## Why SVG plus DOT?

The two formats serve different portability layers:

| Need | Surface | Tradeoff |
|---|---|---|
| Immediate notebook/IDE inspection | Built-in self-contained SVG | Zero setup and no package dependency; intentionally a compact layered layout for modest graphs |
| Files that render anywhere | `save_visualization("graph.svg", x)` | Static rather than interactive |
| Large or publication-oriented layout | `dot_source` or `.dot` / `.gv` export | A downstream Graphviz renderer is needed, but ReactiveKernels does not impose its binary/artifact footprint on every user |

Mermaid was not chosen as the core renderer because rendering it requires a
JavaScript integration in the host page, and syntax/version behavior becomes a
frontend concern. Makie/D3-style interactive views can be useful for filtering,
pan/zoom, or runtime overlays, but their dependency and application surface are
better added later as optional consumers of the stable DOT export. The initial
scope is deliberately inspect/export/save; it does not add graph editing,
runtime profiling, or an interactive frontend.

Successful `Plan`s are acyclic. A raw `Graph` may still contain a cycle; the
built-in SVG keeps the involved nodes visible at a stable final rank so the
backward edges expose the problem, while DOT remains available for a more
sophisticated diagnostic layout.
