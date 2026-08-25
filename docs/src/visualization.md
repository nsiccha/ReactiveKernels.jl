# DAG visualization

ReactiveKernels separates the computation graph from its runtime kernel, so
visualization is also a planning-time operation. It never participates in a
prepared kernel's hot path.

## Quick use

```julia
p = plan(g; have = (x, y), want = (out,))

p                                      # interactive HTML in a rich display
visualize(p)                           # the configurable display object
visualize(p; alternatives = true)     # add unselected candidates
visualize(p; orientation = :vertical) # top-to-bottom instead of left-to-right

save_visualization("plan.html", p)
save_visualization("plan.svg", p)
save_visualization("plan.dot", p)
dot_source(p)
```

`Graph` supports the same calls and shows every registered recipe. A `Plan`
shows only selected recipes by default, because that is usually the answer to
“what will run?”. With `alternatives=true`, backward-reachable candidates that
lost the cost selection remain visible as muted dashed nodes.

The HTML view is a complete, dependency-free component: use **Fit**, **Zoom
in**, and **Zoom out**, drag the canvas to pan, scroll to zoom at the pointer,
or use `+`, `-`, and `0` from the focused canvas. Selecting a value or recipe
opens its full state, detail, and incoming/outgoing connections in the side
inspector. The component keeps its navigation and inspection state when a docs
page moves the original DOM node between layouts.

Documentation code can embed the public object directly—there is no second
graph model or rendering implementation to copy:

```julia
view = visualize(artifact.dag) # artifact.dag::Plan
html = sprint(show, MIME"text/html"(), view)
```

`showable(MIME"text/html"(), view)` is true. If a frontend does not support
HTML, `image/svg+xml` remains the static rich-display fallback.

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

## Why HTML plus SVG plus DOT?

The three formats serve different portability layers:

| Need | Surface | Tradeoff |
|---|---|---|
| Interactive notebook/docs inspection | Built-in self-contained HTML | Fit, pan, zoom, keyboard navigation, and structural inspection; no CDN or runtime package |
| Files that render anywhere | `save_visualization("graph.svg", x)` | Static rather than interactive |
| Standalone interactive artifact | `save_visualization("graph.html", x)` | Opens directly in a browser and contains its own SVG, CSS, and JavaScript |
| Large or publication-oriented layout | `dot_source` or `.dot` / `.gv` export | A downstream Graphviz renderer is needed, but ReactiveKernels does not impose its binary/artifact footprint on every user |

Mermaid was not chosen as the core renderer because rendering it requires a
versioned Mermaid integration in the host page, and syntax/version behavior
becomes a frontend concern. Makie/D3 would add a substantial dependency or
application surface. The small built-in JavaScript progressively enhances the
same accessible SVG that static frontends render; it does not fetch code or
introduce another layout. Graph editing and runtime profiling remain out of
scope.

Successful `Plan`s are acyclic. A raw `Graph` may still contain a cycle; the
built-in SVG keeps the involved nodes visible at a stable final rank so the
backward edges expose the problem, while DOT remains available for a more
sophisticated diagnostic layout.
