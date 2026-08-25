# DAG visualization

ReactiveKernels separates the computation graph from its runtime kernel, so
visualization is also a planning-time operation. It never participates in a
prepared kernel's hot path.

## Quick use

```julia
@kernel g(x, y) = begin
    out = hypot(x, y)
end
p = plan(g)

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

The HTML view is rendered by
[Cytoscape.js](https://js.cytoscape.org/) with the
[ELK layered layout](https://github.com/kieler/elkjs): use **Fit**, **+**, and
**−**, drag the canvas to pan, scroll to zoom at the pointer, or use `+`, `-`,
and `0` from the focused canvas. Selecting a value or recipe focuses its local
neighborhood and opens its full state, detail, and incoming/outgoing
connections in the side inspector. The component keeps its navigation and
inspection state when a docs page moves the original DOM node between layouts.

The published VitePress site bundles all three pinned npm packages locally.
Notebook, IDE, and standalone HTML displays load the same pinned browser
packages on demand; if scripts or networking are unavailable, the embedded SVG
remains visible instead of leaving an empty figure.

Documentation code can embed the public object directly—there is no second
graph model or rendering implementation to copy:

```julia
view = visualize(artifact.dag) # artifact.dag::Plan
html = sprint(show, MIME"text/html"(), view)
```

`showable(MIME"text/html"(), view)` is true. If a frontend does not support
HTML, `image/svg+xml` remains the static rich-display fallback.

## HTMXObjects fragments

`DAGVisualization` also follows the ordinary HTML-showable child contract used
by HTMXObjects. Inside an `@htmx` route, compose the visualization as a
structural child:

```julia
using HTMXObjects

@get dag() = h.div(visualize(plan))
```

The returned fragment keeps the same library-backed fit, pan, zoom, and inspect
behavior, with SVG fallback. A host can bundle Cytoscape.js, ELK, and
cytoscape-elk just as these docs do; otherwise the fragment's pinned loader
fetches them once. Do not stringify the view or wrap it in `HTMX.Raw`: `h.*`
preserves HTML-showable children structurally, while `Raw` is reserved for
complete, trusted JavaScript or CSS payloads. Use an ordinary layout container
rather than a presentational `SemanticCard`, because the DAG contains
interactive buttons.

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

Canvas labels retain the complete value or operation name and wrap when needed.
The node picker and inspector retain the full type, identity, cost, state, and
incoming/outgoing names. Structural-CSE aliases share one value node and list
all alias names.

## Why HTML plus SVG plus DOT?

The three formats serve different portability layers:

| Need | Surface | Tradeoff |
|---|---|---|
| Interactive docs inspection | Cytoscape.js + ELK HTML | Libraries are pinned and bundled into the VitePress assets; fit, pan, zoom, focus, and structural inspection |
| Interactive notebook/IDE inspection | Cytoscape.js + ELK HTML | Pinned packages load on demand; embedded SVG remains usable offline |
| Files that render anywhere | `save_visualization("graph.svg", x)` | Static rather than interactive |
| Standalone artifact | `save_visualization("graph.html", x)` | Opens directly with an embedded SVG; loads the pinned interactive libraries when online |
| Large or publication-oriented layout | `dot_source` or `.dot` / `.gv` export | A downstream Graphviz renderer is needed, but ReactiveKernels does not impose its binary/artifact footprint on every user |

Cytoscape.js owns the interactive graph renderer and event model; ELK owns the
directed layered layout. ReactiveKernels' JavaScript is an adapter for semantic
styling and inspection, not a home-grown layout engine. Mermaid is a good
static diagram format but not as strong for structural inspection; React Flow
would introduce React into the Vue-based docs host; Sigma targets much larger
free-form networks. Graph editing and runtime profiling remain out of scope.

Successful `Plan`s are acyclic. A raw `Graph` may still contain a cycle; the
built-in SVG keeps the involved nodes visible at a stable final rank so the
backward edges expose the problem, while DOT remains available for a more
sophisticated diagnostic layout.
