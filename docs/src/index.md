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
    details: Preparation keeps exactly the steps needed to turn what you have into what you want, picks between alternative ways to compute a value, and reuses shared subexpressions — then generates an ordinary Julia function.
  - title: A prepared kernel is just straight-line code
    details: While it runs there is no graph to walk, nothing to schedule, and no cache to look up. All of that bookkeeping happens once, during preparation, not on every call.
  - title: Optional buffer-reusing execution
    details: prepare_nonallocating rewrites the generated function to reuse the same arrays across calls, so how much it allocates is separate from what the graph computes.
  - title: Deliberately narrow
    details: No scheduler inside a prepared kernel, no probabilistic-programming semantics, and no symbolic algebra. It turns pure dataflow graphs into fast plain-Julia functions and refuses code whose side effects it cannot check.
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

The definition uses ordinary function syntax (as in ReactiveObjects.jl), and
its argument list is the default set of inputs. Functions like `f` and `h` are
inputs too: they run only when the prepared kernel is called. Type annotations
are optional; leaving them off does not stop the generated function from
specializing on the concrete values you pass. Every named value is reachable as
a port (`model.f`, `model.x`, `model.a`, `model.out`). With no explicit
`return`, the derived values such as `out` are the default outputs; a `return`
can pick different defaults without hiding any port. You may refer to a value
before the line that defines it, a tuple assignment is one recipe with several
outputs, and `@recipe (cost = ..., cse_key = ...)` sets a recipe's cost and
reuse hints without dropping to the lower-level graph API.

Trailing positional defaults and fixed keyword arguments follow ordinary Julia
call syntax. A default is evaluated only when you omit that argument, may refer
to earlier arguments, and stays an input port like the required ones. A keyword
with no default is required. The prepared kernel resolves the call signature
before handing off to the same generated function underneath.

The right-hand side of a recipe is ordinary Julia. Each recipe becomes a single
function that closes over the ports it reads, so control-flow and scoping forms
— `try`/`catch`/`finally`, `let`, comprehensions, and `do` blocks — run as plain
Julia inside it. RK still finds the ports a recipe depends on even when they are
used only inside a `try`, `let`, or comprehension, and it renames a `catch`
variable that happens to match a port name so the two never collide. This works
because RK treats each recipe as a black box: it does not track changes line by
line inside the recipe, it just reruns the whole recipe when one of its inputs
changes. The reactive method bodies on the [compiler](compiler.md) page are the
opposite case — there RK tracks changes field by field, so deferred or
exception-conditional execution would break that tracking, and those forms are
rejected.

This function-shaped form describes a graph, not a new object type. Compiled
reactive state lets you change inputs in place through `set!`, `mutate!`, and
`touch!`. A `@kernel` that contains its own inner methods is handled by a
separate, stricter compiler: it reads the method source directly and only allows
calls whose effects it can account for, so it is not an ordinary stateless
recipe.
The [compiler capability and limits](compiler.md) page covers the full planning,
code-generation, state, control-flow, NUTS, and rejection rules.

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

## Reuse a kernel inside another kernel

```julia
@kernel unit_interval(x::Float64) = begin
    magnitude::Float64 = abs(x)
    tail::Float64 = exp(-magnitude)
    probability::Float64 = x >= 0 ? inv(1 + tail) : tail / (1 + tail)
    log_normalizer::Float64 = log1p(tail)
    log_probability::Float64 = x >= 0 ? -log_normalizer : x - log_normalizer
    log_complement::Float64 = x >= 0 ? -x - log_normalizer : -log_normalizer
    log_jacobian::Float64 = log_probability + log_complement
    return (probability, log_jacobian)
end

@kernel transformed_model(x::Float64) = begin
    (probability::Float64, log_jacobian::Float64) = unit_interval(x)
    score::Float64 = probability + log_jacobian
    return (probability, log_jacobian, score)
end
```

A direct call to a stateless `KernelSpec` inside another stateless `@kernel`
is a graph-construction operation, not a runtime call. RK gives every call site
fresh internal value identities, binds the nested HAVE/WANT boundary to the
call arguments and assignment outputs, and splices the recipes into the outer
graph. Preparing only `:probability` therefore omits the logarithmic path;
preparing only `:log_jacobian` omits the probability recipe; requesting both
shares `magnitude` and `tail`. The generated kernel contains no nested call.

Nested graph calls currently take all default HAVE boundary ports positionally
(supply optional positional values explicitly), must destructure the nested
output boundary exactly, and require exact declared types at the outer input
and output ports. Arguments are existing graph ports; assign a larger expression
to its own recipe first. Put `@recipe` metadata inside the reusable kernel rather
than on the call site. Each restriction is checked while the outer graph is
constructed.

The dedicated [Bijectors and constrained parameters](bijectors.md) page applies
this composition rule to reusable support transforms, shows parameters-only and
Jacobian-only pruning, and renders the resulting fused model DAG.

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

A second mode is **incremental**, demand-driven execution: values you have
already computed count as things you now `have`, changing a source input
invalidates only the cached results that actually depended on it, and the
planner reruns just the missing part — with all of that cache bookkeeping
staying outside the generated kernel.

Mathematical `PreparedKernel`s that Reactant can trace can also run on a GPU or
TPU through the optional Reactant extension, and `replica` runs a whole scalar
kernel across an extra batch axis (for example one column per chain). This is
narrower than “run any reactive state on an accelerator.” The optional external
[adaptive-NUTS exemplar](nuts.md#reactant-adaptive-transition-and-multiple-chains)
compiles one full-depth transition to one data-dependent traced `while`, with
pre-generated momentum, direction, and exponential tensors plus explicit
counters, so there is no host RNG inside the trace. It is scoped to `Float64`, a
positive diagonal Euclidean metric, the locked authored control-flow graph, and
the current diagnostics callback. Overflow and unsupported cases reject; the
native adaptive API remains CPU execution. Its example source and executable
acceptance test are linked from the focused NUTS page.

> **Status:** early development — the public API is still being shaped.

See the [non-allocating workflow](nonallocating.md) for persistent array-cache
lowering, or the [API Reference](api.md) for the exported surface.
