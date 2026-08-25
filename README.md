# ReactiveKernels.jl

A **have/want computational-graph kernel layer**. You build a graph of pure
computations, declare which values you already **have** and which you **want**,
and the library prepares a specialized straight-line Julia kernel containing
*only the cheapest necessary computation*. After preparation the hot path is
ordinary Julia — no graph traversal, no dynamic dispatch, no planner overhead.

A thin **reactive** layer on top adds incremental/replay execution
(provenance-aware invalidation, explicit materialization, frozen/checkpoint cut
points) *without changing planner semantics* — nothing reactive ever runs inside
a prepared kernel.

> This is a proof-of-concept implementation of the design brief in
> [this gist](https://gist.github.com/nsiccha/7f8c6802e1522be05f2d3240dba8aa68).
> It contains no probabilistic-programming or domain-specific semantics — it is a
> general dataflow abstraction. Scope covers the design's Phases 1–4 and the
> first Phase 5 physical-lowering integration: an optional
> MutatingFunctions-backed non-allocating preparation path. E-graphs and
> Enzyme/Reactant backends remain extension points.

The runnable [`examples/eight_schools.jl`](examples/eight_schools.jl) shows how
to build PPL semantics manually from ordinary recipes: unconstrained-to-
constrained transforms, an optional log Jacobian, decomposed prior and
likelihood terms, pointwise log likelihoods, total log density, and new-group
prediction. Different `want` sets prune density or generated-quantity work from
the same graph, and the numeric boundary accepts forward-mode AD dual numbers.

## Pipeline

```
user graph + have/want query
        │  dependency pruning
        │  alternative-producer cost selection
        │  CSE / topological scheduling
        ▼
straight-line Julia Expr  ──►  (optional AST passes)  ──►  RuntimeGeneratedFunctions  ──►  PreparedKernel
                                                                                              │
                                                                            kernel(inputs...)  ── millions of serial calls
```

## Quick start

```julia
using ReactiveKernels

g = Graph()
x = value!(g, :x, Float64)
a = value!(g, :a, Float64)
b = value!(g, :b, Float64)

add!(g, x => a, x -> x + 1.0; cost = 1.0)
add!(g, a => b, a -> 2a;      cost = 1.0)

k = prepare(g; have = (x,), want = (b,))
k(1.2)            # 4.4  — a straight-line kernel: b = 2*(x+1)
```

Inspection is first-class:

```julia
p = plan(g; have = (x,), want = (b,))
println(explain(p))     # have/want, selected recipes + costs, alternatives, total cost
code_expr(p)            # the generated Julia Expr, before RGF compilation
inputs(k), outputs(k)   # graph values in call / return order

visualize(p)            # interactive colored DAG in HTML-capable displays
save_visualization("plan.html", p)
save_visualization("plan.svg", p)
dot_source(p)           # portable Graphviz DOT source
```

The diagram uses explicit value and recipe nodes (`value → recipe → value`),
so multi-input and multi-output operations stay unambiguous. A plan shows only
the selected computation by default; `visualize(p; alternatives=true)` adds
backward-reachable alternatives as muted dashed nodes. Both `Graph` and `Plan`
also expose interactive HTML and SVG rich displays directly. The HTML component
adds fit, pan, zoom, keyboard controls, and a structural node inspector without
external assets; `.dot` / `.gv` exports can be fed to Graphviz for
publication-oriented layout without making Graphviz a package dependency.

See the [visualization guide](docs/src/visualization.md) for the visual
semantics, format tradeoffs, and deliberate non-goals.

## Alternative producers (the interesting bit)

The same value may be obtainable several ways, including multi-output recipes.
The planner selects the globally cheapest *set* of recipes — not a per-value
shortest path — accounting for shared work.

```julia
g = Graph()
u, a, b, c, r = (value!(g, s, Float64) for s in (:u, :a, :b, :c, :r))

add!(g, u => a,       cheap_a;     cost = 1.0)   # cheap route to `a` only
add!(g, u => (a, b),  combined_ab; cost = 1.2)   # produces a AND b together
add!(g, a => b,       make_b;      cost = 1.0)
add!(g, a => c,       make_c;      cost = 1.0)
add!(g, (b, c) => r,  finish;      cost = 1.0)

prepare(g; have = (u,), want = (a,))   # chooses cheap_a  (cost 1.0)
prepare(g; have = (u,), want = (r,))   # chooses combined_ab, since b is needed too (cost 3.2 < 4.0)
prepare(g; have = (a, b), want = (r,)) # starts at the boundary: only make_c + finish
```

Values in `have` are authoritative — the planner never emits code to recompute
them.

## Reactive / incremental use

`ReactiveState` maintains materialized values and reuses them as `have` for the
next demand. Invalidation follows *actual* production provenance, so changing an
input that fed an *unused* alternative path does **not** invalidate a cached
value.

```julia
state = ReactiveState(g; materialize = (a,))
set!(state, x, 1.0)
out1 = get!(state, b)        # computes a, caches it

set!(state, x, 1.5)          # only x changes → a's provenance is stale
out2 = get!(state, b)        # a recomputed on demand; unrelated caches survive
```

**Frozen cut points** support cross-phase replay without any `train`/`test`
concept in the core:

```julia
# Phase 1: derive stats, then checkpoint (freeze) them.
set!(state, X, X_train)
loc, sc = get!(state, (location, scale))
cp = checkpoint(state, (location, scale))

# Phase 2: new data, frozen phase-1 stats — their producers never re-run.
state2 = ReactiveState(g; frozen = cp)
set!(state2, X, X_test)
y = get!(state2, standardized_X)
```

## AST-transform boundary

Lowering produces an ordinary `Expr`, the natural place for optional passes
(simplification, mutation/bufferization, backend rewrites) *before* compilation:

```julia
ast  = lower(p)
ast2 = transform(ast, passes...)   # each pass :: Expr -> Expr
k    = compile(ast2)
# or, ergonomically:
k    = prepare(p; passes = (mypass,))
```

## Preexisting ecosystem examples

[`examples/preexisting.jl`](examples/preexisting.jl) ports the examples that
predate this package and exercises them against the public API:

- ReactiveObjects.jl's chain, diamond, and shared-intermediate gallery graphs;
- ReactiveHMC.jl's Euclidean, Riemannian, and SoftAbs phase points, including
  all three relativistic variants;
- leapfrog, generalized leapfrog, implicit midpoint, and multistep integration.

The scalar gallery kernels assert zero allocations after preparation. The HMC
ports prepare their geometry and dynamics kernels once, then count the original
potential/gradient/metric oracles. For example, four generalized-leapfrog
fixed-point iterations call the combined metric-gradient oracle exactly five
times—once for the starting position and once per changed position—and never
call the three dominated partial oracles.

`test/test_handwritten_benchmarks.jl` uses BenchmarkTools to compare the
prepared chain and shared-intermediate kernels with equivalent hand-written
Julia functions. The test prints median timings and their ratio, while its
portable acceptance checks semantic equality, inferred concrete return types,
and zero hot-path allocations rather than asserting a machine-dependent timing
threshold.

The source revisions are pinned in the example files so the compatibility
corpus is auditable.

For documentation, `examples/artifacts.jl` exposes all 13 compatibility cases
as executable `ExampleArtifact` records. Each record carries the pinned raw
source/call and runtime inputs, the real `PreparedKernel` and its executed
output, the exact `code_expr` generated from that kernel, and its selected
`Plan`. The visualization layer consumes the plan directly, so docs can render
the colored compute DAG without reconstructing graph semantics:

```julia
include("examples/artifacts.jl")
using .CompatibilityArtifacts

artifact = only(filter(x -> x.name == :reactiveobjects_chain, all_artifacts()))
artifact.source       # original input/source
artifact.generated    # actual lowered kernel Expr
visualize(artifact.dag) # structured HAVE/WANT/recipe DAG
```

## Non-allocating preparation

`prepare_nonallocating` keeps the same graph and plan, but applies one final AST
pass that routes each selected operation through
`MutatingFunctions.apply!!`. Every single-output recipe gets a typed persistent
cache. MutatingFunctions is currently unregistered, so install the reviewed
revision explicitly and load it to activate ReactiveKernels' optional extension:

```julia
using Pkg
Pkg.add(url = "https://github.com/nsiccha/MutatingFunctions.jl",
        rev = "b353559ef3e391ae2e2d98256b6967903fdfa410")

using ReactiveKernels, MutatingFunctions
```

The first kernel call seeds its caches; later calls offer them back to the
registered mutating implementation:

```julia
g = Graph()
x = value!(g, :x, Vector{Float64})
a = value!(g, :a, Vector{Float64})
b = value!(g, :b, Vector{Float64})

add!(g, x => a, copy)
add!(g, a => b, reverse)

k = prepare_nonallocating(g; have = (x,), want = (b,))
k([1.0, 2.0, 3.0])  # warm-up: seeds the `copy` and `reverse` caches
y = k([4.0, 5.0])    # reuses both caches; y == [5.0, 4.0]
```

This path deliberately inherits the `apply!!` contract:

- steady-state allocation freedom requires every selected operation to have a
  genuinely mutating `apply!!` method for the runtime types; its generic
  fallback is correct but may allocate;
- selected recipes must each have one output, matching `apply!!`'s one-cache /
  one-result interface;
- each cache slot retains whatever its operation first returns: registered
  allocating operations normally seed fresh kernel-retained storage, but an
  aliasing operation may retain caller-owned input, and a no-recipe plan returns
  its `have` value directly; treat mutable results as borrowed values that may
  alias inputs or be overwritten by the next call;
- a prepared instance is stateful, non-reentrant, and not safe for concurrent
  calls — prepare one per independent caller;
- custom `passes` run on the ordinary lowered AST before the cache rewrite.

See the [non-allocating workflow](docs/src/nonallocating.md) for inspection and
allocation-testing examples.

The optional surface has a reproducible exact-revision integration gate:

```sh
julia --startup-file=no test/run_nonallocating_integration.jl
```

The default `Pkg.test()` target intentionally does not install or load
MutatingFunctions; CI runs both paths independently.

## API surface

| Concept | Functions |
|---|---|
| Build | `Graph`, `value`, `value!`, `add!`, `compose` |
| Plan | `plan`, `explain`, `code_expr`, `inputs`, `outputs` |
| Visualize | `visualize`, `dot_source`, `save_visualization` |
| Lower / compile | `lower`, `transform`, `compile`, `prepare`, `prepare_nonallocating` |
| Cache | `PreparationCache`, `prepare!` |
| Reactive | `ReactiveState`, `set!`, `get!`, `freeze!`, `unfreeze!`, `materialize!`, `checkpoint` |

## Performance contract

After warm-up, a prepared kernel over non-allocating scalar operations adds
**zero** orchestration allocations — the generated code calls operation
implementations directly through a positional tuple (no world-age globals, no
graph objects on the hot path). See `test/test_stateless.jl` and
`examples/demo.jl`.

## Running

```julia
julia --project=. -e 'using Pkg; Pkg.test()'   # full package suite
julia --project=. -e 'using Pkg; Pkg.test(test_args=["benchmark"])'
julia --project=. examples/demo.jl             # runnable walkthrough
julia --project=. examples/eight_schools.jl    # manual PPL graph
julia --project=. examples/preexisting.jl      # ReactiveObjects/ReactiveHMC ports
```
