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
> general dataflow abstraction. Scope covers the design's Phases 1–4; the
> Phase 5 hooks (mutation/bufferization AST passes, e-graphs, Enzyme/Reactant
> backends) are intentionally left as documented extension points.

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
```

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

## API surface

| Concept | Functions |
|---|---|
| Build | `Graph`, `value`, `value!`, `add!`, `compose` |
| Plan | `plan`, `explain`, `code_expr`, `inputs`, `outputs` |
| Lower / compile | `lower`, `transform`, `compile`, `prepare` |
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
julia --project=. -e 'using Pkg; Pkg.test()'   # 143 tests
julia --project=. -e 'using Pkg; Pkg.test(test_args=["benchmark"])'
julia --project=. examples/demo.jl             # runnable walkthrough
julia --project=. examples/preexisting.jl      # ReactiveObjects/ReactiveHMC ports
```
