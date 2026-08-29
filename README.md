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
> first Phase 5 physical-lowering integrations: an optional
> MutatingFunctions-backed non-allocating preparation path and an optional
> Reactant extension for traceable mathematical kernels and whole-kernel
> replicas. The optional external
> [adaptive-NUTS exemplar](examples/nuts_runtime/kernel_nuts_reactant.jl) compiles
> one full-depth transition to one data-dependent traced `while`, with
> pre-generated momentum, direction, and exponential tensors plus explicit
> counters, so there is no host RNG inside the trace. That acceptance path is
> scoped to `Float64`, a positive diagonal Euclidean metric, the locked authored
> control-flow graph, and the current diagnostics callback. Overflow and
> unsupported cases reject; the native adaptive API remains CPU execution. See
> the [executable acceptance test](test/test_kernel_nuts_reactant.jl). This is
> not blanket accelerator support for mutable state machines. E-graph
> optimization remains future work.

`DifferentiationInterface` is the package's backend-neutral scalar-gradient
boundary. Concrete AD engines remain optional: in particular, Enzyme is not a
ReactiveKernels dependency, just as Reactant is loaded only when requested.
`prepare_ad` resolves one named active HAVE port once, passes every other
current HAVE value as a DI `Constant`, and reuses the backend preparation.

The runnable [`examples/eight_schools.jl`](examples/eight_schools.jl) shows how
to build PPL semantics manually from ordinary recipes: unconstrained-to-
constrained transforms, an optional log Jacobian, decomposed prior and
likelihood terms, pointwise log likelihoods, total log density, and new-group
prediction. Different `want` sets prune density or generated-quantity work from
the same graph, and the numeric boundary differentiates cleanly through
reverse-mode AD (`DifferentiationInterface` with the Enzyme backend).

## Pipeline

```
declarative kernel + have/want query
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

step(x) = x + 1.0
scale(a) = 2a

@kernel chain(step, scale, x) = begin
    @recipe (cost = 1.0) a = step(x)
    @recipe (cost = 1.0) b = scale(a)
end

k = prepare(chain)
k(step, scale, 1.2)  # 4.4 — a straight-line kernel: b = 2*(x+1)

@kernel affine(x::Float64, factor::Float64 = 2.0;
               offset::Float64 = factor - 1) = begin
    y::Float64 = factor * x + offset
end

affine_kernel = prepare(affine)
affine_kernel(3.0)                  # 7.0
affine_kernel(3.0; offset = 4.0)    # 10.0
```

With an AD backend available, the same authored defaults and keywords carry
through the reusable gradient boundary:

```julia
using DifferentiationInterface
import Enzyme

backend = AutoEnzyme(; mode = Enzyme.Reverse)
affine_ad = prepare_ad(
    affine, backend, 3.0; active = :x, want = :y,
)
ad_gradient(affine_ad, 3.0; offset = 4.0)  # 2.0

# An array-valued active port can also return the value while filling
# caller-owned gradient storage in place.
@kernel quadratic(q::Vector{Float64}; data::Vector{Float64}) = begin
    objective::Float64 = sum(abs2, q .- data)
end
q = [1.0, 2.0]
data = [0.5, -1.0]
quadratic_ad = prepare_ad(
    quadratic, backend, q; data, active = :q, want = :objective,
)
gradient_buffer = similar(q)
value, returned_gradient = ad_value_and_gradient!(
    quadratic_ad, gradient_buffer, q; data,
)
returned_gradient === gradient_buffer  # true; gradient_buffer == [1.0, 6.0]
```

The backend is an `AbstractADType`; core code does not import Enzyme. Plain
reverse mode is sufficient—no runtime-activity mode or function annotation is
part of the RK boundary. `ad_gradient` returns a gradient; the prepared-only
`ad_value_and_gradient!` returns `(value, gradient)` and mutates the supplied
gradient destination. Both rebuild DI `Constant` contexts from the current
inactive arguments on every call.

Like ReactiveObjects.jl's `@reactive` definitions, the primary `@kernel` form
is a normal-looking function definition. Its arguments are the kernel's input
ports, including function-valued inputs such as `step` and `scale`. Type
annotations are optional metadata: generated kernels still specialize on the
concrete values passed at runtime. Every signature argument and assignment is
exposed (`chain.x`, `chain.a`, `chain.b`); without an explicit `return`, the
derived sink `b` is the default output. A `return` can choose another default
output boundary without hiding any named port. Anonymous `@kernel begin ... end`
specifications remain useful for fragments and low-level tooling. Trailing
positional defaults and fixed keyword arguments follow Julia call syntax;
defaults run when omitted and may refer to earlier arguments. They remain
ordinary exposed input ports (`affine.factor`, `affine.offset`).

`@kernel` declares a graph rather than a new object type. Reactive state inputs
can be changed with `set!`, `mutate!`, and `touch!`, but inline method definitions
such as `Base.show(io, __self__) = ...` and magic `__self__` rewriting are not
currently supported. Define external methods on an ordinary wrapper type when
that object-level interface is needed.

### Low-level `Graph` escape hatch (equivalent)

`@kernel` is only hygienic construction of the low-level objects. For tooling
that already owns `Value`s, the imperative API remains available and produces
the same plan and generated code:

```julia
authored = @kernel begin
    x::Float64
    a::Float64 = x + 1.0
    b::Float64 = 2a
    return b
end

low = Graph()
low_x = value!(low, :x, Float64)
low_a = value!(low, :a, Float64)
low_b = value!(low, :b, Float64)
add!(low, low_x => low_a, x -> x + 1.0)
add!(low, low_a => low_b, a -> 2a)

@assert code_expr(plan(authored)) ==
        code_expr(plan(low; have = low_x, want = low_b))
```

Inspection is first-class:

```julia
p = plan(chain)
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
also expose interactive HTML and SVG rich displays directly. Cytoscape.js with
ELK supplies the HTML view's layered layout, routing, fit, pan, and zoom; a
structural side inspector keeps complete node details available without packing
them into the canvas. The docs bundle those libraries locally, while notebook
and standalone displays retain an offline SVG fallback. `.dot` / `.gv` exports
can be fed to Graphviz for publication-oriented layout without making Graphviz
a Julia package dependency.

See the [visualization guide](docs/src/visualization.md) for the visual
semantics, format tradeoffs, and deliberate non-goals.

## Alternative producers (the interesting bit)

The same value may be obtainable several ways, including multi-output recipes.
The planner selects the globally cheapest *set* of recipes — not a per-value
shortest path — accounting for shared work.

```julia
@kernel g(u) = begin
    @recipe (cost = 1.0) a = cheap_a(u)
    @recipe (cost = 1.2) (a, b) = combined_ab(u)
    @recipe (cost = 1.0) b = make_b(a)
    @recipe (cost = 1.0) c = make_c(a)
    @recipe (cost = 1.0) r = finish(b, c)
end

prepare(g; want = :a)                    # chooses cheap_a  (cost 1.0)
prepare(g)                               # chooses combined_ab (cost 3.2 < 4.0)
prepare(g; have = (:a, :b), want = :r)  # only make_c + finish
```

Values in `have` are authoritative — the planner never emits code to recompute
them.

## Composition and extension

Fragments compose through named ports. `merge` is pure: it builds a fresh spec,
unifies same-name/same-type ports, copies both recipe sets, and rejects a type
mismatch before constructing a partial result.

```julia
@kernel base(position) = begin
    energy = abs2(position) / 2
end

@kernel diagnostics(energy) = begin
    energy_squared = abs2(energy)
end

extended = merge(base, diagnostics)
prepare(extended)(2.0)                         # 2.0, same boundary as `base`
prepare(extended; want = (:energy, :energy_squared))(2.0) # (2.0, 4.0)
```

The base boundary is preserved by default, so adding diagnostics does not alter
the ordinary call or return shape and unused observables generate no work. Use
`merge(base, fragment; boundary = :fragment)` only when replacing both default
inputs and outputs is intentional. This is the extension path for augmenting a
NUTS graph with downstream-specific observables while retaining a drop-in
regular-NUTS kernel.

## Reactive / incremental use

`ReactiveState` maintains materialized values and reuses them as `have` for the
next demand. Invalidation follows *actual* production provenance, so changing an
input that fed an *unused* alternative path does **not** invalidate a cached
value.

```julia
state = ReactiveState(g; materialize = :a)
set!(state, g.x, 1.0)
out1 = get!(state, g.b)        # computes a, caches it

set!(state, g.x, 1.5)          # only x changes → a's provenance is stale
out2 = get!(state, g.b)        # a recomputed on demand; unrelated caches survive
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
as executable `ExampleArtifact` records. Each record carries the corresponding
compact `@kernel` source and runtime inputs, the real `PreparedKernel` and its
executed output, the exact `code_expr` generated from that kernel, and its
selected `Plan`. The visualization layer consumes the plan directly, so docs
can render the colored compute DAG without reconstructing graph semantics. The
docs' Generated kernel pane combines that raw AST with the selected plan to show
named operations and retained authored expressions; it never presents the
display-only copy as the compiled `code_expr`:

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
@kernel g(x::Vector{Float64}) = begin
    a::Vector{Float64} = copy(x)
    b::Vector{Float64} = reverse(a)
end

k = prepare_nonallocating(g)
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
| Author / compose | `@kernel`, `KernelSpec`, `merge`, `compose`, `port`, `kernel_graph` |
| Low-level build | `Graph`, `value`, `value!`, `add!` |
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
