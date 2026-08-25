# ReactiveKernels.jl

[![Docs (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://nsiccha.github.io/ReactiveKernels.jl/dev/)

A **have → want** computational-graph kernel layer for Julia.

You describe a computation once as a transparent dataflow graph of named,
typed values and the operations that connect them. You then declare what you
already **have** and what you **want**, and `ReactiveKernels` prepares a
specialized, straight-line Julia function that turns one into the other — with
no graph traversal, no dynamic scheduling, and no interpretation on the hot
path.

The graph is a **planning-time** abstraction, not a runtime scheduler. Once a
kernel is prepared you can call it serially, millions of times, at a speed close
to equivalent hand-written Julia.

```text
   user graph  +  have / want query
                     │
                     ▼
                  planner            dependency pruning
                     │               alternative-producer selection
                     │               CSE / topological scheduling
                     ▼
          straight-line Julia Expr
                     │
                     ▼
        RuntimeGeneratedFunctions.jl
                     │
                     ▼
                PreparedKernel        kernel(inputs...)  ← millions of serial calls
```

## What it does

Given a graph and a `have`/`want` query, the planner:

- **prunes** to only the computations needed to produce `want` from `have`;
- **selects** among alternative ways of producing a value when more than one
  exists;
- performs graph-level **common-subexpression elimination** and topological
  scheduling;
- **emits a specialized Julia function** via
  [`RuntimeGeneratedFunctions.jl`](https://github.com/SciML/RuntimeGeneratedFunctions.jl).

A second, equally important mode is **incremental / demand-driven** execution:
previously materialized values join the effective `have` set, changed source
values invalidate only the cached results whose provenance actually depended on
them, and the same planner prepares and runs just the missing computation. All
invalidation and cache bookkeeping happens **outside** the generated hot kernel.

## What it is not

`ReactiveKernels` is deliberately narrow. It does **not** schedule distributed or
threaded work, execute graph nodes dynamically, do automatic differentiation, or
carry any probabilistic-programming semantics (priors, log densities, transforms).
It is not a symbolic-algebra system and does not depend on a PPL, Symbolics,
Enzyme, or Reactant. A downstream library (for example a PPL) can build
domain-specific graph construction on top and delegate planning and code
generation here.

## Usage

> **Status:** early development — the public API is still being shaped. The
> sketch below follows the design and will track the implementation as it lands.

```julia
using ReactiveKernels

# Values are graph-level identities, not runtime containers.
x   = value(:x, Float64)
y   = value(:y, Float64)
a   = value(:a, Float64)
b   = value(:b, Float64)
c   = value(:c, Float64)
out = value(:out, Float64)

g = Graph()
add!(g, (x, y) => a,   f)
add!(g,  a      => b,  g1)
add!(g,  a      => c,  g2)
add!(g, (b, c)  => out, h)

# Prepare once: select only what's needed to turn (x, y) into out.
k = prepare(g; have = (x, y), want = (out,))

# Call on the hot path — no graph, no planner involved.
result = k(1.0, 2.0)
```

If you instead already **have** `b` and `c`, `prepare(g; have = (b, c), want =
(out,))` emits a kernel morally equivalent to `h(b, c)` — everything upstream is
pruned away.

## Installation

Not yet registered. While in development:

```julia
using Pkg
Pkg.add(url = "https://github.com/nsiccha/ReactiveKernels.jl")
```

## Documentation

- Development docs: <https://nsiccha.github.io/ReactiveKernels.jl/dev/>

## License

MIT — see [`LICENSE`](LICENSE).
