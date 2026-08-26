# NUTS sampling

> [!WARNING]
> **This page documents the current `@reactive` substrate, which is being replaced —
> it is not the final authoring API.** The compiled group shown below (a flat
> `reactive_nuts_group` with three hand-unrolled endpoints authored via `@reactive`) is
> the **legacy implementation / parity oracle**. The approved direction is a unified,
> **method-bearing `@kernel`** surface — a phase-point *endpoint* object with in-place
> `leapfrog!` / `refresh_momentum!` methods, a composed `sampler`, and ordinary tree
> recursion (ReactiveHMC-faithful, `@reactive` **removed**). That surface is **staged,
> not yet canonical**: it is mid-implementation (syntax → poc → HMC). Its **reviewed,
> source-faithful authoring fixture is shown below as a source-only, non-executable
> target** (compiler lowering in progress); the legacy `@reactive` substrate that
> follows it is being replaced.

`ReactiveKernels` includes a multinomial No-U-Turn sampler. Its per-transition
Hamiltonian work is *currently* a compiled reactive kernel — the **legacy substrate**
documented below. The tree-growth recursion, RNG draws, U-turn criteria, leapfrog
integration, and the adaptation/statistics update methods all run *over* the compiled
handles as ordinary inferred Julia; they are not themselves reactive graphs.

The public model boundary is a **scalar potential** plus its gradient callable
`potential_gradient!(gradient, position)`, which fills the gradient buffer in place and
returns the potential. **ReactiveKernels computes no pullbacks itself** — it accepts
*any* consumer gradient: a hand-written analytic one, or an optional automatic-
differentiation integration (e.g.
[DifferentiationInterface](https://github.com/JuliaDiff/DifferentiationInterface.jl)
with reverse-mode [Enzyme](https://github.com/EnzymeAD/Enzyme.jl)). The examples on
this page use the standard Gaussian `U(x) = ‖x‖²/2`, whose gradient is simply `x`, with
an **analytic** callable — so the *kernel*, not any differentiation machinery, is the
visible content, and the timings isolate sampler overhead rather than the gradient. The
complete runnable workflow, including the optional DI + Enzyme boundary, is
[`examples/nuts.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/examples/nuts.jl).

## The reviewed `@kernel` authoring surface

> [!IMPORTANT]
> **Compiler lowering in progress / not executable sampler.** The definitions below are
> the **reviewed, ReactiveHMC.jl-faithful** `@kernel` NUTS authoring surface, sourced
> **drift-proof** from `benchmark/nuts_kernel_authoring_fixture.jl` at reviewed commit
> `725ac9b` and read from the file at build time. This is a **source-only** target:
> it is **not executed**, there is **no generated-kernel or Compute-DAG pane**, and
> **no parity, allocation, performance, or production claim** is made (the stateful
> `@kernel` lowering is mid-implementation — the definitions construct but do not
> compile to a runnable sampler). It is the authoring *shape* to review, not code you
> can run.

Re-authored directly against the ReactiveHMC.jl reference source
([`ca9ea4ca`](https://github.com/nsiccha/ReactiveHMC.jl), `phasepoints.jl` /
`integrators.jl` / `nuts.jl` / `adaptation.jl`), with exactly three named deviations —
`@reactive` → the unified `@kernel`; implicit-field capture / `__self__` → an explicit
`self` first method parameter; and the `@node` caching hint dropped so caching/hoisting
is *inferred* by `@kernel`, not annotated. Everything else (field names, object
composition, the `pot_f`/`grad_f` phase-point shape, the direct in-object state mutation,
the dual-averaging / Welford recurrences, `restore!`/`rcopy!`) is the reference's. One
`@kernel` macro; no `Graph`/`add!`/applier/binding plumbing; `@reactive` removed.

```@eval
Main.ReactiveKernelsDocs.render_authoring_fixture()
```

## Current substrate (legacy `@reactive` group — being replaced)

The sampler is currently implemented as a flat `reactive_nuts_group`: a single
`@reactive` object with the three phase-point endpoints (`init`/`fwd`/`bwd`)
**hand-unrolled ×3**, per-endpoint `_grad_bundle`/`_kin_bundle` recipes and their
projections, a reactive active-endpoint selection (`active_ham`), energy error
(`dham`), and `diverged`; leapfrog and tree recursion run as ordinary Julia over those
handles. Step-size adaptation, Welford metric adaptation, and the trajectory/sampling
statistics are four further `@reactive` programs.

This is the **parity oracle / current substrate**, not the authoring surface to
review — it is exactly what the staged unified `@kernel` surface (see the banner above,
and the [Fused NUTS authoring](nuts-architecture.md) page) replaces. The raw
`@reactive` authoring panels have been **removed from this primary path deliberately**:
the real, method-bearing `@kernel` definitions + interaction will appear here
build-executed and drift-proof — sourced from HMC's reviewed authoring fixture — once
it lands. Nothing on this page is the desired final architecture.

## The legacy substrate's state and interaction

*(This section introspects the current `@reactive` substrate described above — kept for
reference until the unified `@kernel` surface lands; it is not the target API.)*

The compiled group *is* the reactive **state**. Its fields are the sources and derived
nodes of the `@reactive` definition: `init_pos`/`init_mom` and the control/diagnostic
fields are HAVE **sources** you write; the Hamiltonian quantities (`init_ham`, `dham`,
`diverged`, the per-endpoint bundles/projections) are **derived** reactive nodes you
read:

```@example nutsdrive
using LinearAlgebra, Random, ReactiveKernels

# A consumer gradient callable — here the analytic gradient of U(x) = ‖x‖²/2 (∇U = x).
# (RK computes no pullbacks; an optional DI + Enzyme integration would supply the same
# callable — see the intro.)
potential_gradient!(gradient, position) =
    (copyto!(gradient, position); sum(abs2, position) / 2)

# The group is the reactive state — every field is a reactive source or derived node:
group = reactive_nuts_group(potential_gradient!,
    Matrix{Float64}(I, 4, 4), zeros(4), ones(4))
propertynames(group)
```

You **read** a derived getter directly; each recomputes only its invalidated inputs
(a source-slot read is allocation-free):

```@example nutsdrive
(group.init_ham, group.dham, group.diverged)   # derived reactive nodes, read on demand
```

The sampler holds only orchestration scratch — **all** phase-point state lives on the
reactive group, so there is no shadow copy to keep in sync:

```@example nutsdrive
sampler = nuts_state(group;
    rng = Xoshiro(1), step_f = partial(leapfrog!; stepsize = 0.3), max_depth = 5)
fieldnames(typeof(sampler))
```

You **drive** it by running a transition; afterwards the diagnostics are read straight
off the reactive group (the transition wrote its sources, the getters recompute):

```@example nutsdrive
chain = sample!(sampler, 1)                     # one NUTS transition
(chain.diagnostics[1].depth, chain.diagnostics[1].diverged, group.dham)
```

## The sampled path

The same compiled `reactive_nuts_group` program drives warmup and sampling through the
consumer gradient callable (analytic here; an optional DI + Enzyme integration would
supply the same callable):

```@example nuts
using LinearAlgebra
using Random
using ReactiveKernels

# Analytic gradient of U(x) = ‖x‖²/2 (∇U = x); RK computes no pullbacks itself.
potential_gradient!(gradient, position) =
    (copyto!(gradient, position); sum(abs2, position) / 2)
dimension = 4

group = reactive_nuts_group(potential_gradient!,
    Matrix{Float64}(I, dimension, dimension), zeros(dimension), zeros(dimension))
sampler = nuts_state(group;
    rng = Xoshiro(20260825),
    step_f = partial(leapfrog!; stepsize = 0.35),
    max_depth = 7)
warmup = warmup!(sampler, 300)
chain = sample!(sampler, 1_000)

count(diagnostic -> diagnostic.diverged, chain.diagnostics)
```

`warmup!` performs initial step-size search, dual averaging, and windowed diagonal
metric adaptation, reusing the same compiled adaptation programs across every metric
window (no per-window rebuild). `sample!` returns samples and per-transition
diagnostics — acceptance, tree depth, leapfrog count, energy error, and divergence
status. The transition itself reads and writes the group's reactive handles: reading
a Hamiltonian getter recomputes only its invalidated inputs, and the source-slot
control writes are inferred and allocation-free.

## Benchmarks

For a reproducible comparison under identical four-chain settings, run
`julia --startup-file=no benchmark/nuts_comparison.jl`. It creates a temporary
environment and pins AdvancedHMC and DynamicHMC outside the package's dependencies,
differentiating the **same** scalar log density through DI + Enzyme for every
sampler. Read every figure as a separate measurement — it is **not** evidence of
blanket sampler superiority:

- `setup_seconds` (one-time construction: the target's DI preparation and the
  reactive graph preparation) is reported **separately** from `sampling_seconds`,
  which is the **matched timed warmup + draw phase** each sampler runs (warmup plus
  retained draws, not post-warmup draws alone).
- On this workload ReactiveKernels' matched warmup + draw time is roughly **4–7×**
  the AdvancedHMC/DynamicHMC wall time, while allocating far less
  (order **10–15 MiB** versus **57–427 MiB** across the panels). The residual wall
  gap is the reactive selection/diagnostic machinery, not setup or the gradient.
- Effective sample size **per gradient** is the *same order of magnitude* but
  **mixed by model** — ReactiveKernels is between AdvancedHMC and DynamicHMC on the
  Gaussian, lowest on the noncentered and one centered configuration, and in the
  middle on the other — so there is no blanket efficiency claim either.

A decomposed in-repo microbenchmark (`benchmark/nuts_microbench.jl`) and a pinned
public ReactiveHMC ca9 three-way (`benchmark/nuts_microbench_ca9.jl`) further
separate the per-stage costs: source mutation/invalidation, Hamiltonian getters, one
leapfrog, a depth-1 tree, and the full transition, each with allocation and wall
time and typed/LLVM evidence for the compiled `step!` path.
