# Compiled-reactive NUTS sampling

`ReactiveKernels` includes a multinomial No-U-Turn sampler whose per-transition
Hamiltonian work is a genuinely *compiled reactive* kernel. The public workflow —
[`reactive_nuts_group`](@ref) + [`nuts_state`](@ref), with warmup and statistics —
compiles **five** distinct [`ReactiveProgram`](@ref)s. Everything else is ordinary
inferred Julia: the tree-growth recursion, RNG draws, U-turn criteria, leapfrog
integration, and the adaptation/statistics update methods all run *over* these
compiled reactive handles; they are **not** themselves reactive graphs, and no DAG
is manufactured for them.

The public model boundary is a **scalar potential**; its gradient goes through
[DifferentiationInterface](https://github.com/JuliaDiff/DifferentiationInterface.jl)
with reverse-mode [Enzyme](https://github.com/EnzymeAD/Enzyme.jl), prepared once and
written into the sampler's owned buffer in place. The complete runnable workflow is
[`examples/nuts.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/examples/nuts.jl).

## The five compiled reactive programs

Each panel below shows the **actual kernel**, not an opaque constructor call.
**Raw input** leads with the *real `@reactive` authoring* of the kernel — the exact
recipe math (and, for the adaptation kernels, the inner `fit!`/`step!` update method)
read straight out of `src/reactive_nuts.jl` — followed by how you **construct and
interact** with it (build it, then read a getter or run its update method) and the
value that read returns. **Generated kernel** is the real `code_expr` of a selected
load-bearing getter of the actual `reactive_program` (a fused derived getter where
the program has derived nodes, a source-slot getter for a state-only program);
**Compute DAG** is that same `program.plan`, whose interactive graph carries the
*complete* recipe inventory. Choose **Compare all** to inspect the three views side
by side while preserving the DAG's fit, zoom, pan, and node-inspection state.

So you can read the authoring syntax and the interaction directly — the definitions
are the actual source, the getters/update methods are exactly how you drive them —
and give feedback on the API without opening any implementation file.

The docs build asserts a mechanical one-to-one coverage gate: exactly these five
programs, each artifact's program/DAG/getter tied to the live
`reactive_program(object)`, and each program's full source/derived recipe census
matching its declared inventory — so a missing or extra recipe fails the build.

- The **NUTS group** program fuses the whole per-endpoint Hamiltonian: for each of
  `init`/`fwd`/`bwd` an owned value–gradient bundle and kinetic bundle with their
  velocity/kinetic/gradient/potential projections, the Cholesky metric factor and
  per-endpoint hamiltonian, and then the reactive active-endpoint selection
  (`active_ham`), energy error (`dham`), and `diverged` flag. Its control and
  diagnostic surface (`gofwd`, `may_sample`, `may_continue`, `depth`, `n_steps`,
  `acceptance_sum`, and the completed-transition snapshots
  `last_energy_error`/`last_diverged`, plus the `min_dham` threshold) are HAVE
  sources on the same program. The selected `dham` getter shows how all of that
  work fuses into one straight-line function with no graph traversal.
- **DualAveragingState** compiles the Nesterov step-size recurrence
  (`log_current`/`current`/`final`) over its accumulator sources; `fit!` is an
  ordinary method advancing those sources.
- **WelfordVariance**, **TrajectoryStats**, and **SamplingStats** are *state-only*
  reactive programs: their authoritative buffers/counters/history are HAVE sources
  with no derived nodes, and their `step!`/recorder/reduction methods are ordinary
  Julia over those handles.

```@eval
Main.ReactiveKernelsDocs.render_five_programs(@__MODULE__)
```

## The sampled path

The same compiled `reactive_nuts_group` program drives warmup and sampling through
the DI + Enzyme scalar-potential boundary:

```@example nuts
using LinearAlgebra
using Random
using ReactiveKernels
using DifferentiationInterface
import Enzyme

backend = AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
                     function_annotation = Enzyme.Const)
potential(q) = sum(abs2, q) / 2
dimension = 4
preparation = prepare_gradient(potential, backend, zeros(dimension))
potential_gradient!(gradient, position) = first(value_and_gradient!(
    potential, gradient, preparation, backend, position))

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
