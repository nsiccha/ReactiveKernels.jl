# NUTS sampling

```@eval
Main.ReactiveKernelsDocs.render_result_assets()
```

This external NUTS compiler-acceptance exemplar is authored as a
**single, method-bearing `@kernel` surface** — eight named specifications that
together are the whole sampler — modeled on the ReactiveHMC.jl algorithm
structure. This page shows that artifact's **authoring source** and measured
performance. It demonstrates what ReactiveKernels compiles; it is not a sampler
API that ReactiveKernels intends to ship.

The fixture's `nuts!!` entry is **landed as executable compiler evidence on `main`**: `@kernel`
lowers the NUTS source to a sealed, registry-free **native compiled recursion**
(`compile_nuts_native` / `_build_nuts_sampler`). That entry mutates compiler-owned
state in place and returns the **same object** (`result === state`, same concrete
type) at **exact zero allocations**. The docs build loads the reviewed
`benchmark/nuts_kernel_authoring_fixture.jl` in an isolated module, reads the
displayed source from that same file, admits its captured kernels, and executes
the exact source-locked `nuts!!` interaction shown below. The measured
leapfrog-steps/s comparison against DynamicHMC, AdvancedHMC, and nsiccha/NUTS.jl
is recorded in the static receipt [`benchmark/receipts/nuts-g7-v1.toml`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/receipts/nuts-g7-v1.toml)
— parsed here, not re-run in CI. It is a work-normalized inner-loop receipt,
not an end-to-end sampling, adaptation, wall-time, or ESS benchmark.

Packaging matters: “public” inside the fixture comments means the entry of that
sealed external artifact, not an RK package API. The NUTS runtime, native emitter,
compiled-reactive compatibility implementation, and domain types live in the
`ReactiveKernelsNUTSExamples` nested package and are also available through the
thin `ReactiveKernelsNUTSExample` launcher. A bare `using ReactiveKernels` neither loads
nor exports them. See [Compiler capability and
limits](compiler.md#what-the-nuts-proof-does-and-does-not-establish) for the exact
boundary and why the two implementations prove different things.

## Status — read this before the code

```@eval
Main.ReactiveKernelsDocs.render_nuts_status()
```

```@eval
Main.ReactiveKernelsDocs.render_nuts_g7_benchmark()
```

The sealed native compiler (`packages/ReactiveKernelsNUTSExamples/src/nuts_runtime/kernel_nuts_native.jl`,
`_build_nuts_sampler`) and the minimal-reset external authoring fixture are
**on `main` as compiler evidence**. The figures in the G7 panel immediately above
measure that native acceptance artifact and come from the static receipt, not a CI perf
run or the external `CompiledNUTSState` comparison path. RK, AdvancedHMC, and
DynamicHMC used one shared potential-and-gradient authority and matched target,
mass, step size, and RNG schedule; the receipt also checks gradient/work
accounting. It does **not** measure adaptation, retained draws, ESS, or
time-to-effective-sample.

The ReactiveHMC.jl `ca9` structure is an **algorithm-structure reference only** — not
a bitwise or RNG target; improvements may change arithmetic or ordering.

## Reactant adaptive transition and multiple chains

The optional external adaptive-NUTS exemplar compiles one full-depth transition
to one data-dependent traced `while` and uses pre-generated momentum, direction,
and exponential tensors plus explicit counters, so there is no host RNG inside
the trace. This is a deliberately narrow compiler-acceptance path: it is
scoped to `Float64`, a positive diagonal Euclidean metric, the locked authored
control-flow graph, and the current diagnostics callback. Overflow and
unsupported cases reject; the native adaptive API remains CPU execution.

The source authority is
[`packages/ReactiveKernelsNUTSExamples/src/nuts_runtime/kernel_nuts_reactant.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsNUTSExamples/src/nuts_runtime/kernel_nuts_reactant.jl),
and
[`test/test_kernel_nuts_reactant.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/test/test_kernel_nuts_reactant.jl)
is the executable acceptance authority. The test requires one `stablehlo.while`
and checks the native oracle, random-input counters, divergence/nonfinite paths,
and fail-closed specialization guards.

### Measured Reactant performance

The matched benchmark below is the Reactant result that was previously missing
from this page. It executes the **same authored adaptive transition** through the
source-faithful native compiler and Reactant, starting from the same state and
using identical pre-generated random bundles at `max_depth = 10`. State is
independently initialized to the same value for each native/Reactant transition
pair outside the timed region. This prevents accumulated floating-point branch
drift in a chaotic carried chain from silently changing the compared work. A
deterministic candidate stream is screened outside timing, and the receipt
publishes how many candidates were excluded after backend-sensitive transition
parity mismatches. Floating phase-point and diagnostic values must match with
`atol = 128eps(Float64)` and `rtol = 0`; control counters and random consumption
must match exactly. The frozen receipt reports synchronous CPU execution, full-transition wall time,
work-normalized leapfrog steps/s, and compilation separately. Compilation,
host/device transfers, state setup, random-bundle generation, rebundling, and result readback
are outside steady-state timing.

```@eval
Main.ReactiveKernelsDocs.render_nuts_reactant_benchmark()
```

This baseline makes one synchronous compiled call per transition. Batching
independent chains or compiling an outer loop over several sequential
transitions could amortize dispatch and state-machine overhead, but neither is
measured here; the receipt is not evidence that the current single-transition
ratio is an inherent Reactant limit.

The executable source is
[`benchmark/nuts_reactant_comparison.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/nuts_reactant_comparison.jl),
and the immutable input to this panel is
[`benchmark/receipts/nuts-reactant-v1.toml`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/receipts/nuts-reactant-v1.toml).
This is a matched-control compiler/runtime microbenchmark on a fixed target, not adaptation,
retained-draw, ESS, accelerator-transfer, or time-to-effective-sample evidence.

The simpler fixed-step HMC kernel in
[`packages/ReactiveKernelsKernelExamples/src/hmc.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsKernelExamples/src/hmc.jl)
also keeps momentum and its Metropolis uniform explicit, uses a static leapfrog
count, and lets `replica` map the scalar kernel across chains. These are scoped
compatibility statements, not a claim that arbitrary mutable or reactive state
machines are accelerator compatible.

The separate [WALNUTS-D mathematical-kernel page](walnuts.md) keeps the same
phase-point, leapfrog, and depth-10 multinomial-NUTS mathematics, but replaces
each leaf with the released fixed-macro-time dyadic refinement and reverse-grid
test. It shows the exact authored `@kernel` slices consumed by the compiler.

## The mathematical UX — what you write

The design goal is that the *math is the code*: each `@kernel` reads as the recurrence
it implements, and the compiler — not the author — schedules effects, owns storage, and
invalidates stale values.

- **`euclidean_phasepoint(pot_f, grad_f, metric, pos, mom)`** — a phasepoint is
  potential + kinetic energy at `(pos, mom)`. You write both valid ways to obtain
  the potential: `pot = pot_f(pos)` and `pot, dpot_dpos = grad_f(pos)`. The planner
  selects the gradient recipe when `dpot_dpos` is wanted, while retaining both
  callable authorities by identity. Then `chol_metric = cholesky(metric)`, the
  kinetic term `kin = ½(logdet(chol_metric) + momᵀ M⁻¹ mom)`, and `ham = pot + kin`.
  The Hamiltonian-gradient fields are **alias projections that collapse onto the
  canonical owned gradient slots** — `dham_dpos` onto `dpot_dpos`, `dham_dmom` onto
  `dkin_dmom` — one physical slot each, not a second copy.
- **`leapfrog!(phasepoint; stepsize)`** — the leapfrog integrator is its three
  broadcast writes: half-kick the momentum, full-drift the position, half-kick again.
  It is an RK-authored **free** `@kernel` with visible ordered effects; the sampler
  binds `step_f = partial(leapfrog!; stepsize)`.
- **`refresh_momentum!!(phasepoint; rng)`** — momentum resampling `m ~ N(0, M)` is
  exactly `randn!` then `lmul!(L, m)`. The `mom` write **invalidates** the kinetic /
  Hamiltonian closures; the scheduler recomputes them on next read. You write no cache
  update.
- **`nuts_stats!(state)`** — a registered diagnostics callback over compiler-owned
  state: it increments the owned `n_steps` once per leaf and folds the running
  acceptance rate, independent of the gradient and the integrator body. It runs on the
  sealed sampler at zero allocations.
- **`nuts_state(init; step_f, …)`** — the sampler state, authored **implicit-field**
  with **no `Ref`**: explicit fixed physical `init` / `fwd` / `bwd` endpoints, the
  derived `diverged` recipe written once (never imperatively), and the tree-growth /
  U-turn / multinomial-swap logic as ordinary inner methods. Direction is two direct
  physical-endpoint branch calls, not a mutable current-view.
- **`nuts!!(state; rng)`** — the sealed fixture entry: refresh momentum on the owned
  `init`, `step!` the tree, `return state` (result **is** `state`; fixed shape/type).
- **`dual_averaging_state`** / **`welford_var`** — the adaptation recurrences: Nesterov
  dual averaging (`m`/`H`/`μ` + `fit!(x)`) and streaming Welford variance
  (`n`/`mean`/`var` + `step!(x)`), each written as its update rule.

## The full compiled kernel graph

The graph below is the **full compiled NUTS kernel**: `reactive_nuts_group`
compiles the entire per-transition Hamiltonian work into ONE flat
`ReactiveProgram`, and this panel renders `reactive_program(group).plan` directly
— a 67-node Compute DAG. The three phase-point endpoints `init`, `fwd`, and `bwd`
each compute potential, gradient, kinetic term, and Hamiltonian; the three
converge at the active-endpoint selection, the energy error `dham`, and the
`diverged` flag, while the shared metric Cholesky and the potential/gradient
authority fan into all three. This is the mathematical heart the sampler
evaluates on every step.

The tree recursion, leapfrog integration, U-turn criterion, and adaptation are
ordinary type-stable Julia driver methods **outside** any reactive `Plan` — they
*drive* this graph but are deliberately not graph recipes, so there is no single
`Plan` for the whole sampler, only this compiled per-step program driven by
native compiled recursion.

The panel is build-executed. **Raw input** is the group construction;
**Generated kernel** is a readable view of the fused `:dham` (energy-error)
getter — one representative compiled getter, not a whole-program listing; the
exact getter AST remains available through `code_expr`. **Compute DAG** is that
exact `reactive_program(group).plan`. **Compare all** opens the side-by-side
split view. Read the graph as on the [DAG visualization](visualization.md) page:
green nodes are `HAVE` inputs, orange nodes are `WANT` outputs, and blue nodes are
the selected recipes that compute them.

```@eval
Main.ReactiveKernelsDocs.render_nuts_compiled_kernel_dag(@__MODULE__)
```

## A single Hamiltonian kernel (subordinate example)

The graph above is the full compiled per-step kernel. This smaller panel zooms in
on a **single Euclidean phasepoint** — one endpoint's Hamiltonian recurrence —
extracted as an ordinary stateless `Plan`, so the core energy-and-gradient work
can be read on its own through the same three-pane view used by the distribution
and batched examples. It is a subordinate teaching extraction, not the full
compiled sampler kernel: the complete sampler is the method-bearing eight-spec
artifact below, whose recursive tree growth, ordered RNG effects, and in-place
updates are sealed by the native method compiler and do not reduce to one plan.

The panel is build-executed. **Raw input** contains the phasepoint math and the
selected HAVE/WANT boundary; **Generated kernel** is a readable view derived
from the executed kernel and exact `kernel.plan`, while `code_expr(kernel)`
remains the compiled AST. **Compute DAG** is that same plan. The potential-only
recipe is an alternative producer: because the requested outputs include the
position gradient, planning selects the combined value-and-gradient recipe and
does not execute the redundant potential path. **Compare all** opens the standard
side-by-side split view.

```@eval
Main.ReactiveKernelsDocs.render_nuts_phasepoint(@__MODULE__)
```

This docs-scoped stateless extraction mirrors the phasepoint recurrence for
inspection; the full byte-locked eight-spec source later on this page remains the
authoritative sealed NUTS compiler-acceptance fixture.

## The locked compiler/lowering contract

These are the properties the `@kernel` lowering is **locked to** and the landed sealed
fixture entry **satisfies** (see the status table).

- **Captured source and exact call authority.** The external fixture uses visible
  arithmetic/control and captured sibling `@kernel` methods; ordinary unregistered
  helpers reject rather than acquiring inferred authority. The former `@rk_pure`,
  `@rk_borrows`, and `@rk_rng` declarations have been removed. `@node` remains
  supported.
- **Immutable plan.** Construction produces a fixed-shape, fixed-type plan; the sealed
  entry mutates compiler-owned concrete state and returns the same object. The fixture
  identity holds: `nuts!!(state; rng)` returns `result === state` (same object, fixed
  concrete type) at **exact 0-B** — verified on the sealed native sampler.
- **Input isolation and prepared storage.** Construction leaves caller inputs untouched
  and unaliased with writable sampler storage. Preparation creates compiler-owned
  endpoint buffers; repeated calls reuse and mutate those buffers in place, and the
  public result is the prepared state object itself.
- **Owned endpoints vs shared authority.** Read-only authority inputs (`pot_f`, `grad_f`,
  `metric`, the metric-only `chol_metric` closure and the `@node(logdet(chol_metric))`
  value) are **shared by identity** across `init` / `fwd` / `bwd`; the integrator-written
  `pos`, `mom` and their derived closures (`pot`, `dpot_dpos`, `kin`, `ham`, and the
  `dham_*` alias projections) are **owned/distinct** per endpoint. The registered
  structural strong-update `copy!!(dest, src)` moves only the owned closure and leaves
  shared authority untouched; a metric change updates the one shared slot exactly once.
- **No `Ref`.** State is implicit-field; direction is concrete branch calls threading a
  physical endpoint, so there is no aliasing indirection to reason about.

## Build-executed authored sampler entry

The following interaction is evaluated verbatim during the docs build. It
constructs the sealed sampler from the build-loaded fixture, calls that
fixture's authored `nuts!!` entry with an explicit `Xoshiro(1)`, and displays
the resulting diagnostics. Publication fails if the returned object identity or
committed diagnostics drift.

```@eval
Main.ReactiveKernelsDocs.render_nuts_source_interaction()
```

## The authoring source

The block below is the **exact reviewed authoring source** — the surface that `@kernel`
lowers to the sealed, registry-free native NUTS sampler now on `main`. The docs build
reads it directly from the durable fixture
[`benchmark/nuts_kernel_authoring_fixture.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/nuts_kernel_authoring_fixture.jl);
loads all eight definitions in an isolated module, requires the captured MethodIR
for every method-bearing definition, and refuses to publish if the compiled
sampler interaction above fails. This page renders live at
<https://nsiccha.github.io/ReactiveKernels.jl/dev/nuts>.

The build-loaded source below is the macro-free executable fixture as it exists today.
Its helpers are inline arithmetic/control or captured sibling `@kernel` methods;
there are no user-authored effect declarations. The fixture's `@node` use is
unrelated and stays.

::: details Show the complete build-loaded authoring fixture

```@eval
Main.ReactiveKernelsDocs.render_nuts_complete_source()
```

:::

## How correctness is established (separately, not as docs content)

Correctness is established in **isolated verification harnesses**, by **mathematical,
independent** gates — Hamiltonian and gradient identities, leapfrog reversibility and
controlled/expected energy-*error* behavior (not exact energy conservation), NUTS tree /
multinomial / divergence / depth logic, the adaptation recurrences, and analytic target
distributions under justified tolerances. Those harnesses are not shown here, and none of
them is presented on this page as an accepted throughput or ESS result — see the status
table above for exactly what is and is not verified today.
