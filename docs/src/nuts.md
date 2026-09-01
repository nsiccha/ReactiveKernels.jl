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
type) at **exact zero allocations**. The docs build reads
`benchmark/nuts_kernel_authoring_fixture.jl` only as inert text: it does not
include, parse, lower, compile, or execute that fixture. Executable evidence
lives in the test suite, not in this page's build. The measured
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

## Reactant receipt

The frozen adaptive-NUTS Reactant result now lives on the
[static Reactant receipt page](nuts-reactant.md). That page is receipt-only: neither it
nor this sampling page executes NUTS compiler/runtime code during the docs build.

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

## Compiler/runtime execution is disabled in docs

The docs build does not construct the reactive NUTS group, prepare a phasepoint
kernel, generate a getter, build a `ReactiveProgram`, or run the sealed sampler.
Those moving compiler/runtime paths are owned by focused tests. This page keeps
only frozen receipts and the exact authoring source read as text.

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

## Sampler execution: not part of the docs build

The docs build does not construct or run the sealed sampler. The `nuts!!`
identity, zero-allocation, and diagnostics claims above are established by the
test suite against the same source fixture; running the sampler machinery
during every docs build would couple publication to an execution surface that
the compiler work is actively moving. This page displays the exact authored
source and the frozen receipts only.

## The authoring source

The block below is the **exact reviewed authoring source** — the surface that `@kernel`
lowers to the sealed, registry-free native NUTS sampler now on `main`. The docs build
reads it as inert text directly from the durable fixture
[`benchmark/nuts_kernel_authoring_fixture.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/nuts_kernel_authoring_fixture.jl);
it does not load the definitions or request captured MethodIR. This page renders live at
<https://nsiccha.github.io/ReactiveKernels.jl/dev/nuts>.

The source below is the macro-free executable fixture as it exists today.
Its helpers are inline arithmetic/control or captured sibling `@kernel` methods;
there are no user-authored effect declarations. The fixture's `@node` use is
unrelated and stays.

::: details Show the complete authoring fixture

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
