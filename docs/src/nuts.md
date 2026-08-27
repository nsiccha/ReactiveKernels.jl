# NUTS sampling

ReactiveKernels' No-U-Turn sampler is authored as a **single, method-bearing
`@kernel` surface** — eight named specifications that together are the whole sampler
— modeled on the ReactiveHMC.jl algorithm structure. This page shows the **current
authoring source** so you can read and give feedback on the syntax and the compiler
contract while the executable lowering is being finished.

The source below is **byte-synced, drift-proof, from the reviewed fixture commit
`ccb35d3`** (`benchmark/nuts_kernel_authoring_fixture.jl`), read at build time. The WIP
compiler substrate on the `syntax` / `poc` / `hmc` branches already **constructs all
eight `@kernel`s and executes the phasepoint and the leapfrog leaf**; this
`docs`/`main` build deliberately renders the **source only**, because the tree/root
(`nuts!!`) lowering is not yet landed on `main`. So no generated-kernel pane,
Compute-DAG, parity oracle, allocation number, or throughput figure is shown here —
those arrive when the tree/root lowering lands.

## Status — read this before the code

| Piece | State |
|---|---|
| Source contract (the eight `@kernel` specs below, the seven `@rk_*` effect registrations, the plan shape) | **Settled** — this is the reviewed authoring surface on `main`. |
| All eight source specs construct; concrete phasepoint/frame init/recompute/copy verified | **Verified on the WIP `poc`/`hmc` substrate** — executable there; **not** on `docs`/`main`. (DA/Welford runtime construction is a separate receipt, not this one.) |
| Executable leapfrog (leaf scope) | **Verified** at accepted SHA `6085efd` — real `ccb` source; analytic F32/F64; normal gradient Δ1, `@inferred`, exact 0-B; dirty-produced recovery Δ2 analytic; dirty-source reject. |
| Tree / root `nuts!!` lowering (`step!`, tree growth, U-turn, adaptation drive) | **In progress** — not yet lowered/landed. |
| Final acceptance + performance | **Staged; RK arm pending** — the acceptance harness is fully staged/wired and the AHMC / DHMC + analytic arms run; the RK G1–G14 / full perf run awaits the public `nuts!!`. **No** ESS, wall-time, bitwise/RNG oracle, or throughput claim is made anywhere on this page. |

The verified construction/phasepoint/leaf results above live on the WIP `poc`/`hmc`
branches; those compiler commits are **not** on `main`. The `main` docs build carries
the **source contract only**.

The ReactiveHMC.jl `ca9` structure is an **algorithm-structure reference only** — not
a bitwise or RNG target; improvements may change arithmetic or ordering.

## The mathematical UX — what you write

The design goal is that the *math is the code*: each `@kernel` reads as the recurrence
it implements, and the compiler — not the author — schedules effects, owns storage, and
invalidates stale values.

- **`euclidean_phasepoint(grad_f, metric, pos, mom)`** — a phasepoint is
  potential + kinetic energy at `(pos, mom)`. You write the four lines of physics
  directly: one destination-bound gradient recipe produces `pot, dpot_dpos = grad_f(pos)`
  (there is **no** separate `pot_f` producer), `chol_metric = cholesky(metric)`, the
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
  acceptance rate, independent of the gradient and the integrator body. It is not
  executable yet — only its registration, storage, and binding are being built;
  zero-allocation execution is the locked target, not a delivered result.
- **`nuts_state(init; step_f, …)`** — the sampler state, authored **implicit-field**
  with **no `Ref`**: explicit fixed physical `init` / `fwd` / `bwd` endpoints, the
  derived `diverged` recipe written once (never imperatively), and the tree-growth /
  U-turn / multinomial-swap logic as ordinary inner methods. Direction is two direct
  physical-endpoint branch calls, not a mutable current-view.
- **`nuts!!(state; rng)`** — the public compiled entry: refresh momentum on the owned
  `init`, `step!` the tree, `return state` (result **is** `state`; fixed shape/type).
- **`dual_averaging_state`** / **`welford_var`** — the adaptation recurrences: Nesterov
  dual averaging (`m`/`H`/`μ` + `fit!(x)`) and streaming Welford variance
  (`n`/`mean`/`var` + `step!(x)`), each written as its update rule.

## The locked compiler/lowering contract and acceptance target

These are the properties the `@kernel` lowering is **locked to** and the public
`nuts!!` entry is being **accepted against** — the leaf-scope pieces are verified today
(see the status table); the whole-`nuts!!` guarantees are the acceptance target until
the tree/root lowering passes.

- **Captured source, exact effect registrations.** The compiler schedules the public
  `@rk_pure` / `@rk_borrows` / `@rk_rng` helpers by their *registered* effects, never by
  inferring the body. Authors touch no internals.
- **Immutable plan.** Construction produces a fixed-shape, fixed-type plan; the public
  entry mutates compiler-owned concrete state and returns the same object. Today the
  **leaf** returns its *owned endpoint* and is exact 0-B (verified, SHA `6085efd`); the
  full **public identity** — `nuts!!` returning `result === state` at exact 0-B — remains
  the acceptance target, pending the tree/root lowering.
- **Owned endpoints vs shared authority.** Read-only authority inputs (`grad_f`,
  `metric`, the metric-only `chol_metric` closure and the `@node(logdet(chol_metric))`
  value) are **shared by identity** across `init` / `fwd` / `bwd`; the integrator-written
  `pos`, `mom` and their derived closures (`pot`, `dpot_dpos`, `kin`, `ham`, and the
  `dham_*` alias projections) are **owned/distinct** per endpoint. The registered
  structural strong-update `copy!!(dest, src)` moves only the owned closure and leaves
  shared authority untouched; a metric change updates the one shared slot exactly once.
- **No `Ref`.** State is implicit-field; direction is concrete branch calls threading a
  physical endpoint, so there is no aliasing indirection to reason about.

## The authoring source (synced from `ccb35d3`, source-only)

The block below is the exact reviewed source, read drift-proof at build time. It is the
authoring *target* for the in-progress tree/root lowering — **not** an executable
sampler on this build, and it carries no parity, allocation, or performance claim. The
durable, inspectable copy on `main` is
[`benchmark/nuts_kernel_authoring_fixture.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/nuts_kernel_authoring_fixture.jl)
(provenance: reviewed fixture commit `ccb35d3`); this page renders live at
<https://nsiccha.github.io/ReactiveKernels.jl/dev/nuts>.

```@eval
Main.ReactiveKernelsDocs.render_authoring_fixture()
```

## How correctness is established (separately, not as docs content)

Correctness is established in **isolated verification harnesses**, by **mathematical,
independent** gates — Hamiltonian and gradient identities, leapfrog reversibility and
controlled/expected energy-*error* behavior (not exact energy conservation), NUTS tree /
multinomial / divergence / depth logic, the adaptation recurrences, and analytic target
distributions under justified tolerances. Those harnesses are not shown here, and none of
them is presented on this page as an accepted throughput or ESS result — see the status
table above for exactly what is and is not verified today.
