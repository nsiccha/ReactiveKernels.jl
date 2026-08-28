# NUTS sampling

ReactiveKernels' No-U-Turn sampler is authored as a **single, method-bearing
`@kernel` surface** — eight named specifications that together are the whole sampler
— modeled on the ReactiveHMC.jl algorithm structure. This page shows the **authoring
source** below, and — now that the compiler is landed — the **measured performance** of
the compiled sampler.

The public `nuts!!` sampler is **landed and executable on `main`**: `@kernel` lowers the
NUTS source to a sealed, registry-free **native compiled recursion** (`compile_nuts_native`
/ `_build_nuts_sampler`), and the public `nuts!!(state; rng)` mutates compiler-owned state
in place and returns the **same object** (`result === state`, same concrete type) at
**exact zero allocations**. The source below is **byte-synced, drift-proof** from the
reviewed fixture (`benchmark/nuts_kernel_authoring_fixture.jl`), read at build time. The
measured leapfrog-steps/s comparison against DynamicHMC, AdvancedHMC, and nsiccha/NUTS.jl
is recorded in the static receipt [`benchmark/receipts/nuts-g7-v1.toml`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/receipts/nuts-g7-v1.toml)
— parsed here, not re-run in CI. No ESS or wall-time result is claimed anywhere on this page.

## Status — read this before the code

| Piece | State |
|---|---|
| Source contract (the eight `@kernel` specs below, the seven `@rk_*` effect registrations, the plan shape) | **Settled** — this is the reviewed authoring surface on `main`. |
| All eight source specs construct; concrete phasepoint/frame init/recompute/copy verified | **Landed on `main`** — the compiler constructs and runs the whole sampler; sealed production certificate `mode = production`. |
| Executable leapfrog (leaf scope) | **Verified** — analytic F32/F64; normal gradient Δ1, `@inferred`, exact 0-B; dirty-produced recovery analytic; dirty-source reject. |
| Public `nuts!!` sampler (`step!`, tree growth, U-turn) | **Landed on `main`** — sealed registry-free native recursion; `nuts!!(state; rng) === state` (same object, fixed type), **exact 0-B** on the public path. |
| Performance (work-normalized leapfrog-steps/s) | **Measured** — see [`benchmark/receipts/nuts-g7-v1.toml`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/receipts/nuts-g7-v1.toml): RK **beats AdvancedHMC and DynamicHMC** (~1.6–1.7×), and is ~0.86× of nsiccha/NUTS.jl (reported reference), all over ONE shared DifferentiationInterface+Enzyme gradient with matched target/mass/stepsize/RNG. **No** ESS, wall-time, bitwise/RNG oracle claim is made. |

The sealed native compiler (`kernel_nuts_native.jl`, `_build_nuts_sampler`) and the
minimal-reset authoring fixture are **on `main`**; the public `nuts!!` runs there. The
performance figures cited on this page come from the static receipt, not a CI perf run.

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
  acceptance rate, independent of the gradient and the integrator body. It runs on the
  sealed sampler at zero allocations.
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

## The locked compiler/lowering contract

These are the properties the `@kernel` lowering is **locked to** and the landed public
`nuts!!` entry **satisfies** (see the status table).

- **Captured source, exact effect registrations.** The compiler schedules the public
  `@rk_pure` / `@rk_borrows` / `@rk_rng` helpers by their *registered* effects, never by
  inferring the body. Authors touch no internals.
- **Immutable plan.** Construction produces a fixed-shape, fixed-type plan; the public
  entry mutates compiler-owned concrete state and returns the same object. The public
  identity holds: `nuts!!(state; rng)` returns `result === state` (same object, fixed
  concrete type) at **exact 0-B** — verified on the sealed native sampler.
- **Owned endpoints vs shared authority.** Read-only authority inputs (`grad_f`,
  `metric`, the metric-only `chol_metric` closure and the `@node(logdet(chol_metric))`
  value) are **shared by identity** across `init` / `fwd` / `bwd`; the integrator-written
  `pos`, `mom` and their derived closures (`pot`, `dpot_dpos`, `kin`, `ham`, and the
  `dham_*` alias projections) are **owned/distinct** per endpoint. The registered
  structural strong-update `copy!!(dest, src)` moves only the owned closure and leaves
  shared authority untouched; a metric change updates the one shared slot exactly once.
- **No `Ref`.** State is implicit-field; direction is concrete branch calls threading a
  physical endpoint, so there is no aliasing indirection to reason about.

## The authoring source

The block below is the exact reviewed source, read drift-proof at build time — the
authoring surface that `@kernel` lowers to the sealed native sampler now on `main`. The
durable, inspectable copy is
[`benchmark/nuts_kernel_authoring_fixture.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/nuts_kernel_authoring_fixture.jl);
this page renders live at
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
