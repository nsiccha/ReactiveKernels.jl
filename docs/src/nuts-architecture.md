# Fused compiled-NUTS kernel authoring

This page is an **ergonomic acceptance test**. It asks the real question behind the
fused No-U-Turn sampler: *how easy is it to author an efficient, allocation-free,
reusable mathematical kernel in `ReactiveKernels`?* It shows the **approved target
author syntax** (fixture v2) beside the **kernel that actually runs today**, and states
exactly what the compiler already infers versus what this task still has to build —
tracked in [`ReactiveKernels:syntax` todo `152a6td`](https://claude.ai/code). The fused
NUTS endpoint is the canonical usability gate.

Where an *actual* compiled program exists it is shown through the build-executed
**Raw input / Generated kernel / Compute DAG** renderer; where a piece is not yet
production code it is labeled **planned** and no generated pane is invented for it.

```@eval
Main.ReactiveKernelsDocs.render_build_commit()
```

> [!IMPORTANT]
> **This is a feasibility benchmark, not the public sampler.** Everything marked
> **REAL** is build-executed or independently measured at the accepted benchmark tip
> [`71f37a2`](https://github.com/nsiccha/ReactiveKernels.jl/commit/71f37a2) — a
> *borrowed-output prototype* proving the fused leaf is correct and fast. Pieces
> marked **PLANNED** are not implemented as production code. The shipped,
> reviewable-today sampler is the slower reactive path on the
> [NUTS sampling](nuts.md) page; this page explains the fused design that replaces
> its hot leaf, and *how it should be authored*.

## Why a fused leaf — the reactive leaf was slow

The [NUTS sampling](nuts.md) page ships a genuinely compiled-reactive sampler, but a
reactive `get!` still pays, on every leaf, for the generic machinery that makes
reactivity *general*: validity-bit checks, provenance-aware invalidation, and
active-endpoint selection. On the matched four-chain benchmark that overhead makes
the reactive sampler's warmup+draw wall time roughly **4–7×** AdvancedHMC/DynamicHMC
— the residual gap is reactive bookkeeping, not setup or the gradient.

A NUTS transition repeats one **leaf** (a leapfrog step plus its energy error)
thousands of times, always in the same order. That leaf does not need a general
reactive engine — it needs one straight-line schedule compiled once, performing
**no** `get!`, validity, or invalidation work in the hot path. That is the fused
leaf below.

## The usability gate: authoring an endpoint three ways

The same Hamiltonian endpoint, authored three ways. Column 2 is what runs **today**
(and is build-executed in the Unit A panel below); Column 3 is the **approved target
syntax** this task delivers (`ReactiveKernels:syntax` `152a6td`, fixture v2). Column 3
is illustrative — it is **not** executable yet, so no generated/DAG pane is shown for
it; the honest "what exists vs what this task builds" split is spelled out beneath it.

### 1. Handwritten mutable Julia — fast, but hand-everything

```julia
mutable struct HamEndpoint{T,M}
    pos::Vector{T}; mom::Vector{T}; metric::M
    chol; logdetM::T                       # hoisted by hand: recomputed only on metric change
    grad::Vector{T}; vel::Vector{T}        # hand-owned reused buffers
    pot::T; kin::T; ham::T
end
function refresh!(s)                        # author hand-writes ordering + what to recompute
    s.pot = s.potential_gradient!(s.grad, s.pos)
    ldiv!(s.vel, s.chol, s.mom); s.kin = 0.5*(s.logdetM + dot(s.mom, s.vel))
    s.ham = s.pot + s.kin
end
function leapfrog!(s, h)
    @. s.mom -= 0.5h*s.grad; refresh!(s)    # must remember to refresh after each write
    @. s.pos += h*s.vel;     refresh!(s)
    @. s.mom -= 0.5h*s.grad
end
adapt_metric!(s, M) = (s.metric = M; s.chol = cholesky(M); s.logdetM = logdet(s.chol); refresh!(s))
```

0-B once the buffers exist, but there is no incrementality (every `refresh!` reruns the
gradient even when only `mom` moved), the buffers and recompute triggers are hand-managed,
there is no reuse/compose, and accepted-state isolation is a manual copy.

### 2. Current ReactiveKernels — reactive and 0-B, but heavy to author

Reactive and allocation-free, but the author writes, per program: owned-bundle
`mutable struct`s (`_ValueGradient`/`_Kinetic`), named recipe functions + projections,
**three** `_nuts_cache_apply` methods, `_nuts_is_mutating`, `_nuts_prepare`,
`_copy_slot_value(!)` hooks, every `typeof(...)` annotation, and the endpoint block
**hand-unrolled ×3** (no loop grammar). The **Unit A panel below build-executes exactly
this level** — the real graph construction plus its generated fused schedule and DAG.

### 3. Intended public syntax — the approved target (fixture v2, `152a6td`)

> **Illustrative TARGET — not currently executable.** No generated/DAG pane is shown
> for it; the substrate it needs is reviewed but not yet canonical (see below).
>
> **Superseded direction note:** the approved authoring surface now **removes
> `@reactive`** in favour of a single method-bearing **`@kernel`** macro (a phase-point
> *endpoint* object with in-place `leapfrog!`/`refresh_momentum!` methods and a composed
> `sampler`, ReactiveHMC-faithful — approved as V7, mid-implementation). The
> `@reactive endpoint` form below is the earlier shape of that same design; the real
> unified `@kernel` definitions will be shown here build-executed and drift-proof from
> HMC's reviewed authoring fixture once it lands.

```julia
@reactive endpoint(potential_gradient!, metric, pos, mom) = begin
    chol             = cholesky(metric)
    logdetM          = logdet(chol)
    (pot, dpot_dpos) = value_gradient(potential_gradient!, pos)
    dham_dmom        = mass_solve(chol, mom)
    kin              = 0.5 * (logdetM + dot(mom, dham_dmom))
    ham              = pot + kin
    leapfrog!(h) = begin
        @. mom -= 0.5h * dpot_dpos
        @. pos += h * dham_dmom
        @. mom -= 0.5h * dpot_dpos
    end
    adapt_metric!(M) = begin
        metric = M
    end
end
init = endpoint(pgrad!, metric, pos0, mom0)
fwd  = copy(init)
bwd  = copy(init)
@reactive nuts_step(init, fwd, bwd, gofwd, min_dham) = begin
    active_ham = gofwd ? fwd.ham : bwd.ham
    dham       = init.ham - active_ham
    diverged   = dham < min_dham
end
s = nuts_step(init, fwd, bwd, true, -1000.0)
leapfrog!(s.fwd, 0.25)
s.dham
copyto!(s.init, s.fwd)
adapt_metric!(s.fwd, new_metric)
```

- The `(pos, mom)` field writes **infer** the leapfrog changed set; a graph proof
  **hoists** `chol`/`logdetM` because `metric` is unchanged. Assigning `metric` selects
  the other update mode and recomputes them exactly once on the next dependent read.
- RK registers the `value_gradient` / `mass_solve` in-place policies **once**; kernel
  authors write no cache applier, bundle, or ownership metadata.
- The same source specializes over `Float32`/`Float64`, `Matrix`/`Diagonal`, and any
  callable gradient type. The consumer `potential_gradient!(grad, pos) -> value` may be
  analytic **or** an optional DI + Enzyme-prepared callable (permitted).
- `copy` owns distinct buffers; a flattened composition makes cross-endpoint
  invalidation and schedules sound — the endpoint is authored once and replicated by
  `copy`, not unrolled ×3.
- A library author's *custom* in-place op is the only irreducible registration surface.
  V1 (recommended): `@inplace myop!(out, a, b)` generates a logical
  `myop(a, b) -> (value, out)`; kernel authors never call the 3-arg impl. The variant is
  the user's pick — decision `0dvxevh`.

### What the target infers (nothing hand-written) — and eliminates

**Inferred:** want/bindings/graph/schedule/freshness; owned in-place outputs (via the
registered combinator effects); the changed set (field-LHS writes + registered effects);
the persistent partition (graph-hoist proof); copy-isolation (facade); loop-free reuse
(`copy` + compose). **Eliminated vs Column 2:** the bundle structs, named recipes +
projections, all cache appliers, `is_mutating`, the `prepare` hook, the copy hooks, the
`typeof` annotations, and the ×3 unroll.

### Current vs compiler-work-still-required (honesty)

**Exists today** (`61ec0c8` + array `compile_update` `849683`, verified): the `@reactive`
facade (signature sources, derived recipes, invalidation-tracked mutation methods,
`get`/`set!`/`mutate!`/`touch!`/`assign!`, in-place `@. field -= …` writes, facade
`copy`/`copyto!`), `specialize=true`, the `prepare=` hook; the `compile_update` array core
with `_RecipeApplier`/`cache_policy`/`alias_writes`/`bind`; `prepare_reactive_nonallocating`;
KernelSpec `compose`/`merge`. The current 0-B NUTS path works — but only via the
hand-written bundle/cache-applier boilerplate of Column 2.

**This task adds** (not yet built): the `in_place_effect` trait + `@inplace` registration +
the construction-time resolver (roles→Values, `cache_policy` synthesis, auto-hoist proof,
bundle+projection generation); the RK-provided `value_gradient`/`mass_solve` combinators +
their once-registered policies; poc's core hook consuming the resolved policy in
`compile_update` cache-selection + the facade ordered-per-occurrence effect-metadata trace
(enabling changed-inference and nonlocal bundle→projection resolution); and auto-flattening
`compose` across `@reactive` objects (today `reactive_nuts.jl` flattens by hand). So Column
3 is the target; the combinators, inference, auto-hoist, and auto-flatten-compose are the
deliverable, not current API.

The expert escape hatch `compile_update(...; changed, persistent, want, input_binding,
output_binding, cache_policy)` stays public and inspectable (Raw input / Generated kernel /
Compute DAG) — but is never the default author experience.

## What is genuinely easy — and correct — today

Not everything is a defect. Two things work well and should stay as they are:

- **Ordinary Julia stays ordinary.** The dynamic tree control — recursion, the
  U-turn criterion, proposal selection, RNG order, and the control scalars — reads
  the leaf-written slots as plain Julia. It is *not* have→want graph work, and
  forcing it into recipes would only manufacture fake DAGs. The fused driver mirrors
  the reactive oracle's RNG draw order exactly, so tree growth (hence gradient count)
  matches for the same seed.
- **Control flow inside a `@kernel` recipe is allowed.** A recipe right-hand side
  may contain `try`/`catch`/`finally`, `let`, comprehensions, and `do` blocks,
  because each recipe compiles into an **opaque ordinary-Julia `op` closure** over
  its free ports — the control flow runs *inside* the op, where there is no reactive
  invalidation to defeat. Free ports referenced inside such forms are still detected
  as recipe dependencies, and a `catch e` whose variable collides with a port name is
  automatically renamed (hygiene) so it never shadows the port. This is the opposite
  of the higher-level `@reactive` **method** surface, which *rejects* `let` /
  `try`/`catch` / comprehensions / `do` at expansion: a `@reactive` method body is
  invalidation-tracked field-by-field, and those forms' deferred/except execution
  would defeat that tracking. Same principle, opposite verdict — because a recipe op
  is opaque and a reactive method body is tracked.

## The compiler expansion — Unit A, build-executed

The panel is build-executed: **Raw input** is the exact column-2 construction plus
the values it ran on; **Generated kernel** is the real `code_expr` of the fused
non-allocating schedule (one straight-line function over the recipe caches — no graph
traversal); **Compute DAG** is that same `plan`. The docs build asserts a non-vacuous
coverage gate — exactly 9 HAVE ports, 9 WANT outputs, 13 recipes, tied to the live
plan — so a missing or extra recipe fails the build.

```@eval
Main.ReactiveKernelsDocs.render_fused_leaf(@__MODULE__)
```

The leaf's `pos`, `mom`, `old_grad` change every call; `chol`, `stepsize`,
`init_ham`, `threshold`, the `pgrad` closure, and `logdet_chol` are the
**persistent partition**, constant while the metric and step size are fixed. The
panel above uses an **analytic** `pgrad` (∇U(x)=x) so the kernel — not the
differentiation — is the visible content; ReactiveKernels computes no pullbacks
itself and accepts any consumer gradient, including an optional
DifferentiationInterface + reverse-mode Enzyme integration. Exactly one recipe
(`_grad_bundle`) calls the gradient, writing potential *and* gradient into one owned
`_ValueGradient` bundle; `_vg_gradient`/`_vg_value` are borrowed projections.
**One leaf ⇒ one gradient.**

## The four compiled units — honest status

| Unit | What it is | Status |
|---|---|---|
| **A** | Fused Hamiltonian/leapfrog endpoint update (the leaf above) | **REAL** — build-executed |
| **C** | Accepted-state boundary recomputation (one boundary gradient) | **REAL** — benchmark shadow driver |
| **B** | NUTS criterion / diagnostic update | **PLANNED** — not yet production code |
| **D** | Dual averaging, Welford metric adaptation, statistics | **PLANNED** — not yet production code |

### Unit C — accepted-state boundary recomputation (REAL)

When a transition is accepted, the initial endpoint must be made consistent at the
new position and momentum. In the benchmark this is ordinary Julia around the leaf —
**no compiled DAG, so none is shown**:

```julia
# Recompute an endpoint's full derived state from (pos, mom): grad = ∇U (ONE gradient),
# vel = M^-1 mom, kinetic, Hamiltonian. Makes `init` consistent at the boundary.
function _seed_endpoint!(ep, s)
    pot = s.pgrad!(ep.grad, ep.pos)             # ONE call: fills grad = ∇U(pos), returns U
    copyto!(ep.vel, ep.mom); ldiv!(s.chol, ep.vel)
    kin = (s.logdet_chol + dot(ep.mom, ep.vel)) / 2
    ep.pot = pot; ep.kin = kin; ep.ham = pot + kin
    ep
end

_restore_init!(s) = (p = s.proposals[end];      # after an accepted transition
    copyto!(s.init.pos, p.pos); copyto!(s.init.mom, p.mom);
    _seed_endpoint!(s.init, s); s)              # the ONE boundary gradient

# Momentum refresh: pos is unchanged, so the stored potential and gradient stay
# valid — reuse them, NO gradient call (mirrors the reactive refresh).
function fused_refresh!(s)
    randn!(s.rng, s.init.mom); lmul!(s.chol.L, s.init.mom)
    copyto!(s.init.vel, s.init.mom); ldiv!(s.chol, s.init.vel)
    kin = (s.logdet_chol + dot(s.init.mom, s.init.vel)) / 2
    s.init.kin = kin; s.init.ham = s.init.pot + kin; s
end
```

The boundary recompute costs **one gradient per accepted transition — moved, not
added** (the momentum-refresh gradient relocated to the accepted state). With one
gradient per leaf, the fused transition eliminates exactly the two redundant reactive
recomputations per transition that the generic `get!` path incurred.

### Units B and D — PLANNED

- **Unit B — NUTS criterion / diagnostic update.** The U-turn criterion, the running
  summed-momentum reduction, and per-transition diagnostics are ordinary Julia in the
  benchmark; *planned* to become a compiled segment via the effect-metadata layer.
  Not built.
- **Unit D — dual averaging, Welford adaptation, statistics.** These exist today as
  compiled reactive programs on the [NUTS sampling](nuts.md) page. Composing them into
  the *fused* owned-slot transition through the same `compile_update` path is
  *planned*, not built.

No generated panes are shown for B or D — there is no fused production program for
them at `71f37a2`.

## Benchmark and correctness — the accepted receipt (REAL)

These figures are **not** re-measured at docs build (throughput is not stable in CI);
they are the independently accepted receipt at
[`71f37a2`](https://github.com/nsiccha/ReactiveKernels.jl/commit/71f37a2), reproducible
from a tracked-clean detached-HEAD worktree at that SHA. On the controlled **D=8
Gaussian gate**, differentiating the **same** scalar log density through
DifferentiationInterface + reverse-mode Enzyme for every sampler:

| Sampler | Gradients/s | Bytes/transition |
|---|---|---|
| **Fused ReactiveKernels** | **≈ 2.61 M** | **0** |
| AdvancedHMC 0.8.6 | ≈ 2.04 M | ≈ 27.2 KB |
| DynamicHMC 3.6.1 | ≈ 1.77 M | ≈ 31.4 KB |

That is **≈ 1.3× AdvancedHMC** and **≈ 1.5× DynamicHMC** on this gate, at zero bytes
per transition — a **feasibility benchmark on one controlled workload**, not a blanket
superiority claim, and *not yet the public sampler*.

**Reproduce.** `julia benchmark/nuts_bench_setup.jl` builds `benchmark/bench-env`
pinned to MutatingFunctions `b353559`, DifferentiationInterface `0.7.21`, Enzyme
`0.13.199`, AdvancedHMC `0.8.6`, DynamicHMC `3.6.1`, LogDensityProblems `2.2.0`. Then
run the SHA-guarded scripts `benchmark/nuts_fused_parity.jl`,
`benchmark/nuts_statistical_correctness.jl`, `benchmark/nuts_sampler_comparison.jl`.

The prototype is gated three independent ways at `71f37a2`:

- **Full-state + RNG parity — 300 transitions.** Bit-exact position, momentum,
  gradient, velocity, potential, kinetic, Hamiltonian, and NUTS diagnostics against
  the reactive oracle, with an identical RNG-stream continuation and exact gradient
  accounting (one per leaf + one accepted-state boundary gradient).
- **Analytic AR(1) statistical gate.** Against a correlated Gaussian with **known**
  covariance `Σ = [ρ^|i−j|]`, `ρ = 0.5`, sampled by many overdispersed independent
  chains (final state per chain, ≈ i.i.d.): per-dimension means within a z-CI of 0,
  the sample covariance recovers `Σ` within Monte-Carlo tolerance, and per-dimension
  Kolmogorov–Smirnov passes against the analytic marginal `N(0, Σ_dd)`. AdvancedHMC
  and DynamicHMC run the **same** protocol as independent cross-checks. Correctness
  means matching the *true distribution*, not the reactive path.
- **Type/LLVM and allocation gates** on the actual timed windows.

## The production path — reviewed substrate, not yet canonical

> **Reviewed compiler substrate, not canonical production integration.** Scalar
> `compile_update` checkpoint `5817984` and the array/alias-effect overlay `849683`
> have passed core review. They establish the expert
> `compile_update → bind_schedule → begin_updates!/bound()/finish_updates!` machinery,
> authoritative owned-array outputs, cut-point reuse, isolation, typed/LLVM gates, and
> stable-shape 0-B execution. They are approved overlays **awaiting canonical
> landing/consumption**; the public HMC sampler has not yet migrated Units A/C onto
> them, and Units B/D effect-metadata compilation remains open. Therefore the substrate
> is real/reviewed, while the concise target syntax and the complete fused sampler
> remain planned.

Turning the benchmark prototype into the public sampler consumes that reviewed
substrate: the owned-slot `compile_update` path (typed seed-once cache, aliasing owned
outputs, array/alias-effect metadata so alias-writes participate in the freshness
validator), plus the construction-time resolver and combinators of fixture v2 above. The
production sampler must then compile and compose Units B and D through the same path and
**repeat every gate above** before it replaces the reactive hot leaf.

This page stays current under a build-executed drift gate: every generated pane, DAG,
and inventory is regenerated from source at build time and fails the build on drift
(tracked in [`ReactiveKernels:docs` todo `10qxx92`](https://claude.ai/code)).

## See also

- [NUTS sampling](nuts.md) — the shipped compiled-reactive sampler (the slower path
  this design replaces in the hot leaf).
- [Non-allocating kernels](nonallocating.md) — the general non-allocating preparation
  surface the fused leaf specializes.
- [Visualization](visualization.md) — the interactive Compute-DAG renderer used above.
