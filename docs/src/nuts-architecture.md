# Fused compiled-NUTS kernel authoring

This page is an **ergonomic acceptance test**. It asks the real question behind the
fused No-U-Turn sampler: *how easy is it to author an efficient, allocation-free,
reusable mathematical kernel in `ReactiveKernels`?* It shows the **concise author
syntax we intend** beside the **compiler expansion that actually runs**, measures
the author-effort gap, and labels every remaining author-facing workaround as an
**open package defect** — tracked in
[`ReactiveKernels:syntax` todo `152a6td`](https://claude.ai/code). The fused NUTS
leaf is the canonical usability gate.

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

## The usability gate: authoring the fused leaf three ways

Here is the same leaf authored three ways. The middle column — the **current
low-level reality** — is the one that runs today and is build-executed further down.
The goal of `152a6td` is to make the third column produce the middle column's kernel.

### 1. Handwritten mutable Julia — fast, but inline and non-reusable

```julia
# Allocation-free by hand: every buffer, the schedule, and the mutation order are
# hand-managed and hard-wired to this one leaf. Not reusable, not composable.
function leapfrog_leaf!(new_pos, new_mom, new_vel, gradient, half,
                        pos, mom, old_grad, chol, stepsize, init_ham,
                        threshold, potential_gradient!, logdet_chol)
    @. half = mom - 0.5 * stepsize * old_grad
    copyto!(new_vel, half); ldiv!(chol, new_vel)
    @. new_pos = pos + stepsize * new_vel
    value = potential_gradient!(gradient, new_pos)          # the one gradient
    @. new_mom = half - 0.5 * stepsize * gradient
    copyto!(new_vel, new_mom); ldiv!(chol, new_vel)
    kinetic = (logdet_chol + dot(new_mom, new_vel)) / 2
    new_ham = value + kinetic
    dham = init_ham - new_ham
    (; new_pos, new_mom, new_vel, gradient, new_ham,
       dham, diverged = !(dham >= threshold))
end
```

### 2. Current low-level ReactiveKernels — reusable and allocation-free, but heavy to author

This is what an author must write **today** to get a reusable, allocation-free
kernel: build the graph explicitly, split changed from persistent inputs by hand,
hand-write a per-recipe cache policy, and drop to the internal
`_prepare_nonallocating` because the public `prepare_nonallocating` reallocates. It
is build-executed in the panel below. **Every hand-managed concept in it is an open
defect** (see the table that follows).

### 3. Intended public syntax — the acceptance target (design owned by `152a6td`)

```julia
# TARGET (illustrative — design owned by ReactiveKernels:syntax 152a6td):
# mathematics + explicit mutation intent; the compiler derives the changed/persistent
# split, owned/borrowed storage, per-recipe cache policy, and alias/effect metadata.
@kernel leapfrog_leaf(pos, mom, old_grad;                    # per-call inputs
                      chol, stepsize, init_ham, threshold,
                      potential_gradient!, logdet_chol) = begin   # persistent inputs
    half     = mom - 0.5 * stepsize * old_grad
    new_pos  = pos + stepsize * (chol \ half)
    bundle   = potential_gradient!(new_pos)                  # owned value+gradient bundle
    new_mom  = half - 0.5 * stepsize * gradient(bundle)
    kinetic  = (logdet_chol + dot(new_mom, chol \ new_mom)) / 2
    new_ham  = value(bundle) + kinetic
    dham     = init_ham - new_ham
    diverged = !(dham >= threshold)
end
```

The function-shaped `@kernel` surface (mathematics-as-recipes, have→want planning)
is real today; what is **not** yet automatic is deriving the *allocation-free
owned-slot schedule* from it without the column-2 plumbing. Closing that gap is
`152a6td`.

### Author-effort comparison

| Approach | Author writes | Reusable / composable | Allocation-free | Author must manage |
|---|---|---|---|---|
| 1. Handwritten mutable Julia | ~14 lines | ❌ inline, one-off | ✅ by hand | every buffer, schedule, mutation order |
| 2. Current low-level RK *(build-executed below)* | ~40 lines + a hand-written cache policy | ✅ as a kernel | ✅ behind a hand-written concrete barrier | have/want, changed vs persistent, cache policy, `Union` barrier, borrowed outputs, alias-write order |
| 3. Intended `@kernel` *(target `152a6td`)* | ~12 lines | ✅ | ✅ compiler-derived | mathematics + mutation intent only |

## Open author-facing defects (tracked in `152a6td`)

Each row is a workaround the author is currently forced into by column 2. None of
these is *intended* authoring — each is an open product defect, not normal usage.

| Author-facing workaround (today) | Should be | Owner |
|---|---|---|
| Manual `have`/`want`; hand-split changed vs persistent inputs | inferred from usage | `152a6td` |
| Hand-written `cache_apply`; internal `_prepare_nonallocating` because public `prepare_nonallocating` **reallocates** the owned bundles | compiler-derived per-recipe cache policy | `152a6td` |
| `Ref{Union{Nothing,T}}` + a `val::T` boxing barrier | typed seed-once `Ref{T}` at construction | `152a6td` |
| Borrowed outputs copied by hand; owned-slot aliasing wired manually | compiler owned/borrowed inference + safe aliasing | `152a6td` / `poc` |
| Alias-write ordering guaranteed by hand (owned bundle mutates before its projection reads) | compiler alias/effect metadata + freshness validator | `152a6td` / `poc` |
| `logdet(chol)` hoisted out of the leaf by hand | persistent-partition inference | `152a6td` |
| Shape change (dimension, `Matrix` vs `Diagonal`) ⇒ rebuild the kernel by hand | covered shape-change rules | `152a6td` |
| Composing units re-declares shared bindings/cache policy | composition reuses bindings without duplication | `152a6td` |

The low-level `compile_update(...; changed, persistent, want, input_binding,
output_binding, cache_policy)` interface is **planned** as an expert/compiler escape
hatch — not yet built, and explicitly *not* the default author experience.

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

## The production path — PLANNED

Turning the benchmark prototype into the public sampler requires the owned-slot
**`compile_update`** primitive and array/alias-effect contract from the core
(`ReactiveKernels:poc`'s domain): a typed seed-once `Ref{T}` cache, aliasing owned
outputs (side-effect the endpoint arrays into aliased owned slots, return only isbits
scalars), and array/alias-effect metadata so alias-writes participate in the freshness
validator. The production sampler must then compile and compose Units B and D through
the same path and **repeat every gate above** before it replaces the reactive hot leaf.

This page stays current under a build-executed drift gate: every generated pane, DAG,
and inventory is regenerated from source at build time and fails the build on drift
(tracked in [`ReactiveKernels:docs` todo `10qxx92`](https://claude.ai/code)).

## See also

- [NUTS sampling](nuts.md) — the shipped compiled-reactive sampler (the slower path
  this design replaces in the hot leaf).
- [Non-allocating kernels](nonallocating.md) — the general non-allocating preparation
  surface the fused leaf specializes.
- [Visualization](visualization.md) — the interactive Compute-DAG renderer used above.
