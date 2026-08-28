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
type) at **exact zero allocations**. The source below is embedded statically from
the reviewed `benchmark/nuts_kernel_authoring_fixture.jl` and guarded byte-for-byte
by `test/test_nuts_docs_fixture.jl`; it is not read or executed by the docs build. The measured
leapfrog-steps/s comparison against DynamicHMC, AdvancedHMC, and nsiccha/NUTS.jl
is recorded in the static receipt [`benchmark/receipts/nuts-g7-v1.toml`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/receipts/nuts-g7-v1.toml)
— parsed here, not re-run in CI. It is a work-normalized inner-loop receipt,
not an end-to-end sampling, adaptation, wall-time, or ESS benchmark.

Packaging matters: “public” inside the fixture comments means the entry of that
sealed external artifact, not an RK package API. The NUTS runtime, native emitter,
compiled-reactive compatibility implementation, and domain types live under
`examples/nuts_runtime/` and are loaded only through the explicit
`ReactiveKernelsNUTSExample` module. A bare `using ReactiveKernels` neither loads
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

The sealed native compiler (`examples/nuts_runtime/kernel_nuts_native.jl`,
`_build_nuts_sampler`) and the minimal-reset external authoring fixture are
**on `main` as compiler evidence**. The performance figures cited on this page
measure that acceptance artifact and come from the static receipt, not a CI perf
run or the external `CompiledNUTSState` comparison path. RK, AdvancedHMC, and
DynamicHMC used one shared DifferentiationInterface+Enzyme gradient and matched
target, mass, step size, and RNG schedule; the receipt also checks gradient/work
accounting. It does **not** measure adaptation, retained draws, ESS, or
time-to-effective-sample.

The ReactiveHMC.jl `ca9` structure is an **algorithm-structure reference only** — not
a bitwise or RNG target; improvements may change arithmetic or ordering.

## Reactant and multiple chains

Adaptive NUTS is currently a CPU sampler. Its U-turn/divergence exits, ragged tree
depth, proposal swaps, and host RNG are data dependent, so they do not trace as a
static Reactant program. The traceable alternative is the fixed-step HMC kernel in
[`examples/hmc.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/examples/hmc.jl):
momentum and the Metropolis uniform are explicit inputs, the leapfrog count is
static, and `replica` maps that scalar kernel across chains. This is a scoped
compatibility statement, not a claim that arbitrary mutable or reactive state
machines are accelerator compatible.

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

## A selected kernel in the standard compiler view

The complete sampler below is a method-bearing eight-spec artifact, not one
stateless `Plan`: its recursive tree growth, ordered RNG effects, and in-place
updates are sealed by the native method compiler. Its methodless Euclidean
phasepoint recurrence *does* have an ordinary stateless plan, so it is the honest
place to inspect NUTS work through the same three-pane view used by the
distribution and batched examples.

The panel is build-executed. **Raw input** contains the phasepoint math and the
selected HAVE/WANT boundary; **Generated kernel** is the resulting
`code_expr(kernel)`; **Compute DAG** is that exact `kernel.plan`. The potential-only
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

## The authoring source

The block below is the **exact reviewed authoring source** — the surface that `@kernel`
lowers to the sealed, registry-free native NUTS sampler now on `main`. It is reproduced
verbatim from the durable fixture
[`benchmark/nuts_kernel_authoring_fixture.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/nuts_kernel_authoring_fixture.jl);
the drift test `test/test_nuts_docs_fixture.jl` keeps this page byte-identical to that
fixture, and this page renders live at
<https://nsiccha.github.io/ReactiveKernels.jl/dev/nuts>.

The verbatim source below is the macro-free executable fixture as it exists today.
Its helpers are inline arithmetic/control or captured sibling `@kernel` methods;
there are no user-authored effect declarations. The fixture's `@node` use is
unrelated and stays.

::: details Show the complete byte-synchronized authoring fixture

```julia
# ReactiveHMC-STRUCTURE `@kernel` NUTS AUTHORING FIXTURE — FINAL executable-integration surface.
# implicit-field, no-Ref (two-direct-branch direction), runtime rng, RK-visible leapfrog!/refresh_momentum!!/
# copy!!, `!!` public entry, pot_f + grad_f as alternative pot producers (pot_f RESTORED), no author effect
# declarations, and a concrete registered zero-allocation stats callback over compiler-owned diagnostics state.
#
# Algorithm-STRUCTURE reference (NOT a bitwise target): ReactiveHMC.jl v0.1.0 (781sB @ ca9ea4ca) —
# phasepoints.jl / integrators.jl / nuts.jl / adaptation.jl. Semantic fidelity, not token/bitwise.
#
# LOCKED FORMS + integration corrections honored:
#  A) leapfrog! is an RK-authored FREE @kernel with visible ordered effects; step_f=partial(leapfrog!;ε).
#  A2) refresh_momentum!! is an RK-authored FREE @kernel (SOURCE MUTATION ONLY: Random.randn! + LinearAlgebra
#      .lmul!) invoked by public nuts!! on the owned init phasepoint BEFORE step!; the mom write invalidates
#      the kinetic/momentum closure (recomputed by the scheduler on next read), dpot stays current.
#  B) implicit-field methods; NO Ref/fwdbwd/current-view; explicit FIXED physical owned init/fwd/bwd; direction
#     is TWO DIRECT PHYSICAL-ENDPOINT BRANCH CALLS with a CONCRETE endpoint actual threaded as a plain formal.
#  C) public `@kernel nuts!!(state; rng)` refreshes momentum then mutates compiler-owned concrete state +
#     `return state` (result===state, fixed shape/type, 0-B, no RefValue).
#  GRAD: euclidean_phasepoint takes pot_f + grad_f as ALTERNATIVE producers of pot (pot_f RESTORED per user
#     directive 2026-08-27T10:18:56 — its earlier removal was unauthorized); grad_f additionally produces dpot.
#     pot_f is a shared-by-identity authority retained even when the planner selects grad_f for pot. The build
#     hook passes the in-place pgrad!(g,x)::T unchanged; the factory binds it to each endpoint's dpot_dpos + pot.
#  RNG is a TYPED RUNTIME arg, NOT sampler state; `step!(rng)` threads it; `nuts!!` calls refresh + step!(state,rng).
#  RESET/COPY use the RK-CORE registered structural strong-update `copy!!(dest, src)` (result===dest).
#  DIAGNOSTICS: nuts_state owns n_steps/reached_depth/acceptance_rate (+ existing dham/diverged); reset! zeros
#     them per transition; the registered stats_f (nuts_stats!) increments n_steps ONCE per collectstats!/leaf
#     and records acceptance data — n_steps is produced by the callback, independent of pgrad + body marker.
#  T-PRESERVING: all construction literals derive from the endpoint/template (zero/one/oftype/similar).
#  @node(logdet(chol_metric)) preserved. Every hot helper is an expression or captured sibling method whose
#  primitive reads/RNG effects are visible in MethodIR; the production NUTS source needs no `@rk_*` declaration.
#
#  PINNED STRUCTURAL-COPY OWNERSHIP POLICY (`deepcopy(init)` is the STRUCTURAL MARKER):
#   - SHARED BY IDENTITY across init/fwd/bwd: read-only authority inputs pot_f, grad_f, metric, plus the
#     metric-only closure chol_metric and the @node(logdet(chol_metric)) value (one slot). pot_f is the
#     authored alternative pot producer, retained by identity even when its recipe is unselected.
#   - OWNED/DISTINCT per endpoint: integrator-written pos, mom + endpoint closures/caches pot, dpot_dpos,
#     dkin_dmom, kin, ham, dham_dpos, dham_dmom (aliased projections collapse to one physical slot).
#
# STAGE: FINAL integration-input source surface for syntax cherry-pick + POC compile. CONSTRUCTION happens on
# the factory/effects substrate (@kernel source-capture + built-in primitive authority + nuts!! execution seam). NO
# execution/parity/0-B/perf claim here; docs not sourced. Structural verification + lexical-shadowing inventory
# via nuts_authoring_shadowing_gate.jl; executable certification via nuts_acceptance_harness.jl (c83, held).
using ReactiveKernels
using LinearAlgebra, LogExpFunctions, Random

# ---- cold module helpers (construction only) — T-derived, no Int→Float64 ------------------------------
fillf(f::Function, value, n::Int) = [f(value) for _ in 1:n]
# Built-in RNG/effect primitives used by refresh_momentum!!: Random.randn! (ordered RNG, rng arg 1, writes/
# result-aliases dest arg 2), LinearAlgebra.lmul! (reads matrix+dest, writes/aliases dest arg 2). Each
# accept/reject branch below directly exposes the exact built-in ordered Random.randexp(rng) authority.

# T-derived tree/proposal buffers from the phasepoint/template arrays (zero/similar), sentinel via oftype(ham).
trajectory(bwd, fwd) = (; bwd, fwd)
trajectory(v::AbstractVector) = trajectory(zero(v), zero(v))
mv(mom, dham_dmom) = (; mom, dham_dmom)
mv(v::AbstractVector) = mv(zero(v), zero(v))
tree(phasepoint) = (; log_weight = fill(oftype(phasepoint.ham, -Inf), 2),
                      bwd = mv(phasepoint.pos), bwd_fwd = mv(phasepoint.pos), summed_mom = trajectory(phasepoint.pos))

# step_f resolves to the RK-registered leapfrog! token with a bound stepsize source (form A). stats_f resolves
# to the registered nuts_stats! kernel (a concrete zero-allocation callback, NOT nothing).
example_step_binding(stepsize) = partial(leapfrog!; stepsize = stepsize)
example_nuts_binding(init, stepsize) =
    nuts_state(init; step_f = partial(leapfrog!; stepsize), stats_f = nuts_stats!)

# ---- euclidean_phasepoint — pot_f + grad_f are ALTERNATIVE producers of pot; @node preserved -------------
# `pot_f` is the authored alternative potential producer (restored per user directive
# 2026-08-27T10:18:56 — its earlier removal was an unauthorized simplification). It is a
# SHARED-BY-IDENTITY read-only authority, retained by identity even when the planner selects
# `grad_f` (which produces pot as a byproduct of the needed dpot); the two `pot`-producing
# recipes below are alternative producers, plan-resolved (see src/kernel_lowering.jl).
@kernel euclidean_phasepoint(pot_f, grad_f, metric, pos, mom) = begin
    pot = pot_f(pos)                             # pot_f: authored alternative producer of pot (kept by identity)
    pot, dpot_dpos = grad_f(pos)                 # grad_f also produces pot (+dpot); factory binds grad_f=pgrad!(g,x)
    chol_metric = cholesky(metric)
    dkin_dmom = chol_metric \ mom
    kin = oftype(pot, 0.5) * (@node(logdet(chol_metric)) + dot(mom, dkin_dmom))   # typed 1/2 (no Float64 promotion)
    ham = pot + kin
    dham_dpos = dpot_dpos
    dham_dmom = dkin_dmom
end

# ---- FORM A: leapfrog! — RK-authored FREE @kernel with visible ordered effects -----------------------
@kernel leapfrog!(phasepoint; stepsize) = begin
    @. phasepoint.mom -= oftype(stepsize, 0.5) * stepsize * phasepoint.dham_dpos   # typed half-step
    @. phasepoint.pos +=                       stepsize * phasepoint.dham_dmom
    @. phasepoint.mom -= oftype(stepsize, 0.5) * stepsize * phasepoint.dham_dpos
end

# ---- FORM A2: refresh_momentum!! — RK-authored FREE @kernel; SOURCE MUTATION ONLY ---------------------
# mom ~ N(0, metric) = L·z (L = chol_metric.L). The two writes are the whole kernel; dkin_dmom/dham_dmom/kin/
# ham are INVALIDATED by the mom write and recomputed by the scheduler on next read; dpot_dpos/dham_dpos stay
# current. No author cache writes, no @node reference (shared chol/@node stays current, metric unchanged).
@kernel refresh_momentum!!(phasepoint; rng) = begin
    Random.randn!(rng, phasepoint.mom)
    LinearAlgebra.lmul!(phasepoint.chol_metric.L, phasepoint.mom)
    return phasepoint
end

# ---- reset/proposal restore via the RK-CORE registered structural strong-update `copy!!(dest, src)`
# (result === dest): copies the COMPLETE OWNED authoritative closure from src into dest's EXISTING buffers,
# preserves destination object/buffer identity, transfers source currentness, leaves SHARED authority slots
# UNTOUCHED, collapses aliased
# projections to ONE physical copy, rejects incompatible shape/type/shared-authority identity.
#   SHARED-BY-IDENTITY (untouched by copy!!): pot_f, grad_f, metric, chol_metric + @node(logdet(chol_metric)). A
#     metric mutation updates the ONE shared authority + its chol/@node closure EXACTLY once.
#   OWNED/DISTINCT (what copy!! moves): pos, mom + pot, dpot_dpos, dkin_dmom, kin, ham, dham_dpos, dham_dmom
#     (aliased projections collapse to one physical copy). deepcopy(init) is the STRUCTURAL MARKER.

# ---- adaptation.jl: registered zero-allocation stats callback over compiler-owned diagnostics state --------
# collectstats!(__self__) calls stats_f(__self__) ONCE per leaf (from start! at depth==1). nuts_stats!
# increments the owned n_steps (independent of pgrad + leapfrog body marker) and records running acceptance.
@kernel nuts_stats!(state) = begin
    state.n_steps += 1
    state.acceptance_rate = (one(state.dham) - one(state.dham) / state.n_steps) * state.acceptance_rate +
                            (one(state.dham) / state.n_steps) *
                            (state.dham >= zero(state.dham) ? one(state.dham) : exp(state.dham))
    return state
end

# ---- nuts.jl: nuts_state — FORM B: implicit-field; NO Ref; explicit init/fwd/bwd; two-direct-branch ---
# direction (concrete endpoint `ep` threaded); runtime rng; visible reset!/copy!! strong-updates; owned
# diagnostics (n_steps/reached_depth/acceptance_rate) updated by the registered stats_f.
@kernel nuts_state(init; step_f, max_depth = 10, min_dham = oftype(init.ham, -1000), stats_f = nothing) = begin
    gofwd = true
    may_sample = true
    may_continue = true
    fwd = deepcopy(init)                 # explicit FIXED physical owned endpoints (structural copies)
    bwd = deepcopy(init)
    trees = fillf(tree, init, max_depth + 1)
    proposals = fillf(deepcopy, init, max_depth + 2)
    dham = zero(init.ham)
    diverged = !(dham >= min_dham)        # authored ONCE as the derived recipe (never written imperatively)
    n_steps = 0                           # owned diagnostics (compiler-owned storage), reset per transition
    reached_depth = 0
    acceptance_rate = zero(init.ham)

    # The remaining hot helper is a captured sibling method whose complete primitive body is part of
    # MethodIR/native ProgramT.  RNG draws stay directly at their consuming branches, and the explicit
    # reductions in finish! replace the former opaque `dot` + lazy `broadcasted(+)` helpers.  They preserve
    # the real-valued NUTS criterion, while—as the fixture header states—the scalar reduction is not a
    # bitwise BLAS-dot target.
    finiteorneginf(x) = begin
        result = (x - x == zero(x)) ? x : -(one(x) / zero(x))
        result
    end
    reset!() = begin
        gofwd = true
        may_sample = true
        may_continue = true
        dham = zero(init.ham)             # write dham ONLY; `diverged` is the derived recipe (recomputed)
        n_steps = 0
        reached_depth = 0
        acceptance_rate = zero(init.ham)
        copy!!(fwd, init)                 # registered owned-copy (visible)
        copy!!(bwd, init)
        # FAITHFUL RESET: source liveness establishes that the trajectory OVERWRITES every reached tree buffer
        # and every reached proposal before any read; the committed stale-poison D1–D5 battery verifies the
        # censused paths in test_kernel_nuts.jl (the eager-vs-minimal perf A/B is measured EXTERNALLY, not
        # asserted here). Therefore clearing
        # all trees + copying all proposals each transition is dead, O(max_depth) work. Seed ONLY the live-on-
        # entry slots: fwd/bwd (start endpoints) and proposals[1]/proposals[end] (the sample fallbacks read when
        # the sampler takes few/zero steps). trees[1].log_weight is seeded by step! before its first read.
        copy!!(proposals[1], init)
        copy!!(proposals[length(proposals)], init)
    end
    collectstats!() = isnothing(stats_f) || stats_f(__self__)
    logadvanceprob(depth) = trees[depth-1].log_weight[1] - trees[depth].log_weight[1]
    swapproposal!(i, j = length(proposals)) = begin
        proposals[i], proposals[j] = proposals[j], proposals[i]
    end

    step!(rng) = begin
        reset!(__self__)
        gofwd ? (@. bwd.mom *= -1) : (@. fwd.mom *= -1)        # two direct branches; concrete backward
        trees[1].log_weight[1] = zero(init.ham)
        for depth in 1:max_depth
            reached_depth = depth
            rand(rng, Bool) && flip!(__self__, depth)
            gofwd ? finish!(__self__, fwd, depth, rng) : finish!(__self__, bwd, depth, rng)
            may_sample || break
            ((trees[depth].log_weight[1] - trees[depth].log_weight[2]) >
                zero(trees[depth].log_weight[1] - trees[depth].log_weight[2]) ? true :
                -Random.randexp(rng) < (trees[depth].log_weight[1] - trees[depth].log_weight[2])) &&
                swapproposal!(__self__, depth)
            may_continue || break
        end
        copy!!(init, proposals[end])                          # registered owned-copy (visible)
    end
    flip!(depth) = if depth > 1
        gofwd = !gofwd
        gofwd ? flip_neg!(__self__, bwd, depth) : flip_neg!(__self__, fwd, depth)   # concrete backward
    end
    flip_neg!(ep, depth) = begin
        tree = trees[depth]
        @. tree.bwd.mom = -ep.mom
        @. tree.bwd.dham_dmom = -ep.dham_dmom
        @. tree.summed_mom.fwd *= -1
    end
    finish!(ep, depth, rng) = begin
        tree = trees[depth]
        suptree = trees[depth+1]
        tree.log_weight[2] = tree.log_weight[1]
        if depth == 1
            @. suptree.bwd.mom = ep.mom                        # tree-data copies as visible broadcasts
            @. suptree.bwd.dham_dmom = ep.dham_dmom
        else
            @. suptree.bwd.mom = tree.bwd.mom
            @. suptree.bwd.dham_dmom = tree.bwd.dham_dmom
            @. tree.bwd_fwd.mom = ep.mom
            @. tree.bwd_fwd.dham_dmom = ep.dham_dmom
            @. tree.summed_mom.bwd = tree.summed_mom.fwd
        end
        start!(__self__, ep, depth, rng)
        may_continue || return may_sample = false
        suptree.log_weight[1] = logaddexp(tree.log_weight[1], tree.log_weight[2])
        if depth == 1
            @. suptree.summed_mom.fwd = suptree.bwd.mom + ep.mom

            # Seed from the already-typed scalar diagnostic, not an array element: zero-length vector
            # inputs retain Julia's ordinary empty-reduction value and never acquire an implicit [1] read.
            backward_dot = zero(dham)
            forward_dot = zero(dham)
            for i in 1:length(suptree.summed_mom.fwd)
                backward_dot += suptree.summed_mom.fwd[i] * suptree.bwd.dham_dmom[i]
                forward_dot += suptree.summed_mom.fwd[i] * ep.dham_dmom[i]
            end
            may_continue = backward_dot > zero(backward_dot) && forward_dot > zero(forward_dot)
        else
            @. suptree.summed_mom.fwd = tree.summed_mom.bwd + tree.summed_mom.fwd
            base_backward_dot = zero(dham)
            base_forward_dot = zero(dham)
            sum1_backward_dot = zero(dham)
            sum1_forward_dot = zero(dham)
            sum2_backward_dot = zero(dham)
            sum2_forward_dot = zero(dham)
            for i in 1:length(suptree.summed_mom.fwd)
                base_backward_dot += suptree.summed_mom.fwd[i] * suptree.bwd.dham_dmom[i]
                base_forward_dot += suptree.summed_mom.fwd[i] * ep.dham_dmom[i]
                sum1_backward_dot += (tree.summed_mom.bwd[i] + tree.bwd.mom[i]) * suptree.bwd.dham_dmom[i]
                sum1_forward_dot += (tree.summed_mom.bwd[i] + tree.bwd.mom[i]) * tree.bwd.dham_dmom[i]
                sum2_backward_dot += (tree.bwd_fwd.mom[i] + tree.summed_mom.fwd[i]) * tree.bwd_fwd.dham_dmom[i]
                sum2_forward_dot += (tree.bwd_fwd.mom[i] + tree.summed_mom.fwd[i]) * ep.dham_dmom[i]
            end
            may_continue = (
                base_backward_dot > zero(base_backward_dot) &&
                base_forward_dot > zero(base_forward_dot) &&
                sum1_backward_dot > zero(sum1_backward_dot) &&
                sum1_forward_dot > zero(sum1_forward_dot) &&
                sum2_backward_dot > zero(sum2_backward_dot) &&
                sum2_forward_dot > zero(sum2_forward_dot)
            )
        end
    end
    start!(ep, depth, rng) = if depth == 1
        step_f(ep)                                             # registered leapfrog! token, concrete ep
        raw_dham = init.ham - ep.ham
        dham = finiteorneginf(__self__, raw_dham)
        collectstats!(__self__)                                # registered stats_f increments n_steps (per leaf)
        diverged && return may_continue = false
        trees[1].log_weight[1] = dham
        copy!!(proposals[1], ep)
    else
        start!(__self__, ep, depth - 1, rng)
        may_continue || return may_sample = false
        swapproposal!(__self__, depth - 1, depth)
        finish!(__self__, ep, depth - 1, rng)
        if may_sample
            if (trees[depth - 1].log_weight[1] - trees[depth].log_weight[1]) >
                    zero(trees[depth - 1].log_weight[1] - trees[depth].log_weight[1]) ? true :
                    -Random.randexp(rng) < (trees[depth - 1].log_weight[1] - trees[depth].log_weight[1])
                swapproposal!(__self__, depth - 1, depth)
            end
        end
    end
end

# ---- FORM C: public compiled entry — refresh momentum on owned init, mutate state, return SAME object -----
@kernel nuts!!(state; rng) = begin
    refresh_momentum!!(state.init; rng)   # source momentum refresh on the owned init phasepoint, BEFORE step!
    step!(state, rng)                     # threads runtime rng; mutates compiler-owned concrete state
    return state
end

# ---- adaptation.jl: dual_averaging_state (m/H/mu + fit!(x)) — implicit-field; source-typed defaults --------
@kernel dual_averaging_state(init; target = oftype(init, .8), regularization_scale = oftype(init, .05),
                             relaxation_exponent = oftype(init, .75), offset = oftype(init, 10)) = begin
    m = one(init)
    H = zero(init)
    mu = log(oftype(init, 10)) + log(init)
    log_current = mu - sqrt(m) / regularization_scale * H
    log_final = zero(init)
    current = exp(log_current)
    final = exp(log_final)
    fit!(x) = begin
        m += 1
        H += (target - x - H) / (m + offset)
        log_final += m^(-relaxation_exponent) * (log_current - log_final)
    end
end

# ---- adaptation.jl: welford_var (n/mean/var + step!(x; dn)) — template vector, T-derived -------------------
@kernel welford_var(template::AbstractVector) = begin
    n = zero(eltype(template))
    mean = zero(template)
    var = zero(template)
    step!(x::AbstractVector; dn = one(n)) = begin
        n += dn
        w = dn / n
        # Keep the adaptation recurrence self-contained in the sanctioned Base arithmetic surface.  In
        # particular this method must not depend on an author-declared effect helper merely to tell the
        # compiler the helper's arity/result domain: both smoothing operations are the ordinary affine
        # formula, visible in the captured MethodIR.
        @. var = (one(w) - w) * var + w * (x - ((one(w) - w) * mean + w * x)) * (x - mean)
        @. mean = (one(w) - w) * mean + w * x
    end
    step!(x::AbstractMatrix; kwargs...) = for xi in eachcol(x)
        step!(__self__, xi; kwargs...)
    end
end
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
