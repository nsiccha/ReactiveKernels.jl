# NUTS sampling

ReactiveKernels' No-U-Turn sampler is authored as eight method-bearing `@kernel`
specifications modeled on ReactiveHMC.jl's algorithm structure. The public
sampler is compiled, executable, and allocation-free in steady state. This page
separates that implementation claim from the narrower performance evidence and
keeps the complete authoring source available without making it the main reading
flow.

The public `nuts!!` sampler is **landed and executable on `main`**: `@kernel` lowers the
NUTS source to a sealed, registry-free **native compiled recursion** (`compile_nuts_native`
/ `_build_nuts_sampler`), and the public `nuts!!(state; rng)` mutates compiler-owned state
in place and returns the **same object** (`result === state`, same concrete type) at
**exact zero allocations**. The source below is copied from the reviewed fixture
and guarded byte-for-byte by `test/test_nuts_docs_fixture.jl`. The checked-in G7
receipt measures work-normalized leapfrog throughput; it is not an end-to-end
sampling or ESS benchmark.

## Status — read this before the code

| Piece | State |
|---|---|
| Source contract (the eight `@kernel` specs below, the seven `@rk_*` effect registrations, the plan shape) | **Executable current `main`; correction pending.** The fixture currently has one combined `grad_f` producer. The previously removed `pot_f` alternative producer is to be restored with compiler/ownership evidence; until that lands, the source below mirrors current `main` exactly. |
| All eight source specs construct; concrete phasepoint/frame init/recompute/copy verified | **Landed on `main`** — the compiler constructs and runs the whole sampler; sealed production certificate `mode = production`. |
| Executable leapfrog (leaf scope) | **Verified** — analytic F32/F64; normal gradient Δ1, `@inferred`, exact 0-B; dirty-produced recovery analytic; dirty-source reject. |
| Public `nuts!!` sampler (`step!`, tree growth, U-turn) | **Landed on `main`** — sealed registry-free native recursion; `nuts!!(state; rng) === state` (same object, fixed type), **exact 0-B** on the public path. |
| End-to-end sampling time and ESS | **Not measured for the current sealed-native path.** The earlier compiled-reactive implementation was about 4–7× slower than AdvancedHMC/DynamicHMC in matched warmup+draw wall time, so the inner-loop result must not be read as a blanket sampler-speed claim. |
| Work-normalized inner-loop throughput | **Measured, narrow metric** — [`nuts-g7-v1.toml`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/receipts/nuts-g7-v1.toml) records 2.27M leapfrog steps/s for RK, 1.33M for AdvancedHMC, 1.42M for DynamicHMC, and 2.65M for nsiccha/NUTS.jl on the frozen AR(1) setup. |

The sealed native compiler (`kernel_nuts_native.jl`, `_build_nuts_sampler`) and the
minimal-reset authoring fixture are **on `main`**; the public `nuts!!` runs there. The
performance figures cited on this page come from the static receipt, not a CI perf run.
RK, AdvancedHMC, and DynamicHMC used one shared DifferentiationInterface+Enzyme
gradient and matched target, mass, step size, and RNG schedule; the receipt also
checks the gradient/work accounting. It does **not** measure adaptation, retained
draws, ESS, or time-to-effective-sample.

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

- **`euclidean_phasepoint(grad_f, metric, pos, mom)`** — on current `main`, a phasepoint is
  potential + kinetic energy at `(pos, mom)`. You write the four lines of physics
  directly: one destination-bound gradient recipe produces `pot, dpot_dpos = grad_f(pos)`
  (the pending source-contract correction will restore `pot_f` as an unselected
  alternative producer), `chol_metric = cholesky(metric)`, the
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

The block below is the **exact reviewed authoring source** — the surface that `@kernel`
lowers to the sealed, registry-free native NUTS sampler now on `main`. It is reproduced
verbatim from the durable fixture
[`benchmark/nuts_kernel_authoring_fixture.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/nuts_kernel_authoring_fixture.jl);
the drift test `test/test_nuts_docs_fixture.jl` keeps this page byte-identical to that
fixture, and this page renders live at
<https://nsiccha.github.io/ReactiveKernels.jl/dev/nuts>.

The fixture's comment preamble preserves its integration-stage provenance, so its
“docs not sourced” staging line is historical rather than the page's current
status. The table above is authoritative.

::: details Show the complete byte-synchronized authoring fixture

```julia
# ReactiveHMC-STRUCTURE `@kernel` NUTS AUTHORING FIXTURE — FINAL executable-integration surface.
# implicit-field, no-Ref (two-direct-branch direction), runtime rng, RK-visible leapfrog!/refresh_momentum!!/
# copy!!, `!!` public entry, one destination-bound grad recipe (NO pot_f), public @rk_* helper effect
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
#  GRAD: euclidean_phasepoint takes ONLY grad_f — one destination-bound selected grad recipe produces pot+dpot
#     (no redundant pot_f producer, no required-but-unused pot_f authority). The build hook passes the in-place
#     pgrad!(g,x)::T unchanged; the factory binds it to each endpoint's owned dpot_dpos slot + pot scalar.
#  RNG is a TYPED RUNTIME arg, NOT sampler state; `step!(rng)` threads it; `nuts!!` calls refresh + step!(state,rng).
#  RESET/COPY use the RK-CORE registered structural strong-update `copy!!(dest, src)` (result===dest).
#  DIAGNOSTICS: nuts_state owns n_steps/reached_depth/acceptance_rate (+ existing dham/diverged); reset! zeros
#     them per transition; the registered stats_f (nuts_stats!) increments n_steps ONCE per collectstats!/leaf
#     and records acceptance data — n_steps is produced by the callback, independent of pgrad + body marker.
#  T-PRESERVING: all construction literals derive from the endpoint/template (zero/one/oftype/similar).
#  @node(logdet(chol_metric)) preserved. Public @rk_* declarations register the pure/borrowing/rng helpers so
#  the compiler schedules them with visible effects (never body inference).
#
#  PINNED STRUCTURAL-COPY OWNERSHIP POLICY (`deepcopy(init)` is the STRUCTURAL MARKER):
#   - SHARED BY IDENTITY across init/fwd/bwd: read-only authority inputs grad_f, metric, plus the metric-only
#     closure chol_metric and the @node(logdet(chol_metric)) value (one slot). NO pot_f.
#   - OWNED/DISTINCT per endpoint: integrator-written pos, mom + endpoint closures/caches pot, dpot_dpos,
#     dkin_dmom, kin, ham, dham_dpos, dham_dmom (aliased projections collapse to one physical slot).
#
# STAGE: FINAL integration-input source surface for syntax cherry-pick + POC compile. CONSTRUCTION happens on
# the factory/effects substrate (@kernel source-capture + @rk_* declarations + nuts!! execution seam). NO
# execution/parity/0-B/perf claim here; docs not sourced. Structural verification + lexical-shadowing inventory
# via nuts_authoring_shadowing_gate.jl; executable certification via nuts_acceptance_harness.jl (c83, held).
using ReactiveKernels
using LinearAlgebra, LogExpFunctions, Random

# ---- module helpers (algorithm-structure reference) — T-derived, no Int→Float64 -----------------------
fillf(f::Function, value, n::Int) = [f(value) for _ in 1:n]
finiteorneginf(x) = isfinite(x) ? x : typeof(x)(-Inf)
min1exp(x) = x >= 0 ? one(x) : exp(x)
badd(args...) = Base.broadcasted(+, args...)
randbernoullilog(rng, logprob) = logprob > 0 ? true : -randexp(rng) < logprob
logswapprob(tree) = tree.log_weight[1] - tree.log_weight[2]
compute_criterion(mom, bwd_dham_dmom, fwd_dham_dmom) =
    (dot(mom, bwd_dham_dmom) > 0 && dot(mom, fwd_dham_dmom) > 0)

# PUBLIC exact-identity effect declarations (973f7f4/bf7d2ed) — the compiler schedules these with visible
# effects (registered primitives), never by body inference. Authors touch no internals.
@rk_pure finiteorneginf 1
@rk_pure min1exp 1
@rk_borrows badd 2
@rk_rng randbernoullilog 2 1
@rk_pure logswapprob 1
@rk_pure compute_criterion 3
# Built-in RNG/effect primitives used by refresh_momentum!!: Random.randn! (ordered RNG, rng arg 1, writes/
# result-aliases dest arg 2), LinearAlgebra.lmul! (reads matrix+dest, writes/aliases dest arg 2).

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

# ---- euclidean_phasepoint — ONE destination-bound grad recipe (pot+dpot); pot_f DROPPED; @node preserved ----
@kernel euclidean_phasepoint(grad_f, metric, pos, mom) = begin
    pot, dpot_dpos = grad_f(pos)                 # single selected grad recipe; factory binds grad_f=pgrad!(g,x)
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
#   SHARED-BY-IDENTITY (untouched by copy!!): grad_f, metric, chol_metric + @node(logdet(chol_metric)). A
#     metric mutation updates the ONE shared authority + its chol/@node closure EXACTLY once.
#   OWNED/DISTINCT (what copy!! moves): pos, mom + pot, dpot_dpos, dkin_dmom, kin, ham, dham_dpos, dham_dmom
#     (aliased projections collapse to one physical copy). deepcopy(init) is the STRUCTURAL MARKER.

# ---- adaptation.jl: registered zero-allocation stats callback over compiler-owned diagnostics state --------
# collectstats!(__self__) calls stats_f(__self__) ONCE per leaf (from start! at depth==1). nuts_stats!
# increments the owned n_steps (independent of pgrad + leapfrog body marker) and records running acceptance.
@kernel nuts_stats!(state) = begin
    state.n_steps += 1
    state.acceptance_rate = (one(state.dham) - one(state.dham) / state.n_steps) * state.acceptance_rate +
                            (one(state.dham) / state.n_steps) * min1exp(state.dham)
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
            randbernoullilog(rng, logswapprob(trees[depth])) && swapproposal!(__self__, depth)
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
        may_continue = if depth == 1
            @. suptree.summed_mom.fwd = suptree.bwd.mom + ep.mom
            compute_criterion(suptree.summed_mom.fwd, suptree.bwd.dham_dmom, ep.dham_dmom)
        else
            @. suptree.summed_mom.fwd = tree.summed_mom.bwd + tree.summed_mom.fwd
            (
                compute_criterion(suptree.summed_mom.fwd, suptree.bwd.dham_dmom, ep.dham_dmom) &&
                compute_criterion(badd(tree.summed_mom.bwd, tree.bwd.mom),
                                  suptree.bwd.dham_dmom, tree.bwd.dham_dmom) &&
                compute_criterion(badd(tree.bwd_fwd.mom, tree.summed_mom.fwd),
                                  tree.bwd_fwd.dham_dmom, ep.dham_dmom)
            )
        end
    end
    start!(ep, depth, rng) = if depth == 1
        step_f(ep)                                             # registered leapfrog! token, concrete ep
        dham = finiteorneginf(init.ham - ep.ham)
        collectstats!(__self__)                                # registered stats_f increments n_steps (per leaf)
        diverged && return may_continue = false
        trees[1].log_weight[1] = dham
        copy!!(proposals[1], ep)
    else
        start!(__self__, ep, depth - 1, rng)
        may_continue || return may_sample = false
        swapproposal!(__self__, depth - 1, depth)
        finish!(__self__, ep, depth - 1, rng)
        if may_sample && randbernoullilog(rng, logadvanceprob(__self__, depth))
            swapproposal!(__self__, depth - 1, depth)
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
