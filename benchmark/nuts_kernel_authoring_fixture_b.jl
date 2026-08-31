module NUTSBMutationAuthoringFixture

# Parallel mutation-profile-B translation.  The locked fixture remains
# byte-identical in nuts_kernel_authoring_fixture.jl.  This variant keeps bare
# calls only across captured RK kernel/method boundaries, makes stdlib writes
# explicit on their destination LHS, and spells identity-preserving structural
# owner transfer as `destination .= source`. Ordinary `=` retains value/as-if
# semantics; a backend may reuse storage only when that is unobservable. The
# typed compiler lowers structured `.=` through the existing structural-copy
# backend while retaining ordinary array broadcast semantics.

# ReactiveHMC-STRUCTURE — mutation-profile-B parallel translation `@kernel` NUTS AUTHORING FIXTURE — FINAL executable-integration surface.
# implicit-field, no-Ref (two-direct-branch direction), runtime rng, RK-visible leapfrog!/refresh_momentum!!/
# structural copy, `!!` public entry, pot_f + grad_f as alternative pot producers (pot_f RESTORED), no author effect
# declarations, and a concrete registered zero-allocation stats callback over compiler-owned diagnostics state.
#
# Algorithm-STRUCTURE reference (NOT a bitwise target): ReactiveHMC.jl v0.1.0 (781sB @ ca9ea4ca) —
# phasepoints.jl / integrators.jl / nuts.jl / adaptation.jl. Semantic fidelity, not token/bitwise.
#
# LOCKED FORMS + integration corrections honored:
#  A) leapfrog! is an RK-authored FREE @kernel with visible ordered effects; step_f=partial(leapfrog!;ε).
#  A2) refresh_momentum!! is an RK-authored FREE @kernel whose source exposes explicit value writes
#      (`mom = randn!(...)`; `mom = chol.L * mom`). It is invoked by public nuts!! on the owned init
#      phasepoint BEFORE step!; the mom write invalidates
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
#  RESET/COPY use the RK-CORE registered structural strong-update `structural copy(dest, src)` (result===dest).
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
# Random.randn! is the one source-visible built-in effect in refresh_momentum!!: ordered RNG, rng arg 1,
# writes/result-aliases destination arg 2. The mathematical product stays visible as `chol.L * mom`; the
# compiler may bufferize it through registered LinearAlgebra.lmul! only after proving ownership/liveness,
# alias safety, shape compatibility, and the sanctioned matrix domain. Each accept/reject branch below
# directly exposes the exact built-in ordered Random.randexp(rng) authority.

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
    phasepoint.mom = Random.randn!(rng, phasepoint.mom)
    phasepoint.mom = phasepoint.chol_metric.L * phasepoint.mom
    return phasepoint
end

# ---- reset/proposal restore via the RK-CORE registered structural strong-update `structural copy(dest, src)`
# (result === dest): copies the COMPLETE OWNED authoritative closure from src into dest's EXISTING buffers,
# preserves destination object/buffer identity, transfers source currentness, leaves SHARED authority slots
# UNTOUCHED, collapses aliased
# projections to ONE physical copy, rejects incompatible shape/type/shared-authority identity.
#   SHARED-BY-IDENTITY (untouched by structural copy): pot_f, grad_f, metric, chol_metric + @node(logdet(chol_metric)). A
#     metric mutation updates the ONE shared authority + its chol/@node closure EXACTLY once.
#   OWNED/DISTINCT (what structural copy moves): pos, mom + pot, dpot_dpos, dkin_dmom, kin, ham, dham_dpos, dham_dmom
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
# direction (concrete endpoint `ep` threaded); runtime rng; visible reset!/structural copy strong-updates; owned
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
        fwd .= init                       # identity-preserving structured copy (visible)
        bwd .= init
        # FAITHFUL RESET: source liveness establishes that the trajectory OVERWRITES every reached tree buffer
        # and every reached proposal before any read; the committed stale-poison D1–D5 battery verifies the
        # censused paths in test_kernel_nuts.jl (the eager-vs-minimal perf A/B is measured EXTERNALLY, not
        # asserted here). Therefore clearing
        # all trees + copying all proposals each transition is dead, O(max_depth) work. Seed ONLY the live-on-
        # entry slots: fwd/bwd (start endpoints) and proposals[1]/proposals[end] (the sample fallbacks read when
        # the sampler takes few/zero steps). trees[1].log_weight is seeded by step! before its first read.
        proposals[1] .= init
        return proposals[length(proposals)] .= init
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
        return init .= proposals[end]                       # mutate identity + return changed authority
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
        proposals[1] .= ep
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


end # module NUTSBMutationAuthoringFixture
