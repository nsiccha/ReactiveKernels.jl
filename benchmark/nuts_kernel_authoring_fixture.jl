# ReactiveHMC-STRUCTURE `@kernel` NUTS AUTHORING FIXTURE — implicit-field, no-Ref (two-direct-branch
# direction), runtime rng, RK-visible leapfrog!/copy!!, `!!` public entry. Durable CONSUMER contract
# against the canonical LOCKED forms A/B/C + the 22:46 source-contract corrections.
#
# Algorithm-STRUCTURE reference (NOT a bitwise target): ReactiveHMC.jl v0.1.0 (781sB @ ca9ea4ca) —
# phasepoints.jl / integrators.jl / nuts.jl / adaptation.jl. Semantic fidelity, not token/bitwise.
#
# LOCKED FORMS + 22:46 CORRECTIONS honored:
#  A) leapfrog! is an RK-authored FREE @kernel with visible ordered effects; step_f=partial(leapfrog!;ε).
#  B) implicit-field methods; NO Ref/fwdbwd/current-view; explicit FIXED physical owned init/fwd/bwd.
#     Direction is expressed as TWO DIRECT PHYSICAL-ENDPOINT BRANCH CALLS with a CONCRETE endpoint actual
#     (`gofwd ? op!(__self__, fwd, …) : op!(__self__, bwd, …)`) — the conditional NEVER yields an endpoint
#     value/alias/local; the concrete endpoint threads through the recursion as a plain method formal `ep`.
#  C) public `@kernel nuts!!(state; rng)` mutates compiler-owned concrete state + `return state`
#     (result===state, fixed shape/type, 0-B, no RefValue).
#  RNG is a TYPED RUNTIME arg, NOT sampler state: `rng` is removed from nuts_state sources; `step!(rng;…)`
#  threads it through every RNG-using sibling/recursive call; `nuts!!` calls `step!(state, rng)`.
#  RESET/COPY use the RK-CORE registered structural strong-update `copy!!(dest, src)` (result===dest) —
#  NOT opaque restore!/rcopy! and NOT a hand-authored field list. reset! establishes authoritative owned
#  endpoint state via copy!! + INLINE visible buffer clears + control writes (no unregistered helper).
#  step_f resolves to the registered leapfrog! token; a non-nothing stats_f must likewise be a registered
#  kernel (resolver-registered-or-reject at construction). @node(logdet(chol_metric)) preserved.
#
#  PINNED STRUCTURAL-COPY OWNERSHIP POLICY (load-bearing for the one-logdet-cut-point + no-extra-gradient
#  guarantees; `deepcopy(init)` is the STRUCTURAL MARKER for it, not ordinary all-fields deepcopy):
#   - SHARED BY IDENTITY across init/fwd/bwd: read-only authority inputs pot_f, grad_f, metric, plus the
#     metric-only closure chol_metric and the @node(logdet(chol_metric)) value (compiler shares read-only
#     authority closures — exactly one slot).
#   - OWNED/DISTINCT per endpoint: the registered-integrator-written sources pos, mom, plus their
#     endpoint-dependent closures/caches pot, dpot_dpos, dkin_dmom, kin, ham, dham_dpos, dham_dmom
#     (compiler clones writable-source closures — pairwise-distinct owned buffers).
#   copy!! copies EXACTLY the owned set into existing destination buffers; it never touches the shared
#   metric authority. (Runtime slot-identity/counter gates — shared=one slot, owned pairwise-distinct,
#   metric mutation recomputes chol/logdet once, pos/mom leaf schedule contains neither — land with lowering.)
#
# STAGE: durable source-capture CONSUMER surface. CONSTRUCTION-BLOCKED on the current substrate (@node,
# implicit fields + __self__ receiver, free-kernel discrimination pending syntax's source-capture
# substrate) — required-capability signal, NOT a defect. NO execution/parity/0-B/perf claim; docs not
# sourced. Structural verification + lexical-shadowing inventory via nuts_authoring_shadowing_gate.jl.
using ReactiveKernels
using LinearAlgebra, LogExpFunctions, Random

# ---- module helpers (algorithm-structure reference) -------------------------------------------------
fillf(f::Function, value, n::Int) = [f(value) for _ in 1:n]
finiteorneginf(x) = isfinite(x) ? x : typeof(x)(-Inf)
min1exp(x) = x >= 0 ? one(x) : exp(x)
badd(args...) = Base.broadcasted(+, args...)
randbernoullilog(rng, logprob) = logprob > 0 ? true : -randexp(rng) < logprob
logswapprob(tree) = tree.log_weight[1] - tree.log_weight[2]
compute_criterion(mom, bwd_dham_dmom, fwd_dham_dmom) =
    (dot(mom, bwd_dham_dmom) > 0 && dot(mom, fwd_dham_dmom) > 0)
smooth(prev, new, new_weight) = (1 - new_weight) * prev + new_weight * new

trajectory(d::Int) = trajectory(zeros(d), zeros(d))
trajectory(bwd, fwd) = (; bwd, fwd)
mv(mom, dham_dmom) = (; mom, dham_dmom)
mv(d::Int) = mv(zeros(d), zeros(d))
tree(d::Int) = (; log_weight = fill(-Inf, 2), bwd = mv(d), bwd_fwd = mv(d), summed_mom = trajectory(d))
tree(phasepoint) = tree(length(phasepoint.pos))

# The exact registered-token binding nuts_state's step_f expects: step_f resolves to the RK-registered
# leapfrog! kernel with a bound stepsize SOURCE (form A; factory binding static/inlined, hygienic
# identity). Parsed by the structural gate; actual construction happens once the source-capture
# substrate lands.
example_step_binding(stepsize) = partial(leapfrog!; stepsize = stepsize)

# A REAL parsed construction/binding of the sampler (form A step_f token + registered-or-nothing stats_f):
# step_f resolves to the registered leapfrog! kernel with a bound stepsize source; a non-nothing stats_f
# is resolver-registered-or-reject at construction (this TEXT expression is spelling only — hygienic
# registered identity is proven later by the resolver/MethodIR gate, not by the source-call census).
example_nuts_binding(init, stepsize; stats_f = nothing) =
    nuts_state(init; step_f = partial(leapfrog!; stepsize), stats_f)

# ---- euclidean_phasepoint (methodless => stateless) — @node(logdet) PRESERVED ------------------------
@kernel euclidean_phasepoint(pot_f, grad_f, metric, pos, mom) = begin
    pot = pot_f(pos)
    pot, dpot_dpos = grad_f(pos)
    chol_metric = cholesky(metric)
    dkin_dmom = chol_metric \ mom
    kin = .5 * (@node(logdet(chol_metric)) + dot(mom, dkin_dmom))
    ham = pot + kin
    dham_dpos = dpot_dpos
    dham_dmom = dkin_dmom
end

# ---- FORM A: leapfrog! — RK-authored FREE @kernel with visible ordered effects -----------------------
@kernel leapfrog!(phasepoint; stepsize) = begin
    @. phasepoint.mom -= 0.5 * stepsize * phasepoint.dham_dpos
    @. phasepoint.pos +=       stepsize * phasepoint.dham_dmom
    @. phasepoint.mom -= 0.5 * stepsize * phasepoint.dham_dpos
end

# ---- reset/proposal restore uses the RK-CORE registered structural strong-update `copy!!(dest, src)`
# (result === dest) — NOT a hand-authored field list (author never enumerates phase-point fields; a
# field list would drift when the phase-point graph changes). Its EXPLICIT CORE REGISTRATION: copy the
# COMPLETE OWNED authoritative closure from src into dest's EXISTING buffers, preserve destination
# identity/currentness, leave SHARED authority slots UNTOUCHED, collapse aliased projections to ONE
# physical copy, and reject incompatible shape/type/shared-authority identity. The compiler generates the
# minimal alias-aware copy schedule.
#
# EXPECTED OWNERSHIP METADATA (the contract copy!!'s core registration enforces — documented, NOT
# hand-implemented here; "read-only" is relative to endpoint methods):
#   SHARED-BY-IDENTITY authority (one slot, untouched by copy!!): pot_f, grad_f, metric, plus the
#     metric-only closure chol_metric + the @node(logdet(chol_metric)) value. These remain SHARED even
#     when an OWNER-level adaptation method mutates metric — such a mutation updates the ONE shared
#     authority and its chol/@node closure EXACTLY ONCE.
#   OWNED/DISTINCT per endpoint (what copy!! moves): the integrator-written sources pos, mom + their
#     endpoint-dependent closures/caches pot, dpot_dpos, dkin_dmom, kin, ham, dham_dpos, dham_dmom
#     (aliased projections collapse to one physical copy). deepcopy(init) is the STRUCTURAL MARKER.

# ---- nuts.jl: nuts_state — FORM B: implicit-field; NO Ref; explicit init/fwd/bwd; two-direct-branch ---
# direction (concrete endpoint `ep` threaded); runtime rng; visible reset!/copy!! strong-updates.
# step_f resolves to the registered leapfrog! token (see example_step_binding). stats_f is
# registered-or-nothing: a non-nothing stats_f MUST be a registered RK kernel (not an opaque runtime
# Function) so collectstats!(__self__) has a visible registered callback identity.
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

    # reset establishes authoritative owned endpoint state — registered owned copies + INLINE visible
    # buffer clears (no opaque unregistered helper) + control writes.
    reset!() = begin
        gofwd = true
        may_sample = true
        may_continue = true
        dham = zero(init.ham)           # write dham ONLY; `diverged` is the derived recipe (recomputed)
        copy!!(fwd, init)               # registered owned-copy (visible)
        copy!!(bwd, init)
        for p in proposals; copy!!(p, init); end
        for t in trees
            Base.fill!(t.log_weight, -Inf)
            @. t.bwd.mom = 0
            @. t.bwd.dham_dmom = 0
            @. t.bwd_fwd.mom = 0
            @. t.bwd_fwd.dham_dmom = 0
            @. t.summed_mom.bwd = 0
            @. t.summed_mom.fwd = 0
        end
    end
    collectstats!() = isnothing(stats_f) || stats_f(__self__)
    logadvanceprob(depth) = trees[depth-1].log_weight[1] - trees[depth].log_weight[1]
    swapproposal!(i, j = length(proposals)) = begin
        proposals[i], proposals[j] = proposals[j], proposals[i]
    end

    step!(rng) = begin
        reset!(__self__)
        gofwd ? (@. bwd.mom *= -1) : (@. fwd.mom *= -1)        # two direct branches; concrete backward
        trees[1].log_weight[1] = 0.
        for depth in 1:max_depth
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
        collectstats!(__self__)
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

# ---- FORM C: public compiled entry — mutate compiler-owned concrete state + return the SAME object -----
@kernel nuts!!(state; rng) = begin
    step!(state, rng)             # threads runtime rng; mutates compiler-owned concrete state
    return state
end

# ---- adaptation.jl: dual_averaging_state (m/H/mu + fit!(x)) — implicit-field -------------------------
@kernel dual_averaging_state(init; target = .8, regularization_scale = .05,
                             relaxation_exponent = .75, offset = 10) = begin
    m = one(init)
    H = zero(init)
    mu = log(10) + log(init)
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

# ---- adaptation.jl: welford_var (n/mean/var + step!(x; dn)) — implicit-field --------------------------
@kernel welford_var(dim) = begin
    n = 0.
    mean = zeros(dim)
    var = zeros(dim)
    step!(x::AbstractVector; dn = 1.) = begin
        n += dn
        w = dn / n
        @. var = smooth(var, (x - smooth(mean, x, w)) * (x - mean), w)
        @. mean = smooth(mean, x, w)
    end
    step!(x::AbstractMatrix; kwargs...) = for xi in eachcol(x)
        step!(__self__, xi; kwargs...)
    end
end
