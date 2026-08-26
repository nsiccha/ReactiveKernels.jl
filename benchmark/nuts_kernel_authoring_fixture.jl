# ReactiveHMC-STRUCTURE `@kernel` NUTS AUTHORING FIXTURE — implicit-field, no-Ref, RK-visible leapfrog,
# `!!` public entry. Durable CONSUMER contract against the canonical LOCKED forms A/B/C (lead 22:35).
#
# Algorithm-STRUCTURE reference (NOT a bitwise target): ReactiveHMC.jl v0.1.0
# (~/.julia/packages/ReactiveHMC/781sB/src, pinned main@ca9ea4ca) — phasepoints.jl / integrators.jl /
# nuts.jl / adaptation.jl. Semantic fidelity, not token fidelity: improvements may change arithmetic/order
# and must not preserve mistakes. Correctness is independent/mathematical, never bitwise/RNG agreement.
#
# DEVIATIONS from the reference source (semantic, user-ruled — named):
#   (1) sole macro: @reactive -> @kernel.
#   (2) NO Ref current-views (Ref was a source+backend mistake): explicit FIXED physical owned init/fwd/bwd
#       structural copies; every direction-dependent op branches EXPLICITLY on `gofwd` so the compiler emits
#       typed per-direction variants — never a current-view alias / Ref / fwdbwd index.
#   (3) leapfrog! is an RK-authored FREE @kernel (visible ordered effects), passed ordinarily as
#       step_f=partial(leapfrog!;stepsize=ε); it is NOT an opaque Julia call in the hot path.
#   (4) public compiled entry is `@kernel nuts!!(state; rng)` — mutates compiler-owned concrete state and
#       returns the SAME object (result === state), fixed shape/type, 0-B, no RefValue.
#   @node(logdet(chol_metric)) is PRESERVED verbatim (named assignments are nodes automatically; anonymous
#   inline subexpressions become nodes ONLY via @node — no heuristic AST extraction).
#
# STAGE: durable source-capture CONSUMER surface. Syntax is implementing the narrow source-capture
# substrate; until it lands this may be CONSTRUCTION-BLOCKED (required-capability signal, NOT a defect —
# implicit fields + __self__ receiver + free-kernel leapfrog!/nuts!! discrimination + no-Julia-IR capture).
# NO execution/parity/0-B/perf claim here. Structural verification + the non-vacuous lexical-shadowing
# inventory run via benchmark/nuts_authoring_shadowing_gate.jl (parses this file; does not eval @kernel).
using ReactiveKernels
using LinearAlgebra, LogExpFunctions, Random

# ---- nuts.jl / adaptation.jl module helpers (algorithm-structure reference) --------------------------
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
tree(d::Int) = (;
    log_weight = fill(-Inf, 2),
    bwd = mv(d),
    bwd_fwd = mv(d),
    summed_mom = trajectory(d),
)
tree(phasepoint) = tree(length(phasepoint.pos))

# ---- phasepoints.jl: euclidean_phasepoint (methodless => stateless) — @node(logdet) PRESERVED ---------
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

# ---- integrators.jl: leapfrog! — FORM A: RK-authored FREE @kernel with visible ordered effects --------
# Passed ordinarily as step_f=partial(leapfrog!;stepsize=ε); factory binding static/inlined; no author
# type ceremony; no hidden self-prepare. RK sees the ordered pos/mom writes + dham_dpos/dham_dmom reads.
@kernel leapfrog!(phasepoint; stepsize) = begin
    @. phasepoint.mom -= 0.5 * stepsize * phasepoint.dham_dpos
    @. phasepoint.pos +=       stepsize * phasepoint.dham_dmom
    @. phasepoint.mom -= 0.5 * stepsize * phasepoint.dham_dpos
end

# ---- nuts.jl: nuts_state — FORM B: implicit-field methods; NO Ref; explicit physical init/fwd/bwd; -----
# direction resolved by EXPLICIT gofwd branches (typed per-direction variants); __self__ only as a call
# actual. Bare owner fields + explicit child traversal (fwd.pos). `current forward` = gofwd ? fwd : bwd;
# `current backward` = gofwd ? bwd : fwd.
@kernel nuts_state(init; rng, max_depth = 10, min_dham = -1000.,
                   step_f = nothing, stats_f = nothing) = begin
    gofwd = true
    may_sample = true
    may_continue = true
    fwd = deepcopy(init)                 # explicit FIXED physical owned endpoints (structural copies)
    bwd = deepcopy(init)                 # NO Ref, NO fwdbwd index, NO current-view alias
    trees = fillf(tree, init, max_depth + 1)
    proposals = fillf(deepcopy, init, max_depth + 2)
    dham = 0.
    diverged = !(dham >= min_dham)
    stepfwd!() = step_f(gofwd ? fwd : bwd)                       # step the current forward (gofwd branch)
    collectstats!() = isnothing(stats_f) || stats_f(__self__)
    logadvanceprob(depth) = trees[depth-1].log_weight[1] - trees[depth].log_weight[1]
    swapproposal!(i, j = length(proposals)) = begin
        proposals[i], proposals[j] = proposals[j], proposals[i]
    end
    step!(; force = true) = begin
        restore!(__self__; force)
        (gofwd ? bwd : fwd).mom .*= -1                           # negate current backward momentum
        trees[1].log_weight[1] = 0.
        for depth in 1:max_depth
            rand(rng, Bool) && flip!(__self__, depth)
            finish_tree!(__self__, depth)
            may_sample || break
            randbernoullilog(rng, logswapprob(trees[depth])) && swapproposal!(__self__, depth)
            may_continue || break
        end
        rcopy!(init, proposals[end])
    end
    flip!(depth) = if depth > 1
        gofwd = !gofwd
        tree = trees[depth]
        backward = gofwd ? bwd : fwd                            # current backward AFTER toggle (gofwd branch)
        @. tree.bwd.mom = -backward.mom
        @. tree.bwd.dham_dmom = -backward.dham_dmom
        @. tree.summed_mom.fwd *= -1
    end
    finish_tree!(depth) = begin
        tree = trees[depth]
        suptree = trees[depth+1]
        forward = gofwd ? fwd : bwd                             # current forward (gofwd branch)
        tree.log_weight[2] = tree.log_weight[1]
        if depth == 1
            rcopy!(suptree.bwd, (; forward.mom, forward.dham_dmom))
        else
            rcopy!(suptree.bwd, tree.bwd)
            rcopy!(tree.bwd_fwd, (; forward.mom, forward.dham_dmom))
            tree.summed_mom.bwd .= tree.summed_mom.fwd
        end
        start_tree!(__self__, depth)
        may_continue || return may_sample = false
        suptree.log_weight[1] = logaddexp(tree.log_weight[1], tree.log_weight[2])
        may_continue = if depth == 1
            suptree.summed_mom.fwd .= suptree.bwd.mom .+ forward.mom
            compute_criterion(suptree.summed_mom.fwd, suptree.bwd.dham_dmom, forward.dham_dmom)
        else
            suptree.summed_mom.fwd .= tree.summed_mom.bwd .+ tree.summed_mom.fwd
            (
                compute_criterion(suptree.summed_mom.fwd, suptree.bwd.dham_dmom, forward.dham_dmom) &&
                compute_criterion(badd(tree.summed_mom.bwd, tree.bwd.mom),
                                  suptree.bwd.dham_dmom, tree.bwd.dham_dmom) &&
                compute_criterion(badd(tree.bwd_fwd.mom, tree.summed_mom.fwd),
                                  tree.bwd_fwd.dham_dmom, forward.dham_dmom)
            )
        end
    end
    start_tree!(depth) = if depth == 1
        stepfwd!(__self__)
        forward = gofwd ? fwd : bwd                             # current forward (gofwd branch)
        dham = finiteorneginf(init.ham - forward.ham)
        collectstats!(__self__)
        diverged && return may_continue = false
        trees[1].log_weight[1] = dham
        rcopy!(proposals[1], forward)
    else
        start_tree!(__self__, depth - 1)
        may_continue || return may_sample = false
        swapproposal!(__self__, depth - 1, depth)
        finish_tree!(__self__, depth - 1)
        if may_sample && randbernoullilog(rng, logadvanceprob(__self__, depth))
            swapproposal!(__self__, depth - 1, depth)
        end
    end
end

# ---- FORM C: public compiled entry — mutate compiler-owned concrete state + return the SAME object -----
# result === state; fixed shape; same identity/type; 0-B; no RefValue. Shape changes use a separate
# non-hot reconstruction API (not this hot entry). `!!` alias/effect registration drives invalidation.
@kernel nuts!!(state; rng) = begin
    step!(state)          # one multinomial NUTS transition on compiler-owned concrete state
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
