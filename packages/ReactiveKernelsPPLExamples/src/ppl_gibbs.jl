"""
    PPLGibbs

**⚠️ EXPERIMENTAL — NOT REVIEWED / NOT APPROVED. Do NOT build on this.**

A PPL-style Gibbs layer over an `@ppl` model, in the PPL satellite — **never in
ReactiveKernels core** (user decision `0qv3ikg` option C, handed off from
`ReactiveKernels:sampling:gibbs`). Like `PPLMacro`, it is deliberately **not
exported** and not part of the consumer API; reach it via
`ReactiveKernelsPPLExamples.PPLGibbs`.

# Mechanism (built on rk's public surface — nothing in core)

Each block's full conditional is a slice of the ONE joint `@ppl` graph, and a
`ReactiveState` drives the sweep: `set!`-ing a block auto-invalidates only its
Markov blanket, so a block update recomputes only the terms that actually depend
on it (verified against the `ReactiveKernels:sampling:gibbs` PoC on
`kb-impl/ReactiveKernels-sampling-gibbs` @ `c67d401`). The named latents of the
`@ppl` model are the blocking vocabulary.

# Scope

- **Blocks**, generic **Metropolis-within-Gibbs**, incremental through
  `ReactiveState`. Because only the current block changes per step, the full
  `constrained_logdensity` is a correct Metropolis target (the other blocks'
  terms cancel in the ratio) and is recomputed incrementally.
- **Explicit blocking / sampler plan** (`1nwitne`): `blocks` is a
  `Vector{Symbol}` (single-site default, one block per latent) or a `Gibbs`
  spec — `Gibbs(:z, (:a, :b))` groups `(a, b)` into one jointly-updated block;
  `Gibbs(:z => :rw)` forces the random-walk sampler on a block that would
  otherwise draw conjugately. A group updates all its members jointly (one
  accept/reject on the joint conditional).
- **Real / positive / unit support** (`support` kwarg): a `:positive`/`:unit`
  block walks in log/logit space with the Metropolis-Hastings correction, so
  the proposal respects the support and the chain targets the constrained
  density directly. Validated against an analytic Gaussian posterior (real), a
  1-D numerical grid posterior (positive), and the conjugate Beta posterior
  (unit).
- **Automatic conjugacy** (`conjugate` kwarg, on by default): reading the
  `@ppl` structure descriptor (`PPLMacro.model_info`), a block whose full
  conditional is a supported data-only conjugate pair is drawn in closed form
  (exact, always accepted): **Beta-Bernoulli, Beta-Binomial, Gamma-Poisson**.
  Closed-form draws use `Random`-only samplers (Marsaglia–Tsang gamma, beta via
  two gammas). Validated against the exact Beta and Gamma posteriors.
- **Discrete latents** (Gibbs-only): an `@ppl` parameter with a discrete family
  (`bernoulli`) is a latent carried OUTSIDE the continuous `unconstrained`
  vector — its own `Bool` HAVE port, no transform/gradient — and sampled by
  **enumeration** of its full conditional (exact, always accepted). First cut:
  scalar Bernoulli. Validated against the analytic Bernoulli conditional.

`inverse_gamma` and `dirichlet` are now available in
`ReactiveKernelsDistributionKernels` (added by the distributions lane at the
authoring lane's request), so `@ppl` accepts an `inverse_gamma` (positive)
parameter and the Inverse-Gamma-Normal / Dirichlet conjugate pairs are now
buildable. Normal-Normal (mean) needs linear-predictor analysis (deferred).

Follow-up increments toward SSVS (decisions `0hvlzvd` / `0f8thsz`): vector
discrete latents + conditional priors (`β_j | z_j`), the spike-and-slab
acceptance model, and the deferred conjugate pairs above.
"""
module PPLGibbs

using ReactiveKernels
import ..PPLMacro   # experimental structure descriptor for conjugacy detection

export gibbs, Gibbs

# --- Conjugate posterior sampling (Random-only; no Distributions.jl) ---------
# Marsaglia–Tsang draw from Gamma(shape, rate) (`rate` is β, matching the
# `gamma(shape, rate)` distribution object). Shape ≥ 1 uses the squeeze method;
# shape < 1 boosts with the standard `U^(1/shape)` correction (the 1/rate scale
# commutes with that multiply, so applying it to the boosted draw is exact).
function _rand_gamma(rng, shape::Real, rate::Real)
    shape < 1 && return _rand_gamma(rng, shape + 1, rate) * rand(rng)^(1 / shape)
    d = shape - 1 / 3
    c = 1 / sqrt(9d)
    while true
        x = randn(rng)
        v = (1 + c * x)^3
        v <= 0 && continue
        u = rand(rng)
        log(u) < 0.5 * x^2 + d - d * v + d * log(v) && return d * v / rate
    end
end

# Beta(a, b) via two Gamma(·, 1) draws.
function _rand_beta(rng, a::Real, b::Real)
    x = _rand_gamma(rng, a, 1.0)
    y = _rand_gamma(rng, b, 1.0)
    x / (x + y)
end

# A detected conjugate full conditional whose posterior parameters are data-only
# (constant across sweeps): `:beta` → Beta(p1, p2), `:gamma` → Gamma(p1, p2).
struct _Conjugate
    family::Symbol
    p1::Float64
    p2::Float64
end
_draw_conjugate(rng, c::_Conjugate) = c.family === :beta ?
    _rand_beta(rng, c.p1, c.p2) : _rand_gamma(rng, c.p1, c.p2)

# Detect whether `block`'s full conditional is a supported data-only conjugate
# pair, using the `@ppl` structure descriptor. Returns a `_Conjugate` with the
# closed-form posterior parameters, or `nothing` (→ fall back to RW-MH). First
# cut — the three feasible "direct" pairs (block is the sole latent appearing as
# a bare likelihood argument, prior hyperparameters literal): Beta-Bernoulli,
# Beta-Binomial, Gamma-Poisson. Inverse-Gamma-Normal / Dirichlet need
# distributions absent from the distribution-kernels package; Normal-Normal needs
# linear-predictor analysis. Both are deferred (see the module docstring).
function _detect_conjugate(info::PPLMacro.PPLModelInfo, data, block::Symbol)
    p = nothing
    for pi in info.params
        pi.name === block && (p = pi; break)
    end
    (p === nothing || p.is_vector) && return nothing
    all(a -> a isa Real, p.prior_args) || return nothing
    refs = [o for o in info.obs if any(a -> a === block, o.args)]
    isempty(refs) && return nothing

    if p.prior_family === :beta && all(o -> o.family in (:binomial, :bernoulli), refs)
        s = 0.0
        f = 0.0
        for o in refs
            obsv = data[o.data]
            if o.family === :bernoulli
                o.args == Any[block] || return nothing
                s += sum(obsv)
                f += length(obsv) - sum(obsv)
            else                                   # binomial(n, block)
                (length(o.args) == 2 && o.args[2] === block &&
                    o.args[1] isa Symbol && haskey(data, o.args[1])) || return nothing
                n = data[o.args[1]]
                s += sum(obsv)
                f += sum(n) - sum(obsv)
            end
        end
        return _Conjugate(:beta, Float64(p.prior_args[1]) + s,
                          Float64(p.prior_args[2]) + f)
    elseif p.prior_family === :gamma && all(o -> o.family === :poisson, refs)
        c = 0.0
        n = 0
        for o in refs
            o.args == Any[block] || return nothing
            c += sum(data[o.data])
            n += length(data[o.data])
        end
        return _Conjugate(:gamma, Float64(p.prior_args[1]) + c,
                          Float64(p.prior_args[2]) + n)
    end
    return nothing
end

# A discrete latent (support `:discrete`) is sampled by ENUMERATION of its full
# conditional — never a random walk. Returns `(; kind, vals)` — `kind` is
# `:scalar` or `:vector` (a vector latent is updated element-wise), `vals` the
# support values to enumerate — or `nothing` when `block` is not an enumerable
# discrete latent. Bernoulli's port is `Bool`, so `vals = (false, true)`.
function _detect_discrete(info::PPLMacro.PPLModelInfo, block::Symbol)
    for pi in info.params
        pi.name === block || continue
        pi.support === :discrete || return nothing
        pi.prior_family === :bernoulli || return nothing   # other discrete families: later
        return (; kind = pi.is_vector ? :vector : :scalar, vals = (false, true))
    end
    nothing
end

# Sample a category from unnormalized log-weights `logps` over `vals`.
function _sample_categorical(rng, vals, logps)
    w = exp.(logps .- maximum(logps))
    r = rand(rng) * sum(w)
    acc = 0.0
    @inbounds for i in eachindex(vals)
        acc += w[i]
        r <= acc && return vals[i]
    end
    vals[end]
end

# Random-walk proposal in a block's UNCONSTRAINING space, returning the proposal
# together with the log Metropolis-Hastings correction `log q(cur|prop) -
# log q(prop|cur)`. A `:real` block walks symmetrically (correction 0); a
# `:positive`/`:unit` block walks symmetrically in log/logit space, so the
# correction is the log-ratio of the transform derivatives. That correction is
# exactly what lets the chain target the CONSTRAINED density directly
# (`constrained_logdensity`, which carries no unconstraining Jacobian) while
# every proposal respects the block's support.
function _propose_elt(rng, x::Real, step, support::Symbol)
    if support === :real
        return (x + step * randn(rng), 0.0)
    elseif support === :positive
        lx = log(x)
        lp = lx + step * randn(rng)
        return (exp(lp), lp - lx)                      # + log x' - log x
    elseif support === :unit
        l = log(x) - log1p(-x)                         # logit(x)
        lp = l + step * randn(rng)
        xp = 1 / (1 + exp(-lp))                        # logistic(l')
        return (xp, (log(xp) + log1p(-xp)) - (log(x) + log1p(-x)))
    else
        error("PPLGibbs: unknown support $(repr(support)) for a block; expected " *
              ":real, :positive, or :unit.")
    end
end

_propose(rng, cur::Real, step, support::Symbol) =
    _propose_elt(rng, cur, step, support)
function _propose(rng, cur::AbstractVector{<:Real}, step, support::Symbol)
    prop = similar(cur, float(eltype(cur)))
    hastings = 0.0
    for i in eachindex(cur)
        p, h = _propose_elt(rng, cur[i], step, support)
        prop[i] = p
        hastings += h
    end
    (prop, hastings)
end

# Per-block support (default real). `support` is a NamedTuple / Dict of
# block-name => (:real | :positive | :unit), or `nothing` for all-real.
_support_of(::Nothing, ::Symbol) = :real
_support_of(support, b::Symbol) = get(support, b, :real)

# --- Block/sampler specification (explicit user-declared blocks) -------------
# Decision `1nwitne` (resolved "no preference" → my recommendation): the default
# is single-site (one latent per block); a `Gibbs` spec lets the user GROUP
# latents into a jointly-updated block and/or FORCE a sampler. Each entry is a
# latent `:z`, a group `(:a, :b)`, or `<block> => sampler`.
struct _Block
    members::Vector{Symbol}
    sampler::Symbol            # :auto (conjugate-if-possible else RW) | :rw
end

"""
    Gibbs(entries...)

An explicit blocking / sampler plan for [`gibbs`](@ref). Each entry is a single
latent `:z`, a group of latents `(:a, :b)` updated jointly (one accept/reject on
the joint conditional), or a `block => sampler` pair. `sampler` is `:auto`
(closed-form conjugate draw when the block is a single latent with a detected
conjugate conditional, else a support-aware random-walk step — the default) or
`:rw` (force the random-walk step). A plain `Vector{Symbol}` passed to `gibbs` is
shorthand for one single-site `:auto` block per latent.
"""
struct Gibbs
    blocks::Vector{_Block}
    # Explicit inner constructor suppresses the auto-generated converting default
    # `Gibbs(blocks)`, which would otherwise shadow the varargs form below for a
    # single non-`Vector{_Block}` argument (e.g. `Gibbs((:a, :b))`).
    Gibbs(blocks::Vector{_Block}) = new(blocks)
end
Gibbs(entries...) = Gibbs(_Block[_parse_block(e) for e in entries])

_members(s::Symbol) = [s]
_members(t::Tuple) = collect(Symbol, t)
_members(v::AbstractVector) = collect(Symbol, v)
_parse_block(e::Pair) = _Block(_members(e.first), Symbol(e.second))
_parse_block(e) = _Block(_members(e), :auto)

_normalize_blocks(g::Gibbs) = g
_normalize_blocks(v::AbstractVector{Symbol}) =
    Gibbs(_Block[_Block([s], :auto) for s in v])
_normalize_blocks(v::AbstractVector) = Gibbs(_Block[_parse_block(e) for e in v])

# accept_rate key: a single-latent block keys by its symbol (back-compatible with
# the `blocks = [:a, :b]` form); a group keys by the tuple of its members.
_block_key(b::_Block) = length(b.members) == 1 ? b.members[1] : Tuple(b.members)

"""
    gibbs(model; blocks, data, init, iters, rng, step = 0.5, warmup = 0,
          support = nothing, conjugate = true, target = :constrained_logdensity)

Metropolis-within-Gibbs over the named latents of an `@ppl` `model`. `blocks` is
either a `Vector{Symbol}` (one single-site block per latent) or a [`Gibbs`](@ref)
spec that groups latents into jointly-updated blocks and/or forces a sampler.
`data` and `init` are NamedTuples of port name → value. Returns
`(; draws, accept_rate, iters)`, where `draws[latent]` is the retained chain for
each latent and `accept_rate[key]` is per block (`key` is the latent symbol for a
single-site block, the tuple of members for a group).

When `conjugate` is on (default) and the model carries an `@ppl` structure
descriptor, each block whose full conditional is a supported data-only conjugate
pair (Beta-Bernoulli, Beta-Binomial, Gamma-Poisson) is drawn in closed form
(exact, always accepted — `accept_rate` 1.0); every other block uses the
support-aware random-walk step below. Pass `conjugate = false` to force RW-MH for
all blocks.

`support` gives each block's support as a NamedTuple / Dict of block-name =>
`:real` (default) | `:positive` | `:unit`. A `:positive`/`:unit` block walks in
log/logit space with the matching Metropolis-Hastings correction, so every
proposal respects the support while the chain still targets the CONSTRAINED
density directly (the named-latent port takes the constrained value and
`constrained_logdensity` carries no unconstraining Jacobian). Support is passed
here rather than read from the model: `@ppl` exposes no structure descriptor yet
(that is the follow-up conjugacy increment), so the engine stays decoupled from
the macro internals.

The sweep is driven incrementally through a `ReactiveState`: each accepted or
rejected `set!` invalidates only the block's Markov blanket, so the next
`constrained_logdensity` read recomputes only the affected terms.
"""
function gibbs(model; blocks, data, init, iters::Int, rng,
               step = 0.5, warmup::Int = 0, support = nothing,
               conjugate::Bool = true,
               target::Symbol = :constrained_logdensity)
    spec = _normalize_blocks(blocks)
    members = Symbol[]                        # every latent, flattened, in order
    for b in spec.blocks, m in b.members
        m in members || push!(members, m)
    end
    tgt = getproperty(model, target)
    st = ReactiveState(model.graph; materialize = (tgt,))
    for (k, v) in pairs(data)
        set!(st, getproperty(model, k), v)
    end
    ports = Dict(m => getproperty(model, m) for m in members)
    supports = Dict(m => _support_of(support, m) for m in members)
    # Conjugate detection is per SINGLE-latent `:auto` block (a group is not
    # jointly conjugate in general). Automatic when `conjugate` is on and the
    # model carries an `@ppl` structure descriptor.
    conj = Dict{Symbol,Union{Nothing,_Conjugate}}(m => nothing for m in members)
    disc = Dict{Symbol,Any}(m => nothing for m in members)
    if PPLMacro.has_model_info(model)
        info = PPLMacro.model_info(model)
        for b in spec.blocks
            length(b.members) == 1 || continue
            m = b.members[1]
            # A discrete latent MUST be enumerated (never RW), regardless of the
            # `conjugate` kwarg or a `:rw` override.
            disc[m] = _detect_discrete(info, m)
            # Closed-form conjugate draw for a continuous `:auto` block.
            if disc[m] === nothing && conjugate && b.sampler === :auto
                conj[m] = _detect_conjugate(info, data, m)
            end
        end
    end
    cur = Dict{Symbol,Any}(m => init[m] for m in members)
    for m in members
        set!(st, ports[m], cur[m])
    end
    logtarget() = get!(st, tgt)

    draws = Dict{Symbol,Vector{Any}}(m => Any[] for m in members)
    blockkeys = [_block_key(b) for b in spec.blocks]
    accepts = Dict{Any,Int}(k => 0 for k in blockkeys)
    for it in 1:(warmup + iters)
        counted = it > warmup
        for b in spec.blocks
            k = _block_key(b)
            m1 = length(b.members) == 1 ? b.members[1] : nothing
            d = m1 === nothing ? nothing : disc[m1]
            c = m1 === nothing ? nothing : conj[m1]
            if d !== nothing
                # Exact discrete Gibbs draw by enumerating the full conditional.
                if d.kind === :scalar
                    logps = Float64[]
                    for v in d.vals
                        set!(st, ports[m1], v)
                        push!(logps, logtarget())
                    end
                    chosen = _sample_categorical(rng, d.vals, logps)
                    set!(st, ports[m1], chosen)
                    cur[m1] = chosen
                else
                    # Vector latent: single-site enumeration per element, each
                    # conditioned on the current values of the others.
                    zv = collect(cur[m1])
                    for j in eachindex(zv)
                        logps = Float64[]
                        for v in d.vals
                            zv[j] = v
                            set!(st, ports[m1], zv)
                            push!(logps, logtarget())
                        end
                        zv[j] = _sample_categorical(rng, d.vals, logps)
                    end
                    set!(st, ports[m1], zv)
                    cur[m1] = zv
                end
                counted && (accepts[k] += 1)
            elseif c !== nothing
                # Exact conjugate draw (single-latent block) — always accepted.
                m = m1
                prop = _draw_conjugate(rng, c)
                set!(st, ports[m], prop)
                cur[m] = prop
                counted && (accepts[k] += 1)
            else
                # Joint random-walk over the block's members (single-site when
                # one member): propose all, one accept/reject on the joint
                # conditional, restore all on reject.
                lp_cur = logtarget()
                saved = Any[cur[m] for m in b.members]
                hastings = 0.0
                for m in b.members
                    prop, h = _propose(rng, cur[m], step, supports[m])
                    set!(st, ports[m], prop)
                    cur[m] = prop
                    hastings += h
                end
                lp_prop = logtarget()
                if isfinite(lp_prop) && log(rand(rng)) < (lp_prop - lp_cur) + hastings
                    counted && (accepts[k] += 1)
                else
                    for (i, m) in enumerate(b.members)
                        set!(st, ports[m], saved[i])
                        cur[m] = saved[i]
                    end
                end
            end
        end
        if counted
            for m in members
                push!(draws[m], copy(cur[m]))
            end
        end
    end
    accept_rate = Dict(k => accepts[k] / iters for k in blockkeys)
    (; draws, accept_rate, iters)
end

end # module PPLGibbs
