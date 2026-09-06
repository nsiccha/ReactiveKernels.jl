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

- **Single-site blocks**, generic **random-walk Metropolis-within-Gibbs**,
  incremental through `ReactiveState`. Because only one block changes per step,
  the full `constrained_logdensity` is a correct Metropolis target (the other
  blocks' terms cancel in the ratio) and is recomputed incrementally.
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

Feasible-conjugacy note (decision `0hvlzvd` resolved "no preference" → my
recommendation): the recommended set also named Inverse-Gamma-Normal and
Dirichlet, but `inverse_gamma`/`dirichlet` are absent from
`ReactiveKernelsDistributionKernels`, so those pairs cannot be authored in `@ppl`
without the dist-kernels owner adding them (primer finding, not a local
workaround). Normal-Normal (mean) is dist-feasible but needs linear-predictor
analysis. Both are deferred.

Follow-up increments: those deferred conjugate pairs, the explicit block/sampler
API (`1nwitne`: `Gibbs(:z => …, (:a,:b) => …)`), and reproducing the SSVS
spike-and-slab example on an `@ppl` model as the acceptance case.
"""
module PPLGibbs

using ReactiveKernels
import ..PPLMacro   # experimental structure descriptor for conjugacy detection

export gibbs

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

"""
    gibbs(model; blocks, data, init, iters, rng, step = 0.5, warmup = 0,
          support = nothing, conjugate = true, target = :constrained_logdensity)

Single-site Metropolis-within-Gibbs over the named latents of an `@ppl` `model`.
`blocks` is a vector of latent-name `Symbol`s; `data` and `init` are NamedTuples
of port name → value. Returns `(; draws, accept_rate, iters)` where
`draws[block]` is the retained chain for that block.

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
function gibbs(model; blocks::Vector{Symbol}, data, init, iters::Int, rng,
               step = 0.5, warmup::Int = 0, support = nothing,
               conjugate::Bool = true,
               target::Symbol = :constrained_logdensity)
    tgt = getproperty(model, target)
    st = ReactiveState(model.graph; materialize = (tgt,))
    for (k, v) in pairs(data)
        set!(st, getproperty(model, k), v)
    end
    ports = Dict(b => getproperty(model, b) for b in blocks)
    supports = Dict(b => _support_of(support, b) for b in blocks)
    # Per-block conjugate full conditional (or `nothing` → RW-MH). Automatic
    # when `conjugate` is on and the model carries an `@ppl` structure descriptor.
    conj = Dict{Symbol,Union{Nothing,_Conjugate}}(b => nothing for b in blocks)
    if conjugate && PPLMacro.has_model_info(model)
        info = PPLMacro.model_info(model)
        for b in blocks
            conj[b] = _detect_conjugate(info, data, b)
        end
    end
    cur = Dict{Symbol,Any}(b => init[b] for b in blocks)
    for b in blocks
        set!(st, ports[b], cur[b])
    end
    logtarget() = get!(st, tgt)

    draws = Dict{Symbol,Vector{Any}}(b => Any[] for b in blocks)
    accepts = Dict{Symbol,Int}(b => 0 for b in blocks)
    for it in 1:(warmup + iters)
        counted = it > warmup
        for b in blocks
            c = conj[b]
            if c !== nothing
                # Exact conjugate draw from the full conditional — always accepted.
                prop = _draw_conjugate(rng, c)
                set!(st, ports[b], prop)
                cur[b] = prop
                counted && (accepts[b] += 1)
            else
                lp_cur = logtarget()
                cur_val = cur[b]
                prop, hastings = _propose(rng, cur_val, step, supports[b])
                set!(st, ports[b], prop)
                lp_prop = logtarget()
                if isfinite(lp_prop) && log(rand(rng)) < (lp_prop - lp_cur) + hastings
                    cur[b] = prop
                    counted && (accepts[b] += 1)
                else
                    set!(st, ports[b], cur_val)      # reject: restore the block
                end
            end
        end
        if counted
            for b in blocks
                push!(draws[b], copy(cur[b]))
            end
        end
    end
    accept_rate = Dict(b => accepts[b] / iters for b in blocks)
    (; draws, accept_rate, iters)
end

end # module PPLGibbs
