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

Follow-up increments (design settled: decisions `0hvlzvd` conjugacy scope and
`1nwitne` blocking API both resolved "no preference" → my recommendations, a
small fixed conjugate set + explicit user-declared blocks): closed-form conjugate
draws (Normal / Inverse-Gamma / Beta-Bernoulli/Binomial / Dirichlet), the
explicit block/sampler API (`Gibbs(:z => …, (:a,:b) => …)`), and reproducing the
SSVS spike-and-slab example on an `@ppl` model as the acceptance case. Conjugacy
detection will read a structure descriptor exposed by `@ppl`, at which point the
`support` kwarg can default from the model rather than the caller.
"""
module PPLGibbs

using ReactiveKernels

export gibbs

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
          support = nothing, target = :constrained_logdensity)

Single-site random-walk Metropolis-within-Gibbs over the named latents of an
`@ppl` `model`. `blocks` is a vector of latent-name `Symbol`s; `data` and `init`
are NamedTuples of port name → value. Returns `(; draws, accept_rate, iters)`
where `draws[block]` is the retained chain for that block.

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
               target::Symbol = :constrained_logdensity)
    tgt = getproperty(model, target)
    st = ReactiveState(model.graph; materialize = (tgt,))
    for (k, v) in pairs(data)
        set!(st, getproperty(model, k), v)
    end
    ports = Dict(b => getproperty(model, b) for b in blocks)
    supports = Dict(b => _support_of(support, b) for b in blocks)
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
            lp_cur = logtarget()
            cur_val = cur[b]
            prop, hastings = _propose(rng, cur_val, step, supports[b])
            set!(st, ports[b], prop)
            lp_prop = logtarget()
            if isfinite(lp_prop) && log(rand(rng)) < (lp_prop - lp_cur) + hastings
                cur[b] = prop
                counted && (accepts[b] += 1)
            else
                set!(st, ports[b], cur_val)          # reject: restore the block
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
