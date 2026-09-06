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

# First-cut scope

- **Single-site, real-support blocks**, generic **random-walk
  Metropolis-within-Gibbs**, incremental through `ReactiveState`. Because only
  one block changes per step, the full `constrained_logdensity` is a correct
  Metropolis target (the other blocks' terms cancel in the ratio) and is
  recomputed incrementally.

Follow-up increments (design settled: decisions `0hvlzvd` conjugacy scope and
`1nwitne` blocking API both resolved "no preference" → my recommendations, a
small fixed conjugate set + explicit user-declared blocks): closed-form conjugate
draws (Normal / Inverse-Gamma / Beta-Bernoulli/Binomial / Dirichlet),
positive/unit support proposals, the explicit block/sampler API
(`Gibbs(:z => …, (:a,:b) => …)`), and reproducing the SSVS spike-and-slab
example on an `@ppl` model as the acceptance case.
"""
module PPLGibbs

using ReactiveKernels

export gibbs

# Random-walk proposal, shaped by the current value (scalar or vector real).
_propose(rng, cur::Real, step) = cur + step * randn(rng)
_propose(rng, cur::AbstractVector{<:Real}, step) =
    cur .+ step .* randn(rng, length(cur))

"""
    gibbs(model; blocks, data, init, iters, rng, step = 0.5, warmup = 0,
          target = :constrained_logdensity)

Single-site random-walk Metropolis-within-Gibbs over the named latents of an
`@ppl` `model` (real-support blocks, first cut). `blocks` is a vector of
latent-name `Symbol`s; `data` and `init` are NamedTuples of port name → value.
Returns `(; draws, accept_rate, iters)` where `draws[block]` is the retained
chain for that block.

The sweep is driven incrementally through a `ReactiveState`: each accepted or
rejected `set!` invalidates only the block's Markov blanket, so the next
`constrained_logdensity` read recomputes only the affected terms.
"""
function gibbs(model; blocks::Vector{Symbol}, data, init, iters::Int, rng,
               step = 0.5, warmup::Int = 0,
               target::Symbol = :constrained_logdensity)
    tgt = getproperty(model, target)
    st = ReactiveState(model.graph; materialize = (tgt,))
    for (k, v) in pairs(data)
        set!(st, getproperty(model, k), v)
    end
    ports = Dict(b => getproperty(model, b) for b in blocks)
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
            prop = _propose(rng, cur_val, step)
            set!(st, ports[b], prop)
            lp_prop = logtarget()
            if isfinite(lp_prop) && log(rand(rng)) < lp_prop - lp_cur
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
