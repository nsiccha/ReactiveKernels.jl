# ── Linear-Gaussian recurrences as WarmupHMC Markov chains ───────────────
#
# `markov_chains(model, :x)` lowers the latent `@scan` array `x` in its states
# and records, per state, the transition WarmupHMC's `MarkovChain` reads:
# state `t`'s location is `coef[t] * x[t - 1] + drift[t]` (the seed's location
# where `coef[t] = 0`) and its log step SD is `log_scale[t]`.
# `adaptive_centering_problem` then gives the recurrence one `MarkovChain`,
# whose warm-up adapts its centring per time step. A recurrence per `@plate`
# index is one chain over every index's states in plate order: each index's
# first state has `coef = 0`, so the chain restarts there.
#
# The step analysis is `marginalize`'s (`_gaussian_transition`). A
# non-centred step (`e ~ Normal(0, 1); x[t] = a * x[t - 1] + s * e`) is
# re-spelled as its centred law (`x[t] ~ Normal(a * x[t - 1], s)`), so the
# packed coordinates are the states; the density of every other parameter is
# unchanged. The recorded frame is a deterministic scan the density does not
# read (in a cell, a second plate over the same indices).

const _CHAIN_PREFIX = "_rkppl_chain_"
_chain_name(x::Symbol, role::Symbol) = Symbol(_CHAIN_PREFIX, x, :_, role)
const _CHAIN_ROLES = (:coef, :drift, :log_scale)

_chain_fail(x, msg) = _sfail("markov_chains(`$x`): " * msg)

"""
    markov_chains(model, names::Symbol...) -> model

Mark the latent `@scan` arrays `names` as Markov chains for WarmupHMC's
adaptive centring (WarmupHMC 0.3.1 or later). `model` is an `RKPPLModel` or
a binding of one; the result composes with `|`, `condition` and `merge` as
the model does. [`adaptive_centering_problem`](@ref) then gives each marked
recurrence one `WarmupHMC.MarkovChain`, whose warm-up chooses per time step
how far to centre it: centred where observations pin it, the innovation
where they do not.

A marked array `x` qualifies as for [`marginalize`](@ref): its `@scan`
(top level, or a statement of a `@plate` cell) carries `x` alone, seeded once
(`x[1] ~ Normal(m0, s0)` or `x[1] = m0`), and each step is linear-Gaussian in
`x[t - 1]`, centred (`x[t] ~ Normal(a * x[t - 1] + c, s)`) or non-centred
(`e ~ Normal(0, 1); x[t] = a * x[t - 1] + c + s * e`). The coefficients may
read parameters, data, definitions and the loop index, but no value a chain
or an adaptive cell transforms. Unlike `marginalize`, any observation may
read `x`.

The program is lowered in the states: a non-centred step becomes its centred
law, so the packed coordinates are the states of `x` (its sampled seed, then
each step) instead of its innovations. The density of every other parameter
is unchanged. A recurrence per `@plate` index is one chain over
every index's states, restarting at each index. Any other form is refused
with a `SurfaceLoweringError` naming the statement.
"""
function markov_chains(m::RKPPLModel, names::Symbol...)
    isempty(names) && throw(ArgumentError(
        "markov_chains needs the latent array to chain: `markov_chains(model, :x)`"))
    out = copy(m.chains)
    for x in names
        occursin('.', String(x)) && _chain_fail(x, "a recurrence inside a submodel " *
            "is not chained yet; name a `@scan` array of the model")
        x in out && _chain_fail(x, "`$x` is already a Markov chain")
        x in m.marginalized && _chain_fail(x, "`$x` is integrated out by `marginalize`")
        push!(out, x)
    end
    return RKPPLModel(m.ast, m.mod, copy(m.fixed), copy(m.rewrites),
        copy(m.conditioned), copy(m.documentation), copy(m.marginalized), out)
end
markov_chains(m::RKPPLBoundModel, names::Symbol...) =
    RKPPLBoundModel(markov_chains(m.model, names...), copy(m.data))
markov_chains(::StructuralPlan, names::Symbol...) = _sfail("markov_chains(" *
    join(("`$x`" for x in names), ", ") * "): a bound plan is already lowered. " *
    "Mark the model or its binding before observing it: " *
    "`markov_chains(model(; inputs...), :x) | (; y)`")

function _chain_latents(ast::Expr, chains, values, observing)
    for x in chains
        ast = _chain_latent(ast, x, values, observing)
    end
    return ast
end

function _chain_latent(ast::Expr, x::Symbol, values, observing)
    stmts = ast.args
    k = findfirst(st -> _is_scan_of(st, x), stmts)
    if k !== nothing
        block = stmts[k].args[end]
        spec, hi = _parse_marginal_scan(block)
        law = _chain_transition(spec, x, hi)
        frame = _chain_frame_scan(x, law, spec)
        out = Any[]
        for (i, st) in enumerate(stmts)
            if i == k
                push!(out, _scan_macro(_chain_centred_block(block, spec, x, law)))
                append!(out, frame)
            else
                push!(out, st)
            end
        end
        return Expr(:block, out...)
    end
    p = findfirst(st -> _is_plate_scan_of(st, x), stmts)
    p === nothing && _chain_fail(x, any(st -> Meta.isexpr(st, :macrocall) &&
        st.args[1] === Symbol("@plate") && _mentions_symbol(st, x), stmts) ?
        "its `@scan` is not a statement of a `@plate` cell (a recurrence in a " *
        "nested `@plate` is not chained yet)" :
        "no top-level `@scan` and no `@plate` cell `@scan` carries `$x`")
    loop = _plate_loop(stmts[p])
    head = loop.args[1]
    ivar = head.args[1]
    cell = Any[c for c in loop.args[2].args if !(c isa LineNumberNode)]
    kc = findfirst(c -> _is_scan_of(c, x), cell)
    block = cell[kc].args[end]
    spec, hi = _parse_marginal_scan(block)
    law = _chain_transition(spec, x, hi)
    density_cell = copy(cell)
    density_cell[kc] = _scan_macro(_chain_centred_block(block, spec, x, law))
    frame = _chain_frame_scan(x, law, spec; cell = true)
    carried = frame[1]
    outputs = Any[Expr(:(=), Expr(:ref, _chain_name(x, r), ivar), v) for (r, v) in frame[2]]
    side = _cell_side_plate(head, cell, (kc,), observing, Any[carried], outputs) do drawn
        _chain_fail(x, "its transition reads " * join(("`$d`" for d in drawn), ", ") *
            ", drawn inside the `@plate` cell; draw it outside the cell (a declared " *
            "array read as `$(first(drawn))[$ivar]`)")
    end
    out = Any[]
    for (i, st) in enumerate(stmts)
        if i == p
            push!(out, Expr(:macrocall, st.args[1], st.args[2],
                Expr(:for, head, Expr(:block, density_cell...))), side)
        else
            push!(out, st)
        end
    end
    return Expr(:block, out...)
end

# The transition, refused unless it is a stochastic recurrence.
function _chain_transition(spec::ScanSpec, x::Symbol, hi)
    law = try
        _gaussian_transition(spec, x, hi)
    catch e
        e isa SurfaceLoweringError || rethrow()
        # The analysis is marginalize's; name the operation the author called.
        throw(SurfaceLoweringError(replace(e.message, "marginalize(`$x`)" => "markov_chains(`$x`)",
            "the Kalman filter integrates out" => "a Markov chain is")))
    end
    law.innovation_var === nothing && _chain_fail(x, "its steps draw no innovation; " *
        "a deterministic recurrence has no states to centre")
    return law
end

_chain_seed_sampled(spec::ScanSpec) = only(spec.setup).kind === :sample
_chain_centred(spec::ScanSpec) = !any(st -> st.kind === :sample && !st.indexed, spec.step)

# The scan as written when it is centred; otherwise the seed as written and
# each step's centred law, `x[t] ~ Normal(coef * x[t - 1] + shift, sd)`, after
# the step statements the law reads.
function _chain_centred_block(block, spec::ScanSpec, x, law::_GaussianTransition)
    _chain_centred(spec) && return block
    t = law.loopvar
    lag = Expr(:ref, x, Expr(:call, :-, t, 1))
    location = something(_madd(_mmul(law.coef, lag, false), law.shift, false), 0.0)
    step = Expr(:call, :~, Expr(:ref, x, t),
        Expr(:call, :Normal, location, Expr(:call, :sqrt, law.innovation_var)))
    setup = Any[st for st in block.args if !(st isa LineNumberNode) && !Meta.isexpr(st, :for)]
    loop = Expr(:for, Expr(:(=), t, Expr(:call, :(:), 2, law.hi)),
        Expr(:block, law.locals..., step))
    return Expr(:block, setup..., loop)
end

# The recorded frame: a scan carrying each state's coefficient, drift and log
# step SD, then the per-state values the chain reads. With a sampled seed
# every entry is a state and state 1's location is the seed's mean. With a
# fixed seed `x[1] = m0` the states are `x[2:end]`: the first one's location
# reads the seed, `coef[2] * m0 + shift[2]`, so it is its drift and its
# coefficient is zero. Returns the scan and `role => value` pairs: top-level
# definitions, or a cell's per-index outputs (`cell = true`).
function _chain_frame_scan(x, law::_GaussianTransition, spec::ScanSpec; cell = false)
    t = law.loopvar
    sampled = _chain_seed_sampled(spec)
    carried = Dict(r => _chain_name(x, Symbol(:frame_, r)) for r in _CHAIN_ROLES)
    C, D, L = carried[:coef], carried[:drift], carried[:log_scale]
    # Literal coefficients (`x[t - 1] + …` has coefficient `1`) are floats, so
    # each frame holds one element type, also across `ifelse`.
    coef = _mfloat(something(law.coef, 0.0))
    shift = _mfloat(something(law.shift, 0.0))
    m0 = _mfloat(something(law.m0, 0.0))
    log_scale = Expr(:call, :*, 0.5, Expr(:call, :log, law.innovation_var))
    first_step(first, later) = sampled ? later :
        Expr(:call, :ifelse, Expr(:call, :(==), t, 2), first, later)
    body = Expr(:block, law.locals...,
        Expr(:(=), :($C[$t]), first_step(0.0, coef)),
        Expr(:(=), :($D[$t]), first_step(_mfloat(_madd(_mmul(coef, m0, false), shift, false)), shift)),
        Expr(:(=), :($L[$t]), log_scale))
    loop = Expr(:for, Expr(:(=), t, Expr(:call, :(:), 2, law.hi)), body)
    seed_log_scale = sampled ? Expr(:call, :*, 0.5, Expr(:call, :log, law.P0)) : 0.0
    scan = _scan_macro(Expr(:block,
        Expr(:(=), :($C[1]), 0.0), Expr(:(=), :($D[1]), _mfloat(m0)),
        Expr(:(=), :($L[1]), seed_log_scale), loop))
    states(a) = sampled ? a : Expr(:ref, a, Expr(:call, :(:), 2, law.hi))
    values = [r => states(carried[r]) for r in _CHAIN_ROLES]
    cell && return scan, values
    return Any[scan, (Expr(:(=), _chain_name(x, r), v) for (r, v) in values)...]
end
