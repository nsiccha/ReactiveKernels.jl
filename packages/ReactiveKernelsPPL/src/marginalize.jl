# ── Exact marginalization of a linear-Gaussian latent recurrence ─────────
#
# `marginalize(model, :x)` integrates the latent `@scan` array `x` out of the
# density exactly. The model source keeps the recurrence as written; lowering
# proves it linear-Gaussian and rewrites it, together with the Normal
# observation that reads it, into a Kalman filter: one `@scan` carrying the
# filtered state mean and variance and observing each entry through its
# one-step-ahead predictive density (rkppl-use §5 "Observing a carried
# array"). The sampler then has no coordinates for `x`; its exact posterior
# draws come from forward filtering, backward sampling at the query boundary
# (`prepare_marginalized_latents` / `marginalized_latent_draws`), following
# the discrete-latent precedent: no RNG enters an RK kernel.
#
# A deterministic `@scan` beside the filter records each step's transition
# (coefficient, shift, innovation variance) for those draws. Nothing reads it
# for the density, so the density and gradient cuts leave it out.

const _MARGINAL_PREFIX = "_rkppl_marginal_"
_marginal_name(x::Symbol, role::Symbol) = Symbol(_MARGINAL_PREFIX, x, :_, role)
const _MARGINAL_ROLES = (:mean, :var, :coef, :shift, :innovation_var)

"""
    marginalize(model, names::Symbol...; predictive = (;), filtered = (;)) -> model

Integrate finite integer parameters or linear-Gaussian latent `@scan` arrays
`names` out of `model` exactly. `model` is an `RKPPLModel` or a binding of one (`model(; inputs...)`);
the result composes with `|` and `condition` as the model does. The model
source keeps the recurrence as written.

Scalar `Binomial`, `Bernoulli`, `Categorical` and `DiscreteUniform` parameters
are summed over their finite supports with log-sum-exp. Binomial trials and
DiscreteUniform bounds come from data. The integer state occupies no real
coordinates; the generated program iterates over its support at runtime.
[`prepare_discrete_latents`](@ref) returns the joint state probabilities and
named state columns under `parameters`, and [`discrete_latent_draws`](@ref)
recovers a joint draw with a caller-supplied RNG. Several declarations can
be eliminated together, including submodel locals named by their author paths.

An integer array, declared (`z[axes(y, 1)] .~ Categorical(w)`) or written
as the integers of a `@plate` loop (`z[i] ~ Categorical(w)`), is summed out
element by element when every density term that reads it sums cells that
each read it only at their own element (`y .~ Normal.(mu[z], s)`): the
elements are then conditionally independent, and the density holds
`Σ_i logsumexp_s (log p(z_i = s) + log p(y_i | z_i = s))`, in time linear in
the number of elements and in the support size. Elements may each have their
own support (`Binomial.(n, p)` with a vector `n`). A term coupling elements
(one reading `sum(z)`, a recurrence over `z`) is refused: summing it out
would enumerate every joint state. `:pointwise` then holds each element's
predictive density given everything else (sequential in field order when
several responses read the element), and `prepare_discrete_latents` returns
each element's posterior under `parameters.elements`, drawn independently by
`discrete_latent_draws`.

A linear-Gaussian recurrence is lowered as a Kalman filter.
A marginalized array `x` qualifies when

- its `@scan` carries `x` alone, seeded once: `x[1] ~ Normal(m0, s0)` or
  `x[1] = m0`;
- each step is linear-Gaussian in the previous state: centred,
  `x[t] ~ Normal(a * x[t - 1] + c, s)`, or non-centred,
  `e ~ Normal(mu, sigma); x[t] = a * x[t - 1] + c + b * e`, where `a`, `b`, `c`,
  `s`, `mu` and `sigma` may read parameters, data, definitions, the loop
  index and earlier step locals but not `x` or an innovation. Locals that
  are themselves linear in `x[t - 1]` and the innovations (`w = s * e`) may
  enter the write;
- one statement reads it, a Normal observation of a whole observed data
  vector whose location is affine in it and whose scale does not read it:
  `y .~ Normal.(offset .+ loading .* x, noise)`. A `missing` entry of `y` is
  a step that only predicts: the filter carries the state through it
  without an update, and the likelihood has no term for it. `y` may also be
  a vector the model defines without reading `x`, such as measured rows
  merged in time order with bounded latents of censored ones (data
  augmentation: `gap[1:K] .~ restricted.(Flat.(), 0, Inf)`,
  `y = vcat(measured, limit .- gap)[order]`). The filter then observes each
  entry at its value, and the scan's length is supplied. Condition the
  measured rows, `| (; measured)`: their entries' one-step-ahead predictive
  densities are then the likelihood and `:pointwise.measured`, and the other
  entries' are prior factors. Bound as an input instead, `measured` is not
  conditioned, and every entry's density is a prior factor. Rows may instead
  observe the state at their steps, `y .~ Normal.(offset .+ loading .*
  x[step], noise)` with `step` bound integer data: a step may have several
  rows or none. The filter then runs over events (each step, then its rows
  in order), the scan's length is supplied, and each row's pointwise density
  is its predictive given the rows before it in that order.

Any other form is refused with a `SurfaceLoweringError` naming the
statement. Each scalar coefficient may vary by step (irregular gaps
`exp(-gap[t] / range)`, a reset `r = 0` where a new series starts).

A `@scan` carrying several arrays, or reading one at a lag above
1 (seeded at `x[1]`, …, `x[p]`, its loop from `p + 1`), has a vector state:
name every array it carries, `marginalize(model, :level, :slope)`. Each seed
is a Normal draw or a value affine in the seeds above it, and each step
writes every array once, affine in the lags `a[t - k]`, the values written
above it in the same step and Normal innovations (a local linear trend
`level[t] ~ Normal(level[t - 1] + slope[t - 1], s)`, an AR(2)
`x[t] ~ Normal(a1 * x[t - 1] + a2 * x[t - 2], s)`). One statement reads the
arrays, `y .~ Normal.(offset .+ loading_a .* a .+ …, noise)`, an observation
of bound data without missing entries whose location is affine in them. The
filter carries the window `a[t - j]`, `j < p`, of every array, and the draws
return each array under its name. In a plate, each cell has its own window
and observes its own vector `y[g]`.

A step's innovation scale may be 0 (`s * sqrt(1 - r * r)` at `r = 1`, the
state carried over unchanged): the filter is exact there, and native
gradients are finite. In emitted JAX (`jax_source`) a scale that reaches 0
through `sqrt` has a NaN gradient, because JAX's derivative of `sqrt` at 0
is infinite.

A recurrence a submodel holds is named by its author path:
`marginalize(model, Symbol("z.x"))` for the array `x` of `z ~ sm(…)` (a
nested call: `Symbol("z.w.x")`). It integrates out as its top-level spelling
does, and its draws nest under the call, `draws.z.x`, as [`constrain`](@ref)
nests sampled locals.

A recurrence in a `@plate for g in …` cell integrates out one filter per
index, when the cell's `@scan` carries `x` as above and one statement of the
cell observes it, `y[g] .~ Normal.(offset .+ loading .* x, noise)`, with `y`
bound data holding one vector per index. A cell definition or per-index
output read only by that observation (`loc[g] = level[g] .+ x`) is part of
it. Each of `offset`, `loading` and `noise` is one value per index (a
declared array or data vector read as `a[g]`, a parameter, a cell scalar) or
one value per entry of `y[g]` (data holding one vector per index read as
`w[g]`, and broadcasts of these). The draws hold one trajectory per index.

The marginalized program's density is the exact marginal density of the
observations given the remaining parameters, so `coordinate_names` and the
layout no longer contain `x` or its innovations. `:pointwise.y` holds the
one-step-ahead predictive densities (0 at a missing entry), whose sum is the
marginal likelihood; for a merged `y`, `:pointwise.measured` holds those of
the measured entries.
Exact draws of `x` given the parameters come from
[`prepare_marginalized_latents`](@ref) and
[`marginalized_latent_draws`](@ref).

The filter's values become named values of the program when the call names
them. `predictive` names the one-step-ahead predictive law of each observed
entry given the entries before it, any of `predictive = (location = :mu,
variance = :v, scale = :sd)`; `filtered` names the filtered mean and variance
of the state given the observations up to each step, `filtered = (mean = :m,
variance = :v)`, or per array for a call integrating out several,
`filtered = (level = (mean = :level_mean,), slope = (variance = :slope_var,))`.
Each name is an ordinary value: `prepare_query(built, plan, (:mu, :sd))`
returns it, `jax_source(built, plan; values = (:mu, :sd))` emits it in
`constrain`, `kernel_expr` shows it. A predictive value holds one value per
entry of `y` (at a `missing` entry, its forecast from the entries before it),
one per row for rows observing `x[step]`, and in a `@plate` cell one vector
per index (`mu[g]`); a filtered value holds one value per step (one vector per
index in a cell). Nothing is added when nothing is named, and naming values
leaves the density unchanged. A name the model already uses, a name starting
with `_rkppl`, a name given twice, and a call naming values for recurrences
that different statements observe are refused: call `marginalize` once per
recurrence, `marginalize(marginalize(model, :x; …), :z; …)`.

Marginalize the model or its binding before observing it:
`marginalize(model(; inputs...), :x) | (; y)`. A bound plan (the result of
`|`) is already lowered and is refused.
"""
function marginalize(m::RKPPLModel, names::Symbol...; predictive = (;), filtered = (;))
    isempty(names) && throw(ArgumentError(
        "marginalize needs the latent array to integrate out: `marginalize(model, :x)`"))
    out = copy(m.marginalized)
    for x in names
        x in out && _sfail("marginalize(`$x`): `$x` is already marginalized")
        x in m.chains && _sfail("marginalize(`$x`): `$x` is a Markov chain (`markov_chains`)")
        push!(out, x)
    end
    values = copy(m.marginal_values)
    request = _marginal_values_request(collect(Symbol, names), predictive, filtered, values)
    request === nothing || push!(values, request)
    return RKPPLModel(m.ast, m.mod, copy(m.fixed), copy(m.rewrites),
        copy(m.conditioned), copy(m.documentation), out, copy(m.chains), values)
end
marginalize(m::RKPPLBoundModel, names::Symbol...; kwargs...) =
    RKPPLBoundModel(marginalize(m.model, names...; kwargs...), copy(m.data))
marginalize(::StructuralPlan, names::Symbol...; kwargs...) = _sfail("marginalize(" *
    join(("`$x`" for x in names), ", ") * "): a bound plan is already lowered. " *
    "Marginalize the model or its binding before observing it: " *
    "`marginalize(model(; inputs...), :x) | (; y)`")

# ── Named filter values ───────────────────────────────────────────────────
#
# `marginalize(model, :x; predictive = (location = :mu, scale = :sd),
# filtered = (mean = :m, variance = :v))` names values of the filter as
# ordinary values of the marginalized program: each observed entry's
# one-step-ahead predictive location, variance or scale, and the filtered
# mean and variance of each array integrated out. They are definitions (a
# per-index output in a `@plate` cell), so `prepare_query` and `jax_source`
# read them by name; nothing is added when nothing is named.

const _PREDICTIVE_ROLES = (:location, :variance, :scale)
const _FILTERED_ROLES = (:mean, :variance)

_marginal_call(arrays) = "marginalize(" * join((":$a" for a in arrays), ", ") * "; …)"

function _marginal_roles(call, spec, allowed, what)
    spec isa Union{NamedTuple,AbstractDict} || _sfail("$call: `$what` names values " *
        "by role, `$what = (" * join(("$r = :name" for r in allowed), ", ") *
        ")`; got `$(repr(spec))`")
    out = Pair{Symbol,Symbol}[]
    for (role, nm) in pairs(spec)
        role = Symbol(role)
        role in allowed || _sfail("$call: `$what` takes " *
            join(("`$r`" for r in allowed), ", ") * "; `$role` is not one of them")
        nm isa Symbol || _sfail("$call: `$what.$role` is the name of the value, a " *
            "Symbol such as `:$(role)_name`; got `$(repr(nm))`")
        push!(out, role => nm)
    end
    return out
end

function _marginal_values_request(arrays, predictive, filtered, earlier)
    call = _marginal_call(arrays)
    pred = _marginal_roles(call, predictive, _PREDICTIVE_ROLES, "predictive")
    filt = Tuple{Symbol,Symbol,Symbol}[]
    filtered isa Union{NamedTuple,AbstractDict} || _sfail("$call: `filtered` names " *
        "values by role, `filtered = (mean = :name, variance = :name)`; got `$(repr(filtered))`")
    if !isempty(filtered) && all(v -> v isa Symbol, values(filtered))
        # One array: `filtered = (mean = :m, variance = :v)`.
        length(arrays) == 1 || _sfail("$call: this call integrates out several arrays; " *
            "name each array's filtered values under it, `filtered = (" *
            join(("$a = (mean = :name,)" for a in arrays), ", ") * ")`")
        for (role, nm) in _marginal_roles(call, filtered, _FILTERED_ROLES, "filtered")
            push!(filt, (only(arrays), role, nm))
        end
    else
        # Per array: `filtered = (level = (mean = :m,), slope = (variance = :v,))`.
        for (a, spec) in pairs(filtered)
            a = Symbol(a)
            a in arrays || _sfail("$call: `filtered` names `$a`, which this call does " *
                "not integrate out; key it by the call's arrays (" *
                join(("`$x`" for x in arrays), ", ") * ")")
            for (role, nm) in _marginal_roles(call, spec, _FILTERED_ROLES, "filtered.$a")
                push!(filt, (a, role, nm))
            end
        end
    end
    isempty(pred) && isempty(filt) && return nothing
    named = Symbol[last.(pred); [t[3] for t in filt]]
    taken = Set{Symbol}(nm for r in earlier
        for nm in (last.(r.predictive)..., (t[3] for t in r.filtered)...))
    seen = Set{Symbol}()
    for nm in named
        startswith(String(nm), "_rkppl") && _sfail("$call: `$nm` starts with `_rkppl`, " *
            "which names RKPPL's own values; choose another name")
        nm in arrays && _sfail("$call: `$nm` is an array this call integrates out; " *
            "name its filter values otherwise")
        (nm in seen || nm in taken) && _sfail("$call: the value name `$nm` is given twice")
        push!(seen, nm)
    end
    return _MarginalValues(arrays, pred, filt)
end

# The requests of a model's `marginalize` calls by lowered array identifier. A
# value name the model already uses (a parameter, definition or input) is
# refused: the named value would shadow it.
function _marginal_value_requests(requests, ctx, used, values)
    out = Dict{Symbol,_MarginalValues}()
    for r in requests
        for nm in (last.(r.predictive)..., (t[3] for t in r.filtered)...)
            (nm in used || haskey(values, nm)) && _sfail("$(_marginal_call(r.arrays)): " *
                "the value name `$nm` is already used in the model; choose another name")
        end
        ids = Symbol[_marginalized_identifier(a, ctx) for a in r.arrays]
        resolved = _MarginalValues(ids, r.predictive,
            [(_marginalized_identifier(a, ctx), role, nm) for (a, role, nm) in r.filtered])
        for id in ids
            out[id] = resolved
        end
    end
    return out
end

# The one request covering the recurrence `members` integrates out together,
# merging requests of several calls; `nothing` when none names a value. A call
# naming values for arrays that different statements observe is refused: each
# observation has its own predictive law.
function _marginal_request_for(requests, members)
    found = _MarginalValues[]
    for x in members
        r = get(requests, x, nothing)
        r === nothing || any(f -> f === r, found) || push!(found, r)
    end
    isempty(found) && return nothing
    for r in found
        issubset(r.arrays, members) || _marginal_fail(first(members),
            "a `marginalize` call naming `predictive` or `filtered` values integrates out " *
            "recurrences observed by different statements (" *
            join(("`$a`" for a in r.arrays), ", ") * "); call `marginalize` once per " *
            "recurrence, `marginalize(marginalize(model, :x; …), :z; …)`")
    end
    pred = Pair{Symbol,Symbol}[p for r in found for p in r.predictive]
    length(unique(first.(pred))) == length(pred) || _marginal_fail(first(members),
        "two `marginalize` calls name its predictive " *
        join(unique(first(p) for p in pred if count(q -> first(q) == first(p), pred) > 1), ", "))
    filt = Tuple{Symbol,Symbol,Symbol}[t for r in found for t in r.filtered]
    slots = [(a, role) for (a, role, _) in filt]
    length(unique(slots)) == length(slots) || _marginal_fail(first(members),
        "two `marginalize` calls name the same filtered value of an array")
    return _MarginalValues(collect(Symbol, members), pred, filt)
end

# The definitions naming a recurrence's filter values: `loc` and `var` are the
# predictive location and variance of each observed entry, `moments(a)` the
# filtered mean and variance of array `a`. `define(name, value)` writes one: a
# top-level definition, or a per-index output in a `@plate` cell.
function _marginal_value_defs(req, loc, var, moments; scale = :(sqrt.($var)),
        define = (n, v) -> Expr(:(=), n, v))
    req === nothing && return Any[]
    defs = Any[]
    for (role, nm) in req.predictive
        push!(defs, define(nm, role === :location ? loc : role === :variance ? var : scale))
    end
    for (a, role, nm) in req.filtered
        mean, v = moments(a)
        push!(defs, define(nm, role === :mean ? mean : v))
    end
    return defs
end
_stores_predictive(req) = req !== nothing && !isempty(req.predictive)

# The identifier a marginalized name lowers under: a top-level array as it is;
# a submodel's local by its author path (`z.x`, `z.w.x`), resolved through the
# call scopes as a conditioning path is.
function _marginalized_identifier(x::Symbol, ctx)
    occursin('.', String(x)) || return x
    parts = Symbol.(split(String(x), '.'))
    nm = _resolve_scope_properties(foldl((a, b) -> Expr(:., a, QuoteNode(b)), parts), ctx)
    nm isa Symbol || _marginal_fail(x, "`$x` names no local of a submodel call; " *
        "name a `@scan` array by its path (`z.x` for `x` of `z ~ sm(…)`)")
    return nm
end

# ── Affine forms ──────────────────────────────────────────────────────────
#
# `constant + Σ coefs[k] * k` over the random leaves `k`. A `nothing`
# constant is zero. Coefficients and constants are value expressions; the
# builders below fold literal zeros and ones and nothing else, so every
# authored operation stays in the program.

struct _Affine
    constant::Any
    coefs::Dict{Any,Any}
end
_Affine(constant) = _Affine(constant, Dict{Any,Any}())

_is_lit(ex, v) = ex isa Real && ex == v
_op(dotted::Bool, op::Symbol) = dotted ? Symbol('.', op) : op

function _madd(a, b, dotted)
    (a === nothing || _is_lit(a, 0)) && return b
    (b === nothing || _is_lit(b, 0)) && return a
    return Expr(:call, _op(dotted, :+), a, b)
end
function _mmul(a, b, dotted)
    (a === nothing || b === nothing) && return nothing
    (_is_lit(a, 0) || _is_lit(b, 0)) && return nothing
    _is_lit(a, 1) && return b
    _is_lit(b, 1) && return a
    return Expr(:call, _op(dotted, :*), a, b)
end
_mneg(a, dotted) = a === nothing ? nothing :
    a isa Real ? -a : Expr(:call, _op(dotted, :-), a)
_mdiv(a, b, dotted) = a === nothing ? nothing : Expr(:call, _op(dotted, :/), a, b)
# A literal seed of a carried value the steps compute in floating point
# (`x[1] = 0`, `Normal(0, 1)`'s variance `1`) is a float, so the carry keeps
# one type.
_mfloat(ex) = ex isa Real ? Float64(ex) : ex

function _aff_add(a::_Affine, b::_Affine, dotted)
    coefs = copy(a.coefs)
    for (k, c) in b.coefs
        coefs[k] = haskey(coefs, k) ? _madd(coefs[k], c, dotted) : c
    end
    return _Affine(_madd(a.constant, b.constant, dotted), coefs)
end
_aff_scale(a::_Affine, f, dotted) = _Affine(_mmul(a.constant, f, dotted),
    Dict{Any,Any}(k => _mmul(c, f, dotted) for (k, c) in a.coefs))
_aff_scale_right(a::_Affine, f, dotted) = _Affine(_mmul(f, a.constant, dotted),
    Dict{Any,Any}(k => _mmul(f, c, dotted) for (k, c) in a.coefs))
_aff_neg(a::_Affine, dotted) = _Affine(_mneg(a.constant, dotted),
    Dict{Any,Any}(k => _mneg(c, dotted) for (k, c) in a.coefs))
_aff_div(a::_Affine, d, dotted) = _Affine(_mdiv(a.constant, d, dotted),
    Dict{Any,Any}(k => _mdiv(c, d, dotted) for (k, c) in a.coefs))

# `leaf(ex)` returns the affine form of a random leaf (or `nothing`);
# `reads(ex)` is true when `ex` reads any random value. Non-random
# subexpressions are constants. `nothing` means not affine.
function _affine(ex, leaf, reads, dotted::Bool)
    l = leaf(ex)
    l === nothing || return l
    reads(ex) || return _Affine(ex)
    Meta.isexpr(ex, :call) && length(ex.args) >= 2 || return nothing
    f = ex.args[1]
    f isa Symbol || return nothing
    s = String(f)
    op = startswith(s, '.') && length(s) > 1 ? Symbol(s[2:end]) : f
    args = ex.args[2:end]
    part(a) = _affine(a, leaf, reads, dotted)
    if op === :+
        acc = _Affine(nothing)
        for a in args
            p = part(a)
            p === nothing && return nothing
            acc = _aff_add(acc, p, dotted)
        end
        return acc
    elseif op === :- && length(args) == 1
        p = part(args[1])
        return p === nothing ? nothing : _aff_neg(p, dotted)
    elseif op === :- && length(args) == 2
        p, q = part(args[1]), part(args[2])
        (p === nothing || q === nothing) && return nothing
        return _aff_add(p, _aff_neg(q, dotted), dotted)
    elseif op === :*
        random = findall(reads, args)
        length(random) == 1 || return nothing
        j = only(random)
        p = part(args[j])
        p === nothing && return nothing
        for (i, a) in enumerate(args)
            i == j && continue
            p = i < j ? _aff_scale_right(p, a, dotted) : _aff_scale(p, a, dotted)
        end
        return p
    elseif op === :/ && length(args) == 2
        reads(args[2]) && return nothing
        p = part(args[1])
        return p === nothing ? nothing : _aff_div(p, args[2], dotted)
    end
    return nothing
end

# ── The recurrence ────────────────────────────────────────────────────────

_marginal_fail(x, msg) = _sfail("marginalize(`$x`): " * msg)

# The carried arrays a `@scan` block seeds, read syntactically (without
# validating an unrelated block).
function _scan_seed_names(block)
    names = Symbol[]
    block isa Expr || return names
    for st in block.args
        st isa LineNumberNode && continue
        Meta.isexpr(st, :for) && break
        lhs = Meta.isexpr(st, :call, 3) && st.args[1] === :~ ? st.args[2] :
            Meta.isexpr(st, :(=), 2) ? st.args[1] : nothing
        Meta.isexpr(lhs, :ref) && lhs.args[1] isa Symbol &&
            !(lhs.args[1] in names) && push!(names, lhs.args[1])
    end
    return names
end
_is_scan_of(st, x) = Meta.isexpr(st, :macrocall) && length(st.args) >= 3 &&
    st.args[1] === Symbol("@scan") && x in _scan_seed_names(st.args[end])

# The Normal law of a sampled seed or step: its location and scale
# expressions. Truncated, restricted and other families are not Gaussian.
function _marginal_normal(x, st, what)
    (st.family === :normal && st.support === nothing && length(st.args) == 2) ||
        _marginal_fail(x, "$what is not an unbounded Normal draw; the Kalman " *
            "filter integrates out a linear-Gaussian recurrence only")
    return st.args[1], st.args[2]
end

"""The transition `x[t] = coef * x[t - 1] + shift + Normal(0, sqrt(innovation_var))`
of a linear-Gaussian recurrence, with its seed `x[1] ~ Normal(m0, sqrt(P0))`
and the deterministic step statements the coefficients read. `hi` is the
loop's upper bound as written: a literal, a length name, or in a `@plate`
cell any data expression (`length(y[g])`)."""
struct _GaussianTransition
    loopvar::Symbol
    hi::Any
    m0::Any
    P0::Any
    locals::Vector{Any}
    coef::Any
    shift::Any
    innovation_var::Any
end

# A `@scan` block's spec and its loop's upper bound as written. A cell's
# bound may be any data expression (`length(y[g])`), which the shared parser
# does not take; it parses a stand-in name in its place, as the cell scan
# lowering does (`_stochastic_cell_scan`).
function _parse_marginal_scan(block)
    hi = _scan_upper_bound(block)
    (hi isa Int || hi isa Symbol || hi === nothing) && return parse_scan_block(block), hi
    parsed = deepcopy(block)
    forx = last(filter(st -> !(st isa LineNumberNode), parsed.args))
    forx.args[1].args[2].args[3] = :_rkppl_marginal_bound
    return parse_scan_block(parsed), hi
end

function _gaussian_transition(spec::ScanSpec, x::Symbol, hi = spec.hi)
    others = setdiff(spec.states, [x])
    isempty(others) || _marginal_fail(x, "its `@scan` also carries " *
        join(("`$o`" for o in others), ", ") * "; a recurrence carrying " *
        "several arrays (a vector state) is not marginalized yet")
    (length(spec.setup) == 1 && spec.lo == 2) || _marginal_fail(x, "it is " *
        "seeded at $(length(spec.setup)) entries; the filter needs one seed " *
        "`$x[1]` and the loop `for $(spec.loopvar) in 2:…` (a lag above 1 is " *
        "a vector state, not marginalized yet)")
    seed = only(spec.setup)
    if seed.kind === :sample
        m0, s0 = _marginal_normal(x, seed, "the seed `$x[1]`")
        P0 = _mmul(s0, s0, false)
    else
        m0, P0 = seed.expr, 0.0
    end
    t = spec.loopvar
    lag = Expr(:ref, x, Expr(:call, :-, t, 1))
    random = Dict{Symbol,_Affine}()
    reads(ex) = _mentions_symbol(ex, x) || any(k -> _mentions_symbol(ex, k), keys(random))
    leaf(ex) = ex == lag ? _Affine(nothing, Dict{Any,Any}(:__lag => 1)) :
        ex isa Symbol && haskey(random, ex) ? random[ex] : nothing
    locals = Any[]
    write = nothing
    nz = 0
    statement(st) = st.kind === :sample ?
        (st.indexed ? "`$(st.target)[$t] ~ …`" : "`$(st.target) ~ …`") :
        (st.indexed ? "`$(st.target)[$t] = $(st.expr)`" : "`$(st.target) = $(st.expr)`")
    nonlinear(st) = _marginal_fail(x, "the step $(statement(st)) is not linear " *
        "in `$lag` and the Normal innovations; the filter needs " *
        "`$x[$t] = a * $lag + c + b * e` with `a`, `b`, `c` not reading them")
    for st in spec.step
        if st.kind === :sample && !st.indexed
            mu, sigma = _marginal_normal(x, st, "the innovation $(statement(st))")
            (reads(mu) || reads(sigma)) && _marginal_fail(x, "the innovation " *
                "$(statement(st)) reads `$x` or another innovation in its arguments")
            nz += 1
            random[st.target] = _Affine(mu, Dict{Any,Any}(Symbol(:__z, nz) => sigma))
        elseif st.kind === :sample
            mu, sigma = _marginal_normal(x, st, "the step $(statement(st))")
            reads(sigma) && _marginal_fail(x, "the scale of the step " *
                "$(statement(st)) reads `$x` or an innovation")
            a = _affine(mu, leaf, reads, false)
            a === nothing && nonlinear(st)
            nz += 1
            write = _aff_add(a, _Affine(nothing, Dict{Any,Any}(Symbol(:__z, nz) => sigma)), false)
        elseif st.indexed
            a = _affine(st.expr, leaf, reads, false)
            a === nothing && nonlinear(st)
            write = a
        elseif reads(st.expr)
            a = _affine(st.expr, leaf, reads, false)
            a === nothing && nonlinear(st)
            random[st.target] = a
        else
            push!(locals, Expr(:(=), st.target, st.expr))
        end
    end
    write === nothing && _marginal_fail(x, "the loop never writes `$x[$t]`")
    # A compound coefficient or innovation scale is read twice (it is
    # squared): a step local computes it once.
    named(ex, role) = ex isa Expr ?
        (push!(locals, Expr(:(=), _marginal_name(x, role), ex)); _marginal_name(x, role)) : ex
    coef = named(get(write.coefs, :__lag, nothing), :step_coef)
    innovation_var = nothing
    scales = sort!([k for k in keys(write.coefs) if k !== :__lag]; by = String)
    for (i, k) in enumerate(scales)
        c = named(write.coefs[k], Symbol(:innovation_scale_, i))
        innovation_var = _madd(innovation_var, _mmul(c, c, false), false)
    end
    return _GaussianTransition(t, hi, m0, P0, locals, coef,
        write.constant, innovation_var)
end

# ── Shapes of the observation's pieces ────────────────────────────────────
#
# A piece of the observation's broadcast (offset, loading, scale) is read
# per entry inside the filter: `piece[t]` for one value per row, `piece` for
# one value. The classification is syntactic over the program's
# declarations, definitions and bound data; a piece it cannot place is
# refused, never guessed.

struct _MarginalNames
    scalars::Set{Symbol}
    arrays::Set{Symbol}
    defs::Dict{Symbol,Any}
    # Arrays written in `@plate` cells: one entry per index, of a shape the
    # syntax does not tell.
    plate_arrays::Set{Symbol}
end

function _marginal_names(ast::Expr)
    scalars, arrays, defs = Set{Symbol}(), Set{Symbol}(), Dict{Symbol,Any}()
    plate_arrays = Set{Symbol}()
    for st in ast.args
        st isa Expr || continue
        if Meta.isexpr(st, :call, 3) && st.args[1] === :~
            lhs = st.args[2]
            lhs isa Symbol ? push!(scalars, lhs) :
                Meta.isexpr(lhs, :ref) && lhs.args[1] isa Symbol && push!(arrays, lhs.args[1])
        elseif Meta.isexpr(st, :call, 3) && st.args[1] === :.~
            lhs = st.args[2]
            base = Meta.isexpr(lhs, :ref) ? lhs.args[1] : lhs
            base isa Symbol && push!(arrays, base)
        elseif Meta.isexpr(st, :(=), 2) && st.args[1] isa Symbol
            defs[st.args[1]] = st.args[2]
        elseif Meta.isexpr(st, :macrocall) && st.args[1] === Symbol("@scan")
            union!(arrays, _scan_seed_names(st.args[end]))
        elseif Meta.isexpr(st, :macrocall) && st.args[1] === Symbol("@plate")
            _plate_written_arrays!(plate_arrays, st.args[end])
        end
    end
    return _MarginalNames(scalars, union(arrays, plate_arrays), defs, plate_arrays)
end

function _plate_written_arrays!(arrays, ex)
    ex isa Expr || return arrays
    if (Meta.isexpr(ex, :(=), 2) || (Meta.isexpr(ex, :call, 3) && ex.args[1] in (:~, :.~)))
        lhs = ex.head === :(=) ? ex.args[1] : ex.args[2]
        Meta.isexpr(lhs, :ref) && lhs.args[1] isa Symbol && push!(arrays, lhs.args[1])
    end
    foreach(a -> _plate_written_arrays!(arrays, a), ex.args)
    return arrays
end

const _MARGINAL_SCALAR_CALLS = (:sum, :mean, :std, :var, :minimum, :maximum,
    :length, :prod, :first, :last, :dot, :norm)
const _MARGINAL_ARRAY_CALLS = (:fill, :zeros, :ones, :collect, :vcat, :hcat,
    :repeat, :range, :reverse, :cumsum, :cumprod, :diff)

_combine_shapes(shapes) = :unknown in shapes ? :unknown :
    :vector in shapes ? :vector : :scalar

# The shape of `ex`: `:scalar` (one value), `:vector` (one value per entry of
# the observation) or `:unknown`. `leaf(ex)` places names and indexed reads
# (or returns `nothing` for an expression it does not place); literals,
# broadcasts and calls are combined here.
function _shape_of(ex, leaf)
    ex isa Real && return :scalar
    l = leaf(ex)
    l === nothing || return l
    ex isa Expr || return :unknown
    shape(a) = _shape_of(a, leaf)
    if ex.head === :. && length(ex.args) == 2 && Meta.isexpr(ex.args[2], :tuple)
        return _combine_shapes([shape(a) for a in ex.args[2].args])
    elseif ex.head === :call && !isempty(ex.args)
        f = ex.args[1]
        if f isa GlobalRef
            f.name === :_bound_value && return :scalar
            return :unknown
        end
        f isa Symbol || return :unknown
        args = ex.args[2:end]
        startswith(String(f), '.') && length(String(f)) > 1 &&
            return _combine_shapes([shape(a) for a in args])
        f in _MARGINAL_SCALAR_CALLS && return :scalar
        f in _MARGINAL_ARRAY_CALLS && return :unknown
        shapes = [shape(a) for a in args]
        :unknown in shapes && return :unknown
        all(==(:scalar), shapes) && return :scalar
        f in (:*, :/) && count(==(:vector), shapes) == 1 && return :vector
        f in (:+, :-) && all(==(:vector), shapes) && return :vector
        return :unknown
    end
    return :unknown
end

# A bound value: one number, or a vector of numbers. A vector of vectors
# holds one array per index, which only a cell reads (below).
_value_shape(v) = v isa Number ? :scalar :
    v isa AbstractVector && all(e -> e isa Number, v) ? :vector : :unknown

# The shape of one entry `v[g]` of a bound value read in a `@plate` cell.
_entry_shape(v) = !(v isa AbstractVector) ? :unknown :
    eltype(v) <: AbstractVector{<:Number} ? :vector :
    all(e -> e isa Number, v) ? :scalar :
    all(e -> _value_shape(e) === :vector, v) ? :vector : :unknown

# Shapes of the whole program's names: declarations, definitions, bound data.
function _toplevel_shape_leaf(names::_MarginalNames, values)
    active = Set{Symbol}()
    function leaf(ex)
        if ex isa Symbol
            haskey(values, ex) && return _value_shape(values[ex])
            ex in names.scalars && return :scalar
            ex in names.arrays && return :vector
            if haskey(names.defs, ex)
                ex in active && return :unknown
                push!(active, ex)
                sh = _shape_of(names.defs[ex], leaf)
                delete!(active, ex)
                return sh
            end
            ex in (:pi, :π, :ℯ, :Inf, :NaN) && return :scalar
            return :unknown
        elseif Meta.isexpr(ex, :ref)
            _shape_of(ex.args[1], leaf) === :vector || return :unknown
            idx = ex.args[2:end]
            any(i -> i === :(:), idx) && return :vector
            return _combine_shapes([_shape_of(i, leaf) for i in idx])
        end
        return nothing
    end
    return leaf
end

# Shapes inside a `@plate for ivar in …` cell, relative to one index: a
# per-index entry `a[ivar]` of a vector of numbers (bound data, a declared
# array, a vector definition) is one value; of bound data holding one vector
# per index it is one value per entry of the observation. Cell-local
# definitions and per-index outputs are read through their values; a name of
# the whole program counts only as one value.
function _cell_shape_leaf(names::_MarginalNames, values, ivar, defs, outputs, drawn)
    top = _toplevel_shape_leaf(names, values)
    active = Set{Symbol}()
    through(name, value) = begin
        name in active && return :unknown
        push!(active, name)
        sh = _shape_of(value, leaf)
        delete!(active, name)
        sh
    end
    function leaf(ex)
        if ex isa Symbol
            (ex === ivar || ex in drawn) && return :scalar
            haskey(defs, ex) && return through(ex, defs[ex])
            return top(ex) === :scalar ? :scalar : :unknown
        elseif Meta.isexpr(ex, :ref, 2) && ex.args[2] === ivar && ex.args[1] isa Symbol
            a = ex.args[1]
            haskey(outputs, a) && return through(a, outputs[a])
            haskey(values, a) && return _entry_shape(values[a])
            a in names.plate_arrays && return :unknown
            return top(a) === :vector ? :scalar : :unknown
        elseif Meta.isexpr(ex, :ref)
            return :unknown
        end
        return nothing
    end
    return leaf
end

# A piece's value at the seed (`piece[1]` or `piece`) and at step `t`, and
# with `square` its square at each. Compound pieces are named once as
# definitions in `defs` (top-level, or cell-local in a cell); a one-value
# piece's square too, so the loop does not recompute it.
function _marginal_piece!(defs, piece, role, x, y, leaf, t; square = false)
    piece === nothing && return (nothing, nothing, nothing, nothing)
    piece isa Real && return (piece, piece, piece * piece, piece * piece)
    shape = _shape_of(piece, leaf)
    shape === :unknown && _marginal_fail(x, "cannot tell whether the observation's " *
        "$role `$piece` is one value or one value per entry of `$y`; write it " *
        "from declared parameters, bound data and definitions of them")
    name = piece
    if !(piece isa Symbol)
        name = _marginal_name(x, role)
        push!(defs, Expr(:(=), name, piece))
    end
    if shape === :scalar
        square || return (name, name, nothing, nothing)
        sq = _marginal_name(x, Symbol(role, :_squared))
        push!(defs, Expr(:(=), sq, Expr(:call, :*, name, name)))
        return (name, name, sq, sq)
    end
    at(i) = Expr(:ref, name, i)
    return (at(1), at(t), Expr(:call, :*, at(1), at(1)), Expr(:call, :*, at(t), at(t)))
end

# ── The rewrite ───────────────────────────────────────────────────────────

# An unsupplied length name (`for t in 2:T`) is the observed array's length
# (rkppl-use §5). The filter observes that array, but the transition scan for
# the draws observes nothing, so the length is bound to the observation's
# length here, before data definitions are formed, as a supplied `T` is.
function _marginalized_lengths!(values, data, ast::Expr, marginalized)
    for x in marginalized
        stmts = ast.args
        k = findfirst(st -> _is_scan_of(st, x), stmts)
        k === nothing && continue
        hi = _scan_upper_bound(stmts[k].args[end])
        _marginalized_events!(values, data, stmts, k, x, hi)
        (hi isa Symbol && !haskey(values, hi)) || continue
        inlined, _ = _inline_latent_definitions(stmts, k, x)
        for st in inlined
            (Meta.isexpr(st, :call, 3) && st.args[1] === :.~ && st.args[2] isa Symbol &&
                _mentions_symbol(st.args[3], x)) || continue
            _observation_gather(st, x) === nothing || break
            v = get(values, st.args[2], nothing)
            if v isa AbstractVector
                values[hi] = length(v)
                push!(data, hi)
            end
            break
        end
    end
    return values
end

# Rows observing the state at their steps (`y .~ Normal.(… x[step] …)`): the
# filter runs over events, one per step and one per row, so their count
# `steps + rows` is bound here, with a supplied length (rows do not say how
# many steps there are).
function _marginalized_events!(values, data, stmts, k, x, hi)
    inlined, _ = _inline_latent_definitions(stmts, k, x)
    for st in inlined
        idx = _observation_gather(st, x)
        idx === nothing && continue
        steps = hi isa Integer ? hi : get(values, hi, nothing)
        rows = get(values, idx, nothing)
        (steps isa Integer && rows isa AbstractVector) || return
        E = _marginal_name(x, :events)
        values[E] = steps + length(rows)
        push!(data, E)
        return
    end
end

# The events of rows observing steps `step` of a recurrence of `T` steps, in
# order: each step, then each of its rows in their order. `_rkppl_marginal_
# event_rows` holds each event's row (0 for a step), `…_event_steps` its step,
# `…_row_events` each row's event, `…_step_ends` each step's last event.
function _rkppl_marginal_event_rows(step::AbstractVector{<:Integer}, T::Integer)
    all(s -> 1 <= s <= T, step) || throw(ContractValidationError("[bind] a row " *
        "observes a step outside 1:$T of its marginalized recurrence"))
    out = zeros(Int, T + length(step))
    for (i, e) in enumerate(_rkppl_marginal_row_events(step, T))
        out[e] = i
    end
    return out
end
_rkppl_marginal_row_events(step::AbstractVector{<:Integer}, T::Integer) =
    step .+ invperm(sortperm(step))
function _rkppl_marginal_event_steps(step::AbstractVector{<:Integer}, T::Integer)
    rows = _rkppl_marginal_event_rows(step, T)
    return cumsum(rows .== 0)
end
_rkppl_marginal_step_ends(step::AbstractVector{<:Integer}, T::Integer) =
    [s + count(<=(s), step) for s in 1:T]

# Rows observing the state at their steps, `y .~ Normal.(offset .+ loading .*
# x[step], noise)`: the filter runs over events. A step's event applies its
# transition (from the transition scan) and observes nothing; a row's event
# keeps the state and observes the row. Each row's one-step-ahead predictive
# law, at its event, is its observation; the draws read the moments at each
# step's last event.
function _marginalize_gathered(stmts, k, j, x, law, y, form, scale, idx, ast, values, bound,
        req = nothing)
    E = _marginal_name(x, :events)
    (bound && !(Missing <: eltype(values[y]))) || _marginal_fail(x, "its observation " *
        "`$y` reads `$x[$idx]`; rows observing their steps are marginalized when `$y` " *
        "is bound data without missing entries")
    rows = get(values, idx, nothing)
    (rows isa AbstractVector{<:Integer} && length(rows) == length(values[y])) ||
        _marginal_fail(x, "its observation reads `$x[$idx]`; `$idx` must be bound " *
            "integer data, one step per entry of `$y`")
    haskey(values, E) || _marginal_fail(x, "the length `$(law.hi)` of its `@scan` is " *
        "not supplied; rows observing their steps (`$x[$idx]`) do not say how many " *
        "steps there are, so supply it (`model(; $(law.hi) = …)`)")
    T = law.hi
    leaf = _toplevel_shape_leaf(_marginal_names(ast), values)
    name(role) = _marginal_name(x, role)
    ER, ES, RE, ENDS = name(:event_rows), name(:event_steps), name(:row_events), name(:step_ends)
    OBS, G, EY = name(:event_observes), name(:event_row), name(:event_observed)
    C, D, Q = (name(r) for r in (:coef, :shift, :innovation_var))
    EC, ED, EQ = name(:event_coef), name(:event_shift), name(:event_innovation_var)
    call(f, args...) = Expr(:call, GlobalRef(@__MODULE__, f), args...)
    defs = Any[
        :($ER = $(call(:_rkppl_marginal_event_rows, idx, T))),
        :($ES = $(call(:_rkppl_marginal_event_steps, idx, T))),
        :($RE = $(call(:_rkppl_marginal_row_events, idx, T))),
        :($ENDS = $(call(:_rkppl_marginal_step_ends, idx, T))),
        :($OBS = $ER .> 0),
        :($G = max.($ER, 1)),
        :($EY = $y[$G])]
    pieces = _event_pieces!(defs, x, y, form, scale, leaf, G)
    transition_reads = Any[:($EC = $C[$ES]), :($ED = $D[$ES]), :($EQ = $Q[$ES])]
    filter = _event_filter_scan(x, E, OBS, EY, EC, ED, EQ, D, Q, pieces)
    EM, EV = name(:event_location), name(:event_variance)
    PL, PV = name(:predictive_location), name(:predictive_variance)
    tail = Any[filter,
        :($y .~ Normal.($PL[$RE], sqrt.($PV[$RE]))),
        # The draws' filtered moments: each step's last event.
        :($(name(:mean)) = $EM[$ENDS]), :($(name(:var)) = $EV[$ENDS])]
    # Named values: each row's predictive law at its event, in row order, and
    # the filtered moments per step.
    append!(tail, _marginal_value_defs(req, :($PL[$RE]), :($PV[$RE]),
        _ -> (name(:mean), name(:var))))
    out = Any[]
    for (i, st) in enumerate(stmts)
        i == k && continue
        if i == j
            push!(out, _transition_scan(x, law))
            append!(out, defs, transition_reads, tail)
            continue
        end
        push!(out, st)
    end
    return Expr(:block, out...)
end

# The observation's pieces at event `e`: a one-value piece as it is, a piece
# with one value per row gathered at each event's row (`piece[G]`).
function _event_pieces!(defs, x, y, form, scale, leaf, G)
    e = _marginal_name(x, :event)
    function piece(p, role; square = false)
        p === nothing && return (nothing, nothing)
        p isa Real && return (p, p * p)
        shape = _shape_of(p, leaf)
        shape === :unknown && _marginal_fail(x, "cannot tell whether the observation's " *
            "$role `$p` is one value or one value per entry of `$y`; write it " *
            "from declared parameters, bound data and definitions of them")
        n = p
        if !(p isa Symbol)
            n = _marginal_name(x, role)
            push!(defs, Expr(:(=), n, p))
        end
        if shape === :scalar
            square || return (n, nothing)
            sq = _marginal_name(x, Symbol(role, :_squared))
            push!(defs, Expr(:(=), sq, Expr(:call, :*, n, n)))
            return (n, sq)
        end
        ev = _marginal_name(x, Symbol(:event_, role))
        push!(defs, Expr(:(=), ev, Expr(:ref, n, G)))
        at = Expr(:ref, ev, e)
        return (at, Expr(:call, :*, at, at))
    end
    O, _ = piece(form.constant, :offset)
    L, LL = piece(form.coefs[x], :loading; square = true)
    _, SS = piece(scale, :scale; square = true)
    return (O, L, LL, SS)
end

# The filter over events: event 1 is step 1's seed law; a step's event
# predicts through its transition, a row's event updates on its row (its
# gain is the event's `OBS`).
function _event_filter_scan(x, E, OBS, EY, EC, ED, EQ, D, Q, pieces)
    e = _marginal_name(x, :event)
    name(role) = _marginal_name(x, role)
    EM, EV = name(:event_location), name(:event_variance)
    PL, PV = name(:predictive_location), name(:predictive_variance)
    a, c, w = name(:event_step_coef), name(:event_step_shift), name(:event_step_var)
    pm, pv, g = name(:predicted_location), name(:predicted_variance), name(:gain)
    at(v) = Expr(:ref, v, e)
    prev(v) = Expr(:ref, v, Expr(:call, :-, e, 1))
    _, gain, mean_e, var_e, loc_e, fv_e = _kalman_update(at(EY), pm, pv, pieces, g;
        present = at(OBS))
    body = Expr(:block,
        :($a = ifelse($(at(OBS)), 1.0, $(at(EC)))),
        :($c = ifelse($(at(OBS)), 0.0, $(at(ED)))),
        :($w = ifelse($(at(OBS)), 0.0, $(at(EQ)))),
        :($pm = $a * $(prev(EM)) + $c),
        :($pv = $a * $a * $(prev(EV)) + $w),
        Expr(:(=), at(PL), loc_e), Expr(:(=), at(PV), fv_e),
        Expr(:(=), g, gain), Expr(:(=), at(EM), mean_e), Expr(:(=), at(EV), var_e))
    loop = Expr(:for, Expr(:(=), e, Expr(:call, :(:), 2, E)), body)
    return _scan_macro(Expr(:block,
        :($(EM)[1] = $D[1]), :($(EV)[1] = $Q[1]),
        :($(PL)[1] = 0.0), :($(PV)[1] = 1.0), loop))
end

# The upper bound of a `@scan` block's loop, read syntactically.
function _scan_upper_bound(block)
    for st in block.args
        Meta.isexpr(st, :for) || continue
        range = st.args[1]
        Meta.isexpr(range, :(=), 2) && Meta.isexpr(range.args[2], :call, 3) &&
            range.args[2].args[1] === :(:) || return nothing
        return range.args[2].args[3]
    end
    return nothing
end

function _marginalize_latents(ast::Expr, marginalized, values, observing,
        requests = Dict{Symbol,_MarginalValues}())
    done = Set{Symbol}()
    for x in marginalized
        x in done && continue
        k = findfirst(st -> _is_scan_of(st, x), ast.args)
        if k === nothing
            p = findfirst(st -> _marginal_cell_scan(st, x) !== nothing, ast.args)
            if p === nothing
                nested = any(st -> _plate_loop(st) !== nothing && _mentions_symbol(st, x), ast.args)
                _marginal_fail(x, nested ?
                    "its `@scan` is not a statement of a `@plate` cell (a recurrence in a " *
                    "nested `@plate` is not marginalized yet)" :
                    "no top-level `@scan` and no `@plate` cell `@scan` carries `$x`")
            end
            ast, members = _marginalize_cell_latent(ast, p, x, marginalized, values, observing, requests)
        else
            spec = parse_scan_block(ast.args[k].args[end])
            group = _vector_group(spec, x, marginalized)
            members = something(group, Symbol[x])
            req = _marginal_request_for(requests, members)
            ast = if group === nothing
                _marginalize_latent(ast, k, x, _gaussian_transition(spec, x), values, observing, req)
            else
                _marginalize_vector_latent(ast, k, _vector_transition(spec, group), values, observing, req)
            end
        end
        union!(done, members)
    end
    return ast
end

function _marginalize_latent(ast, k, x, law, values, observing, req)
    stmts = ast.args
    stmts, k = _inline_latent_definitions(stmts, k, x)
    j = _marginal_reader(x, stmts, k)
    y, loc, scale, form, idx = _marginal_observation(x, stmts[j], y -> y isa Symbol, "y")
    bound = y in observing && get(values, y, nothing) isa AbstractVector
    idx === nothing || return _marginalize_gathered(stmts, k, j, x, law, y, form, scale,
        idx, ast, values, bound, req)
    # Otherwise a vector the model computes without `x` (data augmentation:
    # measured rows merged with bounded latents of censored rows), which the
    # filter reads entry by entry.
    bound || any(st -> Meta.isexpr(st, :(=), 2) && st.args[1] === y, stmts) ||
        _marginal_fail(x, "its observation `$y` is neither observed data nor a " *
            "definition; marginalize a recurrence observed through bound data " *
            "(`| (; $y)`) or through a vector the model defines (`$y = …`)")
    bound || !(law.hi isa Symbol) || haskey(values, law.hi) ||
        _marginal_fail(x, "the length `$(law.hi)` of its `@scan` is not supplied; " *
            "when the observed vector `$y` is a definition, supply it " *
            "(`model(; $(law.hi) = …)`)")
    leaf = _toplevel_shape_leaf(_marginal_names(ast), values)
    t = law.loopvar
    defs = Any[]
    seed, step = _marginal_pieces!(defs, x, y, form, scale, leaf, t)
    filter = if !bound
        _filter_scan(x, y, law, seed, step; response = y)
    elseif Missing <: eltype(values[y])
        # Missing entries: the filter reads the presence of each entry and
        # the values with absent entries filled; an absent entry only
        # predicts (its gain is zero). The response keeps its presence mask,
        # so its likelihood skips the absent entries.
        present, filled = _marginal_name(x, :present), _marginal_name(x, :observed)
        push!(defs, :($present = $(GlobalRef(@__MODULE__, :_rkppl_presence))($y)),
            :($filled = $(GlobalRef(@__MODULE__, :_rkppl_fill_missing))($y, 0.0)))
        _filter_scan(x, filled, law, seed, step; present, response = y)
    else
        # Observed entry by entry; the predictive moments are carried only
        # when the author names them.
        (_filter_scan(x, y, law, seed, step; store = _stores_predictive(req)),)
    end
    transition = _transition_scan(x, law)
    named = _marginal_value_defs(req, _marginal_name(x, :predictive_location),
        _marginal_name(x, :predictive_variance),
        _ -> (_marginal_name(x, :mean), _marginal_name(x, :var)))
    out = Any[]
    for (i, st) in enumerate(stmts)
        i == k && continue
        if i == j
            append!(out, defs)
            push!(out, filter..., transition, named...)
            continue
        end
        push!(out, st)
    end
    return Expr(:block, out...)
end

# The one statement besides its scan (statement `k`) that reads `x`.
function _marginal_reader(x, stmts, k; observation = "y .~ Normal.(offset .+ loading .* $x, noise)")
    readers = [j for (j, st) in enumerate(stmts)
        if j != k && st isa Expr && _mentions_symbol(st, x)]
    length(readers) == 1 || _marginal_fail(x, "every read of `$x` must be the " *
        "location of one Normal observation `$observation`; " *
        (isempty(readers) ? "nothing reads it" : "it is read by " *
            join(("`$(stmts[j])`" for j in readers), ", ")))
    return only(readers)
end

# The Normal observation `y .~ Normal.(offset .+ loading .* x, noise)` that
# reads `x`: its response `y` (which `response(y)` accepts), location, scale
# and the location's affine form in `x`.
function _marginal_observation(x, obs, response, spelled)
    wrong() = _marginal_fail(x, "it is read by `$obs`; the filter needs a Normal " *
        "observation of a whole observed vector, `$spelled .~ Normal.(offset .+ " *
        "loading .* $x, noise)`, with `offset`, `loading` and `noise` not reading `$x`")
    (Meta.isexpr(obs, :call, 3) && obs.args[1] === :.~) || wrong()
    y, rhs = obs.args[2], obs.args[3]
    (Meta.isexpr(rhs, :., 2) && rhs.args[1] === :Normal &&
        Meta.isexpr(rhs.args[2], :tuple) && length(rhs.args[2].args) in (1, 2)) || wrong()
    response(y) || wrong()
    loc = rhs.args[2].args[1]
    scale = length(rhs.args[2].args) == 2 ? rhs.args[2].args[2] : 1.0
    _mentions_symbol(scale, x) && wrong()
    readsx(ex) = _mentions_symbol(ex, x)
    # `x` itself, or one gather `x[idx]` of it by a name (each row observes
    # the state at its step); never both, never two gathers.
    gathers = Set{Any}()
    function leafx(ex)
        whole = ex === x
        gathered = Meta.isexpr(ex, :ref, 2) && ex.args[1] === x && ex.args[2] isa Symbol
        (whole || gathered) || return nothing
        push!(gathers, whole ? nothing : ex.args[2])
        return _Affine(nothing, Dict{Any,Any}(x => 1))
    end
    form = _affine(loc, leafx, readsx, true)
    (form === nothing || !haskey(form.coefs, x) || length(gathers) != 1) && wrong()
    return y, loc, scale, form, only(gathers)
end

# The gather `idx` of an observation `y .~ Normal.(… x[idx] …)` reading `x`
# through one, read syntactically (`nothing` for any other statement).
function _observation_gather(st, x)
    (Meta.isexpr(st, :call, 3) && st.args[1] === :.~) || return nothing
    found = Ref{Any}(nothing)
    visit(ex) = ex isa Expr && (Meta.isexpr(ex, :ref, 2) && ex.args[1] === x &&
        ex.args[2] isa Symbol ? (found[] = ex.args[2]) : foreach(visit, ex.args))
    visit(st.args[3])
    return found[]
end

# The observation's offset, loading and scale at the seed and at step `t`.
function _marginal_pieces!(defs, x, y, form, scale, leaf, t)
    O1, Ot = _marginal_piece!(defs, form.constant, :offset, x, y, leaf, t)
    L1, Lt, LL1, LLt = _marginal_piece!(defs, form.coefs[x], :loading, x, y, leaf, t;
        square = true)
    _, _, SS1, SSt = _marginal_piece!(defs, scale, :scale, x, y, leaf, t; square = true)
    return (O1, L1, LL1, SS1), (Ot, Lt, LLt, SSt)
end

# ── A recurrence per `@plate` index ───────────────────────────────────────
#
# In `@plate for g in …`, a cell `@scan` carrying `x` and one observation
# `y[g] .~ Normal.(offset .+ loading .* x, noise)` of bound data holding one
# vector per index become the cell's Kalman filter: the observed cell scan
# `y[g][t] ~ …` (rkppl-use §5 "Observing a carried array") carrying the
# filtered mean and variance. A second plate over the same indices, which
# the density does not read, runs the filter again and records each index's
# filtered moments and transition as per-index outputs for the draws.

_plate_loop(st) = Meta.isexpr(st, :macrocall) && length(st.args) == 3 &&
    st.args[1] === Symbol("@plate") && Meta.isexpr(st.args[3], :for) &&
    Meta.isexpr(st.args[3].args[1], :(=), 2) && st.args[3].args[1].args[1] isa Symbol &&
    Meta.isexpr(st.args[3].args[2], :block) ? st.args[3] : nothing

function _is_plate_scan_of(st, x)
    loop = _plate_loop(st)
    loop === nothing && return false
    return any(c -> _is_scan_of(c, x), loop.args[2].args)
end

# The cell carried arrays of the filter (in the density's plate) and of the
# draws' plate, and the per-index outputs the draws query reads.
_marginal_cell_carried(x, role) = _marginal_name(x, Symbol(:cell_, role))
_marginal_draw_carried(x, role) = _marginal_name(x, Symbol(:draw_, role))

# A cell submodel writes a carried array as a column of trajectories,
# `state[g][t]`. Distinguish that from an ordinary cell-local `state[t]`.
function _marginal_cell_scan(st, x)
    loop = _plate_loop(st)
    loop === nothing && return nothing
    ref = Expr(:ref, x, loop.args[1].args[1])
    return findfirst(loop.args[2].args) do c
        _is_cell_scan_block(c) && any(_observed_scan_setups(c.args[end])) do seed
            lhs = _observed_scan_lhs(seed)
            Meta.isexpr(lhs, :ref, 2) && lhs.args[1] in (x, ref)
        end
    end
end

function _marginalize_cell_latent(ast, p, x, marginalized, values, observing, requests)
    st = ast.args[p]
    loop = _plate_loop(st)
    head = loop.args[1]
    ivar = head.args[1]
    cell = copy(loop.args[2].args)
    k = _marginal_cell_scan(st, x)
    others = Any[c for (i, c) in enumerate(ast.args) if i != p]
    indexed = Set(lhs.args[1] for seed in _observed_scan_setups(cell[k].args[end])
        for lhs in (_observed_scan_lhs(seed),) if Meta.isexpr(lhs, :ref, 2) &&
        Meta.isexpr(lhs.args[1], :ref, 2) && lhs.args[1].args[2] === ivar)
    for ref in indexed
        name = ref.args[1]
        name in observing && _marginal_fail(x, "`$name` is observed data, not a latent array")
        any(c -> _mentions_symbol(c, name), others) && _marginal_fail(x,
            "its carried array `$name` is read outside the plate")
        # Other indices or the whole column cannot be integrated independently.
        function localize(ex)
            ex == ref && return name
            ex === name && _marginal_fail(x, "its cell reads `$name` outside `$ref`")
            return ex isa Expr ? Expr(ex.head, map(localize, ex.args)...) : ex
        end
        cell = map(localize, cell)
    end
    spec, hi = _parse_marginal_scan(cell[k].args[end])
    group = _vector_group(spec, x, marginalized)
    members = something(group, Symbol[x])
    req = _marginal_request_for(requests, members)
    for a in members
        cell, k = _inline_cell_latent_definitions(cell, k, a, ivar, others)
    end
    response(y) = Meta.isexpr(y, :ref, 2) && y.args[1] isa Symbol && y.args[2] === ivar
    if group === nothing
        j = _marginal_reader(x, cell, k;
            observation = "y[$ivar] .~ Normal.(offset .+ loading .* $x, noise)")
        y, _, scale, form, idx = _marginal_observation(x, cell[j], response, "y[$ivar]")
        idx === nothing || _marginal_fail(x, "its observation `$y` reads `$x[$idx]`; " *
            "gathered rows in a `@plate` cell are not marginalized yet (at top level " *
            "each row observes the state at its step)")
    else
        j, y, form, scale = _vector_reader(cell, k, group, response, "y[$ivar]")
    end
    yname = y.args[1]
    yvalue = get(values, yname, nothing)
    yvalue isa AbstractVector && any(e -> e isa AbstractVector && Missing <: eltype(e), yvalue) &&
        _marginal_fail(x, "its observation `$y` admits `missing` entries; a missing " *
            "entry in a `@plate` cell's series is not marginalized yet (at top level it " *
            "is a step that only predicts)")
    (yname in observing && _entry_shape(yvalue) === :vector) ||
        _marginal_fail(x, "its observation `$y` is not observed data holding one " *
            "vector per index; marginalize a recurrence observed through bound data " *
            "(`| (; $yname)`, one vector per `$ivar`)")
    defs = Dict{Symbol,Any}(c.args[1] => c.args[2] for c in cell
        if Meta.isexpr(c, :(=), 2) && c.args[1] isa Symbol)
    outputs = Dict{Symbol,Any}(c.args[1].args[1] => c.args[2] for c in cell
        if Meta.isexpr(c, :(=), 2) && Meta.isexpr(c.args[1], :ref, 2) &&
            c.args[1].args[1] isa Symbol && c.args[1].args[2] === ivar)
    # Scalars drawn in the cell: one value per index (the draws' plate refuses
    # them below).
    drawn = Set{Symbol}(c.args[2] for c in cell
        if Meta.isexpr(c, :call, 3) && c.args[1] === :~ && c.args[2] isa Symbol)
    leaf = _cell_shape_leaf(_marginal_names(ast), values, ivar, defs, outputs, drawn)
    if group === nothing
        law = _gaussian_transition(spec, x, hi)
        pieces = Any[]
        seed, step = _marginal_pieces!(pieces, x, y, form, scale, leaf, law.loopvar)
        filter = _filter_scan(x, y, law, seed, step;
            M = _marginal_cell_carried(x, :mean), V = _marginal_cell_carried(x, :var))
        replacement = Any[pieces..., filter]
        side = Any[_cell_draws_plate(x, y, law, seed, step, head, cell, pieces, k, j, observing, req)]
    else
        law = _vector_transition(spec, group, hi)
        replacement = _vector_filter_statements(law, y, form, scale, leaf, req; ivar)
        side = Any[]
    end
    density_cell = copy(cell)
    deleteat!(density_cell, k)
    reader = k < j ? j - 1 : j
    splice!(density_cell, reader:reader, replacement)
    out = copy(ast.args)
    splice!(out, p:p, Any[Expr(:macrocall, st.args[1:2]...,
        Expr(:for, head, Expr(:block, density_cell...))), side...])
    return Expr(:block, out...), members
end

# The cell form of `_inline_latent_definitions`: a cell definition reading
# the latent (`mu = offset .+ loading .* x`) or a per-index output of one
# (`loc[g] = x`) that only the observation reads is substituted there. A
# per-index output read anywhere else in the program has no value once the
# latent is integrated out, so it is refused.
function _inline_cell_latent_definitions(cell, k, x, ivar, others)
    cell = copy(cell)
    while true
        readers = [j for (j, c) in enumerate(cell)
            if j != k && c isa Expr && _mentions_symbol(c, x)]
        length(readers) == 1 || return cell, k
        j = only(readers)
        c = cell[j]
        Meta.isexpr(c, :(=), 2) || return cell, k
        lhs, value = c.args
        output = Meta.isexpr(lhs, :ref, 2) && lhs.args[1] isa Symbol && lhs.args[2] === ivar
        (lhs isa Symbol || output) || return cell, k
        d = output ? lhs.args[1] : lhs
        users = [i for (i, s) in enumerate(cell)
            if i != j && s isa Expr && _mentions_symbol(s, d)]
        outside = output ? [s for s in others if s isa Expr && _mentions_symbol(s, d)] : Any[]
        (length(users) == 1 && isempty(outside)) || _marginal_fail(x, "its definition " *
            "`$c` is read by " * (isempty(users) && isempty(outside) ? "nothing" :
                join(("`$s`" for s in (cell[users]..., outside...)), ", ")) *
            "; a definition reading `$x` must be read by its observation alone")
        i = only(users)
        replaced = _substitute_expr(cell[i], output ? lhs : d, value)
        _mentions_symbol(replaced, d) && _marginal_fail(x, "its definition `$c` is " *
            "read as `$d` in `$(cell[i])`; read it as `$lhs` in the observation")
        cell[i] = replaced
        deleteat!(cell, j)
        j < k && (k -= 1)
    end
end

# The plate of the draws: the cell's definitions the filter needs, the
# observation's pieces, and a deterministic scan running the filter and
# recording the transition, returned per index. The density reads none of it.
# Values the author named (`req`) are per-index outputs of this plate, so the
# density's cell is unchanged.
function _cell_draws_plate(x, y, law, seed, step, head, cell, pieces, k, j, observing,
        req = nothing)
    ivar = head.args[1]
    carried = Tuple(_marginal_draw_carried(x, r) for r in _MARGINAL_ROLES)
    predictive = _stores_predictive(req) ? (_marginal_draw_carried(x, :predictive_location),
        _marginal_draw_carried(x, :predictive_variance)) : nothing
    scan = _cell_draws_scan(x, y, law, seed, step, carried; predictive)
    outputs = [Expr(:(=), Expr(:ref, _marginal_name(x, r), ivar), c)
        for (r, c) in zip(_MARGINAL_ROLES, carried)]
    loc, var = something(predictive, (nothing, nothing))
    append!(outputs, _marginal_value_defs(req, loc, var, _ -> (carried[1], carried[2]);
        define = (n, v) -> Expr(:(=), Expr(:ref, n, ivar), v)))
    return _cell_side_plate(head, cell, (k, j), observing, Any[pieces..., scan], outputs) do drawn
        _marginal_fail(x, "its transition or observation reads " *
            join(("`$d`" for d in drawn), ", ") * ", drawn inside the `@plate` cell; " *
            "draw it outside the cell (a declared array read as `$(first(drawn))[$ivar]`)")
    end
end

# A second plate over a cell's indices that the density does not read: the
# cell's definitions `body` needs (statements `skip` excluded), then `body`,
# then the per-index `outputs`. A value drawn inside the cell has no draw in
# this plate, so a read of one calls `refuse` with the names.
function _cell_side_plate(refuse, head, cell, skip, observing, body, outputs)
    needed = Set{Symbol}()
    _all_symbols!(needed, Expr(:block, body..., outputs...))
    sampled = Set{Symbol}()
    for (i, c) in enumerate(cell)
        i in skip && continue
        Meta.isexpr(c, :call, 3) && c.args[1] in (:~, :.~) || continue
        lhs = c.args[2]
        base = Meta.isexpr(lhs, :ref) ? lhs.args[1] : lhs
        # An observation of bound data is data to this plate.
        base isa Symbol && !(base in observing) && push!(sampled, base)
    end
    kept = Set{Int}()
    changed = true
    while changed
        changed = false
        for (i, c) in enumerate(cell)
            (i in skip || i in kept) && continue
            (Meta.isexpr(c, :(=), 2) && c.args[1] isa Symbol && c.args[1] in needed) ||
                continue
            push!(kept, i)
            _all_symbols!(needed, c.args[2])
            changed = true
        end
    end
    drawn = intersect(needed, sampled)
    isempty(drawn) || refuse(sort!(collect(drawn)))
    stmts = Any[cell[i] for i in sort!(collect(kept))]
    append!(stmts, body, outputs)
    return Expr(:macrocall, Symbol("@plate"), LineNumberNode(0),
        Expr(:for, head, Expr(:block, stmts...)))
end

# `predictive` names carried arrays `(location, variance)` recording each
# entry's one-step-ahead predictive law, or is `nothing`.
function _cell_draws_scan(x, y, law::_GaussianTransition, seed_pieces, step_pieces, carried;
        predictive = nothing)
    t = law.loopvar
    M, V, C, D, Q = carried
    _, _, mean1, var1, loc1, fv1 = _kalman_update(:($y[1]), law.m0, law.P0, seed_pieces,
        nothing)
    pm, pv, g = _marginal_name(x, :predicted_mean),
        _marginal_name(x, :predicted_var), _marginal_name(x, :gain)
    prev(a) = Expr(:ref, a, Expr(:call, :-, t, 1))
    pm_ex = something(_madd(_mmul(law.coef, prev(M), false), law.shift, false), 0.0)
    pv_ex = something(_madd(_mmul(_mmul(law.coef, law.coef, false), prev(V), false),
        law.innovation_var, false), 0.0)
    _, gain, meant, vart, loct, fvt = _kalman_update(:($y[$t]), pm, pv, step_pieces, g)
    record(i, loc, fv) = predictive === nothing ? () :
        (Expr(:(=), Expr(:ref, predictive[1], i), loc),
            Expr(:(=), Expr(:ref, predictive[2], i), fv))
    body = Expr(:block, law.locals...,
        Expr(:(=), pm, pm_ex), Expr(:(=), pv, pv_ex), record(t, loct, fvt)...,
        Expr(:(=), g, gain),
        Expr(:(=), :($M[$t]), meant), Expr(:(=), :($V[$t]), vart),
        Expr(:(=), :($C[$t]), _mfloat(something(law.coef, 0.0))),
        Expr(:(=), :($D[$t]), _mfloat(something(law.shift, 0.0))),
        Expr(:(=), :($Q[$t]), _mfloat(something(law.innovation_var, 0.0))))
    loop = Expr(:for, Expr(:(=), t, Expr(:call, :(:), 2, law.hi)), body)
    return _scan_macro(Expr(:block, record(1, loc1, fv1)...,
        Expr(:(=), :($M[1]), mean1), Expr(:(=), :($V[1]), var1),
        Expr(:(=), :($C[1]), 0.0), Expr(:(=), :($D[1]), _mfloat(something(law.m0, 0.0))),
        Expr(:(=), :($Q[1]), _mfloat(law.P0)), loop))
end

# A definition that reads the latent (`mu = offset .+ loading .* x`) and is
# read by exactly one statement is part of that statement's value: it
# is substituted there, so the observation reads the latent directly. Once
# the latent is integrated out such a definition has no value of the
# parameters, so it leaves the program.
function _inline_latent_definitions(stmts, k, x)
    stmts = copy(stmts)
    while true
        readers = [j for (j, st) in enumerate(stmts)
            if j != k && st isa Expr && _mentions_symbol(st, x)]
        length(readers) == 1 || return stmts, k
        j = only(readers)
        st = stmts[j]
        (Meta.isexpr(st, :(=), 2) && st.args[1] isa Symbol) || return stmts, k
        d, value = st.args[1], st.args[2]
        users = [i for (i, s) in enumerate(stmts)
            if i != j && s isa Expr && _mentions_symbol(s, d)]
        length(users) == 1 || _marginal_fail(x, "its definition `$st` is read " *
            (isempty(users) ? "by nothing" : "by " *
                join(("`$(stmts[i])`" for i in users), ", ")) *
            "; a definition reading `$x` must be read by its observation alone")
        i = only(users)
        stmts[i] = _substitute_name(stmts[i], d, value)
        deleteat!(stmts, j)
        j < k && (k -= 1)
    end
end

_substitute_name(ex, name, value) = ex === name ? value :
    ex isa Expr ? Expr(ex.head, (_substitute_name(a, name, value) for a in ex.args)...) : ex
# Replace every occurrence of the expression `pattern` (`loc[g]`) by `value`.
_substitute_expr(ex, pattern, value) = ex == pattern ? value :
    ex isa Expr ? Expr(ex.head, (_substitute_expr(a, pattern, value) for a in ex.args)...) : ex

_scan_macro(block) = Expr(:macrocall, Symbol("@scan"), LineNumberNode(0), block)

# One Kalman update at the predicted state mean `pm` and variance `pv`, for an
# observation with offset `O`, loading `L` (squared `LL`) and squared scale
# `SS`: the one-step-ahead predictive law, the gain, and the filtered mean and
# variance, which read the gain through `gain` (its expression in a seed, its
# local in a step).
function _kalman_update(yt, pm, pv, (O, L, LL, SS), gain; present = nothing)
    fv = something(_madd(_mmul(LL, pv, false), SS, false), 0.0)
    loc = something(_madd(O, _mmul(L, pm, false), false), 0.0)
    predictive = Expr(:call, :~, yt, Expr(:call, :Normal, loc, Expr(:call, :sqrt, fv)))
    gain_ex = Expr(:call, :/, something(_mmul(L, pv, false), 0.0), fv)
    # An absent entry (`present` false) updates nothing: the filtered state
    # is the prediction.
    present === nothing || (gain_ex = Expr(:call, :*, present, gain_ex))
    gain === nothing && (gain = gain_ex)
    filtered_mean = _madd(pm, _mmul(gain, Expr(:call, :-, yt, loc), false), false)
    filtered_var = Expr(:call, :-, pv, something(_mmul(gain,
        _mmul(L, pv, false), false), 0.0))
    return predictive, gain_ex, filtered_mean, filtered_var, loc, fv
end

#
# Observed through bound data without missing entries, the scan observes
# each entry, `y[t] ~ Normal(…)` (an observed carried array). Otherwise
# (`response`, read by the filter as the values `y`), the scan carries each
# entry's one-step-ahead predictive mean and variance, and an ordinary
# observation `response .~ Normal.(mean, sqrt.(var))` follows it; `present`
# names a vector whose false entries only predict. `store` also carries the
# predictive mean and variance beside an entry-by-entry observation (values the
# author named). Returns the statements.
function _filter_scan(x, y, law::_GaussianTransition, seed_pieces, step_pieces;
        M = _marginal_name(x, :mean), V = _marginal_name(x, :var), present = nothing,
        response = nothing, store = false)
    t = law.loopvar
    entry(i) = present === nothing ? nothing : Expr(:ref, present, i)
    # Not `…_mean`: `_marginalized_names` finds latents by their filtered mean.
    PM, PV = _marginal_name(x, :predictive_location), _marginal_name(x, :predictive_variance)
    moments(i, loc, fv) = (Expr(:(=), Expr(:ref, PM, i), loc), Expr(:(=), Expr(:ref, PV, i), fv))
    observe(obs, i, loc, fv) = response === nothing ?
        (store ? (obs, moments(i, loc, fv)...) : (obs,)) : moments(i, loc, fv)
    obs1, _, mean1, var1, loc1, fv1 = _kalman_update(:($y[1]), law.m0, law.P0,
        seed_pieces, nothing; present = entry(1))
    pm, pv, g = _marginal_name(x, :predicted_mean),
        _marginal_name(x, :predicted_var), _marginal_name(x, :gain)
    prev(a) = Expr(:ref, a, Expr(:call, :-, t, 1))
    pm_ex = something(_madd(_mmul(law.coef, prev(M), false), law.shift, false), 0.0)
    pv_ex = something(_madd(_mmul(_mmul(law.coef, law.coef, false), prev(V), false),
        law.innovation_var, false), 0.0)
    obst, gain, meant, vart, loct, fvt = _kalman_update(:($y[$t]), pm, pv, step_pieces, g;
        present = entry(t))
    body = Expr(:block, law.locals...,
        Expr(:(=), pm, pm_ex), Expr(:(=), pv, pv_ex), observe(obst, t, loct, fvt)...,
        Expr(:(=), g, gain), Expr(:(=), :($M[$t]), meant), Expr(:(=), :($V[$t]), vart))
    loop = Expr(:for, Expr(:(=), t, Expr(:call, :(:), 2, law.hi)), body)
    scan = _scan_macro(Expr(:block, observe(obs1, 1, loc1, fv1)...,
        Expr(:(=), :($M[1]), mean1), Expr(:(=), :($V[1]), var1), loop))
    response === nothing && return scan
    return (scan, :($response .~ Normal.($PM, sqrt.($PV))))
end

function _transition_scan(x, law::_GaussianTransition)
    t = law.loopvar
    C, D, Q = (_marginal_name(x, r) for r in (:coef, :shift, :innovation_var))
    body = Expr(:block, law.locals...,
        Expr(:(=), :($C[$t]), _mfloat(something(law.coef, 0.0))),
        Expr(:(=), :($D[$t]), _mfloat(something(law.shift, 0.0))),
        Expr(:(=), :($Q[$t]), _mfloat(something(law.innovation_var, 0.0))))
    loop = Expr(:for, Expr(:(=), t, Expr(:call, :(:), 2, law.hi)), body)
    return _scan_macro(Expr(:block,
        Expr(:(=), :($C[1]), 0.0),
        Expr(:(=), :($D[1]), _mfloat(something(law.m0, 0.0))),
        Expr(:(=), :($Q[1]), _mfloat(law.P0)), loop))
end

# ── Exact draws of a marginalized recurrence ─────────────────────────────

"""
    MarginalizedLatentQuery

A prepared query of the Kalman filter of every marginalized recurrence
(`prepare_marginalized_latents`). `q(u)` returns, keyed by latent name, the
filtered means and variances and the transition the filter used;
[`marginalized_latent_draws`](@ref) draws the latents. Not thread-safe:
prepare one per thread.
"""
# A vector state's group (marginalize_vector.jl): its key, its window's
# components `(array, j)`, structural transition and noise entries
# `(i, k)`, and whether these fields hold one series per plate index.
const _MarginalGroup = Tuple{Symbol,Vector{Tuple{Symbol,Int}},Vector{Tuple{Int,Int}},
    Vector{Tuple{Int,Int}},Bool}

struct MarginalizedLatentQuery{Q}
    names::Vector{Symbol}
    query::Q
    # Vector states (marginalize_vector.jl).
    groups::Vector{_MarginalGroup}
    # Each lowered name's author path: a submodel's local reads as `z.x`.
    labels::Dict{Symbol,Symbol}
end

# The marginalized latents of a lowered program: those whose filter scan
# (carrying `_rkppl_marginal_<x>_mean`) it holds at top level, whose event
# filter's per-step moments it defines (`_rkppl_marginal_<x>_mean = …`), or
# whose draws' plate returns `_rkppl_marginal_<x>_mean[g]` per index.
function _marginalized_names(plan::StructuralPlan)
    out = Symbol[]
    suffix = "_mean"
    names = Iterators.flatten(((s for sp in plan.scans for s in sp.states),
        (d.name for d in plan.derived), (a.name for a in plan.assignments)))
    for s in names
        str = String(s)
        startswith(str, _MARGINAL_PREFIX) && endswith(str, suffix) || continue
        push!(out, Symbol(str[length(_MARGINAL_PREFIX)+1:end-length(suffix)]))
    end
    return out
end

"""
    prepare_marginalized_latents(built, plan) -> MarginalizedLatentQuery

Prepare the Kalman filter of every recurrence [`marginalize`](@ref) integrated
out of `plan`, on the graph `built` (`build_kernel` of the same binding).
`q(u)` returns a `NamedTuple` keyed by latent name; each entry holds the
filtered `mean[t]` and `var[t]` of the state given the first `t`
observations, and the transition `x[t] = coef[t] * x[t - 1] + shift[t] +
Normal(0, sqrt(innovation_var[t]))` (entry 1 holds the seed's law: `coef[1]
= 0`). A recurrence per `@plate` index holds one such vector per index in
each field. A vector state is one entry keyed by its first array: its
window's `components` (`(array, j)` for `array[t - j]`), its seed depth
`lags`, and per step the window's filtered `mean` and `cov`, and the
transition into it, `coef` and `shift` with noise covariance
`innovation_cov` (entries up to `lags` hold seeds). In a plate the per-step
fields are grouped by plate index. A recurrence a submodel holds is keyed by
its author path (`Symbol("z.x")`). At posterior draws of the parameters these
determine each latent's exact conditional posterior; [`marginalized_latent_draws`](@ref)
samples it. A program with no marginalized recurrence is refused.
"""
function prepare_marginalized_latents(built, plan::StructuralPlan; kwargs...)
    names = _marginalized_names(plan)
    groups = _marginalized_groups(plan)
    isempty(names) && isempty(groups) && throw(ContractValidationError("[query] this program " *
        "integrates out no recurrence: marginalize one with `marginalize(model, :x)`"))
    wants = Tuple([(_marginal_name(x, r) for x in names for r in _MARGINAL_ROLES)...,
        (w for gr in groups for w in _vector_wants(gr...))...])
    paths = _scope_name_paths(plan.submodel_scopes)
    labels = Dict{Symbol,Symbol}(nm => Symbol(join(paths[nm], '.')) for nm in
        Iterators.flatten((names, (a for gr in groups for (a, _) in gr[2]))) if haskey(paths, nm))
    return MarginalizedLatentQuery(names, prepare_query(built, plan, wants; kwargs...), groups,
        labels)
end
prepare_marginalized_latents(built, m::_PlanSource; kwargs...) =
    prepare_marginalized_latents(built, _bound_plan(m); kwargs...)

function (q::MarginalizedLatentQuery)(u::AbstractVector, discrete::Vararg{NamedTuple})
    values = Base.invokelatest(q.query, u, discrete...)
    k = length(_MARGINAL_ROLES)
    entries = Any[NamedTuple{_MARGINAL_ROLES}(Tuple(values[(i - 1) * k + r] for r in 1:k))
        for i in eachindex(q.names)]
    next = k * length(q.names)
    label(nm) = get(q.labels, nm, nm)
    for (g, comps, A, Q, per_cell) in q.groups
        n = length(_vector_wants(g, comps, A, Q))
        f = _vector_filtered(values[next+1:next+n], comps, A, Q, per_cell)
        push!(entries, merge(f, (; components = map(((a, j),) -> (label(a), j), f.components))))
        next += n
    end
    latents = Tuple(label(nm) for nm in (q.names..., (first(gr) for gr in q.groups)...))
    return NamedTuple{latents}(Tuple(entries))
end

"""
    marginalized_latent_draws(q::MarginalizedLatentQuery, u, rng) -> NamedTuple
    marginalized_latent_draws(filtered::NamedTuple, rng) -> NamedTuple

One exact draw of every marginalized recurrence given the parameters `u`,
keyed by latent name (`Vector{Float64}`, one value per entry; for a
recurrence per `@plate` index, one such vector per index), by forward
filtering, backward sampling over `q(u)`. Only `rng` supplies randomness, in
key order and then index order, so a seeded `rng` reproduces the draws. At posterior draws of the
parameters, parameters and latents together are exact draws from the joint
posterior of the model as written.
"""
marginalized_latent_draws(q::MarginalizedLatentQuery, u::AbstractVector, rng) =
    marginalized_latent_draws(q(u), rng)
marginalized_latent_draws(q::MarginalizedLatentQuery, u::AbstractVector,
    discrete::NamedTuple, rng) = marginalized_latent_draws(q(u, discrete), rng)
function marginalized_latent_draws(filtered::NamedTuple, rng)
    out = Pair{Symbol,Any}[]
    for (x, f) in pairs(filtered)
        # A vector state draws its window jointly, then each of its arrays.
        haskey(f, :components) ? append!(out, _backward_sample_vector(rng, f)) :
            push!(out, x => _backward_sample(rng, f))
    end
    return _nest_paths(out)
end

# Draws keyed by author path: a submodel's local `z.x` nests as `draws.z.x`,
# as `constrain` nests sampled locals.
function _nest_paths(pairs)
    out = Pair{Symbol,Any}[]
    nested = Dict{Symbol,Vector{Pair{Symbol,Any}}}()
    for (name, value) in pairs
        head, rest = let s = String(name), i = findfirst('.', s)
            i === nothing ? (name, nothing) : (Symbol(s[1:i-1]), Symbol(s[i+1:end]))
        end
        if rest === nothing
            push!(out, name => value)
        else
            haskey(nested, head) || push!(out, head => nothing)
            push!(get!(nested, head, Pair{Symbol,Any}[]), rest => value)
        end
    end
    return (; (k => (v === nothing ? _nest_paths(nested[k]) : v) for (k, v) in out)...)
end

# A recurrence per `@plate` index holds one filtered series per index: each
# is drawn on its own, in index order.
_backward_sample(rng, f) = f.mean isa AbstractVector{<:Number} ? _backward_sample_series(rng, f) :
    Vector{Float64}[_backward_sample_series(rng, map(r -> r[i], f)) for i in eachindex(f.mean)]

# Backward sampling: `x[T]` from its filtered law, then each `x[t]` from its
# law given the filtered `x[t]` and the drawn `x[t + 1]`.
function _backward_sample_series(rng, f)
    m, P, a, c, q = f.mean, f.var, f.coef, f.shift, f.innovation_var
    T = length(m)
    x = Vector{Float64}(undef, T)
    T == 0 && return x
    x[T] = m[T] + sqrt(_nonnegative_var(P[T], T)) * randn(rng)
    for t in (T - 1):-1:1
        predicted = a[t + 1]^2 * P[t] + q[t + 1]
        if predicted > 0
            gain = P[t] * a[t + 1] / predicted
            mean = m[t] + gain * (x[t + 1] - a[t + 1] * m[t] - c[t + 1])
            var = P[t] - gain * a[t + 1] * P[t]
        else
            mean, var = m[t], P[t]
        end
        x[t] = mean + sqrt(_nonnegative_var(var, t)) * randn(rng)
    end
    return x
end

# Rounding may leave a zero variance slightly negative; anything else is a
# filter that has failed (a non-finite or clearly negative variance).
function _nonnegative_var(v, t)
    isfinite(v) && v >= -1e-12 * (1 + abs(v)) || throw(ArgumentError(
        "[query] the filtered state variance at entry $t is $v"))
    return max(v, 0.0)
end
