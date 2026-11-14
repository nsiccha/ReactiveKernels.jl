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
    marginalize(model, names::Symbol...) -> model

Integrate the latent `@scan` arrays `names` out of `model` exactly, as a Kalman
filter. `model` is an `RKPPLModel` or a binding of one (`model(; inputs...)`);
the result composes with `|` and `condition` as the model does. The model
source keeps the recurrence as written.

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
  `y .~ Normal.(offset .+ loading .* x, noise)`.

Any other form is refused with a `SurfaceLoweringError` naming the
statement. Each scalar coefficient may vary by step (irregular gaps
`exp(-gap[t] / range)`, a reset `r = 0` where a new series starts).

The marginalized program's density is the exact marginal density of the
observations given the remaining parameters, so `coordinate_names` and the
layout no longer contain `x` or its innovations. `:pointwise.y` holds the
one-step-ahead predictive densities, whose sum is the marginal likelihood.
Exact draws of `x` given the parameters come from
[`prepare_marginalized_latents`](@ref) and
[`marginalized_latent_draws`](@ref).

Marginalize the model or its binding before observing it:
`marginalize(model(; inputs...), :x) | (; y)`. A bound plan (the result of
`|`) is already lowered and is refused.
"""
function marginalize(m::RKPPLModel, names::Symbol...)
    isempty(names) && throw(ArgumentError(
        "marginalize needs the latent array to integrate out: `marginalize(model, :x)`"))
    out = copy(m.marginalized)
    for x in names
        occursin('.', String(x)) && _sfail("marginalize(`$x`): a recurrence " *
            "inside a submodel is not marginalized yet; name a top-level `@scan` array")
        x in out && _sfail("marginalize(`$x`): `$x` is already marginalized")
        push!(out, x)
    end
    return RKPPLModel(m.ast, m.mod, copy(m.fixed), copy(m.rewrites),
        copy(m.conditioned), copy(m.documentation), out)
end
marginalize(m::RKPPLBoundModel, names::Symbol...) =
    RKPPLBoundModel(marginalize(m.model, names...), copy(m.data))
marginalize(::StructuralPlan, names::Symbol...) = _sfail("marginalize(" *
    join(("`$x`" for x in names), ", ") * "): a bound plan is already lowered. " *
    "Marginalize the model or its binding before observing it: " *
    "`marginalize(model(; inputs...), :x) | (; y)`")

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
and the deterministic step statements the coefficients read."""
struct _GaussianTransition
    loopvar::Symbol
    hi::Union{Symbol,Int}
    m0::Any
    P0::Any
    locals::Vector{Any}
    coef::Any
    shift::Any
    innovation_var::Any
end

function _gaussian_transition(spec::ScanSpec, x::Symbol)
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
    return _GaussianTransition(t, spec.hi, m0, P0, locals, coef,
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
end

function _marginal_names(ast::Expr)
    scalars, arrays, defs = Set{Symbol}(), Set{Symbol}(), Dict{Symbol,Any}()
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
            _plate_written_arrays!(arrays, st.args[end])
        end
    end
    return _MarginalNames(scalars, arrays, defs)
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

function _marginal_shape(ex, names::_MarginalNames, values, active = Set{Symbol}())
    ex isa Real && return :scalar
    if ex isa Symbol
        if haskey(values, ex)
            v = values[ex]
            return v isa Number ? :scalar : v isa AbstractVector ? :vector : :unknown
        end
        ex in names.scalars && return :scalar
        ex in names.arrays && return :vector
        if haskey(names.defs, ex)
            ex in active && return :unknown
            push!(active, ex)
            sh = _marginal_shape(names.defs[ex], names, values, active)
            delete!(active, ex)
            return sh
        end
        ex in (:pi, :π, :ℯ, :Inf, :NaN) && return :scalar
        return :unknown
    end
    ex isa Expr || return :unknown
    shape(a) = _marginal_shape(a, names, values, active)
    combine(shapes) = :unknown in shapes ? :unknown :
        :vector in shapes ? :vector : :scalar
    if ex.head === :ref
        shape(ex.args[1]) === :vector || return :unknown
        idx = ex.args[2:end]
        any(i -> i === :(:), idx) && return :vector
        return combine([shape(i) for i in idx])
    elseif ex.head === :. && length(ex.args) == 2 && Meta.isexpr(ex.args[2], :tuple)
        return combine([shape(a) for a in ex.args[2].args])
    elseif ex.head === :call && !isempty(ex.args)
        f = ex.args[1]
        if f isa GlobalRef
            f.name === :_bound_value && return :scalar
            return :unknown
        end
        f isa Symbol || return :unknown
        args = ex.args[2:end]
        startswith(String(f), '.') && length(String(f)) > 1 &&
            return combine([shape(a) for a in args])
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

# A piece's value at the seed (`piece[1]` or `piece`) and at step `t`, and
# with `square` its square at each. Compound pieces are named once as
# top-level definitions; a one-value piece's square too, so the loop does not
# recompute it.
function _marginal_piece!(defs, piece, role, x, y, names, values, t; square = false)
    piece === nothing && return (nothing, nothing, nothing, nothing)
    piece isa Real && return (piece, piece, piece * piece, piece * piece)
    shape = _marginal_shape(piece, names, values)
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
        (hi isa Symbol && !haskey(values, hi)) || continue
        inlined, _ = _inline_latent_definitions(stmts, k, x)
        for st in inlined
            (Meta.isexpr(st, :call, 3) && st.args[1] === :.~ && st.args[2] isa Symbol &&
                _mentions_symbol(st.args[3], x)) || continue
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

function _marginalize_latents(ast::Expr, marginalized, values, observing)
    for x in marginalized
        ast = _marginalize_latent(ast, x, values, observing)
    end
    return ast
end

function _marginalize_latent(ast::Expr, x::Symbol, values, observing)
    stmts = ast.args
    k = findfirst(st -> _is_scan_of(st, x), stmts)
    if k === nothing
        nested = any(st -> Meta.isexpr(st, :macrocall) && st.args[1] === Symbol("@plate") &&
            _mentions_symbol(st, x), stmts)
        _marginal_fail(x, nested ?
            "a recurrence inside a `@plate` cell is not marginalized yet" :
            "no top-level `@scan` carries `$x`")
    end
    law = _gaussian_transition(parse_scan_block(stmts[k].args[end]), x)
    stmts, k = _inline_latent_definitions(stmts, k, x)
    readers = [j for (j, st) in enumerate(stmts)
        if j != k && st isa Expr && _mentions_symbol(st, x)]
    length(readers) == 1 || _marginal_fail(x, "every read of `$x` must be the " *
        "location of one Normal observation `y .~ Normal.(offset .+ loading .* $x, " *
        "noise)`; " * (isempty(readers) ? "nothing reads it" : "it is read by " *
        join(("`$(stmts[j])`" for j in readers), ", ")))
    j = only(readers)
    obs = stmts[j]
    wrong() = _marginal_fail(x, "it is read by `$obs`; the filter needs a Normal " *
        "observation of a whole observed vector, `y .~ Normal.(offset .+ loading .* $x, noise)`, " *
        "with `offset`, `loading` and `noise` not reading `$x`")
    (Meta.isexpr(obs, :call, 3) && obs.args[1] === :.~) || wrong()
    y, rhs = obs.args[2], obs.args[3]
    (Meta.isexpr(rhs, :., 2) && rhs.args[1] === :Normal &&
        Meta.isexpr(rhs.args[2], :tuple) && length(rhs.args[2].args) in (1, 2)) || wrong()
    y isa Symbol || wrong()
    (y in observing && get(values, y, nothing) isa AbstractVector) ||
        _marginal_fail(x, "its observation `$y` is not observed data; " *
            "marginalize a recurrence observed through bound data (`| (; $y)`)")
    loc = rhs.args[2].args[1]
    scale = length(rhs.args[2].args) == 2 ? rhs.args[2].args[2] : 1.0
    _mentions_symbol(scale, x) && wrong()
    readsx(ex) = _mentions_symbol(ex, x)
    leafx(ex) = ex === x ? _Affine(nothing, Dict{Any,Any}(x => 1)) : nothing
    form = _affine(loc, leafx, readsx, true)
    (form === nothing || !haskey(form.coefs, x)) && wrong()
    names = _marginal_names(ast)
    t = law.loopvar
    defs = Any[]
    O1, Ot = _marginal_piece!(defs, form.constant, :offset, x, y, names, values, t)
    L1, Lt, LL1, LLt = _marginal_piece!(defs, form.coefs[x], :loading, x, y, names,
        values, t; square = true)
    _, _, SS1, SSt = _marginal_piece!(defs, scale, :scale, x, y, names, values, t;
        square = true)
    filter = _filter_scan(x, y, law, (O1, L1, LL1, SS1), (Ot, Lt, LLt, SSt))
    transition = _transition_scan(x, law)
    out = Any[]
    for (i, st) in enumerate(stmts)
        i == k && continue
        if i == j
            append!(out, defs)
            push!(out, filter, transition)
            continue
        end
        push!(out, st)
    end
    return Expr(:block, out...)
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

_scan_macro(block) = Expr(:macrocall, Symbol("@scan"), LineNumberNode(0), block)

# One Kalman update at the predicted state mean `pm` and variance `pv`, for an
# observation with offset `O`, loading `L` (squared `LL`) and squared scale
# `SS`: the one-step-ahead predictive law, the gain, and the filtered mean and
# variance, which read the gain through `gain` (its expression in a seed, its
# local in a step).
function _kalman_update(yt, pm, pv, (O, L, LL, SS), gain)
    fv = something(_madd(_mmul(LL, pv, false), SS, false), 0.0)
    loc = something(_madd(O, _mmul(L, pm, false), false), 0.0)
    predictive = Expr(:call, :~, yt, Expr(:call, :Normal, loc, Expr(:call, :sqrt, fv)))
    gain_ex = Expr(:call, :/, something(_mmul(L, pv, false), 0.0), fv)
    gain === nothing && (gain = gain_ex)
    filtered_mean = _madd(pm, _mmul(gain, Expr(:call, :-, yt, loc), false), false)
    filtered_var = Expr(:call, :-, pv, something(_mmul(gain,
        _mmul(L, pv, false), false), 0.0))
    return predictive, gain_ex, filtered_mean, filtered_var
end

function _filter_scan(x, y, law::_GaussianTransition, seed_pieces, step_pieces)
    t = law.loopvar
    M, V = _marginal_name(x, :mean), _marginal_name(x, :var)
    obs1, _, mean1, var1 = _kalman_update(:($y[1]), law.m0, law.P0, seed_pieces, nothing)
    pm, pv, g = _marginal_name(x, :predicted_mean),
        _marginal_name(x, :predicted_var), _marginal_name(x, :gain)
    prev(a) = Expr(:ref, a, Expr(:call, :-, t, 1))
    pm_ex = something(_madd(_mmul(law.coef, prev(M), false), law.shift, false), 0.0)
    pv_ex = something(_madd(_mmul(_mmul(law.coef, law.coef, false), prev(V), false),
        law.innovation_var, false), 0.0)
    obst, gain, meant, vart = _kalman_update(:($y[$t]), pm, pv, step_pieces, g)
    body = Expr(:block, law.locals...,
        Expr(:(=), pm, pm_ex), Expr(:(=), pv, pv_ex), obst, Expr(:(=), g, gain),
        Expr(:(=), :($M[$t]), meant), Expr(:(=), :($V[$t]), vart))
    loop = Expr(:for, Expr(:(=), t, Expr(:call, :(:), 2, law.hi)), body)
    return _scan_macro(Expr(:block, obs1,
        Expr(:(=), :($M[1]), mean1), Expr(:(=), :($V[1]), var1), loop))
end

function _transition_scan(x, law::_GaussianTransition)
    t = law.loopvar
    C, D, Q = (_marginal_name(x, r) for r in (:coef, :shift, :innovation_var))
    body = Expr(:block, law.locals...,
        Expr(:(=), :($C[$t]), something(law.coef, 0.0)),
        Expr(:(=), :($D[$t]), something(law.shift, 0.0)),
        Expr(:(=), :($Q[$t]), something(law.innovation_var, 0.0)))
    loop = Expr(:for, Expr(:(=), t, Expr(:call, :(:), 2, law.hi)), body)
    return _scan_macro(Expr(:block,
        Expr(:(=), :($C[1]), 0.0),
        Expr(:(=), :($D[1]), something(law.m0, 0.0)),
        Expr(:(=), :($Q[1]), law.P0), loop))
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
struct MarginalizedLatentQuery{Q}
    names::Vector{Symbol}
    query::Q
end

# The marginalized latents of a lowered program: those whose filter scan
# (carrying `_rkppl_marginal_<x>_mean`) it holds.
function _marginalized_names(plan::StructuralPlan)
    out = Symbol[]
    suffix = "_mean"
    for sp in plan.scans, s in sp.states
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
= 0`). At posterior draws of the parameters these determine each latent's
exact conditional posterior; [`marginalized_latent_draws`](@ref) samples it.
A program with no marginalized recurrence is refused.
"""
function prepare_marginalized_latents(built, plan::StructuralPlan; kwargs...)
    names = _marginalized_names(plan)
    isempty(names) && throw(ContractValidationError("[query] this program " *
        "integrates out no recurrence: marginalize one with `marginalize(model, :x)`"))
    wants = Tuple(_marginal_name(x, r) for x in names for r in _MARGINAL_ROLES)
    return MarginalizedLatentQuery(names, prepare_query(built, plan, wants; kwargs...))
end
prepare_marginalized_latents(built, m::_PlanSource; kwargs...) =
    prepare_marginalized_latents(built, _bound_plan(m); kwargs...)

function (q::MarginalizedLatentQuery)(u::AbstractVector)
    values = Base.invokelatest(q.query, u)
    k = length(_MARGINAL_ROLES)
    entries = map(enumerate(q.names)) do (i, x)
        NamedTuple{_MARGINAL_ROLES}(Tuple(values[(i - 1) * k + r] for r in 1:k))
    end
    return NamedTuple{Tuple(q.names)}(Tuple(entries))
end

"""
    marginalized_latent_draws(q::MarginalizedLatentQuery, u, rng) -> NamedTuple
    marginalized_latent_draws(filtered::NamedTuple, rng) -> NamedTuple

One exact draw of every marginalized recurrence given the parameters `u`,
keyed by latent name (`Vector{Float64}`, one value per entry), by forward
filtering, backward sampling over `q(u)`. Only `rng` supplies randomness, in
key order, so a seeded `rng` reproduces the draws. At posterior draws of the
parameters, parameters and latents together are exact draws from the joint
posterior of the model as written.
"""
marginalized_latent_draws(q::MarginalizedLatentQuery, u::AbstractVector, rng) =
    marginalized_latent_draws(q(u), rng)
marginalized_latent_draws(filtered::NamedTuple, rng) =
    map(f -> _backward_sample(rng, f), filtered)

# Backward sampling: `x[T]` from its filtered law, then each `x[t]` from its
# law given the filtered `x[t]` and the drawn `x[t + 1]`.
function _backward_sample(rng, f)
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
