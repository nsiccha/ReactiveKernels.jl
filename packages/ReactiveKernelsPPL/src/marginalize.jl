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
  `y .~ Normal.(offset .+ loading .* x, noise)`. A `missing` entry of `y` is
  a step that only predicts: the filter carries the state through it
  without an update, and the likelihood has no term for it. `y` may also be
  a vector the model defines without reading `x`, such as measured rows
  merged in time order with bounded latents of censored ones (data
  augmentation: `gap[1:K] .~ restricted.(Flat.(), 0, Inf)`,
  `y = vcat(measured, limit .- gap)[order]`). The filter then observes each
  entry at its value, and the scan's length is supplied.

Any other form is refused with a `SurfaceLoweringError` naming the
statement. Each scalar coefficient may vary by step (irregular gaps
`exp(-gap[t] / range)`, a reset `r = 0` where a new series starts).

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
marginal likelihood.
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
        x in m.chains && _sfail("marginalize(`$x`): `$x` is a Markov chain (`markov_chains`)")
        push!(out, x)
    end
    return RKPPLModel(m.ast, m.mod, copy(m.fixed), copy(m.rewrites),
        copy(m.conditioned), copy(m.documentation), out, copy(m.chains))
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
        p = findfirst(st -> _is_plate_scan_of(st, x), stmts)
        p === nothing || return _marginalize_cell_latent(ast, p, x, values, observing)
        nested = any(st -> Meta.isexpr(st, :macrocall) && st.args[1] === Symbol("@plate") &&
            _mentions_symbol(st, x), stmts)
        _marginal_fail(x, nested ?
            "its `@scan` is not a statement of a `@plate` cell (a recurrence in a " *
            "nested `@plate` is not marginalized yet)" :
            "no top-level `@scan` and no `@plate` cell `@scan` carries `$x`")
    end
    law = _gaussian_transition(parse_scan_block(stmts[k].args[end]), x)
    stmts, k = _inline_latent_definitions(stmts, k, x)
    j = _marginal_reader(x, stmts, k)
    y, loc, scale, form = _marginal_observation(x, stmts[j], y -> y isa Symbol, "y")
    bound = y in observing && get(values, y, nothing) isa AbstractVector
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
        (_filter_scan(x, y, law, seed, step),)
    end
    transition = _transition_scan(x, law)
    out = Any[]
    for (i, st) in enumerate(stmts)
        i == k && continue
        if i == j
            append!(out, defs)
            push!(out, filter..., transition)
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
    leafx(ex) = ex === x ? _Affine(nothing, Dict{Any,Any}(x => 1)) : nothing
    form = _affine(loc, leafx, readsx, true)
    (form === nothing || !haskey(form.coefs, x)) && wrong()
    return y, loc, scale, form
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

function _marginalize_cell_latent(ast::Expr, p, x::Symbol, values, observing)
    stmts = ast.args
    loop = _plate_loop(stmts[p])
    head = loop.args[1]
    ivar = head.args[1]
    cell = Any[c for c in loop.args[2].args if !(c isa LineNumberNode)]
    k = findfirst(c -> _is_scan_of(c, x), cell)
    spec, hi = _parse_marginal_scan(cell[k].args[end])
    law = _gaussian_transition(spec, x, hi)
    others = Any[st for (i, st) in enumerate(stmts) if i != p]
    cell, k = _inline_cell_latent_definitions(cell, k, x, ivar, others)
    j = _marginal_reader(x, cell, k;
        observation = "y[$ivar] .~ Normal.(offset .+ loading .* $x, noise)")
    y, loc, scale, form = _marginal_observation(x, cell[j],
        y -> Meta.isexpr(y, :ref, 2) && y.args[1] isa Symbol && y.args[2] === ivar,
        "y[$ivar]")
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
    t = law.loopvar
    pieces = Any[]
    seed, step = _marginal_pieces!(pieces, x, y, form, scale, leaf, t)
    filter = _filter_scan(x, y, law, seed, step;
        M = _marginal_cell_carried(x, :mean), V = _marginal_cell_carried(x, :var))
    density_cell = Any[]
    for (i, c) in enumerate(cell)
        i == k && continue
        if i == j
            append!(density_cell, pieces)
            push!(density_cell, filter)
            continue
        end
        push!(density_cell, c)
    end
    draws = _cell_draws_plate(x, y, law, seed, step, head, cell, pieces, k, j, observing)
    out = Any[]
    for (i, st) in enumerate(stmts)
        if i == p
            push!(out, Expr(:macrocall, st.args[1], st.args[2],
                Expr(:for, head, Expr(:block, density_cell...))), draws)
        else
            push!(out, st)
        end
    end
    return Expr(:block, out...)
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
function _cell_draws_plate(x, y, law, seed, step, head, cell, pieces, k, j, observing)
    ivar = head.args[1]
    carried = Tuple(_marginal_draw_carried(x, r) for r in _MARGINAL_ROLES)
    scan = _cell_draws_scan(x, y, law, seed, step, carried)
    outputs = [Expr(:(=), Expr(:ref, _marginal_name(x, r), ivar), c)
        for (r, c) in zip(_MARGINAL_ROLES, carried)]
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

function _cell_draws_scan(x, y, law::_GaussianTransition, seed_pieces, step_pieces, carried)
    t = law.loopvar
    M, V, C, D, Q = carried
    _, _, mean1, var1 = _kalman_update(:($y[1]), law.m0, law.P0, seed_pieces, nothing)
    pm, pv, g = _marginal_name(x, :predicted_mean),
        _marginal_name(x, :predicted_var), _marginal_name(x, :gain)
    prev(a) = Expr(:ref, a, Expr(:call, :-, t, 1))
    pm_ex = something(_madd(_mmul(law.coef, prev(M), false), law.shift, false), 0.0)
    pv_ex = something(_madd(_mmul(_mmul(law.coef, law.coef, false), prev(V), false),
        law.innovation_var, false), 0.0)
    _, gain, meant, vart = _kalman_update(:($y[$t]), pm, pv, step_pieces, g)
    body = Expr(:block, law.locals...,
        Expr(:(=), pm, pm_ex), Expr(:(=), pv, pv_ex), Expr(:(=), g, gain),
        Expr(:(=), :($M[$t]), meant), Expr(:(=), :($V[$t]), vart),
        Expr(:(=), :($C[$t]), _mfloat(something(law.coef, 0.0))),
        Expr(:(=), :($D[$t]), _mfloat(something(law.shift, 0.0))),
        Expr(:(=), :($Q[$t]), _mfloat(something(law.innovation_var, 0.0))))
    loop = Expr(:for, Expr(:(=), t, Expr(:call, :(:), 2, law.hi)), body)
    return _scan_macro(Expr(:block,
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
# names a vector whose false entries only predict. Returns the statements.
function _filter_scan(x, y, law::_GaussianTransition, seed_pieces, step_pieces;
        M = _marginal_name(x, :mean), V = _marginal_name(x, :var), present = nothing,
        response = nothing)
    t = law.loopvar
    entry(i) = present === nothing ? nothing : Expr(:ref, present, i)
    # Not `…_mean`: `_marginalized_names` finds latents by their filtered mean.
    PM, PV = _marginal_name(x, :predictive_location), _marginal_name(x, :predictive_variance)
    observe(obs, i, loc, fv) = response === nothing ? (obs,) :
        (Expr(:(=), Expr(:ref, PM, i), loc), Expr(:(=), Expr(:ref, PV, i), fv))
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
struct MarginalizedLatentQuery{Q}
    names::Vector{Symbol}
    query::Q
end

# The marginalized latents of a lowered program: those whose filter scan
# (carrying `_rkppl_marginal_<x>_mean`) it holds at top level, or whose draws'
# plate returns `_rkppl_marginal_<x>_mean[g]` per index.
function _marginalized_names(plan::StructuralPlan)
    out = Symbol[]
    suffix = "_mean"
    names = Iterators.flatten(((s for sp in plan.scans for s in sp.states),
        (d.name for d in plan.derived)))
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
each field. At posterior draws of the parameters these determine each
latent's exact conditional posterior; [`marginalized_latent_draws`](@ref)
samples it. A program with no marginalized recurrence is refused.
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
keyed by latent name (`Vector{Float64}`, one value per entry; for a
recurrence per `@plate` index, one such vector per index), by forward
filtering, backward sampling over `q(u)`. Only `rng` supplies randomness, in
key order and then index order, so a seeded `rng` reproduces the draws. At posterior draws of the
parameters, parameters and latents together are exact draws from the joint
posterior of the model as written.
"""
marginalized_latent_draws(q::MarginalizedLatentQuery, u::AbstractVector, rng) =
    marginalized_latent_draws(q(u), rng)
marginalized_latent_draws(filtered::NamedTuple, rng) =
    map(f -> _backward_sample(rng, f), filtered)

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
