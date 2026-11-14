# ── A vector state ────────────────────────────────────────────────────────
#
# A `@scan` carrying several arrays (a local linear trend: `level`, `slope`)
# or reading one at a lag above one (an AR(p): `x[t - 1]`, …, `x[t - p]`,
# seeded at `x[1..p]`) has a vector state: the window `(a, j) = a[t - j]` of
# every carried array `a` over `j in 0:p-1`, `p` the seed depth. Its Kalman
# filter is written out entry by entry for that fixed dimension, so every
# operation stays a scalar one, as in the scalar filter (marginalize.jl).
#
# The seeds give the joint Normal law of the window at step `p`; the filter's
# seeds condition it on `y[1..p]` in order (`y[k]` reads the window's
# component `(a, p - k)`), and its loop predicts through the step and updates
# on `y[t]`. A draws-only scan records each step's transition, and the draws
# condition backwards from step `T` to step `p`.
#
# Generated names are `_rkppl_marginal_<g>__<part>__…`, `g` the group's first
# array: the draws read the group's layout back from them, so an array name
# holding `__` is refused.

struct _VectorTransition
    group::Symbol
    arrays::Vector{Symbol}
    p::Int
    loopvar::Symbol
    hi::Any
    comps::Vector{Tuple{Symbol,Int}}
    # The window's law at step `p`: means, and covariances `(i, k)` with
    # `i <= k` (absent: structurally zero).
    m0::Vector{Any}
    P0::Dict{Tuple{Int,Int},Any}
    # Top-level definitions the seed law reads (compound seed pieces).
    seed_defs::Vector{Any}
    # Step locals, then the transition `s[t] = A s[t - 1] + shift + noise`
    # with noise covariance `Q` (`(i, k)`, `i <= k`); absent entries are 0.
    locals::Vector{Any}
    A::Dict{Tuple{Int,Int},Any}
    shift::Vector{Any}
    Q::Dict{Tuple{Int,Int},Any}
end

_vname(g::Symbol, parts...) = Symbol(_MARGINAL_PREFIX, g, "__", join(parts, "__"))
_vsym(D, i, k) = get(D, (min(i, k), max(i, k)), nothing)

# Covariance of two affine forms over independent standard-normal draws
# (keys `(:noise, n)`): `Σ_n b_in * b_kn`.
function _noise_cov(fi::_Affine, fk::_Affine)
    acc = nothing
    for (key, b) in fi.coefs
        (key isa Tuple && key[1] === :noise) || continue
        c = get(fk.coefs, key, nothing)
        c === nothing && continue
        acc = _madd(acc, _mmul(b, c, false), false)
    end
    return acc
end

# A compound noise scale is read in every covariance entry: `name!` replaces it
# by a definition (top level for seeds, a step local for the loop).
function _name_noise_scales(f::_Affine, name!)
    coefs = Dict{Any,Any}()
    for (key, c) in f.coefs
        coefs[key] = key isa Tuple && key[1] === :noise && c isa Expr ? name!(key[2], c) : c
    end
    return _Affine(f.constant, coefs)
end

function _vector_transition(spec::ScanSpec, arrays::Vector{Symbol})
    g = first(arrays)
    fail(msg) = _marginal_fail(g, msg)
    p = spec.lo - 1
    t = spec.loopvar
    comps = [(a, j) for a in arrays for j in 0:(p - 1)]
    index = Dict(c => i for (i, c) in enumerate(comps))
    nz = Ref(0)
    noise(scale) = (nz[] += 1; Dict{Any,Any}((:noise, nz[]) => scale))

    # Seeds: each an affine form over earlier seeds and its own draw.
    seeds = Dict{Tuple{Symbol,Int},_Affine}()
    seed_reads(ex) = any(a -> _mentions_symbol(ex, a), arrays)
    seed_leaf(ex) = Meta.isexpr(ex, :ref, 2) && ex.args[1] in arrays &&
        ex.args[2] isa Integer ? get(seeds, (ex.args[1], ex.args[2]), nothing) : nothing
    seed_defs = Any[]
    for st in spec.setup
        what = "the seed `$(st.target)[$(st.index)]`"
        if st.kind === :sample
            mu, s = _marginal_normal(g, st, what)
            seed_reads(s) && fail("the scale of $what reads a seed")
            form = _affine(mu, seed_leaf, seed_reads, false)
            form === nothing && fail("$what is not linear in the seeds above it")
            seeds[(st.target, st.index)] = _aff_add(form, _Affine(nothing, noise(s)), false)
        else
            form = _affine(st.expr, seed_leaf, seed_reads, false)
            form === nothing && fail("$what is not linear in the seeds above it")
            seeds[(st.target, st.index)] = form
        end
    end
    seed_name!(n, c) = (nm = _vname(g, :seed_scale, n); push!(seed_defs, :($nm = $c)); nm)
    window = [_name_noise_scales(seeds[(a, p - j)], seed_name!) for (a, j) in comps]
    m0 = Any[]
    for (i, f) in enumerate(window)
        c = f.constant
        if c isa Expr
            nm = _vname(g, :seed_mean, i)
            push!(seed_defs, :($nm = $c))
            c = nm
        end
        push!(m0, c)
    end
    P0 = Dict{Tuple{Int,Int},Any}()
    for i in eachindex(window), k in i:length(window)
        c = _noise_cov(window[i], window[k])
        c === nothing || (P0[(i, k)] = c)
    end

    # The step: each carried array written once, affine in the lags, the
    # values written above it and the step's Normal innovations.
    random = Dict{Symbol,_Affine}()
    write = Dict{Symbol,_Affine}()
    reads(ex) = any(a -> _mentions_symbol(ex, a), arrays) ||
        any(k -> _mentions_symbol(ex, k), keys(random))
    function leaf(ex)
        if Meta.isexpr(ex, :ref, 2) && ex.args[1] in arrays
            a, i = ex.args
            i === t && return get(write, a, nothing)
            Meta.isexpr(i, :call, 3) && i.args[1] === :- && i.args[2] === t &&
                i.args[3] isa Integer && 1 <= i.args[3] <= p || return nothing
            return _Affine(nothing, Dict{Any,Any}((:state, index[(a, i.args[3] - 1)]) => 1))
        end
        return ex isa Symbol && haskey(random, ex) ? random[ex] : nothing
    end
    locals = Any[]
    statement(st) = st.kind === :sample ?
        (st.indexed ? "`$(st.target)[$t] ~ …`" : "`$(st.target) ~ …`") :
        (st.indexed ? "`$(st.target)[$t] = $(st.expr)`" : "`$(st.target) = $(st.expr)`")
    nonlinear(st) = fail("the step $(statement(st)) is not linear in the lags " *
        "`a[$t - k]` of its carried arrays and the Normal innovations")
    for st in spec.step
        if st.kind === :sample && !st.indexed
            mu, sigma = _marginal_normal(g, st, "the innovation $(statement(st))")
            (reads(mu) || reads(sigma)) && fail("the innovation $(statement(st)) " *
                "reads a carried array or another innovation in its arguments")
            random[st.target] = _Affine(mu, noise(sigma))
        elseif st.kind === :sample
            mu, sigma = _marginal_normal(g, st, "the step $(statement(st))")
            reads(sigma) && fail("the scale of the step $(statement(st)) reads a " *
                "carried array or an innovation")
            a = _affine(mu, leaf, reads, false)
            a === nothing && nonlinear(st)
            write[st.target] = _aff_add(a, _Affine(nothing, noise(sigma)), false)
        elseif st.indexed
            a = _affine(st.expr, leaf, reads, false)
            a === nothing && nonlinear(st)
            write[st.target] = a
        elseif reads(st.expr)
            a = _affine(st.expr, leaf, reads, false)
            a === nothing && nonlinear(st)
            random[st.target] = a
        else
            push!(locals, Expr(:(=), st.target, st.expr))
        end
    end
    step_name!(n, c) = (nm = _vname(g, :step_scale, n); push!(locals, :($nm = $c)); nm)
    A = Dict{Tuple{Int,Int},Any}()
    shift = Any[nothing for _ in comps]
    rows = Dict{Int,_Affine}()
    for (i, (a, j)) in enumerate(comps)
        if j > 0
            A[(i, index[(a, j - 1)])] = 1
            continue
        end
        f = _name_noise_scales(write[a], step_name!)
        rows[i] = f
        shift[i] = f.constant
        for (key, c) in f.coefs
            (key isa Tuple && key[1] === :state) || continue
            # A compound coefficient is read in several covariance entries.
            if c isa Expr
                nm = _vname(g, :step_coef, i, key[2])
                push!(locals, :($nm = $c))
                c = nm
            end
            A[(i, key[2])] = c
        end
    end
    Q = Dict{Tuple{Int,Int},Any}()
    for i in keys(rows), k in keys(rows)
        i <= k || continue
        c = _noise_cov(rows[i], rows[k])
        c === nothing || (Q[(i, k)] = c)
    end
    return _VectorTransition(g, arrays, p, t, spec.hi, comps, m0, P0, seed_defs,
        locals, A, shift, Q)
end

# One Kalman update of the window at predicted means `pm[i]` and covariances
# `pv(i, k)`, for an observation of value `yk` reading component `i` with
# loading `H[i]` (absent: 0), offset `O` and squared scale `SS`. Returns the
# predictive location and variance, the cross covariances `ph[i]`, and the
# filtered means and covariances given the update's gains `gain(i)`.
function _vector_update(pm, pv, H, O, SS, d)
    ph = Any[nothing for _ in 1:d]
    for i in 1:d, (k, h) in H
        ph[i] = _madd(ph[i], _mmul(pv(i, k), h, false), false)
    end
    fv = SS
    loc = O
    for (k, h) in H
        fv = _madd(fv, _mmul(h, ph[k], false), false)
        loc = _madd(loc, _mmul(h, pm[k], false), false)
    end
    return something(loc, 0.0), something(fv, 0.0), ph
end

function _filtered(pm, pv, ph, gain, residual, d)
    means = Any[_mfloat(something(_madd(pm[i], _mmul(gain(i), residual, false), false), 0.0))
        for i in 1:d]
    covs = Dict{Tuple{Int,Int},Any}()
    for i in 1:d, k in i:d
        covs[(i, k)] = _mfloat(something(_madd(pv(i, k),
            _mneg(_mmul(gain(i), ph[k], false), false), false), 0.0))
    end
    return means, covs
end

_vector_mean(law, i) = _vname(law.group, i, law.comps[i]..., :m)
_vector_cov(law, i, k) = _vname(law.group, :v, min(i, k), max(i, k))

# The filter `@scan`: seeds condition the window's law at step `p` on
# `y[1..p]`, the loop predicts and updates on each `y[t]`. `loading(a, k)` is
# array `a`'s loading at entry `k` (absent: 0), `offset(k)` and `ss(k)` the
# offset and the squared noise scale.
function _vector_filter_scan(law::_VectorTransition, y, offset, loading, ss)
    g, p, t, d = law.group, law.p, law.loopvar, length(law.comps)
    M(i, at) = Expr(:ref, _vector_mean(law, i), at)
    V(i, k, at) = Expr(:ref, _vector_cov(law, i, k), at)
    index = Dict(c => i for (i, c) in enumerate(law.comps))
    observed(k, j) = Dict(index[(a, j)] => h for a in law.arrays
        for h in (loading(a, k),) if h !== nothing)
    setup = Any[]
    for k in 1:p
        pm = k == 1 ? law.m0 : Any[M(i, k - 1) for i in 1:d]
        pv = k == 1 ? ((i, l) -> _vsym(law.P0, i, l)) : ((i, l) -> V(min(i, l), max(i, l), k - 1))
        loc, fv, ph = _vector_update(pm, pv, observed(k, p - k), offset(k), ss(k), d)
        yk = Expr(:ref, y, k)
        push!(setup, :($yk ~ Normal($loc, sqrt($fv))))
        gain(i) = ph[i] === nothing ? nothing : Expr(:call, :/, ph[i], fv)
        means, covs = _filtered(pm, pv, ph, gain, Expr(:call, :-, yk, loc), d)
        append!(setup, (Expr(:(=), M(i, k), means[i]) for i in 1:d))
        append!(setup, (Expr(:(=), V(i, l, k), covs[(i, l)]) for i in 1:d for l in i:d))
    end
    prev = Expr(:call, :-, t, 1)
    body = Any[law.locals...]
    pm = Any[_vname(g, :pm, i) for i in 1:d]
    for i in 1:d
        ex = law.shift[i]
        for l in 1:d
            ex = _madd(ex, _mmul(get(law.A, (i, l), nothing), M(l, prev), false), false)
        end
        push!(body, Expr(:(=), pm[i], _mfloat(something(ex, 0.0))))
    end
    pvn = Dict{Tuple{Int,Int},Any}()
    for i in 1:d, k in i:d
        ex = _vsym(law.Q, i, k)
        for (l, r) in Iterators.product(1:d, 1:d)
            ail, akr = get(law.A, (i, l), nothing), get(law.A, (k, r), nothing)
            (ail === nothing || akr === nothing) && continue
            ex = _madd(ex, _mmul(_mmul(ail, akr, false), V(l, r, prev), false), false)
        end
        ex === nothing && continue
        pvn[(i, k)] = _vname(g, :pv, i, k)
        push!(body, Expr(:(=), pvn[(i, k)], ex))
    end
    pv(i, k) = _vsym(pvn, i, k)
    H = observed(t, 0)
    loc, _, ph = _vector_update(pm, pv, H, offset(t), ss(t), d)
    phn = Any[nothing for _ in 1:d]
    for i in 1:d
        ph[i] === nothing && continue
        phn[i] = _vname(g, :ph, i)
        push!(body, Expr(:(=), phn[i], ph[i]))
    end
    # The predictive variance reads the cross covariances through their locals.
    fv_ex = ss(t)
    for (k, h) in H
        fv_ex = _madd(fv_ex, _mmul(h, phn[k], false), false)
    end
    fvn, locn = _vname(g, :fv), _vname(g, :loc)
    fv_ex = something(fv_ex, 0.0)
    yt = Expr(:ref, y, t)
    push!(body, Expr(:(=), fvn, fv_ex), Expr(:(=), locn, loc),
        :($yt ~ Normal($locn, sqrt($fvn))))
    gains = Any[phn[i] === nothing ? nothing : _vname(g, :gain, i) for i in 1:d]
    for i in 1:d
        gains[i] === nothing || push!(body, Expr(:(=), gains[i], Expr(:call, :/, phn[i], fvn)))
    end
    means, covs = _filtered(pm, pv, phn, i -> gains[i], Expr(:call, :-, yt, locn), d)
    append!(body, (Expr(:(=), M(i, t), means[i]) for i in 1:d))
    append!(body, (Expr(:(=), V(i, k, t), covs[(i, k)]) for i in 1:d for k in i:d))
    loop = Expr(:for, Expr(:(=), t, Expr(:call, :(:), p + 1, law.hi)), Expr(:block, body...))
    return _scan_macro(Expr(:block, setup..., loop))
end

# The draws' transition `@scan`: each step's structural transition entries,
# shifts and noise covariances (0 at the seeds, which the draws do not read).
_vector_coef(law, i, k) = _vname(law.group, :a, i, k)
_vector_shift(law, i) = _vname(law.group, :c, i)
_vector_noise(law, i, k) = _vname(law.group, :q, i, k)

function _vector_transition_scan(law::_VectorTransition)
    t, p, d = law.loopvar, law.p, length(law.comps)
    entries = Pair{Symbol,Any}[]
    for (i, k) in sort!(collect(keys(law.A)))
        push!(entries, _vector_coef(law, i, k) => law.A[(i, k)])
    end
    for i in 1:d
        push!(entries, _vector_shift(law, i) => something(law.shift[i], 0.0))
    end
    for (i, k) in sort!(collect(keys(law.Q)))
        push!(entries, _vector_noise(law, i, k) => law.Q[(i, k)])
    end
    setup = Any[Expr(:(=), Expr(:ref, n, s), 0.0) for s in 1:p for (n, _) in entries]
    body = Any[law.locals..., (Expr(:(=), Expr(:ref, n, t), _mfloat(v)) for (n, v) in entries)...]
    loop = Expr(:for, Expr(:(=), t, Expr(:call, :(:), p + 1, law.hi)), Expr(:block, body...))
    return _scan_macro(Expr(:block, setup..., loop))
end

# The arrays a top-level `@scan` carrying `x` integrates out together, or
# `nothing` when it carries `x` alone at lag one (the scalar filter) or when
# no top-level `@scan` carries `x` (a `@plate` cell's, or a refusal).
function _marginal_group(ast::Expr, x::Symbol, marginalized)
    k = findfirst(st -> _is_scan_of(st, x), ast.args)
    k === nothing && return nothing
    spec = parse_scan_block(ast.args[k].args[end])
    (length(spec.states) == 1 && spec.lo == 2) && return nothing
    missing_names = setdiff(spec.states, marginalized)
    isempty(missing_names) || _marginal_fail(x, "its `@scan` also carries " *
        join(("`$o`" for o in missing_names), ", ") * "; a recurrence carrying several " *
        "arrays (a vector state) integrates out together: name every array it " *
        "carries, `marginalize(model, " * join((":$s" for s in spec.states), ", ") * ")`")
    for s in spec.states
        occursin("__", String(s)) && _marginal_fail(s, "a vector state's array name " *
            "must not hold `__`; rename `$s`")
    end
    return spec.states
end

function _marginalize_vector_latent(ast::Expr, arrays::Vector{Symbol}, values, observing)
    g = first(arrays)
    stmts = ast.args
    k = findfirst(st -> _is_scan_of(st, g), stmts)
    law = _vector_transition(parse_scan_block(stmts[k].args[end]), arrays)
    for a in arrays
        stmts, k = _inline_latent_definitions(stmts, k, a)
    end
    readers = [j for (j, st) in enumerate(stmts)
        if j != k && st isa Expr && any(a -> _mentions_symbol(st, a), arrays)]
    spelled = "y .~ Normal.(offset .+ " * join(("loading_$a .* $a" for a in arrays), " .+ ") *
        ", noise)"
    length(readers) == 1 || _marginal_fail(g, "every read of " *
        join(("`$a`" for a in arrays), ", ") * " must be the location of one Normal " *
        "observation `$spelled`; " * (isempty(readers) ? "nothing reads them" :
            "they are read by " * join(("`$(stmts[j])`" for j in readers), ", ")))
    j = only(readers)
    y, form, scale = _vector_observation(g, arrays, stmts[j], spelled)
    (y in observing && get(values, y, nothing) isa AbstractVector) ||
        _marginal_fail(g, "its observation `$y` is not bound data; a vector state is " *
            "marginalized when observed through bound data (`| (; $y)`)")
    Missing <: eltype(values[y]) && _marginal_fail(g, "its observation `$y` admits " *
        "`missing` entries; a vector state is marginalized without them yet")
    leaf = _toplevel_shape_leaf(_marginal_names(ast), values)
    defs = Any[law.seed_defs...]
    offset = _vector_piece!(defs, form.constant, :offset, g, y, leaf)
    loadings = Dict(a => _vector_piece!(defs, get(form.coefs, a, nothing),
        Symbol(:loading_, i), g, y, leaf) for (i, a) in enumerate(arrays))
    ss = _vector_piece!(defs, scale, :scale, g, y, leaf; square = true)
    filter = _vector_filter_scan(law, y, offset, (a, k) -> loadings[a](k), ss)
    transition = _vector_transition_scan(law)
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

# The Normal observation of a whole bound vector whose location is affine in
# the group's arrays: its response, affine form and scale.
function _vector_observation(g, arrays, obs, spelled)
    wrong() = _marginal_fail(g, "it is read by `$obs`; the filter needs a Normal " *
        "observation of a whole observed vector, `$spelled`, with `offset`, the " *
        "loadings and `noise` not reading the carried arrays")
    (Meta.isexpr(obs, :call, 3) && obs.args[1] === :.~ && obs.args[2] isa Symbol) || wrong()
    y, rhs = obs.args[2], obs.args[3]
    (Meta.isexpr(rhs, :., 2) && rhs.args[1] === :Normal &&
        Meta.isexpr(rhs.args[2], :tuple) && length(rhs.args[2].args) in (1, 2)) || wrong()
    loc = rhs.args[2].args[1]
    scale = length(rhs.args[2].args) == 2 ? rhs.args[2].args[2] : 1.0
    reads(ex) = any(a -> _mentions_symbol(ex, a), arrays)
    reads(scale) && wrong()
    gathered = Ref(false)
    function leaf(ex)
        ex isa Symbol && ex in arrays && return _Affine(nothing, Dict{Any,Any}(ex => 1))
        Meta.isexpr(ex, :ref) && ex.args[1] in arrays && (gathered[] = true)
        return nothing
    end
    form = _affine(loc, leaf, reads, true)
    gathered[] && _marginal_fail(g, "its observation `$obs` gathers entries of a " *
        "carried array; rows observing a vector state at their steps are not " *
        "marginalized yet")
    (form === nothing || !any(a -> haskey(form.coefs, a), arrays)) && wrong()
    return y, form, scale
end

# A piece of the observation (offset, a loading, the scale) at entry `k`:
# one value as it is, one value per entry as `piece[k]`; with `square` its
# square. A compound piece is named once as a definition in `defs`.
function _vector_piece!(defs, piece, role, g, y, leaf; square = false)
    piece === nothing && return k -> nothing
    piece isa Real && return k -> square ? piece * piece : piece
    shape = _shape_of(piece, leaf)
    shape === :unknown && _marginal_fail(g, "cannot tell whether the observation's " *
        "$role `$piece` is one value or one value per entry of `$y`; write it " *
        "from declared parameters, bound data and definitions of them")
    name = piece
    if !(piece isa Symbol)
        name = _vname(g, role)
        push!(defs, Expr(:(=), name, piece))
    end
    if shape === :scalar
        square || return k -> name
        sq = _vname(g, Symbol(role, :_squared))
        push!(defs, Expr(:(=), sq, Expr(:call, :*, name, name)))
        return k -> sq
    end
    return k -> square ? Expr(:call, :*, Expr(:ref, name, k), Expr(:ref, name, k)) :
        Expr(:ref, name, k)
end

# ── Exact draws of a vector state ─────────────────────────────────────────

using LinearAlgebra: pinv

# The vector-state groups of a lowered program, read back from the filter's
# carried means `_rkppl_marginal_<g>__<i>__<a>__<j>__m` and the transition's
# entries `…__a__<i>__<k>`, `…__c__<i>`, `…__q__<i>__<k>`.
function _marginalized_groups(plan::StructuralPlan)
    groups = Dict{Symbol,Any}()
    order = Symbol[]
    for sp in plan.scans, s in sp.states
        str = String(s)
        startswith(str, _MARGINAL_PREFIX) || continue
        parts = split(str[length(_MARGINAL_PREFIX)+1:end], "__")
        length(parts) >= 3 || continue
        g = Symbol(parts[1])
        entry = get!(groups, g) do
            push!(order, g)
            (; comps = Dict{Int,Tuple{Symbol,Int}}(), A = Tuple{Int,Int}[],
                Q = Tuple{Int,Int}[])
        end
        if length(parts) == 5 && parts[5] == "m"
            entry.comps[parse(Int, parts[2])] = (Symbol(parts[3]), parse(Int, parts[4]))
        elseif parts[2] == "a"
            push!(entry.A, (parse(Int, parts[3]), parse(Int, parts[4])))
        elseif parts[2] == "q"
            push!(entry.Q, (parse(Int, parts[3]), parse(Int, parts[4])))
        end
    end
    return _MarginalGroup[(g, [groups[g].comps[i] for i in 1:length(groups[g].comps)],
        sort!(groups[g].A), sort!(groups[g].Q)) for g in order]
end

# The query names of one group, in the order `_vector_filtered` reads them.
function _vector_wants(g, comps, A, Q)
    d = length(comps)
    law = (; group = g, comps)
    return Symbol[(_vector_mean(law, i) for i in 1:d)...,
        (_vector_cov(law, i, k) for i in 1:d for k in i:d)...,
        (_vector_coef(law, i, k) for (i, k) in A)...,
        (_vector_shift(law, i) for i in 1:d)...,
        (_vector_noise(law, i, k) for (i, k) in Q)...]
end

# One group's filtered window: per step, its mean vector and covariance, and
# the transition into it (matrices; entry `p` and before hold seeds).
function _vector_filtered(values, comps, A, Q)
    d = length(comps)
    T = length(values[1])
    next = Ref(0)
    take() = values[next[] += 1]
    means = [take() for _ in 1:d]
    covs = Dict((i, k) => take() for i in 1:d for k in i:d)
    coefs = Dict(e => take() for e in A)
    shifts = [take() for _ in 1:d]
    noises = Dict(e => take() for e in Q)
    p = maximum(last, comps) + 1
    mean = [Float64[means[i][t] for i in 1:d] for t in 1:T]
    cov = [Float64[covs[(min(i, k), max(i, k))][t] for i in 1:d, k in 1:d] for t in 1:T]
    coef = [Float64[haskey(coefs, (i, k)) ? coefs[(i, k)][t] : 0.0 for i in 1:d, k in 1:d]
        for t in 1:T]
    shift = [Float64[shifts[i][t] for i in 1:d] for t in 1:T]
    innovation_cov = [Float64[haskey(noises, (min(i, k), max(i, k))) ?
        noises[(min(i, k), max(i, k))][t] : 0.0 for i in 1:d, k in 1:d] for t in 1:T]
    return (; components = Tuple(comps), lags = p, mean, cov, coef, shift, innovation_cov)
end

# Backward sampling of a window: the window at `T` from its filtered law,
# then each window at `t` given the drawn window at `t + 1`, down to step `p`;
# each array's entry `t` is its window component `(a, 0)` at step `t`, and its
# entries before `p` are the window at `p`. A singular predicted covariance
# (a deterministic seed, the window's shifted components) is conditioned on
# through its pseudo-inverse.
function _backward_sample_vector(rng, f)
    comps, p, T = f.components, f.lags, length(f.mean)
    d = length(comps)
    arrays = unique(first.(comps))
    out = Dict(a => Vector{Float64}(undef, T) for a in arrays)
    T == 0 && return [a => out[a] for a in arrays]
    s = _draw_normal(rng, f.mean[T], f.cov[T], T)
    record(s, t) = for (i, (a, j)) in enumerate(comps)
        (j == 0 || t == p) && t - j >= 1 && (out[a][t - j] = s[i])
    end
    record(s, T)
    for t in (T - 1):-1:p
        Ad, P, m = f.coef[t + 1], f.cov[t], f.mean[t]
        predicted = Ad * P * Ad' + f.innovation_cov[t + 1]
        cross = P * Ad'
        J = cross * pinv(Matrix(Symmetric((predicted + predicted') / 2)); rtol = 1e-10)
        mean = m + J * (s - (Ad * m + f.shift[t + 1]))
        cov = P - J * cross'
        s = _draw_normal(rng, mean, cov, t)
        record(s, t)
    end
    return [a => out[a] for a in arrays]
end

function _draw_normal(rng, m, P, t)
    F = eigen(Symmetric((P + P') / 2))
    scale = maximum(abs, F.values; init = 0.0)
    all(v -> isfinite(v) && v >= -1e-9 * (1 + scale), F.values) || throw(ArgumentError(
        "[query] the filtered state covariance at entry $t is not positive semidefinite"))
    return m + F.vectors * (sqrt.(max.(F.values, 0.0)) .* randn(rng, length(m)))
end
