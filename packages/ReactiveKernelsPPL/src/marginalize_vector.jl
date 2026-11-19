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
    # Seed means and their coefficients on independent standard Normal draws.
    m0::Vector{Any}
    seed_noise::Matrix{Any}
    # Top-level definitions the seed law reads (compound seed pieces).
    seed_defs::Vector{Any}
    # Step locals, then the transition `s[t] = A s[t - 1] + shift + noise`
    # with noise coefficients on independent standard Normal draws.
    locals::Vector{Any}
    A::Dict{Tuple{Int,Int},Any}
    shift::Vector{Any}
    step_noise::Matrix{Any}
end

_vname(g::Symbol, parts...) = Symbol(_MARGINAL_PREFIX, g, "__", join(parts, "__"))

# Each column is one shared innovation, even when several state rows read it.
function _noise_factor(rows, draws)
    return Any[get(row.coefs, (:noise, n), nothing) for row in rows, n in draws]
end

# Inner products of rows of symbolic factors. `nothing` denotes a structural zero.
function _factor_dot(x, y)
    total = nothing
    for (a, b) in zip(x, y)
        total = _madd(total, _mmul(a, b, false), false)
    end
    return total
end

# A compound noise scale is read in every covariance entry: `name!` replaces it
# by a definition (top level for seeds, a step local for the loop). One draw
# enters every row that reads it: in `slope[t] ~ Normal(phi * level[t], s)`
# `level[t]`'s innovation reaches `slope`'s row as `phi * <its scale>`, so a
# name belongs to one row and one innovation, never to the innovation alone.
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
    seed_name!(i) = (n, c) -> (nm = _vname(g, :seed_scale, i, n);
        push!(seed_defs, :($nm = $c)); nm)
    window = [_name_noise_scales(seeds[(a, p - j)], seed_name!(i))
        for (i, (a, j)) in enumerate(comps)]
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
    seed_noise = _noise_factor(window, 1:nz[])
    first_step_draw = nz[] + 1

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
    step_name!(i) = (n, c) -> (nm = _vname(g, :step_scale, i, n);
        push!(locals, :($nm = $c)); nm)
    A = Dict{Tuple{Int,Int},Any}()
    shift = Any[nothing for _ in comps]
    rows = [_Affine(nothing) for _ in comps]
    for (i, (a, j)) in enumerate(comps)
        if j > 0
            A[(i, index[(a, j - 1)])] = 1
            continue
        end
        f = _name_noise_scales(write[a], step_name!(i))
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
    return _VectorTransition(g, arrays, p, t, spec.hi, comps, m0, seed_noise, seed_defs,
        locals, A, shift, _noise_factor(rows, first_step_draw:nz[]))
end

# Right Givens rotations triangularize a joint square root, one independent
# noise column at a time. The observation is row 1. Its pivot is the predictive
# scale, the column below it times the standardized residual updates the means,
# and the trailing square block factors the conditional covariance.
# The loops here run over the authored state and innovations, never the data.
function _observation_triangle(factor, loading, scale, save)
    d, n = size(factor)
    joint = Matrix{Any}(nothing, d + 1, max(n, d) + 1)
    joint[1, 1] = save(Expr(:call, GlobalRef(Base, :abs), scale))
    joint[2:end, 2:(n + 1)] = factor
    for col in 1:n
        joint[1, col + 1] = save(_factor_dot(loading, view(factor, :, col)))
    end
    function rotate!(row, col)
        a, b = joint[row, row], joint[row, col]
        b === nothing && return
        if a === nothing
            joint[:, row], joint[:, col] = joint[:, col], joint[:, row]
            return
        end
        # At a zero pair the rotation is the identity. Keep both entries:
        # a zero loading can have a nonzero derivative.
        zero_pair = save(:(($a == 0) & ($b == 0)))
        norm = save(Expr(:call, GlobalRef(Base, :hypot), a, :($zero_pair ? one($b) : $b)))
        x_unit = save(:($a / $norm))
        cosine = save(:($zero_pair ? one($norm) : $x_unit))
        sine = save(:($b / $norm))
        # The protected norm is one at the origin. These quotients preserve
        # the entries there, in the same floating type as every other branch.
        joint[row, row] = save(:($zero_pair ? $x_unit : $norm))
        joint[row, col] = save(:($zero_pair ? $sine : zero($norm)))
        for i in 2:(d + 1)
            i == row && continue
            x, y = joint[i, row], joint[i, col]
            joint[i, row] = save(_factor_dot((cosine, sine), (x, y)))
            joint[i, col] = save(_factor_dot((cosine, _mneg(sine, false)), (y, x)))
        end
    end
    for col in 2:size(joint, 2)
        rotate!(1, col)
    end
    # With zero observation noise, a zero pair can precede a nonzero loading.
    # The remaining row entries are zero-valued, but may have live tangents.
    # Finish their rotations about the final innovation scale: hypot(sd, 0)
    # is sd, hence the cosine is one and the sine is the residual over sd.
    sd = joint[1, 1]
    for col in 2:size(joint, 2)
        residual = joint[1, col]
        residual === nothing && continue
        sine = save(:($residual / $sd))
        for i in 2:(d + 1)
            x, y = joint[i, 1], joint[i, col]
            joint[i, 1] = save(_madd(x, _mmul(sine, y, false), false))
            joint[i, col] = save(_madd(y, _mneg(_mmul(sine, x, false), false), false))
        end
        joint[1, col] = nothing
    end
    for row in 2:(d + 1), col in (row + 1):size(joint, 2)
        rotate!(row, col)
    end
    return joint
end

_vector_mean(law, i) = _vname(law.group, i, law.comps[i]..., :m)
_vector_cov(law, i, k) = _vname(law.group, :v, min(i, k), max(i, k))

# Carried arrays of values the author named (`marginalize(…; predictive,
# filtered)`): each entry's predictive `:location` / `:variance`, and array
# `a`'s filtered mean and variance at each step. Two-part names, never ending
# in `_mean`, so neither `_marginalized_groups` nor `_marginalized_names` reads
# them as a latent.
_vector_predictive(law, role) = _vname(law.group, Symbol(:predictive_, role))
_vector_filtered_mean(law, a) = _vname(law.group, Symbol(:filtered_, a, :_m))
_vector_filtered_var(law, a) = _vname(law.group, Symbol(:filtered_, a, :_v))

# The filter `@scan`: seeds condition the window's law at step `p` on
# `y[1..p]`, the loop predicts and updates on each `y[t]`. `loading(a, k)` is
# array `a`'s loading at entry `k` (absent: 0), `offset(k)` and `scale(k)` give the
# observation offset and noise scale. `predictive` also carries each entry's
# predictive location and variance, and `filtered` each listed array's filtered
# mean and variance: the window component `(a, p - k)` at seed `k` (the window
# then holds the seeds `a[p - j]`), `(a, 0)` in the loop.
function _vector_filter_scan(law::_VectorTransition, y, offset, loading, scale, defs;
        predictive = false, filtered = Symbol[])
    g, p, t, d = law.group, law.p, law.loopvar, length(law.comps)
    M(i, at) = Expr(:ref, _vector_mean(law, i), at)
    V(i, k, at) = Expr(:ref, _vector_cov(law, i, k), at)
    L(i, k, at) = Expr(:ref, _vname(g, :root, i, k), at)
    index = Dict(c => i for (i, c) in enumerate(law.comps))

    # Seeds have scalar definitions outside the scan; the same algebra in the
    # loop gets step locals. Name each expression once at its point of use.
    function store!(stmts, at, ex)
        ex isa Expr || return ex
        name = _vname(g, :filter, at, length(stmts) + 1)
        push!(stmts, Expr(:(=), name, ex))
        return name
    end
    function observe!(writes, locals, at, lag, pm, factor)
        save(ex) = store!(locals, at, ex)
        H = Any[j == lag ? loading(a, at) : nothing for (a, j) in law.comps]
        loc = _mfloat(something(_madd(offset(at), _factor_dot(H, pm), false), 0.0))
        joint = _observation_triangle(factor, H, scale(at), save)
        sd = joint[1, 1]
        yt = Expr(:ref, y, at)
        push!(writes, :($yt ~ Normal($loc, $sd)))
        residual = :($yt - $loc)
        root = joint[2:end, 2:(d + 1)]
        for i in 1:d
            gain = _mdiv(joint[i + 1, 1], sd, false)
            mean = _madd(pm[i], _mmul(gain, residual, false), false)
            push!(writes, Expr(:(=), M(i, at), _mfloat(something(mean, 0.0))))
            for k in 1:d
                push!(writes, Expr(:(=), L(i, k, at), _mfloat(something(root[i, k], 0.0))))
            end
            for k in i:d
                cov = _factor_dot(view(root, i, :), view(root, k, :))
                push!(writes, Expr(:(=), V(i, k, at), _mfloat(something(cov, 0.0))))
            end
        end
        if predictive
            push!(writes, Expr(:(=), Expr(:ref, _vector_predictive(law, :location), at), loc),
                Expr(:(=), Expr(:ref, _vector_predictive(law, :scale), at), sd))
        end
        for a in filtered
            i = index[(a, lag)]
            push!(writes, Expr(:(=), Expr(:ref, _vector_filtered_mean(law, a), at), M(i, at)),
                Expr(:(=), Expr(:ref, _vector_filtered_var(law, a), at), V(i, i, at)))
        end
        return root
    end
    setup = Any[]
    factor = law.seed_noise
    for k in 1:p
        pm = k == 1 ? law.m0 : Any[M(i, k - 1) for i in 1:d]
        factor = observe!(setup, defs, k, p - k, pm, factor)
    end
    prev = Expr(:call, :-, t, 1)
    body = Any[law.locals...]
    A = Any[get(law.A, (i, k), nothing) for i in 1:d, k in 1:d]
    pm = Any[store!(body, t, _madd(law.shift[i],
        _factor_dot(view(A, i, :), (M(k, prev) for k in 1:d)), false)) for i in 1:d]
    # Independent columns of [A * L_previous, process noise factor].
    # At a rank change an entry above the diagonal can be zero with an
    # active tangent. Carry the whole square block after the rotations.
    propagated = Any[store!(body, t, _factor_dot(view(A, i, :),
        (L(l, k, prev) for l in 1:d))) for i in 1:d, k in 1:d]
    observe!(body, body, t, 0, pm, hcat(propagated, law.step_noise))
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
    for i in 1:d, k in i:d
        q = _factor_dot(view(law.step_noise, i, :), view(law.step_noise, k, :))
        q === nothing || push!(entries, _vector_noise(law, i, k) => q)
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

function _marginalize_vector_latent(ast::Expr, arrays::Vector{Symbol}, values, observing,
        req = nothing)
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
    noise = _vector_piece!(defs, scale, :scale, g, y, leaf)
    stored = req === nothing ? Symbol[] : unique(Symbol[a for (a, _, _) in req.filtered])
    filter = _vector_filter_scan(law, y, offset, (a, k) -> loadings[a](k), noise, defs;
        predictive = _stores_predictive(req), filtered = stored)
    transition = _vector_transition_scan(law)
    sd = _vector_predictive(law, :scale)
    named = _marginal_value_defs(req, _vector_predictive(law, :location), :($sd .* $sd),
        a -> (_vector_filtered_mean(law, a), _vector_filtered_var(law, a)); scale = sd)
    out = Any[]
    for (i, st) in enumerate(stmts)
        i == k && continue
        if i == j
            append!(out, defs)
            push!(out, filter, transition, named...)
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
# one value as it is, one value per entry as `piece[k]`. A compound piece is named once as a definition in `defs`.
function _vector_piece!(defs, piece, role, g, y, leaf)
    piece === nothing && return k -> nothing
    piece isa Real && return k -> piece
    shape = _shape_of(piece, leaf)
    shape === :unknown && _marginal_fail(g, "cannot tell whether the observation's " *
        "$role `$piece` is one value or one value per entry of `$y`; write it " *
        "from declared parameters, bound data and definitions of them")
    name = piece
    if !(piece isa Symbol)
        name = _vname(g, role)
        push!(defs, Expr(:(=), name, piece))
    end
    return shape === :scalar ? (k -> name) : (k -> Expr(:ref, name, k))
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
