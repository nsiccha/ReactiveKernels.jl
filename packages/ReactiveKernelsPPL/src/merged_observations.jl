# ── Observed vectors merged with other values (data augmentation) ────────
#
# Data augmentation merges an observed vector with values the model computes,
# typically bounded latents that stand for censored rows:
#
#     below[1:K] .~ restricted.(Flat.(), -Inf, limit)
#     value = vcat(measured, below)[order]
#     value .~ Normal.(mu, s)
#
# with `measured` conditioned (`| (; measured)`). Each entry of `value` is an
# entry of `measured` or one of the other values, so lowering splits the
# statement by entry. The entries of each observed vector are its
# observation, `measured .~ Normal.(mu[rows], s[rows])`, which enters
# `likelihood` and `:pointwise.measured`. The other entries are one density
# factor, `rest = vcat(below)[parts]; rest .~ Normal.(mu[rest_rows], …)`,
# which enters `prior`. `value` stays a definition, so a reader of it sees
# every entry: queries, and a marginalized recurrence's Kalman filter
# (marginalize.jl), which runs before this pass and leaves
# `value .~ Normal.(location, sqrt.(variance))` behind it. The positions are
# data-only definitions that `bind_data` computes from `order` and the
# parts' lengths.

const _MERGED_PREFIX = "_rkppl_merged_"
_merged_name(v::Symbol, role) = Symbol(_MERGED_PREFIX, v, :_, role)

# The parts and index of a merge, `vcat(parts…)[order]` or `vcat(parts…)`
# (also spelled `[a; b]`), or `nothing`.
function _merge_form(rhs)
    order = nothing
    if Meta.isexpr(rhs, :ref, 2)
        order = rhs.args[2]
        rhs = rhs.args[1]
    end
    parts = Meta.isexpr(rhs, :vcat) ? rhs.args :
        Meta.isexpr(rhs, :call) && !isempty(rhs.args) && rhs.args[1] === :vcat ?
            rhs.args[2:end] : nothing
    (parts === nothing || isempty(parts) ||
        any(p -> Meta.isexpr(p, :row) || Meta.isexpr(p, :...), parts)) && return nothing
    return parts, order
end

# A top-level `v .~ F.(args…)` (a built-in family) over a merge definition `v`
# with an observed vector among its parts: `(v, parts, order, F, args)`.
function _merged_observation(st, defs, observed)
    Meta.isexpr(st, :call, 3) && st.args[1] === :.~ || return nothing
    v, rhs = st.args[2], st.args[3]
    (v isa Symbol && haskey(defs, v)) || return nothing
    form = _merge_form(defs[v])
    form === nothing && return nothing
    parts, order = form
    any(p -> p isa Symbol && p in observed, parts) || return nothing
    (Meta.isexpr(rhs, :., 2) && rhs.args[1] isa Symbol &&
        Meta.isexpr(rhs.args[2], :tuple) && !_external_rhs(rhs)) || return nothing
    return v, parts, order, rhs.args[1], rhs.args[2].args
end

# The single-axis extents of top-level declarations, `z[1:K] .~ …`.
function _declared_extents(ast::Expr)
    out = Dict{Symbol,Any}()
    for st in ast.args
        Meta.isexpr(st, :call, 3) && st.args[1] in (:~, :.~) || continue
        lhs = st.args[2]
        Meta.isexpr(lhs, :ref, 2) && lhs.args[1] isa Symbol && (out[lhs.args[1]] = lhs.args[2])
    end
    return out
end

_extent_length(ext) = Meta.isexpr(ext, :call, 3) && ext.args[1] === :(:) &&
    ext.args[2] == 1 ? ext.args[3] : :(length($ext))

# A data expression for the length of a merge's part: bound data, a declared
# array's extent, a gather's index, a range, or the first argument of a
# broadcast whose length is known this way. `nothing` otherwise.
function _merged_length(p, values, defs, extents, active::Set{Symbol} = Set{Symbol}())
    if p isa Symbol
        get(values, p, nothing) isa AbstractVector && return :(length($p))
        haskey(extents, p) && return _extent_length(extents[p])
        (haskey(defs, p) && !(p in active)) || return nothing
        push!(active, p)
        return _merged_length(defs[p], values, defs, extents, active)
    end
    p isa Expr || return nothing
    Meta.isexpr(p, :call, 3) && p.args[1] === :(:) && return :(length($p))
    Meta.isexpr(p, :ref, 2) && return _merged_length(p.args[2], values, defs, extents, active)
    args = Meta.isexpr(p, :., 2) && Meta.isexpr(p.args[2], :tuple) ? p.args[2].args :
        Meta.isexpr(p, :call) && p.args[1] isa Symbol && length(String(p.args[1])) > 1 &&
            startswith(String(p.args[1]), '.') ? p.args[2:end] : nothing
    args === nothing && return nothing
    for a in args
        n = _merged_length(a, values, defs, extents, active)
        n === nothing || return n
    end
    return nothing
end

# A distribution argument read at the merge's positions `at`: one value as it
# is, one value per entry gathered (into a broadcast's arguments, or from a
# definition `named(ex)` names). `nothing` when its shape is unknown.
function _merged_gather(ex, at, leaf, named)
    ex isa Real && return ex
    shape = _shape_of(ex, leaf)
    shape === :scalar && return ex
    shape === :vector || return nothing
    ex isa Symbol && return Expr(:ref, ex, at)
    inner(args) = (g = Any[_merged_gather(a, at, leaf, named) for a in args];
        any(isnothing, g) ? nothing : g)
    if Meta.isexpr(ex, :., 2) && Meta.isexpr(ex.args[2], :tuple)
        g = inner(ex.args[2].args)
        g === nothing || return Expr(:., ex.args[1], Expr(:tuple, g...))
    elseif Meta.isexpr(ex, :call) && ex.args[1] isa Symbol &&
            length(String(ex.args[1])) > 1 && startswith(String(ex.args[1]), '.')
        g = inner(ex.args[2:end])
        g === nothing || return Expr(:call, ex.args[1], g...)
    end
    return Expr(:ref, named(ex), at)
end

function _merged_observations(ast::Expr, values, observing)
    observed = Set{Symbol}(n for n in observing
        if get(values, n, nothing) isa AbstractVector)
    isempty(observed) && return ast
    defs = Dict{Symbol,Any}()
    stated = Set{Symbol}()
    for st in ast.args
        if Meta.isexpr(st, :(=), 2) && st.args[1] isa Symbol
            defs[st.args[1]] = st.args[2]
        elseif Meta.isexpr(st, :call, 3) && st.args[1] in (:~, :.~)
            lhs = st.args[2]
            base = Meta.isexpr(lhs, :ref) ? lhs.args[1] : lhs
            base isa Symbol && push!(stated, base)
        end
    end
    any(st -> _merged_observation(st, defs, observed) !== nothing, ast.args) || return ast
    names = _marginal_names(ast)
    leaf = _toplevel_shape_leaf(names, values)
    extents = _declared_extents(ast)
    merged = Dict{Symbol,Symbol}()
    out = Any[]
    for st in ast.args
        m = _merged_observation(st, defs, observed)
        m === nothing && (push!(out, st); continue)
        append!(out, _split_merged_observation(m..., values, observed, defs, stated,
            merged, leaf, extents))
    end
    return Expr(:block, out...)
end

function _split_merged_observation(v, parts, order, family, args, values, observed, defs,
        stated, merged, leaf, extents)
    fail(msg) = _sfail("`$v .~ $family.(…)` observes `$v = $(defs[v])`, a merge of " *
        "observed data with other values: " * msg)
    responses = [j for (j, p) in enumerate(parts) if p isa Symbol && p in observed]
    others = [j for j in eachindex(parts) if !(j in responses)]
    for j in others
        _reads_values_any(parts[j], observed, defs) && fail("its part `$(parts[j])` reads " *
            "observed data. Merge each observed vector whole, as a part of its own " *
            "(`vcat(y, z)[order]`)")
    end
    for j in responses
        r = parts[j]
        count(==(r), parts) == 1 || fail("`$r` is merged twice; each entry is observed once")
        r in stated && fail("`$r` is observed by a sampling statement of its own too; " *
            "each entry is observed once")
        haskey(merged, r) && fail("`$r` is merged into `$(merged[r])` too; each entry " *
            "is observed once")
        Missing <: eltype(values[r]) && fail("`$r` holds `missing`; missing entries of " *
            "a merged observed vector are not built yet")
        merged[r] = v
    end
    lengths = Any[]
    for p in parts
        n = _merged_length(p, values, defs, extents)
        n === nothing && fail("the length of its part `$p` is not known from the data; " *
            "write each part from bound data, declared arrays (`z[1:K] .~ …`) and " *
            "broadcasts of them")
        push!(lengths, n)
    end
    call(f, a...) = Expr(:call, GlobalRef(@__MODULE__, f), a...)
    defs_out = Any[]
    O = order
    if O === nothing
        O = _merged_name(v, :order)
        push!(defs_out, :($O = $(call(:_rkppl_merged_identity, lengths...))))
    elseif !(O isa Symbol)
        O = _merged_name(v, :order)
        push!(defs_out, :($O = $order))
    end
    rows = Dict(j => _merged_name(v, Symbol(parts[j], :_rows)) for j in responses)
    for j in responses
        push!(defs_out, :($(rows[j]) = $(call(:_rkppl_merged_rows, O, j, lengths...))))
    end
    mask = sum(1 << (j - 1) for j in responses)
    RR, RP, RV = _merged_name(v, :rest_rows), _merged_name(v, :rest_parts), _merged_name(v, :rest)
    if !isempty(others)
        push!(defs_out,
            :($RR = $(call(:_rkppl_merged_rest_rows, O, mask, lengths...))),
            :($RP = $(call(:_rkppl_merged_rest_parts, O, mask, lengths...))),
            :($RV = vcat($(parts[others]...))[$RP]))
    end
    # Arguments with one value per entry that are not broadcasts are named
    # once, then gathered.
    named_args = Dict{Any,Symbol}()
    arg_defs = Any[]
    named(ex) = get!(named_args, ex) do
        nm = _merged_name(v, Symbol(:argument_, length(named_args) + 1))
        push!(arg_defs, Expr(:(=), nm, ex))
        nm
    end
    function law(at)
        g = Any[]
        for a in args
            ga = _merged_gather(a, at, leaf, named)
            ga === nothing && fail("cannot tell whether its argument `$a` is one value " *
                "or one value per entry of `$v`; write it from declared parameters, " *
                "bound data and definitions of them")
            push!(g, ga)
        end
        return Expr(:., family, Expr(:tuple, g...))
    end
    stmts = Any[Expr(:call, :.~, parts[j], law(rows[j])) for j in responses]
    isempty(others) || push!(stmts, Expr(:call, :.~, RV, law(RR)))
    return vcat(defs_out, arg_defs, stmts)
end

# Where the entries of a merge `vcat(parts…)[order]` come from, given the
# parts' lengths `n`: `bind_data` computes these once. Position `i` of the
# merge holds entry `order[i]` of the concatenation.
function _rkppl_merged_check(order, n)
    L = sum(n; init = 0)
    (order isa AbstractVector{<:Integer} && length(order) == L && isperm(order)) ||
        throw(ContractValidationError("[bind] a merge of observed data " *
            "`vcat(parts…)[order]` needs `order` to be a permutation of 1:$L, one " *
            "position per entry of its parts (of lengths $(join(n, ", ")))"))
    return nothing
end
_rkppl_merged_part(ends, c) = searchsortedfirst(ends, c)
_rkppl_merged_identity(n::Integer...) = collect(1:sum(n; init = 0))

# The positions of part `j`'s entries in the merge, in their order in part `j`.
function _rkppl_merged_rows(order, j::Integer, n::Integer...)
    _rkppl_merged_check(order, n)
    start = sum(n[1:(j - 1)]; init = 0)
    return invperm(order)[(start + 1):(start + n[j])]
end

# The positions in the merge of the entries of the parts outside `mask`
# (bit `j - 1` set for an observed part `j`), in increasing order.
function _rkppl_merged_rest_rows(order, mask::Integer, n::Integer...)
    _rkppl_merged_check(order, n)
    ends = cumsum(collect(Int, n))
    return Int[i for (i, c) in enumerate(order)
        if iszero((mask >> (_rkppl_merged_part(ends, c) - 1)) & 1)]
end

# Those entries' positions in the concatenation of the parts outside `mask`.
function _rkppl_merged_rest_parts(order, mask::Integer, n::Integer...)
    rows = _rkppl_merged_rest_rows(order, mask, n...)
    ns = collect(Int, n)
    starts = cumsum(ns) .- ns
    observed_before(c) = sum((clamp(c - 1 - starts[j], 0, ns[j]) for j in eachindex(ns)
        if !iszero((mask >> (j - 1)) & 1)); init = 0)
    return Int[order[i] - observed_before(order[i]) for i in rows]
end

# A `.~` over a definition that reads observed data and a sampled value, in
# a form the split above does not cover: the refusal names both reads.
function _observed_definition_msg(lhs::Symbol, detmap, observed, model)
    haskey(detmap, lhs) || return nothing
    reads(names) = sort!([string(n) for n in names
        if _reads_values_any(detmap[lhs], Set{Symbol}((n,)), detmap)])
    obs, smp = reads(observed), reads(model)
    (isempty(obs) || isempty(smp)) && return nothing
    return "`$lhs .~ …` observes a definition that reads the observed data " *
        join(("`$o`" for o in obs), ", ") * " and the sampled value(s) " *
        join(("`$s`" for s in smp), ", ") * ". An observation does not depend on " *
        "a sampled value. A merge of whole observed vectors with other values, " *
        "`$lhs = vcat(y, z)[order]` observed with `| (; y)`, observes each entry of " *
        "`y` and adds a density factor for the others"
end
