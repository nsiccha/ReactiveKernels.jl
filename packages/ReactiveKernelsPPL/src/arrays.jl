# Declared array-valued parameters ([`ArrayParameter`](@ref)): validation,
# bind-time sizing, the packed layout entries, and the in-graph value,
# gather and prior statements.
#
# An array parameter is a VALUE the model reads by name, with standard
# Julia semantics: `z` is a `Vector{Float64}` (or a `Matrix{Float64}` for a
# two-axis declaration), `L` is the lower-triangular `Matrix{Float64}`
# Cholesky factor. Expressions over them are ordinary Julia (`z[1]`,
# `L[2, 1]`, `sd .* z`, `B * w` with a bound data matrix
# `B`). The one non-positional read is the level lookup: an axis declared
# as `levels(g)` is indexed by the grouping column's VALUES, so `z[g]`
# reads, for every observation, the element of `z` on that observation's
# level (the established `c[levels(g)]` / `c[g]` meaning).

# ── names ────────────────────────────────────────────────────────────

_array_names(plan::StructuralPlan) = Symbol[p.name for p in plan.array_parameters]

_is_array_param(plan::StructuralPlan, name) =
    name isa Symbol && any(p -> p.name === name, plan.array_parameters)

function _array_param(plan::StructuralPlan, name::Symbol)
    i = findfirst(p -> p.name === name, plan.array_parameters)
    i === nothing && throw(ContractValidationError(
        "[arrays] $name is not an array parameter"))
    return plan.array_parameters[i]
end

_is_structured_array(p::ArrayParameter) = p.family === :lkj_cholesky

# The flat (column-major) packed vector an elementwise array constrains
# into: the array's own name for one axis, a hygienic local reshaped into
# the matrix for two.
_array_flat_name(name::Symbol, ndims::Int) =
    ndims == 1 ? name : Symbol(:_ppl_arrflat_, name)

# ── dims: forms and bind-time sizes ──────────────────────────────────

_is_levels_dim(d) = d isa Expr && d.head === :call && length(d.args) == 2 &&
    d.args[1] === :levels && d.args[2] isa Symbol
_is_axis_dim(d) = d isa Expr && d.head === :call && length(d.args) == 3 &&
    (d.args[1] === :axes || d.args[1] === :size) && d.args[2] isa Symbol &&
    (d.args[3] === 1 || d.args[3] === 2)

function _validate_array_dim(p::ArrayParameter, d)
    d isa Int && d >= 1 && return nothing
    (_is_levels_dim(d) || _is_axis_dim(d) || _levels_count(d) !== nothing) &&
        return nothing
    _fail(p.label, "array $(p.name) axis $(repr(d)) is not a size: axes " *
          "are a literal `1:K` (K ≥ 1), `1:length(levels(g)) - k`, " *
          "`levels(g)`, or `axes(M, d)` / `size(M, d)` of a matrix `M` " *
          "(d = 1 or 2)")
end

# Distinct values of a `levels(g)` axis, in the `levels` order every other
# `c[levels(g)]` declaration uses.
function _array_axis_levels(plan::StructuralPlan, p::ArrayParameter, g::Symbol)
    haskey(plan.columns, g) || _fail(p.label, "array $(p.name) axis " *
        "`levels($g)` needs the bound grouping column $g")
    col = _vector_column(plan.columns, g, p.label, "`levels()` grouping column")
    isempty(col) && _fail(p.label, "array $(p.name) axis `levels($g)`: " *
        "column $g is empty")
    return _grouping_levels(col)
end

function _array_dim_size(plan::StructuralPlan, p::ArrayParameter, d)
    d isa Int && return d
    _is_levels_dim(d) && return length(_array_axis_levels(plan, p, d.args[2]))
    cnt = _levels_count(d)
    if cnt !== nothing
        g, k = cnt
        n = length(_array_axis_levels(plan, p, g)) - k
        n >= 1 || _fail(p.label, "array $(p.name) axis `1:$(repr(d))` is " *
            "empty on the bound data ($g has $(n + k) levels)")
        return n
    end
    fn, M, k = d.args[1], d.args[2], d.args[3]
    m = _find_matrix(plan, M)
    if m !== nothing
        return k == 2 ? length(m.columns) : plan.n_obs
    end
    haskey(plan.columns, M) || _fail(p.label, "array $(p.name) axis " *
        "`$fn($M, $k)` needs a bound matrix $M")
    col = plan.columns[M]
    col isa AbstractMatrix || _fail(p.label, "array $(p.name) axis " *
        "`$fn($M, $k)` sizes over a matrix, but $M is a vector column")
    return size(col, k)
end

"""Concrete axis lengths of an array parameter on a bound plan."""
function _array_dims(plan::StructuralPlan, p::ArrayParameter)
    isbound(plan) || throw(ContractValidationError(
        "[arrays] array $(p.name) sizes resolve on a bound plan only"))
    return Int[_array_dim_size(plan, p, d) for d in p.dims]
end

# ── structure validation ─────────────────────────────────────────────

# Value-expression arguments of an elementwise prior (per element):
# literals, names, literal vectors, or expressions over values.
function _validate_array_arg(p::ArrayParameter, key::Symbol, a)
    if a isa Real
        isfinite(a) || _fail(p.label, "array $(p.name) prior $key must " *
            "be finite, got $(repr(a))")
        return nothing
    end
    a isa Symbol && return nothing
    if a isa Expr && a.head === :vect
        isempty(a.args) && _fail(p.label, "array $(p.name) prior $key is " *
            "an empty vector")
        all(x -> x isa Real && isfinite(x), a.args) || _fail(p.label,
            "array $(p.name) prior $key: literal vectors hold finite " *
            "numbers, got $(repr(a))")
        return nothing
    end
    a isa Expr && return nothing
    _fail(p.label, "array $(p.name) prior $key must be a literal, a " *
          "name, a literal vector or an expression, got $(repr(a))")
end

function _validate_array_parameters(plan::StructuralPlan)
    names = _union_names(plan)
    for p in plan.array_parameters
        _check_name_hygiene(p.name)
        nd = length(p.dims)
        1 <= nd <= 2 || _fail(p.label, "array $(p.name) has $nd axes " *
            "(arrays have one or two)")
        for d in p.dims
            _validate_array_dim(p, d)
        end
        if p.family === :lkj_cholesky
            nd == 2 && p.dims[1] == p.dims[2] || _fail(p.label,
                "LKJCholesky factor $(p.name) is square (K×K), got axes " *
                "$(repr(p.dims))")
            keys(p.args) == (:arg1,) || _fail(p.label,
                "LKJCholesky factor $(p.name) takes the shape `eta` only")
            eta = p.args.arg1
            eta isa Real && isfinite(eta) && eta > 0 || _fail(p.label,
                "LKJCholesky factor $(p.name): `eta` must be a finite " *
                "positive literal, got $(repr(eta))")
            p.support_override === nothing || _fail(p.label,
                "LKJCholesky factor $(p.name) carries no support override")
        else
            haskey(SAMPLED_ARITY, p.family) || _fail(p.label,
                "array $(p.name) family $(p.family) unknown (admitted: " *
                "$(join(sort!(collect(keys(SAMPLED_ARITY))), ", ")), " *
                "lkj_cholesky)")
            want = ntuple(i -> Symbol(:arg, i), SAMPLED_ARITY[p.family])
            Tuple(keys(p.args)) == want || _fail(p.label,
                "array $(p.name) family $(p.family) takes positional keys " *
                "$want, got $(Tuple(keys(p.args)))")
            try
                support_of(p.family, p.support_override)
            catch err
                err isa ContractValidationError || rethrow()
                _fail(p.label, "array $(p.name): " * err.message)
            end
            for (k, a) in pairs(p.args)
                _validate_array_arg(p, k, a)
            end
            if p.family === :uniform
                lo, hi = p.args.arg1, p.args.arg2
                lo isa Real && hi isa Real && lo < hi || _fail(p.label,
                    "array $(p.name): `Uniform.(lo, hi)` bounds are " *
                    "literals with lo < hi, got ($(repr(lo)), $(repr(hi)))")
            end
            if nd == 2
                # Two-axis arrays take shared (scalar) prior arguments:
                # per-element arguments would need a matching matrix.
                for (k, a) in pairs(p.args)
                    a isa Real || (a isa Symbol && a in names) || _fail(
                        p.label, "array $(p.name) has two axes; its prior " *
                        "$k must be a literal or a scalar parameter/" *
                        "assignment name, got $(repr(a))")
                end
            end
        end
    end
    return nothing
end

# ── data validation ──────────────────────────────────────────────────

function _validate_array_parameters_data(plan::StructuralPlan)
    for p in plan.array_parameters
        dims = _array_dims(plan, p)
        all(>=(1), dims) || _fail(p.label, "array $(p.name) has an empty " *
            "axis (sizes $(repr(dims)))")
        if p.family === :lkj_cholesky
            dims[1] == dims[2] || _fail(p.label, "LKJCholesky factor " *
                "$(p.name) is not square: $(repr(dims))")
            continue
        end
        _is_structured_array(p) && continue
        length(dims) == 1 || continue
        K = dims[1]
        for (k, a) in pairs(p.args)
            n = _array_arg_length(plan, a)
            n === nothing && continue
            n == K || _fail(p.label, "array $(p.name) has $K elements but " *
                "its prior $k has $n (per-element arguments match the " *
                "array; shared ones are scalars)")
        end
    end
    return nothing
end

# Statically known element count of a per-element prior argument
# (`nothing` = scalar or only known at run time).
function _array_arg_length(plan::StructuralPlan, a)
    a isa Expr && a.head === :vect && return length(a.args)
    a isa Symbol || return nothing
    _is_array_param(plan, a) &&
        return prod(_array_dims(plan, _array_param(plan, a)))
    _is_derived(plan, a) && return plan.n_obs
    if haskey(plan.columns, a)
        col = plan.columns[a]
        col isa AbstractVector || _fail(:plan, "prior argument $a is a " *
            "matrix column (per-element arguments are vectors)")
        return length(col)
    end
    return nothing
end

# ── value expressions over arrays ────────────────────────────────────

# True when `ex` reads an array parameter anywhere, directly or through
# assignments computed from one.
function _mentions_array(ex, plan::StructuralPlan,
        seen::Set{Symbol} = Set{Symbol}())
    if ex isa Symbol
        _is_array_param(plan, ex) && return true
        ex in seen && return false
        push!(seen, ex)
        i = findfirst(a -> a.name === ex, plan.assignments)
        return i !== nothing &&
            _mentions_array(plan.assignments[i].expr, plan, seen)
    end
    ex isa Expr || return false
    return any(a -> _mentions_array(a, plan, seen), ex.args)
end

# An assignment computed from array parameters (`M = (sd .* L)'`) is an
# array value too: readable by position (its axes are its expression's).
_is_array_assignment(plan::StructuralPlan, name) =
    name isa Symbol && any(a -> a.name === name && _mentions_array(a.expr, plan),
        plan.assignments)

# A literal position or a whole axis.
_is_position(i) = (i isa Int && i >= 1) || i === :(:)

# A per-observation index: a column name (data or derived).
_is_row_index(plan::StructuralPlan, i) =
    i isa Symbol && i !== :(:) && !_is_array_param(plan, i) &&
    !(i in _union_names(plan))

# Index forms of `A[...]` over an array parameter `A`:
# `:scalar` (every index a literal Int), `:slice` (literal Ints and `:`),
# `:gather` (the first index a data column — one value per observation,
# the rest literal positions or `:`), or `:invalid`.
function _array_index_kind(plan::StructuralPlan, ex::Expr)
    idx = ex.args[2:end]
    isempty(idx) && return :invalid
    all(i -> i isa Int && i >= 1, idx) && return :scalar
    all(_is_position, idx) && return :slice
    _is_row_index(plan, idx[1]) && all(_is_position, idx[2:end]) &&
        return :gather
    return :invalid
end

# Validate one `A[...]` read and record its references. `allow_gather`:
# only per-observation (derived-column) expressions may gather.
function _collect_array_ref!(refs, ex::Expr, plan::StructuralPlan, label,
        bound::Bool; allow_gather::Bool)
    base = ex.args[1]
    if _is_array_assignment(plan, base)
        all(_is_position, ex.args[2:end]) && !isempty(ex.args[2:end]) ||
            _fail(label, "`$(repr(ex))` reads the array value $base by " *
                "literal positions or `:` only (gather per observation " *
                "from the declared array parameter itself)")
        push!(refs, base)
        return true
    end
    _is_array_param(plan, base) || return false
    p = _array_param(plan, base)
    kind = _array_index_kind(plan, ex)
    kind === :invalid && _fail(label, "indexing `$(repr(ex))` reads array " *
        "$(base) by literal positions (`$(base)[1]`, `$(base)[2, 1]`, " *
        "`$(base)[:, 1]`) or, in a per-observation expression, by one " *
        "data column (`$(base)[g]`)")
    kind === :gather && !allow_gather && _fail(label, "`$(repr(ex))` " *
        "gathers per observation — write it in a vector (per-observation) " *
        "definition, not a scalar one")
    push!(refs, base)
    if kind === :gather
        length(ex.args) - 1 == length(p.dims) || _fail(label,
            "`$(repr(ex))` gathers from the $(length(p.dims))-axis array " *
            "$(base) with $(length(ex.args) - 1) indices (one per axis)")
        p.family === :lkj_cholesky && _fail(label,
            "`$(repr(ex))` gathers from the LKJCholesky factor $(base)")
        g = ex.args[2]
        _is_derived(plan, g) && _fail(label, "`$(repr(ex))` gathers by the " *
            "derived column $g — gathers read raw data columns")
        bound && _validate_gather_data(plan, p, g, label)
    elseif bound
        dims = _array_dims(plan, p)
        idx = ex.args[2:end]
        if length(idx) == length(dims)
            for (i, d) in zip(idx, dims)
                i isa Int && i > d && _fail(label, "`$(repr(ex))` is out of " *
                    "bounds: $(base) has size $(repr(Tuple(dims)))")
            end
        elseif !(length(idx) == 1 && idx[1] isa Int && idx[1] <= prod(dims))
            _fail(label, "`$(repr(ex))` indexes $(length(idx)) axes of the " *
                "$(length(dims))-axis array $(base)")
        end
    end
    return true
end

# A gather `z[g]` at bind: a `levels(h)` axis needs every value of `g` on
# that axis; an integer axis (`1:K`, `axes(M, d)`) needs integer `g` in
# range — plain Julia indexing.
function _validate_gather_data(plan::StructuralPlan, p::ArrayParameter,
        g::Symbol, label)
    haskey(plan.columns, g) || _fail(label, "array $(p.name) is read by " *
        "`$(p.name)[$g]`, but $g is not a bound column")
    col = _vector_column(plan.columns, g, label, "gather index")
    d = p.dims[1]
    if _is_levels_dim(d)
        lv = _array_axis_levels(plan, p, d.args[2])
        codes = _declared_codes(col, lv)
        any(==(0), codes) && _fail(label, "`$(p.name)[$g]` looks values of " *
            "$g up on the axis `levels($(d.args[2]))` of $(p.name), but " *
            "$g holds values not on that axis")
    else
        K = _array_dim_size(plan, p, d)
        eltype(col) <: Integer && eltype(col) !== Bool || _fail(label,
            "`$(p.name)[$g]` indexes the integer axis of $(p.name) " *
            "(size $K) by $g, which does not hold integers — declare the " *
            "axis `levels($g)` to look values up by level")
        all(i -> 1 <= i <= K, col) || _fail(label, "`$(p.name)[$g]`: " *
            "$g holds indices outside 1:$K")
    end
    return nothing
end

# Array-valued expressions (`sd .* z`, `L[:, 1]`, `phi[1] * 2`): every
# leaf is a literal, a scalar name or an array parameter (never a
# per-observation column); dotted and undotted arithmetic, the scalar
# math allowlist (dotted or not), reductions over an array, indexing and
# adjoints admitted. Records references.
function _collect_array_value_refs!(refs, ex, plan::StructuralPlan, label,
        bound::Bool)
    ex isa Number && return nothing
    ex isa LineNumberNode && return nothing
    if ex isa Symbol
        (_is_array_param(plan, ex) || ex in _union_names(plan)) &&
            return push!(refs, ex)
        _is_derived(plan, ex) && _fail(label, "derived column $ex is " *
            "per-observation — it does not combine with arrays here")
        bound && haskey(plan.columns, ex) && _fail(label, "column $ex is " *
            "per-observation — it does not combine with arrays here")
        return push!(refs, ex)
    end
    ex isa Expr || _fail(label, "unsupported literal $(repr(ex))")
    head = ex.head
    if head === :ref
        _collect_array_ref!(refs, ex, plan, label, bound;
            allow_gather = false) || _fail(label, "indexing `$(repr(ex))` " *
            "reads array parameters only")
        return nothing
    end
    if head === Symbol("'")
        for a in ex.args
            _collect_array_value_refs!(refs, a, plan, label, bound)
        end
        return nothing
    end
    if head === :call
        fn = ex.args[1]
        # A reduction over a data column inside an array expression is a
        # scalar subterm (`mean(x) .* z`).
        fn isa Symbol && fn in REDUCTION_FNS && length(ex.args) == 2 &&
            !_mentions_array(ex.args[2], plan) &&
            return _collect_assignment_refs!(refs, ex, plan, label, bound)
        fn isa Symbol && (fn in ASSIGNMENT_FNS || fn in ELEMENTWISE_OPS ||
            fn === :transpose) || _fail(label, "call `$fn` is not in the " *
            "array-value vocabulary (arithmetic, dotted arithmetic, the " *
            "scalar math functions, reductions, `transpose`)")
        for a in ex.args[2:end]
            _collect_array_value_refs!(refs, a, plan, label, bound)
        end
        return nothing
    end
    if head === :.
        length(ex.args) == 2 && ex.args[1] isa Symbol &&
            ex.args[2] isa Expr && ex.args[2].head === :tuple || _fail(label,
            "field access does not lower in array expressions")
        ex.args[1] in ELEMENTWISE_FNS || _fail(label, "`$(ex.args[1]).` " *
            "is not in the elementwise vocabulary")
        for a in ex.args[2].args
            _collect_array_value_refs!(refs, a, plan, label, bound)
        end
        return nothing
    end
    _fail(label, "unsupported expression head $head in an array expression")
end

# `B * w`: a per-observation matrix — a bound data matrix `B`, or the
# rows of an array gathered per observation (`z[g, :]`) — times an
# array-valued expression: one value per observation (standard Julia
# matrix-vector product).
function _is_data_matvec(ex, plan::StructuralPlan)
    ex isa Expr && ex.head === :call && length(ex.args) == 3 &&
        ex.args[1] === :* || return false
    B, w = ex.args[2], ex.args[3]
    rows = (B isa Symbol && !_is_array_param(plan, B) &&
            !(B in _union_names(plan)) && !_is_derived(plan, B)) ||
        _is_row_gather(plan, B)
    return rows && _mentions_array(w, plan)
end

# `z[g, :]`: the rows of a two-axis array, one per observation.
_is_row_gather(plan::StructuralPlan, ex) =
    ex isa Expr && ex.head === :ref && length(ex.args) == 3 &&
    _is_array_param(plan, ex.args[1]) && ex.args[3] === :(:) &&
    _array_index_kind(plan, ex) === :gather

function _collect_data_matvec!(refs, ex::Expr, plan::StructuralPlan, label,
        bound::Bool)
    B, w = ex.args[2], ex.args[3]
    _collect_array_value_refs!(refs, w, plan, label, bound)
    if B isa Expr
        _collect_array_ref!(refs, B, plan, label, bound; allow_gather = true)
        return nothing
    end
    bound || return nothing
    haskey(plan.columns, B) || _fail(label, "`$(repr(ex))`: $B is not a " *
        "bound column")
    col = plan.columns[B]
    col isa AbstractMatrix || _fail(label, "`$(repr(ex))` multiplies $B by " *
        "an array — $B must be a bound matrix (a vector times a vector is " *
        "not a product in Julia; write elementwise math dotted)")
    eltype(col) <: Real || _fail(label, "matrix $B is not numeric")
    if w isa Symbol && _is_array_param(plan, w)
        dims = _array_dims(plan, _array_param(plan, w))
        length(dims) == 1 && dims[1] == size(col, 2) || _fail(label,
            "`$(repr(ex))`: $B has $(size(col, 2)) columns but $w has " *
            "size $(repr(Tuple(dims)))")
    end
    return nothing
end

# ── layout ───────────────────────────────────────────────────────────

# Per-coordinate labels: `z.i` for one axis, Stan's `z.i.j` (row i,
# column j, column-major packing) for two.
function _array_labels(name::Symbol, dims::Vector{Int})
    length(dims) == 1 && return [Symbol(name, ".", i) for i in 1:dims[1]]
    return [Symbol(name, ".", i, ".", j) for j in 1:dims[2] for i in 1:dims[1]]
end

"""Append the packed layout entries of the plan's array parameters
starting at `offset`; returns the next free offset."""
function _array_layout_entries!(entries::Vector{LayoutEntry},
        plan::StructuralPlan, offset::Int)
    for p in plan.array_parameters
        dims = _array_dims(plan, p)
        if p.family === :lkj_cholesky
            K = dims[1]
            packed = K * (K - 1) ÷ 2
            labels = [Symbol(p.name, ".", i) for i in 1:packed]
            push!(entries, LayoutEntry(:cholesky_corr, nothing, p.name,
                labels, offset, packed, :lkj))
            offset += packed
        else
            transform, lo, hi =
                _entry_transform(p.family, p.support_override, p.args)
            n = prod(dims)
            push!(entries, LayoutEntry(:array, nothing, p.name,
                _array_labels(p.name, dims), offset, n, transform, lo, hi,
                dims))
            offset += n
        end
    end
    return offset
end

# In-graph constrain edges of an elementwise array: the plate edges over
# its flat packed block, then (two axes) the column-major reshape.
function _array_transform_statements(e::LayoutEntry)
    flat = _array_flat_name(e.name, length(e.dims))
    fe = LayoutEntry(:plate, nothing, flat, e.labels, e.offset, e.size,
        e.transform, e.lo, e.hi)
    stmts = _plate_transform_statements(fe)
    length(e.dims) == 1 && return stmts
    push!(stmts, :($(e.name)::Matrix{Float64} =
        reshape(Float64.($flat), $(e.dims...))))
    return stmts
end

function _array_jacobian_term(e::LayoutEntry)
    flat = _array_flat_name(e.name, length(e.dims))
    return jacobian_term(LayoutEntry(:plate, nothing, flat, e.labels,
        e.offset, e.size, e.transform, e.lo, e.hi))
end

# ── generator ────────────────────────────────────────────────────────

# `L::Matrix{Float64}` assembled from the LKJ vine's lower-triangle
# scalars (`_ppl_rl_<L>_<i>_<j>`), row-major `hvcat` (zeros above the
# diagonal) — the value the model reads; the prior and the log-Jacobian
# keep reading the scalars.
function _lkj_matrix_statement(L::Symbol, K::Int)
    elems = Any[i >= j ? _rl_name(L, i, j) : 0.0 for i in 1:K for j in 1:K]
    rows = Expr(:tuple, ntuple(_ -> K, K)...)
    return :($L::Matrix{Float64} = hvcat($rows, $(elems...)))
end

# Name of the level-code vector for gathers by column `g` on the
# `levels(h)` axis.
_array_level_index_name(g::Symbol, h::Symbol) = Symbol(:_ppl_lvx_, h, :_, g)

# Rewrite every level gather `z[g]` (a `levels(h)` axis) to an integer
# gather over the level codes; integer axes stay plain Julia indexing.
# Records the (g, h) code vectors it needs.
function _array_gather_rewrite(ex, plan::StructuralPlan,
        needed::Set{Tuple{Symbol,Symbol}})
    ex isa Expr || return ex
    if ex.head === :ref && _is_array_param(plan, ex.args[1]) &&
            _array_index_kind(plan, ex) === :gather
        p = _array_param(plan, ex.args[1])
        d = p.dims[1]
        if _is_levels_dim(d)
            g, h = ex.args[2], d.args[2]
            push!(needed, (g, h))
            return Expr(:ref, p.name, _array_level_index_name(g, h),
                ex.args[3:end]...)
        end
        return ex
    end
    return Expr(ex.head,
        (_array_gather_rewrite(a, plan, needed) for a in ex.args)...)
end

# The level-code vectors a plan's gathers read (data-only: `bound=` folds
# them), one per (index column, axis column).
function _array_level_index_statements(plan::StructuralPlan,
        needed::Set{Tuple{Symbol,Symbol}})
    stmts = Expr[]
    for (g, h) in sort!(collect(needed))
        p = first(q for q in plan.array_parameters
            if _is_levels_dim(q.dims[1]) && q.dims[1].args[2] === h)
        lv = _array_axis_levels(plan, p, h)
        lvlvec = Expr(:vect, (_level_literal(l) for l in lv)...)
        push!(stmts, :($(_array_level_index_name(g, h)) =
            _declared_codes($g, $lvlvec)))
    end
    return stmts
end

# Value statements after the layout transforms: each LKJ factor's matrix.
function _array_value_statements(plan::StructuralPlan)
    stmts = Expr[]
    for p in plan.array_parameters
        p.family === :lkj_cholesky || continue
        push!(stmts, _lkj_matrix_statement(p.name, _array_dims(plan, p)[1]))
    end
    return stmts
end

# Prior nodes `_ppl_prior_<name>` of the plan's array parameters.
function _array_prior_stmts!(stmts::Vector{Expr}, terms::Vector{Any},
        plan::StructuralPlan, needed::Set{Tuple{Symbol,Symbol}})
    for p in plan.array_parameters
        dims = _array_dims(plan, p)
        node = Symbol(:_ppl_prior_, p.name)
        if p.family === :lkj_cholesky
            push!(stmts, :($node::Float64 =
                $(_lkj_prior_terms(p.name, dims[1], Float64(p.args.arg1)))))
            push!(terms, node)
            continue
        end
        if p.family === :flat
            push!(stmts, :($node::Float64 = 0.0))
            push!(terms, node)
            continue
        end
        flat = _array_flat_name(p.name, length(dims))
        args = Pair{Symbol,Any}[]
        for (i, (k, a)) in enumerate(pairs(p.args))
            if a isa Real || a isa Symbol
                push!(args, k => a)
                continue
            end
            local_name = Symbol(:_ppl_parg_, p.name, :_, i)
            val = a.head === :vect ? :(Float64[$(a.args...)]) :
                _array_gather_rewrite(a, plan, needed)
            push!(stmts, :($local_name = $val))
            push!(args, k => local_name)
        end
        _vector_prior_stmts!(stmts, terms, flat, p.family,
            NamedTuple{Tuple(first.(args))}(Tuple(last.(args))),
            p.support_override)
    end
    return nothing
end
