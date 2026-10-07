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

# Multivariate slice families (`mv_slices.jl`): family → (stem,
# slices). `:rows` / `:cols` slice a two-axis array, `:vector` is a
# one-axis array drawn once; simplex and ordered slices are two-axis only
# (their one-vector forms are `phi ~ Dirichlet(alpha)`,
# `c ~ Ordered(Normal(m, s), K)`).
const _MV_SLICE_FAMILIES = Dict{Symbol,Tuple{Symbol,Symbol}}(
    Symbol(stem, :_, sl) => (stem, sl)
    for (stem, sls) in ((:mvnormal_cholesky, (:rows, :cols, :vector)),
        (:mvnormal, (:rows, :cols, :vector)), (:dirichlet, (:rows, :cols)),
        (:ordered_normal, (:rows, :cols)))
    for sl in sls)

_is_slice_array(p::ArrayParameter) = haskey(_MV_SLICE_FAMILIES, p.family)
_slice_stem(p::ArrayParameter) = _MV_SLICE_FAMILIES[p.family][1]
_slice_kind(p::ArrayParameter) = _MV_SLICE_FAMILIES[p.family][2]

_is_structured_array(p::ArrayParameter) =
    p.family in (:lkj_cholesky, :lkj_cholesky_stack) || _is_slice_array(p)

# The flat (column-major) packed vector an elementwise array constrains
# into: the array's own name for one axis, a hygienic local reshaped into
# the matrix for two.
_array_flat_name(name::Symbol, ndims::Int) =
    ndims == 1 ? name : Symbol(:_ppl_arrflat_, name)

# ── dims: forms and bind-time sizes ──────────────────────────────────

_is_levels_dim(d) = d isa Expr && d.head === :call &&
    length(d.args) in (2, 3) && d.args[1] in (:levels, :unique, :_ppl_axis_values) &&
    d.args[2] isa Symbol && (length(d.args) == 2 || d.args[3] isa QuoteNode)
_levels_subset(d) = length(d.args) == 2 ? Colon() : d.args[3].value
_is_axis_dim(d) = d isa Expr && d.head === :call && length(d.args) == 3 &&
    (d.args[1] === :axes || d.args[1] === :size) && d.args[2] isa Symbol &&
    (d.args[3] === 1 || d.args[3] === 2)

function _validate_array_dim(p::ArrayParameter, d)
    d isa Int && d >= 0 && return nothing
    if _is_levels_dim(d)
        _validate_subset_shape(LevelMap(p.name, d.args[2], [], :levels,
            _levels_subset(d)))
        return nothing
    end
    (_is_levels_dim(d) || _is_axis_dim(d) || _levels_count(d) !== nothing) &&
        return nothing
    _fail(p.label, "array $(p.name) axis $(repr(d)) is not a size: axes " *
          "are a literal `1:K` (K ≥ 0), `1:length(levels(g)) - k`, " *
          "`levels(g)`, or `axes(M, d)` / `size(M, d)` of a matrix `M` " *
          "(d = 1 or 2)")
end

# Distinct values of a `levels(g)` axis, in the `levels` order every other
# `c[levels(g)]` declaration uses. `g` is a bound column: raw data or a
# data definition `bind_data` evaluated (`gg = vcat(g1, g2)`). `name` /
# `label` name the array value the axis belongs to (messages only).
function _array_axis_levels(plan::StructuralPlan, name::Symbol, label,
        g::Symbol)
    haskey(plan.columns, g) || _fail(label, "array $name axis " *
        "`levels($g)` needs the bound grouping column $g")
    col = _vector_column(plan.columns, g, label, "`levels()` grouping column")
    return _grouping_levels(col)
end
_array_axis_levels(plan::StructuralPlan, p::ArrayParameter, g::Symbol) =
    _array_axis_levels(plan, p.name, p.label, g)

function _array_axis_levels(plan::StructuralPlan, name::Symbol, label, d::Expr)
    d.args[1] === :levels && return _array_axis_levels(plan, name, label, d.args[2])
    g = d.args[2]
    haskey(plan.columns, g) || _fail(label, "array $name axis needs bound data $g")
    values = _vector_column(plan.columns, g, label, "array level values")
    return d.args[1] === :unique ? unique(values) : collect(values)
end

function _array_dim_size(plan::StructuralPlan, name::Symbol, label, d;
        active = Set{Symbol}())
    d isa Int && return d
    if _is_levels_dim(d)
        levels = _array_axis_levels(plan, name, label, d)
        return length(_apply_subset(levels,
            LevelMap(name, d.args[2], [], :levels, _levels_subset(d))))
    end
    cnt = _levels_count(d)
    if cnt !== nothing
        g, k = cnt
        n = length(_array_axis_levels(plan, name, label, g)) - k
        n >= 0 || _fail(label, "array $name axis `1:$(repr(d))` has " *
            "negative size $n on the bound data ($g has $(n + k) levels)")
        return n
    end
    fn, M, k = d.args[1], d.args[2], d.args[3]
    m = _find_matrix(plan, M)
    if m !== nothing
        return k == 2 ? length(m.columns) : _value_rows(plan, M)
    end
    if !haskey(plan.columns, M)
        shape = _value_axes(plan, M, active; data_axes = true)
        # Julia supplies singleton dimensions beyond an array's rank.
        shape === nothing || return k <= length(shape) ? shape[k] : 1
    end
    haskey(plan.columns, M) || _fail(label, "array $name axis " *
        "`$fn($M, $k)` needs a bound matrix $M")
    col = plan.columns[M]
    col isa AbstractArray || _fail(label, "array $name axis " *
        "`$fn($M, $k)` needs an array, got $(summary(col))")
    return size(col, k)
end

_array_dim_size(plan::StructuralPlan, p::ArrayParameter, d; kwargs...) =
    _array_dim_size(plan, p.name, p.label, d; kwargs...)

"""Concrete axis lengths of an array parameter on a bound plan."""
function _array_dims(plan::StructuralPlan, p::ArrayParameter)
    p.family === :external && return Int[_sampling_extent(plan, d, p.name) for d in p.dims]
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
        for x in a.args
            _validate_array_arg(p, key, x)
        end
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
        if p.family === :external
            _validate_external_parameter(plan, p)
            continue
        end
        nd = length(p.dims)
        stack = p.family === :lkj_cholesky_stack
        (1 <= nd <= 2 || (stack && nd == 3)) || _fail(p.label,
            "array $(p.name) has $nd axes (arrays have one or two; a " *
            "per-level LKJ stack three)")
        for d in p.dims
            _validate_array_dim(p, d)
        end
        if stack
            p.dims[1] isa Int && p.dims[1] >= 2 && p.dims[1] == p.dims[2] &&
                _is_levels_dim(p.dims[3]) || _fail(p.label,
                "per-level LKJCholesky factors $(p.name) have axes " *
                "[K, K, levels(g)] with a literal K ≥ 2, got " *
                "$(repr(p.dims))")
            _validate_lkj_args(p)
            eta = p.args.arg1
            _validate_lkj_eta(p, eta)
            p.support_override === nothing || _fail(p.label,
                "per-level LKJCholesky factors $(p.name) carry no support " *
                "override")
        elseif p.family === :lkj_cholesky
            nd == 2 && p.dims[1] == p.dims[2] || _fail(p.label,
                "LKJCholesky factor $(p.name) is square (K×K), got axes " *
                "$(repr(p.dims))")
            _validate_lkj_args(p)
            eta = p.args.arg1
            _validate_lkj_eta(p, eta)
            p.support_override === nothing || _fail(p.label,
                "LKJCholesky factor $(p.name) carries no support override")
        elseif _is_slice_array(p)
            _validate_array_slices(plan, p)
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
                all(x -> !(x isa Real) || isfinite(x), (lo, hi)) ||
                    _fail(p.label, "array $(p.name): Uniform bounds must be finite")
                if lo isa Real && hi isa Real
                    lo < hi || _fail(p.label, "array $(p.name): " *
                        "Uniform bounds need lo < hi")
                end
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

_lkj_uplo(p::ArrayParameter) = get(p.args, :uplo, 'L')
_is_lkj_stack(t::Symbol) = t in (:lkj_stack, :lkj_stack_upper)
_is_upper_lkj(t::Symbol) = t in (:lkj_upper, :lkj_stack_upper)

function _validate_lkj_args(p::ArrayParameter)
    keys(p.args) in ((:arg1,), (:arg1, :uplo)) || _fail(p.label,
        "LKJCholesky factor $(p.name) takes eta and optional uplo")
    _lkj_uplo(p) in ('L', 'U') || _fail(p.label,
        "LKJCholesky factor $(p.name): uplo is 'L' or 'U'")
end

function _validate_lkj_eta(p::ArrayParameter, eta)
    eta isa Symbol && return nothing
    eta isa Real && !(eta isa Bool) && isfinite(eta) && eta > 0 &&
        return nothing
    _fail(p.label, "LKJCholesky factor $(p.name): `eta` must be a " *
        "positive scalar literal or a declared value, got $(repr(eta))")
end

# ── data validation ──────────────────────────────────────────────────

function _validate_array_parameters_data(plan::StructuralPlan)
    known = union(Set{Symbol}(_all_names(plan)), Set{Symbol}(keys(plan.columns)))
    for p in plan.array_parameters
        dims = _array_dims(plan, p)
        p.family === :external && continue
        all(>=(0), dims) || _fail(p.label, "array $(p.name) has a negative " *
            "axis size (sizes $(repr(dims)))")
        _is_structured_array(p) && !all(>=(1), dims) && _fail(p.label,
            "structured array $(p.name) has an empty axis (sizes $(repr(dims)))")
        if p.family in (:lkj_cholesky, :lkj_cholesky_stack)
            dims[1] == dims[2] || _fail(p.label, "LKJCholesky factor " *
                "$(p.name) is not square: $(repr(dims))")
            eta = p.args.arg1
            if eta isa Symbol
                eta in known || _fail(p.label, "LKJCholesky factor " *
                    "$(p.name) references unknown eta $eta")
                if haskey(plan.columns, eta)
                    v = plan.columns[eta]
                    v isa Real && !(v isa Bool) || _fail(p.label,
                        "LKJCholesky factor $(p.name): eta $eta must be a scalar")
                end
            end
            continue
        end
        if _is_slice_array(p)
            _validate_array_slices_data(plan, p, dims)
            continue
        end
        _is_structured_array(p) && continue
        for (key, arg) in pairs(p.args), name in _expr_value_symbols(arg)
            name in known || _fail(p.label,
                "array $(p.name) prior $key references unknown name $name")
        end
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
    _is_derived(plan, a) && return _value_rows(plan, a)
    if haskey(plan.columns, a)
        col = plan.columns[a]
        col isa Real && return nothing
        col isa AbstractVector || _fail(:plan, "prior argument $a is a " *
            "matrix column (per-element arguments are vectors)")
        return length(col)
    end
    return nothing
end

# ── multivariate slices ──────────────────────────────────────────────

# `zeros(K)` / `fill(a, K)`: a literal-length constant vector (`fill` is
# the symmetric `Dirichlet(K, a)` concentration).
_is_zeros_call(a) = a isa Expr && a.head === :call && length(a.args) == 2 &&
    a.args[1] === :zeros && a.args[2] isa Int && a.args[2] >= 1
_is_fill_call(a) = a isa Expr && a.head === :call && length(a.args) == 3 &&
    a.args[1] === :fill && a.args[3] isa Int && a.args[3] >= 1

# The family argument roles of each slice stem, in positional-key order:
# `:vector` (a K-vector, shared or per slice), `:matrix` (a shared K×K
# matrix), `:scalar` (a shared scalar), `:count` (the literal slice
# length).
_slice_roles(::Val{:mvnormal_cholesky}) = (arg1 = :vector, arg2 = :matrix)
_slice_roles(::Val{:mvnormal}) = (arg1 = :vector, arg2 = :matrix)
_slice_roles(::Val{:dirichlet}) = (arg1 = :vector,)
_slice_roles(::Val{:ordered_normal}) =
    (arg1 = :scalar, arg2 = :scalar, arg3 = :count)
_slice_roles(p::ArrayParameter) = _slice_roles(Val(_slice_stem(p)))

_slice_noun(p::ArrayParameter) = _slice_kind(p) === :rows ? "rows" :
    _slice_kind(p) === :cols ? "columns" : "draw"

# A slice array's arguments are model-level values — literal vectors,
# `zeros(K)` / `fill(a, K)`, array names or expressions over them, a
# per-slice `eachrow(M)` / `eachcol(M)` of one — never per-observation
# data or gathers.
function _validate_array_slices(plan::StructuralPlan, p::ArrayParameter)
    want = _slice_kind(p) === :vector ? 1 : 2
    length(p.dims) == want || _fail(p.label, "array $(p.name) " *
        "$(_slice_noun(p)) need $want axes, got $(length(p.dims))")
    roles = _slice_roles(p)
    keys(p.args) == keys(roles) || _fail(p.label, "array $(p.name) " *
        "family $(p.family) takes keys $(keys(roles)), got " *
        "$(Tuple(keys(p.args)))")
    p.support_override === nothing || _fail(p.label, "array $(p.name) " *
        "carries no support override")
    for (k, a) in pairs(p.args)
        role = roles[k]
        if role === :count
            a isa Int && a >= 1 || _fail(p.label, "array $(p.name): the " *
                "slice length is a literal ≥ 1, got $(repr(a))")
            continue
        end
        if _is_slice_iterator(a)
            role === :vector && _slice_kind(p) !== :vector || _fail(p.label,
                "array $(p.name): argument $k is shared by every slice, " *
                "got the per-slice $(repr(a))")
            a = a.args[2]
        end
        if a isa Real || a isa Bool
            role === :scalar && !(a isa Bool) && isfinite(a) ||
                _fail(p.label, "array $(p.name): argument $k is a " *
                    "$role, got the scalar $(repr(a))")
            continue
        end
        if _is_zeros_call(a) || _is_fill_call(a)
            role === :vector || _fail(p.label, "array $(p.name): " *
                "argument $k is a $role, got the vector $(repr(a))")
            _is_fill_call(a) && !(a.args[2] isa Real) &&
                _collect_array_value_refs!(Symbol[], a.args[2], plan,
                    p.label, isbound(plan))
            continue
        end
        if a isa Expr && a.head === :vect
            role === :vector || _fail(p.label, "array $(p.name): argument " *
                "$k is a $role, got the vector $(repr(a))")
            all(x -> x isa Real && !(x isa Bool) && isfinite(x), a.args) ||
                _fail(p.label, "array $(p.name): a literal vector holds " *
                    "finite numbers, got $(repr(a))")
            continue
        end
        a isa Symbol && a === p.name && _fail(p.label, "array $(p.name) " *
            "cannot parameterize its own prior")
        _collect_array_value_refs!(Symbol[], a, plan, p.label,
            isbound(plan))
    end
    if _slice_stem(p) === :dirichlet
        alpha = p.args.arg1
        lits = alpha isa Expr && alpha.head === :vect ? alpha.args :
            _is_fill_call(alpha) && alpha.args[2] isa Real ? [alpha.args[2]] :
            Any[]
        all(x -> x > 0, lits) || _fail(p.label, "array $(p.name): Dirichlet " *
            "concentrations are positive, got $(repr(alpha))")
    end
    if _slice_stem(p) === :ordered_normal
        sd = p.args.arg2
        sd isa Real && !(sd > 0) && _fail(p.label, "array $(p.name): the " *
            "`Normal` scale is positive, got $(repr(sd))")
    end
    return nothing
end

# Statically known size of a slice argument (`nothing` when only known
# at run time): literal vectors, `zeros(K)` / `fill(a, K)`, declared
# arrays, bound data values.
function _slice_arg_size(plan::StructuralPlan, a)
    a isa Expr && a.head === :vect && return (length(a.args),)
    _is_zeros_call(a) && return (a.args[2],)
    _is_fill_call(a) && return (a.args[3],)
    _is_array_param(plan, a) &&
        return Tuple(_array_dims(plan, _array_param(plan, a)))
    a isa Symbol && haskey(plan.columns, a) && return size(plan.columns[a])
    return nothing
end

# Statically known shapes must agree with the slices: slice length K,
# slice count G (`eachrow`: dims = (G, K); `eachcol`: (K, G); a vector:
# (K,), G = 1). A shared vector argument has size (K,), a per-slice
# `eachrow(M)` (G, K) and `eachcol(M)` (K, G), a matrix (K, K), the
# `Ordered` length K. Computed values are checked when the density runs.
function _validate_array_slices_data(plan::StructuralPlan, p::ArrayParameter,
        dims::Vector{Int})
    kind = _slice_kind(p)
    G, K = kind === :rows ? (dims[1], dims[2]) :
        kind === :cols ? (dims[2], dims[1]) : (1, dims[1])
    roles = _slice_roles(p)
    for (k, a) in pairs(p.args)
        role = roles[k]
        if role === :count
            a == K || _fail(p.label, "array $(p.name) has $(_slice_noun(p)) " *
                "of length $K, but `Ordered` declares length $a")
            continue
        end
        role === :scalar && continue
        want, b = if _is_slice_iterator(a)
            (a.args[1] === :eachrow ? (G, K) : (K, G)), a.args[2]
        else
            (role === :vector ? (K,) : (K, K)), a
        end
        sz = _slice_arg_size(plan, b)
        sz === nothing || sz == want || _fail(p.label, "array $(p.name) " *
            "has $G slice(s) of length $K, so its argument $(repr(a)) has " *
            "size $(repr(want)), but $(repr(b)) has size $(repr(sz))")
    end
    return nothing
end

# ── value expressions over arrays ────────────────────────────────────

# True when `ex` constructs an array value or reads an array parameter,
# directly or through assignments.
# For indexed assignments, `opaque` also includes module results: a Julia
# call may produce an array without reading a declared one.
function _mentions_array(ex, plan::StructuralPlan,
        seen::Set{Symbol} = Set{Symbol}(); opaque::Bool = false)
    _is_bound_array_value_call(ex) && return true
    opaque && _contains_module_call(ex) && !_is_bound_value_call(ex) && return true
    if ex isa Symbol
        _is_array_param(plan, ex) && return true
        ex in seen && return false
        push!(seen, ex)
        i = findfirst(a -> a.name === ex, plan.assignments)
        return i !== nothing &&
            _mentions_array(plan.assignments[i].expr, plan, seen; opaque)
    end
    ex isa Expr || return false
    ex.head === :vect && return true
    _level_plate_axis(ex) === nothing || return true
    return any(a -> _mentions_array(ex.head === :tuple ?
        _tuple_field_value(a) : a, plan, seen; opaque), ex.args)
end

# An assignment computed from array parameters or module calls is a
# Julia value too: readable by position, and per observation along an axis
# its expression carries (`b = z * (sd .* L)'` keeps `z`'s `levels(g)`
# rows, so `b[g, 1]` reads each observation's level).
_is_array_assignment(plan::StructuralPlan, name) =
    name isa Symbol && any(a -> a.name === name && _mentions_array(a.expr, plan; opaque=true),
        plan.assignments)

# ── axes of array values ─────────────────────────────────────────────

# The axes of an array-valued expression, in the dim forms declarations
# use (a literal `K`, `levels(g)`, `axes(M, d)`, ...), following Julia:
# broadcasting keeps the operands' axes (a length-1 axis stretches),
# `A * B` takes `A`'s rows and `B`'s columns, an adjoint or `transpose`
# swaps them (a vector becomes a 1×n row), `M[:, j]` keeps the axes read
# with `:`, and a reduction is a scalar. `Any[]` is a scalar; `nothing`
# means the shape is unavailable (for example, an opaque module call).
# With bound data, data_axes resolves declared dimensions to concrete
# lengths as well, without evaluating any sampled value.
function _value_axes(plan::StructuralPlan, ex,
        seen::Set{Symbol} = Set{Symbol}(); data_axes::Bool = false)
    ex isa Number && return Any[]
    _is_bound_value_call(ex) && !_is_bound_array_value_call(ex) && return Any[]
    if ex isa Symbol
        ex in seen && return nothing
        if data_axes && haskey(plan.columns, ex)
            value = plan.columns[ex]
            return value isa AbstractArray ? Any[size(value)...] : Any[]
        end
        if _is_array_param(plan, ex)
            p = _array_param(plan, ex)
            (data_axes && isbound(plan)) || return Any[p.dims...]
            push!(seen, ex)
            r = p.family === :external ?
                Any[_sampling_extent(plan, d, p.name) for d in p.dims] :
                Any[_array_dim_size(plan, p, d; active = seen) for d in p.dims]
            delete!(seen, ex)
            return r
        end
        i = findfirst(v -> v.name === ex, plan.vector_parameters)
        if i !== nothing
            p = plan.vector_parameters[i]
            sz = data_axes ? _resolve_vector_extent(p, plan, plan.columns,
                plan.responses).size : p.size
            data_axes && sz === nothing && p.family === :simplex_dirichlet &&
                (sz = _dirichlet_size(plan, p.args.arg1, p.label))
            return sz === nothing ? nothing : Any[sz]
        end
        if data_axes
            i = findfirst(p -> p.name === ex, plan.plate_parameters)
            if i !== nothing
                push!(seen, ex)
                n = _plate_rows(plan, plan.plate_parameters[i]; active = seen)
                delete!(seen, ex)
                return Any[n]
            end
            i = findfirst(s -> ex in s.states, plan.scans)
            if i !== nothing
                s = plan.scans[i]
                # A trajectory owns its authored bound, not the row count
                # of a response that happens to consume another value.
                (s.hi isa Int || haskey(plan.columns, s.hi)) || return nothing
                return Any[_scan_length(plan, s)]
            end
        end
        any(p -> p.name === ex, plan.parameters) && return Any[]
        definitions = data_axes ? (plan.assignments..., plan.derived...) : plan.assignments
        j = findfirst(a -> a.name === ex, definitions)
        (j === nothing || ex in seen) && return nothing
        push!(seen, ex)
        r = _value_axes(plan, definitions[j].expr, seen; data_axes)
        delete!(seen, ex)
        return r
    end
    ex isa Expr || return nothing
    levelaxis = _level_plate_axis(ex)
    levelaxis === nothing || return Any[:(levels($levelaxis))]
    ax(a) = _value_axes(plan, a, seen; data_axes)
    head = ex.head
    anchor = _matrix_intercept_anchor(ex)
    if anchor !== nothing
        a = ax(anchor)
        a === nothing && return nothing
        all(d -> d isa Int, a) && return Any[prod(a; init = 1)]
        length(a) == 1 && return a
        return nothing
    end
    # A vector literal keeps its outer axis even when its entries are live
    # scalars. Matrix products contract that axis rather than broadcasting
    # the matrix's columns into the observation domain.
    head === :vect && return Any[length(ex.args)]
    head === Symbol("'") && return _adjoint_axes(ax(ex.args[1]))
    if head === :.
        _is_dotted_call(ex) || return nothing
        return _broadcast_axes(Any[ax(a) for a in ex.args[2].args])
    end
    if head === :ref
        base = ax(ex.args[1])
        idx = ex.args[2:end]
        (base === nothing || length(idx) != length(base) ||
            !all(_is_position, idx)) && return nothing
        return Any[d for (i, d) in zip(idx, base) if i === :(:)]
    end
    head === :call && !isempty(ex.args) || return nothing
    fn = ex.args[1]
    args = ex.args[2:end]
    if fn isa GlobalRef && getglobal(fn.mod, fn.name) === Base.hcat
        return _hcat_axes(Any[ax(a) for a in args])
    end
    # Qualified Base calls have the same shape rules as their bare spelling.
    fn isa GlobalRef && fn.mod === Base && (fn = fn.name)
    fn isa Symbol || return nothing
    fn in REDUCTION_FNS && return Any[]
    fn === :transpose && length(args) == 1 &&
        return _adjoint_axes(ax(args[1]))
    fn === :* && return foldl(_matmul_axes, Any[ax(a) for a in args])
    (fn in ELEMENTWISE_OPS || fn in (:+, :-, :/, :^)) &&
        return _broadcast_axes(Any[ax(a) for a in args])
    if fn in ASSIGNMENT_FNS
        # Undotted scalar math (`exp(s)`) on scalars is a scalar.
        all(a -> ax(a) == Any[], args) && return Any[]
    end
    return nothing
end

# Concatenation shares the same shape walk as broadcasts, products and
# declared arrays. It never runs live values or their element functions.
function _hcat_axes(parts::Vector{Any})
    any(isnothing, parts) && return nothing
    isempty(parts) && return Any[0] # Base.hcat() returns an empty vector.
    any(a -> length(a) > 2, parts) && return nothing
    rows = Any[isempty(a) ? 1 : a[1] for a in parts]
    if all(r -> r isa Int, rows) && !all(==(first(rows)), rows)
        throw(DimensionMismatch("hcat operands have different row counts: $(repr(rows))"))
    end
    cols = Any[length(a) < 2 ? 1 : a[2] for a in parts]
    all(c -> c isa Int, cols) || return nothing
    return Any[first(rows), sum(cols)]
end

_adjoint_axes(a) = a === nothing ? nothing :
    isempty(a) ? Any[] :
    length(a) == 1 ? Any[1, a[1]] :
    length(a) == 2 ? Any[a[2], a[1]] : nothing

_is_singleton_axis(d) = d === 1

# Broadcast axes: per position, the operands' non-singleton axis (a
# `levels(g)` axis wins over a literal size, so a level gather stays
# possible; mismatched lengths fail in Julia when the value is computed).
function _broadcast_axes(axs::Vector{Any})
    any(isnothing, axs) && return nothing
    n = maximum(length, axs; init = 0)
    out = Any[]
    for i in 1:n
        cands = Any[a[i] for a in axs
            if length(a) >= i && !_is_singleton_axis(a[i])]
        if isempty(cands)
            push!(out, 1)
        else
            j = findfirst(_is_levels_dim, cands)
            push!(out, cands[j === nothing ? 1 : j])
        end
    end
    return out
end

function _matmul_axes(A, B)
    (A === nothing || B === nothing) && return nothing
    isempty(A) && return B
    isempty(B) && return A
    length(A) == 2 && length(B) == 2 && return Any[A[1], B[2]]
    length(A) == 2 && length(B) == 1 && return Any[A[1]]
    length(A) == 1 && length(B) == 2 && return Any[A[1], B[2]]
    return nothing
end

# The axes a per-observation read `A[g, ...]` gathers along: a declared
# array's own, an array-valued definition's inferred ones (`nothing` when
# `A` is neither, or its axes are not known).
function _gather_axes(plan::StructuralPlan, base)
    base isa Symbol || return nothing
    _is_array_param(plan, base) && return Any[_array_param(plan, base).dims...]
    _is_array_assignment(plan, base) && return _value_axes(plan, base)
    return nothing
end

# Level lookup applies only to a read with one index per known axis.
# A single index on a matrix is linear, and an opaque call's result has
# no declared axis metadata: both use ordinary Julia positions.
function _gather_axis(plan::StructuralPlan, ex::Expr)
    axs = _gather_axes(plan, ex.args[1])
    return axs !== nothing && !isempty(axs) &&
        length(ex.args) - 1 == length(axs) ?
            axs[_gather_index_axis(plan, ex)] : nothing
end

# A per-observation gather over either axis of a declared array or an
# array-valued definition.
_is_gather_ref(plan::StructuralPlan, ex) =
    ex isa Expr && ex.head === :ref && ex.args[1] isa Symbol &&
    (_is_array_param(plan, ex.args[1]) ||
        _is_array_assignment(plan, ex.args[1])) &&
    _array_index_kind(plan, ex) === :gather

# A literal position or a whole axis.
_is_position(i) = (i isa Int && i >= 1) || i === :(:)

# A per-observation index: a column name (data or derived).
_is_row_index(plan::StructuralPlan, i) =
    i isa Symbol && i !== :(:) && !_is_array_param(plan, i) &&
    !(i in _union_names(plan))

# Index forms of `A[...]` over an array parameter `A`:
# `:scalar` (every index a literal Int), `:slice` (literal Ints and `:`),
# `:gather` (one index a data column — one value per observation,
# the rest literal positions or `:`), or `:invalid`.
_gather_index_axis(plan::StructuralPlan, ex::Expr) =
    findfirst(i -> _is_row_index(plan, i), ex.args[2:end])

function _array_index_kind(plan::StructuralPlan, ex::Expr)
    idx = ex.args[2:end]
    isempty(idx) && return :invalid
    all(i -> i isa Int && i >= 1, idx) && return :scalar
    all(_is_position, idx) && return :slice
    count(i -> _is_row_index(plan, i), idx) == 1 &&
        all(i -> _is_position(i) || _is_row_index(plan, i), idx) &&
        return :gather
    return :invalid
end

# Validate one `A[...]` read and record its references. `allow_gather`:
# only per-observation (derived-column) expressions may gather.
function _collect_array_ref!(refs, ex::Expr, plan::StructuralPlan, label,
        bound::Bool; allow_gather::Bool)
    base = ex.args[1]
    if _is_array_assignment(plan, base)
        idx = ex.args[2:end]
        if !isempty(idx) && all(_is_position, idx)
            push!(refs, base)
            return true
        end
        _array_index_kind(plan, ex) === :gather || _fail(label,
            "`$(repr(ex))` reads the array value $base by literal " *
            "positions or `:` (`$base[1]`, `$base[:, 1]`) or, in a " *
            "per-observation expression, by one data column " *
            "(`$base[g]`, `$base[g, 1]`)")
        allow_gather || _fail(label, "`$(repr(ex))` gathers per " *
            "observation — write it in a vector (per-observation) " *
            "definition, not a scalar one")
        axs = _value_axes(plan, base)
        axs === nothing || length(idx) == 1 ||
            length(idx) == length(axs) || _fail(label, "`$(repr(ex))` " *
            "gathers from the $(length(axs))-axis array value $base with " *
            "$(length(idx)) indices. Give one index per axis or one " *
            "linear index")
        g = ex.args[1 + _gather_index_axis(plan, ex)]
        _reads_data_only(plan, g, Set{Symbol}(_all_names(plan)), Set{Symbol}()) ||
            _fail(label, "`$(repr(ex))` gather index $g must be data")
        push!(refs, base)
        if bound
            d = _gather_axis(plan, ex)
            if d === nothing
                # An opaque result can depend on parameter values; its
                # size is checked by Julia indexing at evaluation time.
                K = axs === nothing ? nothing : prod(
                    _array_dim_size(plan, base, label, a) for a in axs)
                _validate_positional_gather(plan, base, label, g, K)
            else
                _validate_gather_axis(plan, base, label, d, g)
            end
        end
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
        axis = _gather_index_axis(plan, ex)
        g = ex.args[1 + axis]
        _reads_data_only(plan, g, Set{Symbol}(_all_names(plan)), Set{Symbol}()) ||
            _fail(label, "`$(repr(ex))` gather index $g must be data")
        bound && _validate_gather_axis(plan, p.name, label, p.dims[axis], g)
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

# A gather at bind, along the indexed axis `d` of the array value
# `name`: a `levels(h)` axis needs every value of `g` on that axis; an
# integer axis (`1:K`, `axes(M, d)`) needs integer `g` in range — plain
# Julia indexing.
function _validate_gather_axis(plan::StructuralPlan, name::Symbol, label,
        d, g::Symbol)
    _is_levels_dim(d) || return _validate_positional_gather(plan, name,
        label, g, _array_dim_size(plan, name, label, d))
    col = _gather_index_column(plan, name, label, g)
    return _validate_level_gather(plan, name, label, d, g, col)
end

function _validate_level_gather(plan::StructuralPlan, name::Symbol, label,
        d, g::Symbol, col::AbstractVector)
    lv = _array_axis_levels(plan, name, label, d)
    codes = _declared_codes(col, lv)
    any(==(0), codes) && _fail(label, "`$name[$g]` looks values of " *
        "$g up on the axis `levels($(d.args[2]))` of $name, but " *
        "$g holds values not on that axis")
    return nothing
end

function _gather_index_column(plan::StructuralPlan, name::Symbol, label,
        g::Symbol)
    haskey(plan.columns, g) || _fail(label, "array $name is read by " *
        "`$name[$g]`, but $g is not a bound column")
    return _vector_column(plan.columns, g, label, "gather index")
end

function _validate_positional_gather(plan::StructuralPlan, name::Symbol,
        label, g::Symbol, K::Union{Int,Nothing})
    col = _gather_index_column(plan, name, label, g)
    eltype(col) <: Integer && eltype(col) !== Bool || _fail(label,
        "`$name[$g]` indexes $name by position, but $g does not hold " *
        "integers — declare an axis `levels($g)` to look values up by level")
    all(i -> i >= 1 && (K === nothing || i <= K), col) || _fail(label,
        "`$name[$g]`: $g holds indices outside " *
        (K === nothing ? "the positive integers" : "1:$K"))
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
        # Whole-value data have no observation axis. An array definition
        # retained by lowering reads these as bound operands, just like a
        # scalar assignment or an undotted module call.
        bound && haskey(plan.columns, ex) &&
            ex in first(_bound_model_level_inputs(plan)) && return nothing
        _is_derived(plan, ex) && _fail(label, "derived column $ex is " *
            "per-observation — it does not combine with arrays here")
        bound && haskey(plan.columns, ex) && _fail(label, "column $ex is " *
            "per-observation — it does not combine with arrays here")
        return push!(refs, ex)
    end
    ex isa Expr || _fail(label, "unsupported literal $(repr(ex))")
    _is_plate_column_expr(ex) &&
        return _collect_plate_column_refs!(refs, ex, plan, label, bound)
    head = ex.head
    if head === :tuple || head === :vect
        for a in ex.args
            head === :tuple && (a = _tuple_field_value(a))
            _collect_array_value_refs!(refs, a, plan, label, bound)
        end
        return nothing
    end
    if head === :ref
        return _collect_model_value_ref!(refs, ex, plan, label, bound)
    end
    if head === Symbol("'")
        for a in ex.args
            _collect_array_value_refs!(refs, a, plan, label, bound)
        end
        return nothing
    end
    if head === :call
        fn = ex.args[1]
        # A module call takes and returns whole values, just as it does
        # inside scalar and observation expressions (functions as values).
        fn isa GlobalRef &&
            return _collect_opaque_refs!(refs, ex, plan, label, bound)
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
        length(ex.args) == 2 && ex.args[1] isa Union{Symbol,GlobalRef} &&
            ex.args[2] isa Expr && ex.args[2].head === :tuple || _fail(label,
            "field access does not lower in array expressions")
        (ex.args[1] isa GlobalRef || ex.args[1] in ELEMENTWISE_FNS) ||
            _fail(label, "`$(ex.args[1]).` " *
            "is not in the elementwise vocabulary")
        for a in ex.args[2].args
            _collect_array_value_refs!(refs, a, plan, label, bound)
        end
        return nothing
    end
    _fail(label, "unsupported expression head $head in an array expression")
end

# `B * w`: a per-observation matrix — a bound data matrix `B` (supplied, or
# a data-only module value the model computes at bind), or the rows of an
# array gathered per observation (`z[g, :]`) — times an array-valued
# expression: one value or row per observation (standard Julia
# matrix-vector or matrix-matrix product).
function _is_data_matvec(ex, plan::StructuralPlan)
    ex isa Expr && ex.head === :call && length(ex.args) == 3 &&
        ex.args[1] === :* || return false
    B, w = ex.args[2], ex.args[3]
    rows = (B isa Symbol && !_is_array_param(plan, B) &&
            !(B in _union_names(plan)) &&
            (!_is_derived(plan, B) || _is_bind_data_derived(plan, B))) ||
        _is_row_gather(plan, B)
    return rows && _mentions_array(w, plan)
end

# A derived column the model computes at bind from data alone (a module
# call reading only raw columns or other such columns): bound data, so
# its value may be a matrix with one row per observation.
_is_bind_data_derived(plan::StructuralPlan, name::Symbol) =
    any(d -> d.name === name, plan.derived) &&
        name in _module_data_names(plan; unbound=true)

# `z[g, :]` or `z[:, g]'`: one row per observation. Adjoint preserves
# the author's Julia orientation, rather than changing the gather itself.
function _row_gather_ref(plan::StructuralPlan, ex)
    ex isa Expr || return nothing
    if ex.head === :ref && length(ex.args) == 3 &&
            ex.args[3] === :(:) && _is_gather_ref(plan, ex)
        return ex
    elseif ex.head === Symbol("'") && length(ex.args) == 1
        col = ex.args[1]
        col isa Expr && col.head === :ref && length(col.args) == 3 &&
            col.args[2] === :(:) && _is_gather_ref(plan, col) && return col
    end
    return nothing
end

_is_row_gather(plan::StructuralPlan, ex) =
    _row_gather_ref(plan, ex) !== nothing

function _collect_data_matvec!(refs, ex::Expr, plan::StructuralPlan, label,
        bound::Bool)
    B, w = ex.args[2], ex.args[3]
    _collect_array_value_refs!(refs, w, plan, label, bound)
    if B isa Expr
        _collect_array_ref!(refs, _row_gather_ref(plan, B), plan, label,
            bound; allow_gather = true)
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
        length(dims) in (1, 2) && dims[1] == size(col, 2) || _fail(label,
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
        p.name in plan.conditioned && continue
        dims = _array_dims(plan, p)
        if p.family === :external
            entry = _external_layout_entry(plan, p, offset, dims)
            push!(entries, entry)
            offset += entry.size
        elseif p.family === :lkj_cholesky
            K = dims[1]
            packed = K * (K - 1) ÷ 2
            labels = [Symbol(p.name, ".", i) for i in 1:packed]
            transform = _lkj_uplo(p) === 'U' ? :lkj_upper : :lkj
            push!(entries, LayoutEntry(:cholesky_corr, nothing, p.name,
                labels, offset, packed, transform, NaN, NaN, dims))
            offset += packed
        elseif p.family === :lkj_cholesky_stack
            # Level k's K(K-1)/2 vine partials are contiguous (`L.p.k`,
            # P × S column-major): partial p of every level is a strided
            # view, so one broadcast vine serves all S levels.
            K, S = dims[1], dims[3]
            P = K * (K - 1) ÷ 2
            labels = [Symbol(p.name, ".", i, ".", k) for k in 1:S for i in 1:P]
            transform = _lkj_uplo(p) === 'U' ? :lkj_stack_upper : :lkj_stack
            push!(entries, LayoutEntry(:array, nothing, p.name, labels,
                offset, P * S, transform, NaN, NaN, dims))
            offset += P * S
        elseif _is_slice_array(p)
            # Multivariate normal slices are centered: the array's entries
            # are its coordinates. Simplex / ordered slices pack their
            # unconstrained coordinates (K − 1 / K per slice) in the
            # array's orientation.
            transform = _slice_transform(Val(_slice_stem(p)), _slice_kind(p))
            pd = _slice_packed_dims(transform, dims)
            n = prod(pd)
            push!(entries, LayoutEntry(:array, nothing, p.name,
                _array_labels(p.name, pd), offset, n, transform, NaN, NaN,
                dims))
            offset += n
        else
            transform, lo, hi =
                _entry_transform(p.family, _bound_override(plan, p.support_override),
                    p.family === :uniform ? map(x -> _layout_bound(plan, x), p.args) : p.args)
            n = prod(dims)
            push!(entries, LayoutEntry(:array, nothing, p.name,
                _array_labels(p.name, dims), offset, n, transform, lo, hi,
                dims))
            offset += n
        end
    end
    return offset
end

# Layout transform of a slice array: centered multivariate normals are
# the identity; simplex / ordered slices transform along each slice.
_slice_transform(::Val{:mvnormal_cholesky}, kind) = :identity
_slice_transform(::Val{:mvnormal}, kind) = :identity
_slice_transform(::Val{:dirichlet}, kind) = Symbol(:simplex_, kind)
_slice_transform(::Val{:ordered_normal}, kind) = Symbol(:ordered_, kind)

const _SLICE_TRANSFORMS = (:simplex_rows, :simplex_cols, :ordered_rows,
    :ordered_cols)
_is_slice_transform(t::Symbol) = t in _SLICE_TRANSFORMS

# The orientation a slice transform runs along (an `mv_slices.jl` value,
# spliced as a constructor call in the graph).
_slice_transform_orientation(t::Symbol) =
    t in (:simplex_rows, :ordered_rows) ? _SliceRows() : _SliceCols()
_orientation_expr(::_SliceRows) = :(_SliceRows())
_orientation_expr(::_SliceCols) = :(_SliceCols())
_orientation_expr(::_SliceWhole) = :(_SliceWhole())

# Packed (unconstrained) axes of a slice transform over the constrained
# axes `dims`: a simplex slice of length K packs K − 1 coordinates.
_slice_packed_dims(t::Symbol, dims) =
    t === :simplex_rows ? [dims[1], dims[2] - 1] :
    t === :simplex_cols ? [dims[1] - 1, dims[2]] : collect(dims)

_slice_constrain(t::Symbol, o, U) = startswith(String(t), "simplex") ?
    _simplex_slices_constrain(o, U) : _ordered_slices_constrain(o, U)
_slice_unconstrain(t::Symbol, o, X) = startswith(String(t), "simplex") ?
    _simplex_slices_unconstrain(o, X) : _ordered_slices_unconstrain(o, X)
_slice_logjac(t::Symbol, o, U) = startswith(String(t), "simplex") ?
    _simplex_slices_logjac(o, U) : _ordered_slices_logjac(o, U)
_slice_function_names(t::Symbol) = startswith(String(t), "simplex") ?
    (:_simplex_slices_constrain, :_simplex_slices_logjac) :
    (:_ordered_slices_constrain, :_ordered_slices_logjac)

# Host edges of a slice-transformed array entry (the same functions the
# graph calls, so host and graph agree bit-for-bit).
function _array_slices_constrain(e::LayoutEntry, seg)
    U = reshape(Vector{_host_eltype(seg)}(seg), _slice_packed_dims(e.transform, e.dims)...)
    return _slice_constrain(e.transform,
        _slice_transform_orientation(e.transform), U)
end
_array_slices_unconstrain(e::LayoutEntry, X) =
    vec(_slice_unconstrain(e.transform,
        _slice_transform_orientation(e.transform), Matrix{_host_eltype(X)}(X)))
function _array_slices_logjac(e::LayoutEntry, seg)
    U = reshape(Vector{_host_eltype(seg)}(seg), _slice_packed_dims(e.transform, e.dims)...)
    return _slice_logjac(e.transform,
        _slice_transform_orientation(e.transform), U)
end

_array_slices_packed_name(name::Symbol) = Symbol(:_ppl_arru_, name)

# In-graph constrain edges of an elementwise array: the plate edges over
# its flat packed block, then (two axes) the column-major reshape. A
# slice-transformed array reshapes its packed block and transforms every
# slice in one call.
function _array_transform_statements(e::LayoutEntry)
    # There are no bijector cells in an empty elementwise array. Preserve its
    # value shape with an owned empty output, without evaluating a cell.
    e.size == 0 && !_is_slice_transform(e.transform) &&
        !_is_lkj_stack(e.transform) && return Expr[:($(e.name)::Array{Float64,$(length(e.dims))} =
        zeros(Float64, $(e.dims...)))]
    _is_lkj_stack(e.transform) && return _lkj_stack_transform_statements(e)
    if _is_slice_transform(e.transform)
        U = _array_slices_packed_name(e.name)
        lo, hi = e.offset, e.offset + e.size - 1
        cfn, _ = _slice_function_names(e.transform)
        o = _orientation_expr(_slice_transform_orientation(e.transform))
        return Expr[
            :($U::Matrix{Float64} = reshape(Float64.(view(unconstrained,
                $lo:$hi)), $(_slice_packed_dims(e.transform, e.dims)...))),
            :($(e.name)::Matrix{Float64} = $cfn($o, $U)),
        ]
    end
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
    e.size == 0 && return nothing
    _is_lkj_stack(e.transform) &&
        return _lkj_vine_logjac(e.name, e.dims[1]; stacked = true)
    if _is_slice_transform(e.transform)
        _, jfn = _slice_function_names(e.transform)
        o = _orientation_expr(_slice_transform_orientation(e.transform))
        return :($jfn($o, $(_array_slices_packed_name(e.name))))
    end
    flat = _array_flat_name(e.name, length(e.dims))
    return jacobian_term(LayoutEntry(:plate, nothing, flat, e.labels,
        e.offset, e.size, e.transform, e.lo, e.hi))
end

# A per-level LKJ stack (`:lkj_stack`, dims `[K, K, S]`): the stacked
# vine over partial p of every level (a strided view of its packed block,
# core constraint 1: S never multiplies statements), then the value
# `L::Array{Float64,3}` (`L[:, :, k]` is level k's factor) assembled from
# the S-vector entries, column-major over (i, j).
function _lkj_stack_transform_statements(e::LayoutEntry)
    K, S = e.dims[1], e.dims[3]
    P = K * (K - 1) ÷ 2
    read(p) = (lo = e.offset + p - 1;
        :(view(unconstrained, $lo:$P:$(lo + (S - 1) * P))))
    stmts = _lkj_vine_statements(e.name, K, read; stacked = true)
    cols = Any[]
    for j in 1:K, i in 1:K
        row, col = _is_upper_lkj(e.transform) ? (j, i) : (i, j)
        push!(cols, row == 1 && col == 1 ? :(fill(1.0, $S)) :
            row >= col ? _rl_name(e.name, row, col) : :(zeros($S)))
    end
    push!(stmts, :($(e.name)::Array{Float64,3} =
        reshape(permutedims(hcat($(cols...))), $K, $K, $S)))
    return stmts
end

# Host edges of a per-level LKJ stack: level by level, the single-factor
# host functions (the same vine as the stacked graph edges).
function _lkj_stack_constrain(e::LayoutEntry, seg)
    K, S = e.dims[1], e.dims[3]
    P = K * (K - 1) ÷ 2
    T = _host_eltype(seg)
    U = reshape(Vector{T}(seg), P, S)
    out = Array{T,3}(undef, K, K, S)
    for k in 1:S
        L = lkj_chol_constrain(U[:, k], K)
        out[:, :, k] = _is_upper_lkj(e.transform) ? permutedims(L) : L
    end
    return out
end
function _lkj_stack_unconstrain(e::LayoutEntry, X)
    K, S = e.dims[1], e.dims[3]
    return reduce(vcat, [lkj_chol_unconstrain(
        _is_upper_lkj(e.transform) ? permutedims(X[:, :, k]) : X[:, :, k], K)
        for k in 1:S])
end
function _lkj_stack_logjac(e::LayoutEntry, seg)
    K, S = e.dims[1], e.dims[3]
    P = K * (K - 1) ÷ 2
    U = reshape(Vector{_host_eltype(seg)}(seg), P, S)
    return sum(lkj_chol_logjac(U[:, k], K) for k in 1:S)
end

# ── generator ────────────────────────────────────────────────────────

# One triangular transform graph for every declared factor. Preparation fixes
# K and the packed slice; neither a literal nor a data-derived K replicates
# the body. The partials follow Stan's column-block order. Column block j
# carries the running product of √(1 - z²) over its j - 1 partials, starting
# at 1: entry i is its partial times the product so far, and the diagonal is
# the final product. This is the left-associated order of the host
# `lkj_chol_constrain`, so the two agree bit for bit. Each pass is one loop
# over the K(K-1)/2 packed partials, carrying the entry's position (i, j);
# the last partial of a column moves it to the next column. No iteration is
# spent on a masked-out entry. The trip count is fixed at preparation, so a
# compiled reverse pass keeps a statically sized tape, which a column loop
# with an inner `1:(j - 1)` range does not.
_lkj_array_partials(L::Symbol) = Symbol(:_ppl_lkj_partials_, L)
_lkj_array_logjac(L::Symbol) = Symbol(:_ppl_lkj_logjac_, L)
_lkj_array_diagonal(L::Symbol) = Symbol(:_ppl_lkj_diagonal_, L)

function _lkj_array_transform_statements(e::LayoutEntry)
    L, K = e.name, e.dims[1]
    P = K * (K - 1) ÷ 2
    row, col = _is_upper_lkj(e.transform) ? (:i, :j) : (:j, :i)
    z, lj = _lkj_array_partials(L), _lkj_array_logjac(L)
    diagonal = _lkj_array_diagonal(L)
    return Expr[
        :($z::Vector{Float64} = tanh.($(block_read(e.offset, e.size)))),
        # The diagonal is part of constructing the factor. Publish it as a
        # graph value so a guarded prior need not share the whole matrix with
        # response products. Each diagonal entry is still computed once.
        :($diagonal::Vector{Float64} = let
            out = ones(Float64, $K)
            i, j, d = 1, 2, 1.0
            for p in 1:$P
                d = d * sqrt(1 - $z[p]^2)
                if i + 1 < j
                    i = i + 1
                else
                    out[j] = d
                    i, j, d = 1, j + 1, 1.0
                end
            end
            out
        end),
        :($L::Matrix{Float64} = let
            out = zeros(Float64, $K, $K)
            out[1, 1] = 1.0
            i, j, w = 1, 2, 1.0
            for p in 1:$P
                out[$row, $col] = $z[p] * w
                w = w * sqrt(1 - $z[p]^2)
                if i + 1 < j
                    i = i + 1
                else
                    out[j, j] = $diagonal[j]
                    i, j, w = 1, j + 1, 1.0
                end
            end
            out
        end),
        :($lj::Float64 = let
            total = 0.0
            i, j = 1, 2
            for p in 1:$P
                total = total + ((j - i + 1) / 2) * log(1 - $z[p]^2)
                i, j = i + 1 < j ? (i + 1, j) : (1, j + 1)
            end
            total
        end),
    ]
end

# The normalization constant is preparation-only metadata. The diagonal sum
# remains a retained recipe loop even for a conditioned, caller-owned factor.
function _lkj_array_prior_terms(L::Symbol, K::Int, eta::Float64;
        diagonal::Union{Nothing,Symbol} = nothing)
    c = lkj_logconst(K, eta)
    entry = diagonal === nothing ? :($L[i, i]) : :($diagonal[i])
    term = _lkj_prior_diagonal(:(log($entry)), K, :i, eta)
    return :(let
        total = $c
        for i in 2:$K
            total += $term
        end
        total
    end)
end

# Normalization depends only on eta, so keep it separate from the factor's
# diagonal. Preparation evaluates it once when eta is bound or constant;
# live shapes retain their own loop and ordinary generated derivatives.
function _lkj_array_normalizer_terms(K::Int, eta::Symbol)
    lgamma = GlobalRef(DistributionKernelSources, :loggamma)
    return :(if isfinite($eta) && $eta > 0
        let
            total = $(K - 1) * $lgamma($eta + $(0.5 * (K - 1)))
            for k in 1:$(K - 1)
                total = total - 0.5 * k * $(log(pi)) -
                    $lgamma($eta + 0.5 * ($(K - 1) - k))
            end
            total
        end
    else
        -Inf
    end)
end

function _lkj_array_diagonal_terms(L::Symbol, K::Int, eta::Symbol;
        intrinsic_pair::Bool = false, diagonal::Union{Nothing,Symbol} = nothing)
    entry = diagonal === nothing ? :($L[i, i]) : :($diagonal[i])
    # A declared literal 2 × 2 factor has one diagonal contribution. Its
    # scalar equation avoids a second retained loop reading the same factor
    # that downstream matrix products consume. A bound shape, even K = 2,
    # still uses the retained diagonal loop below.
    if intrinsic_pair
        pair_entry = diagonal === nothing ? :($L[2, 2]) : :($diagonal[2])
        return :(if isfinite($eta) && $eta > 0
            (2 * $eta - 2) * log($pair_entry)
        else
            0.0
        end)
    end
    return :(if isfinite($eta) && $eta > 0
        let
            total = 0.0
            for i in 2:$K
                total += ($K - i + 2 * $eta - 2) * log($entry)
            end
            total
        end
    else
        0.0
    end)
end

# Name of the level-code vector for gathers by column `g` on the
# `levels(h)` axis.
_array_level_index_name(g::Symbol, h::Symbol, axis::Int) =
    Symbol(:_ppl_lvx_, h, :_, g, :_axis_, axis)

# Rewrite every level gather `z[g]` (a `levels(h)` axis) to an integer
# gather over the level codes; integer axes stay plain Julia indexing.
# Records the (index column, array value, axis) code vectors it needs. The value
# identifies its declared axes, including a selected subset of levels.
function _array_gather_rewrite(ex, plan::StructuralPlan,
        needed::Set{Tuple{Symbol,Symbol,Int}})
    ex isa Expr || return ex
    # An array-cell plate column: its inputs carry the level codes; the
    # cell body indexes by those codes and is RK's to plan.
    _is_plate_column_expr(ex) && return Expr(:do,
        _array_gather_rewrite(ex.args[1], plan, needed), ex.args[2])
    if ex.head === :call && length(ex.args) == 4 &&
            ex.args[1] === :_ppl_level_gather
        name, g, ld = ex.args[2:end]
        axs = _gather_axes(plan, name)
        d = axs[ld]
        lv = _array_axis_levels(plan, name, :plan, d)
        lv = _apply_subset(lv, LevelMap(name, d.args[2], [], :levels,
            _levels_subset(d)))
        # These codes depend only on bound labels, not the number of cells
        # in the graph. Omitted selected levels use the usual zero row.
        codes = _declared_codes(_array_axis_levels(plan, name, :plan, g), lv)
        source = name
        if _levels_subset(d) !== Colon()
            source = length(axs) == 1 ? :(vcat(0.0, $name)) :
                ld == 1 ? :(vcat(zeros(1, size($name, 2)), $name)) :
                :(cat(zeros(size($name, 1), size($name, 2), 1), $name; dims = 3))
            codes = codes .+ 1
        end
        ix = Expr(:vect, codes...)
        length(axs) == 1 && return Expr(:ref, source, ix)
        if ld == 1
            aligned = Expr(:ref, source, ix, :(:))
            return :(eachcol(permutedims($aligned)))
        end
        aligned = Expr(:ref, source, :(:), :(:), ix)
        return :(eachcol(reshape($aligned, size($name, 1) * size($name, 2), :)))
    end
    if ex.head === :call && !isempty(ex.args) &&
            ex.args[1] in (:_ppl_level_indices, :_ppl_level_values)
        g = ex.args[2]
        lv = _array_axis_levels(plan, g, :plan, g)
        if ex.args[1] === :_ppl_level_indices
            return :(collect(1:$(length(lv))))
        elseif ex.args[1] === :_ppl_level_values
            return Expr(:vect, map(_level_literal, lv)...)
        end
    end
    if ex.head === :call && length(ex.args) == 3 && ex.args[1] === :_ppl_codes
        g, h = ex.args[2], ex.args[3]
        push!(needed, (g, h, 0))
        return _array_level_index_name(g, h, 0)
    end
    if ex.head === :call && length(ex.args) == 4 && ex.args[1] === :_ppl_axis_codes
        g, name, axis = ex.args[2:end]
        push!(needed, (g, name, axis))
        return _array_level_index_name(g, name, axis)
    end
    if _is_gather_ref(plan, ex)
        d = _gather_axis(plan, ex)
        if _is_levels_dim(d)
            axis = _gather_index_axis(plan, ex)
            g, name = ex.args[1 + axis], ex.args[1]
            push!(needed, (g, name, axis))
            codes = _array_level_index_name(g, name, axis)
            idx = copy(ex.args[2:end])
            idx[axis] = codes
            if _levels_subset(d) !== Colon()
                zero = length(_gather_axes(plan, name)) == 1 ? 0.0 :
                    axis == 1 ? :(zeros(1, size($name, 2))) :
                        :(zeros(size($name, 1), 1))
                cat = axis == 1 ? :vcat : :hcat
                idx[axis] = Expr(:call, :.+, codes, 1)
                return Expr(:ref, Expr(:call, cat, zero, name), idx...)
            end
            return Expr(:ref, name, idx...)
        end
        return ex
    end
    return Expr(ex.head,
        (_array_gather_rewrite(a, plan, needed) for a in ex.args)...)
end

# The level-code vectors a plan's gathers read (data-only: `bound=` folds
# them), keyed by an array value or a plate cell's levels column.
function _array_level_index_statements(plan::StructuralPlan,
        needed::Set{Tuple{Symbol,Symbol,Int}})
    stmts = Expr[]
    for (g, name, axis) in sort!(collect(needed))
        axes = _gather_axes(plan, name)
        d = axis == 0 ? Expr(:call, :levels, name) : axes[axis]
        h = d.args[2]
        lv = _array_axis_levels(plan, name, :plan, d)
        lv = _apply_subset(lv, LevelMap(name, h, [], :levels,
            _levels_subset(d)))
        lvlvec = Expr(:vect, (_level_literal(l) for l in lv)...)
        push!(stmts, :($(_array_level_index_name(g, name, axis)) =
            _declared_codes($g, $lvlvec)))
    end
    return stmts
end

# Prior nodes `_ppl_prior_<name>` of the plan's array parameters.
function _array_prior_stmts!(stmts::Vector{Expr}, terms::Vector{Any},
        plan::StructuralPlan, needed::Set{Tuple{Symbol,Symbol,Int}};
        context = plan, pointwise = Pair{Symbol,Any}[])
    for p in plan.array_parameters
        dims = _array_dims(plan, p)
        node = Symbol(:_ppl_prior_, p.name)
        if p.family === :external
            append!(stmts, _external_density_statements(p))
            push!(terms, node)
            push!(pointwise, p.name => Symbol(p.args.broadcast ? :_ppl_pw_prior_ : :_ppl_prior_, p.name))
            continue
        end
        if prod(dims) == 0
            # An elementwise declaration over no elements has no density
            # cells. Scalar priors elsewhere in the model remain intact.
            push!(stmts, :($node::Float64 = 0.0))
            push!(terms, node)
            value = if _is_slice_array(p)
                kind = _slice_kind(p)
                kind === :vector ? node :
                    :(zeros($(kind === :rows ? dims[1] : dims[2])))
            elseif p.family === :lkj_cholesky_stack
                :(zeros($(dims[3])))
            else
                :(zeros($(dims...)))
            end
            push!(pointwise, p.name => value)
            continue
        end
        if p.family === :lkj_cholesky
            factor_diagonal = p.name in plan.conditioned ? nothing :
                _lkj_array_diagonal(p.name)
            eta = p.args.arg1
            if eta isa Symbol
                normalizer, diagonal = Symbol(node, :_normalizer), Symbol(node, :_diagonal)
                push!(stmts, :($normalizer::Float64 = $(_lkj_array_normalizer_terms(dims[1], eta))),
                    :($diagonal::Float64 = $(_lkj_array_diagonal_terms(p.name, dims[1], eta;
                        intrinsic_pair = p.dims == Any[2, 2], diagonal = factor_diagonal))),
                    :($node::Float64 = $normalizer + $diagonal))
            else
                push!(stmts, :($node::Float64 =
                    $(_lkj_array_prior_terms(p.name, dims[1], eta;
                        diagonal = factor_diagonal))))
            end
            push!(terms, node)
            push!(pointwise, p.name => node)
            continue
        end
        if p.family === :lkj_cholesky_stack
            push!(stmts, :($node::Float64 = $(_lkj_prior_terms(p.name,
                dims[1], p.args.arg1; nstack = dims[3]))))
            push!(terms, node)
            # A stack is a broadcast of factor draws, one density per slice.
            if p.name in plan.conditioned
                value = Symbol(:_ppl_pointwise_, p.name)
                push!(stmts, :($value = $(_lkj_prior_terms(p.name,
                    dims[1], Float64(p.args.arg1); nstack = dims[3], pointwise = true))))
                push!(pointwise, p.name => value)
            end
            continue
        end
        if p.family === :flat
            push!(stmts, :($node::Float64 = 0.0))
            push!(terms, node)
            push!(pointwise, p.name => :(0.0 .* $(p.name)))
            continue
        end
        if _is_slice_array(p)
            density = _slice_prior_call!(stmts, p)
            push!(stmts, :($node::Float64 =
                $density))
            push!(terms, node)
            if _slice_kind(p) === :vector
                push!(pointwise, p.name => node)
            elseif p.name in plan.conditioned
                value = Symbol(:_ppl_pointwise_, p.name)
                density = copy(density)
                density.args[1] = Symbol(replace(string(density.args[1]), "_logpdf" => "_pointwise"))
                push!(stmts, :($value = $density))
                push!(pointwise, p.name => value)
            end
            continue
        end
        flat = _array_flat_name(p.name, length(dims))
        arg_names = Symbol[]
        arg_values = Any[]
        for (i, (k, a)) in enumerate(pairs(p.args))
            push!(arg_names, k)
            if a isa Real || a isa Symbol
                push!(arg_values, a)
                continue
            end
            local_name = Symbol(:_ppl_parg_, p.name, :_, i)
            val = a.head === :vect ? :(Float64[$(a.args...)]) :
                _array_gather_rewrite(a, context, needed)
            push!(stmts, :($local_name = $val))
            push!(arg_values, local_name)
        end
        _vector_prior_stmts!(stmts, terms, flat, p.family,
            NamedTuple{Tuple(arg_names)}(Tuple(arg_values)),
            p.support_override; conditioned = p.name in context.conditioned)
        push!(pointwise, p.name => :(reshape($(Symbol(:_ppl_pw_prior_, flat)), size($(p.name)))))
    end
    return nothing
end

# The slice density of a slice array (`mv_slices.jl`), its arguments bound
# to hygienic locals: a shared value as itself (literal vectors and
# `zeros(K)` / `fill(a, K)` as `Float64` vectors), a per-slice
# `eachrow(M)` / `eachcol(M)` as a `_PerSlice` of `M`. The literal
# `Ordered` length is structural and not passed.
_slice_density(::Val{:mvnormal_cholesky}) = :_mvnormal_cholesky_slices_logpdf
_slice_density(::Val{:mvnormal}) = :_mvnormal_slices_logpdf
_slice_density(::Val{:dirichlet}) = :_dirichlet_slices_logpdf
_slice_density(::Val{:ordered_normal}) = :_ordered_normal_slices_logpdf

function _slice_prior_call!(stmts::Vector{Expr}, p::ArrayParameter)
    roles = _slice_roles(p)
    vals = Any[]
    for (i, (k, a)) in enumerate(pairs(p.args))
        roles[k] === :count && continue
        local_name = Symbol(:_ppl_parg_, p.name, :_, i)
        if _is_slice_iterator(a)
            o = _orientation_expr(a.args[1] === :eachrow ? _SliceRows() :
                _SliceCols())
            push!(stmts, :($local_name = _PerSlice($o,
                $(_slice_value_expr(a.args[2])))))
            push!(vals, local_name)
        elseif a isa Symbol
            push!(vals, a)
        elseif a isa Real
            push!(vals, Float64(a))
        else
            push!(stmts, :($local_name = $(_slice_value_expr(a))))
            push!(vals, local_name)
        end
    end
    o = _orientation_expr(_slice_orientation(_slice_kind(p)))
    return :($(_slice_density(Val(_slice_stem(p))))($o, $(p.name), $(vals...)))
end

_slice_value_expr(a) = a isa Expr && a.head === :vect ? :(Float64[$(a.args...)]) :
    _is_zeros_call(a) ? :(zeros(Float64, $(a.args[2]))) :
    _is_fill_call(a) ? :(fill($(a.args[2] isa Real ? Float64(a.args[2]) :
        a.args[2]), $(a.args[3]))) : a
