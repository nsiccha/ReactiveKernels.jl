# Generic data design and offset recipes for declared parameters and values.
using LinearAlgebra: Diagonal, Symmetric, eigen, norm, nullspace, rank

"""Design-matrix recipe name for a predictor (`mu` → `_ppl_design_mu`)."""
design_name(predictor::Symbol) = Symbol(:_ppl_design_, predictor)

"""Offset-vector recipe name for a predictor (`mu` → `_ppl_offset_mu`)."""
offset_name(predictor::Symbol) = Symbol(:_ppl_offset_, predictor)

"""
    _observation_rows(values...) -> Int

Rows of the broadcast of `values` along their first axis (a number has one
row). Generated programs read observation row counts through this call over
bound data, so a built graph takes them from the binding it is prepared with.
"""
_observation_rows(values...) =
    length(only(Base.Broadcast.broadcast_shape(map(v -> (axes(v, 1),), values)...)))

# First-axis lengths broadcast when each is one or a common length.
function _observation_rows_compatible(values)
    lengths = unique!([size(v, 1) for v in values])
    return length(filter(!=(1), lengths)) <= 1
end

"""
    design_recipe(shape, n_rows; plates) -> Union{Nothing,Expr}

`_ppl_design_<pred> = Float64.(hcat(<blocks…>))`, or `nothing` for a
width-0 (offset-only) predictor or a predictor with no static (data-only) blocks. `n_rows` is the design row count
(obs-level: `n_obs`; subject-level: `n_sub`), as a number or as an expression
over bound data (`_predictor_rows_source`). Blocks: intercept → `ones(n)`, continuous
→ the bare column, factor → full-rank dummies over mapped levels,
matrix → one part per element (`ones(n)` at intercept positions, the
bare column otherwise — matrices hold data/derived columns only, so no
latent wrap).
`plates` names the per-cell latent vectors (a `ContinuousTerm` may scale
one — the SB `me` mirror): a latent part wraps as `Float64.(col)`, which
materializes the layout's packed-slice `view` to a dense vector so the
`hcat` stays homogeneous — a mixed `Vector`/`SubArray` `hcat` lowers
through a Union-typed path Enzyme cannot differentiate.
"""
function design_recipe(shape::DesignShape, n_rows;
        plates::AbstractSet{Symbol} = Set{Symbol}())
    shape.width == 0 && return nothing
    # Continuous blocks contribute their bare column (a bound port), not a
    # wrapped Expr — except latent columns, which materialize (above).
    parts = Any[]
    for b in shape.blocks
        if b.kind === InterceptTerm
            push!(parts, :(ones($n_rows)))
        elseif b.kind === ContinuousTerm
            push!(parts, b.column in plates ? :(Float64.($(b.column))) : b.column)
        elseif b.kind === FactorTerm
            push!(parts, _contrast_expr(b))
        elseif b.kind === MatrixTerm
            for e in b.elements
                push!(parts, e === nothing ? :(ones($n_rows)) : e)
            end
        end
    end
    isempty(parts) && return nothing
    matrix = Expr(:call, :hcat, parts...)
    name = design_name(shape.predictor)
    return :($name = Float64.($matrix))
end

"""
    offset_recipe(shape; scalars, n_rows, broadcast) -> Union{Nothing,Expr}

`_ppl_offset_<pred> = col1 .+ col2 .+ …` over offset-term columns, or
`nothing` when the predictor has no offset terms. A sum with a scalar
assignment from `scalars` gets `n_rows` rows (`_ppl_rows`; a number or an
expression over bound data) unless `broadcast` retains its scalar shape.
"""
function offset_recipe(shape::DesignShape; scalars = Set{Symbol}(), n_rows = 0,
        broadcast = false)
    cols = Any[b.column for b in shape.blocks if b.kind === OffsetTerm]
    isempty(cols) && return nothing
    total = foldl((a, c) -> :($a .+ $c), cols)
    !broadcast && any(in(scalars), cols) &&
        (total = Expr(:call, GlobalRef(@__MODULE__, :_ppl_rows), total, n_rows))
    name = offset_name(shape.predictor)
    return :($name = $total)
end

"""
    preprocessing_recipes(plan) -> Vector{Expr}

Data design and offset recipes in `plan.predictors` order. A predictor
with broadcast affine operands uses per-block expressions instead of a
fused design matrix; absent design and offset blocks emit no recipe.
"""
function preprocessing_recipes(plan::StructuralPlan)
    stmts = Expr[]
    plates = Set{Symbol}(p.name for p in plan.plate_parameters)
    for pred in plan.predictors
        shape = design_shape(pred, plan.columns; levelmaps = plan.levelmaps,
            matrices = plan.matrices)
        if !_broadcast_affine(plan, pred)
            rows = _predictor_rows_source(plan, pred.name)
            recipe = design_recipe(shape, rows; plates)
            recipe !== nothing && push!(stmts, recipe)
        end
        scalars = Set{Symbol}(only(t.columns) for t in pred.terms
            if _is_scalar_offset(t, plan))
        rows = isempty(scalars) ? 0 : _located_rows_source(plan, pred)
        off = offset_recipe(shape; scalars, n_rows = rows,
            broadcast = _broadcast_affine(plan, pred))
        off !== nothing && push!(stmts, off)
    end
    # GLM-object response matrices: one `X = Float64.(hcat(...))` recipe
    # per matrix (emitted once, shared across responses). Columns are
    # data-only, so the assembly folds at prepare; the object branch
    # prepends its own ones column downstream.
    seen_matrices = Set{Symbol}()
    for r in plan.responses
        _is_glm_family(r.family) || continue
        r.predictor in seen_matrices && continue
        push!(seen_matrices, r.predictor)
        m = _find_matrix(plan, r.predictor)
        push!(stmts, :($(m.name) = Float64.(hcat($(m.columns...)))))
    end
    return stmts
end

# Affine predictors with singleton, non-vector or live derived operands
# use per-block coefficient expressions. This retains
# Julia's broadcast axes and avoids mixing bound and active columns in a
# generated hcat. Data-only derived columns are already present in columns.
# A proven scalar location keeps its scalar shape too: the observation
# plate broadcasts it as a shared argument, with no row vector.
function _broadcast_affine(plan::StructuralPlan, pred::PredictorSpec)
    _uses_structured_observation_axes(plan) && return false
    _scalar_valued_predictor(plan, pred) && return true
    for t in pred.terms
        t.kind in (ContinuousTerm, OffsetTerm, FactorTerm, ComposedTerm) || continue
        for c in t.columns
            t.kind === ContinuousTerm && !haskey(plan.columns, c) &&
                _is_derived(plan, c) && return true
            col = get(plan.columns, c, nothing)
            col isa AbstractArray || continue
            (ndims(col) != 1 || length(col) != plan.n_obs) && return true
        end
    end
    observations = _observation_axes(plan)
    observations === nothing && return false
    # Scalar-only affine predictors also retain scalar shape beside an
    # array response or several independent observation domains.
    return any(a -> length(a) != 1 || length(only(a)) != plan.n_obs,
        values(observations.domains))
end

# Every summand is a proven scalar (`_value_axes` is `Any[]`): an intercept
# coefficient, a scalar offset or covariate, or a composition of sampled
# scalars, numbers and scalar definitions. An opaque call result has an
# unknown shape and keeps the location's rows.
function _scalar_valued_predictor(plan::StructuralPlan, pred::PredictorSpec)
    isempty(pred.terms) && return false
    scalar(ex) = _value_axes(plan, ex; data_axes = true) == Any[] &&
        !_reads_rank0_data(plan, ex)
    return all(pred.terms) do t
        t.kind === InterceptTerm && return true
        t.kind in (ContinuousTerm, OffsetTerm) && return all(scalar, t.columns)
        t.kind === ComposedTerm || return false
        isempty(t.columns) && isempty(t.options.subs) || return false
        return scalar(t.options.tree)
    end
end

# A rank-0 array has no axes (`_value_axes` gives `Any[]`) yet stays an
# array (`fill(0.5)`), so a value reading one is not a proven `Number`.
function _reads_rank0_data(plan::StructuralPlan, ex,
        seen::Set{Symbol} = Set{Symbol}())
    if ex isa Symbol
        value = get(plan.columns, ex, nothing)
        value isa AbstractArray && return ndims(value) == 0
        ex in seen && return false
        push!(seen, ex)
        for defs in (plan.assignments, plan.derived)
            i = findfirst(a -> a.name === ex, defs)
            i === nothing || return _reads_rank0_data(plan, defs[i].expr, seen)
        end
        return false
    end
    ex isa Expr || return false
    return any(a -> _reads_rank0_data(plan, a, seen), ex.args)
end

"""Design row count of a predictor, resolved from its authored inputs."""
_predictor_rows(plan::StructuralPlan, pname::Symbol) = _value_rows(plan, pname)

function _contrast_expr(b::DesignBlock)
    @assert b.kind === FactorTerm
    lvlvec = Expr(:vect, (_level_literal(lvl) for lvl in b.levels)...)
    # Compare labels with Julia's identity relation, including missing/NaN.
    g = b.column
    perms = :(permutedims($lvlvec))
    return :(Float64.(isequal.($g, $perms)))
end

# One matrix block as an in-graph data expression:
# `Float64.(hcat(ones(n), col, …))` over bound columns (the
# `_contrast_expr` precedent — pure data, so the Enzyme reverse pass
# sees a constant). The generator applies the block's coefficient slice.
function _matrix_block_expr(b::DesignBlock, n_obs)
    @assert b.kind === MatrixTerm
    parts = Any[e === nothing ? :(ones($n_obs)) : e for e in b.elements]
    return :(Float64.($(Expr(:call, :hcat, parts...))))
end

# Nonliteral level values stay quoted data; a Symbol must also be quoted
# so the emitted expression never interprets a label as a variable.
_level_literal(lvl::Union{Number,String,Bool,Char}) = lvl
_level_literal(lvl::Symbol) = QuoteNode(lvl)
_level_literal(lvl) = QuoteNode(lvl)

"""
    _declared_codes(x, levels) -> Vector{Int}

In-model grouping encoder: the 1-based position of each element of raw
grouping column `x` in DECLARED `levels` order — no sorting (SB
numbering parity: SB numbers `CA.levels` order for categorical
groupings, sort order otherwise). The generator splices one call per
grouped column (`_ppl_gidx_<group> = _declared_codes(<group>,
[<levels...>])`); both inputs are bound data, so the call folds under
`bound=` and the Enzyme reverse pass sees no new surface. Validated
plans only: bind-time coverage validation proves every value occurs in
`levels`, so the 0 fallback for uncovered values is unreachable
in-graph (a loud bind error beats a silent wrong gather).
"""
function _declared_codes(x::AbstractVector, levels::AbstractVector)
    codes = Vector{Int}(undef, length(x))
    for (i, v) in enumerate(x)
        c = 0
        for (j, lv) in enumerate(levels)
            if isequal(v, lv)
                c = j
                break
            end
        end
        codes[i] = c
    end
    return codes
end

# Stage lanes of a stopping-ratio response (SB `brm_ordinal` structure 2):
# observation `i` with level `y[i]` passes stages `1..min(y[i], K-1)` —
# survives (logCC) below its level, stops (logF) at it. The per-observation
# stage count is data, so the stages flatten into one lane per
# (observation, stage) pair, built from the bound response: the lane's
# observation index and stage index. Data-only (folded under `bound=`).
@inline function _ordinal_effects_matrix(effects, y, K)
    return _ordinal_effects_matrix(effects, length(y), K)
end
@inline function _ordinal_effects_matrix(effects, n::Integer, K)
    ndims(effects) == 2 && size(effects) == (n, K - 1) ||
        throw(DimensionMismatch("ordinal threshold effects require an N × (K-1) matrix"))
    return effects
end

function _ordinal_stage_obs(y::AbstractVector{<:Integer}, K::Integer)
    out = Int[]
    for (i, v) in enumerate(y), _ in 1:min(v, K - 1)
        push!(out, i)
    end
    return out
end
function _ordinal_stage_idx(y::AbstractVector{<:Integer}, K::Integer)
    out = Int[]
    for v in y, j in 1:min(v, K - 1)
        push!(out, j)
    end
    return out
end

# The values a prior argument of a two-axis elementwise array gives its
# packed cells: a number is shared by every element, and an array of the
# declared size gives each element its own value, read column-major as the
# array packs (`B[a, b] .~ Normal.(M, s)` draws `B[i, j]` from
# `Normal(M[i, j], s)`). A size known at binding is checked there; this
# check covers values computed in the graph. The message is built out of
# line, so the reshape itself stays inlinable.
_array_prior_cells(a::Number, dims::Dims) = a
@traceable function _array_prior_cells(a::AbstractArray, dims::Dims)
    size(a) == dims || _array_prior_cells_mismatch(size(a), dims)
    return vec(a)
end
@noinline _array_prior_cells_mismatch(size, dims) = throw(DimensionMismatch(
    "a per-element prior argument of size $size does not match its array " *
    "of size $dims"))

# Gather observation values after ordinary scalar/singleton broadcasting.
# Stage tables may repeat rows; one-entry operands still supply every lane,
# including zero lanes for an empty response. This stays in the value graph.
@inline _broadcast_gather(value::Number, rows) = fill(value, length(rows))
@inline _broadcast_gather(value, rows) =
    value[length(value) == 1 ? fill(1, length(rows)) : rows]
