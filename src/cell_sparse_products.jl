# A product whose left factor is bound sparse data, `L * eta`, at a cell's rows.
#
# A sparse matrix declares its structure in its type: row `r` of `L * eta` is a
# sum over the entries stored in row `r`, which is exactly how Julia's sparse
# product computes it, so that row reads `eta` only at those entries' columns.
# Numeric zeros of a dense factor are not structure and keep reading the whole
# operand (`0 * Inf` is `NaN` there); an explicitly stored zero is structure and
# is read. A cell therefore computes `eta` at the stored columns of its rows, and
# its footprint holds those columns instead of the whole operand.
#
# Finding a row of a compressed-column matrix costs its whole column count, once
# per cell. So the rows are indexed once, when the data is bound: the plan gets
# the index as one more bound constant (`_cell_sparse_products`), and a cell costs
# its own stored entries however large the matrix is. A plan with no cell, and a
# factor that is not bound, keep the authored product.

"""
    _CellSparseRows(matrix, nrows, ncols, rowptr, colval, nzval)

The entries stored in a bound sparse `matrix`, by row: row `r` holds columns
`colval[rowptr[r]:rowptr[r + 1] - 1]`, ascending, with values `nzval` at the same
positions.
"""
struct _CellSparseRows{M,Tv}
    matrix::M
    nrows::Int
    ncols::Int
    rowptr::Vector{Int}
    colval::Vector{Int}
    nzval::Vector{Tv}
end
Base.show(io::IO, s::_CellSparseRows) = print(io, "stored rows of a ", s.nrows, "×", s.ncols,
    " sparse matrix (", length(s.colval), " entries)")

# The index of a sparse matrix, `nothing` for anything else, and whether a type
# is sparse (ext/ReactiveKernelsSparseArraysExt.jl).
_cell_sparse_rows(::Any) = nothing
_cell_sparse_type(::Type) = false

"""
    _CellSparseProduct()

The operation of a product over a bound sparse factor's index. Called whole, as
when another reader needs every row, it is the authored product.
"""
struct _CellSparseProduct end
@inline (::_CellSparseProduct)(s::_CellSparseRows, W) = s.matrix * W
_opname(::_CellSparseProduct) = "*"

@inline _cell_sparse_stored(s::_CellSparseRows, r::Integer) = s.rowptr[r]:(s.rowptr[r + 1] - 1)

# The columns stored in rows `rows`: ascending and unique, the operand's rows a
# cell asks for.
_cell_sparse_columns(s::_CellSparseRows, r::Integer) = s.colval[_cell_sparse_stored(s, r)]
function _cell_sparse_columns(s::_CellSparseRows, rows::Vector{Int})
    columns = Int[]
    for r in rows
        append!(columns, view(s.colval, _cell_sparse_stored(s, r)))
    end
    unique!(sort!(columns))
end

# The product's full axes from its operand's. Its shape is still checked though
# only some rows of the operand are computed.
function _cell_sparse_shape(s::_CellSparseRows, operand::Tuple)
    length(first(operand)) == s.ncols || throw(DimensionMismatch(
        "a sparse matrix of $(s.ncols) columns times an operand of $(length(first(operand))) rows"))
    (Base.OneTo(s.nrows), Base.tail(operand)...)
end

@inline _cell_sparse_eltype(s::_CellSparseRows{M,Tv}, W) where {M,Tv} =
    Base.promote_op(LinearAlgebra.matprod, Tv, eltype(W))

# Rows `rows` of the product, from the operand's rows at `columns`: each row's
# stored entries in ascending column, the order of Julia's sparse product, so the
# values are the same bit for bit.
function _cell_sparse_product(s::_CellSparseRows, rows, columns::Vector{Int}, W::AbstractVector)
    out = zeros(_cell_sparse_eltype(s, W), length(rows))
    for (i, r) in enumerate(rows), p in _cell_sparse_stored(s, r)
        out[i] += s.nzval[p] * W[searchsortedfirst(columns, s.colval[p])]
    end
    out
end
function _cell_sparse_product(s::_CellSparseRows, rows, columns::Vector{Int}, W::AbstractMatrix)
    out = zeros(_cell_sparse_eltype(s, W), length(rows), size(W, 2))
    for (i, r) in enumerate(rows), p in _cell_sparse_stored(s, r)
        c = searchsortedfirst(columns, s.colval[p])
        v = s.nzval[p]
        for k in axes(W, 2)
            out[i, k] += v * W[c, k]
        end
    end
    out
end

# `p` with every product `L * W` over a bound sparse `L` and a dense `W` reading
# `L`'s index instead. Runs only where cells are lowered. Binding `L` emits a product
# inside a larger expression (`a .+ L * w`) as part of one closure, whose operands
# this cannot separate: it keeps its row form and reads `w` whole.
function _cell_sparse_products(p::Plan; argument_types = nothing, outlined = Set{Int}())
    old = p.graph
    cidof(v) = canon_id(old, v.id)
    factors = Dict{ValueId,Any}()
    for r in p.recipes
        r.op isa _BoundConstant && length(r.outputs) == 1 &&
            _cell_sparse_type(typeof(r.op.value)) &&
            (factors[cidof(only(r.outputs))] = r.op.value)
    end
    isempty(factors) && return p
    types = _cell_value_types(p; argument_types, outlined)
    # The factor's input slot and the operand, when `r` is such a product.
    function product(r)
        length(r.inputs) == 2 && length(r.outputs) == 1 && !r.effectful || return nothing
        f = _cell_form(r; outlined = r.id in outlined)
        f !== nothing && f.kind === :product && f.operand != 0 || return nothing
        j = findfirst(==(:factor), f.roles)
        j !== nothing && j != f.operand && haskey(factors, cidof(r.inputs[j])) || return nothing
        T = get(types, cidof(r.inputs[f.operand]), Any)
        T <: AbstractVecOrMat && !_cell_sparse_type(T) || return nothing
        _cell_sparse_rows(factors[cidof(r.inputs[j])]) === nothing && return nothing
        (j, r.inputs[f.operand])
    end
    any(r -> product(r) !== nothing, p.recipes) || return p
    g = Graph(copy(old.values), copy(old.recipes), copy(old.producers),
              copy(old.aliases), old.version)
    next_id = minimum((r.id for r in p.recipes); init = 0) - 1
    indexed = Dict{ValueId,Any}()
    recipes = Recipe[]
    for r in p.recipes
        found = product(r)
        if found === nothing
            push!(recipes, r)
            continue
        end
        j, operand = found
        factor = cidof(r.inputs[j])
        rows = get!(() -> _cell_sparse_rows(factors[factor]), indexed, factor)
        i = value!(g, Symbol(r.inputs[j].name, :_stored_rows), typeof(rows))
        push!(recipes, Recipe(next_id, (), (i,), _BoundConstant(rows), 0.0, nothing, false, rows))
        next_id -= 1
        push!(recipes, Recipe(r.id, (i, operand), r.outputs, _CellSparseProduct(), r.cost,
                              nothing, false, r.source))
    end
    template = Plan(g, p.have, p.want, p.recipes, p.producer, p.cost, p.candidates)
    _partial_subplan(template, p.have, p.want, recipes)
end
