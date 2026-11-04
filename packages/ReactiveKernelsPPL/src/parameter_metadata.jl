"""
    parameter_metadata(built) -> Vector{<:NamedTuple}
    parameter_metadata(layout::LayoutTable) -> Vector{<:NamedTuple}

Inspect the constrained parameters of an existing [`build_kernel`](@ref)
result (including a BRM backend's `model`) or its layout. Each record contains:

- `path`: a tuple of authored `Symbol` name segments. Submodel namespaces
  remain separate segments; a literal dot in a name remains part of that name.
- `rank`, `shape`: the constrained value's number of dimensions and size tuple.
  A scalar has rank `0` and shape `()`.
- `element_shapes`: `nothing` for ordinary scalar/array parameters; for a
  container of cell values, a fresh vector of size tuples, including scalar
  `()` and empty cells. Its outer `shape` is `(number_of_cells,)`.
- `ncoordinates`: the parameter's packed unconstrained coordinate count.
  This can differ from its constrained size (e.g. a length-three simplex has
  two coordinates). Parameters with zero coordinates are retained.

Records follow the depth-first leaf order of [`constrain`](@ref)'s named
result. Split coefficient runs are grouped under their exposed predictor name.
Only parameters present in the layout are reported, not data, conditioned
values or derived quantities.

This reads stored layout dimensions and bound cell-shape tables. It does not
bind, build, prepare, fit or evaluate a model, bound expression, data operation,
or caller-owned geometry endpoint. Returned containers do not alias the layout.

Dimensions belong to the supplied layout's binding. Reusing a build with new
data (for example BRM `reprocess`) does not update that build's layout. To
inspect a different binding, supply its already-created layout. A bound
`StructuralPlan` alone is not accepted: this accessor does not assign a layout
or infer current dimensions from a different binding.
"""
parameter_metadata(built::NamedTuple) = parameter_metadata(built.layout)

function parameter_metadata(layout::LayoutTable)
    records = NamedTuple[]
    pairs = Pair{Symbol,Int}[]
    groups = _coefficient_groups(layout)
    seen = Set{Symbol}()
    # Structural traversal only: bound dependencies can change which split
    # coefficient group constrain encounters first, without running a bound.
    for e in _bound_entry_order(layout)
        name = e.kind === :coefficient ? e.predictor::Symbol : e.name
        name in seen && continue
        push!(seen, name)
        ncoordinates = e.kind === :coefficient ? sum(x.size for x in groups[name]) : e.size
        shape, element_shapes = _parameter_shape(layout, e, ncoordinates)
        path = get(layout.name_paths, name, (name,))
        push!(records, (; path, rank = length(shape), shape, element_shapes, ncoordinates))
        push!(pairs, name => length(records))
    end
    # Match constrain's flat order and its grouping of interleaved namespaces.
    order = Dict(e.name => i for (i, e) in enumerate(layout.entries))
    sort!(pairs; by = p -> get(order, first(p), 0))
    indices = Int[]
    _parameter_leaf_order!(indices, _scoped_draw_values(layout, pairs))
    return records[indices]
end

function _parameter_leaf_order!(out, values::NamedTuple)
    foreach(value -> _parameter_leaf_order!(out, value), values)
    return out
end
_parameter_leaf_order!(out, index::Int) = push!(out, index)

function _parameter_shape(layout, e, ncoordinates)
    if haskey(layout.cell_geometries, e.name)
        shapes_name = layout.cell_geometries[e.name].shapes
        haskey(layout.bound_values, shapes_name) || throw(ContractValidationError(
            "[layout] parameter metadata for $(e.name) needs bound cell shapes $shapes_name"))
        shapes = layout.bound_values[shapes_name]
    elseif haskey(layout.cell_latents, e.name)
        shapes = _cell_latent_dims(layout, e.name)
    else
        return (_parameter_shape(Val(e.kind), e, ncoordinates), nothing)
    end
    return ((length(shapes),), Tuple{Vararg{Int}}[Tuple(d) for d in shapes])
end

_parameter_shape(::Val{:sampled}, e, n) = ()
_parameter_shape(::Val{:coefficient}, e, n) = (n,)
_parameter_shape(::Val{:plate}, e, n) = (n,)
_parameter_shape(::Val{:external_plate}, e, n) = (n,)
_parameter_shape(::Val{:glm}, e, n) = (n,)
_parameter_shape(::Val{:scan}, e, n) = (n,)
_parameter_shape(::Val{:vector}, e, n) = (_vector_constrained_size(e),)
_parameter_shape(::Val{:array}, e, n) = Tuple(e.dims)
_parameter_shape(::Val{:external}, e, n) = Tuple(e.dims)
_parameter_shape(::Val{:cholesky_corr}, e, n) = (_lkj_dim(n), _lkj_dim(n))
