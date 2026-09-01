# Finite structural-vector lowering for optional tensor backends.
#
# The host value is a fixed-length Vector whose elements may contain builtin
# numeric scalars/arrays, the explicitly supported structural wrappers below,
# and identities declared static by a compiler binding.  The backend value is
# a NamedTuple containing exactly one dense numeric tensor per source-logical
# owned leaf group.  Static identities are held only by the contract and never
# cross the dynamic backend ABI.

abstract type _SMFiniteStructuralNode end

struct _SMFiniteScalarNode{Index,T} <: _SMFiniteStructuralNode end
struct _SMFiniteArrayNode{Index,A,Shape} <: _SMFiniteStructuralNode end
struct _SMFiniteStaticNode{Index,T} <: _SMFiniteStructuralNode end

struct _SMFiniteNamedTupleNode{Names,C<:Tuple} <: _SMFiniteStructuralNode
    children::C
end
struct _SMFiniteTupleNode{C<:Tuple} <: _SMFiniteStructuralNode
    children::C
end
struct _SMFiniteDiagonalNode{C<:_SMFiniteStructuralNode} <:
        _SMFiniteStructuralNode
    child::C
end
struct _SMFiniteCholeskyNode{Uplo,Info,C<:_SMFiniteStructuralNode} <:
        _SMFiniteStructuralNode
    child::C
end

struct _SMFiniteScalarColumnSpec{T} end
struct _SMFiniteArrayColumnSpec{A,Shape,T} end

struct _SMFiniteStructuralContract{
        Element,Capacity,Names,Schema,Specs,StaticValues,Paths,OwnedGroups} <:
        _SMFiniteStructuralPort
    schema::Schema
    specs::Specs
    static_values::StaticValues
    representative_paths::Paths
    owned_array_groups::OwnedGroups
end

function _sm_finite_static_values(port::_StructuredStatePort)
    transition = getfield(port, :transition)
    Groups, ExternalGroups = typeof(transition).parameters[2:3]
    initial = getfield(transition, :initial)
    Tuple(getfield(initial, first(Groups[index]))
          for index in ExternalGroups)
end

mutable struct _SMFiniteStructuralBuilder{S}
    static_values::S
    static_used::BitVector
    array_columns::IdDict{Any,Int}
    specs::Vector{Any}
    representative_paths::Vector{Tuple}
    owned_array_groups::Vector{Vector{Tuple}}
end

function _sm_finite_static_index(static_values::Tuple, value)
    findfirst(candidate -> candidate === value, static_values)
end

function _sm_finite_add_scalar!(builder::_SMFiniteStructuralBuilder,
                               value, path::Tuple)
    index = length(builder.specs) + 1
    T = typeof(value)
    push!(builder.specs, _SMFiniteScalarColumnSpec{T}())
    push!(builder.representative_paths, path)
    push!(builder.owned_array_groups, Tuple[])
    _SMFiniteScalarNode{index,T}()
end

function _sm_finite_add_array!(builder::_SMFiniteStructuralBuilder,
                              value, path::Tuple)
    index = get(builder.array_columns, value, 0)
    if index == 0
        index = length(builder.specs) + 1
        builder.array_columns[value] = index
        A = typeof(value)
        Shape = size(value)
        push!(builder.specs,
              _SMFiniteArrayColumnSpec{A,Shape,eltype(A)}())
        push!(builder.representative_paths, path)
        push!(builder.owned_array_groups, Tuple[path])
    else
        spec = builder.specs[index]
        spec isa _SMFiniteArrayColumnSpec{typeof(value),size(value),eltype(value)} ||
            throw(ArgumentError(
                "finite structural prototype aliases arrays with conflicting logical contracts"))
        push!(builder.owned_array_groups[index], path)
    end
    _SMFiniteArrayNode{index,typeof(value),size(value)}()
end

function _sm_finite_schema!(builder::_SMFiniteStructuralBuilder,
                            value, path::Tuple)
    static_index = _sm_finite_static_index(builder.static_values, value)
    if static_index !== nothing
        builder.static_used[static_index] = true
        return _SMFiniteStaticNode{static_index,typeof(value)}()
    elseif _sm_finite_backend_primitive(typeof(value))
        return _sm_finite_add_scalar!(builder, value, path)
    elseif value isa LinearAlgebra.Diagonal
        child = _sm_finite_schema!(
            builder, value.diag, (path..., :diag))
        return _SMFiniteDiagonalNode{typeof(child)}(child)
    elseif value isa LinearAlgebra.Cholesky
        child = _sm_finite_schema!(
            builder, value.factors, (path..., :factors))
        return _SMFiniteCholeskyNode{
            value.uplo,value.info,typeof(child)}(child)
    elseif value isa NamedTuple
        names = propertynames(value)
        children = Tuple(_sm_finite_schema!(
            builder, getfield(value, name), (path..., name))
            for name in names)
        return _SMFiniteNamedTupleNode{names,typeof(children)}(children)
    elseif value isa Tuple
        children = Tuple(_sm_finite_schema!(
            builder, getfield(value, index), (path..., index))
            for index in eachindex(value))
        return _SMFiniteTupleNode{typeof(children)}(children)
    elseif value isa AbstractArray
        _sm_finite_backend_array(typeof(value)) || throw(ArgumentError(
            "finite structural prototype rejects unsupported backend array `$(typeof(value))` at $path"))
        return _sm_finite_add_array!(builder, value, path)
    end
    throw(ArgumentError(
        "finite structural prototype leaf `$(typeof(value))` at $path is not " *
        "a backend primitive or an explicitly bound static identity"))
end

function _sm_finite_validate_node(
        ::_SMFiniteScalarNode{Index,T}, value, static_values,
        path::Tuple, ::Val{Strict}) where {Index,T,Strict}
    valid = Strict ? typeof(value) === T :
        _sm_functional_argument_type_ok(typeof(value), T)
    valid || throw(ArgumentError(
        "finite structural scalar at $path does not match logical type `$T`"))
    value
end

function _sm_finite_validate_node(
        ::_SMFiniteArrayNode{Index,A,Shape}, value, static_values,
        path::Tuple, ::Val{Strict}) where {Index,A,Shape,Strict}
    valid = Strict ? typeof(value) === A :
        _sm_functional_argument_type_ok(typeof(value), A)
    valid || throw(ArgumentError(
        "finite structural array at $path does not match logical type `$A`"))
    size(value) == Shape || throw(ArgumentError(
        "finite structural array at $path has axes $(size(value)); expected $Shape"))
    value
end

function _sm_finite_validate_node(
        ::_SMFiniteStaticNode{Index,T}, value, static_values,
        path::Tuple, ::Val{Strict}) where {Index,T,Strict}
    value === getfield(static_values, Index) || throw(ArgumentError(
        "finite structural static identity at $path was replaced"))
    value
end

function _sm_finite_validate_node(
        node::_SMFiniteNamedTupleNode{Names}, value, static_values,
        path::Tuple, strict::Val) where {Names}
    value isa NamedTuple && propertynames(value) == Names ||
        throw(ArgumentError(
            "finite structural value at $path has the wrong named-tuple layout"))
    for (name, child) in zip(Names, node.children)
        _sm_finite_validate_node(
            child, getfield(value, name), static_values,
            (path..., name), strict)
    end
    value
end

function _sm_finite_validate_node(
        node::_SMFiniteTupleNode, value, static_values,
        path::Tuple, strict::Val)
    value isa Tuple && length(value) == length(node.children) ||
        throw(ArgumentError(
            "finite structural value at $path has the wrong tuple layout"))
    for index in eachindex(node.children)
        _sm_finite_validate_node(
            getfield(node.children, index), getfield(value, index),
            static_values, (path..., index), strict)
    end
    value
end

function _sm_finite_validate_node(
        node::_SMFiniteDiagonalNode, value, static_values,
        path::Tuple, strict::Val)
    value isa LinearAlgebra.Diagonal || throw(ArgumentError(
        "finite structural value at $path no longer has its Diagonal wrapper"))
    _sm_finite_validate_node(
        node.child, value.diag, static_values, (path..., :diag), strict)
    value
end

function _sm_finite_validate_node(
        node::_SMFiniteCholeskyNode{Uplo,Info}, value, static_values,
        path::Tuple, strict::Val) where {Uplo,Info}
    value isa LinearAlgebra.Cholesky || throw(ArgumentError(
        "finite structural value at $path no longer has its Cholesky wrapper"))
    value.uplo === Uplo && value.info === Info || throw(ArgumentError(
        "finite structural Cholesky metadata at $path was replaced"))
    _sm_finite_validate_node(
        node.child, value.factors, static_values,
        (path..., :factors), strict)
    value
end

function _sm_finite_validate_owned_groups(
        value, groups::Tuple, path_prefix::Tuple=())
    leaders = Any[]
    for group in groups
        leader = _sm_topology_value(value, first(group))
        all(path -> _sm_topology_value(value, path) === leader, group) ||
            throw(ArgumentError(
                "finite structural value breaks required owned alias group $group"))
        push!(leaders, leader)
    end
    for left in eachindex(leaders), right in eachindex(leaders)
        left < right || continue
        leaders[left] !== leaders[right] || throw(ArgumentError(
            "finite structural value merges distinct owned leaf groups " *
            "$(groups[left]) and $(groups[right])"))
    end
    Tuple(leaders)
end

function _sm_finite_validate_element(
        contract::_SMFiniteStructuralContract, value,
        strict::Val=Val(true))
    _sm_finite_validate_node(
        contract.schema, value, contract.static_values, (), strict)
    _sm_finite_validate_owned_groups(
        value, contract.owned_array_groups)
end

function _sm_finite_validate_elements(
        contract::_SMFiniteStructuralContract{Element,Capacity},
        values::Vector) where {Element,Capacity}
    length(values) == Capacity || throw(ArgumentError(
        "finite structural vector has capacity $(length(values)); expected $Capacity"))
    prior_leaders = Any[]
    for (index, value) in enumerate(values)
        strict = typeof(value) === Element
        strict || _sm_functional_argument_type_ok(typeof(value), Element) ||
            throw(ArgumentError(
            "finite structural element $index has type `$(typeof(value))`; expected `$Element`"))
        # Optional tensor backends recursively replace builtin numeric leaves
        # with their exact logical tracer types while retaining the authored
        # aggregate layout.  Validate those leaves through the registered
        # functional-type bridge; ordinary host values still take the exact
        # concrete-type path.  Shape, static-identity, and alias-topology
        # checks remain identical in both modes.
        leaders = _sm_finite_validate_element(
            contract, value, Val(strict))
        for leader in leaders, prior in prior_leaders
            leader !== prior || throw(ArgumentError(
                "finite structural prototype shares owned storage across elements"))
        end
        append!(prior_leaders, leaders)
    end
    values
end

function _sm_finite_structural_contract(
        prototype::Vector; static_values::Tuple=())
    isempty(prototype) && throw(ArgumentError(
        "finite structural prototype must have positive capacity"))
    for left in eachindex(static_values), right in eachindex(static_values)
        left < right || continue
        static_values[left] !== static_values[right] || throw(ArgumentError(
            "finite structural static identities must be unique"))
    end
    builder = _SMFiniteStructuralBuilder(
        static_values, falses(length(static_values)), IdDict{Any,Int}(),
        Any[], Tuple[], Vector{Vector{Tuple}}())
    schema = _sm_finite_schema!(builder, first(prototype), ())
    all(builder.static_used) || throw(ArgumentError(
        "finite structural contract declares an unused static identity"))
    specs = Tuple(builder.specs)
    paths = Tuple(builder.representative_paths)
    groups = Tuple(Tuple(group) for group in builder.owned_array_groups
                   if !isempty(group))
    names = ntuple(index -> Symbol(:leaf_, index), length(specs))
    Element = typeof(first(prototype))
    Capacity = length(prototype)
    contract = _SMFiniteStructuralContract{
        Element,Capacity,names,typeof(schema),typeof(specs),
        typeof(static_values),typeof(paths),typeof(groups)}(
            schema, specs, static_values, paths, groups)
    _sm_finite_validate_elements(contract, prototype)
    contract
end

function _sm_finite_pack_column(
        ::_SMFiniteScalarColumnSpec{T}, values::Vector,
        path::Tuple, capacity::Int) where {T}
    leaves = map(value -> _sm_topology_value(value, path), values)
    typeof(first(leaves)) === T ? T[leaves...] : stack(leaves)
end

function _sm_finite_pack_column(
        ::_SMFiniteArrayColumnSpec{A,Shape,T}, values::Vector,
        path::Tuple, capacity::Int) where {A,Shape,T}
    leaves = map(value -> _sm_topology_value(value, path), values)
    typeof(first(leaves)) === A || return stack(leaves)
    column = Array{T}(undef, Shape..., capacity)
    dimension = length(Shape) + 1
    for index in 1:capacity
        selectdim(column, dimension, index) .= leaves[index]
    end
    column
end

# Optional tensor backends may trace only some leaves of one logical structural
# value.  Give the backend one chance to promote the completed column tuple so
# that the fixed loop carry has a single concrete representation from entry
# through every later read/write.  Native storage remains unchanged.
@inline _sm_finite_pack_backend(raw) = raw

function _sm_finite_structural_pack(
        contract::_SMFiniteStructuralContract{
            Element,Capacity,Names}, values::Vector) where
        {Element,Capacity,Names}
    _sm_finite_validate_elements(contract, values)
    columns = map(contract.specs, contract.representative_paths) do spec, path
        _sm_finite_pack_column(spec, values, path, Capacity)
    end
    raw = _sm_finite_pack_backend(NamedTuple{Names}(columns))
    _sm_finite_validate_raw(contract, raw)
end

function _sm_finite_structural_logical_copy(
        contract::_SMFiniteStructuralContract, values::Vector)
    _sm_finite_structural_unpack(
        contract, _sm_finite_structural_pack(contract, values))
end

function _sm_finite_raw_column_ok(
        column, ::_SMFiniteScalarColumnSpec{T},
        ::Val{Capacity}) where {T,Capacity}
    _sm_functional_argument_type_ok(typeof(column), Vector{T}) &&
        size(column) == (Capacity,)
end

function _sm_finite_raw_column_ok(
        column, ::_SMFiniteArrayColumnSpec{A,Shape,T},
        ::Val{Capacity}) where {A,Shape,T,Capacity}
    Expected = Array{T,length(Shape) + 1}
    _sm_functional_argument_type_ok(typeof(column), Expected) &&
        size(column) == (Shape..., Capacity)
end

function _sm_finite_validate_raw(
        contract::_SMFiniteStructuralContract{Element,Capacity,Names}, raw) where
        {Element,Capacity,Names}
    raw isa NamedTuple && propertynames(raw) == Names ||
        throw(ArgumentError(
            "finite structural raw ABI has missing, extra, or reordered columns"))
    columns = values(raw)
    length(columns) == length(contract.specs) || throw(ArgumentError(
        "finite structural raw ABI has the wrong column count"))
    for (index, (column, spec)) in enumerate(zip(columns, contract.specs))
        _sm_finite_raw_column_ok(column, spec, Val(Capacity)) ||
            throw(ArgumentError(
                "finite structural raw column $index has the wrong numeric type, axes, or capacity"))
    end
    raw
end

_sm_finite_capacity(
    ::_SMFiniteStructuralContract{Element,Capacity}) where
    {Element,Capacity} = Capacity

function _sm_finite_raw_type(
        contract::_SMFiniteStructuralContract{Element,Capacity,Names}) where
        {Element,Capacity,Names}
    column_types = map(contract.specs) do spec
        if spec isa _SMFiniteScalarColumnSpec
            Vector{typeof(spec).parameters[1]}
        else
            _, shape, element = typeof(spec).parameters
            Array{element,length(shape) + 1}
        end
    end
    NamedTuple{Names,Tuple{column_types...}}
end

@generated function _sm_finite_read_array_column(
        column, index, ::Val{Rank}) where {Rank}
    indices = Any[:(Colon()) for _ in 1:Rank]
    :(_sm_functional_index(column, $(indices...), index))
end

@inline _sm_finite_read_position(
        column, ::_SMFiniteScalarColumnSpec, ::Val{Position}) where {Position} =
    maximum(_sm_functional_index(column, Position:Position))

@inline _sm_finite_read_position(
        column, ::_SMFiniteArrayColumnSpec{A,Shape},
        ::Val{Position}) where {A,Shape,Position} =
    _sm_finite_read_array_column(
        column, Position, Val(length(Shape)))

# The structural capacity is part of the compiled ABI.  Enumerate it at
# generation time and select complete leaves rather than applying a traced
# scalar index to a backend column.  This is the same bounded gather contract
# used by the functional control-frame store.
@generated function _sm_finite_read_column(
        column, index, spec::Spec, ::Val{Capacity}) where {Spec,Capacity}
    selected = :(_sm_finite_read_position(column, spec, Val(1)))
    for position in 2:Capacity
        selected = :(_sm_predicated_select(
            index .== oftype(index, $position),
            _sm_finite_read_position(column, spec, Val($position)),
            $selected))
    end
    selected
end

@inline _sm_finite_reconstruct(
        ::_SMFiniteScalarNode{Index}, leaves, static_values) where {Index} =
    getfield(leaves, Index)
@inline _sm_finite_reconstruct(
        ::_SMFiniteArrayNode{Index}, leaves, static_values) where {Index} =
    getfield(leaves, Index)
@inline _sm_finite_reconstruct(
        ::_SMFiniteStaticNode{Index}, leaves, static_values) where {Index} =
    getfield(static_values, Index)
@inline function _sm_finite_reconstruct(
        node::_SMFiniteNamedTupleNode{Names}, leaves,
        static_values) where {Names}
    NamedTuple{Names}(map(child -> _sm_finite_reconstruct(
        child, leaves, static_values), node.children))
end
@inline function _sm_finite_reconstruct(
        node::_SMFiniteTupleNode, leaves, static_values)
    map(child -> _sm_finite_reconstruct(
        child, leaves, static_values), node.children)
end
@inline _sm_finite_reconstruct(
        node::_SMFiniteDiagonalNode, leaves, static_values) =
    LinearAlgebra.Diagonal(
        _sm_finite_reconstruct(node.child, leaves, static_values))
@inline function _sm_finite_reconstruct(
        node::_SMFiniteCholeskyNode{Uplo,Info}, leaves,
        static_values) where {Uplo,Info}
    _sm_cholesky_reconstruct(
        _sm_finite_reconstruct(node.child, leaves, static_values),
        Uplo, Info)
end

# Optional backends may need representation-only wrappers while tracing (for
# example Reactant's BatchedCholesky).  Reusable results cross back to the
# source-logical ABI before their next invocation: retain the backend arrays,
# but rebuild every structural wrapper from the frozen source schema.  This is
# intentionally separate from `_sm_finite_reconstruct`, whose backend-aware
# wrapper choice is required inside the compiled program.
@inline _sm_finite_restore_logical(
        ::_SMFiniteScalarNode, value, static_values) = value
@inline _sm_finite_restore_logical(
        ::_SMFiniteArrayNode, value, static_values) = value
@inline _sm_finite_restore_logical(
        ::_SMFiniteStaticNode{Index}, value, static_values) where {Index} =
    getfield(static_values, Index)
@inline function _sm_finite_restore_logical(
        node::_SMFiniteNamedTupleNode{Names}, value,
        static_values) where {Names}
    NamedTuple{Names}(map(Names, node.children) do name, child
        _sm_finite_restore_logical(
            child, getfield(value, name), static_values)
    end)
end
@inline function _sm_finite_restore_logical(
        node::_SMFiniteTupleNode, value, static_values)
    map(eachindex(node.children), node.children) do index, child
        _sm_finite_restore_logical(
            child, getfield(value, index), static_values)
    end
end
@inline _sm_finite_restore_logical(
        node::_SMFiniteDiagonalNode, value, static_values) =
    LinearAlgebra.Diagonal(_sm_finite_restore_logical(
        node.child, getfield(value, :diag), static_values))
@inline function _sm_finite_restore_logical(
        node::_SMFiniteCholeskyNode{Uplo,Info}, value,
        static_values) where {Uplo,Info}
    LinearAlgebra.Cholesky(
        _sm_finite_restore_logical(
            node.child, getfield(value, :factors), static_values),
        Uplo, Info)
end

function _sm_finite_restore_logical_elements(
        contract::_SMFiniteStructuralContract, values::Vector)
    _sm_finite_validate_elements(contract, values)
    restored = [_sm_finite_restore_logical(
        contract.schema, value, contract.static_values) for value in values]
    _sm_finite_validate_elements(contract, restored)
end

@inline function _sm_finite_inbounds(index, ::Val{Capacity}) where {Capacity}
    (index >= one(index)) .& (index <= oftype(index, Capacity))
end

@inline function _sm_finite_safe_index(index, ::Val{Capacity}) where {Capacity}
    clamp.(index, one(index), oftype(index, Capacity))
end

function _sm_finite_column_values(
        contract::_SMFiniteStructuralContract{Element,Capacity},
        raw, index) where {Element,Capacity}
    map(values(raw), contract.specs) do column, spec
        _sm_finite_read_column(
            column, index, spec, Val(Capacity))
    end
end

function _sm_finite_structural_read(
        contract::_SMFiniteStructuralContract{Element,Capacity},
        raw, index, active=true) where {Element,Capacity}
    _sm_finite_validate_raw(contract, raw)
    valid = _sm_finite_inbounds(index, Val(Capacity))
    safe = _sm_finite_safe_index(index, Val(Capacity))
    leaves = _sm_finite_column_values(contract, raw, safe)
    value = _sm_finite_reconstruct(
        contract.schema, leaves, contract.static_values)
    overflow = _sm_predicated_and(active, _sm_predicated_not(valid))
    (; storage=raw, value, overflow)
end

function _sm_finite_encode_element(
        contract::_SMFiniteStructuralContract, value)
    _sm_finite_validate_element(contract, value, Val(false))
    map(path -> _sm_topology_value(value, path),
        contract.representative_paths)
end

@inline _sm_finite_scalar_candidate(column, value) =
    zero.(column) .+ value
@inline function _sm_finite_bool_candidate(column, active)
    result = copy(column)
    result .= active
    result
end
@inline _sm_finite_scalar_candidate(column::Array{Bool}, value::Bool) =
    _sm_finite_bool_candidate(column, value)

@inline function _sm_finite_array_candidate(
        column, value, ::Val{Rank}) where {Rank}
    zero.(column) .+ reshape(value, (size(value)..., 1))
end
@inline function _sm_finite_array_candidate(
        column::Array{Bool}, value::Array{Bool,Rank},
        ::Val{Rank}) where {Rank}
    _sm_finite_bool_candidate(
        column, reshape(value, (size(value)..., 1)))
end

@inline _sm_finite_select(active, candidate, prior) =
    _sm_predicated_select(active, candidate, prior)
@inline function _sm_finite_select(
        active, candidate::Array{Bool,N}, prior::Array{Bool,N}) where {N}
    result = copy(prior)
    result .= _sm_predicated_select(active, candidate, prior)
    result
end

@inline function _sm_finite_store_column(
        column, value, index, active,
        spec::_SMFiniteScalarColumnSpec)
    positions = collect(axes(column, 1))
    selected = _sm_predicated_and(active, positions .== index)
    candidate = _sm_finite_scalar_candidate(column, value)
    _sm_finite_select(selected, candidate, column)
end

@inline function _sm_finite_store_column(
        column, value, index, active,
        spec::_SMFiniteArrayColumnSpec{A,Shape}) where {A,Shape}
    rank = length(Shape)
    positions = collect(axes(column, rank + 1))
    selector_shape = (ntuple(_ -> 1, rank)..., length(positions))
    selected = _sm_predicated_and(
        active, reshape(positions .== index, selector_shape))
    candidate = _sm_finite_array_candidate(column, value, Val(rank))
    _sm_finite_select(selected, candidate, column)
end

function _sm_finite_write_encoded(
        contract::_SMFiniteStructuralContract{Element,Capacity,Names},
        raw, index, leaves, active) where {Element,Capacity,Names}
    columns = map(values(raw), leaves, contract.specs) do column, value, spec
        _sm_finite_store_column(column, value, index, active, spec)
    end
    NamedTuple{Names}(columns)
end

function _sm_finite_structural_write(
        contract::_SMFiniteStructuralContract{Element,Capacity},
        raw, index, value, active=true) where {Element,Capacity}
    _sm_finite_validate_raw(contract, raw)
    leaves = _sm_finite_encode_element(contract, value)
    valid = _sm_finite_inbounds(index, Val(Capacity))
    safe = _sm_finite_safe_index(index, Val(Capacity))
    commit = _sm_predicated_and(active, valid)
    storage = _sm_finite_write_encoded(
        contract, raw, safe, leaves, commit)
    overflow = _sm_predicated_and(active, _sm_predicated_not(valid))
    (; storage, overflow)
end

function _sm_finite_structural_copy(
        contract::_SMFiniteStructuralContract{Element,Capacity},
        raw, destination, source, active=true) where {Element,Capacity}
    _sm_finite_validate_raw(contract, raw)
    destination_valid = _sm_finite_inbounds(destination, Val(Capacity))
    source_valid = _sm_finite_inbounds(source, Val(Capacity))
    valid = _sm_predicated_and(destination_valid, source_valid)
    safe_destination = _sm_finite_safe_index(destination, Val(Capacity))
    safe_source = _sm_finite_safe_index(source, Val(Capacity))
    leaves = _sm_finite_column_values(contract, raw, safe_source)
    commit = _sm_predicated_and(active, valid)
    storage = _sm_finite_write_encoded(
        contract, raw, safe_destination, leaves, commit)
    overflow = _sm_predicated_and(active, _sm_predicated_not(valid))
    (; storage, overflow)
end

function _sm_finite_swap_column(
        column, left, right, active, spec, capacity::Val)
    left_value = _sm_finite_read_column(
        column, left, spec, capacity)
    right_value = _sm_finite_read_column(
        column, right, spec, capacity)
    with_left = _sm_finite_store_column(
        column, right_value, left, active, spec)
    _sm_finite_store_column(
        with_left, left_value, right, active, spec)
end

function _sm_finite_structural_swap(
        contract::_SMFiniteStructuralContract{
            Element,Capacity,Names}, raw, left, right,
        active=true) where {Element,Capacity,Names}
    _sm_finite_validate_raw(contract, raw)
    left_valid = _sm_finite_inbounds(left, Val(Capacity))
    right_valid = _sm_finite_inbounds(right, Val(Capacity))
    valid = _sm_predicated_and(left_valid, right_valid)
    safe_left = _sm_finite_safe_index(left, Val(Capacity))
    safe_right = _sm_finite_safe_index(right, Val(Capacity))
    commit = _sm_predicated_and(active, valid)
    columns = map(values(raw), contract.specs) do column, spec
        _sm_finite_swap_column(
            column, safe_left, safe_right, commit, spec,
            Val(Capacity))
    end
    storage = NamedTuple{Names}(columns)
    overflow = _sm_predicated_and(active, _sm_predicated_not(valid))
    (; storage, overflow)
end

function _sm_finite_structural_select(
        contract::_SMFiniteStructuralContract{Element,Capacity,Names},
        active, candidate, prior) where {Element,Capacity,Names}
    _sm_finite_validate_raw(contract, candidate)
    _sm_finite_validate_raw(contract, prior)
    columns = map(_sm_finite_select,
                  ntuple(_ -> active, length(contract.specs)),
                  values(candidate), values(prior))
    NamedTuple{Names}(columns)
end

function _sm_finite_structural_unpack(
        contract::_SMFiniteStructuralContract{Element,Capacity}, raw) where
        {Element,Capacity}
    # This validation deliberately precedes reconstruction.  Static identity
    # restoration and alias canonicalization must never mask a counterfeit raw
    # result from a reusable backend executable.
    _sm_finite_validate_raw(contract, raw)
    [_sm_finite_structural_read(contract, raw, index).value
     for index in 1:Capacity]
end

struct _SMFiniteValidatedCall{C,F}
    contract::C
    compiled::F
end

_sm_finite_validated_call(compiled, contract::_SMFiniteStructuralContract) =
    _SMFiniteValidatedCall(contract, compiled)

function (validated::_SMFiniteValidatedCall)(raw, arguments...)
    contract = getfield(validated, :contract)
    _sm_finite_validate_raw(contract, raw)
    output = getfield(validated, :compiled)(raw, arguments...)
    _sm_finite_validate_raw(contract, output)
end
