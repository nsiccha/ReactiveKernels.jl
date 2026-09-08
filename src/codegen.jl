# Lowering a Plan to ordinary straight-line Julia, the optional AST-transform
# boundary, and final RGF compilation into a PreparedKernel.
#
# Operations are *not* referenced as globals in the generated code (that would
# be fragile under world age; gist §9). Instead every selected recipe's `op` is
# passed positionally in an `__ops__` tuple and called by literal index, so the
# generated body is closed over nothing and specializes on the concrete op
# types. The hot path therefore touches no graph object.

const _OPS_ARG = :__ops__
const _CACHES_ARG = :__caches__
const _CACHE_APPLY_ARG = :__cache_apply__

# Assign a globally unique source-variable Symbol to every canonical value in
# the plan. User names are diagnostic hints, not binding authority: they may
# collide with one another, with a generated disambiguation such as `a_12`, or
# with the hidden `__ops__` argument.
function _varnames(p::Plan)
    g = p.graph
    ids = Int[]
    for v in p.have; push!(ids, canon_id(g, v.id)); end
    for r in p.recipes, o in r.outputs; push!(ids, canon_id(g, o.id)); end
    for w in p.want; push!(ids, canon_id(g, w.id)); end
    unique!(ids)
    used = Set{Symbol}((_OPS_ARG, _CACHES_ARG, _CACHE_APPLY_ARG))
    out = Dict{Int,Symbol}()
    for id in ids
        base = p.graph.values[id].name
        candidate = base
        suffix = 0
        while candidate in used
            candidate = suffix == 0 ? Symbol(base, :_, id) :
                        Symbol(base, :_, id, :_, suffix)
            suffix += 1
        end
        out[id] = candidate
        push!(used, candidate)
    end
    out
end

_embedded_kernel(op) = nothing

# A first-class authored plate keeps its scalar pointwise kernel as compiler
# metadata.  The ordinary call method is a semantic fallback (and is useful to
# low-level consumers); `_lower_with_ops` recognizes the operation and emits
# the fused native/tensorized loop products directly.
struct _AuthoredPlateOp{K,A}
    kernel::K
    # Cold lowering metadata: each tuple names the non-atomic arguments of an
    # absorbed plate. Its original broadcast domain must still be valid even
    # when a later plate adds axes (or the scalar body ignores an argument).
    axis_checks::Tuple
end
_AuthoredPlateOp{K,A}(kernel::K) where {K,A} =
    _AuthoredPlateOp{K,A}(kernel, ())

"The transparent scalar plan captured by an authored `plate(...) do` recipe."
function plate_body(recipe::Recipe)
    recipe.op isa _AuthoredPlateOp || throw(ArgumentError(
        "recipe $(recipe.id) is not an authored plate"))
    recipe.op.kernel.plan
end

# Marker discovery must not invoke arbitrary iteration or `broadcastable`
# machinery: outer HAVE values can be atomic structures whose derived fields
# eventually feed a plate.  Restrict axis recognition to the collection shapes
# that the native authored-plate lowering consumes directly.  Backends can
# extend this internal trait for another array representation when necessary.
@inline _authored_plate_is_axis(x) = false
@inline _authored_plate_is_axis(x::AbstractArray) = !isempty(axes(x))
@inline _authored_plate_is_axis(x::Tuple) = !isempty(axes(x))
@inline _authored_plate_is_axis(x::Base.Broadcast.Broadcasted) =
    !isempty(axes(x))

# Compile-time counterpart of `_authored_plate_is_axis`, over a port's declared
# value TYPE. The native plate lowering uses it to bind the axis marker to a
# specific argument statically instead of via a runtime `findfirst` (which
# defeats reverse-mode Enzyme's static-activity analysis — see
# `_lower_authored_plate_native!`). It is deliberately CONSERVATIVE: it returns
# `:axis` / `:not_axis` only when the type PROVES what `_authored_plate_is_axis`
# would decide at runtime for every value of that type, and `:ambiguous`
# otherwise. A metadata-`Any` port (which may hold a runtime scalar), an abstract
# array type, a `Broadcasted`, or a non-axis struct is `:ambiguous`, so the caller
# keeps the runtime marker for it. A rank-0 array has empty axes, so — like a
# scalar — it is `:not_axis`.
@inline _static_plate_axis_class(::Type) = :ambiguous
@inline _static_plate_axis_class(::Type{<:Number}) = :not_axis
@inline _static_plate_axis_class(::Type{<:AbstractArray{<:Any,0}}) = :not_axis
@inline _static_plate_axis_class(::Type{<:AbstractArray{<:Any,N}}) where {N} =
    :axis
@inline _static_plate_axis_class(::Type{<:Tuple}) = :axis

@inline _authored_plate_argument(::Val{A}, index, arg) where {A} =
    index in A ? Ref(arg) : arg

# Keep construction of the immutable Broadcasted descriptor inside the lowered
# plate call.  On Julia 1.10, returning a multi-axis descriptor across either
# helper boundary defeats scalar replacement and allocates on every reduction.
@inline function _authored_plate_arguments(::Val{A}, args...) where {A}
    ntuple(length(args)) do index
        _authored_plate_argument(Val(A), index, getfield(args, index))
    end
end

@inline function _authored_plate_broadcast(::Val{A}, args...) where {A}
    wrapped = _authored_plate_arguments(Val(A), args...)
    broadcasted = Base.broadcasted(tuple, wrapped...)
    isempty(axes(broadcasted)) && throw(ArgumentError(
        "an authored plate requires at least one non-Ref batched argument"))
    # Instantiation performs Julia's ordinary broadcast-axis compatibility
    # check before any scalar endpoint recipe or output mutation executes.
    Base.Broadcast.instantiate(broadcasted)
end

function _authored_plate_marker(::Val{A}, args...) where {A}
    index = findfirst(eachindex(args)) do position
        !(position in A) && _authored_plate_is_axis(getfield(args, position))
    end
    index === nothing && throw(ArgumentError(
        "an authored plate requires at least one batched argument"))
    getfield(args, index)
end
@inline _authored_plate_zero(::Type{T}, marker) where {T} = zero(T)
@inline _authored_plate_zero(::Type{Any}, marker) = zero(eltype(marker))

# The pointwise buffer's container type and shape come from the axis marker. For
# an array marker `similar(marker, T, output_axes)` is the single-allocation fast
# path (unchanged). A `Tuple` axis marker — a tuple-valued batched argument, or a
# plate whose sole axis is a tuple — has no `similar(::Tuple, T, axes)` method, so
# the native/nonallocating pointwise allocation used to crash on it even though
# the Reactant path materialized it. Julia's broadcast rule collapses a `Tuple`
# against arrays (or a lone `Tuple`) to a plain `Array`, so allocate the `Array`
# of the combined output shape directly, matching the container both `_plate_similar`
# (style-combining) and the Reactant oracle already produce for a tuple axis.
# (`_plate_similar` alone does not cover a plate whose SOLE axis is a tuple — a
# lone `Style{Tuple}` would materialize a tuple, not an array — so this marker
# dispatch is the general fix. The name retains `similar` so the lowering's
# materialization proxy in `test_authored_plate.jl` still matches.)
@inline _plate_similar_output(marker, ::Type{T}, output_axes) where {T} =
    similar(marker, T, output_axes)
@inline _plate_similar_output(marker::Tuple, ::Type{T}, output_axes) where {T} =
    similar(Array{T}, output_axes)

# An untyped plate cell exposes its result through a metadata-`Any` boundary
# port (type annotations are optional metadata in an authored kernel), so the
# native lowering allocates its pointwise container as a boxed `Vector{Any}` —
# even when every cell yields the same concrete type. That boxed container is
# not promotable at the Reactant host-operand boundary
# (`unwrapped_eltype(::Type{Any})` has no method), so a transformed-data plate
# over BOUND data (evaluated natively, then handed to a traced likelihood)
# breaks an otherwise-tracing posterior. Narrow the filled container once to its
# concrete element type, mirroring `map(identity, ·)`: a homogeneous cell
# collapses to `Vector{Float64}`, while a genuinely heterogeneous cell keeps its
# wide container. Reached only when the boundary type is `Any`; a typed cell
# keeps the original single-allocation fast path untouched.
@inline _narrow_plate_output(x::AbstractArray) = _narrow_plate_output(x, eltype(x))
@inline _narrow_plate_output(x::AbstractArray, ::Type) = x
function _narrow_plate_output(x::AbstractArray, ::Type{Any})
    isempty(x) && return x
    narrowed = mapreduce(typeof, Base.promote_typejoin, x)
    narrowed === Any && return x
    result = similar(x, narrowed)
    copyto!(result, x)
    result
end

# Concrete element type of an authored plate's pointwise result, inferred from
# the scalar body kernel over the per-coordinate argument types. Each argument is
# projected to what the scalar body actually receives per coordinate: an
# atomic/`Ref` or `Number` argument whole, an axis argument by its `eltype`.
# `Base.promote_op` over the whole (nested) plate body kernel narrows to that
# concrete type at ordinary call sites; only inside a RuntimeGeneratedFunction
# does it fail to const-fold, so the generated native lowering derives the same
# type through the individual recipe ops instead. This types the pointwise buffer
# and total accumulator UP FRONT — no boxed `Vector{Any}` is ever built and the
# accumulator seed matches the summed cells (`_narrow_plate_output` above is a
# post-hoc narrowing that still allocates the boxed container first and leaves the
# accumulator untyped).
#
# When `promote_op` cannot pin a CONCRETE type — a genuinely uninferrable body,
# but also a STATIC call over an UNTYPED batched port, whose projected element
# type is `Any` (`_plate_cache_slot`/`_plate_result_type` call this with the
# plan-level HAVE types, where an unannotated port is `Any`) — fall back to the
# scalar body kernel's DECLARED output valtype. For a typed body (`::Float64`)
# that valtype is authoritative and concrete, so the nonallocating cache slot is
# seeded `Array{Float64}` instead of a boxed `Array{Any}` that would re-box every
# element through `broadcast!` on each call (the `want=:pointwise` regression:
# 16 KB/call for a length-1000 untyped plate whose cells are Float64). A body
# with no concrete declared type stays `Any`, where `_narrow_plate_output` still
# recovers a homogeneous element type at runtime.
function _authored_plate_result_eltype(op::_AuthoredPlateOp{K,A},
                                       argtypes) where {K,A}
    element_types = ntuple(length(argtypes)) do position
        argtype = argtypes[position]
        (position in A || argtype <: Number) ? argtype : eltype(argtype)
    end
    T = Base.promote_op(op.kernel, element_types...)
    isconcretetype(T) && return T
    declared = valtype(only(outputs(op.kernel)))
    isconcretetype(declared) && return declared
    T === Union{} ? declared : T
end

# A plate is a pure graph map/reduction. Once Julia has instantiated the
# broadcast axes, a recipe only needs to run again when a dimension kept by
# one of its transitive HAVE roots changes in Cartesian iteration order. The
# cached value is a scalar local, never an axis-sized intermediate.
@inline _plate_dependency_changed(index, previous,
                                  arg::Union{Number,Ref}) = false

@inline function _plate_dependency_changed(
        index, previous, arg::Base.Broadcast.Extruded)
    @inbounds for dimension in eachindex(arg.keeps)
        arg.keeps[dimension] && index[dimension] != previous[dimension] &&
            return true
    end
    false
end

@inline function _plate_dependency_changed(index, previous, arg)
    argument_axes = axes(arg)
    @inbounds for dimension in eachindex(argument_axes)
        length(argument_axes[dimension]) == 1 && continue
        index[dimension] != previous[dimension] && return true
    end
    false
end

function _plate_similar(arguments::Tuple, ::Type{T}, output_axes) where {T}
    style = Base.Broadcast.combine_styles(arguments...)
    broadcasted = Base.Broadcast.Broadcasted(
        style, identity, arguments, output_axes)
    similar(broadcasted, T)
end

@inline function _plate_require_axes(output_axes)
    isempty(output_axes) && throw(ArgumentError(
        "a plate requires at least one batched broadcast axis"))
    output_axes
end

function (op::_AuthoredPlateOp{K,A})(args...) where {K,A}
    batch = _authored_plate_broadcast(Val(A), args...)
    marker = _authored_plate_marker(Val(A), args...)
    # Type the buffer up front from the inferred pointwise element type, so a
    # homogeneous untyped cell fills a `Vector{Float64}` directly instead of a
    # boxed `Vector{Any}`.
    result = _plate_similar_output(marker,
                                   _authored_plate_result_eltype(op, map(typeof, args)),
                                   axes(batch))
    for index in eachindex(batch)
        scalar_args = batch[index]
        result[index] = op.kernel(scalar_args...)
    end
    # Fallback runtime narrowing when the element type was genuinely
    # uninferrable (`valtype(output) === Any` and `promote_op` could not narrow):
    # a no-op for the typed buffer above, recovering a homogeneous element type
    # otherwise so the result stays promotable at the Reactant host boundary.
    _narrow_plate_output(result)
end

# A first-class authored SEQUENTIAL scan.  Unlike `plate` (a pure broadcast map),
# it threads a carry through an ordered loop; its scalar step is a 2-`want` kernel
# `(carry, x, shared...) -> (new_carry, output)`. Native lowering inlines that
# step and can stream its output into a reducing plate. The tensorized product
# calls `_tensorized_scan`, specialized by a backend to a `stablehlo.while`.
# The op's argument order is fixed by authoring: index 1 = carry seed, index 2 =
# the sequence `xs`, index 3+ = Ref-shared operands; `A` records the atomic
# (broadcast-invariant) indices {1, 3, 4, …}.
struct _AuthoredScanOp{K,A}
    kernel::K
end

"The transparent scalar step plan captured by an authored `scan(...) do` recipe."
function scan_body(recipe::Recipe)
    recipe.op isa _AuthoredScanOp || throw(ArgumentError(
        "recipe $(recipe.id) is not an authored scan"))
    recipe.op.kernel.plan
end

function (op::_AuthoredScanOp{K,A})(args...) where {K,A}
    length(args) >= 2 || throw(ArgumentError(
        "an authored scan expects (init, xs, shared...) arguments"))
    _tensorized_scan(op.kernel, args[2], args[1], args[3:end]...)
end

function _lhs_symbols!(symbols::Set{Symbol}, lhs)
    lhs isa Symbol && return push!(symbols, lhs)
    lhs isa Expr || return symbols
    if lhs.head === :(::)
        _lhs_symbols!(symbols, lhs.args[1])
    elseif lhs.head === :tuple
        foreach(item -> _lhs_symbols!(symbols, item), lhs.args)
    end
    symbols
end

function _local_symbols!(symbols::Set{Symbol}, node)
    node isa Expr || return symbols
    node.head === :quote && return symbols
    if node.head === :(=)
        _lhs_symbols!(symbols, node.args[1])
    elseif node.head === :for
        iteration = node.args[1]
        iteration isa Expr && iteration.head === :(=) &&
            _lhs_symbols!(symbols, iteration.args[1])
    end
    foreach(child -> _local_symbols!(symbols, child), node.args)
    symbols
end

_signature_name(name::Symbol) = name
_signature_name(annotation::Expr) = annotation.head === :(::) ?
    _signature_name(annotation.args[1]) : throw(ArgumentError(
        "embedded kernel has an unsupported argument form: $annotation"))

function _rewrite_embedded(node, names::Dict{Symbol,Any}, op_offset::Int)
    node isa Symbol && return get(names, node, node)
    node isa Expr || return node
    node.head === :quote && return node
    if node.head === :ref && length(node.args) == 2 &&
       node.args[1] === _OPS_ARG && node.args[2] isa Int
        return Expr(:ref, _OPS_ARG, op_offset + node.args[2])
    end
    Expr(node.head,
         (_rewrite_embedded(child, names, op_offset) for child in node.args)...)
end

function _embedded_statements(ast::Expr, callargs, lhs, op_offset::Int)
    ast.head === :function || throw(ArgumentError(
        "nested preparation requires an RK-generated function expression"))
    signature, body = ast.args
    signature isa Expr && signature.head === :tuple || throw(ArgumentError(
        "embedded kernel has an unsupported signature"))
    params = signature.args[2:end]
    length(params) == length(callargs) || throw(ArgumentError(
        "embedded kernel expected $(length(params)) inputs, got $(length(callargs))"))

    names = Dict{Symbol,Any}()
    for (param, callarg) in zip(params, callargs)
        names[_signature_name(param)] = callarg
    end
    locals = _local_symbols!(Set{Symbol}(), body)
    for local_name in locals
        haskey(names, local_name) ||
            (names[local_name] = gensym(Symbol(:embedded_, local_name)))
    end

    statements = Any[]
    bodyargs = body.args
    return_index = findlast(statement ->
        statement isa Expr && statement.head === :return, bodyargs)
    return_index === nothing && throw(ArgumentError(
        "embedded RK kernel has no terminal return"))
    any(statement -> statement isa Expr && statement.head === :return,
        bodyargs[1:(return_index - 1)]) && throw(ArgumentError(
            "embedded RK kernel has an early return"))
    for statement in bodyargs[1:(return_index - 1)]
        statement isa LineNumberNode && continue
        push!(statements, _rewrite_embedded(statement, names, op_offset))
    end
    returned = _rewrite_embedded(
        bodyargs[return_index].args[1], names, op_offset)
    push!(statements, Expr(:(=), lhs, returned))
    statements
end

function _authored_plate_sum_recipe(p::Plan, plate_recipe::Recipe)
    pointwise = only(plate_recipe.outputs)
    pointwise_id = canon_id(p.graph, pointwise.id)
    matches = Recipe[]
    for recipe in p.recipes
        recipe === plate_recipe && continue
        recipe.effectful && continue
        length(recipe.inputs) == 1 || continue
        canon_id(p.graph, only(recipe.inputs).id) == pointwise_id || continue
        source = recipe.source
        source isa Expr && source.head === :call && length(source.args) == 2 || continue
        callee = source.args[1]
        (callee === :sum || callee === GlobalRef(Base, :sum)) || continue
        source.args[2] === pointwise.name || continue
        (recipe.op === sum || recipe.op isa _KernelSourceOp) || continue
        push!(matches, recipe)
    end
    length(matches) <= 1 || throw(ArgumentError(
        "an authored plate pointwise port has more than one selected sum consumer"))
    isempty(matches) ? nothing : only(matches)
end

# Inline the scalar step into the ordered native loop. Calling the nested
# PreparedKernel from a loop with a changing carry defeats inference across the
# RGF boundary; the same step AST and operation table specialize normally here.
function _lower_authored_scan_native!(body, op::_AuthoredScanOp, callargs, lhs,
                                      offset; consumer = nothing)
    step = op.kernel
    indices, index, carry, output = gensym.((:scan_indices, :scan_index,
                                            :scan_carry, :scan_output))
    xs = callargs[2]
    arguments = Any[carry, Expr(:ref, xs, index), callargs[3:end]...]
    step_body = Expr(:block, _embedded_statements(
        step.ast, arguments, Expr(:tuple, carry, output), offset)...)
    invariant = Any[]
    initial_output, loop_output, final_output = Expr(:block), Expr(:block), Expr(:block)
    if lhs !== nothing
        push!(initial_output.args, :($lhs = $(GlobalRef(Base, :similar))(
            $xs, $(GlobalRef(Base, :typeof))($output))))
        push!(initial_output.args, :($lhs[$index] = $output))
        push!(loop_output.args, :($lhs[$index] = $output))
    end
    if consumer !== nothing
        cell, args, positions, cell_offset, pointwise_lhs, total_lhs = consumer
        cell_output = gensym(:scan_cell)
        cell_args = Any[i in positions ? output : arg for (i, arg) in enumerate(args)]
        invariant, dynamic, cell_type = _authored_scan_cell_statements(
            cell, cell_args, positions, cell_output, cell_offset)
        append!(step_body.args, dynamic)
        # Match ordinary plate lowering's inferred result type, including
        # heterogeneous cells; the first value alone cannot type that buffer.
        plate_eltype = gensym(:scan_plate_eltype)
        push!(initial_output.args, :($plate_eltype = $cell_type))
        if pointwise_lhs !== nothing
            push!(initial_output.args, :($pointwise_lhs = $(GlobalRef(Base, :similar))(
                $xs, $plate_eltype)))
            push!(initial_output.args, :($pointwise_lhs[$index] = $cell_output))
            push!(loop_output.args, :($pointwise_lhs[$index] = $cell_output))
            push!(final_output.args, :($pointwise_lhs =
                $(GlobalRef(@__MODULE__, :_narrow_plate_output))($pointwise_lhs)))
        end
        push!(initial_output.args,
              :($total_lhs = $(GlobalRef(Base, :zero))(
                  $plate_eltype === Any ? $(GlobalRef(Base, :typeof))($output) :
                  $plate_eltype) + $cell_output))
        push!(loop_output.args, :($total_lhs = $total_lhs + $cell_output))
    end
    push!(body.args, :($indices = $(GlobalRef(Base, :eachindex))($xs)))
    push!(body.args, :($(GlobalRef(Base, :isempty))($indices) &&
        throw(ArgumentError("scan requires a non-empty sequence"))))
    append!(body.args, invariant)
    push!(body.args, :($carry = $(callargs[1])))
    push!(body.args, :($index = $(GlobalRef(Base, :first))($indices)))
    append!(body.args, step_body.args)
    append!(body.args, initial_output.args)
    push!(body.args, Expr(:for,
        :($index = $(GlobalRef(Base.Iterators, :drop))($indices, 1)),
        Expr(:block, step_body.args..., loop_output.args...)))
    append!(body.args, final_output.args)
    body
end

function _authored_scan_cell_statements(cell, args, positions, output, offset)
    inner = cell.plan
    length(cell.ops) == length(inner.recipes) || throw(ArgumentError(
        "an authored plate body must lower to one operation per transparent scalar recipe"))
    roots = Set(canon_id(inner.graph, inner.have[i].id) for i in positions)
    dependencies = _plate_dependencies(inner, roots).recipes
    locals = Dict(canon_id(inner.graph, v.id) => arg
                  for (v, arg) in zip(inner.have, args))
    types = Dict{Int,Any}(canon_id(inner.graph, v.id) =>
        Expr(:call, GlobalRef(Base, :typeof), arg) for (v, arg) in zip(inner.have, args))
    invariant, dynamic = Any[], Any[]
    for (i, recipe) in enumerate(inner.recipes)
        length(recipe.outputs) == 1 || throw(ArgumentError(
            "an authored plate currently requires single-output scalar recipes"))
        value = only(recipe.outputs)
        result = gensym(Symbol(:scan_cell_, value.name))
        inputs = Any[locals[canon_id(inner.graph, v.id)] for v in recipe.inputs]
        statement = :($result = $(_OPS_ARG)[$(offset + i)]($(inputs...)))
        push!(isempty(dependencies[i]) ? invariant : dynamic, statement)
        locals[canon_id(inner.graph, value.id)] = result
        input_types = Any[types[canon_id(inner.graph, v.id)] for v in recipe.inputs]
        types[canon_id(inner.graph, value.id)] = Expr(:call,
            GlobalRef(Base, :promote_op), Expr(:ref, _OPS_ARG, offset + i), input_types...)
    end
    push!(dynamic, :($output = $(locals[canon_id(inner.graph, only(inner.want).id)])))
    invariant, dynamic, types[canon_id(inner.graph, only(inner.want).id)]
end

# Fuse only a sole scalar plate consumer with a selected sum. Other array
# operands need full broadcast-axis validation; other consumers or WANTs need
# the materialized scan. In either case ordinary native scan lowering applies.
function _authored_scan_sum_consumer(p::Plan, scan_recipe::Recipe)
    output_id = canon_id(p.graph, only(scan_recipe.outputs).id)
    any(w -> canon_id(p.graph, w.id) == output_id, p.want) && return nothing
    consumers = filter(p.recipes) do recipe
        any(v -> canon_id(p.graph, v.id) == output_id, recipe.inputs)
    end
    length(consumers) == 1 || return nothing
    consumer = only(consumers)
    consumer.op isa _AuthoredPlateOp || return nothing
    # Composed plate chains retain separate broadcast-domain checks. Let their
    # ordinary lowering validate those domains against a materialized scan.
    isempty(consumer.op.axis_checks) || return nothing
    _authored_plate_sum_recipe(p, consumer) === nothing && return nothing
    atomic = typeof(consumer.op).parameters[2]
    for (i, input) in enumerate(consumer.inputs)
        if canon_id(p.graph, input.id) == output_id
            i in atomic && return nothing # Ref(errors) consumes the whole vector.
        elseif !(i in atomic || valtype(input) <: Number)
            return nothing
        end
    end
    consumer
end

# Compose selected scalar DAGs only at code generation. The public graph and
# Plan retain every named array node, so another WANT/HAVE query can still
# materialize it or cut the graph there. A whole-array/Ref use is a boundary,
# as is any additional selected consumer; no source expression is rewritten.
function _compose_authored_plates(g::Graph, producer::Recipe, consumer::Recipe)
    scalar_graph = Graph()
    callvalues = Value[]
    scalar_have = Value[]
    atomic = Int[]
    positions = Dict{Tuple{Int,Bool},Int}()
    checks = Tuple[]
    readable_recipes = Dict{Int,Recipe}()
    producer_id = canon_id(g, only(producer.outputs).id)

    function append_body(recipe, replacement = nothing)
        op = recipe.op
        inner = op.kernel.plan
        old_atomic = typeof(op).parameters[2]
        mapped = Dict{Int,Value}()
        input_positions = Vector{Tuple}(undef, length(recipe.inputs))
        for (index, (outer, input)) in enumerate(zip(recipe.inputs, inner.have))
            cid = canon_id(g, outer.id)
            if replacement !== nothing && cid == producer_id
                mapped[canon_id(inner.graph, input.id)] = replacement.value
                input_positions[index] = replacement.axes
                continue
            end
            key = (cid, index in old_atomic)
            position = get!(positions, key) do
                push!(callvalues, outer)
                push!(scalar_have, value!(scalar_graph, input.name, valtype(input)))
                last(key) && push!(atomic, length(callvalues))
                length(callvalues)
            end
            mapped[canon_id(inner.graph, input.id)] = scalar_have[position]
            input_positions[index] = (position,)
        end
        # Preserve all absorbed domain checks, including singleton/empty axes
        # and unused formals. Flattening only the scalar dependencies loses them.
        remap(group) = Tuple(unique(Int[
            position for index in group for position in input_positions[index]]))
        append!(checks, (remap(group) for group in op.axis_checks))
        axes = remap(Tuple(index for index in eachindex(recipe.inputs)
                           if !(index in old_atomic)))
        push!(checks, axes)
        for (recipe_index, r) in enumerate(inner.recipes)
            ins = Tuple(mapped[canon_id(inner.graph, v.id)] for v in r.inputs)
            outs = Tuple(get!(mapped, canon_id(inner.graph, v.id)) do
                value!(scalar_graph, v.name, valtype(v))
            end for v in r.outputs)
            copied = add!(scalar_graph, ins => outs, r.op; cost = r.cost,
                          effectful = r.effectful, source = r.source)
            readable_recipes[copied.id] = op.kernel.lowered_recipes[recipe_index]
        end
        (; value = mapped[canon_id(inner.graph, only(inner.want).id)], axes)
    end

    intermediate = append_body(producer)
    result = append_body(consumer, intermediate)
    # Keep the complete HAVE boundary, including unused axis arguments.
    scalar_plan = plan(scalar_graph; have = scalar_have, want = (result.value,))
    kernel = prepare(scalar_plan)
    # Source RHSs use the original scalar formal names. Keep their original
    # recipe metadata in operation-table order so readable code binds those
    # names to the newly projected scalar arguments, including across chains.
    kernel = PreparedKernel(kernel.f, kernel.ops, kernel.inputs, kernel.outputs,
        kernel.plan, kernel.ast,
        Tuple(readable_recipes[r.id] for r in scalar_plan.recipes))
    op = _AuthoredPlateOp{typeof(kernel),Tuple(atomic)}(kernel, Tuple(unique(checks)))
    Recipe(consumer.id, Tuple(callvalues), consumer.outputs, op,
           producer.cost + consumer.cost, nothing, false, consumer.source)
end

function _fuse_authored_plate_chains(p::Plan)
    recipes = Union{Nothing,Recipe}[p.recipes...]
    boundary = Set(canon_id(p.graph, v.id) for v in (p.have..., p.want...))
    changed = false
    for index in eachindex(recipes)
        producer = recipes[index]
        producer === nothing && continue
        producer.op isa _AuthoredPlateOp || continue
        length(producer.outputs) == 1 || continue
        cid = canon_id(p.graph, only(producer.outputs).id)
        cid in boundary && continue
        consumers = findall(recipes) do candidate
            candidate === nothing && return false
            any(input -> canon_id(p.graph, input.id) == cid, candidate.inputs)
        end
        length(consumers) == 1 || continue
        consumer_index = only(consumers)
        consumer_index > index || continue
        consumer = recipes[consumer_index]
        consumer.op isa _AuthoredPlateOp || continue
        consumer_atomic = typeof(consumer.op).parameters[2]
        any(position -> position in consumer_atomic &&
            canon_id(p.graph, consumer.inputs[position].id) == cid,
            eachindex(consumer.inputs)) && continue
        recipes[consumer_index] = _compose_authored_plates(p.graph, producer, consumer)
        recipes[index] = nothing
        changed = true
    end
    changed || return p
    selected = Recipe[r for r in recipes if r !== nothing]
    producer = Dict(canon_id(p.graph, v.id) => r for r in selected for v in r.outputs)
    Plan(p.graph, p.have, p.want, selected, producer, p.cost, p.candidates)
end

function _plate_dependencies(plan::Plan, root_ids::Set{Int})
    graph = plan.graph
    dependencies = Dict{Int,Set{Int}}()
    for input in plan.have
        cid = canon_id(graph, input.id)
        dependencies[cid] = cid in root_ids ? Set((cid,)) : Set{Int}()
    end
    recipe_dependencies = Vector{Set{Int}}(undef, length(plan.recipes))
    for (recipe_index, recipe) in enumerate(plan.recipes)
        roots = Set{Int}()
        for input in recipe.inputs
            union!(roots, get(dependencies, canon_id(graph, input.id), Set{Int}()))
        end
        recipe_dependencies[recipe_index] = roots
        for output in recipe.outputs
            cid = canon_id(graph, output.id)
            haskey(dependencies, cid) || (dependencies[cid] = copy(roots))
        end
    end
    (; values = dependencies, recipes = recipe_dependencies)
end

function _authored_plate_condition(callargs, roots, positions, atomic)
    tests = Any[Expr(:call, GlobalRef(@__MODULE__, :_authored_plate_is_axis),
                     callargs[positions[root]]) for root in sort!(collect(roots))
                if !(positions[root] in atomic)]
    isempty(tests) && return false
    foldl((left, right) -> Expr(:||, left, right), tests)
end

function _authored_plate_changed(callargs, roots, positions, atomic,
                                 index, previous)
    tests = Any[
        Expr(:call, GlobalRef(@__MODULE__, :_plate_dependency_changed),
             index, previous, callargs[positions[root]])
        for root in sort!(collect(roots)) if !(positions[root] in atomic)
    ]
    isempty(tests) && return false
    foldl((left, right) -> Expr(:||, left, right), tests)
end

function _authored_plate_recipe_groups(inner::Plan, dependencies,
                                       positions, atomic, callvalues)
    groups = Tuple{Set{Int},Vector{Int}}[]
    for (recipe_index, recipe) in enumerate(inner.recipes)
        length(recipe.outputs) == 1 || throw(ArgumentError(
            "an authored plate currently requires single-output scalar recipes"))
        roots = get(dependencies,
                    canon_id(inner.graph, only(recipe.outputs).id), Set{Int}())
        dynamic = Set(root for root in roots
                      if !(positions[root] in atomic) &&
                         !(valtype(callvalues[positions[root]]) <: Number))
        if !isempty(groups) && first(last(groups)) == dynamic
            push!(last(groups)[2], recipe_index)
        else
            push!(groups, (dynamic, [recipe_index]))
        end
    end
    groups
end

# CartesianIndices advances its first dimension at every coordinate. A recipe
# with any one-dimensional array/tuple root can therefore be evaluated
# unconditionally in the scalar loop: even when that root is singleton-expanded,
# recomputation is semantically identical under the plate's pure-recipe
# contract. Keeping this decision in generated code removes loop-carried
# scheduler control from the common vector-plate kernel while retaining the
# dependency scheduler for higher-dimensional partial-axis broadcasts.
function _authored_plate_unconditional_group(
        roots, positions, atomic, callvalues)
    any(roots) do root
        position = positions[root]
        position in atomic && return false
        type = valtype(callvalues[position])
        type <: AbstractVector || type <: Tuple
    end
end

function _authored_plate_scalar_ref(inner::Plan, locals, callargs,
                                    callvalues, prepared_arguments, atomic,
                                    input::Value, index, looped::Bool)
    graph = inner.graph
    cid = canon_id(graph, input.id)
    have_index = findfirst(value -> canon_id(graph, value.id) == cid, inner.have)
    if have_index !== nothing
        arg = callargs[have_index]
        # Numbers and explicit `Ref` arguments are statically scalar. Keep them
        # as ordinary loop invariants instead of routing them through a
        # broadcast wrapper and indexed projection on every coordinate.
        statically_scalar = have_index in atomic ||
                            valtype(callvalues[have_index]) <: Number
        return looped && !statically_scalar ?
            Expr(:call, GlobalRef(Base.Broadcast, :_broadcast_getindex),
                 prepared_arguments[have_index], index) : arg
    end
    locals[cid]
end

function _lower_authored_plate_native!(body, runtime_ops, runtime_recipes,
                                       op::_AuthoredPlateOp, callargs, callvalues,
                                       pointwise_lhs, total_lhs)
    inner_kernel = op.kernel
    inner = inner_kernel.plan
    length(inner.want) == 1 || throw(ArgumentError(
        "an authored plate body must have exactly one distinguished result"))
    length(inner_kernel.ops) == length(inner.recipes) || throw(ArgumentError(
        "an authored plate body must lower to one operation per transparent scalar recipe"))

    atomic = typeof(op).parameters[2]
    # The distinguished result's element type governs both the materialized
    # pointwise buffer and the total accumulator seed. The plan-level `valtype`
    # is `Any` whenever the plate body's scalar result carries no explicit
    # annotation (a bare arithmetic cell, an unannotated `plate(x) do e; f(e) end`);
    # baking that literal `Any` into `similar`/`zero` allocates a boxed
    # `Vector{Any}` and seeds the accumulator with a type that disagrees with the
    # summed cells, both of which defeat scalar replacement and reverse-mode AD (a
    # `Vector{Int}` axis then seeds `zero(eltype(marker)) = zero(Int)` against
    # `Float64` cells, producing a `Union` accumulator Enzyme rejects). Recover the
    # concrete element type by inferring the scalar body kernel over the actual
    # per-coordinate argument types (`plate_eltype`, bound below), deferring to the
    # runtime `eltype(marker)` seed only when the body is genuinely uninferrable.
    # (`_narrow_plate_output` after the loop still recovers a homogeneous element
    # type at runtime in that residual `Any` case.)
    plate_eltype = gensym(:plate_eltype)
    needs_marker = pointwise_lhs !== nothing || total_lhs !== nothing
    marker = gensym(:plate_axis)
    index = gensym(:plate_index)
    previous = gensym(:plate_previous)
    first_coordinate = gensym(:plate_first)
    atomic_val = Expr(:call, GlobalRef(Base, :Val), QuoteNode(atomic))
    if needs_marker
        # The marker supplies only the pointwise buffer's container type/shape and
        # the `Any`-fallback element type — never a differentiated value. Emitting
        # a runtime `_authored_plate_marker(Val(atomic), callargs...)` here (a
        # `findfirst` over the whole argument tuple) makes the selected marker
        # CONDITIONALLY active whenever a data-only `Vector{Int}` axis sits beside
        # an active/`Constant` `Vector{Float64}` — the posteriordb M0/Mh/seeds
        # shape — which reverse-mode Enzyme rejects with an
        # `EnzymeRuntimeActivityError` under the standing static-activity config
        # (no `set_runtime_activity`). So bind the marker to a SPECIFIC argument at
        # lowering time — but only when the port value types PROVE it is the same
        # axis the runtime `findfirst` would pick. Walk the non-atomic positions in
        # authored order: skip provable non-axes (scalars, rank-0 arrays), take the
        # first provable axis, and FALL BACK to the runtime marker the moment a
        # preceding candidate is `:ambiguous` (a metadata-`Any` port that may hold a
        # runtime scalar, an abstract array type, a non-axis struct) — because the
        # runtime `_authored_plate_is_axis` could then skip it and select a later
        # argument, and a static pick that is not provably the same axis would build
        # the pointwise buffer from the wrong argument (e.g. a scalar `Any` port).
        axis_position = nothing
        for position in eachindex(callargs)
            position in atomic && continue
            class = _static_plate_axis_class(valtype(callvalues[position]))
            class === :not_axis && continue
            class === :axis && (axis_position = position)
            break
        end
        marker_source = axis_position === nothing ?
            :($(GlobalRef(@__MODULE__, :_authored_plate_marker))(
                $atomic_val, $(callargs...))) :
            callargs[axis_position]
        push!(body.args, :($marker = $marker_source))
    end

    # Reuse Base's ordinary broadcast preparation one argument at a time. A
    # composite `Broadcasted(tuple, ...)` is convenient for primal execution,
    # but carrying that tuple-producing expression through reverse AD leaves a
    # much larger derivative loop. `broadcastable` + `preprocess` preserves
    # scalar, Ref, singleton-expansion, and custom-axis semantics while letting
    # the lowered scalar recipes consume only the projected values they need.
    raw_arguments = Union{Nothing,Symbol}[]
    prepared_arguments = Union{Nothing,Symbol}[]
    for (position, arg) in enumerate(callargs)
        statically_scalar = position in atomic ||
                            valtype(callvalues[position]) <: Number
        if statically_scalar
            push!(raw_arguments, nothing)
            push!(prepared_arguments, nothing)
            continue
        end
        raw = gensym(:plate_argument)
        prepared = gensym(:plate_prepared)
        wrapped = Expr(:call, GlobalRef(Base, :broadcastable), arg)
        push!(body.args, Expr(:(=), raw, wrapped))
        push!(body.args, Expr(:(=), prepared,
            Expr(:call, GlobalRef(Base.Broadcast, :preprocess), nothing, raw)))
        push!(raw_arguments, raw)
        push!(prepared_arguments, prepared)
    end
    output_axes = gensym(:plate_axes)
    # A fused plate chain absorbs one axis-check group per sub-plate
    # (`_compose_authored_plates`). Emitting every one added extra pre-loop
    # `_plate_require_axes(combine_axes(...))` calls that an equivalent single
    # `plate` never has, measurably slowing the primal (~1.56× on the snag
    # `composed-authore` bakeoff). Skip a group ONLY when the always-emitted
    # `combined_axes` check below fully subsumes it — both conditions required:
    #   (a) its operands ⊆ the combined operand set, so broadcast-compatibility
    #       is already validated (the superset's `combine_axes` throws the same
    #       `DimensionMismatch`); AND
    #   (b) at least one operand is a STATICALLY provable ≥1-dim axis
    #       (`_static_plate_axis_class === :axis`), so this sub-plate's own
    #       "at least one batched broadcast axis" guard (`_plate_require_axes`
    #       non-empty) cannot fire.
    # Without (b) an all-scalar sub-plate (a fused producer over untyped ports
    # bound to runtime scalars) would wrongly pass on an axis contributed by
    # ANOTHER sub-plate — the `test_authored_plate` `multi`/`unused`
    # ArgumentError guards. A chain over typed/bound array ports (the common
    # case) meets both and drops the redundant checks, matching the single plate.
    combined_positions = Set(position for position in eachindex(raw_arguments)
                             if raw_arguments[position] !== nothing)
    for group in op.axis_checks
        group_positions = Int[position for position in group
                              if raw_arguments[position] !== nothing]
        # Skip ONLY when the combined check subsumes this group — BOTH required:
        # (a) operands ⊆ the combined operand set, and (b) at least one operand
        # is a statically provable ≥1-dim axis. An EMPTY `group_positions` (every
        # original operand filtered out because it is Ref-atomic or a static
        # `Number`) is NOT skippable: it means this sub-plate has NO batched axis
        # of its own, and the group must still lower to `combine_axes()` ⇒
        # `_plate_require_axes(())` ⇒ `ArgumentError`, exactly as the pre-fusion
        # single plate would. Both `issubset` (empty ⊆ anything) and `any`
        # (`false` over empty) already give the right verdict, so no special-case
        # skip — a `continue` here would let an axis-less producer silently borrow
        # a sibling sub-plate's axis (regression caught in review of the first
        # candidate; `snag composed-authore` negative-domain tests below).
        if issubset(group_positions, combined_positions) &&
           any(p -> _static_plate_axis_class(valtype(callvalues[p])) === :axis,
               group_positions)
            continue
        end
        group_axes = Expr(:call, GlobalRef(Base.Broadcast, :combine_axes),
            (raw_arguments[position] for position in group_positions)...)
        push!(body.args, Expr(:call, GlobalRef(@__MODULE__, :_plate_require_axes),
                              group_axes))
    end
    combined_axes = Expr(:call, GlobalRef(Base.Broadcast, :combine_axes),
                         (arg for arg in raw_arguments if arg !== nothing)...)
    push!(body.args, Expr(:(=), output_axes,
        Expr(:call, GlobalRef(@__MODULE__, :_plate_require_axes), combined_axes)))

    root_positions = Dict(
        canon_id(inner.graph, input.id) => position
        for (position, input) in enumerate(inner.have)
    )
    root_ids = Set(keys(root_positions))
    dependencies = _plate_dependencies(inner, root_ids).values
    locals = Dict{Int,Symbol}()
    for recipe in inner.recipes, output in recipe.outputs
        cid = canon_id(inner.graph, output.id)
        get!(locals, cid) do
            gensym(Symbol(:plate_, output.name))
        end
    end
    op_offset = length(runtime_ops)
    append!(runtime_ops, inner_kernel.ops)
    append!(runtime_recipes, inner_kernel.lowered_recipes)
    # Bind the concrete pointwise element type by propagating inferred types
    # through the plate body's scalar recipe DAG. Each individual `__ops__`
    # recipe op is an ordinary callable (a `_KernelSourceOp`, not the opaque
    # nested `PreparedKernel`), so `Base.promote_op` over it const-folds inside
    # the generated function from the actual per-coordinate argument types Julia
    # infers here — the `Vector{Int}`-axis element / atomic scalar types — rather
    # than the plan-level `Any`. Seeding `promote_op` over the whole plate body
    # kernel instead does NOT fold (nested-RGF inference is opaque) and would
    # inject a runtime inference call. This is a compile-time expression, so it
    # also covers an empty axis with no representative element. A genuinely
    # uninferrable recipe yields `Any`, reproducing the previous `Vector{Any}` /
    # `_authored_plate_zero(Any, marker)` behavior (then narrowed at runtime by
    # `_narrow_plate_output`) rather than regressing.
    plate_type_exprs = Dict{Int,Any}()
    for (position, input) in enumerate(inner.have)
        cid = canon_id(inner.graph, input.id)
        plate_type_exprs[cid] =
            (position in atomic || valtype(callvalues[position]) <: Number) ?
                Expr(:call, GlobalRef(Base, :typeof), callargs[position]) :
                Expr(:call, GlobalRef(Base, :eltype), raw_arguments[position])
    end
    for (recipe_index, recipe) in enumerate(inner.recipes)
        input_type_exprs = Any[
            get(plate_type_exprs, canon_id(inner.graph, input.id),
                GlobalRef(Core, :Any)) for input in recipe.inputs]
        output_cid = canon_id(inner.graph, only(recipe.outputs).id)
        plate_type_exprs[output_cid] = Expr(:call, GlobalRef(Base, :promote_op),
            Expr(:ref, _OPS_ARG, op_offset + recipe_index), input_type_exprs...)
    end
    push!(body.args, Expr(:(=), plate_eltype,
        get(plate_type_exprs, canon_id(inner.graph, only(inner.want).id),
            GlobalRef(Core, :Any))))
    groups = _authored_plate_recipe_groups(
        inner, dependencies, root_positions, atomic, callvalues)
    has_scheduled_groups = any(groups) do (roots, _)
        !isempty(roots) && !_authored_plate_unconditional_group(
            roots, root_positions, atomic, callvalues)
    end

    # Recipes with the same dynamic root set share one scheduling guard. This
    # preserves partial-dimension invariant caching. Common vector-plate groups
    # are known to be safe to recompute at every coordinate and need no guard in
    # either the primal or differentiated native kernel.
    for (roots, recipe_indices) in groups
        _authored_plate_unconditional_group(
            roots, root_positions, atomic, callvalues) && continue
        assignments = Expr(:block)
        for recipe_index in recipe_indices
            recipe = inner.recipes[recipe_index]
            length(recipe.outputs) == 1 || throw(ArgumentError(
                "an authored plate currently requires single-output scalar recipes"))
            output = only(recipe.outputs)
            out = locals[canon_id(inner.graph, output.id)]
            args = Any[_authored_plate_scalar_ref(
                inner, locals, callargs, callvalues, prepared_arguments, atomic,
                input, index, false) for input in recipe.inputs]
            call = Expr(:call, Expr(:ref, _OPS_ARG, op_offset + recipe_index),
                        args...)
            push!(assignments.args, Expr(:(=), out, call))
        end
        condition = _authored_plate_condition(
            callargs, roots, root_positions, atomic)
        push!(body.args, Expr(:if,
            Expr(:call, GlobalRef(Base, :!), condition), assignments))
    end

    if pointwise_lhs !== nothing
        push!(body.args,
            :($pointwise_lhs = $(GlobalRef(@__MODULE__, :_plate_similar_output))(
                $marker, $plate_eltype, $output_axes)))
    end
    accumulator = total_lhs === nothing ? nothing : gensym(:plate_total)
    if accumulator !== nothing
        # `_authored_plate_zero(::Type{T}, marker) = zero(T)` for the inferred
        # concrete `T`, and falls back to `zero(eltype(marker))` only when
        # inference yielded `Any` — never a `zero(Int)` seed against `Float64`
        # cells.
        push!(body.args, Expr(:(=), accumulator,
            Expr(:call, GlobalRef(@__MODULE__, :_authored_plate_zero),
                 plate_eltype, marker)))
    end

    loopbody = Expr(:block)
    for (roots, recipe_indices) in groups
        isempty(roots) && continue
        assignments = Expr(:block)
        for recipe_index in recipe_indices
            recipe = inner.recipes[recipe_index]
            output = only(recipe.outputs)
            out = locals[canon_id(inner.graph, output.id)]
            args = Any[_authored_plate_scalar_ref(
                inner, locals, callargs, callvalues, prepared_arguments, atomic,
                input, index, true) for input in recipe.inputs]
            call = Expr(:call, Expr(:ref, _OPS_ARG, op_offset + recipe_index),
                        args...)
            push!(assignments.args, Expr(:(=), out, call))
        end
        if _authored_plate_unconditional_group(
                roots, root_positions, atomic, callvalues)
            append!(loopbody.args, assignments.args)
        else
            has_axis = _authored_plate_condition(
                callargs, roots, root_positions, atomic)
            changed = _authored_plate_changed(
                callargs, roots, root_positions, atomic, index, previous)
            condition = Expr(:&&, has_axis,
                Expr(:||, first_coordinate, changed))
            push!(loopbody.args, Expr(:if, condition, assignments))
        end
    end
    # The distinguished result is usually a recipe output bound in `locals`, but
    # an identity/passthrough cell (`plate(x) do v; v end`) names an input as its
    # result. That input never appears in `locals`, so reuse the shared scalar
    # projection, which resolves a HAVE input to its per-coordinate reference and
    # otherwise falls back to `locals`.
    scalar_result = _authored_plate_scalar_ref(
        inner, locals, callargs, callvalues, prepared_arguments, atomic,
        only(inner.want), index, true)
    pointwise_lhs === nothing ||
        push!(loopbody.args, :($pointwise_lhs[$index] = $scalar_result))
    accumulator === nothing ||
        push!(loopbody.args, :($accumulator += $scalar_result))
    if has_scheduled_groups
        push!(loopbody.args, :($previous = $index))
        push!(loopbody.args, :($first_coordinate = false))
    end
    iteration = Expr(:call, GlobalRef(Base, :CartesianIndices), output_axes)
    if has_scheduled_groups
        push!(body.args, :($previous = nothing))
        push!(body.args, :($first_coordinate = true))
    end
    push!(body.args, Expr(:for, Expr(:(=), index, iteration), loopbody))
    # Fallback runtime narrowing of the pointwise container. `plate_eltype`
    # already types the buffer up front, so for an inferable body this is a
    # compile-time no-op (`_narrow_plate_output` dispatches on `eltype`); it only
    # does work in the residual case where inference yielded `Any` and the buffer
    # is a boxed `Vector{Any}`, recovering a homogeneous element type so a
    # transformed-data plate over bound data stays promotable at the Reactant
    # host-operand boundary. The total accumulator is already typed via
    # `_authored_plate_zero`, so this touches only the pointwise materialization.
    if pointwise_lhs !== nothing
        push!(body.args, :($pointwise_lhs =
            $(GlobalRef(@__MODULE__, :_narrow_plate_output))($pointwise_lhs)))
    end
    total_lhs === nothing || push!(body.args, :($total_lhs = $accumulator))
    body
end

function _lower_authored_plate_tensorized!(body, runtime_ops, runtime_recipes,
                                           op::_AuthoredPlateOp, callargs,
                                           pointwise_lhs, total_lhs)
    inner_kernel = op.kernel
    inner = inner_kernel.plan
    length(inner.want) == 1 || throw(ArgumentError(
        "an authored plate body must have exactly one distinguished result"))
    length(inner_kernel.ops) == length(inner.recipes) || throw(ArgumentError(
        "an authored plate body must lower to one operation per transparent scalar recipe"))

    atomic = typeof(op).parameters[2]
    locals = Dict{Int,Any}()
    for (index, input) in enumerate(inner.have)
        arg = callargs[index]
        locals[canon_id(inner.graph, input.id)] = index in atomic ?
            Expr(:call, GlobalRef(Base, :Ref), arg) : arg
    end
    op_offset = length(runtime_ops)
    append!(runtime_ops, inner_kernel.ops)
    append!(runtime_recipes, inner_kernel.lowered_recipes)
    for (recipe_index, recipe) in enumerate(inner.recipes)
        length(recipe.outputs) == 1 || throw(ArgumentError(
            "an authored plate currently requires single-output scalar recipes"))
        output = only(recipe.outputs)
        out = gensym(Symbol(:plate_, output.name))
        args = Any[locals[canon_id(inner.graph, input.id)] for input in recipe.inputs]
        operation = Expr(:ref, _OPS_ARG, op_offset + recipe_index)
        call = Expr(:call, GlobalRef(@__MODULE__, :_tensorized_plate_call),
                    operation, args...)
        push!(body.args, Expr(:(=), out, call))
        locals[canon_id(inner.graph, output.id)] = out
    end
    scalar_result = locals[canon_id(inner.graph, only(inner.want).id)]
    materialized = Expr(:call,
        GlobalRef(@__MODULE__, :_tensorized_plate_materialize), scalar_result)
    pointwise_lhs === nothing ||
        push!(body.args, :($pointwise_lhs = $materialized))
    total = Expr(:call,
        GlobalRef(@__MODULE__, :_tensorized_plate_sum), scalar_result)
    total_lhs === nothing || push!(body.args, :($total_lhs = $total))
    body
end

function _lower_with_ops(p::Plan; tensorized::Bool = false,
                         inline_embedded::Bool = true)
    g = p.graph
    names = _varnames(p)
    nm(v) = names[canon_id(g, v.id)]
    argexprs = Any[_OPS_ARG]
    for v in p.have
        push!(argexprs, :($(nm(v))::$(valtype(v))))
    end
    body = Expr(:block)
    runtime_ops = Any[]
    runtime_recipes = Recipe[]
    skipped_recipes = Set{Int}()
    pending_scans = Dict{Int,Any}()
    # HAVE is authoritative, and the first selected producer of any other
    # logical value owns its binding. Later recipes may emit that value as a
    # collateral multi-output; execute the recipe but discard the duplicate so
    # neither authoritative inputs nor earlier logical values are overwritten.
    assigned = Set(canon_id(g, v.id) for v in p.have)
    for r in p.recipes
        r.id in skipped_recipes && continue
        callargs = Any[nm(inp) for inp in r.inputs]

        if inline_embedded && r.op isa _AuthoredPlateOp
            length(r.outputs) == 1 || throw(ArgumentError(
                "an authored plate recipe must have exactly one pointwise output"))
            pointwise = only(r.outputs)
            pointwise_id = canon_id(g, pointwise.id)
            sum_recipe = _authored_plate_sum_recipe(p, r)
            pointwise_needed = pointwise_id in Set(canon_id(g, w.id) for w in p.want) ||
                any(candidate -> candidate !== sum_recipe &&
                    any(input -> canon_id(g, input.id) == pointwise_id,
                        candidate.inputs), p.recipes)
            pointwise_lhs = pointwise_needed ? nm(pointwise) : nothing
            total_lhs = sum_recipe === nothing ? nothing : nm(only(sum_recipe.outputs))
            pointwise_lhs === nothing || push!(assigned, pointwise_id)
            if sum_recipe !== nothing
                push!(assigned, canon_id(g, only(sum_recipe.outputs).id))
                push!(skipped_recipes, sum_recipe.id)
            end
            if !tensorized && haskey(pending_scans, r.id)
                scan_recipe, scan_args, scan_offset = pending_scans[r.id]
                output_id = canon_id(g, only(scan_recipe.outputs).id)
                positions = findall(
                    v -> canon_id(g, v.id) == output_id, r.inputs)
                cell_offset = length(runtime_ops)
                append!(runtime_ops, r.op.kernel.ops)
                append!(runtime_recipes, r.op.kernel.lowered_recipes)
                consumer = (r.op.kernel, callargs, positions, cell_offset,
                            pointwise_lhs, total_lhs)
                _lower_authored_scan_native!(
                    body, scan_recipe.op, scan_args, nothing, scan_offset; consumer)
            elseif tensorized
                _lower_authored_plate_tensorized!(
                    body, runtime_ops, runtime_recipes, r.op, callargs,
                    pointwise_lhs, total_lhs)
            else
                _lower_authored_plate_native!(
                    body, runtime_ops, runtime_recipes, r.op, callargs, r.inputs,
                    pointwise_lhs, total_lhs)
            end
            continue
        end

        lhsnames = Any[]
        for output in r.outputs
            cid = canon_id(g, output.id)
            if cid in assigned
                push!(lhsnames, gensym(Symbol(nm(output), :_discard)))
            else
                push!(assigned, cid)
                push!(lhsnames, nm(output))
            end
        end
        lhs = length(lhsnames) == 1 ? only(lhsnames) : Expr(:tuple, lhsnames...)
        if inline_embedded && r.op isa _AuthoredScanOp
            # Both products keep the same table: the traced path calls the scan
            # op, while the native path calls its inlined scalar operations.
            push!(runtime_ops, r.op)
            push!(runtime_recipes, r)
            scan_index = length(runtime_ops)
            append!(runtime_ops, r.op.kernel.ops)
            append!(runtime_recipes, r.op.kernel.lowered_recipes)
            if tensorized
                call = Expr(:call, Expr(:ref, _OPS_ARG, scan_index),
                            callargs...)
                push!(body.args, Expr(:(=), lhs, call))
            else
                consumer = _authored_scan_sum_consumer(p, r)
                if consumer === nothing
                    _lower_authored_scan_native!(body, r.op, callargs, lhs, scan_index)
                else
                    # Emit at the plate, where all its scalar inputs are ready.
                    # The reserved table slots keep both backend products equal.
                    pending_scans[consumer.id] = (r, callargs, scan_index)
                end
            end
            continue
        end
        embedded = inline_embedded ? _embedded_kernel(r.op) : nothing
        if embedded === nothing
            push!(runtime_ops, r.op)
            push!(runtime_recipes, r)
            call = Expr(:call, Expr(:ref, _OPS_ARG, length(runtime_ops)),
                        callargs...)
            push!(body.args, Expr(:(=), lhs, call))
        else
            inner_ast = _embedded_ast(embedded, tensorized)
            op_offset = length(runtime_ops)
            append!(runtime_ops, embedded.ops)
            append!(runtime_recipes, embedded.lowered_recipes)
            append!(body.args,
                    _embedded_statements(inner_ast, callargs, lhs, op_offset))
        end
    end
    retval = length(p.want) == 1 ? nm(p.want[1]) :
             Expr(:tuple, (nm(w) for w in p.want)...)
    push!(body.args, Expr(:return, retval))
    Expr(:function, Expr(:tuple, argexprs...), body),
    Tuple(runtime_ops), Tuple(runtime_recipes)
end

"""
    lower(p::Plan) -> Expr

Lower a plan to an ordinary anonymous-function `Expr` of the form

    function (__ops__, x::T1, y::T2)
        a = __ops__[1](x, y)
        ...
        return out
    end

This `Expr` is a first-class artifact: it may be inspected (`code_expr`) and
rewritten (`transform`) before compilation (gist §9).
"""
lower(p::Plan) = first(_lower_with_ops(p; inline_embedded = false))
_lower_unembedded(p::Plan) = lower(p)

function _batched_dependency_analysis(p::Plan, batched)
    graph = p.graph
    names = Tuple(batched isa Symbol ? (batched,) : batched)
    isempty(names) && throw(ArgumentError(
        "lower_batched requires at least one batched HAVE port"))
    all(name -> name isa Symbol, names) || throw(ArgumentError(
        "lower_batched batched ports must be Symbols; got $(names)"))
    length(unique(names)) == length(names) || throw(ArgumentError(
        "lower_batched batched ports must be unique; got $(names)"))

    have_by_name = Dict(value.name => value for value in p.have)
    mapped = Value[]
    for name in names
        value = get(have_by_name, name, nothing)
        value === nothing && throw(ArgumentError(
            "lower_batched port :$name is not in the plan HAVE boundary"))
        push!(mapped, value)
    end
    mapped_ids = Set(canon_id(graph, value.id) for value in mapped)
    dependencies = _plate_dependencies(p, mapped_ids)
    want = only(p.want)
    want_dependencies = get(
        dependencies.values, canon_id(graph, want.id), Set{Int}())
    isempty(want_dependencies) && throw(ArgumentError(
        "lower_batched: want :$(want.name) is loop-invariant (does not depend on a batched port); nothing to vectorize"))
    (; mapped = Tuple(mapped), mapped_ids,
       recipe_dependencies = dependencies.recipes, want)
end

"""
    lower_batched(p::Plan; batched, reduce = :+) -> Expr

Lower a pure plan to a batched, **dependency-stratified** kernel. The `batched`
HAVE ports participate in ordinary Julia broadcasting: compatible axes zip,
singleton dimensions expand, scalars repeat, and `Ref(x)` keeps an array-valued
input atomic. Broadcast axes are instantiated and checked before any recipe or
output mutation executes.

Every recipe inherits the transitive set of batched HAVE ports it depends on.
During Cartesian broadcast traversal, its scalar result is recomputed only when
a kept broadcast dimension of one of those roots changes. This is equivalent to
placing the recipe at the narrowest valid nested-loop boundary: an outer-only
scale transform runs once per scale coordinate and is reused through all inner
coordinates without an intermediate buffer. Recipes independent of every
batched port are emitted once above the traversal.

The single scalar `want` is accumulated by `reduce` (default `:+`, a sum).
Pass `reduce=nothing` to collect the broadcast-shaped pointwise wants instead;
only that requested output is materialized.

Restricted (this lowering) to single-output recipes and a single scalar `want`
that is itself batched (the per-element density). `reduce` is spliced as a bare
callee, so use a `Base` reducer symbol (`:+`).
"""
function lower_batched(p::Plan; batched, reduce = :+)
    g = p.graph
    length(p.want) == 1 || throw(ArgumentError(
        "lower_batched requires a single scalar want; got $(length(p.want))"))
    for r in p.recipes
        length(r.outputs) == 1 || throw(ArgumentError(
            "lower_batched requires single-output recipes; recipe $(r.id) has $(length(r.outputs)) outputs"))
    end
    analysis = _batched_dependency_analysis(p, batched)
    names = _varnames(p)
    nm(v) = names[canon_id(g, v.id)]
    mapped(value) = canon_id(g, value.id) in analysis.mapped_ids

    index = gensym(:plate_index)
    previous = gensym(:plate_previous)
    first_coordinate = gensym(:plate_first)
    raw_arguments = Dict{Int,Symbol}()
    prepared_arguments = Dict{Int,Symbol}()
    for value in analysis.mapped
        cid = canon_id(g, value.id)
        raw_arguments[cid] = gensym(Symbol(:plate_argument_, value.name))
        prepared_arguments[cid] = gensym(Symbol(:plate_prepared_, value.name))
    end

    # `_broadcast_getindex` implements Julia's scalar, Ref, and singleton
    # projection at the current Cartesian coordinate.
    function argref(input)
        cid = canon_id(g, input.id)
        haskey(prepared_arguments, cid) || return nm(input)
        Expr(:call, GlobalRef(Base.Broadcast, :_broadcast_getindex),
             prepared_arguments[cid], index)
    end
    callexpr(k, r) = Expr(:call, Expr(:ref, _OPS_ARG, k),
                          (argref(inp) for inp in r.inputs)...)

    argexprs = Any[_OPS_ARG]
    for v in p.have
        push!(argexprs, mapped(v) ? nm(v) : :($(nm(v))::$(valtype(v))))
    end

    body = Expr(:block)
    for value in analysis.mapped
        cid = canon_id(g, value.id)
        raw = raw_arguments[cid]
        prepared = prepared_arguments[cid]
        push!(body.args, Expr(:(=), raw,
            Expr(:call, GlobalRef(Base, :broadcastable), nm(value))))
        push!(body.args, Expr(:(=), prepared,
            Expr(:call, GlobalRef(Base.Broadcast, :preprocess), nothing, raw)))
    end
    broadcast_arguments = Expr(
        :tuple, (raw_arguments[canon_id(g, value.id)]
                 for value in analysis.mapped)...)
    output_axes = gensym(:plate_axes)
    combined_axes = Expr(:call, GlobalRef(Base.Broadcast, :combine_axes),
        (raw_arguments[canon_id(g, value.id)]
         for value in analysis.mapped)...)
    push!(body.args, Expr(:(=), output_axes,
        Expr(:call, GlobalRef(@__MODULE__, :_plate_require_axes),
             combined_axes)))

    assigned = Set(canon_id(g, v.id) for v in p.have)
    for (k, r) in enumerate(p.recipes)
        isempty(analysis.recipe_dependencies[k]) || continue
        out = only(r.outputs)
        cid = canon_id(g, out.id)
        cid in assigned && continue
        push!(assigned, cid)
        push!(body.args, Expr(:(=), nm(out), callexpr(k, r)))
    end

    loopbody = Expr(:block)
    loop_assigned = copy(assigned)
    for (k, r) in enumerate(p.recipes)
        roots = analysis.recipe_dependencies[k]
        isempty(roots) && continue
        out = only(r.outputs)
        cid = canon_id(g, out.id)
        cid in loop_assigned && continue
        push!(loop_assigned, cid)
        push!(body.args, Expr(:local, Expr(:(::), nm(out), valtype(out))))
        changed = foldl((left, right) -> Expr(:||, left, right),
            (Expr(:call, GlobalRef(@__MODULE__, :_plate_dependency_changed),
                  index, previous, prepared_arguments[root])
             for root in sort!(collect(roots))))
        condition = Expr(:||, first_coordinate, changed)
        push!(loopbody.args,
              Expr(:if, condition, Expr(:(=), nm(out), callexpr(k, r))))
    end

    want = analysis.want
    result = gensym(reduce === nothing ? :collected : :accumulator)
    if reduce === nothing
        push!(body.args, Expr(:(=), result,
            Expr(:call, GlobalRef(@__MODULE__, :_plate_similar),
                 broadcast_arguments, valtype(want), output_axes)))
        push!(loopbody.args,
              Expr(:(=), Expr(:ref, result, index), nm(want)))
    else
        push!(body.args, Expr(:(=), result,
            Expr(:call, GlobalRef(Base, :zero), valtype(want))))
        push!(loopbody.args, Expr(:(=), result,
            Expr(:call, reduce, result, nm(want))))
    end

    push!(loopbody.args, Expr(:(=), previous, index))
    push!(loopbody.args, Expr(:(=), first_coordinate, false))
    pushfirst!(loopbody.args, Expr(:if, first_coordinate,
                                   Expr(:(=), previous, index)))
    push!(body.args, Expr(:(=), previous, nothing))
    push!(body.args, Expr(:(=), first_coordinate, true))
    iteration = Expr(:call, GlobalRef(Base, :CartesianIndices), output_axes)
    push!(body.args, Expr(:for, Expr(:(=), index, iteration), loopbody))
    push!(body.args, Expr(:return, result))
    Expr(:function, Expr(:tuple, argexprs...), body)
end

# Reactant deliberately rejects scalar indexing of traced arrays.  Keep the
# ordinary loop lowering above as the native hot path, and compile this eager
# tensor lowering alongside it for array-tracing backends.  Each dependent
# recipe is materialized before the next one consumes it; Reactant sees those
# broadcasts and the final reduction as tensor operations and can fuse them in
# the compiled program.  This body is never selected for ordinary Julia arrays,
# so the native exact-zero-allocation reducing contract is unchanged.
function _lower_batched_tensorized(p::Plan; batched, reduce = :+)
    g = p.graph
    length(p.want) == 1 || throw(ArgumentError(
        "lower_batched requires a single scalar want; got $(length(p.want))"))
    for r in p.recipes
        length(r.outputs) == 1 || throw(ArgumentError(
            "lower_batched requires single-output recipes; recipe $(r.id) has $(length(r.outputs)) outputs"))
    end
    batched_names = Set{Symbol}(batched isa Symbol ? (batched,) : batched)
    names = _varnames(p)
    nm(v) = names[canon_id(g, v.id)]

    batched_input_ids = Set{Int}()
    for v in p.have
        v.name in batched_names && push!(batched_input_ids, canon_id(g, v.id))
    end
    isempty(batched_input_ids) && throw(ArgumentError(
        "lower_batched: none of the have ports are batched (batched = $(sort(collect(batched_names))))"))

    batched_vals = Set{Int}(batched_input_ids)
    recipe_is_batched = falses(length(p.recipes))
    for (k, r) in enumerate(p.recipes)
        dependent = any(canon_id(g, inp.id) in batched_vals for inp in r.inputs)
        recipe_is_batched[k] = dependent
        dependent && for output in r.outputs
            push!(batched_vals, canon_id(g, output.id))
        end
    end

    want = only(p.want)
    canon_id(g, want.id) in batched_vals || throw(ArgumentError(
        "lower_batched: want :$(want.name) is loop-invariant (does not depend on a batched port); nothing to vectorize"))

    argexprs = Any[_OPS_ARG]
    for v in p.have
        push!(argexprs, canon_id(g, v.id) in batched_input_ids ?
              nm(v) : :($(nm(v))::$(valtype(v))))
    end

    body = Expr(:block)
    assigned = Set(canon_id(g, v.id) for v in p.have)
    for (k, r) in enumerate(p.recipes)
        out = only(r.outputs)
        cid = canon_id(g, out.id)
        cid in assigned && continue
        push!(assigned, cid)
        op = Expr(:ref, _OPS_ARG, k)
        args = Any[nm(inp) for inp in r.inputs]
        call = recipe_is_batched[k] ?
               Expr(:call, GlobalRef(Base, :broadcast), op, args...) :
               Expr(:call, op, args...)
        push!(body.args, Expr(:(=), nm(out), call))
    end

    retval = if reduce === nothing
        nm(want)
    elseif reduce === :+
        Expr(:call, GlobalRef(Base, :sum), nm(want))
    else
        Expr(:call, GlobalRef(Base, :reduce), reduce, nm(want))
    end
    push!(body.args, Expr(:return, retval))
    Expr(:function, Expr(:tuple, argexprs...), body)
end

"""
    transform(ast, passes...) -> Expr

Apply zero or more AST passes (each an `Expr -> Expr` function) in order. This
is the extension point for simplification, mutation/bufferization, or
backend-specific rewrites; it must not change planning semantics (gist §9).
"""
transform(ast::Expr, passes...) = foldl((a, pass) -> pass(a), passes; init = ast)

"""
    compile(ast::Expr) -> callable

Compile a lowered `Expr` into a native Julia function via
`RuntimeGeneratedFunctions`. The returned callable takes `(__ops__, args...)`.
"""
compile(ast::Expr) = @RuntimeGeneratedFunction(ast)

# A plated kernel owns two compiled bodies but presents exactly the same
# PreparedKernel API as every scalar kernel.  The batched input position is a
# type parameter so choosing the native body remains inferred and allocation
# free.  Optional backend extensions specialize `_batched_call` on their traced
# array marker; ordinary arrays always take `native`.
abstract type _ArrayFunctionPair end

struct _BatchedFunctionPair{I,B,R,N,T} <: _ArrayFunctionPair
    native::N
    tensorized::T
end

@inline function (f::_BatchedFunctionPair{I})(ops, args...) where {I}
    traced = _dynamic_tensorized_marker(args)
    marker = traced === nothing ? getfield(args, I) : traced
    _batched_call(f, ops, args, marker)
end

struct _EmbeddedFunctionPair{I,N,T,A} <: _ArrayFunctionPair
    native::N
    tensorized::T
    tensorized_ast::A
end

@inline function (f::_EmbeddedFunctionPair{I})(ops, args...) where {I}
    traced = _dynamic_tensorized_marker(args)
    marker = traced === nothing ? getfield(args, I) : traced
    _batched_call(f, ops, args, marker)
end

# An untyped authored signature still specializes on its concrete call-site
# argument types.  Keep every graph-proven candidate axis as type metadata and
# prefer the first runtime broadcast axis when choosing native vs. tensorized
# execution.  The native call ignores the marker and may derive its plate axis
# only after projecting an atomic boundary, so exhausting the candidates falls
# back to a non-axis sentinel.  Atomic `Ref(port)` inputs are excluded from the
# axis candidates, so an array-valued atom cannot be mistaken for the plate
# axis; but when every axis operand is bound and no axis candidate remains,
# `_embedded_marker_candidates` admits array-valued HAVE ports (atomic ones
# included) as backend markers, since only the marker TYPE — never its axis —
# selects native vs. tensorized execution there.
struct _DynamicEmbeddedFunctionPair{I,N,T,A} <: _ArrayFunctionPair
    native::N
    tensorized::T
    tensorized_ast::A
end

@inline _dynamic_embedded_marker(args, ::Val{()}) = nothing

# A graph-proven candidate can be an atomic model boundary rather than the
# array consumed by a downstream plate.  Traverse only deliberately supported
# structures; generic structs remain atomic instead of being reflected over or
# passed to `broadcastable`.  Backend extensions may specialize this trait for
# their own transparent runtime carrier.
@inline _embedded_marker_values(x) = ()
@inline _embedded_marker_values(x::NamedTuple) = values(x)

@inline function _embedded_axis_marker(value)
    _authored_plate_is_axis(value) && return value
    _embedded_axis_marker_values(_embedded_marker_values(value))
end

@inline _embedded_axis_marker_values(::Tuple{}) = nothing

@inline function _embedded_axis_marker_values(values::Tuple)
    marker = _embedded_axis_marker(first(values))
    marker === nothing ?
        _embedded_axis_marker_values(Base.tail(values)) : marker
end

@inline function _dynamic_embedded_marker(args, ::Val{I}) where {I}
    index = first(I)
    marker = _embedded_axis_marker(getfield(args, index))
    marker === nothing || return marker
    _dynamic_embedded_marker(args, Val(Base.tail(I)))
end

@inline _requires_tensorized_marker(marker) = false
@inline _dynamic_tensorized_marker(::Tuple{}) = nothing

@inline function _dynamic_tensorized_value_marker(value)
    _requires_tensorized_marker(value) && return value
    _dynamic_tensorized_marker(_embedded_marker_values(value))
end

@inline function _dynamic_tensorized_marker(args::Tuple)
    marker = _dynamic_tensorized_value_marker(first(args))
    marker === nothing ? _dynamic_tensorized_marker(Base.tail(args)) : marker
end

@inline function (f::_DynamicEmbeddedFunctionPair{I})(ops, args...) where {I}
    traced = _dynamic_tensorized_marker(args)
    marker = traced === nothing ?
             _dynamic_embedded_marker(args, Val(I)) : traced
    _batched_call(f, ops, args, marker)
end

@inline _batched_call(f::_ArrayFunctionPair, ops, args, marker) =
    f.native(ops, args...)

"""
    PreparedKernel

A small callable object holding the RGF-generated function, the positional
`ops` tuple, and metadata (graph values in call/return order, the plan, and the
lowered AST). Runtime invocation does not consult any planning logic.
"""
struct PreparedKernel{F,O,IN,OUT,RR}
    f::F
    ops::O
    inputs::IN
    outputs::OUT
    plan::Plan
    ast::Expr
    lowered_recipes::RR
end

# Partial evaluation deliberately stores hoisted values in the prepared
# operation tuple so the public residual kernel has only its unbound HAVE
# ports. Compiler and differentiation backends must nevertheless be able to
# receive array-valued constants as explicit inactive operands: capturing them
# in the callable can either turn a whole dataset into source-level literals or
# obscure its read-only activity. This compact call boundary replaces only
# array-valued `_BoundConstant`s with trailing hidden operands. The public
# PreparedKernel is unchanged; ordinary execution and inspection retain the
# residual ABI.
struct _ExternalBoundArraySlot{I} end

struct _ExternalizedBoundArrayCall{F,O,I}
    f::F
    ops::O
end

@generated function (call::_ExternalizedBoundArrayCall{F,O,I})(
        args::Vararg{Any,N}) where {F,O,I,N}
    external_count = length(I)
    public_count = N - external_count
    public_count >= 0 || return :(throw(ArgumentError(
        "externalized bound-array call is missing hidden operands")))
    replacements = Dict(index => slot for (slot, index) in enumerate(I))
    operations = Any[]
    for index in 1:fieldcount(O)
        if haskey(replacements, index)
            argument = public_count + replacements[index]
            push!(operations,
                  :(_BoundConstant(getfield(args, $argument))))
        else
            push!(operations,
                  :(getfield(getfield(call, :ops), $index)))
        end
    end
    public_args = Any[
        :(getfield(args, $index)) for index in 1:public_count]
    :(getfield(call, :f)(($(operations...),), $(public_args...)))
end

"""
    _externalize_bound_arrays(kernel; min_elements = 0) -> (call, values)

Return an internal backend call plus the array-valued partial-evaluation
constants it expects as trailing hidden operands. If the kernel contains no
such constants, return the kernel itself and an empty tuple. Scalar and other
small static constants remain in the operation table; `min_elements` lets a
backend keep arrays below that element count in the table as well, so a small
static dataset stays a compiler literal it can fold rather than a runtime
operand it must read. By default every array is externalized.

This is a backend ABI adapter, not a different model boundary: `kernel` keeps
its original public inputs, and `call(public_args..., values...)` is exactly
equivalent to `kernel(public_args...)`.
"""
function _externalize_bound_array_call(f, ops;
                                       min_elements::Integer = 0)
    positions = Tuple(
        index for (index, op) in pairs(ops)
        if op isa _BoundConstant && op.value isa AbstractArray &&
           length(op.value) >= min_elements)
    isempty(positions) && return nothing, ()
    values = Tuple(ops[index].value for index in positions)
    stripped = ntuple(length(ops)) do index
        slot = findfirst(==(index), positions)
        slot === nothing ? ops[index] :
            _ExternalBoundArraySlot{slot}()
    end
    call = _ExternalizedBoundArrayCall{
        typeof(f),typeof(stripped),positions}(f, stripped)
    call, values
end

function _externalize_bound_arrays(kernel::PreparedKernel;
                                   min_elements::Integer = 0)
    call, values = _externalize_bound_array_call(
        kernel.f, kernel.ops; min_elements)
    call === nothing ? (kernel, ()) : (call, values)
end

# Prepared RK kernels are compiler-owned program structure when used as recipe
# operations. Flatten them automatically so ordinary composition of `plate`
# and scalar prepared kernels produces one generated outer program rather than
# an opaque nested callback.
_embedded_kernel(kernel::PreparedKernel) = kernel

_batched_options(::_BatchedFunctionPair{I,B,R}) where {I,B,R} = (B, R)
function _embedded_ast(kernel::PreparedKernel, tensorized::Bool)
    tensorized || return kernel.ast
    if kernel.f isa _BatchedFunctionPair
        batched, reduce = _batched_options(kernel.f)
        return _lower_batched_tensorized(
            kernel.plan; batched = batched, reduce = reduce)
    elseif kernel.f isa Union{_EmbeddedFunctionPair,
                              _DynamicEmbeddedFunctionPair}
        # The tensorized product may already contain arbitrarily nested plates
        # and user AST passes. Preserve that exact prepared artifact instead of
        # rebuilding from the native `kernel.ast`, which would silently restore
        # scalar-indexing loops at the next composition level.
        return kernel.f.tensorized_ast
    end
    kernel.ast
end

function _needs_embedded_tensorization(p::Plan)
    any(p.recipes) do recipe
        recipe.op isa _AuthoredPlateOp && return true
        recipe.op isa _AuthoredScanOp && !isempty(p.have) && return true
        kernel = _embedded_kernel(recipe.op)
        kernel !== nothing && kernel.f isa _ArrayFunctionPair
    end
end

_array_marker_positions(::_ArrayFunctionPair) = ()
_array_marker_positions(::_BatchedFunctionPair{I}) where {I} = (I,)
_array_marker_positions(::_EmbeddedFunctionPair{I}) where {I} = (I,)
_array_marker_positions(::_DynamicEmbeddedFunctionPair{I}) where {I} = I

function _embedded_marker_candidates(p::Plan)
    root_positions = Dict(
        canon_id(p.graph, input.id) => position
        for (position, input) in enumerate(p.have)
    )
    dependencies = _plate_dependencies(
        p, Set(keys(root_positions))).values
    candidates = Int[]
    for recipe in p.recipes
        positions = if recipe.op isa _AuthoredScanOp
            # With the sequence bound, a traced carry/shared operand still
            # selects the tensorized product through its runtime HAVE roots.
            (2, 1, (3:length(recipe.inputs))...)
        elseif recipe.op isa _AuthoredPlateOp
            atomic = typeof(recipe.op).parameters[2]
            Tuple(index for index in eachindex(recipe.inputs)
                  if !(index in atomic))
        else
            kernel = _embedded_kernel(recipe.op)
            kernel === nothing ? () : _array_marker_positions(kernel.f)
        end
        for position in positions
            input = recipe.inputs[position]
            roots = sort!(collect(get(
                dependencies, canon_id(p.graph, input.id), Set{Int}())))
            for root in roots
                position = root_positions[root]
                position in candidates || push!(candidates, position)
            end
        end
    end
    if isempty(candidates)
        # No non-atomic (axis-defining) plate operand traces back to an active
        # HAVE port: every axis operand is bound (`bound=`), so the plate axis is
        # a compile-time constant baked into both bodies (the native body derives
        # its own axis; the tensorized body takes the marker-less axis fallback).
        # Backend selection still needs a runtime marker, so admit any
        # array-valued active HAVE port as one — including a whole-vector
        # `Ref`-captured parameter that is atomic for broadcasting yet remains a
        # live array HAVE. Only the marker TYPE is consulted (`_batched_call`
        # dispatches native vs. tensorized on it), never its axis, so an atomic
        # array admitted here cannot be mistaken for the plate axis.
        for (position, input) in enumerate(p.have)
            valtype(input) <: AbstractArray && push!(candidates, position)
        end
    end
    Tuple(candidates)
end

"""
    ReplicatedKernel

A callable produced by [`replica`](@ref). It preserves a scalar prepared
kernel as the single source of truth and maps that complete callable over a
trailing replica axis on selected HAVE ports.
"""
struct ReplicatedKernel{B,BT,OT,K,IN,OUT}
    target::K
    inputs::IN
    outputs::OUT
end

@inline function (k::ReplicatedKernel{B})(args...) where {B}
    length(args) == length(k.inputs) || throw(MethodError(k, args))
    _replica_call(k, args, getfield(args, first(B)))
end

inputs(k::ReplicatedKernel) = k.inputs
outputs(k::ReplicatedKernel) = k.outputs
code_expr(k::ReplicatedKernel) = code_expr(k.target)

function Base.show(io::IO, k::ReplicatedKernel{B}) where {B}
    names = Tuple(k.inputs[i].name for i in B)
    print(io, "ReplicatedKernel(batched=", names, ", target=")
    show(io, k.target)
    print(io, ")")
end

_replica_rank(::Type{T}) where {T<:Number} = 0
_replica_rank(::Type{T}) where {T<:AbstractArray} = ndims(T)
_replica_rank(::Type{T}) where {T} = throw(ArgumentError(
    "replica batched ports must be Numbers or AbstractArrays; got $T"))

function _replica(target, batched)
    boundary = inputs(target)
    names = Tuple(batched isa Symbol ? (batched,) : batched)
    isempty(names) && throw(ArgumentError("replica requires at least one batched port"))
    length(unique(names)) == length(names) || throw(ArgumentError(
        "replica batched port names must be unique; got $(names)"))
    all(name -> name isa Symbol, names) || throw(ArgumentError(
        "replica batched ports must be Symbols; got $(names)"))

    indices = map(names) do name
        index = findfirst(value -> value.name === name, boundary)
        index === nothing && throw(ArgumentError(
            "replica batched port :$name is not in the prepared HAVE boundary"))
        index
    end
    indices = Tuple(sort(collect(indices)))
    input_types = Tuple{(valtype(boundary[i]) for i in indices)...}
    foreach(_replica_rank, input_types.parameters)
    output_types = Tuple{(valtype(value) for value in outputs(target))...}
    all(type -> type <: Union{Number,AbstractArray}, output_types.parameters) ||
        throw(ArgumentError(
            "replica outputs must be Numbers or AbstractArrays; got $(output_types.parameters)"))
    ReplicatedKernel{indices,input_types,output_types,typeof(target),
                     typeof(boundary),typeof(outputs(target))}(
        target, boundary, outputs(target))
end

"""
    replica(kernel::PreparedKernel; batched) -> ReplicatedKernel

Lift a complete scalar prepared kernel over a trailing replica axis. `batched`
names the HAVE ports that receive that extra final dimension. Scalar batched
ports therefore become vectors, vectors become matrices, and so on; shared
ports retain their scalar-kernel shapes. Outputs receive the same trailing
replica axis.

The scalar kernel remains the only mathematical definition. In particular,
reductions inside it still reduce only their original dimensions, so a scalar
`dot(q, q)` becomes one dot product per replica rather than a reduction across
replicas. Native Julia evaluates scalar replicas and stacks their results;
optional array-compiler extensions may lower the same map to a backend batch
primitive.
"""
replica(kernel::PreparedKernel; batched) = _replica(kernel, batched)

@inline _replica_native_arg(arg, ::Type{T}, replica_index) where {T<:Number} =
    arg[replica_index]
@inline _replica_native_arg(arg, ::Type{T}, replica_index) where {T<:AbstractArray} =
    copy(selectdim(arg, ndims(arg), replica_index))

@generated function _replica_native_inputs(
        ::Val{B}, ::Type{BT}, args::A, replica_index) where {B,BT,A}
    batched_lookup = Dict(index => position for (position, index) in enumerate(B))
    values = Any[]
    for index in 1:length(A.parameters)
        if haskey(batched_lookup, index)
            position = batched_lookup[index]
            push!(values, :(_replica_native_arg(
                getfield(args, $index), $(BT.parameters[position]), replica_index)))
        else
            push!(values, :(getfield(args, $index)))
        end
    end
    Expr(:tuple, values...)
end

_replica_stack(values, ::Type{T}) where {T<:Number} = collect(values)
_replica_stack(values, ::Type{T}) where {T<:AbstractArray} = stack(values)

function _replica_native_outputs(results, ::Type{OT}) where {OT<:Tuple}
    output_types = OT.parameters
    if length(output_types) == 1
        return _replica_stack(results, only(output_types))
    end
    ntuple(length(output_types)) do output_index
        _replica_stack((result[output_index] for result in results),
                       output_types[output_index])
    end
end

function _replica_call(k::ReplicatedKernel{B,BT,OT}, args, marker) where {B,BT,OT}
    replica_count = size(marker, ndims(marker))
    for index in B
        arg = getfield(args, index)
        expected_rank = _replica_rank(valtype(k.inputs[index])) + 1
        ndims(arg) == expected_rank || throw(DimensionMismatch(
            "replica port :$(k.inputs[index].name) has rank $(ndims(arg)); " *
            "expected $expected_rank (scalar rank plus one trailing replica axis)"))
        size(arg, ndims(arg)) == replica_count || throw(DimensionMismatch(
            "replica port :$(k.inputs[index].name) has " *
            "$(size(arg, ndims(arg))) replicas; expected $replica_count"))
    end
    results = map(1:replica_count) do replica_index
        scalar_args = _replica_native_inputs(
            Val(B), BT, args, replica_index)
        k.target(scalar_args...)
    end
    _replica_native_outputs(results, OT)
end

# Emit positional arguments explicitly: on Julia 1.12, splatting the captured
# `args` tuple into some RGF call shapes allocates even though the emitted
# function itself is allocation-free. Keep the public call nongenerated so
# reflection over it continues to accept abstract argument types.
@generated function _prepared_call(k::PreparedKernel, args::A, ::Val{N}) where {A<:Tuple,N}
    positional = [:(getfield(args, $index)) for index in 1:N]
    :(k.f(k.ops, $(positional...)))
end

@inline function (k::PreparedKernel{F,O,IN,OUT})(
        args::Vararg{Any,N}) where {F,O,IN,OUT,N}
    N == fieldcount(IN) || throw(MethodError(k, args))
    _prepared_call(k, args, Val(N))
end

function _prepare(p::Plan, ast::Expr, ops::Tuple, recipes::Tuple)
    f = compile(ast)
    PreparedKernel(f, ops, Tuple(p.have), Tuple(p.want), p, ast, recipes)
end

"""
    _prepare_batched(p::Plan; batched, reduce = :+) -> PreparedKernel

Internal constructor shared by public plate authoring and optional array
compiler extensions.  It compiles both the allocation-free native loop and an
eager tensorized body, then selects between them by the runtime array type.
"""
function _prepare_batched(p::Plan; batched, reduce = :+)
    batched_names = Set{Symbol}(batched isa Symbol ? (batched,) : batched)
    input_index = findfirst(v -> v.name in batched_names, p.have)
    input_index === nothing && throw(ArgumentError(
        "lower_batched: none of the have ports are batched (batched = $(sort(collect(batched_names))))"))

    native_ast = lower_batched(p; batched = batched, reduce = reduce)
    tensorized_ast = _lower_batched_tensorized(p; batched = batched, reduce = reduce)
    native = compile(native_ast)
    tensorized = compile(tensorized_ast)
    batched_ports = Tuple(value.name for value in p.have
                          if value.name in batched_names)
    f = _BatchedFunctionPair{
        input_index,batched_ports,reduce,typeof(native),typeof(tensorized)}(
            native, tensorized)
    ops = ntuple(i -> p.recipes[i].op, length(p.recipes))
    PreparedKernel(f, ops, Tuple(p.have), Tuple(p.want), p, native_ast,
                   Tuple(p.recipes))
end

# The optional MutatingFunctions extension uses one typed cache cell per
# selected recipe. Its `nothing` value requests the allocating implementation
# on first use; later calls feed the stored value back to the extension helper.
_cache_slot(::Value{T}) where {T} = Ref{Union{Nothing,T}}(nothing)

function _rewrite_nonallocating_calls(node)
    node isa Expr || return node
    args = map(_rewrite_nonallocating_calls, node.args)
    rewritten = Expr(node.head, args...)
    if rewritten.head === :call && rewritten.args[1] isa Expr
        callee = rewritten.args[1]
        if callee.head === :ref && length(callee.args) == 2 &&
           callee.args[1] === _OPS_ARG && callee.args[2] isa Int
            cache = Expr(:ref, _CACHES_ARG, callee.args[2])
            return Expr(:call, _CACHE_APPLY_ARG, cache, callee,
                        rewritten.args[2:end]...)
        end
    end
    rewritten
end

function _nonallocating_ast(ast::Expr)
    ast.head === :function ||
        throw(ArgumentError("non-allocating preparation requires a function Expr"))
    signature = ast.args[1]
    signature isa Expr && signature.head === :tuple &&
        !isempty(signature.args) && first(signature.args) === _OPS_ARG ||
        throw(ArgumentError("non-allocating preparation requires the lowered __ops__ signature"))
    args = Expr(:tuple, _OPS_ARG, _CACHES_ARG, _CACHE_APPLY_ARG,
                signature.args[2:end]...)
    Expr(:function, args, _rewrite_nonallocating_calls(ast.args[2]))
end

"""
    NonAllocatingKernel

A stateful prepared kernel whose selected single-output recipes are invoked
through `MutatingFunctions.apply!!`. Each recipe owns a persistent typed cache:
the first call seeds it and later calls offer it back for in-place reuse.

Each cache slot retains whatever its operation returned on the first call.
Registered allocating operations such as `copy` normally seed fresh
kernel-retained storage, but aliasing operations may retain caller-owned
inputs, and a no-recipe plan returns its `have` value directly. Treat mutable
results as borrowed values that may alias inputs or be overwritten by later
calls. A kernel instance is therefore neither reentrant nor safe for concurrent
calls; prepare one instance per independent caller.
"""
struct NonAllocatingKernel{F,O,C,A,IN,OUT}
    f::F
    ops::O
    caches::C
    cache_apply::A
    inputs::IN
    outputs::OUT
    plan::Plan
    ast::Expr
end

@inline function (k::NonAllocatingKernel)(args...)
    length(args) == length(k.inputs) || throw(MethodError(k, args))
    k.f(args...)
end

function _prepare_nonallocating(p::Plan, ast::Expr, cache_apply)
    for r in p.recipes
        length(r.outputs) == 1 || throw(ArgumentError(
            "prepare_nonallocating requires single-output recipes; recipe $(r.id) has $(length(r.outputs)) outputs"))
    end
    rewritten, ops, caches = _nonallocating_program(p, ast)
    # Compile with the operation and cache tuples bound as constants inside the
    # body. Passing them as call arguments re-tuples the non-isbits operation
    # table on every invocation at the runtime-generated call boundary — a
    # measured fixed per-call heap cost. `k.ast` keeps the unbound, readable
    # form; the tuples are empty/unseeded at preparation, so embedding them is
    # cheap and the compiled body sees them as constants.
    f = compile(_bind_nonallocating_constants(rewritten, ops, caches,
                                              cache_apply))
    NonAllocatingKernel(f, ops, caches, cache_apply, Tuple(p.have),
                        Tuple(p.want), p, rewritten)
end

function _bind_nonallocating_constants(ast::Expr, ops::Tuple, caches::Tuple,
                                       cache_apply)
    signature = ast.args[1]
    runtime_args = signature.args[4:end]
    body = ast.args[2]
    Expr(:function, Expr(:tuple, runtime_args...),
         Expr(:block,
              Expr(:(=), _OPS_ARG, ops),
              Expr(:(=), _CACHES_ARG, caches),
              Expr(:(=), _CACHE_APPLY_ARG, cache_apply),
              body.args...))
end

"""
    prepare(p::Plan; passes=(), bound=()) -> PreparedKernel
    prepare(g::Graph; have, want, passes=(), bound=()) -> PreparedKernel

Ergonomic composition of `plan -> lower -> transform -> compile`. `passes` is a
tuple of AST passes applied before compilation.

`bound` opts into the [`partial_evaluation`](@ref) pre-pass: pass one
`Value => data` pair (or an iterable of them) naming HAVE ports whose runtime
values are fixed for this preparation. The data-only subgraph reachable from
only those ports runs once, here, and the returned kernel takes just the
remaining HAVE ports positionally (in their original relative order); the
hoisted values are baked in as constants. With `bound = ()` (the default)
behavior is unchanged. `passes` apply to the residual (per-call) kernel.

Selected authored plates with a single plate consumer are composed during
lowering, eliminating the intermediate array. Named ports remain in `p`: asking
for an intermediate, supplying it as HAVE, or selecting another consumer keeps
that boundary. Whole-array (`Ref`) consumers and opaque intervening recipes
also retain their materialization boundary.
"""
function prepare(p::Plan; passes = (), bound = ())
    p = _partial_apply(p, bound)
    lowered_plan = _fuse_authored_plate_chains(p)
    native_ast, ops, recipes = _lower_with_ops(lowered_plan)
    isempty(passes) || (native_ast = transform(native_ast, passes...))
    if !_needs_embedded_tensorization(p)
        return _prepare(p, native_ast, ops, recipes)
    end

    tensorized_ast, tensorized_ops, tensorized_recipes =
        _lower_with_ops(lowered_plan; tensorized = true)
    tensorized_ops == ops || throw(ArgumentError(
        "embedded native and tensorized kernels produced different operation tables"))
    tensorized_recipes == recipes || throw(ArgumentError(
        "embedded native and tensorized kernels produced different readable recipes"))
    isempty(passes) ||
        (tensorized_ast = transform(tensorized_ast, passes...))
    native = compile(native_ast)
    tensorized = compile(tensorized_ast)
    candidates = _embedded_marker_candidates(p)
    isempty(candidates) && throw(ArgumentError(
        "an embedded plate requires an array-valued HAVE port in the outer kernel"))
    typed_candidate = findfirst(
        index -> valtype(p.have[index]) <: AbstractArray, candidates)
    f = if typed_candidate === nothing
        _DynamicEmbeddedFunctionPair{
            candidates,typeof(native),typeof(tensorized),typeof(tensorized_ast)}(
                native, tensorized, tensorized_ast)
    else
        input_index = candidates[typed_candidate]
        _EmbeddedFunctionPair{
            input_index,typeof(native),typeof(tensorized),typeof(tensorized_ast)}(
                native, tensorized, tensorized_ast)
    end
    PreparedKernel(f, ops, Tuple(p.have), Tuple(p.want), p, native_ast, recipes)
end

function prepare(g::Graph; have = (), want = (), passes = (), bound = ())
    p = plan(g; have = have, want = want)
    prepare(p; passes = passes, bound = bound)
end

"""
    prepare_nonallocating(p::Plan; passes=()) -> NonAllocatingKernel
    prepare_nonallocating(g::Graph; have, want, passes=()) -> NonAllocatingKernel

Optional MutatingFunctions-backed preparation interface. Install and load
`MutatingFunctions` alongside `ReactiveKernels` to activate the package
extension that supplies these methods (for a `Plan`, a `Graph`, or an
already-prepared `PreparedKernel`'s plan). The extension prepares the same
straight-line plan as [`prepare`](@ref), then applies a final AST transform
that routes every selected operation through `MutatingFunctions.apply!!` and a
persistent per-step cache. User `passes` run before this final transform.

Operations synthesized from captured `@kernel` source are decomposed into
destination-passing steps where the captured expression allows: lazy wrappers
and isbits-valued calls run inline, broadcast materializations, `vcat`, range
`getindex`, and `zeros`/`ones` reuse typed destination buffers, and every
other resolved call becomes its own cache step so registered `apply!!`
coverage (e.g. `mul!`-backed `*`) applies per step. Source shapes outside
that grammar keep the whole-recipe cache step.

The first invocation populates the caches and may allocate. Later invocations
reuse them when the selected operations provide allocation-free `apply!!`
methods for the runtime argument and cache types. MutatingFunctions' generic
fallback preserves semantics but may still allocate, so allocation freedom is
a property of the complete lowered operation set rather than a planner
guarantee.

Every selected recipe must have exactly one output. Each slot retains the first
object returned by its operation: that is fresh kernel-retained storage for
ordinary allocating/registered operations, but it may be a caller-owned input
for aliasing operations. A no-recipe plan returns its input directly. Treat
mutable results as borrowed values that may alias inputs or be overwritten by
the next call; a prepared instance is not reentrant or thread-safe.
"""
function prepare_nonallocating(args...; kwargs...)
    throw(ArgumentError(
        "prepare_nonallocating requires the optional MutatingFunctions extension; " *
        "install MutatingFunctions and load it with `using MutatingFunctions`. " *
        "With the extension loaded it accepts a Plan, a Graph (with have/want), " *
        "or a PreparedKernel"))
end

"Graph values in positional call order."
inputs(k::PreparedKernel) = k.inputs
inputs(k::NonAllocatingKernel) = k.inputs
inputs(p::Plan) = Tuple(p.have)
"Graph values in return order."
outputs(k::PreparedKernel) = k.outputs
outputs(k::NonAllocatingKernel) = k.outputs
outputs(p::Plan) = Tuple(p.want)

"""
    code_expr(p) -> Expr

The generated Julia `Expr` before RGF compilation. Accepts a `Plan` or a
`PreparedKernel`. Useful for asserting that unused operations are literally
absent from the kernel (gist §20).
"""
code_expr(p::Plan) = lower(p)
code_expr(k::PreparedKernel) = k.ast
code_expr(k::NonAllocatingKernel) = k.ast

# --- explanation -----------------------------------------------------------

_opname(op) = try
    n = nameof(op)
    startswith(string(n), "#") ? string(op) : string(n)
catch
    string(op)
end
_opname(::_AuthoredPlateOp) = "plate"
_opname(::_AuthoredScanOp) = "scan"

function _readable_callee(op)
    name = try
        nameof(op)
    catch
        nothing
    end
    name isa Symbol && !startswith(string(name), "#") ? name : :operation
end

function _operation_slot(node)
    node isa Expr && node.head === :ref && length(node.args) == 2 &&
        node.args[1] === _OPS_ARG && node.args[2] isa Int || return nothing
    node.args[2]
end

function _readable_recipe_call(recipe::Recipe, args)
    op = recipe.op
    source = recipe.source
    if !(source isa _NoKernelSource)
        rhs = deepcopy(source)
        bindings = Any[]
        for (input, arg) in zip(recipe.inputs, args)
            input.name === arg && continue
            push!(bindings, Expr(:(=), input.name, arg))
        end
        isempty(bindings) && return rhs
        header = length(bindings) == 1 ? only(bindings) : Expr(:block, bindings...)
        return Expr(:let, header, Expr(:block, rhs))
    end
    Expr(:call, _readable_callee(op), args...)
end

function _readable_expr(node, recipes)
    node isa Expr || return node

    # A BARE operation slot — `__ops__[k]` passed as a value rather than invoked
    # — renders as its readable operation name, not the raw index. The authored
    # plate lowering does this when it seeds the pointwise element type with
    # `Base.promote_op(__ops__[k], …)`; without this branch the reference would
    # survive the readable rewrite as `__ops__[\d+]`.
    let slot = _operation_slot(node)
        slot !== nothing && 1 <= slot <= length(recipes) &&
            return _readable_callee(recipes[slot].op)
    end

    # Ordinary lowered kernels and pure reactive getters invoke an operation
    # slot directly. In-place variants wrap the same slot in cache plumbing;
    # the readable view deliberately shows the authored operation, not that
    # execution-only machinery.
    if node.head === :call && !isempty(node.args)
        slot = _operation_slot(node.args[1])
        if slot !== nothing && 1 <= slot <= length(recipes)
            args = map(arg -> _readable_expr(arg, recipes), node.args[2:end])
            return _readable_recipe_call(recipes[slot], args)
        end
        if node.args[1] === _CACHE_APPLY_ARG && length(node.args) >= 3
            slot = _operation_slot(node.args[3])
            if slot !== nothing && 1 <= slot <= length(recipes)
                args = map(arg -> _readable_expr(arg, recipes), node.args[4:end])
                return _readable_recipe_call(recipes[slot], args)
            end
        end
    end

    args = map(arg -> _readable_expr(arg, recipes), node.args)
    if node.head === :function && !isempty(args)
        signature = args[1]
        if signature isa Expr && signature.head === :tuple
            hidden = (_OPS_ARG, _CACHES_ARG, _CACHE_APPLY_ARG)
            signature = Expr(:tuple, (arg for arg in signature.args
                                      if !(arg isa Symbol && arg in hidden))...)
            args[1] = signature
        end
    end
    Expr(node.head, args...)
end

# Build a display-only copy of a lowered kernel or reactive getter. Positional
# `__ops__[k]` calls become the selected recipe's named operation or its retained
# authored RHS, and hidden operation/cache arguments are removed from the shown
# signature. Passing a PreparedKernel uses its flattened recipe sequence, so a
# nested generated kernel never leaves mismatched or opaque operation slots.
# This internal explanatory expression is not executable authority; `code_expr`
# remains the exact compiled AST.
_readable_expr(ast::Expr, plan::Plan) = _readable_expr(ast, plan.recipes)
_readable_expr(ast::Expr, kernel::PreparedKernel) =
    _readable_expr(ast, kernel.lowered_recipes)

function _recipe_line(r::Recipe)
    ins = join([string(v.name) for v in r.inputs], ", ")
    outs = length(r.outputs) == 1 ? string(r.outputs[1].name) :
           "(" * join([string(v.name) for v in r.outputs], ", ") * ")"
    "$outs = $(_opname(r.op))($ins)"
end

"""
    explain(p::Plan) -> String

Human-readable account of the plan: the have/want boundary, the selected
recipes with costs, the total cost, and the backward-reachable alternatives
that were not selected (gist §16).
"""
function explain(p::Plan)
    io = IOBuffer()
    println(io, "Have:")
    println(io, "  ", isempty(p.have) ? "(none)" : join([string(v.name) for v in p.have], ", "))
    println(io, "Want:")
    println(io, "  ", join([string(v.name) for v in p.want], ", "))
    println(io, "Selected recipes:")
    if isempty(p.recipes)
        println(io, "  (none — all wanted values are already in HAVE)")
    else
        lines = [_recipe_line(r) for r in p.recipes]
        w = maximum(length, lines)
        for (r, l) in zip(p.recipes, lines)
            println(io, "  ", rpad(l, w + 2), "cost ", r.cost)
        end
    end
    selected = Set(r.id for r in p.recipes)
    unused = [r for r in p.candidates if !(r.id in selected)]
    if !isempty(unused)
        println(io, "Alternatives not selected:")
        for r in unused
            println(io, "  ", rpad(_recipe_line(r), 0), "  (cost ", r.cost, ")")
        end
    end
    print(io, "Total graph cost: ", p.cost)
    String(take!(io))
end

Base.show(io::IO, p::Plan) = print(io, explain(p))
function Base.show(io::IO, k::PreparedKernel)
    print(io, "PreparedKernel(", join([string(v.name) for v in k.inputs], ", "),
          " -> ", join([string(v.name) for v in k.outputs], ", "), ")")
end

function Base.show(io::IO, k::NonAllocatingKernel)
    print(io, "NonAllocatingKernel(", join([string(v.name) for v in k.inputs], ", "),
          " -> ", join([string(v.name) for v in k.outputs], ", "), ")")
end
