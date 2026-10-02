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

# The first non-atomic argument that is an axis, as an unrolled conditional
# chain over the argument positions. `_authored_plate_is_axis` is decided by
# the argument's type for every collection the lowering accepts, so the chain
# folds to one argument when the generated body is compiled for concrete types;
# a `findfirst` closure over the argument tuple instead allocated per call
# (304 bytes for a three-port plate whose domain port carries no declared type).
@generated function _authored_plate_marker(::Val{A}, args...) where {A}
    body = :(throw(ArgumentError(
        "an authored plate requires at least one batched argument")))
    for position in reverse(eachindex(args))
        position in A && continue
        body = :(_authored_plate_is_axis(getfield(args, $position)) ?
            getfield(args, $position) : $body)
    end
    body
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
# (broadcast-invariant) indices {1, 3, 4, …}. `I` is the authored
# `include_init` literal: when `true` the result is `[init, output…]`, one
# element longer than the sequences, written into one buffer. `H` is true for
# `history = h0`: then `h0` is the LAST argument (atomic) and the step's last
# formal reads the outputs written so far.
struct _AuthoredScanOp{K,A,I,H}
    kernel::K
end
_AuthoredScanOp{K,A}(kernel::K) where {K,A} = _AuthoredScanOp{K,A,false,false}(kernel)
_scan_has_history(::_AuthoredScanOp{K,A,I,H}) where {K,A,I,H} = H

_scan_includes_init(::_AuthoredScanOp{K,A,I}) where {K,A,I} = I

"The transparent scalar step plan captured by an authored `scan(...) do` recipe."
function scan_body(recipe::Recipe)
    recipe.op isa _AuthoredScanOp || throw(ArgumentError(
        "recipe $(recipe.id) is not an authored scan"))
    recipe.op.kernel.plan
end

# `A` is the atomic (broadcast-invariant) index set over the op's argument tuple
# `(init, operand...)`: index 1 (the carry seed) plus every `Ref`-shared operand.
# The iterated sequences are the remaining operand indices (>= 2, not in `A`).
# Split them at compile time so `_tensorized_scan` receives the sequence tuple and
# the shared tuple explicitly.
@generated function (op::_AuthoredScanOp{K,A,I,H})(args...) where {K,A,I,H}
    n = length(args)
    n >= 2 + H || return :(throw(ArgumentError(
        "an authored scan expects (init, sequence, shared...) arguments")))
    last_shared = H ? n - 1 : n
    iterated = Expr(:tuple, (:(args[$i]) for i in 2:last_shared if !(i in A))...)
    shared = Expr(:tuple, (:(args[$i]) for i in 2:last_shared if i in A)...)
    H && return :(_tensorized_scan_history(op.kernel, args[1], args[$n],
                                           $iterated, $shared))
    :(_tensorized_scan(op.kernel, args[1], $iterated, $shared, Val($I)))
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

# The static type of an inlined scan step's per-step output, as a chain of
# `Base.promote_op` over the step's own operations: the type the first step's
# output would have, without running it. The authored-plate element type uses
# the same chain; unlike inferring the nested PreparedKernel as a whole, each
# operation infers exactly inside the enclosing generated body. `nothing` when
# the step is not one plain operation per recipe (a nested plate, scan or
# embedded kernel inside the step).
function _authored_scan_step_output_type(step, input_types, offset)
    p = step.plan
    length(step.ops) == length(p.recipes) || return nothing
    any(op -> op isa Union{_AuthoredPlateOp,_AuthoredScanOp} ||
        _embedded_kernel(op) !== nothing, step.ops) && return nothing
    types = Dict{Int,Any}(canon_id(p.graph, v.id) => T
                          for (v, T) in zip(p.have, input_types))
    for (i, recipe) in enumerate(p.recipes)
        all(v -> haskey(types, canon_id(p.graph, v.id)), recipe.inputs) ||
            return nothing
        T = Expr(:call, GlobalRef(Base, :promote_op), Expr(:ref, _OPS_ARG, offset + i),
                 (types[canon_id(p.graph, v.id)] for v in recipe.inputs)...)
        if length(recipe.outputs) == 1
            types[canon_id(p.graph, only(recipe.outputs).id)] = T
        else
            for (j, output) in enumerate(recipe.outputs)
                types[canon_id(p.graph, output.id)] =
                    Expr(:call, GlobalRef(Base, :fieldtype), T, j)
            end
        end
    end
    length(p.want) == 2 || return nothing
    get(types, canon_id(p.graph, p.want[2].id), nothing)
end

# Inline the scalar step into the ordered native loop. Calling the nested
# PreparedKernel from a loop with a changing carry defeats inference across the
# RGF boundary; the same step AST and operation table specialize normally here.
# An empty sequence runs no step: the scan port is an empty vector of the
# step's output type and a fused plate consumer sums nothing. An
# `include_init = true` scan writes the carry seed and then each output into
# one buffer one element longer than the sequences (`[init]` when empty).
function _lower_authored_scan_native!(body, op::_AuthoredScanOp, callargs, lhs,
                                      offset; consumer = nothing, recycled = nothing,
                                      pointwise_recycled = nothing)
    _scan_has_history(op) && return _lower_authored_scan_history_native!(
        body, op, callargs, lhs, offset; recycled)
    step = op.kernel
    indices, index, carry, output, output_type, position = gensym.((:scan_indices,
        :scan_index, :scan_carry, :scan_output, :scan_output_type, :scan_position))
    # `A` marks the atomic operands: index 1 (the carry seed) plus the `Ref`
    # shareds.  The iterated sequences are the remaining operand indices; each is
    # indexed by the loop counter per step, while shared operands pass whole.
    atomic = typeof(op).parameters[2]
    iterated_positions = [i for i in 2:length(callargs) if !(i in atomic)]
    shared_positions = [i for i in 2:length(callargs) if i in atomic]
    seqs = Any[callargs[i] for i in iterated_positions]
    xs = first(seqs)                                    # the axis-defining sequence
    arguments = Any[carry]
    # One element of each sequence per step. The index comes from
    # `eachindex(seqs...)`, valid for every sequence, so the read needs no
    # bounds check (the hand loop's `@inbounds`).
    elements = Any[]
    for i in iterated_positions
        element = gensym(:scan_element)
        push!(elements, :($element = $(_inbounds_value(:($(callargs[i])[$index])))))
        push!(arguments, element)
    end
    for i in shared_positions
        push!(arguments, callargs[i])
    end
    step_body = Expr(:block, elements..., _embedded_statements(
        step.ast, arguments, Expr(:tuple, carry, output), offset)...)
    # The step's static output type for an empty sequence: the step's argument
    # types are the carry seed, one element of each sequence and the shared
    # operands, exactly as the first step would receive them.
    typeof_ref, eltype_ref = GlobalRef(Base, :typeof), GlobalRef(Base, :eltype)
    step_input_types = Any[Expr(:call, typeof_ref, callargs[1])]
    for i in iterated_positions
        push!(step_input_types, Expr(:call, eltype_ref, callargs[i]))
    end
    for i in shared_positions
        push!(step_input_types, Expr(:call, typeof_ref, callargs[i]))
    end
    empty_output_type = something(
        _authored_scan_step_output_type(step, step_input_types, offset), Any)
    invariant = Any[]
    initial_output, loop_output, final_output = Expr(:block), Expr(:block), Expr(:block)
    empty_output = Expr(:block, :($output_type = $empty_output_type))
    if lhs !== nothing && _scan_includes_init(op)
        seed, similar_ref = callargs[1], GlobalRef(Base, :similar)
        promote_ref, length_ref = GlobalRef(Base, :promote_type), GlobalRef(Base, :length)
        trajectory_type = :($promote_ref($typeof_ref($seed), $typeof_ref($output)))
        allocation = :($similar_ref($xs, $trajectory_type, $length_ref($indices) + 1))
        push!(initial_output.args, :($lhs = $(recycled === nothing ? allocation :
            _lane_allocation(recycled, trajectory_type,
                :(($(GlobalRef(Base, :OneTo))($length_ref($indices) + 1),)), allocation))))
        push!(initial_output.args, :($lhs[1] = $seed))
        push!(initial_output.args, :($lhs[2] = $output))
        push!(initial_output.args, :($position = 2))
        push!(loop_output.args, :($position += 1))
        # `position` stays within the `length(indices) + 1` buffer.
        push!(loop_output.args, _inbounds_expr(:($lhs[$position] = $output)))
        push!(empty_output.args, :($lhs = $similar_ref($xs,
            $promote_ref($typeof_ref($seed), $output_type), 1)))
        push!(empty_output.args, :($lhs[1] = $seed))
    elseif lhs !== nothing
        allocation = :($(GlobalRef(Base, :similar))($xs, $typeof_ref($output)))
        push!(initial_output.args, :($lhs = $(recycled === nothing ? allocation :
            _lane_allocation(recycled, :($typeof_ref($output)),
                :($(GlobalRef(Base, :axes))($xs)), allocation))))
        push!(initial_output.args, :($lhs[$index] = $output))
        # The buffer shares the sequences' axes, which `index` comes from.
        push!(loop_output.args, _inbounds_expr(:($lhs[$index] = $output)))
        push!(empty_output.args, :($lhs = $(GlobalRef(Base, :similar))(
            $xs, $output_type)))
    end
    if consumer !== nothing
        cell, args, positions, cell_offset, pointwise_lhs, total_lhs = consumer
        cell_output = gensym(:scan_cell)
        cell_args = Any[i in positions ? output : arg for (i, arg) in enumerate(args)]
        invariant, dynamic, cell_type = _authored_scan_cell_statements(
            cell, cell_args, positions, cell_output, cell_offset; output_type)
        append!(step_body.args, dynamic)
        # Match ordinary plate lowering's inferred result type, including
        # heterogeneous cells; the first value alone cannot type that buffer.
        plate_eltype = gensym(:scan_plate_eltype)
        push!(initial_output.args, :($output_type = $typeof_ref($output)))
        for block in (initial_output, empty_output)
            push!(block.args, :($plate_eltype = $cell_type))
        end
        if pointwise_lhs !== nothing
            allocation = :($(GlobalRef(Base, :similar))($xs, $plate_eltype))
            push!(initial_output.args, :($pointwise_lhs = $(
                pointwise_recycled === nothing ? allocation :
                _lane_allocation(pointwise_recycled, plate_eltype,
                    :($(GlobalRef(Base, :axes))($xs)), allocation))))
            push!(empty_output.args, :($pointwise_lhs = $allocation))
            push!(initial_output.args, :($pointwise_lhs[$index] = $cell_output))
            push!(loop_output.args, :($pointwise_lhs[$index] = $cell_output))
            push!(final_output.args, :($pointwise_lhs =
                $(GlobalRef(@__MODULE__, :_narrow_plate_output))($pointwise_lhs)))
        end
        total_zero = :($(GlobalRef(Base, :zero))(
            $plate_eltype === Any ? $output_type : $plate_eltype))
        push!(initial_output.args, :($total_lhs = $total_zero + $cell_output))
        push!(empty_output.args, :($total_lhs = $total_zero))
        push!(loop_output.args, :($total_lhs = $total_lhs + $cell_output))
    end
    # `eachindex(seqs...)` throws `DimensionMismatch` unless every iterated
    # sequence shares axes, giving the lockstep length check for free.
    push!(body.args, :($indices = $(GlobalRef(Base, :eachindex))($(seqs...))))
    append!(body.args, invariant)
    nonempty = Expr(:block)
    push!(nonempty.args, :($carry = $(callargs[1])))
    push!(nonempty.args, :($index = $(GlobalRef(Base, :first))($indices)))
    append!(nonempty.args, step_body.args)
    append!(nonempty.args, initial_output.args)
    push!(nonempty.args, Expr(:for,
        :($index = $(GlobalRef(Base.Iterators, :drop))($indices, 1)),
        Expr(:block, step_body.args..., loop_output.args...)))
    push!(body.args, Expr(:if, :($(GlobalRef(Base, :isempty))($indices)),
                          empty_output, nonempty))
    append!(body.args, final_output.args)
    body
end

# A `history = h0` scan: the result vector is filled with `h0` before the first
# step and each step writes its output into it, so the step's last argument —
# a read-only view of that vector — holds the outputs of the earlier steps and
# `h0` at and after the current one: the in-place hand loop. Its element type
# is `typeof(h0)`; an empty sequence runs no step and yields an empty vector.
function _lower_authored_scan_history_native!(body, op::_AuthoredScanOp, callargs,
                                              lhs, offset; recycled = nothing)
    lhs === nothing && throw(ArgumentError(
        "a `history =` scan materializes its output vector"))
    step = op.kernel
    indices, index, carry, output, earlier = gensym.((:scan_indices,
        :scan_index, :scan_carry, :scan_output, :scan_history))
    atomic = typeof(op).parameters[2]
    fill_value = callargs[end]
    iterated_positions = [i for i in 2:(length(callargs) - 1) if !(i in atomic)]
    shared_positions = [i for i in 2:(length(callargs) - 1) if i in atomic]
    seqs = Any[callargs[i] for i in iterated_positions]
    arguments = Any[carry]
    for i in iterated_positions
        push!(arguments, Expr(:ref, callargs[i], index))
    end
    for i in shared_positions
        push!(arguments, callargs[i])
    end
    push!(arguments, earlier)
    step_body = _embedded_statements(
        step.ast, arguments, Expr(:tuple, carry, output), offset)
    module_ref(name) = GlobalRef(@__MODULE__, name)
    record = Any[
        :($carry = $(module_ref(:_scan_history_carry))($carry)),
        :($lhs[$index] = $output)]
    push!(body.args, :($indices = $(GlobalRef(Base, :eachindex))($(seqs...))))
    push!(body.args, :($lhs = $(module_ref(:_scan_history_buffer))(
        $((recycled === nothing ? () : (recycled,))...), $(first(seqs)), $fill_value)))
    push!(body.args, :($earlier = $(module_ref(:_ScanHistory))($lhs)))
    nonempty = Expr(:block,
        :($carry = $(callargs[1])),
        :($index = $(GlobalRef(Base, :first))($indices)),
        step_body..., record...,
        Expr(:for, :($index = $(GlobalRef(Base.Iterators, :drop))($indices, 1)),
             Expr(:block, step_body..., record...)))
    push!(body.args, Expr(:if,
        :(!$(GlobalRef(Base, :isempty))($indices)), nonempty))
    body
end

function _authored_scan_cell_statements(cell, args, positions, output, offset;
                                        output_type = nothing)
    inner = cell.plan
    length(cell.ops) == length(inner.recipes) || throw(ArgumentError(
        "an authored plate body must lower to one operation per transparent scalar recipe"))
    roots = Set(canon_id(inner.graph, inner.have[i].id) for i in positions)
    dependencies = _plate_dependencies(inner, roots).recipes
    locals = Dict(canon_id(inner.graph, v.id) => arg
                  for (v, arg) in zip(inner.have, args))
    # The scan output's type is supplied by the caller when given, so the empty
    # sequence (which has no output value) types the plate the same way.
    types = Dict{Int,Any}(canon_id(inner.graph, v.id) =>
        (output_type !== nothing && i in positions ? output_type :
         Expr(:call, GlobalRef(Base, :typeof), arg))
        for (i, (v, arg)) in enumerate(zip(inner.have, args)))
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
    # The fused loop streams one cell per step; the seed slot of an
    # init-including scan would need a cell of its own, and a history scan
    # reads its own output vector, so both materialize.
    (_scan_includes_init(scan_recipe.op) || _scan_has_history(scan_recipe.op)) &&
        return nothing
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

# The same decision from the VALUE a call receives. A port without a declared
# type (`observations = domain(plan)`, one graph serving several plan types)
# keeps the scheduling guard above at lowering time, but the generated body is
# compiled for the concrete argument types: this trait is a constant there, so
# a one-dimensional domain drops the guard and its per-coordinate bookkeeping
# exactly as a declared `UnitRange`/`Vector` domain does, while a runtime
# scalar keeps the guard (and its value computed once above the loop).
@inline _plate_unconditional_root(::AbstractVector) = true
@inline _plate_unconditional_root(::Tuple) = true
@inline _plate_unconditional_root(_) = false

function _authored_plate_runtime_unconditional(callargs, roots, positions, atomic)
    tests = Any[Expr(:call, GlobalRef(@__MODULE__, :_plate_unconditional_root),
                     callargs[positions[root]]) for root in sort!(collect(roots))
                if !(positions[root] in atomic)]
    isempty(tests) && return false
    foldl((left, right) -> Expr(:||, left, right), tests)
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

# Dose-outer lowering of a gathered generator-sum cell (`_KernelReduction`).
#
# A cell `sum(w[j] * get(u, t - s[j], 0.0) for j in eachindex(s); init = 0.0)`
# runs observation-outer as authored: per cell, one pass over the doses, each
# testing whether its lag is in range. With the doses and `u` shared by every
# cell, the same sum runs dose-outer — one pass over the cells per dose, each
# cell accumulating `add_sum(acc, term)` in the authored dose order, so every
# cell's value is the authored fold, bitwise — and when the gather index moves
# by exactly one per cell (checked at run time, `_plate_affine_index`), each
# pass splits at the window where the index is in range: a contiguous read of
# `u` without the test there, the default outside. That is the shape of a
# hand-written shifted-slice accumulation loop.
#
# The plan is static: the materialized pointwise result is the reduction
# recipe's output, every other cell recipe is a plate invariant (computed once
# above the loop), and the iterator and gather source read invariants only.
# The types are checked when the body is compiled (`_plate_reduction_ready`);
# otherwise the ordinary cell loop runs.
function _plate_reduction_plan(inner::Plan, inner_kernel, dependencies,
                               root_positions, atomic, callvalues, pointwise_lhs)
    pointwise_lhs === nothing && return nothing
    graph = inner.graph
    want = canon_id(graph, only(inner.want).id)
    dynamic(cid) = Set(root for root in get(dependencies, cid, Set{Int}())
                       if !(root_positions[root] in atomic) &&
                          !(valtype(callvalues[root_positions[root]]) <: Number))
    recipe_index = nothing
    for (position, recipe) in enumerate(inner.recipes)
        length(recipe.outputs) == 1 || return nothing
        output = canon_id(graph, only(recipe.outputs).id)
        if output == want
            recipe_index = position
        elseif !isempty(dynamic(output))
            return nothing
        end
    end
    recipe_index === nothing && return nothing
    op = inner_kernel.ops[recipe_index]
    op isa _KernelSourceOp && op.f isa _KernelReduction || return nothing
    recipe = inner.recipes[recipe_index]
    roots = dynamic(want)
    isempty(roots) && return nothing
    II, XI, KI, AI = typeof(op.f).parameters
    for position in (II..., AI...)
        isempty(dynamic(canon_id(graph, recipe.inputs[position].id))) ||
            return nothing
    end
    (; recipe_index, recipe, roots, iterator = II, init = XI, index = KI,
       source = AI)
end

# Fill a recycled lane buffer (`_lane_reuse`) instead of evaluating
# `allocation` when it fits; `recycled` is a recycle argument of
# `_lower_with_ops`.
function _lane_allocation(recycled, eltype, output_axes, allocation)
    buffer = gensym(:lane_buffer)
    Expr(:block,
        :($buffer = $(GlobalRef(@__MODULE__, :_lane_reuse))(
            $recycled, $eltype, $output_axes)),
        :($buffer === nothing ? $allocation : $buffer))
end

# `@inbounds` for generated code, which carries no macro calls.
_inbounds_expr(ex) = Expr(:block, Expr(:inbounds, true), ex, Expr(:inbounds, :pop))
function _inbounds_value(ex)
    value = gensym(:inbounds_value)
    Expr(:block, Expr(:inbounds, true), Expr(:local, Expr(:(=), value, ex)),
         Expr(:inbounds, :pop), value)
end

function _lower_plate_reduction_native(reduction, inner::Plan, locals, callargs,
                                       callvalues, raw_arguments, prepared_arguments,
                                       atomic, root_positions, plate_type_exprs, op_offset,
                                       plate_eltype, output_axes, pointwise_lhs,
                                       accumulator, cell_loop)
    recipe = reduction.recipe
    graph = inner.graph
    arguments(cell) = Any[_authored_plate_scalar_ref(
        inner, locals, callargs, callvalues, prepared_arguments, atomic,
        input, cell, true) for input in recipe.inputs]
    select(positions, cell) = arguments(cell)[collect(Int, positions)]
    # Argument types for the compile-time check: a HAVE port's per-cell
    # element (or atomic value) type, an invariant local's own type.
    types = Any[haskey(root_positions, canon_id(graph, input.id)) ?
                plate_type_exprs[canon_id(graph, input.id)] :
                Expr(:call, GlobalRef(Base, :typeof), locals[canon_id(graph, input.id)])
                for input in recipe.inputs]
    ready = Expr(:call, GlobalRef(@__MODULE__, :_plate_reduction_ready),
                 :($(Expr(:ref, _OPS_ARG, op_offset + reduction.recipe_index)).f),
                 plate_eltype, Expr(:curly, GlobalRef(Core, :Tuple), types...))
    if !_authored_plate_unconditional_group(
            reduction.roots, root_positions, atomic, callvalues)
        # The cell must run at every coordinate, as it does for a domain value
        # that is a vector or tuple when called.
        ready = Expr(:&&, _authored_plate_runtime_unconditional(
            callargs, reduction.roots, root_positions, atomic), ready)
    end

    parts = gensym(:plate_reduction)
    cells = gensym(:plate_cells)
    count = gensym(:plate_count)
    iterator = gensym(:plate_iterator)
    source = gensym(:plate_source)
    element = gensym(:plate_element)
    index_of = gensym(:plate_index_of)
    affine = gensym(:plate_affine)
    first_index = gensym(:plate_first_index)
    lo = gensym(:plate_lo)
    hi = gensym(:plate_hi)
    position = gensym(:plate_position)
    cell = gensym(:plate_cell)
    entry = :($pointwise_lhs[$cell])
    step(f, extra...) = _inbounds_expr(:($entry = $parts.$f(
        $entry, $element, $(extra...), $(arguments(cell)...))))
    window(range, body) = Expr(:for, Expr(:(=), position, range),
        Expr(:block, Expr(:(=), cell, _inbounds_value(:($cells[$position]))), body))
    every(body) = Expr(:for, Expr(:(=), cell,
        :($(GlobalRef(@__MODULE__, :_plate_cells))($cells))), Expr(:block, body))
    # The index function passed to the out-of-line check (`_PlateCellIndex`).
    # Its per-cell arguments are the raw plate arguments, read at the cell:
    # each must span the plate's axes (no singleton expansion), which also
    # puts every cell in bounds.
    index_arguments = Any[]
    spanning = Any[]
    for position in reduction.index
        cid = canon_id(graph, recipe.inputs[position].id)
        have_index = get(root_positions, cid, nothing)
        if have_index === nothing
            push!(index_arguments, Expr(:call,
                GlobalRef(@__MODULE__, :_PlateSharedArgument), locals[cid]))
        elseif have_index in atomic || valtype(callvalues[have_index]) <: Number
            push!(index_arguments, Expr(:call,
                GlobalRef(@__MODULE__, :_PlateSharedArgument), callargs[have_index]))
        else
            push!(index_arguments, Expr(:call,
                GlobalRef(@__MODULE__, :_PlateCellArgument), raw_arguments[have_index]))
            push!(spanning, raw_arguments[have_index])
        end
    end
    isempty(spanning) || (ready = Expr(:&&, ready, Expr(:call,
        GlobalRef(@__MODULE__, :_plate_spans), output_axes, spanning...)))
    index_function = Expr(:call, GlobalRef(@__MODULE__, :_PlateCellIndex),
        :($parts.index), element, Expr(:tuple, index_arguments...))
    gathered = _inbounds_value(:($source[$first_index + ($position - 1)]))
    passes = quote
        $iterator = $parts.iterator($(select(reduction.iterator, nothing)...))
        $source = $parts.array($(select(reduction.source, nothing)...))
        $(every(_inbounds_expr(:($entry = $parts.init($(select(reduction.init, cell)...))))))
        for $element in $iterator
            $index_of = $index_function
            ($affine, $first_index) =
                $(GlobalRef(@__MODULE__, :_plate_affine_index))($index_of, $cells)
            if $affine
                ($lo, $hi) = $(GlobalRef(@__MODULE__, :_plate_gather_window))(
                    $source, $first_index, $count)
                $(window(:(1:($lo - 1)), step(:step_out)))
                $(window(:($lo:$hi), step(:step_in, gathered)))
                $(window(:(($hi + 1):$count), step(:step_out)))
            else
                $(every(step(:step)))
            end
        end
    end
    interchanged = quote
        $parts = $(Expr(:ref, _OPS_ARG, op_offset + reduction.recipe_index)).f
        $cells = $(GlobalRef(Base, :CartesianIndices))($output_axes)
        $count = length($cells)
        $count > 0 && $passes
    end
    if accumulator !== nothing
        # The total adds the cells in coordinate order, as the cell loop does.
        push!(interchanged.args, every(:($accumulator += $(_inbounds_value(entry)))))
    end
    Expr(:if, ready, interchanged, cell_loop)
end

function _lower_authored_plate_native!(body, runtime_ops, runtime_recipes,
                                       op::_AuthoredPlateOp, callargs, callvalues,
                                       pointwise_lhs, total_lhs; recycled = nothing)
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
    # (`_compose_authored_plates`), so it emits extra pre-loop
    # `_plate_require_axes(combine_axes(...))` calls that an equivalent single
    # `plate` never has — the ONLY codegen difference between the two forms
    # (verified: `code_expr` is otherwise identical modulo gensyms). They
    # validate nothing new (the always-emitted `combined_axes` check below
    # subsumes broadcast-compatibility), so eliding them is a STRUCTURAL CLEANUP
    # that makes a fused chain lower identically to a single plate. (snag
    # `composed-authore` reported a ~1.56× composed-vs-single-plate primal gap
    # under concurrent benchmark load; it is NOT reproducible under controlled
    # conditions and its mechanism is UNLOCATED — this elision is a cleanup, not
    # a proven speedup.) Skip a group ONLY when the `combined_axes` check fully
    # subsumes it — both conditions required:
    #   (a) its operands ⊆ the combined operand set, so broadcast-compatibility
    #       is already validated (the superset's `combine_axes` throws the same
    #       `DimensionMismatch`); AND
    #   (b) at least one operand is a STATICALLY provable ≥1-dim axis
    #       (`_static_plate_axis_class === :axis`), so this sub-plate's own
    #       "at least one batched broadcast axis" guard cannot fire.
    # Without (b) an all-scalar sub-plate (e.g. a fused producer over untyped
    # ports bound to runtime scalars) would wrongly pass on an axis contributed
    # by ANOTHER sub-plate — the `test_authored_plate` `multi`/`unused`
    # rejection guards. A chain over typed/bound array ports (the common case)
    # meets both and drops the redundant checks, matching the single plate.
    combined_positions = Set(position for position in eachindex(raw_arguments)
                             if raw_arguments[position] !== nothing)
    for group in op.axis_checks
        group_positions = Int[position for position in group
                              if raw_arguments[position] !== nothing]
        # Skip ONLY when the combined check subsumes this group — BOTH required:
        # (a) operands ⊆ the combined operand set, and (b) at least one operand
        # is a statically provable ≥1-dim axis. An EMPTY `group_positions` (every
        # original operand filtered out because it is Ref-atomic or a static
        # `Number`) is NOT skippable: this sub-plate has NO batched axis of its
        # own and must still be REJECTED, exactly as the pre-fusion single plate.
        # Both `issubset` (empty ⊆ anything) and `any` (`false` over empty)
        # already give that verdict, so no special-case skip: the group then
        # lowers to a zero-operand `combine_axes()`, which throws (currently a
        # `MethodError` — there is no zero-arg method), rejecting the axis-less
        # sub-plate. (An UNTYPED all-scalar operand instead survives the filter
        # as a 0-dim arg and is rejected one step later by `_plate_require_axes`
        # over an empty axes tuple ⇒ `ArgumentError` — see the `multi`/`unused`
        # tests; both paths reject.) A `continue` here would instead let an
        # axis-less producer silently borrow a sibling's axis (regression caught
        # in review of the first candidate; negative-domain tests below).
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
        allocation = :($(GlobalRef(@__MODULE__, :_plate_similar_output))(
            $marker, $plate_eltype, $output_axes))
        push!(body.args, Expr(:(=), pointwise_lhs, recycled === nothing ? allocation :
            _lane_allocation(recycled, plate_eltype, output_axes, allocation)))
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
            condition = Expr(:||,
                _authored_plate_runtime_unconditional(
                    callargs, roots, root_positions, atomic),
                Expr(:&&, has_axis, Expr(:||, first_coordinate, changed)))
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
    cell_loop = Expr(:for, Expr(:(=), index, iteration), loopbody)
    reduction = _plate_reduction_plan(
        inner, inner_kernel, dependencies, root_positions, atomic, callvalues,
        pointwise_lhs)
    if reduction === nothing
        push!(body.args, cell_loop)
    else
        push!(body.args, _lower_plate_reduction_native(
            reduction, inner, locals, callargs, callvalues, raw_arguments,
            prepared_arguments, atomic, root_positions, plate_type_exprs, op_offset, plate_eltype,
            output_axes, pointwise_lhs, accumulator, cell_loop))
    end
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
                                           op::_AuthoredPlateOp, callargs, callvalues,
                                           pointwise_lhs, total_lhs)
    inner_kernel = op.kernel
    inner = inner_kernel.plan
    length(inner.want) == 1 || throw(ArgumentError(
        "an authored plate body must have exactly one distinguished result"))
    length(inner_kernel.ops) == length(inner.recipes) || throw(ArgumentError(
        "an authored plate body must lower to one operation per transparent scalar recipe"))

    atomic = typeof(op).parameters[2]
    # A cell recipe whose transitive roots are all atomic (`Ref`) arguments or
    # statically scalar ports is loop-invariant: the native lowering evaluates
    # it ONCE above the cell loop (an empty-`dynamic` group in
    # `_lower_authored_plate_native!`). The tensorized body must give its value
    # the same standing. A cell-local ARRAY built from atomic operands only
    # (`shifts = schedule_plan.shifts` read from a `Ref`-wrapped host plan) is
    # one shared value, not a lane axis, so it is evaluated once with the
    # unwrapped atomic operands and re-enters every downstream tensorized plate
    # call wrapped in `Ref`, exactly like an atomic argument. Before this, the
    # bare invariant array met the lane axis in `combine_axes` and the whole
    # plate failed with `DimensionMismatch` under Reactant while the native
    # kernel was correct (snag `plate-cell-gathe-94d4a929`). The distinguished
    # result is never treated as invariant, so a degenerate constant cell keeps
    # its former lowering.
    root_positions = Dict(
        canon_id(inner.graph, input.id) => position
        for (position, input) in enumerate(inner.have))
    dependencies = _plate_dependencies(inner, Set(keys(root_positions)))
    # Statically scalar means the OUTER argument's value type (`callvalues`),
    # exactly as the native lowering decides — never the cell formal's inferred
    # element type, which is scalar for the batched axis itself.
    invariant_root(root) = root_positions[root] in atomic ||
        valtype(callvalues[root_positions[root]]) <: Number
    result_cid = canon_id(inner.graph, only(inner.want).id)
    locals = Dict{Int,Any}()
    shared = Dict{Int,Any}()
    for (index, input) in enumerate(inner.have)
        arg = callargs[index]
        cid = canon_id(inner.graph, input.id)
        locals[cid] = index in atomic ?
            Expr(:call, GlobalRef(Base, :Ref), arg) : arg
        shared[cid] = arg
    end
    op_offset = length(runtime_ops)
    append!(runtime_ops, inner_kernel.ops)
    append!(runtime_recipes, inner_kernel.lowered_recipes)
    for (recipe_index, recipe) in enumerate(inner.recipes)
        length(recipe.outputs) == 1 || throw(ArgumentError(
            "an authored plate currently requires single-output scalar recipes"))
        output = only(recipe.outputs)
        output_cid = canon_id(inner.graph, output.id)
        out = gensym(Symbol(:plate_, output.name))
        operation = Expr(:ref, _OPS_ARG, op_offset + recipe_index)
        if output_cid != result_cid &&
           all(invariant_root, dependencies.recipes[recipe_index])
            args = Any[shared[canon_id(inner.graph, input.id)]
                       for input in recipe.inputs]
            push!(body.args, Expr(:(=), out, Expr(:call, operation, args...)))
            locals[output_cid] = Expr(:call, GlobalRef(Base, :Ref), out)
            shared[output_cid] = out
            continue
        end
        args = Any[locals[canon_id(inner.graph, input.id)] for input in recipe.inputs]
        call = Expr(:call, GlobalRef(@__MODULE__, :_tensorized_plate_call),
                    operation, args...)
        push!(body.args, Expr(:(=), out, call))
        locals[output_cid] = out
    end
    scalar_result = locals[result_cid]
    materialized = Expr(:call,
        GlobalRef(@__MODULE__, :_tensorized_plate_materialize), scalar_result)
    pointwise_lhs === nothing ||
        push!(body.args, :($pointwise_lhs = $materialized))
    total = Expr(:call,
        GlobalRef(@__MODULE__, :_tensorized_plate_sum), scalar_result)
    total_lhs === nothing || push!(body.args, :($total_lhs = $total))
    body
end

# A recipe output authored with a type (`weights::Vector{Float64} = ...`) is
# emitted as a typed local of the NATIVE product: Julia converts the assigned
# value to the declared type (an identity for a value already of that type, a
# loud error for an inconvertible one) and, decisively, keeps the value's type
# known to inference for every consumer. Without it a nested prepared kernel
# called from inside a recipe -- a prepared scan child in a lazy arm, a prepared
# plate child -- re-enters the same generic RK call operators that are already
# on the inference stack (`_KernelSourceOp`, `_kernel_source_call`,
# `_prepared_call`, RGF `generated_callfunc`); Julia's recursion heuristic then
# widens the re-entered signatures and the child's result reaches the parent as
# an abstract type (`Vector`, `Any`) even though the child alone infers exactly.
# Every consumer of that value dispatches dynamically -- inside an embedded plate
# loop once per cell, boxing the coordinate and the result (measured on the
# ShinyRK simulation graph: 2.4x the read time and 3x the bytes of the children's
# sum; snag `embedded-prepare-612e5944`). The declaration restores the authored
# contract at the assignment. Only a product that is host-only by construction
# may declare: the tensorized product binds traced arrays to the same names, and
# a PLAIN prepared kernel (no plate, scan or embedded pair, hence no tensorized
# product) has its native body traced by Reactant directly, where a host-typed
# local would `convert` a traced array. So `_lower_with_ops` declares only in
# the native body of a kernel that `prepare` builds as a native/tensorized pair
# (`_needs_embedded_tensorization`), `lower_batched` in its native loop, and
# the position driver (`_lower_replicated_with_ops`) in both of its parts.
# A bound constant (`_BoundConstant`) is exempt: inference already knows its
# exact value type, and a declaration would only convert a deliberately bound
# view or range into an owning copy. `Any` declares nothing.
function _declare_typed_output!(body::Expr, value::Value, name, declared::Set{Symbol})
    name isa Symbol || return
    name in declared && return
    T = valtype(value)
    T === Any && return
    push!(declared, name)
    push!(body.args, Expr(:local, Expr(:(::), name, T)))
end

# `recycle` lists (value id => argument name) pairs, appended to the signature
# after the HAVE ports: each argument is `nothing` or a buffer that the value's
# authored plate or scan may fill instead of allocating its output (the
# position driver hands back the previous position's lane buffer,
# `_lower_replicated_with_ops`). Any other producer ignores it.
function _lower_with_ops(p::Plan; tensorized::Bool = false,
                         inline_embedded::Bool = true,
                         declare::Bool = !tensorized && inline_embedded &&
                                         _needs_embedded_tensorization(p),
                         recycle::Vector{Pair{Int,Symbol}} = Pair{Int,Symbol}[])
    g = p.graph
    names = _varnames(p)
    nm(v) = names[canon_id(g, v.id)]
    argexprs = Any[_OPS_ARG]
    for v in p.have
        push!(argexprs, :($(nm(v))::$(valtype(v))))
    end
    append!(argexprs, (last(pair) for pair in recycle))
    recycled(cid) = (index = findfirst(pair -> first(pair) == cid, recycle);
                     index === nothing ? nothing : last(recycle[index]))
    body = Expr(:block)
    runtime_ops = Any[]
    runtime_recipes = Recipe[]
    skipped_recipes = Set{Int}()
    pending_scans = Dict{Int,Any}()
    declared = Set{Symbol}()
    declare_types = declare
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
            if declare_types
                pointwise_lhs === nothing ||
                    _declare_typed_output!(body, pointwise, pointwise_lhs, declared)
                total_lhs === nothing || _declare_typed_output!(
                    body, only(sum_recipe.outputs), total_lhs, declared)
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
                    body, scan_recipe.op, scan_args, nothing, scan_offset; consumer,
                    pointwise_recycled = recycled(pointwise_id))
            elseif tensorized
                _lower_authored_plate_tensorized!(
                    body, runtime_ops, runtime_recipes, r.op, callargs, r.inputs,
                    pointwise_lhs, total_lhs)
            else
                _lower_authored_plate_native!(
                    body, runtime_ops, runtime_recipes, r.op, callargs, r.inputs,
                    pointwise_lhs, total_lhs; recycled = recycled(pointwise_id))
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
                declare_types && !(r.op isa _BoundConstant) &&
                    _declare_typed_output!(body, output, nm(output), declared)
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
                    output = only(r.outputs)
                    _lower_authored_scan_native!(body, r.op, callargs, lhs, scan_index;
                        recycled = lhs === nm(output) ?
                            recycled(canon_id(g, output.id)) : nothing)
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
rewritten (`transform`) before compilation (gist §9). Pair it with its
operation table via [`lower_with_ops`](@ref).
"""
lower(p::Plan) = first(_lower_with_ops(p; inline_embedded = false))
_lower_unembedded(p::Plan) = lower(p)

"""
    lower_with_ops(p::Plan; tensorized=false, inline_embedded=false) -> (ast, ops, recipes)

Lower a plan to an executable straight-line function `Expr` plus the operation
table it closes over. `ast` is `function (__ops__, ports...) ... end` over the
plan HAVE ports in order; `ops` is the tuple of callables with `ops[i]`
evaluating the `__ops__[i]` references in the body; `recipes` is the parallel
human-readable recipe metadata.

Evaluate functionally with [`compile`](@ref) (or `eval`) and call with the ops
tuple first:

    ast, ops, _ = lower_with_ops(p)
    f = compile(ast)
    f(ops, port_values...)

`lower(p)` is `first(lower_with_ops(p))`. With `tensorized = true` the body
takes the backend tensorized form over the same operation table; with
`inline_embedded = true` embedded prepared kernels are spliced as statements
(the form [`prepare`](@ref) compiles) instead of opaque `__ops__` calls.
"""
function lower_with_ops(p::Plan; tensorized::Bool = false,
                        inline_embedded::Bool = false)
    _lower_with_ops(
        p; tensorized = tensorized, inline_embedded = inline_embedded)
end

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


_replicated_graph_plan(p::Plan) = all(!recipe.effectful for recipe in p.recipes)

function _replicated_dependency_analysis(p::Plan, batched)
    graph = p.graph
    names = Tuple(batched isa Symbol ? (batched,) : batched)
    isempty(names) && throw(ArgumentError(
        "position batching requires at least one batched HAVE port"))
    all(name -> name isa Symbol, names) || throw(ArgumentError(
        "position batching batched ports must be Symbols; got $(names)"))
    length(unique(names)) == length(names) || throw(ArgumentError(
        "position batching batched ports must be unique; got $(names)"))
    have_by_name = Dict(value.name => value for value in p.have)
    mapped = Value[]
    for name in names
        value = get(have_by_name, name, nothing)
        value === nothing && throw(ArgumentError(
            "position batching port :$name is not in the plan HAVE boundary"))
        push!(mapped, value)
    end
    mapped_ids = Set(canon_id(graph, value.id) for value in mapped)
    dependencies = _plate_dependencies(p, mapped_ids)
    positions = Tuple(
        findfirst(value -> value.name === name, p.have) for name in names)
    input_types = Tuple{(valtype(p.have[index]) for index in positions)...}
    foreach(_replica_expected_rank, input_types.parameters)
    (; mapped = Tuple(mapped), mapped_ids,
       recipe_dependencies = dependencies.recipes, positions, input_types)
end

@inline function _replicated_validate_axes(
        args::Tuple, ::Val{B}, ::Type{BT})::Int where {B,BT}
    replica_count = _replica_batch_count(getfield(args, first(B)), BT.parameters[1])
    for (position, index) in enumerate(B)
        count = _replica_batch_count(getfield(args, index), BT.parameters[position])
        count == replica_count || throw(DimensionMismatch(
            "position-batched ports disagree on batch length; got " *
            "$count and $replica_count"))
    end
    replica_count
end

@inline _replicated_project(arg::AbstractVector, replica_index) = arg[replica_index]
@inline function _replicated_project(
        arg::AbstractArray{T,N}, replica_index) where {T,N}
    copy(selectdim(arg, N, replica_index))
end
# Expand only the record's type-defined fields. Recursing through `map` at
# runtime can widen nested leaf types inside a generated position residual,
# turning the authored arithmetic loops into boxed dynamic dispatch. Array
# lengths and the position count never participate in this structural walk.
function _replicated_record_projection(arg, T, index, lane = nothing)
    (T <: Union{Tuple,NamedTuple} && isconcretetype(T)) || return lane === nothing ?
        :(_replicated_project($arg, $index)) :
        :(_replicated_project!($lane, $arg, $index))
    fields = Any[_replicated_record_projection(
        :(getfield($arg, $i)), fieldtype(T, i), index,
        lane === nothing ? nothing : :(getfield($lane, $i))) for i in 1:fieldcount(T)]
    values = Expr(:tuple, fields...)
    T <: NamedTuple ? :(NamedTuple{$(QuoteNode(fieldnames(T)))}($values)) : values
end
@inline @generated function _replicated_project(arg::T, index) where {T<:Union{Tuple,NamedTuple}}
    isconcretetype(T) ? _replicated_record_projection(:arg, T, :index) :
        :(map(value -> _replicated_project(value, index), arg))
end

# A batched dense numeric array port projects every position into one lane
# buffer of the projection's own type (`Array{T,N-1}`), so the scalar residual
# sees exactly what `_replicated_project` gives it without allocating per
# position. The driver copies every WANT into the stacked result before the
# next position overwrites the lane, and a residual recipe is pure, so no
# position's value is read after its lane is reused. Other leaves keep the
# ordinary projection (`nothing` lane).
_replicated_lane(arg) = nothing
_replicated_lane(arg::Vector{<:Number}) = nothing
_replicated_lane(arg::Array{T,N}) where {T<:Number,N} =
    Array{T,N - 1}(undef, Base.front(size(arg)))
_replicated_lane(arg::Union{Tuple,NamedTuple}) = map(_replicated_lane, arg)

# Dense trailing-axis slices are contiguous. Only expose one when the scalar
# port admits its type; concrete Array ports retain their copying lane.
struct _ReplicatedViewLane end
_replicated_lane(arg, ::Type) = _replicated_lane(arg)
@inline @generated function _replicated_lane(arg::Array{T,N}, ::Type{V}) where {T<:Number,N,V}
    _replicated_column_admitted(Array{T,N}, V) ?
        :(_ReplicatedViewLane()) : :(_replicated_lane(arg))
end
# A borrowed reader keeps its dense lanes between calls (`_borrowed_batch`).
_replicated_lane!(slot, arg) = _replicated_lane(arg)
_replicated_lane!(slot, arg::Vector{<:Number}) = nothing
@inline function _replicated_lane!(slot, arg::Array{T,N}) where {T<:Number,N}
    lane = slot[]
    lane isa Array{T,N - 1} && size(lane) == Base.front(size(arg)) && return lane
    fresh = _replicated_lane(arg)
    slot[] = fresh
    fresh
end
_replicated_lane!(slot, arg, ::Type) = _replicated_lane!(slot, arg)
@inline @generated function _replicated_lane!(slot, arg::Array{T,N}, ::Type{V}) where {T<:Number,N,V}
    _replicated_column_admitted(Array{T,N}, V) ?
        :(_ReplicatedViewLane()) : :(_replicated_lane!(slot, arg))
end
@inline _replicated_project!(::Nothing, arg, index) = _replicated_project(arg, index)
@inline _replicated_project!(::_ReplicatedViewLane, arg::Array{T,N}, index) where {T,N} =
    selectdim(arg, N, index)
@inline function _replicated_project!(lane::Array{T}, arg::Array{T,N}, index) where {T,N}
    count = length(lane)
    copyto!(lane, 1, arg, (index - 1) * count + 1, count)
end
@inline @generated function _replicated_project!(lane::Tuple, arg::T, index) where {T<:Tuple}
    isconcretetype(T) ? _replicated_record_projection(:arg, T, :index, :lane) :
        :(map((item, value) -> _replicated_project!(item, value, index), lane, arg))
end
@inline @generated function _replicated_project!(lane::NamedTuple{K}, arg::T, index) where {K,T<:NamedTuple{K}}
    isconcretetype(T) ? _replicated_record_projection(:arg, T, :index, :lane) :
        :(map((item, value) -> _replicated_project!(item, value, index), lane, arg))
end

# A borrowed reader's first-position scratch. Its producer checks element
# type and shape in `_lane_reuse`, including for undeclared array WANTs.
@inline function _replicated_recycled(slot, ::Type{V}) where {V}
    buffer = slot[]
    buffer isa V ? buffer : nothing
end
_replicated_output_lane(destination, ::Type, index) = nothing
@inline @generated function _replicated_output_lane(destination::Array{T,N}, ::Type{V}, index) where {T,N,V}
    _replicated_column_admitted(Array{T,N}, V) ?
        :(selectdim(destination, $N, index)) : nothing
end

@inline _replicated_same_column(value, destination, index) = false
@inline function _replicated_same_column(value::SubArray, destination, index)
    parent(value) === destination &&
        parentindices(value) == (ntuple(d -> Base.Slice(Base.OneTo(size(destination, d))), ndims(destination) - 1)..., index)
end

@inline function _replicated_output(::Type{T}, replica_count) where {T<:Number}
    Vector{T}(undef, replica_count)
end
@inline function _replicated_output(
        ::Type{A}, replica_count) where {T,N,A<:AbstractArray{T,N}}
    Array{T,N + 1}(undef, ntuple(_ -> 0, N)..., replica_count)
end
@inline function _replicated_output(
        ::Type{A}, replica_count) where {A<:AbstractArray}
    size = ntuple(_ -> 0, ndims(A))
    similar(A, (size..., replica_count))
end

@inline function _replicated_store!(destination, replica_index, value)
    size(selectdim(destination, ndims(destination), replica_index)) == size(value) ||
        throw(DimensionMismatch("position outputs must have the same shape at every position"))
    eltype(destination) === eltype(value) || throw(ArgumentError(
        "position outputs must have the same numeric element type at every position"))
    _replicated_same_column(value, destination, replica_index) && return destination
    copyto!(selectdim(destination, ndims(destination), replica_index), value)
    destination
end
@inline function _replicated_store!(destination::Tuple, index, value::Tuple)
    length(destination) == length(value) || throw(ArgumentError(
        "position outputs must have the same tuple length"))
    map((out, item) -> _replicated_store!(out, index, item), destination, value)
end
@inline function _replicated_store!(destination::NamedTuple{K}, index,
                                    value::NamedTuple{L}) where {K,L}
    K == L || throw(ArgumentError("position outputs must have the same record fields"))
    map((out, item) -> _replicated_store!(out, index, item), destination, value)
end
@inline _replicated_output(value::Number, replica_count) =
    Vector{typeof(value)}(undef, replica_count)
@inline function _replicated_output(value::AbstractArray, replica_count)
    eltype(value) <: Number || throw(ArgumentError(
        "position output arrays must have numeric elements; put record fields in a tuple or named tuple"))
    similar(value, (size(value)..., replica_count))
end
@inline _replicated_output(value::NamedTuple, count) =
    map(item -> _replicated_output(item, count), value)
@inline _replicated_output(value::Tuple, count) =
    map(item -> _replicated_output(item, count), value)
_replicated_output(value, count) = throw(ArgumentError(
    "position outputs must contain numbers, arrays, tuples or named tuples; got $(typeof(value))"))
_replicated_output(::Type{T}, count) where {T} = throw(ArgumentError(
    "an empty position batch needs declared output types; cannot infer $T without a position"))
_replicated_output(::Type{T}, count) where {T<:Tuple} =
    map(type -> _replicated_output(type, count), fieldtypes(T))
_replicated_output(::Type{T}, count) where {T<:NamedTuple} =
    NamedTuple{fieldnames(T)}(map(type -> _replicated_output(type, count), fieldtypes(T)))

# A recipe-free batched HAVE already has the desired stacked layout when its
# leaves are dense numeric arrays. Vectors of scalar records and custom arrays
# still need ordinary projection and stacking; copying their outer container
# would change the output layout or scalar representation.
_replicated_passthrough_compatible(value) = false
_replicated_passthrough_compatible(::Array{T,N}) where {T,N} =
    T <: Number && (N > 1 || (N == 1 && isconcretetype(T)))
_replicated_passthrough_compatible(value::Union{Tuple,NamedTuple}) =
    all(_replicated_passthrough_compatible, value)

_replicated_passthrough_output(value::Array) = copy(value)
_replicated_passthrough_output(value::Union{Tuple,NamedTuple}) =
    map(_replicated_passthrough_output, value)
_replicated_passthrough_reuse(cache, value) = _replicated_passthrough_output(value)
@inline function _replicated_passthrough_reuse(cache, value::Array)
    cache isa typeof(value) && size(cache) == size(value) ?
        copyto!(cache, value) : _replicated_passthrough_output(value)
end
@inline function _replicated_passthrough_reuse(cache::Tuple, value::Tuple)
    length(cache) == length(value) || return _replicated_passthrough_output(value)
    map(_replicated_passthrough_reuse, cache, value)
end
@inline function _replicated_passthrough_reuse(
        cache::NamedTuple{K}, value::NamedTuple{L}) where {K,L}
    K == L || return _replicated_passthrough_output(value)
    map(_replicated_passthrough_reuse, cache, value)
end
@inline function _replicated_passthrough_output!(slot, value)
    output = _replicated_passthrough_reuse(slot[], value)
    slot[] = output
    # Mapping may concretize explicitly abstract NamedTuple fields, just as
    # ordinary projection/stacking does. Its result need not have input's type.
    output
end
@inline function _replicated_passthrough_output!(slot, value::Array{T,N}) where {T,N}
    output = _replicated_passthrough_reuse(slot[], value)::Array{T,N}
    slot[] = output
    output
end
@inline function _replicated_fill!(output, value, count)
    for index in Base.OneTo(count)
        _replicated_store!(output, index, value)
    end
    output
end

"""
    lower_replicated(p::Plan; batched, reuse = false) -> Expr

Lower a scalar plan over a shared trailing **position axis**. Recipes that
transitively depend on a batched HAVE port execute once per position; recipes
independent of every batched port execute once above the loop. This is position
batching, not broadcast data batching: each batched array port is projected to
its scalar-kernel argument at one position.

The plan is split at its batched HAVE ports exactly as `bound=` partial
evaluation splits it at bound ports (`_replicated_parts`): a shared prefix and
a per-position residual. Each part is lowered by the ordinary native scalar
lowering (`_lower_with_ops`) and spliced into the position driver, so authored
plates emit their fused loops, scans inline their step, and embedded kernels
splice, exactly as [`prepare`](@ref) lowers the same recipes. Their internal
runtime control flow stays inside that per-position body. The operation table
the returned function expects is the second result of
`_lower_replicated_with_ops`.
"""
lower_replicated(p::Plan; batched, reuse = false) =
    first(_lower_replicated_with_ops(p; batched, reuse))

# The shared prefix is every recipe computable from the non-batched HAVE ports
# alone (the same first-producer-wins split `bound=` partial evaluation uses,
# `_partial_split`); the residual is the rest. The residual's HAVE boundary is
# the batched ports plus every prefix-owned value it or the WANT list reads,
# so a shared-only WANT passes through the residual unchanged.
function _replicated_parts(p::Plan, analysis)
    g = p.graph
    mapped_have = Value[v for v in p.have if canon_id(g, v.id) in analysis.mapped_ids]
    shared_have = Value[v for v in p.have if !(canon_id(g, v.id) in analysis.mapped_ids)]
    prefix, residual, prefix_owned = _partial_split(
        p, Set(canon_id(g, v.id) for v in shared_have))
    constants = _partial_constants(p, prefix_owned, residual)
    have_ids = Set(canon_id(g, v.id) for v in p.have)
    prefix_want = Value[v for v in constants if !(canon_id(g, v.id) in have_ids)]
    (; prefix = _partial_subplan(p, shared_have, prefix_want, prefix),
       residual = _partial_subplan(p, vcat(mapped_have, constants), p.want, residual))
end

# Lower one part as `prepare` lowers a scalar kernel. Every output is declared,
# as the position driver always did: its body is host-only by construction
# (a traced batch calls the scalar target through `_replica`).
_replicated_part_ast(part::Plan; recycle = Pair{Int,Symbol}[]) =
    _lower_with_ops(_fuse_authored_plate_chains(part); declare = true, recycle)

# A bound constant keeps its exact value type (a bound view stays a view).
function _replicated_declares(p::Plan, value::Value)
    producer = get(p.producer, canon_id(p.graph, value.id), nothing)
    !(producer isa Recipe && producer.op isa _BoundConstant)
end

function _lower_replicated_with_ops(p::Plan; batched, reuse = false)
    _replicated_graph_plan(p) || throw(ArgumentError(
        "position batching requires pure recipes"))
    analysis = _replicated_dependency_analysis(p, batched)
    g = p.graph
    names = _varnames(p)
    nm(v) = names[canon_id(g, v.id)]
    mapped(cid) = cid in analysis.mapped_ids
    parts = _replicated_parts(p, analysis)
    have_ids = Set(canon_id(g, v.id) for v in p.have)
    declared = Set{Symbol}()

    prefix_ast, prefix_ops = if isempty(parts.prefix.want)
        nothing, ()
    else
        ast, ops, _ = _replicated_part_ast(parts.prefix)
        ast, ops
    end
    # A WANT whose residual producer is an authored plate or scan receives the
    # first position's value as scratch (`recycle`, see `_lower_with_ops`).
    # Later positions write into the owned destination column when its type
    # admits a view, or reuse the scratch. A borrowed reader keeps that scratch
    # between calls, with its producer checking element type and axes.
    nout, nhave = length(p.want), length(p.have)
    recycle = Pair{Int,Symbol}[]
    recycle_wants = Int[]
    for (output_index, v) in enumerate(p.want)
        cid = canon_id(g, v.id)
        cid in have_ids && continue
        any(pair -> first(pair) == cid, recycle) && continue
        producer = get(parts.residual.producer, cid, nothing)
        producer isa Recipe &&
            producer.op isa Union{_AuthoredPlateOp,_AuthoredScanOp} || continue
        push!(recycle, cid => gensym(Symbol(v.name, :_recycled)))
        push!(recycle_wants, output_index)
    end
    recycle_vars = Any[gensym(Symbol(p.want[k].name, :_lane)) for k in recycle_wants]
    residual_ast, residual_ops, _ = _replicated_part_ast(parts.residual; recycle)
    residual_offset = length(prefix_ops)
    lane_vars = Dict(canon_id(g, v.id) => gensym(Symbol(v.name, :_lane))
                     for v in parts.residual.have if mapped(canon_id(g, v.id)))

    # One per-position block: project the batched ports into their lanes, then
    # the spliced residual body binds one fresh local per WANT.
    # `_embedded_statements` renames every residual local, so the
    # first-position block and the loop body are independent copies of the
    # same scalar program.
    function position_block!(destination, replica_index, want_vars, recycled)
        callargs = Any[]
        for v in parts.residual.have
            if mapped(canon_id(g, v.id))
                projected = gensym(Symbol(v.name, :_position))
                push!(destination.args, :($projected = $(GlobalRef(@__MODULE__,
                    :_replicated_project!))($(lane_vars[canon_id(g, v.id)]),
                    $(nm(v)), $replica_index)))
                push!(callargs, projected)
            else
                push!(callargs, nm(v))
            end
        end
        append!(callargs, recycled)
        for (v, variable) in zip(p.want, want_vars)
            canon_id(g, v.id) in have_ids && continue
            _replicated_declares(p, v) &&
                _declare_typed_output!(destination, v, variable, declared)
        end
        lhs = length(want_vars) == 1 ? only(want_vars) : Expr(:tuple, want_vars...)
        append!(destination.args,
                _embedded_statements(residual_ast, callargs, lhs, residual_offset))
    end

    argexprs = Any[_OPS_ARG]
    reuse && push!(argexprs, :__output_caches__)
    for v in p.have
        push!(argexprs, nm(v))
    end
    body = Expr(:block)
    runtime_args = Expr(:tuple, (nm(v) for v in p.have)...)
    push!(body.args, :(replica_count = _replicated_validate_axes(
        $runtime_args, Val($(analysis.positions)),
        $(analysis.input_types))))
    empty_outputs = Any[Expr(:call, GlobalRef(@__MODULE__, :_replicated_output),
                            valtype(v), 0) for v in p.want]
    empty_result = length(empty_outputs) == 1 ? only(empty_outputs) :
                   Expr(:tuple, empty_outputs...)
    push!(body.args, :(if replica_count == 0
        return $empty_result
    end))
    if isempty(p.recipes)
        compatible = Expr(:call, GlobalRef(Base, :all),
            GlobalRef(@__MODULE__, :_replicated_passthrough_compatible),
            Expr(:tuple, (nm(v) for v in p.want if mapped(canon_id(g, v.id)))...))
        passthrough_outputs = map(enumerate(p.want)) do (output_index, v)
            value = nm(v)
            if mapped(canon_id(g, v.id))
                reuse ? Expr(:call, GlobalRef(@__MODULE__, :_replicated_passthrough_output!),
                             Expr(:ref, :__output_caches__, output_index), value) :
                        Expr(:call, GlobalRef(@__MODULE__, :_replicated_passthrough_output), value)
            else
                allocation = reuse ?
                    Expr(:call, GlobalRef(@__MODULE__, :_replicated_output!),
                         Expr(:ref, :__output_caches__, output_index), value, :replica_count) :
                    Expr(:call, GlobalRef(@__MODULE__, :_replicated_output), value, :replica_count)
                Expr(:call, GlobalRef(@__MODULE__, :_replicated_fill!),
                     allocation, value, :replica_count)
            end
        end
        passthrough_result = length(p.want) == 1 ? only(passthrough_outputs) :
                             Expr(:tuple, passthrough_outputs...)
        push!(body.args, Expr(:if, compatible, Expr(:return, passthrough_result)))
    end
    if prefix_ast !== nothing
        prefix_names = Any[nm(v) for v in parts.prefix.want]
        for v in parts.prefix.want
            _replicated_declares(p, v) &&
                _declare_typed_output!(body, v, nm(v), declared)
        end
        lhs = length(prefix_names) == 1 ? only(prefix_names) :
              Expr(:tuple, prefix_names...)
        append!(body.args, _embedded_statements(
            prefix_ast, Any[nm(v) for v in parts.prefix.have], lhs, 0))
    end

    # Borrowed cache slots: the stacked outputs, then one lane per HAVE port,
    # then one recycled buffer per WANT (`_borrowed_batch_slot_count`).
    slot(index) = Expr(:ref, :__output_caches__, index)
    for (position, v) in enumerate(p.have)
        lane = get(lane_vars, canon_id(g, v.id), nothing)
        lane === nothing && continue
        allocation = reuse ?
            Expr(:call, GlobalRef(@__MODULE__, :_replicated_lane!),
                 slot(nout + position), nm(v), valtype(v)) :
            Expr(:call, GlobalRef(@__MODULE__, :_replicated_lane), nm(v), valtype(v))
        push!(body.args, Expr(:(=), lane, allocation))
    end
    first_recycled = Any[reuse ?
        :($(GlobalRef(@__MODULE__, :_replicated_recycled))(
            $(slot(nout + nhave + k)), $(valtype(p.want[k])))) :
        nothing for k in recycle_wants]

    output_vars = Dict(canon_id(g, v.id) => gensym(Symbol(v.name, :_batched))
                       for v in p.want)
    first_index = gensym(:replica_index)
    first_vars = Any[gensym(Symbol(v.name, :_first)) for v in p.want]
    push!(body.args, :($first_index = 1))
    position_block!(body, first_index, first_vars, first_recycled)
    for (lane, k) in zip(recycle_vars, recycle_wants)
        push!(body.args, :($lane = $(first_vars[k])))
    end
    for (output_index, v) in enumerate(p.want)
        cid = canon_id(g, v.id)
        value = first_vars[output_index]
        allocation = reuse ?
            Expr(:call, GlobalRef(@__MODULE__, :_replicated_output!),
                 Expr(:ref, :__output_caches__, output_index), value, :replica_count) :
            Expr(:call, GlobalRef(@__MODULE__, :_replicated_output), value, :replica_count)
        push!(body.args, Expr(:(=), output_vars[cid], allocation))
        push!(body.args,
              Expr(:call, GlobalRef(@__MODULE__, :_replicated_store!),
                   output_vars[cid], first_index, value))
    end

    rest_body = Expr(:block)
    rest_vars = Any[gensym(Symbol(v.name, :_position)) for v in p.want]
    rest_recycled = Any[:(let column = $(GlobalRef(@__MODULE__, :_replicated_output_lane))(
        $(output_vars[canon_id(g, p.want[k].id)]), $(valtype(p.want[k])), replica_index)
        column === nothing ? $lane : column
    end) for (lane, k) in zip(recycle_vars, recycle_wants)]
    position_block!(rest_body, :replica_index, rest_vars, rest_recycled)
    for (output_index, v) in enumerate(p.want)
        push!(rest_body.args,
              Expr(:call, GlobalRef(@__MODULE__, :_replicated_store!),
                   output_vars[canon_id(g, v.id)], :replica_index,
                   rest_vars[output_index]))
    end
    # Keep the first lane as scratch across calls, never a view of an escaped
    # output. All later positions use it or their own destination column.
    push!(body.args, Expr(:for,
        Expr(:(=), :replica_index,
             Expr(:call, GlobalRef(Base, :OneTo), :replica_count)),
        Expr(:if, Expr(:call, GlobalRef(Base, :(==)), :replica_index, 1),
             Expr(:block, Expr(:continue)), rest_body)))
    if reuse
        for (lane, k) in zip(recycle_vars, recycle_wants)
            push!(body.args, :($(slot(nout + nhave + k))[] = $lane))
        end
    end
    retval = length(p.want) == 1 ? only(values(output_vars)) :
             Expr(:tuple, (output_vars[canon_id(g, v.id)] for v in p.want)...)
    push!(body.args, Expr(:return, retval))
    Expr(:function, Expr(:tuple, argexprs...), body),
    (prefix_ops..., residual_ops...)
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
    declared = Set{Symbol}()
    for (k, r) in enumerate(p.recipes)
        isempty(analysis.recipe_dependencies[k]) || continue
        out = only(r.outputs)
        cid = canon_id(g, out.id)
        cid in assigned && continue
        push!(assigned, cid)
        r.op isa _BoundConstant || _declare_typed_output!(body, out, nm(out), declared)
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
        if recipe_is_batched[k]
            # A batched recipe maps over the lane axis; every operand that is
            # NOT batched — a shared scalar, or a shared ARRAY port such as the
            # whole unit-response vector a per-dose cell gathers from — is one
            # atomic value per lane. It enters the call wrapped in `Ref`, and
            # the call routes through `_tensorized_plate_call`, whose backend
            # extension batches a shared array payload (`Ops.batch`) instead
            # of expanding it as a broadcast axis. Before this, a shared array
            # HAVE was broadcast as an axis (`DimensionMismatch`, snag
            # `plate-cell-gathe-94d4a929`) while `lower_batched` hoisted it
            # correctly natively.
            shared = Any[
                canon_id(g, inp.id) in batched_vals ? nm(inp) :
                    Expr(:call, GlobalRef(Base, :Ref), nm(inp))
                for inp in r.inputs]
            call = Expr(:call, GlobalRef(@__MODULE__, :_tensorized_plate_call),
                        op, shared...)
        else
            call = Expr(:call, op, args...)
        end
        push!(body.args, Expr(:(=), nm(out), call))
    end

    materialized = Expr(:call,
        GlobalRef(@__MODULE__, :_tensorized_plate_materialize), nm(want))
    retval = if reduce === nothing
        materialized
    elseif reduce === :+
        Expr(:call, GlobalRef(@__MODULE__, :_tensorized_plate_sum), nm(want))
    else
        Expr(:call, GlobalRef(Base, :reduce), reduce, materialized)
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
Prepared callables may be stored as package-level constants: during package
precompilation, the package being compiled owns the generated expression cache.
Global names in the lowered body still resolve in `ReactiveKernels`.
"""
function compile(ast::Expr)
    ast = _canonical_locals(ast)
    cache_module = _generated_function_cache_module()
    cache_module === (@__MODULE__) &&
        return RuntimeGeneratedFunction(@__MODULE__, @__MODULE__, ast)
    Base.invokelatest(RuntimeGeneratedFunctions.init, cache_module)
    # The cache and generated method must share an owner on Julia 1.12: a
    # generator defined in an older dependency cannot read new global bindings.
    # Julia's macro hygiene resolves globals in RK while preserving lexical
    # scopes, rather than guessing which symbols in the body are free names.
    qualified = macroexpand(@__MODULE__, Expr(:macrocall,
        GlobalRef(@__MODULE__, Symbol("@_native_context")), LineNumberNode(0), ast))
    # Hygiene renames every local through the global `gensym` counter again.
    qualified = _canonical_locals(_native_parameter_names(ast, qualified))
    f = Base.invokelatest(RuntimeGeneratedFunction, cache_module, cache_module, qualified)
    # Only functions prepared before the new context method becomes visible
    # need this barrier, e.g. a builder that immediately warms its first kernel.
    applicable(RuntimeGeneratedFunctions.generated_callfunc, f) ? f :
        _PrecompileWarmFunction(f)
end

macro _native_context(ast)
    ast
end

# Lowering mints its scratch locals with `gensym` (plate axes and indices, scan
# carries, the renamed locals of a spliced embedded kernel), and `gensym` draws
# on a process-global counter: two lowerings of one unchanged plan differ in
# exactly those names. `RuntimeGeneratedFunctions` keys its body cache on a
# content hash of the expression and carries that hash in the callable's TYPE,
# so every fresh lowering would otherwise mint a new `RuntimeGeneratedFunction`
# type — and with it a new `PreparedKernel`/`_EmbeddedFunctionPair` constructor
# specialization at `prepare` time plus a new `generated_callfunc` expansion at
# the first call — on EVERY `prepare` of an unchanged graph (snag
# `prepare-with-bou-2b4faf57`: ~15–60 ms of compilation per request-time
# `prepare(...; bound = ...)` of a graph embedding a prepared plate child).
# Alpha-rename the gensym'd locals in first-occurrence order before the body
# reaches the cache: identical lowerings then produce byte-identical bodies,
# one callable type, and one compilation per graph shape. Distinct originals
# map to distinct names (the map is a bijection on the symbols it touches), so
# hygiene is preserved; quoted data and every non-`#` symbol are untouched.
_lowering_gensym(s::Symbol) = (str = String(s); !isempty(str) && str[1] == '#')

function _canonical_locals(ast::Expr)
    names = Dict{Symbol,Symbol}()
    function canonical(s::Symbol)
        get!(names, s) do
            Symbol(replace(String(s), r"#\d+" => ""), '#', length(names) + 1)
        end
    end
    function walk(node)
        node isa Symbol && return _lowering_gensym(node) ? canonical(node) : node
        node isa Expr || return node
        node.head === :quote && return node
        Expr(node.head, map(walk, node.args)...)
    end
    walk(ast)
end

# Preserve the lowered argument names used by operation-table inspection.
# Macro hygiene still owns every local binding and qualifies every global.
function _native_parameter_names(original, qualified)
    original.args[1] isa Expr && original.args[1].head === :tuple &&
        qualified.args[1] isa Expr && qualified.args[1].head === :tuple ||
        return qualified
    names = Dict{Symbol,Symbol}()
    for (old, new) in zip(original.args[1].args, qualified.args[1].args)
        old = old isa Expr && old.head === :(::) ? old.args[1] : old
        new = new isa Expr && new.head === :(::) ? new.args[1] : new
        old isa Symbol && new isa Symbol && (names[new] = old)
    end
    function restore(node)
        node isa Symbol && return get(names, node, node)
        node isa Expr || return node
        node.head === :quote && return node
        Expr(node.head, map(restore, node.args)...)
    end
    restore(qualified)
end

struct _PrecompileWarmFunction{F} <: Function
    f::F
end

@inline function (f::_PrecompileWarmFunction)(args...)
    # During ordinary execution the image's context method is already loaded.
    # Keep latest-world dispatch confined to execution while building an image.
    ccall(:jl_generating_output, Cint, ()) == 0 ? f.f(args...) :
        _precompile_warm_call(f.f, args...)
end

@inline function _precompile_warm_call(f, args...)
    result_type = Core.Compiler.return_type(f, typeof(args))
    # The first call can precede the new method's world. After load, retain
    # the native return type rather than leaking invokelatest's Any result.
    Base.invokelatest(f, args...)::(result_type === Union{} ? Any : result_type)
end

_native_generated_function(f) = f
_native_generated_function(f::_PrecompileWarmFunction) = f.f
RuntimeGeneratedFunctions.get_expression(f::_PrecompileWarmFunction) =
    Base.invokelatest(RuntimeGeneratedFunctions.get_expression, f.f)
RuntimeGeneratedFunctions.drop_expr(f::_PrecompileWarmFunction) =
    _PrecompileWarmFunction(Base.invokelatest(RuntimeGeneratedFunctions.drop_expr, f.f))
@inline RuntimeGeneratedFunctions.generated_callfunc(f::_PrecompileWarmFunction, args...) =
    f(args...)

# RGF caches created in an already-loaded dependency are not part of a
# consumer's package image. Use Julia's actual precompilation target rather
# than the graph's author module: consumers can prepare imported graphs, and
# preparation can run inside helpers or submodules. This is cold-path loader
# state only. Ordinary runtime preparation still uses RK's existing context.
function _generated_function_cache_module()
    (ccall(:jl_generating_output, Cint, ()) == 0 ||
     Base.JLOptions().incremental == 0) && return @__MODULE__
    target = @static if isdefined(Base, :precompilation_target)
        # Julia 1.10 / 1.11.
        Base.precompilation_target
    elseif isdefined(Base, :precompilation_stack)
        # Julia 1.12+ records nested package precompilation in order.
        isempty(Base.precompilation_stack) ? nothing : last(Base.precompilation_stack)
    else
        error("cannot identify the package owning generated functions during precompilation")
    end
    # Non-package output generation has no package-loader target.
    target === nothing && return @__MODULE__
    Base.root_module(target)
end

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
# `_embedded_marker_candidates` admits every HAVE port not provably scalar
# (array-valued and ambiguous/`Any` ones, atomic ones included) as backend
# markers, since only the marker TYPE — never its axis — selects native vs.
# tensorized execution there.
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
# A view or other array wrapper must not hide its traced storage from dispatch.
@inline _embedded_marker_values(x::AbstractArray) = parent(x) === x ? () : (parent(x),)

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

# A prepared kernel is statically untraced. `Recipe.cse_key` provenance tuples
# carry output `Type`s as data (authoring.jl `_kernel_provenance_key`), and
# ReactantCore's `::Type` `is_traced` early-out covers only the 1-arg call
# while the structural recursion is 2-arg, so walking a kernel reaches type
# internals and throws `type DataType has no field var` (upstream ReactantCore
# gap). Kernels are immutable compile-time metadata built before tracing and
# never contain tracers; `@trace` still takes the traced path when a loop's
# DATA operands are traced.
ReactantCore.is_traced(::PreparedKernel) = false
ReactantCore.is_traced(::PreparedKernel, ::Base.IdSet) = false

# Partial evaluation deliberately stores hoisted values in the prepared
# operation tuple so the public residual kernel has only its unbound HAVE
# ports. Compiler and differentiation backends must nevertheless be able to
# receive array-valued constants as explicit inactive operands: capturing them
# in the callable can either turn a whole dataset into source-level literals or
# obscure its read-only activity. This compact call boundary replaces only
# array-containing `_BoundConstant`s with trailing hidden operands. The public
# PreparedKernel is unchanged; ordinary execution and inspection retain the
# residual ABI.
struct _ExternalBoundArraySlot{I} end

struct _ExternalizedBoundArrayCall{F,O,I,H}
    f::F
    ops::O
end

# Build a call to the existing body without the RGF vararg wrapper, which
# packs all operands into one tuple. Keeping readonly structured operands
# beside live storage in that temporary obscures their activity. This changes
# only the call boundary, not the generated model body or its operands.
function _native_body_call_expr(::Type{F}, f, ops, args) where {F}
    if F <: RuntimeGeneratedFunctions.RuntimeGeneratedFunction
        return :(Base.@inline RuntimeGeneratedFunctions.generated_callfunc(
            $f, $ops, $(args...)))
    elseif F <: _PrecompileWarmFunction
        # Preserve latest-world execution while building a consumer image;
        # after loading it, enter the wrapped generated body directly.
        direct = :(Base.@inline RuntimeGeneratedFunctions.generated_callfunc(
            getfield($f, :f), $ops, $(args...)))
        return :(ccall(:jl_generating_output, Cint, ()) == 0 ?
                 $direct : $f($ops, $(args...)))
    end
    :($f($ops, $(args...)))
end

@generated function (call::_ExternalizedBoundArrayCall{F,O,I,H})(
        args::Vararg{Any,N}) where {F,O,I,H,N}
    external_count = length(I)
    public_count = N - external_count
    public_count >= 0 || return :(throw(ArgumentError(
        "externalized bound-array call is missing hidden operands")))
    if H
        forwarded = Any[:(getfield(args,$index)) for index in 1:N]
        return _native_body_call_expr(
            F, :(getfield(call,:f)), :(getfield(call,:ops)), forwarded)
    end
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
    _native_body_call_expr(
        F, :(getfield(call, :f)), Expr(:tuple, operations...), public_args)
end

"""
    _externalize_bound_arrays(kernel; min_elements = 0) -> (call, values)

Return an internal backend call plus the array-containing partial-evaluation
constants it expects as trailing hidden operands. If the kernel contains no
such constants, return the kernel itself and an empty tuple. Scalar and other
small static constants remain in the operation table; `min_elements` lets a
backend keep arrays below that element count in the table as well, so a small
static dataset stays a compiler literal it can fold rather than a runtime
operand it must read. By default every array is externalized. A tuple or named
tuple containing arrays crosses without capturing the arrays in the
differentiated callable, and it crosses when any of its array leaves meets
`min_elements`. When the compiled body reads the operation table at literal
positions, each array leaf crosses as its own operand (depth-first order) and
the body rebuilds the tuple where it was read, with non-array leaves as
literals; otherwise the value crosses as one structured operand.

This is a backend ABI adapter, not a different model boundary: `kernel` keeps
its original public inputs, and `call(public_args..., values...)` is exactly
equivalent to `kernel(public_args...)`.

With `materialize_view_copies`, a bound `SubArray` crosses as an owning copy
(`collect`) of its elements instead of the prebuilt view. A `SubArray`-typed
`Constant` operand defeats static-activity analysis under reverse-mode
automatic differentiation (it
unboxes the parent pointer into an active slot), while an owning array with
identical contents differentiates cleanly (snag plain-enzyme-rev-3dc5d563).
"""
function _externalize_bound_array_call(f, ops;
                                       min_elements::Integer = 0,
                                       materialize_view_copies::Bool = false)
    positions = Tuple(
        index for (index, op) in pairs(ops)
        if op isa _BoundConstant &&
           _has_external_bound_array(op.value, min_elements))
    isempty(positions) && return nothing, ()
    values = Tuple(
        _externalize_bound_value(ops[index].value, materialize_view_copies)
        for index in positions)
    stripped = ntuple(length(ops)) do index
        slot = findfirst(==(index), positions)
        slot === nothing ? ops[index] :
            _ExternalBoundArraySlot{slot}()
    end
    structured = any(value -> value isa Union{Tuple,NamedTuple}, values)
    external_body = structured ?
        _externalize_bound_array_body(f, positions, values) : nothing
    callable = external_body === nothing ? f :
        RuntimeGeneratedFunctions.drop_expr(external_body)
    call = _ExternalizedBoundArrayCall{
        typeof(callable),typeof(stripped),positions,
        external_body !== nothing}(callable, stripped)
    call, external_body === nothing ? values : _bound_array_leaves(values)
end

# A rewritten body takes every array leaf of a structured bound value as its
# own operand and rebuilds the tuple or named tuple where the table was read;
# non-array leaves stay literals of that rebuild. A multi-field structure
# crossing as one operand beside a view of the active input fails native
# Enzyme's static activity analysis, while its leaves cross cleanly (snag
# `interpolated-err-41772e14`).
_bound_array_leaves(values::Tuple) =
    Tuple(Iterators.flatten(map(_bound_array_leaves, values)))
_bound_array_leaves(value::NamedTuple) = _bound_array_leaves(Tuple(value))
_bound_array_leaves(value::AbstractArray) = (value,)
_bound_array_leaves(value) = ()

function _bound_rebuild_expr(value::NamedTuple, ports, next::Ref{Int})
    fields = Any[_bound_rebuild_expr(v, ports, next) for v in value]
    :(NamedTuple{$(keys(value))}(($(fields...),)))
end
_bound_rebuild_expr(value::Tuple, ports, next::Ref{Int}) =
    Expr(:tuple, Any[_bound_rebuild_expr(v, ports, next) for v in value]...)
_bound_rebuild_expr(::AbstractArray, ports, next::Ref{Int}) =
    ports[next[] += 1]
_bound_rebuild_expr(value, ports, next::Ref{Int}) = QuoteNode(value)

# A compiled body whose operation-table accesses are literal can load hidden
# bound operands directly. This changes only its internal argument boundary;
# every loop, branch and arithmetic expression stays intact. Rebuilding an
# array-containing operation tuple beside an active argument can obscure its
# activity before the tuple is optimized away. Source transforms that inspect
# or dynamically access the table keep the reconstruction path above.
_externalize_bound_array_body(f, positions, values) = nothing
function _externalize_bound_array_body(
        f::Union{RuntimeGeneratedFunctions.RuntimeGeneratedFunction,
                 _PrecompileWarmFunction}, positions, values)
    ast = deepcopy(RuntimeGeneratedFunctions.get_expression(f))
    operands = Symbol[]
    loads = Dict{Int,Any}()
    for (index, value) in zip(positions, values)
        ports = [gensym(:_rk_bound_operand)
                 for _ in 1:length(_bound_array_leaves(value))]
        append!(operands, ports)
        loads[index] = _bound_rebuild_expr(value, ports, Ref(0))
    end
    valid = true
    function replace_load(node)
        if node === _OPS_ARG
            valid = false
            return node
        end
        node isa Expr || return node
        node.head === :quote && return node
        if node.head === :call && length(node.args) == 1
            slot = _operation_slot(node.args[1])
            haskey(loads,slot) && return loads[slot]
        end
        slot = _operation_slot(node)
        if slot !== nothing
            haskey(loads,slot) && (valid = false)
            return node
        end
        Expr(node.head,map(replace_load,node.args)...)
    end
    body = replace_load(ast.args[2])
    valid || return nothing
    append!(ast.args[1].args, operands)
    ast.args[2] = body
    compile(ast)
end

_has_external_bound_array(value, min_elements) = false
_has_external_bound_array(value::AbstractArray, min_elements) =
    length(value) >= min_elements
_has_external_bound_array(value::Union{Tuple,NamedTuple}, min_elements) =
    any(v -> _has_external_bound_array(v, min_elements), value)

function _externalize_bound_arrays(kernel::PreparedKernel;
                                   min_elements::Integer = 0,
                                   materialize_view_copies::Bool = false)
    call, values = _externalize_bound_array_call(
        kernel.f, kernel.ops; min_elements, materialize_view_copies)
    call === nothing ? (kernel, ()) : (call, values)
end

# Decide how one bound array crosses the externalized boundary. Owning arrays
# cross whole; with `materialize_view_copies`, a `SubArray` crosses as an
# owning copy with identical contents, axes, and element type. The copy
# snapshots the view at preparation time; bound data is documented as fixed at
# preparation, so unlike the primal's aliasing view this cannot observe a later
# mutation of the parent — the same freeze `prepare_ad` already applies to the
# externalized operand objects themselves.
function _externalize_bound_value(value::AbstractArray,
                                  materialize_view_copies::Bool)
    (value isa SubArray && materialize_view_copies) || return value
    collected = collect(value)
    axes(collected) == axes(value) || return value
    collected
end

_externalize_bound_value(value, materialize_view_copies::Bool) = value
_externalize_bound_value(value::Union{Tuple,NamedTuple},
                         materialize_view_copies::Bool) =
    map(v -> _externalize_bound_value(v, materialize_view_copies), value)

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
        # Backend selection still needs a runtime marker, so admit every active
        # HAVE port the static axis class does not PROVE scalar (`:not_axis`) —
        # including a whole-vector `Ref`-captured parameter that is atomic for
        # broadcasting yet remains a live array HAVE, and an untyped (`Any`)
        # port whose live values are arrays. Only the marker TYPE is consulted
        # (`_batched_call` dispatches native vs. tensorized on it), never its
        # axis, so an admitted port can never be mistaken for the plate axis;
        # the dynamic runtime selection skips scalar values. Only provably
        # scalar live ports (e.g. `::Float64`) still throw below.
        for (position, input) in enumerate(p.have)
            _static_plate_axis_class(valtype(input)) === :not_axis ||
                push!(candidates, position)
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
    marker = _dynamic_tensorized_marker(args)
    _replica_call(k, args, marker === nothing ? getfield(args, first(B)) : marker)
end

inputs(k::ReplicatedKernel) = k.inputs
outputs(k::ReplicatedKernel) = k.outputs
code_expr(k::ReplicatedKernel) = code_expr(k.target)

# Same statically-untraced contract as `PreparedKernel` above: the whole
# object is captured by the `@trace` loop in `_replica_call`.
ReactantCore.is_traced(::ReplicatedKernel) = false
ReactantCore.is_traced(::ReplicatedKernel, ::Base.IdSet) = false

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

# Known numeric ranks retain their validation. Untyped and record boundaries
# derive their layout from the runtime container, without narrowing the scalar
# graph's HAVE types or inventing a second mathematical graph.
_replica_expected_rank(::Type{T}) where {T<:Number} = 1
_replica_expected_rank(::Type{<:AbstractArray{T,N}}) where {T,N} = N + 1
_replica_expected_rank(::Type{<:AbstractArray}) = nothing
_replica_expected_rank(::Type{Any}) = nothing
_replica_expected_rank(::Type{T}) where {T<:Union{Tuple,NamedTuple}} = nothing
_replica_expected_rank(::Type{T}) where {T} = throw(ArgumentError(
    "position-batched ports must be numeric, tuple or named-tuple values; got $T"))

@inline function _replica_batch_count(arg::AbstractArray, ::Type{T}) where {T}
    expected = _replica_expected_rank(T)
    ndims(arg) > 0 || throw(DimensionMismatch("a position batch needs a trailing axis"))
    expected === nothing || ndims(arg) == expected || throw(DimensionMismatch(
        "position-batched input has rank $(ndims(arg)); expected $expected"))
    Base.require_one_based_indexing(arg)
    size(arg, ndims(arg))
end
@inline _replica_field_types(::Type{Any}, arg) = map(_ -> Any, arg)
@inline _replica_field_types(::Type{T}, arg::Tuple) where {T<:Tuple} =
    isconcretetype(T) ? fieldtypes(T) : map(_ -> Any, arg)
@inline function _replica_field_types(::Type{T}, arg::NamedTuple{K}) where
        {K,T<:NamedTuple}
    isconcretetype(T) || return map(_ -> Any, arg)
    fieldnames(T) == K || throw(ArgumentError("position record fields differ from the scalar port"))
    NamedTuple{K}(fieldtypes(T))
end
@inline function _replica_batch_count(arg::Union{Tuple,NamedTuple}, ::Type{T}) where {T}
    isempty(arg) && throw(ArgumentError("a position record needs at least one batched leaf"))
    types = _replica_field_types(T, arg)
    length(types) == length(arg) || throw(ArgumentError(
        "position tuple length differs from the scalar port"))
    counts = map(_replica_batch_count, arg, types)
    count = first(counts)
    all(==(count), counts) || throw(DimensionMismatch(
        "position record leaves disagree on batch length"))
    count
end
_replica_batch_count(arg, ::Type{T}) where {T} = throw(ArgumentError(
    "position batches need arrays or tuple/named-tuple trees of arrays; got $(typeof(arg))"))

_replica_output_type(::Type{Any}) = true
_replica_output_type(::Type{T}) where {T<:Union{Number,AbstractArray}} = true
_replica_output_type(::Type{T}) where {T<:Union{Tuple,NamedTuple}} =
    !isconcretetype(T) || all(_replica_output_type, fieldtypes(T))
_replica_output_type(::Type) = false

function _replica_batch_indices(boundary, batched)
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
    Tuple(sort!(collect(indices)))
end

function _replica(target, batched)
    boundary = inputs(target)
    indices = _replica_batch_indices(boundary, batched)
    input_types = Tuple{(valtype(boundary[i]) for i in indices)...}
    foreach(_replica_expected_rank, input_types.parameters)
    output_types = Tuple{(valtype(value) for value in outputs(target))...}
    all(_replica_output_type, output_types.parameters) ||
        throw(ArgumentError(
            "replica outputs must be numeric or tuple/named-tuple trees; got $(output_types.parameters)"))
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

"""
    GraphReplicatedKernel

Native position-batched lowering for a straight-line scalar plan. Shared-only
recipes are evaluated once above the position loop; recipes depending on a
batched HAVE port execute once per position. The scalar `PreparedKernel` is
retained as the mathematical authority and as the fallback for array-compiler
batching.
"""
struct GraphReplicatedKernel{B,BT,OT,F,O,P,K,IN,OUT}
    target::K
    native::F
    ops::O
    plan::P
    ast::Expr
    inputs::IN
    outputs::OUT
end

function _replicated_backend_call end

@inline function (k::GraphReplicatedKernel{B})(args...) where {B}
    length(args) == length(k.inputs) || throw(MethodError(k, args))
    if _dynamic_tensorized_marker(args) !== nothing
        return _replicated_backend_call(k, args)
    end
    k.native(k.ops, args...)
end

inputs(k::GraphReplicatedKernel) = k.inputs
outputs(k::GraphReplicatedKernel) = k.outputs
code_expr(k::GraphReplicatedKernel) = k.ast
plan(k::GraphReplicatedKernel) = k.plan

function Base.show(io::IO, k::GraphReplicatedKernel{B}) where {B}
    names = Tuple(k.inputs[i].name for i in B)
    print(io, "GraphReplicatedKernel(batched=", names, ", target=")
    show(io, k.target)
    print(io, ")")
end

function _replica_graph(target::PreparedKernel, batched)
    indices = _replica_batch_indices(inputs(target), batched)
    _replicated_graph_plan(target.plan) || throw(ArgumentError(
        "scalar kernel requires the complete-callable replica fallback"))
    ast, ops = _lower_replicated_with_ops(target.plan; batched)
    native = compile(ast)
    boundary = inputs(target)
    input_types = Tuple{(valtype(boundary[i]) for i in indices)...}
    output_types = Tuple{(valtype(value) for value in outputs(target))...}
    GraphReplicatedKernel{indices,input_types,output_types,
                          typeof(native),typeof(ops),typeof(target.plan),
                          typeof(target),typeof(boundary),
                          typeof(outputs(target))}(
        target, native, ops, target.plan, ast,
        boundary, outputs(target))
end

function replica_graph(kernel::PreparedKernel; batched)
    _replica_graph(kernel, batched)
end

# Reuse only the final stacked buffers. Intermediate allocation remains the
# scalar recipe's responsibility, and the owning surface has no mutable cache.
_replicated_reuse(cache, value, count) = _replicated_output(value, count)
@inline function _replicated_reuse(cache::AbstractArray, value::Number, count)
    cache isa Vector{typeof(value)} && length(cache) == count ?
        cache : _replicated_output(value, count)
end
@inline function _replicated_reuse(cache::AbstractArray, value::Array{T,N}, count) where {T,N}
    cache isa Array{T,N+1} && size(cache) == (size(value)..., count) ?
        cache : _replicated_output(value, count)
end
@inline function _replicated_reuse(cache::AbstractArray, value::SubArray{T,N,P}, count) where {T,N,P<:Array}
    cache isa Array{T,N+1} && size(cache) == (size(value)..., count) ?
        cache : _replicated_output(value, count)
end
@inline function _replicated_reuse(cache::Tuple, value::Tuple, count)
    length(cache) == length(value) || return _replicated_output(value, count)
    map((out, item) -> _replicated_reuse(out, item, count), cache, value)
end
@inline function _replicated_reuse(cache::NamedTuple{K}, value::NamedTuple{L}, count) where {K,L}
    K == L || return _replicated_output(value, count)
    map((out, item) -> _replicated_reuse(out, item, count), cache, value)
end
_replicated_destination_type(::Type) = Any
_replicated_destination_type(::Type{T}) where {T<:Number} =
    isconcretetype(T) ? Vector{T} : Any
_replicated_destination_type(::Type{Array{T,N}}) where {T,N} = Array{T,N+1}
_replicated_destination_type(::Type{<:SubArray{T,N,P}}) where {T,N,P<:Array} = Array{T,N+1}
function _replicated_destination_type(::Type{T}) where {T<:Tuple}
    isconcretetype(T) || return Any
    Tuple{map(_replicated_destination_type, fieldtypes(T))...}
end
function _replicated_destination_type(::Type{T}) where {T<:NamedTuple}
    isconcretetype(T) || return Any
    leaves = Tuple{map(_replicated_destination_type, fieldtypes(T))...}
    # NamedTuple's tuple parameter is invariant: a predicted Any field would
    # reject a concrete custom-array destination. Keep such records broad.
    isconcretetype(leaves) ? NamedTuple{fieldnames(T),leaves} : Any
end

@inline @generated function _replicated_output!(slot, value::V, count) where {V}
    # Only the scalar value's structural type determines this assertion, never
    # a data length or shape. Cache slots still accept later shapes/types, and
    # custom `similar` results retain the broad fallback. Emit the type itself
    # so nested native records stay concrete before the retained position loop.
    destination = _replicated_destination_type(V)
    quote
        output = _replicated_reuse(slot[], value, count)
        slot[] = output
        output::$destination
    end
end

_replicated_alias(cache::AbstractArray, arg::AbstractArray) =
    Base.mightalias(cache, arg) ||
    (!(eltype(arg) <: Number) && any(item -> _replicated_alias(cache, item), arg))
_replicated_alias(cache::AbstractArray, arg::Union{Tuple,NamedTuple}) =
    any(item -> _replicated_alias(cache, item), arg)
_replicated_alias(cache, arg) = false
_replicated_aliases(cache::AbstractArray, args) =
    any(arg -> _replicated_alias(cache, arg), args)
_replicated_aliases(cache::Union{Tuple,NamedTuple}, args) =
    any(item -> _replicated_aliases(item, args), cache)
_replicated_aliases(cache, args) = false

struct BorrowedBatchedKernel{K,F,O,C}
    target::K
    native::F
    ops::O
    caches::C
    ast::Expr
end
# One slot per stacked output, then one input lane per HAVE port and one
# recycled lane buffer per WANT (`_lower_replicated_with_ops`).
_borrowed_batch_slot_count(target) =
    2 * length(outputs(target)) + length(inputs(target))
_borrowed_batch_caches(count::Int) = ntuple(_ -> Ref{Any}(nothing), count)
function _borrowed_batch(target::GraphReplicatedKernel)
    ast, ops = _lower_replicated_with_ops(
        target.plan; batched=batched_ports(target), reuse=true)
    BorrowedBatchedKernel(target, compile(ast), ops,
        _borrowed_batch_caches(_borrowed_batch_slot_count(target)), ast)
end

# A new native execution instance shares the read-only computation, not the
# buffers of an earlier call. No planning, lowering or compilation occurs.
Base.copy(kernel::BorrowedBatchedKernel) =
    BorrowedBatchedKernel(kernel.target, kernel.native, kernel.ops,
                         _borrowed_batch_caches(length(kernel.caches)), kernel.ast)

@inline function (kernel::BorrowedBatchedKernel)(args::Vararg{Any,N}) where {N}
    length(args) == length(inputs(kernel)) || throw(MethodError(kernel, args))
    _dynamic_tensorized_marker(args) === nothing || throw(ArgumentError(
        "reuse=true borrows native output buffers; compile the owning batch instead"))
    for index in 1:length(outputs(kernel))
        # A caller may feed an earlier borrowed output into the next call.
        # Detach in that case so no input is modified through an output alias.
        # The lane slots after the outputs never leave the call.
        slot = kernel.caches[index]
        _replicated_aliases(slot[], args) && (slot[] = nothing)
    end
    kernel.native(kernel.ops, kernel.caches, args...)
end
inputs(kernel::BorrowedBatchedKernel) = inputs(kernel.target)
outputs(kernel::BorrowedBatchedKernel) = outputs(kernel.target)
plan(kernel::BorrowedBatchedKernel) = plan(kernel.target)
code_expr(kernel::BorrowedBatchedKernel) = kernel.ast

@inline _replica_native_arg(arg, ::Type{T}, replica_index) where {T<:Number} =
    arg[replica_index]
@inline _replica_native_arg(arg, ::Type{T}, replica_index) where {T<:AbstractArray} =
    copy(selectdim(arg, ndims(arg), replica_index))
@inline _replica_native_arg(arg, ::Type{T}, index) where {T} =
    _replicated_project(arg, index)

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
function _replica_stack(values, ::Type{T}) where {T}
    isempty(values) && return _replicated_output(T, 0)
    result = _replicated_output(first(values), length(values))
    for (index, value) in enumerate(values)
        _replicated_store!(result, index, value)
    end
    result
end

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
    replica_count = _replicated_validate_axes(args, Val(B), BT)
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

# Emit positional arguments explicitly: splatting the captured `args` tuple
# into the RGF call allocates (one tuple box per call) even though the emitted
# program itself is allocation-free. Keep the public call nongenerated so
# reflection over it continues to accept abstract argument types.
@generated function _nonallocating_call(
        k::NonAllocatingKernel, args::A, ::Val{N}) where {A<:Tuple,N}
    positional = [:(getfield(args, $index)) for index in 1:N]
    :(k.f($(positional...)))
end

@inline function (k::NonAllocatingKernel{F,O,C,A,IN,OUT})(
        args::Vararg{Any,N}) where {F,O,C,A,IN,OUT,N}
    N == fieldcount(IN) || throw(MethodError(k, args))
    _nonallocating_call(k, args, Val(N))
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

Preparing a `Graph` (or an authored `KernelSpec`) with a non-empty `bound`
reuses everything that does not depend on the bound values — the plan, the
compiled data-only prefix and the compiled residual — from earlier bindings of
the same graph version, HAVE/WANT boundary, bound port set and passes, exactly
as [`prepare!`](@ref) does with a [`PreparationCache`](@ref). A binding of new
values then costs the prefix's execution, not a fresh planning, lowering and
compilation. The graph holds these entries, so they live as long as the graph,
a mutation of the graph starts afresh, and they keep no bound values. Passes
that are not singletons (closures) are not retained: such a binding is
prepared afresh. A `Plan` is always prepared afresh.

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
    _reuses_bound_preparation(bound, passes) &&
        return _graph_bound_preparation(g, have, want, passes, bound)
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

# A `_BoundConstant` recipe carries a value that `partial_evaluation` computed
# once at bind time and baked into the residual kernel (its authored `source` was
# consumed by the hoisted prefix, so it renders with no source and no callable
# `nameof`). The readable view shows that value as the literal constant it is, so
# a `bound=` kernel reads like the unbound kernel with its data-only bindings
# folded: a scalar binding shows the same literal it would unbound, and a hoisted
# data port shows the bound data. Display-only, like the rest of this renderer;
# `code_expr` still lowers the constant through the executable `__ops__` slot.
_readable_bound_constant(value) = value

function _readable_recipe_call(recipe::Recipe, args)
    op = recipe.op
    op isa _BoundConstant && return _readable_bound_constant(op.value)
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
    # Synthesized tuple unpacks read as field access, not as accessor calls.
    if length(r.inputs) == 1
        suffix = _unpack_access_suffix(r.op)
        suffix !== nothing && return "$outs = $(only(r.inputs).name)$suffix"
    end
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
