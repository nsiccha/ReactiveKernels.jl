module ReactiveKernelsMutatingFunctionsExt

using ReactiveKernels
# The name the core's body entry (`ReactiveKernels._native_body_call_expr`)
# emits, for the nested plate driver below.
using ReactiveKernels: RuntimeGeneratedFunctions
import LinearAlgebra
import MutatingFunctions

mutable struct _NestedPlateCache{K,R}
    kernel::K
    result::R
end

function ReactiveKernels._nested_plate_cache(op::ReactiveKernels._AuthoredPlateOp{K,A},
        argtypes, ::Type{T}, N) where {K,A,T<:AbstractArray}
    isconcretetype(T) && N !== nothing || return nothing
    celltypes = Tuple((i in A || argtypes[i] <: Number) ? argtypes[i] :
        eltype(argtypes[i]) for i in eachindex(argtypes))
    all(isconcretetype, celltypes) || return nothing
    p = op.kernel.plan
    all(r -> length(r.outputs) == 1, p.recipes) || return nothing
    cell = ReactiveKernels._prepare_nonallocating(p, cache_apply!;
        have_types=celltypes)
    # The plate keeps an owned copy of each cell's result (`cache_apply!`), so
    # its elements have the copy's type: a cell that reads a slice in place
    # returns a `SubArray`, which the plate stores as the `Vector` it copies to.
    R = Base.promote_op(copy, T)
    isconcretetype(R) || return nothing
    result = Array{R,N}(undef, ntuple(_ -> 0, N)...)
    Ref(_NestedPlateCache(cell, result))
end

@generated function cache_apply!(slot::Base.RefValue{C},
        op::ReactiveKernels._AuthoredPlateOp{K,A}, args::Vararg{Any,N}) where {C<:_NestedPlateCache,K,A,N}
    wrapped = gensym.([Symbol(:plate_operand_, i) for i in 1:N])
    bindings = [:($(wrapped[i]) = args[$i]) for i in 1:N]
    # Ref operands have no broadcast axes and enter the cell unchanged.
    # Fold the remaining axes explicitly: a wide varargs combine_axes call
    # loses inference on Julia 1.10 and makes those temporary Refs escape.
    combined = foldl((ax, i) -> :(Base.Broadcast.broadcast_shape(
        $ax, Base.axes($(wrapped[i])))), (i for i in 1:N if !(i in A)); init=:(()))
    index = gensym(:plate_index)
    cellargs = [i in A ? wrapped[i] :
        :(Base.Broadcast._broadcast_getindex($(wrapped[i]), $index)) for i in 1:N]
    # Enter the cell kernel's body directly, as `_nonallocating_call` does, not
    # through the kernel's call: the enclosing kernel's own `_nonallocating_call`
    # would otherwise reach itself through this cell. Inference does not cache a
    # method in such a cycle, and Julia 1.13 then calls the enclosing
    # `_nonallocating_call` through boxed arguments: a copy of the whole kernel
    # and its argument tuple on every call (832 B for a plate of scan cells).
    # The cell kernel is prepared over exactly these operands.
    cell = gensym(:cell_kernel)
    F = fieldtype(fieldtype(C, :kernel), :f)
    body = ReactiveKernels._native_body_call_expr(F, :(getfield($cell, :f)),
        Any[:(getfield($cell, :tables)), cellargs...])
    quote
        $(Expr(:meta, :inline))
        $(bindings...)
        cache = slot[]
        combined_axes = $combined
        isempty(combined_axes) && throw(ArgumentError(
            "an authored plate requires at least one non-Ref batched argument"))
        result = _authored_plate_array_cache(cache.result, combined_axes)
        $cell = cache.kernel
        for $index in CartesianIndices(combined_axes)
            value = $body
            if isassigned(result, $index) && axes(result[$index]) == axes(value)
                copyto!(result[$index], value)
            else
                result[$index] = copy(value)
            end
        end
        cache.result = result
        result
    end
end

@inline function cache_apply!(slot::Base.RefValue{S}, op, args...) where {S}
    R = ReactiveKernels._nonalloc_step_type(S, op, args...)
    R isa Type && isconcretetype(R) && return _cache_apply_as!(slot, R, op, args...)
    slot[] = MutatingFunctions.apply!!(slot[], op, args...)
end

# A slot whose value type preparation could not infer (`Ref{Any}`) is read as
# the type the step's own call infers here, `R`: its buffer is offered to
# `apply!!` once it is an `R`, and the result keeps that type. Read back as
# `Any`, the result reached every later use, each plate cell's included, as an
# untyped value that dispatched at run time and boxed its numbers.
@inline function _cache_apply_as!(slot::Base.RefValue, ::Type{R}, op,
                                  args...) where {R}
    cache = slot[]
    result = cache isa R ? MutatingFunctions.apply!!(cache, op, args...) :
        MutatingFunctions.apply!!(nothing, op, args...)
    slot[] = result
    result
end

# A step whose result is a tuple of a buffer and numbers, such as RKPPL's
# simplex and its log-Jacobian, keeps that tuple in a `Union{Nothing,T}`
# slot, which stores it boxed: storing every call's result would allocate the
# box every call. The slot is the destination offered to `apply!!`, which
# reads its buffers, never its numbers, so a result that kept the slot's
# buffers is not stored again. The differentiated driver below never stores.
@inline function cache_apply!(slot::Base.RefValue{Union{Nothing,T}}, op,
                              args...) where {T<:Tuple}
    cache = slot[]
    result = MutatingFunctions.apply!!(cache, op, args...)
    _keeps_buffers(cache, result) || (slot[] = result)
    result
end

@inline _keeps_buffers(cache, result) = false
@inline _keeps_buffers(cache::T, result::T) where {T<:Tuple} =
    all(map((c, r) -> isbits(c) || c === r, cache, result))

# The differentiated program's driver (`ReactiveKernels._ad_na_cache_driver`):
# its slots are seeded and read-only. `apply!!` fills the slot's buffer when
# the result fits it and returns any other result, which is not kept. A nested
# plate's slot holds the cell kernel and its arrays, and keeps its driver.
@inline cache_fill!(slot::Base.RefValue, op, args...) =
    MutatingFunctions.apply!!(slot[], op, args...)
@inline cache_fill!(slot::Base.RefValue{<:_NestedPlateCache},
                    op::ReactiveKernels._AuthoredPlateOp, args...) =
    cache_apply!(slot, op, args...)
ReactiveKernels._ad_na_cache_driver(::typeof(cache_apply!)) = cache_fill!

# The reactive getter injects a `cache_apply(cache, op, args...) -> newcache`
# helper that takes the slot VALUE (not the Ref) and returns the value the
# getter stores back. `MutatingFunctions.apply!!` already has that exact
# contract — including immutable/isbits passthrough — so it IS the reactive hook.
@inline reactive_cache_apply(cache, op, args...) =
    MutatingFunctions.apply!!(cache, op, args...)

# An authored plate's pointwise result is one ordinary single-output recipe.
# Reuse that recipe's borrowed cache after the first call while preserving the
# exact broadcast/Ref argument semantics used by allocating execution.
@inline function _apply_authored_plate!(
        cache, op::ReactiveKernels._AuthoredPlateOp{K,A}, args...) where {K,A}
    wrapped = ReactiveKernels._authored_plate_arguments(Val(A), args...)
    combined_axes = Base.Broadcast.combine_axes(wrapped...)
    isempty(combined_axes) && throw(ArgumentError(
        "an authored plate requires at least one non-Ref batched argument"))
    # Materialize into the cache's own element type. The cache slot is typed by
    # `_authored_plate_result_eltype` at preparation (a concrete element type for
    # an inferable body, `Any` only when genuinely uninferrable), so this trusts
    # that inferred type instead of the plan-level `valtype(output)`, which is
    # `Any` for any unannotated plate body and would box a `Vector{Any}` and
    # disagree bit-for-bit with the ordinary native lowering's typed buffer.
    output_type = eltype(cache)
    result = if cache isa Array{output_type}
        _authored_plate_array_cache(cache, combined_axes)
    elseif Base.axes(cache) == combined_axes && eltype(cache) == output_type
        cache
    else
        similar(cache, output_type, combined_axes)
    end
    Base.Broadcast.broadcast!(op.kernel, result, wrapped...)
    result
end

@inline function _authored_plate_array_cache(
        cache::Array{T}, combined_axes::Tuple{Vararg{Any,N}}) where {T,N}
    result = cache isa Array{T,N} && Base.axes(cache) == combined_axes ?
             cache : similar(cache, T, combined_axes)
    result::Array{T,N}
end


@inline MutatingFunctions.apply!!(
    cache::AbstractArray, op::ReactiveKernels._AuthoredPlateOp, args...) =
        _apply_authored_plate!(cache, op, args...)

# The generated nonallocating program loads the plate cache through its slot.
# Preserve the concrete dimensionality recovered from the broadcast axes before
# storing it back; routing through the generic Union/Array cache helper boxes
# this embedded operation even after the cache has been seeded. The slot may be
# rank-concrete (`Ref{Vector{T}}`) when the batched argument types were
# statically known at preparation, or the unranked `Ref{Array{T}}` fallback.
@inline function cache_apply!(
        slot::Base.RefValue{A},
        op::ReactiveKernels._AuthoredPlateOp, args...) where {A<:Array}
    result = _apply_authored_plate!(slot[], op, args...)
    slot[] = result
    result
end

# Fixed-arity plate steps: the slurped `args...` tuple above round-trips
# through `ntuple` construction and two splats, which boxes one tuple per call
# on Julia 1.10 even though every element is stack-allocatable. Bind every
# argument explicitly for the arities programs actually emit; the varargs
# methods stay as the fallback for degenerate (0-argument, which throws) and
# very wide plates. The Array branch still refines through
# `_authored_plate_array_cache`, which recovers the rank from the inferred
# axes-tuple type: an untyped batched port seeds the unranked `Ref{Array{T}}`
# slot, and `ndims` on that rank-unknown cache dynamically dispatches (and
# leaves `broadcast!` rank-unknown), allocating hundreds of bytes per call.
for _PLATE_N in 1:8
    _a = [Symbol(:_plate_a_, i) for i in 1:_PLATE_N]
    _w = [Symbol(:_plate_w_, i) for i in 1:_PLATE_N]
    _binds = [quote
        $(_w[i]) = ReactiveKernels._authored_plate_argument(Val(A), $i,
            $(_a[i]))
    end for i in 1:_PLATE_N]
    @eval @inline function cache_apply!(slot::Base.RefValue{SA},
            op::ReactiveKernels._AuthoredPlateOp{K,A},
            $(_a...)) where {K,A,SA<:Array}
        $(_binds...)
        combined_axes = Base.Broadcast.combine_axes($(_w...))
        isempty(combined_axes) && throw(ArgumentError(
            "an authored plate requires at least one non-Ref batched argument"))
        cache = slot[]
        output_type = eltype(cache)
        result = if cache isa Array{output_type}
            _authored_plate_array_cache(cache, combined_axes)
        elseif Base.axes(cache) == combined_axes &&
               eltype(cache) == output_type
            cache
        else
            similar(cache, output_type, combined_axes)
        end
        Base.Broadcast.broadcast!(op.kernel, result, $(_w...))
        slot[] = result
        result
    end
end

# --- destination-passing step coverage --------------------------------------
# The core decomposes fused captured sources into step operations whose
# CALLABLE form is the exact allocating semantics; these methods add the
# buffer-reusing halves. Each reuses the cache only when shape and eltype
# match the allocating result exactly, so value parity is preserved
# bit-for-bit and a mismatch simply reseeds the cache.

@inline function MutatingFunctions.apply!!(
        cache::AbstractArray, ::ReactiveKernels._MaterializeStep,
        bc::Base.Broadcast.Broadcasted)
    instantiated = Base.Broadcast.instantiate(bc)
    if Base.axes(cache) == Base.axes(instantiated) &&
       Base.Broadcast.combine_eltypes(instantiated.f, instantiated.args) ===
       eltype(cache)
        return Base.Broadcast.materialize!(cache, instantiated)
    end
    Base.Broadcast.materialize(instantiated)
end

const _LA = ReactiveKernels.LinearAlgebra

@inline _matmul_destination_matches(cache::AbstractVector, A, B::AbstractVector) =
    size(cache) == (size(A, 1),)
@inline _matmul_destination_matches(cache::AbstractMatrix, A, B::AbstractMatrix) =
    size(cache) == (size(A, 1), size(B, 2))
@inline _matmul_destination_matches(cache, A, B) = false

# Guarded matrix product: `mul!` follows the pre-sized destination convention,
# so reuse the cache only when it already has the exact result shape and
# eltype; any mismatch (e.g. a batch-size change) recomputes and reseeds. An
# operand BLAS cannot take (a view of RKPPL's compact-cell overlay) is summed
# into the cache by the native body's own loop, with the same values: the
# generic `mul!` allocates two tile buffers per call on Julia 1.10 and reaches
# its kernel dynamically under native Reverse (snag `compact-cell-gra-ebf0a114`).
@inline function MutatingFunctions.apply!!(
        cache::AbstractVecOrMat, op::ReactiveKernels._MatMulStep,
        A::AbstractVecOrMat, B::AbstractVecOrMat)
    T = Base.promote_op(_LA.matprod, eltype(A), eltype(B))
    if eltype(cache) === T && _matmul_destination_matches(cache, A, B)
        ReactiveKernels._native_mul_loops(A, B) &&
            return ReactiveKernels._native_mul_accumulate!(fill!(cache, zero(T)), A, B)
        return _LA.mul!(cache, A, B)
    end
    op(A, B)
end

# Whether `vcat(args...)` fits the cache: every operand an
# `AbstractArray{T,N}` with the cache's trailing sizes, their heights summing
# to the cache's. The operands of one `vcat` are often of different array
# types (a fresh `Matrix` row over a parameter block read in place as a
# reshaped view), so the checks run over the operand tuple one inlined call
# per operand: a loop over it reads each operand as a `Union` and builds its
# size closure at run time, which the type-stability check reports and native
# Enzyme reverse rejects (snag `rkppl-vcat-of-pa-decf6725`).
@inline _vcat_trailing_size(a, ::Val{M}) where {M} =
    ntuple(d -> size(a, d + 1), Val(M))
@inline _vcat_operands_fit(::Type{T}, ::Val{N}, trailing, ::Tuple{}) where {T,N} =
    true
@inline _vcat_operands_fit(::Type{T}, ::Val{N}, trailing,
                           args::Tuple) where {T,N} =
    first(args) isa AbstractArray{T,N} &&
    _vcat_trailing_size(first(args), Val(N - 1)) == trailing &&
    _vcat_operands_fit(T, Val(N), trailing, Base.tail(args))
@inline _vcat_destination_matches(cache::Array{T,N}, args) where {T,N} =
    _vcat_operands_fit(T, Val(N), _vcat_trailing_size(cache, Val(N - 1)), args) &&
    size(cache, 1) == ReactiveKernels._native_cat_height_sum(args)

# Copy each operand's rows into the cache in order, one call per operand.
@inline _vcat_copy_rows!(cache::Array, row, ::Tuple{}) = cache
@inline function _vcat_copy_rows!(cache::Array{T,N}, row,
                                  args::Tuple) where {T,N}
    a = first(args)
    rows = size(a, 1)
    copyto!(view(cache, row+1:row+rows, ntuple(_ -> Colon(), Val(N - 1))...), a)
    _vcat_copy_rows!(cache, row + rows, Base.tail(args))
end

# Elementwise `+`/`-` over dense arrays. `Base.+`/`Base.-` on arrays check
# exact shape agreement and then broadcast (`arraymath.jl`), so on matching
# shapes `broadcast!` into the cache computes bit-identical values with no
# temporary; anything else (shape mismatch, mixed eltypes, exotic operands)
# falls back to the twin call, preserving its exact result and errors. Only
# `+` has an N-ary method (`-(a, b, c)` is a `MethodError`), so only `+` gets
# the 3-argument and slurped forms. Restricted to exact `Array` operands: a
# structured array could shadow the generic method with non-elementwise
# semantics, and those stay on the allocating fallback.
@inline function _add_destination_matches(cache::Array{T,N}, args) where {T,N}
    length(args) >= 2 || return false
    all(a -> a isa Array{T,N}, args) || return false
    ax = axes(args[1])
    all(a -> axes(a) == ax, args) || return false
    axes(cache) == ax
end

@inline function MutatingFunctions.apply!!(
        cache::Array{T,N}, ::typeof(+), a::Array{T,N}, b::Array{T,N}) where {T,N}
    _add_destination_matches(cache, (a, b)) || return a + b
    Base.Broadcast.broadcast!(+, cache, a, b)
    cache
end

@inline function MutatingFunctions.apply!!(
        cache::Array{T,N}, ::typeof(+),
        a::Array{T,N}, b::Array{T,N}, c::Array{T,N}) where {T,N}
    _add_destination_matches(cache, (a, b, c)) || return a + b + c
    Base.Broadcast.broadcast!(+, cache, a, b, c)
    cache
end

@inline function MutatingFunctions.apply!!(
        cache::Array{T,N}, ::typeof(+),
        a::Array{T,N}, b::Array{T,N}, c::Array{T,N}, d::Array{T,N},
        rest::Array{T,N}...) where {T,N}
    args = (a, b, c, d, rest...)
    _add_destination_matches(cache, args) || return +(args...)
    Base.Broadcast.broadcast!(+, cache, args...)
    cache
end

# A square dense matrix plus a `UniformScaling` (`K + sigma * I`), into a
# matching cache (`ReactiveKernels._uniform_add_into!`).
@inline MutatingFunctions.apply!!(cache::Matrix{T}, ::typeof(+), A::Matrix{T},
        J::LinearAlgebra.UniformScaling) where {T} =
    ReactiveKernels._uniform_add_into!(cache, A, J)

@inline function MutatingFunctions.apply!!(
        cache::Array{T,N}, ::typeof(-), a::Array{T,N}, b::Array{T,N}) where {T,N}
    _add_destination_matches(cache, (a, b)) || return a - b
    Base.Broadcast.broadcast!(-, cache, a, b)
    cache
end

@inline function MutatingFunctions.apply!!(
        cache::Array{T,N}, op::ReactiveKernels._ConcatenateStep{typeof(vcat)},
        args...) where {T,N}
    _vcat_destination_matches(cache, args) || return op.f(args...)
    _vcat_copy_rows!(cache, 0, args)
end

# `vcat` of dense matrices of the cache's element type (a `Matrix`, or a
# contiguous parameter block reshaped to one, `_NativeMatrix`) into a matrix
# cache: the operands' rows in order through the native body's unrolled copy,
# as for vectors below, rather than a view of the cache per operand, which
# native Enzyme reverse records on its tape on every gradient call. Operands
# of other widths, or one that may share memory with the cache, return the
# step's own result. Other operands take the general method above.
@inline function MutatingFunctions.apply!!(
        cache::Matrix{T}, op::ReactiveKernels._ConcatenateStep{typeof(vcat)},
        a::ReactiveKernels._NativeMatrix,
        rest::Vararg{ReactiveKernels._NativeMatrix,N}) where {T,N}
    args = (a, rest...)
    all(x -> eltype(x) === T, args) || return op.f(args...)
    width = size(a, 2)
    size(cache, 2) == width && ReactiveKernels._native_cat_widths_equal(width, rest) ||
        return op.f(args...)
    n = ReactiveKernels._native_cat_height_sum(args)
    size(cache, 1) == n || return op.f(args...)
    ReactiveKernels._lane_vcat_aliases(cache, args) && return op.f(args...)
    ReactiveKernels._native_vcat_copy!(cache, n, 0, args)
end

# `vcat` of dense vectors of the cache's element type into a vector cache:
# the operands' rows in order, through the native body's unrolled copy
# (`_native_vcat_copy!`) rather than a loop over the operand tuple, whose
# per-operand views and loads native Enzyme reverse records on its tape on
# every gradient call. Other operands take the general method above.
@inline function MutatingFunctions.apply!!(
        cache::Vector{T}, op::ReactiveKernels._ConcatenateStep{typeof(vcat)},
        a::ReactiveKernels._NativeVector,
        rest::Vararg{ReactiveKernels._NativeVector,N}) where {T,N}
    args = (a, rest...)
    all(x -> eltype(x) === T, args) || return op.f(args...)
    n = ReactiveKernels._native_cat_height_sum(args)
    length(cache) == n || return op.f(args...)
    ReactiveKernels._lane_vcat_aliases(cache, args) && return op.f(args...)
    ReactiveKernels._native_vcat_copy!(cache, n, 0, args)
end

# `copy` of a contiguous view of a vector, as RKPPL reads each scalar layout
# entry (`copy(view(unconstrained, i:i))`): the same values through the native
# body's element copy. Base's generic `copyto!` from a view iterates paired
# index ranges whose state native Enzyme reverse records on its tape on every
# gradient call (144 B per copy). A cache of another length or one that may
# share memory with the view takes the step's own result.
@inline function MutatingFunctions.apply!!(
        cache::Vector{T}, ::typeof(copy),
        A::ReactiveKernels._NativeVectorView) where {T}
    eltype(A) === T && length(cache) == length(A) || return copy(A)
    Base.mightalias(cache, A) && return copy(A)
    ReactiveKernels._native_vcat_copy!(cache, length(cache), 0, (A,))
end

# The native body's `vcat` of numbers and dense vectors (`_native_vcat`), the
# step a vector literal of numbers concatenates through: the operands fill a
# matching vector cache in order, as `_native_vcat` fills its fresh vector.
# Another element type or length, an operand that may share memory with the
# cache, or other operands return the step's own result instead.
@inline function MutatingFunctions.apply!!(
        cache::Vector{T},
        op::ReactiveKernels._ConcatenateStep{typeof(ReactiveKernels._native_vcat)},
        a::ReactiveKernels._NativeColumnOperand,
        rest::Vararg{ReactiveKernels._NativeColumnOperand,N}) where {T,N}
    args = (a, rest...)
    isbitstype(T) && Base.promote_eltypeof(args...) === T || return op.f(args...)
    n = ReactiveKernels._native_cat_height_sum(args)
    length(cache) == n || return op.f(args...)
    ReactiveKernels._lane_vcat_aliases(cache, args) && return op.f(args...)
    ReactiveKernels._native_vcat_copy!(cache, n, 0, args)
end

# One-dimensional gather without constructing the view object: contiguous
# ranges copy with offsets, anything else 1-D copies elementwise in index
# order — both bit-identical to `x[idx]`, whose bounds behavior they preserve
# (`copyto!`/scalar indexing throw the same `BoundsError`). Any shape the loop
# does not cover (non-vectors, empty-vs-nonempty mismatches, exotic indices)
# falls back to the twin call, which reseeds the cache.
@inline function MutatingFunctions.apply!!(
        cache::Vector{T}, ::ReactiveKernels._GatherStep,
        x::AbstractVector{T}, idx::AbstractRange) where {T}
    n = length(idx)
    n == 0 && return length(cache) == 0 ? cache : x[idx]
    length(cache) == n || return x[idx]
    if idx isa Union{UnitRange,Base.OneTo}
        return copyto!(cache, 1, x, first(idx), n)
    end
    for (i, j) in enumerate(idx)
        cache[i] = x[j]
    end
    cache
end

@inline function MutatingFunctions.apply!!(
        cache::Vector{T}, ::ReactiveKernels._GatherStep,
        x::AbstractVector{T}, idx::AbstractVector{<:Integer}) where {T}
    eltype(idx) === Bool && return _gather_slow!(cache, x, idx)
    n = length(idx)
    n == 0 && return length(cache) == 0 ? cache : x[idx]
    length(cache) == n || return x[idx]
    for i in 1:n
        cache[i] = x[idx[i]]
    end
    cache
end

# Shapes the fast loops do not cover (non-vectors, multiple indices, logical
# masks, exotic ranges): the previous view-plus-materialize computation,
# reusing the tested `_MaterializeStep` method, so behavior never regresses
# from before `_GatherStep` existed.
@inline function _gather_slow!(cache, x, idx...)
    MutatingFunctions.apply!!(cache, ReactiveKernels._MaterializeStep(),
        Base.Broadcast.broadcasted(Base.identity, view(x, idx...)))
end

@inline function MutatingFunctions.apply!!(
        cache::AbstractArray, ::ReactiveKernels._GatherStep, x, idx...)
    _gather_slow!(cache, x, idx...)
end

# Partitioned-plate lane gathers: `x[lanes]` elementwise into the cache, or a
# fill when the argument is a singleton (the twin's indexed-view branch).
# The guards mirror the twin's branch order exactly — the singleton check
# first, so out-of-range lane indices on a length-1 argument fill rather than
# throwing. Resize the owned vector when the lane count changes; shape errors
# still call the twin, preserving its error. Bool lane masks use the twin's
# selection and copy its view into owned storage.
#
# The gather loop itself is `@inbounds`, with the twin's `BoundsError` on
# out-of-range lanes preserved by an explicit integer-range validation (pinned
# by tests): the guards above establish `length(cache) == length(lanes)`, so
# only the data-driven `x[lane]` index needs the manual check. Spelled-out
# integer comparisons stay allocation-free where the per-element `checkbounds`
# calls do not: under reverse-mode Enzyme on Julia 1.12 those checks allocate
# (~17.6 bytes/element measured), defeating the non-allocating gradient.
@inline function MutatingFunctions.apply!!(
        cache::Vector{T}, gather::ReactiveKernels._LaneGather,
        x::AbstractVector{T}, lanes::AbstractVector{<:Integer}) where {T}
    eltype(lanes) === Bool && return copy(gather(x, lanes))
    if length(x) == 1
        length(cache) == length(lanes) || resize!(cache, length(lanes))
        fill!(cache, x[1])
        return cache
    end
    ndims(x) == 1 && length(x) == gather.n || return copy(gather(x, lanes))
    length(cache) == length(lanes) || resize!(cache, length(lanes))
    n = length(x)
    @inbounds for (i, lane) in enumerate(lanes)
        (1 <= lane <= n) || throw(BoundsError(x, lane))
        cache[i] = x[lane]
    end
    cache
end

@inline function MutatingFunctions.apply!!(
        cache::Array{Float64,N}, op::ReactiveKernels._FillConstructorStep,
        dims::Vararg{Integer,N}) where {N}
    size(cache) == dims || return op(dims...)
    fill!(cache, ReactiveKernels._fill_constructor_value(op))
    cache
end

# Row/column reduction with a preparation-constant `dims`. Base's
# `sum!`/`prod!`/`minimum!`/`maximum!` run the same per-lane `mapreducedim`
# pass the allocating twin (`f(A; dims = ...)` uses, so values are
# bit-identical; the cache must already have the exact result shape and
# eltype, and anything else reseeds through the twin call.
@inline function MutatingFunctions.apply!!(
        cache::Array{T}, op::ReactiveKernels._RowReduceStep, A::Array) where {T}
    T === Base.promote_op(getfield(op, :f), eltype(A)) || return op(A)
    axes(cache) == Base.reduced_indices(axes(A), getfield(op, :dims)) ||
        return op(A)
    _rk_rowreduce_inplace!(getfield(op, :f), cache, A)
    cache
end

@inline _rk_rowreduce_inplace!(::typeof(sum), r, A) = Base.sum!(r, A)
@inline _rk_rowreduce_inplace!(::typeof(prod), r, A) = Base.prod!(r, A)
@inline _rk_rowreduce_inplace!(::typeof(minimum), r, A) = Base.minimum!(r, A)
@inline _rk_rowreduce_inplace!(::typeof(maximum), r, A) = Base.maximum!(r, A)

# --- cache ownership ---------------------------------------------------------
# A step keeps a cache slot only when a destination-passing method fills owned
# storage (`ReactiveKernels._nonalloc_destination`). Besides the core's own
# fresh-storage steps, that is a registered `apply!!` method selected for the
# step's concrete cache and argument types: its first result is the operation's
# own fresh result, and later calls fill it in place. MutatingFunctions' generic
# fallback is not one: it allocates the result and copies it into whatever the
# slot holds, which may be caller data.
const _GENERIC_APPLY = which(MutatingFunctions.apply!!, Tuple{Any,Any,Vararg{Any}})

function ReactiveKernels._nonalloc_destination(::typeof(cache_apply!), op,
                                               T, argtypes)
    ReactiveKernels._nonalloc_fresh_step(op) && return true
    argtypes === nothing && return false
    T isa Type && isconcretetype(T) &&
        all(t -> t isa Type && isconcretetype(t), argtypes) || return false
    method = try
        which(MutatingFunctions.apply!!, Tuple{T,typeof(op),argtypes...})
    catch
        return false
    end
    method !== _GENERIC_APPLY
end

# The reactive layer seeds a slot with its recipe's first result in the same
# way, so it reuses a slot in place under the same rule, typed by the recipe's
# declared port types.
ReactiveKernels._reactive_inplace_destination(::typeof(reactive_cache_apply),
                                              recipe::ReactiveKernels.Recipe) =
    ReactiveKernels._nonalloc_destination(cache_apply!, recipe.op,
        ReactiveKernels.valtype(only(recipe.outputs)),
        map(ReactiveKernels.valtype, Tuple(recipe.inputs)))

# --- public entry points ----------------------------------------------------

function ReactiveKernels.prepare_reactive_nonallocating(graph::ReactiveKernels.Graph;
        have = (), want = (),
        is_mutating = ReactiveKernels._default_is_mutating)
    ReactiveKernels._prepare_reactive(graph;
        have = have, want = want,
        cache_apply = reactive_cache_apply, is_mutating = is_mutating)
end

function ReactiveKernels.prepare_reactive_nonallocating(spec::ReactiveKernels.KernelSpec;
        is_mutating = ReactiveKernels._default_is_mutating, kwargs...)
    ReactiveKernels._prepare_reactive(spec;
        cache_apply = reactive_cache_apply, is_mutating = is_mutating, kwargs...)
end

function ReactiveKernels.prepare_nonallocating(p::ReactiveKernels.Plan,
                                               exemplars...;
                                               passes = (), bound = (),
                                               simplify = nothing)
    ReactiveKernels._with_simplify(simplify) do
        # As ordinary `prepare`: cell-only producers compose before the bound split.
        fused = bound === () ? p :
            ReactiveKernels._fuse_authored_plate_chains(p; cells_only = true)
        applied = ReactiveKernels._partial_apply(fused, bound)
        have_types = ReactiveKernels._nonalloc_exemplar_types(applied, exemplars)
        ReactiveKernels._prepare_nonallocating(applied, cache_apply!; have_types,
            exact_signature = have_types !== nothing, passes)
    end
end

function ReactiveKernels.prepare_nonallocating(g::ReactiveKernels.Graph,
                                                exemplars...;
                                                have = (), want = (), passes = (),
                                                bound = (), simplify = nothing)
    p = ReactiveKernels.plan(g; have = have, want = want)
    ReactiveKernels.prepare_nonallocating(p, exemplars...; passes = passes,
                                          bound = bound, simplify = simplify)
end

# The natural consumer call after benchmarking an ordinary prepared kernel:
# reuse its already-selected plan.
function ReactiveKernels.prepare_nonallocating(k::ReactiveKernels.PreparedKernel,
                                                exemplars...; passes = (),
                                                simplify = nothing)
    ReactiveKernels.prepare_nonallocating(k.plan, exemplars...; passes = passes,
                                          simplify = simplify)
end

end # module ReactiveKernelsMutatingFunctionsExt
