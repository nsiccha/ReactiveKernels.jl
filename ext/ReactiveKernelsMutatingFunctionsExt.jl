module ReactiveKernelsMutatingFunctionsExt

using ReactiveKernels
import MutatingFunctions

@inline function cache_apply!(slot::Base.RefValue, op, args...)
    slot[] = MutatingFunctions.apply!!(slot[], op, args...)
end

# The reactive getter injects a `cache_apply(cache, op, args...) -> newcache`
# helper that takes the slot VALUE (not the Ref) and returns the value the
# getter stores back. `MutatingFunctions.apply!!` already has that exact
# contract — including immutable/isbits passthrough — so it IS the reactive hook.
@inline reactive_cache_apply(cache, op, args...) =
    MutatingFunctions.apply!!(cache, op, args...)

# An authored plate's pointwise result is one ordinary single-output recipe.
# Reuse that recipe's borrowed cache after the first call while preserving the
# exact broadcast/Ref argument semantics used by allocating execution.
@inline function _authored_plate_array_cache(
        cache::Array{T}, combined_axes::Tuple{Vararg{Any,N}}) where {T,N}
    result = cache isa Array{T,N} && Base.axes(cache) == combined_axes ?
             cache : similar(cache, T, combined_axes)
    result::Array{T,N}
end

@inline function _apply_authored_plate!(
        cache, op::ReactiveKernels._AuthoredPlateOp{K,A}, args...) where {K,A}
    wrapped = ReactiveKernels._authored_plate_arguments(Val(A), args...)
    combined_axes = Base.Broadcast.combine_axes(wrapped...)
    isempty(combined_axes) && throw(ArgumentError(
        "an authored plate requires at least one non-Ref batched argument"))
    output = only(ReactiveKernels.outputs(op.kernel))
    output_type = ReactiveKernels.valtype(output)
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
# eltype; any mismatch (e.g. a batch-size change) recomputes and reseeds.
@inline function MutatingFunctions.apply!!(
        cache::AbstractVecOrMat, op::ReactiveKernels._MatMulStep,
        A::AbstractVecOrMat, B::AbstractVecOrMat)
    T = Base.promote_op(_LA.matprod, eltype(A), eltype(B))
    if eltype(cache) === T && _matmul_destination_matches(cache, A, B)
        return _LA.mul!(cache, A, B)
    end
    A * B
end

@inline function _vcat_destination_matches(cache::Array{T,N}, args) where {T,N}
    all(a -> a isa AbstractArray{T,N}, args) || return false
    trailing = ntuple(d -> size(cache, d + 1), N - 1)
    rows = 0
    for a in args
        ntuple(d -> size(a, d + 1), N - 1) == trailing || return false
        rows += size(a, 1)
    end
    size(cache, 1) == rows
end

@inline function MutatingFunctions.apply!!(
        cache::Array{T,N}, op::ReactiveKernels._ConcatenateStep{typeof(vcat)},
        args...) where {T,N}
    _vcat_destination_matches(cache, args) || return op.f(args...)
    offset = 0
    for a in args
        rows = size(a, 1)
        copyto!(view(cache, offset+1:offset+rows,
                     ntuple(_ -> Colon(), N - 1)...), a)
        offset += rows
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

function ReactiveKernels.prepare_nonallocating(p::ReactiveKernels.Plan; passes = ())
    # Embedded prepared recipes are already allocation-free executable kernels.
    # Keep them as one cache operation here; ordinary `prepare` is the boundary
    # that splices their generated bodies into a flat operation table.
    ast = ReactiveKernels._lower_unembedded(p)
    isempty(passes) || (ast = ReactiveKernels.transform(ast, passes...))
    ReactiveKernels._prepare_nonallocating(p, ast, cache_apply!)
end

function ReactiveKernels.prepare_nonallocating(g::ReactiveKernels.Graph;
                                                have = (), want = (), passes = ())
    p = ReactiveKernels.plan(g; have = have, want = want)
    ReactiveKernels.prepare_nonallocating(p; passes = passes)
end

# The natural consumer call after benchmarking an ordinary prepared kernel:
# reuse its already-selected plan.
function ReactiveKernels.prepare_nonallocating(k::ReactiveKernels.PreparedKernel;
                                                passes = ())
    ReactiveKernels.prepare_nonallocating(k.plan; passes = passes)
end

end # module ReactiveKernelsMutatingFunctionsExt
