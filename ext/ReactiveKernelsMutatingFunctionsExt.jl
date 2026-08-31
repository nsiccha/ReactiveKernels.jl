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
# this embedded operation even after the cache has been seeded.
@inline function cache_apply!(
        slot::Base.RefValue{Array{T}},
        op::ReactiveKernels._AuthoredPlateOp, args...) where {T}
    result = _apply_authored_plate!(slot[], op, args...)
    slot[] = result
    result
end

# Seed an authored plate with a concrete empty Array cache instead of the
# generic `Union{Nothing,T}` slot.  The first call resizes/replaces that empty
# cache for the broadcast axes; subsequent calls load a concrete Array and the
# cache-apply path stays allocation-free rather than boxing a Nothing/Array
# union around the embedded plate operation.
function authored_plate_cache_slot(recipe::ReactiveKernels.Recipe)
    if recipe.op isa ReactiveKernels._AuthoredPlateOp
        T = ReactiveKernels.valtype(only(ReactiveKernels.outputs(recipe.op.kernel)))
        return Ref{Array{T}}(Vector{T}())
    end
    ReactiveKernels._cache_slot(only(recipe.outputs))
end

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
    ReactiveKernels._prepare_nonallocating(
        p, ReactiveKernels._nonallocating_ast(ast), cache_apply!;
        cache_slot = authored_plate_cache_slot)
end

function ReactiveKernels.prepare_nonallocating(g::ReactiveKernels.Graph;
                                                have = (), want = (), passes = ())
    p = ReactiveKernels.plan(g; have = have, want = want)
    ReactiveKernels.prepare_nonallocating(p; passes = passes)
end

end # module ReactiveKernelsMutatingFunctionsExt
