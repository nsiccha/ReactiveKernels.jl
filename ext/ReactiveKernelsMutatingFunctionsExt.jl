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
function MutatingFunctions.apply!!(
        cache::AbstractArray, op::ReactiveKernels._AuthoredPlateOp{K,A},
        args...) where {K,A}
    wrapped = ReactiveKernels._authored_plate_arguments(Val(A), args...)
    combined_axes = Base.Broadcast.combine_axes(wrapped...)
    isempty(combined_axes) && throw(ArgumentError(
        "an authored plate requires at least one non-Ref batched argument"))
    output = only(ReactiveKernels.outputs(op.kernel))
    result = axes(cache) == combined_axes &&
             eltype(cache) == ReactiveKernels.valtype(output) ?
             cache : similar(cache, ReactiveKernels.valtype(output), combined_axes)
    Base.Broadcast.broadcast!(op.kernel, result, wrapped...)
    result
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
        p, ReactiveKernels._nonallocating_ast(ast), cache_apply!)
end

function ReactiveKernels.prepare_nonallocating(g::ReactiveKernels.Graph;
                                                have = (), want = (), passes = ())
    p = ReactiveKernels.plan(g; have = have, want = want)
    ReactiveKernels.prepare_nonallocating(p; passes = passes)
end

end # module ReactiveKernelsMutatingFunctionsExt
