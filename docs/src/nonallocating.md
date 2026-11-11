# Reusable buffers

Use `prepare_nonallocating` when repeated direct calls should reuse kernel-owned intermediate storage. Load MutatingFunctions to activate the extension. The documentation setup installs its pinned source dependency.

## Returned-value lifetime

The following graph copies and reverses an input vector. The prepared callable retains storage for operations with supported in-place forms.

```@example buffers
using ReactiveKernels, MutatingFunctions

@kernel reverse_copy(x::Vector{Float64}) = begin
    copied::Vector{Float64} = copy(x)
    reversed::Vector{Float64} = reverse(copied)
    return reversed
end

run = prepare_nonallocating(reverse_copy)
input = [1.0, 2.0, 3.0]
borrowed = run(input)
saved = copy(borrowed)
next_result = run([4.0, 5.0, 6.0])
@assert input == [1.0, 2.0, 3.0]
@assert saved == reverse(input)
(; saved, next_result=copy(next_result), input_unchanged=input == [1.0, 2.0, 3.0])
```

Treat mutable results as borrowed. A later call can overwrite a kernel-owned array; a result that is only an input or field read can alias caller-owned storage. Copy a result if it must survive another call.

The interface selects the same mathematical recipes as ordinary preparation. It routes supported operations through reusable buffers. Operations without an in-place form can still allocate, so the name alone does not establish an allocation count for an arbitrary graph.

## Generated program

The generated expression makes the buffer boundary inspectable.

```@example buffers
using Main.Tutorials
Tutorials.generated(run)
```

A prepared object with mutable caches is for direct use by one caller at a time. Give concurrent callers their own prepared objects. For differentiation, use the prepared AD APIs that manage the derivative storage contract; do not assume that wrapping an arbitrary mutating callable preserves its primal state for reverse evaluation.

The [bound-regression example](bound-regression.md) shows how to separate data-dependent work before deciding whether a repeated computation also needs reusable buffers.
