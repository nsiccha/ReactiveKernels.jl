module ReactiveKernelsMutatingFunctionsExt

using ReactiveKernels
import MutatingFunctions

@inline function cache_apply!(slot::Base.RefValue, op, args...)
    slot[] = MutatingFunctions.apply!!(slot[], op, args...)
end

function ReactiveKernels.prepare_nonallocating(p::ReactiveKernels.Plan; passes = ())
    ast = isempty(passes) ? ReactiveKernels.lower(p) :
          ReactiveKernels.transform(ReactiveKernels.lower(p), passes...)
    ReactiveKernels._prepare_nonallocating(
        p, ReactiveKernels._nonallocating_ast(ast), cache_apply!)
end

function ReactiveKernels.prepare_nonallocating(g::ReactiveKernels.Graph;
                                                have = (), want = (), passes = ())
    p = ReactiveKernels.plan(g; have = have, want = want)
    ReactiveKernels.prepare_nonallocating(p; passes = passes)
end

end # module ReactiveKernelsMutatingFunctionsExt
