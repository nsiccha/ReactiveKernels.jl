module ReactiveKernelsReactantExt

using ReactiveKernels
import Reactant

@inline ReactiveKernels._kernel_source_arg_style(
    arg::Reactant.TracedType) = Val(:tensorized)

# Prepared kernels are immutable compiled programs.  Their graph/plan/AST
# fields are inspection metadata, not runtime arguments.  Leaving Reactant's
# generic struct traversal in charge would recursively trace that metadata (and
# eventually encounter types such as Tuple{Vararg{Value}}), even though kernel
# execution only needs the already-compiled callable and operation tuple.
function Reactant.make_tracer(seen, previous::ReactiveKernels.PreparedKernel,
                              path, mode; kwargs...)
    previous
end

function Reactant.traced_type_inner(
        ::Type{T}, seen, mode, track_numbers, ndevices, runtime) where
        {T<:ReactiveKernels.PreparedKernel}
    T
end

# Named/defaulted @kernel signatures wrap a PreparedKernel plus immutable
# default providers.  The whole wrapper is likewise static program structure.
function Reactant.make_tracer(
        seen, previous::ReactiveKernels._KernelSignatureCallable,
        path, mode; kwargs...)
    previous
end

function Reactant.traced_type_inner(
        ::Type{T}, seen, mode, track_numbers, ndevices, runtime) where
        {T<:ReactiveKernels._KernelSignatureCallable}
    T
end

# Native Julia arrays keep the fused scalar loop.  Reactant arrays select the
# separately generated eager broadcast/reduction body, avoiding forbidden
# scalar indexing while leaving XLA free to fuse the tensor operations.
@inline function ReactiveKernels._batched_call(
        f::ReactiveKernels._BatchedFunctionPair, ops, args,
        marker::Reactant.RArray)
    f.tensorized(ops, args...)
end

end # module ReactiveKernelsReactantExt
