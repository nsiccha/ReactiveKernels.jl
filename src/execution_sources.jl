# Planning retains each authored closure and its definition token. Execution
# needs the captured source program, already compiled for the world-age
# fallback, and its concrete lexical captures. Reusing that program prevents
# eval-created closure types and provenance tokens from specializing the same
# executable on every fresh build. No plan or source recipe is replaced.
# Provenance stays with the plan and lowered recipes. Every field here is part
# of the differentiated callable, so it holds only what execution reads; the
# authoring module `M` is shared by repeated builds and costs no field.
struct _ExecutableSourceFunction{F,C,M} <: Function
    f::F
    captures::C
end

@inline @generated function (f::_ExecutableSourceFunction)(args::Vararg{Any,N}) where {N}
    forwarded = [:(getfield(args, $i)) for i in 1:N]
    Expr(:block, Expr(:meta, :inline),
        :(Base.@inline RuntimeGeneratedFunctions.generated_callfunc(
            getfield(f, :f), getfield(f, :captures), $(forwarded...))))
end

_kernel_endpoint_source_module(::_ExecutableSourceFunction{F,C,M}) where {F,C,M} = M

_execution_source(@nospecialize(x)) = x
Base.@nospecializeinfer function _execution_source(@nospecialize(xs::Tuple))
    values = Vector{Any}(undef, nfields(xs))
    for i in eachindex(values)
        values[i] = _execution_source(getfield(xs, i))
    end
    Tuple(values)
end

Base.@nospecializeinfer function _execution_source(@nospecialize(f::_KernelSourceFunction))
    runtime = _execution_runtime(getfield(f, :runtime_f))
    # Julia retains ownership of local functions, generators, exception scopes
    # and boxed lexical cells; they have no generated-source companion.
    runtime === nothing && return f
    captures = getfield(f, :captures)
    mod = parentmodule(typeof(getfield(f, :f)))
    T = _ExecutableSourceFunction{typeof(runtime),typeof(captures),mod}
    _kernel_new_instance(T, (runtime, captures))
end

# A companion compiled while a package image is built, in the same world that
# defines its generated method, is wrapped in `_PrecompileWarmFunction`. The
# wrapper tests for image generation and keeps a latest-world call on every
# invocation, so an execution copy holding it ran a dynamic call in every cell
# of a plate from a package-defined graph (snag `generator-sum-in-862b7041`).
# Once the method is visible, as in every process that loads the image, the
# copy holds the generated function itself, as for a graph built at run time.
# While it is not, keep the guarded closure; its method query selects the
# native closure after loading.
_execution_runtime(@nospecialize(f)) = f
_execution_runtime(f::_PrecompileWarmFunction) =
    applicable(RuntimeGeneratedFunctions.generated_callfunc, f.f) ? f.f : nothing

_execution_source(op::_KernelSourceOp{nothing}) = op
Base.@nospecializeinfer function _execution_source(@nospecialize(op::_KernelSourceOp))
    f = _execution_source(getfield(op, :f))
    tensor = _execution_source(getfield(op, :tensor_f))
    ignored = _execution_source(getfield(op, :ignored_throws))
    identity = getfield(op, :call_identity)
    form = (typeof(op)::DataType).parameters[2]
    T = _KernelSourceOp{nothing,form,typeof(f),typeof(tensor),typeof(ignored),typeof(identity)}
    _kernel_new_instance(T, (f, tensor, ignored, identity))
end

Base.@nospecializeinfer function _execution_source(@nospecialize(split::_KernelSplit))
    spec = (typeof(split)::DataType).parameters[1]
    _KernelSplit(Val(spec), _execution_source(getfield(split, :call)),
        _execution_source(getfield(split, :wholes)), _execution_source(getfield(split, :splits)),
        _execution_source(getfield(split, :root)))
end

Base.@nospecializeinfer function _execution_source(@nospecialize(apply::_KernelSplitApply))
    positions = (typeof(apply)::DataType).parameters[1]
    split = _execution_source(getfield(apply, :split))
    _kernel_new_instance(_KernelSplitApply{positions,typeof(split)}, (split,))
end

Base.@nospecializeinfer function _execution_source(@nospecialize(branch::_KernelBranch))
    CI, TI, EI = (typeof(branch)::DataType).parameters[1:3]
    _KernelBranch(Val(CI), Val(TI), Val(EI), _execution_source(getfield(branch, :call)),
        _execution_source(getfield(branch, :condition)), _execution_source(getfield(branch, :then_arm)),
        _execution_source(getfield(branch, :else_arm)),
        getfield(branch, :then_spec), getfield(branch, :else_spec))
end

Base.@nospecializeinfer function _execution_source(@nospecialize(op::_KernelReduction))
    II, XI, KI, AI = (typeof(op)::DataType).parameters[1:4]
    values = Tuple(Any[_execution_source(getfield(op, i)) for i in 1:8])
    types = map(typeof, values)
    T = _KernelReduction{II,XI,KI,AI,types...}
    _kernel_new_instance(T, values)
end
