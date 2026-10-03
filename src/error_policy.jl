# Opt-in source policy, not exception catching. Only macro-captured source
# participates; ordinary Julia methods stay opaque and keep their behavior.

@inline _ignored_throw_call(f, args::Vararg{Any,N}) where {N} =
    _ignored_throw_call(_kernel_source_style(args), f, args...)
@inline _ignored_throw_call(::Val{:native}, f, args::Vararg{Any,N}) where {N} = f(args...)
@inline _ignored_throw_call(::Val{:tensorized}, f, args::Vararg{Any,N}) where {N} =
    traced(f, args...)

# These are explicit captured checks, so the existing opt-in policy strips
# them as it strips a captured throw. Ordinary preparation retains them.
@inline _ignored_throw_call(::Val{:native}, ::typeof(_runtime_check), valid, error) = nothing
@inline _ignored_throw_call(::Val{:tensorized}, ::typeof(_runtime_check), valid, error) = nothing

struct _IgnoredThrowFunction{F}
    f::F
end
@inline (f::_IgnoredThrowFunction)(args::Vararg{Any,N}) where {N} =
    _ignored_throw_call(f.f, args...)

function _kernel_visible_throw(ex, mod, names)
    ex isa Expr && ex.head === :call && length(ex.args) == 2 || return false
    callee = ex.args[1]
    callee isa Symbol && callee in names && return false
    callee isa GlobalRef && return isdefined(callee.mod, callee.name) &&
        getglobal(callee.mod, callee.name) === throw
    mod isa Module && _kernel_resolve_binding(mod, callee) === throw
end

function _kernel_base_or_core_callee(callee, mod)
    binding = callee isa GlobalRef ?
        (isdefined(callee.mod, callee.name) ? getglobal(callee.mod, callee.name) : nothing) :
        mod isa Module ? _kernel_resolve_binding(mod, callee) : nothing
    binding isa Union{Function,Type} &&
        Base.moduleroot(parentmodule(binding)) in (Base, Core)
end

function _kernel_ignore_throw_source(ex, mod, names; statement = false)
    ex isa Expr || return ex
    ex.head in (:quote, :inert) && return ex
    if ex.head in (:function, :(=)) && length(ex.args) == 2
        signature = ex.args[1]
        while signature isa Expr && signature.head in (:where, :(::))
            signature = signature.args[1]
        end
        if signature isa Expr && signature.head === :call
            inner = copy(names)
            _lhs_symbols!(inner, signature)
            return Expr(ex.head, ex.args[1],
                        _kernel_ignore_throw_source(ex.args[2], mod, inner))
        end
    elseif ex.head === :-> && length(ex.args) == 2
        inner = _lhs_symbols!(copy(names), ex.args[1])
        return Expr(:->, ex.args[1], _kernel_ignore_throw_source(ex.args[2], mod, inner))
    end
    # Inspect the expanded assertion, not its message: a removed throw never
    # evaluates its exception constructor or message arguments.
    if ex.head === :macrocall && mod isa Module
        return _kernel_ignore_throw_source(macroexpand(mod, ex), mod, names;
                                           statement)
    end
    _kernel_visible_throw(ex, mod, names) && return nothing
    if statement && ex.head in (:&&, :||) && length(ex.args) == 2 &&
            _kernel_visible_throw(ex.args[2], mod, names)
        # A discarded guard result need not merge Bool with Nothing. Retain
        # evaluation of its predicate (which may itself have side effects).
        return Expr(:block, _kernel_ignore_throw_source(ex.args[1], mod, names), nothing)
    elseif ex.head === :block
        names = copy(names)
        for arg in ex.args
            arg isa Expr && arg.head in (:(=), :local) && !isempty(arg.args) &&
                _lhs_symbols!(names, arg.args[1])
        end
        last_value = findlast(arg -> !(arg isa LineNumberNode), ex.args)
        return Expr(:block, (_kernel_ignore_throw_source(arg, mod, names;
                     statement = statement || i != last_value) for (i, arg) in enumerate(ex.args))...)
    elseif ex.head in (:for, :while) && length(ex.args) == 2
        return Expr(ex.head, _kernel_ignore_throw_source(ex.args[1], mod, names),
                    _kernel_ignore_throw_source(ex.args[2], mod, names; statement = true))
    elseif ex.head in (:if, :elseif)
        return Expr(ex.head, _kernel_ignore_throw_source(ex.args[1], mod, names),
                    (_kernel_ignore_throw_source(arg, mod, names; statement)
                     for arg in ex.args[2:end])...)
    elseif ex.head === :. && length(ex.args) == 2 &&
            ex.args[2] isa Expr && ex.args[2].head === :tuple
        _kernel_base_or_core_callee(ex.args[1], mod) &&
            return Expr(:., ex.args[1], _kernel_ignore_throw_source(ex.args[2], mod, names))
        callee = Expr(:call, GlobalRef(@__MODULE__, :_IgnoredThrowFunction), ex.args[1])
        return Expr(:., callee, _kernel_ignore_throw_source(ex.args[2], mod, names))
    elseif ex.head === :call && !isempty(ex.args)
        # Calls carrying keyword syntax keep their ordinary callable. The
        # transparent @traceable contract itself is positional-only.
        args = Any[_kernel_ignore_throw_source(arg, mod, names) for arg in ex.args[2:end]]
        any(arg -> arg isa Expr && arg.head in (:parameters, :kw), args) &&
            return Expr(:call, ex.args[1], args...)
        callee = ex.args[1]
        callee isa Symbol && _is_broadcast_operator(callee) &&
            return Expr(:call, callee, args...)
        # Keep structural iterator/reduction syntax visible to the existing
        # loop lowering. Base/Core mathematics is opaque to this source policy.
        if _kernel_base_or_core_callee(callee, mod)
            return Expr(:call, callee, args...)
        end
        return Expr(:call, GlobalRef(@__MODULE__, :_ignored_throw_call), ex.args[1], args...)
    end
    Expr(ex.head, (_kernel_ignore_throw_source(arg, mod, names) for arg in ex.args)...)
end

function _kernel_ignored_traceable_methods(callee, formals, wheres, body, mod, names)
    ignored = _kernel_ignore_throw_source(body, mod, names)
    native = _kernel_native_body(ignored, mod, names)
    tensorized = _kernel_tensorized_rhs(_traceable_tail_body(ignored, callee),
                                      names, mod, copy(names))
    methods = Expr[]
    for (style, rhs) in ((:native, native), (:tensorized, tensorized))
        signature = Expr(:call, GlobalRef(@__MODULE__, :_ignored_throw_call),
            Expr(:(::), Expr(:curly, GlobalRef(Base, :Val), QuoteNode(style))),
            Expr(:(::), Expr(:call, GlobalRef(Core, :typeof), callee)), formals...)
        for layer in reverse(wheres)
            signature = Expr(:where, signature, layer...)
        end
        push!(methods, Expr(:(=), signature, rhs))
    end
    methods
end

function _kernel_with_ignored_throws(op::_KernelSourceOp{Token,Form}, ignored) where {Token,Form}
    _KernelSourceOp(Val(Token), Val(Form), op.f, op.tensor_f, ignored)
end

_kernel_ignore_throw_op(op) = _IgnoredThrowFunction(op)
_kernel_ignore_throw_op(op::_KernelSourceOp) =
    op.ignored_throws === nothing ? op : op.ignored_throws
function _kernel_ignore_throw_op(op::_AuthoredPlateOp{K,A}) where {K,A}
    kernel = prepare(op.kernel.plan; on_error = :ignore)
    _AuthoredPlateOp{typeof(kernel),A}(kernel, op.axis_checks)
end
function _kernel_ignore_throw_op(op::_AuthoredScanOp{K,A,I,H}) where {K,A,I,H}
    kernel = prepare(op.kernel.plan; on_error = :ignore)
    _AuthoredScanOp{typeof(kernel),A,I,H}(kernel)
end
_kernel_ignore_throw_op(op::PreparedKernel) = prepare(op.plan; on_error = :ignore)

function _kernel_error_policy(p::Plan, on_error)
    on_error === nothing && return p
    on_error === :ignore || throw(ArgumentError(
        "the only supported on_error policy is :ignore (strip visible throws)"))
    function rewrite(r)
        Recipe(r.id, r.inputs, r.outputs, _kernel_ignore_throw_op(r.op),
               r.cost, r.cse_key, r.effectful, r.source)
    end
    # Before partial evaluation: bound-data mathematics uses exactly the same
    # opted-in operations. The original graph and its ordinary cache are intact.
    recipes = map(rewrite, p.recipes)
    by_id = Dict(r.id => r for r in recipes)
    producer = Dict(cid => by_id[old.id] for (cid, old) in p.producer)
    Plan(p.graph, p.have, p.want, recipes, producer, p.cost, map(rewrite, p.candidates))
end
