# Prototype for the operation-table study (todo `0pro4di`): a recipe's authored
# source emitted directly into the generated body.
#
# Today every authored recipe is one entry of the operation table `__ops__`, a
# `_KernelSourceOp` around a separately generated function of the recipe's
# source, and the body calls `lhs = __ops__[k](x, y)`. With
# `RK_INLINE_SOURCES=1` the body carries the source itself,
#
#     lhs = let x#1 = x, y#2 = y
#         <the recipe's native source, its names bound hygienically>
#     end
#
# and the slot keeps `_InlinedSource(captures)`: the recipe's lexical captures
# as data, read by the body where the source reads them. The table's type then
# names no source function. The plan, its recipes and `Recipe.source` are
# untouched, so `explain`, `recipe_inventory` and the structural graph read as
# before, and `code_expr` shows the code that executes.
#
# A recipe keeps its table call when its source is outside this emitter's
# reach: an exact bare call (`cholesky(x)`, a raw function in the table), a
# plate, a scan, a bound constant, an embedded kernel, the `on_error = :ignore`
# twin (its body differs from the source), a closure Julia owns (a boxed
# capture), or a source whose free names this emitter cannot resolve.

# Off under `on_error = :ignore`: the throw-stripped twin that policy selects
# keeps the authored source on its recipe, so the source is not its code.
const _INLINE_SOURCES_OFF = :ReactiveKernels_inline_sources_off
_inline_sources_enabled() = get(ENV, "RK_INLINE_SOURCES", "0") == "1" &&
    !get(task_local_storage(), _INLINE_SOURCES_OFF, false)

struct _InlinedSource{C}
    captures::C
end
(::_InlinedSource)(args...) = throw(ArgumentError(
    "an inlined source's operation slot is a data placeholder; its code is in the generated body"))

# Heads whose symbol arguments are syntax, not names.
const _INLINE_OPAQUE_HEADS = (:quote, :inert, :meta, :inbounds, :boundscheck,
                              :loopinfo, :simdloop, :gc_preserve_begin, :gc_preserve_end)
# Scoped constructs: a name they bind is local to them, so a same-named module
# global read outside them would be mistaken for the local by a flat rename.
const _INLINE_SCOPE_HEADS = (:->, :function, :let, :for, :generator, :comprehension,
                             :typed_comprehension, :do, :try, :while)

# The inline form of recipe `r` over `callargs` (its operation at table slot
# `slot`), with the recipe's captures, or `nothing` where it keeps its call.
function _inline_source_form(r::Recipe, callargs, slot::Int)
    op = r.op
    op isa _KernelSourceOp || return nothing
    kernel_sourceop_form(op) === :fused || return nothing
    source = r.source
    source isa Expr || return nothing
    params = _recipe_source_names(r)
    length(params) == length(callargs) && allunique(params) || return nothing
    mod = _kernel_endpoint_source_module(op)
    mod isa Module || return nothing
    closure = _kernel_native_source(_source_closure(op.f))
    T = typeof(closure)
    T isa DataType && closure isa Function || return nothing
    names = fieldnames(T)
    values = Any[getfield(closure, i) for i in 1:length(names)]
    any(v -> v isa Core.Box, values) && return nothing
    captures = NamedTuple{names}(Tuple(values))
    native = _kernel_native_body(source, mod, Set{Symbol}(params))
    # A source whose closures Julia must own (a local function, a generator or
    # comprehension, an exception scope) keeps its native closure today, with
    # no generated companion (`_kernel_source_function`): inside a generated
    # body Julia creates such a closure at run time and every value derived
    # from it dispatches dynamically. It keeps its table call here too.
    _kernel_source_native_scope(native) && return nothing
    body = try
        macroexpand(mod, native)
    catch
        return nothing
    end
    _kernel_source_native_scope(body) && return nothing
    _inline_contains_return(body) && return nothing
    scoped = Set{Symbol}()
    bound = _inline_bound_names!(Set{Symbol}(), body, scoped, false)
    bound === nothing && return nothing
    for name in scoped
        # A name bound in a nested scope and also a module global: a read
        # outside that scope means the global. Keep the call instead.
        _global_value(mod, name) === nothing || return nothing
    end
    renamed = Dict{Symbol,Symbol}()
    for name in params
        renamed[name] = gensym(name)
    end
    for name in names
        haskey(renamed, name) || (renamed[name] = gensym(name))
    end
    for name in bound
        haskey(renamed, name) || (renamed[name] = gensym(name))
    end
    rewritten = _inline_qualify(body, mod, renamed)
    rewritten === nothing && return nothing
    bindings = Any[]
    for (name, arg) in zip(params, callargs)
        push!(bindings, Expr(:(=), renamed[name], arg))
    end
    if !isempty(names)
        captured = Expr(:call, GlobalRef(Core, :getfield),
                        Expr(:call, GlobalRef(Core, :getfield), _OPS_ARG, slot),
                        QuoteNode(:captures))
        for name in names
            push!(bindings, Expr(:(=), renamed[name],
                                 Expr(:call, GlobalRef(Core, :getfield), captured,
                                      QuoteNode(name))))
        end
    end
    Expr(:let, Expr(:block, bindings...), rewritten), captures
end

function _inline_contains_return(ex)
    ex isa Expr || return false
    ex.head === :return && return true
    ex.head in (:->, :function) && return false
    any(_inline_contains_return, ex.args)
end

# Every name `ex` binds anywhere (one flat set), and in `scoped` those bound by
# a scoped construct. `nothing` for a construct outside the emitter's reach.
function _inline_bound_names!(names::Set{Symbol}, ex, scoped::Set{Symbol}, inscope::Bool)
    ex isa Expr || return names
    head = ex.head
    head in _INLINE_OPAQUE_HEADS && return names
    head === :global && return nothing
    head === :macrocall && return nothing
    local_names = Set{Symbol}()
    if head === :(=) || _kernel_assignment_head(head)
        lhs = ex.args[1]
        if lhs isa Expr && lhs.head in (:call, :where)
            # A local function definition binds its name and its parameters.
            _kernel_function_name!(local_names, lhs)
            _inline_parameter_names!(local_names, lhs) === nothing && return nothing
        else
            _lhs_symbols!(local_names, lhs)
        end
    elseif head === :local
        for declaration in ex.args
            target = declaration isa Expr && declaration.head === :(=) ?
                     declaration.args[1] : declaration
            _lhs_symbols!(local_names, target)
        end
    elseif head === :function
        isempty(ex.args) && return nothing
        signature = ex.args[1]
        signature isa Expr && signature.head === :call &&
            _kernel_function_name!(local_names, signature)
        _inline_parameter_names!(local_names, signature) === nothing && return nothing
    elseif head === :->
        _inline_parameter_names!(local_names, ex.args[1]) === nothing && return nothing
    elseif head === :let
        header = ex.args[1]
        for binding in (header isa Expr && header.head === :block ? header.args : Any[header])
            if binding isa Symbol
                push!(local_names, binding)
            elseif binding isa Expr && binding.head === :(=)
                _lhs_symbols!(local_names, binding.args[1])
            else
                return nothing
            end
        end
    elseif head === :for
        iteration = ex.args[1]
        for item in (iteration isa Expr && iteration.head === :block ? iteration.args : Any[iteration])
            item isa Expr && item.head === :(=) || return nothing
            _lhs_symbols!(local_names, item.args[1])
        end
    elseif head === :generator
        for item in ex.args[2:end]
            clause = item isa Expr && item.head === :filter ? item.args[2:end] : Any[item]
            for binding in clause
                binding isa Expr && binding.head === :(=) || return nothing
                _lhs_symbols!(local_names, binding.args[1])
            end
        end
    elseif head === :try
        length(ex.args) >= 2 && ex.args[2] isa Symbol && push!(local_names, ex.args[2])
    elseif head === :tuple || head === :parameters
        # `(a = 1,)` and `(; a = x)` label named-tuple fields; walk their values.
        for item in ex.args
            if item isa Expr && item.head in (:(=), :kw) && length(item.args) == 2
                _inline_bound_names!(names, item.args[2], scoped, inscope) === nothing && return nothing
            else
                _inline_bound_names!(names, item, scoped, inscope) === nothing && return nothing
            end
        end
        return names
    elseif head === :kw
        return _inline_bound_names!(names, ex.args[end], scoped, inscope)
    end
    scoping = inscope || head in _INLINE_SCOPE_HEADS
    for name in local_names
        push!(names, name)
        scoping && push!(scoped, name)
    end
    for arg in ex.args
        _inline_bound_names!(names, arg, scoped, scoping) === nothing && return nothing
    end
    names
end

# The names a lambda or function signature binds, or `nothing`.
function _inline_parameter_names!(names::Set{Symbol}, signature)
    signature isa Symbol && return push!(names, signature)
    signature isa Expr || return nothing
    if signature.head === :tuple || signature.head === :parameters
        for item in signature.args
            _inline_parameter_names!(names, item) === nothing && return nothing
        end
        return names
    elseif signature.head === :call
        for item in signature.args[2:end]
            _inline_parameter_names!(names, item) === nothing && return nothing
        end
        return names
    elseif signature.head === :(::)
        return length(signature.args) == 2 ? _inline_parameter_names!(names, signature.args[1]) :
               names
    elseif signature.head === :kw
        return _inline_parameter_names!(names, signature.args[1])
    elseif signature.head === :...
        return _inline_parameter_names!(names, only(signature.args))
    elseif signature.head === :where
        return _inline_parameter_names!(names, signature.args[1])
    end
    nothing
end

# `ex` with every renamed name replaced and every other free name qualified in
# `mod`, or `nothing` for a free name `mod` does not bind.
function _inline_qualify(ex, mod::Module, renamed::Dict{Symbol,Symbol})
    if ex isa Symbol
        haskey(renamed, ex) && return renamed[ex]
        ex in (:end, :begin) && return ex
        _global_value(mod, ex) === nothing && return nothing
        return GlobalRef(mod, ex)
    end
    ex isa Expr || return ex
    head = ex.head
    head in _INLINE_OPAQUE_HEADS && return ex
    if head === :kw && length(ex.args) == 2
        # A keyword label stays (a signature's defaulted parameter is renamed
        # by `_inline_qualify_signature`).
        value = _inline_qualify(ex.args[2], mod, renamed)
        value === nothing && return nothing
        return Expr(:kw, ex.args[1], value)
    elseif head in (:->, :function) && length(ex.args) == 2
        signature = _inline_qualify_signature(ex.args[1], mod, renamed)
        signature === nothing && return nothing
        body = _inline_qualify(ex.args[2], mod, renamed)
        body === nothing && return nothing
        return Expr(head, signature, body)
    elseif head === :(=) && length(ex.args) == 2 && ex.args[1] isa Expr &&
           ex.args[1].head in (:call, :where)
        signature = _inline_qualify_signature(ex.args[1], mod, renamed)
        signature === nothing && return nothing
        body = _inline_qualify(ex.args[2], mod, renamed)
        body === nothing && return nothing
        return Expr(:(=), signature, body)
    elseif head in (:tuple, :parameters)
        args = Any[]
        for item in ex.args
            if item isa Expr && item.head === :(=) && length(item.args) == 2 &&
                    item.args[1] isa Symbol
                # A named-tuple field label.
                value = _inline_qualify(item.args[2], mod, renamed)
                value === nothing && return nothing
                push!(args, Expr(:(=), item.args[1], value))
            else
                rewritten = _inline_qualify(item, mod, renamed)
                rewritten === nothing && return nothing
                push!(args, rewritten)
            end
        end
        return Expr(head, args...)
    elseif head === :. && length(ex.args) == 2 && ex.args[2] isa QuoteNode
        owner = _inline_qualify(ex.args[1], mod, renamed)
        owner === nothing && return nothing
        return Expr(:., owner, ex.args[2])
    end
    args = Any[]
    for arg in ex.args
        rewritten = _inline_qualify(arg, mod, renamed)
        rewritten === nothing && return nothing
        push!(args, rewritten)
    end
    Expr(head, args...)
end

# A lambda or function signature: its parameters renamed (defaulted and typed
# ones included), its types and default values qualified.
function _inline_qualify_signature(signature, mod::Module, renamed::Dict{Symbol,Symbol})
    signature isa Symbol && return get(renamed, signature, signature)
    signature isa Expr || return signature
    head = signature.head
    if head === :kw && length(signature.args) == 2
        name = _inline_qualify_signature(signature.args[1], mod, renamed)
        value = _inline_qualify(signature.args[2], mod, renamed)
        (name === nothing || value === nothing) && return nothing
        return Expr(:kw, name, value)
    elseif head === :(::) && length(signature.args) == 2
        name = _inline_qualify_signature(signature.args[1], mod, renamed)
        type = _inline_qualify(signature.args[2], mod, renamed)
        (name === nothing || type === nothing) && return nothing
        return Expr(:(::), name, type)
    elseif head in (:tuple, :parameters, :call, :...)
        args = Any[]
        for item in signature.args
            rewritten = _inline_qualify_signature(item, mod, renamed)
            rewritten === nothing && return nothing
            push!(args, rewritten)
        end
        return Expr(head, args...)
    end
    nothing
end
