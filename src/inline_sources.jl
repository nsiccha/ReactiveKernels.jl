# A recipe's authored source emitted directly into the generated body
# (operation-table migration, stage 1; decision `0dgfkf5`).
#
# An authored recipe used to be one entry of the operation table `__ops__`, a
# `_KernelSourceOp` around a separately generated function of the recipe's
# source, called as `lhs = __ops__[k](x, y)`. The body now carries the source
# itself,
#
#     lhs = let var"#opk" = __ops__[k], var"#x#opk" = x, var"#y#opk" = y,
#               var"#c#opk" = getfield(__ops__[k], :c)
#         <the recipe's native source, its names bound hygienically>
#     end
#
# and the slot keeps the recipe's authored closure (`_kernel_shared_closure`
# gives fresh builds of one program one closure type), whose fields are the
# source's lexical captures: the body reads them through the literal slot,
# which every slot analysis (`_operation_slot`, `_native_table`, the AD
# externalization) recognizes, and the slot stays an ordinary callable, so the
# type expressions `_promote_op(__ops__[k], …)` of cells and scan steps keep
# working. The table's type then names no generated source function. The plan,
# its recipes and `Recipe.source` are untouched, so `explain`,
# `recipe_inventory` and the structural graph read as before, and `code_expr`
# shows the code that executes (`docs/src/constraints.md`).
#
# A recipe keeps its table call when its source is outside this emitter's
# reach: an exact bare call (`cholesky(x)`, a raw function in the table), a
# plate, a scan, a bound constant, an embedded kernel, the `on_error = :ignore`
# twin (its body differs from the source), a closure Julia owns (a boxed
# capture), or a source whose free names this emitter cannot resolve.

# Off under `on_error = :ignore`: the throw-stripped twin that policy selects
# keeps the authored source on its recipe, so the source is not its code.
# `RK_INLINE_SOURCES=0` keeps every call in the table (the comparison baseline
# of the migration's measurements).
const _INLINE_SOURCES_OFF = :ReactiveKernels_inline_sources_off
_inline_sources_enabled() = get(ENV, "RK_INLINE_SOURCES", "1") != "0" &&
    !get(task_local_storage(), _INLINE_SOURCES_OFF, false)

# The statement's locals are named from the recipe's slot, never minted with
# `gensym`: two lowerings of one plan then emit one expression (`lower(p)`
# equals `lower_with_ops(p)`'s, and a prepared kernel's `code_expr` is a
# function of its plan). A leading `#` keeps them apart from every authored
# identifier, and the `#op` infix from the `#name#N` locals that
# `_value_local_name` derives from gensyms and from `gensym`'s own `##name#N`.
# `_canonical_locals` treats them as the lowering's locals, as it does gensyms.
_inline_slot_name(slot::Int) = Symbol("#op", slot)
_inline_local_name(name::Symbol, slot::Int) = Symbol('#', name, "#op", slot)

# A source that is its own value: a number, a string, a character, `nothing`
# or a quoted symbol.
_inline_literal(x) = x isa Union{Number,AbstractString,Char,Nothing,QuoteNode}

# Heads whose symbol arguments are syntax, not names.
const _INLINE_OPAQUE_HEADS = (:quote, :inert, :meta, :inbounds, :boundscheck,
                              :loopinfo, :simdloop, :gc_preserve_begin, :gc_preserve_end)
# Scoped constructs: a name they bind is local to them, so a same-named module
# global read outside them would be mistaken for the local by a flat rename.
const _INLINE_SCOPE_HEADS = (:->, :function, :let, :for, :generator, :comprehension,
                             :typed_comprehension, :do, :try, :while)

# The inline form of recipe `r` over `callargs` (its operation at table slot
# `slot`) and the authored closure the slot keeps, or `nothing` where the
# recipe keeps its call.
function _inline_source_form(r::Recipe, callargs, slot::Int)
    op = r.op
    op isa _KernelSourceOp || return nothing
    kernel_sourceop_form(op) === :fused || return nothing
    # A plain source function inlines, and so does a fused source with
    # data-only split nodes (`_KernelSplit`): a residual plan calls it whole,
    # partial evaluation reads its parts from the plan's recipe, never from the
    # slot, so the whole source is its code. A branch or a reduction keeps its
    # operation: the plate lowerings read its parts through the slot
    # (`getfield(__ops__[k], :f)`), and its source is not its code.
    f = getfield(op, :f)
    f isa _KernelSplit && (f = getfield(f, :call))
    f isa _KernelSourceFunction || return nothing
    # The throw-stripped twin (`on_error = :ignore`) carries the authored
    # source, which is not its code: it keeps its call until it inlines its
    # own stripped source (`_kernel_ignore_throw_source`). An authored
    # operation is the one that holds a twin.
    getfield(op, :ignored_throws) === nothing && return nothing
    source = r.source
    if _inline_literal(source)
        # A literal statement (`a::Float64 = -1.5`): the literal is its code.
        # The statement still names the slot, which keeps the authored
        # closure, so the passes keyed on a statement's recipe read it as
        # every other inline statement.
        isempty(callargs) || return nothing
        closure = _kernel_native_source(f)
        T = typeof(closure)
        T isa DataType && closure isa Function && fieldcount(T) == 0 || return nothing
        shared = _kernel_shared_closure(closure, Expr(:->, Expr(:tuple), source))
        return Expr(:let, Expr(:block, Expr(:(=), _inline_slot_name(slot), Expr(:ref, _OPS_ARG, slot))),
                    source), shared
    end
    if source isa Symbol
        # An alias (`y::Float64 = x`): the statement binds its one argument
        # and reads it back.
        params = _recipe_source_names(r)
        length(params) == length(callargs) == 1 && only(params) === source || return nothing
        closure = _kernel_native_source(f)
        T = typeof(closure)
        T isa DataType && closure isa Function && fieldcount(T) == 0 || return nothing
        shared = _kernel_shared_closure(closure, Expr(:->, Expr(:tuple, source), source))
        name = _inline_local_name(source, slot)
        return Expr(:let, Expr(:block, Expr(:(=), _inline_slot_name(slot), Expr(:ref, _OPS_ARG, slot)),
                               Expr(:(=), name, only(callargs))), name), shared
    end
    source isa Expr || return nothing
    params = _recipe_source_names(r)
    length(params) == length(callargs) && allunique(params) || return nothing
    mod = _kernel_endpoint_source_module(op)
    mod isa Module || return nothing
    closure = _kernel_native_source(f)
    T = typeof(closure)
    T isa DataType && closure isa Function || return nothing
    names = fieldnames(T)
    any(i -> getfield(closure, i) isa Core.Box, 1:length(names)) && return nothing
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
    for name in Iterators.flatten((params, names, bound))
        haskey(renamed, name) || (renamed[name] = _inline_local_name(name, slot))
    end
    rewritten = _inline_qualify(body, mod, renamed)
    rewritten === nothing && return nothing
    # The first binding names the recipe's slot: the passes that key on a
    # statement's recipe (`_inline_statement_call`) read it there, and the
    # slot analyses see the slot as read.
    bindings = Any[Expr(:(=), _inline_slot_name(slot), Expr(:ref, _OPS_ARG, slot))]
    for (name, arg) in zip(params, callargs)
        push!(bindings, Expr(:(=), renamed[name], arg))
    end
    # A capture is a field of the closure in the slot, read through the
    # literal slot (never the bare table, which the slot analyses read as
    # "the whole table" and `_rewrite_embedded` cannot shift).
    for name in names
        push!(bindings, Expr(:(=), renamed[name],
                             Expr(:call, GlobalRef(Core, :getfield),
                                  Expr(:ref, _OPS_ARG, slot), QuoteNode(name))))
    end
    # The slot keeps the closure under the type every build of this source
    # shares (`_kernel_shared_closure`), so one program's fresh builds have
    # one table type and one compiled body.
    shared = _kernel_shared_closure(closure, Expr(:->, Expr(:tuple, params...), source))
    Expr(:let, Expr(:block, bindings...), rewritten), shared
end

# An inlined statement's right-hand side read back as the call it replaced:
# `(slot, callargs)` for `let var"#opk" = __ops__[k], var"#x#opk" = x, …; body
# end`, or `nothing` for any other expression. A capture binding
# (`var"#c#opk" = getfield(__ops__[k], :c)`) is not an argument.
function _inline_statement_call(rhs)
    rhs isa Expr && rhs.head === :let && length(rhs.args) == 2 || return nothing
    header = rhs.args[1]
    header isa Expr && header.head === :block && !isempty(header.args) || return nothing
    first_binding = header.args[1]
    first_binding isa Expr && first_binding.head === :(=) || return nothing
    slot = _operation_slot(first_binding.args[2])
    slot === nothing && return nothing
    callargs = Any[]
    for binding in header.args[2:end]
        binding isa Expr && binding.head === :(=) || return nothing
        value = binding.args[2]
        _inline_capture_read(value, slot) && continue
        push!(callargs, value)
    end
    slot, callargs
end
_inline_capture_read(value, slot) = value isa Expr && value.head === :call &&
    length(value.args) == 3 && value.args[1] == GlobalRef(Core, :getfield) &&
    _operation_slot(value.args[2]) == slot && value.args[3] isa QuoteNode

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
        # A source reading a binding the module can reassign keeps its call:
        # its generated source function was compiled when the kernel was
        # defined and names the value the binding held then (`compile`,
        # `_canonical_globals`), where the body compiles at preparation and
        # would read the binding as it is by then (`test_authoring.jl`,
        # "lowered exact calls retain composition identity").
        _global_isconst(mod, ex) || return nothing
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
            elseif head === :parameters && item isa Symbol
                # `(; x)` and `f(; x)` name the field or keyword `x` by the
                # value `x` reads: the label stays, the value is renamed.
                value = _inline_qualify(item, mod, renamed)
                value === nothing && return nothing
                push!(args, Expr(:kw, item, value))
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
