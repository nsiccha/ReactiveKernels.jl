# A recipe's authored source emitted directly into the generated body
# (operation-table migration, stages 1 and 2; decision `0dgfkf5`).
#
# An authored recipe used to be one entry of the operation table, a
# `_KernelSourceOp` around a separately generated function of the recipe's
# source, called as `lhs = __ops__[k](x, y)`. The body carries the source
# itself,
#
#     lhs = let var"#x#opk" = x, var"#y#opk" = y
#         <the recipe's native source, its names bound hygienically>
#     end
#
# with its locals named from the recipe's position `k` in the lowered recipes
# (`_inline_local_name`), so the statement names its recipe for the passes
# that read one (`_inline_statement_call`) and references no table. A source
# with lexical captures reads them from the constants table, where its
# recipe's position holds the record's captured values
# (`_lowering_tables`): `var"#c#opk" = getfield(__constants__[k], :c)`.
# A literal statement is its literal (`shift::Float64 = -1.5`), an alias its
# argument (`y = x`). The plan, its recipes and `Recipe.source` are untouched,
# so `explain`, `recipe_inventory` and the structural graph read as before,
# and `code_expr` shows the code that executes (`docs/src/constraints.md`).
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

# The statement's locals are named from the recipe's position in the lowered
# recipes, never minted with `gensym`: two lowerings of one plan then emit one
# expression (`lower(p)` equals `lower_with_ops(p)`'s, and a prepared kernel's
# `code_expr` is a function of its plan). A leading `#` keeps them apart from
# every authored identifier, and the `#op` infix from the `#name#N` locals
# that `_value_local_name` derives from gensyms and from `gensym`'s own
# `##name#N`. `_canonical_locals` treats them as the lowering's locals, as it
# does gensyms; an embedding shifts the position with the recipes
# (`_embedded_statements`).
_inline_local_name(name::Symbol, slot::Int) = Symbol('#', name, "#op", slot)
# The position an inline statement's local was named from, or `nothing`.
function _inline_local_index(name::Symbol)
    text = String(name)
    startswith(text, '#') || return nothing
    i = findlast("#op", text)
    i === nothing && return nothing
    digits = SubString(text, last(i) + 1)
    isempty(digits) || !all(isdigit, digits) ? nothing : parse(Int, digits)
end
_inline_shifted_local(name::Symbol, offset::Int) = (index = _inline_local_index(name);
    index === nothing ? name : Symbol(SubString(String(name), 1, findlast("#op", String(name))[1] - 1),
                                      "#op", index + offset))

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

# The inline form of recipe `r` over `callargs` (the recipe at position `slot`
# of the lowered recipes) and the entry the callables table keeps for it (its
# record, or the metadata around one; a prepared kernel drops it unless a
# plate lowering reads it), or `nothing` where the recipe keeps its call.
function _inline_source_form(r::Recipe, callargs, slot::Int)
    op = r.op
    op isa _KernelSourceOp || return nothing
    kernel_sourceop_form(op) === :fused || return nothing
    # A plain source inlines, and so do a fused source with data-only split
    # nodes (`_KernelSplit`) and a lazy branch (`_KernelBranch`): a residual
    # plan calls the whole source, and partial evaluation and the plate
    # lowerings read the parts from the plan's recipe or from the metadata at
    # the table position (`_recipe_table_entry`), never from this statement,
    # so the whole source (`c ? a : b` as an `if`, as lazy) is its code. A
    # generator sum (`_KernelReduction`) keeps its call: its source is one
    # Julia owns.
    # The table keeps the operation's whole entry at the position (the
    # branch or split metadata, whose parts a plate lowering reads there);
    # the source inlined is that of the whole call's record.
    entry = getfield(op, :f)
    f = entry
    f isa _KernelBranch && (f = getfield(f, :call))
    f isa _KernelSplit && (f = getfield(f, :call))
    f isa _KernelSource || return nothing
    # A source whose closure Julia must own (a local function, a generator or
    # comprehension, an exception scope, a boxed capture) keeps its native
    # closure and its call: inside a generated body Julia creates such a
    # closure at run time and every value derived from it dispatches
    # dynamically.
    _source_native(f) === nothing || return nothing
    # The throw-stripped twin (`on_error = :ignore`) carries the authored
    # source, which is not its code; `prepare(…; on_error = :ignore)` turns
    # this emitter off for that preparation (`_INLINE_SOURCES_OFF`), so an
    # operation without a twin here is one built without one: a split part,
    # an outlined cell subexpression (`_cell_outline`), a hand-built
    # operation. Its source is its code.
    source = r.source
    captures = _source_captures(f)
    if _inline_literal(source)
        # A literal statement (`a::Float64 = -1.5`): the literal is its code.
        isempty(callargs) && isempty(captures) || return nothing
        return source, entry
    end
    if source isa Symbol
        # An alias (`y::Float64 = x`): the statement is its one argument.
        params = _recipe_source_names(r)
        length(params) == length(callargs) == 1 && only(params) === source || return nothing
        isempty(captures) || return nothing
        return only(callargs), entry
    end
    source isa Expr || return nothing
    params = _recipe_source_names(r)
    length(params) == length(callargs) && allunique(params) || return nothing
    mod = _kernel_endpoint_source_module(op)
    mod isa Module || return nothing
    names = keys(captures)
    native = _kernel_native_body(source, mod, Set{Symbol}(params))
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
    # The parameters bound to the call's arguments, then the captures: each
    # a field of the captured values the constants table holds at the
    # recipe's position, read through that literal entry (never the bare
    # table, which the table analyses read as "the whole table" and
    # `_rewrite_embedded` cannot shift).
    bindings = Any[Expr(:(=), renamed[name], arg) for (name, arg) in zip(params, callargs)]
    for name in names
        push!(bindings, Expr(:(=), renamed[name], _inline_capture_expr(slot, name)))
    end
    isempty(bindings) && return rewritten, entry
    Expr(:let, Expr(:block, bindings...), rewritten), entry
end

# `getfield(__constants__[k], :name)`: the captured value `name` of the source
# at position `k`, whose captured values the constants table holds there.
_inline_capture_expr(slot::Int, name::Symbol) =
    Expr(:call, GlobalRef(Core, :getfield), Expr(:ref, _CONSTANTS_ARG, slot),
         QuoteNode(name))

# An inlined statement's right-hand side read back as the call it replaced:
# `(slot, callargs)` for `let var"#x#opk" = x, …; body end`, the position `k`
# its locals were named from and the bound arguments, or `nothing` for any
# other expression. A capture binding (`var"#c#opk" = getfield(__constants__[k], :c)`)
# is not an argument. A statement with no binding (a literal, an alias, a
# source over no argument) names no recipe.
function _inline_statement_call(rhs)
    rhs isa Expr && rhs.head === :let && length(rhs.args) == 2 || return nothing
    header = rhs.args[1]
    header isa Expr && header.head === :block && !isempty(header.args) || return nothing
    first_binding = header.args[1]
    first_binding isa Expr && first_binding.head === :(=) &&
        first_binding.args[1] isa Symbol || return nothing
    slot = _inline_local_index(first_binding.args[1])
    slot === nothing && return nothing
    callargs = Any[]
    for binding in header.args
        binding isa Expr && binding.head === :(=) || return nothing
        value = binding.args[2]
        _inline_capture_read(value, slot) && continue
        push!(callargs, value)
    end
    slot, callargs
end
_inline_capture_read(value, slot) = value isa Expr && value.head === :call &&
    length(value.args) == 3 && value.args[1] == GlobalRef(Core, :getfield) &&
    _constant_slot(value.args[2]) == slot && value.args[3] isa QuoteNode

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
