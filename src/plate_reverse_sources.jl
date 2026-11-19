# A registered source describes one Julia method. Select a concrete scalar
# signature, then require the caller's inferred cell reads to prove that
# signature before entering the rule. A different overload keeps ordinary AD.
struct _PRMissingScalarSource end
const _PR_SOURCE_STACK = gensym(:plate_reverse_sources)
_pr_contains_source_guard(x) = x isa Expr && (Meta.isexpr(x, :return, 1) &&
    _index_run_call(only(x.args), GlobalRef(@__MODULE__, :_PRMissingScalarSource), 0) ||
    any(_pr_contains_source_guard, x.args))

Base.@nospecializeinfer function _pr_source_cut(@nospecialize(f::Function), ref, args, flags)
    types = Tuple(_pr_source_argtype(a) for a in args)
    stack = get(task_local_storage(), _PR_SOURCE_STACK, ())
    key = (f, types)
    key in stack && _pr_fail("recursive scalar source for `$(nameof(f))`")
    source = _traceable_source(typeof(f), types...)
    if source !== nothing
        selected = Base._which(Tuple{typeof(f),types...})
        records = Tuple{typeof(_traceable_source),Type{typeof(f)},(Type{T} for T in types)...}
        record = Base._which(records)
        if selected === nothing || selected.method.sig != source[4] ||
                selected.method.primary_world > record.method.primary_world ||
                !isempty(source[5]) || source[6]
            source = nothing
        end
    end
    source === nothing && (source = _provided_source(f, types))
    if source === nothing && !isempty(stack)
        source = _pr_forward_source(f, types)
    end
    source === nothing && return nothing
    native_type = Base.promote_op(f, types...)
    native_type in (Float16, Float32, Float64) || return nothing
    names, body, mod = source
    expression = Expr(:function, Expr(:tuple, names...), _pr_source_constants(body))
    cut = task_local_storage(_PR_SOURCE_STACK, (stack..., key)) do
        _pr_expression_cut(expression, mod, args, flags, ())
    end
    cut === nothing && return nothing
    matches = foldl((a, b) -> Expr(:&&, a, b),
                    (:($x isa $T) for (x, T) in zip(args, types)); init = true)
    result = gensym(:source_result)
    quote
        if !$matches
            return $(GlobalRef(@__MODULE__, :_PRMissingScalarSource))()
        end
        $result = $cut
        # A native return annotation can convert the raw value of the
        # registered expression. Propagate the cut only across an identity.
        if !(first($result) isa $native_type)
            return $(GlobalRef(@__MODULE__, :_PRMissingScalarSource))()
        end
        $result
    end
end

_pr_source_argtype(x::QuoteNode) = Core.Typeof(x.value)
_pr_source_argtype(x::Number) = typeof(x)
_pr_source_argtype(x) = Float64

function _pr_source_literal(x)
    x isa QuoteNode && return Some(x.value)
    x isa Union{Type,Module,Function,Number} && return Some(x)
    if x isa GlobalRef && isdefined(x.mod, x.name) && isconst(x.mod, x.name)
        return Some(getglobal(x.mod, x.name))
    end
    nothing
end

function _pr_foldable_call(f, signature)
    effects = Base.infer_effects(f, signature)
    Core.Compiler.is_foldable(effects) && Core.Compiler.is_nothrow(effects)
end

# Providers can encode a call's lexical module or dispatch tag as a literal
# type application and a singleton constructor. Preserve any constructor that
# can throw or has observable effects, including those inside a lazy arm.
function _pr_source_constants(x)
    x isa Expr && !(x.head in (:quote, :inert, :meta)) || return x
    args = map(_pr_source_constants, x.args)
    if x.head === :. && length(args) == 2 && args[2] isa QuoteNode
        owner = _pr_source_literal(args[1])
        name = args[2].value
        if owner !== nothing && something(owner) isa Module && name isa Symbol
            ref = GlobalRef(something(owner), name)
            _pr_source_literal(ref) === nothing || return ref
        end
    elseif x.head === :curly
        literals = map(_pr_source_literal, args)
        if all(!isnothing, literals) && something(first(literals)) isa UnionAll
            try
                return QuoteNode(Core.apply_type(map(something, literals)...))
            catch err
                err isa Union{TypeError,ArgumentError} || rethrow()
            end
        end
    elseif x.head === :call && length(args) == 1
        constructor = _pr_source_literal(only(args))
        if constructor !== nothing
            T = something(constructor)
            if T isa DataType && Base.issingletontype(T) && Base.promote_op(T) === T &&
                    _pr_foldable_call(T, Tuple{})
                return QuoteNode(T.instance)
            end
        end
    end
    Expr(x.head, args...)
end

function _pr_ir_value(code, x)
    x isa Core.SSAValue && return code.ssavaluetypes[x.id]
    x isa Core.SlotNumber && return code.slottypes[x.id]
    literal = _pr_source_literal(x)
    literal === nothing ? Any : Core.Const(something(literal))
end

function _pr_ir_callee(code, x)
    value = _pr_ir_value(code, x)
    value isa Core.Const ? value.val : nothing
end

# An inferred constant callee may have come from an effectful expression.
# Elide its lookup only if it is a constant binding or a foldable, nonthrowing call.
function _pr_ir_static_lookup(code, stmt)
    stmt isa GlobalRef && return _pr_source_literal(stmt) !== nothing
    Meta.isexpr(stmt, :call) || return false
    lookup = _pr_ir_callee(code, first(stmt.args))
    lookup === nothing && return false
    types = map(x -> Core.Compiler.widenconst(_pr_ir_value(code, x)), stmt.args[2:end])
    _pr_foldable_call(lookup, Tuple{types...})
end

# Follow a straight-line scalar call inside a registered source. For a
# dispatcher that splats its vararg slot, recover exactly those arguments.
# Every other numeric statement retains its position and its SSA dependencies.
function _pr_forward_source(f, types)
    Base.issingletontype(typeof(f)) || return nothing
    f isa Union{Core.Builtin,Core.IntrinsicFunction} && return nothing
    selected = Base._which(Tuple{typeof(f),types...})
    selected === nothing && return nothing
    entries = Base.code_typed(f, Tuple{types...}; optimize = false)
    length(entries) == 1 || return nothing
    code = first(only(entries))
    code isa Core.CodeInfo || return nothing
    result = last(code.code)
    result isa Core.ReturnNode && isdefined(result, :val) || return nothing
    result.val isa Core.SSAValue && result.val.id == length(code.code) - 1 || return nothing

    method = selected.method
    names = [gensym(:argument) for _ in types]
    bindings = Dict{Int,Any}()
    function operand(x)
        x isa Core.SSAValue && return get(bindings, x.id, nothing)
        if x isa Core.SlotNumber && 2 <= x.id <= length(names) + 1 &&
                (!method.isva || x.id < method.nargs)
            return names[x.id - 1]
        end
        _pr_source_literal(x) === nothing ? nothing : x
    end

    body = Expr(:block)
    for (i, stmt) in enumerate(@view code.code[1:end-1])
        value = code.ssavaluetypes[i]
        if value isa Core.Const && value.val isa Union{Function,Type,Module}
            _pr_ir_static_lookup(code, stmt) || return nothing
            bindings[i] = QuoteNode(value.val)
            continue
        end
        Meta.isexpr(stmt, :call) || return nothing
        T = Core.Compiler.widenconst(value)
        T isa Type && T !== Union{} && T <: Real || return nothing
        called = _pr_ir_callee(code, first(stmt.args))
        if called === Core._apply_iterate
            length(stmt.args) == 4 && _pr_ir_callee(code, stmt.args[2]) === Base.iterate &&
                method.isva && stmt.args[4] isa Core.SlotNumber &&
                stmt.args[4].id == method.nargs || return nothing
            called = _pr_ir_callee(code, stmt.args[3])
            args = names[method.nargs-1:end]
        else
            args = map(operand, stmt.args[2:end])
            any(isnothing, args) && return nothing
        end
        called isa Function && called !== f || return nothing
        name = bindings[i] = gensym(:forwarded)
        push!(body.args, Expr(:(=), name, Expr(:call, QuoteNode(called), args...)))
    end
    push!(body.args, bindings[result.val.id])
    (names, body, method.module)
end
