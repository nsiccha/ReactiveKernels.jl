# Ordinary helper comprehensions, recovered from Julia's lowered code. Only a
# straight-line constructor of one Generator/collect is admitted. Its element
# program is retained as scalar source; array reads must be at that element.
# No helper is executed to discover its dependencies.
struct _CellHelperArg
    expr::Any
    type::Any
end
struct _CellHelperClosure
    type::Any
    fields::Vector{Any}
end
struct _CellHelperGenerator
    closure::_CellHelperClosure
    iterator::_CellHelperArg
end

_cell_helper_type(x::_CellHelperArg) = x.type
_cell_helper_type(x::_CellHelperClosure) = x.type
_cell_helper_type(x) = typeof(x)
_cell_helper_expr(x::_CellHelperArg) = x.expr
_cell_helper_expr(x) = QuoteNode(x)

function _cell_helper_code(F, types)
    signature = Tuple{F, types...}
    # `which` on abstract arguments can choose a fallback although a concrete
    # subtype dispatches elsewhere. A row proof must cover every admitted call.
    matches = Base._methods_by_ftype(signature, -1, Base.get_world_counter())
    matches === nothing && return nothing
    length(matches) == 1 && only(matches).fully_covers || return nothing
    method = only(matches).method
    method.isva && return nothing
    ci = Base.uncompressed_ast(method)
    ci isa Core.CodeInfo || return nothing
    any(st -> st isa Union{Core.GotoNode,Core.GotoIfNot,Core.PhiNode,Core.PiNode} ||
        Meta.isexpr(st, :static_parameter), ci.code) && return nothing
    ci
end

# Reconstruct one unbranched method. At the outer level only closure/generator
# construction and wrapper calls are admitted. Inside the element, retain all
# statements (including unused calls), without duplicating SSA computations.
function _cell_helper_ir(F, arguments; closure = nothing, scalar = false,
                         stack = Set{Any}())
    key = (F, Tuple(_cell_helper_type.(arguments)))
    key in stack && return nothing
    ci = _cell_helper_code(F, last(key))
    ci === nothing && return nothing
    push!(stack, key)
    try
        slots = Dict{Int,Any}(i + 1 => a for (i, a) in enumerate(arguments))
        slots[1] = closure === nothing ? (Base.issingletontype(F) ? F.instance : nothing) : closure
        ssa = Dict{Int,Any}()
        statements = Any[]
        maps = 0
        value(x) = x isa Core.SSAValue ? get(ssa, x.id, nothing) :
            x isa Core.SlotNumber ? get(slots, x.id, nothing) :
            x isa Core.Argument ? get(slots, x.n, nothing) :
            x isa GlobalRef ? (isdefined(x.mod, x.name) && isconst(x.mod, x.name) ?
                getglobal(x.mod, x.name) : nothing) :
            x isa QuoteNode ? x.value : x isa Expr ? nothing : x
        function rhs(ex)
            ex isa Expr || return value(ex)
            if ex.head === :new && !scalar
                T = value(ex.args[1])
                T isa DataType && T <: Function || return nothing
                fields = Any[value(a) for a in ex.args[2:end]]
                length(fields) == fieldcount(T) || return nothing
                return _CellHelperClosure(T, fields)
            end
            ex.head === :call || return nothing
            f = value(ex.args[1])
            args = Any[value(a) for a in ex.args[2:end]]
            any(isnothing, args) && return nothing
            if f === Core.getfield && length(args) == 2 && args[1] isa _CellHelperClosure
                c, field = args
                j = field isa Symbol ? findfirst(==(field), fieldnames(c.type)) : field
                j isa Int && 1 <= j <= length(c.fields) || return nothing
                return c.fields[j]
            end
            if f === Core.apply_type &&
                    all(a -> a isa Type || a isa Symbol || isbits(a), args)
                return Core.apply_type(args...)
            end
            if scalar
                f isa Union{Function,Type} || return nothing
                callee = f isa Type ? QuoteNode(f) :
                    something(_const_call_ref(f), QuoteNode(f))
                ex = f === Base.getindex ? Expr(:ref, _cell_helper_expr.(args)...) :
                    Expr(:call, callee, _cell_helper_expr.(args)...)
                return _CellHelperArg(ex, Any)
            end
            f === Core.typeof && length(args) == 1 && return _cell_helper_type(only(args))
            if f === Base.Generator && length(args) == 2 &&
                    args[1] isa _CellHelperClosure && args[2] isa _CellHelperArg
                return _CellHelperGenerator(args...)
            end
            if f === Base.collect && length(args) == 1 && only(args) isa _CellHelperGenerator
                maps += 1
                return only(args)
            end
            f isa Function && Base.issingletontype(typeof(f)) &&
                !(parentmodule(f) in (Core, Base)) || return nothing
            nested = _cell_helper_ir(typeof(f), args; stack)
            nested isa _CellHelperGenerator || return nothing
            maps += 1
            nested
        end
        for (i, st) in enumerate(ci.code)
            if st isa Core.ReturnNode
                isdefined(st, :val) || return nothing
                result = value(st.val)
                if scalar
                    result === nothing && return nothing
                    push!(statements, _cell_helper_expr(result))
                    return Expr(:block, statements...)
                end
                return maps == 1 && result isa _CellHelperGenerator ? result : nothing
            elseif st === nothing || st isa LineNumberNode || st isa Core.NewvarNode ||
                   st isa Expr && st.head in (:meta, :code_coverage_effect, :inbounds)
                continue
            end
            assigned = Meta.isexpr(st, :(=), 2)
            v = rhs(assigned ? st.args[2] : st)
            v === nothing && return nothing
            if scalar && v isa _CellHelperArg && v.expr isa Expr
                name = gensym(:helper_value)
                push!(statements, Expr(:(=), name, v.expr))
                v = _CellHelperArg(name, v.type)
            end
            ssa[i] = v
            if assigned
                st.args[1] isa Core.SlotNumber || return nothing
                slots[st.args[1].id] = v
            end
        end
        nothing
    finally
        delete!(stack, key)
    end
end

function _cell_helper_call(r::Recipe, input_types; outlined = false, stable = false)
    input_types === nothing && return nothing
    length(r.outputs) == 1 && !r.effectful || return nothing
    if r.op isa _KernelSourceOp
        source = _cell_native(r; outlined)
        source === nothing && return nothing
        ex, params, mod = source
        Meta.isexpr(ex, :call) || return nothing
        _native_callee_shadowed(ex.args[1], params) && return nothing
        stable && !_kernel_constant_callable(mod, ex.args[1]) && return nothing
        f = _cell_binding(ex.args[1], mod)
        all(a -> a in params, ex.args[2:end]) || return nothing
        positions = [findfirst(==(a), params) for a in ex.args[2:end]]
    else
        f = r.op
        params = Symbol[v.name for v in r.inputs]
        positions = collect(eachindex(params))
    end
    args = [_CellHelperArg(params[j], input_types[j]) for j in positions]
    (; f, params, args)
end

function _cell_helper_form(r::Recipe, input_types; outlined = false)
    call = _cell_helper_call(r, input_types; outlined)
    call === nothing && return nothing
    (; f, params, args) = call
    f isa Function && Base.issingletontype(typeof(f)) &&
        !(parentmodule(f) in (Core, Base)) || return nothing
    mapped = _cell_helper_ir(typeof(f), args)
    mapped isa _CellHelperGenerator || return nothing
    iterator = findfirst(==(mapped.iterator.expr), params)
    iterator === nothing && return nothing
    I = input_types[iterator]
    # These iterators collect in linear-index order. An arbitrary vector may
    # override `iterate`, so its indexing alone cannot prove the same rows.
    I <: Union{Vector{<:Integer},UnitRange{<:Integer},
               StepRange{<:Integer,<:Integer},Base.OneTo{<:Integer}} &&
        !(eltype(I) <: Bool) || return nothing
    index = gensym(:helper_index)
    body = _cell_helper_ir(mapped.closure.type, [_CellHelperArg(index, eltype(I))];
                          closure = mapped.closure, scalar = true)
    body === nothing && return nothing
    roles = fill(:whole, length(params))
    roles[iterator] = :index
    indices = [Int[] for _ in params]
    tails = Any[nothing for _ in params]
    for (j, param) in enumerate(params)
        input_types[j] <: AbstractArray || continue
        reads, suffixes = Any[], Any[]
        for st in body.args
            ex = Meta.isexpr(st, :(=), 2) ? st.args[2] : st
            _cell_capture_reads!(reads, ex, Set([param]), @__MODULE__; tails = suffixes) ||
                return nothing
        end
        isempty(reads) && continue
        j == iterator && return nothing
        all(==(index), reads) || return nothing
        # Linear indexing has the iterator's vector shape. Other axes need
        # a separate shape proof; keep the ordinary whole-value path for now.
        all(isempty, suffixes) || return nothing
        roles[j] = :linear_gather
        indices[j] = [iterator]
        tails[j] = [(iterator, ())]
    end
    # As for authored plates, infer the buffer's element type from the actual
    # prepared operands. Abstract array ports need not expose their getindex
    # return type during graph analysis. This program is only used for type
    # inference; evaluation retains the recovered body in the runtime loop.
    element_program = compile(Expr(:->, Expr(:tuple, params..., index), body))
    rowmap = (; params, iterator, index, body, element_program)
    _CellForm(:pointwise, roles, Int[isempty(ix) ? 0 : first(ix) for ix in indices],
        [iterator], 0, indices, Any[], _cell_direct_reads(indices), nothing,
        tails, nothing, rowmap)
end

# Ordinary array arithmetic has stricter shape checks than broadcasting. Only
# Base's array methods over standard storage have this row law. A custom array
# can specialize either arithmetic or broadcasting, and an abstract array type
# is not evidence that its concrete values use the standard methods.
# A custom numeric element type can own array/broadcast overloads too.
_cell_arithmetic_number(T) = T in (Bool, Int8, Int16, Int32, Int64, Int128,
    UInt8, UInt16, UInt32, UInt64, UInt128, Float16, Float32, Float64, BigInt, BigFloat)
_cell_arithmetic_number(::Type{Complex{T}}) where {T} = _cell_arithmetic_number(T)
_cell_arithmetic_number(::Type{Rational{T}}) where {T} = _cell_arithmetic_number(T)
_cell_arithmetic_array(T) = false
_cell_arithmetic_array(::Type{<:Array{T}}) where {T} = _cell_arithmetic_number(T)
_cell_arithmetic_array(::Type{<:SubArray{<:Number,N,P}}) where {N,P} =
    _cell_arithmetic_array(P)

# Abstract ports can still produce dense gathers. Prove the arithmetic for a
# concrete storage type, then check the original producer's inferred type in
# the emitted program before using that proof. A custom array is never treated
# as dense merely because its port is abstract.
function _cell_arithmetic_type(T)
    _cell_arithmetic_array(T) && return T
    T isa DataType && T <: AbstractArray{<:Number} || return nothing
    isconcretetype(T) && return nothing
    E = eltype(T)
    _cell_arithmetic_number(E) || return nothing
    S = Array{E,ndims(T)}
    S <: T ? S : nothing
end

@inline function _cell_check_arithmetic_type(actual, expected)
    actual <: expected || throw(ArgumentError(
        "local array arithmetic requires an original operand of type $expected; inferred $actual"))
    nothing
end

function _cell_forwarded_arithmetic(f, args, stack = Set{Any}())
    if f in (Base.:+, Base.:-) && length(args) == 2
        types = _cell_arithmetic_type.([a.type for a in args])
        any(isnothing, types) && return nothing
        matches = Base._methods_by_ftype(Tuple{typeof(f), types...},
                                        -1, Base.get_world_counter())
        matches === nothing && return nothing
        length(matches) == 1 && only(matches).fully_covers || return nothing
        method = only(matches).method
        method.module === Base || return nothing
        standard = which(f, Tuple{AbstractArray,AbstractArray})
        dense = which(f, Tuple{Array,Array})
        method in (standard, dense) || return nothing
        return (; f, args, types)
    end
    f isa Function && Base.issingletontype(typeof(f)) &&
        !(parentmodule(f) in (Core, Base)) || return nothing
    key = (typeof(f), Tuple(a.type for a in args))
    key in stack && return nothing
    ci = _cell_helper_code(key...)
    ci === nothing && return nothing
    # A transparent wrapper does exactly one call and returns that call. In
    # particular, do not discard unused calls, checks, conversions or effects.
    statements = [(i, st) for (i, st) in enumerate(ci.code) if
        !(st === nothing || st isa LineNumberNode || st isa Core.NewvarNode ||
          st isa Expr && st.head in (:meta, :code_coverage_effect, :inbounds))]
    length(statements) == 2 || return nothing
    (index, ex), (_, ret) = statements
    Meta.isexpr(ex, :call) && ret isa Core.ReturnNode && isdefined(ret, :val) &&
        ret.val === Core.SSAValue(index) || return nothing
    callee = ex.args[1]
    callee isa GlobalRef && isdefined(callee.mod, callee.name) &&
        isconst(callee.mod, callee.name) || return nothing
    all(a -> a isa Core.SlotNumber && 2 <= a.id <= length(args) + 1,
        ex.args[2:end]) || return nothing
    push!(stack, key)
    try
        _cell_forwarded_arithmetic(getglobal(callee.mod, callee.name),
            [args[a.id - 1] for a in ex.args[2:end]], stack)
    finally
        delete!(stack, key)
    end
end

function _cell_arithmetic_form(r::Recipe, input_types; outlined = false)
    call = _cell_helper_call(r, input_types; outlined, stable = true)
    call === nothing && return nothing
    arithmetic = _cell_forwarded_arithmetic(call.f, call.args)
    arithmetic === nothing && return nothing
    positions = Int[findfirst(==(a.expr), call.params) for a in arithmetic.args]
    roles = fill(:whole, length(r.inputs))
    roles[positions] .= :leaf
    _CellForm(:array_arithmetic, roles, zeros(Int, length(roles)), positions, 0,
        [Int[] for _ in roles], Any[], [Int[] for _ in roles],
        (; f = arithmetic.f, positions, types = arithmetic.types))
end

function _cell_helper_window!(e, mapping, args)
    result, position = gensym(:helper_rows), gensym(:helper_position)
    parameters = mapping.params
    iterator = parameters[mapping.iterator]
    bindings = Expr(:block, [Expr(:(=), p, a) for (p, a) in zip(parameters, args)]...)
    types = Any[Expr(:call, GlobalRef(Core, :typeof), p) for p in parameters]
    push!(types, Expr(:call, GlobalRef(Base, :eltype), iterator))
    element = Expr(:call, GlobalRef(@__MODULE__, :_promote_op),
                   QuoteNode(mapping.element_program), types...)
    body = Expr(:block,
        :($result = $(GlobalRef(Base, :Vector)){$element}(
            $(GlobalRef(Base, :undef)), $(GlobalRef(Base, :length))($iterator))),
        Expr(:for, :($position = $(GlobalRef(Base, :eachindex))($iterator)), Expr(:block,
            :($(mapping.index) = $iterator[$position]),
            Expr(:(=), Expr(:ref, result, position), mapping.body))), result)
    _cell_bind!(e, :cell_rows, Expr(:let, bindings, body))
end
