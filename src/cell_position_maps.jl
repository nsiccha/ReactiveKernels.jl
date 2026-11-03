# Row forms of ordinary helpers that map positions to rows, recovered from the
# helper's lowered Julia code. No helper is executed to discover them.
#
# A helper qualifies when its result has one row per position of the vectors
# it iterates and row `g` reads each "position" argument only at `g`:
#
# - a comprehension or `map` over `eachindex(x)`, `axes(x, 1)`, `1:length(x)`,
#   `Base.OneTo(length(x))`, `x` itself, `enumerate(...)` or `zip(...)` of
#   those (destructured or not), including straight-line wrappers;
# - a loop over one such iterator that builds the result by exactly one
#   unconditional `push!(out, row)` onto an empty vector, or `out[g] = row`
#   into `similar(x)`, `zeros(T, length(x))` or `Vector{T}(undef, length(x))`,
#   after a prologue of type computations only. The body may hold branches,
#   inner loops and nested calls on per-position values; its locals must be
#   assigned in each iteration before they are read.
#
# Every other argument (`field` in `field[ix]` with `ix = paths[g]`) is passed
# whole, so it must be computed whole: the ordinary locality checks decide
# whether that is legitimate (a shared or bound value) or refused. The
# position `g` itself may only index a position argument; an expression of it
# (`x[g - 1]`), a read of the result being built, a value carried from one
# iteration to the next, `length(x)` of a position argument read as a value,
# and mutation of anything not created in the iteration all keep the helper
# without a row form.
#
# A cell's window calls the authored helper on fresh dense vectors of its
# position arguments' requested rows, and its other arguments unchanged. Both
# the original and the window argument types must dispatch to the same single
# method, so the authored loops, branches and errors run as written.

struct _CellPositionMap
    # The recipe inputs whose length gives the row count (`zip`: the least).
    sources::Vector{Int}
    # Facts the proof assumed of an input whose static type does not show
    # them: `(input, :standard)`, a vector iterated in its linear order from 1;
    # `(input, :eltype)`, a concrete element type, so `eltype(x[1])` is the same
    # for every element. Checked on the original producer's type at run time.
    checks::Vector{Tuple{Int,Symbol}}
end

struct _PMFail <: Exception end
struct _PMRestart <: Exception end
_pm_fail() = throw(_PMFail())

# An abstract value of the analysis:
# `:const` (data: the value), `:param` (an argument of the analyzed helper),
# `:pos` (the iteration position), `:elem` (an argument read at it), `:local`
# (a value computed in one iteration), `:typeval` (a type or a value computed
# from types only), `:typeof`/`:typeread`/`:len` (an argument's type, an
# element read only for its type, an argument's length), `:iter` (data: an
# iterator descriptor), `:istate`/`:state`/`:ibool` (the iteration protocol),
# `:tuple` (data: its items), `:closure`/`:closuretype`/`:gen`, `:acc` (the
# result being built; data `:push` or `(:prealloc, j)`), `:rows` (a result).
struct _PMValue
    kind::Symbol
    data::Any
end
_PMValue(kind::Symbol) = _PMValue(kind, nothing)
Base.:(==)(a::_PMValue, b::_PMValue) = a.kind === b.kind && isequal(a.data, b.data)
Base.hash(a::_PMValue, h::UInt) = hash(a.data, hash(a.kind, hash(:_PMValue, h)))

# Iterator descriptors: `(:positions, j)`, `(:over, j)`, `(:enumerate, d)` and
# `(:zip, ds)`. The sources are the arguments whose positions are iterated.
_pm_sources(d) = d[1] in (:positions, :over) ? Int[d[2]] :
    d[1] === :enumerate ? _pm_sources(d[2]) :
    unique!(reduce(vcat, map(_pm_sources, d[2]); init = Int[]))
_pm_element(d) = d[1] === :positions ? _PMValue(:pos) :
    d[1] === :over ? _PMValue(:elem, d[2]) :
    d[1] === :enumerate ? _PMValue(:tuple, Any[_PMValue(:pos), _pm_element(d[2])]) :
    _PMValue(:tuple, Any[_pm_element(x) for x in d[2]])

mutable struct _PMState
    types::Vector{Any}         # the analyzed helper's argument types
    whole::Set{Int}            # arguments read whole
    iterated::Set{Int}
    calls::Vector{Any}         # (callee type, abstract arguments, method)
    stack::Set{Any}
    checks::Set{Tuple{Int,Symbol}}
end

# Vectors whose iteration is their linear indexing from 1, as `eachindex`
# reports it. An arbitrary vector may override `iterate` or its axes.
const _PM_STANDARD = Union{Array{<:Any,1},UnitRange{<:Integer},StepRange{<:Integer,<:Integer},
                           Base.OneTo{<:Integer},StepRangeLen}
_pm_onebased_index(@nospecialize(I)) = I isa Type && I <: Union{Integer,UnitRange,StepRange,
    Base.OneTo,Base.Slice{<:Base.OneTo},Array{<:Integer,1}}
# A vector Base iterates in its linear order from position 1: one of the
# types above, a view of one-based indices, or a concrete vector whose
# `iterate`, `eachindex` and `axes` are Base's generic array methods. An
# overridden iteration or offset axes (`OffsetArray`, identity ranges) are not.
function _pm_standard_vector(T)
    T isa Type && T !== Union{} && T <: AbstractVector || return false
    T <: _PM_STANDARD && return true
    if T isa DataType && T <: SubArray
        I = T.parameters[4]
        return I isa DataType && all(_pm_onebased_index, I.parameters)
    end
    isconcretetype(T) && !(T <: Union{Base.IdentityUnitRange,Base.Slice}) || return false
    generic(f, sig) = (m = which(f, sig); m.module === Base && m.sig isa DataType &&
                       m.sig.parameters[2] in (AbstractArray, AbstractVector, Any))
    generic(iterate, Tuple{T}) && generic(eachindex, Tuple{T}) && generic(axes, Tuple{T})
end
_pm_vector_eltype(T) = T isa Type && T !== Union{} && T <: AbstractArray &&
    isconcretetype(eltype(T))

# A fact of argument `j`'s type: proven statically, checked at run time when
# its static type is abstract (the original producer's inferred type), or
# failed when its concrete static type lacks it.
function _pm_require!(st::_PMState, j::Int, fact::Symbol)
    T = st.types[j]
    holds = fact === :standard ? _pm_standard_vector(T) : _pm_vector_eltype(T)
    holds && return nothing
    T === Any || T isa Type && !isconcretetype(T) && T <: AbstractArray || _pm_fail()
    push!(st.checks, (j, fact))
    nothing
end

function _pm_whole!(st::_PMState, j::Int)
    j in st.iterated && _pm_fail()
    j in st.whole && return
    push!(st.whole, j)
    throw(_PMRestart())
end

function _pm_position!(st::_PMState, j::Int)
    j in st.whole && _pm_fail()
    T = st.types[j]
    T === Any || T isa Type && T <: AbstractVector || _pm_whole!(st, j)
    nothing
end

function _pm_iterated!(st::_PMState, j::Int)
    _pm_position!(st, j)
    _pm_require!(st, j, :standard)
    push!(st.iterated, j)
    nothing
end

_pm_static_type(st::_PMState, v::_PMValue) =
    v.kind === :param ? st.types[v.data] :
    v.kind === :const ? Core.Typeof(v.data) : Any

# The lowered code of the single method every call of `F` with these argument
# types dispatches to, or `nothing`.
function _pm_method(F, types)
    matches = Base._methods_by_ftype(Tuple{F, types...}, -1, Base.get_world_counter())
    matches === nothing && return nothing
    length(matches) == 1 && only(matches).fully_covers || return nothing
    method = only(matches).method
    method.isva && return nothing
    ci = Base.uncompressed_ast(method)
    ci isa Core.CodeInfo || return nothing
    any(st -> st isa Union{Core.PhiNode,Core.PiNode,Core.UpsilonNode,Core.PhiCNode}, ci.code) &&
        return nothing
    (method, ci)
end

const _PM_NOOP_HEADS = (:meta, :code_coverage_effect, :inbounds, :loopinfo, :aliasscope,
                        :popaliasscope)
_pm_noop(st) = st === nothing || st isa LineNumberNode || st isa Core.NewvarNode ||
    st isa Expr && st.head in _PM_NOOP_HEADS

# Type-level functions: their results depend only on the types of their
# arguments, so they are the same for an argument and its window.
const _PM_TYPE_FUNCTIONS = (Base.eltype, Base.promote_type, Base.typejoin, Base.float,
                            Core.apply_type, Base.promote_eltype, Base.nonmissingtype)

_pm_mutating(f) = f isa Function && endswith(string(nameof(f)), "!") ||
    f === Core.setfield! || f === Base.setproperty! || f === Core.arrayset

# One function's lowered code. `iteration` selects the rules inside one row's
# computation (a comprehension element or a loop body); outside it only the
# iteration's construction, closures and type computations are admitted.
mutable struct _PMFrame
    ci::Core.CodeInfo
    slots::Dict{Int,_PMValue}
    ssa::Dict{Int,_PMValue}
    iteration::Bool
    sparams::Bool
    accumulations::Vector{Int}
    accumulator::Union{Nothing,_PMValue}
    statement::Int
end

function _pm_value(fr::_PMFrame, x)
    x isa Core.SSAValue && return get(() -> _pm_fail(), fr.ssa, x.id)
    x isa Core.SlotNumber && return get(() -> _pm_fail(), fr.slots, x.id)
    x isa Core.Argument && return get(() -> _pm_fail(), fr.slots, x.n)
    if x isa GlobalRef
        isdefined(x.mod, x.name) && isconst(x.mod, x.name) || _pm_fail()
        return _PMValue(:const, getglobal(x.mod, x.name))
    end
    x isa QuoteNode && return _PMValue(:const, x.value)
    x isa Expr && _pm_fail()
    _PMValue(:const, x)
end

_pm_typelike(v::_PMValue) = v.kind in (:const, :typeval)
_pm_value_kind(v::_PMValue) = v.kind in (:const, :typeval, :local, :elem)

# An iterator from a value: a descriptor, an argument iterated directly.
function _pm_iterator(st::_PMState, v::_PMValue)
    v.kind === :iter && return v.data
    if v.kind === :param
        _pm_iterated!(st, v.data)
        return (:over, v.data)
    end
    _pm_fail()
end

# A value used whole as an argument of an ordinary call inside an iteration:
# an argument is then read whole; iteration machinery may not escape.
function _pm_operand!(st::_PMState, v::_PMValue)
    k = v.kind
    k in (:const, :typeval, :local, :elem) && return
    k === :tuple && return foreach(x -> _pm_operand!(st, x), v.data)
    k in (:param, :len, :typeof, :typeread) && return _pm_whole!(st, v.data)
    _pm_fail()
end

function _pm_call(st::_PMState, fr::_PMFrame, f::_PMValue, args::Vector{_PMValue})
    n = length(args)
    if f.kind === :typeval
        # A computed type called as a constructor: `Vector{T}()`.
        if fr.iteration
            foreach(a -> _pm_operand!(st, a), args)
            return _PMValue(:local)
        end
        n == 0 && return _PMValue(:acc, :push)
        n == 2 && args[1].kind === :const && args[1].data === undef &&
            args[2].kind === :len && return _PMValue(:acc, (:prealloc, args[2].data))
        _pm_fail()
    end
    f.kind === :const || _pm_fail()
    fv = f.data
    # Protocol and structure, in and outside an iteration.
    if fv === Core.getfield && n >= 2
        a, field = args[1], args[2]
        if a.kind === :closure && field.kind === :const
            T, fields = a.data
            j = field.data isa Symbol ? findfirst(==(field.data), fieldnames(T)) : field.data
            j isa Int && 1 <= j <= length(fields) || _pm_fail()
            return fields[j]
        elseif a.kind === :istate && field.kind === :const && field.data in (1, 2)
            return field.data == 1 ? _pm_element(a.data) : _PMValue(:state, a.data)
        elseif a.kind === :tuple && field.kind === :const && field.data isa Int
            1 <= field.data <= length(a.data) || _pm_fail()
            return a.data[field.data]
        end
    end
    if fv === Base.indexed_iterate && n in (2, 3) && args[1].kind === :tuple &&
            args[2].kind === :const && args[2].data isa Int
        i = args[2].data
        1 <= i <= length(args[1].data) || _pm_fail()
        return _PMValue(:tuple, Any[args[1].data[i], _PMValue(:const, i + 1)])
    end
    if fv === Core.:(===) && n == 2 && any(a -> a.kind === :istate, args)
        any(a -> a.kind === :const && a.data === nothing, args) || _pm_fail()
        return _PMValue(:ibool)
    end
    fv === Base.not_int && n == 1 && args[1].kind === :ibool && return _PMValue(:ibool)
    if fv === Base.eltype && n == 1
        a = only(args)
        a.kind === :param && return _PMValue(:typeval)
        if a.kind === :typeread
            _pm_require!(st, a.data, :eltype)
            return _PMValue(:typeval)
        end
    end
    if fv === Core.typeof && n == 1
        a = only(args)
        a.kind === :param && return _PMValue(:typeof, a.data)
        if a.kind === :typeread
            _pm_require!(st, a.data, :eltype)
            return _PMValue(:typeval)
        end
        _pm_typelike(a) && return _PMValue(:typeval)
    end
    if fv === Core.apply_type && n >= 1 && args[1].kind === :const &&
            args[1].data isa Type && args[1].data <: Function
        # A closure's type over its captures' types: the method is the same
        # for every parameter.
        all(a -> _pm_typelike(a) || a.kind === :typeof, args[2:end]) || _pm_fail()
        return _PMValue(:closuretype, args[1].data)
    end
    if fv in _PM_TYPE_FUNCTIONS
        for a in args
            a.kind === :typeof && _pm_whole!(st, a.data)
        end
        all(_pm_typelike, args) && return _PMValue(:typeval)
    end
    fr.iteration ? _pm_iteration_call(st, fr, fv, args) : _pm_outer_call(st, fr, fv, args)
end

# Outside an iteration: the iterator, the closure, the result's construction
# and wrappers. Anything else has no proven row form.
function _pm_outer_call(st::_PMState, fr::_PMFrame, fv, args::Vector{_PMValue})
    n = length(args)
    param(k) = n >= k && args[k].kind === :param
    if fv === Base.eachindex && n == 1 && param(1)
        _pm_iterated!(st, args[1].data)
        return _PMValue(:iter, (:positions, args[1].data))
    elseif fv === Base.axes && n == 2 && param(1) && args[2].kind === :const && args[2].data == 1
        _pm_iterated!(st, args[1].data)
        return _PMValue(:iter, (:positions, args[1].data))
    elseif fv === Base.length && n == 1 && param(1)
        return _PMValue(:len, args[1].data)
    elseif fv === Base.:(:) && n == 2 && args[1].kind === :const && args[1].data === 1 &&
            args[2].kind === :len
        _pm_iterated!(st, args[2].data)
        return _PMValue(:iter, (:positions, args[2].data))
    elseif fv === Base.OneTo && n == 1 && args[1].kind === :len
        _pm_iterated!(st, args[1].data)
        return _PMValue(:iter, (:positions, args[1].data))
    elseif fv === Base.enumerate && n == 1
        return _PMValue(:iter, (:enumerate, _pm_iterator(st, args[1])))
    elseif fv === Base.zip && n >= 1
        return _PMValue(:iter, (:zip, Any[_pm_iterator(st, a) for a in args]))
    elseif fv === Base.Generator && n == 2 && args[1].kind === :closure
        return _PMValue(:gen, (args[1], _pm_iterator(st, args[2])))
    elseif fv === Base.collect && n == 1 && args[1].kind === :gen
        return _pm_comprehension(st, args[1].data...)
    elseif fv === Base.map && n == 2 && args[1].kind === :closure
        return _pm_comprehension(st, args[1], _pm_iterator(st, args[2]))
    elseif fv === Base.iterate && n == 1
        return _PMValue(:istate, _pm_iterator(st, args[1]))
    elseif fv === Base.getindex && n == 1 && _pm_typelike(args[1])
        return _PMValue(:acc, :push)               # `T[]`
    elseif fv === Base.getindex && n == 2 && param(1) && args[2].kind === :const &&
            args[2].data isa Integer
        _pm_position!(st, args[1].data)
        return _PMValue(:typeread, args[1].data)   # `x[1]`, only for its type
    elseif fv isa Type && n == 0
        return _PMValue(:acc, :push)               # `Vector{T}()`
    elseif fv isa Type && n == 2 && args[1].kind === :const && args[1].data === undef &&
            args[2].kind === :len
        return _PMValue(:acc, (:prealloc, args[2].data))
    elseif fv === Base.similar && n in (1, 2) && param(1) && (n == 1 || _pm_typelike(args[2]))
        _pm_position!(st, args[1].data)
        return _PMValue(:acc, (:prealloc, args[1].data))
    elseif fv === Base.similar && n in (2, 3) && args[end].kind === :len &&
            (n == 2 || _pm_typelike(args[2]))
        args[1].kind === :param && _pm_whole!(st, args[1].data)
        args[1].kind in (:param, :const, :typeval) || _pm_fail()
        return _PMValue(:acc, (:prealloc, args[end].data))
    elseif fv in (Base.zeros, Base.ones) && n in (1, 2) && args[end].kind === :len &&
            (n == 1 || _pm_typelike(args[1]))
        return _PMValue(:acc, (:prealloc, args[end].data))
    elseif fv === Base.fill && n == 2 && _pm_typelike(args[1]) && args[2].kind === :len
        return _PMValue(:acc, (:prealloc, args[2].data))
    end
    # A wrapper: another helper over arguments, constants and types.
    if fv isa Function && Base.issingletontype(typeof(fv)) &&
            !(parentmodule(fv) in (Core, Base)) &&
            all(a -> a.kind in (:param, :const, :typeval), args)
        return _pm_function(st, typeof(fv), args)
    end
    _pm_fail()
end

# Inside one row's computation.
function _pm_iteration_call(st::_PMState, fr::_PMFrame, fv, args::Vector{_PMValue})
    n = length(args)
    any(a -> a.kind === :pos, args) && !(fv === Base.getindex && n == 2 && args[2].kind === :pos) &&
        !(fv === Base.setindex! && n == 3 && args[1].kind === :acc && args[3].kind === :pos) &&
        _pm_fail()
    if fv === Base.getindex && n == 2 && args[2].kind === :pos
        a = args[1]
        a.kind === :param || _pm_fail()
        _pm_position!(st, a.data)
        return _PMValue(:elem, a.data)
    end
    if any(a -> a.kind === :acc, args)
        acc = fr.accumulator
        acc === nothing && _pm_fail()
        ok = fv === Base.push! && n == 2 && args[1].kind === :acc && acc.data === :push ||
            fv === Base.setindex! && n == 3 && args[1].kind === :acc &&
                args[3].kind === :pos && acc.data isa Tuple
        ok || _pm_fail()
        _pm_operand!(st, args[2])
        push!(fr.accumulations, fr.statement)
        return _PMValue(:local)
    end
    _pm_mutating(fv) && n >= 1 && args[1].kind !== :local && _pm_fail()
    foreach(a -> _pm_operand!(st, a), args)
    _PMValue(:local)
end

# A statement's right-hand side.
function _pm_rhs(st::_PMState, fr::_PMFrame, ex)
    ex isa Expr || return _pm_value(fr, ex)
    if ex.head === :call
        f = _pm_value(fr, ex.args[1])
        args = _PMValue[_pm_value(fr, a) for a in ex.args[2:end]]
        return _pm_call(st, fr, f, args)
    elseif ex.head === :new
        T = _pm_value(fr, ex.args[1])
        fields = _PMValue[_pm_value(fr, a) for a in ex.args[2:end]]
        if fr.iteration
            T.kind in (:const, :closuretype, :typeval) || _pm_fail()
            T.kind === :const && T.data isa Type && T.data <: Core.Box && _pm_fail()
            foreach(a -> _pm_operand!(st, a), fields)
            return _PMValue(:local)
        end
        T.kind === :closuretype || _pm_fail()
        return _PMValue(:closure, (T.data, fields))
    elseif ex.head === :static_parameter
        return _PMValue(:typeval)
    elseif ex.head === :boundscheck
        return _PMValue(fr.iteration ? :local : :typeval)
    end
    _pm_fail()
end

function _pm_statement!(st::_PMState, fr::_PMFrame, i::Int)
    s = fr.ci.code[i]
    fr.statement = i
    _pm_noop(s) && return nothing
    s isa Union{Core.GotoNode,Core.GotoIfNot,Core.ReturnNode} && _pm_fail()
    assigned = Meta.isexpr(s, :(=), 2)
    v = _pm_rhs(st, fr, assigned ? s.args[2] : s)
    fr.ssa[i] = v
    if assigned
        lhs = s.args[1]
        lhs isa Core.SlotNumber || _pm_fail()
        old = get(fr.slots, lhs.id, nothing)
        fr.slots[lhs.id] = old === nothing ? v : _pm_join(old, v)
    end
    v
end

# Two values one slot holds in one row's computation.
function _pm_join(a::_PMValue, b::_PMValue)
    a == b && return a
    _pm_value_kind(a) && _pm_value_kind(b) && return _PMValue(:local)
    _pm_fail()
end

# A comprehension or `map`: its element program, analyzed as one iteration.
function _pm_comprehension(st::_PMState, closure::_PMValue, iterator)
    T, _ = closure.data
    found = _pm_method(T, (Any,))
    found === nothing && _pm_fail()
    method, ci = found
    push!(st.calls, (T, nothing, method))
    fr = _PMFrame(ci, Dict(1 => closure, 2 => _pm_element(iterator)),
                  Dict{Int,_PMValue}(), true, false, Int[], nothing, 0)
    # Each element call has fresh locals; their values are joined over two
    # passes so a branch or an inner loop sees every assignment.
    for _ in 1:2, i in eachindex(ci.code)
        s = ci.code[i]
        if s isa Core.ReturnNode
            isdefined(s, :val) || continue      # unreachable
            _pm_operand!(st, _pm_value(fr, s.val))
        elseif s isa Core.GotoNode
            continue
        elseif s isa Core.GotoIfNot
            _pm_operand!(st, _pm_value(fr, s.cond))
        elseif Meta.isexpr(s, :(=), 2) || !(_pm_noop(s))
            _pm_statement!(st, fr, i)
        end
    end
    _PMValue(:rows, iterator)
end

# One helper method over abstract arguments.
function _pm_function(st::_PMState, F, args::Vector{_PMValue})
    key = (F, Tuple(args))
    key in st.stack && _pm_fail()
    types = Tuple(_pm_static_type(st, a) for a in args)
    found = _pm_method(F, types)
    found === nothing && _pm_fail()
    method, ci = found
    push!(st.calls, (F, args, method))
    push!(st.stack, key)
    try
        slots = Dict{Int,_PMValue}(k + 1 => a for (k, a) in enumerate(args))
        slots[1] = Base.issingletontype(F) ? _PMValue(:const, F.instance) : _pm_fail()
        fr = _PMFrame(ci, slots, Dict{Int,_PMValue}(), false, false, Int[], nothing, 0)
        any(s -> s isa Union{Core.GotoNode,Core.GotoIfNot}, ci.code) &&
            return _pm_loop(st, fr)
        for (i, s) in enumerate(ci.code)
            if s isa Core.ReturnNode
                isdefined(s, :val) || _pm_fail()
                v = _pm_value(fr, s.val)
                v.kind === :rows || _pm_fail()
                return v
            end
            _pm_statement!(st, fr, i)
        end
        _pm_fail()
    finally
        delete!(st.stack, key)
    end
end

_pm_slot(x) = x isa Core.SlotNumber ? x.id : nothing
_pm_ssa(x, i) = x isa Core.SSAValue && x.id == i
_pm_iterate_call(ex, n) = Meta.isexpr(ex, :call, n + 1) && ex.args[1] == GlobalRef(Base, :iterate)

# The lowered `for` loop: a straight prologue, `I = iterate(it)` and its exit
# check, the head reading `I`, the body, `I = iterate(it, state)`, the check
# and the back edge, then `return out`.
function _pm_loop(st::_PMState, fr::_PMFrame)
    code = fr.ci.code
    setup = findfirst(s -> Meta.isexpr(s, :(=), 2) && _pm_slot(s.args[1]) !== nothing &&
                           _pm_iterate_call(s.args[2], 1), code)
    setup === nothing && _pm_fail()
    for i in 1:setup-1
        code[i] isa Union{Core.GotoNode,Core.GotoIfNot,Core.ReturnNode} && _pm_fail()
        _pm_statement!(st, fr, i)
    end
    I = _pm_slot(code[setup].args[1])
    iterator = code[setup].args[2].args[2]
    state = _pm_statement!(st, fr, setup)
    state.kind === :istate || _pm_fail()
    desc = state.data
    function check(k)
        k + 2 <= length(code) || _pm_fail()
        a, b, c = code[k], code[k+1], code[k+2]
        Meta.isexpr(a, :call, 3) && a.args[1] == GlobalRef(Core, :(===)) &&
            _pm_slot(a.args[2]) == I && a.args[3] === nothing &&
            Meta.isexpr(b, :call, 2) && b.args[1] == GlobalRef(Base, :not_int) &&
            _pm_ssa(b.args[2], k) && c isa Core.GotoIfNot && _pm_ssa(c.cond, k + 1) ||
            _pm_fail()
        c.dest
    end
    exit = check(setup + 1)
    head = setup + 4
    head < exit - 1 && _pm_slot(code[head]) == I || _pm_fail()
    back = exit - 1
    code[back] isa Core.GotoNode && code[back].label == head || _pm_fail()
    step = back - 4
    step > head || _pm_fail()
    s = code[step]
    Meta.isexpr(s, :(=), 2) && _pm_slot(s.args[1]) == I && _pm_iterate_call(s.args[2], 2) &&
        s.args[2].args[2] == iterator || _pm_fail()
    check(step + 1) == exit || _pm_fail()
    # The epilogue returns the result being built.
    acc = nothing
    for i in exit:length(code)
        r = code[i]
        _pm_noop(r) && continue
        if r isa Core.ReturnNode
            isdefined(r, :val) || _pm_fail()
            x = r.val
            x isa Core.SSAValue && (x = code[x.id])
            acc = _pm_slot(x)
            acc === nothing && _pm_fail()
            break
        elseif _pm_slot(r) !== nothing && i < length(code)
            continue
        end
        _pm_fail()
    end
    acc === nothing && _pm_fail()
    accumulator = get(fr.slots, acc, nothing)
    accumulator !== nothing && accumulator.kind === :acc || _pm_fail()
    prealloc = accumulator.data isa Tuple
    prealloc && _pm_sources(desc) != Int[accumulator.data[2]] && _pm_fail()
    body = head+1:step-1
    # The head's element and state; the step's state is the head's.
    fr.ssa[head] = state
    fr.iteration = true
    fr.accumulator = accumulator
    assigned = Set{Int}()
    for i in body
        r = code[i]
        r isa Core.ReturnNode && _pm_fail()
        if r isa Union{Core.GotoNode,Core.GotoIfNot}
            label = r isa Core.GotoNode ? r.label : r.dest
            label in body || label == step || _pm_fail()
        end
        Meta.isexpr(r, :(=), 2) && push!(assigned, _pm_slot(r.args[1]))
    end
    (I in assigned || acc in assigned || nothing in assigned) && _pm_fail()
    for k in assigned
        # A slot the prologue set and the body sets again carries its value
        # into the first iteration.
        haskey(fr.slots, k) && _pm_fail()
    end
    for _ in 1:2
        empty!(fr.accumulations)
        for i in body
            r = code[i]
            if r isa Core.GotoIfNot
                _pm_operand!(st, _pm_value(fr, r.cond))
            elseif !(r isa Core.GotoNode)
                _pm_statement!(st, fr, i)
            end
        end
    end
    v = _pm_value(fr, code[step].args[2].args[3])
    v.kind === :state && v.data == desc || _pm_fail()
    # Exactly one accumulation, executed once per iteration: no jump within
    # the body passes over it or returns to before it.
    length(fr.accumulations) == 1 || _pm_fail()
    p = only(fr.accumulations)
    for i in body
        r = code[i]
        r isa Union{Core.GotoNode,Core.GotoIfNot} || continue
        label = r isa Core.GotoNode ? r.label : r.dest
        min(i, label) < p < max(i, label) && _pm_fail()
    end
    _pm_assigned_before_read(code, head + 1, step, assigned) || _pm_fail()
    _PMValue(:rows, desc)
end

# Whether every read of a slot in `assigned` within `first:last` follows an
# assignment on every path from `first`: no value crosses iterations.
function _pm_assigned_before_read(code, first::Int, last::Int, assigned::Set{Int})
    range = first:last
    successors(i) = let s = code[i]
        s isa Core.GotoNode ? [s.label] :
        s isa Core.GotoIfNot ? [i + 1, s.dest] : [i + 1]
    end
    defined = Dict{Int,Set{Int}}()           # assigned on entry to a statement
    defined[first] = Set{Int}()
    work = [first]
    changed = true
    while changed
        changed = false
        for i in range
            haskey(defined, i) || continue
            entry = defined[i]
            s = code[i]
            out = copy(entry)
            Meta.isexpr(s, :(=), 2) && push!(out, s.args[1].id)
            for t in successors(i)
                t in range || continue
                t == first && return false
                if !haskey(defined, t)
                    defined[t] = out
                    changed = true
                else
                    meet = intersect(defined[t], out)
                    if meet != defined[t]
                        defined[t] = meet
                        changed = true
                    end
                end
            end
        end
    end
    function reads(x, found)
        x isa Core.SlotNumber && push!(found, x.id)
        x isa Expr && foreach(a -> reads(a, found), x.args)
        found
    end
    for i in range
        haskey(defined, i) || continue
        s = code[i]
        ex = Meta.isexpr(s, :(=), 2) ? s.args[2] : s isa Core.GotoIfNot ? s.cond : s
        for k in reads(ex, Set{Int}())
            k in assigned && !(k in defined[i]) && return false
        end
    end
    true
end

# The proof for a helper of type `F` over argument types `types`: which
# arguments are read at the row only, which are iterated, or `nothing`.
function _cell_position_proof(F, types::Vector{Any})
    whole = Set{Int}()
    for _ in 0:length(types)
        st = _PMState(types, whole, Set{Int}(), Any[], Set{Any}(), Set{Tuple{Int,Symbol}}())
        result = try
            _pm_function(st, F, _PMValue[_PMValue(:param, j) for j in eachindex(types)])
        catch err
            err isa _PMRestart && continue
            err isa _PMFail && return nothing
            rethrow()
        end
        result.kind === :rows || return nothing
        sources = _pm_sources(result.data)
        isempty(sources) && return nothing
        positions = Int[j for j in eachindex(types) if !(j in st.whole) &&
                        (types[j] === Any || types[j] isa Type && types[j] <: AbstractVector)]
        all(j -> j in positions, sources) || return nothing
        # The window call must reach the same methods as the original.
        window(j) = j in positions ? Vector{<:eltype(types[j])} : types[j]
        for (G, args, method) in st.calls
            args === nothing && continue
            wtypes = Tuple(a.kind === :param ? window(a.data) : _pm_static_type(st, a)
                           for a in args)
            found = _pm_method(G, wtypes)
            found === nothing && return nothing
            first(found) === method || return nothing
        end
        return (; positions, sources, checks = sort!(collect(st.checks)))
    end
    nothing
end

function _cell_position_form(r::Recipe, input_types; outlined = false)
    call = _cell_helper_call(r, input_types; outlined)
    call === nothing && return nothing
    (; f, params, args) = call
    f isa Function && Base.issingletontype(typeof(f)) &&
        !(parentmodule(f) in (Core, Base)) || return nothing
    proof = _cell_position_proof(typeof(f), Any[a.type for a in args])
    proof === nothing && return nothing
    inputs = Int[findfirst(==(a.expr), params) for a in args]
    roles = fill(:whole, length(params))
    for j in unique(inputs)
        # One input passed as several arguments has one role.
        rows = [k in proof.positions for k in eachindex(inputs) if inputs[k] == j]
        all(rows) ? (roles[j] = :row) : any(rows) && return nothing
    end
    sources = unique!(Int[inputs[k] for k in proof.sources])
    checks = unique!(Tuple{Int,Symbol}[(inputs[k], fact) for (k, fact) in proof.checks])
    n = length(params)
    indices = [Int[] for _ in 1:n]
    _CellForm(:position_map, roles, zeros(Int, n), sources, 0, indices, Any[],
        _cell_direct_reads(indices), nothing, Any[nothing for _ in 1:n], nothing,
        _CellPositionMap(sources, checks))
end

# ------------------------------------------------------------------ runtime

# The row axis of a position map: its source's, or the least of several
# (`zip` stops at its shortest argument).
@inline _cell_position_shape(shape::Tuple) = (Base.OneTo(length(first(shape))),)
@inline _cell_position_shape(first_shape::Tuple, shapes::Tuple...) =
    (Base.OneTo(minimum(s -> length(first(s)), (first_shape, shapes...))),)

# A fresh dense vector of rows of a position argument, in the requested order.
function _cell_dense_rows(x::AbstractVector)
    rows = Vector{eltype(x)}(undef, length(x))
    @inbounds for k in eachindex(rows, x)
        rows[k] = x[k]
    end
    rows
end
_cell_dense_rows(x) = throw(ArgumentError(
    "a helper's per-row argument must be a vector at a cell's rows; got $(typeof(x))"))

@inline function _cell_check_position_type(::Type{T}, ::Val{:standard}) where {T}
    _pm_standard_vector(T) || throw(ArgumentError(
        "a helper iterates an argument of type $T at a cell's rows; its rows are proved " *
        "only for vectors Base iterates in linear order from position 1"))
    nothing
end
@inline function _cell_check_position_type(::Type{T}, ::Val{:eltype}) where {T}
    _pm_vector_eltype(T) || throw(ArgumentError(
        "a helper reads an element of an argument of type $T for its type at a cell's rows; " *
        "that requires a concrete element type"))
    nothing
end

# A pushed argument is never computed whole: its rows come from the same RK
# producer and element-type inference as its whole value would. Its element
# type is checked on those rows; RK producers build Base arrays.
@inline _cell_check_position_rows(rows::AbstractVector, ::Val{:eltype}) =
    _cell_check_position_type(typeof(rows), Val(:eltype))
@inline _cell_check_position_rows(rows, ::Val{:standard}) = nothing

@inline _cell_position_empty(::Type{T}) where {T<:AbstractVector} =
    isconcretetype(T) ? T(undef, 0) : Any[]
@inline _cell_position_empty(@nospecialize(T)) = Any[]

# The window of a position map at rows `U`: the helper over its position
# arguments' rows and its other arguments whole.
function _cell_position_window!(e, r::Recipe, f, args, U::Symbol)
    d = e.pushdowns
    for (j, fact) in f.rowmap.checks
        input = canon_id(d.graph, r.inputs[j].id)
        push!(e.body.args, input in d.pushed ?
            Expr(:call, _cell_ref(:_cell_check_position_rows), args[j], Val(fact)) :
            Expr(:call, _cell_ref(:_cell_check_position_type),
                 Expr(:call, GlobalRef(Core, :typeof), e.names[input]), Val(fact)))
    end
    call = _slot_call_expr(r, d.slots[r.id], args)
    T = _slot_type_expr(r, d.slots[r.id],
        Any[Expr(:call, GlobalRef(Core, :typeof), a) for a in args])
    _cell_bind!(e, :cell_rows, Expr(:if, Expr(:call, GlobalRef(Base, :isempty), U),
        Expr(:call, _cell_ref(:_cell_position_empty), T), call))
end
