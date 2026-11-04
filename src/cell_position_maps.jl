# Row forms of ordinary helpers that map positions to rows, recovered from the
# helper's lowered Julia code. No helper is executed to discover them.
#
# A helper qualifies when its result has one row per position of the vectors
# it iterates and row `g` reads each "position" argument only at `g`:
#
# - a comprehension or `map` over `eachindex(x)`, `axes(x, 1)`, `1:length(x)`,
#   `Base.OneTo(length(x))`, `x` itself, `enumerate(...)` or `zip(...)` of
#   those (destructured or not), including straight-line wrappers;
# - a loop over one such iterator that builds the result by `push!(out, row)`
#   onto an empty vector, or `out[g] = row` into `similar(x)`,
#   `zeros(T, length(x))` or `Vector{T}(undef, length(x))`, once on every path
#   through the body (each arm of a branch may write its own row). The body may
#   hold branches, inner loops and nested calls on per-position values; its
#   locals must be assigned in each iteration before they are read.
#
# A matrix argument is a position argument over its first axis: the loop runs
# over `axes(x, 1)` (or `1:size(x, 1)`) and row `g` reads `x[g, d]` for any
# `d`; `size(x, 2)` and `axes(x, 2)` are the same for its rows. A result
# filled before the loop, `zeros(T, size(x, 1), m...)` or `ones`, may also be
# read and written at its own row any number of times (`out[g, k] += ...`): a
# row no iteration reaches keeps the fill value, as in the whole helper. Each
# trailing extent `m` is a literal or the `size` of an argument, so the
# result's shape is known before the call. An unfilled result must be written
# once per row, as a whole row (`out[g, :] = row`).
#
# Before the loop, the prologue may compute values from the arguments passed
# whole, literals and those extents (`coefficients = L * reshape(z, k, :)`,
# `scale = sd / sqrt(size(frequency, 2))`). Such a value is the same for a
# window as for the whole helper; a row may read and index it. No prologue
# call may mutate (`!`), and none may read a position argument or a row count.
#
# Every other argument (`field` in `field[ix]` with `ix = paths[g]`) is passed
# whole, so it must be computed whole: the ordinary locality checks decide
# whether that is legitimate (a shared or bound value) or refused. A row may
# hand such an array to a closure it builds (`sum(w[t] * cs[ix[t]] for t in
# eachindex(ix))`) or to another helper (`mean_of(natural, ix, w, k)`) when that
# code only indexes it. Its work is then the row's own, however long the array
# is; any other use (`sum(natural)`) is cross-row work. The position `g` itself
# may only index a position argument; an expression of it
# (`x[g - 1]`), a read of another row of the result being built, a value
# carried from one iteration to the next, `length(x)` of a position argument
# read as a value, and mutation of anything not created in the iteration all
# keep the helper without a row form.
#
# A cell's window calls the authored helper on fresh dense vectors (matrices)
# of its position arguments' requested rows, and its other arguments
# unchanged. Both the original and the window argument types must dispatch to
# the same single method, so the authored loops, branches and errors run as
# written.

struct _CellPositionMap
    # The recipe inputs whose length gives the row count (`zip`: the least).
    sources::Vector{Int}
    # Facts the proof assumed of an input whose static type does not show
    # them: `(input, :standard)`, a vector iterated in its linear order from 1;
    # `(input, :standard_rows)`, a matrix whose first axis starts at 1;
    # `(input, :eltype)`, a concrete element type, so `eltype(x[1])` is the same
    # for every element. Checked on the original producer's type at run time.
    checks::Vector{Tuple{Int,Symbol}}
    # The inputs read at rows of a matrix (`x[g, d]`).
    matrices::Vector{Int}
    # The result's axes after its rows, for an array result built row by row:
    # `(:const, m)` or `(:size, input, axis)`. Empty for a vector result.
    extents::Vector{Tuple{Symbol,Vararg{Int}}}
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
# result being built; data `:push` or `(:prealloc, j, filled, extents)`),
# `:accread` (that result read at the current row), `:shared` (a value the
# prologue computed from arguments passed whole), `:extent` (data `(j, k)`:
# `size(x, k)` of an argument, the same for its rows when `k >= 2`), `:rows`
# (a result; data `(iterator, extents)`).
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
# A matrix whose first axis is `Base.OneTo`, as `axes(x, 1)` iterates its
# rows: an `Array`, a view of one-based indices, or a concrete matrix whose
# `axes` is Base's generic method. Offset axes are not.
function _pm_standard_matrix(T)
    T isa Type && T !== Union{} && T <: AbstractMatrix || return false
    T <: Array{<:Any,2} && return true
    if T isa DataType && T <: SubArray
        I = T.parameters[4]
        return I isa DataType && all(_pm_onebased_index, I.parameters)
    end
    isconcretetype(T) && !(T <: Base.ReinterpretArray) || return false
    m = which(axes, Tuple{T})
    m.module === Base && m.sig isa DataType && m.sig.parameters[2] in (AbstractArray, Any)
end
_pm_is_matrix(T) = T isa Type && T !== Union{} && T <: AbstractMatrix
_pm_matrix(st, j::Int) = _pm_is_matrix(st.types[j])

# A fact of argument `j`'s type: proven statically, checked at run time when
# its static type is abstract (the original producer's inferred type), or
# failed when its concrete static type lacks it.
function _pm_require!(st::_PMState, j::Int, fact::Symbol)
    T = st.types[j]
    holds = fact === :standard ? _pm_standard_vector(T) :
        fact === :standard_rows ? _pm_standard_matrix(T) : _pm_vector_eltype(T)
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
    T === Any || T isa Type && (T <: AbstractVector || _pm_is_matrix(T)) || _pm_whole!(st, j)
    nothing
end

# Argument `j` iterated by row: a vector by position, a matrix by its first axis.
function _pm_iterated!(st::_PMState, j::Int)
    _pm_position!(st, j)
    _pm_require!(st, j, _pm_matrix(st, j) ? :standard_rows : :standard)
    push!(st.iterated, j)
    nothing
end
# Argument `j` iterated by its linear positions or elements (`eachindex(x)`,
# `for v in x`): one row per position of a vector only.
function _pm_linear_iterated!(st::_PMState, j::Int)
    _pm_matrix(st, j) && _pm_fail()
    _pm_iterated!(st, j)
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
    # Whether some write sets part of a row (`out[g, k] = v`).
    partial::Bool
end
_PMFrame(ci, slots, ssa, iteration, sparams, accumulations, accumulator, statement) =
    _PMFrame(ci, slots, ssa, iteration, sparams, accumulations, accumulator, statement, false)

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
_pm_value_kind(v::_PMValue) = v.kind in (:const, :typeval, :local, :elem, :accread, :shared,
                                         :extent)
# The same for a window as for the whole helper.
_pm_invariant(v::_PMValue) = v.kind in (:const, :typeval, :shared, :extent)

# An iterator from a value: a descriptor, an argument iterated directly.
function _pm_iterator(st::_PMState, v::_PMValue)
    v.kind === :iter && return v.data
    if v.kind === :param
        _pm_linear_iterated!(st, v.data)
        return (:over, v.data)
    end
    _pm_fail()
end

# A value the prologue computes by calling `fv` on arguments passed whole,
# literals, types and extents: the same for a window as for the whole helper.
# A call that may mutate is not one, and a position argument, its row count
# or its type read this way is read whole.
function _pm_shared_call!(st::_PMState, fv, args::Vector{_PMValue})
    _pm_mutating(fv) && _pm_fail()
    foreach(a -> _pm_shared_operand!(st, a), args)
    # Its inferred type: a row may read a number, but only index an array.
    T = Core.Compiler.return_type(fv, Tuple{Any[_pm_shared_type(st, a) for a in args]...})
    _PMValue(:shared, T)
end
_pm_shared_type(st::_PMState, a::_PMValue) =
    a.kind === :const ? Core.Typeof(a.data) :
    a.kind === :param ? st.types[a.data] :
    a.kind === :shared ? a.data :
    a.kind in (:extent, :len) ? Int :
    a.kind === :tuple ? Tuple{Any[_pm_shared_type(st, x) for x in a.data]...} : Any
# A prologue value a row hands to a call: a number, or an integer range it
# iterates (`axes(x, 2)`); an array (`coefficients`) is only indexed.
_pm_shared_scalar(T) = T isa Type && T !== Union{} &&
    (T <: Number || T <: AbstractRange{<:Integer})
function _pm_shared_operand!(st::_PMState, a::_PMValue)
    k = a.kind
    _pm_invariant(a) && return
    k === :tuple && return foreach(x -> _pm_shared_operand!(st, x), a.data)
    k in (:param, :len, :typeof, :typeread) && return _pm_whole!(st, a.data)
    _pm_fail()
end

# `size(x, k)`: an extent the same for x's rows when `k >= 2` and x is an
# array; the row count when `k == 1`. An argument of unknown rank is whole.
function _pm_size!(st::_PMState, j::Int, k::Int)
    k >= 1 || _pm_fail()
    k == 1 && return _PMValue(:len, j)
    T = st.types[j]
    T isa Type && T !== Union{} && T <: AbstractArray || _pm_whole!(st, j)
    _PMValue(:extent, (j, k))
end

# A result `fill`ed before the loop, or not, with rows from argument `j` and
# trailing axes `rest`: each a literal, an extent or the row count of an
# argument passed whole.
function _pm_prealloc!(st::_PMState, j::Int, rest, filled::Bool)
    extents = Tuple{Symbol,Vararg{Int}}[]
    for a in rest
        if a.kind === :const && a.data isa Integer
            push!(extents, (:const, Int(a.data)))
        elseif a.kind === :extent
            push!(extents, (:size, a.data...))
        elseif a.kind === :len
            _pm_whole!(st, a.data)
            push!(extents, (:size, a.data, 1))
        else
            _pm_fail()
        end
    end
    _PMValue(:acc, (:prealloc, j, filled, Tuple(extents)))
end

# A value used as an argument of an ordinary call inside an iteration. An
# array argument a row hands whole to a call (a reduction, iteration, a whole
# broadcast) is cross-row work: no row form. A scalar argument is read whole;
# iteration machinery may not escape.
_pm_array_argument(st::_PMState, j::Int) =
    (T = st.types[j]; T === Any || T isa Type && T <: AbstractArray)
function _pm_operand!(st::_PMState, v::_PMValue)
    k = v.kind
    k in (:const, :typeval, :local, :elem, :accread, :extent) && return
    k === :shared && return _pm_shared_scalar(v.data) ? nothing : _pm_fail()
    k === :tuple && return foreach(x -> _pm_operand!(st, x), v.data)
    k === :param && _pm_array_argument(st, v.data) && _pm_fail()
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
        n >= 2 && args[1].kind === :const && args[1].data === undef &&
            args[2].kind === :len &&
            return _pm_prealloc!(st, args[2].data, args[3:end], false)
        _pm_fail()
    end
    f.kind === :const || _pm_fail()
    fv = f.data
    # Extents: `size(x, k)` and `axes(x, k)` of an argument; past its first
    # axis, the same for its rows.
    if fv in (Base.size, Base.axes) && n == 2 && args[1].kind === :param &&
            args[2].kind === :const && args[2].data isa Integer && args[2].data >= 2
        extent = _pm_size!(st, args[1].data, Int(args[2].data))
        return fv === Base.size ? extent : _PMValue(:shared, Base.OneTo{Int})
    end
    if fv === Base.size && n == 2 && args[1].kind === :param &&
            args[2].kind === :const && args[2].data == 1
        return _pm_size!(st, args[1].data, 1)
    end
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
        _pm_invariant(a) && return _PMValue(:typeval)
    end
    if fv === Core.apply_type && n >= 1 && args[1].kind === :const &&
            args[1].data isa Type && args[1].data <: Function
        # A closure's type over its captures' types: the method is the same
        # for every parameter. In a row, a capture may be a value of that row.
        all(a -> _pm_typelike(a) || a.kind === :typeof || fr.iteration && a.kind === :local,
            args[2:end]) || _pm_fail()
        return _PMValue(:closuretype, args[1].data)
    end
    if fv in _PM_TYPE_FUNCTIONS
        for a in args
            a.kind === :typeof && _pm_whole!(st, a.data)
        end
        # The type of an argument passed whole is the original's.
        all(a -> _pm_typelike(a) || a.kind === :typeof, args) && return _PMValue(:typeval)
    end
    fr.iteration ? _pm_iteration_call(st, fr, fv, args) : _pm_outer_call(st, fr, fv, args)
end

# Outside an iteration: the iterator, the closure, the result's construction
# and wrappers. Anything else has no proven row form.
function _pm_outer_call(st::_PMState, fr::_PMFrame, fv, args::Vector{_PMValue})
    n = length(args)
    param(k) = n >= k && args[k].kind === :param
    if fv === Base.eachindex && n == 1 && param(1)
        _pm_linear_iterated!(st, args[1].data)
        return _PMValue(:iter, (:positions, args[1].data))
    elseif fv === Base.axes && n == 2 && param(1) && args[2].kind === :const && args[2].data == 1
        _pm_iterated!(st, args[1].data)
        return _PMValue(:iter, (:positions, args[1].data))
    elseif fv === Base.length && n == 1 && param(1)
        # A matrix's length is not its row count.
        _pm_matrix(st, args[1].data) && return _pm_shared_call!(st, fv, args)
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
    elseif fv isa Type && n >= 2 && args[1].kind === :const && args[1].data === undef &&
            args[2].kind === :len
        # `Vector{T}(undef, length(x))`, `Matrix{T}(undef, size(x, 1), m)`
        return _pm_prealloc!(st, args[2].data, args[3:end], false)
    elseif fv === Base.similar && n in (1, 2) && param(1) && (n == 1 || _pm_typelike(args[2]))
        j = args[1].data
        _pm_position!(st, j)
        return _pm_prealloc!(st, j, _pm_matrix(st, j) ? [_PMValue(:extent, (j, 2))] : (), false)
    elseif fv === Base.similar && n in (2, 3) && args[end].kind === :len &&
            (n == 2 || _pm_typelike(args[2]))
        args[1].kind === :param && _pm_whole!(st, args[1].data)
        args[1].kind in (:param, :const, :typeval) || _pm_fail()
        return _pm_prealloc!(st, args[end].data, (), false)
    elseif fv in (Base.zeros, Base.ones) && n >= 1
        # `zeros(T, size(x, 1), m...)`: every entry holds its fill value.
        dims = _pm_typelike(args[1]) && !(args[1].kind === :const && args[1].data isa Integer) ?
            args[2:end] : args
        !isempty(dims) && dims[1].kind === :len || _pm_fail()
        return _pm_prealloc!(st, dims[1].data, dims[2:end], true)
    elseif fv === Base.fill && n >= 2 && _pm_typelike(args[1]) && args[2].kind === :len
        # A number fills; another value is one object every row would share.
        filled = args[1].kind === :const && args[1].data isa Number
        return _pm_prealloc!(st, args[2].data, args[3:end], filled)
    end
    # A wrapper: another helper over arguments, constants and types. One
    # that maps no positions is a value of the prologue.
    if fv isa Function && Base.issingletontype(typeof(fv)) &&
            !(parentmodule(fv) in (Core, Base)) &&
            all(a -> a.kind in (:param, :const, :typeval), args) &&
            any(a -> a.kind === :param && !(a.data in st.whole), args)
        calls, checks, iterated = copy(st.calls), copy(st.checks), copy(st.iterated)
        try
            return _pm_function(st, typeof(fv), args)
        catch err
            err isa _PMFail || rethrow()
            st.calls, st.checks, st.iterated = calls, checks, iterated
        end
    end
    fv isa Function || fv isa Type || _pm_fail()
    _pm_shared_call!(st, fv, args)
end

# Inside one row's computation.
function _pm_iteration_call(st::_PMState, fr::_PMFrame, fv, args::Vector{_PMValue})
    n = length(args)
    # The result being built, at its own row and one index per trailing axis.
    acc = fr.accumulator
    at_row(first) = n >= first && args[first].kind === :pos && args[1].kind === :acc &&
        args[1] == acc && acc.data isa Tuple && n - first == length(acc.data[4]) &&
        !any(a -> a.kind === :pos, args[first+1:end])
    if fv === Base.getindex && n >= 2 && args[2].kind === :pos
        a = args[1]
        if a.kind === :acc
            # Only a filled result holds a value before the row writes it.
            at_row(2) && acc.data[3] || _pm_fail()
            foreach(x -> _pm_operand!(st, x), args[3:end])
            return _PMValue(:accread)
        end
        a.kind === :param || _pm_fail()
        _pm_position!(st, a.data)
        any(x -> x.kind === :pos, args[3:end]) && _pm_fail()
        # A vector at the position; a matrix at its row (`x[g, d]`), never by
        # the linear index `x[g]`.
        n == (_pm_matrix(st, a.data) ? 3 : 2) || _pm_fail()
        foreach(x -> _pm_operand!(st, x), args[3:end])
        return _PMValue(:elem, a.data)
    end
    if fv === Base.setindex! && n >= 3 && args[3].kind === :pos
        at_row(3) || _pm_fail()
        foreach(x -> _pm_operand!(st, x), args[4:end])
        _pm_operand!(st, args[2])
        push!(fr.accumulations, fr.statement)
        # A whole row: one index (a vector) or `out[g, :]`.
        all(x -> x.kind === :const && x.data isa Colon, args[4:end]) || (fr.partial = true)
        return _PMValue(:local)
    end
    any(a -> a.kind === :pos, args) && _pm_fail()
    if fv === Base.getindex && n >= 2 && args[1].kind === :param
        # A lookup into an argument passed whole (`field[ix]`, `ix = paths[g]`).
        _pm_whole!(st, args[1].data)
        foreach(a -> _pm_operand!(st, a), args[2:end])
        return _PMValue(:local)
    end
    if fv === Base.getindex && n >= 2 && args[1].kind === :shared
        # A lookup into a prologue value (`coefficients[k, offset + j]`).
        foreach(a -> _pm_operand!(st, a), args[2:end])
        return _PMValue(:local)
    end
    if any(a -> a.kind === :acc, args)
        acc === nothing && _pm_fail()
        fv === Base.push! && n == 2 && args[1] == acc && acc.data === :push || _pm_fail()
        _pm_operand!(st, args[2])
        push!(fr.accumulations, fr.statement)
        return _PMValue(:local)
    end
    _pm_mutating(fv) && n >= 1 && args[1].kind !== :local && _pm_fail()
    # A helper handed a whole array: its own body must only index it.
    if fv isa Function && Base.issingletontype(typeof(fv)) &&
            !(parentmodule(fv) in (Core, Base)) && any(a -> _pm_whole_array(st, a), args)
        return _pm_row_function!(st, typeof(fv), args)
    end
    foreach(a -> _pm_operand!(st, a), args)
    _PMValue(:local)
end

# A value only an index read may use inside a row: an array argument passed
# whole, or an array the prologue computed.
_pm_whole_array(st::_PMState, a::_PMValue) =
    a.kind === :param && _pm_array_argument(st, a.data) ||
    a.kind === :shared && !_pm_shared_scalar(a.data)

# One row's computation in a function body: a comprehension element, a closure
# the row builds, a helper the row calls. Each call has fresh locals; their
# values are joined over two passes so a branch or an inner loop sees every
# assignment.
function _pm_row_body!(st::_PMState, fr::_PMFrame)
    code = fr.ci.code
    for _ in 1:2, i in eachindex(code)
        s = code[i]
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
    nothing
end

# A closure a row builds over a whole array: however it is called, its element
# is a value of this row, so its body is that row's work.
function _pm_row_closure!(st::_PMState, T, fields::Vector{_PMValue})
    found = _pm_method(T, (Any,))
    found === nothing && _pm_fail()
    method, ci = found
    push!(st.calls, (T, nothing, method))
    closure = _PMValue(:closure, (T, fields))
    fr = _PMFrame(ci, Dict(1 => closure, 2 => _PMValue(:local)),
                  Dict{Int,_PMValue}(), true, false, Int[], nothing, 0)
    _pm_row_body!(st, fr)
    _PMValue(:local)
end

# A helper a row calls with a whole array: its body runs under the row's rules.
function _pm_row_function!(st::_PMState, F, args::Vector{_PMValue})
    key = (F, Tuple(args))
    key in st.stack && _pm_fail()
    found = _pm_method(F, Tuple(_pm_static_type(st, a) for a in args))
    found === nothing && _pm_fail()
    method, ci = found
    push!(st.calls, (F, args, method))
    push!(st.stack, key)
    try
        slots = Dict{Int,_PMValue}(k + 1 => a for (k, a) in enumerate(args))
        slots[1] = _PMValue(:const, F.instance)
        _pm_row_body!(st, _PMFrame(ci, slots, Dict{Int,_PMValue}(), true, false, Int[], nothing, 0))
    finally
        delete!(st.stack, key)
    end
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
            T.kind === :closuretype && any(a -> _pm_whole_array(st, a), fields) &&
                return _pm_row_closure!(st, T.data, fields)
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
    # A number read whole is one of the row's values, in any branch.
    if assigned && fr.iteration && v.kind === :param && !_pm_array_argument(st, v.data)
        _pm_whole!(st, v.data)
        v = _PMValue(:local)
    end
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
    _pm_row_body!(st, _PMFrame(ci, Dict(1 => closure, 2 => _pm_element(iterator)),
                               Dict{Int,_PMValue}(), true, false, Int[], nothing, 0))
    _PMValue(:rows, (iterator, ()))
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
    # The loop's top may declare the body's locals (`NewvarNode`) before the
    # head reads `I`; the back edge returns to the top.
    top = setup + 4
    head = top
    while head < exit - 1 && _pm_noop(code[head])
        head += 1
    end
    head < exit - 1 && _pm_slot(code[head]) == I || _pm_fail()
    back = exit - 1
    code[back] isa Core.GotoNode && code[back].label == top || _pm_fail()
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
    # A filled result may be read and written at its row any number of times;
    # a row it never writes keeps the fill value. Otherwise every path through
    # the body writes one whole row exactly once.
    if !(prealloc && accumulator.data[3])
        fr.partial && _pm_fail()
        _pm_writes_once(code, head + 1, step, fr.accumulations) || _pm_fail()
    end
    _pm_assigned_before_read(code, head + 1, step, assigned) || _pm_fail()
    _PMValue(:rows, (desc, prealloc ? accumulator.data[4] : ()))
end

# Whether every path from `first` to `last` passes exactly one of `writes`. A
# write in an inner loop reaches `last` with a count above one.
function _pm_writes_once(code, first::Int, last::Int, writes::Vector{Int})
    counts = Dict{Int,Set{Int}}(first => Set(0))
    work = [first]
    while !isempty(work)
        i = pop!(work)
        i == last && continue
        s = code[i]
        out = Set(min(c + (i in writes), 2) for c in counts[i])
        for t in (s isa Core.GotoNode ? (s.label,) :
                  s isa Core.GotoIfNot ? (i + 1, s.dest) : (i + 1,))
            first <= t <= last || return false
            seen = get!(Set{Int}, counts, t)
            issubset(out, seen) && continue
            union!(seen, out)
            push!(work, t)
        end
    end
    get(counts, last, Set{Int}()) == Set(1)
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
        desc, extents = result.data
        sources = _pm_sources(desc)
        isempty(sources) && return nothing
        positions = Int[j for j in eachindex(types) if !(j in st.whole) &&
                        (types[j] === Any || types[j] isa Type &&
                         (types[j] <: AbstractVector || _pm_is_matrix(types[j])))]
        all(j -> j in positions, sources) || return nothing
        matrices = Int[j for j in positions if _pm_is_matrix(types[j])]
        # The window call must reach the same methods as the original.
        window(j) = !(j in positions) ? types[j] :
            j in matrices ? Matrix{<:eltype(types[j])} : Vector{<:eltype(types[j])}
        for (G, args, method) in st.calls
            args === nothing && continue
            wtypes = Tuple(a.kind === :param ? window(a.data) : _pm_static_type(st, a)
                           for a in args)
            found = _pm_method(G, wtypes)
            found === nothing && return nothing
            first(found) === method || return nothing
        end
        return (; positions, sources, checks = sort!(collect(st.checks)), matrices,
                extents = collect(Tuple{Symbol,Vararg{Int}}, extents))
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
    matrices = unique!(Int[inputs[k] for k in proof.matrices])
    extents = Tuple{Symbol,Vararg{Int}}[e[1] === :size ? (:size, inputs[e[2]], e[3]) : e
                                        for e in proof.extents]
    n = length(params)
    indices = [Int[] for _ in 1:n]
    _CellForm(:position_map, roles, zeros(Int, n), sources, 0, indices, Any[],
        _cell_direct_reads(indices), nothing, Any[nothing for _ in 1:n], nothing,
        _CellPositionMap(sources, checks, matrices, extents))
end

# ------------------------------------------------------------------ runtime

# The row axis of a position map: its source's, or the least of several
# (`zip` stops at its shortest argument).
@inline _cell_position_shape(shape::Tuple) = (Base.OneTo(length(first(shape))),)
@inline _cell_position_shape(first_shape::Tuple, shapes::Tuple...) =
    (Base.OneTo(minimum(s -> length(first(s)), (first_shape, shapes...))),)
# An array result's full axes: its rows, then its trailing extents.
@inline _cell_position_extents(shape::Tuple, extents::Tuple) =
    (first(shape), map(Base.OneTo, extents)...)

# A fresh dense vector of rows of a position argument, in the requested order.
function _cell_dense_rows(x::AbstractVector)
    rows = Vector{eltype(x)}(undef, length(x))
    @inbounds for k in eachindex(rows, x)
        rows[k] = x[k]
    end
    rows
end
# A fresh dense matrix of rows of a matrix position argument.
function _cell_dense_rows(x::AbstractMatrix)
    rows = Matrix{eltype(x)}(undef, size(x, 1), size(x, 2))
    @inbounds for (j, jx) in enumerate(axes(x, 2)), (i, ix) in enumerate(axes(x, 1))
        rows[i, j] = x[ix, jx]
    end
    rows
end
_cell_dense_rows(x) = throw(ArgumentError(
    "a helper's per-row argument must be a vector or a matrix at a cell's rows; " *
    "got $(typeof(x))"))

@inline function _cell_check_position_type(::Type{T}, ::Val{:standard}) where {T}
    _pm_standard_vector(T) || throw(ArgumentError(
        "a helper iterates an argument of type $T at a cell's rows; its rows are proved " *
        "only for vectors Base iterates in linear order from position 1"))
    nothing
end
@inline function _cell_check_position_type(::Type{T}, ::Val{:standard_rows}) where {T}
    _pm_standard_matrix(T) || throw(ArgumentError(
        "a helper iterates the rows of an argument of type $T at a cell's rows; its rows " *
        "are proved only for matrices whose first axis starts at 1"))
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
@inline _cell_check_position_rows(rows, ::Val{:standard_rows}) = nothing

@inline _cell_position_empty(::Type{T}) where {T<:AbstractVector} =
    isconcretetype(T) ? T(undef, 0) : Any[]
@inline _cell_position_empty(@nospecialize(T)) = Any[]
# No rows of an array result keep its trailing axes.
@inline _cell_position_empty(::Type{T}, shape::Tuple) where {T<:AbstractArray} =
    isconcretetype(T) ? T(undef, 0, map(length, Base.tail(shape))...) :
    Array{Any}(undef, 0, map(length, Base.tail(shape))...)
@inline _cell_position_empty(@nospecialize(T), shape::Tuple) =
    Array{Any}(undef, 0, map(length, Base.tail(shape))...)

# The window of a position map at rows `U`: the helper over its position
# arguments' rows and its other arguments whole.
function _cell_position_window!(e, cid, r::Recipe, f, args, U::Symbol)
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
    empty = isempty(f.rowmap.extents) ?
        Expr(:call, _cell_ref(:_cell_position_empty), T) :
        Expr(:call, _cell_ref(:_cell_position_empty), T, e.shapes[cid])
    _cell_bind!(e, :cell_rows, Expr(:if, Expr(:call, GlobalRef(Base, :isempty), U),
        empty, call))
end

# The shape expression of a position map: the row axis of its sources
# (`parts`), then, for an array result, each trailing extent from a literal or
# an input's shape (`input_shape`, by input id).
function _cell_position_shape_expr!(e, r::Recipe, f, parts, input_shape)
    rows = Expr(:call, _cell_ref(:_cell_position_shape), parts...)
    isempty(f.rowmap.extents) && return rows
    d = e.pushdowns
    extents = map(f.rowmap.extents) do x
        x[1] === :const && return x[2]
        shape = input_shape(canon_id(d.graph, r.inputs[x[2]].id))
        Expr(:call, GlobalRef(Base, :length), Expr(:call, GlobalRef(Base, :getindex), shape, x[3]))
    end
    Expr(:call, _cell_ref(:_cell_position_extents), rows, Expr(:tuple, extents...))
end

# `vec(x)` of an array: position `u` of the result is a position of one row
# of `x` (`_cell_vec_rows`), so the result has a row form whenever `x`'s rows
# are known, and reads only the rows of `x` the requested positions lie in.
function _cell_vec_form(r::Recipe, input_types; outlined = false)
    call = _cell_helper_call(r, input_types; outlined, stable = true)
    call === nothing && return nothing
    call.f === Base.vec && length(call.args) == 1 || return nothing
    T = only(call.args).type
    T isa Type && T !== Union{} && T <: AbstractArray || return nothing
    j = findfirst(==(only(call.args).expr), call.params)
    roles = fill(:whole, length(call.params))
    roles[j] = :vec_source
    _CellForm(:vec, roles, zeros(Int, length(roles)), [j], 0)
end
