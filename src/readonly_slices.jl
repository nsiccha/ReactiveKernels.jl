# A slice that a plate cell passes only to calls that read it.
#
#     @plate for s in eachindex(block)
#         value[s] = field_state(frequency, lengths, coefficients[block[s], :], coordinates[s])
#     end
#
# `coefficients[block[s], :]` copies the row at every cell, and native Enzyme
# reverse allocates the copy's shadow and adds its adjoint back into the
# parent's. A cell that only reads the row can read it in place. Whether it only
# reads it is decided per callee, at preparation, from concrete types: Julia's
# own effects cannot say so (a callee that writes a local buffer of its own reads
# `!e`), and reflection is refused inside the generated functions that would
# otherwise see the types at compile time. So the proof needs the exemplar
# evidence `prepare(...; exemplars)` supplies; without it the program copies.
#
# The cell reads a view (`_readonly_slice`) when
#
# - the recipe's whole source is `A[i...]` with some non-scalar index, over a
#   capture-free source (`_readonly_slice_form`), of bits elements;
# - `A` is not a bound constant: `bound=` holds the caller's data, and a view
#   of it would hold the caller's array across the differentiated call;
# - the slice is not the cell's result;
# - every reader is an authored or a bare call, which over the view and over the copy
#   infers one concrete result type, which is proved to only read the view
#   (`_reads_only`), and which cannot tell the view from the copy
#   (`_reads_through_interface`); and
# - the view's type is concrete and is not the copy's.
#
# Anything else keeps the copy, so a refusal costs what the program costs today.
# `JULIA_DEBUG=ReactiveKernels` logs why a slice keeps its copy (`_readonly_keeps_copy`).
# A position residual keeps its lane slots (`_lane_source_expr`), and a scan
# step and a plate nested in a cell keep their own lowering.
# A slice nested in a call is one recipe with that call and not a slice recipe;
# RKPPL's cells split it (`__rk_split`), and a `@kernel` plate's do not.
#
# A view is not a `Vector`, so a callee could tell them apart. The optimized code
# the proof reads has already selected the view's branch and method, so a second
# gate reads the lowered code of every user-level frame the slice reaches
# (`_reads_through_interface`): the frame must touch it only through the array
# interface, and a function with several methods must not choose among them by
# the container. So `x isa Vector`, `typeof(x)`, `parent(x)` and a `Vector` method
# beside an `AbstractVector` one all keep the copy.
#
# `RK_READONLY_SLICES=0` keeps every copy (the comparison baseline).
_readonly_slices_enabled() = get(ENV, "RK_READONLY_SLICES", "1") != "0"

# The slice as the cell reads it. Base's `view` throws the `BoundsError` the
# slice throws and has its axes, element type and values. A gather of a vector
# by a `Vector{Int}` takes `_readonly_gather`, which does not materialize
# `range[indices]` to flatten a pair of views.
@inline _readonly_slice(A, I...) = view(A, I...)
@inline _readonly_slice(A::AbstractVector, I::Vector{Int}) = _readonly_gather(A, I)
# A cell window (`_CellWindow`) resolves its row as its own `getindex` does, then
# views its data. That view's parent need not be the type the proof ran over, and
# the proof still holds: what a callee writes or returns does not depend on the
# parent, since the lowered gate refuses any code that tells containers apart. Any
# other read of a window, a gather of rows included, keeps its `getindex`.
@inline _readonly_slice(w::_CellWindow, I...) = w[I...]
@inline _readonly_slice(w::_CellWindow, i::Integer, j::Union{Integer,Colon},
                        rest::Union{Integer,Colon}...) =
    _readonly_slice(w.data, _cell_local_row(w, i, (i, j, rest...)), j, rest...)

# The proof. It walks optimized typed IR from an argument, tracking every value
# derived from it until the value holds no reference. Only reads are known;
# anything else that meets a derived value refuses, so an IR form this walk does
# not know costs the copy and never a wrong value.
#
# A frame's verdict on its argument is `:reads` (it only reads it, and returns
# nothing derived from it), `:returns` (it only reads it, but may return a value
# derived from it) or `:refuse` (no proof). A caller counts a call's result as
# derived only when the callee may return one, so a callee that computes a fresh
# vector from the slice's values and returns it (a vector-valued cell) only reads
# the slice.
#
# The memo holds each frame's verdict, and every frame that refused with the
# statement it refused at, innermost first, for the preparation log
# (`_readonly_keeps_copy`).
struct _ReadOnlyMemo
    verdicts::Dict{Tuple{Any,Int},Symbol}
    refusals::Vector{Any}
end
_ReadOnlyMemo() = _ReadOnlyMemo(Dict{Tuple{Any,Int},Symbol}(), Any[])
_readonly_refuse!(memo::_ReadOnlyMemo, @nospecialize(sig), position::Int, @nospecialize(at)) =
    (push!(memo.refusals, (; sig, position, at)); :refuse)

# Bits values with no pointer: a value of this type cannot reach the argument.
_readonly_holds_no_reference(@nospecialize(T)) =
    isbitstype(T) && !_readonly_holds_pointer(T)
_readonly_holds_pointer(@nospecialize(T)) =
    T <: Ptr || T <: Core.LLVMPtr || any(_readonly_holds_pointer, fieldtypes(T))

# The builtins that read a derived value. The writing ones (`arrayset`,
# `setfield!`, `memoryrefset!`, ...) are not here, nor is anything that turns a
# pointer into a number.
const _READONLY_BUILTINS = let names = (:arrayref, :arraysize, :arraylen, :memoryrefget,
        :memoryrefnew, :memoryrefoffset, :memoryref_isassigned)
    found = Any[Core.getfield, Core.:(===), Core.isa, Core.typeof, Core.sizeof,
                Core.ifelse, Core.tuple]
    for name in names, m in (Core, Base, Core.Intrinsics)
        isdefined(m, name) && push!(found, getglobal(m, name))
    end
    Tuple(unique(found))
end

_readonly_callee(f::GlobalRef) = isdefined(f.mod, f.name) ? getglobal(f.mod, f.name) : nothing
_readonly_callee(f::QuoteNode) = f.value
_readonly_callee(f) = f

_readonly_method_instance(mi::Core.MethodInstance) = mi
_readonly_method_instance(ci) = ci.def  # A `CodeInstance`, from Julia 1.12.

"""
    _reads_only(sig, position) -> Bool

Whether the call of signature `sig` only reads its argument at `position` (the
index in `sig`, the function being 1): nothing it reaches writes that argument
or anything derived from it, stores it, or passes it to code the walk cannot
read, and the call does not return it. `true` is a proof; `false` means no proof.
"""
_reads_only(@nospecialize(sig::Type{<:Tuple}), position::Int) =
    _reads_only!(_ReadOnlyMemo(), sig, position) === :reads

# The verdict of the call of signature `sig` on its argument at `position`.
function _reads_only!(memo::_ReadOnlyMemo, @nospecialize(sig), position::Int)
    key = (sig, position)
    verdicts = memo.verdicts
    haskey(verdicts, key) && return verdicts[key]
    # A recursive call adds no sink beyond the ones its frames show, but its
    # result may be derived until its frame has been read.
    verdicts[key] = :returns
    verdicts[key] = _reads_only_frame!(memo, sig, position)
end

function _reads_only_frame!(memo::_ReadOnlyMemo, @nospecialize(sig), position::Int)
    typed = Base.code_typed_by_type(sig; optimize = true)
    length(typed) == 1 || return _readonly_refuse!(memo, sig, position, "no unique typed body")
    code = first(only(typed))
    code isa Core.CodeInfo || return _readonly_refuse!(memo, sig, position, "no typed body")
    method = try
        Base._which(sig).method
    catch
        return _readonly_refuse!(memo, sig, position, "no unique method")
    end
    derived = _readonly_argument_roots(code, method, position)
    changed = true
    while changed
        changed = false
        for (i, statement) in enumerate(code.code)
            verdict = _reads_only_statement!(memo, code, i, statement, derived)
            verdict === :refuse && return _readonly_refuse!(memo, sig, position,
                (; statement = Core.SSAValue(i), code = statement))
            if verdict === :derived && !(Core.SSAValue(i) in derived)
                push!(derived, Core.SSAValue(i))
                changed = true
            end
        end
    end
    returns = any(s -> s isa Core.ReturnNode && isdefined(s, :val) && s.val in derived, code.code)
    return returns ? :returns : :reads
end

# The values the walk starts from: the argument at `position`. A vararg method
# holds its trailing arguments in one tuple slot. When the body reads that tuple
# only by constant `getfield`, as a forwarding shim does (`_KernelSource`, the
# callable of every authored cell call), the walk starts from the reads of the one
# element at `position`, and the tuple's other elements are the frame's other
# arguments, as a fixed-arity method's are. Tracking the whole tuple instead made
# every array argument of the call count as the slice, so a callee that passed
# another array to BLAS (`correlation * raw`) kept the slice's copy. Any other
# use of the tuple (a splat, a dynamic index, the tuple itself) tracks it whole.
function _readonly_argument_roots(code::Core.CodeInfo, method::Method, position::Int)
    nargs = Int(method.nargs)
    method.isva && position >= nargs || return Set{Any}((Core.Argument(position),))
    slot = Core.Argument(nargs)
    element = position - nargs + 1
    roots = Set{Any}()
    for (i, statement) in enumerate(code.code)
        _readonly_mentions(statement, slot) || continue
        k = _readonly_tuple_element(statement, slot)
        k === nothing && return Set{Any}((slot,))
        k == element && push!(roots, Core.SSAValue(i))
    end
    roots
end

# The constant index of a statement `getfield(slot, k)` (with or without its
# bounds-check flag), else `nothing`.
function _readonly_tuple_element(@nospecialize(statement), slot::Core.Argument)
    statement isa Expr && statement.head === :call || return nothing
    args = statement.args
    length(args) in (3, 4) && args[2] === slot && args[3] isa Int &&
        _readonly_callee(args[1]) === Core.getfield || return nothing
    length(args) == 3 || args[4] isa Bool || return nothing
    args[3]::Int
end

# Whether an optimized IR statement uses `slot` anywhere.
_readonly_mentions(@nospecialize(x), slot::Core.Argument) = x === slot
_readonly_mentions(x::Expr, slot::Core.Argument) = any(a -> _readonly_mentions(a, slot), x.args)
_readonly_mentions(x::Core.ReturnNode, slot::Core.Argument) =
    isdefined(x, :val) && _readonly_mentions(x.val, slot)
_readonly_mentions(x::Core.PiNode, slot::Core.Argument) = _readonly_mentions(x.val, slot)
_readonly_mentions(x::Core.GotoIfNot, slot::Core.Argument) = _readonly_mentions(x.cond, slot)
_readonly_mentions(x::Core.UpsilonNode, slot::Core.Argument) =
    isdefined(x, :val) && _readonly_mentions(x.val, slot)
_readonly_mentions(x::Union{Core.PhiNode,Core.PhiCNode}, slot::Core.Argument) =
    any(k -> isassigned(x.values, k) && _readonly_mentions(x.values[k], slot), eachindex(x.values))

function _reads_only_statement!(memo, code::Core.CodeInfo, i::Int, statement, derived)
    is_derived(x) = x in derived
    result(any_derived) = any_derived && !_readonly_holds_no_reference(
        Core.Compiler.widenconst(code.ssavaluetypes[i])) ? :derived : :clean
    if statement isa Core.ReturnNode
        # What a frame returns is its verdict (`_reads_only_frame!`).
        return :clean
    elseif statement isa Core.PhiNode
        values = statement.values
        return result(any(k -> isassigned(values, k) && is_derived(values[k]), eachindex(values)))
    elseif statement isa Core.PiNode
        return result(is_derived(statement.val))
    elseif statement isa Core.SSAValue || statement isa Core.Argument
        return result(is_derived(statement))
    elseif statement isa Core.UpsilonNode || statement isa Core.PhiCNode
        # Values crossing a `try`: a caught exception can hold a derived value
        # the walk no longer sees.
        return :refuse
    elseif isdefined(Core, :EnterNode) && statement isa getglobal(Core, :EnterNode)
        return :refuse
    elseif statement isa Expr
        args = statement.args
        if statement.head === :invoke
            mi = _readonly_method_instance(args[1])
            uses = [k for k in 2:length(args) if is_derived(args[k])]
            isempty(uses) && return :clean
            verdicts = [_reads_only!(memo, mi.specTypes, k - 1) for k in uses]
            any(==(:refuse), verdicts) && return :refuse
            return result(any(==(:returns), verdicts))
        elseif statement.head === :call
            any(is_derived, @view args[2:end]) || return :clean
            f = _readonly_callee(args[1])
            # A thrown value leaves every frame of this walk: none holds a `try`.
            f === Core.throw && return :clean
            any(g -> f === g, _READONLY_BUILTINS) || return :refuse
            return result(true)
        elseif statement.head === :new
            return result(any(is_derived, @view args[2:end]))
        elseif statement.head === :enter
            return :refuse
        elseif statement.head in (:boundscheck, :meta, :inbounds, :loopinfo, :aliasscope,
                :popaliasscope, :gc_preserve_begin, :gc_preserve_end, :code_coverage_effect,
                :isdefined, :throw_undef_if_not, :leave, :pop_exception, :coverageeffect)
            return :clean
        end
        return any(is_derived, args) ? :refuse : :clean
    end
    return :clean
end

# The array interface: what these return for a view and for its copy is one
# value, bit for bit, whichever method computes it. Nothing here hands the caller
# the container's storage, type or parent. `sum` and `prod` are not in it: Base
# may fold a view in another order than an array (a slice it cannot index
# linearly, and on Julia 1.13 a row too), so their last bits can differ.
const _READONLY_INTERFACE = (Base.getindex, Base.size, Base.length, Base.axes, Base.eachindex,
    Base.firstindex, Base.lastindex, Base.iterate, Base.first, Base.last, Base.eltype,
    Base.ndims, Base.isempty, Base.maximum, Base.minimum, Base.copy, Base.collect)
_readonly_interface(@nospecialize(f)) = any(g -> f === g, _READONLY_INTERFACE)

# A frame's callee: a global, directly or through the statement that reads it.
function _readonly_lowered_callee(code::Core.CodeInfo, @nospecialize(f))
    f isa Core.SSAValue && (f = code.code[f.id])
    f isa GlobalRef ? _readonly_callee(f) : f isa QuoteNode ? f.value :
        f isa Function || f isa Type ? f : nothing
end

"""
    _reads_through_interface(f, position, nargs, viewed, copied, seen) -> Bool

Whether a call of `f` with `nargs` arguments cannot tell apart `viewed` and
`copied` at `position`: Base's array interface, or a user function whose methods
admit both alike and whose lowered bodies use the argument only through the
interface. Anything else, Base's or Core's other functions included, is `false`.
"""
function _reads_through_interface(@nospecialize(f), position::Int, nargs::Int,
        @nospecialize(viewed), @nospecialize(copied), seen::Set{Any})
    _readonly_interface(f) && return true
    f === nothing && return false
    f isa Function || return false
    owner = parentmodule(f)
    (owner === Base || owner === Core || parentmodule(owner) === Base) && return false
    F = typeof(f)
    at(T) = Tuple{F, ntuple(k -> k == position ? T : Any, nargs)...}
    found = false
    for m in methods(f)
        admits_view = typeintersect(m.sig, at(viewed)) !== Union{}
        admits_copy = typeintersect(m.sig, at(copied)) !== Union{}
        admits_view == admits_copy || return false
        admits_view || continue
        found = true
        m.isva && return false
        key = (m, position)
        key in seen && continue
        push!(seen, key)
        Base.hasgenerator(m) && return false
        isdefined(m, :source) && m.source !== nothing || return false
        _lowered_reads_through_interface(Base.uncompressed_ast(m), position + 1,
                                         viewed, copied, seen) || return false
    end
    found
end

# Whether a lowered body uses its argument in `slot` only through the interface.
function _lowered_reads_through_interface(code::Core.CodeInfo, slot::Int,
        @nospecialize(viewed), @nospecialize(copied), seen::Set{Any})
    derived = Set{Any}((Core.SlotNumber(slot),))
    is_derived(x) = x in derived
    mentions(ex) = is_derived(ex) || ex isa Expr && any(mentions, ex.args)
    function read_ok(call::Expr)
        args = call.args
        callee = _readonly_lowered_callee(code, args[1])
        for k in 2:length(args)
            is_derived(args[k]) || continue
            _reads_through_interface(callee, k - 1, length(args) - 1, viewed, copied, seen) ||
                return false
        end
        !any(k -> args[k] isa Expr && mentions(args[k]), 2:length(args))
    end
    # Aliases are found in statement order, which a lowered body assigns in;
    # a second pass covers an alias read before its assignment in a loop.
    for _ in 1:2
        for (i, statement) in enumerate(code.code)
            ssa = Core.SSAValue(i)
            if statement isa Expr && statement.head === :(=)
                target, value = statement.args
                if is_derived(value)
                    push!(derived, target)
                elseif value isa Expr && value.head === :call
                    any(mentions, @view value.args[2:end]) && !read_ok(value) && return false
                elseif mentions(value)
                    return false
                end
            elseif statement isa Expr && statement.head === :call
                any(mentions, @view statement.args[2:end]) && !read_ok(statement) && return false
                mentions(statement.args[1]) && return false
            elseif statement isa Core.ReturnNode
                isdefined(statement, :val) && is_derived(statement.val) && return false
            elseif is_derived(statement)
                push!(derived, ssa)
            elseif statement isa Expr && mentions(statement)
                return false
            elseif statement isa Core.GotoIfNot && is_derived(statement.cond)
                return false
            end
        end
    end
    true
end

# The same test over a reader's authored source, the entry to the user's code:
# `name` may only be an argument of a call or the array of a `ref`. A dotted call
# is not read here: the proof refuses it (Base's broadcast takes the operand's
# address to test aliasing), so it keeps its copy.
function _source_reads_through_interface(ex, name::Symbol, mod,
        @nospecialize(viewed), @nospecialize(copied), seen::Set{Any})
    ex === name && return false
    ex isa Expr || return true
    recur(a) = _source_reads_through_interface(a, name, mod, viewed, copied, seen)
    if ex.head === :ref
        return all(recur, ex.args[1] === name ? @view(ex.args[2:end]) : ex.args)
    elseif ex.head === :call
        args = ex.args
        for k in 2:length(args)
            args[k] === name || continue
            _reads_through_interface(_cell_binding(args[1], mod), k - 1, length(args) - 1,
                                     viewed, copied, seen) || return false
        end
        return all(a -> a === name || recur(a), @view args[2:end])
    elseif ex.head === :block
        return all(recur, ex.args)
    end
    # Any other form that holds the name is not a form this test can read.
    !_readonly_source_holds(ex, name)
end
_readonly_source_holds(ex, name) = ex === name ||
    ex isa Expr && any(a -> _readonly_source_holds(a, name), ex.args)

# The parts of a recipe whose whole source is one slice `A[i...]` of its inputs:
# the input position of `A`, and per index an input position (`Int`), the whole
# axis (`Colon()`) or a literal (`Some`). `nothing` for any other recipe,
# including an `end`/`begin` in an index and every source over captures.
function _readonly_slice_form(r::Recipe)
    length(r.outputs) == 1 || return nothing
    closure = _lane_source_closure(r)
    closure === nothing && return nothing
    _, params, mod = closure
    source = _lane_strip_lines(r.source)
    _lane_slice_node(source) || return nothing
    position(name) = name isa Symbol ? findfirst(==(name), params) : nothing
    array = position(source.args[1])
    array === nothing && return nothing
    parts = Any[]
    for index in @view source.args[2:end]
        k = position(index)
        if k !== nothing
            push!(parts, k)
        elseif index === :(:) || _cell_colon(index, mod)
            push!(parts, Colon())
        elseif index isa Integer
            push!(parts, Some(index))
        else
            return nothing
        end
    end
    (; array, parts)
end

_readonly_part_type(part::Int, input_types) = input_types[part]
_readonly_part_type(::Colon, input_types) = Colon
_readonly_part_type(part::Some, input_types) = typeof(something(part))

# Whether the cell of plate recipe `r` holds a recipe that is one slice, so that
# typing the program can be skipped when none does.
_readonly_slice_candidate(r::Recipe) = _readonly_slices_enabled() &&
    any(q -> _readonly_slice_form(q) !== nothing, getfield(r.op.kernel, :plan).recipes)

# The value types of plan `p` over its exemplars' types, through plates, scans
# and embedded kernels, as the generated body types them (`_uninferred_values!`).
_readonly_value_types(p::Plan, exemplar_types) = _uninferred_values!(_UninferredValue[], p,
    collect(Any, exemplar_types), Symbol[], _owns_array_regions(p))

"""
    _readonly_cell_slices(p, r, types) -> Set{Int}

The ids of the recipes in the cell plan of plate recipe `r` of `p` whose slices
the cell reads in place, given the types of `p`'s values
(`_readonly_value_types`). The comment at the top of this file states the
conditions.
"""
function _readonly_cell_slices(p::Plan, r::Recipe, types)
    marked = Set{Int}()
    inner = getfield(r.op.kernel, :plan)
    g, cell_graph = p.graph, inner.graph
    atomic = (typeof(r.op)::DataType).parameters[2]
    bound = Set(canon_id(g, only(q.outputs).id) for q in p.recipes if q.op isa _BoundConstant)
    bound_ports = Set(canon_id(cell_graph, port.id) for (position, port) in enumerate(inner.have)
                      if canon_id(g, r.inputs[position].id) in bound)
    cell_types = _uninferred_values!(_UninferredValue[], inner,
        Any[_plate_argument_type(get(types, canon_id(g, v.id), Any), position in atomic)
            for (position, v) in enumerate(r.inputs)], Symbol[], false)
    result = Set(canon_id(cell_graph, v.id) for v in inner.want)
    input_types(q) = Any[get(cell_types, canon_id(cell_graph, v.id), Any) for v in q.inputs]
    for q in inner.recipes
        form = _readonly_slice_form(q)
        form === nothing && continue
        slice = canon_id(cell_graph, only(q.outputs).id)
        slice in result && (_readonly_keeps_copy(q, "the slice is the cell's result"); continue)
        canon_id(cell_graph, q.inputs[form.array].id) in bound_ports &&
            (_readonly_keeps_copy(q, "its array is bound data (`bound=`)"); continue)
        types_of_q = input_types(q)
        A = types_of_q[form.array]
        A isa Type && isconcretetype(A) && A <: AbstractArray && isbitstype(eltype(A)) ||
            (_readonly_keeps_copy(q, "its array's type is not a concrete array of bits " *
                "values (a preparation without `exemplars` types none)"; array = A); continue)
        index_types = Any[_readonly_part_type(part, types_of_q) for part in form.parts]
        all(T -> T isa Type && isconcretetype(T), index_types) ||
            (_readonly_keeps_copy(q, "an index type is not concrete"; index_types); continue)
        # A scalar read is a number, not a slice.
        all(T -> T <: Integer, index_types) && continue
        copied = get(cell_types, slice, Any)
        viewed = Core.Compiler.return_type(_readonly_slice, Tuple{A,index_types...})
        copied isa Type && isconcretetype(copied) && isconcretetype(viewed) &&
            viewed !== copied || (_readonly_keeps_copy(q, "the copy's or the view's type is " *
                "not concrete, or they are one type"; copied, viewed); continue)
        readers = [reader for reader in inner.recipes
                   if any(v -> canon_id(cell_graph, v.id) == slice, reader.inputs)]
        isempty(readers) && continue
        refusal = nothing
        for reader in readers
            refusal = _readonly_reader(reader, input_types(reader), slice, cell_graph,
                                       viewed, copied)
            refusal === nothing || (_readonly_keeps_copy(q, refusal.why;
                reader = _lane_strip_lines(reader.source), viewed,
                refused = _readonly_refusals_text(refusal.frames)); break)
        end
        refusal === nothing && push!(marked, q.id)
    end
    marked
end

# The preparation log's account of a slice that keeps its copy, for a reader who
# expected a view: `JULIA_DEBUG=ReactiveKernels` shows it. Logging macros evaluate
# their values only when the level is enabled, so this costs nothing otherwise.
_readonly_keeps_copy(q::Recipe, why::String; detail...) =
    @debug "ReactiveKernels: a plate cell keeps its copy of a slice: $why" slice =
        _lane_strip_lines(q.source) detail... _group = :readonly_slices

# Each frame that refused the proof, innermost first: its signature, the argument
# the walk followed, and the statement of its optimized code it refused at.
_readonly_refusals_text(frames) = join((string("argument ", frame.position, " of ",
    frame.sig, ": ", frame.at isa String ? frame.at :
        string(frame.at.statement, " = ", frame.at.code)) for frame in frames), "\n")

# Whether `reader`, an authored call or an exact bare call (whose operation is the
# function itself), only reads the value `slice` over `viewed`, cannot tell it from
# `copied`, and computes the same concrete type of result over both: `nothing` when
# it does, else why not (`why`) and the frames that refused the proof (`frames`).
function _readonly_reader(reader::Recipe, input_types, slice, cell_graph,
                          @nospecialize(viewed), @nospecialize(copied))
    refuse(why) = (; why, frames = ())
    length(reader.outputs) == 1 || return refuse("its reader has several outputs")
    all(T -> T isa Type, input_types) || return refuse("its reader's input types are unknown")
    op = reader.op
    positions = [k for (k, v) in enumerate(reader.inputs) if canon_id(cell_graph, v.id) == slice]
    seen = Set{Any}()
    f = if op isa _KernelSourceOp
        getfield(op, :f) isa _KernelBranch && return refuse("its reader is a branch")
        closure = _lane_source_closure(reader)
        closure === nothing && return refuse("its reader's source has captures")
        _, params, mod = closure
        source = _lane_strip_lines(reader.source)
        all(k -> _source_reads_through_interface(source, params[k], mod, viewed, copied, seen),
            positions) || return refuse("its reader could tell a view from a copy " *
                "(it uses the slice other than through the array interface)")
        _lane_source_callable(getfield(op, :f))
    elseif op isa Function
        all(k -> _reads_through_interface(op, k, length(input_types), viewed, copied, seen),
            positions) || return refuse("its reader could tell a view from a copy " *
                "(it uses the slice other than through the array interface)")
        op
    else
        return refuse("its reader is not a call")
    end
    over(T) = Tuple{Any[k in positions ? T : input_types[k] for k in eachindex(input_types)]...}
    result = Core.Compiler.return_type(f, over(viewed))
    isconcretetype(result) && result === Core.Compiler.return_type(f, over(copied)) ||
        return refuse("its reader does not infer one concrete result type over the view " *
            "and the copy")
    signature = Tuple{typeof(f), over(viewed).parameters...}
    memo = _ReadOnlyMemo()
    verdicts = [_reads_only!(memo, signature, k + 1) for k in positions]
    all(==(:reads), verdicts) && return nothing
    any(==(:refuse), verdicts) || return refuse("its reader may return the slice, or a " *
        "value that holds it")
    (; why = "its reader is not proved to only read it (`_reads_only`)", frames = memo.refusals)
end

# The statement that reads the slice in place: `_readonly_slice(A, i...)` over the
# cell's argument expressions, each evaluated once as the call would.
function _readonly_slice_statements!(body, recipe::Recipe, args, out)
    form = _readonly_slice_form(recipe)
    names = Any[gensym(:plate_slice_argument) for _ in args]
    append!(body.args, (Expr(:(=), name, arg) for (name, arg) in zip(names, args)))
    parts = Any[part isa Int ? names[part] :
                part isa Colon ? Expr(:call, GlobalRef(Base, :Colon)) : something(part)
                for part in form.parts]
    push!(body.args, Expr(:(=), out, Expr(:call, GlobalRef(@__MODULE__, :_readonly_slice),
                                          names[form.array], parts...)))
    body
end
