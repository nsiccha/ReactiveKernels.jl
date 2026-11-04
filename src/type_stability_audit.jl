# Typed-IR audit of prepared programs: the code half of the type-stability check
# (`type_stability.jl`), and the gate the RK and RKPPL suites hold RK's own
# lowering to (`test/type_stability_audit.jl`, `test/test_type_stability.jl`).
# `_code_findings(f, argtypes; recurse)` infers and optimizes `f` for concrete
# argument types and reports, in the body and in every statically invoked
# callee that `recurse` accepts:
#
# - `:nonconcrete`: a used value whose inferred type is not concrete. A small
#   union of concrete types (`iterate`'s `Union{Nothing,Tuple{...}}`) is
#   accepted, an array of non-concrete elements is not;
# - `:dynamic`: a call Julia dispatches at run time (not a builtin), including
#   a splatted `Core._apply_iterate` call.
#
# Statements on a path that cannot return (throw paths) are skipped, and so is
# an unused call's result, whose inferred type Julia widens to `Any`. A
# non-concrete value whose arguments are all concrete is the source of the
# instability and is reported as `:root`. Statements inlined from a call chain
# in `_CODE_FINDINGS_JUSTIFIED` are exempt; each entry carries its
# justification. So are the statements of an expansion of Julia's logging
# macros (`_code_logging`). A finding records where its statement was written
# (`_code_frames`): after inlining, its method alone does not say.

# One frame of a statement's source: a function or a macro expansion, and the
# line it was written on.
struct _CodeFrame
    name::Symbol
    file::Symbol
    line::Int
end

Base.show(io::IO, f::_CodeFrame) = print(io, f.name, " @ ", f.file, ":", f.line)

struct _CodeFinding
    kind::Symbol
    method::String
    index::Int
    type::Any
    statement::String
    location::Vector{_CodeFrame} # innermost first
end

function Base.show(io::IO, f::_CodeFinding)
    print(io, f.kind, " in ", f.method, " %", f.index, " :: ", f.type, "  ", f.statement)
    isempty(f.location) || print(io, "  at ", first(f.location))
end

const _CODE_FINDINGS_MAX_UNION = 4

# RK-introduced residue that stays, with the reason it stays (decision
# `1hui2qi`: fix every site or justify it explicitly). Each entry is a call
# chain; a statement whose inlining chain contains all of its names, in any
# order, is exempt. None is exempt now: the one former entry, the Julia 1.12+
# `hasmethod` → `to_tuple_type` loop of the guarded source call, is gone since
# its method query is `Core._hasmethod` on the complete signature.
const _CODE_FINDINGS_JUSTIFIED = ()

_code_widen(@nospecialize(T)) = Core.Compiler.widenconst(T)

function _code_stable(@nospecialize(T))
    T === Union{} && return true
    if T isa Union
        members = Base.uniontypes(T)
        return length(members) <= _CODE_FINDINGS_MAX_UNION && all(_code_stable, members)
    end
    T isa DataType || return false
    T <: Type && return true
    isconcretetype(T) || return false
    T <: AbstractArray && return _code_stable(eltype(T))
    true
end

function _code_successors(code, types, i)
    statement = code[i]
    statement isa Core.ReturnNode && return Int[]
    statement isa Core.GotoNode && return [statement.label]
    statement isa Core.GotoIfNot && return [i + 1, statement.dest]
    if isdefined(Core, :EnterNode) && statement isa getfield(Core, :EnterNode)
        return [i + 1, statement.catch_dest]
    end
    statement isa Expr && statement.head === :enter && return [i + 1, statement.args[1]]
    types[i] === Union{} && return Int[]
    i < length(code) ? [i + 1] : Int[]
end

# Statements from which a value-returning `return` is reachable.
function _code_returning(ci::Core.CodeInfo)
    code, types = ci.code, ci.ssavaluetypes
    n = length(code)
    next = [_code_successors(code, types, i) for i in 1:n]
    live = [code[i] isa Core.ReturnNode && isdefined(code[i], :val) for i in 1:n]
    changed = true
    while changed
        changed = false
        for i in n:-1:1
            live[i] && continue
            if any(s -> 1 <= s <= n && live[s], next[i])
                live[i] = changed = true
            end
        end
    end
    live
end

# SSA values some statement reads.
function _code_used(ci::Core.CodeInfo)
    reads = falses(length(ci.code))
    function walk(x)
        x isa Core.SSAValue && 1 <= x.id <= length(reads) && (reads[x.id] = true)
        if x isa Expr
            foreach(walk, x.args)
        elseif x isa Core.PhiNode
            foreach(i -> isassigned(x.values, i) && walk(x.values[i]), eachindex(x.values))
        elseif x isa Core.PhiCNode
            foreach(walk, x.values)
        elseif x isa Core.PiNode || x isa Core.UpsilonNode
            isdefined(x, :val) && walk(x.val)
        elseif x isa Core.ReturnNode
            isdefined(x, :val) && walk(x.val)
        elseif x isa Core.GotoIfNot
            walk(x.cond)
        end
        nothing
    end
    foreach(walk, ci.code)
    reads
end

_code_argument_type(ci, x::Core.SSAValue) = _code_widen(ci.ssavaluetypes[x.id])
_code_argument_type(ci, x::Core.Argument) = _code_widen(ci.slottypes[x.n])
_code_argument_type(ci, x::GlobalRef) =
    isdefined(x.mod, x.name) ? Core.Typeof(getglobal(x.mod, x.name)) : Any
_code_argument_type(ci, x::QuoteNode) = Core.Typeof(x.value)
_code_argument_type(ci, x) = Core.Typeof(x)

_code_callee(ci, f::GlobalRef) = isdefined(f.mod, f.name) ? getglobal(f.mod, f.name) : nothing
function _code_callee(ci, f::Core.SSAValue)
    T = _code_widen(ci.ssavaluetypes[f.id])
    Base.issingletontype(T) ? T.instance : nothing
end
_code_callee(ci, f::QuoteNode) = f.value
_code_callee(ci, f) = f

function _code_dynamic(ci, statement::Expr)
    statement.head === :call || return false
    f = _code_callee(ci, statement.args[1])
    f isa Core.Builtin || return true
    f === Core._apply_iterate || f === Core._call_latest
end

_code_call_arguments(statement::Expr) =
    statement.head === :invoke ? statement.args[3:end] :
    statement.head === :call ? statement.args[2:end] : Any[]

_code_valued(statement) =
    statement isa Expr && statement.head in (:call, :invoke, :new, :foreigncall, :splatnew) ||
    statement isa Core.PhiNode || statement isa Core.PiNode || statement isa GlobalRef

# Julia 1.14 also embeds resolved global bindings in optimized IR. Query
# the inferred partition, preserving its world, rather than the current global.
@static if isdefined(Core.Compiler, :partition_singleton)
    _code_callee(ci, x::Core.BindingPartition) = Core.Compiler.partition_singleton(x)
    _code_argument_type(ci, x::Core.BindingPartition) = Core.Compiler.partition_rt_widened(x)
    _code_valued(::Core.BindingPartition) = true
end

function _code_method_instance(statement::Expr)
    mi = statement.args[1]
    isdefined(Core, :CodeInstance) && mi isa Core.CodeInstance && (mi = mi.def)
    mi isa Core.MethodInstance ? mi : nothing
end

_code_label(m::Method) = string(m.module, ".", m.name, " @ ", m.file, ":", m.line)

# A statement as a reader recognizes it: a statically invoked call as the
# signature it calls (`invoke f(::Vector{Float64}, ::Float64)`), anything else
# unquoted (`%2 * _2`).
function _code_statement_text(statement)
    if statement isa Expr && statement.head === :invoke
        mi = _code_method_instance(statement)
        mi === nothing || return "invoke " * _code_signature_text(mi.specTypes)
    end
    sprint(Base.show_unquoted, statement)
end

function _code_signature_text(@nospecialize(tt))
    tt = Base.unwrap_unionall(tt)
    tt isa DataType && tt <: Tuple && !isempty(tt.parameters) || return string(tt)
    F, arguments = tt.parameters[1], tt.parameters[2:end]
    callee = F isa DataType && Base.issingletontype(F) && F <: Function ?
        string(F.instance) : "(::$F)"
    callee * "(" * join(("::$T" for T in arguments), ", ") * ")"
end

_code_frame_name(m::Core.MethodInstance) = _code_frame_name(m.def)
_code_frame_name(m::Method) = m.name
_code_frame_name(m) = m

_code_frame_name(m::Core.CodeInstance) = _code_frame_name(m.def)

_code_frame(@nospecialize(method), file, line) =
    _CodeFrame(Symbol(_code_frame_name(method)), Symbol(file), Int(line))

# The source of statement `i`, innermost frame first: the function it was
# written in, each function that was inlined into, and the macro expansions in
# between (needs the code inferred with `debuginfo = :source`). A frame of a body
# RuntimeGeneratedFunctions generates has no line, `none:0`. From Julia 1.12 the
# frames are the `DebugInfo` line tables and inlining edges, which the compiler's
# own walk (`append_scopes!`) lists outermost first; before, a `LineInfoNode`
# chain from the innermost frame.
function _code_frames(ci::Core.CodeInfo, i)
    frames = _CodeFrame[]
    if isdefined(ci, :debuginfo)
        scopes = Base.IRShow.LineInfoNode[]
        Base.IRShow.append_scopes!(scopes, i, ci.debuginfo, nothing)
        for node in Iterators.reverse(scopes)
            push!(frames, _code_frame(node.method, node.file, node.line))
        end
        return frames
    end
    location = ci.codelocs[i]
    while location != 0
        node = ci.linetable[location]
        push!(frames, _code_frame(node.method, node.file, node.line))
        location = node.inlined_at
    end
    frames
end

_code_justified(frames) = let chain = (f.name for f in frames)
    any(entry -> all(in(chain), entry), _CODE_FINDINGS_JUSTIFIED)
end

# Base's logging source, where the code its logging macros (`@debug`, `@info`,
# `@warn`, `@error`, `@logmsg`) expand to is written.
const _LOGGING_SOURCES = Set{Symbol}(m.file for m in methods(Base.CoreLogging.logmsg_code))

# Whether a statement with these frames belongs to an expansion of Julia's
# logging macros. That code is Julia's own, written into the caller's body by
# the macro, and it dispatches at run time by design: the logger is a dynamically
# scoped `AbstractLogger`, and `shouldlog` and `handle_message` are called in the
# latest world. It runs only when its log statement is reached, and dispatches
# only when a logger accepts the record, so no client can make it concrete.
# The audit therefore treats it as it treats the bodies of Julia's own
# functions, which it never enters. This covers the record's message and
# fields, which run only for an accepted record. Measured on Julia 1.10.12,
# 1.12.7 and 1.13.1 (snag `type-stability-a-a6941de1`: the `@debug` calls on
# the underflow branches of SpecialFunctions' `beta_inc` refused a
# negative-binomial likelihood at `type_stability_level = :all`).
_code_logging(frames) =
    any(f -> f.name === Symbol("macro expansion") && f.file in _LOGGING_SOURCES, frames)

function _within_module(m::Module, root::Module)
    while true
        m === root && return true
        parent = parentmodule(m)
        parent === m && return false
        m = parent
    end
end

"""
    _rk_or_generated(modules...) -> predicate

Recurse into ReactiveKernels (and its submodules), into the bodies RK
generates (`RuntimeGeneratedFunctions`), and into methods of `modules`, the
authoring modules of the audited kernels.
"""
function _rk_or_generated(modules::Module...)
    m -> _within_module(m.module, @__MODULE__) ||
         nameof(m.module) === :RuntimeGeneratedFunctions ||
         any(root -> _within_module(m.module, root), modules)
end

"""
    _code_findings(f, argtypes; recurse = _rk_or_generated()) -> Vector{_CodeFinding}

Audit `f` called with arguments of types `argtypes`, recursing into every
statically invoked callee whose method `recurse` accepts.
"""
_code_findings(@nospecialize(f), @nospecialize(argtypes); recurse = _rk_or_generated()) =
    _code_audit!(_CodeFinding[], Set{Any}(), Base.signature_type(f, argtypes), "call",
                 recurse)

function _code_audit!(found, seen, @nospecialize(tt), where, recurse)
    tt in seen && return found
    push!(seen, tt)
    for (ci, _) in Base.code_typed_by_type(tt; optimize = true, debuginfo = :source)
        ci isa Core.CodeInfo || continue
        live, reads = _code_returning(ci), _code_used(ci)
        for (i, statement) in enumerate(ci.code)
            live[i] || continue
            # A return produces no SSA value or call to audit; its operand's
            # producer is checked separately. Julia can synthesize a lone
            # constant return with an SSA count instead of a type vector.
            statement isa Core.ReturnNode && continue
            T = _code_widen(ci.ssavaluetypes[i])
            unstable = reads[i] && _code_valued(statement) && !_code_stable(T)
            dynamic = statement isa Expr && _code_dynamic(ci, statement)
            mi = statement isa Expr && statement.head === :invoke ?
                _code_method_instance(statement) : nothing
            callee = mi === nothing ? nothing : mi.def
            recursing = callee isa Method && recurse(callee)
            # The exemptions read the statement's debug info, so only for a
            # statement the audit would report or enter.
            unstable || dynamic || recursing || continue
            frames = _code_frames(ci, i)
            (_code_justified(frames) || _code_logging(frames)) && continue
            if unstable
                root = statement isa Expr && statement.head in (:call, :invoke) &&
                    all(x -> _code_stable(_code_argument_type(ci, x)),
                        _code_call_arguments(statement))
                push!(found, _CodeFinding(root ? :root : :nonconcrete, where, i, T,
                                          _code_statement_text(statement), frames))
            end
            dynamic && push!(found, _CodeFinding(:dynamic, where, i, T,
                                                 _code_statement_text(statement), frames))
            recursing && _code_audit!(found, seen, mi.specTypes, _code_label(callee), recurse)
        end
    end
    found
end
