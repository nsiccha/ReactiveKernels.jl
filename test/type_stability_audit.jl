# Typed-IR audit of prepared programs, shared by the ReactiveKernels and
# ReactiveKernelsPPL type-stability gates. `findings(f, argtypes)` infers and
# optimizes `f` for concrete argument types and reports, in the body and in
# every statically invoked callee that `recurse` accepts:
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
# instability and is reported as `:root`.
module TypeStabilityAudit

export findings, ad_call_findings, rk_or_generated

import ReactiveKernels
import DifferentiationInterface

struct Finding
    kind::Symbol
    method::String
    index::Int
    type::Any
    statement::String
end

Base.show(io::IO, f::Finding) = print(io, f.kind, " in ", f.method, " %", f.index,
    " :: ", f.type, "  ", f.statement)

const MAX_UNION = 4

widen(@nospecialize(T)) = Core.Compiler.widenconst(T)

function stable(@nospecialize(T))
    T === Union{} && return true
    if T isa Union
        members = Base.uniontypes(T)
        return length(members) <= MAX_UNION && all(stable, members)
    end
    T isa DataType || return false
    T <: Type && return true
    isconcretetype(T) || return false
    T <: AbstractArray && return stable(eltype(T))
    true
end

function successors(code, types, i)
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
function returning(ci::Core.CodeInfo)
    code, types = ci.code, ci.ssavaluetypes
    n = length(code)
    next = [successors(code, types, i) for i in 1:n]
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
function used(ci::Core.CodeInfo)
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

argument_type(ci, x::Core.SSAValue) = widen(ci.ssavaluetypes[x.id])
argument_type(ci, x::Core.Argument) = widen(ci.slottypes[x.n])
argument_type(ci, x::GlobalRef) =
    isdefined(x.mod, x.name) ? Core.Typeof(getglobal(x.mod, x.name)) : Any
argument_type(ci, x::QuoteNode) = Core.Typeof(x.value)
argument_type(ci, x) = Core.Typeof(x)

callee(ci, f::GlobalRef) = isdefined(f.mod, f.name) ? getglobal(f.mod, f.name) : nothing
function callee(ci, f::Core.SSAValue)
    T = widen(ci.ssavaluetypes[f.id])
    Base.issingletontype(T) ? T.instance : nothing
end
callee(ci, f::QuoteNode) = f.value
callee(ci, f) = f

function dynamic(ci, statement::Expr)
    statement.head === :call || return false
    f = callee(ci, statement.args[1])
    f isa Core.Builtin || return true
    f === Core._apply_iterate || f === Core._call_latest
end

call_arguments(statement::Expr) =
    statement.head === :invoke ? statement.args[3:end] :
    statement.head === :call ? statement.args[2:end] : Any[]

valued(statement) =
    statement isa Expr && statement.head in (:call, :invoke, :new, :foreigncall, :splatnew) ||
    statement isa Core.PhiNode || statement isa Core.PiNode || statement isa GlobalRef

function method_instance(statement::Expr)
    mi = statement.args[1]
    isdefined(Core, :CodeInstance) && mi isa Core.CodeInstance && (mi = mi.def)
    mi isa Core.MethodInstance ? mi : nothing
end

label(m::Method) = string(m.module, ".", m.name, " @ ", m.file, ":", m.line)

"""
    findings(f, argtypes; recurse = rk_or_generated) -> Vector{Finding}

Audit `f` called with arguments of types `argtypes`, recursing into every
statically invoked callee whose method `recurse` accepts.
"""
findings(@nospecialize(f), @nospecialize(argtypes); recurse = rk_or_generated) =
    audit!(Finding[], Set{Any}(), Base.signature_type(f, argtypes), "call", recurse)

function audit!(found, seen, @nospecialize(tt), where, recurse)
    tt in seen && return found
    push!(seen, tt)
    for (ci, _) in Base.code_typed_by_type(tt; optimize = true)
        ci isa Core.CodeInfo || continue
        live, reads = returning(ci), used(ci)
        for (i, statement) in enumerate(ci.code)
            live[i] || continue
            T = widen(ci.ssavaluetypes[i])
            if reads[i] && valued(statement) && !stable(T)
                root = statement isa Expr && statement.head in (:call, :invoke) &&
                    all(x -> stable(argument_type(ci, x)), call_arguments(statement))
                push!(found, Finding(root ? :root : :nonconcrete, where, i, T,
                                     first(sprint(show, statement), 300)))
            end
            statement isa Expr && dynamic(ci, statement) &&
                push!(found, Finding(:dynamic, where, i, T, first(sprint(show, statement), 300)))
            if statement isa Expr && statement.head === :invoke
                mi = method_instance(statement)
                mi === nothing && continue
                m = mi.def
                m isa Method && recurse(m) || continue
                audit!(found, seen, mi.specTypes, label(m), recurse)
            end
        end
    end
    found
end

function within(m::Module, root::Module)
    while true
        m === root && return true
        parent = parentmodule(m)
        parent === m && return false
        m = parent
    end
end

"""
    rk_or_generated(modules...) -> predicate

Recurse into ReactiveKernels (and its submodules), into the bodies RK
generates (`RuntimeGeneratedFunctions`), and into methods of `modules`, the
authoring modules of the audited kernels.
"""
function rk_or_generated(modules::Module...)
    m -> within(m.module, ReactiveKernels) ||
         nameof(m.module) === :RuntimeGeneratedFunctions ||
         any(root -> within(m.module, root), modules)
end
rk_or_generated(m::Method) = rk_or_generated()(m)

"""
    ad_call_findings(prepared, args...; recurse) -> Vector{Finding}

Audit the callable a `PreparedADKernel` hands its backend, at the point and
contexts `prepared` builds from `args`.
"""
function ad_call_findings(prepared, args...; recurse = rk_or_generated)
    point, contexts = ReactiveKernels._ad_prepared_arguments(prepared, args, (;))
    types = (Core.Typeof(point),
             map(c -> Core.Typeof(DifferentiationInterface.unwrap(c)), contexts)...)
    findings(prepared.call, types; recurse)
end

end # module
