# Registry-free native NUTS control metadata.
#
# This file deliberately separates STRUCTURE from EMISSION.  A cold compiler lowers the detached
# MethodIR values into zero-field node TYPES.  Generated native methods may later inspect only those
# type parameters and concrete cfg fields; they never consult a module-level table, `Core.eval`, or a
# later-world method.  Captured callable values are validated eagerly in the owner's world and kept in
# one concrete Tuple, addressed from the type tree by an integer index.

abstract type _NativeNode end
abstract type _NativeValue <: _NativeNode end
abstract type _NativeStmt <: _NativeNode end

struct _NNSelf <: _NativeValue end
struct _NNSelfField{Path} <: _NativeValue end
struct _NNLocal{N} <: _NativeValue end
struct _NNFormal{N,P,K} <: _NativeValue end
struct _NNLit{V} <: _NativeValue end
struct _NNExt{K} <: _NativeValue end                 # the only structural externals are Bool / indexing end
struct _NNIndex{B,I} <: _NativeValue end             # B node type; I Tuple node type
struct _NNGetfield{B,F} <: _NativeValue end
struct _NNShort{Op,L,R} <: _NativeValue end
struct _NNIfExpr{C,T,E} <: _NativeValue end
struct _NNBlockExpr{S,V} <: _NativeValue end
struct _NNTupleExpr{E} <: _NativeValue end
struct _NNNamedTuple{Names,V} <: _NativeValue end
struct _NNRegistered{I,Intrinsic,Token,A,Kw,Dot} <: _NativeValue end
struct _NNFieldCall{Path,A,Kw,Hint} <: _NativeValue end
struct _NNCallExpr{Name,Mid,T,A,Kw} <: _NativeValue end

struct _NNLocalAssign{L,R} <: _NativeStmt end
struct _NNCall{Name,Mid,T,A,Kw} <: _NativeStmt end
struct _NNExprStmt{E} <: _NativeStmt end
struct _NNIf{C,T,E} <: _NativeStmt end
struct _NNFor{V,I,B} <: _NativeStmt end
struct _NNWhile{C,B} <: _NativeStmt end
struct _NNGuard{Op,C,B} <: _NativeStmt end
struct _NNReturn{V} <: _NativeStmt end                # V === Nothing means a bare return
struct _NNBreak <: _NativeStmt end
struct _NNContinue <: _NativeStmt end
struct _NNPlaceWrite{T,Root,Owner,Alias,R,Dot} <: _NativeStmt end
struct _NNPlaceSwap{W} <: _NativeStmt end
struct _NNWriteReturn{W} <: _NativeStmt end           # write and native return are one first-class node

struct _NNKw{Name,V} end
struct _NNFormalDesc{Name,Kind,Required,Default} end
struct _NNMethod{Mid,Name,Formals,Body} end
struct _NNNoStats end
struct _NNStats{Method} end
struct _NativeProgram{OwnerToken,PlanT,RootMid,Methods,Derived,Stats} end

_native_tuple_type(xs) = Tuple{xs...}
_native_params(::Type{T}) where {T} = T.parameters

struct _NativeEncodeReject <: Exception
    reason::String
end
Base.showerror(io::IO, e::_NativeEncodeReject) = print(io, "native NUTS encoder: ", e.reason)
_native_reject(msg) = throw(_NativeEncodeReject(msg))

# A local cold-compile authority table.  It is intentionally a mutable local scratch object; only its
# immutable `Tuple(values)` result survives.  Keys are the complete authored `_CapturedCalleeRef` values
# (module identity + slot + optional qualified field), never final Symbols or resolved target names.
mutable struct _NativeCalleeBuilder
    refs::Vector{_CapturedCalleeRef}
    regs::Vector{_KernelRegistration}
    values::Vector{Any}
    index::Dict{_CapturedCalleeRef,Int}
end
_NativeCalleeBuilder() = _NativeCalleeBuilder(_CapturedCalleeRef[], _KernelRegistration[], Any[],
                                               Dict{_CapturedCalleeRef,Int}())

function _native_validate_registered(x::_RegisteredCall)
    reg = x.registration
    if reg.kind === :intrinsic
        reg.token === Symbol("__rk_intrinsic_copy!!__") ||
            _native_reject("unsupported intrinsic token $(reg.token)")
        length(x.args) == 2 || _native_reject("copy!! requires exactly two positional arguments")
        isempty(x.kw) || _native_reject("copy!! does not admit keyword arguments")
        x.broadcast && _native_reject("copy!! does not admit broadcast syntax")
        return nothing
    end
    callee = _lf_callee(x) # eager owner-world rebind validation; never called by generated code
    pe = reg.primitive_effect
    if reg.kind in (:primitive, :declared_effect)
        pe === nothing && _native_reject("$(reg.kind) call has no detached effect descriptor")
        length(x.args) == pe.arity || _native_reject(
            "captured $(reg.kind) call arity $(length(x.args)) disagrees with descriptor arity $(pe.arity)")
    elseif !(reg.kind in (:pure_primitive, :free_method, :object_kernel, :stateless))
        _native_reject("unsupported captured registration kind $(reg.kind)")
    end
    isempty(x.kw) || _native_reject("registered-call keywords are not encoded by the native NUTS surface")
    callee
end

function _native_callee_index!(b::_NativeCalleeBuilder, x::_RegisteredCall)
    value = _native_validate_registered(x)
    x.registration.kind === :intrinsic && return 0
    if haskey(b.index, x.ref)
        i = b.index[x.ref]
        b.regs[i] == x.registration || _native_reject(
            "one captured reference carries two different detached registrations: $(x.ref)")
        b.values[i] === value || _native_reject(
            "one captured reference resolved to two different callable identities: $(x.ref)")
        return i
    end
    push!(b.refs, x.ref); push!(b.regs, x.registration); push!(b.values, value)
    i = length(b.refs); b.index[x.ref] = i; i
end

function _native_encode_kw(kw, b)
    out = Any[]
    for p in kw
        p.first === _KMIR_KWSPLAT && _native_reject("keyword splats are not admitted by native NUTS")
        push!(out, _NNKw{p.first,_native_encode(p.second, b)})
    end
    _native_tuple_type(out)
end

function _native_single_mid(name, candidates)
    length(candidates) == 1 || _native_reject(
        "sibling call `$name` has $(length(candidates)) candidates; exact MethodId narrowing is required")
    candidates[1].id.decl
end

function _native_encode(x, b::_NativeCalleeBuilder)
    if x isa _SelfRef
        _NNSelf
    elseif x isa _SelfField
        _NNSelfField{x.path}
    elseif x isa _LocalRef
        _NNLocal{x.name}
    elseif x isa _FormalRef
        x.kind in (:possplat, :kwsplat) && _native_reject("formal splat `$(x.arg)` is not admitted")
        _NNFormal{x.arg,x.pos,x.kind}
    elseif x isa _Lit
        isbits(x.value) || _native_reject("literal $(typeof(x.value)) is not type-encodable")
        _NNLit{x.value}
    elseif x isa _ExtRef
        x.ref.name === :Bool ? _NNExt{:Bool} :
        x.ref.name === :end  ? _NNExt{:end}  :
        _native_reject("external $(x.ref) is not a sanctioned structural Bool/end reference")
    elseif x isa _Index
        _NNIndex{_native_encode(x.base,b),_native_tuple_type(map(y -> _native_encode(y,b), x.idxs))}
    elseif x isa _Getfield
        _NNGetfield{_native_encode(x.base,b),x.field}
    elseif x isa _Short
        x.op in (:&&, :||) || _native_reject("invalid short-circuit operator $(x.op)")
        _NNShort{x.op,_native_encode(x.lhs,b),_native_encode(x.rhs,b)}
    elseif x isa _IfExpr
        _NNIfExpr{_native_encode(x.cond,b),_native_encode(x.thenv,b),_native_encode(x.elsev,b)}
    elseif x isa _BlockExpr
        _NNBlockExpr{_native_tuple_type(map(y -> _native_encode(y,b),x.stmts)),_native_encode(x.value,b)}
    elseif x isa _TupleExpr
        _NNTupleExpr{_native_tuple_type(map(y -> _native_encode(y,b),x.elts))}
    elseif x isa _NamedTuple
        _NNNamedTuple{x.names,_native_tuple_type(map(y -> _native_encode(y,b),x.vals))}
    elseif x isa _RegisteredCall
        i = _native_callee_index!(b, x); reg = x.registration
        _NNRegistered{i,reg.kind === :intrinsic,reg.token,
            _native_tuple_type(map(y -> _native_encode(y,b),x.args)),_native_encode_kw(x.kw,b),x.broadcast}
    elseif x isa _FieldCall
        _NNFieldCall{x.path,_native_tuple_type(map(y -> _native_encode(y,b),x.pos)),
                     _native_encode_kw(x.kw,b),x.hint}
    elseif x isa _CallExpr
        x.target isa _SelfRef || _native_reject("sibling value call target is not __self__")
        _NNCallExpr{x.name,_native_single_mid(x.name,x.candidates),_native_encode(x.target,b),
                    _native_tuple_type(map(y -> _native_encode(y,b),x.pos)),_native_encode_kw(x.kw,b)}
    elseif x isa _LocalAssign
        _NNLocalAssign{x.lhs,_native_encode(x.rhs,b)}
    elseif x isa _Call
        x.target isa _SelfRef || _native_reject("sibling statement call target is not __self__")
        _NNCall{x.name,_native_single_mid(x.name,x.candidates),_native_encode(x.target,b),
                _native_tuple_type(map(y -> _native_encode(y,b),x.pos)),_native_encode_kw(x.kw,b)}
    elseif x isa _ExprStmt
        _NNExprStmt{_native_encode(x.expr,b)}
    elseif x isa _If
        _NNIf{_native_encode(x.cond,b),_native_tuple_type(map(y -> _native_encode(y,b),x.thenb)),
              _native_tuple_type(map(y -> _native_encode(y,b),x.elseb))}
    elseif x isa _For
        length(x.var) == 1 || _native_reject("destructuring for variables are not admitted")
        _NNFor{x.var[1],_native_encode(x.iter,b),_native_tuple_type(map(y -> _native_encode(y,b),x.body))}
    elseif x isa _While
        _NNWhile{_native_encode(x.cond,b),_native_tuple_type(map(y -> _native_encode(y,b),x.body))}
    elseif x isa _Guard
        x.op in (:&&, :||) || _native_reject("invalid guard operator $(x.op)")
        _NNGuard{x.op,_native_encode(x.cond,b),_native_tuple_type(map(y -> _native_encode(y,b),x.body))}
    elseif x isa _Return
        _NNReturn{x.value === nothing ? Nothing : _native_encode(x.value,b)}
    elseif x isa _Break
        _NNBreak
    elseif x isa _Continue
        _NNContinue
    elseif x isa _PlaceWrite
        _NNPlaceWrite{_native_encode(x.target,b),x.root,x.owner,x.alias,_native_encode(x.rhs,b),x.dot}
    elseif x isa _PlaceSwap
        _NNPlaceSwap{_native_tuple_type(map(y -> _native_encode(y,b),x.targets))}
    elseif x isa _SetReturn
        _NNWriteReturn{_native_encode(x.write,b)}
    elseif x isa _OpCall
        _native_reject("opaque/operator call $(x.op) lacks a detached captured registration")
    elseif x isa _Comparison
        _native_reject("chained comparison is outside the current native NUTS surface")
    elseif x isa _NodeExpr
        _native_reject("@node raw payload is not valid native control metadata")
    elseif x isa _SubjectMethodCall
        _native_reject("subject-method calls are not valid inside nuts_state methods")
    else
        _native_reject("unsupported node $(typeof(x))")
    end
end

function _native_encode_formals(ir, b)
    out = Any[]
    for f in ir.formals
        f.kind in (:pos,) || _native_reject(
            "native nuts method $(ir.id.name) formal $(f.name) has unsupported kind $(f.kind)")
        default = f.default === nothing ? Nothing : _native_encode(f.default, b)
        push!(out, _NNFormalDesc{f.name,f.kind,f.required,default})
    end
    _native_tuple_type(out)
end

function _native_encode_method(ir, b)
    ir.ok || _native_reject("MethodIR $(ir.id.name)#$(ir.id.decl) is not ok: $(ir.reason)")
    _NNMethod{ir.id.decl,ir.id.name,_native_encode_formals(ir,b),
              _native_tuple_type(map(y -> _native_encode(y,b),ir.body))}
end

function _native_encode_program(irs, ::Type{PlanT}, owner_token;
                                root_name::Symbol=:step!, derived=(), stats_ir=nothing) where {PlanT}
    ordered = sort!(collect(irs), by = ir -> ir.id.decl)
    mids = Int[ir.id.decl for ir in ordered]
    length(unique(mids)) == length(mids) || _native_reject("duplicate MethodId declaration ordinal")
    root = only(ir for ir in ordered if ir.id.name === root_name)
    b = _NativeCalleeBuilder()
    methods = _native_tuple_type(map(ir -> _native_encode_method(ir,b), ordered))
    stats = if stats_ir === nothing
        _NNNoStats
    else
        _validate_stats_body(stats_ir, ()) # caller normally supplies the exact produced slots; structural fallback below
        _NNStats{_native_encode_method(stats_ir,b)}
    end
    P = _NativeProgram{owner_token,PlanT,root.id.decl,methods,Tuple(sort!(collect(Symbol,derived))),stats}
    (program=P, callees=Tuple(b.values), refs=Tuple(b.refs), registrations=Tuple(b.regs))
end

# Total type-tree census used by both the production compiler and its source-derived gate.  Method/formal/
# Tuple carriers are transparent; only encoded `_MExpr`/`_MStmt` counterparts contribute to the count.
function _native_type_node_count(::Type{T}) where {T}
    n = T <: _NativeNode ? 1 : 0
    T isa DataType || return n
    for p in T.parameters
        p isa Type || continue
        if p <: _NativeNode || p <: _NNMethod || p <: _NNFormalDesc || p <: _NNStats || p <: Tuple
            n += _native_type_node_count(p)
        end
    end
    n
end
function _native_program_node_count(::Type{<:_NativeProgram{OT,PT,RM,Methods,D,S}}) where {OT,PT,RM,Methods,D,S}
    n = sum(_native_type_node_count(m) for m in Methods.parameters)
    S <: _NNStats && (n += _native_type_node_count(S.parameters[1]))
    n
end
