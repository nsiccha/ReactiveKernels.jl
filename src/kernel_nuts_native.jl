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
    owner_mod::Module
    refs::Vector{_CapturedCalleeRef}
    regs::Vector{_KernelRegistration}
    values::Vector{Any}
    index::Dict{_CapturedCalleeRef,Int}
end
_NativeCalleeBuilder(owner_mod::Module) = _NativeCalleeBuilder(owner_mod, _CapturedCalleeRef[],
    _KernelRegistration[], Any[], Dict{_CapturedCalleeRef,Int}())

function _native_validate_registered(x::_RegisteredCall)
    reg = x.registration
    if reg.kind === :intrinsic
        reg.token === Symbol("__rk_intrinsic_copy!!__") ||
            _native_reject("unsupported intrinsic token $(reg.token)")
        length(x.args) == 2 || _native_reject("copy!! requires exactly two positional arguments")
        isempty(x.kw) || _native_reject("copy!! does not admit keyword arguments")
        x.broadcast && _native_reject("copy!! does not admit broadcast syntax")
        return reg.source
    end
    callee = try
        _lf_callee(x) # eager owner-world rebind validation; never called by generated code
    catch e
        e isa _LLowerReject || rethrow()
        _native_reject(sprint(showerror,e))
    end
    pe = reg.primitive_effect
    if reg.kind === :primitive
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
        (x.value isa Bool || x.value isa Int) || _native_reject(
            "literal $(repr(x.value))::$(typeof(x.value)) is outside the sanctioned Bool/Int NUTS set")
        _NNLit{x.value}
    elseif x isa _ExtRef
        x.ref.mod === b.owner_mod || _native_reject(
            "structural external $(x.ref) is not rooted in the owner module $(b.owner_mod)")
        x.ref.name === :Bool ? (isdefined(x.ref.mod,:Bool) && getglobal(x.ref.mod,:Bool) === Bool ?
            _NNExt{:Bool} : _native_reject("owner Bool binding was rebound")) :
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
        f.type === nothing || _native_reject(
            "native nuts method $(ir.id.name) formal $(f.name) has an unsupported type annotation")
        default = f.default === nothing ? Nothing : _native_encode(f.default, b)
        push!(out, _NNFormalDesc{f.name,f.kind,f.required,default})
    end
    _native_tuple_type(out)
end

# Independent semantic fingerprints.  The raw walker and type-tree walker are intentionally separate
# implementations: the gate proves that equal node counts did not hide a collapsed target/path/callee/default.
_native_ref_fp(r::_CapturedCalleeRef) = (r.slot.mod,r.slot.name,r.field)
_native_reg_fp(r::_KernelRegistration) =
    (r.token,r.kind,r.subject,r.write_roots,r.read_roots,r.is_bang_bang,r.primitive_effect)
_native_kw_fp(kw) = Tuple((p.first,_native_ir_fp(p.second)) for p in kw)
function _native_ir_fp(x)
    x isa _SelfRef && return (:self,)
    x isa _SelfField && return (:selffield,x.path)
    x isa _LocalRef && return (:local,x.name)
    x isa _FormalRef && return (:formal,x.arg,x.pos,x.kind)
    x isa _Lit && return (:lit,x.value)
    x isa _ExtRef && return (:ext,x.ref.name)
    x isa _Index && return (:index,_native_ir_fp(x.base),Tuple(map(_native_ir_fp,x.idxs)))
    x isa _Getfield && return (:getfield,_native_ir_fp(x.base),x.field)
    x isa _Short && return (:short,x.op,_native_ir_fp(x.lhs),_native_ir_fp(x.rhs))
    x isa _IfExpr && return (:ifexpr,_native_ir_fp(x.cond),_native_ir_fp(x.thenv),_native_ir_fp(x.elsev))
    x isa _BlockExpr && return (:blockexpr,Tuple(map(_native_ir_fp,x.stmts)),_native_ir_fp(x.value))
    x isa _TupleExpr && return (:tupleexpr,Tuple(map(_native_ir_fp,x.elts)))
    x isa _NamedTuple && return (:namedtuple,x.names,Tuple(map(_native_ir_fp,x.vals)))
    x isa _RegisteredCall && return (:registered,_native_ref_fp(x.ref),_native_reg_fp(x.registration),
        x.intrinsic,Tuple(map(_native_ir_fp,x.args)),_native_kw_fp(x.kw),x.broadcast)
    x isa _FieldCall && return (:fieldcall,x.path,Tuple(map(_native_ir_fp,x.pos)),_native_kw_fp(x.kw),x.hint)
    x isa _CallExpr && return (:callexpr,x.name,_native_single_mid(x.name,x.candidates),
        _native_ir_fp(x.target),Tuple(map(_native_ir_fp,x.pos)),_native_kw_fp(x.kw))
    x isa _LocalAssign && return (:localassign,x.lhs,_native_ir_fp(x.rhs))
    x isa _Call && return (:call,x.name,_native_single_mid(x.name,x.candidates),_native_ir_fp(x.target),
        Tuple(map(_native_ir_fp,x.pos)),_native_kw_fp(x.kw))
    x isa _ExprStmt && return (:exprstmt,_native_ir_fp(x.expr))
    x isa _If && return (:if,_native_ir_fp(x.cond),Tuple(map(_native_ir_fp,x.thenb)),Tuple(map(_native_ir_fp,x.elseb)))
    x isa _For && return (:for,x.var[1],_native_ir_fp(x.iter),Tuple(map(_native_ir_fp,x.body)))
    x isa _While && return (:while,_native_ir_fp(x.cond),Tuple(map(_native_ir_fp,x.body)))
    x isa _Guard && return (:guard,x.op,_native_ir_fp(x.cond),Tuple(map(_native_ir_fp,x.body)))
    x isa _Return && return (:return,x.value===nothing ? nothing : _native_ir_fp(x.value))
    x isa _Break && return (:break,)
    x isa _Continue && return (:continue,)
    x isa _PlaceWrite && return (:write,_native_ir_fp(x.target),x.root,x.owner,x.alias,_native_ir_fp(x.rhs),x.dot)
    x isa _PlaceSwap && return (:swap,Tuple(map(_native_ir_fp,x.targets)))
    x isa _SetReturn && return (:writeret,_native_ir_fp(x.write))
    _native_reject("fingerprint has no raw rule for $(typeof(x))")
end
function _native_raw_program_fingerprint(irs; stats_ir=nothing)
    ms=Tuple((ir.id.decl,ir.id.name,
        Tuple((f.name,f.kind,f.required,f.type,f.default===nothing ? nothing : _native_ir_fp(f.default)) for f in ir.formals),
        Tuple(map(_native_ir_fp,ir.body))) for ir in sort!(collect(irs),by=ir->ir.id.decl))
    (methods=ms,stats=stats_ir===nothing ? nothing :
        (stats_ir.id.decl,stats_ir.id.name,Tuple(map(_native_ir_fp,stats_ir.body))))
end

function _native_encoded_kw_fp(K::Type{<:Tuple},refs,regs)
    Tuple((x.parameters[1],_native_type_fp(x.parameters[2],refs,regs)) for x in K.parameters)
end
function _native_type_fp(T::Type,refs,regs)
    T<:_NNSelf && return (:self,)
    T<:_NNSelfField && return (:selffield,T.parameters[1])
    T<:_NNLocal && return (:local,T.parameters[1])
    T<:_NNFormal && return (:formal,T.parameters...)
    T<:_NNLit && return (:lit,T.parameters[1])
    T<:_NNExt && return (:ext,T.parameters[1])
    T<:_NNIndex && return (:index,_native_type_fp(T.parameters[1],refs,regs),
        Tuple(_native_type_fp(x,refs,regs) for x in T.parameters[2].parameters))
    T<:_NNGetfield && return (:getfield,_native_type_fp(T.parameters[1],refs,regs),T.parameters[2])
    T<:_NNShort && return (:short,T.parameters[1],_native_type_fp(T.parameters[2],refs,regs),_native_type_fp(T.parameters[3],refs,regs))
    T<:_NNIfExpr && return (:ifexpr,(_native_type_fp(x,refs,regs) for x in T.parameters)...)
    T<:_NNBlockExpr && return (:blockexpr,Tuple(_native_type_fp(x,refs,regs) for x in T.parameters[1].parameters),
        _native_type_fp(T.parameters[2],refs,regs))
    T<:_NNTupleExpr && return (:tupleexpr,Tuple(_native_type_fp(x,refs,regs) for x in T.parameters[1].parameters))
    T<:_NNNamedTuple && return (:namedtuple,T.parameters[1],Tuple(_native_type_fp(x,refs,regs) for x in T.parameters[2].parameters))
    if T<:_NNRegistered
        I,Intrinsic,Token,A,Kw,Dot=T.parameters
        ref=_native_ref_fp(refs[I]); reg=_native_reg_fp(regs[I])
        return (:registered,ref,reg,Intrinsic,Tuple(_native_type_fp(x,refs,regs) for x in A.parameters),
                _native_encoded_kw_fp(Kw,refs,regs),Dot)
    end
    T<:_NNFieldCall && return (:fieldcall,T.parameters[1],Tuple(_native_type_fp(x,refs,regs) for x in T.parameters[2].parameters),
        _native_encoded_kw_fp(T.parameters[3],refs,regs),T.parameters[4])
    T<:_NNCallExpr && return (:callexpr,T.parameters[1],T.parameters[2],_native_type_fp(T.parameters[3],refs,regs),
        Tuple(_native_type_fp(x,refs,regs) for x in T.parameters[4].parameters),_native_encoded_kw_fp(T.parameters[5],refs,regs))
    T<:_NNLocalAssign && return (:localassign,T.parameters[1],_native_type_fp(T.parameters[2],refs,regs))
    T<:_NNCall && return (:call,T.parameters[1],T.parameters[2],_native_type_fp(T.parameters[3],refs,regs),
        Tuple(_native_type_fp(x,refs,regs) for x in T.parameters[4].parameters),_native_encoded_kw_fp(T.parameters[5],refs,regs))
    T<:_NNExprStmt && return (:exprstmt,_native_type_fp(T.parameters[1],refs,regs))
    T<:_NNIf && return (:if,_native_type_fp(T.parameters[1],refs,regs),
        Tuple(_native_type_fp(x,refs,regs) for x in T.parameters[2].parameters),Tuple(_native_type_fp(x,refs,regs) for x in T.parameters[3].parameters))
    T<:_NNFor && return (:for,T.parameters[1],_native_type_fp(T.parameters[2],refs,regs),Tuple(_native_type_fp(x,refs,regs) for x in T.parameters[3].parameters))
    T<:_NNWhile && return (:while,_native_type_fp(T.parameters[1],refs,regs),Tuple(_native_type_fp(x,refs,regs) for x in T.parameters[2].parameters))
    T<:_NNGuard && return (:guard,T.parameters[1],_native_type_fp(T.parameters[2],refs,regs),Tuple(_native_type_fp(x,refs,regs) for x in T.parameters[3].parameters))
    T<:_NNReturn && return (:return,T.parameters[1]===Nothing ? nothing : _native_type_fp(T.parameters[1],refs,regs))
    T<:_NNBreak && return (:break,)
    T<:_NNContinue && return (:continue,)
    T<:_NNPlaceWrite && return (:write,_native_type_fp(T.parameters[1],refs,regs),T.parameters[2],T.parameters[3],T.parameters[4],
        _native_type_fp(T.parameters[5],refs,regs),T.parameters[6])
    T<:_NNPlaceSwap && return (:swap,Tuple(_native_type_fp(x,refs,regs) for x in T.parameters[1].parameters))
    T<:_NNWriteReturn && return (:writeret,_native_type_fp(T.parameters[1],refs,regs))
    _native_reject("fingerprint has no encoded rule for $T")
end
function _native_program_fingerprint(::Type{P},refs,regs) where {P<:_NativeProgram}
    q=_native_program_parts(P)
    ms=Tuple(begin
        mid,name,F,B=M.parameters
        (mid,name,Tuple((x.parameters[1],x.parameters[2],x.parameters[3],nothing,
            x.parameters[4]===Nothing ? nothing : _native_type_fp(x.parameters[4],refs,regs)) for x in F.parameters),
            Tuple(_native_type_fp(x,refs,regs) for x in B.parameters))
    end for M in q.methods.parameters)
    st=q.stats<:_NNNoStats ? nothing : begin
        M=q.stats.parameters[1]; mid,name,F,B=M.parameters
        (mid,name,Tuple(_native_type_fp(x,refs,regs) for x in B.parameters))
    end
    (methods=ms,stats=st)
end

function _native_encode_method(ir, b)
    ir.ok || _native_reject("MethodIR $(ir.id.name)#$(ir.id.decl) is not ok: $(ir.reason)")
    _NNMethod{ir.id.decl,ir.id.name,_native_encode_formals(ir,b),
              _native_tuple_type(map(y -> _native_encode(y,b),ir.body))}
end

function _native_encode_program(irs, ::Type{PlanT}, owner_token, owner_mod::Module;
                                root_name::Symbol=:step!, derived=(), stats_ir=nothing,
                                stats_produced=()) where {PlanT}
    ordered = sort!(collect(irs), by = ir -> ir.id.decl)
    mids = Int[ir.id.decl for ir in ordered]
    length(unique(mids)) == length(mids) || _native_reject("duplicate MethodId declaration ordinal")
    roots = [ir for ir in ordered if ir.id.name === root_name]
    length(roots) == 1 || _native_reject("expected exactly one `$root_name` method; found $(length(roots))")
    root = roots[1]
    called = Set{Int}()
    _midwalk(x) = begin
        (x isa _Call || x isa _CallExpr) && push!(called,_native_single_mid(x.name,x.candidates))
        if x isa Tuple || x isa AbstractVector; foreach(_midwalk,x)
        elseif x isa Pair; _midwalk(x.second)
        elseif x isa _MExpr || x isa _MStmt; foreach(f->_midwalk(getfield(x,f)),fieldnames(typeof(x))) end
    end
    foreach(ir -> begin _midwalk(ir.body); foreach(f -> f.default===nothing || _midwalk(f.default),ir.formals) end, ordered)
    missing = sort!(collect(setdiff(called,Set(mids))))
    isempty(missing) || _native_reject("sibling calls reference unencoded MethodIds $missing")
    b = _NativeCalleeBuilder(owner_mod)
    methods = _native_tuple_type(map(ir -> _native_encode_method(ir,b), ordered))
    stats = if stats_ir === nothing
        _NNNoStats
    else
        _validate_stats_body(stats_ir, stats_produced)
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

# -------------------------------------------------------------------------------------------------
# Native emitter.  Every decision below is a pure function of `ProgramT`'s type parameters.  Runtime
# callable authorities are loaded only from the concrete cfg tuple by literal index.

mutable struct _NativeEmitCtx
    program::Type
    plan::Type
    frame::Symbol
    cfg::Symbol
    locals::Dict{Symbol,Symbol}
    kinds::Dict{Symbol,Symbol}
    methods::Dict{Int,Type}
    derived::Tuple
    value_method::Bool
    instrumented::Bool
    scratch::Symbol
end

function _native_call_expr_mids(T::Type,out=Set{Int}())
    T<:_NNCallExpr && push!(out,T.parameters[2])
    T isa DataType || return out
    for p in T.parameters
        p isa Type || continue
        (p<:_NativeNode || p<:Tuple || p<:_NNMethod) && _native_call_expr_mids(p,out)
    end
    out
end
function _native_value_mids(::Type{P}) where {P<:_NativeProgram}
    out=Set{Int}(); foreach(m->_native_call_expr_mids(m,out),_native_program_parts(P).methods.parameters); out
end

_nn_local(n::Symbol) = Symbol("__nn_local_", n)
_nn_formal(n::Symbol) = Symbol("__nn_formal_", n)
_nn_cfg(C::_NativeEmitCtx, f::Symbol) = Expr(:call, GlobalRef(Core, :getfield), C.cfg, QuoteNode(f))
_nn_callee(C::_NativeEmitCtx, i::Int) = Expr(:call, GlobalRef(Core, :getfield),
    _nn_cfg(C, :callees), i)

function _native_program_parts(::Type{<:_NativeProgram{OT,PT,RM,M,D,S}}) where {OT,PT,RM,M,D,S}
    (owner=OT, plan=PT, root=RM, methods=M, derived=D, stats=S)
end
function _native_method_map(::Type{P}) where {P<:_NativeProgram}
    out = Dict{Int,Type}()
    for m in _native_program_parts(P).methods.parameters
        mid = m.parameters[1]
        haskey(out, mid) && _native_reject("duplicate encoded method $mid")
        out[mid] = m
    end
    out
end
function _native_plan_field(::Type{PlanT}, field::Symbol) where {PlanT<:_KernelPlan}
    key = PlanT.parameters[1]
    sig = key[2]
    i = findfirst(t -> t[1] == (field,), sig)
    i === nothing && _native_reject("plan has no named endpoint field `$field`")
    (sig[i][3], sig[i][4])
end
_native_plan_slot(P::Type{<:_KernelPlan}, field::Symbol) = _native_plan_field(P, field)[2]

_nn_tuple_params(T::Type{<:Tuple}) = T.parameters

function _nn_kind(T::Type, C::_NativeEmitCtx)
    if T <: _NNSelf
        :self
    elseif T <: _NNSelfField
        path = T.parameters[1]; f = path[1]
        length(path) > 1 && return f in _EP_SELF ? :canonfield : :tree
        f in _EP_SELF ? :endpoint : f === :trees ? :treevec : f === :proposals ? :epvec : :scalar
    elseif T <: _NNFormal
        get(C.kinds, T.parameters[1], :scalar)
    elseif T <: _NNLocal
        get(C.kinds, T.parameters[1], :scalar)
    elseif T <: _NNIndex
        k = _nn_kind(T.parameters[1], C)
        k === :treevec ? :tree : k === :epvec ? :endpoint : :scalar
    elseif T <: _NNGetfield
        _nn_kind(T.parameters[1], C) === :endpoint ? :canonfield : :tree
    else
        :scalar
    end
end

function _nn_ensure(C::_NativeEmitCtx, ep, field::Symbol)
    ensure = Expr(:call, GlobalRef(Core, :getfield), _nn_cfg(C, :ensures), QuoteNode(field))
    shared = Expr(:call, GlobalRef(Core, :getfield), C.frame, QuoteNode(:shared))
    role, slot = _native_plan_field(C.plan, field)
    endpoint = :__nn_ensure_endpoint
    object = role === :owned ? endpoint : role === :shared ? shared :
        _native_reject("derived field `$field` has unsupported plan role `$role`")
    current = Expr(:call, GlobalRef(@__MODULE__, :_canon_current), object, :(Val($slot)))
    args=Any[endpoint,shared,_nn_cfg(C,:handles)]
    C.instrumented && push!(args,C.scratch)
    repair = Expr(:call, ensure, args...)
    value = Expr(:call, GlobalRef(@__MODULE__, :_canon_slot), object, :(Val($slot)))
    # Most recursive NUTS reads are repeats of a value already repaired in this transaction.  Keep the
    # currentness test in the generated caller so the hot/current path does not cross the separately compiled
    # ensure-function boundary.  The cold/dirty path still invokes the exact existing ensure, preserving its
    # recursive producer ordering, kill/bless protocol, and exception prefix; reread the physical slot only
    # after that call succeeds.
    # The endpoint expression itself is evaluated exactly once, matching the old function-call boundary even
    # for an indexed endpoint.  A lexical `let` makes nested ensures collision-free without a mutable gensym.
    Expr(:let, Expr(:(=), endpoint, ep), Expr(:block, Expr(:if, :(!$current), repair), value))
end

function _nn_self_read(T::Type{<:_NNSelfField}, C::_NativeEmitCtx)
    path = T.parameters[1]; f = path[1]
    fg = Expr(:call, GlobalRef(Core, :getfield), C.frame, QuoteNode(f))
    if length(path) == 1
        f === :stats_f && return (_native_program_parts(C.program).stats <: _NNNoStats ? :(nothing) :
                                  Expr(:call, GlobalRef(@__MODULE__, :nuts_frame_stats), C.frame))
        f === :step_f && return Expr(:call, GlobalRef(@__MODULE__, :nuts_frame_step), C.frame)
        _is_diag(f) && return Expr(:call, GlobalRef(@__MODULE__, :_diag_slot),
                                   Expr(:call, GlobalRef(Core, :getfield), C.frame, QuoteNode(:diag)),
                                   :(Val($(_diag_index(f)))))
        return fg
    end
    f in _EP_SELF || _native_reject("multi-segment self field has non-endpoint base $f")
    length(path) == 2 || _native_reject("endpoint path deeper than two: $path")
    fld = path[2]
    fld in C.derived ? _nn_ensure(C, fg, fld) :
        Expr(:call, GlobalRef(@__MODULE__, :_canon_slot), fg, :(Val($(_native_plan_slot(C.plan,fld)))))
end

_nn_is_end(T::Type) = T <: _NNExt && T.parameters[1] === :end

function _nn_index(T::Type{<:_NNIndex}, C::_NativeEmitCtx)
    B, Is = T.parameters
    base = _nn_val(B, C); it = _nn_tuple_params(Is)
    if !any(_nn_is_end, it)
        return Expr(:ref, base, (_nn_val(i,C) for i in it)...)
    end
    # A lexical `let` gives each index expression a once-evaluated base without any global/gensym counter.
    bs = :__nn_index_base
    parts = Any[]
    for (d,i) in enumerate(it)
        push!(parts, _nn_is_end(i) ? (length(it) == 1 ? :(lastindex($bs)) : :(lastindex($bs,$d))) : _nn_val(i,C))
    end
    Expr(:let, Expr(:(=),bs,base), Expr(:ref,bs,parts...))
end

function _nn_call_expr(callee, args, kws::Type{<:Tuple}, C)
    kwexprs = Any[]
    for K in kws.parameters
        name, V = K.parameters
        push!(kwexprs, Expr(:kw, name, _nn_val(V,C)))
    end
    isempty(kwexprs) ? Expr(:call, callee, args...) :
        Expr(:call, callee, Expr(:parameters,kwexprs...), args...)
end

function _nn_sibling_call(mid::Int, As::Type{<:Tuple}, Kws::Type{<:Tuple}, C::_NativeEmitCtx)
    isempty(Kws.parameters) || _native_reject("native sibling keyword calls are not admitted")
    args = Any[_nn_val(a,C) for a in As.parameters]
    length(args) <= 3 || _native_reject("native sibling call arity $(length(args)) exceeds fixed ABI 3")
    family = C.instrumented ? Symbol("_nni_method",length(args)) : Symbol("_nn_method",length(args))
    prefix = C.instrumented ? Any[C.program,:(Val($mid)),C.cfg,C.frame,C.scratch] :
                              Any[C.program,:(Val($mid)),C.cfg,C.frame]
    Expr(:call, GlobalRef(@__MODULE__, family), prefix..., args...)
end

function _nn_stats(C::_NativeEmitCtx)
    S = _native_program_parts(C.program).stats
    S <: _NNNoStats && return :nothing
    M = S.parameters[1]; body = M.parameters[4]
    xs = Any[_nn_stmt(s,C) for s in body.parameters]
    # `_validate_stats_body` has already proven the final statement is `return __self__`; a field call
    # inlines only the preceding diagnostic writes, exactly like the control emitter.
    pop!(xs)
    C.instrumented ? Expr(:block, xs...,
        Expr(:call,GlobalRef(@__MODULE__,:_nuts_instrument_marker!),C.scratch,
             _nn_cfg(C,:stats_site),:(Val(:diagnostics))), :nothing) :
        Expr(:block, xs..., :nothing)
end

function _nn_val(T::Type, C::_NativeEmitCtx)
    if T <: _NNSelf
        C.frame
    elseif T <: _NNSelfField
        _nn_self_read(T,C)
    elseif T <: _NNLocal
        get(C.locals,T.parameters[1]) do; _native_reject("unbound local $(T.parameters[1])") end
    elseif T <: _NNFormal
        get(C.locals,T.parameters[1]) do; _native_reject("unbound formal $(T.parameters[1])") end
    elseif T <: _NNLit
        QuoteNode(T.parameters[1])
    elseif T <: _NNExt
        T.parameters[1] === :Bool ? GlobalRef(Core,:Bool) :
            _native_reject("indexing end is only valid inside an encoded index")
    elseif T <: _NNIndex
        _nn_index(T,C)
    elseif T <: _NNGetfield
        B,F = T.parameters; be = _nn_val(B,C)
        _nn_kind(B,C) === :endpoint ?
            (F in C.derived ? _nn_ensure(C,be,F) : Expr(:call,GlobalRef(@__MODULE__,:_canon_slot),be,:(Val($(_native_plan_slot(C.plan,F)))))) :
            Expr(:.,be,QuoteNode(F))
    elseif T <: _NNShort
        Op,L,R = T.parameters; Expr(Op,_nn_val(L,C),_nn_val(R,C))
    elseif T <: _NNIfExpr
        X,Y,Z = T.parameters; Expr(:if,_nn_val(X,C),_nn_val(Y,C),_nn_val(Z,C))
    elseif T <: _NNBlockExpr
        S,V = T.parameters; Expr(:block,(_nn_stmt(s,C) for s in S.parameters)...,_nn_val(V,C))
    elseif T <: _NNTupleExpr
        Expr(:tuple,(_nn_val(x,C) for x in T.parameters[1].parameters)...)
    elseif T <: _NNNamedTuple
        names,vals=T.parameters; :(NamedTuple{$names}(($(map(v->_nn_val(v,C),vals.parameters)...),)))
    elseif T <: _NNRegistered
        I,Intrinsic,Token,A,Kw,Dot = T.parameters
        args = Any[_nn_val(a,C) for a in A.parameters]
        if Intrinsic
            Token === Symbol("__rk_intrinsic_copy!!__") || _native_reject("unknown intrinsic $Token")
            length(args)==2 || _native_reject("copy!! encoded arity drift")
            Expr(:call,GlobalRef(@__MODULE__,:_canon_copy_endpoint!),args...)
        else
            _nn_call_expr(_nn_callee(C,I),args,Kw,C)
        end
    elseif T <: _NNFieldCall
        Path,A,Kw,Hint = T.parameters; isempty(Kw.parameters) || _native_reject("field-call keywords unsupported")
        fld = Path[end]
        fld === :step_f ? begin
            length(A.parameters)==1 || _native_reject("step_f requires one endpoint")
            ep=_nn_val(A.parameters[1],C)
            leafargs=Any[ep,Expr(:call,GlobalRef(Core,:getfield),C.frame,QuoteNode(:shared)),
                         _nn_cfg(C,:handles),_nn_cfg(C,:stepkw)]
            if C.instrumented
                push!(leafargs,C.scratch)
                Expr(:block,Expr(:call,_nn_cfg(C,:leaf),leafargs...),
                     Expr(:call,GlobalRef(@__MODULE__,:_nuts_instrument_marker!),C.scratch,
                          _nn_cfg(C,:leaf_site),:(Val(:leaf_body))))
            else
                Expr(:call,_nn_cfg(C,:leaf),leafargs...)
            end
        end : fld === :stats_f ? _nn_stats(C) : _native_reject("unsupported field callable $fld ($Hint)")
    elseif T <: _NNCallExpr
        Name,Mid,Target,A,Kw=T.parameters
        Target <: _NNSelf || _native_reject("non-self sibling value target")
        _nn_sibling_call(Mid,A,Kw,C)
    else
        _native_reject("native value emitter does not support $T")
    end
end

function _nn_dest(T::Type,C::_NativeEmitCtx)
    if T <: _NNGetfield
        B,F=T.parameters; be=_nn_val(B,C)
        _nn_kind(B,C)===:endpoint ? Expr(:call,GlobalRef(@__MODULE__,:_canon_slot),be,:(Val($(_native_plan_slot(C.plan,F))))) :
            Expr(:.,be,QuoteNode(F))
    elseif T <: _NNIndex
        _nn_index(T,C)
    elseif T <: _NNSelfField
        path=T.parameters[1]
        length(path)==1 ? Expr(:.,C.frame,QuoteNode(path[1])) : _nn_self_read(T,C)
    else
        _native_reject("unsupported native destination $T")
    end
end

function _nn_rhs_dot(T::Type,C::_NativeEmitCtx)
    if T <: _NNRegistered
        I,Intrinsic,Token,A,Kw,Dot=T.parameters
        Intrinsic && _native_reject("copy!! cannot occur under broadcast")
        isempty(Kw.parameters) || _native_reject("broadcast keywords unsupported")
        Expr(:call,GlobalRef(Base,:broadcasted),_nn_callee(C,I),(_nn_rhs_dot(a,C) for a in A.parameters)...)
    else
        _nn_val(T,C)
    end
end

function _nn_write(W::Type{<:_NNPlaceWrite},C::_NativeEmitCtx; rhs_override=nothing)
    T,Root,Owner,Alias,R,Dot=W.parameters
    rhs = rhs_override === nothing ? _nn_val(R,C) : rhs_override
    if T <: _NNSelfField && length(T.parameters[1])==1
        f=T.parameters[1][1]
        if _is_diag(f)
            set=Expr(:call,GlobalRef(@__MODULE__,:_diag_set_value!),
                     Expr(:call,GlobalRef(Core,:getfield),C.frame,QuoteNode(:diag)),:(Val($(_diag_index(f)))),rhs)
            return f===:dham ? Expr(:block,set,Expr(:call,GlobalRef(@__MODULE__,:_nuts_produce_diverged!),C.frame)) : set
        end
        return Expr(:(=),Expr(:.,C.frame,QuoteNode(f)),rhs)
    end
    dest=_nn_dest(T,C)
    Dot ? Expr(:call,GlobalRef(Base,:materialize!),dest,_nn_rhs_dot(R,C)) : Expr(:(=),dest,rhs)
end

function _nn_stmt(T::Type,C::_NativeEmitCtx)
    if T <: _NNLocalAssign
        lhs,R=T.parameters; length(lhs)==1 || _native_reject("destructuring local assignment unsupported")
        n=lhs[1]; s=_nn_local(n); C.locals[n]=s; C.kinds[n]=_nn_kind(R,C)
        Expr(:(=),s,_nn_val(R,C))
    elseif T <: _NNCall
        Name,Mid,Target,A,Kw=T.parameters; Target<:_NNSelf || _native_reject("non-self sibling target")
        _nn_sibling_call(Mid,A,Kw,C)
    elseif T <: _NNExprStmt
        _nn_val(T.parameters[1],C)
    elseif T <: _NNIf
        X,Y,Z=T.parameters
        Expr(:if,_nn_val(X,C),Expr(:block,(_nn_stmt(s,C) for s in Y.parameters)...),
             Expr(:block,(_nn_stmt(s,C) for s in Z.parameters)...))
    elseif T <: _NNFor
        V,I,B=T.parameters; s=_nn_local(V); old=get(C.locals,V,nothing); C.locals[V]=s; C.kinds[V]=:scalar
        body=Expr(:block,(_nn_stmt(x,C) for x in B.parameters)...)
        old===nothing ? delete!(C.locals,V) : (C.locals[V]=old)
        Expr(:for,Expr(:(=),s,_nn_val(I,C)),body)
    elseif T <: _NNWhile
        X,B=T.parameters; Expr(:while,_nn_val(X,C),Expr(:block,(_nn_stmt(s,C) for s in B.parameters)...))
    elseif T <: _NNGuard
        Op,X,B=T.parameters; Expr(Op,_nn_val(X,C),Expr(:block,(_nn_stmt(s,C) for s in B.parameters)...))
    elseif T <: _NNReturn
        V=T.parameters[1]
        if C.value_method
            V===Nothing ? Expr(:return) : Expr(:return,_nn_val(V,C))
        elseif V===Nothing
            Expr(:return,C.frame)
        else
            Expr(:block,_nn_val(V,C),Expr(:return,C.frame))
        end
    elseif T <: _NNBreak
        Expr(:break)
    elseif T <: _NNContinue
        Expr(:continue)
    elseif T <: _NNPlaceWrite
        _nn_write(T,C)
    elseif T <: _NNPlaceSwap
        W=T.parameters[1].parameters
        lhs=Any[_nn_dest(w.parameters[1],C) for w in W]
        rhs=Any[_nn_val(w.parameters[5],C) for w in W]
        Expr(:(=),Expr(:tuple,lhs...),Expr(:tuple,rhs...))
    elseif T <: _NNWriteReturn
        W=T.parameters[1]; W.parameters[6] && _native_reject("broadcast set-return unsupported")
        # One lexical temporary makes the contract explicit: evaluate RHS once, write it, then RETURN now.
        tmp=:__nn_writeret_value; rhs=_nn_val(W.parameters[5],C)
        Expr(:let,Expr(:(=),tmp,rhs),Expr(:block,_nn_write(W,C;rhs_override=tmp),
             Expr(:return,C.value_method ? tmp : C.frame)))
    else
        _native_reject("native statement emitter does not support $T")
    end
end

function _nn_generated_body(ProgramT,Mid,argsyms::Tuple; instrumented::Bool=false)
    parts=_native_program_parts(ProgramT); methods=_native_method_map(ProgramT)
    haskey(methods,Mid) || return :(throw(ArgumentError("native NUTS call to unencoded MethodId $Mid")))
    M=methods[Mid]; formals=M.parameters[3]; body=M.parameters[4]
    length(argsyms)<=length(formals.parameters) || return :(throw(MethodError($(Symbol("_nn_method",length(argsyms))), (Val($Mid),))))
    value_method=Mid in _native_value_mids(ProgramT)
    C=_NativeEmitCtx(ProgramT,parts.plan,:frame,:cfg,Dict{Symbol,Symbol}(),Dict{Symbol,Symbol}(),methods,
                     parts.derived,value_method,instrumented,:scratch)
    pro=Any[]
    for (i,F) in enumerate(formals.parameters)
        Name,Kind,Required,Default=F.parameters
        Kind===:pos || return :(throw(ArgumentError("native NUTS formal kind $Kind is unsupported")))
        s=_nn_formal(Name); C.locals[Name]=s; Name===:ep && (C.kinds[Name]=:endpoint)
        if i<=length(argsyms)
            push!(pro,Expr(:(=),s,argsyms[i]))
        elseif Default!==Nothing
            push!(pro,Expr(:(=),s,_nn_val(Default,C)))
        else
            return :(throw(MethodError($(Symbol("_nn_method",length(argsyms))), (Val($Mid),))))
        end
    end
    xs=Any[_nn_stmt(s,C) for s in body.parameters]
    value_method ? Expr(:block,pro...,xs...) : Expr(:block,pro...,xs...,Expr(:return,:frame))
end

@generated _nn_method0(::Type{P},::Val{Mid},cfg,frame) where {P<:_NativeProgram,Mid} =
    _nn_generated_body(P,Mid,())
@generated _nn_method1(::Type{P},::Val{Mid},cfg,frame,a1) where {P<:_NativeProgram,Mid} =
    _nn_generated_body(P,Mid,(:a1,))
@generated _nn_method2(::Type{P},::Val{Mid},cfg,frame,a1,a2) where {P<:_NativeProgram,Mid} =
    _nn_generated_body(P,Mid,(:a1,:a2))
@generated _nn_method3(::Type{P},::Val{Mid},cfg,frame,a1,a2,a3) where {P<:_NativeProgram,Mid} =
    _nn_generated_body(P,Mid,(:a1,:a2,:a3))

@generated _nni_method0(::Type{P},::Val{Mid},cfg,frame,scratch) where {P<:_NativeProgram,Mid} =
    _nn_generated_body(P,Mid,();instrumented=true)
@generated _nni_method1(::Type{P},::Val{Mid},cfg,frame,scratch,a1) where {P<:_NativeProgram,Mid} =
    _nn_generated_body(P,Mid,(:a1,);instrumented=true)
@generated _nni_method2(::Type{P},::Val{Mid},cfg,frame,scratch,a1,a2) where {P<:_NativeProgram,Mid} =
    _nn_generated_body(P,Mid,(:a1,:a2);instrumented=true)
@generated _nni_method3(::Type{P},::Val{Mid},cfg,frame,scratch,a1,a2,a3) where {P<:_NativeProgram,Mid} =
    _nn_generated_body(P,Mid,(:a1,:a2,:a3);instrumented=true)

struct _NutsNoInstrument end
struct _NutsEmissionSite{Real,Instrument} end

mutable struct _NutsEmissionBuilder
    instrumented::Bool
    next_real::Int
    ops::Vector{Any}
end
_NutsEmissionBuilder(instrumented::Bool) = _NutsEmissionBuilder(instrumented, 1, Any[])

function _nuts_emission_site!(b::_NutsEmissionBuilder, kind::Symbol, authority, counter)
    id=b.next_real; b.next_real+=1
    real=_NutsRealOp{id,kind,authority}
    push!(b.ops,real)
    instr=_NutsNoInstrument
    if b.instrumented
        iid=100000+id
        instr=_NutsInstrumentWrite{iid,id,counter}
        # The TYPE tape is appended by the same call that hands the corresponding site to the executable
        # emitter.  Thus the certificate never reconstructs a parallel list after code emission.
        push!(b.ops,instr)
    end
    _NutsEmissionSite{real,instr}()
end
_nuts_emission_ops(b::_NutsEmissionBuilder) = _native_tuple_type(b.ops)

function _native_recipe_manifest(pf::_PreparedFactory)
    owner=kernel_prepared_token(pf); plan=kernel_prepared_plan(pf)
    desc=Any[]
    for (rid,seam,h) in zip(kernel_plan_recipes(plan),kernel_plan_recipe_seam(plan),
                            kernel_prepared_handles(pf))
        rid==seam[1] || _native_reject("recipe manifest order drift")
        push!(desc,_NutsRecipeDescriptor{owner,rid,typeof(recipe_handle_op(h)),recipe_handle_mode(h),
                                         seam[2],seam[3],seam[4]})
    end
    _native_tuple_type(desc)
end

function _native_recipe_hook(builder::_NutsEmissionBuilder, pf::_PreparedFactory, scratch::Symbol)
    recs=kernel_plan_recipes(kernel_prepared_plan(pf)); manifest=_native_recipe_manifest(pf)
    (i,h)->begin
        rid=recs[i]; D=manifest.parameters[i]
        (D.parameters[2] == rid && D.parameters[3] === typeof(recipe_handle_op(h)) &&
         D.parameters[4] === recipe_handle_mode(h)) || _native_reject("recipe emission/manifest drift")
        site=_nuts_emission_site!(builder,:recipe,D,(:recipe,i))
        builder.instrumented ? Expr(:call,GlobalRef(@__MODULE__,:_nuts_recipe_complete!),scratch,
                                    :(Val($i)),QuoteNode(site)) : nothing
    end
end

function _native_leaf_write_hook(builder::_NutsEmissionBuilder,integrator,scratch::Symbol)
    (ordinal,pw,canon)->begin
        D=_NutsLeafWriteDescriptor{integrator,ordinal,canon,pw.dot}
        site=_nuts_emission_site!(builder,:leaf_write,D,(:leaf_write,ordinal))
        builder.instrumented ? Expr(:call,GlobalRef(@__MODULE__,:_nuts_instrument_site!),scratch,
                                    QuoteNode(site)) : nothing
    end
end

struct _CompiledNutsRootNative{ProgramT,RecipeManifest,RootManifest,LeafManifest,EmissionManifest,
                               Refresh,Cfg,H}
    refresh::Refresh
    cfg::Cfg
    handles::H
end

struct _CompiledNutsRootInstrumented{ProgramT,RecipeManifest,RootManifest,LeafManifest,EmissionManifest,
                                     Refresh,Cfg,H,S}
    refresh::Refresh
    cfg::Cfg
    handles::H
    scratch::S
end

mutable struct _NutsInstrumentationScratch{OwnerToken,RootToken,PlanT,ProgramT,EmittedOps,
                                           RecipeManifest,GradIndex,TraceCapacity}
    transitions::Int
    refreshes::Int
    leaf_bodies::Int
    diagnostics::Int
    metric_mutations::Int
    recipe_counts::Vector{Int}
    trace::Vector{Int}
    trace_pairs::Int
    trace_overflows::Int
end
function _nuts_instrumentation_scratch(::Val{OwnerToken},::Val{RootToken},::Type{PlanT},
        ::Type{ProgramT},::Type{EmittedOps},::Type{RecipeManifest},::Val{GradIndex},
        ::Val{TraceCapacity}=Val(256)) where {OwnerToken,RootToken,PlanT,ProgramT,EmittedOps,
                                            RecipeManifest,GradIndex,TraceCapacity}
    iseven(TraceCapacity) && TraceCapacity > 0 || throw(ArgumentError("trace capacity must be positive/even"))
    trace=zeros(Int,TraceCapacity)
    _NutsInstrumentationScratch{OwnerToken,RootToken,PlanT,ProgramT,EmittedOps,RecipeManifest,
                                GradIndex,TraceCapacity}(0,0,0,0,0,zeros(Int,length(RecipeManifest.parameters)),
                                                        trace,0,0)
end

struct _NutsInstrumentationCounts
    transitions::Int
    refreshes::Int
    leaf_bodies::Int
    gradients::Int
    diagnostics::Int
    metric_mutations::Int
    trace_length::Int
    trace_overflows::Int
end

@inline function _nuts_trace_site!(s::_NutsInstrumentationScratch,
        ::_NutsEmissionSite{R,I}) where {RealId,Kind,Authority,InstrumentId,Counter,
        R<:_NutsRealOp{RealId,Kind,Authority},
        I<:_NutsInstrumentWrite{InstrumentId,RealId,Counter}}
    cap=length(getfield(s,:trace)); pair=getfield(s,:trace_pairs)
    off=2*(pair % (cap>>>1))
    @inbounds begin
        getfield(s,:trace)[off+1]=RealId
        getfield(s,:trace)[off+2]=InstrumentId
    end
    setfield!(s,:trace_pairs,pair+1)
    pair >= (cap>>>1) && setfield!(s,:trace_overflows,getfield(s,:trace_overflows)+1)
    nothing
end

@inline function _nuts_instrument_marker!(s::_NutsInstrumentationScratch,site::_NutsEmissionSite,
        ::Val{Kind}) where {Kind}
    Kind === :transition ? setfield!(s,:transitions,getfield(s,:transitions)+1) :
    Kind === :refresh ? setfield!(s,:refreshes,getfield(s,:refreshes)+1) :
    Kind === :leaf_body ? setfield!(s,:leaf_bodies,getfield(s,:leaf_bodies)+1) :
    Kind === :diagnostics ? setfield!(s,:diagnostics,getfield(s,:diagnostics)+1) :
    throw(ArgumentError("unsupported NUTS instrumentation marker `$Kind`"))
    _nuts_trace_site!(s,site)
end

@inline function _nuts_instrument_site!(s::_NutsInstrumentationScratch,site::_NutsEmissionSite)
    _nuts_trace_site!(s,site)
end

@inline function _nuts_recipe_complete!(s::_NutsInstrumentationScratch,::Val{I},
        site::_NutsEmissionSite) where {I}
    @inbounds getfield(s,:recipe_counts)[I]+=1
    _nuts_trace_site!(s,site)
end

# Compiler-owned probes use the identical prepared-handle emitter as the public leaf/root, but are not
# members of the public transition's lexical op tape.  They therefore update the complete recipe counter
# bank without appending a fictitious public-root site to the trace.
@inline function _nuts_recipe_probe_complete!(s::_NutsInstrumentationScratch,::Val{I}) where {I}
    @inbounds getfield(s,:recipe_counts)[I]+=1
    nothing
end

function _nuts_instrumentation_counts(s::_NutsInstrumentationScratch{OT,RT,PT,PG,Ops,RM,GI}) where {
        OT,RT,PT,PG,Ops,RM,GI}
    grad=@inbounds getfield(s,:recipe_counts)[GI]
    _NutsInstrumentationCounts(s.transitions,s.refreshes,s.leaf_bodies,grad,s.diagnostics,
                               s.metric_mutations,min(2*s.trace_pairs,length(s.trace)),s.trace_overflows)
end

function _nuts_validate_emitted_ops(mode::Symbol,Ops::Type{<:Tuple})
    ts=Ops.parameters
    # Every compiled NUTS root necessarily has root-entry, refresh, and transition sites.  Empty is not a
    # degenerate valid program; accepting it would let a fabricated certificate vacuously satisfy G15.
    isempty(ts) && return false
    all(t->t isa DataType && t<:_NutsEmittedOp,ts) || return false
    ids=Int[t.parameters[1] for t in ts]
    length(ids)==length(unique(ids)) || return false
    realids=Set{Int}(t.parameters[1] for t in ts if t<:_NutsRealOp)
    instrids=Set{Int}(t.parameters[1] for t in ts if t<:_NutsInstrumentWrite)
    isempty(intersect(realids,instrids)) || return false
    mode === :production && return isempty(instrids)
    mode === :instrumented || return false
    # Instrumented emission is TOTAL, not merely locally well-formed: every real site has exactly one
    # immediately-following write, and every write belongs to exactly that predecessor.  Without the
    # cardinality/alternation rule `Real,Write,Real` falsely validated while silently omitting one counter.
    iseven(length(ts)) || return false
    length(realids)==length(instrids)==(length(ts)>>>1) || return false
    for i in 1:2:length(ts)
        real=ts[i]; write=ts[i+1]
        real<:_NutsRealOp || return false
        write<:_NutsInstrumentWrite || return false
        write.parameters[2] == real.parameters[1] || return false
        counter=write.parameters[3]
        ((counter === real.parameters[2]) ||
         (counter === :leaf_body && real.parameters[2] === :inlined) ||
         (counter isa Tuple && !isempty(counter) && counter[1] === real.parameters[2])) || return false
    end
    true
end

_nuts_real_op_signature(Ops::Type{<:Tuple}) = Tuple((t.parameters[1],t.parameters[2],t.parameters[3])
    for t in Ops.parameters if t<:_NutsRealOp)

_native_root_program(::Type{<:_CompiledNutsRootNative{ProgramT}}) where {ProgramT} = ProgramT
_native_root_program(::Type{<:_CompiledNutsRootInstrumented{ProgramT}}) where {ProgramT} = ProgramT
function _native_root_manifests(::Type{<:_CompiledNutsRootNative{P,RM,RootM,LeafM,Ops}}) where {
        P,RM,RootM,LeafM,Ops}; (RM,RootM,LeafM,Ops); end
function _native_root_manifests(::Type{<:_CompiledNutsRootInstrumented{P,RM,RootM,LeafM,Ops}}) where {
        P,RM,RootM,LeafM,Ops}; (RM,RootM,LeafM,Ops); end

# Construct the zero-field seal only after the REAL native root, scratch, and frame exist.  All structural
# facts come from the compiler-owned Plan/Program types; the caller supplies no evidence tuple. Production and
# instrumented modes accept only their corresponding concrete root families.
function _native_nuts_certificate(::Val{Mode}, pf::_PreparedFactory, skel,
        ::Val{RootToken}, root, scratch, frame::_NutsFrame) where {Mode,RootToken}
    Mode in (:production,:instrumented) || throw(ArgumentError(
        "unsupported native NUTS certificate mode `$Mode`"))
    ProgramT=_native_root_program(typeof(root))
    Mode === :production && !(root isa _CompiledNutsRootNative) && throw(ArgumentError(
        "production certificate requires the production native root"))
    Mode === :instrumented && !(root isa _CompiledNutsRootInstrumented) && throw(ArgumentError(
        "instrumented certificate requires the instrumented native root"))
    parts = _native_program_parts(ProgramT)
    OwnerToken = kernel_token(skel)
    parts.owner === OwnerToken || throw(ArgumentError("native certificate owner/program mismatch"))
    plan = kernel_prepared_plan(pf)
    PlanT = typeof(plan)
    parts.plan === PlanT || throw(ArgumentError("native certificate plan/program mismatch"))
    PlanKey = kernel_plan_key(plan)
    # The integrator authority is the frame's validated prepared callable/binder, not a PlanKey tuple
    # convention.  PlanT/PlanKey separately retain the endpoint-plan definition authority.
    Integrator = prepared_callable_token(nuts_frame_step(frame))
    SelectedRecipes = PlanKey[5]
    # Full selected physical role signature, including paths and aliases.  This is intentionally redundant
    # with PlanKey: a gate can census roles without interpreting private Key tuple positions, while exact Key
    # identity remains separately present.
    Roles = Tuple((t[1], t[2], t[3], t[4]) for t in PlanKey[2])
    ControlFingerprint = _NutsControlFingerprint{ProgramT,parts.root,
                                                 _native_program_node_count(ProgramT)}
    RecipeManifest,RootManifest,LeafManifest,EmittedOps=_native_root_manifests(typeof(root))
    RecipeManifest === _native_recipe_manifest(pf) || throw(ArgumentError(
        "native root recipe manifest is detached from the prepared handles"))
    _nuts_validate_emitted_ops(Mode,EmittedOps) || throw(ArgumentError(
        "compiler produced an invalid native emitted-op stream"))
    Cert = _NutsCertificate{Mode,OwnerToken,RootToken,PlanT,PlanKey,ProgramT,
        ControlFingerprint,RecipeManifest,RootManifest,LeafManifest,EmittedOps,SelectedRecipes,Roles,
        Integrator,typeof(root),typeof(scratch),typeof(frame)}
    Cert()
end
@inline function (r::_CompiledNutsRootNative{P})(fr,::Tuple{},rng) where {P}
    _diagnostics_reset!(getfield(fr,:diag)); _nuts_invalidate_diverged!(fr)
    try
        r.refresh(getfield(fr,:init),getfield(fr,:shared),r.handles,rng)
        _nn_method1(P,Val(_native_program_parts(P).root),r.cfg,fr,rng)
        _diagnostics_root_commit!(getfield(fr,:diag)); _nuts_derived_root_commit!(fr)
    catch
        setfield!(getfield(fr,:diag),:pending,UInt(0)); _nuts_invalidate_diverged!(fr); rethrow()
    end
    fr
end

@inline function (r::_CompiledNutsRootInstrumented{P})(fr,
        sc::_NutsInstrumentationScratch,rng) where {P}
    sc === getfield(r,:scratch) || throw(ArgumentError(
        "instrumented root requires its compiler-owned scratch authority"))
    _diagnostics_reset!(getfield(fr,:diag))
    _nuts_instrument_site!(sc,getfield(r.cfg,:diag_reset_site))
    _nuts_invalidate_diverged!(fr)
    _nuts_instrument_site!(sc,getfield(r.cfg,:invalidate_entry_site))
    try
        r.refresh(getfield(fr,:init),getfield(fr,:shared),r.handles,rng)
        _nuts_instrument_marker!(sc,getfield(r.cfg,:refresh_site),Val(:refresh))
        _nni_method1(P,Val(_native_program_parts(P).root),r.cfg,fr,sc,rng)
        _diagnostics_root_commit!(getfield(fr,:diag))
        _nuts_instrument_site!(sc,getfield(r.cfg,:diag_commit_site))
        _nuts_derived_root_commit!(fr)
        _nuts_instrument_site!(sc,getfield(r.cfg,:derived_commit_site))
        _nuts_instrument_marker!(sc,getfield(r.cfg,:transition_site),Val(:transition))
    catch
        setfield!(getfield(fr,:diag),:pending,UInt(0))
        _nuts_instrument_site!(sc,getfield(r.cfg,:catch_pending_site))
        _nuts_invalidate_diverged!(fr)
        _nuts_instrument_site!(sc,getfield(r.cfg,:catch_invalidate_site))
        rethrow()
    end
    fr
end

# Compile one demanded ensure through the same recipe-completion seam as the leaf.  `builder` is cold,
# per-compilation scratch; its type tape is frozen into the root/certificate before the sampler escapes.
function _compile_native_ensure(pf::_PreparedFactory,::Type{OW},::Type{SH},field::Symbol,
        builder::_NutsEmissionBuilder) where {OW,SH}
    plan=kernel_prepared_plan(pf); hs=kernel_prepared_handles(pf); fc=_lf_canon_map(plan)
    producer=Dict{Int,Int}(c=>r for (c,r) in kernel_plan_producer(plan))
    recs=kernel_plan_recipes(plan)
    hidx=Dict{Int,Tuple{Any,Int}}(recs[i]=>(hs[i],i) for i in eachindex(hs))
    haskey(fc,field) || _native_reject("demanded ensure has no canon for `$field`")
    c=fc[field]; stmts=Any[]; current=Set{Int}(); stale=Set{Int}()
    hook=_native_recipe_hook(builder,pf,:scratch)
    _lf_ensure!(stmts,c,current,stale,plan,producer,hidx,OW,SH;recipe_hook=hook)
    ret=_pp_read(plan,c)
    builder.instrumented ?
        compile(:((owned,shared,handles,scratch)->$(Expr(:block,stmts...,:(return $ret))))) :
        compile(:((owned,shared,handles)->$(Expr(:block,stmts...,:(return $ret)))))
end

function _native_demanded_fields(pf,skel)
    deriv=derived_fields(pf); demanded=Set{Symbol}()
    scan(x)=begin
        if x isa _Getfield && x.field in deriv
            push!(demanded,x.field)
        elseif x isa _SelfField && length(x.path)==2 && x.path[1] in _EP_SELF && x.path[2] in deriv
            push!(demanded,x.path[2])
        end
        if x isa Tuple || x isa AbstractVector
            foreach(scan,x)
        elseif x isa Pair
            scan(x.second)
        elseif x isa _MExpr || x isa _MStmt
            foreach(f->scan(getfield(x,f)),fieldnames(typeof(x)))
        end
    end
    foreach(ir->scan(ir.body),method_irs(skel))
    sort!(collect(demanded))
end

function _compile_native_metric_update(pf::_PreparedFactory,::Type{OW},::Type{SH},
        ::Val{Mode}) where {OW,SH,Mode}
    Mode in (:production,:instrumented) || _native_reject("unsupported metric-probe mode $Mode")
    plan=kernel_prepared_plan(pf); hs=kernel_prepared_handles(pf); fc=_lf_canon_map(plan)
    metric=get(fc,:metric,nothing); metric===nothing && _native_reject("plan has no metric canon")
    affected=Set{Int}((metric,_lf_kill_closure(plan,metric)...))
    stmts=Any[]
    for c in sort!(collect(affected)); _lf_mask!(stmts,plan,c,:kill); end
    push!(stmts,:(copyto!($(_pp_read(plan,metric)),new_metric)))
    _lf_mask!(stmts,plan,metric,:bless)
    hook=Mode===:instrumented ?
        ((i,h)->Expr(:call,GlobalRef(@__MODULE__,:_nuts_recipe_probe_complete!),:scratch,:(Val($i)))) :
        nothing
    # Only shared-authority producers execute during the metric mutation itself.  Owned kinetic closure is
    # intentionally left dirty and is repaired by the real phasepoint read, matching the Plan invalidation.
    for (i,h) in enumerate(hs)
        outs=collect(h.owned)
        isempty(outs) && continue
        all(c->kernel_plan_field(plan,c)[1]===:shared,outs) || continue
        any(in(affected),outs) || continue
        _pp_emit_handle!(stmts,plan,h,i,OW,SH;recipe_hook=hook)
    end
    args=Mode===:instrumented ? :((owned,shared,handles,scratch,new_metric)) :
                                :((owned,shared,handles,new_metric))
    compile(:($args -> $(Expr(:block,stmts...,:(return owned)))))
end

struct _NutsRefreshBodyMarker{RefreshT} end
struct _NutsLeafBodyMarker{ProgramT} end

function _native_node_manifest(pf::_PreparedFactory,RM::Type{<:Tuple})
    plan=kernel_prepared_plan(pf)
    nodes=Any[]
    for (i,D) in enumerate(RM.parameters)
        # A compiler node is a selected recipe whose producer-owned result is shared and whose inputs are
        # shared.  This identifies the fixture's declared logdet node structurally, without operator names.
        ins,owned=D.parameters[5],D.parameters[7]
        !isempty(owned) || continue
        all(c->kernel_plan_field(plan,c)[1]===:shared,ins) || continue
        all(c->kernel_plan_field(plan,c)[1]===:shared,owned) || continue
        push!(nodes,D)
    end
    _native_tuple_type(nodes)
end

function _native_manifest_slice(ops::Vector{Any},lo::Int,hi::Int)
    hi < lo ? Tuple{} : _native_tuple_type(ops[lo:hi])
end

function _compile_native_components(pf::_PreparedFactory,skel,refresh_skel,frame::_NutsFrame,
        base,::Type{ProgramT},::Val{Mode}) where {ProgramT<:_NativeProgram,Mode}
    instrumented=Mode===:instrumented
    Mode in (:production,:instrumented) || _native_reject("unsupported native component mode $Mode")
    b=_NutsEmissionBuilder(instrumented)
    RM=_native_recipe_manifest(pf)
    diag_reset_site=_nuts_emission_site!(b,:root_write,(base.RootToken,:diagnostics_reset),
                                         (:root_write,:diagnostics_reset))
    invalidate_entry_site=_nuts_emission_site!(b,:root_write,(base.RootToken,:derived_invalidate),
                                                (:root_write,:derived_invalidate))
    refresh_site=_nuts_emission_site!(b,:refresh,typeof(base.refresh),:refresh)
    EPT=typeof(getfield(frame,:fwd)); SH=typeof(getfield(frame,:shared))
    fields=_native_demanded_fields(pf,skel)
    ensures=NamedTuple{Tuple(fields)}(Tuple(_compile_native_ensure(pf,EPT,SH,f,b) for f in fields))
    leaf_start=length(b.ops)+1
    leaf_ir,stepkw=prepared_callable_leaf(nuts_frame_step(frame))
    hook=_native_recipe_hook(b,pf,:__lf_instrumentation)
    write_hook=_native_leaf_write_hook(b,prepared_callable_token(nuts_frame_step(frame)),
                                       :__lf_instrumentation)
    leaf=instrumented ? compile_leapfrog_instrumented(pf,EPT,SH,leaf_ir;
                                                       recipe_hook=hook,write_hook=write_hook) :
                        _compile_leapfrog_native(pf,EPT,SH,leaf_ir,false;
                                                 recipe_hook=hook,write_hook=write_hook)
    metric_update=_compile_native_metric_update(pf,EPT,SH,Val(Mode))
    LeafMarker=_NutsLeafBodyMarker{ProgramT}
    leaf_site=_nuts_emission_site!(b,:inlined,LeafMarker,:leaf_body)
    stats=stats_binding_token(nuts_frame_stats(frame))
    stats_site=stats===nothing ? _NutsEmissionSite{_NutsRealOp{0,:none,Nothing},_NutsNoInstrument}() :
        _nuts_emission_site!(b,:diagnostics,stats,:diagnostics)
    leaf_end=length(b.ops)
    diag_commit_site=_nuts_emission_site!(b,:root_write,(base.RootToken,:diagnostics_commit),
                                          (:root_write,:diagnostics_commit))
    derived_commit_site=_nuts_emission_site!(b,:root_write,(base.RootToken,:derived_commit),
                                             (:root_write,:derived_commit))
    transition_site=_nuts_emission_site!(b,:transition,base.RootToken,:transition)
    catch_pending_site=_nuts_emission_site!(b,:root_write,(base.RootToken,:catch_pending_clear),
                                            (:root_write,:catch_pending_clear))
    catch_invalidate_site=_nuts_emission_site!(b,:root_write,(base.RootToken,:catch_derived_invalidate),
                                               (:root_write,:catch_derived_invalidate))
    Ops=_nuts_emission_ops(b)
    root_types=Any[b.ops[i] for i in eachindex(b.ops) if i<leaf_start || i>leaf_end]
    root_ops=_native_tuple_type(root_types)
    leaf_ops=_native_manifest_slice(b.ops,leaf_start,leaf_end)
    nodes=_native_node_manifest(pf,RM)
    RootM=_NutsRootManifest{kernel_token(refresh_skel),_NutsRefreshBodyMarker{typeof(base.refresh)},root_ops}
    LeafM=_NutsLeafManifest{prepared_callable_token(nuts_frame_step(frame)),:step_f,
                            LeafMarker,leaf_ops,nodes}
    cfg=(leaf=leaf,handles=base.cfg.handles,stepkw=stepkw,ensures=ensures,
         callees=base.cfg.callees,callee_refs=base.cfg.callee_refs,
         callee_registrations=base.cfg.callee_registrations,
         metric_update=metric_update,
         diag_reset_site=diag_reset_site,invalidate_entry_site=invalidate_entry_site,
         refresh_site=refresh_site,leaf_site=leaf_site,stats_site=stats_site,
         diag_commit_site=diag_commit_site,derived_commit_site=derived_commit_site,
         transition_site=transition_site,catch_pending_site=catch_pending_site,
         catch_invalidate_site=catch_invalidate_site)
    (;cfg,RM,RootM,LeafM,Ops)
end

# Cold native entry using the existing validated compiler solely to construct the captured
# leaf/ensure/refresh authorities.  The hot root and recursive program are wholly native and registry-free;
# the later immutable state-threading ABI can reuse the same ProgramT and compiler-owned manifests.
function compile_nuts_native(pf::_PreparedFactory,skel,refresh_skel,nuts_root_skel,frame::_NutsFrame;
                             root_name::Symbol=:step!)
    base=compile_nuts(pf,skel,refresh_skel,nuts_root_skel,frame;root_name=root_name)
    irs=method_irs(skel); binding=nuts_frame_stats(frame); stats_ir=nothing
    if stats_binding_registration(binding)!==nothing
        stats_ir=only(method_irs(stats_binding_source(binding)))
        _validate_stats_body(stats_ir,stats_binding_produced(binding))
    end
    produced=stats_ir===nothing ? () : stats_binding_produced(binding)
    E=_native_encode_program(irs,typeof(kernel_prepared_plan(pf)),kernel_token(skel),kernel_module(skel);
                             root_name=root_name,derived=derived_fields(pf),stats_ir=stats_ir,
                             stats_produced=produced)
    authority=(refresh=base.refresh,RootToken=base.RootToken,
        cfg=(handles=base.cfg.handles,callees=E.callees,callee_refs=E.refs,
             callee_registrations=E.registrations))
    C=_compile_native_components(pf,skel,refresh_skel,frame,authority,E.program,Val(:production))
    cfg=C.cfg
    H=kernel_prepared_handles(pf)
    root=_CompiledNutsRootNative{E.program,C.RM,C.RootM,C.LeafM,C.Ops,
                                 typeof(base.refresh),typeof(cfg),typeof(H)}(base.refresh,cfg,H)
    certificate=_native_nuts_certificate(Val(:production),pf,skel,Val(base.RootToken),root,(),frame)
    (root! = root,scratch=(),RootToken=base.RootToken,cfg=cfg,refresh=base.refresh,program=E.program,
     refs=E.refs,registrations=E.registrations,control=base,certificate=certificate)
end


"""Compile the genuine instrumented sibling of the production native root.

The ProgramT and all real operations are shared structurally with `compile_nuts_native`, while the leaf,
native method family, root, certificate, and scratch are independently constructed.  Instrument writes
touch only `_NutsInstrumentationScratch`; none are present in the production root or cfg.
"""
function compile_nuts_native_instrumented(pf::_PreparedFactory,skel,refresh_skel,nuts_root_skel,
        frame::_NutsFrame;root_name::Symbol=:step!)
    prod=compile_nuts_native(pf,skel,refresh_skel,nuts_root_skel,frame;root_name=root_name)
    authority=(refresh=prod.refresh,RootToken=prod.RootToken,cfg=prod.cfg)
    C=_compile_native_components(pf,skel,refresh_skel,frame,authority,prod.program,Val(:instrumented))
    cfg=C.cfg
    H=kernel_prepared_handles(pf); P=prod.program
    recs=kernel_plan_recipes(kernel_prepared_plan(pf))
    gradidx=findfirst(==(kernel_prepared_grad_recipe(pf)),recs)
    gradidx===nothing && _native_reject("prepared gradient recipe is absent from selected recipe order")
    scratch=_nuts_instrumentation_scratch(Val(kernel_token(skel)),Val(prod.RootToken),
                                          typeof(kernel_prepared_plan(pf)),P,C.Ops,C.RM,Val(gradidx))
    root=_CompiledNutsRootInstrumented{P,C.RM,C.RootM,C.LeafM,C.Ops,
                                       typeof(prod.refresh),typeof(cfg),typeof(H),typeof(scratch)}(
                                       prod.refresh,cfg,H,scratch)
    certificate=_native_nuts_certificate(Val(:instrumented),pf,skel,Val(prod.RootToken),
                                         root,scratch,frame)
    _nuts_certificate_parts(certificate).emitted_ops === C.Ops || throw(ArgumentError(
        "instrumented scratch/certificate emitted-op mismatch"))
    (root! = root,scratch=scratch,RootToken=prod.RootToken,cfg=cfg,refresh=prod.refresh,
     program=P,refs=prod.refs,registrations=prod.registrations,control=prod.control,
     certificate=certificate,production=prod)
end

function _nuts_instrumented_scratch(k::KernelObject)
    h=_nuts_sealed_handles(k); p=_nuts_certificate_parts(getfield(h,:certificate))
    p.mode === :instrumented || throw(ArgumentError("sampler is not the sealed instrumented sibling"))
    sc=getfield(h,:scratch)
    sc isa _NutsInstrumentationScratch || throw(ArgumentError(
        "instrumented certificate is detached from compiler-owned instrumentation scratch"))
    q=typeof(sc).parameters
    (q[1]===p.owner && q[2]===p.root_token && q[3]===p.plan && q[4]===p.program &&
     q[5]===p.emitted_ops && q[6]===p.recipe_manifest) || throw(ArgumentError(
        "instrumented scratch token/plan/program/op-stream provenance mismatch"))
    sc
end

nuts_sealed_op_stream(k::KernelObject) =
    _nuts_certificate_parts(nuts_sealed_certificate(k)).emitted_ops
nuts_instrumented_counts(k::KernelObject) =
    _nuts_instrumentation_counts(_nuts_instrumented_scratch(k))
function nuts_instrumented_recompute(k::KernelObject)
    sc=_nuts_instrumented_scratch(k)
    RM=_nuts_certificate_parts(nuts_sealed_certificate(k)).recipe_manifest
    Tuple((owner=D.parameters[1],recipe=D.parameters[2],count=sc.recipe_counts[i])
          for (i,D) in enumerate(RM.parameters))
end
function nuts_instrumented_trace(k::KernelObject)
    sc=_nuts_instrumented_scratch(k); cap=length(sc.trace); np=min(sc.trace_pairs,cap>>>1)
    np==0 && return ()
    firstpair=sc.trace_pairs<=cap>>>1 ? 0 : sc.trace_pairs % (cap>>>1)
    Tuple(sc.trace[2*((firstpair+j)%(cap>>>1))+d] for j in 0:np-1 for d in 1:2)
end
function _nuts_schedule_op(T::Type{<:_NutsRealOp},refresh_marker=Nothing)
    _,kind,authority=T.parameters
    if kind === :recipe
        D=authority
        return (kind=:recipe,id=D.parameters[2],owner=D.parameters[1],residual=false)
    elseif kind === :refresh && refresh_marker !== Nothing
        return (kind=:inlined,id=refresh_marker,owner=nothing,residual=false)
    elseif kind === :inlined
        return (kind=:inlined,id=authority,owner=nothing,residual=false)
    end
    (kind=kind,id=authority,owner=nothing,residual=false)
end
_nuts_schedule_ops(T::Type{<:Tuple},refresh_marker=Nothing) =
    Tuple(_nuts_schedule_op(x,refresh_marker) for x in T.parameters if x<:_NutsRealOp)

function _nuts_recipe_roles(p)
    RM=p.recipe_manifest
    gradidx=findfirst(D->D.parameters[4]===:destination,RM.parameters)
    cholidx=findfirst(D->D.parameters[3]===typeof(cholesky),RM.parameters)
    logidx=findfirst(D->D.parameters[3]===typeof(logdet),RM.parameters)
    any(isnothing,(gradidx,cholidx,logidx)) && throw(ArgumentError(
        "instrumented recipe manifest lacks grad/cholesky/logdet roles"))
    role(i)=(RM.parameters[i].parameters[1],RM.parameters[i].parameters[2])
    (chol_role=role(cholidx),logdet_role=role(logidx),grad_role=role(gradidx))
end

function nuts_instrumented_schedule(k::KernelObject)
    p=_nuts_certificate_parts(nuts_sealed_certificate(k)); _nuts_instrumented_scratch(k)
    rootp=p.root_manifest.parameters; leafp=p.leaf_manifest.parameters; rr=_nuts_recipe_roles(p)
    plan=(selected_recipe_keys=Tuple((D.parameters[1],D.parameters[2])
                                    for D in p.recipe_manifest.parameters),rr...)
    integrator=(leapfrog_token=p.integrator,stepf_slot=leafp[2],refresh_token=rootp[1],
                body_marker=leafp[3],refresh_body_marker=rootp[2])
    (plan=plan,integrator=integrator,root_ops=_nuts_schedule_ops(rootp[3],rootp[2]),
     leaf_ops=_nuts_schedule_ops(leafp[4]),
     nodes=Tuple((owner=D.parameters[1],id=D.parameters[2]) for D in leafp[5].parameters),
     recompute=nuts_instrumented_recompute(k))
end

function nuts_instrumented_mutate_metric!(k::KernelObject,new_metric)
    sc=_nuts_instrumented_scratch(k); h=_nuts_sealed_handles(k); root=getfield(h,:root)
    frame=getfield(h,:frame); cfg=getfield(root,:cfg)
    typeof(new_metric) === typeof(nuts_sealed_metric(k)) || throw(ArgumentError(
        "metric probe must preserve the sealed metric representation"))
    getfield(cfg,:metric_update)(getfield(frame,:init),getfield(frame,:shared),
        getfield(root,:handles),sc,new_metric)
    setfield!(sc,:metric_mutations,getfield(sc,:metric_mutations)+1)
    nuts_sealed_metric(k)
end

function _nuts_native_slot(p,ep,name::Symbol)
    _canon_slot(ep,Val(_native_plan_slot(p.plan,name)))
end
function _nuts_instrumented_phasepoint(k::KernelObject,ep)
    p=_nuts_certificate_parts(nuts_sealed_certificate(k)); sc=_nuts_instrumented_scratch(k)
    root=nuts_sealed_root(k); cfg=getfield(root,:cfg); frame=nuts_sealed_frame(k)
    ens=getfield(cfg,:ensures)
    getfield(ens,:dham_dmom)(ep,getfield(frame,:shared),getfield(root,:handles),sc)
    getfield(ens,:ham)(ep,getfield(frame,:shared),getfield(root,:handles),sc)
    (pos=_nuts_native_slot(p,ep,:pos),mom=_nuts_native_slot(p,ep,:mom),
     pot=_nuts_native_slot(p,ep,:pot),dpot_dpos=_nuts_native_slot(p,ep,:dpot_dpos),
     dkin_dmom=_nuts_native_slot(p,ep,:dkin_dmom),kin=_nuts_native_slot(p,ep,:kin),
     ham=_nuts_native_slot(p,ep,:ham),dham_dmom=_nuts_native_slot(p,ep,:dham_dmom))
end
nuts_instrumented_phasepoint(k::KernelObject) =
    _nuts_instrumented_phasepoint(k,getfield(nuts_sealed_frame(k),:init))

function nuts_instrumented_leaf_probe!(k::KernelObject,pos,mom)
    sc=_nuts_instrumented_scratch(k); root=nuts_sealed_root(k); frame=nuts_sealed_frame(k)
    src=getfield(frame,:init); ep=getfield(frame,:fwd)
    isequal(pos,_nuts_native_slot(_nuts_certificate_parts(nuts_sealed_certificate(k)),src,:pos)) ||
        throw(ArgumentError("leaf probe position must come from the sealed phasepoint"))
    isequal(mom,_nuts_native_slot(_nuts_certificate_parts(nuts_sealed_certificate(k)),src,:mom)) ||
        throw(ArgumentError("leaf probe momentum must come from the sealed phasepoint"))
    _canon_copy_endpoint!(ep,src)
    cfg=getfield(root,:cfg)
    getfield(cfg,:leaf)(ep,getfield(frame,:shared),getfield(root,:handles),getfield(cfg,:stepkw),sc)
    _nuts_instrument_marker!(sc,getfield(cfg,:leaf_site),Val(:leaf_body))
    _nuts_instrumented_phasepoint(k,ep)
end

function nuts_instrumentation_equivalent(production::KernelObject,instrumented::KernelObject)
    hp=_nuts_sealed_handles(production); hi=_nuts_sealed_handles(instrumented)
    cp=_nuts_certificate_parts(getfield(hp,:certificate)); ci=_nuts_certificate_parts(getfield(hi,:certificate))
    cp.mode === :production && ci.mode === :instrumented || return false
    getfield(hp,:root) !== getfield(hi,:root) || return false
    getfield(hp,:frame) !== getfield(hi,:frame) || return false
    _nuts_instrumented_scratch(instrumented)
    for f in (:owner,:root_token,:plan,:plan_key,:program,:control,:recipe_manifest,:recipes,:roles,:integrator)
        getfield(cp,f) === getfield(ci,f) || return false
    end
    _nuts_real_op_signature(cp.emitted_ops) === _nuts_real_op_signature(ci.emitted_ops) || return false
    _nuts_validate_emitted_ops(:production,cp.emitted_ops) || return false
    _nuts_validate_emitted_ops(:instrumented,ci.emitted_ops) || return false
    true
end

function _build_nuts_instrumented_sampler(pf::_PreparedFactory,endpoint_values,nuts_state_skel,
        refresh_skel::_Mode2KernelSkeleton,nuts_skel::_Mode2KernelSkeleton;
        step_f,max_depth::Int,min_dham,stats_f)
    frame=_prepare_nuts_frame(pf,endpoint_values,max_depth;step_f,stats_f,min_dham)
    C=compile_nuts_native_instrumented(pf,nuts_state_skel,refresh_skel,nuts_skel,frame)
    _sealed_nuts_sampler(Val(kernel_token(nuts_state_skel)),Val(C.RootToken),frame,
                         C.root!,C.scratch,C.certificate)
end
