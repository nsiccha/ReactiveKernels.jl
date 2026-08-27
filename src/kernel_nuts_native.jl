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
function _native_plan_slot(::Type{PlanT}, field::Symbol) where {PlanT<:_KernelPlan}
    key = PlanT.parameters[1]
    sig = key[2]
    i = findfirst(t -> t[1] == (field,), sig)
    i === nothing && _native_reject("plan has no named endpoint field `$field`")
    sig[i][4]
end

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
    Expr(:call, ensure, ep, shared, _nn_cfg(C, :handles))
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
    Expr(:call, GlobalRef(@__MODULE__, Symbol("_nn_method",length(args))), C.program, :(Val($mid)),
         C.cfg, C.frame, args...)
end

function _nn_stats(C::_NativeEmitCtx)
    S = _native_program_parts(C.program).stats
    S <: _NNNoStats && return :nothing
    M = S.parameters[1]; body = M.parameters[4]
    xs = Any[_nn_stmt(s,C) for s in body.parameters]
    # `_validate_stats_body` has already proven the final statement is `return __self__`; a field call
    # inlines only the preceding diagnostic writes, exactly like the control emitter.
    pop!(xs)
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
            Expr(:call,_nn_cfg(C,:leaf),ep,
                 Expr(:call,GlobalRef(Core,:getfield),C.frame,QuoteNode(:shared)),
                 _nn_cfg(C,:handles),_nn_cfg(C,:stepkw))
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

function _nn_generated_body(ProgramT,Mid,argsyms::Tuple)
    parts=_native_program_parts(ProgramT); methods=_native_method_map(ProgramT)
    haskey(methods,Mid) || return :(throw(ArgumentError("native NUTS call to unencoded MethodId $Mid")))
    M=methods[Mid]; formals=M.parameters[3]; body=M.parameters[4]
    length(argsyms)<=length(formals.parameters) || return :(throw(MethodError($(Symbol("_nn_method",length(argsyms))), (Val($Mid),))))
    value_method=Mid in _native_value_mids(ProgramT)
    C=_NativeEmitCtx(ProgramT,parts.plan,:frame,:cfg,Dict{Symbol,Symbol}(),Dict{Symbol,Symbol}(),methods,
                     parts.derived,value_method)
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

struct _CompiledNutsRootNative{ProgramT,Refresh,Cfg,H}
    refresh::Refresh
    cfg::Cfg
    handles::H
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

# Cold prototype entry using the existing validated compiler solely for leaf/ensure/refresh/public-root setup.
# The hot root and recursive program are wholly native and registry-free.  This factoring is temporary: it keeps
# the correctness slice narrow while the final immutable state-threading compiler reuses the same ProgramT.
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
    cfg=(leaf=base.cfg.leaf,handles=base.cfg.handles,stepkw=base.cfg.stepkw,
         ensures=base.cfg.ensures,callees=E.callees,callee_refs=E.refs,
         callee_registrations=E.registrations)
    H=kernel_prepared_handles(pf)
    root=_CompiledNutsRootNative{E.program,typeof(base.refresh),typeof(cfg),typeof(H)}(base.refresh,cfg,H)
    (root! = root,scratch=(),RootToken=base.RootToken,cfg=cfg,refresh=base.refresh,program=E.program,
     refs=E.refs,registrations=E.registrations,control=base)
end
