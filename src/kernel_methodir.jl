# Normalized method-body IR (MethodIR) for `@kernel` methods — the IMPLICIT-FIELD / NO-Ref
# representation increment (V7 pivot, poc lane). Runs at RUNTIME over the substrate's STORED method
# ASTs + captured recipe/body SOURCE (`kernel_methods`/`kernel_recipe_ast`/`kernel_module`/
# `kernel_subject`/`kernel_registration`, `kernel_stateful.jl`) and builds immutable MethodIR VALUES,
# attached by the definition-unique Token. Its consumer is the LATER package graph-Value/effect
# lowering (NOT Julia SSA/IR).
#
# COMPILER BOUNDARY (RK gate 8): semantics are discovered ONLY from macro-captured @kernel source +
# explicit registrations. NO `code_lowered`/`code_typed`/`CodeInfo`/`Core.Compiler`/inference/
# arbitrary-function-body analysis — this walks stored `Expr` ASTs and reads module bindings BY
# IDENTITY (`kernel_registration`), never eval'ing a call. Ordinary Julia stays OPAQUE unless
# registered.
#
# SCOPE: REPRESENTATION + classification ONLY. It FAITHFULLY represents ordinary in-object Julia on
# the IMPLICIT-FIELD surface — a nested method declares NO receiver formal; bare unshadowed names are
# the owner's FIELDS (`_SelfField`), a lexical local/formal SHADOWS a field; the ONLY object-pass
# spelling is `__self__` as a sibling call's first actual. Mode-2 FREE methods (`leapfrog!`, `nuts!!`)
# instead carry an EXPLICIT subject formal (the receiver). Represented shapes: DIRECT self/owned-alias/
# indexed-nested PLACE writes interleaved through loops/branches/recursion, two-direct-branch direction
# (a concrete endpoint threaded as a plain formal `ep`), set-then-return, registered `copy!!`
# intrinsic, registered `step_f`/`stats_f` callable fields, `__self__` sibling calls, kwargs forwarding.
# NO Ref / dynamic current-views (superseded — the no-Ref concrete backend). NEVER rejects a faithful
# shape.
#
# Classification is THREE ORTHOGONAL FACTS on `MethodIR`: `control` (:straight/:branch/:loop/
# :recursive), the LOCAL `effects` set, and `resolution_deps` (what a later typed call-graph / trait /
# effect-closure pass must resolve — operator/rng/opaque/intrinsic/registered/field calls, mixed
# overloads, deferred aliases; never a falsely-final "pure" claim). `kind` is a SECONDARY derived
# convenience, NEVER an authority. Ordered STRUCTURED access events + call edges are derived from the
# body (`access_events`/`read_roots`/`write_roots`/`call_edges`) — branch-scoped + loop-carry aware,
# every call a read (the sound schedule input).
#
# It does NOT do graph-Value/effect lowering, `compile_update`, epoch codegen, factory construction,
# execution, `@node` double-promotion, migration, or merge — those are later increments.

# ---- IR value types (immutable; internal, unexported) -----------------------

"A hygienic reference to a complex/parametric declared type (`Vector{Float64}`, `Matrix{T}`) — the
definition module + the type AST, resolved by the later graph-Value/effect lowering. Bare concrete
types are a `GlobalRef`; where-bound typevars are their Symbol."
struct _KTypeRef; mod::Module; ast::Any; end

"""
    MethodId

DECLARATION-unique, typed overload key for a `@kernel` method. `decl` is the per-definition
declaration ORDINAL (so two same-name/same-arity overloads never collide, and classification is keyed
per declaration, never by name). `argtypes` are the HYGIENIC declared POSITIONAL types (`GlobalRef`
for a bare concrete type resolved in the definition module, a `Symbol` for a where-bound typevar, a
`_KTypeRef` for a complex/parametric type, `nothing` when unannotated); `wheres` is the `where`-clause
typevar/constraint AST list. A sentinel `npos_req == -1` marks an unclassifiable sig.
"""
struct MethodId
    name::Symbol
    decl::Int
    npos_req::Int
    npos_opt::Int
    kw_req::Tuple{Vararg{Symbol}}
    kw_opt::Tuple{Vararg{Symbol}}
    argtypes::Tuple{Vararg{Any}}
    wheres::Tuple{Vararg{Any}}
end
_kmethodid_opaque(name::Symbol, decl::Int) = MethodId(name, decl, -1, -1, (), (), (), ())
# Sentinel kw name marking a kwargs-splat: on a MethodId's `kw_opt` it means "accepts ANY keyword"
# (a `; kwargs...` formal); as a call actual it means a `; kwargs...` FORWARD (matches any overload).
const _KMIR_KWSPLAT = Symbol("kwargs...")
const _KMIR_POSSPLAT = 1_000_000     # a `xs...` positional splat -> effectively unbounded arity

abstract type _MExpr end
abstract type _MStmt end   # forward-declared: a value-position `_BlockExpr` holds `_MStmt` effects
"The stateful object / Mode-2 subject itself — the implicit receiver (`__self__` sibling actual) or
the explicit Mode-2 subject formal."
struct _SelfRef   <: _MExpr end
"A read of an owner-PATH of fields off the implicit receiver — bare `mom` -> `(:mom,)`, `fwd.mom` ->
`(:fwd, :mom)` (a Mode-2 `phasepoint.mom` -> `(:mom,)`). A single-element path is an OWNED field; a
longer path threads composed sub-objects. Resolution of shared-vs-owned is a later increment; emission
records the structural path (fields by NAME)."
struct _SelfField <: _MExpr; path::Tuple{Vararg{Symbol}}; end
"A read of a method-local binding."
struct _LocalRef  <: _MExpr; name::Symbol; end
"A read of a method formal argument. `pos` is its logical index (positional-then-keyword,
receiver-stripped); `kind` is `:pos`/`:kw`/`:possplat`/`:kwsplat`."
struct _FormalRef <: _MExpr; arg::Symbol; pos::Int; kind::Symbol; end
"A literal constant baked into the body."
struct _Lit       <: _MExpr; value::Any; end
"An external binding in VALUE position — a hygienic `GlobalRef` (deferred resolution in the definition
module), NOT an eagerly-evaluated value."
struct _ExtRef    <: _MExpr; ref::GlobalRef; end
"An OPERATOR / OPAQUE-external application `op(args...; kw...)`; `op` is a hygienic `GlobalRef` (the
later trait consumer resolves it for `typeof`-dispatch). `broadcast` marks an expression-position dotted
operator (`a .+ b`). `hint` ∈ (`:operator_candidate` — an operator callee, a HINT not proof;
`:opaque` — a named UNREGISTERED external, footprint proven later). A DIRECT registered @kernel /
intrinsic callee is instead a `_RegisteredCall` (resolved from the captured snapshot); RNG is NOT a
hint (never name-classified). Usable value OR discarded-value (statement)."
struct _OpCall    <: _MExpr
    op::GlobalRef
    args::Tuple{Vararg{_MExpr}}
    kw::Tuple{Vararg{Pair{Symbol,_MExpr}}}
    broadcast::Bool
    hint::Symbol
end
_OpCall(op::GlobalRef, args, kw, broadcast) = _OpCall(op, args, kw, broadcast, :operator_candidate)
"Indexing `base[idxs...]`."
struct _Index     <: _MExpr; base::_MExpr; idxs::Tuple{Vararg{_MExpr}}; end
"A non-receiver property access `base.field` (base not rooted at the receiver)."
struct _Getfield  <: _MExpr; base::_MExpr; field::Symbol; end
"Short-circuit `lhs && rhs` / `lhs || rhs` in expression position (op ∈ (:&&, :||))."
struct _Short     <: _MExpr; op::Symbol; lhs::_MExpr; rhs::_MExpr; end
"A CHAINED comparison `a < b <= c` — ordered `operands` (N+1 values) + `ops` (N operators) + the captured
EXACT PURE-PRIMITIVE provenance of each operator (`refs`/`regs`, parallel to `ops`; RK 06:08 — a chained
comparison operator is NOT spelling-authorized, it resolves through the SAME definition-time pure identity
as a call and is rebind-checked at emission or rejected). A faithful value node; the `&&`-lowering with the
shared middle temp is left to the later lowering."
struct _Comparison <: _MExpr
    operands::Tuple{Vararg{_MExpr}}
    ops::Tuple{Vararg{Symbol}}
    refs::Tuple{Vararg{_CapturedCalleeRef}}
    regs::Tuple{Vararg{_KernelRegistration}}
end
"Ternary `cond ? then : else` in expression position."
struct _IfExpr    <: _MExpr; cond::_MExpr; thenv::_MExpr; elsev::_MExpr; end
"A value-position BLOCK — an if-EXPRESSION branch (`may_continue = if c; <effects…>; <value> end`):
ordered effect `stmts` then the terminal `value` (the block's result)."
struct _BlockExpr <: _MExpr; stmts::Tuple{Vararg{_MStmt}}; value::_MExpr; end
"A tuple constructor `(a, b, …)` in expression position."
struct _TupleExpr <: _MExpr; elts::Tuple{Vararg{_MExpr}}; end
"A `@node(expr)` lifted graph recipe used in VALUE position. `@node` is ALREADY a real lifted graph
recipe (RK): representation records it as a distinguished node and does NOT double-promote — but its
inner expression IS source-normalized into `inner` (an `_MExpr`) so its dependency READS are emitted
(a lifted recipe never drops its deps). `raw` retains the verbatim inner AST + `mod` for the later graph
binding."
struct _NodeExpr  <: _MExpr; inner::_MExpr; raw::Any; mod::Module; end
"A SIBLING method call used as a VALUE (expression position / a terminal reducer result). `candidates`
is the arity/kw-matching overload set; `target` is the callee's receiver actual (`__self__`);
`pos`/`kw` the remaining actuals."
struct _CallExpr <: _MExpr
    name::Symbol
    candidates::Tuple{Vararg{Any}}    # _CallCandidate (defined below)
    target::_MExpr
    pos::Tuple{Vararg{_MExpr}}
    kw::Tuple{Vararg{Pair{Symbol,_MExpr}}}
end
"A call whose CALLEE is a receiver CALLABLE-FIELD path — `step_f(ep)` / `stats_f(__self__)`. This is a
consumer-supplied callable held in a reactive field. `hint` records the construction-time resolution
policy: `:registered` (must resolve to a registered @kernel/intrinsic at construction — `step_f`;
registered-or-REJECT, NEVER opaque) or `:registered_or_nothing` (registered @kernel OR `nothing` —
`stats_f`, guarded by an `isnothing` short-circuit). `path` is the callee field path, `pos`/`kw` the
actuals. A callable field is NEVER an opaque runtime Function — it carries a visible registered
identity resolved by the later resolver."
struct _FieldCall <: _MExpr
    path::Tuple{Vararg{Symbol}}
    pos::Tuple{Vararg{_MExpr}}
    kw::Tuple{Vararg{Pair{Symbol,_MExpr}}}
    hint::Symbol
end
_FieldCall(path, pos, kw) = _FieldCall(path, pos, kw, :registered)
"A named-tuple construction with property shorthand — `(; fwd.mom, fwd.dham_dmom)` →
`names = (:mom, :dham_dmom)`, `vals` the field-read `_MExpr`s."
struct _NamedTuple <: _MExpr
    names::Tuple{Vararg{Symbol}}
    vals::Tuple{Vararg{_MExpr}}
end
"A call to a DIRECT REGISTERED @kernel / RK-core intrinsic (`copy!!(dest, src)`), resolved from the
substrate's DEFINITION-TIME detached registration SNAPSHOT (`kernel_callee_registrations`) — NOT a
reread of the module global at analysis time (a rebind before analysis would silently substitute a
different callee). `ref` is the authored binding slot (`_CapturedCalleeRef`); `registration` is the
immutable captured `_KernelRegistration` (Token / kind / subject / effect-roots / `!!` contract /
source) — the IDENTITY. `intrinsic` marks kind `:intrinsic`. `args`/`kw`/`broadcast` the actuals. A
later pass re-validates the slot with `kernel_rebound`; drift detected at emission is rejected."
struct _RegisteredCall <: _MExpr
    ref::_CapturedCalleeRef
    registration::_KernelRegistration
    intrinsic::Bool
    args::Tuple{Vararg{_MExpr}}
    kw::Tuple{Vararg{Pair{Symbol,_MExpr}}}
    broadcast::Bool
end
"A PENDING call to a METHOD OF THE SUBJECT object (`nuts!!`'s `step!(state, rng)`): the subject is a
compiler-owned KernelObject and `name` is one of its registered methods, resolved AT CONSTRUCTION from
the object's Token/MethodId — never a global external/opaque callee. `name` is the method name;
`subject` the subject receiver `_MExpr` (`_SelfRef`); `pos`/`kw` the remaining actuals."
struct _SubjectMethodCall <: _MExpr
    name::Symbol
    subject::_MExpr
    pos::Tuple{Vararg{_MExpr}}
    kw::Tuple{Vararg{Pair{Symbol,_MExpr}}}
end

"""
    _Formal

One authored (receiver-stripped for Mode-2; no strip for implicit-field) method formal: `name`, `kind`
(`:pos`/`:kw`/`:possplat`/`:kwsplat`), `required` (no default), the declared `type` annotation AST (or
`nothing`), and the normalized `default` `_MExpr` (in the METHOD's scope) when optional.
"""
struct _Formal
    name::Symbol
    kind::Symbol
    required::Bool
    type::Any
    default::Union{Nothing,_MExpr}
end

"A method-local (re)bind `lhs = rhs`; `lhs` is 1+ names (a tuple for destructuring). A compound
`x op= rhs` is normalized so `rhs` reads the pre-write value of `x`."
struct _LocalAssign <: _MStmt; lhs::Tuple{Vararg{Symbol}}; rhs::_MExpr; end
"One arity/kw-matching overload candidate of a sibling call: its declaration-unique `MethodId` + the
callee's SECONDARY derived `kind` (informational; identity is the `id`)."
struct _CallCandidate; id::MethodId; kind::Symbol; end

"A sibling method call (`flip!(__self__, depth)`). `name` is the callee name; `candidates` is the
ARITY/KW-matching overload set (NOT erased to one — the later typed call-graph narrows by arg-node
types), each with its declaration-unique id + kind. `target` is the callee's receiver actual
(`__self__` → `_SelfRef`); `pos`/`kw` are the remaining caller actuals."
struct _Call <: _MStmt
    name::Symbol
    candidates::Tuple{Vararg{_CallCandidate}}
    target::_MExpr
    pos::Tuple{Vararg{_MExpr}}
    kw::Tuple{Vararg{Pair{Symbol,_MExpr}}}
end
"A bare expression evaluated for effect / a trailing result."
struct _ExprStmt <: _MStmt; expr::_MExpr; end
"`if cond; thenb…; else elseb…; end` (elseif chains nest inside `elseb`)."
struct _If    <: _MStmt; cond::_MExpr; thenb::Tuple{Vararg{_MStmt}}; elseb::Tuple{Vararg{_MStmt}}; end
"`for var in iter; body…; end` (retains loop induction + the loop-local binding)."
struct _For   <: _MStmt; var::Tuple{Vararg{Symbol}}; iter::_MExpr; body::Tuple{Vararg{_MStmt}}; end
"`while cond; body…; end`."
struct _While <: _MStmt; cond::_MExpr; body::Tuple{Vararg{_MStmt}}; end
"Statement-level short-circuit GUARD `cond && (effect)` / `cond || (effect)` — the RHS is an EFFECT
(assignment / sibling call / field call / break/continue/return / block), op ∈ (:&&, :||)."
struct _Guard <: _MStmt; op::Symbol; cond::_MExpr; body::Tuple{Vararg{_MStmt}}; end
"A `return value` — trailing (the method result) OR an early return."
struct _Return <: _MStmt; value::Union{Nothing,_MExpr}; end
"`break` / `continue` — ordinary loop control."
struct _Break    <: _MStmt end
struct _Continue <: _MStmt end
"A write to a mutable PLACE — a location rooted at an implicit owned FIELD (`gofwd`, `trees[1].log_weight[1]`,
`bwd.mom`; a Mode-2 `phasepoint.mom`), an OWNED-ALIAS local (`tree = trees[depth]` then `@. tree.bwd.mom
= …`), or a plain/loop-local temp. `target` is the full LHS `_MExpr` place; `root` ∈ (`:self`, `:alias`,
`:deferred`); `owner` is the owning field path prefix for `:self`/`:alias` (a write-ROOT for the
footprint) else `nothing`; `alias` names the resolved owned-alias local for `:alias`. `rhs` folds a
compound `op=`; `dot` marks broadcast/`@.`. Represented in EVERY control shape — never rejected."
struct _PlaceWrite <: _MStmt
    target::_MExpr
    root::Symbol
    owner::Union{Nothing,Tuple{Vararg{Symbol}}}
    alias::Union{Nothing,Symbol}
    rhs::_MExpr
    dot::Bool
end
"A tuple-destructuring swap/assignment — Julia semantics PINNED: every RHS source is evaluated FIRST,
THEN stored. Covers the proposal swap `proposals[i], proposals[j] = proposals[j], proposals[i]`."
struct _PlaceSwap <: _MStmt
    targets::Tuple{Vararg{_PlaceWrite}}
end
"A set-then-return `cond || return may_sample = false` — the write is performed and its value returned.
`write` is the performed `_PlaceWrite`."
struct _SetReturn <: _MStmt
    write::_PlaceWrite
end

"""
    MethodIR

The immutable ordered structured IR of one `@kernel` method body. Classification is THREE ORTHOGONAL
FACTS (primary; none gates legality of a direct in-object write):
- `control` ∈ (`:straight`, `:branch`, `:loop`, `:recursive`) — the most-complex control shape present.
- `effects` — a SET (Tuple) ⊆ (`:place_write`, `:opaque_call`); the empty tuple is a body with no LOCAL
  effect. Transitive sibling/registered/opaque/RNG effects are NOT baked in (a later
  call-graph fixed point produces closed summaries).
- `resolution_deps` — what must be resolved later: sibling-overload sets, opaque-callee traits,
  registered/intrinsic-callee (captured Token) identities, callable-field identities, subject-method
  calls, unresolved aliases; never a falsely-final "resolved purity" claim. RNG is NOT a dep here — it
  is never name-classified; RNG effect/ordering is deferred to a later trait / typed-RuntimeArg fact.
`kind` is a SECONDARY derived convenience (`:segment`/`:orchestration`/`:unresolved`), never the
authority. `ok` is `false` only for a genuine grammar violation (a faithful in-object write is NEVER a
violation). `signature` retains the FULL authored signature AST; `self` is the receiver symbol (`__self__`
for implicit-field, the subject for Mode-2). Ordered STRUCTURED access events + call EDGES are derived
from `body` via `access_events`/`read_roots`/`write_roots`/`call_edges`."
"""
struct MethodIR
    id::MethodId
    self::Symbol
    formals::Tuple{Vararg{_Formal}}
    body::Tuple{Vararg{_MStmt}}
    control::Symbol
    effects::Tuple{Vararg{Symbol}}
    resolution_deps::Tuple{Vararg{Any}}
    kind::Symbol
    ok::Bool
    reason::Union{Nothing,String}
    signature::Any
end

# Contract-violation signal, caught at the method boundary — never escapes emission.
struct _KMIRReject <: Exception; reason::String; end
_kmir_reject(reason::AbstractString) = throw(_KMIRReject(String(reason)))

# ---- analysis context -------------------------------------------------------

# `mode` ∈ (:implicit, :mode2). In :implicit a nested method has NO receiver formal — bare unshadowed
# names in `fields` are owned FIELDS; `__self__` is the sibling receiver actual. In :mode2 the receiver
# is the explicit `self` subject formal (`phasepoint.field`), and there are no sibling methods.
struct _KMIRCtx
    fields::Set{Symbol}                          # the object's port/derived field names (owned roots)
    methods::Set{Symbol}                         # sibling method names
    specs::Dict{Symbol,Vector{_CallCandidate}}   # method name -> its overload candidates (id + kind)
    formalpos::Dict{Symbol,Int}
    formalkind::Dict{Symbol,Symbol}
    self::Symbol                                 # receiver symbol (`__self__` implicit / subject Mode-2)
    mod::Module
    kind::Symbol                                 # the CURRENT method's SECONDARY kind (informational)
    mode::Symbol                                 # :implicit | :mode2
    callee_regs::Tuple                            # def-time DETACHED registration snapshot (`_CapturedCallee`s)
    optional_here::Set{Symbol}                   # fields under a DOMINATING `isnothing(f) ||` guard
                                                 #   at the CURRENT dominated call site (scope-live, `||` only)
    maybe::Set{Symbol}                           # locals bound on only SOME path (maybe-bound) -> a later
                                                 #   read is unsound; rejected actionably (definite-assignment)
    aliases::Dict{Symbol,Tuple{Vararg{Symbol}}}  # SCOPE-LIVE owned-alias local -> owning field path
    reassigned::Set{Symbol}                      # locals bound MORE THAN ONCE -> never a definite alias
end
_KMIRCtx(fields, methods, specs, formalpos, formalkind, self, mod, kind, mode, callee_regs) =
    _KMIRCtx(fields, methods, specs, formalpos, formalkind, self, mod, kind, mode, callee_regs,
             Set{Symbol}(), Set{Symbol}(), Dict{Symbol,Tuple{Vararg{Symbol}}}(), Set{Symbol}())

const _KMIR_SELFACTUAL = :__self__               # the object-pass receiver actual (implicit-field)

# ---- signature parsing (reuses the existing authoring adapter; RK gate 2) ----

_kmir_peel(sig) = _kernel_peel_signature(sig)

# The `where`-clause typevar/constraint list of a signature (outermost first).
function _kmir_wheres(sig)
    ws = Any[]
    while sig isa Expr && sig.head === :where
        append!(ws, sig.args[2:end]); sig = sig.args[1]
    end
    ws
end
# The return `::` annotation of a signature (or nothing).
function _kmir_retann(sig)
    while sig isa Expr && sig.head === :where; sig = sig.args[1]; end
    (sig isa Expr && sig.head === :(::) && length(sig.args) == 2) ? sig.args[2] : nothing
end
# The where-bound typevar names of a signature (LHS of each `where` constraint).
function _kmir_where_vars(sig)
    vars = Set{Symbol}()
    for w in _kmir_wheres(sig); push!(vars, _kmir_where_var(w)); end
    vars
end
_kmir_where_var(w) = w isa Symbol ? w :
    (w isa Expr && w.head in (:<:, :>:) && w.args[1] isa Symbol ? w.args[1] : Symbol(string(w)))

# A HYGIENIC reference to a declared type annotation: a where-bound typevar stays its Symbol; a bare
# concrete type resolves to a `GlobalRef` in the definition module; a complex/parametric type becomes a
# `_KTypeRef` (module + AST). `nothing` = unannotated.
_kmir_type_ref(::Nothing, ::Module, ::Set{Symbol}) = nothing
function _kmir_type_ref(ast, mod::Module, wherevars::Set{Symbol})
    ast isa Symbol && return ast in wherevars ? ast : GlobalRef(mod, ast)
    _KTypeRef(mod, ast)
end

# The base name of a formal declaration, peeling ::T and =default.
function _kmir_base_name(arg)
    arg isa Symbol && return arg
    (arg isa Expr && arg.head in (:(::), :kw, :(=))) && return _kmir_base_name(arg.args[1])
    _kmir_reject("unsupported formal declaration `$(repr(arg))`")
end
# The declared TYPE annotation of a formal (or nothing).
function _kmir_formal_type(arg)
    arg isa Expr || return nothing
    arg.head === :(::) && return arg.args[2]
    (arg.head === :kw || arg.head === :(=)) && return _kmir_formal_type(arg.args[1])
    nothing
end
# (name, kind, required, type, default_expr) for one formal.
function _kmir_formal(arg, kind::Symbol)
    if arg isa Symbol
        return (arg, kind, true, nothing, nothing)
    elseif arg isa Expr
        arg.head === :(::) && return (_kmir_base_name(arg.args[1]), kind, true, arg.args[2], nothing)
        (arg.head === :kw || arg.head === :(=)) &&
            return (_kmir_base_name(arg.args[1]), kind, false, _kmir_formal_type(arg.args[1]), arg.args[2])
        (arg.head === :(...) && length(arg.args) == 1 && arg.args[1] isa Symbol) &&
            return (arg.args[1], kind === :kw ? :kwsplat : :possplat, false, nothing, nothing)
    end
    _kmir_reject("unsupported formal declaration `$(repr(arg))`")
end

# Raw formal descriptors, positional then keyword. `strip_first` drops the first positional (the Mode-2
# subject receiver); implicit-field methods strip nothing (they declare no receiver formal). Throws
# _KMIRReject on an unsupported formal.
function _kmir_raw_formals(call, strip_first::Bool)
    (call isa Expr && call.head === :call) || _kmir_reject("malformed method signature")
    args = call.args[2:end]
    positional = Any[a for a in args if !(a isa Expr && a.head === :parameters)]
    strip_first && !isempty(positional) && (positional = positional[2:end])
    out = Tuple{Symbol,Symbol,Bool,Any,Any}[]
    for a in positional; push!(out, _kmir_formal(a, :pos)); end
    for a in args
        (a isa Expr && a.head === :parameters) || continue
        for kw in a.args; push!(out, _kmir_formal(kw, :kw)); end
    end
    out
end

# Does a candidate `MethodId` match a call of `npos` positional actuals + `kwset` keyword names?
function _kmir_kwarity_match(id::MethodId, npos::Int, kwset::Set{Symbol})
    (id.npos_req <= npos <= id.npos_req + id.npos_opt) || return false
    (_KMIR_KWSPLAT in kwset) && return true
    cand_any = _KMIR_KWSPLAT in id.kw_opt
    cand_any ? issubset(Set(id.kw_req), kwset) :
        (issubset(kwset, union(Set(id.kw_req), Set(id.kw_opt))) && issubset(Set(id.kw_req), kwset))
end

# Declaration-unique typed MethodId for one method declaration (`decl` = its ordinal).
function _kmir_methodid(name::Symbol, decl::Int, raw, mod::Module, wherevars::Set{Symbol}, wheres)
    haskwsplat = any(d -> d[2] === :kwsplat, raw)   # `; kwargs...` -> accepts ANY keyword
    hasposplat = any(d -> d[2] === :possplat, raw)  # `xs...`        -> unbounded positional arity
    npos_req = count(d -> d[2] === :pos && d[3], raw)
    npos_opt = count(d -> d[2] === :pos && !d[3], raw) + (hasposplat ? _KMIR_POSSPLAT : 0)
    kw_req = Tuple(d[1] for d in raw if d[2] === :kw && d[3])
    kw_opt = (Tuple(d[1] for d in raw if d[2] === :kw && !d[3])..., (haskwsplat ? (_KMIR_KWSPLAT,) : ())...)
    argtypes = Tuple(_kmir_type_ref(d[4], mod, wherevars) for d in raw if d[2] === :pos)
    MethodId(name, decl, npos_req, npos_opt, kw_req, kw_opt, argtypes, Tuple(wheres))
end

# Resolve a sibling call to its arity/kw-matching CANDIDATE SET; no match / an opaque candidate rejects.
function _kmir_resolve_call(name::Symbol, npos::Int, kwnames::Vector{Symbol},
                            specs::Dict{Symbol,Vector{_CallCandidate}})
    cands = get(specs, name, _CallCandidate[])
    isempty(cands) && _kmir_reject("sibling call `$name` has no matching @kernel method")
    any(c -> c.id.npos_req == -1, cands) &&
        _kmir_reject("sibling call `$name` targets a method with an unclassifiable signature")
    kwset = Set(kwnames)
    matches = filter(c -> _kmir_kwarity_match(c.id, npos, kwset), cands)
    isempty(matches) && _kmir_reject(
        "sibling call `$name` ($npos positional, kw $(sort(collect(kwset)))) matches no @kernel overload")
    Tuple(matches)
end

# ---- receiver owner-path detection ------------------------------------------

# If `ex` is a getproperty chain rooted at the Mode-2 subject (`phasepoint.a.b`), return `(:a, :b)`;
# else `nothing`. (Mode-2 explicit receiver.)
function _kmir_subject_path(ex, self::Symbol)
    ex === self && return ()
    (ex isa Expr && ex.head === :. && length(ex.args) == 2 && ex.args[2] isa QuoteNode) || return nothing
    base = _kmir_subject_path(ex.args[1], self)
    base === nothing ? nothing : (base..., ex.args[2].value)
end
# If `ex` is a getproperty chain rooted at an UNSHADOWED owned field (implicit-field), return the FULL
# field path INCLUDING the root field (`fwd.mom` -> `(:fwd, :mom)`, bare `gofwd` -> `(:gofwd,)`); else
# `nothing`. A shadowed name / a formal / a non-field root is NOT a field path.
function _kmir_field_path(ex, ctx::_KMIRCtx, shadow::Set{Symbol})
    if ex isa Symbol
        (ex in shadow || haskey(ctx.formalpos, ex)) && return nothing
        return (ex in ctx.fields) ? (ex,) : nothing
    end
    (ex isa Expr && ex.head === :. && length(ex.args) == 2 && ex.args[2] isa QuoteNode) || return nothing
    base = _kmir_field_path(ex.args[1], ctx, shadow)
    base === nothing ? nothing : (base..., ex.args[2].value)
end
# Unified receiver field-path (mode-dependent), or nothing.
_kmir_recv_path(ex, ctx::_KMIRCtx, shadow::Set{Symbol}) =
    ctx.mode === :mode2 ? _kmir_subject_path(ex, ctx.self) : _kmir_field_path(ex, ctx, shadow)

# Peel `.`/`[]`/`::` navigation down to the base symbol/expr (for place-root classification).
_kmir_nav_root(x::Symbol) = x
_kmir_nav_root(x) = (x isa Expr && x.head in (:., :ref, :(::))) ? _kmir_nav_root(x.args[1]) : x

# The receiver field path of a PLACE lhs, INDEX-TRANSPARENT (an index `[]` navigates WITHIN a field) —
# `phasepoint.mom` -> `(:mom,)` (Mode-2 subject), `bwd.mom` -> `(:bwd, :mom)`, `trees[1].log_weight[1]`
# -> `(:trees, :log_weight)`, bare `gofwd` -> `(:gofwd,)`. `nothing` for a non-receiver-rooted place
# (a loop-local `t.bwd.mom`, an alias local). Distinct from `_kmir_recv_path` (index-OPAQUE, for value
# reads that must preserve `[]`).
function _kmir_owner_path(ex, ctx::_KMIRCtx, shadow::Set{Symbol})
    if ex isa Expr && ex.head === :ref
        return _kmir_owner_path(ex.args[1], ctx, shadow)
    elseif ex isa Expr && ex.head === :. && length(ex.args) == 2 && ex.args[2] isa QuoteNode
        base = _kmir_owner_path(ex.args[1], ctx, shadow)
        return base === nothing ? nothing : (base..., ex.args[2].value)
    elseif ex isa Symbol
        if ctx.mode === :mode2
            return ex === ctx.self ? () : nothing
        else
            (ex in shadow || haskey(ctx.formalpos, ex)) && return nothing
            return (ex in ctx.fields) ? (ex,) : nothing
        end
    end
    nothing
end

# ---- dotted-operator + compound helpers -------------------------------------

_kmir_dotted_op(callee) = callee isa Symbol &&
    (s = String(callee); length(s) >= 2 && s[1] == '.' && s[2] != '.')
_kmir_base_op(callee) = Symbol(String(callee)[2:end])
function _kmir_compound_op(head::Symbol)
    (head === :(=) || head === :(.=)) && return nothing
    s = String(head)
    (endswith(s, "=") && length(s) > 1 && s[end-1] != '=' &&
        !(s in ("!=", "<=", ">=", ".!=", ".<=", ".>="))) || return nothing
    Symbol(s[1:end-1])
end
_kmir_is_dotassign(head::Symbol) =
    head === :(.=) || (startswith(String(head), ".") && endswith(String(head), "=") &&
                       _kmir_compound_op(head) !== nothing)
_kmir_is_assign(head::Symbol) =
    head === :(=) || head === :(.=) || _kmir_compound_op(head) !== nothing
_kmir_is_dotmacro(ex) = ex isa Expr && ex.head === :macrocall &&
    !isempty(ex.args) && ex.args[1] === Symbol("@__dot__")
# A `@node`-SPELLED macrocall (a candidate) — spelling only; genuine RK-@node IDENTITY is decided by
# `_kernel_marker_kind` (resolves the `@node` binding to RK's `var"@node"`, defeating a foreign/local
# spoof). Spelling alone NEVER authorizes promotion (RK marker-hygiene).
_kmir_node_candidate(ex) = _kernel_marker_candidate(ex) === :node

# ---- expression normalization -> _MExpr VALUE -------------------------------

# A human-readable authored-slot name for an error message.
_kmir_ref_str(ref::_CapturedCalleeRef) =
    ref.field === nothing ? String(ref.slot.name) : "$(ref.slot.name).$(ref.field)"

# Look up a DIRECT callee in the def-time DETACHED registration SNAPSHOT (finding 2). `callee` is a bare
# Symbol (`copy!!`) or a qualified `Mod.f`. Returns the captured `_CapturedCallee` — the authored slot
# key + the immutable `_KernelRegistration` IDENTITY (Token/kind/subject/effect/!!) — or `nothing`. The
# snapshot (not a reread of the module global) is the semantic authority; a rebind is only VALIDATED:
# if the authored slot RE-RESOLVES to a different definition than captured, REJECT (drift detected).
function _kmir_lookup_captured(callee, ctx::_KMIRCtx)
    ref = if callee isa Symbol
        _CapturedCalleeRef(GlobalRef(ctx.mod, callee), nothing)
    elseif callee isa Expr && callee.head === :. && length(callee.args) == 2 &&
           callee.args[1] isa Symbol && callee.args[2] isa QuoteNode
        _CapturedCalleeRef(GlobalRef(ctx.mod, callee.args[1]), callee.args[2].value)
    else
        return nothing
    end
    cc = nothing
    for c in ctx.callee_regs
        (c.ref == ref) && (cc = c; break)
    end
    cc === nothing && return nothing
    kernel_rebound(cc.registration, _kernel_resolve_captured_ref(cc.ref)) && _kmir_reject(
        "registered callee `$(_kmir_ref_str(ref))` was REBOUND after definition — the captured " *
        "registration drifted from the current binding (stale snapshot)")
    cc
end

# Resolve a BASE operator identity (`op` a bare Symbol, already de-dotted) to a captured EXACT
# PURE-PRIMITIVE `_CapturedCallee`, or `nothing` (RK 06:01). This closes the operator loophole:
# ordinary/dotted/compound operators resolve through the SAME definition-time pure provenance as
# named calls (rebind-validated by `_kmir_lookup_captured`), NOT `Base.isoperator` spelling.
function _kmir_pure_op_captured(op::Symbol, ctx::_KMIRCtx)
    cc = _kmir_lookup_captured(op, ctx)
    (cc !== nothing && cc.registration.kind === :pure_primitive) ? cc : nothing
end

# The synthesized binary-operator application of a COMPOUND assignment (`a += b` → `a + b`): a captured
# EXACT pure-primitive base op -> a `_RegisteredCall` (definition-time identity, rebind-checked); else a
# hinted `_OpCall`. So compound operators are NOT a spelling loophole either (RK 06:01).
function _kmir_binop_expr(base::Symbol, args::Tuple, broadcast::Bool, ctx::_KMIRCtx)
    cc = _kmir_pure_op_captured(base, ctx)
    cc !== nothing && return _RegisteredCall(cc.ref, cc.registration, false, args, (), broadcast)
    _OpCall(GlobalRef(ctx.mod, base), args, (), broadcast)
end

# Classify a bare/qualified callee -> a hygienic GlobalRef, else reject.
function _kmir_callee_ref(callee, ctx::_KMIRCtx, shadow::Set{Symbol})
    if callee isa Symbol
        callee in ctx.methods &&
            _kmir_reject("sibling method call `$callee` in expression position is not compilable")
        (callee === ctx.self || callee in shadow || haskey(ctx.formalpos, callee)) &&
            _kmir_reject("a dynamic callable from a formal/local/receiver `$callee` is not compilable")
        return GlobalRef(ctx.mod, callee)
    elseif callee isa Expr && callee.head === :. && length(callee.args) == 2 &&
           callee.args[1] isa Symbol && callee.args[2] isa QuoteNode
        m = try getproperty(ctx.mod, callee.args[1]) catch; nothing end
        m isa Module && return GlobalRef(m, callee.args[2].value)
        _kmir_reject("qualified callee `$(callee.args[1]).$(callee.args[2].value)` — " *
                     "`$(callee.args[1])` is not a module in the definition scope (unresolved qualifier)")
    end
    _kmir_reject("a computed/dynamic callee `$(repr(callee))` is not compilable")
end

# Parse one keyword actual into a `name => value` pair.
function _kmir_kw_pair(k, ctx::_KMIRCtx, shadow::Set{Symbol}; bcast::Bool=false)
    k isa Symbol && return (k => _kmexpr(k, ctx, shadow; bcast=bcast))
    if k isa Expr
        k.head === :kw && return (k.args[1] => _kmexpr(k.args[2], ctx, shadow; bcast=bcast))
        (k.head === :(...) && length(k.args) == 1) &&
            return (_KMIR_KWSPLAT => _kmexpr(k.args[1], ctx, shadow; bcast=bcast))
        (k.head === :. && length(k.args) == 2 && k.args[2] isa QuoteNode) &&
            return (k.args[2].value => _kmexpr(k, ctx, shadow; bcast=bcast))
    end
    _kmir_reject("unsupported keyword `$(repr(k))` in call")
end

# The effect/resolution HINT for an OPAQUE/operator `_OpCall` (registered/intrinsic callees are a
# separate `_RegisteredCall`, resolved from the captured snapshot BEFORE this): `:operator_candidate`
# for an operator callee; else `:opaque`. RNG is NOT classified here — a call is not "rng" because an
# actual is spelled `rng` (unsound: false-positives a global/renamed collision, misses a differently-
# named typed RuntimeArg). An unregistered RNG helper is still opaque; exact built-ins such as `rand`/
# `randexp` resolve earlier as `_RegisteredCall`s; NUTS keeps its registered draws directly in the
# consuming branch MethodIR.
function _kmir_call_hint(callee, ex, ctx::_KMIRCtx)
    ((callee isa Symbol && Base.isoperator(callee)) || _kmir_dotted_op(callee)) && return :operator_candidate
    :opaque
end

# Positional + keyword actuals of a call, as `_MExpr` values.
function _kmir_pos_kw(ex, ctx::_KMIRCtx, shadow::Set{Symbol}; bcast::Bool=false)
    pos = _MExpr[]; kwp = Pair{Symbol,_MExpr}[]
    for a in ex.args[2:end]
        if a isa Expr && a.head === :parameters
            for k in a.args; push!(kwp, _kmir_kw_pair(k, ctx, shadow; bcast=bcast)); end
        elseif a isa Expr && a.head === :kw
            push!(kwp, a.args[1] => _kmexpr(a.args[2], ctx, shadow; bcast=bcast))
        elseif a isa Expr && a.head === :(...)
            _kmir_reject("a splatted positional argument is not compilable")
        else
            push!(pos, _kmexpr(a, ctx, shadow; bcast=bcast))
        end
    end
    (Tuple(pos), Tuple(kwp))
end

# The construction-time resolution hint for a callable-FIELD call (`step_f`/`stats_f`): a field call at a
# site DOMINATED by an `isnothing(<field>) ||` guard (this specific dominated call, scope-live) is
# registered-OR-NOTHING; ANY other site — unguarded, a 2nd call, or an `isnothing(f) &&` — is
# registered-or-REJECT. (`&&` never marks optional: `isnothing(f) && f(x)` only calls f when it IS nothing.)
function _kmir_fieldcall_hint(path::Tuple, ctx::_KMIRCtx)
    (length(path) == 1 && path[1] in ctx.optional_here) ? :registered_or_nothing : :registered
end

function _kmexpr(ex, ctx::_KMIRCtx, shadow::Set{Symbol}; bcast::Bool=false)
    # the implicit receiver actual (only legal as a sibling first-actual; as a value -> the object)
    ex === _KMIR_SELFACTUAL && ctx.mode === :implicit && return _SelfRef()
    # Finding 4: a maybe-bound local (bound on only some path) read after the branch/guard is UNSOUND —
    # reject actionably rather than silently resolve it as an unconditional local / field / global.
    (ex isa Symbol && ex in ctx.maybe && !(ex in shadow)) && _kmir_reject(
        "maybe-bound local `$ex` is read after a conditional/guard binding that may not have run")
    # receiver owner-path (bare field / field.a.b ; Mode-2 subject / subject.a.b)
    rp = _kmir_recv_path(ex, ctx, shadow)
    if rp !== nothing
        return isempty(rp) ? _SelfRef() : _SelfField(rp)
    end
    if ex isa Symbol
        (ex in shadow) && return _LocalRef(ex)
        haskey(ctx.formalpos, ex) && return _FormalRef(ex, ctx.formalpos[ex], ctx.formalkind[ex])
        return _ExtRef(GlobalRef(ctx.mod, ex))
    end
    (ex isa Expr) || return _Lit(ex)
    ex.head === :quote && return _Lit(ex)
    # @node promotion is IDENTITY-aware (RK marker-hygiene): a `@node`-spelled macrocall promotes to a
    # `_NodeExpr` (a lifted recipe; inner normalized for dep reads, NO double-promote) ONLY when its
    # `@node` binding resolves to RK's genuine `@node` (`_kernel_marker_kind === :node`). A foreign/local
    # `@node` macro (a spoof) is ordinary Julia macro semantics — OPAQUE under the compiler boundary —
    # and is REJECTED actionably, never silently promoted on spelling.
    if _kmir_node_candidate(ex)
        _kernel_marker_kind(ctx.mod, ex) === :node && return _NodeExpr(
            _kmexpr(ex.args[end], ctx, shadow; bcast=bcast), ex.args[end], ctx.mod)
        _kmir_reject("a `@node`-spelled macro whose `@node` does NOT resolve to RK's `@node` (a foreign/" *
                     "local macro binding) is not RK node provenance — ordinary Julia macro semantics are " *
                     "opaque under the compiler boundary")
    end
    m(node) = _kmexpr(node, ctx, shadow; bcast=bcast)
    if ex.head === :. && length(ex.args) == 2 && ex.args[2] isa QuoteNode
        return _Getfield(m(ex.args[1]), ex.args[2].value)
    elseif ex.head === :ref
        return _Index(m(ex.args[1]), Tuple(m(a) for a in ex.args[2:end]))
    elseif ex.head === :call
        callee = ex.args[1]
        # (a) callee is a receiver CALLABLE-FIELD path (`step_f(ep)`) -> a `_FieldCall` (registered).
        cp = _kmir_recv_path(callee, ctx, shadow)
        if cp !== nothing && !isempty(cp)
            pos, kwp = _kmir_pos_kw(ex, ctx, shadow; bcast=bcast)
            return _FieldCall(cp, pos, kwp, _kmir_fieldcall_hint(cp, ctx))
        end
        # (b) callee is a SIBLING method used as a VALUE -> a `_CallExpr`.
        if callee isa Symbol && callee in ctx.methods
            nm, cands, tgt, ps, kw = _kmir_resolve_sibling(ex, ctx, shadow)
            return _CallExpr(nm, cands, tgt, ps, kw)
        end
        # remaining callees (global/qualified/operator): parse the actuals once.
        bc = _kmir_dotted_op(callee)
        pos, kwp = _kmir_pos_kw(ex, ctx, shadow; bcast=bcast)
        # (c) a DIRECT REGISTERED @kernel / intrinsic / exact pure-primitive callee -> `_RegisteredCall`
        #     from the captured def-time snapshot (finding 2), carrying the immutable registration
        #     identity. Ordinary spelling resolves directly; a DOTTED pure operator (`.+`, bc) resolves
        #     through its captured BASE identity with broadcast set (RK 06:01 — no spelling loophole).
        cc = bc ? _kmir_pure_op_captured(_kmir_base_op(callee), ctx) : _kmir_lookup_captured(callee, ctx)
        cc !== nothing && return _RegisteredCall(cc.ref, cc.registration,
            cc.registration.kind === :intrinsic, pos, kwp, bcast || bc)
        # (d) a call passing the RECEIVER/subject as its FIRST actual to a plain (non-op) name that is
        #     neither sibling, field, nor a captured global -> a PENDING subject-method call (finding 1;
        #     `nuts!!`'s `step!(state, rng)`), resolved at construction via the object's Token/MethodId.
        if callee isa Symbol && !bc && !Base.isoperator(callee) && !isempty(pos) && pos[1] isa _SelfRef
            return _SubjectMethodCall(callee, pos[1], pos[2:end], kwp)
        end
        # (e) an operator / opaque external call -> `_OpCall` with a HINT.
        op = bc ? GlobalRef(ctx.mod, _kmir_base_op(callee)) : _kmir_callee_ref(callee, ctx, shadow)
        return _OpCall(op, pos, kwp, bc || bcast, _kmir_call_hint(callee, ex, ctx))
    elseif ex.head === :macrocall && _kmir_is_dotmacro(ex)
        return _kmexpr(ex.args[end], ctx, shadow; bcast=true)
    elseif ex.head in (:&&, :||)
        return _Short(ex.head, m(ex.args[1]), m(ex.args[2]))
    elseif ex.head === :comparison
        operands = _MExpr[m(ex.args[i]) for i in 1:2:length(ex.args)]
        ops = Symbol[ex.args[i] for i in 2:2:length(ex.args)]
        # RK 06:08: each chained-comparison operator must resolve to a captured EXACT pure primitive by
        # its base identity (rebind-checked) — a rebound / locally-shadowed / unregistered `<` REJECTS,
        # never spelling-authorized. Store the captured provenance parallel to `ops`.
        refs = _CapturedCalleeRef[]; regs = _KernelRegistration[]
        for op in ops
            base = _kmir_dotted_op(op) ? _kmir_base_op(op) : op
            cc = _kmir_pure_op_captured(base, ctx)
            cc === nothing && _kmir_reject(
                "chained-comparison operator `$op` does not resolve to a captured exact pure primitive " *
                "(rebound / locally shadowed / unregistered) — it cannot be spelling-authorized")
            push!(refs, cc.ref); push!(regs, cc.registration)
        end
        return _Comparison(Tuple(operands), Tuple(ops), Tuple(refs), Tuple(regs))
    elseif ex.head === :if && length(ex.args) == 3
        return _IfExpr(m(ex.args[1]), m(ex.args[2]), m(ex.args[3]))
    elseif ex.head === :block
        stmts = [s for s in ex.args if !(s isa LineNumberNode)]
        isempty(stmts) && _kmir_reject("an empty value-position block has no result")
        eff = _MStmt[]; sh = copy(shadow)
        for s in stmts[1:end-1]
            node, nb = _kmstmt(s, ctx, sh); node === nothing || push!(eff, node); union!(sh, nb)
        end
        return _BlockExpr(Tuple(eff), _kmexpr(stmts[end], ctx, sh; bcast=bcast))
    elseif ex.head === :tuple
        pidx = findfirst(a -> a isa Expr && a.head === :parameters, ex.args)
        if pidx !== nothing
            names = Symbol[]; vals = _MExpr[]
            for k in ex.args[pidx].args
                nm_v = _kmir_kw_pair(k, ctx, shadow; bcast=bcast)
                push!(names, nm_v.first); push!(vals, nm_v.second)
            end
            return _NamedTuple(Tuple(names), Tuple(vals))
        end
        return _TupleExpr(Tuple(m(a) for a in ex.args))
    end
    _kmir_reject("expression form `$(ex.head)` is not compilable")
end

# ---- statement normalization -> (_MStmt-or-nothing, newly-bound names) -------

_kmir_stmts(node) = (node isa Expr && node.head === :block) ? node.args : Any[node]

function _kmstmt(ex, ctx::_KMIRCtx, shadow::Set{Symbol})
    ex isa LineNumberNode && return (nothing, Set{Symbol}())
    if ex isa Symbol || !(ex isa Expr)
        return (_ExprStmt(_kmexpr(ex, ctx, shadow)), Set{Symbol}())
    end
    if ex.head === :return
        if !isempty(ex.args) && ex.args[1] isa Expr && _kmir_is_assign(ex.args[1].head)
            a = ex.args[1]
            _kmir_is_place_lhs(a.args[1], ctx, shadow) ||
                _kmir_reject("a `return <local-assignment>` is not compilable")
            w = _kmir_placewrite(a.args[1], a.args[2], _kmir_compound_op(a.head),
                                 _kmir_is_dotassign(a.head), ctx, shadow)
            return (_SetReturn(w), Set{Symbol}())
        end
        val = isempty(ex.args) ? nothing : _kmexpr(ex.args[1], ctx, shadow)
        return (_Return(val), Set{Symbol}())
    elseif ex.head === :break
        return (_Break(), Set{Symbol}())
    elseif ex.head === :continue
        return (_Continue(), Set{Symbol}())
    elseif ex.head === :quote
        return (_ExprStmt(_kmexpr(ex, ctx, shadow)), Set{Symbol}())
    elseif ex.head === :macrocall && _kmir_is_dotmacro(ex)
        return _kmstmt_dotmacro(ex, ctx, shadow)
    elseif _kmir_is_assign(ex.head)
        return _kmstmt_assign(ex, ctx, shadow)
    elseif ex.head === :call
        return _kmstmt_call(ex, ctx, shadow)
    elseif ex.head === :if
        return _kmstmt_if(ex, ctx, shadow)
    elseif ex.head in (:&&, :||)
        return _kmstmt_guard(ex, ctx, shadow)
    elseif ex.head === :for
        return _kmstmt_for(ex, ctx, shadow)
    elseif ex.head === :while
        return _kmstmt_while(ex, ctx, shadow)
    end
    (_ExprStmt(_kmexpr(ex, ctx, shadow)), Set{Symbol}())
end

# --- owned-alias pre-scan + place-root resolution ---

# The owning FIELD (top of the receiver-rooted place path), or nothing.
function _kmir_place_owner_field(lhs, ctx::_KMIRCtx, shadow::Set{Symbol})
    p = _kmir_owner_path(lhs, ctx, shadow)
    (p === nothing || isempty(p)) ? nothing : (p[1],)
end
# Locals bound MORE THAN ONCE anywhere in the body — never a definite owned alias (reassignment).
function _kmir_reassigned_locals(stmts)
    reassigned = Set{Symbol}(); seen = Set{Symbol}()
    walk(x) = begin
        if x isa Expr && x.head !== :quote
            (_kmir_is_assign(x.head) && x.args[1] isa Symbol) &&
                ((x.args[1] in seen) ? push!(reassigned, x.args[1]) : push!(seen, x.args[1]))
            foreach(walk, x.args)
        end
        nothing
    end
    for s in stmts; walk(s); end
    reassigned
end
# (root, owner, alias) for a write LHS: `:self` (receiver-rooted, owner = top owned field), `:alias`
# (an owned-alias local), or `:deferred` (a property/index write we cannot own-resolve yet — e.g. a
# loop-element local `t` ranging over a field).
function _kmir_place_root(lhs, ctx::_KMIRCtx, shadow::Set{Symbol})
    ownf = _kmir_place_owner_field(lhs, ctx, shadow)
    ownf !== nothing && return (:self, ownf, nothing)
    r = _kmir_nav_root(lhs)
    (r isa Symbol && haskey(ctx.aliases, r)) && return (:alias, ctx.aliases[r], r)
    (:deferred, nothing, nothing)
end
# Build a `_PlaceWrite` (never rejected — a faithful direct in-object mutation).
function _kmir_placewrite(lhs, rhs, op, dot, ctx::_KMIRCtx, shadow::Set{Symbol})
    root, owner, alias = _kmir_place_root(lhs, ctx, shadow)
    target = _kmexpr(lhs, ctx, shadow)
    rexpr  = _kmexpr(rhs, ctx, shadow)
    if op !== nothing
        base = _kmir_dotted_op(op) ? _kmir_base_op(op) : op
        rexpr = _kmir_binop_expr(base, (_kmexpr(lhs, ctx, shadow), rexpr), dot || _kmir_dotted_op(op), ctx)
    end
    _PlaceWrite(target, root, owner, alias, rexpr, dot)
end

function _kmstmt_assign(ex, ctx::_KMIRCtx, shadow::Set{Symbol})
    lhs = ex.args[1]; rhs = ex.args[2]
    op  = _kmir_compound_op(ex.head)
    dot = _kmir_is_dotassign(ex.head)
    # tuple-destructure: a place-bearing multi-assign -> `_PlaceSwap`; else a plain-local rebind.
    if lhs isa Expr && lhs.head === :tuple
        if any(e -> _kmir_is_place_lhs(e, ctx, shadow), lhs.args)
            (op === nothing && !dot) || _kmir_reject("a compound/dotted tuple place-assignment is not compilable")
            (rhs isa Expr && rhs.head === :tuple && length(rhs.args) == length(lhs.args)) ||
                _kmir_reject("a place tuple-assignment needs a matching-arity tuple RHS")
            targets = Tuple(_kmir_placewrite(l, r, nothing, false, ctx, shadow)
                            for (l, r) in zip(lhs.args, rhs.args))
            return (_PlaceSwap(targets), Set{Symbol}())
        end
        dot && _kmir_reject("a dotted destructuring bind is not compilable")
        names = Symbol[]; _kmir_assign_names!(names, lhs)
        return (_LocalAssign(Tuple(names), _kmexpr(rhs, ctx, shadow)), Set{Symbol}(names))
    end
    # a PLACE write (field-rooted / owned-alias / indexed-nested / deferred) — never rejected.
    if _kmir_is_place_lhs(lhs, ctx, shadow)
        return (_kmir_placewrite(lhs, rhs, op, dot, ctx, shadow), Set{Symbol}())
    end
    # a plain method-local (re)bind — maintain SCOPE-LIVE owned aliases. A local bound ONCE to a SINGLE
    # field place (`tree = trees[depth]`) is a definite alias in this scope; a reassigned / multi-owner
    # / non-field binding is NOT — resolve `:deferred`.
    if lhs isa Symbol
        dot && _kmir_reject("a dotted assignment to a method-local `$(repr(lhs))` is not compilable")
        rexpr = _kmexpr(rhs, ctx, shadow)
        if op !== nothing
            base = _kmir_dotted_op(op) ? _kmir_base_op(op) : op
            rexpr = _kmir_binop_expr(base, (_kmexpr(lhs, ctx, shadow), rexpr), _kmir_dotted_op(op), ctx)
        end
        p = (op === nothing) ? _kmir_recv_path(_kmir_nav_root(rhs), ctx, shadow) : nothing
        # an alias binds to the RHS field root ONLY when the RHS is a field-rooted place (index/getfield
        # off a bare field), never a bare field copy or a non-field expression.
        aliasable = p !== nothing && !isempty(p) && rhs isa Expr && rhs.head in (:ref, :.) &&
                    !(lhs in ctx.reassigned)
        if aliasable
            ctx.aliases[lhs] = (p[1],)
        else
            delete!(ctx.aliases, lhs)
        end
        delete!(ctx.maybe, lhs)   # an UNCONDITIONAL rebind makes a previously maybe-bound local definite
        return (_LocalAssign((lhs,), rexpr), Set{Symbol}((lhs,)))
    end
    _kmir_reject("a mutation through `$(repr(lhs))` is not compilable")
end

function _kmir_assign_names!(names, lhs)
    if lhs isa Symbol
        push!(names, lhs)
    elseif lhs isa Expr && lhs.head === :(::)
        _kmir_assign_names!(names, lhs.args[1])
    elseif lhs isa Expr && lhs.head === :tuple
        for e in lhs.args; _kmir_assign_names!(names, e); end
    end
    names
end

function _kmstmt_dotmacro(ex, ctx::_KMIRCtx, shadow::Set{Symbol})
    inner = ex.args[end]
    if inner isa Expr && _kmir_is_assign(inner.head) && _kmir_is_place_lhs(inner.args[1], ctx, shadow)
        return (_kmir_placewrite(inner.args[1], inner.args[2], _kmir_compound_op(inner.head), true, ctx, shadow),
                Set{Symbol}())
    end
    _kmir_reject("`@.`/broadcast not writing a place (bare field / owned alias) is not compilable")
end

# Route a statement-position call — ONE representation, usable value or discarded-value:
#  (a) callee is a receiver CALLABLE-FIELD path (`step_f(ep)`) -> a `_FieldCall` (registered);
#  (b) callee is a SIBLING method (`flip!(__self__, …)`) -> a sibling `_Call`;
#  (c) any other callee (operator / registered / intrinsic `copy!!` / opaque external) -> `_OpCall`.
function _kmstmt_call(ex, ctx::_KMIRCtx, shadow::Set{Symbol})
    callee = ex.args[1]
    cp = _kmir_recv_path(callee, ctx, shadow)
    if cp !== nothing && !isempty(cp)
        return (_ExprStmt(_kmexpr(ex, ctx, shadow)), Set{Symbol}())    # callable-FIELD path
    elseif callee isa Symbol && callee in ctx.methods
        return (_kmstmt_sibling_call(ex, ctx, shadow), Set{Symbol}())  # sibling _Call
    end
    (_ExprStmt(_kmexpr(ex, ctx, shadow)), Set{Symbol}())               # external/registered _OpCall
end

# Is the RHS of a short-circuit an EFFECT (a guard body) rather than a value? An assignment /
# break/continue/return / SIBLING call / callable-FIELD call is an effect; a comparison/arithmetic/
# primitive/short-circuit is a VALUE.
function _kmir_effect_rhs(rhs, ctx::_KMIRCtx, shadow::Set{Symbol})
    rhs isa Expr || return false
    _kmir_is_assign(rhs.head) && return true
    rhs.head in (:break, :continue, :return) && return true
    rhs.head === :block && return any(s -> _kmir_effect_rhs(s, ctx, shadow), rhs.args)
    if rhs.head === :call && !isempty(rhs.args)
        callee = rhs.args[1]
        (callee isa Symbol && callee in ctx.methods) && return true                    # sibling call
        cp = _kmir_recv_path(callee, ctx, shadow); (cp !== nothing && !isempty(cp)) && return true  # field call
        _kmir_lookup_captured(callee, ctx) !== nothing && return true                  # registered/intrinsic call
    end
    false
end

# Resolve a sibling call to (name, candidate set, receiver target, positional actuals, kw). The FIRST
# positional is the callee's receiver actual (`__self__` implicit / the subject Mode-2) — stripped.
function _kmir_resolve_sibling(ex, ctx::_KMIRCtx, shadow::Set{Symbol})
    name = ex.args[1]
    rawpos = Any[]; kwp = Pair{Symbol,_MExpr}[]
    for a in ex.args[2:end]
        if a isa Expr && a.head === :parameters
            for k in a.args; push!(kwp, _kmir_kw_pair(k, ctx, shadow)); end
        elseif a isa Expr && a.head === :kw
            push!(kwp, a.args[1] => _kmexpr(a.args[2], ctx, shadow))
        elseif a isa Expr && a.head === :(...)
            _kmir_reject("a splatted positional argument in sibling call `$name` is not compilable")
        else
            push!(rawpos, a)
        end
    end
    isempty(rawpos) && _kmir_reject("sibling call `$name` needs a `__self__`/receiver first argument")
    target = _kmexpr(rawpos[1], ctx, shadow)
    posargs = Tuple(_kmexpr(a, ctx, shadow) for a in rawpos[2:end])
    kwnames = sort!(Symbol[p.first for p in kwp])
    candidates = _kmir_resolve_call(name, length(posargs), kwnames, ctx.specs)
    (name, candidates, target, posargs, Tuple(kwp))
end

function _kmstmt_sibling_call(ex, ctx::_KMIRCtx, shadow::Set{Symbol})
    nm, cands, tgt, ps, kw = _kmir_resolve_sibling(ex, ctx, shadow)
    _Call(nm, cands, tgt, ps, kw)
end

# Process a nested BLOCK body: owned aliases bound INSIDE the block are SCOPE-LOCAL.
function _kmbody_scoped(stmts, ctx::_KMIRCtx, shadow::Set{Symbol})
    saved = copy(ctx.aliases)
    body, binds = _kmbody(stmts, ctx, shadow)
    empty!(ctx.aliases); merge!(ctx.aliases, saved)
    (body, binds)
end

function _kmstmt_if(ex, ctx::_KMIRCtx, shadow::Set{Symbol})
    cond = _kmexpr(ex.args[1], ctx, shadow)
    thenb, tb = _kmbody_scoped(_kmir_stmts(ex.args[2]), ctx, copy(shadow))
    eb = _MStmt[]; ebinds = Set{Symbol}(); has_else = false
    if length(ex.args) == 3
        has_else = true
        e = ex.args[3]
        if e isa Expr && e.head === :elseif
            en, eb2 = _kmstmt_if(Expr(:if, e.args...), ctx, copy(shadow)); push!(eb, en); union!(ebinds, eb2)
        else
            eb, ebinds = _kmbody_scoped(_kmir_stmts(e), ctx, copy(shadow))
        end
    end
    # Finding 4 (sound definite assignment): a local is definitely bound after the `if` ONLY if bound in
    # BOTH arms AND an `else` exists. A one-arm / no-else bind does NOT escape as an unconditional local —
    # it becomes maybe-bound (a later read is rejected actionably). Both-arms binds remain definite.
    definite = has_else ? intersect(tb, ebinds) : Set{Symbol}()
    union!(ctx.maybe, setdiff(union(tb, ebinds), definite, shadow))
    (_If(cond, Tuple(thenb), Tuple(eb)), definite)
end

# A guard condition `isnothing(<field>)` over an owned field symbol -> that field (else nothing).
function _kmir_isnothing_field(cond_ast, ctx::_KMIRCtx)
    (cond_ast isa Expr && cond_ast.head === :call && length(cond_ast.args) == 2 &&
     cond_ast.args[1] === :isnothing && cond_ast.args[2] isa Symbol &&
     cond_ast.args[2] in ctx.fields) ? cond_ast.args[2] : nothing
end

function _kmstmt_guard(ex, ctx::_KMIRCtx, shadow::Set{Symbol})
    if _kmir_effect_rhs(ex.args[2], ctx, shadow)
        cond = _kmexpr(ex.args[1], ctx, shadow)   # cond evaluated BEFORE the body (field not yet optional)
        # Finding 6: ONLY `isnothing(f) ||` DOMINATES the body's call to f as registered-OR-NOTHING —
        # scoped to THIS guard body. `isnothing(f) &&` never marks f optional (it calls f only when f IS
        # nothing). Any call to f outside this dominated body stays registered-or-reject.
        optf = (ex.head === :(||)) ? _kmir_isnothing_field(ex.args[1], ctx) : nothing
        added = optf !== nothing && !(optf in ctx.optional_here)
        added && push!(ctx.optional_here, optf)
        body, binds = _kmbody_scoped(_kmir_stmts(ex.args[2]), ctx, copy(shadow))
        added && delete!(ctx.optional_here, optf)
        # Finding 4: a guard body MAY NOT RUN — its new binds do NOT escape as definite locals; they
        # become maybe-bound (a later read is rejected actionably).
        union!(ctx.maybe, setdiff(binds, shadow))
        return (_Guard(ex.head, cond, Tuple(body)), Set{Symbol}())
    end
    (_ExprStmt(_kmexpr(ex, ctx, shadow)), Set{Symbol}())
end

function _kmstmt_for(ex, ctx::_KMIRCtx, shadow::Set{Symbol})
    spec = ex.args[1]
    (spec isa Expr && spec.head === :(=)) || _kmir_reject("only `for x in itr` loops are compilable")
    varnames = Symbol[]; _kmir_assign_names!(varnames, spec.args[1])
    iter = _kmexpr(spec.args[2], ctx, shadow)
    body, binds = _kmbody_scoped(_kmir_stmts(ex.args[2]), ctx, union(shadow, Set{Symbol}(varnames)))
    (_For(Tuple(varnames), iter, Tuple(body)), Set{Symbol}())
end

function _kmstmt_while(ex, ctx::_KMIRCtx, shadow::Set{Symbol})
    cond = _kmexpr(ex.args[1], ctx, shadow)
    body, _ = _kmbody_scoped(_kmir_stmts(ex.args[2]), ctx, copy(shadow))
    (_While(cond, Tuple(body)), Set{Symbol}())
end

function _kmbody(stmts, ctx::_KMIRCtx, shadow::Set{Symbol})
    out = _MStmt[]; binds = Set{Symbol}(); sh = copy(shadow)
    for st in stmts
        st isa LineNumberNode && continue
        if st isa Expr && st.head === :block
            inner, ib = _kmbody(st.args, ctx, sh); append!(out, inner); union!(binds, ib); union!(sh, ib); continue
        end
        node, nb = _kmstmt(st, ctx, sh)
        node === nothing || push!(out, node)
        union!(binds, nb); union!(sh, nb)
    end
    (out, binds)
end

# ---- classification: control shape (PER DECLARATION) ------------------------

function _kmir_has_loop(node)
    node isa Expr || return false
    (node.head in (:for, :while)) && return true
    node.head === :quote && return false
    any(_kmir_has_loop, node.args)
end
function _kmir_contains_head(node, heads)
    node isa Expr || return false
    node.head === :quote && return false
    node.head in heads && return true
    any(a -> _kmir_contains_head(a, heads), node.args)
end

# Sibling CALL SITES of a body: (name, receiver-stripped positional arity, kw names) per call.
function _kmir_call_sites!(acc::Vector{Tuple{Symbol,Int,Vector{Symbol}}}, node, methods::Set{Symbol})
    node isa Expr || return acc
    node.head === :quote && return acc
    if node.head === :call && !isempty(node.args) && node.args[1] isa Symbol && node.args[1] in methods
        npos = 0; kwn = Symbol[]
        _kwname(k) = k isa Symbol ? k :
            (k isa Expr && k.head === :kw ? k.args[1] :
             k isa Expr && k.head === :(...) ? _KMIR_KWSPLAT :
             k isa Expr && k.head === :. && length(k.args) == 2 && k.args[2] isa QuoteNode ? k.args[2].value :
             nothing)
        for a in node.args[2:end]
            if a isa Expr && a.head === :parameters
                for k in a.args; (n = _kwname(k)) === nothing || push!(kwn, n); end
            elseif a isa Expr && a.head === :kw
                push!(kwn, a.args[1])
            else
                npos += 1
            end
        end
        push!(acc, (node.args[1], max(npos - 1, 0), sort!(kwn)))   # strip the receiver actual
    end
    for a in node.args; _kmir_call_sites!(acc, a, methods); end
    acc
end

function _kmir_match_ords(name::Symbol, npos::Int, kwnames::Vector{Symbol}, specs_ord)
    cands = get(specs_ord, name, Tuple{Int,MethodId}[])
    kwset = Set(kwnames)
    Int[o for (o, id) in cands if id.npos_req != -1 && _kmir_kwarity_match(id, npos, kwset)]
end

function _kmir_recursive_ordinals(adj::Vector{Vector{Int}}, N::Int)
    rec = Set{Int}()
    for i in 1:N
        seen = Set{Int}(); stack = copy(adj[i])
        while !isempty(stack)
            j = pop!(stack)
            j == i && (push!(rec, i); break)
            (j in seen) && continue
            push!(seen, j); append!(stack, adj[j])
        end
    end
    rec
end

# ---- LOCAL effect predicates (over the EMITTED IR) --------------------------

# The LHS of an assignment / `@.` is a PLACE (owned field / owned-alias / indexed-nested / deferred
# property) vs a plain-local rebind (a bare non-field Symbol). A receiver-rooted target, a property/
# index chain, or an alias local is a place.
function _kmir_is_place_lhs(lhs, ctx::_KMIRCtx, shadow::Set{Symbol})
    lhs isa Expr && lhs.head === :tuple && return any(e -> _kmir_is_place_lhs(e, ctx, shadow), lhs.args)
    _kmir_recv_path(lhs, ctx, shadow) !== nothing && return true        # bare field / receiver-rooted
    (lhs isa Expr && lhs.head in (:., :ref)) && return true             # property/index place
    (lhs isa Symbol && haskey(ctx.aliases, lhs)) && return true         # owned-alias local
    false
end
_kmir_has_branch(node) = _kmir_contains_head(node, (:if, :elseif, :&&, :||))

# The `control` shape of each method DECLARATION + a PRELIMINARY control-only secondary kind. The
# COMPLETE effects/deps/FINAL kind are computed from the EMITTED IR, never from the AST.
function _kmir_controls(method_bodies, methods::Set{Symbol}, specs_ord)
    N = length(method_bodies)
    sites = [_kmir_call_sites!(Tuple{Symbol,Int,Vector{Symbol}}[], b, methods) for b in method_bodies]
    resolved = [[_kmir_match_ords(s[1], s[2], s[3], specs_ord) for s in sites[i]] for i in 1:N]
    adj = [Int[] for _ in 1:N]
    for i in 1:N, ords in resolved[i]; length(ords) == 1 && push!(adj[i], ords[1]); end
    rec = _kmir_recursive_ordinals(adj, N)
    controls = [(i in rec) ? :recursive : _kmir_has_loop(b) ? :loop :
                _kmir_has_branch(b) ? :branch : :straight for (i, b) in enumerate(method_bodies)]
    prelim = [c in (:loop, :recursive) ? :orchestration : :segment for c in controls]
    (controls, prelim)
end

# Compute the COMPLETE local effect SET + resolution DEPENDENCIES + secondary kind from an EMITTED body.
# `effects` ⊆ (:place_write, :opaque_call). Deps carry HYGIENIC identity:
#   (:operator_trait, GlobalRef) | (:opaque_call, GlobalRef) |
#   (:intrinsic_call, GlobalRef) | (:registered_call, GlobalRef) | (:registered_field, path) |
#   (:registered_or_nothing_field, path) | (:sibling_overload, name) | (:deferred_alias, alias-or-target).
# One node's contribution to the (effects, deps) census — shared by body statements AND formal-default
# expression trees (RK 06:21b), so a default's callees are never a hidden opaque edge.
function _kmir_node_fact!(effects, deps, n)
    if n isa _PlaceWrite
        push!(effects, :place_write)
        (n.root === :deferred) && push!(deps, (:deferred_alias, n.alias === nothing ? n.target : n.alias))
    elseif n isa _OpCall
        if n.hint === :opaque
            push!(effects, :opaque_call); push!(deps, (:opaque_call, n.op))
        else                                          # :operator_candidate — a HINT, not proof
            push!(deps, (:operator_trait, n.op))
        end
    elseif n isa _RegisteredCall
        # the CAPTURED Token is the identity (finding 2); its effect footprint is resolved later
        # from the registration's roots — no falsely-final local effect.
        push!(deps, (n.intrinsic ? :intrinsic_call : :registered_call, n.registration.token))
    elseif n isa _Comparison
        # RK 06:08: a chained comparison's operators carry captured EXACT pure-primitive identities
        # (no opaque edge, no spelling authority) — record each as a registered-call dep.
        for r in n.regs; push!(deps, (:registered_call, r.token)); end
    elseif n isa _SubjectMethodCall
        push!(deps, (:subject_method, n.name))         # resolved at construction (Token/MethodId)
    elseif n isa _FieldCall
        push!(deps, (n.hint === :registered_or_nothing ? :registered_or_nothing_field :
                     :registered_field, n.path))
    elseif (n isa _Call || n isa _CallExpr) && length(n.candidates) > 1
        push!(deps, (:sibling_overload, n.name))
    end
    nothing
end

function _kmir_facts_from_body(body, control::Symbol, defaults = ())
    effects = Set{Symbol}(); deps = Any[]
    for s in body;      _kmir_walk((n) -> _kmir_node_fact!(effects, deps, n), s); end
    for d in defaults;  _kmir_walk((n) -> _kmir_node_fact!(effects, deps, n), d); end   # RK 06:21b
    deps = unique(deps)
    undecided = any(d -> d isa Tuple && d[1] in (:opaque_call, :sibling_overload, :deferred_alias), deps)
    kind = control in (:loop, :recursive) ? :orchestration : undecided ? :unresolved : :segment
    (Tuple(sort!(collect(effects))), Tuple(deps), kind)
end

# ---- per-method emission ----------------------------------------------------

# One method descriptor for emission: name, the peeled `:call`, the body AST, the full signature, and
# whether the FIRST positional is a receiver subject to strip (Mode-2) — implicit-field strips nothing.
struct _KMIRMethodDesc
    name::Symbol
    call::Any
    body::Any
    signature::Any
    strip_first::Bool
end

# Build the MethodIR VALUE for one method DECLARATION.
function _kmir_emit_method(m::_KMIRMethodDesc, decl::Int, control::Symbol, prelim_kind::Symbol,
                           self::Symbol, mode::Symbol, fields::Set{Symbol}, methods::Set{Symbol},
                           specs::Dict{Symbol,Vector{_CallCandidate}}, mod::Module, callee_regs::Tuple)
    name = m.name
    _mir(id, formals, body, ok, reason; effects=(), deps=(), kind=prelim_kind) =
        MethodIR(id, self, formals, body, control, effects, deps, kind, ok, reason, m.signature)
    raw = try
        _kmir_raw_formals(m.call, m.strip_first)
    catch err
        err isa _KMIRReject || rethrow()
        return _mir(_kmethodid_opaque(name, decl), (), (), false, err.reason)
    end
    wherevars = _kmir_where_vars(m.signature)
    id = _kmir_methodid(name, decl, raw, mod, wherevars, _kmir_wheres(m.signature))
    formalpos = Dict{Symbol,Int}(d[1] => i for (i, d) in enumerate(raw))
    formalkind = Dict{Symbol,Symbol}(d[1] => d[2] for d in raw)
    plain_formals = Tuple(_Formal(d[1], d[2], d[3], _kmir_type_ref(d[4], mod, wherevars), nothing) for d in raw)
    try
        # DEFAULTS left-to-right: the i-th default sees ONLY the formals declared before it + owner
        # FIELDS (an implicit-field default may read a field, e.g. `j = length(proposals)`).
        formals = _Formal[]
        priorpos = Dict{Symbol,Int}(); priorkind = Dict{Symbol,Symbol}()
        for (i, d) in enumerate(raw)
            def = nothing
            if d[5] !== nothing
                dctx = _KMIRCtx(fields, methods, specs, copy(priorpos), copy(priorkind), self, mod,
                                prelim_kind, mode, callee_regs)
                def = _kmexpr(d[5], dctx, Set{Symbol}())
            end
            push!(formals, _Formal(d[1], d[2], d[3], _kmir_type_ref(d[4], mod, wherevars), def))
            priorpos[d[1]] = i; priorkind[d[1]] = d[2]
        end
        stmts = collect(_kmir_stmts(m.body))
        ctx = _KMIRCtx(fields, methods, specs, formalpos, formalkind, self, mod, prelim_kind, mode,
                       callee_regs, Set{Symbol}(), Set{Symbol}(), Dict{Symbol,Tuple{Vararg{Symbol}}}(),
                       _kmir_reassigned_locals(stmts))
        body, binds = _kmbody(stmts, ctx, Set{Symbol}())
        if !isempty(body) && body[end] isa _ExprStmt
            body[end] = _Return(body[end].expr)
        elseif !isempty(body) && body[end] isa _Call
            c = body[end]
            body[end] = _Return(_CallExpr(c.name, c.candidates, c.target, c.pos, c.kw))
        end
        defexprs = Any[f.default for f in formals if f.default !== nothing]
        effects, deps, kind = _kmir_facts_from_body(body, control, defexprs)   # RK 06:21b: fold defaults
        return _mir(id, Tuple(formals), Tuple(body), true, nothing; effects, deps, kind)
    catch err
        err isa _KMIRReject || rethrow()
        return _mir(id, plain_formals, (), false, err.reason)
    end
end

# The object's field set from the substrate's DETACHED IMMUTABLE port-name accessor (finding 11) — the
# authored field/port names captured at definition, NEVER a live-mutable shared `KernelSpec.ports`.
_kmir_field_set(skel) = Set{Symbol}(kernel_port_names(skel))

# ---- structured, ordered ACCESS EVENTS (the sound schedule input) -----------
# Branch-scoped groups (mutually-exclusive then/else), loop-carry marking (a for/while body MAY run
# zero times and carries across iterations), guard bodies (may-not-run), and EVERY call's argument
# reads (a call is a read of its actuals + a resolvable effect). This lets a later schedule derive
# sound read-before-first-write across mutually-exclusive branches and zero-iteration loops — NOT a
# flat stream of two concatenated branch bodies.

abstract type _Access end
"A value/address READ of a receiver field path."
struct _ARead   <: _Access; path::Tuple{Vararg{Symbol}}; end
"A WRITE to a place root `(root, owner)`."
struct _AWrite  <: _Access; root::Symbol; owner::Any; end
"A CALL effect (`kind` ∈ :sibling/:field/:registered/:intrinsic/:subject_method/:opaque/:operator) with
its ordered argument-read events (`reads`). Every call IS a read of its actuals. RNG is never a call
kind — it is not name-classified (finding 10)."
struct _ACall   <: _Access; kind::Symbol; id::Any; reads::Vector{_Access}; end
"Two MUTUALLY-EXCLUSIVE ordered event groups (`thn`/`els`) — an `if`/ternary branch. A read in one arm
is NOT sequenced before a write in the other."
struct _ABranch <: _Access; thn::Vector{_Access}; els::Vector{_Access}; end
"A LOOP body's events — `may_run_zero` is always true for a `for`/`while` (the body may not execute)
and the body is loop-CARRIED (a read may see a prior iteration's write, or the pre-loop value)."
struct _ALoop   <: _Access; body::Vector{_Access}; may_run_zero::Bool; end
"A short-circuit GUARD body that MAY NOT RUN (`cond && body` / `cond || body`), with the `cond` reads."
struct _AGuard  <: _Access; cond::Vector{_Access}; body::Vector{_Access}; end

# value-position reads (ordered) of an _MExpr into `ev`.
function _kmir_val_reads!(ev, x)
    if x isa _SelfField
        push!(ev, _ARead(x.path))
    elseif x isa _FieldCall
        rd = _Access[]
        for a in x.pos; _kmir_val_reads!(rd, a); end; for p in x.kw; _kmir_val_reads!(rd, p.second); end
        push!(ev, _ACall(:field, x.path, rd))
    elseif x isa _OpCall
        rd = _Access[]
        for a in x.args; _kmir_val_reads!(rd, a); end; for p in x.kw; _kmir_val_reads!(rd, p.second); end
        push!(ev, _ACall(_kmir_opcall_kind(x), x.op, rd))
    elseif x isa _RegisteredCall
        rd = _Access[]
        for a in x.args; _kmir_val_reads!(rd, a); end; for p in x.kw; _kmir_val_reads!(rd, p.second); end
        push!(ev, _ACall(x.intrinsic ? :intrinsic : :registered, x.registration.token, rd))
    elseif x isa _SubjectMethodCall
        rd = _Access[]
        _kmir_val_reads!(rd, x.subject)
        for a in x.pos; _kmir_val_reads!(rd, a); end; for p in x.kw; _kmir_val_reads!(rd, p.second); end
        push!(ev, _ACall(:subject_method, x.name, rd))
    elseif x isa _CallExpr
        rd = _Access[]
        _kmir_val_reads!(rd, x.target)
        for a in x.pos; _kmir_val_reads!(rd, a); end; for p in x.kw; _kmir_val_reads!(rd, p.second); end
        push!(ev, _ACall(:sibling, x.name, rd))
    elseif x isa _BlockExpr
        for st in x.stmts; _kmir_access_stmt!(ev, st); end
        _kmir_val_reads!(ev, x.value)
    elseif x isa _IfExpr
        _kmir_val_reads!(ev, x.cond)
        tb = _Access[]; eb = _Access[]
        _kmir_val_reads!(tb, x.thenv); _kmir_val_reads!(eb, x.elsev)
        push!(ev, _ABranch(tb, eb))
    elseif x isa _Short
        _kmir_val_reads!(ev, x.lhs)
        rb = _Access[]; _kmir_val_reads!(rb, x.rhs)
        push!(ev, _AGuard(_Access[], rb))                # RHS may not run (short-circuit)
    elseif x isa _Comparison
        # RK 06:08/06:17: `a < b <= c` is `(a<b) && (b<=c)` — a SHORT-CIRCUIT chain, the shared middle
        # operand evaluated ONCE. op1 is a REGISTERED pure call reading operands 1,2 GUARANTEED; each
        # later op reads only its NEW operand (the middle reused) under a progressively nested `_AGuard`
        # (may not run). Faithful conditional reads, no opaque/external edge, no spelling authority.
        n = length(x.regs)
        rd1 = _Access[]
        _kmir_val_reads!(rd1, x.operands[1]); _kmir_val_reads!(rd1, x.operands[2])
        push!(ev, _ACall(:registered, x.regs[1].token, rd1))
        inner = _Access[]                                # build the guard chain innermost-first
        for i in n:-1:2
            body = _Access[]
            rdi = _Access[]; _kmir_val_reads!(rdi, x.operands[i + 1])
            push!(body, _ACall(:registered, x.regs[i].token, rdi))
            isempty(inner) || push!(body, _AGuard(_Access[], inner))
            inner = body
        end
        n >= 2 && push!(ev, _AGuard(_Access[], inner))
    elseif x isa _MExpr
        for fn in fieldnames(typeof(x)); _kmir_vr_field!(ev, getfield(x, fn)); end
    end
    nothing
end
_kmir_opcall_kind(x::_OpCall) = x.hint === :opaque ? :opaque : :operator
_kmir_vr_field!(ev, v::_MExpr) = _kmir_val_reads!(ev, v)
_kmir_vr_field!(ev, v::Tuple) = (foreach(e -> _kmir_vr_field!(ev, e), v); nothing)
_kmir_vr_field!(ev, v::Pair) = _kmir_vr_field!(ev, v.second)
_kmir_vr_field!(ev, v) = nothing
# ADDRESS reads of a write TARGET — index + container navigation, but NOT the terminal stored place.
function _kmir_addr_reads!(ev, target, top::Bool)
    if target isa _SelfField
        top || push!(ev, _ARead(target.path))
    elseif target isa _Index
        _kmir_addr_reads!(ev, target.base, false); for i in target.idxs; _kmir_val_reads!(ev, i); end
    elseif target isa _Getfield
        _kmir_addr_reads!(ev, target.base, false)
    else
        _kmir_val_reads!(ev, target)
    end
    nothing
end
function _kmir_access_stmt!(ev, s)
    if s isa _PlaceWrite
        _kmir_addr_reads!(ev, s.target, true); _kmir_val_reads!(ev, s.rhs)
        push!(ev, _AWrite(s.root, s.owner))
    elseif s isa _PlaceSwap
        # Julia tuple assignment: evaluate ALL RHS sources first, THEN all LHS addresses, THEN store.
        for t in s.targets; _kmir_val_reads!(ev, t.rhs); end            # (1) all RHS value reads
        for t in s.targets; _kmir_addr_reads!(ev, t.target, true); end  # (2) all LHS address reads
        for t in s.targets; push!(ev, _AWrite(t.root, t.owner)); end    # (3) all stores
    elseif s isa _SetReturn
        _kmir_access_stmt!(ev, s.write)
    elseif s isa _Call
        rd = _Access[]
        _kmir_val_reads!(rd, s.target)
        for a in s.pos; _kmir_val_reads!(rd, a); end; for p in s.kw; _kmir_val_reads!(rd, p.second); end
        push!(ev, _ACall(:sibling, s.name, rd))
    elseif s isa _If
        _kmir_val_reads!(ev, s.cond)
        tb = _Access[]; eb = _Access[]
        for b in s.thenb; _kmir_access_stmt!(tb, b); end
        for b in s.elseb; _kmir_access_stmt!(eb, b); end
        push!(ev, _ABranch(tb, eb))
    elseif s isa _For
        _kmir_val_reads!(ev, s.iter)
        bb = _Access[]; for b in s.body; _kmir_access_stmt!(bb, b); end
        push!(ev, _ALoop(bb, true))
    elseif s isa _While
        # the INITIAL condition is evaluated exactly ONCE, guaranteed, BEFORE the body — even for a
        # zero-iteration loop. Emit it OUTSIDE the loop group; the loop body then carries (body + the
        # end-of-iteration RETEST of the condition), may-run-zero.
        _kmir_val_reads!(ev, s.cond)                                    # guaranteed initial condition
        bb = _Access[]
        for b in s.body; _kmir_access_stmt!(bb, b); end
        _kmir_val_reads!(bb, s.cond)                                    # loop-carried retest
        push!(ev, _ALoop(bb, true))
    elseif s isa _Guard
        cb = _Access[]; _kmir_val_reads!(cb, s.cond)
        bb = _Access[]; for b in s.body; _kmir_access_stmt!(bb, b); end
        push!(ev, _AGuard(cb, bb))
    elseif s isa _Return
        s.value === nothing || _kmir_val_reads!(ev, s.value)
    elseif s isa _LocalAssign
        _kmir_val_reads!(ev, s.rhs)
    elseif s isa _ExprStmt
        _kmir_val_reads!(ev, s.expr)
    end
    nothing
end

"Ordered STRUCTURED LOCAL ACCESS EVENTS of a method IR — a `Vector{_Access}` of `_ARead`/`_AWrite`/
`_ACall`/`_ABranch`/`_ALoop`/`_AGuard`. Branch arms are MUTUALLY EXCLUSIVE groups, loop bodies are
carried + may-run-zero, guard bodies may not run, and every call carries its argument reads — the
SOUND input (never a flat two-branch concatenation) to the later read-before-first-write / loop-carried
/ write-only schedule derivation."
function access_events(ir::MethodIR)
    ev = _Access[]
    # DEFAULT expressions (RK 06:21b) evaluate at entry, left-to-right, and MAY NOT RUN (the caller may
    # supply the arg) — each is a guarded read/call prelude, never an unconditional read.
    for f in ir.formals
        f.default === nothing && continue
        db = _Access[]; _kmir_val_reads!(db, f.default)
        isempty(db) || push!(ev, _AGuard(_Access[], db))
    end
    for s in ir.body; _kmir_access_stmt!(ev, s); end
    ev
end
# Recursively flatten the structured events into a flat leaf stream (reads/writes/calls).
function _kmir_flatten!(out, ev)
    for e in ev
        if e isa _ARead || e isa _AWrite
            push!(out, e)
        elseif e isa _ACall
            _kmir_flatten!(out, e.reads); push!(out, e)
        elseif e isa _ABranch
            _kmir_flatten!(out, e.thn); _kmir_flatten!(out, e.els)
        elseif e isa _ALoop
            _kmir_flatten!(out, e.body)
        elseif e isa _AGuard
            _kmir_flatten!(out, e.cond); _kmir_flatten!(out, e.body)
        end
    end
    out
end
_kmir_leaves(ir::MethodIR) = _kmir_flatten!(_Access[], access_events(ir))

"Distinct LOCAL write place-ROOTS `(root, owner)` of a method IR (over the structured access events)."
write_roots(ir::MethodIR) = unique(Any[(e.root, e.owner) for e in _kmir_leaves(ir) if e isa _AWrite])
"Distinct LOCAL field read ROOTS of a method IR — value reads + write-target ADDRESS reads + call
argument reads, but NOT a write-only terminal place (over the structured access events)."
read_roots(ir::MethodIR) = unique(Any[e.path for e in _kmir_leaves(ir) if e isa _ARead])
"LOCAL call EDGES of a method IR — EVERY call requiring later resolution, with HYGIENIC identity:
`(:sibling, name, n_candidates)` / `(:field, path, hint)` / `(:registered|:intrinsic, ref, token)` /
`(:subject_method, name)` / `(:external, op::GlobalRef, hint)`. NOT transitively closed."
function call_edges(ir::MethodIR)
    out = Any[]
    nodes = Any[ir.body...]
    for f in ir.formals; f.default === nothing || push!(nodes, f.default); end   # RK 06:21b: default callees
    for s in nodes
        _kmir_walk(s) do n
            n isa _Call            && push!(out, (:sibling, n.name, length(n.candidates)))
            n isa _CallExpr        && push!(out, (:sibling, n.name, length(n.candidates)))
            n isa _FieldCall       && push!(out, (:field, n.path, n.hint))
            n isa _RegisteredCall  && push!(out, (n.intrinsic ? :intrinsic : :registered, n.ref, n.registration.token))
            n isa _SubjectMethodCall && push!(out, (:subject_method, n.name))
            n isa _OpCall          && push!(out, (:external, n.op, n.hint))
            n isa _Comparison      && foreach((r, g) -> push!(out, (:registered, r, g.token)), n.refs, n.regs)
        end
    end
    unique(out)
end

# ---- IR tree walk -----------------------------------------------------------

_kmir_walk(f, x::Union{_MExpr,_MStmt}) =
    (f(x); for fn in fieldnames(typeof(x)); _kmir_walk_field(f, getfield(x, fn)); end; nothing)
_kmir_walk_field(f, v::Union{_MExpr,_MStmt}) = _kmir_walk(f, v)
_kmir_walk_field(f, v::Tuple) = (foreach(e -> _kmir_walk_field(f, e), v); nothing)
_kmir_walk_field(f, v::Pair) = (_kmir_walk_field(f, v.second); nothing)
_kmir_walk_field(f, v) = nothing

# ---- method_irs entry -------------------------------------------------------

"""
    method_irs(skel_or_object) -> Tuple{Vararg{MethodIR}}

The immutable ordered `MethodIR` table for a `@kernel` definition (one entry per authored method
DECLARATION), emitted from the substrate's stored method/body ASTs. Consumed by the later
graph-Value/effect lowering / trait-resolution / schedule-gen increments. `()` for a methodless
(stateless) value.
"""
function method_irs end
method_irs(::Any) = ()

function method_irs(skel::_StatefulKernelSkeleton)
    meta = kernel_methods(skel)
    isempty(meta) && return ()
    mod = kernel_module(skel)
    fields = _kmir_field_set(skel)                       # finding 11: detached immutable port names
    regs = kernel_callee_registrations(skel)             # finding 2: def-time registration snapshot
    methods = Set{Symbol}(m.name for m in meta)
    descs = [_KMIRMethodDesc(m.name, m.call, m.body, m.signature, false) for m in meta]
    _method_irs_common(descs, _KMIR_SELFACTUAL, :implicit, fields, methods, mod, regs)
end

function method_irs(skel::_Mode2KernelSkeleton)
    subject = kernel_subject(skel)
    mod = kernel_module(skel)
    body = kernel_recipe_ast(skel)                       # the free-method body (Expr :block)
    regs = kernel_callee_registrations(skel)             # finding 2: def-time registration snapshot
    # Finding 3: CONSUME the substrate's frozen raw Mode-2 signature/call (`kernel_methods` now carries
    # the deeply-frozen `signature`/`call`, thawed fresh) — never synthesize from `keys(spec.ports)`
    # (lossy on pos-vs-kw / default / type / order / where / return).
    m = kernel_methods(skel)[1]
    fields = Set{Symbol}()                               # Mode-2 has no owned fields — subject is explicit
    descs = [_KMIRMethodDesc(m.name, m.call, body, m.signature, true)]
    _method_irs_common(descs, subject, :mode2, fields, Set{Symbol}(), mod, regs)
end
kernel_name(skel::_Mode2KernelSkeleton) = getfield(skel, :name)
kernel_name(skel::_StatefulKernelSkeleton) = getfield(skel, :name)

# Shared emission pipeline: ids -> control shapes -> per-method IR.
function _method_irs_common(descs::Vector{_KMIRMethodDesc}, self::Symbol, mode::Symbol,
                            fields::Set{Symbol}, methods::Set{Symbol}, mod::Module, callee_regs::Tuple)
    ids = Vector{MethodId}(undef, length(descs))
    specs_ord = Dict{Symbol,Vector{Tuple{Int,MethodId}}}()
    for (i, d) in enumerate(descs)
        wherevars = _kmir_where_vars(d.signature)
        raw = try _kmir_raw_formals(d.call, d.strip_first) catch err
            err isa _KMIRReject ? nothing : rethrow()
        end
        ids[i] = raw === nothing ? _kmethodid_opaque(d.name, i) :
                 _kmir_methodid(d.name, i, raw, mod, wherevars, _kmir_wheres(d.signature))
        push!(get!(specs_ord, d.name, Tuple{Int,MethodId}[]), (i, ids[i]))
    end
    controls, prelim = _kmir_controls([d.body for d in descs], methods, specs_ord)
    specs = Dict{Symbol,Vector{_CallCandidate}}()
    for (i, d) in enumerate(descs)
        push!(get!(specs, d.name, _CallCandidate[]), _CallCandidate(ids[i], prelim[i]))
    end
    Tuple(_kmir_emit_method(d, i, controls[i], prelim[i], self, mode, fields, methods, specs, mod, callee_regs)
          for (i, d) in enumerate(descs))
end
