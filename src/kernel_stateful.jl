# Stateful `@kernel` authoring — Increment 1 SUBSTRATE (V7 implicit-field pivot).
#
# `@kernel` has THREE modes (discriminated in `authoring.jl`): a methodless recipe
# body ⇒ byte-identical stateless `KernelSpec`; a body with nested methods ⇒ a
# stateful OBJECT kernel with an IMPLICIT synthesized receiver (no `self` formal,
# bare fields, `__self__` only as a sibling object-pass actual); a methodless body
# mutating its first positional subject's fields — or any `!!` name — ⇒ a free
# METHOD. This increment establishes ONLY the substrate: the discriminators, the
# unique per-definition Token, the `KernelObject`/`KernelView` skeletons, implicit-
# receiver + Mode-2 method extraction, the frozen detached child/recipe/body source,
# the registered-kernel/intrinsic resolver + hygiene/rebind, and the
# `@node`/`deepcopy`/`partial`/`!!` source-marker recognition.
#
# It deliberately contains NO effect-lowering: no MethodIR/SSA, no `compile_update`
# consumption, no epoch codegen, no legacy-facade deletion, no port migration, no
# factory/execution. Those are later increments (poc's effect-lowering lane retargets
# MethodIR onto this substrate only after this SHA clears review).

# --- (1)+(2) method-presence discriminator ----------------------------------

# Peel a `where` clause and a return `::` annotation off a method signature to
# reach its underlying `:call`, so `m!(self::T, x) where {T}` and
# `function m!(self)::Nothing … end` are still recognised as methods.
function _kernel_peel_signature(sig)
    while sig isa Expr && (sig.head === :where ||
          (sig.head === :(::) && length(sig.args) == 2))
        sig = sig.args[1]
    end
    sig
end

# A body statement is an inner METHOD DEFINITION when its (peeled) signature is a
# `:call` — SHORT form `f!(self, …) = …` or LONG form `function f!(self, …) … end`.
# A recipe/input has a Symbol / tuple / `::`-typed-Symbol LHS, which never peels to
# a `:call`, keeping the two disjoint.
function _kernel_stmt_method_form(stmt)
    if stmt isa Expr && stmt.head === :(=) && length(stmt.args) == 2
        s = _kernel_peel_signature(stmt.args[1])
        s isa Expr && s.head === :call && return :short
    elseif stmt isa Expr && stmt.head === :function && length(stmt.args) == 2
        s = _kernel_peel_signature(stmt.args[1])
        s isa Expr && s.head === :call && return :long
    end
    nothing
end

_kernel_body_statements(block) =
    block isa Expr && block.head === :block ? block.args : Any[block]

"Does a `@kernel` body define at least one inner method (⇒ stateful)?"
function _kernel_body_has_methods(block)
    for stmt in _kernel_body_statements(block)
        _kernel_is_line(stmt) && continue
        _kernel_stmt_method_form(stmt) === nothing || return true
    end
    false
end

# --- (2) explicit-self stateful object + typed view skeletons ---------------

# A method-bearing `@kernel` object. `Token` is the unique per-definition type
# minted at expansion (below), so method dispatch + overload identity are typed
# and never collide across definitions. `State`/`Handles` are placeholders filled
# by later increments (the flattened compiled state + typed value handles).
struct KernelObject{Token,State,Handles}
    state::State
    handles::Handles
end

kernel_token(::KernelObject{Token}) where {Token} = Token
kernel_token(::Type{<:KernelObject{Token}}) where {Token} = Token

# A typed VIEW into an owner `KernelObject`: `Parent` is the owner's concrete type,
# `Path` a compile-time view path (e.g. `:fwd`). It carries the owner by value so a
# call like `leapfrog!(sampler.fwd)` selects the owner-owned per-path schedule from
# the `Path` TYPE PARAMETER — a compile-time selection, never a runtime lookup.
struct KernelView{Parent,Path}
    parent::Parent
end

kernel_view_path(::KernelView{Parent,Path}) where {Parent,Path} = Path
kernel_view_parent(view::KernelView) = getfield(view, :parent)

# --- (4)+(5) short/long method extraction, IMPLICIT-RECEIVER contract --------
#
# V7 implicit-field pivot (RK 2026-08-26): a nested method declares NO receiver
# formal. Bare unshadowed names are the owner's fields (poc's IR resolves
# field-vs-local by lexical scope from the stored AST + the recipe-portion
# `kernel_spec` ports); a lexical local/formal shadows a field. The ONLY
# object-pass spelling is `__self__` as the FIRST positional actual of a sibling
# call (`flip!(__self__, depth)`), recorded here as a receiver edge. A
# `self`/`__self__` FORMAL, or `__self__` used as a field qualifier / anywhere
# but an object-pass actual, is REJECTED.

struct _KernelMethod
    name::Symbol
    form::Symbol             # :short | :long
    argnames::Tuple{Vararg{Symbol}}   # SCALAR positional formal names (implicit receiver — NO self)
    vararg::Union{Symbol,Nothing}       # positional slurp `xs...` name, or nothing
    kwargs_splat::Union{Symbol,Nothing} # keyword slurp `; kwargs...` name, or nothing
    signature::Any           # the FULL AUTHORED signature AST (where binders +
                             #   constraints + return `::` annotation + splats preserved)
    call::Any                # the peeled `:call` (name + formals, splats retained
                             #   verbatim for forwarding) used for extraction
    body::Any                # the raw method-body AST (bare fields; handed to poc's emission)
    sibling_calls::Tuple{Vararg{Symbol}}  # callee names invoked with `__self__` (receiver edges)
end

# A vararg/kwargs SLURP formal (`xs...` positionally, `; kwargs...` in a keyword
# `:parameters` block) is `Expr(:..., inner)`. It is a bound name but NOT a scalar
# value, so callers record it separately from ordinary argnames.
_kernel_is_splat(a) = a isa Expr && a.head === :...

_kernel_arg_name(a) =
    a isa Symbol ? a :
    (a isa Expr && a.head === :...) ? _kernel_arg_name(a.args[1]) :
    (a isa Expr && a.head === :(::) && a.args[1] isa Symbol) ? a.args[1] :
    (a isa Expr && a.head === :kw && a.args[1] isa Symbol) ? a.args[1] :
    (a isa Expr && a.head === :kw && a.args[1] isa Expr &&
        a.args[1].head === :(::) && a.args[1].args[1] isa Symbol) ? a.args[1].args[1] :
    (a isa Expr && a.head === :(::) && a.args[1] isa Expr &&
        a.args[1].head === :kw) ? a.args[1].args[1] : nothing

# Validate/recognize `__self__` usage in a method body and collect the sibling
# receiver edges. `__self__` is legal ONLY as the FIRST positional actual of a
# call (`sibling!(__self__, …)`); `__self__.field` (field qualifier) and any other
# bare `__self__` are rejected. Returns the tuple of callee names so invoked.
function _kernel_self_usage(mname::Symbol, body)
    callees = Symbol[]
    local walk
    walk = function (x, self_ok::Bool)
        if x === :__self__
            self_ok || throw(ArgumentError(
                "stateful @kernel method :$mname uses `__self__` outside an object-pass call " *
                "actual — it is only the synthetic receiver in `sibling!(__self__, …)`"))
            return
        end
        x isa Expr || return
        if x.head === :(.) && !isempty(x.args) && x.args[1] === :__self__
            throw(ArgumentError(
                "stateful @kernel method :$mname uses `__self__.` as a field qualifier; " *
                "fields are BARE (no `self.`/`__self__.`) and `__self__` is only an object-pass actual"))
        elseif x.head === :call
            callee = x.args[1]
            pos = Any[]
            kws = Any[]
            for a in x.args[2:end]
                (a isa Expr && a.head === :parameters) ? append!(kws, a.args) : push!(pos, a)
            end
            if !isempty(pos) && pos[1] === :__self__ && callee isa Symbol
                push!(callees, callee)
            end
            walk(callee, false)
            for (i, a) in enumerate(pos)
                walk(a, i == 1)          # only the first positional slot admits a bare `__self__`
            end
            for a in kws
                walk(a, false)
            end
        else
            for a in x.args
                walk(a, false)
            end
        end
        return
    end
    walk(body, false)
    Tuple(callees)
end

function _kernel_extract_method(stmt, form)
    raw_sig = stmt.args[1]                       # full authored signature (unpeeled)
    call = _kernel_peel_signature(raw_sig)       # the underlying :call
    body = stmt.args[2]
    name = call.args[1]
    name isa Symbol || throw(ArgumentError(
        "stateful @kernel method must have a plain name; got $(repr(name))"))
    formals = call.args[2:end]
    # Julia places a keyword `:parameters` block BEFORE the positional formals.
    kwparams = Any[]
    positionals = formals
    if !isempty(formals) && formals[1] isa Expr && formals[1].head === :parameters
        kwparams = formals[1].args
        positionals = formals[2:end]
    end
    # IMPLICIT RECEIVER: methods declare NO `self` formal (zero positionals is fine).
    # ALL positionals are ordinary formals. A positional `xs...` slurp and a keyword
    # `; kwargs...` slurp are recorded SEPARATELY: their names are bindings, not
    # scalar values, so a consumer iterating `argnames` never mistakes a slurp for a
    # single argument — while the raw `signature`/`call` retain the splat AST verbatim.
    argnames = Symbol[]
    vararg = nothing
    for a in positionals
        if _kernel_is_splat(a)
            vararg === nothing || throw(ArgumentError(
                "stateful @kernel method :$name has more than one positional vararg"))
            n = _kernel_arg_name(a)
            n isa Symbol || throw(ArgumentError(
                "stateful @kernel method :$name has an unsupported positional vararg $(repr(a))"))
            vararg = n
        else
            n = _kernel_arg_name(a)
            n === nothing && throw(ArgumentError(
                "stateful @kernel method :$name has an unsupported argument $(repr(a))"))
            push!(argnames, n)
        end
    end
    kwargs_splat = nothing
    for a in kwparams
        if _kernel_is_splat(a)
            kwargs_splat === nothing || throw(ArgumentError(
                "stateful @kernel method :$name has more than one keyword splat"))
            n = _kernel_arg_name(a)
            n isa Symbol || throw(ArgumentError(
                "stateful @kernel method :$name has an unsupported keyword splat $(repr(a))"))
            kwargs_splat = n
        else
            n = _kernel_arg_name(a)
            n === nothing && throw(ArgumentError(
                "stateful @kernel method :$name has an unsupported keyword $(repr(a))"))
            push!(argnames, n)
        end
    end
    # Zero self/__self__ formals: a formal literally named `self` or `__self__` is
    # REJECTED — the receiver is synthesized, never author-written.
    for n in argnames
        (n === :self || n === :__self__) && throw(ArgumentError(
            "stateful @kernel method :$name declares a `$n` formal; the receiver is implicit " *
            "(zero self/__self__ formals) — bare unshadowed names are the owner's fields"))
    end
    (vararg === :self || vararg === :__self__ || kwargs_splat === :self ||
        kwargs_splat === :__self__) && throw(ArgumentError(
        "stateful @kernel method :$name declares a `self`/`__self__` splat formal; the receiver is implicit"))
    # Recognize/validate `__self__` object-pass usage and record the receiver edges.
    sibling_calls = _kernel_self_usage(name, body)
    _KernelMethod(name, form, Tuple(argnames), vararg, kwargs_splat,
                  raw_sig, call, body, sibling_calls)
end

# --- (3) FROZEN, detached child-capture snapshot ----------------------------

# An IMMUTABLE, DETACHED, structurally-frozen snapshot of a composed child
# `KernelSpec`, captured by VALUE when the owner definition executes. `Value` and
# `Recipe` are already immutable; only a `Graph`'s Dict/Vector containers are
# mutable, so freezing the containers into Tuples makes the whole snapshot
# structurally immutable. It preserves ALL graph, port, boundary, and signature
# metadata and reconstructs a fresh `KernelSpec` for planning, so the owner never
# re-reads (nor can it mutate) the child.
struct _ChildSnapshot{VS,RS,PS,AS,PT,PO,HN,WN,CS}
    name::Symbol
    values::VS               # Tuple of Value (immutable)
    recipes::RS              # Tuple of Recipe (immutable)
    producers::PS            # Tuple of (id::Int, Tuple of recipe indices)
    aliases::AS              # Tuple of Pair{Int,Int}
    version::Int
    ports::PT                # Tuple of Pair{Symbol,Int} (port name => value id)
    port_order::PO           # Tuple{Vararg{Symbol}}
    have_names::HN           # Tuple{Vararg{Symbol}}
    want_names::WN           # Tuple{Vararg{Symbol}}
    call_signature::CS
end

"Capture a frozen, detached immutable snapshot of a child `KernelSpec` value."
function _kernel_capture_child(name::Symbol, child::KernelSpec)
    g = child.graph
    _ChildSnapshot(
        name,
        Tuple(Base.values(g.values)),
        Tuple(g.recipes),
        Tuple((id, Tuple(idxs)) for (id, idxs) in g.producers),
        Tuple(k => v for (k, v) in g.aliases),
        g.version,
        Tuple(k => v.id for (k, v) in child.ports),
        Tuple(child.port_order),
        Tuple(child.have_names),
        Tuple(child.want_names),
        child.call_signature,
    )
end

"Reconstruct a fresh `KernelSpec` from a frozen child snapshot (for planning)."
function _kernel_reconstruct(snap::_ChildSnapshot)
    values = Dict{Int,Value}(v.id => v for v in snap.values)
    recipes = Recipe[r for r in snap.recipes]
    producers = Dict{Int,Vector{Int}}(id => collect(idxs) for (id, idxs) in snap.producers)
    aliases = Dict{Int,Int}(p.first => p.second for p in snap.aliases)
    graph = Graph(values, recipes, producers, aliases, snap.version)
    ports = Dict{Symbol,Value}(p.first => values[p.second] for p in snap.ports)
    # `collect(Symbol, …)` (not bare `collect`) so an EMPTY boundary rebuilds as a
    # `Vector{Symbol}`, not a `Vector{Union{}}` — a ports-only Mode-2 spec has no wants.
    KernelSpec(graph, ports, collect(Symbol, snap.port_order),
               collect(Symbol, snap.have_names), collect(Symbol, snap.want_names),
               snap.call_signature)
end

_kernel_snapshot_recipe_count(snap::_ChildSnapshot) = length(snap.recipes)
_kernel_snapshot_port_names(snap::_ChildSnapshot) = snap.port_order

# --- recipe-SOURCE provenance: detached, recursively-frozen AST -------------

# `Recipe.op` is an opaque lowered closure, so poc's stateful MethodIR needs the
# ORIGINAL recipe-statement source. We store it as an IMMUTABLE, recursively-frozen
# node form — NO `Expr` / `Vector` / `Dict` survives in the stored value — and hand
# it back only by THAWING a FRESH mutable `Expr` copy per call, so a caller that
# mutates the returned AST can never reach the stored source (nor a second reader).

# Frozen wrappers, so NO mutable object survives in the stored form:
#   `Expr`      → `_FrozenExpr(head, Tuple of frozen args)`
#   `QuoteNode` → `_FrozenQuoteNode(frozen value)`  (its value may wrap an `Expr`)
#   `Vector{T}` → `_FrozenVector{T}(Tuple of frozen elems)`  (exact `eltype` stored)
#   `Dict{K,V}` → `_FrozenDict{K,V}(Tuple of frozen `key => value` pairs)`  (exact types)
# Each is an immutable `struct` whose children are `Tuple`s (never a `Vector`) of frozen
# nodes / immutable leaves. The `Vector`/`Dict` wrappers carry the EXACT concrete element/
# key/value types so thaw rebuilds `typeof(src)` faithfully (not a `Vector{Any}`/base
# `Dict`), which matters because a programmatically-supplied AST literal dispatches on its
# concrete type.
struct _FrozenExpr
    head::Symbol
    args::Tuple
end
struct _FrozenQuoteNode
    value::Any
end
struct _FrozenVector{T}
    elems::Tuple             # frozen elements; `T` = exact `eltype`, rebuilt on thaw
end
struct _FrozenDict{K,V}
    pairs::Tuple             # frozen `key => value` pairs; `K`/`V` rebuilt on thaw
end

# Freeze source AST → frozen form. `Expr`/`QuoteNode` and the mutable containers
# `Vector`/`Dict` are wrapped; the immutable COMPOSITES `Tuple`/`NamedTuple`/`Pair`
# keep their own type but with every element frozen (an immutable `Tuple` can still
# hide a mutable `Vector`, so it MUST recurse). An immutable SCALAR leaf (`Symbol`,
# numeric/string/char/bool literal, `LineNumberNode`, an immutable struct, …) is
# retained as-is — it cannot be mutated. Any OTHER mutable leaf (a `Set`, a
# `mutable struct`, a `Matrix`, …) is REJECTED with an actionable error rather than
# silently retained, so the guarantee "no mutable object survives" holds for ANY
# input, not only for authored recipe source (which never contains a bare container).
_kernel_freeze_ast(x::Expr) =
    _FrozenExpr(x.head, Tuple(_kernel_freeze_ast(a) for a in x.args))
_kernel_freeze_ast(x::QuoteNode) = _FrozenQuoteNode(_kernel_freeze_ast(x.value))
_kernel_freeze_ast(x::Vector) =
    _FrozenVector{Base.eltype(x)}(Tuple(_kernel_freeze_ast(e) for e in x))
# NB: qualify `Base.keytype`/`Base.valtype` — ReactiveKernels defines its own `valtype`
# (for `Value`/`ReactiveValue`), which shadows `Base.valtype` inside this module.
_kernel_freeze_ast(x::Dict) =
    _FrozenDict{Base.keytype(x),Base.valtype(x)}(
        Tuple(_kernel_freeze_ast(k) => _kernel_freeze_ast(v) for (k, v) in x))
# A non-`Dict` `AbstractDict` (`IdDict`, an ordered/custom dict, …) cannot be rebuilt to
# its exact `typeof` generically, so REJECT it rather than silently thawing a base `Dict`
# of a different type.
_kernel_freeze_ast(x::AbstractDict) = throw(ArgumentError(
    "cannot freeze recipe-source AST: unsupported `AbstractDict` subtype $(typeof(x)) — " *
    "only the concrete `Dict` is faithfully reconstructable."))
_kernel_freeze_ast(x::Tuple) = map(_kernel_freeze_ast, x)
_kernel_freeze_ast(x::NamedTuple) = NamedTuple{keys(x)}(map(_kernel_freeze_ast, values(x)))
_kernel_freeze_ast(x::Pair) = _kernel_freeze_ast(x.first) => _kernel_freeze_ast(x.second)
# `Symbol` is `ismutable`-true in Julia's type system but is an interned, un-mutatable
# atom (0 fields, no `setfield!`), so it is a safe immutable leaf — retain it, ahead of
# the catch-all below.
_kernel_freeze_ast(x::Symbol) = x
# Any other leaf is retained ONLY if it has NO reachable mutable state; otherwise it is
# REJECTED, never silently retained. `ismutable` alone is insufficient: a `Set` is an
# IMMUTABLE struct wrapping a mutable `Dict`, so the check must recurse into fields.
function _kernel_freeze_ast(x)
    _kernel_leaf_deeply_immutable(x) && return x
    throw(ArgumentError(
        "cannot freeze recipe-source AST: unsupported leaf of type $(typeof(x)) with " *
        "reachable mutable state. Recipe source may contain only ASTs (`Expr`/`QuoteNode`), " *
        "deeply-immutable literals, and `Vector`/`Dict`/`Tuple`/`NamedTuple`/`Pair` of them."))
end

# True iff `x` has NO reachable mutable state: an `isbits` value, the two builtin
# interned/immutable atoms `Symbol` and `String` (both report `ismutable`-true despite
# immutable user semantics), or an immutable struct whose every field is deeply immutable.
# NOT safe: a mutable object (`Array`/`Dict`); an immutable struct WRAPPING one (`Set`,
# whose backing `Dict` is mutable); a mutable or custom `AbstractString` that reaches
# mutable storage (e.g. one backed by a `Vector{UInt8}`) — the exceptional atom is `String`
# ONLY, never `AbstractString` broadly. A genuinely-immutable string wrapper (`SubString`,
# whose fields are a `String` + `Int`s) still passes via the generic recursive field proof
# below. Immutable structs cannot form cycles, so the recursion terminates.
_kernel_leaf_deeply_immutable(::Symbol) = true
_kernel_leaf_deeply_immutable(::String) = true
function _kernel_leaf_deeply_immutable(x)
    isbits(x) && return true
    ismutable(x) && return false
    return all(i -> _kernel_leaf_deeply_immutable(getfield(x, i)), 1:nfields(x))
end

# Thaw frozen form → a FRESH mutable AST. Each call rebuilds new `Expr`/`Vector`/`Dict`
# nodes, so the result shares no mutable structure with the stored frozen value or with
# any previously thawed copy. Immutable leaves are returned as-is (a `Symbol`/literal
# cannot be mutated, so sharing it is safe).
_kernel_thaw_ast(x::_FrozenExpr) =
    Expr(x.head, Any[_kernel_thaw_ast(a) for a in x.args]...)
_kernel_thaw_ast(x::_FrozenQuoteNode) = QuoteNode(_kernel_thaw_ast(x.value))
_kernel_thaw_ast(x::_FrozenVector{T}) where {T} =
    T[_kernel_thaw_ast(e) for e in x.elems]
_kernel_thaw_ast(x::_FrozenDict{K,V}) where {K,V} =
    Dict{K,V}(_kernel_thaw_ast(p.first) => _kernel_thaw_ast(p.second) for p in x.pairs)
_kernel_thaw_ast(x::Tuple) = map(_kernel_thaw_ast, x)
_kernel_thaw_ast(x::NamedTuple) = NamedTuple{keys(x)}(map(_kernel_thaw_ast, values(x)))
_kernel_thaw_ast(x::Pair) = _kernel_thaw_ast(x.first) => _kernel_thaw_ast(x.second)
_kernel_thaw_ast(x) = x

# --- (3)+(2) stateful expansion (skeleton) ----------------------------------

# The substrate object bound by a method-bearing `@kernel` in Increment 1. It
# carries the definition name, the unique Token type, the recipe-portion
# `KernelSpec`, and the extracted method metadata. Later increments replace this
# with the specializing factory + typed views + compiled schedules; for now it is
# an inspectable skeleton so tests can assert the discrimination/detection/Token.
# Thaw the FRESH/FROZEN method metadata: each stored method meta carries its
# `signature`/`call`/`body` deeply FROZEN (see `_FrozenExpr`); every `kernel_methods`
# call rebuilds fresh thawed `Expr`s so a caller can never reach the store or a
# previous read (mutation isolation), and the AST type/structure round-trips faithfully.
function _kernel_thaw_method(m::NamedTuple)
    vals = map(keys(m), Tuple(m)) do k, v
        (k === :signature || k === :call || k === :body) ? _kernel_thaw_ast(v) : v
    end
    NamedTuple{keys(m)}(vals)
end
_kernel_thaw_methods(methods) = Tuple(_kernel_thaw_method(m) for m in methods)

# --- pure endpoint templates -------------------------------------------------

"""
Construction-time templates for the pure, straight-line methods of a
method-bearing object kernel.  `builder(owner_spec, bindings)` returns detached
endpoint snapshots; it is generated by `@kernel`, so endpoint extraction never
evaluates source ASTs or introduces an opaque runtime call.
"""
struct _KernelEndpointTemplates{B,N,A,F}
    binding_names::B
    method_names::N
    method_args::A
    builder::F
end

_kernel_endpoint_is_bang(name::Symbol) = endswith(String(name), "!")

function _kernel_endpoint_has_nonstraight(x)
    x isa Expr || return false
    x.head in (:if, :for, :while, :try, :&&, :||, :let, :function, :->,
               :comprehension, :generator) && return true
    any(_kernel_endpoint_has_nonstraight, x.args)
end

function _kernel_endpoint_has_mutation(x)
    x isa Expr || return false
    if x.head in (:(+=), :(-=), :(*=), :(/=), :(.=))
        return true
    elseif x.head === :(=)
        lhs = x.args[1]
        (lhs isa Expr && lhs.head in (:(.), :ref)) && return true
    elseif x.head === :call && !isempty(x.args)
        callee = x.args[1]
        callee isa Symbol && _kernel_endpoint_is_bang(callee) && return true
    end
    any(_kernel_endpoint_has_mutation, x.args)
end

_kernel_endpoint_candidate(m::_KernelMethod) =
    m.name !== :inv && !_kernel_endpoint_is_bang(m.name) &&
    isempty(m.sibling_calls) && !_kernel_endpoint_has_nonstraight(m.body) &&
    !_kernel_endpoint_has_mutation(m.body) && m.vararg === nothing &&
    m.kwargs_splat === nothing

function _kernel_endpoint_return_type(signature)
    while signature isa Expr && signature.head === :where
        signature = signature.args[1]
    end
    signature isa Expr && signature.head === :(::) && length(signature.args) == 2 ?
        signature.args[2] : GlobalRef(Core, :Any)
end

function _kernel_endpoint_formals(m::_KernelMethod)
    call = m.call
    formals = call.args[2:end]
    if !isempty(formals) && formals[1] isa Expr && formals[1].head === :parameters
        isempty(formals[1].args) || throw(ArgumentError(
            "pure endpoint :$(m.name) does not support keyword arguments; " *
            "make data dependencies positional named ports"))
        formals = formals[2:end]
    end
    out = Tuple{Symbol,Any}[]
    for formal in formals
        name = _kernel_arg_name(formal)
        name isa Symbol || throw(ArgumentError(
            "pure endpoint :$(m.name) has unsupported formal $(repr(formal))"))
        type_expr = formal isa Expr && formal.head === :(::) ? formal.args[2] :
                    GlobalRef(Core, :Any)
        push!(out, (name, type_expr))
    end
    out
end

function _kernel_endpoint_owner_names(signature_inputs, recipe_stmts)
    names = Symbol[name for (name, _) in signature_inputs]
    for raw in recipe_stmts
        _kernel_is_line(raw) && continue
        assignment, _ = _kernel_recipe(raw)
        assignment isa Expr && assignment.head === :(=) || continue
        lhs = assignment.args[1]
        items = lhs isa Expr && lhs.head === :tuple ? lhs.args : Any[lhs]
        for item in items
            name, _ = _kernel_port_decl(item)
            name in names || push!(names, name)
        end
    end
    names
end

function _kernel_endpoint_property_target(x, owner_names::Set{Symbol})
    x isa Expr && x.head === :(.) && length(x.args) == 2 || return nothing
    root = x.args[1]
    leaf = x.args[2]
    root isa Symbol && root in owner_names || return nothing
    leaf = leaf isa QuoteNode ? leaf.value : leaf
    leaf isa Symbol || return nothing
    (root, leaf)
end

function _kernel_endpoint_binding_names(methods, owner_names)
    owners = Set(owner_names)
    found = Symbol[]
    local walk
    walk = function (x)
        x isa Expr || return
        if x.head === :call && !isempty(x.args)
            target = _kernel_endpoint_property_target(x.args[1], owners)
            target === nothing || (target[1] in found || push!(found, target[1]))
        end
        foreach(walk, x.args)
    end
    for m in methods
        _kernel_endpoint_candidate(m) && walk(m.body)
    end
    found
end

function _kernel_endpoint_dependencies(m, endpoint_names::Set{Symbol})
    deps = Symbol[]
    local walk
    walk = function (x)
        x isa Expr || return
        if x.head === :call && !isempty(x.args) && x.args[1] isa Symbol
            name = x.args[1]
            name in endpoint_names && name !== m.name &&
                (name in deps || push!(deps, name))
        end
        foreach(walk, x.args)
    end
    walk(m.body)
    deps
end

function _kernel_endpoint_toposort(methods)
    byname = Dict(m.name => m for m in methods)
    names = Set(keys(byname))
    state = Dict{Symbol,UInt8}()
    order = Symbol[]
    local visit
    visit = function (name)
        mark = get(state, name, 0x00)
        mark == 0x02 && return
        mark == 0x01 && throw(ArgumentError(
            "pure endpoint dependency cycle contains :$name"))
        state[name] = 0x01
        for dep in _kernel_endpoint_dependencies(byname[name], names)
            visit(dep)
        end
        state[name] = 0x02
        push!(order, name)
    end
    for m in methods
        visit(m.name)
    end
    order
end

function _kernel_endpoint_body_parts(body, name::Symbol)
    statements = body isa Expr && body.head === :block ?
                 Any[x for x in body.args if !(x isa LineNumberNode)] : Any[body]
    isempty(statements) && throw(ArgumentError("pure endpoint :$name has an empty body"))
    result = pop!(statements)
    if result isa Expr && result.head === :return
        length(result.args) == 1 || throw(ArgumentError(
            "pure endpoint :$name has an empty return"))
        result = result.args[1]
    elseif result isa Expr && result.head === :(=)
        push!(statements, result)
        lhs = result.args[1]
        lhs = lhs isa Expr && lhs.head === :(::) ? lhs.args[1] : lhs
        lhs isa Symbol || throw(ArgumentError(
            "pure endpoint :$name must finish with a scalar named result"))
        result = lhs
    end
    any(x -> x isa Expr && x.head === :return, statements) && throw(ArgumentError(
        "pure endpoint :$name supports one distinguished final return"))
    statements, result
end

function _kernel_endpoint_assignment_names(statements)
    names = Symbol[]
    for raw in statements
        assignment, _ = _kernel_recipe(raw)
        assignment isa Expr && assignment.head === :(=) || throw(ArgumentError(
            "pure endpoint statements before the distinguished result must be recipe assignments"))
        lhs = assignment.args[1]
        items = lhs isa Expr && lhs.head === :tuple ? lhs.args : Any[lhs]
        for item in items
            name, _ = _kernel_port_decl(item)
            name in names || push!(names, name)
        end
    end
    names
end

function _kernel_endpoint_rename(x, renames::Dict{Symbol,Symbol})
    x isa Symbol && return get(renames, x, x)
    x isa QuoteNode && return x
    x isa Expr || return x
    if x.head === :(.) && length(x.args) == 2
        return Expr(:(.), _kernel_endpoint_rename(x.args[1], renames), x.args[2])
    elseif x.head === :(::)
        return Expr(:(::), _kernel_endpoint_rename(x.args[1], renames), x.args[2:end]...)
    end
    Expr(x.head, (_kernel_endpoint_rename(a, renames) for a in x.args)...)
end

mutable struct _KernelEndpointBuildState
    owner_names::Vector{Symbol}
    owner_set::Set{Symbol}
    residual_owner_names::Vector{Symbol}
    endpoint_names::Set{Symbol}
    endpoint_vars::Dict{Symbol,Symbol}
    endpoint_formals::Dict{Symbol,Vector{Tuple{Symbol,Any}}}
    endpoint_returns::Dict{Symbol,Any}
    inverse_targets::Dict{Symbol,Tuple{Symbol,Any}}
    bindings_var::Symbol
    binding_specs::Dict{Tuple{Symbol,Symbol},Symbol}
    binding_prelude::Vector{Any}
    nested_specs::Dict{Symbol,Any}
    lifted_names::Dict{Symbol,Int}
    current_method::Symbol
end

function _kernel_endpoint_named_output!(st::_KernelEndpointBuildState, base::Symbol)
    count = get(st.lifted_names, base, 0) + 1
    st.lifted_names[base] = count
    count == 1 ? base : Symbol(st.current_method, "__", base, "__", count)
end

function _kernel_endpoint_binding_spec!(st::_KernelEndpointBuildState,
                                        root::Symbol, method::Symbol)
    key = (root, method)
    get!(st.binding_specs, key) do
        var = gensym(Symbol(root, :_, method, :_endpoint))
        bound = Expr(:call, GlobalRef(Core, :getfield), st.bindings_var, QuoteNode(root))
        rhs = Expr(:call, GlobalRef(Base, :getproperty), bound, QuoteNode(method))
        push!(st.binding_prelude, :($var = $rhs))
        var
    end
end

function _kernel_endpoint_lift!(st::_KernelEndpointBuildState, x, lifted::Vector{Any})
    x isa Expr || return x
    x.head in (:quote, :inert) && return x
    if x.head === :call && !isempty(x.args)
        callee = x.args[1]
        args = Any[_kernel_endpoint_lift!(st, a, lifted) for a in x.args[2:end]]
        if callee isa Symbol && callee in st.endpoint_names
            all(arg -> arg isa Symbol, args) || throw(ArgumentError(
                "pure endpoint :$(st.current_method) must assign expression arguments " *
                "to named ports before calling :$callee"))
            spec = st.endpoint_vars[callee]
            placeholder = gensym(Symbol(callee, :_call))
            st.nested_specs[placeholder] = spec
            out = _kernel_endpoint_named_output!(st, callee)
            out_type = st.endpoint_returns[callee]
            actuals = Any[st.residual_owner_names...; args...]
            push!(lifted, Expr(:(=), Expr(:(::), out, out_type),
                               Expr(:call, placeholder, actuals...)))
            return out
        elseif callee === :inv && length(args) == 2 && args[1] isa Symbol &&
               haskey(st.inverse_targets, args[1])
            target = args[1]
            value = args[2]
            value isa Symbol || throw(ArgumentError(
                "inv($target, value) in :$(st.current_method) requires a named value port"))
            input_name, input_type = st.inverse_targets[target]
            forward = st.endpoint_vars[target]
            inverse_spec = Expr(:call, GlobalRef(@__MODULE__, :extract),
                Expr(:parameters,
                     Expr(:kw, :have, Expr(:tuple,
                         (QuoteNode(n) for n in st.residual_owner_names)...,
                         QuoteNode(target))),
                     Expr(:kw, :want, QuoteNode(input_name))),
                forward)
            placeholder = gensym(Symbol(target, :_inverse_call))
            st.nested_specs[placeholder] = inverse_spec
            out = _kernel_endpoint_named_output!(st, input_name)
            actuals = Any[st.residual_owner_names...; value]
            push!(lifted, Expr(:(=), Expr(:(::), out, input_type),
                               Expr(:call, placeholder, actuals...)))
            return out
        end
        target = _kernel_endpoint_property_target(callee, st.owner_set)
        if target !== nothing
            root, method = target
            spec = _kernel_endpoint_binding_spec!(st, root, method)
            all(arg -> arg isa Symbol, args) || throw(ArgumentError(
                "bound endpoint `$root.$method` in :$(st.current_method) requires named port arguments"))
            placeholder = gensym(Symbol(root, :_, method, :_call))
            st.nested_specs[placeholder] = spec
            out = _kernel_endpoint_named_output!(st, Symbol(root, "__", method))
            out_type = Expr(:call, GlobalRef(@__MODULE__, :valtype),
                            Expr(:call, GlobalRef(Base, :only),
                                 Expr(:call, GlobalRef(@__MODULE__, :outputs), spec)))
            push!(lifted, Expr(:(=), Expr(:(::), out, out_type),
                               Expr(:call, placeholder, args...)))
            return out
        end
        return Expr(:call, _kernel_endpoint_lift!(st, callee, lifted), args...)
    end
    Expr(x.head, (_kernel_endpoint_lift!(st, a, lifted) for a in x.args)...)
end

function _kernel_endpoint_lower_body(m::_KernelMethod, st::_KernelEndpointBuildState,
                                     inverse)
    statements, result = _kernel_endpoint_body_parts(m.body, m.name)
    formals = Set(first.(st.endpoint_formals[m.name]))
    protected = union(st.owner_set, st.endpoint_names, formals)
    renames = Dict{Symbol,Symbol}()
    for name in _kernel_endpoint_assignment_names(statements)
        name in protected || (renames[name] = Symbol(m.name, "__", name))
    end
    statements = Any[_kernel_endpoint_rename(x, renames) for x in statements]
    result = _kernel_endpoint_rename(result, renames)

    block = Any[]
    for raw in statements
        assignment, metadata = _kernel_recipe(raw)
        lhs, rhs = assignment.args
        lifted = Any[]
        lowered = _kernel_endpoint_lift!(st, rhs, lifted)
        append!(block, lifted)
        rebuilt = Expr(:(=), lhs, lowered)
        if isempty(metadata)
            push!(block, rebuilt)
        else
            entries = Expr(:tuple, (Expr(:(=), k, v) for (k, v) in metadata)...)
            push!(block, Expr(:macrocall, GlobalRef(@__MODULE__, Symbol("@recipe")),
                              LineNumberNode(0), entries, rebuilt))
        end
    end
    lifted = Any[]
    result = _kernel_endpoint_lift!(st, result, lifted)
    append!(block, lifted)
    push!(block, Expr(:(=), Expr(:(::), m.name, st.endpoint_returns[m.name]), result))

    if inverse !== nothing
        input_name, input_type = st.inverse_targets[m.name]
        _, inverse_arg = inverse
        inv_statements, inv_result = _kernel_endpoint_body_parts(inverse_arg.body, :inv)
        isempty(inv_statements) || throw(ArgumentError(
            "inverse declaration for :$(m.name) must be one straight-line expression"))
        inverse_formals = _kernel_endpoint_formals(inverse_arg)
        z_name = inverse_formals[2][1]
        inv_result = _kernel_endpoint_rename(inv_result, Dict(z_name => m.name))
        inv_lifted = Any[]
        inv_result = _kernel_endpoint_lift!(st, inv_result, inv_lifted)
        append!(block, inv_lifted)
        push!(block, Expr(:(=), Expr(:(::), input_name, input_type), inv_result))
    end
    push!(block, Expr(:return, m.name))
    Expr(:block, block...)
end

function _kernel_endpoint_templates_expr(signature_inputs, recipe_stmts,
                                         methods::Vector{_KernelMethod}, mod::Module)
    endpoints = _KernelMethod[m for m in methods if _kernel_endpoint_candidate(m)]
    owner_names = _kernel_endpoint_owner_names(signature_inputs, recipe_stmts)
    seen_endpoints = Set{Symbol}()
    for endpoint in endpoints
        endpoint.name in seen_endpoints && throw(ArgumentError(
            "pure endpoint :$(endpoint.name) is defined more than once; " *
            "transparent kernel objects require one distinguished return per endpoint name"))
        push!(seen_endpoints, endpoint.name)
        endpoint.name in owner_names && throw(ArgumentError(
            "pure endpoint :$(endpoint.name) collides with an owner port of the same name; " *
            "rename either the port or the endpoint explicitly"))
    end
    binding_names = _kernel_endpoint_binding_names(endpoints, owner_names)
    residual_owner_names = Symbol[n for n in owner_names if !(n in binding_names)]
    endpoint_names = Set(m.name for m in endpoints)

    inverses = Dict{Symbol,_KernelMethod}()
    for m in methods
        m.name === :inv || continue
        formals = _kernel_endpoint_formals(m)
        length(formals) == 2 || throw(ArgumentError(
            "inv(endpoint, z) declarations require exactly the endpoint marker and one value formal"))
        target = formals[1][1]
        target in endpoint_names || throw(ArgumentError(
            "inverse declaration names unknown pure endpoint :$target"))
        haskey(inverses, target) && throw(ArgumentError(
            "pure endpoint :$target has more than one inverse declaration"))
        inverses[target] = m
    end

    endpoint_formals = Dict(m.name => _kernel_endpoint_formals(m) for m in endpoints)
    endpoint_returns = Dict(m.name => _kernel_endpoint_return_type(m.signature) for m in endpoints)
    inverse_targets = Dict{Symbol,Tuple{Symbol,Any}}()
    for m in endpoints
        haskey(inverses, m.name) || continue
        formals = endpoint_formals[m.name]
        length(formals) == 1 || throw(ArgumentError(
            "inverse declaration requires unary endpoint :$(m.name)"))
        inverse_targets[m.name] = formals[1]
    end

    owner_var = gensym(:endpoint_owner_spec)
    bindings_var = gensym(:endpoint_bindings)
    endpoint_vars = Dict(name => gensym(Symbol(name, :_endpoint_spec)) for name in endpoint_names)
    state = _KernelEndpointBuildState(
        owner_names, Set(owner_names), residual_owner_names, endpoint_names,
        endpoint_vars, endpoint_formals, endpoint_returns, inverse_targets,
        bindings_var, Dict{Tuple{Symbol,Symbol},Symbol}(), Any[],
        Dict{Symbol,Any}(), Dict{Symbol,Int}(), :none,
    )

    endpoint_exprs = Dict{Symbol,Any}()
    for name in _kernel_endpoint_toposort(endpoints)
        m = only(filter(x -> x.name === name, endpoints))
        state.current_method = name
        empty!(state.nested_specs)
        empty!(state.lifted_names)
        body = _kernel_endpoint_lower_body(
            m, state, haskey(inverses, name) ? (name, inverses[name]) : nothing)
        signature = Tuple{Symbol,Any}[]
        for owner_name in residual_owner_names
            T = Expr(:call, GlobalRef(@__MODULE__, :valtype),
                     Expr(:ref, owner_var, QuoteNode(owner_name)))
            push!(signature, (owner_name, T))
        end
        append!(signature, endpoint_formals[name])
        endpoint_exprs[name] = _kernel_expand(
            body, signature, nothing, mod; nested_specs = copy(state.nested_specs))
    end

    builder_body = Any[state.binding_prelude...]
    for name in _kernel_endpoint_toposort(endpoints)
        push!(builder_body, :($(endpoint_vars[name]) = $(endpoint_exprs[name])))
    end
    snapshots = Expr(:tuple, (
        Expr(:call, _kernel_capture_child, QuoteNode(m.name), endpoint_vars[m.name])
        for m in endpoints)...)
    push!(builder_body, snapshots)
    builder = Expr(:->, Expr(:tuple, owner_var, bindings_var), Expr(:block, builder_body...))
    arg_meta = Tuple(m.name => Tuple(first.(endpoint_formals[m.name])) for m in endpoints)
    Expr(:call, _KernelEndpointTemplates, QuoteNode(Tuple(binding_names)),
         QuoteNode(Tuple(m.name for m in endpoints)), QuoteNode(arg_meta), builder)
end

struct _CapturedTypeAuthority{T}
    ref::GlobalRef
    value::T
end
function _kernel_capture_type_authorities(mod::Module, names::Tuple)
    out = Any[]
    for name in names
        isdefined(mod, name) || continue
        value = getglobal(mod, name)
        value isa Type || continue
        push!(out, _CapturedTypeAuthority(GlobalRef(mod, name), value))
    end
    Tuple(out)
end

struct _StatefulKernelSkeleton{Token,SN,M,R,C,T,E}
    name::Symbol
    mod::Module              # the DEFINITION module (for hygienic op GlobalRefs)
    spec_snapshot::SN        # DETACHED IMMUTABLE owner-spec provenance (a `_ChildSnapshot`);
                             #   the field/port metadata is captured, never a live mutable spec
    methods::M               # per-method meta with DEEPLY FROZEN signature/call/body
    recipe_source::R         # frozen, detached recipe-statement source (see _FrozenExpr)
    callee_regs::C           # immutable def-time snapshot: authored callee ref => registration
    type_authorities::T      # immutable def-time snapshot: bare formal annotation ref => exact Type
    endpoint_templates::E    # construction-time pure-method KernelSpec templates
end
# `Token` is a phantom type parameter — the unique definition token, a per-expansion
# gensym Symbol carried as `Val{Token}` (unique, typed overload identity, no minted
# struct / no world-age hazard / no mutable registry). Supplied explicitly.
_StatefulKernelSkeleton(name::Symbol, ::Val{Token}, mod::Module, spec_snapshot::SN,
                        methods::M, recipe_source::R,
                        callee_regs::C, type_authorities::T,
                        endpoint_templates::E) where {Token,SN,M,R,C,T,E} =
    _StatefulKernelSkeleton{Token,SN,M,R,C,T,E}(
        name, mod, spec_snapshot, methods, recipe_source, callee_regs,
        type_authorities, endpoint_templates)

kernel_token(skel::_StatefulKernelSkeleton{Token}) where {Token} = Token
kernel_module(skel::_StatefulKernelSkeleton) = getfield(skel, :mod)
# D: immutable def-time snapshot of direct registered callees (detached identities).
kernel_callee_registrations(skel::_StatefulKernelSkeleton) = getfield(skel, :callee_regs)
kernel_type_authorities(skel::_StatefulKernelSkeleton) = getfield(skel, :type_authorities)
# FRESH reconstructed planning `KernelSpec` per call from the detached snapshot — never
# a shared live authority, so mutating a returned spec/graph cannot change a later read.
kernel_spec(skel::_StatefulKernelSkeleton) = _kernel_reconstruct(getfield(skel, :spec_snapshot))
# Immutable port metadata surface (the authored field/port names, a Tuple).
kernel_port_names(skel::_StatefulKernelSkeleton) =
    _kernel_snapshot_port_names(getfield(skel, :spec_snapshot))
# FRESH thawed method ASTs each call (mutation isolation).
kernel_methods(skel::_StatefulKernelSkeleton) = _kernel_thaw_methods(getfield(skel, :methods))
kernel_endpoint_names(skel::_StatefulKernelSkeleton) =
    getfield(getfield(skel, :endpoint_templates), :method_names)

# Thaw a FRESH copy of the captured recipe-statement source (`Expr(:block, …)`), in
# authored order, with full conditional/index/call structure preserved and definition
# line numbers stripped. Each call rebuilds new mutable nodes, so mutating the result
# never affects the stored frozen source or a subsequent read. poc's stateful MethodIR
# consumes this because `Recipe.op` is opaque; pair it with `kernel_module` for
# hygienic resolution of the names inside.
kernel_recipe_ast(skel::_StatefulKernelSkeleton) =
    Expr(:block, (_kernel_thaw_ast(s) for s in getfield(skel, :recipe_source))...)

function Base.show(io::IO, skel::_StatefulKernelSkeleton)
    print(io, "@kernel object :", skel.name, " (", length(skel.methods),
          " method(s), ", length(kernel_endpoint_names(skel)), " pure endpoint(s))")
end

"""
    KernelObjectSpec

A transparent specialization produced by binding kernel-valued ports of a
method-bearing `@kernel` object. Pure endpoint properties such as
`normal.logpdf` are ordinary [`KernelSpec`](@ref) views; use [`extract`](@ref)
to choose any other named HAVE/WANT cut.
"""
struct KernelObjectSpec{S,B}
    skeleton::S
    bindings::B
end

kernel_token(obj::KernelObjectSpec) = kernel_token(getfield(obj, :skeleton))
kernel_module(obj::KernelObjectSpec) = kernel_module(getfield(obj, :skeleton))
kernel_methods(obj::KernelObjectSpec) = kernel_methods(getfield(obj, :skeleton))
kernel_endpoint_names(obj::KernelObjectSpec) =
    kernel_endpoint_names(getfield(obj, :skeleton))

_kernel_object_skeleton(skel::_StatefulKernelSkeleton) = skel
_kernel_object_skeleton(obj::KernelObjectSpec) = getfield(obj, :skeleton)
_kernel_object_bindings(::_StatefulKernelSkeleton) = NamedTuple()
_kernel_object_bindings(obj::KernelObjectSpec) = getfield(obj, :bindings)

function _kernel_object_owner_spec(object)
    skel = _kernel_object_skeleton(object)
    spec = kernel_spec(skel)
    bindings = _kernel_object_bindings(object)
    bound = Set(Symbol.(keys(bindings)))
    isempty(bound) && return spec

    for recipe in spec.graph.recipes
        touched = Symbol[value.name for value in (recipe.inputs..., recipe.outputs...)
                         if value.name in bound]
        isempty(touched) || throw(ArgumentError(
            "kernel-valued binding :$(first(touched)) is used by an owner recipe; " *
            "bound kernels may only be applied through transparent endpoint calls"))
    end
    ports = Dict{Symbol,Value}(name => value for (name, value) in spec.ports
                               if !(name in bound))
    order = Symbol[name for name in spec.port_order if !(name in bound)]
    have = Symbol[name for name in spec.have_names if !(name in bound)]
    want = Symbol[name for name in spec.want_names if !(name in bound)]
    KernelSpec(spec.graph, ports, order, have, want, nothing)
end

kernel_spec(obj::KernelObjectSpec) = _kernel_object_owner_spec(obj)

function _kernel_object_full_spec(object)
    skel = _kernel_object_skeleton(object)
    bindings = _kernel_object_bindings(object)
    templates = getfield(skel, :endpoint_templates)
    missing = Symbol[name for name in templates.binding_names if !(name in keys(bindings))]
    isempty(missing) || throw(ArgumentError(
        "kernel object :$(getfield(skel, :name)) still needs transparent kernel binding(s): " *
        join(string.(missing), ", ")))

    owner = _kernel_object_owner_spec(object)
    combined = owner
    snapshots = templates.builder(owner, bindings)
    for snapshot in snapshots
        combined = merge(combined, _kernel_reconstruct(snapshot); boundary = :base)
    end
    combined
end

function _kernel_object_default_have(object, wanted)
    skel = _kernel_object_skeleton(object)
    templates = getfield(skel, :endpoint_templates)
    names = copy(_kernel_object_owner_spec(object).have_names)
    wanted_items = wanted isa Tuple || wanted isa AbstractVector ? wanted : (wanted,)
    argmap = Dict(templates.method_args)
    for wanted_name in wanted_items
        wanted_name isa Symbol || continue
        for arg in get(argmap, wanted_name, ())
            arg in names || push!(names, arg)
        end
    end
    names
end

"""
    extract(object; have, want) -> KernelSpec

Splice every pure endpoint template and bound child kernel into one ordinary
named graph, then return the requested boundary view.  Supplying an endpoint
as `want` automatically adds its parameter ports to the default HAVE boundary.
"""
function extract(object::Union{_StatefulKernelSkeleton,KernelObjectSpec};
                 have = _KERNEL_DEFAULT_BOUNDARY,
                 want = _KERNEL_DEFAULT_BOUNDARY)
    full = _kernel_object_full_spec(object)
    chosen_want = want === _KERNEL_DEFAULT_BOUNDARY ? Tuple(full.want_names) : want
    chosen_have = have === _KERNEL_DEFAULT_BOUNDARY ?
                  _kernel_object_default_have(object, chosen_want) : have
    view = extract(full; have = chosen_have, want = chosen_want)
    if have === _KERNEL_DEFAULT_BOUNDARY
        wanted = chosen_want isa Tuple || chosen_want isa AbstractVector ?
                 Tuple(chosen_want) : (chosen_want,)
        if length(wanted) == 1
            templates = getfield(_kernel_object_skeleton(object), :endpoint_templates)
            argmap = Dict(templates.method_args)
            endpoint = wanted[1]
            if endpoint isa Symbol && haskey(argmap, endpoint)
                implicit = Tuple(_kernel_object_owner_spec(object).have_names)
                explicit = Tuple(argmap[endpoint])
                signature = _kernel_endpoint_call_signature(Val(implicit), Val(explicit))
                return KernelSpec(view.graph, view.ports, view.port_order,
                                  view.have_names, view.want_names, signature)
            end
        end
    end
    view
end

function _kernel_bind_object(object, args)
    skel = _kernel_object_skeleton(object)
    templates = getfield(skel, :endpoint_templates)
    existing = _kernel_object_bindings(object)
    remaining = Symbol[name for name in templates.binding_names if !(name in keys(existing))]
    isempty(args) && throw(ArgumentError(
        "kernel object :$(getfield(skel, :name)) construction needs at least one bound kernel"))
    length(args) <= length(remaining) || throw(ArgumentError(
        "kernel object :$(getfield(skel, :name)) has $(length(remaining)) unbound kernel port(s), " *
        "but received $(length(args))"))
    for (name, arg) in zip(remaining, args)
        arg isa Union{KernelSpec,_StatefulKernelSkeleton,KernelObjectSpec} ||
            throw(ArgumentError(
                "kernel-valued port :$name requires a transparent KernelSpec or @kernel object, " *
                "got $(typeof(arg))"))
    end
    additions = NamedTuple{Tuple(remaining[1:length(args)])}(Tuple(args))
    KernelObjectSpec(skel, merge(existing, additions))
end

(skel::_StatefulKernelSkeleton)(arg, rest...) = _kernel_bind_object(skel, (arg, rest...))
(obj::KernelObjectSpec)(arg, rest...) = _kernel_bind_object(obj, (arg, rest...))

function (object::Union{_StatefulKernelSkeleton,KernelObjectSpec})(;
        have = _KERNEL_DEFAULT_BOUNDARY, want = _KERNEL_DEFAULT_BOUNDARY)
    extract(object; have = have, want = want)
end

function Base.getproperty(skel::_StatefulKernelSkeleton, name::Symbol)
    name in fieldnames(typeof(skel)) && return getfield(skel, name)
    name in kernel_endpoint_names(skel) && return extract(skel; want = name)
    getfield(skel, name)
end

function Base.getproperty(obj::KernelObjectSpec, name::Symbol)
    name in fieldnames(typeof(obj)) && return getfield(obj, name)
    name in kernel_endpoint_names(obj) && return extract(obj; want = name)
    getfield(obj, name)
end

Base.propertynames(skel::_StatefulKernelSkeleton, private::Bool = false) =
    Tuple(unique((fieldnames(typeof(skel))..., kernel_endpoint_names(skel)...)))
Base.propertynames(obj::KernelObjectSpec, private::Bool = false) =
    Tuple(unique((fieldnames(typeof(obj))..., kernel_endpoint_names(obj)...)))

Base.keys(object::Union{_StatefulKernelSkeleton,KernelObjectSpec}) =
    keys(_kernel_object_full_spec(object))
Base.getindex(object::Union{_StatefulKernelSkeleton,KernelObjectSpec}, name::Symbol) =
    _kernel_object_full_spec(object)[name]
Base.haskey(object::Union{_StatefulKernelSkeleton,KernelObjectSpec}, name::Symbol) =
    haskey(_kernel_object_full_spec(object), name)
kernel_graph(object::Union{_StatefulKernelSkeleton,KernelObjectSpec}) =
    kernel_graph(_kernel_object_full_spec(object))

# Route a method-bearing `@kernel` here. Increment-1 SKELETON: split the body into
# recipe statements vs method definitions, extract the methods (short/long,
# explicit self, keyword params before self, positional `xs...` + keyword
# `; kwargs...` splats), and expand the recipe portion through the
# EXISTING stateless machinery (`_kernel_expand`, the shared parser/adapter — RK
# gate 2). It does NOT yet build a factory/schedule; it returns the substrate so
# later increments (and poc's MethodIR) can attach by Token.
function _kernel_stateful_expand(name, signature_inputs, call_signature, block,
                                 __module__)
    recipe_stmts = Any[]
    method_stmts = Tuple{Symbol,Any}[]   # (form, stmt)
    for stmt in _kernel_body_statements(block)
        form = _kernel_is_line(stmt) ? nothing : _kernel_stmt_method_form(stmt)
        if form === nothing
            push!(recipe_stmts, stmt)
        else
            push!(method_stmts, (form, stmt))
        end
    end

    # Validate/extract methods at expansion (deterministic, reported errors).
    methods = _KernelMethod[]
    for (form, stmt) in method_stmts
        push!(methods, _kernel_extract_method(stmt, form))
    end

    recipe_block = Expr(:block, recipe_stmts...)
    spec_expr = _kernel_expand(recipe_block, signature_inputs, call_signature, __module__)

    token_sym = gensym(Symbol(name, :_token))
    # The exposed per-method metadata carries the FULL authored signature (where
    # binders + return annotation), the peeled call, and the raw body — enough for
    # Increment 2's MethodIR emission (typed MethodId, return-annotation semantics).
    # `signature`/`call`/`body` are DEEPLY FROZEN (line-stripped) exactly like
    # `recipe_source`, so no shared mutable `Expr` survives in the stored form;
    # `kernel_methods` thaws a fresh copy per call (mutation isolation, type fidelity).
    method_meta = Tuple(
        (; name = m.name, form = m.form, argnames = m.argnames,
           vararg = m.vararg, kwargs_splat = m.kwargs_splat,
           signature = _kernel_freeze_ast(Base.remove_linenums!(deepcopy(m.signature))),
           call = _kernel_freeze_ast(Base.remove_linenums!(deepcopy(m.call))),
           body = _kernel_freeze_ast(Base.remove_linenums!(deepcopy(m.body))),
           sibling_calls = m.sibling_calls)
        for m in methods)

    annotation_names = Symbol[]
    for m in methods, annotation in _kernel_signature_annotations(m.signature)
        annotation isa Symbol && !(annotation in annotation_names) && push!(annotation_names, annotation)
    end

    # Capture the recipe-portion SOURCE as a detached, recursively-frozen AST (poc's
    # stateful MethodIR needs it — `Recipe.op` is opaque). Deep-copy each recipe
    # statement before stripping line numbers so the originals feeding `spec_expr`
    # stay untouched; drop bare line-number statements; keep source order + the full
    # conditional/index/call structure. The frozen value is embedded as an immutable
    # literal (exactly like `method_meta`), so it self-quotes into the expansion.
    recipe_source = Tuple(
        _kernel_freeze_ast(s isa Expr ? Base.remove_linenums!(deepcopy(s)) : s)
        for s in recipe_stmts if !(s isa LineNumberNode))

    endpoint_templates = _kernel_endpoint_templates_expr(
        signature_inputs, recipe_stmts, methods, __module__)

    # A method-bearing `@kernel` binds its OWNER name via a `const` so the bound
    # skeleton value is a stable, immutable binding (safe to capture by identity in
    # a composing owner). A `const` in an unsupported local scope is a Julia error,
    # which IS the required deterministic unsupported-local-scope rejection. The
    # token is a per-expansion gensym Symbol carried as `Val(token_sym)`.
    Expr(:const, Expr(:(=), esc(name),
        Expr(:call, _StatefulKernelSkeleton, QuoteNode(name),
             Expr(:call, Val, QuoteNode(token_sym)), __module__,
             # E: capture a DETACHED IMMUTABLE snapshot of the owner spec at definition
             # time instead of storing the live mutable authority.
             Expr(:call, _kernel_capture_child, QuoteNode(name), esc(spec_expr)),
             method_meta, recipe_source,
             # D: snapshot each direct registered callee's identity, PER METHOD BODY —
             # sibling/owner-field/formal/local names excluded (pending/factory-time).
             _kernel_callee_capture_expr(
                 _kernel_mode1_callee_pairs(signature_inputs, recipe_stmts, methods),
                 __module__),
             Expr(:call, _kernel_capture_type_authorities, __module__, QuoteNode(Tuple(annotation_names))),
             endpoint_templates)))
end

# --- Mode-2 free-method recognition (V7 implicit-field pivot) ----------------
#
# A METHODLESS `@kernel name(sig) = body` whose body MUTATES a field of the FIRST
# positional subject (`subject.field = …` / `subject.field op= …`, incl. `@.`
# broadcast) is a Mode-2 FREE METHOD (e.g. `leapfrog!(phasepoint; stepsize)`),
# authored by the sole `@kernel` so RK owns its visible effects — independent of
# `!` spelling (bang is a naming convention, not the discriminator). This is
# provenance/recognition ONLY: the shallow-declared write/read roots on the
# subject, the frozen body source, and a def-unique Token — NO MethodIR / effect
# closure / lowering / execution.

# An assignment-like head: `=` or an augmented `op=` (`+=`, `-=`, `.=`, …), but
# NOT a comparison that merely ends in `=` (`==`, `<=`, `===`, `.==`, …).
function _kernel_is_assign_head(h)
    h isa Symbol || return false
    s = String(h)
    endswith(s, "=") || return false
    h in (:(==), :(!=), :(<=), :(>=), :(===), :(!==),
          :(.==), :(.!=), :(.<=), :(.>=)) && return false
    true
end

# Root of a (possibly nested) `a.b.c` property chain: `a`.
_kernel_dot_root(x) = (x isa Expr && x.head === :(.)) ? _kernel_dot_root(x.args[1]) : x

# Is `place` a property access rooted at `subject` (`subject.f`, `subject.a.b`)?
_kernel_is_subject_place(place, subject::Symbol) =
    place isa Expr && place.head === :(.) && _kernel_dot_root(place) === subject

# The field DIRECTLY under `subject` in a `subject.a.b` chain (the top owned field
# = the write/read ROOT): `subject.mom` → `:mom`, `subject.a.b` → `:a`.
function _kernel_top_field(place, subject::Symbol)
    node = place
    while node isa Expr && node.head === :(.)
        if node.args[1] === subject
            f = length(node.args) >= 2 ? node.args[2] : nothing
            return f isa QuoteNode ? f.value : (f isa Symbol ? f : nothing)
        end
        node = node.args[1]
    end
    nothing
end

# Does the body mutate a field of `subject` anywhere (through `@.` too)? — the
# Mode-2 discriminator. Disjoint from stateless recipes (bare-Symbol LHS) and from
# Mode-1 (nested methods, tested first).
function _kernel_body_mutates_subject(block, subject::Symbol)
    found = false
    walk(x) = begin
        (found || !(x isa Expr)) && return
        if _kernel_is_assign_head(x.head) && length(x.args) >= 1 &&
           _kernel_is_subject_place(x.args[1], subject)
            found = true
            return
        end
        for a in x.args
            walk(a)
        end
    end
    walk(block)
    found
end

# Shallow-DECLARED effect roots on the subject: top-fields written (assignment LHS
# `subject.f …`) vs read (`subject.f` elsewhere). Provenance for the resolver — NOT
# an effect closure (poc derives precise per-owner effects itself).
function _kernel_subject_effect_roots(block, subject::Symbol)
    writes = Symbol[]
    reads = Symbol[]
    pushuniq!(v, x) = (x isa Symbol && !(x in v) && push!(v, x); nothing)
    local walk
    walk = function (x)
        x isa Expr || return
        if _kernel_is_assign_head(x.head) && length(x.args) >= 2
            lhs = x.args[1]
            if _kernel_is_subject_place(lhs, subject)
                pushuniq!(writes, _kernel_top_field(lhs, subject))
                # a direct field place has no reachable sub-reads; do NOT re-count it
                # as a read.
            else
                walk(lhs)
            end
            for a in x.args[2:end]
                walk(a)
            end
        elseif x.head === :(.) && _kernel_dot_root(x) === subject
            pushuniq!(reads, _kernel_top_field(x, subject))
            walk(x.args[1])          # the object chain (the field is a QuoteNode leaf)
        else
            for a in x.args
                walk(a)
            end
        end
        return
    end
    walk(block)
    (writes = Tuple(writes), reads = Tuple(reads))
end

# `@kernel name!!` — the strong same-object update registration (marker #4). The
# `!!` suffix is recognized off the stored name; no extra storage is needed.
_kernel_is_bangbang_name(name::Symbol) = endswith(String(name), "!!")

# The substrate object bound by a Mode-2 free `@kernel` in Increment 1. It carries
# the def name + unique Token, the subject (first positional), the shallow-declared
# write/read effect roots, the ports-only recipe `spec` (the signature ports; the
# body holds mutations, not owner recipes), the frozen body source (poc lowers it),
# and the `!!` registration flag. Later increments add the specializing factory.
struct _Mode2KernelSkeleton{Token,SN,M,B,C}
    name::Symbol
    mod::Module
    subject::Symbol
    spec_snapshot::SN        # DETACHED IMMUTABLE owner-spec provenance (a `_ChildSnapshot`)
    write_roots::Tuple{Vararg{Symbol}}
    read_roots::Tuple{Vararg{Symbol}}
    method::M                # (; name, subject, write_roots, read_roots, signature[frozen], call[frozen])
    body_source::B           # frozen, detached body-statement source (see _FrozenExpr)
    is_bang_bang::Bool
    callee_regs::C           # immutable def-time snapshot: authored callee ref => registration
end
_Mode2KernelSkeleton(name::Symbol, ::Val{Token}, mod::Module, subject::Symbol,
                     spec_snapshot::SN, write_roots, read_roots, method::M,
                     body_source::B, is_bb::Bool, callee_regs::C) where {Token,SN,M,B,C} =
    _Mode2KernelSkeleton{Token,SN,M,B,C}(name, mod, subject, spec_snapshot, Tuple(write_roots),
                                         Tuple(read_roots), method, body_source, is_bb, callee_regs)

kernel_token(skel::_Mode2KernelSkeleton{Token}) where {Token} = Token
kernel_module(skel::_Mode2KernelSkeleton) = getfield(skel, :mod)
kernel_callee_registrations(skel::_Mode2KernelSkeleton) = getfield(skel, :callee_regs)
kernel_spec(skel::_Mode2KernelSkeleton) = _kernel_reconstruct(getfield(skel, :spec_snapshot))
kernel_port_names(skel::_Mode2KernelSkeleton) =
    _kernel_snapshot_port_names(getfield(skel, :spec_snapshot))
kernel_subject(skel::_Mode2KernelSkeleton) = getfield(skel, :subject)
kernel_write_roots(skel::_Mode2KernelSkeleton) = getfield(skel, :write_roots)
kernel_read_roots(skel::_Mode2KernelSkeleton) = getfield(skel, :read_roots)
# FRESH thawed method AST (the frozen signature/call) each call — mutation isolation.
kernel_methods(skel::_Mode2KernelSkeleton) = (_kernel_thaw_method(getfield(skel, :method)),)
kernel_is_bangbang(skel::_Mode2KernelSkeleton) = getfield(skel, :is_bang_bang)

# Thaw a FRESH copy of the captured Mode-2 body source (`Expr(:block, …)`), in
# authored order, structure intact, line numbers stripped — the mutation body poc
# lowers into the free-method schedule.
kernel_recipe_ast(skel::_Mode2KernelSkeleton) =
    Expr(:block, (_kernel_thaw_ast(s) for s in getfield(skel, :body_source))...)

function Base.show(io::IO, skel::_Mode2KernelSkeleton)
    print(io, "Mode-2 @kernel :", skel.name, " (subject :", skel.subject,
          skel.is_bang_bang ? "; !! strong-update" : "",
          "; Increment-1 skeleton)")
end

# Route a Mode-2 free `@kernel` here. Increment-1 SKELETON: scan the subject's
# declared write/read roots, build a ports-only recipe spec from the signature,
# freeze the body source, and bind the substrate by Token. It does NOT lower a
# schedule or generate a callable dispatch; later increments add those.
function _kernel_mode2_expand(name, signature_inputs, call_signature, block,
                              subject::Symbol, raw_signature, __module__)
    roots = _kernel_subject_effect_roots(block, subject)
    # A: retain the FULL authored raw signature + peeled call (positional/keyword order,
    # required/default status, annotations, where/return where supported), DEEPLY FROZEN
    # so poc reads them verbatim rather than reconstructing from spec port keys.
    frozen_sig = _kernel_freeze_ast(Base.remove_linenums!(deepcopy(raw_signature)))
    frozen_call = _kernel_freeze_ast(
        Base.remove_linenums!(deepcopy(_kernel_peel_signature(raw_signature))))
    method_meta = (; name = name, subject = subject,
                     write_roots = roots.writes, read_roots = roots.reads,
                     signature = frozen_sig, call = frozen_call)
    # Ports-only spec: the signature ports (the body carries mutations, not recipes).
    spec_expr = _kernel_expand(Expr(:block), signature_inputs, call_signature, __module__)
    body_source = Tuple(
        _kernel_freeze_ast(s isa Expr ? Base.remove_linenums!(deepcopy(s)) : s)
        for s in _kernel_body_statements(block) if !(s isa LineNumberNode))
    token_sym = gensym(Symbol(name, :_token))
    is_bb = _kernel_is_bangbang_name(name)
    Expr(:const, Expr(:(=), esc(name),
        Expr(:call, _Mode2KernelSkeleton, QuoteNode(name),
             Expr(:call, Val, QuoteNode(token_sym)), __module__, QuoteNode(subject),
             # E: detached immutable owner-spec snapshot, not the live authority.
             Expr(:call, _kernel_capture_child, QuoteNode(name), esc(spec_expr)),
             roots.writes, roots.reads, method_meta, body_source, is_bb,
             # D: snapshot each direct registered callee's identity — the free method's
             # own name + subject/formals + body locals excluded (pending/factory-time).
             _kernel_callee_capture_expr(
                 _kernel_mode2_callee_pairs(name, signature_inputs, block, raw_signature), __module__))))
end

# --- registered-kernel resolver + hygiene/rebind discriminator ---------------
#
# The single load-bearing new substrate accessor (RK 2026-08-26; poc gate 4):
# maps a captured callable NAME/VALUE identity to its registration — def-unique
# Token + declared subject/effect-root metadata — so a cross-kernel call
# (`leapfrog!(fwd)`, and the inner callee of `partial(leapfrog!; …)` /
# `deepcopy(child)`) resolves to a REGISTERED call instead of an opaque Julia
# call. Source/registration PROVENANCE only — no effect closure/lowering. An
# ordinary Julia callable resolves to `nothing` (opaque, per the compiler
# boundary: ordinary Julia is opaque unless explicitly registered).

# --- identity-bound RK-core primitive effect registry ------------------------
#
# A call touching a reactive/self/subject place is admissible ONLY when its CAPTURED
# CALLABLE IDENTITY has a detached effect descriptor — a registered @kernel / sibling
# method / core intrinsic, OR an RK-core PRIMITIVE registered here by EXACT identity.
# Qualification and `!`-spelling are NOT effect evidence (locked compiler boundary:
# ordinary Julia is opaque unless explicitly registered). `Base.fill!` is sanctioned for
# the RIGHT reason — its exact identity is registered with a declared destination write —
# NOT because it is `.`-qualified or bang-suffixed. Captured DETACHED at owner definition
# (`_kernel_capture_callees`, kind `:primitive`), rebind-checked; never reread at analysis.

# The COMPLETE positional-effect descriptor of an RK-core primitive: `token` a stable,
# DISTINCT per-primitive identity (so two primitives never collide in `_RegisteredCall`
# edges / plan caches — the whole registration is keyed by it); `writes` the positional
# actuals written; `reads` the positional actuals read; `result_alias` the positional the
# RETURNED value aliases (`Base.fill!`/`Base.copyto!` return their destination), or
# `nothing`. So `x = Base.fill!(buf, 0)` is known to preserve `x === buf` in later lowering.
# `copy!!` is NOT one of these — it is the RK-core `:intrinsic` with STRONGER STRUCTURAL
# semantics (destination owned-CLOSURE copy, shared authority untouched, `result === dest`,
# shape/type checks), carried by its own registration, never flattened to a positional here.
struct _PrimitiveEffect
    token::Any                        # stable, exact built-in identity token. NEVER inferred from
                                      #   name/spelling (which could collide cross-module).
    arity::Int                        # EXACT accepted positional arity — valid ONLY for that arity
    writes::Tuple{Vararg{Int}}        # positional actuals WRITTEN
    reads::Tuple{Vararg{Int}}         # positional actuals READ
    result_alias::Union{Nothing,Int}  # positional the RESULT aliases (fill!/copyto!/lmul!: dest)
    kind::Symbol                      # :effect (positional writer) | :pure | :rng
    order::Symbol                     # :none | :ordered (rng/effect sequencing, no-CSE)
    borrows::Tuple{Vararg{Int}}       # actual positions the RESULT BORROWS (a lazy view may borrow
                                      #   all) — must NOT be cached/materialized as authoritative
    rng_arg::Union{Nothing,Int}       # the runtime RNG arg position (:rng only) — explicit, not by kind
end
# Compact constructor for an RK-core positional-writer primitive (`Base.fill!`/`copyto!`).
_PrimitiveEffect(token, arity, writes, reads, result_alias) =
    _PrimitiveEffect(token, arity, writes, reads, result_alias, :effect, :none, (), nothing)

# VALUE equality over the WHOLE descriptor: a rebind check must detect any built-in contract drift
# (arity/kind/order/writes/reads/result_alias/borrows/rng_arg), not merely compare the token.
Base.:(==)(a::_PrimitiveEffect, b::_PrimitiveEffect) =
    a.token === b.token && a.arity == b.arity && a.writes == b.writes && a.reads == b.reads &&
    a.result_alias == b.result_alias && a.kind === b.kind && a.order === b.order &&
    a.borrows == b.borrows && a.rng_arg == b.rng_arg

# Pure DISPATCH on the resolved VALUE identity — used AT OWNER DEFINITION (capture time),
# so its result is SNAPSHOTTED detached (no mutable registry, no analysis-time reread).
# `nothing` = not an RK-core positional primitive. Each entry carries a DISTINCT token.
#
# ARITY IS PART OF THE CONTRACT. Definition-time callee capture is arity-BLIND (it keys on
# the callable identity, not the call shape), so a later `Base.copyto!(dest,src,do,so,n)`
# resolves to this SAME registration. The descriptor therefore pins the EXACT positional
# arity it describes (2), and the MethodIR/fixed-point consumer MUST match `length(args) ==
# arity` and DETERMINISTICALLY REJECT any other arity — never silently summarize a
# 5-positional `copyto!` with the 2-arg effect (args 3-5 dropped).
_kernel_primitive_effect(@nospecialize(v)) =
    v === Base.fill!   ? _PrimitiveEffect(Symbol("__rk_primitive_Base_fill!__"),   2, (1,), (2,), 1) :
    v === Base.copyto! ? _PrimitiveEffect(Symbol("__rk_primitive_Base_copyto!__"), 2, (1,), (2,), 1) :
    # RK-core built-in RNG/effect primitives for `refresh_momentum!!` (RK 2026-08-27):
    #   randn!(rng, dest): ordered RNG (arg 1 is the RNG), writes dest (arg 2), result aliases dest.
    v === Random.randn! ?
        _PrimitiveEffect(Symbol("__rk_rng_Random_randn!__"), 2, (2,), (1,), 2, :rng, :ordered, (), 1) :
    #   lmul!(A, dest): reads matrix A (arg 1) + dest (arg 2), writes dest (arg 2), result aliases dest.
    v === LinearAlgebra.lmul! ?
        _PrimitiveEffect(Symbol("__rk_effect_LinearAlgebra_lmul!__"), 2, (2,), (1, 2), 2, :effect, :none, (), nothing) :
    #   rand(rng, T): the 2-positional ordered-RNG form (`rand(rng, Bool)` in step!) — RNG is arg 1,
    #   no reactive WRITE (returns a fresh value); reads both actuals.
    v === Random.rand ?
        _PrimitiveEffect(Symbol("__rk_rng_Random_rand__"), 2, (), (1, 2), nothing, :rng, :ordered, (), 1) :
    #   randexp(rng): the exact 1-positional scalar exponential draw used by the authored NUTS
    #   accept/reject helper.  It reads only the RNG, writes no reactive place, returns a fresh scalar,
    #   and is ordered with every other RNG effect.  This is a built-in authority, not an author macro.
    v === Random.randexp ?
        _PrimitiveEffect(Symbol("__rk_rng_Random_randexp__"), 1, (), (1,), nothing, :rng, :ordered, (), 1) :
    #   eachcol(A): NOT pure — reads A (arg 1) and returns a lazy VIEW iterator BORROWING arg 1 (RK
    #   06:10), so a yielded column aliases A and must NOT be treated as an independent value. No write.
    v === Base.eachcol ?
        _PrimitiveEffect(Symbol("__rk_borrows_Base_eachcol__"), 1, (), (1,), nothing, :pure, :none, (1,), nothing) :
    nothing

# --- exact-identity PURE Base/stdlib primitive SET (RK 2026-08-27 provenance fix) -------------
#
# The locked compiler boundary: an ordinary call receiving a reactive place has UNKNOWN mutation
# unless a DETACHED descriptor exists; `!`-spelling / qualification / `Base.isoperator` are NOT
# evidence. These exact CALLABLE IDENTITIES are RK-core PURE primitives — arbitrary arity, reads
# EVERY actual, writes nothing. They are captured at owner definition as kind `:pure_primitive`
# (token `typeof(target)`, source the exact value), so admission is DEFINITION-TIME + rebind-checked
# and poc emits the captured authored slot — NEVER a live analysis-time spelling lookup. Operators
# (ordinary/dotted/compound) resolve through THIS SAME set by their base identity (RK 06:01), so a
# same-spelled rebound operator cannot bypass the provenance boundary.
# A DEEPLY-IMMUTABLE Tuple of exact callable identities — NOT a mutable IdSet/Dict registry (RK
# 06:08: the locked contract forbids a mutable registry in this provenance path; relocating one would
# not change that). Cold capture, so an identity `===` scan is fine.
#
# NARROWED (RK 06:09): only sound reads-all/no-writes identities — operators, scalar math, predicates,
# type/zero constructors, and non-borrowing navigation. HIGHER-ORDER Base generics (`map`/`filter`/
# `reduce`/`mapreduce`/`sum`/`prod`/`broadcasted`/`materialize`) are EXCLUDED: they take a callable arg
# whose effects are arbitrary, so an arity-blind reads-all descriptor is unsound. `eachcol`/`eachrow`
# are EXCLUDED too — they BORROW arg 1 (return views), carried instead by an exact built-in `borrows`
# descriptor in `_kernel_primitive_effect` (RK 06:10). An unlisted helper stays OPAQUE (negative test).
#
# TRULY MINIMAL (RK 06:21): only the exact identities INDEPENDENTLY OBSERVED as method-body calls in the
# final fixture MethodIR — arithmetic/comparison OPERATORS plus `zero`, `one`, `oftype`, `isnothing`, and
# the range `Colon`, plus `logaddexp`. This category's contract is reads-all AND a FRESH result aliasing
# NO actual, so possibly-aliasing helpers are OUT: `identity`/`getindex`/`first`/`last` (return an actual),
# `min`/`max`/`clamp`/`ifelse` (return an actual), `convert`/`axes`/`eachindex` (can return/borrow arg),
# and the whole higher-order family (`map`/`reduce`/`sum`/…, arbitrary callable-arg effects). `eachcol`
# BORROWS arg 1 → the explicit built-in descriptor in `_kernel_primitive_effect` instead. `dot`/`cholesky`/
# `logdet` appears only as a TOP-LEVEL GRAPH RECIPE in the fixture. An authored consumer additionally exercises
# exact `abs(::builtin AbstractFloat)` and `div(::builtin Integer, ::same Integer)` method-body calls; both are
# admitted only through the same exact-identity, specialization-gated contract. `oftype` delegates to
# `convert` and CAN return a mutable arg2 — the
# final uses are numeric SCALARS (value semantics), so it is retained on the understanding that
# SPECIALIZATION (poc lowering, where concrete types are known) gates it to scalar/deeply-immutable
# actuals and rejects a mutable actual (coordinate with poc). Expansion is an explicit later step, never a
# broad possibly-aliasing whitelist.
const _KERNEL_PURE_PRIMS = (Base.:+, Base.:-, Base.:*, Base.:/, Base.:\, Base.:^, Base.:%,
          Base.:(==), Base.:(!=), Base.:<, Base.:>, Base.:<=, Base.:>=, Base.:!, Base.:&, Base.:|, Base.xor,
          Base.zero, Base.one, Base.oftype, Base.isnothing, Base.length, Base.:(:),
          Base.copy, Base.map,
          Base.exp, Base.log, Base.sqrt, Base.abs, Base.div,
          LogExpFunctions.logaddexp)
# EXACT-IDENTITY VALUE test (`===` scan over the immutable tuple) — used at capture (definition-time
# snapshot) AND in `kernel_rebound` to confirm the current binding is STILL an exact registered pure
# primitive (spoof/rebind rejection). No mutable membership structure.
_kernel_pure_primitive_value(@nospecialize(v)) = any(x -> x === v, _KERNEL_PURE_PRIMS)

# ---- PER-CALLEE SPECIALIZATION domain contracts (RK 06:29/06:30/06:33) -----------------------------
#
# Exact generic identity is NECESSARY but NOT SUFFICIENT: Base generics are EXTENSIBLE, so a custom type
# could define a mutating/opaque method under the SAME `Base.:+`/`length`/`isnothing`/… identity. The
# captured category is therefore a definition-time identity capture PLUS a detached SPECIALIZATION-TIME
# domain contract validated against the CONCRETE argument types (never IR-inspecting the dispatched
# method). The contract is PER CALLEE/TOKEN (and arity) — a single all-args-numeric predicate is wrong:
# `length(proposals)` sees an `Array{<endpoint-state>}` (non-numeric eltype) and `isnothing(stats_f)` sees
# `Nothing` or a registered kernel object (RK 06:33). LOAD-BEARING: `kernel_pure_primitive_domain_ok` is
# the exact predicate poc/the factory call at specialization; a same-generic overload on a custom
# container/number REJECTS unless it is expressed through a registered kernel sibling or sanctioned builtin.
#
# Type-domain leaves. A numeric SCALAR must be RECURSIVELY-SAFE CONCRETE REPRESENTATION, not merely
# `parentmodule ∈ Base/Core + isbits Number` (RK 06:43b): a Base-owned wrapper parameterized by a user
# type — `Rational{UserInt}`, `Complex{UserReal}` — still dispatches user code. Whitelist the primitive
# builtin bit integers/Bool and IEEE floats; admit `Complex`/`Rational` ONLY when every parameter is
# itself admitted (recursive). Final NUTS uses Float32/Float64/Int and passes.
const _KERNEL_SAFE_LEAF_NUMS = (Bool, Int8, Int16, Int32, Int64, Int128,
                                UInt8, UInt16, UInt32, UInt64, UInt128, Float16, Float32, Float64)
const _KERNEL_SAFE_LEAF_INTS = (Bool, Int8, Int16, Int32, Int64, Int128,
                                UInt8, UInt16, UInt32, UInt64, UInt128)
_kernel_dom_builtin(::Type{T}) where {T} = parentmodule(T) in (Base, Core)
function _kernel_dom_num_scalar(::Type{T}) where {T}
    any(x -> x === T, _KERNEL_SAFE_LEAF_NUMS) && return true
    (T <: Complex || T <: Rational) && T isa DataType && length(T.parameters) == 1 &&
        return _kernel_dom_num_scalar(T.parameters[1])       # recursively safe wrapper
    false
end
_kernel_dom_int_scalar(::Type{T}) where {T} = any(x -> x === T, _KERNEL_SAFE_LEAF_INTS)
# a numeric VALUE: a scalar leaf OR a concrete Base numeric array/range (broadcast/array arithmetic).
function _kernel_dom_num_value(::Type{T}) where {T}
    _kernel_dom_num_scalar(T) && return true
    _kernel_dom_builtin(T) || return false
    (T <: Array || T <: AbstractRange) && return _kernel_dom_num_scalar(eltype(T))
    false
end
# a concrete Base CONTAINER (Array/Tuple/range) of ANY element type — the `length` domain.
_kernel_dom_container(::Type{T}) where {T} =
    _kernel_dom_builtin(T) && (T <: Array || T <: Tuple || T <: AbstractRange)
# a builtin RNG — the rng-arg domain of effect primitives.
_kernel_dom_rng(::Type{T}) where {T} =
    T <: Random.AbstractRNG && parentmodule(T) in (Base, Core, Random)

# `isnothing`'s domain: `Nothing`, or a COMPILER-OWNED registered callable/state category with a trusted
# direct lowering (its `=== nothing` cannot be user-subverted) — e.g. a `stats_f=nuts_stats!` kernel
# object. Types resolved at CALL time (forward-declared skeletons/intrinsic are defined below).
_kernel_dom_isnothing(::Type{T}) where {T} =
    T === Nothing || T <: _Mode2KernelSkeleton || T <: _StatefulKernelSkeleton || T <: _KernelIntrinsic

# PER-CALLEE dispatch on the captured exact identity (`reg.source`).
function _kernel_pure_callee_domain_ok(@nospecialize(f), argtypes)
    isempty(argtypes) && return false
    f === Base.copy && return length(argtypes) == 1 && _kernel_dom_num_array(argtypes[1])
    if f === Base.map
        length(argtypes) == 2 || return false
        argtypes[1] === typeof(Base.copy) || return false
        T = argtypes[2]
        T <: Tuple || return false
        return !isempty(T.parameters) &&
            all(t -> t isa Type && _kernel_dom_num_array(t), T.parameters)
    end
    f === Base.length   && return length(argtypes) == 1 && _kernel_dom_container(argtypes[1])
    f === Base.isnothing && return length(argtypes) == 1 && _kernel_dom_isnothing(argtypes[1])
    f === Base.:(:)     && return all(_kernel_dom_int_scalar, argtypes)            # Colon: integer numeric
    f === Base.oftype   && return length(argtypes) == 2 && all(_kernel_dom_num_scalar, argtypes)
    f === Base.abs      && return length(argtypes) == 1 &&
        _kernel_dom_num_scalar(argtypes[1]) && argtypes[1] <: AbstractFloat
    f === Base.div      && return length(argtypes) == 2 && argtypes[1] === argtypes[2] &&
        _kernel_dom_int_scalar(argtypes[1]) && argtypes[1] !== Bool
    (f === Base.zero || f === Base.one) && return length(argtypes) == 1 && _kernel_dom_num_value(argtypes[1])
    # unary transcendentals (RK 14:35 / POC G3): SCALAR-only Real domain — a single-application `exp/log/sqrt`
    # of a numeric scalar leaf is pure and 0-B; NOT admitted over arrays (matrix `exp` is different semantics).
    (f === Base.exp || f === Base.log) &&
        return length(argtypes) == 1 && _kernel_dom_num_scalar(argtypes[1])
    f === Base.sqrt && return length(argtypes) == 1 &&
        (_kernel_dom_num_scalar(argtypes[1]) || _kernel_dom_diag(argtypes[1]))
    f === Base.:* && length(argtypes) == 2 &&
        _kernel_dom_diag(argtypes[1]) && _kernel_dom_num_array(argtypes[2]) &&
        eltype(argtypes[1]) === eltype(argtypes[2]) && return true
    # arithmetic / comparison / logical / logaddexp: numeric leaves OR numeric arrays (broadcast eltype).
    all(_kernel_dom_num_value, argtypes)
end

"""
    kernel_pure_primitive_domain_ok(reg::_KernelRegistration, argtypes) -> Bool

Per-callee SPECIALIZATION admission for a captured `:pure_primitive` call (RK 06:33): `true` iff `reg` is
a pure-primitive registration AND the CONCRETE `argtypes` are in THIS callee's supported domain (numeric
leaf for arithmetic/zero/one/oftype/logaddexp, integer for `Colon`, any-eltype Base container for
`length`, `Nothing`/compiler-owned callable for `isnothing`). A mutating/opaque overload of the same
generic over an unsupported domain is REJECTED.
"""
kernel_pure_primitive_domain_ok(reg, argtypes) =
    reg.kind === :pure_primitive && _kernel_pure_callee_domain_ok(reg.source, argtypes)

# The SAME principle for RK-core BUILT-IN EFFECT primitives (RK 06:30/06:37): each descriptor is an
# authoritative contract ONLY over its supported builtin concrete domain — validated PER exact primitive +
# arity/position (a single coarse all-args predicate is wrong: `lmul!` arg1 is a LinearAlgebra structured
# matrix, `rand`'s arg2 is a `Type{Bool}`). A user method extending the SAME generic over an unsupported
# domain REJECTS. Only these exact built-ins (`:primitive`) receive a detached effect contract.
_kernel_dom_num_array(::Type{T}) where {T} =
    T <: Array && _kernel_dom_builtin(T) && _kernel_dom_num_scalar(eltype(T))
_kernel_dom_num_matrix(::Type{T}) where {T} =
    T <: Matrix && _kernel_dom_builtin(T) && _kernel_dom_num_scalar(eltype(T))
# sanctioned builtin LinearAlgebra structured matrices (or a dense Base matrix) over builtin numeric.
# HARDENING (RK 06:43): the wrapper parentmodule/eltype is NOT enough — a structured wrapper can BACK a
# custom AbstractMatrix/Vector whose getindex/lmul! has arbitrary effects while the outer type still
# belongs to LinearAlgebra. Require the BACKING STORAGE type parameter to itself be a concrete Base numeric
# Matrix (Vector for `Diagonal`). Final `cholesky(m).L :: LowerTriangular{T,Matrix{T}}` passes.
# A concrete builtin numeric `Diagonal{numeric, Vector{numeric}}` (RK 18:34) — the SINGLE shared predicate for
# a sanctioned diagonal mass: the backing storage parameter must be a CONCRETE Base numeric `Vector` (via
# `_kernel_dom_num_array`). A `Diagonal` over a custom `AbstractVector`, a non-`Diagonal` structured wrapper,
# or `UniformScaling` all REJECT. Shared by the `lmul!` lhs domain and the cholesky/logdet/ldiv recipe domain.
function _kernel_dom_diag(::Type{T}) where {T}
    T <: LinearAlgebra.Diagonal || return false
    (T isa DataType && length(T.parameters) >= 2) || return false
    S = T.parameters[2]
    S isa Type && S <: Vector && _kernel_dom_num_array(S)
end

function _kernel_dom_lmul_lhs(::Type{T}) where {T}
    _kernel_dom_num_matrix(T) && return true                   # a dense concrete Base numeric Matrix
    parentmodule(T) === LinearAlgebra || return false
    (T isa DataType && length(T.parameters) >= 2) || return false
    S = T.parameters[2]                                        # the backing storage type
    S isa Type || return false
    if T <: LinearAlgebra.Diagonal
        return _kernel_dom_diag(T)                             # Diagonal backs a concrete Base numeric Vector
    elseif T <: LinearAlgebra.LowerTriangular
        return _kernel_dom_lower_backing(S)                   # concrete Base Matrix OR the canonicalized Adjoint view
    elseif T <: LinearAlgebra.UpperTriangular ||
           T <: LinearAlgebra.UnitLowerTriangular || T <: LinearAlgebra.UnitUpperTriangular ||
           T <: LinearAlgebra.Symmetric || T <: LinearAlgebra.Hermitian
        return _kernel_dom_num_matrix(S)                      # MATRIX-ONLY: no consumer emits an Adjoint backing here
    end
    false
end
# The backing storage of a sanctioned LOWER-triangular wrapper: a concrete Base numeric `Matrix`, OR the
# ALLOCATION-FREE `Adjoint` of one (RK 13:52/13:59). Julia CANONICALIZES `adjoint(UpperTriangular(F))` — POC's
# uplo='U' Cholesky-L view (kernel_nuts.jl:54) — to `LowerTriangular{T,Adjoint{T,Matrix{T}}}`, so the outer
# wrapper is a LowerTriangular whose BACKING is `Adjoint{T,Matrix{T}}`; that adjoint just re-indexes the same
# concrete `F` with no allocation, so `lmul!(chol.L, mom)` stays 0-B. This Adjoint-backing allowance is
# LOWER-triangular ONLY — the exact emitted view — never widened to Upper/Symmetric/Hermitian, which have no
# such consumer and stay matrix-only. Recursively prove the `Adjoint`'s parent is a concrete Base numeric
# `Matrix`; a custom backing under the `Adjoint` (or directly) is rejected.
function _kernel_dom_lower_backing(::Type{S}) where {S}
    _kernel_dom_num_matrix(S) && return true                  # concrete Base numeric Matrix
    S <: LinearAlgebra.Adjoint || return false                # only the adjoint VIEW is additionally allowed
    (S isa DataType && length(S.parameters) >= 2) || return false
    P = S.parameters[2]                                        # the Adjoint's parent (the re-indexed storage)
    P isa Type && _kernel_dom_num_matrix(P)                   # ... which must be a concrete Base numeric Matrix
end
# a supported `rand` sample spec at pos2: a `Type{S}` of a builtin sampleable numeric/Bool scalar, or a
# numeric value spec.
function _kernel_dom_sample_spec(::Type{T}) where {T}
    T <: Type || return _kernel_dom_num_value(T)
    (T isa DataType && length(T.parameters) == 1) || return false
    S = T.parameters[1]
    S isa Type && _kernel_dom_num_scalar(S)
end

# PER-PRIMITIVE dispatch on the captured exact identity (`reg.source`) + arity/position (RK 06:37).
function _kernel_effect_callee_domain_ok(@nospecialize(f), argtypes)
    f === Base.fill!     && return length(argtypes) == 2 && _kernel_dom_num_array(argtypes[1]) &&
                                   _kernel_dom_num_scalar(argtypes[2])
    f === Base.copyto!   && return length(argtypes) == 2 && _kernel_dom_num_array(argtypes[1]) &&
                                   _kernel_dom_num_array(argtypes[2])
    f === Random.randn!  && return length(argtypes) == 2 && _kernel_dom_rng(argtypes[1]) &&
                                   _kernel_dom_num_array(argtypes[2])
    f === LinearAlgebra.lmul! && return length(argtypes) == 2 && _kernel_dom_lmul_lhs(argtypes[1]) &&
                                   _kernel_dom_num_array(argtypes[2])
    f === Random.rand    && return length(argtypes) == 2 && _kernel_dom_rng(argtypes[1]) &&
                                   _kernel_dom_sample_spec(argtypes[2])
    f === Random.randexp && return length(argtypes) == 1 && _kernel_dom_rng(argtypes[1])
    f === Base.eachcol   && return length(argtypes) == 1 && _kernel_dom_num_matrix(argtypes[1])
    false
end

"""
    kernel_builtin_primitive_domain_ok(reg, argtypes) -> Bool

Per-callee/arity SPECIALIZATION admission for a captured RK-core built-in EFFECT primitive (`reg.kind ===
:primitive`, RK 06:37): dispatches on the exact identity — `fill!`(Base Array dest + numeric scalar),
`copyto!`(Base Array dest+src), `randn!`(builtin RNG + numeric Array), `lmul!`(sanctioned LinearAlgebra/
dense matrix + numeric Array), `rand`(builtin RNG + sample `Type`), `randexp`(builtin RNG),
`eachcol`(Base numeric Matrix). A
custom overload over an unsupported domain REJECTS.
"""
kernel_builtin_primitive_domain_ok(reg, argtypes) =
    reg.kind === :primitive && _kernel_effect_callee_domain_ok(reg.source, argtypes)

struct _KernelRegistration
    token::Any               # def-unique/intrinsic Token, or `nothing` for a stateless spec
    kind::Symbol             # :free_method | :object_kernel | :stateless | :intrinsic |
                             # :primitive | :pure_primitive
    subject::Union{Symbol,Nothing}    # Mode-2/intrinsic subject (first positional), else nothing
    write_roots::Tuple{Vararg{Symbol}}
    read_roots::Tuple{Vararg{Symbol}}
    is_bang_bang::Bool       # `!!` strong same-object update registration
    source::Any              # the originating value — the ONLY stable identity handle for a
                             #   token-less stateless spec (so rebind stays sound; see kernel_rebound)
    # `:primitive` only: the detached positional-write descriptor (`Base.fill!`). `nothing`
    # for every other kind — its effect is carried by write_roots/the intrinsic Token.
    primitive_effect::Union{Nothing,_PrimitiveEffect}
end
# 7-arg convenience: every non-primitive registration defaults `primitive_effect=nothing`.
_KernelRegistration(token, kind, subject, wr, rr, bb, source) =
    _KernelRegistration(token, kind, subject, wr, rr, bb, source, nothing)

# A tiny RK-CORE registered intrinsic (NOT authored via `@kernel`): a strong
# structural update carrying a stable, def-unique intrinsic Token so the general
# resolver recognizes it exactly like an `@kernel` definition token (RK 2026-08-26
# source-contract refinement). Provenance only in Increment 1 — the alias-aware
# minimal owned-closure copy is lowered later; no manual field list is stored.
struct _KernelIntrinsic
    name::Symbol
    token::Symbol            # stable, def-unique intrinsic Token (never a gensym)
    subject::Union{Symbol,Nothing}
    is_bang_bang::Bool
end
kernel_token(i::_KernelIntrinsic) = getfield(i, :token)
kernel_subject(i::_KernelIntrinsic) = getfield(i, :subject)
kernel_is_bangbang(i::_KernelIntrinsic) = getfield(i, :is_bang_bang)

# Direct/unprepared execution REJECTS ACTIONABLY (RK 2026-08-26): in Increment 1 an
# intrinsic is captured for PROVENANCE only — there is NO hidden plan, generic
# `deepcopy`, or fallback. The alias-aware minimal owned-closure copy is realized by
# later lowering; until then a bare call must fail loudly, not silently copy.
(i::_KernelIntrinsic)(args...; kwargs...) = throw(ArgumentError(
    "$(getfield(i, :name)) is a registered RK-core intrinsic captured for PROVENANCE " *
    "only in Increment 1 (kind :intrinsic, Token $(getfield(i, :token)), subject " *
    ":$(getfield(i, :subject))); it is NOT executable until lowering lands — no hidden " *
    "plan, generic deepcopy, or fallback. Compose/prepare it through the owning kernel."))

"""
    copy!!(dest, src)

RK-core registered structural strong-update intrinsic (RK 2026-08-26). Contract
(realized by later lowering): returns `dest` identically; fixed shape/type; copies
the compiler-inferred OWNED authoritative closure into existing buffers; leaves
SHARED authority identity/closure untouched; alias-aware minimal physical copies;
incompatible shape / shared authority rejects. Increment 1 registers its
PROVENANCE ONLY — it is recognized through the general registered-kernel/intrinsic
resolver + `!!` metadata, with NO manual field list and NO external dependency.
"""
const copy!! = _KernelIntrinsic(Symbol("copy!!"), Symbol("__rk_intrinsic_copy!!__"),
                                :dest, true)

"""
    kernel_registration(value) -> _KernelRegistration | nothing
    kernel_registration(mod::Module, name::Symbol) -> _KernelRegistration | nothing

Registration provenance of a registered `@kernel` (Mode-1/Mode-2, Token-registered),
a stateless `KernelSpec` (recognized by value, no Token in Increment 1), or an
RK-core `_KernelIntrinsic` (intrinsic Token) — else `nothing` (an ordinary Julia
callable ⇒ opaque, per the compiler boundary). The `(mod, name)` form is the
hygienic name→registration hop poc uses at composition — it reads the module
binding BY IDENTITY, never eval'ing the call.
"""
kernel_registration(::Any) = nothing
kernel_registration(skel::_Mode2KernelSkeleton) =
    _KernelRegistration(kernel_token(skel), :free_method, kernel_subject(skel),
                        kernel_write_roots(skel), kernel_read_roots(skel),
                        kernel_is_bangbang(skel), skel)
kernel_registration(skel::_StatefulKernelSkeleton) =
    _KernelRegistration(kernel_token(skel), :object_kernel, nothing, (), (), false, skel)
kernel_registration(spec::KernelSpec) =
    _KernelRegistration(nothing, :stateless, nothing,
                        Tuple(spec.have_names), Tuple(spec.want_names), false, spec)
kernel_registration(i::_KernelIntrinsic) =
    _KernelRegistration(kernel_token(i), :intrinsic, kernel_subject(i), (), (),
                        kernel_is_bangbang(i), i)
function kernel_registration(mod::Module, name::Symbol)
    isdefined(mod, name) || return nothing
    kernel_registration(getglobal(mod, name))
end

"""
    kernel_rebound(captured::_KernelRegistration, current) -> Bool

Hygiene/rebind discriminator: `true` iff `current` (a value or its registration)
is NOT the same definition the `captured` registration recorded — the name was
rebound to a different `@kernel`/intrinsic, or is no longer a registered kernel. A
Token-registered capture compares by Token; a token-less stateless capture compares
by captured VALUE identity, so two distinct stateless specs read as a rebind — it
NEVER reports unchanged merely because there is no Token.
"""
function kernel_rebound(captured::_KernelRegistration, current)
    if captured.kind === :pure_primitive
        # An exact-identity RK-core PURE primitive (RK 06:01): validated by IDENTITY — `current` must
        # still BE the captured source value AND remain an exact registered pure primitive. A rebind
        # of the authored slot to a different callable (or to a non-pure one) reads as REBOUND, so a
        # same-spelled operator/helper cannot bypass the provenance boundary.
        if current isa _KernelRegistration
            return current.kind !== :pure_primitive || current.source !== captured.source
        end
        return current !== captured.source || !_kernel_pure_primitive_value(current)
    end
    if captured.kind === :primitive
        # An RK-core primitive's target is not a registered kernel, so validate it by re-deriving
        # the detached built-in descriptor and comparing the ENTIRE contract, not merely its token.
        capdesc = captured.primitive_effect
        if current isa _KernelRegistration
            return current.kind !== captured.kind || current.primitive_effect === nothing ||
                   current.primitive_effect != capdesc
        end
        pe = current === nothing ? nothing : _kernel_primitive_effect(current)
        return pe === nothing || pe != capdesc
    end
    cur = current isa _KernelRegistration ? current : kernel_registration(current)
    cur === nothing && return true                 # binding gone / no longer a kernel
    if captured.token !== nothing
        return cur.token !== captured.token         # Token-registered: identity by Token
    end
    # Token-less (stateless): value identity is the only sound handle — two DISTINCT
    # specs are a rebind. NEVER report unchanged just because there is no Token.
    cur.source !== captured.source
end

# --- D: registered-callee identity SNAPSHOT (definition-time provenance) ------
#
# Direct registered callees appearing in captured method bodies get their
# registration identity SNAPSHOTTED at owner/free-kernel definition time — not
# re-resolved from the module global when method_irs later runs. So a callee name
# rebound BEFORE analysis never silently substitutes the rebound definition: the
# detached captured `_KernelRegistration` is authoritative, and `kernel_rebound`
# detects the drift. Callable FIELDS (e.g. a `step_f` kwarg) are factory-time
# registered values, NOT global snapshots, and are out of scope here.

# The binding name(s) on an assignment LHS: `x`, `x::T`, or a `(a, b)` tuple.
function _kernel_binding_names(lhs)
    if lhs isa Symbol
        Symbol[lhs]
    elseif lhs isa Expr && lhs.head === :(::) && lhs.args[1] isa Symbol
        Symbol[lhs.args[1]]
    elseif lhs isa Expr && lhs.head === :tuple
        reduce(vcat, (_kernel_binding_names(a) for a in lhs.args); init = Symbol[])
    else
        Symbol[]
    end
end

# The owner FIELD/PORT names — signature inputs plus recipe-output LHS names — which are
# callable-field/factory-time values, NOT global callees.
function _kernel_owner_field_names(signature_inputs, recipe_stmts)
    names = Set{Symbol}(first(i) for i in signature_inputs)
    for s in recipe_stmts
        s isa LineNumberNode && continue
        stmt = (s isa Expr && s.head === :macrocall) ? s.args[end] : s   # peel @recipe meta
        (stmt isa Expr && stmt.head === :(=)) || continue
        union!(names, _kernel_binding_names(stmt.args[1]))
    end
    names
end

# Names bound by a plain `=` assignment anywhere in `body` — method LOCALS (a callee
# spelled like a local is a local, not a global).
function _kernel_body_local_names(body)
    locals = Symbol[]
    local walk
    walk = function (x)
        x isa Expr || return
        x.head === :(=) && length(x.args) >= 1 && append!(locals, _kernel_binding_names(x.args[1]))
        for a in x.args
            walk(a)
        end
        return
    end
    walk(body)
    locals
end

# The base name of ONE authored formal declaration (peel `::T`, `= default`/`:kw`, and `xs...`).
function _kernel_formal_base_name(a)
    a isa Symbol && return a
    if a isa Expr
        a.head === :(::) && return _kernel_formal_base_name(a.args[1])
        (a.head === :kw || a.head === :(=)) && return _kernel_formal_base_name(a.args[1])
        a.head === :... && return _kernel_formal_base_name(a.args[1])
    end
    nothing
end

# The authored DEFAULT value expressions of a peeled `:call` signature, each paired with the formal
# names IN SCOPE for it (RK 06:21b/06:28). Defaults evaluate LEFT-TO-RIGHT: the i-th formal's default
# sees ONLY the formals declared BEFORE it (positionals precede keywords) — NOT the current or any LATER
# formal. So `f(; x = g(), g = 1)` resolves `g` in `x`'s default to the GLOBAL `g` (Julia:
# `g()=99; f(;x=g(),g=1)=x; f()==99`), and a later formal name matching a global callable must NOT
# suppress capture in an earlier default. Returns `(default_expr, prior_formal_names)` in authored order.
function _kernel_call_default_pairs(call)
    out = Any[]
    (call isa Expr && call.head === :call) || return out
    posargs = Any[]; kwargs = Any[]
    for a in call.args[2:end]
        (a isa Expr && a.head === :parameters) ? append!(kwargs, a.args) : push!(posargs, a)
    end
    prior = Symbol[]
    for a in Iterators.flatten((posargs, kwargs))     # positionals precede keywords in scope order
        (a isa Expr && a.head === :kw) && push!(out, (a.args[2], copy(prior)))
        nm = _kernel_formal_base_name(a); nm === nothing || push!(prior, nm)
    end
    out
end

# PER-METHOD `(body, exclusions)` pairs for a Mode-1 owner. SHARED across methods:
# sibling method names + owner field/port names. PER-METHOD: that method's formals +
# that body's locals — so a formal/local `f` in method A never suppresses a genuine
# registered global `f` call in method B. Each method also contributes a DEFAULTS pair
# (RK 06:21b): defaults evaluate at entry in left-to-right formal scope BEFORE the body — exclude
# siblings/fields + the formals (a later formal cannot appear in an earlier default), NOT body locals.
function _kernel_mode1_callee_pairs(signature_inputs, recipe_stmts, methods)
    shared = Set{Symbol}(m.name for m in methods)
    union!(shared, _kernel_owner_field_names(signature_inputs, recipe_stmts))
    pairs = Any[]
    for m in methods
        excl = copy(shared)
        union!(excl, m.argnames)
        m.vararg === nothing || push!(excl, m.vararg)
        m.kwargs_splat === nothing || push!(excl, m.kwargs_splat)
        union!(excl, _kernel_body_local_names(m.body))
        push!(pairs, (m.body, excl))
        for (dexpr, prior) in _kernel_call_default_pairs(m.call)   # one pair PER default, left-to-right
            dexcl = copy(shared); union!(dexcl, prior)             # only PRIOR formals in scope (RK 06:28)
            push!(pairs, (dexpr, dexcl))
        end
    end
    pairs
end

# The `(body, exclusions)` pair(s) for a Mode-2 free method: its own name + the subject/formals
# (signature inputs) + body locals; plus a DEFAULTS pair (RK 06:21b) in formal scope (siblings has only
# the method's own name; a default sees the subject/formals + globals).
function _kernel_mode2_callee_pairs(name, signature_inputs, block, raw_signature)
    formals = Set{Symbol}(first(i) for i in signature_inputs)
    excl = Set{Symbol}((name,)); union!(excl, formals); union!(excl, _kernel_body_local_names(block))
    pairs = Any[(block, excl)]
    for (dexpr, prior) in _kernel_call_default_pairs(_kernel_peel_signature(raw_signature))
        dexcl = Set{Symbol}((name,)); union!(dexcl, prior)         # own name + only PRIOR formals (RK 06:28)
        push!(pairs, (dexpr, dexcl))
    end
    pairs
end

# De-dot an operator-callee spelling to its BASE identity (RK 06:01: `a .+ b` is `Expr(:call, :.+, …)`,
# and `.+` is not a binding — its base `+` is). A field access / `..` is NOT a dotted operator.
_kernel_dedot_op(s::Symbol) =
    (str = String(s); length(str) >= 2 && str[1] == '.' && str[2] != '.') ? Symbol(str[2:end]) : s
# The base operator of a COMPOUND-assignment head so its identity is captured too (`+=`→`+`, `.+=`→`+`).
# `nothing` for plain `=`/`.=` and for comparisons (`==`/`!=`/`<=`/`>=` and their dotted forms).
function _kernel_compound_base_op(head::Symbol)
    s = String(head)
    startswith(s, ".") && (s = s[2:end])
    (endswith(s, "=") && length(s) > 1 && s[end-1] != '=' && !(s in ("!=", "<=", ">="))) || return nothing
    Symbol(s[1:end-1])
end

# Distinct DIRECT callee references at `:call` heads (and the base operator of COMPOUND-assignment
# heads — RK 06:01, so a `+=`/`.+` operator resolves through the SAME captured pure identity, not a
# spelling loophole) in ONE `body`, EXCLUDING every name in `exclusions` — sibling/subject-method
# names, owner field/port names, and THIS body's formals/locals — which are pending/factory-time
# values that take PRECEDENCE over any same-spelled global. `:quote`/deferred syntax is not descended
# into. Bare `f` → `:f`; dotted `.+` → its base `:+`; `Mod.f` → `(:Mod, :f)`; all immutable.
function _kernel_body_callee_refs(body, exclusions)
    refs = Any[]
    seen = Set{Any}()
    add! = function (ref)
        ref !== nothing && !(ref in seen) && (push!(seen, ref); push!(refs, ref))
    end
    local walk
    walk = function (x)
        x isa Expr || return
        x.head === :quote && return                       # skip quoted/deferred syntax
        if x.head === :call && !isempty(x.args)
            c = x.args[1]
            if c isa Symbol
                b = _kernel_dedot_op(c)                    # `.+` → `+`; bare name unchanged
                add!(b in exclusions ? nothing : b)
            elseif c isa Expr && c.head === :(.) && length(c.args) == 2 &&
                   c.args[1] isa Symbol && c.args[2] isa QuoteNode
                add!((c.args[1], c.args[2].value))
            end
            # The nonseparable integrators use the ordinary Julia ownership-copy
            # idiom `map(copy, tuple)`. Capture the callable VALUE as well as the
            # call head so MethodIR can carry both definition-time identities.
            # This is not spelling authority: specialization later requires exact
            # Base.map/Base.copy identities and a tuple of builtin numeric arrays.
            map_head = c === :map || (c isa Expr && c.head === :(.) &&
                length(c.args) == 2 && c.args[1] === :Base &&
                c.args[2] isa QuoteNode && c.args[2].value === :map)
            if map_head && length(x.args) >= 2
                callable = x.args[2]
                if callable isa Symbol
                    add!(callable in exclusions ? nothing : callable)
                elseif callable isa Expr && callable.head === :(.) &&
                       length(callable.args) == 2 && callable.args[1] isa Symbol &&
                       callable.args[2] isa QuoteNode
                    add!((callable.args[1], callable.args[2].value))
                end
            end
        elseif x.head === :comparison                      # CHAINED `a < b <= c` (RK 06:08): the operators
            for i in 2:2:length(x.args)                    #   are bare Symbols at even positions — capture
                o = x.args[i]                              #   each base identity so it is not spelling-only
                o isa Symbol || continue
                b = _kernel_dedot_op(o)
                add!(b in exclusions ? nothing : b)
            end
        else
            b = _kernel_compound_base_op(x.head)           # `a += b` / `a .+= b` → base op `+`
            b !== nothing && add!(b in exclusions ? nothing : b)
        end
        for a in x.args
            walk(a)
        end
        return
    end
    walk(body)
    refs
end

# The IMMUTABLE AUTHORED-REFERENCE key — the binding SLOT the author actually wrote,
# NOT the resolved target (a target-only key collapses `f` with `M.f` and two module
# aliases, and cannot be re-validated after the qualifier is rebound). `slot` is the
# authored binding in the owner module (`GlobalRef(owner_mod, :f)` bare, or the module
# slot `GlobalRef(owner_mod, :M)` qualified); `field` is `nothing` (bare) or the field
# name (`:f` for `M.f`).
struct _CapturedCalleeRef
    slot::GlobalRef
    field::Union{Symbol,Nothing}
end
# One captured direct callee: the authored slot key + the DETACHED registration payload
# (Token/source/effect metadata) + the resolved target identity (for rebind validation).
struct _CapturedCallee
    ref::_CapturedCalleeRef
    registration::_KernelRegistration
    target::Any
end

# VALUE-SEMANTIC key equality: an independently reconstructed key (same authored module
# IDENTITY + slot name + field) matches the stored one — lookup never depends on reusing
# the captured `cc.ref` object. Module compared by identity; name/field by value.
Base.:(==)(a::_CapturedCalleeRef, b::_CapturedCalleeRef) =
    a.slot.mod === b.slot.mod && a.slot.name === b.slot.name && a.field === b.field
Base.isequal(a::_CapturedCalleeRef, b::_CapturedCalleeRef) = a == b
Base.hash(r::_CapturedCalleeRef, h::UInt) =
    hash(r.field, hash(r.slot.name, hash(objectid(r.slot.mod), h)))

# Re-resolve an authored slot ref through the CURRENT module state (for rebind checks):
# the resolved value, or `nothing` if the slot/qualifier no longer resolves.
function _kernel_resolve_captured_ref(ref::_CapturedCalleeRef)
    slot = ref.slot
    isdefined(slot.mod, slot.name) || return nothing
    base = getglobal(slot.mod, slot.name)
    ref.field === nothing && return base
    (base isa Module && isdefined(base, ref.field)) || return nothing
    getglobal(base, ref.field)
end

# At DEFINITION time: for each authored callee ref build its authored-slot key, resolve
# the target + registration, and keep the REGISTERED ones. An ordinary Julia callee is
# omitted. A tokenless STATELESS direct-call category is REJECTED deterministically (its
# live `KernelSpec` must not be retained as semantic source) until the factory consumes it.
function _kernel_capture_callees(mod::Module, refs)
    caps = _CapturedCallee[]
    for ref in refs
        if ref isa Symbol
            cref = _CapturedCalleeRef(GlobalRef(mod, ref), nothing)
            isdefined(mod, ref) || continue
            target = getglobal(mod, ref)
        else                                       # (M, f) authored qualified
            outer, nm = ref
            cref = _CapturedCalleeRef(GlobalRef(mod, outer), nm)
            isdefined(mod, outer) || continue
            om = getglobal(mod, outer)
            (om isa Module && isdefined(om, nm)) || continue
            target = getglobal(om, nm)
        end
        reg = kernel_registration(target)
        if reg === nothing
            # Not a registered @kernel/intrinsic. Is it an identity-bound RK-core PRIMITIVE
            # (exact `Base.fill!`)? Capture it DETACHED with kind :primitive + its positional
            # descriptor, keyed by the SAME authored-slot ref + rebind-checked by target
            # identity. An ordinary Julia callee (`evil!`, a foreign function) has no
            # descriptor → omitted → opaque → REJECTED by the interprocedural closure. A
            # LOCAL/formal `fill!` never reaches here (excluded at ref collection).
            peff = _kernel_primitive_effect(target)
            if peff === nothing
                # Is it an exact-identity RK-core PURE primitive (including an operator's base
                # identity)? Capture kind :pure_primitive (token
                # typeof(target), source the exact value, no descriptor — reads-all/no-writes, any
                # arity); else omit (opaque → the ownership closure rejects by name).
                _kernel_pure_primitive_value(target) || continue
                reg = _KernelRegistration(typeof(target), :pure_primitive, nothing,
                                          (), (), false, target, nothing)
                push!(caps, _CapturedCallee(cref, reg, target))
                continue
            end
            # keyed by the primitive's OWN distinct token (no cross-primitive collision)
            reg = _KernelRegistration(peff.token, :primitive, nothing,
                                      (), (), false, target, peff)
            push!(caps, _CapturedCallee(cref, reg, target))
            continue
        end
        reg.kind === :stateless && throw(ArgumentError(
            "direct call to a stateless @kernel `$(cref.field === nothing ? cref.slot.name : cref.field)` " *
            "is not a captured provenance category in Increment 1 (its live KernelSpec must " *
            "not be retained as semantic source); compose/prepare it through the factory instead."))
        push!(caps, _CapturedCallee(cref, reg, target))
    end
    Tuple(caps)
end

function _kernel_find_captured_callee(skel, ref::_CapturedCalleeRef)
    for cc in kernel_callee_registrations(skel)
        cc.ref == ref && return cc
    end
    nothing
end

# Look up the captured registration by AUTHORED-SLOT key — no current-global resolution
# (the payload is detached), so a rebind never downgrades this to opaque.
function kernel_callee_registration(skel, ref::_CapturedCalleeRef)
    cc = _kernel_find_captured_callee(skel, ref)
    cc === nothing ? nothing : cc.registration
end

# `true` iff the authored slot/qualifier no longer resolves to the captured target — a
# rebind of the name (or, for `M.f`, of the module `M`). Throws on an unknown ref.
function kernel_callee_rebound(skel, ref::_CapturedCalleeRef)
    cc = _kernel_find_captured_callee(skel, ref)
    cc === nothing && throw(ArgumentError("no captured callee for $(ref)"))
    _kernel_resolve_captured_ref(ref) !== cc.target
end

# The macro-time capture expression: `_kernel_capture_callees(__module__, (refs…))`,
# each ref an IMMUTABLE quoted value (Symbol or `(Symbol,Symbol)`), never a QuoteNode
# wrapping a mutable Expr.
function _kernel_callee_capture_expr(body_excl_pairs, __module__)
    refs = Any[]
    seen = Set{Any}()
    for (body, excl) in body_excl_pairs                # PER-BODY exclusions, unioned refs
        for r in _kernel_body_callee_refs(body, excl)
            !(r in seen) && (push!(seen, r); push!(refs, r))
        end
    end
    ref_exprs = map(refs) do r
        r isa Symbol ? QuoteNode(r) : Expr(:tuple, QuoteNode(r[1]), QuoteNode(r[2]))
    end
    Expr(:call, _kernel_capture_callees, __module__, Expr(:tuple, ref_exprs...))
end

# --- source-marker recognition: @node / deepcopy / partial -------------------
#
# The remaining recognized authoring markers (RK 2026-08-26 GO #5). `@node` is
# PORT-LIFTED into a real recipe node (below), so a marked anonymous subexpression
# becomes a schedulable graph node/port — NOT expanded away. `deepcopy`/`partial`
# are captured in the frozen source; their callee identity is resolved through the
# IDENTITY-CONFIRMED classifier + `kernel_registration`. `!!` (marker #4) is
# recognized off the name. Recognition/promotion only — no effect closure/lowering.

"""
    @node(expr)

Anonymous-node promoter. Inside a `@kernel` recipe, `@node(expr)` is PORT-LIFTED
into a distinct named recipe node (a schedulable graph node/port) — the ONLY way an
anonymous subexpression becomes an independent node (named recipe assignments
auto-node; no arbitrary AST extraction, no cost heuristics). It adds NO second
object-definition surface. Outside a `@kernel` (or if it survives to runtime) it is
a harmless identity.
"""
macro node(ex)
    esc(ex)
end

# (`_kernel_is_node_macro` / `_kernel_has_node_marker` / `_kernel_lift_nodes` — the
# `@node` port-lift — live in authoring.jl ahead of `_kernel_expand`, since that file
# loads first and expands a `@doc`-embedded `@kernel` at load time.)

# The base callee name of a `:call` head (`deepcopy` / `Base.deepcopy` → :deepcopy).
_kernel_callee_name(c) =
    c isa Symbol ? c :
    (c isa Expr && c.head === :(.) && length(c.args) >= 2 && c.args[2] isa QuoteNode) ?
        c.args[2].value : nothing

# (`_kernel_resolve_binding` — the identity-resolving binding lookup — lives in
# authoring.jl alongside the `@node` lift, which also needs it at load time.)

"""
    _kernel_marker_candidate(ex) -> Symbol | nothing

SYNTACTIC, NON-AUTHORITATIVE candidate classification of a call/macrocall AST:
`:node` / `:deepcopy` / `:partial` / `nothing`, matched by NAME only. A foreign
`Evil.deepcopy` or `Evil.@node` matches too — callers MUST confirm identity via
`_kernel_marker_kind(mod, ex)` before treating a candidate as authoritative.
"""
function _kernel_marker_candidate(ex)
    ex isa Expr || return nothing
    if ex.head === :macrocall && !isempty(ex.args) && _kernel_is_node_macro(ex.args[1])
        return :node
    elseif ex.head === :call && !isempty(ex.args)
        name = _kernel_callee_name(ex.args[1])
        name === :deepcopy && return :deepcopy
        name === :partial && return :partial
    end
    nothing
end

"""
    _kernel_marker_kind(mod::Module, ex) -> Symbol | nothing

AUTHORITATIVE marker classification: the syntactic candidate CONFIRMED by resolving
the callee's actual binding in `mod` to the genuine primitive — RK's `@node` macro,
`Base.deepcopy`, or RK's `partial`. A spoof (`Evil.deepcopy`, `Evil.@node`, or a
rebound name) resolves to `nothing`.
"""
function _kernel_marker_kind(mod::Module, ex)
    cand = _kernel_marker_candidate(ex)
    cand === nothing && return nothing
    bound = _kernel_resolve_binding(mod, ex.args[1])
    if cand === :node
        bound === var"@node" ? :node : nothing
    elseif cand === :deepcopy
        bound === Base.deepcopy ? :deepcopy : nothing
    elseif cand === :partial
        bound === partial ? :partial : nothing
    else
        nothing
    end
end
