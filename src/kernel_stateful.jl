# Stateful `@kernel` authoring — Increment 1 SUBSTRATE SKELETON (V7 architecture GO).
#
# A method-bearing `@kernel` body (one that defines inner mutation/orchestration
# methods) is routed here; a methodless body keeps the current byte-identical
# stateless expansion in `authoring.jl` untouched. This increment establishes ONLY
# the substrate: the method-presence discriminator, the unique per-definition Token,
# the explicit-self `KernelObject` / `KernelView` type skeletons, short/long method
# detection with local-scope rejection, and the frozen detached child-capture
# snapshot.
#
# It deliberately contains NO effect-lowering: no MethodIR/SSA, no `compile_update`
# consumption, no epoch codegen, no `@reactive` deletion, no port migration. Those
# are later increments (poc's effect-lowering lane retargets MethodIR onto this
# substrate only after this SHA clears review).

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

# --- (4)+(5) short/long method extraction, explicit-self contract -----------

struct _KernelMethod
    name::Symbol
    form::Symbol             # :short | :long
    self::Symbol             # the explicit first formal (no __self__ magic)
    argnames::Tuple{Vararg{Symbol}}   # immutable — SCALAR (non-splat) formal names
    vararg::Union{Symbol,Nothing}       # positional slurp `xs...` name, or nothing
    kwargs_splat::Union{Symbol,Nothing} # keyword slurp `; kwargs...` name, or nothing
    signature::Any           # the FULL AUTHORED signature AST (where binders +
                             #   constraints + return `::` annotation + splats preserved)
    call::Any                # the peeled `:call` (name + formals, splats retained
                             #   verbatim for forwarding) used for extraction
    body::Any                # the raw method-body AST (handed to poc's emission)
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

function _kernel_extract_method(stmt, form)
    raw_sig = stmt.args[1]                       # full authored signature (unpeeled)
    call = _kernel_peel_signature(raw_sig)       # the underlying :call
    body = stmt.args[2]
    name = call.args[1]
    name isa Symbol || throw(ArgumentError(
        "stateful @kernel method must have a plain name; got $(repr(name))"))
    formals = call.args[2:end]
    # Julia places a keyword `:parameters` block BEFORE the positional formals, so
    # separate kwargs before locating the explicit `self` (the first POSITIONAL).
    kwparams = Any[]
    positionals = formals
    if !isempty(formals) && formals[1] isa Expr && formals[1].head === :parameters
        kwparams = formals[1].args
        positionals = formals[2:end]
    end
    isempty(positionals) && throw(ArgumentError(
        "stateful @kernel method :$name needs an explicit `self` first positional " *
        "formal (explicit self is required — no hidden __self__)"))
    # Self is the first POSITIONAL and must be an ordinary formal — never a slurp
    # (`self...` is nonsensical, and its inner name would otherwise be mis-read as
    # self by the recursive `_kernel_arg_name`).
    _kernel_is_splat(positionals[1]) && throw(ArgumentError(
        "stateful @kernel method :$name self formal cannot be a vararg splat " *
        "$(repr(positionals[1]))"))
    self = _kernel_arg_name(positionals[1])
    self isa Symbol || throw(ArgumentError(
        "stateful @kernel method :$name has an unsupported self formal $(repr(positionals[1]))"))
    # Ordinary (scalar) formal names go into `argnames`. A positional `xs...` slurp
    # and a keyword `; kwargs...` slurp are each recorded SEPARATELY: their names are
    # bindings, not scalar values, so a consumer that iterates `argnames` never
    # mistakes a slurp for a single argument — while the raw `signature`/`call`
    # retain the splat AST verbatim for forwarding.
    argnames = Symbol[]
    vararg = nothing
    for a in positionals[2:end]
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
    _KernelMethod(name, form, self, Tuple(argnames), vararg, kwargs_splat,
                  raw_sig, call, body)
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
    KernelSpec(graph, ports, collect(snap.port_order), collect(snap.have_names),
               collect(snap.want_names), snap.call_signature)
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
struct _StatefulKernelSkeleton{Token,S,M,R}
    name::Symbol
    mod::Module              # the DEFINITION module (for hygienic op GlobalRefs)
    spec::S
    methods::M
    recipe_source::R         # frozen, detached recipe-statement source (see _FrozenExpr)
end
# `Token` is a phantom type parameter — the unique definition token, a per-expansion
# gensym Symbol carried as `Val{Token}` (unique, typed overload identity, no minted
# struct / no world-age hazard / no mutable registry). Supplied explicitly.
_StatefulKernelSkeleton(name::Symbol, ::Val{Token}, mod::Module, spec::S, methods::M,
                        recipe_source::R) where {Token,S,M,R} =
    _StatefulKernelSkeleton{Token,S,M,R}(name, mod, spec, methods, recipe_source)

kernel_token(skel::_StatefulKernelSkeleton{Token}) where {Token} = Token
kernel_module(skel::_StatefulKernelSkeleton) = getfield(skel, :mod)
kernel_spec(skel::_StatefulKernelSkeleton) = getfield(skel, :spec)
kernel_methods(skel::_StatefulKernelSkeleton) = getfield(skel, :methods)

# Thaw a FRESH copy of the captured recipe-statement source (`Expr(:block, …)`), in
# authored order, with full conditional/index/call structure preserved and definition
# line numbers stripped. Each call rebuilds new mutable nodes, so mutating the result
# never affects the stored frozen source or a subsequent read. poc's stateful MethodIR
# consumes this because `Recipe.op` is opaque; pair it with `kernel_module` for
# hygienic resolution of the names inside.
kernel_recipe_ast(skel::_StatefulKernelSkeleton) =
    Expr(:block, (_kernel_thaw_ast(s) for s in getfield(skel, :recipe_source))...)

function Base.show(io::IO, skel::_StatefulKernelSkeleton)
    print(io, "stateful @kernel :", skel.name, " (", length(skel.methods),
          " method(s); Increment-1 skeleton)")
end

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
    spec_expr = _kernel_expand(recipe_block, signature_inputs, call_signature)

    token_sym = gensym(Symbol(name, :_token))
    # The exposed per-method metadata carries the FULL authored signature (where
    # binders + return annotation), the peeled call, and the raw body — enough for
    # Increment 2's MethodIR emission (typed MethodId, return-annotation semantics).
    # The AST fields are embedded as values (immutable, inspectable).
    method_meta = Tuple(
        (; name = m.name, form = m.form, self = m.self, argnames = m.argnames,
           vararg = m.vararg, kwargs_splat = m.kwargs_splat,
           signature = m.signature, call = m.call, body = m.body)
        for m in methods)

    # Capture the recipe-portion SOURCE as a detached, recursively-frozen AST (poc's
    # stateful MethodIR needs it — `Recipe.op` is opaque). Deep-copy each recipe
    # statement before stripping line numbers so the originals feeding `spec_expr`
    # stay untouched; drop bare line-number statements; keep source order + the full
    # conditional/index/call structure. The frozen value is embedded as an immutable
    # literal (exactly like `method_meta`), so it self-quotes into the expansion.
    recipe_source = Tuple(
        _kernel_freeze_ast(s isa Expr ? Base.remove_linenums!(deepcopy(s)) : s)
        for s in recipe_stmts if !(s isa LineNumberNode))

    # A method-bearing `@kernel` binds its OWNER name via a `const` so the bound
    # skeleton value is a stable, immutable binding (safe to capture by identity in
    # a composing owner). A `const` in an unsupported local scope is a Julia error,
    # which IS the required deterministic unsupported-local-scope rejection. The
    # token is a per-expansion gensym Symbol carried as `Val(token_sym)`.
    Expr(:const, Expr(:(=), esc(name),
        Expr(:call, _StatefulKernelSkeleton, QuoteNode(name),
             Expr(:call, Val, QuoteNode(token_sym)), __module__,
             esc(spec_expr), method_meta, recipe_source)))
end
