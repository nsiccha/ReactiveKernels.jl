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
        (; name = m.name, form = m.form, argnames = m.argnames,
           vararg = m.vararg, kwargs_splat = m.kwargs_splat,
           signature = m.signature, call = m.call, body = m.body,
           sibling_calls = m.sibling_calls)
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
struct _Mode2KernelSkeleton{Token,S,M,B}
    name::Symbol
    mod::Module
    subject::Symbol
    spec::S
    write_roots::Tuple{Vararg{Symbol}}
    read_roots::Tuple{Vararg{Symbol}}
    method::M                # (; name, subject, write_roots, read_roots)
    body_source::B           # frozen, detached body-statement source (see _FrozenExpr)
    is_bang_bang::Bool
end
_Mode2KernelSkeleton(name::Symbol, ::Val{Token}, mod::Module, subject::Symbol, spec::S,
                     write_roots, read_roots, method::M, body_source::B,
                     is_bb::Bool) where {Token,S,M,B} =
    _Mode2KernelSkeleton{Token,S,M,B}(name, mod, subject, spec, Tuple(write_roots),
                                      Tuple(read_roots), method, body_source, is_bb)

kernel_token(skel::_Mode2KernelSkeleton{Token}) where {Token} = Token
kernel_module(skel::_Mode2KernelSkeleton) = getfield(skel, :mod)
kernel_spec(skel::_Mode2KernelSkeleton) = getfield(skel, :spec)
kernel_subject(skel::_Mode2KernelSkeleton) = getfield(skel, :subject)
kernel_write_roots(skel::_Mode2KernelSkeleton) = getfield(skel, :write_roots)
kernel_read_roots(skel::_Mode2KernelSkeleton) = getfield(skel, :read_roots)
kernel_methods(skel::_Mode2KernelSkeleton) = (getfield(skel, :method),)
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
                              subject::Symbol, __module__)
    roots = _kernel_subject_effect_roots(block, subject)
    method_meta = (; name = name, subject = subject,
                     write_roots = roots.writes, read_roots = roots.reads)
    # Ports-only spec: the signature ports (the body carries mutations, not recipes).
    spec_expr = _kernel_expand(Expr(:block), signature_inputs, call_signature)
    body_source = Tuple(
        _kernel_freeze_ast(s isa Expr ? Base.remove_linenums!(deepcopy(s)) : s)
        for s in _kernel_body_statements(block) if !(s isa LineNumberNode))
    token_sym = gensym(Symbol(name, :_token))
    is_bb = _kernel_is_bangbang_name(name)
    Expr(:const, Expr(:(=), esc(name),
        Expr(:call, _Mode2KernelSkeleton, QuoteNode(name),
             Expr(:call, Val, QuoteNode(token_sym)), __module__,
             QuoteNode(subject), esc(spec_expr),
             roots.writes, roots.reads, method_meta, body_source, is_bb)))
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

struct _KernelRegistration
    token::Any               # def-unique/intrinsic Token, or `nothing` for a stateless spec
    kind::Symbol             # :free_method | :object_kernel | :stateless | :intrinsic
    subject::Union{Symbol,Nothing}    # Mode-2/intrinsic subject (first positional), else nothing
    write_roots::Tuple{Vararg{Symbol}}
    read_roots::Tuple{Vararg{Symbol}}
    is_bang_bang::Bool       # `!!` strong same-object update registration
end

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
                        kernel_is_bangbang(skel))
kernel_registration(skel::_StatefulKernelSkeleton) =
    _KernelRegistration(kernel_token(skel), :object_kernel, nothing, (), (), false)
kernel_registration(spec::KernelSpec) =
    _KernelRegistration(nothing, :stateless, nothing,
                        Tuple(spec.have_names), Tuple(spec.want_names), false)
kernel_registration(i::_KernelIntrinsic) =
    _KernelRegistration(kernel_token(i), :intrinsic, kernel_subject(i), (), (),
                        kernel_is_bangbang(i))
function kernel_registration(mod::Module, name::Symbol)
    isdefined(mod, name) || return nothing
    kernel_registration(getglobal(mod, name))
end

"""
    kernel_rebound(captured::_KernelRegistration, current) -> Bool

Hygiene/rebind discriminator: `true` iff `current` (a value or its registration)
is NOT the same definition the `captured` registration recorded — the name was
rebound to a different `@kernel`, or is no longer a registered kernel. The Token
is the stable hygienic identity; a stateless capture (no Token) is treated as
un-reboundable here (identity is by value at the call site).
"""
function kernel_rebound(captured::_KernelRegistration, current)
    cur = current isa _KernelRegistration ? current : kernel_registration(current)
    cur === nothing && return true                 # binding gone / no longer a kernel
    captured.token === nothing && return false     # stateless: no Token identity
    cur.token !== captured.token
end
