# Method-bearing `@kernel` objects.
#
# `@kernel` has two modes here (discriminated in `authoring.jl`): a methodless
# recipe body ⇒ a stateless `KernelSpec`; a body with nested methods ⇒ an OBJECT
# kernel with an IMPLICIT synthesized receiver (no `self` formal, bare fields,
# `__self__` only as a sibling object-pass actual) whose pure, straight-line
# methods are transparent endpoint graphs. This file holds the method
# discriminator and extraction, the detached owner-spec snapshot, the pure
# endpoint templates, the object skeleton with its binding/`extract` surface, the
# recognition of stateful free methods (which `@kernel` rejects here), and `@node`.

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
# Tuple lengths describe graph metadata, not numerical execution types. Keep
# them out of the snapshot type so reconstruction does not compile afresh for
# each graph topology; the frozen contents and operation identities stay exact.
# The call signature describes the argument ABI. Retain its type so stateful
# construction can infer the signature binder's resolved argument tuple.
struct _ChildSnapshot{S}
    name::Symbol
    values::Tuple{Vararg{Value}}
    recipes::Tuple{Vararg{Recipe}}
    producers::Tuple{Vararg{Tuple{ValueId,Tuple{Vararg{Int}}}}}
    aliases::Tuple{Vararg{Pair{ValueId,ValueId}}}
    version::Int
    ports::Tuple{Vararg{Pair{Symbol,ValueId}}}
    port_order::Tuple{Vararg{Symbol}}
    have_names::Tuple{Vararg{Symbol}}
    want_names::Tuple{Vararg{Symbol}}
    call_signature::S
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
    values = Dict{ValueId,Value}(v.id => v for v in snap.values)
    recipes = Recipe[r for r in snap.recipes]
    producers = Dict{ValueId,Vector{Int}}(id => collect(idxs) for (id, idxs) in snap.producers)
    aliases = Dict{ValueId,ValueId}(p.first => p.second for p in snap.aliases)
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
function _kernel_endpoint_is_bang(name)
    name isa Expr && name.head === :(.) && length(name.args) == 2 || return false
    leaf = name.args[2]
    leaf = leaf isa QuoteNode ? leaf.value : leaf
    leaf isa Symbol && _kernel_endpoint_is_bang(leaf)
end

# A pure endpoint body is straight-line: loops, closures, comprehensions and
# statement-position branches are control flow.  A branch in VALUE position
# — a ternary, an `if` expression that is the body's result, `a && b` — is a
# lazy selection the recipe lowering owns (`_kernel_branch_parts`), so it
# is straight-line like any other expression.  An authored `plate(...) do`
# cell is likewise an RK subgraph the recipe lowering owns
# (`_kernel_authored_plate_expr`), not an opaque closure: it is straight-line
# whenever its threaded arguments and its cell body are.  A `let` only scopes
# its bindings, which the recipe lowering resolves (`_kernel_free_ports_flow!`),
# so it is straight-line whenever its bindings and body are.
_kernel_endpoint_callee_name(callee::Symbol) = callee
_kernel_endpoint_callee_name(callee::GlobalRef) = callee.name
function _kernel_endpoint_callee_name(callee)
    callee isa Expr && callee.head === :(.) && length(callee.args) == 2 || return nothing
    leaf = callee.args[2]
    leaf isa QuoteNode ? leaf.value : nothing
end
_kernel_endpoint_plate_call(call) =
    call isa Expr && call.head === :call && !isempty(call.args) &&
    _kernel_endpoint_callee_name(call.args[1]) === :plate

function _kernel_endpoint_has_nonstraight(x)
    x isa Expr || return false
    if x.head === :do && length(x.args) == 2 && _kernel_endpoint_plate_call(x.args[1])
        call, lambda = x.args
        lambda isa Expr && lambda.head === :(->) && length(lambda.args) == 2 ||
            return true
        return any(_kernel_endpoint_has_nonstraight, call.args[2:end]) ||
               _kernel_endpoint_has_nonstraight(lambda.args[2])
    end
    x.head in (:for, :while, :try, :function, :->,
               :comprehension, :generator) && return true
    if x.head === :block
        statements = Any[arg for arg in x.args if !(arg isa LineNumberNode)]
        for (index, statement) in enumerate(statements)
            statement isa Expr && statement.head === :if &&
                index != length(statements) && return true
            _kernel_endpoint_has_nonstraight(statement) && return true
        end
        return false
    end
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
        _kernel_endpoint_is_bang(x.args[1]) && return true
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
            # `inv(target, z)` reads the forward endpoint `target`'s graph.
            name === :inv && length(x.args) == 3 && x.args[2] isa Symbol &&
                (name = x.args[2])
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
        # Property access stores its field as a QuoteNode and must not rename
        # it.  Dotted-call syntax (`f.(x)`), however, stores the call arguments
        # in an Expr(:tuple, ...).  Those arguments may refer to method-local
        # recipes and therefore need the same hygienic rename as an ordinary
        # call argument.
        tail = x.args[2]
        renamed_tail = tail isa Expr && tail.head === :tuple ?
                       _kernel_endpoint_rename(tail, renames) : tail
        return Expr(:(.), _kernel_endpoint_rename(x.args[1], renames), renamed_tail)
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
    # In-place endpoint calls (`_kernel_endpoint_in_place_call!`): callee key =>
    # (builder local, spec expression, input count, display name), and the keys
    # the current method reads.
    arm_children::Dict{Any,Tuple{Symbol,Any,Int,Symbol}}
    method_arm_keys::Vector{Any}
end

function _kernel_endpoint_named_output!(st::_KernelEndpointBuildState, base::Symbol)
    count = get(st.lifted_names, base, 0) + 1
    st.lifted_names[base] = count
    count == 1 ? base : Symbol(st.current_method, "__", base, "__", count)
end

_kernel_scoped_port(scope::Symbol, name::Symbol) = Symbol(scope, ".", name)

"""
Clone a bound child endpoint under its authored owner-port name. The scoped
names are graph metadata only: operations remain the same positional Julia
callables, while plans and extracted ports can distinguish `standard.logpdf`
from the enclosing object's public `logpdf`.
"""
function _kernel_scope_spec(spec::KernelSpec, scope::Symbol)
    graph = Graph()
    mapped = Dict{ValueId,Value}()
    for id in sort!(collect(keys(spec.graph.values)))
        source = spec.graph.values[id]
        mapped[id] = value!(graph, _kernel_scoped_port(scope, source.name),
                            valtype(source))
    end
    for id in sort!(collect(keys(spec.graph.aliases)))
        _kernel_inline_alias!(graph, mapped[id],
            mapped[canon_id(spec.graph, id)], "scoped child alias")
    end
    for recipe in spec.graph.recipes
        _add_recipe!(graph,
            Tuple(mapped[value.id] for value in recipe.inputs),
            Tuple(mapped[value.id] for value in recipe.outputs),
            recipe.op, recipe.cost, recipe.cse_key,
            recipe.effectful, recipe.source)
    end
    _reindex_producers!(graph)
    scoped(name) = _kernel_scoped_port(scope, name)
    ports = Dict{Symbol,Value}(
        scoped(name) => mapped[value.id] for (name, value) in spec.ports)
    KernelSpec(graph, ports, scoped.(spec.port_order), scoped.(spec.have_names),
               scoped.(spec.want_names), spec.call_signature)
end

function _kernel_endpoint_binding_spec!(st::_KernelEndpointBuildState,
                                        root::Symbol, method::Symbol)
    key = (root, method)
    get!(st.binding_specs, key) do
        var = gensym(Symbol(root, :_, method, :_endpoint))
        bound = Expr(:call, GlobalRef(Core, :getfield), st.bindings_var, QuoteNode(root))
        endpoint = Expr(:call, GlobalRef(Base, :getproperty), bound, QuoteNode(method))
        rhs = Expr(:call, GlobalRef(@__MODULE__, :_kernel_scope_spec),
                   endpoint, QuoteNode(root))
        push!(st.binding_prelude, :($var = $rhs))
        var
    end
end

# A straight endpoint call (a sibling, a bound child or an `inv`) is lifted
# into its own recipe ahead of the statement that spells it, where it splices
# as graph recipes. A call in a lazy branch arm, or one reading a name a
# surrounding `let` binds, cannot leave its position: lifting the first would
# evaluate it when the arm is not taken, and the second would read an
# unrelated port of the same name. Such a call stays in place and applies the
# endpoint's graph prepared once by the object's builder
# (`_kernel_endpoint_in_place_call!`). `scoped` holds the names a surrounding
# `let` binds; they shadow endpoint and owner names. `lazy` marks a position
# Julia evaluates only on one branch.
function _kernel_endpoint_lift!(st::_KernelEndpointBuildState, x, lifted::Vector{Any},
                                scoped::Set{Symbol} = Set{Symbol}(), lazy::Bool = false)
    x isa Expr || return x
    x.head in (:quote, :inert) && return x
    x.head === :let && return _kernel_endpoint_lift_let!(st, x, lifted, scoped, lazy)
    if x.head === :call && !isempty(x.args)
        callee = x.args[1]
        args = Any[_kernel_endpoint_lift!(st, a, lifted, scoped, lazy) for a in x.args[2:end]]
        in_place = lazy || any(arg -> _kernel_endpoint_reads(arg, scoped), args)
        if callee isa Symbol && callee in st.endpoint_names && !(callee in scoped)
            in_place && return _kernel_endpoint_in_place_call!(
                st, callee, callee, () -> st.endpoint_vars[callee],
                Any[st.residual_owner_names...; args...], scoped)
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
        elseif callee === :inv && !(callee in scoped) && length(args) == 2 &&
               args[1] isa Symbol && !(args[1] in scoped) &&
               haskey(st.inverse_targets, args[1])
            target = args[1]
            value = args[2]
            input_name, input_type = st.inverse_targets[target]
            inverse_spec() = Expr(:call, GlobalRef(@__MODULE__, :extract),
                Expr(:parameters,
                     Expr(:kw, :have, Expr(:tuple,
                         (QuoteNode(n) for n in st.residual_owner_names)...,
                         QuoteNode(target))),
                     Expr(:kw, :want, QuoteNode(input_name))),
                st.endpoint_vars[target])
            in_place && return _kernel_endpoint_in_place_call!(
                st, (:inv, target), Symbol("inv(", target, ")"), inverse_spec,
                Any[st.residual_owner_names...; value], scoped)
            value isa Symbol || throw(ArgumentError(
                "inv($target, value) in :$(st.current_method) requires a named value port"))
            placeholder = gensym(Symbol(target, :_inverse_call))
            st.nested_specs[placeholder] = inverse_spec()
            out = _kernel_endpoint_named_output!(st, input_name)
            actuals = Any[st.residual_owner_names...; value]
            push!(lifted, Expr(:(=), Expr(:(::), out, input_type),
                               Expr(:call, placeholder, actuals...)))
            return out
        end
        target = _kernel_endpoint_property_target(callee, st.owner_set)
        if target !== nothing && !(target[1] in scoped)
            root, method = target
            spec = _kernel_endpoint_binding_spec!(st, root, method)
            in_place && return _kernel_endpoint_in_place_call!(
                st, (root, method), _kernel_scoped_port(root, method), () -> spec,
                args, scoped)
            all(arg -> arg isa Symbol, args) || throw(ArgumentError(
                "bound endpoint `$root.$method` in :$(st.current_method) requires named port arguments"))
            placeholder = gensym(Symbol(root, :_, method, :_call))
            st.nested_specs[placeholder] = spec
            out = _kernel_endpoint_named_output!(
                st, _kernel_scoped_port(root, method))
            out_type = Expr(:call, GlobalRef(@__MODULE__, :valtype),
                            Expr(:call, GlobalRef(Base, :only),
                                 Expr(:call, GlobalRef(@__MODULE__, :outputs), spec)))
            push!(lifted, Expr(:(=), Expr(:(::), out, out_type),
                               Expr(:call, placeholder, args...)))
            return out
        end
        return Expr(:call, _kernel_endpoint_lift!(st, callee, lifted, scoped, lazy), args...)
    end
    Expr(x.head, (_kernel_endpoint_lift!(st, a, lifted, scoped,
                                         lazy || _kernel_endpoint_lazy_child(x, i))
                  for (i, a) in enumerate(x.args))...)
end

# The positions of `?:`/`if`/`elseif`, `&&` and `||` that Julia evaluates only
# on one branch: everything but the condition and the first operand
# (`_kernel_straight_child` is the recipe lowering's twin).
_kernel_endpoint_lazy_child(x::Expr, i::Int) =
    x.head in (:if, :elseif, :(&&), :(||)) && i > 1

_kernel_endpoint_reads(x, names::Set{Symbol}) =
    x isa Symbol ? x in names :
    x isa Expr && !(x.head in (:quote, :inert)) &&
        any(arg -> _kernel_endpoint_reads(arg, names), x.args)

# An endpoint call left at its position. The object's builder binds the
# callee's graph, prepared once, to a local the call reads
# (`_kernel_endpoint_arm_child`). The local is named `prepared <callee>`: no
# port name contains a space, so no port of the method can shadow it, and the
# name has no leading `#`, so lowering keeps it as spelled
# (`_lowering_gensym`) and `code_expr` contains it. It is assigned once,
# before the first endpoint that reads it. `actuals` are the callee's inputs in order: the residual owner
# ports, then the call's arguments. An owner port a surrounding `let` rebinds
# would pass the `let`'s value instead, so that is refused.
function _kernel_endpoint_in_place_call!(st::_KernelEndpointBuildState, key, name::Symbol,
                                         spec_expr, actuals::Vector{Any},
                                         scoped::Set{Symbol})
    for owner in st.residual_owner_names
        owner in scoped && throw(ArgumentError(
            "pure endpoint :$(st.current_method) calls :$name inside a `let` that " *
            "binds :$owner, but :$name reads the owner port :$owner; rename the " *
            "`let` binding"))
    end
    child = get!(st.arm_children, key) do
        (Symbol("prepared ", name), spec_expr(), length(actuals), name)
    end
    key in st.method_arm_keys || push!(st.method_arm_keys, key)
    Expr(:call, child[1], actuals...)
end

# Bindings are sequential: each sees the earlier ones, and the body sees all of
# them plus the names it assigns itself (also local to the `let`).
function _kernel_endpoint_lift_let!(st::_KernelEndpointBuildState, x, lifted::Vector{Any},
                                    scoped::Set{Symbol}, lazy::Bool)
    bindings = x.args[1]
    list = Meta.isexpr(bindings, :block) ? bindings.args : Any[bindings]
    inner = copy(scoped)
    rebuilt = Any[]
    for binding in list
        if Meta.isexpr(binding, :(=)) && length(binding.args) == 2
            push!(rebuilt, Expr(:(=), binding.args[1],
                _kernel_endpoint_lift!(st, binding.args[2], lifted, inner, lazy)))
            _kernel_bound_names!(inner, binding.args[1])
        else
            push!(rebuilt, binding)
            _kernel_bound_names!(inner, binding)
        end
    end
    _kernel_endpoint_let_assigned!(inner, x.args[2])
    Expr(:let, Meta.isexpr(bindings, :block) ? Expr(:block, rebuilt...) : only(rebuilt),
         _kernel_endpoint_lift!(st, x.args[2], lifted, inner, lazy))
end

function _kernel_endpoint_let_assigned!(names::Set{Symbol}, x)
    x isa Expr || return names
    x.head in (:quote, :inert) && return names
    Meta.isexpr(x, :(=)) && length(x.args) == 2 &&
        _kernel_bound_names!(names, x.args[1])
    foreach(arg -> _kernel_endpoint_let_assigned!(names, arg), x.args)
    names
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
        Dict{Any,Tuple{Symbol,Any,Int,Symbol}}(), Any[],
    )

    endpoint_exprs = Dict{Symbol,Any}()
    endpoint_arm_keys = Dict{Symbol,Vector{Any}}()
    for name in _kernel_endpoint_toposort(endpoints)
        m = only(filter(x -> x.name === name, endpoints))
        state.current_method = name
        empty!(state.nested_specs)
        empty!(state.lifted_names)
        empty!(state.method_arm_keys)
        body = _kernel_endpoint_lower_body(
            m, state, haskey(inverses, name) ? (name, inverses[name]) : nothing)
        endpoint_arm_keys[name] = copy(state.method_arm_keys)
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
    prepared_children = Set{Any}()
    for name in _kernel_endpoint_toposort(endpoints)
        # Each in-place callee is prepared once, after its own graph and before
        # the first endpoint that reads it (a single assignment, so the
        # endpoint's recipe closures capture it unboxed).
        for key in endpoint_arm_keys[name]
            key in prepared_children && continue
            push!(prepared_children, key)
            var, spec_expr, nactuals, display = state.arm_children[key]
            push!(builder_body, :($var = $(GlobalRef(@__MODULE__, :_kernel_endpoint_arm_child))(
                $(QuoteNode(display)), $spec_expr, $nactuals)))
        end
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

# The callee of an in-place endpoint call (`_kernel_endpoint_in_place_call!`):
# its graph prepared once, applied to the call's actuals in input order.
function _kernel_endpoint_arm_child(name::Symbol, spec::KernelSpec, nactuals::Int)
    n = length(inputs(spec))
    n == nactuals || throw(ArgumentError(
        "endpoint :$name is applied in place to $nactuals input(s) but its graph has $n " *
        "($(join((v.name for v in inputs(spec)), ", ")))"))
    _KernelPreparedChild(name, prepare(spec))
end

"""
    _KernelObjectSkeleton

Supertype of the object a method-bearing `@kernel` binds. The pure object
machinery here (endpoint properties, `extract`, binding kernel-valued ports,
endpoint splicing inside another `@kernel`) reads an object only through
[`_kernel_pure_skeleton`](@ref), so a downstream front end can define a subtype
that carries further metadata alongside a pure `_StatefulKernelSkeleton`.
"""
abstract type _KernelObjectSkeleton end

# The object a method-bearing `@kernel` binds here: the owner spec, detached as
# an immutable snapshot, and the construction-time pure endpoint templates.
struct _StatefulKernelSkeleton{SN,E} <: _KernelObjectSkeleton
    name::Symbol
    spec_snapshot::SN        # DETACHED IMMUTABLE owner-spec provenance (a `_ChildSnapshot`)
    endpoint_templates::E    # construction-time pure-method KernelSpec templates
end

"""
    _kernel_pure_skeleton(object::_KernelObjectSkeleton) -> _StatefulKernelSkeleton

The pure skeleton behind `object`; a subtype defined downstream returns the one
it wraps.
"""
_kernel_pure_skeleton(skel::_StatefulKernelSkeleton) = skel

# FRESH reconstructed planning `KernelSpec` per call from the detached snapshot — never
# a shared live authority, so mutating a returned spec/graph cannot change a later read.
kernel_spec(skel::_KernelObjectSkeleton) =
    _kernel_reconstruct(getfield(_kernel_pure_skeleton(skel), :spec_snapshot))
# Immutable port metadata surface (the authored field/port names, a Tuple).
kernel_port_names(skel::_KernelObjectSkeleton) =
    _kernel_snapshot_port_names(getfield(_kernel_pure_skeleton(skel), :spec_snapshot))
kernel_endpoint_names(skel::_KernelObjectSkeleton) =
    getfield(_kernel_endpoint_templates(skel), :method_names)
_kernel_endpoint_templates(skel::_KernelObjectSkeleton) =
    getfield(_kernel_pure_skeleton(skel), :endpoint_templates)
_kernel_object_name(skel::_KernelObjectSkeleton) =
    getfield(_kernel_pure_skeleton(skel), :name)

function Base.show(io::IO, skel::_StatefulKernelSkeleton)
    print(io, "@kernel object :", getfield(skel, :name), " (",
          length(kernel_endpoint_names(skel)), " pure endpoint(s))")
end

"""
    KernelObjectSpec

A transparent specialization produced by binding kernel-valued ports of a
method-bearing `@kernel` object. Pure endpoint properties such as
`normal.logpdf` are ordinary [`KernelSpec`](@ref) views; use [`extract`](@ref)
to choose any other named HAVE/WANT cut. Inside another `@kernel`, a constructed
endpoint may bind owner graph ports by name, for example
`normal(; location = μ, scale = σ, log_scale = log_σ).logpdf(x)`. Those names
form an authoritative HAVE cut and the endpoint graph is spliced transparently.
Literal or computed owner values are materialized as hygienic caller recipes.
"""
struct KernelObjectSpec{S,B}
    skeleton::S
    bindings::B
end

kernel_endpoint_names(obj::KernelObjectSpec) =
    kernel_endpoint_names(getfield(obj, :skeleton))

_kernel_object_skeleton(skel::_KernelObjectSkeleton) = skel
_kernel_object_skeleton(obj::KernelObjectSpec) = getfield(obj, :skeleton)
_kernel_object_bindings(::_KernelObjectSkeleton) = NamedTuple()
_kernel_object_bindings(obj::KernelObjectSpec) = getfield(obj, :bindings)

function _kernel_object_owner_spec(object)
    skel = _kernel_object_skeleton(object)
    spec = kernel_spec(skel)
    bindings = _kernel_object_bindings(object)
    bound = Set{Symbol}()
    for name in keys(bindings)
        push!(bound, name)
    end
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
    templates = _kernel_endpoint_templates(skel)
    missing = Symbol[name for name in templates.binding_names if !(name in keys(bindings))]
    isempty(missing) || throw(ArgumentError(
        "kernel object :$(_kernel_object_name(skel)) still needs transparent kernel binding(s): " *
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
    templates = _kernel_endpoint_templates(skel)
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
function extract(object::Union{_KernelObjectSkeleton,KernelObjectSpec};
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
            templates = _kernel_endpoint_templates(_kernel_object_skeleton(object))
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
    templates = _kernel_endpoint_templates(skel)
    existing = _kernel_object_bindings(object)
    remaining = Symbol[name for name in templates.binding_names if !(name in keys(existing))]
    isempty(args) && throw(ArgumentError(
        "kernel object :$(_kernel_object_name(skel)) construction needs at least one bound kernel"))
    length(args) <= length(remaining) || throw(ArgumentError(
        "kernel object :$(_kernel_object_name(skel)) has $(length(remaining)) unbound kernel port(s), " *
        "but received $(length(args))"))
    for (name, arg) in zip(remaining, args)
        arg isa Union{KernelSpec,_KernelObjectSkeleton,KernelObjectSpec} ||
            throw(ArgumentError(
                "kernel-valued port :$name requires a transparent KernelSpec or @kernel object, " *
                "got $(typeof(arg))"))
    end
    additions = NamedTuple{Tuple(remaining[1:length(args)])}(Tuple(args))
    KernelObjectSpec(skel, merge(existing, additions))
end

(skel::_KernelObjectSkeleton)(arg, rest...) = _kernel_bind_object(skel, (arg, rest...))
(obj::KernelObjectSpec)(arg, rest...) = _kernel_bind_object(obj, (arg, rest...))

function (object::Union{_KernelObjectSkeleton,KernelObjectSpec})(;
        have = _KERNEL_DEFAULT_BOUNDARY, want = _KERNEL_DEFAULT_BOUNDARY)
    extract(object; have = have, want = want)
end

function Base.getproperty(skel::_KernelObjectSkeleton, name::Symbol)
    name in fieldnames(typeof(skel)) && return getfield(skel, name)
    name in kernel_endpoint_names(skel) && return extract(skel; want = name)
    getfield(skel, name)
end

function Base.getproperty(obj::KernelObjectSpec, name::Symbol)
    name in fieldnames(typeof(obj)) && return getfield(obj, name)
    name in kernel_endpoint_names(obj) && return extract(obj; want = name)
    getfield(obj, name)
end

Base.propertynames(skel::_KernelObjectSkeleton, private::Bool = false) =
    Tuple(unique((fieldnames(typeof(skel))..., kernel_endpoint_names(skel)...)))
Base.propertynames(obj::KernelObjectSpec, private::Bool = false) =
    Tuple(unique((fieldnames(typeof(obj))..., kernel_endpoint_names(obj)...)))

Base.keys(object::Union{_KernelObjectSkeleton,KernelObjectSpec}) =
    keys(_kernel_object_full_spec(object))
Base.getindex(object::Union{_KernelObjectSkeleton,KernelObjectSpec}, name::Symbol) =
    _kernel_object_full_spec(object)[name]
Base.haskey(object::Union{_KernelObjectSkeleton,KernelObjectSpec}, name::Symbol) =
    haskey(_kernel_object_full_spec(object), name)
kernel_graph(object::Union{_KernelObjectSkeleton,KernelObjectSpec}) =
    kernel_graph(_kernel_object_full_spec(object))

# Route a method-bearing `@kernel` here: split the body into recipe statements
# and method definitions, extract the methods, expand the recipe portion through
# the stateless machinery (`_kernel_expand`) and build the pure endpoint
# templates. `_kernel_object_definition` returns those parts together with the
# expression constructing the pure skeleton, for a front end that adds metadata.
function _kernel_object_definition(name, signature_inputs, call_signature, block,
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
    endpoint_templates = _kernel_endpoint_templates_expr(
        signature_inputs, recipe_stmts, methods, __module__)
    skeleton = Expr(:call, _StatefulKernelSkeleton, QuoteNode(name),
        # Capture a DETACHED IMMUTABLE snapshot of the owner spec at definition
        # time instead of storing the live mutable authority.
        Expr(:call, _kernel_capture_child, QuoteNode(name), esc(spec_expr)),
        # Endpoint recipes are authored in the caller's module just like the
        # owner spec. Escaping the generated builder preserves imported and
        # module-qualified pure callees (`erfc`, `SpecialFunctions.erfc`) as
        # ordinary Julia global bindings instead of resolving them in RK.
        esc(endpoint_templates))
    (; recipe_stmts, methods, skeleton)
end

# A method-bearing `@kernel` binds its OWNER name via a `const` so the bound
# skeleton value is a stable, immutable binding (safe to capture by identity in
# a composing owner). A `const` in an unsupported local scope is a Julia error,
# which IS the required deterministic unsupported-local-scope rejection.
function _kernel_stateful_expand(name, signature_inputs, call_signature, block,
                                 __module__)
    definition = _kernel_object_definition(
        name, signature_inputs, call_signature, block, __module__)
    Expr(:const, Expr(:(=), esc(name), definition.skeleton))
end

# --- Mode-2 free-method recognition (V7 implicit-field pivot) ----------------
#
# A METHODLESS `@kernel name(sig) = body` whose body MUTATES a field of the FIRST
# positional subject (`subject.field = …` / `subject.field op= …`, incl. `@.`
# broadcast) is a Mode-2 FREE METHOD (e.g. `leapfrog!(phasepoint; stepsize)`),
# independent of `!` spelling (bang is a naming convention, not the
# discriminator); so is any `!!` name. ReactiveKernels compiles pure kernels
# only, so `@kernel` recognizes such a body in order to reject it. These
# recognizers are exposed for a front end that compiles stateful methods.

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

# `@kernel name!!` — the strong same-object update registration (marker #4). The
# `!!` suffix is recognized off the stored name; no extra storage is needed.
_kernel_is_bangbang_name(name::Symbol) = endswith(String(name), "!!")

# --- @node ---------------------------------------------------------------------
#
# `@node` is PORT-LIFTED into a real recipe node (below), so a marked anonymous
# subexpression becomes a schedulable graph node/port — NOT expanded away.

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
