# Declarative construction of the existing Graph/Recipe model.
#
# This file deliberately adds no planner or runtime abstraction. `@kernel`
# expands to ordinary `Graph`, `Value`, and `Recipe` construction, while
# `KernelSpec` only remembers stable names and a default have/want boundary.

"""
    KernelSpec

A declaratively authored kernel graph. `graph` is the ordinary [`Graph`](@ref)
consumed by the planner, `ports` maps author-facing names to graph [`Value`](@ref)
objects, and `inputs(spec)` / `outputs(spec)` report the default have/want
boundary declared by [`@kernel`](@ref).

Look up a named port with `spec.name` or `spec[:name]`. Dot names matching a
`KernelSpec` storage field (`graph`, `ports`, `port_order`, `have_names`,
`want_names`, or `call_signature`) return that field; a colliding port therefore
uses bracket lookup.
Planning and preparation accept either port names or `Value`s in their `have`
and `want` overrides.
"""
struct KernelSpec{S}
    graph::Graph
    ports::Dict{Symbol,Value}
    port_order::Vector{Symbol}
    have_names::Vector{Symbol}
    want_names::Vector{Symbol}
    call_signature::S
end

KernelSpec(graph::Graph, ports::Dict{Symbol,Value}, port_order::Vector{Symbol},
           have_names::Vector{Symbol}, want_names::Vector{Symbol}) =
    KernelSpec(graph, ports, port_order, have_names, want_names, nothing)

struct _KernelRequiredArgument end
const _KERNEL_REQUIRED_ARGUMENT = _KernelRequiredArgument()

struct _KernelCallSignature{P,K,M,D}
    defaults::D
end

# A pure object endpoint has a full graph boundary, but authored endpoint
# application spells only the method parameters.  Owner HAVE ports are filled
# transparently by same-name caller ports during graph construction.
struct _KernelEndpointCallSignature{Implicit,Explicit} end
_kernel_endpoint_call_signature(::Val{I}, ::Val{E}) where {I,E} =
    _KernelEndpointCallSignature{I,E}()

function _kernel_call_signature(::Val{P}, ::Val{K}, ::Val{M}, defaults::D) where {P,K,M,D}
    _KernelCallSignature{P,K,M,D}(defaults)
end

struct _KernelSignatureCallable{F,S}
    target::F
    signature::S
end

@inline function (callable::_KernelSignatureCallable)(args...; kwargs...)
    _kernel_signature_invoke(callable, args, NamedTuple(kwargs))
end

inputs(callable::_KernelSignatureCallable) = inputs(callable.target)
outputs(callable::_KernelSignatureCallable) = outputs(callable.target)
code_expr(callable::_KernelSignatureCallable) = code_expr(callable.target)

function Base.show(io::IO, callable::_KernelSignatureCallable)
    print(io, "KernelSignatureCallable(")
    show(io, callable.target)
    print(io, ")")
end

@generated function _kernel_signature_invoke(
    callable::_KernelSignatureCallable{F,S}, args::A,
    kwargs::NamedTuple{N,KT},
) where {F,S,A,N,KT}
    positional_names, keyword_names, default_mask = S.parameters[1:3]
    positional_count = length(positional_names)
    supplied_count = length(A.parameters)
    first_default = findfirst(identity, default_mask[1:positional_count])
    required_positionals = first_default === nothing ? positional_count : first_default - 1

    if supplied_count < required_positionals || supplied_count > positional_count
        return :(throw(MethodError(callable, args)))
    end
    unknown = Tuple(name for name in N if !(name in keyword_names))
    if !isempty(unknown)
        accepted = join(string.(keyword_names), ", ")
        message = isempty(accepted) ?
                  "kernel does not accept keyword arguments; got $(unknown)" :
                  "unknown kernel keyword $(unknown); accepted keywords: $accepted"
        return :(throw(ArgumentError($message)))
    end

    body = Expr(:block)
    resolved = Symbol[]
    defaults = :(getfield(getfield(callable, :signature), :defaults))
    for index in 1:positional_count
        value = gensym(positional_names[index])
        if index <= supplied_count
            push!(body.args, :(local $value = args[$index]))
        else
            provider = :(getfield($defaults, $index))
            push!(body.args, :(local $value = $provider($(resolved...))))
        end
        push!(resolved, value)
    end
    for (keyword_index, name) in enumerate(keyword_names)
        index = positional_count + keyword_index
        value = gensym(name)
        supplied_index = findfirst(==(name), N)
        if supplied_index !== nothing
            push!(body.args, :(local $value = getfield(kwargs, $supplied_index)))
        elseif default_mask[index]
            provider = :(getfield($defaults, $index))
            push!(body.args, :(local $value = $provider($(resolved...))))
        else
            push!(body.args, :(throw(UndefKeywordError($(QuoteNode(name))))))
        end
        push!(resolved, value)
    end
    target = :(getfield(callable, :target))
    push!(body.args, :(return $target($(resolved...))))
    body
end

_kernel_signature_callable(target, ::Nothing) = target
_kernel_signature_callable(target, ::_KernelEndpointCallSignature) = target
function _kernel_signature_callable(
    target, signature::_KernelCallSignature{P,K,M},
) where {P,K,M}
    isempty(K) && !any(M) && return target
    _KernelSignatureCallable(target, signature)
end

function Base.show(io::IO, spec::KernelSpec)
    have = join(string.(spec.have_names), ", ")
    want = join(string.(spec.want_names), ", ")
    print(io, "KernelSpec(", have, " -> ", want, "; ",
          length(spec.ports), " ports, ", length(spec.graph.recipes), " recipes)")
end

Base.keys(spec::KernelSpec) = Tuple(spec.port_order)
Base.haskey(spec::KernelSpec, name::Symbol) = haskey(spec.ports, name)

function Base.getproperty(spec::KernelSpec, name::Symbol)
    name in fieldnames(KernelSpec) && return getfield(spec, name)
    ports = getfield(spec, :ports)
    haskey(ports, name) && return ports[name]
    getfield(spec, name)
end

function Base.propertynames(spec::KernelSpec, private::Bool = false)
    names = (fieldnames(KernelSpec)..., spec.port_order...)
    Tuple(unique(names))
end

function Base.getindex(spec::KernelSpec, name::Symbol)
    haskey(spec.ports, name) && return spec.ports[name]
    available = join(string.(spec.port_order), ", ")
    throw(KeyError("kernel has no port :$name; available ports: $available"))
end

"The underlying low-level [`Graph`](@ref) for a declaratively authored kernel."
kernel_graph(spec::KernelSpec) = spec.graph

"Look up a named graph port. Equivalent to `spec[name]`."
port(spec::KernelSpec, name::Symbol) = spec[name]

function _kernel_declare!(graph::Graph, ports::Dict{Symbol,Value},
                          order::Vector{Symbol}, name::Symbol, T)
    T isa Type || throw(ArgumentError(
        "kernel port :$name must be declared with a Julia type, got $(repr(T))"))
    if haskey(ports, name)
        existing = ports[name]
        valtype(existing) == T || throw(ArgumentError(
            "kernel port :$name has type $(valtype(existing)); cannot redeclare it as $T"))
        return existing
    end
    value = value!(graph, name, T)
    ports[name] = value
    push!(order, name)
    value
end

function _kernel_push_unique!(names::Vector{Symbol}, name::Symbol)
    name in names || push!(names, name)
    names
end

function _kernel_add!(graph::Graph, ins, outs, op, cost, cse_key, effectful,
                      source = _NO_KERNEL_SOURCE)
    add!(graph; inputs = ins, outputs = outs, op = op,
         cost = cost, cse_key = cse_key, effectful = effectful, source = source)
end

# A definition-stable structural key for compiler-owned source operations.  It
# is deliberately unavailable to arbitrary Graph closures: only `@kernel` can
# mint `_KernelSourceOp{Token}`.  Reusing a captured child therefore reuses the
# key, while separately authored formulas have different Token parameters.
struct _KernelProvenanceKey{Token} end

_kernel_provenance_key(recipe::Recipe) =
    _kernel_provenance_key(recipe.op, recipe.cse_key, recipe.effectful)
_kernel_provenance_key(op, key, ::Bool) = key
function _kernel_provenance_key(op::_KernelSourceOp{Token}, key, effectful::Bool) where {Token}
    key === nothing && !effectful ? _KernelProvenanceKey{Token}() : key
end

_kernel_inline_signature_supported(::Any) = false
_kernel_inline_signature_supported(::Nothing) = true
_kernel_inline_signature_supported(::_KernelEndpointCallSignature) = true
function _kernel_inline_signature_supported(
    ::_KernelCallSignature{P,K,M},
) where {P,K,M}
    isempty(K)
end

function _kernel_inline_call_inputs(ins, source, signature = nothing)
    signature isa _KernelEndpointCallSignature && return Tuple(ins)
    source isa Expr && source.head === :call || return Tuple(ins)
    available = Dict(value.name => value for value in ins)
    args = source.args[2:end]
    all(arg -> arg isa Symbol && haskey(available, arg), args) || return Tuple(ins)
    Tuple(available[arg] for arg in args)
end

function _kernel_inline_alias!(graph::Graph, from::Value, to::Value, context)
    from_type = valtype(from)
    to_type = valtype(to)
    from_type == to_type || throw(ArgumentError(
        "nested kernel $context type mismatch: :$(from.name) has type $from_type, " *
        "but :$(to.name) has type $to_type; declare the caller boundary with the " *
        "exact nested boundary type"))
    source = canon_id(graph, from.id)
    target = canon_id(graph, to.id)
    source == target && return graph
    graph.aliases[source] = target
    graph.version += 1
    graph
end

"""
Splice a direct stateless `KernelSpec` call into its caller's graph.

The callee is cloned with fresh `Value` identities on every call, then its
default HAVE boundary is aliased to the call arguments and its default WANT
boundary to the assignment outputs. Consequently the outer planner sees every
callee recipe and can prune, fuse, CSE, visualize, or prepare it normally; no
runtime `KernelSpec` call remains.
"""
function _kernel_add!(graph::Graph, ins, outs, spec::KernelSpec, cost, cse_key,
                      effectful, source = _NO_KERNEL_SOURCE)
    cost == 1.0 && cse_key === nothing && effectful === false ||
        throw(ArgumentError(
            "nested KernelSpec calls do not accept call-site @recipe metadata; " *
            "put cost, cse_key, or effectful metadata on the nested kernel's recipes"))
    _kernel_inline_signature_supported(spec.call_signature) || throw(ArgumentError(
        "nested KernelSpec calls currently require a positional-only boundary; " *
        "prepare kernels with keyword inputs separately"))

    child_inputs = inputs(spec)
    child_outputs = outputs(spec)
    call_inputs = _kernel_inline_call_inputs(ins, source, spec.call_signature)
    length(call_inputs) == length(child_inputs) || throw(ArgumentError(
        "nested KernelSpec call has $(length(call_inputs)) argument(s), but its default HAVE " *
        "boundary has $(length(child_inputs)) port(s)"))
    length(outs) == length(child_outputs) || throw(ArgumentError(
        "nested KernelSpec call assigns $(length(outs)) output(s), but its default WANT " *
        "boundary has $(length(child_outputs)) port(s); destructure that boundary exactly"))

    child_graph = spec.graph
    cloned = Dict{Int,Value}()
    for id in sort!(collect(keys(child_graph.values)))
        value = child_graph.values[id]
        cloned[id] = value!(graph, value.name, valtype(value))
    end

    # Preserve all proven aliases before attaching the cloned boundary. Alias
    # chains remain valid if a later boundary attachment reparents their final
    # canonical representative.
    for id in sort!(collect(keys(child_graph.aliases)))
        _kernel_inline_alias!(
            graph,
            cloned[id],
            cloned[canon_id(child_graph, id)],
            "internal alias",
        )
    end
    for (actual, formal) in zip(call_inputs, child_inputs)
        _kernel_inline_alias!(graph, cloned[formal.id], actual, "input")
    end
    for (actual, formal) in zip(outs, child_outputs)
        _kernel_inline_alias!(graph, actual, cloned[formal.id], "output")
    end

    for recipe in child_graph.recipes
        add!(graph;
             inputs = Tuple(cloned[value.id] for value in recipe.inputs),
             outputs = Tuple(cloned[value.id] for value in recipe.outputs),
             op = recipe.op, cost = recipe.cost,
             cse_key = _kernel_provenance_key(recipe),
             effectful = recipe.effectful, source = recipe.source)
    end
    _reindex_producers!(graph)
end

# A SINGLE-DEFINITION bare-identity recipe `b = a` (RHS a bare port, not a call) makes `b` a
# CANONICAL ALIAS of `a` (RK 2026-08-27): both authored names resolve to ONE canonical Value —
# one physical slot, one shared validity/currentness — instead of a distinct Value joined by an
# opaque identity Recipe. Both authored names stay in `ports`/`port_order` for reporting.
#
# SOUNDNESS: it merges canonical CLASSES (`src=canon_id(from)`, `dst=canon_id(to)`) and is a
# NO-OP when they already coincide, so a reverse/transitive `a=b; b=a` cannot build a cycle or
# reparent incorrectly. It collapses ONLY a PROVEN same-declared-type identity; a typed
# conversion (`b::T = a::U`, T≠U) is uncertain, so the ordinary identity Recipe is kept instead.
# The caller hard-aliases only outputs with EXACTLY ONE authored definition (`b=a; b=c` are
# alternative producers, not a proof `a===c`).
function _kernel_alias!(graph::Graph, from::Value, to::Value, op, cost,
                        source = _NO_KERNEL_SOURCE)
    src = canon_id(graph, from.id)
    dst = canon_id(graph, to.id)
    src == dst && return graph                       # already one class (reverse/transitive)
    if valtype(from) == valtype(to)                  # proven same-type identity → collapse
        graph.aliases[src] = dst
        graph.version += 1                           # a real canonical mutation (like CSE/merge)
    else                                             # typed conversion → keep the ordinary recipe
        add!(graph; inputs = (to,), outputs = (from,), op = op,
             cost = cost, cse_key = nothing, effectful = false, source = source)
    end
    graph
end

# --- macro parsing ---------------------------------------------------------

_kernel_is_line(ex) = ex isa LineNumberNode

function _kernel_port_decl(ex; typed::Bool = false)
    if ex isa Symbol
        typed && throw(ArgumentError(
            "new kernel port :$ex needs an explicit type annotation, for example $ex::Float64"))
        return ex, nothing
    end
    if ex isa Expr && ex.head === :(::) && length(ex.args) == 2 &&
       ex.args[1] isa Symbol
        return ex.args[1], ex.args[2]
    end
    throw(ArgumentError(
        "kernel ports must be names or typed names such as x::Float64; got $(repr(ex))"))
end

function _kernel_bound_names!(names::Set{Symbol}, ex)
    ex isa Symbol && (push!(names, ex); return names)
    ex isa Expr || return names
    if ex.head === :(::) || ex.head === :kw || ex.head === :(=)
        _kernel_bound_names!(names, ex.args[1])
    elseif ex.head === :tuple || ex.head === :parameters
        for arg in ex.args
            _kernel_bound_names!(names, arg)
        end
    elseif ex.head === :(...)
        _kernel_bound_names!(names, ex.args[1])
    end
    names
end

function _kernel_assignment_names!(names::Set{Symbol}, lhs)
    if lhs isa Symbol
        push!(names, lhs)
    elseif lhs isa Expr && lhs.head in (:(::), :kw, :(...))
        _kernel_assignment_names!(names, lhs.args[1])
    elseif lhs isa Expr && lhs.head in (:tuple, :parameters)
        for arg in lhs.args
            _kernel_assignment_names!(names, arg)
        end
    elseif lhs isa Expr && lhs.head === :call && lhs.args[1] isa Symbol
        # Short-form local function definition: f(args...) = body.
        push!(names, lhs.args[1])
    elseif lhs isa Expr && lhs.head === :where
        _kernel_assignment_names!(names, lhs.args[1])
    end
    names
end

function _kernel_assignment_lhs_reads!(out::Vector{Symbol}, seen::Set{Symbol}, lhs,
                                       known::Set{Symbol}, assigned::Set{Symbol},
                                       hidden::Set{Symbol})
    lhs isa Symbol && return assigned
    lhs isa Expr || return assigned
    if lhs.head === :(::)
        _kernel_assignment_lhs_reads!(
            out, seen, lhs.args[1], known, assigned, hidden,
        )
        for annotation in lhs.args[2:end]
            _kernel_free_ports_flow!(
                out, seen, annotation, known, assigned, hidden,
            )
        end
    elseif lhs.head in (:tuple, :parameters)
        for target in lhs.args
            _kernel_assignment_lhs_reads!(
                out, seen, target, known, assigned, hidden,
            )
        end
    elseif lhs.head === :kw
        _kernel_assignment_lhs_reads!(
            out, seen, lhs.args[1], known, assigned, hidden,
        )
        _kernel_free_ports_flow!(
            out, seen, lhs.args[2], known, assigned, hidden,
        )
    elseif lhs.head === :(...)
        _kernel_assignment_lhs_reads!(
            out, seen, lhs.args[1], known, assigned, hidden,
        )
    else
        # Mutation targets such as `a[i]` and `obj.field` evaluate only their
        # receiver/index components; the ordinary expression walker already
        # treats property labels as inert.
        _kernel_free_ports_flow!(out, seen, lhs, known, assigned, hidden)
    end
    assigned
end

function _kernel_function_parameters!(names::Set{Symbol}, signature)
    if signature isa Expr && signature.head === :(::)
        _kernel_function_parameters!(names, signature.args[1])
    elseif signature isa Expr && signature.head === :where
        _kernel_function_parameters!(names, signature.args[1])
        for parameter in signature.args[2:end]
            target = parameter isa Expr && parameter.head in (:<:, :>:) ?
                     parameter.args[1] : parameter
            _kernel_bound_names!(names, target)
        end
    elseif signature isa Expr && signature.head === :call
        signature.args[1] isa Symbol && push!(names, signature.args[1])
        for arg in signature.args[2:end]
            _kernel_bound_names!(names, arg)
        end
    else
        _kernel_bound_names!(names, signature)
    end
    names
end

function _kernel_function_name!(names::Set{Symbol}, signature)
    if signature isa Expr && signature.head in (:(::), :where)
        _kernel_function_name!(names, signature.args[1])
    elseif signature isa Expr && signature.head === :call
        signature.args[1] isa Symbol && push!(names, signature.args[1])
    elseif signature isa Symbol
        push!(names, signature)
    end
    names
end

function _kernel_callable_name!(names::Set{Symbol}, signature)
    while signature isa Expr && signature.head in (:(::), :where)
        signature = signature.args[1]
    end
    if signature isa Expr && signature.head === :call &&
       signature.args[1] isa Symbol
        push!(names, signature.args[1])
    end
    names
end

function _kernel_where_names!(names::Set{Symbol}, signature)
    signature isa Expr || return names
    if signature.head === :where
        for parameter in signature.args[2:end]
            target = parameter isa Expr && parameter.head in (:<:, :>:) ?
                     parameter.args[1] : parameter
            _kernel_bound_names!(names, target)
        end
        _kernel_where_names!(names, signature.args[1])
    elseif signature.head === :(::)
        _kernel_where_names!(names, signature.args[1])
    end
    names
end

function _kernel_signature_annotations!(annotations::Vector{Any}, signature)
    signature isa Expr || return annotations
    if signature.head === :(::)
        _kernel_signature_annotations!(annotations, signature.args[1])
        append!(annotations, signature.args[2:end])
    elseif signature.head === :where
        _kernel_signature_annotations!(annotations, signature.args[1])
        for parameter in signature.args[2:end]
            if parameter isa Expr && parameter.head in (:<:, :>:)
                append!(annotations, parameter.args[2:end])
            end
        end
    elseif signature.head in (:call, :parameters, :tuple)
        start = signature.head === :call ? 2 : 1
        for argument in signature.args[start:end]
            _kernel_signature_annotations!(annotations, argument)
        end
    elseif signature.head === :kw || signature.head === :(...)
        _kernel_signature_annotations!(annotations, signature.args[1])
    end
    annotations
end

_kernel_signature_annotations(signature) =
    _kernel_signature_annotations!(Any[], signature)

function _kernel_function_arguments(signature)
    while signature isa Expr && signature.head in (:(::), :where)
        signature = signature.args[1]
    end
    signature isa Expr && signature.head === :call || return Any[]
    positional = Any[]
    keywords = Any[]
    for argument in signature.args[2:end]
        if argument isa Expr && argument.head === :parameters
            append!(keywords, argument.args)
        else
            push!(positional, argument)
        end
    end
    append!(positional, keywords)
end

_kernel_iterators(ex::Expr) = ex.head === :block ? ex.args : (ex,)

# Precollect only lexical declarations owned by the current hard scope. Plain
# assignments are handled in statement order below; nested loop/catch/function
# scopes must not turn a conditional write into an unconditional shadow.
function _kernel_explicit_locals!(names::Set{Symbol}, ex)
    ex isa Expr || return names
    ex.head === :quote && return names
    if ex.head === :function
        _kernel_function_name!(names, ex.args[1])
        return names
    elseif ex.head === :(=) && ex.args[1] isa Expr &&
           ex.args[1].head in (:call, :where)
        _kernel_function_name!(names, ex.args[1])
        return names
    elseif ex.head in (:local, :global)
        for declaration in ex.args
            target = declaration isa Expr && declaration.head === :(=) ?
                     declaration.args[1] : declaration
            _kernel_assignment_names!(names, target)
        end
        return names
    elseif ex.head in (:->, :let, :generator, :comprehension,
                       :for, :while, :try)
        return names
    end
    for arg in ex.args
        _kernel_explicit_locals!(names, arg)
    end
    names
end

function _kernel_port_read!(out::Vector{Symbol}, seen::Set{Symbol}, name::Symbol,
                            known::Set{Symbol}, assigned::Set{Symbol},
                            hidden::Set{Symbol})
    if name in known && !(name in assigned) && !(name in hidden) && !(name in seen)
        push!(out, name)
        push!(seen, name)
    end
    assigned
end

function _kernel_function_default_reads!(
    out::Vector{Symbol}, seen::Set{Symbol}, signature, known::Set{Symbol},
    assigned::Set{Symbol}, hidden::Set{Symbol},
)
    signature_hidden = copy(hidden)
    _kernel_function_name!(signature_hidden, signature)
    _kernel_where_names!(signature_hidden, signature)
    for argument in _kernel_function_arguments(signature)
        if argument isa Expr && argument.head === :kw
            # Earlier positional/keyword parameters are in scope, while the
            # parameter currently being defaulted is not yet bound.
            _kernel_free_ports_flow!(
                out, seen, argument.args[2], known,
                copy(assigned), signature_hidden,
            )
            parameter_names = Set{Symbol}()
            _kernel_bound_names!(parameter_names, argument.args[1])
            union!(signature_hidden, parameter_names)
        else
            _kernel_bound_names!(signature_hidden, argument)
        end
    end
    nothing
end

function _kernel_validate_function_signature!(
    signature, known::Set{Symbol}, assigned::Set{Symbol}, hidden::Set{Symbol},
)
    # Outer locals/definite writes cannot make a runtime value legal in a local
    # method signature. Only signature-owned lexical names are exempt.
    signature_hidden = Set{Symbol}()
    _kernel_callable_name!(signature_hidden, signature)
    _kernel_where_names!(signature_hidden, signature)
    for annotation in _kernel_signature_annotations(signature)
        reads = Symbol[]
        _kernel_free_ports_flow!(
            reads, Set{Symbol}(), annotation, known,
            Set{Symbol}(), signature_hidden,
        )
        isempty(reads) || throw(ArgumentError(
            "graph port :$(first(reads)) cannot appear in a nested local function " *
            "signature annotation; Julia method signatures cannot capture runtime " *
            "local types. Use a runtime isa/convert check or a module-qualified " *
            "static type instead"))
    end
    nothing
end

function _kernel_generator_iterator_flow!(
    out::Vector{Symbol}, seen::Set{Symbol}, iterator, known::Set{Symbol},
    assigned::Set{Symbol}, hidden::Set{Symbol},
)
    if iterator isa Expr && iterator.head in (:(=), :in)
        _kernel_free_ports_flow!(
            out, seen, iterator.args[2], known, assigned, hidden,
        )
        _kernel_assignment_lhs_reads!(
            out, seen, iterator.args[1], known, assigned, hidden,
        )
        iterator_names = Set{Symbol}()
        _kernel_bound_names!(iterator_names, iterator.args[1])
        union!(hidden, iterator_names)
        union!(assigned, iterator_names)
    elseif iterator isa Expr && iterator.head === :filter
        # Parser order is predicates first and the iterator source last, while
        # comma-form iterator sources all resolve in the incoming outer scope.
        # Only after every source/type has been evaluated do their binders hide
        # graph ports for the predicate and body.
        _kernel_generator_binding_group_flow!(
            out, seen, iterator.args[2:end], known, assigned, hidden,
        )
        for predicate in iterator.args[1:1]
            _kernel_free_ports_flow!(
                out, seen, predicate, known, assigned, hidden,
            )
        end
    elseif iterator isa Expr && iterator.head === :block
        for nested in iterator.args
            _kernel_generator_iterator_flow!(
                out, seen, nested, known, assigned, hidden,
            )
        end
    else
        _kernel_free_ports_flow!(
            out, seen, iterator, known, assigned, hidden,
        )
    end
    nothing
end

function _kernel_generator_binding_group_flow!(
    out::Vector{Symbol}, seen::Set{Symbol}, iterators, known::Set{Symbol},
    assigned::Set{Symbol}, hidden::Set{Symbol},
)
    incoming_hidden = copy(hidden)
    iterator_names = Set{Symbol}()
    for iterator in iterators
        if iterator isa Expr && iterator.head in (:(=), :in)
            _kernel_free_ports_flow!(
                out, seen, iterator.args[2], known, assigned, incoming_hidden,
            )
            _kernel_assignment_lhs_reads!(
                out, seen, iterator.args[1], known, assigned, incoming_hidden,
            )
            _kernel_bound_names!(iterator_names, iterator.args[1])
        else
            _kernel_free_ports_flow!(
                out, seen, iterator, known, assigned, incoming_hidden,
            )
        end
    end
    union!(hidden, iterator_names)
    union!(assigned, iterator_names)
    nothing
end

function _kernel_generator_flow!(
    out::Vector{Symbol}, seen::Set{Symbol}, ex::Expr, known::Set{Symbol},
    assigned::Set{Symbol}, hidden::Set{Symbol},
)
    if ex.head === :comprehension || ex.head === :flatten
        for nested in ex.args
            nested isa Expr && _kernel_generator_flow!(
                out, seen, nested, known, copy(assigned), copy(hidden),
            )
        end
        return nothing
    end
    ex.head === :generator || return _kernel_free_ports_flow!(
        out, seen, ex, known, assigned, hidden,
    )

    generator_hidden = copy(hidden)
    generator_assigned = copy(assigned)
    iterators = ex.args[2:end]
    if length(iterators) > 1
        _kernel_generator_binding_group_flow!(
            out, seen, iterators, known, generator_assigned, generator_hidden,
        )
    else
        for iterator in iterators
            _kernel_generator_iterator_flow!(
                out, seen, iterator, known, generator_assigned, generator_hidden,
            )
        end
    end
    body = ex.args[1]
    if body isa Expr && body.head in (:generator, :flatten)
        _kernel_generator_flow!(
            out, seen, body, known, generator_assigned, generator_hidden,
        )
    else
        _kernel_explicit_locals!(generator_hidden, body)
        _kernel_free_ports_flow!(
            out, seen, body, known, generator_assigned, generator_hidden,
        )
    end
    nothing
end

function _kernel_free_ports_flow!(out::Vector{Symbol}, seen::Set{Symbol}, ex,
                                  known::Set{Symbol}, assigned::Set{Symbol},
                                  hidden::Set{Symbol})
    ex isa Symbol && return _kernel_port_read!(
        out, seen, ex, known, assigned, hidden,
    )
    ex isa Expr || return assigned
    ex.head === :quote && return assigned

    if ex.head === :block
        for statement in ex.args
            _kernel_free_ports_flow!(out, seen, statement, known, assigned, hidden)
        end
    elseif ex.head === :tuple
        # In tuple expression context `name = value` / `name = value` under a
        # `:parameters` node is a named-tuple field, not a local assignment.
        for item in ex.args
            fields = item isa Expr && item.head === :parameters ? item.args : (item,)
            for field in fields
                if field isa Expr && field.head in (:(=), :kw)
                    _kernel_free_ports_flow!(
                        out, seen, field.args[2], known, assigned, hidden,
                    )
                else
                    _kernel_free_ports_flow!(
                        out, seen, field, known, assigned, hidden,
                    )
                end
            end
        end
    elseif ex.head === :kw
        _kernel_free_ports_flow!(out, seen, ex.args[2], known, assigned, hidden)
    elseif ex.head === :.
        _kernel_free_ports_flow!(out, seen, ex.args[1], known, assigned, hidden)
        if length(ex.args) > 1 && !(ex.args[2] isa QuoteNode) &&
           !(ex.args[2] isa Symbol)
            _kernel_free_ports_flow!(out, seen, ex.args[2], known, assigned, hidden)
        end
    elseif ex.head === :(=) && ex.args[1] isa Expr &&
           ex.args[1].head in (:call, :where)
        _kernel_validate_function_signature!(
            ex.args[1], known, assigned, hidden,
        )
        _kernel_function_default_reads!(
            out, seen, ex.args[1], known, assigned, hidden,
        )
        function_names = Set{Symbol}()
        _kernel_function_parameters!(function_names, ex.args[1])
        body_hidden = union(copy(hidden), function_names)
        _kernel_explicit_locals!(body_hidden, ex.args[2])
        _kernel_free_ports_flow!(
            out, seen, ex.args[2], known, Set{Symbol}(), body_hidden,
        )
        union!(assigned, function_names)
    elseif ex.head === :(=)
        # Pure binder leaves are not reads. Typed binders evaluate their type,
        # while indexed/property targets evaluate receiver/index components.
        # Julia evaluates the RHS first, so its definite writes are visible to
        # the later target evaluation.
        _kernel_free_ports_flow!(out, seen, ex.args[2], known, assigned, hidden)
        _kernel_assignment_lhs_reads!(
            out, seen, ex.args[1], known, assigned, hidden,
        )
        written = Set{Symbol}()
        _kernel_assignment_names!(written, ex.args[1])
        union!(assigned, written)
    elseif ex.head === :local
        for declaration in ex.args
            if declaration isa Expr && declaration.head === :(=)
                _kernel_free_ports_flow!(
                    out, seen, declaration.args[2], known, assigned, hidden,
                )
                _kernel_assignment_lhs_reads!(
                    out, seen, declaration.args[1], known, assigned, hidden,
                )
                _kernel_assignment_names!(assigned, declaration.args[1])
            else
                _kernel_assignment_lhs_reads!(
                    out, seen, declaration, known, assigned, hidden,
                )
                _kernel_assignment_names!(assigned, declaration)
            end
        end
    elseif ex.head === :global
        for declaration in ex.args
            if declaration isa Expr && declaration.head === :(=)
                _kernel_free_ports_flow!(
                    out, seen, declaration.args[2], known, assigned, hidden,
                )
            end
        end
    elseif ex.head === :if
        _kernel_free_ports_flow!(out, seen, ex.args[1], known, assigned, hidden)
        incoming = copy(assigned)
        then_assigned = copy(incoming)
        _kernel_free_ports_flow!(
            out, seen, ex.args[2], known, then_assigned, hidden,
        )
        else_assigned = copy(incoming)
        if length(ex.args) >= 3 && ex.args[3] !== nothing
            _kernel_free_ports_flow!(
                out, seen, ex.args[3], known, else_assigned, hidden,
            )
        end
        empty!(assigned)
        union!(assigned, intersect(then_assigned, else_assigned))
    elseif ex.head in (:&&, :||)
        _kernel_free_ports_flow!(out, seen, ex.args[1], known, assigned, hidden)
        branch_assigned = copy(assigned)
        _kernel_free_ports_flow!(
            out, seen, ex.args[2], known, branch_assigned, hidden,
        )
    elseif ex.head === :for
        loop_hidden = copy(hidden)
        for iterator in _kernel_iterators(ex.args[1])
            if iterator isa Expr && iterator.head in (:(=), :in)
                _kernel_free_ports_flow!(
                    out, seen, iterator.args[2], known, assigned, loop_hidden,
                )
                _kernel_assignment_lhs_reads!(
                    out, seen, iterator.args[1], known, assigned, loop_hidden,
                )
                iterator_names = Set{Symbol}()
                _kernel_bound_names!(iterator_names, iterator.args[1])
                union!(loop_hidden, iterator_names)
            else
                _kernel_free_ports_flow!(
                    out, seen, iterator, known, assigned, loop_hidden,
                )
            end
        end
        body_hidden = copy(loop_hidden)
        _kernel_explicit_locals!(body_hidden, ex.args[2])
        _kernel_free_ports_flow!(
            out, seen, ex.args[2], known, copy(assigned), body_hidden,
        )
    elseif ex.head === :while
        _kernel_free_ports_flow!(out, seen, ex.args[1], known, assigned, hidden)
        body_hidden = copy(hidden)
        _kernel_explicit_locals!(body_hidden, ex.args[2])
        _kernel_free_ports_flow!(
            out, seen, ex.args[2], known, copy(assigned), body_hidden,
        )
    elseif ex.head === :generator || ex.head === :comprehension
        _kernel_generator_flow!(
            out, seen, ex, known, assigned, hidden,
        )
    elseif ex.head === :let
        let_hidden = copy(hidden)
        let_assigned = copy(assigned)
        for binding in ex.args[1:end-1]
            if binding isa Expr && binding.head in (:(=), :kw)
                _kernel_free_ports_flow!(
                    out, seen, binding.args[2], known, let_assigned, let_hidden,
                )
                _kernel_assignment_lhs_reads!(
                    out, seen, binding.args[1], known, let_assigned, let_hidden,
                )
                binding_names = Set{Symbol}()
                _kernel_bound_names!(binding_names, binding.args[1])
                union!(let_hidden, binding_names)
                union!(let_assigned, binding_names)
            else
                _kernel_assignment_lhs_reads!(
                    out, seen, binding, known, let_assigned, let_hidden,
                )
                _kernel_bound_names!(let_hidden, binding)
                _kernel_bound_names!(let_assigned, binding)
            end
        end
        _kernel_explicit_locals!(let_hidden, ex.args[end])
        _kernel_free_ports_flow!(
            out, seen, ex.args[end], known, let_assigned, let_hidden,
        )
    elseif ex.head === :->
        _kernel_validate_function_signature!(
            ex.args[1], known, assigned, hidden,
        )
        function_hidden = copy(hidden)
        _kernel_bound_names!(function_hidden, ex.args[1])
        _kernel_explicit_locals!(function_hidden, ex.args[2])
        _kernel_free_ports_flow!(
            out, seen, ex.args[2], known, Set{Symbol}(), function_hidden,
        )
    elseif ex.head === :function
        _kernel_validate_function_signature!(
            ex.args[1], known, assigned, hidden,
        )
        _kernel_function_default_reads!(
            out, seen, ex.args[1], known, assigned, hidden,
        )
        function_hidden = copy(hidden)
        function_names = Set{Symbol}()
        _kernel_function_parameters!(function_names, ex.args[1])
        union!(function_hidden, function_names)
        _kernel_explicit_locals!(function_hidden, ex.args[2])
        _kernel_free_ports_flow!(
            out, seen, ex.args[2], known, Set{Symbol}(), function_hidden,
        )
        union!(assigned, function_names)
    elseif ex.head === :try
        incoming = copy(assigned)
        try_hidden = copy(hidden)
        _kernel_explicit_locals!(try_hidden, ex.args[1])
        _kernel_free_ports_flow!(
            out, seen, ex.args[1], known, copy(incoming), try_hidden,
        )
        if length(ex.args) >= 3 && ex.args[3] !== false
            catch_hidden = copy(hidden)
            ex.args[2] !== false && _kernel_bound_names!(catch_hidden, ex.args[2])
            _kernel_explicit_locals!(catch_hidden, ex.args[3])
            _kernel_free_ports_flow!(
                out, seen, ex.args[3], known, copy(incoming), catch_hidden,
            )
        end
        if length(ex.args) >= 4 && ex.args[4] !== false
            finally_hidden = copy(hidden)
            _kernel_explicit_locals!(finally_hidden, ex.args[4])
            _kernel_free_ports_flow!(
                out, seen, ex.args[4], known, copy(incoming), finally_hidden,
            )
        end
    else
        for arg in ex.args
            _kernel_free_ports_flow!(out, seen, arg, known, assigned, hidden)
        end
    end
    assigned
end

function _kernel_free_ports(ex, known::Set{Symbol})
    out = Symbol[]
    hidden = Set{Symbol}()
    _kernel_explicit_locals!(hidden, ex)
    _kernel_free_ports_flow!(
        out, Set{Symbol}(), ex, known, Set{Symbol}(), hidden,
    )
    out
end

function _kernel_hygienic_catches(ex, known::Set{Symbol})
    ex isa Expr || return ex
    ex.head === :quote && return ex
    if ex.head === :try
        args = Any[_kernel_hygienic_catches(arg, known) for arg in ex.args]
        if length(args) >= 3 && args[2] isa Symbol && args[2] in known &&
           args[3] !== false
            original = args[2]
            fresh = gensym(original)
            args[2] = fresh
            args[3] = Expr(:let, Expr(:(=), original, fresh), args[3])
        end
        return Expr(:try, args...)
    end
    Expr(ex.head, (_kernel_hygienic_catches(arg, known) for arg in ex.args)...)
end

function _kernel_recipe(ex)
    if !(ex isa Expr && ex.head === :macrocall && length(ex.args) >= 4 &&
         ex.args[1] === Symbol("@recipe"))
        return ex, Dict{Symbol,Any}()
    end
    args = Any[a for a in ex.args[3:end] if !_kernel_is_line(a)]
    assignment = pop!(args)
    if length(args) == 1 && args[1] isa Expr && args[1].head === :tuple
        args = Any[args[1].args...]
    end
    metadata = Dict{Symbol,Any}()
    for arg in args
        arg isa Expr && arg.head === :(=) && arg.args[1] isa Symbol ||
            throw(ArgumentError(
                "@recipe metadata must use name=value entries before the assignment"))
        name = arg.args[1]
        name in (:cost, :cse_key, :effectful) || throw(ArgumentError(
            "unknown @recipe option :$name; expected cost, cse_key, or effectful"))
        haskey(metadata, name) && throw(ArgumentError("duplicate @recipe option :$name"))
        metadata[name] = arg.args[2]
    end
    assignment, metadata
end

function _kernel_return_names(ex)
    items = ex isa Expr && ex.head === :tuple ? ex.args : Any[ex]
    names = Symbol[]
    for item in items
        item isa Symbol || throw(ArgumentError(
            "@kernel return values must be port names, got $(repr(item))"))
        push!(names, item)
    end
    names
end

const _KERNEL_RETURN_PORT = Symbol("__return__")

# `return expr` is graph authoring, not an eager Julia return.  Normalize the
# expression to one ordinary recipe feeding a stable distinguished port, then
# let the existing named-boundary machinery do the rest.  A named/tuple return
# remains byte-for-byte compatible with the established grammar.
function _kernel_normalize_return_expressions(statements)
    normalized = Any[]
    for raw in statements
        if raw isa Expr && raw.head === :return && length(raw.args) == 1
            returned = only(raw.args)
            named = returned isa Symbol ||
                    (returned isa Expr && returned.head === :tuple &&
                     all(item -> item isa Symbol, returned.args))
            if !named
                push!(normalized, Expr(:(=), _KERNEL_RETURN_PORT, returned))
                push!(normalized, Expr(:return, _KERNEL_RETURN_PORT))
                continue
            end
        end
        push!(normalized, raw)
    end
    normalized
end

function _kernel_constructed_endpoint(ex, mod, locals::Set{Symbol},
                                      nested_specs::Dict{Symbol,Any},
                                      local_types::Dict{Symbol,Any};
                                      context = "inside @kernel",
                                      materialized = Tuple{Symbol,Any,Any}[])
    ex isa Expr || return ex, nothing
    ex.head in (:quote, :inert) && return ex, nothing

    # `object(owner_args...).endpoint(endpoint_args...)` is source-level graph
    # application.  Resolve the object and endpoint while the macro expands,
    # retarget its implicit owner signature to the caller ports, and let the
    # established nested-KernelSpec splicer clone the complete endpoint graph.
    if ex.head === :call && !isempty(ex.args)
        callee = ex.args[1]
        if callee isa Expr && callee.head === :(.) && length(callee.args) == 2 &&
           callee.args[1] isa Expr && callee.args[1].head === :call &&
           callee.args[2] isa QuoteNode
            constructor = callee.args[1]
            object = _kernel_resolve_binding(mod, constructor.args[1])
            endpoint_name = callee.args[2].value
            if object isa Union{_StatefulKernelSkeleton,KernelObjectSpec} &&
               endpoint_name in kernel_endpoint_names(object)
                endpoint_actuals = ex.args[2:end]
                default_endpoint = extract(object; want = endpoint_name)
                signature = default_endpoint.call_signature
                signature isa _KernelEndpointCallSignature || throw(ArgumentError(
                    "constructed object endpoint :$endpoint_name has no transparent call signature"))
                default_implicit, explicit = typeof(signature).parameters

                constructor_args = constructor.args[2:end]
                parameter_nodes = Any[arg for arg in constructor_args
                                      if arg isa Expr && arg.head === :parameters]
                length(parameter_nodes) <= 1 || throw(ArgumentError(
                    "kernel object constructor for :$endpoint_name has multiple keyword lists"))
                positional = Any[arg for arg in constructor_args
                                 if !(arg isa Expr && arg.head === :parameters)]

                owner_formals = Symbol[]
                owner_actuals = Any[]
                endpoint = default_endpoint
                if isempty(parameter_nodes)
                    append!(owner_formals, default_implicit)
                    append!(owner_actuals, positional)
                else
                    isempty(positional) || throw(ArgumentError(
                        "kernel object constructor for :$endpoint_name cannot mix positional " *
                        "and named owner bindings"))
                    for binding in only(parameter_nodes).args
                        name, actual = if binding isa Symbol
                            (binding, binding)
                        elseif binding isa Expr && binding.head in (:kw, :(=)) &&
                               length(binding.args) == 2 && binding.args[1] isa Symbol
                            (binding.args[1], binding.args[2])
                        else
                            throw(ArgumentError(
                                "kernel object constructor for :$endpoint_name requires named " *
                                "bindings such as `location = caller_port`"))
                        end
                        name in owner_formals && throw(ArgumentError(
                            "duplicate kernel object binding :$name for endpoint :$endpoint_name"))
                        name in explicit && throw(ArgumentError(
                            "kernel object binding :$name duplicates an explicit " *
                            "endpoint argument of :$endpoint_name"))
                        push!(owner_formals, name)
                        push!(owner_actuals, actual)
                    end
                    full = _kernel_object_full_spec(object)
                    available = Tuple(keys(full))
                    for name in owner_formals
                        haskey(full, name) || throw(ArgumentError(
                            "unknown kernel object binding :$name for endpoint :$endpoint_name; " *
                            "available graph ports: $(join(string.(available), ", "))"))
                    end
                    endpoint = extract(object;
                        have = Tuple((owner_formals..., explicit...)),
                        want = endpoint_name)
                end

                length(owner_actuals) == length(owner_formals) || throw(ArgumentError(
                    "kernel object constructor for :$endpoint_name expects $(length(owner_formals)) " *
                    "owner argument(s), got $(length(owner_actuals))"))
                length(endpoint_actuals) == length(explicit) || throw(ArgumentError(
                    "endpoint :$endpoint_name expects $(length(explicit)) argument(s), " *
                    "got $(length(endpoint_actuals))"))
                all(arg -> arg isa Symbol && arg in locals, endpoint_actuals) ||
                    throw(ArgumentError(
                        "constructed endpoint arguments $context must be named caller ports"))

                # Owner bindings are graph inputs, but their source spelling need not
                # already be a named caller port. Materialize literals and computed
                # expressions as hygienic caller recipes, typed by the selected child
                # boundary, then alias those generated ports into the cloned endpoint.
                # This preserves ordinary Julia-like construction such as
                # `normal(0.0, 5.0).logpdf(x)` without hiding runtime object calls.
                endpoint_inputs = inputs(endpoint)
                owner_ports = Symbol[]
                for (index, (formal, actual)) in
                        enumerate(zip(owner_formals, owner_actuals))
                    if actual isa Symbol && actual in locals
                        push!(owner_ports, actual)
                        continue
                    elseif actual isa Symbol
                        throw(ArgumentError(
                            "kernel object binding :$formal $context refers to undeclared " *
                            "caller port :$actual"))
                    end
                    rewritten_actual, _ = _kernel_constructed_endpoint(
                        actual, mod, locals, nested_specs, local_types;
                        context = context, materialized = materialized)
                    generated_port = gensym(Symbol(formal, :_binding))
                    generated_type = valtype(endpoint_inputs[index])
                    push!(materialized,
                          (generated_port, generated_type, rewritten_actual))
                    push!(locals, generated_port)
                    push!(owner_ports, generated_port)
                end

                actuals = Symbol[owner_ports...; endpoint_actuals...]
                for (actual, formal) in zip(actuals, inputs(endpoint))
                    T = valtype(formal)
                    if haskey(local_types, actual) && local_types[actual] != T
                        throw(ArgumentError(
                            "plate do-block argument :$actual is used with incompatible " *
                            "endpoint types $(local_types[actual]) and $T"))
                    end
                    local_types[actual] = T
                end

                generated = gensym(Symbol(endpoint_name, :_endpoint))
                retargeted = _kernel_endpoint_call_signature(
                    Val(Tuple(owner_ports)), Val(explicit))
                nested_specs[generated] = KernelSpec(
                    endpoint.graph, endpoint.ports, endpoint.port_order,
                    endpoint.have_names, endpoint.want_names, retargeted)
                return Expr(:call, generated, endpoint_actuals...),
                       valtype(only(outputs(endpoint)))
            end
        end
    end

    rewritten = Any[]
    child_types = Any[]
    for arg in ex.args
        child, child_type = _kernel_constructed_endpoint(
            arg, mod, locals, nested_specs, local_types;
            context = context, materialized = materialized)
        push!(rewritten, child)
        push!(child_types, child_type)
    end
    inferred = if ex.head === :block
        index = findlast(i -> !_kernel_is_line(ex.args[i]), eachindex(ex.args))
        index === nothing ? nothing : child_types[index]
    else
        nothing
    end
    Expr(ex.head, rewritten...), inferred
end

function _kernel_authored_plate_expr(rhs, mod)
    rhs isa Expr && rhs.head === :do && length(rhs.args) == 2 || return nothing
    call, lambda = rhs.args
    call isa Expr && call.head === :call && !isempty(call.args) || return nothing
    _kernel_resolve_binding(mod, call.args[1]) === plate || return nothing
    lambda isa Expr && lambda.head === :(->) && length(lambda.args) == 2 ||
        throw(ArgumentError("plate do-block requires an ordinary argument list and body"))
    formals_expr, scalar_body = lambda.args
    formals = formals_expr isa Symbol ? Symbol[formals_expr] :
              formals_expr isa Expr && formals_expr.head === :tuple &&
              all(arg -> arg isa Symbol, formals_expr.args) ?
              Symbol[formals_expr.args...] : throw(ArgumentError(
                  "plate do-block arguments must be bare names"))
    authored_arguments = call.args[2:end]
    length(authored_arguments) == length(formals) || throw(ArgumentError(
        "plate received $(length(authored_arguments)) argument(s), but its do-block has " *
        "$(length(formals)) parameter(s)"))
    arguments = Symbol[]
    atomic = Int[]
    materialized_arguments = Tuple{Symbol,Any}[]
    for (index, argument) in enumerate(authored_arguments)
        if argument isa Symbol
            push!(arguments, argument)
        elseif argument isa Expr && argument.head === :call &&
               length(argument.args) == 2 &&
               _kernel_resolve_binding(mod, argument.args[1]) === Ref
            value = argument.args[2]
            name = if value isa Symbol
                value
            else
                generated = gensym(:plate_argument)
                push!(materialized_arguments, (generated, value))
                generated
            end
            push!(arguments, name)
            push!(atomic, index)
        else
            generated = gensym(:plate_argument)
            push!(materialized_arguments, (generated, argument))
            push!(arguments, generated)
        end
    end
    length(unique(formals)) == length(formals) || throw(ArgumentError(
        "plate do-block argument names must be unique"))

    nested_specs = Dict{Symbol,Any}()
    local_types = Dict{Symbol,Any}()
    materialized = Tuple{Symbol,Any,Any}[]
    rewritten, inferred = _kernel_constructed_endpoint(
        scalar_body, mod, Set(formals), nested_specs, local_types;
        context = "inside plate", materialized = materialized)
    signature = Tuple{Symbol,Any}[
        (name, get(local_types, name, GlobalRef(Core, :Any))) for name in formals]
    scalar_graph_body = _kernel_expression_result_body(
        :__plate_value__, signature, rewritten, true)
    if !isempty(materialized)
        lifted = Any[
            Expr(:(=), Expr(:(::), name, T), rhs)
            for (name, T, rhs) in materialized
        ]
        scalar_graph_body = if scalar_graph_body isa Expr &&
                               scalar_graph_body.head === :block
            Expr(:block, lifted..., scalar_graph_body.args...)
        else
            Expr(:block, lifted..., scalar_graph_body)
        end
    end
    if inferred !== nothing && scalar_graph_body isa Expr &&
       scalar_graph_body.head === :block
        assignment_index = findfirst(scalar_graph_body.args) do statement
            statement isa Expr && statement.head === :(=) &&
                statement.args[1] === :__plate_value__
        end
        assignment_index === nothing ||
            (scalar_graph_body.args[assignment_index].args[1] =
                Expr(:(::), :__plate_value__, inferred))
    end
    scalar_spec = _kernel_expand(
        scalar_graph_body, signature, nothing, mod; nested_specs = nested_specs)
    operation = Expr(:call, GlobalRef(@__MODULE__, :_kernel_authored_plate),
                     scalar_spec,
                     Expr(:call, GlobalRef(Base, :Val), QuoteNode(Tuple(atomic))))
    (; arguments, operation, inferred, atomic = Tuple(atomic),
       materialized_arguments)
end

function _kernel_authored_plate(spec::KernelSpec, ::Val{A}) where {A}
    kernel = prepare(spec)
    _AuthoredPlateOp{typeof(kernel),A}(kernel)
end

# A dotted operator such as `.*` is broadcast *syntax*, not a bound function:
# `a .* b` lowers to `broadcast(*, a, b)` and there is no callable named `.*`.
# Splicing the bare symbol as a recipe callee errors at construction with
# `UndefVarError: .* not defined`, so such recipes must take the closure path.
function _is_broadcast_operator(sym::Symbol)
    s = String(sym)
    length(s) >= 2 && s[1] === '.' && Base.isoperator(Symbol(s[2:end]))
end

# Cat-family callees rewritten in the tensorized body only.  The wrappers
# (core.jl) let a tracing backend promote untraced constant-array operands to
# traced constants before concatenating; a callee shadowed by a graph port
# keeps its port meaning and is never rewritten.
_tensorized_callee_replacement(callee::Symbol) =
    callee === :vcat ? :_tensorized_vcat :
    callee === :hcat ? :_tensorized_hcat :
    callee === :cat ? :_tensorized_cat : nothing
_tensorized_callee_replacement(callee) = nothing

function _kernel_tensorized_rhs(ex, known::Set{Symbol} = Set{Symbol}())
    ex isa Expr || return ex
    ex.head in (:quote, :inert) && return ex
    if ex.head === :if && length(ex.args) == 3
        # `broadcast(ifelse, ...)` is the common scalar/tensor select.  It is
        # still scalar for scalar branches, while a traced scalar predicate is
        # broadcast across array branches (Reactant deliberately has no
        # `ifelse(::TracedBool, ::TracedArray, ::TracedArray)` method).
        return Expr(:call, GlobalRef(Base, :broadcast),
                    GlobalRef(Base, :ifelse),
                    (_kernel_tensorized_rhs(arg, known) for arg in ex.args)...)
    elseif ex.head === :&& && length(ex.args) == 2
        return Expr(:call, GlobalRef(Base, :broadcast),
                    GlobalRef(Base, :ifelse),
                    _kernel_tensorized_rhs(ex.args[1], known),
                    _kernel_tensorized_rhs(ex.args[2], known), false)
    elseif ex.head === :|| && length(ex.args) == 2
        return Expr(:call, GlobalRef(Base, :broadcast),
                    GlobalRef(Base, :ifelse),
                    _kernel_tensorized_rhs(ex.args[1], known), true,
                    _kernel_tensorized_rhs(ex.args[2], known))
    elseif ex.head === :call && !isempty(ex.args) &&
           ex.args[1] isa Symbol && !(ex.args[1] in known)
        replacement = _tensorized_callee_replacement(ex.args[1])
        replacement === nothing || return Expr(:call,
            GlobalRef(@__MODULE__, replacement),
            (_kernel_tensorized_rhs(arg, known) for arg in ex.args[2:end])...)
    elseif ex.head in (:vcat, :hcat) &&
           !any(arg -> arg isa Expr && arg.head === :row, ex.args)
        # `[a; b]` / `[a b]` concatenation syntax; `:row`-bearing forms are
        # `hvcat` semantics and keep their native lowering.
        return Expr(:call,
            GlobalRef(@__MODULE__, ex.head === :vcat ?
                :_tensorized_vcat : :_tensorized_hcat),
            (_kernel_tensorized_rhs(arg, known) for arg in ex.args)...)
    end
    Expr(ex.head, (_kernel_tensorized_rhs(arg, known) for arg in ex.args)...)
end

function _kernel_operation(rhs, deps::Vector{Symbol}, known::Set{Symbol};
                           tensorize::Bool = true,
                           mod::Union{Module,Nothing} = nothing,
                           nested_specs = Dict{Symbol,Any}())
    form = :fused
    if rhs isa Expr && rhs.head === :call && !isempty(rhs.args)
        callee = rhs.args[1]
        args = rhs.args[2:end]
        callee_is_port = callee isa Symbol && callee in known
        generated_spec = callee isa Symbol ? get(nested_specs, callee, nothing) : nothing
        callee_is_spec = generated_spec !== nothing ||
                         (mod isa Module &&
                          _kernel_resolve_binding(mod, callee) isa KernelSpec)
        if callee_is_spec && !callee_is_port
            all(arg -> arg isa Symbol && arg in known, args) || throw(ArgumentError(
                "nested KernelSpec call arguments must be declared graph ports; " *
                "assign an expression to a port before passing it to the nested kernel"))
            return generated_spec === nothing ? callee : generated_spec
        end
        if callee isa Symbol && !callee_is_port && !_is_broadcast_operator(callee) &&
           length(args) == length(deps) &&
           all(i -> args[i] === deps[i], eachindex(args))
            return callee                                   # BARE exact identity — stays raw (validated)
        end
        # a call THROUGH A PORT — `callable(args…)` where the callee itself is a port (RK 07:24): the
        # first dep is the callable source, the rest are ordered args. Tagged `:portcall` so a prepared
        # handle self-derives the destination contract from source shape + typed slots.
        callee_is_port && !isempty(deps) && deps[1] === callee && (form = :portcall)
    end
    # A recipe operation synthesized from captured @kernel source, wrapped as COMPILER-OWNED provenance
    # with a definition-unique gensym token (RK 07:21) + its Form, so a prepared handle distinguishes it
    # from a manually-inserted raw (opaque) closure without IR inspection. Call forwards inline.
    deftoken = gensym(:rk_srcop)
    Expr(:call, GlobalRef(@__MODULE__, :_KernelSourceOp),
         Expr(:call, GlobalRef(Base, :Val), QuoteNode(deftoken)),
         Expr(:call, GlobalRef(Base, :Val), QuoteNode(form)),
         Expr(:->, Expr(:tuple, deps...), rhs),
         Expr(:->, Expr(:tuple, deps...),
              tensorize ? _kernel_tensorized_rhs(rhs, known) : rhs))
end

function _kernel_nested_endpoint_deps(rhs, deps::Vector{Symbol}, known::Set{Symbol},
                                      mod, nested_specs)
    rhs isa Expr && rhs.head === :call && !isempty(rhs.args) || return deps
    callee = rhs.args[1]
    generated = callee isa Symbol ? get(nested_specs, callee, nothing) : nothing
    spec = generated isa KernelSpec ? generated :
           (mod isa Module ? _kernel_resolve_binding(mod, callee) : nothing)
    spec isa KernelSpec || return deps
    signature = spec.call_signature
    signature isa _KernelEndpointCallSignature || return deps
    implicit, explicit = typeof(signature).parameters
    args = rhs.args[2:end]
    length(args) == length(explicit) || throw(ArgumentError(
        "endpoint application expects $(length(explicit)) explicit argument(s) " *
        "($(join(string.(explicit), ", "))), got $(length(args))"))
    all(arg -> arg isa Symbol && arg in known, args) || throw(ArgumentError(
        "endpoint application arguments must be named caller ports"))
    for name in implicit
        name in known || throw(ArgumentError(
            "endpoint application needs implicit owner port :$name in the caller graph"))
    end
    Symbol[implicit...; args...]
end

# `@node(expr)` recipe-node promotion, used by `_kernel_expand`. Kept HERE (ahead of
# `_kernel_expand`, not in kernel_stateful.jl) because authoring.jl loads first and a
# `@doc`-embedded `@kernel` expands at load time — a forward ref would be undefined.

# Is a macrocall head NAMED `@node` (bare or module-qualified)? A SYNTACTIC CANDIDATE
# only — identity (the genuine RK `@node` binding vs a foreign `Evil.@node`) is
# confirmed via `_kernel_resolve_binding`.
_kernel_is_node_macro(m) =
    m === Symbol("@node") ||
    (m isa Expr && m.head === :(.) && length(m.args) >= 2 &&
        m.args[2] == QuoteNode(Symbol("@node")))

# Any `@node(...)` CANDIDATE (by name) anywhere in `x`?
_kernel_has_node_marker(x) =
    x isa Expr && (
        (x.head === :macrocall && !isempty(x.args) && _kernel_is_node_macro(x.args[1])) ||
        any(_kernel_has_node_marker, x.args))

# Resolve a callee/macro-head AST (bare `name` or `Mod.name`) to its BINDING VALUE in
# `mod`, or `nothing`. Reads bindings only (no call eval) — inside the compiler boundary.
function _kernel_resolve_binding(mod::Module, callee)
    if callee isa Symbol
        isdefined(mod, callee) ? getglobal(mod, callee) : nothing
    elseif callee isa Expr && callee.head === :(.) && length(callee.args) == 2 &&
           callee.args[1] isa Symbol && callee.args[2] isa QuoteNode
        outer = callee.args[1]
        isdefined(mod, outer) || return nothing
        outerval = getglobal(mod, outer)
        inner = callee.args[2].value
        if outerval isa Module
            isdefined(outerval, inner) ? getglobal(outerval, inner) : nothing
        elseif outerval isa Union{_StatefulKernelSkeleton,KernelObjectSpec}
            inner in kernel_endpoint_names(outerval) ? getproperty(outerval, inner) : nothing
        else
            nothing
        end
    else
        nothing
    end
end

# Heads under which lifting an `@node` OUT would change semantics: either CONTROL-
# DEPENDENT (branch/loop — the node would become unconditional in the static graph) or
# a DEFERRED/LEXICAL scope (lambda/function/do/let/quote — the node would escape a local
# binding it references, an accidental capture). A genuine `@node` beneath any of these
# is rejected, never silently hoisted. (Syntactic only — no Julia IR inference.)
_kernel_is_nonstraight_head(h) =
    h in (:if, :(&&), :(||), :for, :while, :comprehension, :generator, :try,
          :(->), :function, :do, :let, :quote)

# Promote every GENUINE RK `@node(expr)` in a recipe block into a distinct hygienic
# recipe node (a `gensym`ed name, collision-free with any authored port), prepended in
# authored order, replacing each occurrence with its node name. Three soundness rules
# (RK 2026-08-26): (a) IDENTITY-AWARE — a foreign `@node` (e.g. `Evil.@node`) sharing
# only the name is NOT promoted, it is left to expand as its own macro; (b) COLLISION-
# FREE — the generated name is a `gensym`, so it cannot alias an authored port; (c) a
# genuine `@node` in a NON-straight-line (branch/loop) context is REJECTED rather than
# silently made unconditional. Returns the SAME block object (byte-identical) when
# nothing is promoted. `mod === nothing` (no definition module) never promotes.
function _kernel_lift_nodes(block, mod)
    (block isa Expr && block.head === :block) || return block
    _kernel_has_node_marker(block) || return block
    lifted = Any[]
    promoted = Ref(0)
    is_rk_node = head -> mod isa Module && _kernel_resolve_binding(mod, head) === var"@node"
    local rewrite
    rewrite = function (x, straight::Bool)
        x isa Expr || return x
        if x.head === :macrocall && !isempty(x.args) && _kernel_is_node_macro(x.args[1])
            if is_rk_node(x.args[1])
                straight || throw(ArgumentError(
                    "@node is only valid in a straight-line recipe context — not beneath a " *
                    "branch/loop (?:/if/&&/||/for/while/try) or a deferred/lexical scope " *
                    "(->/function/do/let/quote), whose local bindings a hoisted node would " *
                    "escape. Increment 1 rejects it rather than silently changing semantics."))
                inner = rewrite(x.args[end], straight)      # nested @node promoted first
                nm = gensym(:node)
                push!(lifted, Expr(:(=), nm, inner))
                promoted[] += 1
                return nm
            else
                # a foreign `@node` by name only — leave it for its own macro to expand.
                return Expr(x.head, Any[rewrite(a, straight) for a in x.args]...)
            end
        end
        sub = _kernel_is_nonstraight_head(x.head) ? false : straight
        Expr(x.head, Any[rewrite(a, sub) for a in x.args]...)
    end
    new_stmts = Any[rewrite(s, true) for s in block.args]
    promoted[] == 0 && return block            # nothing genuine promoted ⇒ unchanged
    Expr(:block, vcat(lifted, new_stmts)...)
end

function _kernel_expand(block, signature_inputs = Tuple{Symbol,Any}[],
                        call_signature = nothing, mod::Union{Module,Nothing} = nothing;
                        nested_specs = Dict{Symbol,Any}())
    # Promote genuine RK `@node(expr)` markers into distinct schedulable recipe nodes
    # (identity-aware, collision-free, straight-line-only). A no-op — byte-identical —
    # for bodies with no `@node` (or only foreign `@node`).
    block = _kernel_lift_nodes(block, mod)
    raw_statements = block isa Expr && block.head === :block ? block.args : Any[block]
    statements = _kernel_normalize_return_expressions(raw_statements)
    graph_var = gensym(:kernel_graph)
    ports_var = gensym(:kernel_ports)
    order_var = gensym(:kernel_port_order)
    have_var = gensym(:kernel_have)
    want_var = gensym(:kernel_want)
    port_vars = Dict{Symbol,Symbol}()
    port_order = Symbol[]
    annotations = Dict{Symbol,Vector{Any}}()
    have_names = Symbol[]
    want_names = Symbol[]
    entries = Any[]
    saw_return = false
    inferred_port_types = Dict{Symbol,Any}()

    declare_ref = GlobalRef(@__MODULE__, :_kernel_declare!)
    push_unique_ref = GlobalRef(@__MODULE__, :_kernel_push_unique!)
    add_ref = GlobalRef(@__MODULE__, :_kernel_add!)
    alias_ref = GlobalRef(@__MODULE__, :_kernel_alias!)
    spec_ref = GlobalRef(@__MODULE__, :KernelSpec)
    graph_ref = GlobalRef(@__MODULE__, :Graph)
    value_ref = GlobalRef(@__MODULE__, :Value)
    dict_ref = GlobalRef(Base, :Dict)
    symbol_ref = GlobalRef(Core, :Symbol)

    function register!(name::Symbol, type_expr)
        if !(name in port_order)
            push!(port_order, name)
            annotations[name] = Any[]
        end
        type_expr === nothing || push!(annotations[name], type_expr)
        name
    end

    # Function-shaped definitions put the default HAVE boundary in the
    # signature, matching ordinary Julia and ReactiveObjects authoring. An
    # omitted annotation is metadata-only `Any`; Julia still specializes the
    # prepared straight-line function on the concrete runtime argument types.
    for (name, type_expr) in signature_inputs
        register!(name, type_expr)
        _kernel_push_unique!(have_names, name)
        push!(entries, (:input, name))
    end

    # Pass 1: collect the complete named-port namespace, all type annotations,
    # and the boundary. Recipe dependency inference must not depend on statement
    # order: forward inputs and forward intermediates are ordinary graph edges.
    for raw in statements
        _kernel_is_line(raw) && continue
        saw_return && throw(ArgumentError("@kernel statements cannot follow return"))

        if raw isa Expr && raw.head === :return
            saw_return = true
            returned = _kernel_return_names(only(raw.args))
            for name in returned
                _kernel_push_unique!(want_names, name)
            end
            push!(entries, (:return, returned))
            continue
        end

        assignment, metadata = _kernel_recipe(raw)
        if assignment isa Expr && assignment.head === :(=) &&
           !(assignment.args[1] isa Symbol &&
             assignment.args[2] isa Expr && assignment.args[2].head === :function)
            lhs, rhs = assignment.args
            lhs_items = lhs isa Expr && lhs.head === :tuple ? lhs.args : Any[lhs]
            outputs = Tuple{Symbol,Any}[]
            for item in lhs_items
                name, type_expr = _kernel_port_decl(item)
                register!(name, type_expr)
                push!(outputs, (name, type_expr))
            end
            plate_expr = _kernel_authored_plate_expr(rhs, mod)
            if plate_expr !== nothing
                length(outputs) == 1 || throw(ArgumentError(
                    "an authored plate produces exactly one named pointwise port"))
                # A derived iterable is ordinary graph work outside the plate.
                # Materialize it as a hygienic named recipe, then feed that port
                # through the same broadcast/Ref boundary as explicit authoring.
                # This is source sugar for `columns = eachcol(matrix); plate(columns)`;
                # the native/tensorized plate lowerers still own the sole traversal.
                for (name, expression) in plate_expr.materialized_arguments
                    register!(name, nothing)
                    push!(entries,
                          (:recipe, Tuple{Symbol,Any}[(name, nothing)], expression,
                           Dict{Symbol,Any}(), nothing))
                end
                if outputs[1][2] === nothing && plate_expr.inferred !== nothing
                    push!(annotations[outputs[1][1]],
                          :(Array{$(plate_expr.inferred)}))
                    inferred_port_types[outputs[1][1]] = plate_expr.inferred
                end
            elseif length(outputs) == 1 && outputs[1][2] === nothing &&
                   outputs[1][1] === _KERNEL_RETURN_PORT &&
                   rhs isa Expr && rhs.head === :call && length(rhs.args) == 2 &&
                   (rhs.args[1] === :sum || rhs.args[1] === GlobalRef(Base, :sum)) &&
                   rhs.args[2] isa Symbol && haskey(inferred_port_types, rhs.args[2])
                push!(annotations[outputs[1][1]], inferred_port_types[rhs.args[2]])
            end
            push!(entries, (:recipe, outputs, rhs, metadata, plate_expr))
            continue
        end

        isempty(metadata) || throw(ArgumentError("@recipe must wrap a recipe assignment"))
        if raw isa Symbol
            name = register!(raw, nothing)
            _kernel_push_unique!(have_names, name)
            push!(entries, (:input, name))
            continue
        elseif raw isa Expr && raw.head === :(::) && raw.args[1] isa Symbol
            name, type_expr = _kernel_port_decl(raw)
            register!(name, type_expr)
            _kernel_push_unique!(have_names, name)
            push!(entries, (:input, name))
            continue
        end
        throw(ArgumentError(
            "unsupported @kernel statement $(repr(raw)); expected an input name, recipe assignment, or return"))
    end

    known = Set(port_order)
    for name in want_names
        name in known || throw(ArgumentError("@kernel returns undeclared port :$name"))
    end

    # Lower constructed object endpoints in ordinary recipes as well as authored
    # plate bodies. Named constructor bindings select the exact owner HAVE cut;
    # endpoint arguments remain the explicit method boundary. The resulting
    # generated KernelSpec is handled by the same transparent nested splicer as
    # a bare `object.endpoint(args...)` call, so no KernelObjectSpec survives to
    # runtime.
    rewritten_entries = Any[]
    for entry in entries
        if entry[1] === :recipe && entry[5] === nothing
            _, outputs, authored_rhs, metadata, plate_expr = entry
            local_types = Dict{Symbol,Any}()
            materialized = Tuple{Symbol,Any,Any}[]
            rewritten, inferred = _kernel_constructed_endpoint(
                authored_rhs, mod, known, nested_specs, local_types;
                materialized = materialized)
            materialized_names = Set{Symbol}()
            for (name, T, rhs) in materialized
                register!(name, T)
                push!(known, name)
                push!(materialized_names, name)
                push!(rewritten_entries,
                      (:recipe, Tuple{Symbol,Any}[(name, T)], rhs,
                       Dict{Symbol,Any}(), nothing))
            end
            for (name, T) in local_types
                name in materialized_names && continue
                push!(annotations[name], T)
            end
            if length(outputs) == 1 && outputs[1][2] === nothing && inferred !== nothing
                push!(annotations[outputs[1][1]], inferred)
            end
            push!(rewritten_entries,
                  (:recipe, outputs, rewritten, metadata, plate_expr))
        else
            push!(rewritten_entries, entry)
        end
    end
    entries = rewritten_entries

    for name in port_order
        port_vars[name] = gensym(name)
    end

    # Pass 2: declare every port before adding any recipe, then emit recipes in
    # authored order. Repeated annotations are checked by `_kernel_declare!` so
    # incompatible declarations fail during construction with their real types.
    prelude = Any[]
    for name in port_order
        value_var = port_vars[name]
        type_exprs = isempty(annotations[name]) ? Any[GlobalRef(Core, :Any)] :
                     annotations[name]
        for type_expr in type_exprs
            push!(prelude, :($value_var = $declare_ref(
                $graph_var, $ports_var, $order_var, $(QuoteNode(name)), $type_expr)))
        end
    end
    for name in have_names
        push!(prelude, :($push_unique_ref($have_var, $(QuoteNode(name)))))
    end

    # Count authored recipe DEFINITIONS per output name — only a SINGLE-definition output may be
    # hard-aliased (`b=a; b=c` are alternative producers of `b`, never a proof `a===c`).
    def_count = Dict{Symbol,Int}()
    for entry in entries
        entry[1] === :recipe || continue
        for (name, _) in entry[2]
            def_count[name] = get(def_count, name, 0) + 1
        end
    end

    body = Any[]
    consumed_names = Set{Symbol}()
    produced_names = Symbol[]
    for entry in entries
        entry[1] === :recipe || continue
        _, outputs, authored_rhs, metadata, plate_expr = entry
        # ALIAS-AT-EXPANSION (RK 2026-08-27): a bare-identity `b = a` — single output, RHS a bare
        # DECLARED port (`known` is the full predeclared port namespace, forward refs included),
        # not `b` itself, no `@recipe` metadata, and `b` has exactly ONE authored definition — is
        # emitted as a canonical alias (proven same-type; typed conversions keep their recipe).
        # No identity recipe is emitted for the collapsed pair; both labels stay for reporting.
        if length(outputs) == 1 && authored_rhs isa Symbol && authored_rhs in known &&
           authored_rhs != outputs[1][1] && isempty(metadata) &&
           def_count[outputs[1][1]] == 1
            op = _kernel_operation(authored_rhs, Symbol[authored_rhs], known)
            cost = 1.0
            push!(body, :($alias_ref($graph_var, $(port_vars[outputs[1][1]]),
                                     $(port_vars[authored_rhs]), $op, $cost,
                                     $(QuoteNode(authored_rhs)))))
            _kernel_push_unique!(produced_names, outputs[1][1])
            push!(consumed_names, authored_rhs)
            continue
        end
        rhs = _kernel_hygienic_catches(authored_rhs, known)
        deps = plate_expr === nothing ? _kernel_free_ports(rhs, known) :
               plate_expr.arguments
        deps = plate_expr === nothing ? _kernel_nested_endpoint_deps(
            rhs, deps, known, mod, nested_specs) : deps
        all(name -> name in known, deps) || throw(ArgumentError(
            "authored plate arguments must be declared graph ports"))
        union!(consumed_names, deps)
        for (name, _) in outputs
            _kernel_push_unique!(produced_names, name)
        end
        dep_values = Expr(:tuple, (port_vars[name] for name in deps)...)
        out_values = Expr(:tuple, (port_vars[name] for (name, _) in outputs)...)
        cost = get(metadata, :cost, 1.0)
        cse_key = get(metadata, :cse_key, nothing)
        effectful = get(metadata, :effectful, false)
        op = plate_expr === nothing ? _kernel_operation(
            rhs, deps, known; tensorize = !effectful, mod = mod,
            nested_specs = nested_specs) : plate_expr.operation
        push!(body, :($add_ref($graph_var, $dep_values, $out_values,
                               $op, $cost, $cse_key, $effectful,
                               $(QuoteNode(rhs)))))
    end
    if !saw_return
        for name in produced_names
            name in consumed_names || _kernel_push_unique!(want_names, name)
        end
    end
    for name in want_names
        push!(body, :($push_unique_ref($want_var, $(QuoteNode(name)))))
    end

    quote
        let
            $graph_var = $graph_ref()
            $ports_var = $dict_ref{$symbol_ref,$value_ref}()
            $order_var = $symbol_ref[]
            $have_var = $symbol_ref[]
            $want_var = $symbol_ref[]
            $(prelude...)
            $(body...)
            $spec_ref($graph_var, $ports_var, $order_var, $have_var, $want_var,
                      $call_signature)
        end
    end
end

function _kernel_signature_argument(argument)
    has_default = argument isa Expr && argument.head === :kw
    declaration = has_default ? argument.args[1] : argument
    declaration isa Expr && declaration.head === :(...) && throw(ArgumentError(
        "@kernel signatures do not support variadic positional or keyword ports"))
    name, type_expr = _kernel_port_decl(declaration)
    default_expr = has_default ? argument.args[2] : nothing
    (; name, type_expr, has_default, default_expr)
end

function _kernel_call_signature_expr(positional, keywords)
    positional_names = Tuple(argument.name for argument in positional)
    keyword_names = Tuple(argument.name for argument in keywords)
    arguments = (positional..., keywords...)
    default_mask = Tuple(argument.has_default for argument in arguments)
    previous = Symbol[]
    providers = Any[]
    required_ref = GlobalRef(@__MODULE__, :_KERNEL_REQUIRED_ARGUMENT)
    for argument in arguments
        if argument.has_default
            parameters = Expr(:tuple, previous...)
            push!(providers, Expr(:->, parameters, argument.default_expr))
        else
            push!(providers, required_ref)
        end
        push!(previous, argument.name)
    end
    signature_ref = GlobalRef(@__MODULE__, :_kernel_call_signature)
    val_ref = GlobalRef(Base, :Val)
    :($signature_ref(
        $val_ref($(QuoteNode(positional_names))),
        $val_ref($(QuoteNode(keyword_names))),
        $val_ref($(QuoteNode(default_mask))),
        $(Expr(:tuple, providers...)),
    ))
end

function _kernel_named_signature(signature)
    signature isa Expr && signature.head === :call || return nothing
    name = first(signature.args)
    name isa Symbol || throw(ArgumentError(
        "@kernel function name must be a symbol, got $(repr(name))"))
    positional_raw = Any[]
    keyword_raw = Any[]
    for argument in signature.args[2:end]
        if argument isa Expr && argument.head === :parameters
            append!(keyword_raw, argument.args)
        else
            push!(positional_raw, argument)
        end
    end
    positional = [_kernel_signature_argument(argument) for argument in positional_raw]
    keywords = [_kernel_signature_argument(argument) for argument in keyword_raw]

    optional_seen = false
    for argument in positional
        if argument.has_default
            optional_seen = true
        elseif optional_seen
            throw(ArgumentError(
                "@kernel optional positional arguments must occur at the end"))
        end
    end

    seen = Set{Symbol}()
    for argument in (positional..., keywords...)
        argument.name in seen && throw(ArgumentError(
            "@kernel signature repeats port :$(argument.name)"))
        push!(seen, argument.name)
    end
    inputs = Tuple{Symbol,Any}[
        (argument.name, argument.type_expr)
        for argument in (positional..., keywords...)
    ]
    # The first positional name is the Mode-2 subject candidate (a body mutating its
    # fields selects the free-method path); keywords are never the subject.
    positional_names = Tuple(argument.name for argument in positional)
    name, inputs, _kernel_call_signature_expr(positional, keywords), positional_names
end

function _kernel_definition_parts(ex::Expr)
    # Returns (name, inputs, call_signature, positional_names, raw_signature, block).
    # `raw_signature` = `ex.args[1]` — the FULL authored signature AST (Mode-2 retains it).
    if ex.head === :(=) && length(ex.args) == 2
        named = _kernel_named_signature(ex.args[1])
        named === nothing || return (named..., ex.args[1], ex.args[2])
    elseif ex.head === :function && length(ex.args) == 2
        named = _kernel_named_signature(ex.args[1])
        named === nothing || return (named..., ex.args[1], ex.args[2])
    end
    nothing
end

# A short, expression-bodied stateless definition exposes its value as a port
# named after the kernel:
#
#     @kernel square(x) = x * x     # HAVE x, WANT square
#
# Keep block bodies byte-identical: their established contract is graph-shaped
# (implicit derived sinks, or an explicit return selecting existing ports). The
# synthetic result assignment is introduced only after the Mode-1/Mode-2
# discriminators have rejected the stateful paths.
function _kernel_expression_result_body(name::Symbol, inputs, body, short_form::Bool)
    # Julia wraps the RHS of `f(x) = expr` in a two-item block containing a
    # LineNumberNode and `expr`. An explicit `begin ... end` has its own inner
    # line marker/statements and stays on the established graph-body path.
    short_form || return body
    body isa Expr && body.head === :block && length(body.args) == 2 &&
        _kernel_is_line(body.args[1]) || return body
    result = body.args[2]
    # Preserve the pre-existing compact graph forms `f(x) = (y = rhs)` and
    # `f(x) = (x::T)`. Expression-result sugar applies to value expressions.
    result isa Expr && (result.head === :(=) || result.head === :return ||
        (result.head === :(::) && result.args[1] isa Symbol)) && return body

    any(input -> input[1] === name, inputs) && throw(ArgumentError(
        "expression-bodied @kernel `$name` cannot expose its result as :$name because " *
        "the signature already declares an input port with that name; use a block body " *
        "and choose an explicit result port"))

    # A bare input result is a proven identity. Preserve its declared metadata
    # type on the synthetic result so `_kernel_alias!` can collapse both labels
    # to one canonical value/physical slot. Untyped inputs and computed results
    # retain the ordinary metadata-only `Any` behavior.
    lhs = name
    if result isa Symbol
        input_index = findfirst(input -> input[1] === result, inputs)
        if input_index !== nothing
            input_type = inputs[input_index][2]
            input_type === nothing || (lhs = Expr(:(::), name, input_type))
        end
    end
    Expr(:block, Expr(:(=), lhs, result), Expr(:return, name))
end

"""
    @kernel model(f, x) = begin
        y = f(x)
    end

    kernel = prepare(model)
    kernel(sin, 1.0)

    @kernel affine(x, scale = 2; offset = scale - 1) = begin
        y = scale * x + offset
    end

    prepare(affine)(3; offset = 4)

    @kernel square(x) = x * x

    prepare(square)(3)

    @kernel begin
        x::Float64
        y::Float64 = f(x)
        return y
    end

Build a named [`KernelSpec`](@ref) without executing any recipe RHS. The primary
form mirrors an ordinary Julia function definition: its signature names the
default `have` ports, including function-valued inputs. The macro binds the
function name to the resulting `KernelSpec`; [`prepare`](@ref) turns it into the
callable straight-line kernel. Both short and long definitions are supported.
Trailing positional defaults and fixed keyword arguments use ordinary Julia
call syntax. Defaults are evaluated at call time, from left to right, and may
refer to earlier positional or keyword arguments. A keyword without a default
is required. Defaulted and keyword arguments remain exposed `have` ports; the
signature adapter only fills their values before entering the same generated
positional kernel. Variadic positional and keyword splats are not supported.

Type annotations are optional metadata. Omitting them does not add dynamic
dispatch to a prepared kernel: Julia specializes the generated function on the
concrete operation and input types at the call site. An assignment creates one
recipe; tuple assignment creates a multi-output recipe, and assigning an output
again creates an alternative producer. Declarations and producers may be
forward-referenced: the complete named-port namespace is collected before
dependencies are inferred. When a caller global has the same name as a port,
qualify the global with its module name.

Every signature argument and assignment is exposed by name. `return` is
optional: without one, derived sink ports form the default `want` boundary;
with one, the returned port names replace that default. Either way, every port
remains selectable through `spec[:name]` or `want = :name`.

A direct call to another stateless `KernelSpec` on a recipe RHS splices that
kernel's recipes into this graph during construction. Pass existing typed ports
positionally and destructure the nested default output boundary exactly. Every
call site receives fresh internal value identities, so repeated calls remain
independent while the outer planner can prune or CSE their recipes. No nested
runtime call remains in a prepared kernel.

A short expression body is a compact single-result kernel. Its value is exposed
as a port named after the kernel, and that port is the default `want` boundary:
`@kernel square(x) = x * x` creates ports `:x` and `:square`. A bare identity
such as `@kernel passthrough(x) = x` aliases the result port directly to the
input without an identity recipe or an additional physical slot. Block bodies
retain the graph-oriented sink/explicit-return behavior described above.

Recipe metadata uses the compact form

    @recipe (cost = 1.2, cse_key = :combined) (a::T, b::T) = combined(u)

All recipe bodies resolve and capture names in the caller's scope. The macro
only constructs closures and graph metadata; recipe bodies run only when a
prepared kernel is invoked.

`@kernel` supports three authoring modes, discriminated by the body. A methodless
recipe body authors a stateless graph `KernelSpec` — this docstring's primary form,
byte-identical to earlier releases. A body with nested method definitions authors an
OBJECT kernel with an IMPLICIT synthesized receiver: pure straight-line methods are
transparent named endpoints, while effectful or control-flow methods remain available
to the stricter stateful compiler. Nested methods declare no `self`/`__self__` formal,
bare unshadowed names are the owner's fields, and `__self__` appears only as a sibling
object-pass actual (`flip!(__self__, depth)`). A methodless body that mutates a field
of its first positional subject — or ANY `!!` name (a strong same-object update) —
authors a free METHOD (e.g.
`leapfrog!(phasepoint; stepsize)`, `nuts!!(state; rng)`). Reactive mutation of a
compiled stateless kernel remains available through [`prepare_reactive`](@ref),
[`set!`](@ref), [`mutate!`](@ref), and [`touch!`](@ref).
"""
macro kernel(ex)
    definition = ex isa Expr ? _kernel_definition_parts(ex) : nothing
    if definition === nothing && ex isa Expr && ex.head === :(=) &&
       length(ex.args) == 2 && ex.args[1] isa Symbol &&
       ex.args[2] isa Expr && ex.args[2].head === :call
        target = _kernel_resolve_binding(__module__, ex.args[2].args[1])
        if target isa Union{_StatefulKernelSkeleton,KernelObjectSpec}
            # Construction-time specialization sugar.  The call returns a
            # transparent object specification; the new binding remains a
            # stable top-level definition just like a method-bearing @kernel.
            return esc(Expr(:const, ex))
        end
    end
    definition === nothing &&
        return esc(_kernel_expand(ex, Tuple{Symbol,Any}[], nothing, __module__))
    name, inputs, call_signature, positional_names, raw_signature, block = definition
    # Discriminator (V7): nested methods ⇒ Mode-1 object kernel; else a methodless
    # body that MUTATES a field of the FIRST positional subject ⇒ Mode-2 free method
    # (independent of `!` spelling); else the byte-identical stateless expansion.
    _kernel_body_has_methods(block) &&
        return _kernel_stateful_expand(name, inputs, call_signature, block, __module__)
    # `!!` is an EXPLICIT strong same-object update registration (locked Form C): it
    # routes Mode-2 regardless of whether the mutation is direct or through a
    # registered/delegated call (`nuts!!(state; rng) = begin step!(state, rng); state end`).
    if _kernel_is_bangbang_name(name)
        isempty(positional_names) && throw(ArgumentError(
            "stateful @kernel `$name` (`!!` strong same-object update) needs a first " *
            "positional subject to update in place"))
        return _kernel_mode2_expand(name, inputs, call_signature, block,
                                    positional_names[1], raw_signature, __module__)
    end
    if !isempty(positional_names) &&
       _kernel_body_mutates_subject(block, positional_names[1])
        return _kernel_mode2_expand(name, inputs, call_signature, block,
                                    positional_names[1], raw_signature, __module__)
    end
    stateless_body = _kernel_expression_result_body(name, inputs, block, ex.head === :(=))
    esc(Expr(:(=), name,
             _kernel_expand(stateless_body, inputs, call_signature, __module__)))
end

# --- named boundaries ------------------------------------------------------

struct _KernelDefaultBoundary end
const _KERNEL_DEFAULT_BOUNDARY = _KernelDefaultBoundary()

function _kernel_selection(spec::KernelSpec, selection, defaults::Vector{Symbol},
                           label::Symbol)
    chosen = selection === _KERNEL_DEFAULT_BOUNDARY ? Tuple(defaults) : selection
    items = chosen isa Tuple || chosen isa AbstractVector ? chosen : (chosen,)
    values = Value[]
    for item in items
        if item isa Symbol
            push!(values, spec[item])
        elseif item isa Value
            owned = get(spec.graph.values, item.id, nothing)
            if owned === nothing || typeof(owned) !== typeof(item) ||
               owned.id != item.id || owned.name !== item.name
                throw(ArgumentError(
                "$label value $(item) does not belong to this KernelSpec"))
            end
            push!(values, owned)
        else
            throw(ArgumentError(
                "$label entries must be port names or Value objects, got $(repr(item))"))
        end
    end
    Tuple(values)
end

"""
    extract(spec::KernelSpec; have=inputs(spec), want=outputs(spec)) -> KernelSpec

Create a transparent named-boundary view of `spec`.  The returned value shares
the same ordinary graph and named ports; only its default HAVE/WANT boundary is
changed.  Planning and preparation therefore use exactly the same recipes as
an explicit `plan(spec; have, want)` call.
"""
function extract(spec::KernelSpec; have = _KERNEL_DEFAULT_BOUNDARY,
                 want = _KERNEL_DEFAULT_BOUNDARY)
    selected_have = _kernel_selection(spec, have, spec.have_names, :have)
    selected_want = _kernel_selection(spec, want, spec.want_names, :want)
    KernelSpec(spec.graph, spec.ports, spec.port_order,
               Symbol[value.name for value in selected_have],
               Symbol[value.name for value in selected_want], nothing)
end

# Construction sugar for an already-authored boundary.  Runtime values still
# go through `prepare(spec)(args...)`; calling the declarative spec itself is a
# graph-view operation and cannot hide execution.
(spec::KernelSpec)(; have = _KERNEL_DEFAULT_BOUNDARY,
                   want = _KERNEL_DEFAULT_BOUNDARY) =
    extract(spec; have = have, want = want)

# Positional application requests the distinguished default output.  This is a
# convenience/compiler-entry surface; performance-sensitive callers should
# still retain `prepare(spec)` so compilation is paid once.
(spec::KernelSpec)(arg, args...) = prepare(spec)(arg, args...)

function plan(spec::KernelSpec; have = _KERNEL_DEFAULT_BOUNDARY,
              want = _KERNEL_DEFAULT_BOUNDARY)
    plan(spec.graph;
         have = _kernel_selection(spec, have, spec.have_names, :have),
         want = _kernel_selection(spec, want, spec.want_names, :want))
end

function prepare(spec::KernelSpec; have = _KERNEL_DEFAULT_BOUNDARY,
                 want = _KERNEL_DEFAULT_BOUNDARY, passes = ())
    prepared = prepare(plan(spec; have = have, want = want); passes = passes)
    have === _KERNEL_DEFAULT_BOUNDARY || return prepared
    _kernel_signature_callable(prepared, spec.call_signature)
end

function prepare!(cache::PreparationCache, spec::KernelSpec;
                  have = _KERNEL_DEFAULT_BOUNDARY,
                  want = _KERNEL_DEFAULT_BOUNDARY, passes = ())
    prepared = prepare!(cache, spec.graph;
                        have = _kernel_selection(spec, have, spec.have_names, :have),
                        want = _kernel_selection(spec, want, spec.want_names, :want),
                        passes = passes)
    have === _KERNEL_DEFAULT_BOUNDARY || return prepared
    _kernel_signature_callable(prepared, spec.call_signature)
end

function prepare_nonallocating(spec::KernelSpec;
                               have = _KERNEL_DEFAULT_BOUNDARY,
                               want = _KERNEL_DEFAULT_BOUNDARY, passes = ())
    prepared = prepare_nonallocating(spec.graph;
        have = _kernel_selection(spec, have, spec.have_names, :have),
        want = _kernel_selection(spec, want, spec.want_names, :want),
        passes = passes)
    have === _KERNEL_DEFAULT_BOUNDARY || return prepared
    _kernel_signature_callable(prepared, spec.call_signature)
end

"""
    replica(spec::KernelSpec; batched, have=inputs(spec), want=outputs(spec), passes=())

Prepare the scalar authored kernel and lift the complete callable over a shared
trailing replica axis. See [`replica(::PreparedKernel)`](@ref).
"""
function replica(spec::KernelSpec;
                 batched,
                 have = _KERNEL_DEFAULT_BOUNDARY,
                 want = _KERNEL_DEFAULT_BOUNDARY,
                 passes = ())
    prepared = prepare(plan(spec; have = have, want = want); passes = passes)
    replicated = replica(prepared; batched = batched)
    have === _KERNEL_DEFAULT_BOUNDARY || return replicated
    _kernel_signature_callable(replicated, spec.call_signature)
end

function replica(callable::_KernelSignatureCallable; batched)
    replicated = _replica(callable.target, batched)
    _kernel_signature_callable(replicated, callable.signature)
end

"""
    plate(spec::KernelSpec; have, want, batched, reduce = :+) -> PreparedKernel

Prepare a batched, loop-invariant-hoisting kernel from a scalar `@kernel` spec.
The `batched` HAVE ports (a name or a collection of names) are passed as arrays;
recipes that depend only on shared scalar ports are computed once, and the
single scalar `want` is mapped over the batched values and reduced (a sum by
default). Pass `reduce = nothing` to collect the per-element wants instead.

The generated map/reduction form is both allocation-free for native reducing
execution and traceable by array compilers such as Reactant. See
[`lower_batched`](@ref).
"""
function plate(spec::KernelSpec; have, want, batched, reduce = :+)
    p = plan(spec; have = have, want = want)
    _prepare_batched(p; batched = batched, reduce = reduce)
end

inputs(spec::KernelSpec) = _kernel_selection(
    spec, _KERNEL_DEFAULT_BOUNDARY, spec.have_names, :have)
outputs(spec::KernelSpec) = _kernel_selection(
    spec, _KERNEL_DEFAULT_BOUNDARY, spec.want_names, :want)

# --- pure named-port composition ------------------------------------------

function _kernel_clone_value_map!(graph::Graph, ports::Dict{Symbol,Value},
                                  spec::KernelSpec)
    public_ids = Dict(value.id => ports[name] for (name, value) in spec.ports)
    mapped = Dict{Int,Value}()
    for id in sort!(collect(keys(spec.graph.values)))
        source = spec.graph.values[id]
        mapped[id] = get(public_ids, id) do
            # Nested inclusions carry graph-visible values which intentionally
            # are not public ports.  Preserve them by identity mapping rather
            # than assuming every recipe value can be recovered by name.
            value!(graph, source.name, valtype(source))
        end
    end
    mapped
end

function _kernel_clone_recipes!(graph::Graph, mapped::Dict{Int,Value},
                                spec::KernelSpec)
    for recipe in spec.graph.recipes
        ins = Tuple(mapped[value.id] for value in recipe.inputs)
        outs = Tuple(mapped[value.id] for value in recipe.outputs)
        add!(graph; inputs = ins, outputs = outs, op = recipe.op,
             cost = recipe.cost, cse_key = _kernel_provenance_key(recipe),
             effectful = recipe.effectful, source = recipe.source)
    end
    graph
end

function _kernel_clone_aliases!(graph::Graph, mapped::Dict{Int,Value},
                                spec::KernelSpec)
    relations = Tuple{Int,Int}[]
    for alias_id in keys(spec.graph.aliases)
        canonical_id = canon_id(spec.graph, alias_id)
        push!(relations, (alias_id, canonical_id))
    end
    sort!(relations)
    for (source_id, target_id) in relations
        alias = mapped[source_id]
        canonical = mapped[target_id]
        alias_id = alias.id
        canonical_id = canon_id(graph, canonical.id)
        source_id = canon_id(graph, alias_id)
        source_id == canonical_id && continue
        valtype(graph.values[source_id]) == valtype(graph.values[canonical_id]) ||
            throw(ArgumentError(
                "cannot merge structural aliases for value :$(alias.name): " *
                "$(valtype(graph.values[source_id])) does not match " *
                "$(valtype(graph.values[canonical_id]))"))
        if source_id != alias_id
            # A repeated transparent inclusion can attach the same public
            # boundary to another fresh internal child value.  Union that new
            # root into the already-established canonical class instead of
            # requiring one alias source to point at two targets.
            graph.aliases[canonical_id] = source_id
        else
            graph.aliases[source_id] = canonical_id
        end
        graph.version += 1
    end
    graph
end

function _kernel_reindex_producers!(graph::Graph)
    empty!(graph.producers)
    for recipe in graph.recipes, output in recipe.outputs
        producers = get!(graph.producers, canon_id(graph, output.id), Int[])
        recipe.id in producers || push!(producers, recipe.id)
    end
    graph.version += 1
    graph
end

"""
    merge(base::KernelSpec, fragment::KernelSpec; boundary=:base) -> KernelSpec

Create a fresh composed kernel by unifying same-name, same-type ports and
copying both recipe sets into a new ordinary graph. Type mismatches are rejected
before construction. The inputs and outputs of `base` remain the default, so a
fragment that merely adds observables preserves the base kernel's drop-in call
contract. Use the explicit `boundary=:fragment` option to replace both defaults
with the fragment boundary.
"""
function Base.merge(base::KernelSpec, fragment::KernelSpec; boundary::Symbol = :base)
    boundary in (:base, :fragment) || throw(ArgumentError(
        "kernel merge boundary must be :base or :fragment, got $(repr(boundary))"))

    shared = sort!(collect(intersect(keys(base.ports), keys(fragment.ports))))
    for name in shared
        left = valtype(base.ports[name])
        right = valtype(fragment.ports[name])
        left == right || throw(ArgumentError(
            "cannot merge kernel port :$name: base type $left does not match fragment type $right"))
    end

    graph = Graph()
    ports = Dict{Symbol,Value}()
    order = Symbol[]
    for spec in (base, fragment), name in spec.port_order
        haskey(ports, name) && continue
        _kernel_declare!(graph, ports, order, name, valtype(spec.ports[name]))
    end
    base_values = _kernel_clone_value_map!(graph, ports, base)
    fragment_values = _kernel_clone_value_map!(graph, ports, fragment)
    # Boundary/internal aliases must be established before recipe insertion so
    # provenance CSE sees the final canonical inputs.
    _kernel_clone_aliases!(graph, base_values, base)
    _kernel_clone_aliases!(graph, fragment_values, fragment)
    _kernel_clone_recipes!(graph, base_values, base)
    _kernel_clone_recipes!(graph, fragment_values, fragment)
    _kernel_reindex_producers!(graph)

    chosen = boundary === :base ? base : fragment
    KernelSpec(graph, ports, order, copy(chosen.have_names), copy(chosen.want_names),
               chosen.call_signature)
end

"""
    compose(specs::KernelSpec...; boundary=:base) -> KernelSpec

Compose declarative specs by named ports. The default preserves the first
spec's boundary through every merge. `boundary=:fragment` explicitly adopts
the final spec's boundary.
"""
function compose(first::KernelSpec, rest::KernelSpec...; boundary::Symbol = :base)
    boundary in (:base, :fragment) || throw(ArgumentError(
        "kernel composition boundary must be :base or :fragment, got $(repr(boundary))"))
    isempty(rest) && return Base.merge(first, @kernel(begin end); boundary = :base)
    combined = first
    for fragment in rest
        combined = Base.merge(combined, fragment; boundary = :base)
    end
    boundary === :base && return combined
    last = rest[end]
    KernelSpec(combined.graph, combined.ports, combined.port_order,
               copy(last.have_names), copy(last.want_names), last.call_signature)
end
