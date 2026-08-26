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

function _kernel_add!(graph::Graph, ins, outs, op, cost, cse_key, effectful)
    add!(graph; inputs = ins, outputs = outs, op = op,
         cost = cost, cse_key = cse_key, effectful = effectful)
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

# A dotted operator such as `.*` is broadcast *syntax*, not a bound function:
# `a .* b` lowers to `broadcast(*, a, b)` and there is no callable named `.*`.
# Splicing the bare symbol as a recipe callee errors at construction with
# `UndefVarError: .* not defined`, so such recipes must take the closure path.
function _is_broadcast_operator(sym::Symbol)
    s = String(sym)
    length(s) >= 2 && s[1] === '.' && Base.isoperator(Symbol(s[2:end]))
end

function _kernel_operation(rhs, deps::Vector{Symbol}, known::Set{Symbol})
    if rhs isa Expr && rhs.head === :call && !isempty(rhs.args)
        callee = rhs.args[1]
        args = rhs.args[2:end]
        callee_is_port = callee isa Symbol && callee in known
        if callee isa Symbol && !callee_is_port && !_is_broadcast_operator(callee) &&
           length(args) == length(deps) &&
           all(i -> args[i] === deps[i], eachindex(args))
            return callee
        end
    end
    Expr(:->, Expr(:tuple, deps...), rhs)
end

# `@node(expr)` recipe-node promotion, used by `_kernel_expand`. Kept HERE (ahead of
# `_kernel_expand`, not in kernel_stateful.jl) because authoring.jl loads first and a
# `@doc`-embedded `@kernel` expands at load time — a forward ref would be undefined.

# Is a macrocall head the `@node` promoter by NAME (bare or module-qualified)? A
# SYNTACTIC candidate only; identity is confirmed by `_kernel_marker_kind(mod, ex)`.
_kernel_is_node_macro(m) =
    m === Symbol("@node") ||
    (m isa Expr && m.head === :(.) && length(m.args) >= 2 &&
        m.args[2] == QuoteNode(Symbol("@node")))

# Any `@node(...)` marker anywhere in `x`?
_kernel_has_node_marker(x) =
    x isa Expr && (
        (x.head === :macrocall && !isempty(x.args) && _kernel_is_node_macro(x.args[1])) ||
        any(_kernel_has_node_marker, x.args))

# Promote every `@node(expr)` in a recipe block into a distinct named recipe node
# (`#node#k = expr`), prepended in authored order, replacing each occurrence with its
# node name. Returns the SAME block object (byte-identical) when no `@node` is present,
# so ordinary recipes are unchanged. `#`-prefixed names are hygienic (non-authorable)
# and deterministic (stable within one expansion). Nested `@node` is promoted inner-first.
function _kernel_lift_nodes(block)
    (block isa Expr && block.head === :block) || return block
    _kernel_has_node_marker(block) || return block          # no @node ⇒ byte-identical
    lifted = Any[]
    counter = Ref(0)
    local rewrite
    rewrite = function (x)
        x isa Expr || return x
        if x.head === :macrocall && !isempty(x.args) && _kernel_is_node_macro(x.args[1])
            inner = rewrite(x.args[end])                    # nested @node promoted first
            counter[] += 1
            nm = Symbol("#node#", counter[])
            push!(lifted, Expr(:(=), nm, inner))
            return nm
        end
        Expr(x.head, Any[rewrite(a) for a in x.args]...)
    end
    new_stmts = Any[rewrite(s) for s in block.args]
    Expr(:block, vcat(lifted, new_stmts)...)
end

function _kernel_expand(block, signature_inputs = Tuple{Symbol,Any}[],
                        call_signature = nothing)
    # Promote `@node(expr)` markers into distinct named recipe nodes so a marked
    # anonymous subexpression becomes a real schedulable graph node/port (RK 2026-08-26).
    # A no-op returning the SAME block — byte-identical — when no `@node` is present.
    block = _kernel_lift_nodes(block)
    statements = block isa Expr && block.head === :block ? block.args : Any[block]
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

    declare_ref = GlobalRef(@__MODULE__, :_kernel_declare!)
    push_unique_ref = GlobalRef(@__MODULE__, :_kernel_push_unique!)
    add_ref = GlobalRef(@__MODULE__, :_kernel_add!)
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
            push!(entries, (:recipe, outputs, rhs, metadata))
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

    body = Any[]
    consumed_names = Set{Symbol}()
    produced_names = Symbol[]
    for entry in entries
        entry[1] === :recipe || continue
        _, outputs, authored_rhs, metadata = entry
        rhs = _kernel_hygienic_catches(authored_rhs, known)
        deps = _kernel_free_ports(rhs, known)
        union!(consumed_names, deps)
        for (name, _) in outputs
            _kernel_push_unique!(produced_names, name)
        end
        dep_values = Expr(:tuple, (port_vars[name] for name in deps)...)
        out_values = Expr(:tuple, (port_vars[name] for (name, _) in outputs)...)
        op = _kernel_operation(rhs, deps, known)
        cost = get(metadata, :cost, 1.0)
        cse_key = get(metadata, :cse_key, nothing)
        effectful = get(metadata, :effectful, false)
        push!(body, :($add_ref($graph_var, $dep_values, $out_values,
                               $op, $cost, $cse_key, $effectful)))
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
    if ex.head === :(=) && length(ex.args) == 2
        named = _kernel_named_signature(ex.args[1])
        named === nothing || return (named..., ex.args[2])
    elseif ex.head === :function && length(ex.args) == 2
        named = _kernel_named_signature(ex.args[1])
        named === nothing || return (named..., ex.args[2])
    end
    nothing
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

Recipe metadata uses the compact form

    @recipe (cost = 1.2, cse_key = :combined) (a::T, b::T) = combined(u)

All recipe bodies resolve and capture names in the caller's scope. The macro
only constructs closures and graph metadata; recipe bodies run only when a
prepared kernel is invoked.

`@kernel` supports three authoring modes, discriminated by the body. A methodless
recipe body authors a stateless graph `KernelSpec` — this docstring's primary form,
byte-identical to earlier releases. A body with nested method definitions authors a
stateful OBJECT kernel with an IMPLICIT synthesized receiver: nested methods declare
no `self`/`__self__` formal, bare unshadowed names are the owner's fields, and
`__self__` appears only as a sibling object-pass actual (`flip!(__self__, depth)`). A
methodless body that mutates a field of its first positional subject — or ANY `!!`
name (a strong same-object update) — authors a free METHOD (e.g.
`leapfrog!(phasepoint; stepsize)`, `nuts!!(state; rng)`). Reactive mutation of a
compiled stateless kernel remains available through [`prepare_reactive`](@ref),
[`set!`](@ref), [`mutate!`](@ref), and [`touch!`](@ref).
"""
macro kernel(ex)
    definition = ex isa Expr ? _kernel_definition_parts(ex) : nothing
    definition === nothing && return esc(_kernel_expand(ex))
    name, inputs, call_signature, positional_names, block = definition
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
                                    positional_names[1], __module__)
    end
    if !isempty(positional_names) &&
       _kernel_body_mutates_subject(block, positional_names[1])
        return _kernel_mode2_expand(name, inputs, call_signature, block,
                                    positional_names[1], __module__)
    end
    esc(Expr(:(=), name, _kernel_expand(block, inputs, call_signature)))
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

inputs(spec::KernelSpec) = _kernel_selection(
    spec, _KERNEL_DEFAULT_BOUNDARY, spec.have_names, :have)
outputs(spec::KernelSpec) = _kernel_selection(
    spec, _KERNEL_DEFAULT_BOUNDARY, spec.want_names, :want)

# --- pure named-port composition ------------------------------------------

function _kernel_clone_recipes!(graph::Graph, ports::Dict{Symbol,Value},
                                spec::KernelSpec)
    for recipe in spec.graph.recipes
        ins = Tuple(ports[value.name] for value in recipe.inputs)
        outs = Tuple(ports[value.name] for value in recipe.outputs)
        add!(graph; inputs = ins, outputs = outs, op = recipe.op,
             cost = recipe.cost, cse_key = recipe.cse_key,
             effectful = recipe.effectful)
    end
    graph
end

function _kernel_clone_aliases!(graph::Graph, ports::Dict{Symbol,Value},
                                spec::KernelSpec)
    relations = Tuple{Symbol,Symbol}[]
    for alias_id in keys(spec.graph.aliases)
        canonical_id = canon_id(spec.graph, alias_id)
        push!(relations, (
            spec.graph.values[alias_id].name,
            spec.graph.values[canonical_id].name,
        ))
    end
    sort!(relations)
    for (alias_name, canonical_name) in relations
        alias_id = ports[alias_name].id
        canonical_id = canon_id(graph, ports[canonical_name].id)
        alias_id == canonical_id && continue
        if haskey(graph.aliases, alias_id)
            canon_id(graph, alias_id) == canonical_id || throw(ArgumentError(
                "cannot merge structural aliases for port :$alias_name"))
            continue
        end
        canon_id(graph, canonical_id) == alias_id && throw(ArgumentError(
            "cannot merge structural aliases: :$alias_name and :$canonical_name would form a cycle"))
        graph.aliases[alias_id] = canonical_id
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
    _kernel_clone_recipes!(graph, ports, base)
    _kernel_clone_aliases!(graph, ports, base)
    _kernel_clone_recipes!(graph, ports, fragment)
    _kernel_clone_aliases!(graph, ports, fragment)
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
