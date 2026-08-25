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

Look up a named port with `spec[:name]`. Planning and preparation accept either
port names or `Value`s in their `have` and `want` overrides.
"""
struct KernelSpec
    graph::Graph
    ports::Dict{Symbol,Value}
    port_order::Vector{Symbol}
    have_names::Vector{Symbol}
    want_names::Vector{Symbol}
end

function Base.show(io::IO, spec::KernelSpec)
    have = join(string.(spec.have_names), ", ")
    want = join(string.(spec.want_names), ", ")
    print(io, "KernelSpec(", have, " -> ", want, "; ",
          length(spec.ports), " ports, ", length(spec.graph.recipes), " recipes)")
end

Base.keys(spec::KernelSpec) = Tuple(spec.port_order)
Base.haskey(spec::KernelSpec, name::Symbol) = haskey(spec.ports, name)

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

function _kernel_free_ports!(out::Vector{Symbol}, seen::Set{Symbol}, ex,
                             known::Set{Symbol}, bound::Set{Symbol})
    if ex isa Symbol
        if ex in known && !(ex in bound) && !(ex in seen)
            push!(out, ex)
            push!(seen, ex)
        end
        return out
    end
    (ex isa Expr) || return out
    ex.head === :quote && return out

    if ex.head === :->
        local_bound = copy(bound)
        _kernel_bound_names!(local_bound, ex.args[1])
        _kernel_free_ports!(out, seen, ex.args[2], known, local_bound)
        return out
    elseif ex.head === :kw
        # The first argument is a keyword or named-tuple label, not a value
        # reference. Only its value expression can depend on graph ports.
        _kernel_free_ports!(out, seen, ex.args[2], known, bound)
        return out
    elseif ex.head === :.
        # Property names are normally QuoteNodes, but treating this form
        # explicitly also keeps a future parser representation from turning a
        # field label into a graph edge.
        _kernel_free_ports!(out, seen, ex.args[1], known, bound)
        if length(ex.args) > 1 && !(ex.args[2] isa QuoteNode) &&
           !(ex.args[2] isa Symbol)
            _kernel_free_ports!(out, seen, ex.args[2], known, bound)
        end
        return out
    elseif ex.head === :let
        local_bound = copy(bound)
        for binding in ex.args[1:end-1]
            if binding isa Expr && (binding.head === :(=) || binding.head === :kw)
                _kernel_free_ports!(out, seen, binding.args[2], known, local_bound)
                _kernel_bound_names!(local_bound, binding.args[1])
            else
                _kernel_bound_names!(local_bound, binding)
            end
        end
        _kernel_free_ports!(out, seen, ex.args[end], known, local_bound)
        return out
    elseif ex.head === :generator || ex.head === :comprehension
        # Generators store their body first and iterator bindings afterwards.
        # Walk iterator domains before extending the local binding set, then
        # visit the body with all generator-local names hidden.
        args = ex.head === :comprehension && length(ex.args) == 1 &&
               ex.args[1] isa Expr && ex.args[1].head === :generator ?
               ex.args[1].args : ex.args
        local_bound = copy(bound)
        for iterator in args[2:end]
            if iterator isa Expr && iterator.head in (:(=), :in)
                _kernel_free_ports!(out, seen, iterator.args[2], known, local_bound)
                _kernel_bound_names!(local_bound, iterator.args[1])
            else
                _kernel_free_ports!(out, seen, iterator, known, local_bound)
            end
        end
        _kernel_free_ports!(out, seen, args[1], known, local_bound)
        return out
    elseif ex.head === :for
        local_bound = copy(bound)
        iterator = ex.args[1]
        if iterator isa Expr && iterator.head in (:(=), :in)
            _kernel_free_ports!(out, seen, iterator.args[2], known, local_bound)
            _kernel_bound_names!(local_bound, iterator.args[1])
        end
        _kernel_free_ports!(out, seen, ex.args[2], known, local_bound)
        return out
    elseif ex.head === :function
        local_bound = copy(bound)
        _kernel_bound_names!(local_bound, ex.args[1])
        _kernel_free_ports!(out, seen, ex.args[2], known, local_bound)
        return out
    end

    for arg in ex.args
        _kernel_free_ports!(out, seen, arg, known, bound)
    end
    out
end

function _kernel_free_ports(ex, known::Set{Symbol})
    out = Symbol[]
    _kernel_free_ports!(out, Set{Symbol}(), ex, known, Set{Symbol}())
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

function _kernel_expand(block)
    statements = block isa Expr && block.head === :block ? block.args : Any[block]
    graph_var = gensym(:kernel_graph)
    ports_var = gensym(:kernel_ports)
    order_var = gensym(:kernel_port_order)
    have_var = gensym(:kernel_have)
    want_var = gensym(:kernel_want)
    port_vars = Dict{Symbol,Symbol}()
    known = Set{Symbol}()
    body = Any[]
    saw_return = false

    declare_ref = GlobalRef(@__MODULE__, :_kernel_declare!)
    push_unique_ref = GlobalRef(@__MODULE__, :_kernel_push_unique!)
    add_ref = GlobalRef(@__MODULE__, :_kernel_add!)
    spec_ref = GlobalRef(@__MODULE__, :KernelSpec)
    graph_ref = GlobalRef(@__MODULE__, :Graph)
    value_ref = GlobalRef(@__MODULE__, :Value)

    function declare!(name::Symbol, type_expr; input::Bool)
        if !haskey(port_vars, name)
            type_expr === nothing && throw(ArgumentError(
                "new kernel port :$name needs an explicit type annotation, for example $name::Float64"))
            port_vars[name] = gensym(name)
            push!(known, name)
        elseif type_expr === nothing
            input && push!(body, :($push_unique_ref($have_var, $(QuoteNode(name)))))
            return port_vars[name]
        end
        if type_expr !== nothing
            value_var = port_vars[name]
            push!(body, :($value_var = $declare_ref(
                $graph_var, $ports_var, $order_var, $(QuoteNode(name)), $type_expr)))
        end
        input && push!(body, :($push_unique_ref($have_var, $(QuoteNode(name)))))
        port_vars[name]
    end

    for raw in statements
        _kernel_is_line(raw) && continue
        saw_return && throw(ArgumentError("@kernel statements cannot follow return"))

        if raw isa Expr && raw.head === :return
            saw_return = true
            returned = _kernel_return_names(only(raw.args))
            for name in returned
                name in known || throw(ArgumentError(
                    "@kernel returns undeclared port :$name"))
                push!(body, :($push_unique_ref($want_var, $(QuoteNode(name)))))
            end
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
                if !(name in known) && type_expr === nothing
                    throw(ArgumentError(
                        "new kernel output :$name needs an explicit type annotation"))
                end
                push!(outputs, (name, type_expr))
            end

            deps = _kernel_free_ports(rhs, known)
            for (name, type_expr) in outputs
                declare!(name, type_expr; input = false)
            end
            dep_values = Expr(:tuple, (port_vars[name] for name in deps)...)
            out_values = Expr(:tuple, (port_vars[name] for (name, _) in outputs)...)
            op = Expr(:->, Expr(:tuple, deps...), rhs)
            cost = get(metadata, :cost, 1.0)
            cse_key = get(metadata, :cse_key, nothing)
            effectful = get(metadata, :effectful, false)
            push!(body, :($add_ref($graph_var, $dep_values, $out_values,
                                   $op, $cost, $cse_key, $effectful)))
            continue
        end

        isempty(metadata) || throw(ArgumentError("@recipe must wrap a recipe assignment"))
        if raw isa Expr && raw.head === :(::) && raw.args[1] isa Symbol
            name, type_expr = _kernel_port_decl(raw)
            declare!(name, type_expr; input = true)
            continue
        end
        throw(ArgumentError(
            "unsupported @kernel statement $(repr(raw)); expected a typed input, recipe assignment, or return"))
    end

    quote
        let
            $graph_var = $graph_ref()
            $ports_var = Dict{Symbol,$value_ref}()
            $order_var = Symbol[]
            $have_var = Symbol[]
            $want_var = Symbol[]
            $(body...)
            $spec_ref($graph_var, $ports_var, $order_var, $have_var, $want_var)
        end
    end
end

"""
    @kernel begin
        x::Float64
        y::Float64 = f(x)
        return y
    end

Build a named, typed [`KernelSpec`](@ref) without executing any recipe RHS.
Bare typed declarations are default `have` ports. An assignment creates one
recipe; newly introduced outputs need a type annotation, tuple assignment
creates a multi-output recipe, and assigning an existing output again creates
an alternative producer. `return` declares the default `want` boundary.

Recipe metadata uses the compact form

    @recipe (cost = 1.2, cse_key = :combined) (a::T, b::T) = combined(u)

All recipe bodies resolve and capture names in the caller's scope. The macro
only constructs closures and graph metadata; recipe bodies run only when a
prepared kernel is invoked.
"""
macro kernel(block)
    esc(_kernel_expand(block))
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
            haskey(spec.graph.values, item.id) || throw(ArgumentError(
                "$label value $(item) does not belong to this KernelSpec"))
            push!(values, item)
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
    prepare(plan(spec; have = have, want = want); passes = passes)
end

function prepare!(cache::PreparationCache, spec::KernelSpec;
                  have = _KERNEL_DEFAULT_BOUNDARY,
                  want = _KERNEL_DEFAULT_BOUNDARY, passes = ())
    prepare!(cache, spec.graph;
             have = _kernel_selection(spec, have, spec.have_names, :have),
             want = _kernel_selection(spec, want, spec.want_names, :want),
             passes = passes)
end

function prepare_nonallocating(spec::KernelSpec;
                               have = _KERNEL_DEFAULT_BOUNDARY,
                               want = _KERNEL_DEFAULT_BOUNDARY, passes = ())
    prepare_nonallocating(spec.graph;
        have = _kernel_selection(spec, have, spec.have_names, :have),
        want = _kernel_selection(spec, want, spec.want_names, :want),
        passes = passes)
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

    chosen = boundary === :base ? base : fragment
    KernelSpec(graph, ports, order, copy(chosen.have_names), copy(chosen.want_names))
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
               copy(last.have_names), copy(last.want_names))
end
