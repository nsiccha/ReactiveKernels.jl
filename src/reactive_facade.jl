# The public `@reactive`-style object/method authoring facade.
#
# `@reactive` mirrors ReactiveHMC.jl/ReactiveObjects.jl's `@reactive`: a
# near-verbatim function-shaped definition whose signature arguments become
# mutable HAVE sources, whose body `lhs = rhs` assignments become compiled
# reactive derived nodes, and whose inner method definitions become ordinary
# type-stable Julia methods dispatched on the generated object. The reactive
# STATE and its invalidation are the compiled `ReactiveProgram` (built by
# reusing the `@kernel` parser/declaration machinery); the methods are ordinary
# Julia exactly as upstream. This is the object layer above `prepare_reactive`;
# it adds no HMC-specific core path.

"""
    ReactiveObject{Name,S,H}

A stateful reactive object produced by an [`@reactive`](@ref) definition.
`Name` is the definition name (a `Symbol` type parameter) so generated methods
dispatch per definition; `state` is a [`CompiledReactiveState`](@ref) and
`handles` is a `NamedTuple` of typed [`ReactiveValue`](@ref) handles keyed by
port/field name. Reading a field lazily recomputes invalid dependencies;
assigning or dotted-updating a HAVE source field invalidates its downstream.
"""
struct ReactiveObject{Name,S,H}
    state::S
    handles::H
end

ReactiveObject{Name}(state::S, handles::H) where {Name,S,H} =
    ReactiveObject{Name,S,H}(state, handles)

_reactive_name(::ReactiveObject{Name}) where {Name} = Name

@inline Base.getproperty(object::ReactiveObject, name::Symbol) =
    getproperty(object, Val(name))
@inline Base.getproperty(object::ReactiveObject, ::Val{:state}) =
    getfield(object, :state)
@inline Base.getproperty(object::ReactiveObject, ::Val{:handles}) =
    getfield(object, :handles)
@inline Base.getproperty(object::ReactiveObject, ::Val{:program}) =
    getfield(object, :state).program

@generated function Base.getproperty(object::ReactiveObject{Name,S,H},
                                     ::Val{name}) where {Name,S,H,name}
    name in fieldnames(H) || return :(getfield(object, $(QuoteNode(name))))
    quote
        handles = getfield(object, :handles)
        handle = getfield(handles, $(QuoteNode(name)))
        get!(getfield(object, :state), handle)
    end
end

@inline Base.setproperty!(object::ReactiveObject, name::Symbol, value) =
    setproperty!(object, Val(name), value)

@generated function Base.setproperty!(object::ReactiveObject{Name,S,H},
                                      ::Val{name}, value) where {Name,S,H,name}
    name in fieldnames(H) || return :(throw(ArgumentError(
        "ReactiveObject $($(QuoteNode(Name))) has no field $($(QuoteNode(name)))")))
    quote
        handles = getfield(object, :handles)
        assign!(getfield(object, :state),
                getfield(handles, $(QuoteNode(name))), value)
    end
end

Base.propertynames(object::ReactiveObject, private::Bool = false) =
    private ? (:state, :handles, :program,
               propertynames(getfield(object, :handles))...) :
              (:program, propertynames(getfield(object, :handles))...)

struct _ReactiveObjectProperty{P,H}
    object::P
    handle::H
end

@inline Base.dotgetproperty(object::ReactiveObject, name::Symbol) =
    Base.dotgetproperty(object, Val(name))

@generated function Base.dotgetproperty(object::ReactiveObject{Name,S,H},
                                        ::Val{name}) where {Name,S,H,name}
    name in fieldnames(H) || return :(getproperty(object, Val(name)))
    :(_ReactiveObjectProperty(
        object, getfield(getfield(object, :handles), $(QuoteNode(name)))))
end

function Base.materialize!(destination::_ReactiveObjectProperty,
                           broadcasted::Base.Broadcast.Broadcasted)
    assign!(getfield(destination, :object).state, destination.handle) do value
        Base.materialize!(value, broadcasted)
    end
end

@inline mutate!(f, object::ReactiveObject, name::Symbol) =
    mutate!(f, object, Val(name))

@generated function mutate!(f, object::ReactiveObject{Name,S,H},
                            ::Val{name}) where {Name,S,H,name}
    name in fieldnames(H) || return :(throw(ArgumentError(
        "ReactiveObject $($(QuoteNode(Name))) has no field $($(QuoteNode(name)))")))
    quote
        mutate!(f, getfield(object, :state),
                getfield(getfield(object, :handles), $(QuoteNode(name))))
    end
end

reactive_program(object::ReactiveObject) = getfield(object, :state).program

Base.copy(object::ReactiveObject{Name}) where {Name} =
    ReactiveObject{Name}(copy(getfield(object, :state)), getfield(object, :handles))

function Base.copyto!(destination::ReactiveObject{Name},
                      source::ReactiveObject{Name}) where {Name}
    getfield(destination, :handles) === getfield(source, :handles) ||
        throw(ArgumentError("reactive objects belong to different programs"))
    copyto!(getfield(destination, :state), getfield(source, :state))
    destination
end

# Build the exposed-handles NamedTuple for a prepared program: every port/field
# name the selected plan actually kept a slot for, keyed by name. Used by the
# constructor `@reactive` generates.
function _reactive_handles(program::ReactiveProgram, spec::KernelSpec, names)
    exposed = Tuple(name for name in names
                    if haskey(program.index, canon_id(spec.graph, spec[name].id)))
    NamedTuple{exposed}(Tuple(statevalue(program, spec[name]) for name in exposed))
end

# --- @reactive macro ---------------------------------------------------------

_reactive_is_method(stmt) =
    stmt isa Expr && (stmt.head === :function ||
        (stmt.head === :(=) && stmt.args[1] isa Expr && stmt.args[1].head === :call))

# The declared name(s) on the LHS of a kernel-body assignment (symbol, x::T, or
# a destructuring tuple for a multi-output recipe).
function _reactive_assign_names!(names, lhs)
    if lhs isa Symbol
        push!(names, lhs)
    elseif lhs isa Expr && lhs.head === :(::)
        _reactive_assign_names!(names, lhs.args[1])
    elseif lhs isa Expr && lhs.head === :tuple
        for element in lhs.args
            _reactive_assign_names!(names, element)
        end
    end
    names
end

# Names bound by a method signature (its parameters / where-vars), which must
# NOT be rewritten as fields inside that method body.
function _reactive_signature_bindings(signature)
    bound = Set{Symbol}()
    if signature isa Expr && signature.head === :call
        for arg in signature.args[2:end]
            _kernel_bound_names!(bound, arg)   # positional, typed, kw, and :parameters
        end
    end
    _kernel_where_names!(bound, signature)
    bound
end

# Rewrite a method body so field/port reads become getproperty, field writes
# become setproperty!, and sibling-method calls receive the object. `shadow`
# holds names locally bound (method params, loop/let/local vars) that keep their
# ordinary meaning. Ordinary Julia control flow is preserved verbatim.
function _reactive_rewrite(ex, fields::Set{Symbol}, methods::Set{Symbol},
                           self::Symbol, shadow::Set{Symbol})
    if ex isa Symbol
        (ex in fields && !(ex in shadow)) &&
            return Expr(:call, getproperty, self, QuoteNode(ex))
        return ex
    end
    ex isa Expr || return ex
    ex.head === :quote && return ex
    rw(node) = _reactive_rewrite(node, fields, methods, self, shadow)

    if ex.head === :(=)
        lhs = ex.args[1]
        if lhs isa Symbol && lhs in fields && !(lhs in shadow)
            return Expr(:call, setproperty!, self, QuoteNode(lhs), rw(ex.args[2]))
        end
        # a local binding shadows any same-named field within the rest of scope
        local_names = Set{Symbol}()
        _reactive_assign_names!(local_names, lhs)
        inner = union(shadow, local_names)
        return Expr(:(=), lhs, _reactive_rewrite(ex.args[2], fields, methods, self, inner))
    elseif ex.head === :call && ex.args[1] isa Symbol && ex.args[1] in methods
        # sibling-method call: inject the object as the first POSITIONAL arg,
        # keeping any :parameters (keyword) block in its AST position.
        kwc = nothing; posc = Any[]
        for a in ex.args[2:end]
            if a isa Expr && a.head === :parameters
                kwc = Expr(:parameters,
                           (k isa Expr && k.head === :kw ?
                            Expr(:kw, k.args[1], rw(k.args[2])) : rw(k) for k in a.args)...)
            else
                push!(posc, rw(a))
            end
        end
        out = Any[ex.args[1]]
        kwc === nothing || push!(out, kwc)
        push!(out, self)
        append!(out, posc)
        return Expr(:call, out...)
    elseif ex.head === :. && length(ex.args) == 2 && ex.args[2] isa QuoteNode
        # obj.field property access: rewrite the object, keep the field literal
        return Expr(:., rw(ex.args[1]), ex.args[2])
    elseif ex.head in (:for, :while)
        bind = Set{Symbol}()
        ex.head === :for && _reactive_assign_names!(bind, ex.args[1] isa Expr &&
            ex.args[1].head === :(=) ? ex.args[1].args[1] : ex.args[1])
        inner = union(shadow, bind)
        return Expr(ex.head, (_reactive_rewrite(a, fields, methods, self, inner)
                              for a in ex.args)...)
    end
    Expr(ex.head, (rw(arg) for arg in ex.args)...)
end

# Parse a facade definition `name(positional...; keyword...) = body`. Unlike
# @kernel's parser this accepts a `:parameters` (keyword) block: keyword ports
# are HAVE sources initialized from their defaults (or caller-supplied kwargs).
# Returns (name, positional::[(port,type)], keywords::[(port,type,hasdefault,default)], body).
function _reactive_signature_parts(signature)
    signature isa Expr && signature.head === :call || return nothing
    name = signature.args[1]
    name isa Symbol || throw(ArgumentError(
        "@reactive name must be a symbol, got $(repr(name))"))
    positional = Tuple{Symbol,Any}[]
    keywords = Tuple{Symbol,Any,Bool,Any}[]
    for arg in signature.args[2:end]
        if arg isa Expr && arg.head === :parameters
            for kw in arg.args
                if kw isa Expr && kw.head === :kw
                    port, type = _kernel_port_decl(kw.args[1])
                    push!(keywords, (port, type, true, kw.args[2]))
                else
                    port, type = _kernel_port_decl(kw)
                    push!(keywords, (port, type, false, nothing))
                end
            end
        else
            port, type = _kernel_port_decl(arg)
            push!(positional, (port, type))
        end
    end
    (name, positional, keywords)
end

function _reactive_definition_parts(def)
    def isa Expr && (def.head === :(=) || def.head === :function) &&
        length(def.args) == 2 || return nothing
    parts = _reactive_signature_parts(def.args[1])
    parts === nothing && return nothing
    (parts..., def.args[2])
end

function _reactive_expand(def)
    parts = _reactive_definition_parts(def)
    parts === nothing && throw(ArgumentError(
        "@reactive expects `name(args...; kw...) = begin ... end`, got $(repr(def))"))
    name, positional, keywords, body = parts
    body isa Expr && body.head === :block ||
        (body = Expr(:block, body))

    inputs = Tuple{Symbol,Any}[]
    for (p, t) in positional; push!(inputs, (p, t)); end
    for (p, t, _, _) in keywords; push!(inputs, (p, t)); end
    port_names = Symbol[p for (p, _) in inputs]
    method_defs = Any[]
    kernel_stmts = Any[]
    field_names = Symbol[]
    method_names = Symbol[]
    for stmt in body.args
        stmt isa LineNumberNode && continue
        if _reactive_is_method(stmt)
            sig = stmt.args[1]
            mname = Set{Symbol}(); _kernel_callable_name!(mname, sig)
            isempty(mname) || push!(method_names, first(mname))
            push!(method_defs, stmt)
        else
            push!(kernel_stmts, stmt)
            (stmt isa Expr && stmt.head === :(=)) &&
                _reactive_assign_names!(field_names, stmt.args[1])
        end
    end

    fields = Set{Symbol}(vcat(port_names, field_names))
    methods = Set{Symbol}(method_names)
    self = Symbol("__self__")

    # The reactive graph: reuse @kernel's parser over the recipe statements, with
    # the signature args as HAVE ports. Every field is materialized (want) so the
    # object can expose it.
    kernel_body = Expr(:block, kernel_stmts...)
    spec_build = _kernel_expand(kernel_body, inputs)
    want_tuple = Expr(:tuple, (QuoteNode(f) for f in field_names)...)
    expose_tuple = Expr(:tuple, (QuoteNode(n) for n in vcat(port_names, field_names))...)

    # Build everything with plain symbols and `esc` the whole output once so the
    # generated `__self__`, user param names, and user calls share one hygiene
    # context; runtime helpers are interpolated as VALUES (immune to esc).
    spec_sym = gensym(:spec); prog_sym = gensym(:program); handles_sym = gensym(:handles)
    pos_names = Symbol[p for (p, _) in positional]
    # Julia AST: the :parameters (keyword) block comes right after the callee.
    ctor_sig_args = Any[name]
    if !isempty(keywords)
        kw_params = Any[hasdefault ? Expr(:kw, p, default) : p
                        for (p, _, hasdefault, default) in keywords]
        push!(ctor_sig_args, Expr(:parameters, kw_params...))
    end
    append!(ctor_sig_args, pos_names)
    ctor_body = quote
        $spec_sym = $spec_build
        $prog_sym = $prepare_reactive($spec_sym; want = $want_tuple)
        $handles_sym = $_reactive_handles($prog_sym, $spec_sym, $expose_tuple)
        $(ReactiveObject){$(QuoteNode(name))}(
            $prog_sym($(port_names...)), $handles_sym)
    end
    constructor = Expr(:function, Expr(:call, ctor_sig_args...), ctor_body)

    method_code = Any[]
    self_arg = Expr(:(::), self, Expr(:curly, ReactiveObject, QuoteNode(name)))
    for mdef in method_defs
        sig = mdef.args[1]; mbody = mdef.args[2]
        mname = first(let s = Set{Symbol}(); _kernel_callable_name!(s, sig); s end)
        bound = _reactive_signature_bindings(sig)
        # A default value may reference a field; self is in scope in the signature.
        rw_default(p) = (p isa Expr && p.head === :kw) ?
            Expr(:kw, p.args[1], _reactive_rewrite(p.args[2], fields, methods, self, bound)) : p
        kwblock = nothing; posparams = Any[]
        for pr in sig.args[2:end]
            pr isa Expr && pr.head === :parameters ?
                (kwblock = Expr(:parameters, map(rw_default, pr.args)...)) :
                push!(posparams, rw_default(pr))
        end
        new_body = _reactive_rewrite(mbody, fields, methods, self, bound)
        sig_args = Any[mname]
        kwblock === nothing || push!(sig_args, kwblock)   # :parameters right after callee
        push!(sig_args, self_arg)
        append!(sig_args, posparams)
        push!(method_code, Expr(:function, Expr(:call, sig_args...), new_body))
    end

    esc(Expr(:block, constructor, method_code..., name))
end

"""
    @reactive name(args...) = begin
        field = expr(args...)      # compiled reactive derived node
        method!(x) = ...           # ordinary Julia method: method!(obj, x)
    end

Author a stateful reactive object. Signature arguments become mutable HAVE
sources; body `lhs = rhs` assignments become compiled reactive derived nodes
(their invalidation is the underlying [`ReactiveProgram`](@ref)); inner function
definitions become ordinary type-stable methods that take the object as their
first argument, with field references routed through the object. Returns a
constructor bound to `name` producing a [`ReactiveObject`](@ref).
"""
macro reactive(def)
    _reactive_expand(def)
end
