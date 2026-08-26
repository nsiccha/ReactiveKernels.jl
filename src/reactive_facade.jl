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
        assign!(getfield(object, :state),
                getfield(getfield(object, :handles), $(QuoteNode(name))), value)
        value                       # a Julia assignment returns the assigned value
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

# In-place root-field mutation boundary. Every indexed/property-chain/compound/
# dotted/destructuring/@. field write the facade generates goes through this,
# which delegates to the poc-approved `assign!` in-place form: it materializes the
# root slot buffer, runs `f` (the substituted store), and marks valid + invalidates
# the root's dependents in a `finally` (exception-safe, even on a partial write).
# The facade therefore reaches NONE of the guarded core internals directly — only
# `assign!` — and the field is a `Val` type parameter, so there is no runtime
# Symbol lookup. RHS evaluation order is preserved by the caller staging the RHS
# before this call for the plain `=`/compound cases.
@generated function _reactive_inplace!(f, object::ReactiveObject{Name,S,H},
                                       ::Val{name}) where {Name,S,H,name}
    name in fieldnames(H) || return :(throw(ArgumentError(
        "ReactiveObject $($(QuoteNode(Name))) has no field $($(QuoteNode(name)))")))
    quote
        assign!(f, getfield(object, :state),
                getfield(getfield(object, :handles), $(QuoteNode(name))))
    end
end

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

# Validate a program returned by an injected `prepare=` callable (or built in the
# specialized constructor) against the authored spec: it must be built from the
# SAME graph, its HAVE inputs must equal the signature ports in exact order, and
# every exposed/requested port must be present (so `_reactive_handles` never
# silently drops a field). Returns the validated program.
function _reactive_validate_program(program::ReactiveProgram, spec::KernelSpec,
                                    expose_names)
    program.graph === spec.graph || throw(ArgumentError(
        "@reactive prepare callable returned a program built from a different " *
        "graph/KernelSpec than the authored object"))
    expected = Tuple(canon_id(spec.graph, spec[name].id) for name in spec.have_names)
    actual = Tuple(canon_id(spec.graph, value.id) for value in program.inputs)
    expected == actual || throw(ArgumentError(
        "@reactive prepare callable returned a program whose HAVE inputs do not " *
        "match the signature ports $(spec.have_names) in order"))
    for name in expose_names
        haskey(program.index, canon_id(spec.graph, spec[name].id)) || throw(ArgumentError(
            "@reactive prepare callable returned a program missing the exposed " *
            "port :$(name)"))
    end
    program
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

# The root symbol at the base of an assignment target chain (or nothing).
_assign_root(x::Symbol) = x
function _assign_root(x)
    x isa Expr || return nothing
    x.head in (:ref, :., :(::)) && return _assign_root(x.args[1])
    if x.head === :tuple                      # destructuring: all roots must agree
        roots = collect(Iterators.filter(!isnothing, (_assign_root(a) for a in x.args)))
        (!isempty(roots) && all(==(first(roots)), roots)) ? first(roots) : nothing
    else
        nothing
    end
end

# The binary operator of a compound-assignment head (:+=, :.*=, …); nothing for
# a plain `=`, a broadcast `.=`, or a non-assignment head.
function _reactive_compound_op(head::Symbol)
    (head === :(=) || head === :(.=)) && return nothing
    s = String(head)
    (endswith(s, "=") && length(s) > 1 && s[end-1] != '=' &&
        !(s in ("!=", "<=", ">=", ".!=", ".<=", ".>="))) || return nothing
    Symbol(s[1:end-1])
end

_reactive_is_dotassign(head::Symbol) =
    head === :(.=) || (startswith(String(head), ".") && endswith(String(head), "=") &&
                       _reactive_compound_op(head) !== nothing)

_reactive_is_assign(head::Symbol) =
    head === :(=) || head === :(.=) || _reactive_compound_op(head) !== nothing

_reactive_is_dotmacro(ex) = ex isa Expr && ex.head === :macrocall &&
    !isempty(ex.args) && ex.args[1] === Symbol("@__dot__")

# Sibling-method call forwarding, shared by rewrite and subst: inject the object
# as the first POSITIONAL arg, keeping any :parameters block in its AST position.
function _reactive_sibling_call(ex, self, rec)
    kwc = nothing; posc = Any[]
    for a in ex.args[2:end]
        if a isa Expr && a.head === :parameters
            kwc = Expr(:parameters, (k isa Expr && k.head === :kw ?
                Expr(:kw, k.args[1], rec(k.args[2])) : rec(k) for k in a.args)...)
        else
            push!(posc, rec(a))
        end
    end
    out = Any[ex.args[1]]; kwc === nothing || push!(out, kwc)
    # Near-verbatim ca9 often passes __self__ explicitly (flip!(__self__, depth));
    # don't inject a second self when the first positional arg is already it.
    (!isempty(posc) && posc[1] === self) || push!(out, self)
    append!(out, posc)
    Expr(:call, out...)
end

# Substitute inside an in-place mutation body: the root field becomes `buffer`,
# other field reads become getproperty, sibling calls forward the object, and the
# assignment STRUCTURE is preserved (we are mutating buffer in place).
function _reactive_subst(ex, fields::Set{Symbol}, methods::Set{Symbol},
                         self::Symbol, shadow::Set{Symbol}, root::Symbol, buffer::Symbol;
                         dot::Bool = false)
    if ex isa Symbol
        ex === root && return buffer
        if ex in fields && !(ex in shadow)
            read = Expr(:call, getproperty, self, QuoteNode(ex))
            # Inside a `@.`, protect the inserted getproperty CALL from being
            # broadcast (`getproperty.(self,:f)`) by escaping it with `$(...)`; the
            # destination `.field` property access is left for @. to handle.
            return dot ? Expr(:$, read) : read
        end
        return ex
    end
    ex isa Expr || return ex
    ex.head === :quote && return ex
    s(n) = _reactive_subst(n, fields, methods, self, shadow, root, buffer; dot = dot)
    if ex.head === :. && length(ex.args) == 2 && ex.args[2] isa QuoteNode
        return Expr(:., s(ex.args[1]), ex.args[2])
    elseif ex.head === :call && ex.args[1] isa Symbol && ex.args[1] in methods
        return _reactive_sibling_call(ex, self, s)
    end
    Expr(ex.head, (s(a) for a in ex.args)...)
end

# If `rhs` is a simple reference INTO a currently-visible field (`field`,
# `field[...]`, `field.x`, or a nested chain of those), return that field root;
# else nothing. Binding a local to such an expression aliases the field's live
# array, so mutating the local must invalidate the field's dependents.
function _reactive_alias_target(rhs, fields::Set{Symbol}, shadow::Set{Symbol})
    r = rhs
    while r isa Expr && r.head in (:ref, :.)
        r = r.args[1]
    end
    (r isa Symbol && r in fields && !(r in shadow)) ? r : nothing
end

# The set of field roots a binding RHS can alias. A plain reference into a field
# gives its single root; a ternary `c ? A : B` (Expr(:if) with 3 args) is sound
# ONLY if every branch aliases the SAME single field — otherwise (distinct roots,
# or alias-vs-nonalias) it yields a conflicted set (>= 2, with the sentinel) so a
# later mutation of the local is rejected at expansion. Empty = not an alias.
function _reactive_alias_roots(rhs, fields::Set{Symbol}, shadow::Set{Symbol})
    if rhs isa Expr && rhs.head === :if && length(rhs.args) == 3
        ra = _reactive_alias_roots(rhs.args[2], fields, shadow)
        rb = _reactive_alias_roots(rhs.args[3], fields, shadow)
        (isempty(ra) && isempty(rb)) && return Set{Symbol}()
        (length(ra) == 1 && ra == rb) && return ra
        return union(ra, rb, Set([_REACTIVE_ALIAS_CONFLICT]))
    elseif rhs isa Expr && rhs.head in (:&&, :||)
        # A short-circuit `c && a` / `c || a` yields the alias on one path and a
        # non-alias (the bool / other operand) on the short-circuit path, so if any
        # operand aliases a field it is an alias-vs-nonalias conflict -> reject.
        rs = Set{Symbol}()
        for op in rhs.args
            union!(rs, _reactive_alias_roots(op, fields, shadow))
        end
        return isempty(rs) ? rs : union(rs, Set([_REACTIVE_ALIAS_CONFLICT]))
    end
    t = _reactive_alias_target(rhs, fields, shadow)
    t === nothing ? Set{Symbol}() : Set([t])
end

_copy_aliases(a::Dict{Symbol,Set{Symbol}}) =
    Dict{Symbol,Set{Symbol}}(k => copy(v) for (k, v) in a)

# The empty symbol marks a control-flow-CONFLICTED alias (root differs across
# paths, or the local is an alias on only some reachable paths). It forces the
# possible-root set to size >= 2 so a later mutation of that local is rejected at
# macro expansion; it is filtered out of the diagnostic.
const _REACTIVE_ALIAS_CONFLICT = Symbol("")

# Does this if/elseif chain end in a plain `else` (so there is no fall-through)?
function _has_final_else(ex)
    (ex isa Expr && ex.head in (:if, :elseif) && length(ex.args) >= 3) || return false
    last = ex.args[3]
    (last isa Expr && last.head === :elseif) ? _has_final_else(last) : true
end

# STRICT control-flow merge: a local stays a sound alias ONLY if every reachable
# path binds it to the SAME single field root. If roots differ, or the local is an
# alias on only some paths (alias-vs-nonalias), it is marked conflicted (a mutation
# then errors at expansion). `paths` are the per-path alias maps (branch results,
# plus the incoming snapshot when a fall-through is reachable).
function _reactive_merge_strict!(dest::Dict{Symbol,Set{Symbol}}, paths)
    paths = collect(paths)
    empty!(dest)
    ks = Set{Symbol}(); for p in paths; union!(ks, keys(p)); end
    for k in ks
        roots = Set{Symbol}(); sound = true
        for p in paths
            if haskey(p, k)
                union!(roots, p[k])
                length(p[k]) == 1 || (sound = false)
            else
                sound = false                       # alias on only some paths
            end
        end
        dest[k] = (sound && length(roots) == 1) ? Set([first(roots)]) :
                  union(roots, Set([_REACTIVE_ALIAS_CONFLICT]))
    end
    dest
end

# Replace every free occurrence of symbol `old` with `new` (used to redirect an
# alias mutation onto a fresh single-assignment local); quoted subtrees are left
# untouched.
function _reactive_replace_symbol(ex, old::Symbol, new)
    ex === old && return new
    ex isa Expr || return ex
    ex.head === :quote && return ex
    Expr(ex.head, (_reactive_replace_symbol(a, old, new) for a in ex.args)...)
end

# Route an alias-local mutation to the field it aliases. A straight-line
# single-field alias (`tree = trees[d]; tree.x = …`) lowers to one assign!-backed
# inplace closure that invalidates exactly that field's dependents. A
# branch/loop-DIVERGENT alias (bound to different fields on different control-flow
# paths) is REJECTED at macro expansion with an actionable error: a compile-time
# over-approximation would assign! every candidate — materializing a non-chosen
# invalid recipe and throwing if a non-chosen candidate is frozen — which is
# unsound, and there is no in-band runtime field identity to dispatch on without a
# Symbol/materialization. ca9's aliases are all straight-line single-field.
function _reactive_alias_dispatch(self, aliasname, roots, body)
    if length(roots) == 1
        # Bind a FRESH single-assignment local to the alias's current value and
        # capture THAT in the assign! closure. The user's alias local may be
        # conditionally rebound (`c && (w = weights[2])`), which — captured in a
        # closure — Julia boxes (Core.Box: Any-typed, allocating). The fresh local
        # is assigned exactly once, so the closure stays concrete/type-stable/0 B.
        fresh = gensym(:alias)
        body2 = _reactive_replace_symbol(body, aliasname, fresh)
        return Expr(:let, Expr(:(=), fresh, aliasname),
                    _reactive_inplace_call(self, first(roots), gensym(:buffer), body2))
    end
    named = sort!(collect(r for r in roots if r != _REACTIVE_ALIAS_CONFLICT))
    throw(ArgumentError(string(
        "@reactive: local `", aliasname, "` aliases a reactive field only on some ",
        "control-flow paths, or different fields on different paths (",
        isempty(named) ? "no consistent field" : join(named, ", "),
        "); a control-flow-divergent alias cannot be soundly invalidated. Bind it ",
        "to one field on every path, or mutate the field directly.")))
end

# `_reactive_inplace!(self, Val(root)) do buffer; body; end`.
_reactive_inplace_call(self, root, buffer, body) =
    Expr(:call, _reactive_inplace!, Expr(:(->), buffer, Expr(:block, body)),
         self, Expr(:call, Val, QuoteNode(root)))

# Lower an in-place root-field mutation through the assign!-backed
# `_reactive_inplace!`. For a plain `=` or a (non-dotted) compound assignment the
# RHS is STAGED into a temp before the call, so a side-effecting RHS (e.g. one
# that mutates an upstream field of the root) is evaluated before the buffer is
# materialized — Julia's exact RHS-before-receiver order. Dotted/broadcast and
# destructuring mutations use the closure form (root -> buffer) directly, which
# is correct for pure RHS and avoids materializing a broadcast result.
function _reactive_inplace(ex, fields, methods, self, shadow, root::Symbol,
                           aliases::Dict{Symbol,Set{Symbol}} = Dict{Symbol,Set{Symbol}}())
    buffer = gensym(:buffer)
    lhs = ex.args[1]; rhs = ex.args[2]
    if ex.head === :(=)
        # Plain `=`: Julia evaluates RHS before the receiver/index
        # (`v[idx()] = rhs()` traces rhs, idx, set), so stage the RHS first.
        tmp = gensym(:rhs)
        rhs_staged = _reactive_rewrite(rhs, fields, methods, self, shadow, aliases)
        lhs_b = _reactive_subst(lhs, fields, methods, self, shadow, root, buffer)
        return Expr(:block,
            Expr(:(=), tmp, rhs_staged),
            _reactive_inplace_call(self, root, buffer, Expr(:(=), lhs_b, tmp)),
            tmp)   # a Julia assignment returns the assigned value
    end
    # Compound (`v[idx()] += rhs()` traces idx, get, rhs, set) and dotted/
    # broadcast (`.=` / `.op=`): the assign!-buffer closure naturally preserves
    # Julia's evaluation order, so substitute root -> buffer and run in place.
    body = _reactive_subst(ex, fields, methods, self, shadow, root, buffer)
    _reactive_inplace_call(self, root, buffer, body)
end

# Rewrite a method body: field reads -> getproperty, whole-field writes ->
# setproperty!, scalar compound writes -> setproperty!(read op rhs), rooted /
# dotted / destructuring / @. mutations -> in-place assign!, and sibling calls
# forwarded the object. `shadow` holds locally-bound names.
function _reactive_rewrite(ex, fields::Set{Symbol}, methods::Set{Symbol},
                           self::Symbol, shadow::Set{Symbol},
                           aliases::Dict{Symbol,Set{Symbol}} = Dict{Symbol,Set{Symbol}}())
    if ex isa Symbol
        (ex in fields && !(ex in shadow)) &&
            return Expr(:call, getproperty, self, QuoteNode(ex))
        return ex
    end
    ex isa Expr || return ex
    ex.head === :quote && return ex
    rw(node) = _reactive_rewrite(node, fields, methods, self, shadow, aliases)

    if _reactive_is_dotmacro(ex)
        inner = ex.args[end]
        if inner isa Expr && _reactive_is_assign(inner.head)
            root = _assign_root(inner.args[1])
            if root !== nothing && root in fields && !(root in shadow)
                # dot forms read the destination first, so the buffer closure
                # (root -> buffer) preserves order; run @. over the buffer.
                buffer = gensym(:buffer)
                subst = _reactive_subst(inner, fields, methods, self, shadow, root, buffer; dot = true)
                dotted = Expr(:macrocall, ex.args[1], ex.args[2], subst)
                return _reactive_inplace_call(self, root, buffer, dotted)
            elseif root isa Symbol && haskey(aliases, root)
                # local alias root inside @. (`tr = trees[d]; @. tr.mom = -bwd.mom`):
                # keep the alias, rewrite other fields to getproperty, and wrap the
                # WHOLE dotted mutation in the assign!-backed closure (ignoring the
                # buffer) so assign!'s finally invalidates the aliased field.
                subst = _reactive_subst(inner, fields, methods, self, shadow,
                                        gensym(:__noroot__), gensym(:__nobuf__); dot = true)
                dotted = Expr(:macrocall, ex.args[1], ex.args[2], subst)
                return _reactive_alias_dispatch(self, root, aliases[root], dotted)
            end
        end
        return Expr(:macrocall, ex.args[1], ex.args[2],
                    (rw(a) for a in ex.args[3:end])...)
    elseif _reactive_is_assign(ex.head)
        lhs = ex.args[1]; rhs = ex.args[2]
        root = _assign_root(lhs)
        if root === nothing || !(root in fields) || root in shadow
            # Mutation THROUGH a local alias of a field (`w = trees[d]; w[..] = …`):
            # run the mutation inside the assign!-backed inplace closure — the
            # closure ignores the buffer and uses the alias, so assign!'s finally
            # invalidates the aliased field's dependents on success AND on a
            # throwing partial write, without reaching any guarded internals.
            if lhs !== root && root isa Symbol && haskey(aliases, root)
                mut = Expr(ex.head, lhs,
                           _reactive_rewrite(rhs, fields, methods, self, shadow, aliases))
                return _reactive_alias_dispatch(self, root, aliases[root], mut)
            end
            # (Re)binding a local: record a field alias if the RHS references into a
            # field, else clear any prior alias for that name.
            if lhs isa Symbol
                roots = _reactive_alias_roots(rhs, fields, shadow)
                isempty(roots) ? delete!(aliases, lhs) : (aliases[lhs] = roots)
            end
            local_names = Set{Symbol}(); _reactive_assign_names!(local_names, lhs)
            inner = union(shadow, local_names)
            return Expr(ex.head, lhs,
                        _reactive_rewrite(rhs, fields, methods, self, inner, aliases))
        end
        op = _reactive_compound_op(ex.head)
        if lhs === root && ex.head === :(=)
            return Expr(:call, setproperty!, self, QuoteNode(root), rw(rhs))
        elseif lhs === root && op !== nothing && !_reactive_is_dotassign(ex.head)
            return Expr(:call, setproperty!, self, QuoteNode(root),
                        Expr(:call, op, Expr(:call, getproperty, self, QuoteNode(root)),
                             rw(rhs)))
        else
            return _reactive_inplace(ex, fields, methods, self, shadow, root, aliases)
        end
    elseif ex.head === :call && ex.args[1] isa Symbol && ex.args[1] in methods
        return _reactive_sibling_call(ex, self, rw)
    elseif ex.head === :. && length(ex.args) == 2 && ex.args[2] isa QuoteNode
        return Expr(:., rw(ex.args[1]), ex.args[2])
    elseif ex.head in (:if, :elseif)
        # Rewrite the condition, then each branch on its OWN copy of the alias map,
        # and STRICT-merge the reachable paths: a local stays an alias only if every
        # reachable path binds it to the SAME single field; otherwise a later
        # mutation of it errors at expansion. The incoming snapshot is a reachable
        # path only when there is no final `else` (fall-through).
        cond = rw(ex.args[1])
        snapshot = _copy_aliases(aliases)
        out = Any[cond]; branches = Dict{Symbol,Set{Symbol}}[]
        for br in ex.args[2:end]
            bd = _copy_aliases(snapshot)
            push!(out, _reactive_rewrite(br, fields, methods, self, shadow, bd))
            push!(branches, bd)
        end
        paths = _has_final_else(ex) ? branches : push!(copy(branches), snapshot)
        _reactive_merge_strict!(aliases, paths)
        return Expr(ex.head, out...)
    elseif ex.head in (:&&, :||)
        # Short-circuit `cond && rhs` / `cond || rhs`: the condition always runs,
        # the RHS runs only conditionally. Strict-merge the short-circuit path (RHS
        # not evaluated -> incoming snapshot) with the executed path (RHS result),
        # so an alias (re)bound only inside a short-circuit RHS becomes conflicted.
        cond = rw(ex.args[1])
        snapshot = _copy_aliases(aliases); rhs_scope = _copy_aliases(aliases)
        rhs_out = _reactive_rewrite(ex.args[2], fields, methods, self, shadow, rhs_scope)
        _reactive_merge_strict!(aliases, (snapshot, rhs_scope))
        return Expr(ex.head, cond, rhs_out)
    elseif ex.head in (:for, :while)
        bind = Set{Symbol}()
        ex.head === :for && _reactive_assign_names!(bind, ex.args[1] isa Expr &&
            ex.args[1].head === :(=) ? ex.args[1].args[1] : ex.args[1])
        inner = union(shadow, bind)
        # A loop runs 0+ times, so after it a local may hold its incoming alias
        # (0 iterations) OR a loop-body binding (>=1). Rewrite the body on a copy,
        # then merge incoming + body-scope (union) back into the outer map.
        incoming = _copy_aliases(aliases); scope = _copy_aliases(aliases)
        body = Expr(ex.head, (_reactive_rewrite(a, fields, methods, self, inner, scope)
                              for a in ex.args)...)
        # Both zero iterations (incoming) and >=1 (body result) are reachable;
        # strict-merge rejects a local whose alias changed across them.
        _reactive_merge_strict!(aliases, (incoming, scope))
        return body
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

function _reactive_expand(def; prepare = nothing, specialize = false)
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
    # Per-construction specialization (poc-approved opt-in): the HAVE ports are
    # typed from the RUNTIME arguments (`typeof(arg)`) so the same authored object
    # yields a distinct concrete program per concrete signature (Float32/Float64,
    # Matrix/Diagonal, distinct closure types) with no Value{Any}. The default mode
    # types ports from the declared annotations and caches one program in a const.
    spec_inputs = specialize === true ?
        Tuple{Symbol,Any}[(p, :(typeof($p))) for (p, _) in inputs] : inputs
    spec_build = _kernel_expand(kernel_body, spec_inputs)
    want_tuple = Expr(:tuple, (QuoteNode(f) for f in field_names)...)
    expose_tuple = Expr(:tuple, (QuoteNode(n) for n in vcat(port_names, field_names))...)

    # Build everything with plain symbols and `esc` the whole output once so the
    # generated `__self__`, user param names, and user calls share one hygiene
    # context; runtime helpers are interpolated as VALUES (immune to esc).
    spec_sym = gensym(:spec); prog_sym = gensym(:program); handles_sym = gensym(:handles)
    # Preserve the original positional/keyword TYPE annotations so the generated
    # constructor keeps its dispatch (an untyped port stays untyped).
    typed(p, t) = t === nothing ? p : Expr(:(::), p, t)
    # Julia AST: the :parameters (keyword) block comes right after the callee.
    ctor_sig_args = Any[name]
    if !isempty(keywords)
        kw_params = Any[]
        for (p, t, hasdefault, default) in keywords
            tp = typed(p, t)
            push!(kw_params, hasdefault ? Expr(:kw, tp, default) : tp)
        end
        push!(ctor_sig_args, Expr(:parameters, kw_params...))
    end
    for (p, t) in positional; push!(ctor_sig_args, typed(p, t)); end
    # Cache (program, handles) once per definition in a UNIQUE per-expansion const
    # so independent instances share the compiled program (copyto!-compatible),
    # setup runs once, and the constructor stays type-stable (the const is a typed
    # tuple, not a Symbol=>Any dict). The gensym'd name cannot collide across
    # definitions or modules.
    cache_name = gensym(Symbol(name, :__program))
    # Plain default (no prepare=, no specialize): byte-for-byte the original
    # expansion. Otherwise (custom prepare OR specialized construction) validate the
    # built program isa ReactiveProgram AND matches the authored graph/inputs/
    # exposure — so a misbehaving callable fails with an actionable error rather
    # than silently dropping fields.
    built_expr = (prepare === nothing && specialize !== true) ?
        :($prepare_reactive($spec_sym; want = $want_tuple)) :
        quote
            let program = $(prepare === nothing ? prepare_reactive : prepare)(
                    $spec_sym; want = $want_tuple)
                program isa $ReactiveProgram || throw(ArgumentError(
                    "@reactive prepare callable must return a ReactiveProgram; got " *
                    string(typeof(program))))
                $_reactive_validate_program(program, $spec_sym, $expose_tuple)
            end
        end
    if specialize === true
        # Move spec-build + prepare into the constructor: each call builds a
        # runtime-typed program and its handles. The constructor's inference may be
        # unstable, but the returned ReactiveObject is concrete-at-runtime, so a
        # function barrier / state-parametric caller gets an inferred, 0-B hot loop.
        cache_def = nothing
        ctor_body = quote
            $spec_sym = $spec_build
            $prog_sym = $built_expr
            $handles_sym = $_reactive_handles($prog_sym, $spec_sym, $expose_tuple)
            $(ReactiveObject){$(QuoteNode(name))}(
                $prog_sym($(port_names...)), $handles_sym)
        end
    else
        # Default: cache (program, handles) once per definition in a UNIQUE
        # per-expansion const — byte-for-byte the original expansion.
        cache_def = Expr(:const, Expr(:(=), cache_name, quote
            let
                $spec_sym = $spec_build
                built = $built_expr
                (built, $_reactive_handles(built, $spec_sym, $expose_tuple))
            end
        end))
        ctor_body = quote
            ($prog_sym, $handles_sym) = $cache_name
            $(ReactiveObject){$(QuoteNode(name))}(
                $prog_sym($(port_names...)), $handles_sym)
        end
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

    top = Any[]
    cache_def === nothing || push!(top, cache_def)
    push!(top, constructor)
    append!(top, method_code)
    push!(top, name)
    esc(Expr(:block, top...))
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

Options (before the definition):
- `@reactive prepare=<callable> name(...) = ...` — the injected `<callable>`
  receives the `KernelSpec` and a `want` keyword and must return a
  [`ReactiveProgram`](@ref) built from that spec (e.g. one selecting the
  non-allocating in-place preparation). The macro still owns spec/want/handles and
  validates the returned program's graph/inputs/exposure.
- `@reactive specialize=true name(...) = ...` — build a fresh runtime-typed program
  in the CONSTRUCTOR (not a per-definition const), typing every otherwise-untyped
  HAVE port from `typeof(runtime_value)` so the object is concrete-at-runtime and
  generic over argument precision/storage. The default (`specialize=false`) caches
  one program per definition in a const and is unchanged.
"""
macro reactive(args...)
    isempty(args) && throw(ArgumentError("@reactive requires a definition"))
    def = last(args)
    # Additive, non-HMC option: `@reactive prepare=<callable> name(...) = ...`.
    # The macro still owns the spec/want/handles; the injected callable receives
    # the KernelSpec and a `want` keyword and must return a ReactiveProgram (e.g.
    # one selecting the non-allocating cache_apply/is_mutating preparation). With
    # no option the expansion is byte-for-byte identical to before.
    prepare = nothing
    specialize = false
    seen = Set{Symbol}()
    for option in args[1:(end - 1)]
        (option isa Expr && option.head === :(=) &&
         option.args[1] isa Symbol) || throw(ArgumentError(
            "@reactive options must be `key=value`; got $(repr(option))"))
        key = option.args[1]
        key in (:prepare, :specialize) || throw(ArgumentError(
            "@reactive supports options `prepare=<callable>` and " *
            "`specialize=<bool>`; got `$(key)=`"))
        key in seen && throw(ArgumentError("@reactive got a duplicate `$(key)=` option"))
        push!(seen, key)
        key === :prepare && (prepare = option.args[2])
        if key === :specialize
            (option.args[2] isa Bool) || throw(ArgumentError(
                "@reactive `specialize=` requires a literal Bool (true/false); got " *
                repr(option.args[2])))
            specialize = option.args[2]
        end
    end
    _reactive_expand(def; prepare = prepare, specialize = specialize)
end
