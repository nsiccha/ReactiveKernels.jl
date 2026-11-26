# An independent indexed store whose branches select on the sign of a prepared
# entry. Build one scalar program per authored arm, then map each over precisely
# the entries that select it. The same emitted module accepts arbitrary counts.
_jax_map_statements(x) = x isa Expr && x.head === :block ?
    [a for a in x.args if !(a isa LineNumberNode)] : Any[x]
_jax_map_assignment(x) = Meta.isexpr(x, :(=), 2)
# Resolve the authored comparison through the normal scalar emitter. A wrapper
# can implement Base's comparison too. Only a single pure comparison of the
# entry against zero qualifies, including the semantics of unordered values.
function _jax_map_sign(ctx, condition, op, entry, T)
    Meta.isexpr(condition, :call) && length(condition.args) >= 3 &&
        condition.args[end-1] === entry && condition.args[end] isa Real &&
        iszero(condition.args[end]) || return false
    scope = _JaxScope(ctx.scope)
    inner = _JaxContext(ctx.mod, scope, ctx.recipe, ctx.source_module, copy(ctx.locals), ctx.captures)
    inner.traced_branches = ctx.traced_branches
    variable = _jax_identifier(scope.used, :map_entry)
    inner.locals[entry] = _PyVal(variable, T)
    before = _JAX_EFFECTS[]
    result = _jax_expr(inner, condition)
    isempty(scope.lines) && _JAX_EFFECTS[] == before &&
        result.code in (variable * " " * string(op) * " 0", variable * " " * string(op) * " 0.0")
end

# Numerical straight-line expressions only. Arrays other than the one
# selected entry, writes in a call's arguments and nested control flow are
# outside this independence proof.
function _jax_map_scalar_expr(ctx, x, locals)
    x isa Symbol && return x in locals || haskey(ctx.locals, x) && _jax_is_scalar(ctx.locals[x].T)
    x isa Number && return true
    x isa QuoteNode && return x.value isa Union{Number,Symbol,Type}
    x isa GlobalRef && return isdefined(x.mod, x.name) &&
        getfield(x.mod, x.name) isa Union{Number,Function,Type}
    Meta.isexpr(x, :call) || return false
    # Typed singleton constructors occur in dispatch wrappers for arithmetic.
    # A closure or an indexed callee can capture a mutable array.
    head = first(x.args)
    f = if Meta.isexpr(head, :., 2) && head.args[1] isa Module && head.args[2] isa QuoteNode
        mod, name = head.args[1], head.args[2].value
        name isa Symbol && isdefined(mod, name) ? getglobal(mod, name) : nothing
    else
        _jax_call_callee(ctx, head)
    end
    (f isa Union{Function,Type} || length(x.args) == 1 && Meta.isexpr(head, :curly)) || return false
    all(a -> _jax_map_scalar_expr(ctx, a, locals), x.args[2:end])
end

function _jax_map_bindings(ctx, statements, names)
    for st in statements
        _jax_map_assignment(st) || return false
        lhs, rhs = st.args
        lhs isa Symbol && !(lhs in names) && !haskey(ctx.locals, lhs) || return false
        _jax_map_scalar_expr(ctx, rhs, names) || return false
        push!(names, lhs)
    end
    true
end

function _jax_signed_map(ctx, i, body, lo, hi)
    statements = _jax_map_statements(body)
    length(statements) >= 2 && _jax_map_assignment(first(statements)) || return nothing
    entry, read = first(statements).args
    entry isa Symbol && !haskey(ctx.locals, entry) && Meta.isexpr(read, :ref, 2) &&
        read.args[2] === i || return nothing
    source = get(ctx.locals, read.args[1], nothing)
    source === nothing && return nothing
    _jax_lane_number_vector(source) || return nothing
    key = get(ctx.scope.prepared_keys, source.code, nothing)
    key === nothing && return nothing
    branch = last(statements)
    Meta.isexpr(branch, :if, 3) && _jax_map_sign(ctx, branch.args[1], :>, entry, eltype(source.T)) || return nothing
    negative = branch.args[3]
    Meta.isexpr(negative, :elseif) || return nothing
    conditions = _jax_map_statements(negative.args[1])
    length(conditions) == 1 && _jax_map_sign(ctx, only(conditions), :<, entry, eltype(source.T)) || return nothing
    length(negative.args) == 2 || return nothing
    prefix = statements[2:end-1]
    names = Set([entry])
    _jax_map_bindings(ctx, prefix, names) || return nothing
    arms = _jax_map_statements.((branch.args[2], negative.args[2]))
    out = nothing
    for arm in arms
        isempty(arm) && return nothing
        allowed = copy(names)
        _jax_map_bindings(ctx, arm[1:end-1], allowed) || return nothing
        st = arm[end]
        _jax_map_assignment(st) || return nothing
        lhs, rhs = st.args
        Meta.isexpr(lhs, :ref, 2) && lhs.args[2] === i && lhs.args[1] isa Symbol &&
            _jax_map_scalar_expr(ctx, rhs, allowed) || return nothing
        out === nothing || lhs.args[1] === out || return nothing
        out = lhs.args[1]
    end
    original = get(ctx.locals, out, nothing)
    original === nothing && return nothing
    _jax_lane_number_vector(original) && !_jax_slots(original) || return nothing
    outer = ctx.scope
    saved = copy(ctx.locals)
    functions = Pair{String,Vector{String}}[]
    before = _JAX_EFFECTS[]
    try
        for arm in arms
            f = _jax_fresh(ctx, :mapped_arm)
            inner = _JaxScope(outer)
            inner.indent = outer.indent + 2
            item = _jax_identifier(inner.used, entry)
            ctx.scope = inner
            ctx.locals = copy(saved)
            ctx.locals[entry] = _PyVal(item, eltype(source.T))
            _jax_statements(ctx, prefix; value=false)
            # The default arm leaves the old slot intact. Dropping its
            # unused prefix is safe only for total scalar arithmetic.
            all(line -> occursin(r"^ *[A-Za-z_][A-Za-z_0-9]* = [A-Za-z_0-9. ()+*\-]+$", line) &&
                        !occursin(r"[A-Za-z_0-9] *\(|\*\*", line), inner.lines) || return nothing
            _jax_statements(ctx, arm[1:end-1]; value=false)
            v = _jax_expr(ctx, arm[end].args[2])
            _jax_is_scalar(v.T) || return nothing
            stored = _jax_stored(ctx, eltype(original.T), v)
            _jax_emit!(ctx, "return " * stored.code)
            _JAX_EFFECTS[] == before || return nothing
            push!(functions, f => ["    "^(outer.indent + 1) * "def " * f * "(" * item * "):"; inner.lines])
            ctx.scope = outer
        end
    finally
        ctx.scope = outer
        ctx.locals = saved
        _JAX_EFFECTS[] = before
    end
    schedule = _jax_prepared_bound!(ctx, :sign_groups, key)
    result = _jax_fresh(ctx, :mapped)
    _jax_emit!(ctx, "if " * hi.code * " == " * source.code * ".shape[0] == " * original.code * ".shape[0]:")
    outer.indent += 1
    parts = String[]
    for (j, (f, lines)) in enumerate(functions)
        append!(outer.lines, lines)
        indices = schedule * "[" * string(j-1) * "]"
        # An absent arm does not even trace its body.
        part = _jax_fresh(ctx, :mapped_part)
        _jax_emit!(ctx, part * " = jax.vmap(" * f * ")(" * source.code * "[" * indices *
                       "]) if " * indices * ".shape[0] else " * original.code * "[:0]")
        push!(parts, part)
    end
    push!(parts, original.code * "[" * schedule * "[2]]")
    _jax_emit!(ctx, result * " = jnp.concatenate((" * join(parts, ", ") * "))[" * schedule * "[3]]")
    outer.indent -= 1
    # The authored range can cover a prefix or exceed either vector. Keep
    # the usual loop, with its original bounds and index semantics, then.
    _jax_emit!(ctx, "else:")
    outer.indent += 1
    try
        _jax_sequential_range(ctx, i, body, lo, hi)
        _jax_emit!(ctx, result * " = " * ctx.locals[out].code)
    finally
        outer.indent -= 1
    end
    ctx.locals[out] = _PyVal(result, original.T)
    _PyVal("None", Nothing, true)
end
