# An independent indexed store whose branches select on the sign of a prepared
# entry. Build one scalar program per authored arm, then map each over precisely
# the entries that select it. The same emitted module accepts arbitrary counts.
_jax_map_statements(x) = x isa Expr && x.head === :block ?
    [a for a in x.args if !(a isa LineNumberNode)] : Any[x]
_jax_map_assignment(x) = Meta.isexpr(x, :(=), 2)

# A local may own an array across branches and iterations. Verify its whole
# lexical lifetime: the constructor binds it once, indexed writes keep it
# local, and only the function result can publish it. A reshape, closure,
# alias, argument to another call or rebinding makes the proof fail.
function _jax_private_array_uses(ex, name, constructor; top=true, tail=true)
    !_kernel_mentions(ex, name) && return true
    ex === name && return tail
    ex isa Expr || return false
    if ex.head === :block
        statements = _jax_map_statements(ex)
        return all(enumerate(statements)) do (i, st)
            _jax_private_array_uses(st, name, constructor; top, tail=tail && i == length(statements))
        end
    elseif ex.head === :return
        return ex.args == [name]
    elseif ex.head === :for || ex.head === :if
        return !_kernel_mentions(first(ex.args), name) &&
            all(st -> _jax_private_array_uses(st, name, constructor; top=false, tail=false),
                ex.args[2:end])
    elseif _jax_map_assignment(ex)
        lhs, rhs = ex.args
        lhs === name && return top && rhs === constructor && !_kernel_mentions(rhs, name)
        return Meta.isexpr(lhs, :ref) && first(lhs.args) === name &&
            !any(i -> _kernel_mentions(i, name), lhs.args[2:end]) && !_kernel_mentions(rhs, name)
    end
    false
end

# These Base methods allocate new storage. Check the resolved method and
# operand types, including through a Julia helper whose entire lowered body
# forwards its arguments to such a constructor.
function _jax_dense_constructor(f, types)
    (f === fill || f === zeros || f === ones) || return false
    dims = f === fill ? types[2:end] : types
    length(dims) in (1, 2) && all(==(Int), dims) || return false
    f === fill && !(first(types) <: _JaxBranchReal && isprimitivetype(first(types))) && return false
    which(f, Tuple{types...}).module === Base
end

function _jax_record_private_array!(ctx, name, constructor, f, values)
    frame = _jax_frame()
    frame !== nothing && frame.function_body && ctx.scope === frame.scope &&
        frame.lets == 0 && !(name in frame.names) && !haskey(ctx.locals, name) &&
        _jax_private_array_uses(frame.source, name, constructor) || return
    types = _jax_argument_type.(values)
    all(t -> t isa Type && isconcretetype(t), types) || return
    if !_jax_dense_constructor(f, types)
        f isa Function && Base.issingletontype(typeof(f)) || return
        infos = try
            _jax_lowered_code(f, types)
        catch
            return
        end
        length(infos) == 1 || return
        code = only(infos).code
        length(code) == 2 || return
        call, ret = code
        ret isa Core.ReturnNode && isdefined(ret, :val) && ret.val === Core.SSAValue(1) &&
            Meta.isexpr(call, :call) || return
        callee = first(call.args)
        callee isa GlobalRef && isdefined(callee.mod, callee.name) && isconst(callee.mod, callee.name) || return
        operands = call.args[2:end]
        all(x -> x isa Core.SlotNumber && 2 <= x.id <= length(types) + 1, operands) || return
        _jax_dense_constructor(getglobal(callee.mod, callee.name),
                               [types[x.id - 1] for x in operands]) || return
    end
    push!(frame.private_arrays, name)
end

function _jax_plain_copy_array(ctx, value)
    value !== nothing && value.T isa Type && value.T <: Array &&
        eltype(value.T) <: _JaxBranchReal &&
        isempty(value.extent) && !_jax_slots(value) && !(value.origin isa _JaxRowView ||
        value.origin isa _JaxRowShape && (value.origin.stored || _jax_borrowed_buffer(ctx, value)))
end

# A read-only reshape of a dense array still reads later writes to its
# parent. The sequential loop must reshape the current carry for each read,
# even though the alias's shape was established before the loop.
function _jax_readonly_reshape_use(ex, alias, parent)
    ex isa Expr || return ex !== alias
    if ex.head in (:let, :function, :->, :macrocall)
        return !_kernel_mentions(ex, alias) && !_kernel_mentions(ex, parent)
    elseif Meta.isexpr(ex, :ref) && first(ex.args) === alias
        return all(a -> !_kernel_mentions(a, alias), ex.args[2:end])
    elseif (_jax_map_assignment(ex) || haskey(_JAX_UPDATING, ex.head)) &&
            _kernel_mentions(first(ex.args), alias)
        return false
    end
    all(a -> _jax_readonly_reshape_use(a, alias, parent), ex.args)
end

function _jax_record_matrix_reshape!(ctx, alias, ex, parent)
    frame = _jax_frame()
    frame !== nothing && ctx.scope === frame.scope && !haskey(ctx.locals, alias) || return
    array = get(ctx.locals, parent, nothing)
    _jax_plain_copy_array(ctx, array) && isprimitivetype(eltype(array.T)) || return
    statements = _jax_map_statements(frame.source)
    at = findfirst(st -> _jax_map_assignment(st) && st.args[1] === alias && st.args[2] === ex, statements)
    at === nothing && return
    rebound, written = Set{Symbol}(), Set{Symbol}()
    for i in at+1:length(statements)
        st = statements[i]
        _jax_readonly_reshape_use(st, alias, parent) || return
        _jax_write_targets!(rebound, written, st)
    end
    parent in written && !(parent in rebound) && (frame.matrix_reshape_sources[alias] = parent)
end

function _jax_read_reshaped_matrix(ctx, name, value)
    frame = _jax_frame()
    frame === nothing && return value
    parent = get(frame.matrix_reshape_sources, name, nothing)
    parent === nothing && return value
    _PyVal("jnp.reshape(" * ctx.locals[parent].code * ", " * value.code *
           ".shape, order=\"F\")", value.T)
end

function _jax_check_copy_index!(ctx, index, first, last, message)
    interval = _jax_static_integer_bounds(index)
    if interval !== nothing
        lo, hi = interval
        _jax_emit!(ctx, "if not (" * first * " <= " * lo * " and " * hi * " <= " * last * "):")
        ctx.scope.indent += 1
    end
    valid = _PyVal("((" * index.code * " >= " * first * ") & (" * index.code * " <= " * last * "))", Bool)
    _jax_masked_check(ctx, valid, repr(message))
    interval === nothing || (ctx.scope.indent -= 1)
end

# Recover a bound for the offset using the same primitive integer operations
# that the scalar emitter produced. Other expressions stay inside the loop.
function _jax_copy_integer(ctx, ex, operands)
    ex isa Int && return _jax_literal(ex)
    ex isa Symbol && return get(operands, ex, nothing)
    Meta.isexpr(ex, :call) && first(ex.args) in (:+, :-, :*) || return nothing
    args = map(x -> _jax_copy_integer(ctx, x, operands), ex.args[2:end])
    any(isnothing, args) && return nothing
    _jax_call(ctx, getfield(Base, first(ex.args)), _PyVal[args...])
end

function _jax_copy_offset(ctx, index, var)
    Meta.isexpr(index, :call) && last(index.args) === var &&
        _jax_map_scalar_expr(ctx, index, Set([var])) || return nothing
    # Translate through the ordinary scalar dispatch, including wrappers.
    # Only a pure sum ending in the induction variable is a contiguous copy.
    scope = _JaxScope(ctx.scope)
    inner = _JaxContext(ctx.mod, scope, ctx.recipe, ctx.source_module, copy(ctx.locals), ctx.captures)
    inner.traced_branches = ctx.traced_branches
    item = _jax_identifier(scope.used, :copy_index)
    inner.locals[var] = _PyVal(item, Int)
    before = _JAX_EFFECTS[]
    result = _jax_expr(inner, index)
    result.T === Int && isempty(scope.lines) && _JAX_EFFECTS[] == before || return nothing
    parsed = Meta.parse(result.code; raise=false)
    Meta.isexpr(parsed, :call) && length(parsed.args) >= 3 &&
        first(parsed.args) === :+ && last(parsed.args) === Symbol(item) || return nothing
    terms = parsed.args[2:end-1]
    operands = Dict(Symbol(v.code) => v for v in values(ctx.locals) if v.T === Int)
    _jax_copy_integer(ctx, length(terms) == 1 ? only(terms) : Expr(:call, :+, terms...), operands)
end

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

# Copy an aligned prefix, fill a diagonal, or write a contiguous strip into
# a privately allocated array. Match the store before emitting any code.
function _jax_prefix_store(ctx, var, body, hi)
    hi.T isa Type && hi.T <: Integer && hi.T !== Bool || return nothing
    vars = Symbol[var]
    stops = String[hi.code]
    shape_width = false
    statements = _jax_map_statements(body)
    length(statements) == 1 || return nothing
    store = only(statements)
    if Meta.isexpr(store, :for, 2)
        binding, inner = store.args
        Meta.isexpr(binding, :(=), 2) || return nothing
        index, range = binding.args
        index isa Symbol && index !== var && Meta.isexpr(range, :call, 3) &&
            range.args[1] === :(:) && range.args[2] === 1 || return nothing
        stop = range.args[3]
        stop === var && return nothing
        width = stop isa Int ? _jax_literal(stop) : get(ctx.locals, stop, nothing)
        width !== nothing && width.static && width.T === Int || return nothing
        shape_width = !(stop isa Int)
        push!(vars, index)
        push!(stops, width.code)
        statements = _jax_map_statements(inner)
        length(statements) == 1 || return nothing
        store = only(statements)
    end
    _jax_map_assignment(store) || return nothing
    lhs, rhs = store.args
    Meta.isexpr(lhs, :ref) && first(lhs.args) isa Symbol || return nothing
    name, indices = lhs.args[1], lhs.args[2:end]
    dest = get(ctx.locals, name, nothing)
    _jax_plain_copy_array(ctx, dest) && ndims(dest.T) == length(indices) || return nothing
    frame = _jax_frame()
    private = frame !== nothing && name in frame.private_arrays &&
        hi.T === Int && isprimitivetype(eltype(dest.T))
    shape_width && !private && return nothing
    row = nothing
    offset = nothing
    diagonal = length(vars) == 1 && indices == [var, var] &&
        rhs isa Number && typeof(rhs) === eltype(dest.T)
    if diagonal
        axes = [0, 1]
        bounds = [hi.code * " <= " * dest.code * ".shape[" * string(ax) * "]" for ax in axes]
        positions = "jnp.arange(" * hi.code * ")"
        selection = positions * ", " * positions
        value = _jax_literal(rhs).code
    else
        Meta.isexpr(rhs, :ref) && rhs.args[2:end] == vars || return nothing
        source = first(rhs.args)
        source isa Symbol && !(source in vars) || return nothing
        # Reads through this reshape must see writes to its parent in order.
        frame !== nothing && haskey(frame.matrix_reshape_sources, source) && return nothing
        src = get(ctx.locals, source, nothing)
        _jax_plain_copy_array(ctx, src) && ndims(src.T) == length(vars) &&
            eltype(src.T) === eltype(dest.T) || return nothing
        if indices == vars
            bounds = [stop * " <= " * v.code * ".shape[" * string(ax - 1) * "]"
                      for (ax, stop) in enumerate(stops) for v in (dest, src)]
        elseif private && length(vars) == length(indices) == 1 && source !== name
            offset = _jax_copy_offset(ctx, only(indices), var)
            offset === nothing && return nothing
            bounds = [hi.code * " <= " * v.code * ".shape[0]" for v in (src, dest)]
        else
            private && length(vars) == 1 && length(indices) == 2 &&
                indices[2] === var && indices[1] !== var || return nothing
            at = first(indices)
            row = at isa Int ? _jax_literal(at) : get(ctx.locals, at, nothing)
            row !== nothing && row.T === Int || return nothing
            bounds = [hi.code * " <= " * dest.code * ".shape[1]",
                      hi.code * " <= " * src.code * ".shape[0]", dest.code * ".shape[0] > 0"]
        end
        selection = join((":" * stop for stop in stops), ", ")
        value = src.code * "[" * selection * "]"
    end

    # Python resolves shape tests at trace time. On an inactive Julia branch
    # an impossible range must only raise when that branch is selected.
    result = _jax_fresh(ctx, :block)
    _jax_emit!(ctx, "if " * join((stop * " <= 0" for stop in stops), " or ") * ":")
    _jax_emit!(ctx, "    " * result * " = " * dest.code)
    if !diagonal && !private && length(vars) > 1
        # Reshapes can share storage with a different leading dimension. Such
        # a copy can read a preceding store as the authored rows advance.
        _jax_emit!(ctx, "elif " * dest.code * ".shape[0] != " * src.code * ".shape[0] and " * join(bounds, " and ") * ":")
        ctx.scope.indent += 1
        _jax_sequential_range(ctx, var, body, _jax_literal(1), hi)
        _jax_emit!(ctx, result * " = " * ctx.locals[name].code)
        ctx.scope.indent -= 1
    end
    _jax_emit!(ctx, "elif " * join(bounds, " and ") * ":")
    if offset !== nothing
        ctx.scope.indent += 1
        _jax_check_copy_index!(ctx, offset, "0", dest.code * ".shape[0] - " * hi.code,
                               "BoundsError: block exceeds array dimensions")
        _jax_emit!(ctx, result * " = jax.lax.dynamic_update_slice(" * dest.code * ", " * value *
            ", (" * offset.code * ",))")
        ctx.scope.indent -= 1
    elseif row === nothing
        _jax_emit!(ctx, "    " * result * " = " * dest.code * ".at[" * selection * "].set(" * value * ")")
    else
        ctx.scope.indent += 1
        _jax_check_copy_index!(ctx, row, "1", dest.code * ".shape[0]", "BoundsError: row exceeds array dimensions")
        offset = _jax_minus_one(row)
        _jax_emit!(ctx, result * " = jax.lax.dynamic_update_slice(" * dest.code * ", " *
            src.code * "[None, :" * hi.code * "], (" * offset * ", jnp.zeros_like(" * offset * ")))")
        ctx.scope.indent -= 1
    end
    _jax_emit!(ctx, "else:")
    ctx.scope.indent += 1
    _jax_masked_check(ctx, _PyVal("False", Bool), repr("BoundsError: block exceeds array dimensions"))
    _jax_emit!(ctx, result * " = " * dest.code)
    ctx.scope.indent -= 1
    ctx.locals[name] = _PyVal(result, dest.T)
    _PyVal("None", Nothing, true)
end
