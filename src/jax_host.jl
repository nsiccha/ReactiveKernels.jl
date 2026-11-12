# Host NumPy code for `jax_source`: the module's `prepare_data` (decision
# `1ooufd4`, option 1).
#
# A program's data-only prefix, the recipes that read data alone (the part
# `prepare(...; bound)` folds at bind time, `_partial_split`), runs once per
# dataset in `prepare_data`, as NumPy on concrete values. The JAX functions
# then take its results. The prefix is translated by the same structural
# translation as the JAX code, with Python control flow throughout: values are
# concrete, so a condition on data is a Python `if`, a loop over data a Python
# `for`, and a value's length may depend on data (stage lanes, missing
# positions, level codes). Nothing is unrolled: a loop in the source is a loop
# in Python.
#
# Locals keep one Python name per frame, so loops, `break`, `continue` and an
# early `return` keep Julia's meaning. Arrays are written in place; an array
# the frame did not allocate is copied before its first write, so a source
# never writes its inputs. A vector built with `push!` is a Python list, read
# as an array wherever it is not appended to, indexed or measured. Ragged
# values are lists of arrays; `prepare_data` pads the ones the JAX functions
# take.

# The host state of one frame (a recipe's source, a helper's body).
mutable struct _HostState
    frame::Set{String}   # Python names of this frame's locals: rebinding them is safe
    safe::Set{String}    # names holding arrays this frame allocated: written in place
    deep::Set{String}    # names holding collections whose elements this frame
                         # allocated: their elements written in place
    lists::Set{String}   # names holding Python lists (`push!` targets)
    in_function::Bool    # a function body, where `return` is a statement
    depth::Int           # nesting in conditionals and loops
    returns::Vector{Any} # the types an explicit `return` returns
end
_HostState(frame::Set{String} = Set{String}(); in_function::Bool = false) =
    _HostState(frame, Set{String}(), Set{String}(), Set{String}(), in_function, 0, Any[])

# --- text -----------------------------------------------------------------------------

# The NumPy and SciPy spellings of the `jax.numpy` and `jax.scipy` names the
# translation writes: the array API the emitter uses is common to both.
const _HOST_NAMES = (r"\bjnp\." => "np.", r"\bjsp\." => "ssp.",
                     r"\bjax\.scipy\.linalg\." => "scipy.linalg.",
                     r"\bjax\.nn\.sigmoid\(" => "ssp.expit(",
                     r"\bjax\.nn\.log_sigmoid\(" => "ssp.log_expit(",
                     r"\bjax\.nn\.softplus\(" => "np.logaddexp(0.0, ")
const _PY_STRING = r"\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'"

# `text` with the substitutions applied outside its string literals.
function _host_text(text::AbstractString)
    out = IOBuffer()
    last = 1
    for m in eachmatch(_PY_STRING, text)
        print(out, _host_names(SubString(text, last, prevind(text, m.offset))), m.match)
        last = m.offset + ncodeunits(m.match)
    end
    print(out, _host_names(SubString(text, last)))
    String(take!(out))
end
# A JAX construct left in host text (outside its string literals): that text
# has no host translation.
_host_leftover(text::AbstractString) =
    occursin(r"\b(?:jax|lax)\.|\.at\[", replace(text, _PY_STRING => "\"\""))

# A helper's host text; a JAX construct left in it has no host translation.
function _host_helper_text(ctx::_JaxContext, text::AbstractString)
    text = _host_text(text)
    for line in split(text, '\n')
        _host_leftover(line) &&
            _jax_error("the line `", strip(line), "` has no host NumPy translation",
                       _jax_where(ctx))
    end
    text
end
_host_names(s::AbstractString) = foldl((t, r) -> replace(t, r), _HOST_NAMES; init = String(s))

# The NumPy spelling of an expression over static operands, or `nothing` when
# it has no host translation.
function _host_static_text(code::AbstractString)
    text = _host_text(code)
    _host_leftover(text) ? nothing : text
end

# Rewrite the lines a recipe emitted from `first` on; a JAX construct left in
# them has no host translation.
function _host_lines!(ctx::_JaxContext, first::Int)
    lines = ctx.scope.lines
    for i in first:length(lines)
        lines[i] = _host_text(lines[i])
        _host_leftover(lines[i]) &&
            _jax_error("the line `", strip(lines[i]), "` has no host NumPy translation",
                       _jax_where(ctx))
    end
end

# Host integers stay int64 in every module: NumPy computes the data with
# them, and JAX converts them when a float32 module's functions read them.
_host_dtype(@nospecialize(E)) = E isa Type && E <: Bool ? "np.bool_" :
    E isa Type && E <: Integer ? "np.int64" :
    E isa Type && E <: AbstractFloat ? "np." * _jax_dtype_name(E) :
    "object"
_host_is_numeric_array(@nospecialize(T)) = T isa Type && T <: AbstractArray && !_jax_ragged(T) &&
    eltype(T) isa Type && eltype(T) <: Number

# A module-level helper's NumPy twin, `<name>_np`.
function _host_helper!(ctx::_JaxContext, name::AbstractString, text::AbstractString)
    host = name * "_np"
    key = (:helper, host)
    haskey(ctx.mod.helper_names, key) && return host
    ctx.mod.helper_names[key] = host
    push!(ctx.mod.used, host)
    renamed = replace(text, Regex("\\bdef " * name * "\\(") => "def " * host * "(")
    push!(ctx.mod.helpers, _host_helper_text(ctx, rstrip(renamed)))
    host
end

# Run `f`, taking the lines it emits out of the current scope.
function _host_capture(f, ctx::_JaxContext)
    n = length(ctx.scope.lines)
    v = f()
    v, splice!(ctx.scope.lines, n+1:length(ctx.scope.lines))
end
_host_emit_lines!(ctx::_JaxContext, lines, extra::Int = 0) =
    append!(ctx.scope.lines, ("    "^extra * line for line in lines))

# --- locals ---------------------------------------------------------------------------

# A local read as a value: a list is read as an array.
_host_read(ctx::_JaxContext, v::_PyVal) = ctx.host !== nothing && v.code in ctx.host.lists ?
    _PyVal("np.asarray(" * v.code * ", dtype=" * _host_dtype(_jax_eltype(v.T)) * ")", v.T, true) : v
# A local read in place (appended to, indexed, measured, iterated).
_host_raw(ctx::_JaxContext, ex) = ex isa Symbol && haskey(ctx.locals, ex) ? ctx.locals[ex] :
    _jax_expr(ctx, ex)
_host_list(ctx::_JaxContext, @nospecialize(T)) = _PyVal("[]", T, true)

# The frame's Python name for local `s`, allocated on its first assignment.
function _host_name!(ctx::_JaxContext, s::Symbol)
    current = get(ctx.locals, s, nothing)
    current !== nothing && current.code in ctx.host.frame && return current.code
    name = _jax_fresh(ctx, s)
    push!(ctx.host.frame, name)
    name
end

# An allocation: its array may be written in place.
const _HOST_ALLOCATIONS = ("jnp.zeros(", "jnp.ones(", "jnp.full(", "np.zeros(", "np.ones(",
                           "np.full(", "np.empty(")
_host_fresh(code::AbstractString) = code in ("[]", "{}") || endswith(code, ".copy()") ||
    any(p -> startswith(code, p) && _host_one_call(code), _HOST_ALLOCATIONS)
# A list of allocations (`[np.zeros(n) for k in ...]`): each element is a
# fresh array (a list, a dictionary), so its elements may be written in place.
function _host_fresh_elements(code::AbstractString)
    startswith(code, "[") && endswith(code, "]") || return false
    clauses = findfirst(" for ", code)
    clauses === nothing && return false
    _host_fresh(code[nextind(code, 1):prevind(code, first(clauses))])
end
# `code` is one call, not a call followed by more operations.
function _host_one_call(code::AbstractString)
    depth = 0
    for (k, c) in enumerate(code)
        c == '(' && (depth += 1)
        c == ')' && (depth -= 1; depth == 0 && return k == lastindex(code))
    end
    false
end

function _host_bind!(ctx::_JaxContext, lhs::Symbol, v::_PyVal)
    _jax_translation_only(v) && (ctx.locals[lhs] = v; return)
    v.T === Union{} && return
    h = ctx.host
    name = _host_name!(ctx, lhs)
    name == v.code || _jax_emit!(ctx, name * " = " * v.code)
    v.code == "[]" || v.code in h.lists ? push!(h.lists, name) : delete!(h.lists, name)
    deep = _host_fresh_elements(v.code) || v.code in h.deep
    deep ? push!(h.deep, name) : delete!(h.deep, name)
    deep || _host_fresh(v.code) || v.code in h.safe ? push!(h.safe, name) : delete!(h.safe, name)
    ctx.locals[lhs] = _PyVal(name, v.T, true)
    nothing
end

function _host_assign!(ctx::_JaxContext, lhs, v::_PyVal)
    lhs isa Symbol && return _host_bind!(ctx, lhs, v)
    lhs isa Expr && lhs.head === :ref && lhs.args[1] isa Symbol && haskey(ctx.locals, lhs.args[1]) &&
        return _host_write!(ctx, lhs, v)
    root = _host_ref_root(lhs)
    root === nothing || haskey(ctx.locals, root) && return _host_nested_write!(ctx, lhs, root, v)
    _jax_error("the assignment target `", lhs, "` has no host translation", _jax_where(ctx))
end

# The local an indexed target `A[i][j]` writes into (`A`), and how many
# indexings deep (2); `nothing` and 0 for any other target.
function _host_ref_depth(lhs)
    depth = 0
    while lhs isa Expr && lhs.head === :ref
        lhs = lhs.args[1]
        depth += 1
    end
    lhs isa Symbol ? (lhs, depth) : (nothing, 0)
end
_host_ref_root(lhs) = (r = _host_ref_depth(lhs); last(r) >= 2 ? first(r) : nothing)

# Make local `s` an array this frame may write: a copy, unless it allocated it.
function _host_own!(ctx::_JaxContext, s::Symbol)
    v = ctx.locals[s]
    v.code in ctx.host.safe && return v
    name = _host_name!(ctx, s)
    _jax_emit!(ctx, name * " = " * v.code * ".copy()")
    v.code in ctx.host.lists && push!(ctx.host.lists, name)
    push!(ctx.host.safe, name)
    ctx.locals[s] = _PyVal(name, v.T, true)
end

# `A[i] = v`, `d[k] = v`: written in place.
function _host_write!(ctx::_JaxContext, lhs::Expr, v::_PyVal)
    s = lhs.args[1]
    target = ctx.locals[s]
    if target.T isa Type && target.T <: AbstractDict
        length(lhs.args) == 2 || _jax_error("a dictionary key of several indices", _jax_where(ctx))
        key = _jax_expr(ctx, lhs.args[2])
        _jax_emit!(ctx, target.code * "[" * key.code * "] = " * v.code)
        return
    end
    target = _host_own!(ctx, s)
    indexed = _jax_index(ctx, target, lhs.args[2:end]; target = true)
    startswith(indexed.code, target.code * "[") ||
        _jax_error("the indexed write `", lhs, "` has no host translation", _jax_where(ctx))
    _jax_emit!(ctx, indexed.code * " = " * v.code)
    nothing
end

const _HOST_DEEPCOPY = """
def _rk_deepcopy(x):
    # A copy of a collection and of its elements, whose elements the source
    # then writes in place.
    import copy
    return copy.deepcopy(x)
"""
# Make local `s` a collection whose elements this frame may write: a copy of
# it and its elements, unless it allocated them (a list of allocations).
function _host_own_deep!(ctx::_JaxContext, s::Symbol)
    v = ctx.locals[s]
    v.code in ctx.host.deep && return v
    name = _host_name!(ctx, s)
    _jax_emit!(ctx, name * " = " * _jax_helper!(ctx, "_rk_deepcopy", _HOST_DEEPCOPY) *
                    "(" * v.code * ")")
    v.code in ctx.host.lists && push!(ctx.host.lists, name)
    push!(ctx.host.safe, name)
    push!(ctx.host.deep, name)
    ctx.locals[s] = _PyVal(name, v.T, true)
end

# `A[i][j] = v`: an element of an array that a collection holds (a list of
# arrays this source builds), written in place.
function _host_nested_write!(ctx::_JaxContext, lhs::Expr, root::Symbol, v::_PyVal)
    _host_own_deep!(ctx, root)
    inner = _jax_expr(ctx, lhs.args[1])
    if inner.T isa Type && inner.T <: AbstractDict
        length(lhs.args) == 2 || _jax_error("a dictionary key of several indices", _jax_where(ctx))
        key = _jax_expr(ctx, lhs.args[2])
        _jax_emit!(ctx, _py_atom(inner.code) * "[" * key.code * "] = " * v.code)
        return
    end
    indexed = _jax_index(ctx, inner, lhs.args[2:end])
    startswith(indexed.code, _py_atom(inner.code) * "[") ||
        _jax_error("the indexed write `", lhs, "` has no host translation", _jax_where(ctx))
    _jax_emit!(ctx, indexed.code * " = " * v.code)
    nothing
end

# The locals `exs` assign, the arrays they write by index, and the
# collections whose elements they write by index (`A[i][j] = v`).
function _host_targets(exs)
    assigned, written, nested = Set{Symbol}(), Set{Symbol}(), Set{Symbol}()
    walk(ex) = begin
        ex isa Expr || return
        ex.head in (:quote, :inert, :->, :function) && return
        if (ex.head === :(=) || haskey(_JAX_UPDATING, ex.head)) && length(ex.args) == 2
            target = ex.args[1]
            target isa Expr && target.head === :(::) && (target = target.args[1])
            if target isa Expr && target.head === :ref
                root, depth = _host_ref_depth(target)
                root === nothing || push!(written, root)
                depth >= 2 && push!(nested, root)
            elseif target isa Symbol
                push!(assigned, target)
            elseif target isa Expr && target.head === :tuple
                foreach(t -> t isa Symbol && push!(assigned, t), target.args)
            end
        end
        foreach(walk, ex.args)
    end
    foreach(walk, exs)
    assigned, written, nested
end

# Before a loop or conditional: the outer locals its body assigns take names
# of this frame, so every pass assigns the same Python variable; arrays it
# writes are made this frame's own here, not on each pass. A conditional's new
# locals take their names here too, so both arms assign the same one.
function _host_prepare!(ctx::_JaxContext, bodies; conditional::Bool = false)
    assigned, written, nested = _host_targets(bodies)
    for s in sort!(collect(union(assigned, written)))
        if haskey(ctx.locals, s)
            v = ctx.locals[s]
            _jax_translation_only(v) && continue
            if s in nested
                _host_own_deep!(ctx, s)
            elseif s in written && !(v.T isa Type && v.T <: AbstractDict)
                _host_own!(ctx, s)
            elseif !(v.code in ctx.host.frame)
                name = _host_name!(ctx, s)
                _jax_emit!(ctx, name * " = " * v.code)
                v.code in ctx.host.lists && push!(ctx.host.lists, name)
                ctx.locals[s] = _PyVal(name, v.T, true)
            end
        elseif conditional && s in assigned
            name = _jax_fresh(ctx, s)
            push!(ctx.host.frame, name)
            ctx.locals[s] = _PyVal(name, Union{}, true)
        end
    end
end

# The locals of `before`, as a body left them.
_host_outer_locals(ctx::_JaxContext, before) =
    Dict{Symbol,_PyVal}(s => get(ctx.locals, s, v) for (s, v) in before)

# --- statements -----------------------------------------------------------------------

function _host_return(ctx::_JaxContext, ex)
    # A list returned (`return push!(v, x)`) leaves as the array it stands for.
    v = _host_read(ctx, _jax_expr(ctx, ex.args[1]))
    ctx.host.in_function || ctx.host.depth == 0 ||
        _jax_error("a `return` inside a conditional or loop of a recipe's source has no ",
                   "host translation", _jax_where(ctx))
    ctx.host.in_function || return v
    v.T === Union{} || _jax_emit!(ctx, "return " * v.code)
    push!(ctx.host.returns, v.T)
    _PyVal("None", Union{}, true)
end
_host_jump(ctx::_JaxContext, keyword) = (_jax_emit!(ctx, keyword); _PyVal("None", Union{}, true))

# One arm of a conditional: its statements one level in; `result`, when given,
# takes its value. Returns its value's type.
function _host_arm!(ctx::_JaxContext, build, result::AbstractString)
    ctx.scope.indent += 1
    ctx.host.depth += 1
    n = length(ctx.scope.lines)
    T = try
        v = build()
        isempty(result) || v.T === Union{} || _jax_emit!(ctx, result * " = " * v.code)
        v.T
    finally
        length(ctx.scope.lines) == n && _jax_emit!(ctx, "pass")
        ctx.scope.indent -= 1
        ctx.host.depth -= 1
    end
    T
end

_host_join(A, B) = A === Union{} ? B : B === Union{} ? A : promote_type(_jax_typeof_T(A), _jax_typeof_T(B))
_jax_typeof_T(@nospecialize(T)) = T isa Type ? T : Any

# An `if` (`value`: its value is used).
function _host_if(ctx::_JaxContext, ex, value::Bool)
    condition = _jax_expr(ctx, ex.args[1])
    condition.code in ("True", "False") && return _host_decided_if(ctx, ex, condition, value)
    _host_prepare!(ctx, ex.args[2:end]; conditional = true)
    before = copy(ctx.locals)
    values = Set(keys(ctx.scope.values))
    result = value ? _jax_fresh(ctx, :value) : ""
    arm(body) = value ? () -> _jax_expr(ctx, body) :
        () -> (_jax_effect(ctx, body); _PyVal("None", Nothing, true))
    _jax_emit!(ctx, "if " * condition.code * ":")
    T1 = _host_arm!(ctx, arm(ex.args[2]), result)
    after_then = copy(ctx.locals)
    _host_restore!(ctx, before, values)
    T2 = Nothing
    if length(ex.args) >= 3 || value
        _jax_emit!(ctx, "else:")
        T2 = length(ex.args) >= 3 ? _host_arm!(ctx, arm(ex.args[3]), result) :
            _host_arm!(ctx, () -> _PyVal("None", Nothing, true), result)
    end
    after_else = copy(ctx.locals)
    _host_restore!(ctx, before, values)
    for (s, v) in before
        ctx.locals[s] = _PyVal(v.code, _host_join(after_then[s].T, after_else[s].T), true)
    end
    # An arm may have bound a local to an input: it is no longer known to be
    # this frame's allocation or a list.
    for s in first(_host_targets(ex.args[2:end]))
        haskey(ctx.locals, s) || continue
        delete!(ctx.host.safe, ctx.locals[s].code)
        delete!(ctx.host.deep, ctx.locals[s].code)
    end
    value ? _PyVal(result, _host_join(T1, T2), true) : _PyVal("None", Nothing, true)
end
# An `if` whose condition is decided while emitting (`eltype(x) <: Integer`,
# `length(t) > 0` of a tuple) takes its arm alone, in line: Julia never runs
# the other, whose calls may have no method for these types. As in an arm, a
# recipe's `return` there has no host translation.
function _host_decided_if(ctx::_JaxContext, ex, condition::_PyVal, value::Bool)
    taken = condition.code == "True" ? ex.args[2] : length(ex.args) >= 3 ? ex.args[3] : nothing
    taken === nothing && return _PyVal("None", Nothing, true)
    ctx.host.depth += 1
    try
        value && return _jax_expr(ctx, taken)
        _jax_effect(ctx, taken)
        _PyVal("None", Nothing, true)
    finally
        ctx.host.depth -= 1
    end
end
function _host_restore!(ctx::_JaxContext, before, values)
    empty!(ctx.locals); merge!(ctx.locals, before)
    for id in collect(keys(ctx.scope.values))
        id in values || delete!(ctx.scope.values, id)
    end
end

# A conditional between two arms built by `then_build` and `else_build` (a
# lazy branch, a cell's branch called as a value).
function _host_conditional(ctx::_JaxContext, condition::_PyVal, then_build, else_build)
    condition.code in ("True", "False") &&
        return (condition.code == "True" ? then_build : else_build)()
    before = copy(ctx.locals)
    values = Set(keys(ctx.scope.values))
    result = _jax_fresh(ctx, :value)
    _jax_emit!(ctx, "if " * condition.code * ":")
    T1 = _host_arm!(ctx, then_build, result)
    _host_restore!(ctx, before, values)
    _jax_emit!(ctx, "else:")
    T2 = _host_arm!(ctx, else_build, result)
    _host_restore!(ctx, before, values)
    _PyVal(result, _host_join(T1, T2), true)
end

# `a && b`, `a || b`: the right side runs only when Julia's does.
function _host_short_circuit(ctx::_JaxContext, ex, is_or::Bool)
    a = _jax_expr(ctx, ex.args[1])
    # A left side decided while emitting is the value, or runs the right side
    # alone.
    a.code == (is_or ? "True" : "False") && return a
    a.code == (is_or ? "False" : "True") && return _jax_expr(ctx, ex.args[2])
    b, lines = _host_capture(() -> _jax_expr(ctx, ex.args[2]), ctx)
    isempty(lines) && b.T !== Union{} &&
        return _PyVal(_py_infix(is_or ? "or" : "and", [a.code, b.code]), _host_join(Bool, b.T), true)
    name = _jax_fresh(ctx, :condition)
    _jax_emit!(ctx, name * " = " * a.code)
    _jax_emit!(ctx, (is_or ? "if not " : "if ") * name * ":")
    _host_emit_lines!(ctx, lines, 1)
    b.T === Union{} || _jax_emit!(ctx, "    " * name * " = " * b.code)
    isempty(lines) && b.T === Union{} && _jax_emit!(ctx, "    pass")
    _PyVal(name, b.T === Union{} ? Bool : _host_join(Bool, b.T), true)
end

# A domain check: raises when the program runs on values that fail it.
function _host_check(ctx::_JaxContext, valid::_PyVal, message::AbstractString)
    _jax_emit!(ctx, "if not np.all(" * valid.code * "):")
    _jax_emit!(ctx, "    raise ValueError(" * message * ")")
    _PyVal("None", Nothing, true)
end

function _host_let(ctx::_JaxContext, ex)
    bindings, body = ex.args
    before = copy(ctx.locals)
    shadowed = Set{Symbol}()
    for binding in (bindings isa Expr && bindings.head === :block ? bindings.args : Any[bindings])
        binding isa LineNumberNode && continue
        binding isa Symbol && continue
        binding isa Expr && binding.head === :(=) && binding.args[1] isa Symbol ||
            _jax_error("the let binding `", binding, "` has no host translation", _jax_where(ctx))
        s = binding.args[1]
        v = _jax_expr(ctx, binding.args[2])
        # A let binding is a new local, whatever an outer one is called.
        delete!(ctx.locals, s)
        push!(shadowed, s)
        _host_bind!(ctx, s, v)
    end
    result = _jax_bind!(ctx, :let_value, _jax_expr(ctx, body))
    for (s, v) in before
        ctx.locals[s] = s in shadowed ? v : get(ctx.locals, s, v)
    end
    for s in collect(keys(ctx.locals))
        haskey(before, s) || delete!(ctx.locals, s)
    end
    result
end

# --- loops ----------------------------------------------------------------------------

# The header of one iteration specification (`i = 1:n`, `(i, v) = enumerate(x)`),
# with its targets bound as locals.
function _host_iteration!(ctx::_JaxContext, spec)
    spec isa Expr && spec.head in (:(=), :in) && length(spec.args) == 2 ||
        _jax_error("the iteration `", spec, "` has no host translation", _jax_where(ctx))
    lhs, iter = spec.args
    iterable, E = _host_iterable(ctx, iter)
    targets = _host_targets!(ctx, lhs, E)
    "for " * targets * " in " * iterable
end

# The Python iterable of `iter` and its element type.
function _host_iterable(ctx::_JaxContext, iter)
    if iter isa Expr && iter.head === :call
        f = iter.args[1] isa Symbol && _jax_has_local(ctx, iter.args[1]) ? nothing :
            _jax_resolve(ctx, iter.args[1])
        args = iter.args[2:end]
        if f === Colon() && length(args) == 2
            lo, hi = _jax_expr(ctx, args[1]), _jax_expr(ctx, args[2])
            foreach(v -> _jax_python_value(ctx, v), (lo, hi))
            return "range(" * lo.code * ", " * _jax_plus_one(hi) * ")", Int
        elseif f === Colon() && length(args) == 3
            lo, step, hi = (_jax_expr(ctx, a) for a in args)
            foreach(v -> _jax_python_value(ctx, v), (lo, step, hi))
            stop = "(" * hi.code * ") + (1 if " * step.code * " > 0 else -1)"
            return "range(" * lo.code * ", " * stop * ", " * step.code * ")", Int
        elseif (f === eachindex || f === keys || f === values || f === pairs) && length(args) == 1
            x = _host_raw(ctx, args[1])
            f in (eachindex, keys) || _jax_python_value(ctx, x)
            if x.T isa Type && x.T <: AbstractDict
                _jax_python_value(ctx, x)
                f === pairs && return x.code * ".items()", Tuple{Base.keytype(x.T),Base.valtype(x.T)}
                f === values && return x.code * ".values()", Base.valtype(x.T)
                return x.code * ".keys()", Base.keytype(x.T)
            end
            f === values && return x.code, _jax_eltype(x.T)
            f === pairs && return "enumerate(" * x.code * ", 1)", Tuple{Int,_jax_eltype(x.T)}
            if f === eachindex && _jax_is_array(x.T)
                n = _jax_python_value(ctx, _jax_function_call(ctx, length, [x]))
                return "range(1, " * _jax_plus_one(n) * ")", Int
            end
            return "range(1, len(" * x.code * ") + 1)", Int
        elseif f === axes && length(args) == 2
            x = _host_raw(ctx, args[1])
            return "range(1, " * _jax_dim_code(ctx, x, parse(Int, string(args[2]))) * " + 1)", Int
        elseif f === enumerate && length(args) == 1
            x, E = _host_iterable(ctx, args[1])
            return "enumerate(" * x * ", 1)", Tuple{Int,E}
        elseif f === zip
            parts = [_host_iterable(ctx, a) for a in args]
            return "zip(" * join(first.(parts), ", ") * ")", Tuple{last.(parts)...}
        elseif f === reverse && length(args) == 1
            x, E = _host_iterable(ctx, args[1])
            return "reversed(list(" * x * "))", E
        end
    end
    x = _host_raw(ctx, iter)
    _jax_python_value(ctx, x)
    T = x.T
    T isa Type && T <: AbstractDict && return x.code * ".items()", Tuple{Base.keytype(T),Base.valtype(T)}
    T isa Type && T <: Tuple && return x.code, promote_type(fieldtypes(T)...)
    T isa Type && T <: AbstractArray && ndims(T) > 1 &&
        return "np.ravel(" * x.code * ", order=\"F\")", eltype(T)
    T isa Type && T <: AbstractArray || T isa Type && T <: AbstractRange ||
        _jax_error("iterating a `", T, "` has no host translation", _jax_where(ctx))
    x.code, eltype(T)
end

# Host preparation may read all concrete data. A Python iterator in a JAX
# function may read only values available while tracing; converting a tracer
# to a string would otherwise silently print the tracer instead of its value.
function _jax_python_value(ctx::_JaxContext, v::_PyVal)
    ctx.host !== nothing || v.static ||
        _jax_error("a Python iterable or string operation needs values known while JAX traces", _jax_where(ctx))
    v
end

# Bind a loop's target (`i`, `(i, v)`) to fresh locals of element type `E`.
function _host_targets!(ctx::_JaxContext, lhs, @nospecialize(E))
    if lhs isa Symbol
        name = _jax_fresh(ctx, lhs)
        ctx.host === nothing || push!(ctx.host.frame, name)
        ctx.locals[lhs] = _PyVal(name, E, true)
        return name
    end
    lhs isa Expr && lhs.head === :tuple ||
        _jax_error("the loop target `", lhs, "` has no host translation", _jax_where(ctx))
    parts = map(enumerate(lhs.args)) do (k, t)
        Ek = E isa Type && E <: Tuple && k <= fieldcount(E) ? fieldtype(E, k) : Any
        _host_targets!(ctx, t, Ek)
    end
    "(" * join(parts, ", ") * (length(parts) == 1 ? ",)" : ")")
end

_host_statements(body) = body isa Expr && body.head === :block ? body.args : Any[body]

function _host_body!(ctx::_JaxContext, body)
    ctx.host.depth += 1
    n = length(ctx.scope.lines)
    try
        _jax_statements(ctx, _host_statements(body); value = false)
    finally
        length(ctx.scope.lines) == n && _jax_emit!(ctx, "pass")
        ctx.host.depth -= 1
    end
end

function _host_for(ctx::_JaxContext, ex)
    binding, body = ex.args
    specs = binding isa Expr && binding.head === :block ?
        Any[a for a in binding.args if !(a isa LineNumberNode)] : Any[binding]
    _host_prepare!(ctx, Any[body])
    before = copy(ctx.locals)
    indent = ctx.scope.indent
    try
        for spec in specs
            _jax_emit!(ctx, _host_iteration!(ctx, spec) * ":")
            ctx.scope.indent += 1
        end
        _host_body!(ctx, body)
    finally
        ctx.scope.indent = indent
    end
    # The outer locals keep their names; the body's own end with it.
    after = _host_outer_locals(ctx, before)
    empty!(ctx.locals); merge!(ctx.locals, after)
    _PyVal("None", Nothing, true)
end

function _host_while(ctx::_JaxContext, ex)
    condition_ex, body = ex.args
    _host_prepare!(ctx, Any[condition_ex, body])
    before = copy(ctx.locals)
    condition, lines = _host_capture(() -> _jax_expr(ctx, condition_ex), ctx)
    if isempty(lines)
        _jax_emit!(ctx, "while " * condition.code * ":")
    else
        # A condition with statements of its own is evaluated before each pass.
        _jax_emit!(ctx, "while True:")
        _host_emit_lines!(ctx, lines, 1)
        _jax_emit!(ctx, "    if not (" * condition.code * "):")
        _jax_emit!(ctx, "        break")
    end
    ctx.scope.indent += 1
    try
        _host_body!(ctx, body)
    finally
        ctx.scope.indent -= 1
    end
    after = _host_outer_locals(ctx, before)
    empty!(ctx.locals); merge!(ctx.locals, after)
    _PyVal("None", Nothing, true)
end

# --- comprehensions, generators, do-blocks -------------------------------------------

# A nested function over `params` (Julia names, of `types`) whose body `build`
# translates, in a frame of its own. Returns its name and its value's type.
function _host_nested_function!(ctx::_JaxContext, hint, params::Vector{Symbol}, types, build)
    name = _jax_fresh(ctx, hint)
    outer = ctx.scope
    inner = _JaxScope(outer)
    inner.indent = outer.indent + 1
    names = [_jax_identifier(inner.used, p) for p in params]
    saved = (copy(ctx.locals), ctx.host)
    ctx.host = saved[2] === nothing ? nothing : _HostState(Set{String}(names); in_function = true)
    ctx.scope = inner
    v = try
        for (p, n, T) in zip(params, names, types)
            ctx.locals[p] = _PyVal(n, T, true)
        end
        _jax_python_value(ctx, _host_read(ctx, build()))
    finally
        ctx.scope = outer
        empty!(ctx.locals); merge!(ctx.locals, saved[1])
        ctx.host = saved[2]
    end
    push!(outer.lines, "    "^outer.indent * "def " * name * "(" * join(names, ", ") * "):")
    append!(outer.lines, inner.lines)
    v.T === Union{} || push!(outer.lines, "    "^inner.indent * "return " * v.code)
    name, v.T
end

# The clauses of a generator (`for i in r if c`), its element's Python
# expression and type. Iterables after the first must be expressions.
function _host_generator(ctx::_JaxContext, gen::Expr; column_major::Bool = false)
    body = gen.args[1]
    specs = Any[gen.args[2:end]...]
    condition = nothing
    if length(specs) == 1 && specs[1] isa Expr && specs[1].head === :filter
        condition = specs[1].args[1]
        specs = Any[specs[1].args[2:end]...]
    end
    before = copy(ctx.locals)
    clauses = String[]
    bound = Symbol[]
    try
        for (k, spec) in enumerate(specs)
            header, lines = _host_capture(() -> _host_iteration!(ctx, spec), ctx)
            k == 1 ? _host_emit_lines!(ctx, lines) : isempty(lines) ||
                _jax_error("a generator whose inner range needs statements has no host ",
                           "translation", _jax_where(ctx))
            push!(clauses, header)
            append!(bound, _host_target_symbols(spec.args[1]))
        end
        if column_major && length(specs) > 1
            # Julia's product iterator varies its first index fastest. Python
            # product varies the last; reverse both inputs and yielded tuples.
            # Each input is evaluated once, in authored order.
            product = _jax_helper!(ctx, "_rk_product", raw"""
def _rk_product(*iterables):
    import itertools
    return (tuple(reversed(item)) for item in itertools.product(*reversed(iterables)))
""")
            parts = [split(c[5:end], " in "; limit = 2) for c in clauses]
            clauses = ["for (" * join(first.(parts), ", ") * ") in " * product *
                       "(" * join(last.(parts), ", ") * ")"]
        end
        element, E = _host_inline_or_function(ctx, :element, bound, body)
        if condition !== nothing
            test, _ = _host_inline_or_function(ctx, :keep, bound, condition)
            push!(clauses, "if " * test)
        end
        return element, join(clauses, " "), E, length(specs)
    finally
        empty!(ctx.locals); merge!(ctx.locals, before)
    end
end
_host_target_symbols(lhs) = lhs isa Symbol ? Symbol[lhs] :
    vcat((_host_target_symbols(t) for t in lhs.args)...)

# `ex` over the generator targets `bound`: an expression when it needs no
# statements, else a call of a nested function of the targets.
function _host_inline_or_function(ctx::_JaxContext, hint, bound::Vector{Symbol}, ex)
    v, lines = _host_capture(() -> _jax_expr(ctx, ex), ctx)
    _jax_python_value(ctx, v)
    isempty(lines) && return v.code, v.T
    names = [ctx.locals[s].code for s in bound]
    types = Any[ctx.locals[s].T for s in bound]
    fname, T = _host_nested_function!(ctx, hint, bound, types, () -> _jax_expr(ctx, ex))
    fname * "(" * join(names, ", ") * ")", T
end

# A list of `E` values as the Julia vector (or matrix, `rank` 2) it stands for.
function _host_collected(code::AbstractString, @nospecialize(E), rank::Int = 1)
    E isa Type && E <: Number || return _PyVal(code, Array{E,rank}, true)
    _PyVal("np.asarray(" * code * ", dtype=" * _host_dtype(E) * ")", Array{E,rank}, true)
end

function _host_comprehension(ctx::_JaxContext, gen::Expr)
    element, clauses, E, n = _host_generator(ctx, gen)
    if n == 2 && !occursin(" if ", clauses)
        # `[f(i, j) for i in a, j in b]` is the matrix of rows `i`.
        outer, inner = split(clauses, r" (?=for )"; limit = 2)
        return _host_collected("[[" * element * " " * inner * "] " * outer * "]", E, 2)
    end
    n == 1 || occursin(" if ", clauses) ||
        _jax_error("a comprehension over ", n, " ranges has no host translation", _jax_where(ctx))
    _host_collected("[" * element * " " * clauses * "]", E)
end

const _HOST_REDUCTIONS = Dict{Any,String}(sum => "sum", maximum => "max", minimum => "min",
                                          any => "any", all => "all")
function _host_generator_reduce(ctx::_JaxContext, f, gen::Expr, kwargs)
    element, clauses, E, _ = _host_generator(ctx, gen)
    code = f === prod ? "np.prod([" * element * " " * clauses * "])" :
        _HOST_REDUCTIONS[f] * "(" * element * " " * clauses * ")"
    result = _PyVal(code, f in (any, all) ? Bool : E, true)
    for (k, v) in kwargs
        k === :init && f in (sum, prod) ||
            _jax_error("the keyword `", k, "` of `", f, "` over a generator", _jax_where(ctx))
        init = _jax_expr(ctx, v)
        result = _PyVal(_py_infix(f === sum ? "+" : "*", [init.code, result.code]),
                        promote_type(_jax_typeof(init), E), true)
    end
    result
end

# `map(xs...) do args... body end`: the body as a function of one element of
# each collection, called on each. A `plate` over data (a binder's data-only
# column, such as a per-group step count) is the same map on the host.
function _host_do(ctx::_JaxContext, ex)
    call, lambda = ex.args
    call isa Expr && call.head === :call && lambda isa Expr && lambda.head === :-> ||
        _jax_error("the do-block `", ex, "` has no host translation", _jax_where(ctx))
    _jax_resolve(ctx, call.args[1]) in (map, plate) ||
        _jax_error("a do-block on `", call.args[1], "` has no host translation", _jax_where(ctx))
    params = lambda.args[1]
    names = params isa Symbol ? Symbol[params] : Symbol[p for p in params.args]
    assigned, written = _host_targets(Any[lambda.args[2]])
    any(s -> haskey(ctx.locals, s), union(assigned, written)) &&
        _jax_error("a do-block that assigns a local of its caller has no host translation",
                   _jax_where(ctx))
    collections = [_host_iterable(ctx, a) for a in call.args[2:end]]
    length(names) == length(collections) ||
        _jax_error("a do-block with ", length(names), " arguments over ", length(collections),
                   " collections", _jax_where(ctx))
    types = Any[last(c) for c in collections]
    fname, E = _host_nested_function!(ctx, :element, names, types, () -> _jax_expr(ctx, lambda.args[2]))
    targets = [_jax_fresh(ctx, n) for n in names]
    iterable = length(collections) == 1 ? first(only(collections)) :
        "zip(" * join(first.(collections), ", ") * ")"
    _host_collected("[" * fname * "(" * join(targets, ", ") * ") for " *
                    (length(targets) == 1 ? only(targets) : join(targets, ", ")) * " in " *
                    iterable * "]", E)
end

# --- calls ----------------------------------------------------------------------------

# Calls that read a collection in place; `nothing` for any other.
function _host_call(ctx::_JaxContext, f, positional, kwargs)
    isempty(kwargs) || return nothing
    (f === sort! || f === unique!) && length(positional) == 1 &&
        return _host_rewrite!(ctx, f, positional[1])
    listed = _host_list_read(ctx, f, positional)
    listed === nothing || return listed
    if f === push! || f === append!
        length(positional) >= 2 || return nothing
        c = _host_raw(ctx, positional[1])
        c.code in ctx.host.lists ||
            _jax_error("`", f, "` onto a value that is not a vector this source builds ",
                       "(`T[]`)", _jax_where(ctx))
        for a in positional[2:end]
            v = _jax_expr(ctx, a)
            _jax_emit!(ctx, c.code * (f === push! ? ".append(" : ".extend(") * v.code * ")")
        end
        return c
    end
    length(positional) == 1 && (f === length || f === isempty) || length(positional) == 2 &&
        f in (haskey, in, ∈) || length(positional) == 3 && f in (get, get!) ||
        length(positional) == 2 && f === delete! || return nothing
    c = _host_raw(ctx, positional[f in (in, ∈) ? 2 : 1])
    f === length && return _host_length(ctx, c)
    f === isempty && return _PyVal("(" * _host_length(ctx, c).code * " == 0)", Bool, true)
    key = _jax_expr(ctx, positional[f in (in, ∈) ? 1 : 2])
    (f === haskey || f === in || f === ∈) && return _PyVal("(" * key.code * " in " * c.code * ")", Bool, true)
    c.T isa Type && c.T <: AbstractDict ||
        _jax_error("`", f, "` of a `", c.T, "` has no host translation", _jax_where(ctx))
    if f === delete!
        _jax_emit!(ctx, c.code * ".pop(" * key.code * ", None)")
        return c
    end
    default = _jax_expr(ctx, positional[3])
    v = _PyVal(c.code * (f === get ? ".get(" : ".setdefault(") * key.code * ", " * default.code * ")",
               Base.valtype(c.T), true)
    # `get!` writes the dictionary: bind it where it runs, so the write
    # happens even when the source discards its value (a statement
    # `get!(d, k, v)`), and before any later read of `d`.
    f === get! ? _jax_bind!(ctx, :value, v) : v
end

function _host_length(ctx::_JaxContext, x::_PyVal)
    T = x.T
    T isa Type && T <: Tuple && return _PyVal(string(fieldcount(T)), Int, true)
    T isa Type && T <: AbstractArray && ndims(T) > 1 && !(x.code in ctx.host.lists) &&
        return _PyVal("np.size(" * x.code * ")", Int, true)
    _PyVal("len(" * x.code * ")", Int, true)
end

# `x[i]` of a dictionary, a list or a ragged list of arrays; `nothing` for an
# array, which indexes as JAX's does.
function _host_index(ctx::_JaxContext, x::_PyVal, indices)
    T = x.T
    if T isa Type && T <: AbstractDict
        length(indices) == 1 || _jax_error("a dictionary key of several indices", _jax_where(ctx))
        key = _jax_expr(ctx, indices[1])
        return _PyVal(_py_atom(x.code) * "[" * key.code * "]", Base.valtype(T), true)
    end
    x.code in ctx.host.lists || _jax_ragged(T) || return nothing
    length(indices) == 1 || _jax_error("indexing a list with ", length(indices), " indices",
                                       _jax_where(ctx))
    E = _jax_eltype(T)
    indices[1] === :(:) && return _PyVal("list(" * x.code * ")", T, true)
    i = _jax_bound_expr(ctx, x, 1, indices[1])
    i.T isa Type && i.T <: Integer &&
        return _PyVal(_py_atom(x.code) * "[" * _jax_minus_one(i) * "]", E, true)
    _jax_is_array(i.T) || i.T isa Type && i.T <: AbstractRange ||
        _jax_error("the index of type `", i.T, "` into a list", _jax_where(ctx))
    k = _jax_fresh(ctx, :k)
    _host_collected("[" * _py_atom(x.code) * "[" * k * " - 1] for " * k * " in " * i.code * "]", E)
end

# A Julia type called as a constructor; `nothing` where JAX's translation serves.
function _host_construct(ctx::_JaxContext, T::Type, args::Vector{_PyVal})
    T <: AbstractDict && isempty(args) && return _PyVal("{}", T isa UnionAll ? Dict{Any,Any} : T, true)
    T <: AbstractVector && isempty(args) &&
        return _host_list(ctx, T isa UnionAll ? Vector{Any} : T)
    if T <: AbstractArray && !isempty(args) && args[1].T === UndefInitializer
        dims = args[2:end]
        E = eltype(T)
        E isa TypeVar && (E = Float64)
        shape, N = _jax_dims(ctx, T, dims)
        return _PyVal("np.zeros(" * shape * ", dtype=" * _host_dtype(E) * ")", Array{E,N}, true)
    end
    length(args) == 1 && !_jax_is_array(args[1].T) || return nothing
    T <: Bool && return _PyVal("bool(" * args[1].code * ")", Bool, true)
    T <: Integer && return _PyVal("int(" * args[1].code * ")", T, true)
    T <: AbstractFloat && return _PyVal(_host_dtype(T) * "(" * args[1].code * ")", T, true)
    nothing
end

# Scalar predicates of the host's values.
const _HOST_ISEQUAL = """
def _rk_isequal(a, b):
    # Julia's isequal on scalars: NaN equals NaN, -0.0 differs from 0.0, and
    # missing (None) equals only missing.
    if a is None or b is None:
        return a is b
    if a != a and b != b:
        return True
    if not bool(a == b):
        return False
    if isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)) and a == 0:
        return bool(np.signbit(a) == np.signbit(b))
    return True
"""

# --- order: sort, unique, searchsortedfirst --------------------------------------------

# Julia's `isless` order on the host. A float array orders by integer keys:
# -0.0 before 0.0 and every NaN last, with equal keys exactly for the values
# `isequal` treats as equal, so `sort`, `unique` and `searchsortedfirst`
# agree with Julia's on every value. Integer and string arrays are their own
# keys.
const _HOST_ORDER = Dict{Symbol,Pair{String,String}}(
    :key => "_rk_order_key_np" => """
def _rk_order_key_np(x):
    # Julia's isless order of an array as sortable keys: a float is an integer
    # key, -0.0 before 0.0 and every NaN last, equal for values isequal treats
    # as equal. Integer and string arrays are their own keys.
    x = np.asarray(x)
    if x.dtype.kind != "f":
        return x
    bits = np.where(np.isnan(x), np.nan, x).astype(np.float64).view(np.int64)
    return np.where(bits < 0, bits ^ np.int64(0x7FFFFFFFFFFFFFFF), bits)
""",
    :sort => "_rk_sort_np" => """
def _rk_sort_np(x):
    # Julia's sort of a vector: ascending in isless order, equal values in
    # their order.
    x = np.asarray(x)
    return x[np.argsort(_rk_order_key_np(x), kind="stable")]
""",
    :unique => "_rk_unique_np" => """
def _rk_unique_np(x):
    # Julia's unique of a vector: the first value of each isequal class, in
    # order.
    x = np.asarray(x)
    _, first = np.unique(_rk_order_key_np(x), return_index=True)
    return x[np.sort(first)]
""",
    :isless => "_rk_isless_key_np" => """
def _rk_isless_key_np(a):
    # One value's place in Julia's isless order: every NaN last, -0.0 before
    # 0.0, equal places for values isequal treats as equal.
    if isinstance(a, (float, np.floating)):
        return (1, 0.0, 0) if a != a else (0, a, 0 if np.signbit(a) else 1)
    return (0, a, 1)
""",
    :search => "_rk_searchsorted_np" => """
def _rk_searchsorted_np(v, x, side):
    # The number of entries of the sorted vector v (an array or a list) before
    # x in isless order (side "left"), or before or equal to it ("right"):
    # Julia's searchsortedfirst(v, x) - 1 and searchsortedlast(v, x). A
    # bisection, so O(log n) comparisons.
    import bisect
    search = bisect.bisect_left if side == "left" else bisect.bisect_right
    return search(v, _rk_isless_key_np(x), key=_rk_isless_key_np)
""")
const _HOST_ORDER_KEYS = Dict(:sort => (:key, :sort), :unique => (:key, :unique),
                              :search => (:isless, :search))

# The host helper `which` (with the order key it calls), defined once.
function _host_order_helper!(ctx::_JaxContext, which::Symbol)
    name = ""
    for k in _HOST_ORDER_KEYS[which]
        name, text = _HOST_ORDER[k]
        key = (:helper, name)
        haskey(ctx.mod.helper_names, key) && continue
        ctx.mod.helper_names[key] = name
        push!(ctx.mod.used, name)
        push!(ctx.mod.helpers, rstrip(text))
    end
    name
end

# The element type of a vector `f` orders on the host: numbers and strings,
# which order as Julia's `isless` does.
function _host_ordered_eltype(ctx::_JaxContext, f, @nospecialize(T))
    E = T isa Type && T <: AbstractVector ? eltype(T) : nothing
    E isa Type && (E <: Real || E <: AbstractString) && return E
    _jax_error("`", f, "` of a `", T, "` has no host translation (vectors of numbers or ",
               "strings order on the host)", _jax_where(ctx))
end

function _host_order_call(ctx::_JaxContext, f, which::Symbol, args::Vector{_PyVal})
    length(args) == 1 ||
        _jax_error("`", f, "` with ", length(args), " arguments has no host translation",
                   _jax_where(ctx))
    E = _host_ordered_eltype(ctx, f, args[1].T)
    _PyVal(_host_order_helper!(ctx, which) * "(" * args[1].code * ")", Vector{E}, true)
end
for (f, which) in ((sort, :sort), (unique, :unique))
    @eval function _jax_call_positional(ctx::_JaxContext, ::typeof($f), args::Vector{_PyVal})
        ctx.host === nothing &&
            return invoke(_jax_call_positional, Tuple{_JaxContext,Function,Vector{_PyVal}},
                          ctx, $f, args)
        _host_order_call(ctx, $f, $(QuoteNode(which)), args)
    end
end
# `searchsortedfirst(v, x)` and `searchsortedlast(v, x)`, 1-based as Julia's:
# a bisection of `v` (an array, or a list searched in place).
function _host_search(ctx::_JaxContext, f, v::_PyVal, x::_PyVal)
    _host_ordered_eltype(ctx, f, v.T)
    X = x.T
    X isa Type && (X <: Real || X <: AbstractString) ||
        _jax_error("`", f, "` of a `", X, "` value has no host translation (a number or a ",
                   "string has)", _jax_where(ctx))
    _PyVal(_host_order_helper!(ctx, :search) * "(" * v.code * ", " * x.code *
           (f === searchsortedfirst ? ", \"left\") + 1" : ", \"right\")"), Int, true)
end
for f in (searchsortedfirst, searchsortedlast)
    @eval function _jax_call_positional(ctx::_JaxContext, ::typeof($f), args::Vector{_PyVal})
        ctx.host === nothing &&
            return invoke(_jax_call_positional, Tuple{_JaxContext,Function,Vector{_PyVal}},
                          ctx, $f, args)
        length(args) == 2 ||
            _jax_error("`", $f, "` with ", length(args), " arguments has no host translation ",
                       "(`", $f, "(v, x)` has)", _jax_where(ctx))
        _host_search(ctx, $f, args[1], args[2])
    end
end

# A local list read in place where Julia reads one entry or searches it, so a
# loop that reads it as it grows (`last(v)` of a run test) stays linear rather
# than converting the whole list on each pass; `nothing` for any other call.
function _host_list_read(ctx::_JaxContext, f, positional)
    s = isempty(positional) ? nothing : positional[1]
    s isa Symbol && haskey(ctx.locals, s) && ctx.locals[s].code in ctx.host.lists ||
        return nothing
    c = ctx.locals[s]
    if (f === first || f === last) && length(positional) == 1
        return _PyVal(_py_atom(c.code) * (f === last ? "[-1]" : "[0]"), _jax_eltype(c.T), true)
    elseif (f === searchsortedfirst || f === searchsortedlast) && length(positional) == 2
        return _host_search(ctx, f, c, _jax_expr(ctx, positional[2]))
    end
    nothing
end

# The local an argument names, or that a call rewriting a local in place
# returns (`sort!(v)` in `unique!(sort!(v))`), with its value; `nothing` and
# the value for any other expression (a temporary such as `copy(x)`).
function _host_target(ctx::_JaxContext, ex)
    ex isa Symbol && haskey(ctx.locals, ex) && return ex, ctx.locals[ex]
    v = _jax_expr(ctx, ex)
    for (s, l) in ctx.locals
        l.code == v.code && return s, l
    end
    nothing, v
end

# `sort!(v)`, `unique!(v)`: Julia rewrites `v` and returns it. A vector this
# source builds with `push!` (a list) is rewritten in place and stays a list,
# so every name for it sees the result and it may grow on. An array is
# sorted in place, made this frame's own first as any write does; a NumPy
# array keeps its length, so `unique!` of one has no host translation. A
# temporary's rewritten value is the result.
function _host_rewrite!(ctx::_JaxContext, f, ex)
    s, c = _host_target(ctx, ex)
    E = _host_ordered_eltype(ctx, f, c.T)
    helper = _host_order_helper!(ctx, f === sort! ? :sort : :unique)
    s === nothing && return _PyVal(helper * "(" * _host_read(ctx, c).code * ")", Vector{E}, true)
    if c.code in ctx.host.lists
        _jax_emit!(ctx, c.code * "[:] = " * helper * "(" * _host_read(ctx, c).code * ").tolist()")
        return c
    end
    f === unique! &&
        _jax_error("`unique!` of `", s, "`, an array that this source does not build with ",
                   "`push!` (`T[]`), has no host translation: a NumPy array keeps its length",
                   _jax_where(ctx))
    v = _host_own!(ctx, s)
    _jax_emit!(ctx, v.code * "[:] = " * helper * "(" * v.code * ")")
    v
end
function _jax_call_positional(ctx::_JaxContext, ::typeof(isequal), args::Vector{_PyVal})
    (ctx.host === nothing || length(args) != 2) &&
        return _jax_function_call(ctx, isequal, args)
    name = _jax_helper!(ctx, "_rk_isequal", _HOST_ISEQUAL)
    _PyVal(name * "(" * _jax_args(args) * ")", Bool, true)
end

# Search concrete host data in order, stopping before evaluating any later
# predicate. The result is Julia's one-based index, or nothing when absent.
function _jax_function_call(ctx::_JaxContext, ::typeof(findfirst), args::Vector{_PyVal})
    if ctx.host === nothing || length(args) != 2 ||
       !(args[2].T isa Type && args[2].T <: AbstractVector)
        return _jax_traceable_or_lowered(ctx, findfirst, args)
    end
    predicate = _jax_function_of(args[1])
    values = args[2]
    result = _jax_fresh(ctx, :found)
    index = _jax_fresh(ctx, :position)
    element = _jax_fresh(ctx, :element)
    _jax_emit!(ctx, result * " = None")
    _jax_emit!(ctx, "for " * index * ", " * element * " in enumerate(" * values.code * ", 1):")
    ctx.scope.indent += 1
    try
        test = _jax_bind!(ctx, :matched,
                         _jax_call(ctx, predicate, [_PyVal(element, eltype(values.T), true)]))
        _jax_emit!(ctx, "if not isinstance(" * test.code * ", (bool, np.bool_)):")
        _jax_emit!(ctx, "    raise TypeError(\"findfirst predicate must return Bool\")")
        _jax_emit!(ctx, "if " * test.code * ":")
        _jax_emit!(ctx, "    " * result * " = " * index)
        _jax_emit!(ctx, "    break")
    finally
        ctx.scope.indent -= 1
    end
    _PyVal(result, Union{Nothing,Int}, true)
end
function _jax_call_positional(ctx::_JaxContext, ::typeof(ismissing), args::Vector{_PyVal})
    ctx.host === nothing && return _jax_function_call(ctx, ismissing, args)
    _PyVal("(" * only(args).code * " is None)", Bool, true)
end
function _jax_function_call(ctx::_JaxContext, ::typeof(copy), args::Vector{_PyVal})
    x = only(args)
    ctx.host === nothing && return x
    _PyVal(_py_atom(x.code) * ".copy()", x.T, true)
end

# --- axes tuples --------------------------------------------------------------------------
#
# On the host an axes tuple, `axes(x)`, is a Python tuple of `range(1, n + 1)`,
# one per dimension: its lengths and entries are Julia's `Base.OneTo`s'. Data
# shapes computed from data (a shape per plate cell) take this form.

function _host_axes(ctx::_JaxContext, x::_PyVal)
    x.T isa Type && x.T <: Number && return _PyVal("()", Tuple{}, true)
    _jax_is_array(x.T) && !_jax_ragged(x.T) ||
        _jax_error("`axes` of a `", x.T, "` has no host translation", _jax_where(ctx))
    _PyVal("tuple(range(1, n + 1) for n in np.shape(" * x.code * "))",
           NTuple{ndims(x.T),Base.OneTo{Int}}, true)
end

const _HOST_BROADCAST_SHAPE = """
def _rk_broadcast_shape(*shapes):
    # Julia's Broadcast.broadcast_shape of axes tuples: a length-1 axis
    # stretches, the other lengths must agree.
    out = []
    for d in range(max([0] + [len(s) for s in shapes])):
        lengths = set(len(s[d]) for s in shapes if d < len(s)) - {1}
        if len(lengths) > 1:
            raise ValueError("DimensionMismatch: arrays could not be broadcast to a common size")
        out.append(range(1, (lengths.pop() if lengths else 1) + 1))
    return tuple(out)
"""
function _jax_function_call(ctx::_JaxContext, ::typeof(Base.Broadcast.broadcast_shape),
                            args::Vector{_PyVal})
    ctx.host === nothing &&
        return _jax_traceable_or_lowered(ctx, Base.Broadcast.broadcast_shape, args)
    T = _jax_return_type(Base.Broadcast.broadcast_shape, Any[a.T for a in args])
    T isa Type && T <: Tuple && isconcretetype(T) ||
        _jax_error("`Broadcast.broadcast_shape` of `", join((a.T for a in args), "`, `"),
                   "` has no host translation", _jax_where(ctx))
    name = _jax_helper!(ctx, "_rk_broadcast_shape", _HOST_BROADCAST_SHAPE)
    _PyVal(name * "(" * _jax_args(args) * ")", T, true)
end

# --- plates and scans -------------------------------------------------------------------

# A plate over vectors: the cell as a function, called on each element.
function _host_plate(ctx::_JaxContext, recipe::Recipe, op::_AuthoredPlateOp{K,A}, inputs) where {K,A}
    body = op.kernel.plan
    inputs = [!(i in A) && _jax_numeric_tuple(v.T) ?
              _PyVal("np.stack(" * v.code * ")", Vector{eltype(v.T)}, true) : v
              for (i, v) in enumerate(inputs)]
    mapped = Bool[!(i in A) && _jax_is_array(v.T) for (i, v) in enumerate(inputs)]
    any(mapped) || _jax_error("a plate without a batched array argument", _jax_where(ctx))
    ranks = unique(_jax_rank(inputs[i].T) for i in eachindex(inputs) if mapped[i])
    ranks == [1] || _jax_error("a plate over arrays of rank ", join(string.(ranks), ", "),
                               " has no host translation yet", _jax_where(ctx))
    cell_types = Any[mapped[i] ? eltype(v.T) : _jax_unref(v.T) for (i, v) in enumerate(inputs)]
    cell, wants, _ = _jax_plan_function!(ctx, :cell, body, cell_types)
    E = _jax_typeof(only(wants))
    targets = String[]
    args = String[]
    for (i, v) in enumerate(inputs)
        if mapped[i]
            t = _jax_fresh(ctx, :element)
            push!(targets, t); push!(args, t)
        else
            push!(args, v.code)
        end
    end
    sources = [inputs[i].code for i in eachindex(inputs) if mapped[i]]
    iterable = length(sources) == 1 ? only(sources) : "zip(" * join(sources, ", ") * ")"
    [_host_collected("[" * cell * "(" * join(args, ", ") * ") for " * join(targets, ", ") *
                     " in " * iterable * "]", E)]
end

# A scan: the step called on each element of the sequences in turn.
function _host_scan(ctx::_JaxContext, recipe::Recipe, op::_AuthoredScanOp{K,A,I,H},
                    inputs) where {K,A,I,H}
    body = op.kernel.plan
    iterated = [i for i in 2:length(inputs) if !(i in A)]
    shared = [i for i in 2:length(inputs) if i in A]
    order = [1; iterated; shared]
    types = Any[inputs[1].T; [eltype(inputs[i].T) for i in iterated]; [inputs[i].T for i in shared]]
    step, wants, _ = _jax_plan_function!(ctx, :step, body, types)
    E = _jax_typeof(wants[2])
    carry = _jax_fresh(ctx, :carry)
    outputs = _jax_fresh(ctx, :outputs)
    output = _jax_fresh(ctx, :output)
    targets = [_jax_fresh(ctx, :element) for _ in iterated]
    _jax_emit!(ctx, carry * " = " * inputs[1].code)
    _jax_emit!(ctx, outputs * " = []")
    sequences = [inputs[i].code for i in iterated]
    iterable = length(sequences) == 1 ? only(sequences) : "zip(" * join(sequences, ", ") * ")"
    _jax_emit!(ctx, "for " * join(targets, ", ") * " in " * iterable * ":")
    _jax_emit!(ctx, "    " * carry * ", " * output * " = " * step * "(" *
                    join([carry; targets; [inputs[i].code for i in shared]], ", ") * ")")
    _jax_emit!(ctx, "    " * outputs * ".append(" * output * ")")
    I && _jax_emit!(ctx, outputs * ".insert(0, " * inputs[1].code * ")")
    if _jax_numeric_namedtuple(E)
        I && _jax_error("a scan that keeps its initial value and outputs `", E, "`", _jax_where(ctx))
        columns = "tuple(np.asarray([o[k] for o in " * outputs * "]) for k in range(" *
                  string(fieldcount(E)) * "))"
        return [_PyVal(columns, Vector{E}, true)]
    end
    E <: Number || _jax_error("a scan whose step outputs `", E, "` has no host translation",
                              _jax_where(ctx))
    [_host_collected(outputs, E)]
end

# --- helpers called from host code ----------------------------------------------------

# Function identities and types belong to translation, not a Python argument
# list. A closure or aggregate can also carry runtime values: pass those leaves
# and rebuild its translation metadata using names in the helper's own scope.
function _host_function_input!(scope, params, passed, hint, v::_PyVal)
    _host_function_input!(scope, params, passed, hint, v, v.T)
end
function _host_function_input!(scope, params, passed, hint, v::_PyVal, T)
    if T isa Type && T <: Function && Base.issingletontype(T)
        return _PyVal(v.code, _JaxCallable(T.instance), true)
    end
    type = _jax_type_of_value(v)
    type === nothing || return _PyVal(v.code, _JaxCallable(type), true)
    name = _jax_identifier(scope.used, hint)
    push!(params, name)
    push!(passed, v)
    _PyVal(name, T, true)
end
function _host_function_input!(scope, params, passed, hint, v::_PyVal, T::_JaxCallable)
    T.f isa _JaxClosure || return v
    captures = [_host_function_input!(scope, params, passed, Symbol(hint, :_, i), a)
                for (i, a) in enumerate(T.f.captures)]
    _PyVal(v.code, _JaxCallable(_JaxClosure(T.f.F, captures)), true)
end
function _host_function_input!(scope, params, passed, hint, v::_PyVal, T::_JaxAggregate)
    fields = [_host_function_input!(scope, params, passed, Symbol(hint, :_, i), a)
              for (i, a) in enumerate(T.fields)]
    _PyVal(v.code, _JaxAggregate(T.T, fields), true)
end

# A `@traceable` helper on the host: a module-level function of its body. The
# call's `args` select it; `inputs` are the values its formals `names` take
# (a Vararg formal's is the tuple of the trailing arguments).
function _host_function_call(ctx::_JaxContext, f, names::Vector{Symbol}, body, mod::Module,
                             args::Vector{_PyVal}, inputs::Vector{_PyVal} = args)
    types = Any[_jax_argument_type(a) for a in args]
    key = (:host_traceable, objectid(f), Tuple(a.T for a in args))
    scope = _JaxScope()
    params, passed = String[], _PyVal[]
    locals = Dict{Symbol,_PyVal}(n => _host_function_input!(scope, params, passed, n, a)
                                for (n, a) in zip(names, inputs))
    name = get(ctx.mod.helper_names, key, nothing)
    if name === nothing
        name = _jax_private_identifier(ctx.mod.used, Symbol("_rk_", nameof_or(f), "_np"))
        ctx.mod.helper_names[key] = name
        inner = _JaxContext(ctx.mod, scope, ctx.recipe, mod, locals, nothing)
        inner.host = _HostState(Set{String}(params); in_function = true)
        # A list it ends with (`unique!(sort!(v))`) leaves as an array.
        v = _host_read(inner, _jax_expr(inner, body))
        lines = String["def " * name * "(" * join(params, ", ") * "):"]
        append!(lines, scope.lines)
        v.T === Union{} || push!(lines, "    return " * v.code)
        push!(ctx.mod.helpers, _host_helper_text(ctx, join(lines, "\n")))
        inferred = _jax_return_type(f, types)
        T = inferred isa Type && inferred !== Union{} && isconcretetype(inferred) ? inferred :
            foldl(_host_join, inner.host.returns; init = v.T)
        ctx.mod.helper_names[(:host_traceable_type, name)] = T
    end
    T = ctx.mod.helper_names[(:host_traceable_type, name)]
    _jax_bind!(ctx, :value, _PyVal(name * "(" * _jax_args(passed) * ")", T, true))
end

# --- prepare_data ---------------------------------------------------------------------

# The host-side helper that pads a ragged value (a list of arrays) for the
# JAX functions.
# A prepared integer scalar (a parameter block's end, a scan length, a level
# count) sets shapes in the functions, so JAX must know it while it traces,
# also when the prepared data are `jax.jit` arguments. `prepare_data` returns
# it as a `StaticInt`: an `int` that `jax.jit` treats as static (a pytree
# node without leaves), retracing only for a new value.
const _JAX_STATIC_INT = """
class StaticInt(int):
    \"\"\"An integer of the prepared data that sets a shape.

    It is an `int` everywhere, and `jax.jit` treats it as static, so the
    functions jit with the prepared data as arguments; a new value traces
    them again.
    \"\"\"

    def __repr__(self):
        return "StaticInt(%d)" % int(self)


jax.tree_util.register_pytree_node(StaticInt, lambda s: ((), int(s)), lambda v, _: StaticInt(v))
"""

# Static bounds of prepared integer arrays, for ranges whose ends are their
# entries (`_jax_trip_bound`): `prepare_data` returns them with the data. An
# array without entries gives 0 (no range over its entries runs) and a span
# of -1 (no steps).
const _JAX_PREPARED_BOUNDS = Dict{Symbol,Tuple{String,String}}(
    :min => ("_rk_bound_min", """
def _rk_bound_min(a):
    \"\"\"The least entry of a prepared integer array, static (0 without entries).\"\"\"
    a = np.asarray(a)
    return StaticInt(int(a.min()) if a.size else 0)
"""),
    :max => ("_rk_bound_max", """
def _rk_bound_max(a):
    \"\"\"The largest entry of a prepared integer array, static (0 without entries).\"\"\"
    a = np.asarray(a)
    return StaticInt(int(a.max()) if a.size else 0)
"""),
    :span => ("_rk_bound_span", """
def _rk_bound_span(first, last):
    \"\"\"The largest `last - first` of two prepared integer arrays' entries at one
    position (over the positions both have), static: the steps of the longest
    range `first[i]:last[i]`, less one (-1 without entries).
    \"\"\"
    first, last = np.asarray(first), np.asarray(last)
    if first.ndim != last.ndim:
        return StaticInt((int(last.max()) if last.size else 0) -
                         (int(first.min()) if first.size else 0))
    block = tuple(slice(0, min(m, n)) for m, n in zip(first.shape, last.shape))
    d = last[block] - first[block]
    return StaticInt(int(d.max()) if d.size else -1)
"""))

# `prepare_data`'s `lines` with the static bounds `bounds` (`(name, kind,
# keys)`, `_jax_prepared_bounds`) added to the dict it returns, computed from
# the prepared arrays they bound.
function _jax_prepare_bounds_lines(mod::_JaxModule, lines::Vector{String}, bounds)
    isempty(bounds) && return lines
    last_line = lines[end]
    prefix = "    return {"
    startswith(last_line, prefix) || error("prepare_data does not end in its dict")
    for helper in ("StaticInt" => _JAX_STATIC_INT, (_JAX_PREPARED_BOUNDS[k] for (_, k, _) in bounds)...)
        name, text = helper
        haskey(mod.helper_names, (:helper, name)) && continue
        mod.helper_names[(:helper, name)] = name
        push!(mod.used, name)
        push!(mod.helpers, rstrip(text))
    end
    result = "_rk_prepared"
    out = lines[1:end-1]
    push!(out, "    " * result * " = " * last_line[length("    return ")+1:end])
    for (name, kind, keys) in bounds
        push!(out, "    " * result * "[\"" * name * "\"] = " * first(_JAX_PREPARED_BOUNDS[kind]) * "(" *
                   join((result * "[\"" * k * "\"]" for k in keys), ", ") * ")")
    end
    push!(out, "    return " * result)
    out
end

const _JAX_PAD_ROWS = """
def _rk_pad_rows(rows, dtype):
    # A ragged value as the functions take it: the padded array (each row
    # repeats its last entry; an empty row is zeros), the lengths, and the
    # positions of the valid entries in the flattened padded array, whose
    # length is the total. `dtype` is the element type the program declares,
    # so an empty row (`[]`) keeps an integer column integer.
    rows = [np.asarray(r, dtype=dtype) for r in rows]
    lengths = np.asarray([r.shape[0] for r in rows], dtype=np.int64)
    width = max([0] + [int(n) for n in lengths])
    padded = np.stack([np.pad(r, (0, width - r.shape[0]), mode="edge") if r.shape[0] > 0
                       else np.zeros(width, dtype=dtype) for r in rows]) if rows \
        else np.zeros((0, width), dtype=dtype)
    flat = np.flatnonzero(np.arange(width)[None, :] < lengths[:, None])
    return padded, lengths, flat
"""

# --- plate cells' data-only recipes -------------------------------------------------
#
# A top-level plate whose cell reads data (a bound input, or a value
# `prepare_data` computes) runs whole in the functions. A data-only recipe of
# its cell that the functions cannot compute (a `@traceable` helper that builds
# a list, sorts, or loops to a data length) is computed by `prepare_data`
# instead, as native preparation with bound data evaluates a cell's data-only
# recipes once per cell (`_partial_plate_recipe`, partial_evaluation.jl): a
# host loop over the cells calls a function of the data-only recipes it needs,
# and the plate reads each cell's value as an input of its own. The values of
# every cell are a list, so a vector per cell is a ragged input (padded, with
# its lengths). A recipe the functions can compute stays in the cell, where a
# length read from a padded row keeps the static limit the row gives it.
#
# Only a top-level plate's own cell recipes are split, not those of a plate
# nested in the cell, a scan step or a branch arm.

# The cell recipes of a plate that `prepare_data` computes: `plan` (HAVE the
# plate's data inputs it reads, in order; WANT the values the rest of the
# cell reads), each input passed whole (`Ref`) or iterated, whether it runs
# per cell (it reads an iterated input) or once, the plate recipe's id, and
# the ids of the cell recipes split off (`targets`).
struct _JaxCellPrefix
    plan::Plan
    atomic::Vector{Bool}
    percell::Bool
    plate::Int
    targets::Vector{Int}
end

# The id of the `prepare_data` recipe of plate `plate`'s cell values: apart
# from every graph recipe's id.
const _JAX_CELL_PREFIX_BASE = -(Int(1) << 40)
_jax_cell_prefix_id(plate::Int) = _JAX_CELL_PREFIX_BASE - plate

# A data-only recipe of a plate's cell that the functions cannot compute
# (`plate` and `recipe` are recipe ids): the module is emitted again with
# `prepare_data` computing it.
struct _CellHostNeeded <: Exception
    plate::Int
    recipe::Int
    error::JAXEmissionError
end
# Cell recipes split off to `prepare_data` (`recipes`) whose computation
# there has no host translation (`error`): they stay in the cell.
struct _HostUnsupportedCell <: Exception
    plate::Int
    recipes::Vector{Int}
    error::JAXEmissionError
end

# Plate recipe `r` split so that `prepare_data` computes the cell recipes
# `hoisted` (ids in the cell's graph): `(; prefix, plate)`, the `prepare_data`
# recipe of their values and the plate reading them as inputs, or `nothing`
# when none of them reads only the plate's data inputs (those whose canonical
# id in `g` is in `dependent`).
function _jax_cell_split(g::Graph, r::Recipe, dependent::Set{ValueId}, hoisted::Set{Int})
    op = r.op
    body = op.kernel.plan
    atomic_inputs = typeof(op).parameters[2]
    length(body.have) == length(r.inputs) || return nothing
    cg = body.graph
    data = Bool[canon_id(g, v.id) in dependent for v in r.inputs]
    bound = Set{ValueId}(canon_id(cg, body.have[i].id) for i in eachindex(data) if data[i])
    pre, _, _ = _partial_split(body, bound)
    targets = Recipe[x for x in pre if x.id in hoisted]
    isempty(targets) && return nothing
    target_ids = Set{Int}(x.id for x in targets)
    # The targets and the data-only recipes they read, transitively.
    selected = copy(target_ids)
    needed = Set{ValueId}(canon_id(cg, v.id) for x in targets for v in x.inputs)
    for x in Iterators.reverse(pre)
        x.id in selected && continue
        any(o -> canon_id(cg, o.id) in needed, x.outputs) || continue
        push!(selected, x.id)
        foreach(v -> push!(needed, canon_id(cg, v.id)), x.inputs)
    end
    computed = Recipe[x for x in pre if x.id in selected]
    # The rest of the cell, without the recipes only the targets read.
    need = Set{ValueId}(canon_id(cg, v.id) for v in body.want)
    rest = Recipe[]
    for x in Iterators.reverse(body.recipes)
        x.id in target_ids && continue
        any(o -> canon_id(cg, o.id) in need, x.outputs) || continue
        push!(rest, x)
        foreach(v -> push!(need, canon_id(cg, v.id)), x.inputs)
    end
    reverse!(rest)
    # The targets' values the rest reads, in order of first use.
    produced = Set{ValueId}(canon_id(cg, o.id) for x in targets for o in x.outputs)
    frontier = Value[]
    seen = Set{ValueId}()
    for v in Iterators.flatten(((v for x in rest for v in x.inputs), body.want))
        cid = canon_id(cg, v.id)
        cid in produced && !(cid in seen) || continue
        push!(seen, cid)
        push!(frontier, get(cg.values, cid, v))
    end
    isempty(frontier) && return nothing
    # The plate's data inputs those recipes read: they run per cell when one
    # of them is iterated, else once, their values passed to every cell whole.
    read = Set{ValueId}(canon_id(cg, v.id) for x in computed for v in x.inputs)
    inputs = Int[i for i in eachindex(data)
                 if data[i] && canon_id(cg, body.have[i].id) in read]
    atomic = Bool[i in atomic_inputs for i in inputs]
    percell = !all(atomic)
    plan = _partial_subplan(body, Value[body.have[i] for i in inputs], frontier, computed)
    values = Value[Value(v.name, Any) for v in frontier]
    prefix = Recipe(_jax_cell_prefix_id(r.id), Tuple(r.inputs[i] for i in inputs),
                    Tuple(values),
                    _JaxCellPrefix(plan, atomic, percell, r.id, Int[x.id for x in targets]),
                    0.0, nothing, false)
    cell = _partial_subplan(body, Value[body.have; frontier], body.want, rest)
    kernel = (; plan = cell)
    added = percell ? () : Tuple(length(r.inputs) .+ eachindex(values))
    plate = Recipe(r.id, (r.inputs..., values...), r.outputs,
                   _AuthoredPlateOp{typeof(kernel),(atomic_inputs..., added...)}(
                       kernel, op.axis_checks),
                   r.cost, r.cse_key, r.effectful, r.source)
    (; prefix, plate)
end

# `_partial_constants` that also knows the values only `prepare_data`
# computes (`hosted`, by id), which the program's graph does not hold.
function _jax_partial_constants(p::Plan, owned::Set{ValueId}, residual::Vector{Recipe},
                                hosted::Dict{ValueId,Value})
    isempty(hosted) && return _partial_constants(p, owned, residual)
    g = p.graph
    constants = Value[]
    seen = Set{ValueId}()
    for v in Iterators.flatten(((v for r in residual for v in r.inputs), p.want))
        cid = canon_id(g, v.id)
        cid in owned && !(cid in seen) || continue
        push!(seen, cid)
        push!(constants, get(() -> g.values[cid], hosted, cid))
    end
    constants
end

# A plate cell's data-only recipes in `prepare_data`: a function of them,
# called on each cell's data (or once, when they read no iterated input),
# and each of its values for every cell as a list.
function _jax_recipe(ctx::_JaxContext, p::Plan, recipe::Recipe, op::_JaxCellPrefix, inputs)
    ctx.host === nothing && _jax_error("a plate cell's data-only recipes outside `prepare_data`",
                                       _jax_where(ctx))
    inputs = [!op.atomic[i] && _jax_numeric_tuple(v.T) ?
              _PyVal("np.stack(" * v.code * ")", Vector{eltype(v.T)}, true) : v
              for (i, v) in enumerate(inputs)]
    mapped = Bool[op.percell && !op.atomic[i] && _jax_is_array(v.T)
                  for (i, v) in enumerate(inputs)]
    op.percell && !any(mapped) && _jax_error("a plate cell's data-only recipes that read ",
        "no array of its cells", _jax_where(ctx))
    ranks = unique(_jax_rank(inputs[i].T) for i in eachindex(inputs) if mapped[i])
    isempty(ranks) || ranks == [1] || _jax_error("a plate over arrays of rank ",
        join(string.(ranks), ", "), " whose cell's data-only recipes have no host translation",
        _jax_where(ctx))
    types = Any[mapped[i] ? eltype(v.T) : _jax_unref(v.T) for (i, v) in enumerate(inputs)]
    cell, wants, _ = _jax_plan_function!(ctx, :cell_data, op.plan, types)
    targets = String[]
    args = String[]
    for (i, v) in enumerate(inputs)
        if mapped[i]
            t = _jax_fresh(ctx, :element)
            push!(targets, t); push!(args, t)
        else
            push!(args, v.code)
        end
    end
    call = cell * "(" * join(args, ", ") * ")"
    n = length(wants)
    if !op.percell
        n == 1 && return [_PyVal(call, _jax_typeof(only(wants)), true)]
        shared = _jax_bind!(ctx, :cell_data, _PyVal(call, Any, true))
        return [_PyVal(shared.code * "[" * string(j - 1) * "]", _jax_typeof(wants[j]), true)
                for j in 1:n]
    end
    sources = [inputs[i].code for i in eachindex(inputs) if mapped[i]]
    iterable = length(sources) == 1 ? only(sources) : "zip(" * join(sources, ", ") * ")"
    each = "[" * call * " for " * join(targets, ", ") * " in " * iterable * "]"
    n == 1 && return [_host_collected(each, _jax_typeof(only(wants)))]
    rows = _jax_bind!(ctx, :cell_data, _PyVal(each, Any, true))
    row = _jax_fresh(ctx, :row)
    [_host_collected("[" * row * "[" * string(j - 1) * "] for " * row * " in " * rows.code * "]",
                     _jax_typeof(wants[j]))
     for j in 1:n]
end

# The cell recipes split off by `op` that its host translation's error `e`
# leaves to the functions: those reading the cell recipe that raised it,
# transitively, or all of them when it was raised outside the cell's recipes.
function _jax_cell_refused(mod::_JaxModule, op::_JaxCellPrefix, e::JAXEmissionError)
    path = get(mod.failure_paths, e, Tuple{Plan,Recipe}[])
    at = findfirst(step -> first(step) === op.plan, path)
    at === nothing && return copy(op.targets)
    g = op.plan.graph
    failed = Set{ValueId}(canon_id(g, o.id) for o in last(path[at]).outputs)
    refused = Set{Int}((last(path[at]).id,))
    for x in op.plan.recipes
        any(v -> canon_id(g, v.id) in failed, x.inputs) || continue
        push!(refused, x.id)
        foreach(o -> push!(failed, canon_id(g, o.id)), x.outputs)
    end
    targets = Int[t for t in op.targets if t in refused]
    isempty(targets) ? copy(op.targets) : targets
end

# Remember data-only cell recipes even when their device translation works:
# a later loop may need their integer result as a prepared static bound.
function _jax_cell_data!(mod::_JaxModule, split, rp::Plan)
    prepared = Set{ValueId}(canon_id(rp.graph, v.id) for v in split.constants)
    for plate in rp.recipes
        plate.op isa _AuthoredPlateOp || continue
        cell = plate.op.kernel.plan
        length(cell.have) == length(plate.inputs) || continue
        bound = Set{ValueId}(canon_id(cell.graph, cell.have[i].id) for i in eachindex(plate.inputs)
                             if canon_id(rp.graph, plate.inputs[i].id) in prepared)
        pre, _, _ = _partial_split(cell, bound)
        union!(mod.cell_data, ((plate.id, r.id) for r in pre))
    end
end

# The error `e` of a module function `rp` (the residual plan of `split`)
# raised by a data-only recipe of a top-level plate's cell: a
# `_CellHostNeeded` for it, or, when `prepare_data` has no host translation
# of it either, the error naming both reasons. `nothing` for any other.
function _jax_cell_failure(mod::_JaxModule, split, rp::Plan, e::JAXEmissionError)
    path = get(mod.failure_paths, e, nothing)
    (path === nothing || length(path) < 2) && return nothing
    (outer, plate), (cell, recipe) = path[1], path[2]
    outer === rp && plate.op isa _AuthoredPlateOp && cell === plate.op.kernel.plan ||
        return nothing
    length(cell.have) == length(plate.inputs) || return nothing
    g = rp.graph
    prepared = Set{ValueId}(canon_id(g, v.id) for v in split.constants)
    cg = cell.graph
    bound = Set{ValueId}(canon_id(cg, cell.have[i].id) for i in eachindex(plate.inputs)
                         if canon_id(g, plate.inputs[i].id) in prepared)
    pre, _, _ = _partial_split(cell, bound)
    any(x -> x.id == recipe.id, pre) || return nothing
    reason = get(mod.cell_reasons, (plate.id, recipe.id), nothing)
    reason === nothing && return _CellHostNeeded(plate.id, recipe.id, e)
    JAXEmissionError(rstrip(e.message, '.') * ". The recipe reads data alone, and " *
                     "`prepare_data` does not compute it on the host either: " * reason.message)
end

"""
    _jax_prepare_split(plans, data_ports) -> (; prefix, constants, residual)

The data-only part common to the module's plans (`plans`, function name =>
(plan, WANT)) over the HAVE ports named `data_ports`: the recipes that read
data alone (in an order that computes each after its inputs), the boundary
values the functions read from them, and each function's residual plan over
its remaining ports followed by those values.
"""
function _jax_prepare_split(plans, data_ports, excluded::Set{Int} = Set{Int}(); keep = (),
                            cells = Dict{Int,Set{Int}}())
    p1 = first(plans).second[1]
    g = p1.graph
    prefix = Recipe[]
    seen_recipes = Set{Int}()
    constants = Value[]
    seen_values = Set{ValueId}()
    boundaries = []
    # The plates whose cells `prepare_data` computes in part (`cells`), split
    # once for every function that has them.
    rewrites = Dict{Int,Any}()
    hosted = Dict{ValueId,Value}()
    for (fname, (p, w)) in plans
        bound = Value[v for v in p.have if v.name in data_ports]
        ids = Set{ValueId}(canon_id(g, v.id) for v in bound)
        pre, residual, _ = _partial_split(p, ids, excluded)
        # A recipe of the prefix that reads no data (a literal, a prior's
        # scale) stays in the functions too, where JAX folds it: only values
        # computed from data cross to them.
        dependent = copy(ids)
        independent = Recipe[]
        for r in pre
            if any(v -> canon_id(g, v.id) in dependent, r.inputs)
                foreach(o -> push!(dependent, canon_id(g, o.id)), r.outputs)
            else
                push!(independent, r)
            end
        end
        # Of those, the ones the functions read.
        needed = Set{ValueId}(canon_id(g, v.id) for v in p.want)
        for r in residual, v in r.inputs
            push!(needed, canon_id(g, v.id))
        end
        kept = Recipe[]
        for r in Iterators.reverse(independent)
            any(o -> canon_id(g, o.id) in needed, r.outputs) || continue
            push!(kept, r)
            foreach(v -> push!(needed, canon_id(g, v.id)), r.inputs)
        end
        residual = [reverse!(kept); residual]
        # A plate whose cell has data-only recipes the functions cannot
        # compute: `prepare_data` computes them for every cell (a recipe after
        # the producers of the plate's data inputs, which `pre` holds), and the
        # plate reads their values as inputs.
        for (k, r) in enumerate(residual)
            haskey(cells, r.id) && r.op isa _AuthoredPlateOp || continue
            _jax_cell_prefix_id(r.id) in excluded && continue
            rewrite = get!(() -> _jax_cell_split(g, r, dependent, cells[r.id]), rewrites, r.id)
            rewrite === nothing && continue
            pre = [pre; rewrite.prefix]
            residual[k] = rewrite.plate
            for v in rewrite.prefix.outputs
                push!(dependent, canon_id(g, v.id))
                hosted[v.id] = v
            end
        end
        for r in pre
            r.id in seen_recipes && continue
            push!(seen_recipes, r.id)
            push!(prefix, r)
        end
        boundary = _jax_partial_constants(p, dependent, residual, hosted)
        for v in boundary
            cid = canon_id(g, v.id)
            cid in seen_values && continue
            push!(seen_values, cid)
            push!(constants, v)
        end
        w === nothing || push!(boundaries, (fname, p, w, residual))
    end
    # The data values `keep` names are boundary values too, which
    # `prepare_data` returns although no function reads them (the layout's
    # data-derived positions, read by the caller).
    for name in keep
        cid = canon_id(g, _jax_keep_value(g, name).id)
        found = [o for r in prefix if !(r.op isa _JaxCellPrefix) for o in r.outputs
                 if canon_id(g, o.id) == cid]
        isempty(found) && append!(found, (v for v in p1.have
            if canon_id(g, v.id) == cid && v.name in data_ports))
        v = isempty(found) ? nothing : first(found)
        # The caller refuses it, unless a definition gives a value it reads
        # (`_jax_module_source`).
        v === nothing && throw(_KeepUnavailable(name))
        cid in seen_values && continue
        push!(seen_values, cid)
        push!(constants, v)
    end
    # Only the recipes the boundary values need run in `prepare_data`.
    needed = Set{ValueId}(canon_id(g, v.id) for v in constants)
    kept = Recipe[]
    for r in Iterators.reverse(prefix)
        any(o -> canon_id(g, o.id) in needed, r.outputs) || continue
        push!(kept, r)
        foreach(v -> push!(needed, canon_id(g, v.id)), r.inputs)
    end
    prefix = reverse!(kept)
    raw = Value[v for v in p1.have if v.name in data_ports]
    remaining = Value[v for v in p1.have if !(v.name in data_ports)]
    residuals = [fname => (_partial_subplan(p, [remaining; constants], p.want, residual), w)
                 for (fname, p, w, residual) in boundaries]
    left = Recipe[]
    for (_, (p, _)) in plans, r in p.recipes
        r.id in excluded && !any(x -> x.id == r.id, left) && push!(left, r)
    end
    (; prefix, constants, raw, remaining, residuals, excluded = left,
       plan = _partial_subplan(p1, raw, constants, prefix))
end

# A data-only recipe that has no host translation: the JAX functions compute
# it instead.
struct _HostUnsupported <: Exception
    recipe::Int
    error::JAXEmissionError
end
# A data definition (`jax_source`'s `definitions`) without a host translation:
# `prepare_data` takes its value as an input instead.
struct _HostUnsupportedDefinition <: Exception
    name::Symbol
    error::JAXEmissionError
end

const _JAX_FIELDS = """
def _rk_fields(x, names):
    # A named-tuple input of `prepare_data` (the value of a definition it takes
    # as an input, `UNTRANSLATED`): a dict by field name, or its values in
    # field order.
    if isinstance(x, dict):
        return tuple(x[n] for n in names)
    x = tuple(x)
    if len(x) != len(names):
        raise ValueError("expected the fields (%s), got %d values" % (", ".join(names), len(x)))
    return x
"""

const _JAX_KEYWORD_INPUTS = """
def _rk_keyword_inputs(data, names):
    # The `prepare_data` inputs no parameter can be named for (`class`), which
    # it takes by their own names through `**data`: their values, in `names`
    # order. Another key, or a missing one, raises as a parameter would.
    unexpected = [k for k in data if k not in names]
    if unexpected:
        raise TypeError("prepare_data() got unexpected keyword arguments: " + ", ".join(map(repr, unexpected)))
    missing = [n for n in names if n not in data]
    if missing:
        raise TypeError("prepare_data() missing required arguments: " + ", ".join(map(repr, missing)))
    return tuple(data[n] for n in names)
"""

# A raw `prepare_data` input as the host code reads it: a numeric array as a
# NumPy array of its element type, a ragged one as a list of them, and a named
# tuple (the value of a definition without a host translation, `g =
# prep(labels)`) as the tuple of its fields, each read the same way. Returns
# the expression; a named tuple's fields are read into a local first.
function _host_input!(ctx::_JaxContext, code::AbstractString, @nospecialize(T))
    _jax_ragged(T) && return "[np.asarray(a, dtype=" * _host_dtype(eltype(eltype(T))) *
        ") for a in " * code * "]"
    _jax_is_array(T) && return "np.asarray(" * code * ", dtype=" * _host_dtype(eltype(T)) * ")"
    T isa Type && T <: NamedTuple && isconcretetype(T) || return code
    mod = ctx.mod
    if !haskey(mod.helper_names, (:helper, "_rk_fields"))
        mod.helper_names[(:helper, "_rk_fields")] = "_rk_fields"
        push!(mod.used, "_rk_fields")
        push!(mod.helpers, rstrip(_JAX_FIELDS))
    end
    names = String.(fieldnames(T))
    fields = _jax_fresh(ctx, occursin(r"^[A-Za-z_][A-Za-z0-9_]*$", code) ?
                             Symbol(code, :_fields) : :fields)
    _jax_emit!(ctx, fields * " = _rk_fields(" * code * ", (" * join(repr.(names), ", ") *
                    (length(names) == 1 ? ",))" : "))"))
    parts = [_host_input!(ctx, fields * "[" * string(k - 1) * "]", fieldtype(T, k))
             for k in 1:fieldcount(T)]
    "(" * join(parts, ", ") * (length(parts) == 1 ? ",)" : ")")
end

# `prepare_data` over the raw data ports of `split`: the host code of its
# prefix, returning a `dict` of the boundary values under `names` (a ragged
# value padded, with its `<name>_lengths`). `types` are the boundary values'
# types, or `nothing` for the types the translation infers. Returns the
# function's lines and the boundary values' types.
function _jax_prepare_function!(mod::_JaxModule, split, raw_types, names, types;
                                params = split.plan.have, definitions = (),
                                untranslated = Dict{Symbol,JAXEmissionError}())
    p = split.plan
    scope = _JaxScope()
    ctx = _JaxContext(mod, scope, nothing, Main, Dict{Symbol,_PyVal}(), nothing)
    ctx.host = _HostState()
    # The caller passes the data by these names: each input's key is its own
    # name (`_jax_key_name`), held in a Python local of the same name. A name
    # Python reserves (`class`) is held in a respelled local (`class_`), and
    # `prepare_data` takes that input by its own name through `**data`.
    param_names = String[]
    input_locals = String[]
    keys_used = Set{String}()
    for v in params
        key = _jax_key_name(keys_used, v.name; stable = false)
        push!(param_names, key)
        push!(input_locals, _jax_param_name(scope.used, key; stable = false))
    end
    keyword = [k for k in eachindex(param_names) if param_names[k] != input_locals[k]]
    keyword_data = isempty(keyword) ? "" : _jax_identifier(scope.used, :data)
    if !isempty(keyword)
        if !haskey(mod.helper_names, (:helper, "_rk_keyword_inputs"))
            mod.helper_names[(:helper, "_rk_keyword_inputs")] = "_rk_keyword_inputs"
            push!(mod.used, "_rk_keyword_inputs")
            push!(mod.helpers, rstrip(_JAX_KEYWORD_INPUTS))
        end
        py_tuple(xs) = join(xs, ", ") * (length(xs) == 1 ? "," : "")
        _jax_emit!(ctx, py_tuple(input_locals[keyword]) * " = _rk_keyword_inputs(" * keyword_data *
                        ", (" * py_tuple(repr.(param_names[keyword])) * "))")
    end
    known = Dict{Symbol,_PyVal}()
    for (v, T, param) in zip(params, raw_types, input_locals)
        known[v.name] = _PyVal(param, T, true)
        code = _host_input!(ctx, param, T)
        code == param || _jax_emit!(ctx, param * " = " * code)
    end
    # The data definitions, in order, over the parameters and the definitions
    # before them. A data value takes the type the binding's own value has; a
    # local of the definitions (`T0 === nothing`), the translation's. Their
    # globals resolve in the module the caller names (`definitions_module`),
    # never in whichever module called `jax_source`.
    ctx.source_module = mod.definitions_module
    for (name, ex, T0) in definitions
        # One reading a definition without a host translation that is no input
        # either (the caller gave no value of it) has none: it says why.
        for r in _jax_symbols_read!(Symbol[], ex)
            haskey(untranslated, r) && !haskey(known, r) &&
                throw(_HostUnsupportedDefinition(name, JAXEmissionError(
                    "it reads `" * string(r) * "`, which `prepare_data` neither computes " *
                    "nor takes as an input: " * untranslated[r].message)))
        end
        ctx.locals = copy(known)
        first_line = length(scope.lines) + 1
        T = T0
        local_name = try
            v = _jax_expr(ctx, ex)
            T === nothing && (T = v.T)
            local_name = _jax_fresh(ctx, name)
            _jax_emit!(ctx, local_name * " = " * (_host_is_numeric_array(T) ?
                "np.asarray(" * v.code * ", dtype=" * _host_dtype(eltype(T)) * ")" : v.code))
            # Host text, as each recipe's (`_host_lines!`).
            _host_lines!(ctx, first_line)
            local_name
        catch e
            e isa JAXEmissionError || rethrow()
            throw(_HostUnsupportedDefinition(name, e))
        finally
            ctx.locals = Dict{Symbol,_PyVal}()
        end
        known[name] = _PyVal(local_name, T, true)
    end
    ctx.source_module = Main
    for v in p.have
        scope.values[canon_id(p.graph, v.id)] = known[v.name]
    end
    for recipe in p.recipes
        try
            _jax_plan_recipe!(ctx, p, recipe)
        catch e
            e isa JAXEmissionError || rethrow()
            recipe.op isa _JaxCellPrefix || throw(_HostUnsupported(recipe.id, e))
            throw(_HostUnsupportedCell(recipe.op.plate,
                                       _jax_cell_refused(mod, recipe.op, e), e))
        end
    end
    values = [_jax_lookup(ctx, p.graph, v) for v in split.constants]
    # A value without a binding's type (`types[i] === nothing`, one only
    # `prepare_data` computes) takes the translation's.
    types = types === nothing ? Any[v.T for v in values] :
        Any[T === nothing ? v.T : T for (v, T) in zip(values, types)]
    entries = String[]
    for (v, T, name) in zip(values, types, names)
        if _jax_ragged(T)
            if !haskey(mod.helper_names, (:helper, "_rk_pad_rows"))
                mod.helper_names[(:helper, "_rk_pad_rows")] = "_rk_pad_rows"
                push!(mod.helpers, rstrip(_JAX_PAD_ROWS))
            end
            padded = _jax_fresh(ctx, Symbol(name, :_padded))
            lengths = _jax_fresh(ctx, Symbol(name, :_lengths))
            flat = _jax_fresh(ctx, Symbol(name, :_flat))
            _jax_emit!(ctx, padded * ", " * lengths * ", " * flat * " = _rk_pad_rows(" * v.code *
                            ", " * _host_dtype(eltype(eltype(T))) * ")")
            push!(entries, "\"" * name * "\": " * padded, "\"" * name * "_lengths\": " * lengths,
                  "\"" * name * "_flat\": " * flat)
        elseif _host_is_numeric_array(T)
            push!(entries, "\"" * name * "\": np.asarray(" * v.code * ", dtype=" *
                           _host_dtype(eltype(T)) * ")")
        elseif T isa Type && T <: Integer
            if !haskey(mod.helper_names, (:helper, "StaticInt"))
                mod.helper_names[(:helper, "StaticInt")] = "StaticInt"
                push!(mod.used, "StaticInt")
                push!(mod.helpers, rstrip(_JAX_STATIC_INT))
            end
            push!(entries, "\"" * name * "\": StaticInt(" * v.code * ")")
        else
            push!(entries, "\"" * name * "\": " * v.code)
        end
    end
    annotation(T) = _jax_ragged(T) ? "list" : T isa Type && T <: NamedTuple ? "dict" : _jax_annotation(T)
    signature = String[n * ": " * annotation(T) for (k, (n, T)) in enumerate(zip(param_names, raw_types))
                       if !(k in keyword)]
    if !isempty(keyword)
        kinds = unique(annotation(T) for T in raw_types[keyword])
        push!(signature, "**" * keyword_data * ": " * (length(kinds) == 1 ? only(kinds) : "Any"))
    end
    lines = String["def prepare_data(" * join(signature, ", ") * ") -> dict:"]
    keyword_doc = isempty(keyword) ? "" :
        "\n    `**" * keyword_data * "` takes " * join(("`" * n * "`" for n in param_names[keyword]), ", ") *
        " by name, which no parameter can have:\n    pass the data by name, `prepare_data(**raw)`.\n"
    push!(lines, "    \"\"\"The functions' data inputs, computed from the raw data once per dataset.\n\n" *
                 "    Pass the data by the names the model reads them; a list of arrays is a ragged\n" *
                 "    column. Call the other functions with the result: `f(u, **prepared)`.\n" *
                 keyword_doc * "    \"\"\"")
    for r in split.excluded
        push!(lines, "    # Computed by the functions (no host translation): " * _recipe_line(r))
        reason = get(mod.host_reasons, r.id, nothing)
        reason === nothing ||
            push!(lines, "    #   " * replace(reason.message, r"\s*\n\s*" => " "))
    end
    append!(lines, scope.lines)
    push!(lines, "    return {" * join(entries, ", ") * "}")
    lines, types, param_names
end
