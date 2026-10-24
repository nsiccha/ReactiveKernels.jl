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
    lists::Set{String}   # names holding Python lists (`push!` targets)
    in_function::Bool    # a function body, where `return` is a statement
    depth::Int           # nesting in conditionals and loops
    returns::Vector{Any} # the types an explicit `return` returns
end
_HostState(frame::Set{String} = Set{String}(); in_function::Bool = false) =
    _HostState(frame, Set{String}(), Set{String}(), in_function, 0, Any[])

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
# A helper's host text; a JAX construct left in it has no host translation.
function _host_helper_text(ctx::_JaxContext, text::AbstractString)
    text = _host_text(text)
    for line in split(text, '\n')
        occursin(r"\b(?:jax|lax)\.|\.at\[", replace(line, _PY_STRING => "\"\"")) &&
            _jax_error("the line `", strip(line), "` has no host NumPy translation",
                       _jax_where(ctx))
    end
    text
end
_host_names(s::AbstractString) = foldl((t, r) -> replace(t, r), _HOST_NAMES; init = String(s))

# Rewrite the lines a recipe emitted from `first` on; a JAX construct left in
# them has no host translation.
function _host_lines!(ctx::_JaxContext, first::Int)
    lines = ctx.scope.lines
    for i in first:length(lines)
        lines[i] = _host_text(lines[i])
        code = replace(lines[i], _PY_STRING => "\"\"")
        occursin(r"\b(?:jax|lax)\.|\.at\[", code) &&
            _jax_error("the line `", strip(lines[i]), "` has no host NumPy translation",
                       _jax_where(ctx))
    end
end

_host_dtype(@nospecialize(E)) = E isa Type && E <: Bool ? "np.bool_" :
    E isa Type && E <: Integer ? "np.int64" : E isa Type && E <: AbstractFloat ? "np.float64" :
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
_host_read(ctx::_JaxContext, v::_PyVal) = v.code in ctx.host.lists ?
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
    v.T isa _JaxCallable && (ctx.locals[lhs] = v; return)
    v.T === Union{} && return
    h = ctx.host
    name = _host_name!(ctx, lhs)
    name == v.code || _jax_emit!(ctx, name * " = " * v.code)
    v.code == "[]" || v.code in h.lists ? push!(h.lists, name) : delete!(h.lists, name)
    _host_fresh(v.code) || v.code in h.safe ? push!(h.safe, name) : delete!(h.safe, name)
    ctx.locals[lhs] = _PyVal(name, v.T, true)
    nothing
end

function _host_assign!(ctx::_JaxContext, lhs, v::_PyVal)
    lhs isa Symbol && return _host_bind!(ctx, lhs, v)
    lhs isa Expr && lhs.head === :ref && lhs.args[1] isa Symbol && haskey(ctx.locals, lhs.args[1]) &&
        return _host_write!(ctx, lhs, v)
    _jax_error("the assignment target `", lhs, "` has no host translation", _jax_where(ctx))
end

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
    indexed = _jax_index(ctx, target, lhs.args[2:end])
    startswith(indexed.code, target.code * "[") ||
        _jax_error("the indexed write `", lhs, "` has no host translation", _jax_where(ctx))
    _jax_emit!(ctx, indexed.code * " = " * v.code)
    nothing
end

# The locals `exs` assign and the arrays they write by index.
function _host_targets(exs)
    assigned, written = Set{Symbol}(), Set{Symbol}()
    walk(ex) = begin
        ex isa Expr || return
        ex.head in (:quote, :inert, :->, :function) && return
        if (ex.head === :(=) || haskey(_JAX_UPDATING, ex.head)) && length(ex.args) == 2
            target = ex.args[1]
            target isa Expr && target.head === :(::) && (target = target.args[1])
            if target isa Expr && target.head === :ref
                while target isa Expr && target.head === :ref
                    target = target.args[1]
                end
                target isa Symbol && push!(written, target)
            elseif target isa Symbol
                push!(assigned, target)
            elseif target isa Expr && target.head === :tuple
                foreach(t -> t isa Symbol && push!(assigned, t), target.args)
            end
        end
        foreach(walk, ex.args)
    end
    foreach(walk, exs)
    assigned, written
end

# Before a loop or conditional: the outer locals its body assigns take names
# of this frame, so every pass assigns the same Python variable; arrays it
# writes are made this frame's own here, not on each pass. A conditional's new
# locals take their names here too, so both arms assign the same one.
function _host_prepare!(ctx::_JaxContext, bodies; conditional::Bool = false)
    assigned, written = _host_targets(bodies)
    for s in sort!(collect(union(assigned, written)))
        if haskey(ctx.locals, s)
            v = ctx.locals[s]
            v.T isa _JaxCallable && continue
            if s in written && !(v.T isa Type && v.T <: AbstractDict)
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
    v = _jax_expr(ctx, ex.args[1])
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
    end
    value ? _PyVal(result, _host_join(T1, T2), true) : _PyVal("None", Nothing, true)
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
            return "range(" * lo.code * ", " * _jax_plus_one(hi) * ")", Int
        elseif f === Colon() && length(args) == 3
            lo, step, hi = (_jax_expr(ctx, a) for a in args)
            stop = "(" * hi.code * ") + (1 if " * step.code * " > 0 else -1)"
            return "range(" * lo.code * ", " * stop * ", " * step.code * ")", Int
        elseif (f === eachindex || f === keys || f === values || f === pairs) && length(args) == 1
            x = _host_raw(ctx, args[1])
            if x.T isa Type && x.T <: AbstractDict
                f === pairs && return x.code * ".items()", Tuple{Base.keytype(x.T),Base.valtype(x.T)}
                f === values && return x.code * ".values()", Base.valtype(x.T)
                return x.code * ".keys()", Base.keytype(x.T)
            end
            f === values && return x.code, _jax_eltype(x.T)
            f === pairs && return "enumerate(" * x.code * ", 1)", Tuple{Int,_jax_eltype(x.T)}
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
    T = x.T
    T isa Type && T <: AbstractDict && return x.code * ".items()", Tuple{Base.keytype(T),Base.valtype(T)}
    T isa Type && T <: Tuple && return x.code, promote_type(fieldtypes(T)...)
    T isa Type && T <: AbstractArray && ndims(T) > 1 &&
        return "np.ravel(" * x.code * ", order=\"F\")", eltype(T)
    T isa Type && T <: AbstractArray || T isa Type && T <: AbstractRange ||
        _jax_error("iterating a `", T, "` has no host translation", _jax_where(ctx))
    x.code, eltype(T)
end

# Bind a loop's target (`i`, `(i, v)`) to fresh locals of element type `E`.
function _host_targets!(ctx::_JaxContext, lhs, @nospecialize(E))
    if lhs isa Symbol
        name = _jax_fresh(ctx, lhs)
        push!(ctx.host.frame, name)
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
    ctx.host = _HostState(Set{String}(names); in_function = true)
    ctx.scope = inner
    v = try
        for (p, n, T) in zip(params, names, types)
            ctx.locals[p] = _PyVal(n, T, true)
        end
        build()
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
function _host_generator(ctx::_JaxContext, gen::Expr)
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
# each collection, called on each.
function _host_do(ctx::_JaxContext, ex)
    call, lambda = ex.args
    call isa Expr && call.head === :call && lambda isa Expr && lambda.head === :-> ||
        _jax_error("the do-block `", ex, "` has no host translation", _jax_where(ctx))
    _jax_resolve(ctx, call.args[1]) === map ||
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
    _PyVal(c.code * (f === get ? ".get(" : ".setdefault(") * key.code * ", " * default.code * ")",
           Base.valtype(c.T), true)
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
        E = T isa UnionAll ? Float64 : eltype(T)
        shape, N = _jax_dims(ctx, T, dims)
        return _PyVal("np.zeros(" * shape * ", dtype=" * _host_dtype(E) * ")", Array{E,N}, true)
    end
    length(args) == 1 && !_jax_is_array(args[1].T) || return nothing
    T <: Bool && return _PyVal("bool(" * args[1].code * ")", Bool, true)
    T <: Integer && return _PyVal("int(" * args[1].code * ")", T, true)
    T <: AbstractFloat && return _PyVal("float(" * args[1].code * ")", T, true)
    nothing
end

# Scalar predicates of the host's values.
const _HOST_ISEQUAL = """
def _rk_isequal(a, b):
    # Julia's isequal on scalars: NaN equals NaN, and missing (None) equals
    # only missing.
    if a is None or b is None:
        return a is b
    return bool(a == b) or (a != a and b != b)
"""
function _jax_call_positional(ctx::_JaxContext, ::typeof(isequal), args::Vector{_PyVal})
    ctx.host === nothing && return _jax_function_call(ctx, isequal, args)
    name = _jax_helper!(ctx, "_rk_isequal", _HOST_ISEQUAL)
    _PyVal(name * "(" * _jax_args(args) * ")", Bool, true)
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

# A `@traceable` helper on the host: a module-level function of its body.
function _host_function_call(ctx::_JaxContext, f, names::Vector{Symbol}, body, mod::Module,
                             args::Vector{_PyVal})
    types = Any[a.T for a in args]
    key = (:host_traceable, objectid(f), Tuple(types))
    name = get(ctx.mod.helper_names, key, nothing)
    if name === nothing
        name = _jax_identifier(ctx.mod.used, Symbol("_rk_", nameof_or(f), "_np"))
        ctx.mod.helper_names[key] = name
        scope = _JaxScope()
        params = [_jax_identifier(scope.used, n) for n in names]
        inner = _JaxContext(ctx.mod, scope, ctx.recipe, mod,
                            Dict{Symbol,_PyVal}(n => _PyVal(p, T, true)
                                                for (n, p, T) in zip(names, params, types)),
                            nothing)
        inner.host = _HostState(Set{String}(params); in_function = true)
        v = _jax_expr(inner, body)
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
    _jax_bind!(ctx, :value, _PyVal(name * "(" * _jax_args(args) * ")", T, true))
end

# --- prepare_data ---------------------------------------------------------------------

# The host-side helper that pads a ragged value (a list of arrays) for the
# JAX functions.
const _JAX_PAD_ROWS = """
def _rk_pad_rows(rows):
    # A ragged value as the functions take it: the padded array (each row
    # repeats its last entry), the lengths, and the positions of the valid
    # entries in the flattened padded array, whose length is the total.
    rows = [np.asarray(r) for r in rows]
    lengths = np.asarray([r.shape[0] for r in rows], dtype=np.int64)
    width = max([1] + [int(n) for n in lengths])
    padded = np.stack([np.pad(r, (0, width - r.shape[0]), mode="edge") if r.shape[0] > 0
                       else np.zeros(width) for r in rows]) if rows else np.zeros((0, width))
    flat = np.flatnonzero(np.arange(width)[None, :] < lengths[:, None])
    return padded, lengths, flat
"""

"""
    _jax_prepare_split(plans, data_ports) -> (; prefix, constants, residual)

The data-only part common to the module's plans (`plans`, function name =>
(plan, WANT)) over the HAVE ports named `data_ports`: the recipes that read
data alone (in an order that computes each after its inputs), the boundary
values the functions read from them, and each function's residual plan over
its remaining ports followed by those values.
"""
function _jax_prepare_split(plans, data_ports, excluded::Set{Int} = Set{Int}())
    p1 = first(plans).second[1]
    g = p1.graph
    prefix = Recipe[]
    seen_recipes = Set{Int}()
    constants = Value[]
    seen_values = Set{ValueId}()
    boundaries = []
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
        for r in pre
            r.id in seen_recipes && continue
            push!(seen_recipes, r.id)
            push!(prefix, r)
        end
        boundary = _partial_constants(p, dependent, residual)
        for v in boundary
            cid = canon_id(g, v.id)
            cid in seen_values && continue
            push!(seen_values, cid)
            push!(constants, v)
        end
        push!(boundaries, (fname, p, w, residual))
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

# `prepare_data` over the raw data ports of `split`: the host code of its
# prefix, returning a `dict` of the boundary values under `names` (a ragged
# value padded, with its `<name>_lengths`). `types` are the boundary values'
# types, or `nothing` for the types the translation infers. Returns the
# function's lines and the boundary values' types.
function _jax_prepare_function!(mod::_JaxModule, split, raw_types, names, types)
    p = split.plan
    scope = _JaxScope()
    ctx = _JaxContext(mod, scope, nothing, Main, Dict{Symbol,_PyVal}(), nothing)
    ctx.host = _HostState()
    params = String[]
    for (v, T) in zip(p.have, raw_types)
        param = _jax_param_name(scope.used, v.name)
        push!(params, param)
        scope.values[canon_id(p.graph, v.id)] = _PyVal(param, T, true)
        _jax_ragged(T) ? _jax_emit!(ctx, param * " = [np.asarray(a) for a in " * param * "]") :
            _jax_is_array(T) && _jax_emit!(ctx, param * " = np.asarray(" * param * ", dtype=" *
                                           _host_dtype(eltype(T)) * ")")
    end
    for recipe in p.recipes
        try
            _jax_plan_recipe!(ctx, p, recipe)
        catch e
            e isa JAXEmissionError || rethrow()
            throw(_HostUnsupported(recipe.id, e))
        end
    end
    values = [_jax_lookup(ctx, p.graph, v) for v in split.constants]
    types = types === nothing ? Any[v.T for v in values] : types
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
            _jax_emit!(ctx, padded * ", " * lengths * ", " * flat * " = _rk_pad_rows(" * v.code * ")")
            push!(entries, "\"" * name * "\": " * padded, "\"" * name * "_lengths\": " * lengths,
                  "\"" * name * "_flat\": " * flat)
        elseif _host_is_numeric_array(T)
            push!(entries, "\"" * name * "\": np.asarray(" * v.code * ", dtype=" *
                           _host_dtype(eltype(T)) * ")")
        else
            push!(entries, "\"" * name * "\": " * v.code)
        end
    end
    lines = String["def prepare_data(" * join(params, ", ") * "):"]
    push!(lines, "    \"\"\"The functions' data inputs, computed from the raw data once per dataset.\"\"\"")
    for r in split.excluded
        push!(lines, "    # Computed by the functions (no host translation): " * _recipe_line(r))
    end
    append!(lines, scope.lines)
    push!(lines, "    return {" * join(entries, ", ") * "}")
    lines, types
end
