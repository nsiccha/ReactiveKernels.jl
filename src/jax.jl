# JAX source emission (todo `1l3nni7`; route: agent call `0rnbx63`).
#
# `jax_source` translates a planned program into a Python module whose
# function computes the plan's WANT values from its HAVE values with
# `jax.numpy`. The translation is structural: the plan is walked in execution
# order, an ordinary recipe becomes one assignment translated from its authored
# right-hand side, a plate becomes `jax.vmap` of its cell, a scan becomes
# `jax.lax.scan` of its step and a lazy branch becomes `jax.lax.cond` of its
# arms. Nothing is unrolled per element (`docs/src/constraints.md`, "Preserve
# data-dependent iteration"), and sizes are read from the inputs, so one
# emitted module serves every data size of the same program. Gradients are
# `jax.grad` of the emitted primal; derivative rules contribute only their
# primal cut. Construct the emitter cannot translate raises a
# `JAXEmissionError` naming the recipe and the construct.
#
# Conventions. Arrays keep Julia's dimension order (a Julia `n × p` matrix is
# an `(n, p)` array), and Julia's broadcasting, which aligns leading
# dimensions, is reproduced by padding trailing axes from the known ranks.
# Integer index data stay 1-based; the emitted code subtracts one where it
# indexes.

"""
    JAXEmissionError(message)

A program, or a construct in one of its recipes, that [`jax_source`](@ref)
cannot translate to JAX. The message names the recipe and the construct.
"""
struct JAXEmissionError <: Exception
    message::String
end
Base.showerror(io::IO, e::JAXEmissionError) = print(io, "JAXEmissionError: ", e.message)

_jax_error(message...) = throw(JAXEmissionError(string(message...)))

# A translated value: its Python expression and the Julia type it stands for.
# `static` marks a value known while JAX traces the function (a literal, a
# shape, or arithmetic on those): Python control flow may branch on it.
#
# Ragged values. An array of arrays of different lengths (a ragged plate's
# per-group data) is a padded array, one row per group, whose `extent` is the
# Python expression of its lengths vector. Inside a cell, one group's padded
# vector has the scalar length as its `extent`. Padding repeats each row's last
# valid entry, so every padded slot holds a valid value: computations over it
# stay finite, and reductions mask by length.
#
# `limit` is a static upper bound of a data-dependent integer (a padded group
# vector's length is at most its padded size): loops run to it and arrays of
# that length are allocated at it.
struct _PyVal
    code::String
    T::Any
    static::Bool
    extent::String
    limit::String
end
_PyVal(code::AbstractString, @nospecialize(T)) = _PyVal(String(code), T, false, "", "")
_PyVal(code::AbstractString, @nospecialize(T), static::Bool) =
    _PyVal(String(code), T, static, "", "")
_PyVal(code::AbstractString, @nospecialize(T), static::Bool, extent::AbstractString) =
    _PyVal(String(code), T, static, String(extent), "")
_jax_static(v::_PyVal) = _PyVal(v.code, v.T, true, v.extent, v.limit)
_jax_ragged(@nospecialize(T)) = T isa Type && T <: AbstractArray && eltype(T) isa Type &&
    eltype(T) <: AbstractArray
function _jax_extent(args)
    for a in args
        isempty(a.extent) || return a.extent
    end
    ""
end
# A padded vector's valid-entry mask.
_jax_mask(v::_PyVal) = "(jnp.arange(" * v.code * ".shape[0]) < " * v.extent * ")"

# The Python keywords and module names an emitted identifier must not take.
const _JAX_RESERVED = Set{String}(split("""
    False None True and as assert async await break class continue def del elif
    else except finally for from global if import in is lambda nonlocal not or
    pass raise return try while with yield jax jnp jsp lax np math tuple list dict len
    range sum min max abs str int float bool print type object id iter next map zip
    all any round pow filter sorted set""" ))

# One Python function body under construction.
mutable struct _JaxScope
    lines::Vector{String}
    indent::Int
    used::Set{String}
    values::Dict{ValueId,_PyVal}
    parent::Union{Nothing,_JaxScope}
end
_JaxScope(parent::Union{Nothing,_JaxScope} = nothing) =
    _JaxScope(String[], 1, parent === nothing ? Set{String}() : copy(parent.used),
              Dict{ValueId,_PyVal}(), parent)

# The whole module: its helper definitions and module-level constants.
mutable struct _JaxModule
    helpers::Vector{String}
    helper_names::Dict{Any,Any}
    used::Set{String}
    counter::Int
end
_JaxModule() = _JaxModule(String[], Dict{Any,Any}(), Set{String}(), 0)

# Translation state: the module, the current function scope, the current
# recipe (for messages) and the module that resolves its source's globals.
mutable struct _JaxContext
    mod::_JaxModule
    scope::_JaxScope
    recipe::Any
    source_module::Module
    locals::Dict{Symbol,_PyVal}
    captures::Any
    traced_branches::Int
end
_JaxContext(mod, scope, recipe, source_module, locals, captures) =
    _JaxContext(mod, scope, recipe, source_module, locals, captures, 0)

_jax_where(ctx::_JaxContext) = ctx.recipe isa Recipe ?
    " in recipe `" * _recipe_line(ctx.recipe) * "`" : ""

_jax_emit!(ctx::_JaxContext, line::AbstractString) =
    push!(ctx.scope.lines, "    "^ctx.scope.indent * line)

# A fresh Python identifier derived from `hint`.
function _jax_identifier(used::Set{String}, hint)
    text = replace(String(Symbol(hint)), r"[^A-Za-z0-9_]" => "_")
    text = replace(text, r"_+" => "_")
    text = strip(text, '_')
    isempty(text) && (text = "v")
    isdigit(first(text)) && (text = "v_" * text)
    text in _JAX_RESERVED && (text *= "_")
    candidate = text
    n = 1
    while candidate in used
        n += 1
        candidate = string(text, "_", n)
    end
    push!(used, candidate)
    candidate
end

function _jax_fresh(ctx::_JaxContext, hint)
    name = _jax_identifier(union(ctx.scope.used, ctx.mod.used), hint)
    push!(ctx.scope.used, name)
    name
end

# Bind `value` to a Python local holding `v`; a plain name is reused.
function _jax_bind!(ctx::_JaxContext, hint, v::_PyVal)
    occursin(r"^[A-Za-z_][A-Za-z0-9_]*$", v.code) && return v
    name = _jax_fresh(ctx, hint)
    _jax_emit!(ctx, name * " = " * v.code)
    _PyVal(name, v.T, v.static, v.extent, v.limit)
end

# --- types and ranks ----------------------------------------------------------------

_jax_rank(@nospecialize(T)) = T isa Type && T <: AbstractArray ? ndims(T) : 0
_jax_is_array(@nospecialize(T)) = T isa Type && T <: AbstractArray
_jax_is_scalar(@nospecialize(T)) = T isa Type && T <: Number
_jax_eltype(@nospecialize(T)) = T isa Type && T <: AbstractArray ? eltype(T) : T

function _jax_return_type(f, argtypes)
    T = try
        Core.Compiler.return_type(f, Tuple{argtypes...})
    catch
        Any
    end
    T
end

# --- literals -----------------------------------------------------------------------

_jax_literal(x::Bool) = _PyVal(x ? "True" : "False", Bool, true)
_jax_literal(x::Integer) = _PyVal(string(x), typeof(x), true)
function _jax_literal(x::AbstractFloat)
    isnan(x) && return _PyVal("jnp.nan", typeof(x), true)
    isinf(x) && return _PyVal(x > 0 ? "jnp.inf" : "(-jnp.inf)", typeof(x), true)
    text = repr(Float64(x))
    _PyVal(x < 0 ? "(" * text * ")" : text, typeof(x), true)
end
_jax_literal(x::Irrational{:π}) = _PyVal("jnp.pi", Float64, true)
_jax_literal(x::Irrational{:ℯ}) = _PyVal("jnp.e", Float64, true)
_jax_literal(x::Irrational) = _jax_literal(Float64(x))
_jax_literal(::Nothing) = _PyVal("None", Nothing, true)
_jax_literal(x::AbstractString) = _PyVal(repr(String(x)), String, true)
_jax_literal(x::Symbol) = _PyVal(repr(String(x)), Symbol, true)

function _jax_literal(x::AbstractArray{<:Real})
    T = eltype(x)
    body = _jax_nested_list(x)
    dtype = T <: Bool ? "jnp.bool_" : T <: Integer ? "jnp.int64" : "jnp.float64"
    _PyVal("jnp.asarray(" * body * ", dtype=" * dtype * ")", typeof(x))
end
# Python nesting is row-major: the outer list runs over the first dimension.
function _jax_nested_list(x::AbstractArray)
    ndims(x) == 0 && return _jax_literal(x[]).code
    ndims(x) == 1 && return "[" * join((_jax_scalar_text(v) for v in x), ", ") * "]"
    "[" * join((_jax_nested_list(selectdim(x, 1, i)) for i in axes(x, 1)), ", ") * "]"
end
_jax_scalar_text(v::Bool) = v ? "True" : "False"
_jax_scalar_text(v::Integer) = string(v)
function _jax_scalar_text(v::AbstractFloat)
    isnan(v) && return "float('nan')"
    isinf(v) && return v > 0 ? "float('inf')" : "float('-inf')"
    repr(Float64(v))
end
_jax_literal(x::Tuple) = _PyVal("(" * join((_jax_literal(v).code for v in x), ", ") *
                               (length(x) == 1 ? ",)" : ")"), typeof(x))
_jax_literal(x::NamedTuple) = _PyVal(_jax_literal(Tuple(x)).code, typeof(x))
_jax_literal(x) = _jax_error("no JAX literal for a value of type `", typeof(x), "`")

# A constant the source reads (a captured array, a module constant): small
# values inline, arrays become module-level constants.
function _jax_constant(ctx::_JaxContext, hint, x)
    x isa AbstractArray || return _jax_literal(x)
    name = get(ctx.mod.helper_names, (:constant, objectid(x)), nothing)
    if name === nothing
        name = _jax_identifier(ctx.mod.used, uppercase(string(hint)))
        push!(ctx.mod.helpers, name * " = " * _jax_literal(x).code)
        ctx.mod.helper_names[(:constant, objectid(x))] = name
    end
    _PyVal(name, typeof(x))
end

# --- expression translation ---------------------------------------------------------

# Resolve a callee or global in the source's module.
function _jax_resolve(ctx::_JaxContext, ex)
    ex isa QuoteNode && return ex.value
    ex isa GlobalRef && return getglobal(ex.mod, ex.name)
    if ex isa Symbol
        ctx.captures isa NamedTuple && haskey(ctx.captures, ex) &&
            return getfield(ctx.captures, ex)
        isdefined(ctx.source_module, ex) ||
            _jax_error("`", ex, "` is not defined in module `", ctx.source_module, "`",
                       _jax_where(ctx))
        return getglobal(ctx.source_module, ex)
    end
    if ex isa Expr && ex.head === :. && length(ex.args) == 2 && ex.args[2] isa QuoteNode
        owner = _jax_resolve(ctx, ex.args[1])
        return getproperty(owner, ex.args[2].value)
    end
    if ex isa Expr && ex.head === :curly
        return _jax_resolve(ctx, ex.args[1]){map(a -> _jax_type_argument(ctx, a), ex.args[2:end])...}
    end
    # A value spliced into generated source (a module, a function, a type).
    ex isa Expr || return ex
    _jax_error("cannot resolve `", ex, "`", _jax_where(ctx))
end

_jax_has_local(ctx::_JaxContext, s::Symbol) = haskey(ctx.locals, s)

# A type parameter in source: a local holding a type value, or a global.
function _jax_type_argument(ctx::_JaxContext, a)
    a isa Symbol && haskey(ctx.locals, a) && return _jax_callable(ctx.locals[a])
    a isa Integer && return a
    _jax_resolve(ctx, a)
end

function _jax_expr(ctx::_JaxContext, ex)
    ex === nothing && return _PyVal("None", Nothing, true)
    ex isa LineNumberNode && return _PyVal("None", Nothing, true)
    ex isa Number && return _jax_literal(ex)
    ex isa QuoteNode && return _jax_value(ctx, ex.value, :constant)
    ex isa Symbol && return _jax_symbol(ctx, ex)
    ex isa GlobalRef && return _jax_value(ctx, getglobal(ex.mod, ex.name), ex.name)
    ex isa Expr || return _jax_value(ctx, ex, :constant)
    _jax_expr(ctx, Val(ex.head), ex)
end

function _jax_symbol(ctx::_JaxContext, s::Symbol)
    haskey(ctx.locals, s) && return ctx.locals[s]
    s === :nothing && return _PyVal("None", Nothing)
    s === :Inf && return _jax_literal(Inf)
    s === :NaN && return _jax_literal(NaN)
    _jax_value(ctx, _jax_resolve(ctx, s), s)
end

# A Julia value met in source (a resolved global or capture).
_jax_value(ctx::_JaxContext, x::Union{Number,Nothing,AbstractString}, hint) = _jax_literal(x)
_jax_value(ctx::_JaxContext, x::AbstractArray{<:Real}, hint) = _jax_constant(ctx, hint, x)
_jax_value(ctx::_JaxContext, x::Tuple, hint) = _jax_literal(x)
# A function passed as a value (`sum(abs2, z)`): its type carries it, and only
# a call that takes a function argument reads it.
_jax_value(ctx::_JaxContext, x::Function, hint) =
    Base.issingletontype(typeof(x)) ? _PyVal("<function " * string(hint) * ">", typeof(x)) :
    _jax_error("the closure `", hint, "` passed as a value has no JAX translation",
               _jax_where(ctx))
_jax_value(ctx::_JaxContext, x, hint) =
    _jax_error("the value `", hint, "` of type `", typeof(x), "` has no JAX translation",
               _jax_where(ctx))

_jax_expr(ctx::_JaxContext, ::Val{H}, ex) where {H} =
    _jax_error("the construct `", H, "` (`", ex, "`) has no JAX translation yet",
               _jax_where(ctx))

_jax_expr(ctx::_JaxContext, ::Val{:block}, ex) = _jax_statements(ctx, ex.args)
_jax_expr(ctx::_JaxContext, ::Val{:(::)}, ex) = _jax_expr(ctx, ex.args[1])
_jax_expr(ctx::_JaxContext, ::Val{:return}, ex) = _jax_expr(ctx, ex.args[1])
function _jax_expr(ctx::_JaxContext, ::Val{:macrocall}, ex)
    name = ex.args[1]
    name in (Symbol("@inbounds"), Symbol("@fastmath"), Symbol("@inline"),
             Symbol("@noinline")) ||
        (name isa GlobalRef && name.name in (Symbol("@inbounds"), Symbol("@inline"))) ||
        _jax_error("the macro call `", name, "` has no JAX translation", _jax_where(ctx))
    _jax_expr(ctx, ex.args[end])
end

# A sequence of statements: assignments bind locals, the last value is the
# result.
function _jax_statements(ctx::_JaxContext, statements)
    result = _PyVal("None", Nothing)
    for statement in statements
        statement isa LineNumberNode && continue
        result = _jax_statement(ctx, statement)
    end
    result
end

const _JAX_UPDATING = Dict{Symbol,Symbol}(:+= => :+, :-= => :-, :*= => :*, :/= => :/,
                                           :^= => :^)

function _jax_statement(ctx::_JaxContext, ex)
    if ex isa Expr && ex.head === :(=) && length(ex.args) == 2
        return _jax_assign!(ctx, ex.args[1], ex.args[2])
    end
    if ex isa Expr && haskey(_JAX_UPDATING, ex.head) && length(ex.args) == 2
        lhs = ex.args[1]
        return _jax_assign!(ctx, lhs, Expr(:call, _JAX_UPDATING[ex.head], lhs, ex.args[2]))
    end
    if ex isa Expr && ex.head === :local
        return _PyVal("None", Nothing)
    end
    _jax_expr(ctx, ex)
end

function _jax_assign!(ctx::_JaxContext, lhs, rhs)
    v = _jax_expr(ctx, rhs)
    _jax_destructure!(ctx, lhs, v)
    v
end

function _jax_destructure!(ctx::_JaxContext, lhs, v::_PyVal)
    lhs isa Expr && lhs.head === :(::) && (lhs = lhs.args[1])
    if lhs isa Symbol
        # A type or callable value exists only while translating.
        v.T isa _JaxCallable && (ctx.locals[lhs] = v; return)
        name = _jax_fresh(ctx, lhs)
        _jax_emit!(ctx, name * " = " * v.code)
        ctx.locals[lhs] = _PyVal(name, v.T, v.static, v.extent, v.limit)
        return
    end
    if lhs isa Expr && lhs.head === :ref && lhs.args[1] isa Symbol && haskey(ctx.locals, lhs.args[1])
        # `A[i] = v`: a functional update rebinding `A`.
        target = ctx.locals[lhs.args[1]]
        indexed = _jax_index(ctx, target, lhs.args[2:end])
        position = indexed.code[ncodeunits(target.code)+1:end]
        name = _jax_fresh(ctx, lhs.args[1])
        _jax_emit!(ctx, name * " = " * target.code * ".at" * position * ".set(" * v.code * ")")
        ctx.locals[lhs.args[1]] = _PyVal(name, target.T, false, target.extent, target.limit)
        return
    end
    if lhs isa Expr && lhs.head === :tuple
        bound = _jax_bind!(ctx, :tuple, v)
        for (i, target) in enumerate(lhs.args)
            T = v.T isa Type && v.T <: Tuple && i <= fieldcount(v.T) ? fieldtype(v.T, i) : Any
            _jax_destructure!(ctx, target, _PyVal(bound.code * "[" * string(i - 1) * "]", T))
        end
        return
    end
    _jax_error("the assignment target `", lhs, "` has no JAX translation", _jax_where(ctx))
end

function _jax_expr(ctx::_JaxContext, ::Val{:let}, ex)
    bindings, body = ex.args
    saved = copy(ctx.locals)
    try
        list = bindings isa Expr && bindings.head === :block ? bindings.args : Any[bindings]
        for binding in list
            binding isa LineNumberNode && continue
            binding isa Symbol && continue
            binding isa Expr && binding.head === :(=) ||
                _jax_error("the let binding `", binding, "` has no JAX translation",
                           _jax_where(ctx))
            _jax_assign!(ctx, binding.args[1], binding.args[2])
        end
        result = _jax_expr(ctx, body)
        return _jax_bind!(ctx, :let_value, result)
    finally
        empty!(ctx.locals)
        merge!(ctx.locals, saved)
    end
end

function _jax_expr(ctx::_JaxContext, ::Val{:tuple}, ex)
    if !isempty(ex.args) && all(a -> a isa Expr && a.head in (:(=), :kw), ex.args)
        names = Tuple(Symbol(a.args[1]) for a in ex.args)
        values = [_jax_expr(ctx, a.args[2]) for a in ex.args]
        return _PyVal(_jax_tuple_code(values),
                      NamedTuple{names,Tuple{(v.T for v in values)...}})
    end
    if length(ex.args) == 1 && ex.args[1] isa Expr && ex.args[1].head === :parameters
        params = ex.args[1].args
        names = Tuple(Symbol(a isa Symbol ? a : a.args[1]) for a in params)
        values = [_jax_expr(ctx, a isa Symbol ? a : a.args[2]) for a in params]
        return _PyVal(_jax_tuple_code(values),
                      NamedTuple{names,Tuple{(v.T for v in values)...}})
    end
    values = [_jax_expr(ctx, a) for a in ex.args]
    _PyVal(_jax_tuple_code(values), Tuple{(v.T for v in values)...}, false, _jax_tuple_extent(values))
end
# The extents of a tuple's padded elements, as a Python tuple (`None` for an
# element that is not padded), or "" when none is.
_jax_tuple_extent(values) = all(v -> isempty(v.extent), values) ? "" :
    "(" * join((isempty(v.extent) ? "None" : v.extent for v in values), ", ") *
    (length(values) == 1 ? ",)" : ")")
_jax_tuple_code(values) = "(" * join((v.code for v in values), ", ") *
                          (length(values) == 1 ? ",)" : ")")

function _jax_expr(ctx::_JaxContext, ::Val{:vect}, ex)
    values = [_jax_expr(ctx, a) for a in ex.args]
    T = isempty(values) ? Float64 : promote_type((_jax_eltype(v.T) for v in values)...)
    _PyVal("jnp.stack([" * join((v.code for v in values), ", ") * "])", Vector{T})
end

# `x[i...]`, or a typed array literal `T[a, b]`.
function _jax_expr(ctx::_JaxContext, ::Val{:ref}, ex)
    head = ex.args[1]
    if head isa Symbol && !_jax_has_local(ctx, head) && isdefined(ctx.source_module, head) &&
       getglobal(ctx.source_module, head) isa Type
        T = getglobal(ctx.source_module, head)
        values = [_jax_expr(ctx, a) for a in ex.args[2:end]]
        dtype = T <: Bool ? "jnp.bool_" : T <: Integer ? "jnp.int64" : "jnp.float64"
        isempty(values) && return _PyVal("jnp.zeros((0,), dtype=" * dtype * ")", Vector{T})
        return _PyVal("jnp.asarray([" * join((v.code for v in values), ", ") *
                      "], dtype=" * dtype * ")", Vector{T})
    end
    x = _jax_expr(ctx, head)
    _jax_index(ctx, x, ex.args[2:end])
end

# Index `x` with Julia indices `indices` (source expressions).
function _jax_index(ctx::_JaxContext, x::_PyVal, indices)
    if x.T isa Type && x.T <: Tuple
        length(indices) == 1 && indices[1] isa Integer ||
            _jax_error("a tuple is indexed with a literal integer only", _jax_where(ctx))
        k = indices[1]
        T = fieldtype(x.T, k)
        extent = isempty(x.extent) ? "" : x.extent * "[" * string(k - 1) * "]"
        return _PyVal(x.code * "[" * string(k - 1) * "]", T, false,
                      _jax_is_array(T) ? extent : "")
    end
    _jax_is_array(x.T) || _jax_error("indexing a value of type `", x.T, "`", _jax_where(ctx))
    N = ndims(x.T)
    length(indices) == N ||
        (length(indices) == 1 && N == 1) ||
        _jax_error("indexing a rank-", N, " array with ", length(indices),
                   " indices (linear indexing is not translated)", _jax_where(ctx))
    parts = String[]
    kept = 0
    for (dim, index) in enumerate(indices)
        part, keeps = _jax_index_part(ctx, x, dim, index)
        push!(parts, part)
        kept += keeps
    end
    E = eltype(x.T)
    T = kept == 0 ? E : Array{E,kept}
    if _jax_ragged(x.T)
        isempty(x.extent) && _jax_error("a ragged array without its lengths", _jax_where(ctx))
        # One group's row and length, or several groups' rows and lengths.
        return _PyVal(x.code * "[" * join(parts, ", ") * "]", T, false,
                      x.extent * "[" * join(parts, ", ") * "]")
    end
    if !isempty(x.extent) && kept > 0
        # A gather from a padded group vector by an index vector: the indices
        # are valid entries, and the result is padded as the index is.
        gather = length(indices) == 1 ? _jax_bound_expr(ctx, x, 1, indices[1]) : nothing
        gather !== nothing && _jax_is_array(gather.T) && eltype(gather.T) <: Integer ||
            _jax_error("slicing a padded group vector has no JAX translation", _jax_where(ctx))
        return _PyVal(x.code * "[" * join(parts, ", ") * "]", T, false, gather.extent)
    end
    _PyVal(x.code * "[" * join(parts, ", ") * "]", T, false, kept > 0 ? _jax_index_extent(ctx, indices) : "")
end
# The extent of a gather result: its index vector's, when that is padded.
_jax_index_extent(ctx, indices) = ""

# One index position: the Python index text and whether it keeps a dimension.
function _jax_index_part(ctx::_JaxContext, x::_PyVal, dim::Int, index)
    index === :(:) && return (":", 1)
    if index isa Expr && index.head === :call && index.args[1] === :(:) &&
       length(index.args) == 3
        lo = _jax_bound_expr(ctx, x, dim, index.args[2])
        hi = _jax_bound_expr(ctx, x, dim, index.args[3])
        return (_jax_minus_one(lo) * ":" * hi.code, 1)
    end
    i = _jax_bound_expr(ctx, x, dim, index)
    _jax_is_scalar(i.T) && i.T <: Integer && return (_jax_minus_one(i), 0)
    if _jax_is_array(i.T) && eltype(i.T) <: Integer
        ndims(i.T) == 1 || _jax_error("a gather by a rank-", ndims(i.T), " index array",
                                      _jax_where(ctx))
        return ("(" * i.code * ") - 1", 1)
    end
    if i.T isa Type && i.T <: AbstractUnitRange
        m = match(r"^jnp\.arange\((.*), (.*)\)$", i.code)
        m === nothing || return (_jax_minus_one(_PyVal(m.captures[1], Int)) * ":" *
                                 _jax_minus_one(_PyVal(m.captures[2], Int)), 1)
        return ("(" * i.code * ") - 1", 1)
    end
    _jax_error("the index `", index, "` (of type `", i.T, "`) has no JAX translation",
               _jax_where(ctx))
end

# An index expression in which `end` and `begin` name the bounds of `x`'s
# dimension `dim`.
function _jax_bound_expr(ctx::_JaxContext, x::_PyVal, dim::Int, index)
    index === :end && return _PyVal(x.code * ".shape[" * string(dim - 1) * "]", Int, true)
    index === :begin && return _PyVal("1", Int)
    if index isa Expr && (_jax_mentions(index, :end) || _jax_mentions(index, :begin))
        saved = copy(ctx.locals)
        try
            ctx.locals[:end] = _PyVal(x.code * ".shape[" * string(dim - 1) * "]", Int, true)
            ctx.locals[:begin] = _PyVal("1", Int)
            return _jax_expr(ctx, index)
        finally
            empty!(ctx.locals); merge!(ctx.locals, saved)
        end
    end
    _jax_expr(ctx, index)
end
_jax_mentions(ex, s::Symbol) = ex === s ||
    (ex isa Expr && !(ex.head === :ref) && any(a -> _jax_mentions(a, s), ex.args))

function _jax_minus_one(i::_PyVal)
    m = match(r"^-?[0-9]+$", i.code)
    m === nothing ? "(" * i.code * ") - 1" : string(parse(Int, i.code) - 1)
end

# `x.field` on a named tuple, or a dotted call `f.(args...)`.
function _jax_expr(ctx::_JaxContext, ::Val{:.}, ex)
    if length(ex.args) == 2 && ex.args[2] isa QuoteNode
        field = ex.args[2].value
        owner = ex.args[1]
        if owner isa Symbol && !_jax_has_local(ctx, owner)
            return _jax_value(ctx, _jax_resolve(ctx, ex), field)
        end
        x = _jax_expr(ctx, owner)
        x.T isa Type && x.T <: NamedTuple ||
            _jax_error("the field read `", ex, "` on a value of type `", x.T, "`",
                       _jax_where(ctx))
        k = findfirst(==(field), fieldnames(x.T))
        k === nothing && _jax_error("`", x.T, "` has no field `", field, "`", _jax_where(ctx))
        return _PyVal(x.code * "[" * string(k - 1) * "]", fieldtype(x.T, k))
    end
    if length(ex.args) == 2 && ex.args[2] isa Expr && ex.args[2].head === :tuple
        f = _jax_resolve(ctx, ex.args[1])
        args = [_jax_expr(ctx, a) for a in ex.args[2].args]
        return _jax_broadcast(ctx, f, args)
    end
    _jax_error("the construct `", ex, "` has no JAX translation", _jax_where(ctx))
end

function _jax_expr(ctx::_JaxContext, ::Val{Symbol("'")}, ex)
    x = _jax_expr(ctx, ex.args[1])
    _jax_call(ctx, adjoint, [x])
end

# Short-circuit operators: a static left side decides while tracing, so the
# right side runs only when Julia's would (a guard `n == k || throw(...)`); a
# traced left side combines both sides, whose right side must not throw.
_jax_expr(ctx::_JaxContext, ::Val{:&&}, ex) = _jax_short_circuit(ctx, ex, false)
_jax_expr(ctx::_JaxContext, ::Val{:||}, ex) = _jax_short_circuit(ctx, ex, true)
function _jax_short_circuit(ctx::_JaxContext, ex, is_or::Bool)
    a = _jax_expr(ctx, ex.args[1])
    if a.static
        rest = () -> _jax_expr(ctx, ex.args[2])
        decided = () -> _jax_literal(is_or)
        return is_or ? _jax_conditional(ctx, a, decided, rest) :
                       _jax_conditional(ctx, a, rest, decided)
    end
    ctx.traced_branches += 1
    b = try
        _jax_expr(ctx, ex.args[2])
    finally
        ctx.traced_branches -= 1
    end
    _PyVal((is_or ? "jnp.logical_or(" : "jnp.logical_and(") * a.code * ", " * b.code * ")", Bool)
end

# `map(xs...) do args... body end`: the body as a function of one element of
# each collection, vectorized over them (padded collections stay padded).
function _jax_expr(ctx::_JaxContext, ::Val{:do}, ex)
    call, lambda = ex.args
    call isa Expr && call.head === :call && lambda isa Expr && lambda.head === :-> ||
        _jax_error("the do-block `", ex, "` has no JAX translation", _jax_where(ctx))
    f = _jax_resolve(ctx, call.args[1])
    f === map || _jax_error("a do-block on `", f, "` has no JAX translation", _jax_where(ctx))
    collections = [_jax_expr(ctx, a) for a in call.args[2:end]]
    all(c -> _jax_rank(c.T) == 1, collections) ||
        _jax_error("`map` over collections that are not vectors", _jax_where(ctx))
    params = lambda.args[1]
    names = params isa Symbol ? Symbol[params] : Symbol[p for p in params.args]
    length(names) == length(collections) ||
        _jax_error("a do-block with ", length(names), " arguments over ", length(collections),
                   " collections", _jax_where(ctx))
    elements = [_PyVal(_jax_fresh(ctx, n), eltype(c.T)) for (n, c) in zip(names, collections)]
    fname = _jax_fresh(ctx, :element)
    outer = ctx.scope
    inner = _JaxScope(outer)
    inner.indent = outer.indent + 1
    saved = copy(ctx.locals)
    ctx.scope = inner
    result = try
        for (n, e) in zip(names, elements)
            ctx.locals[n] = e
        end
        _jax_expr(ctx, lambda.args[2])
    finally
        ctx.scope = outer
        empty!(ctx.locals); merge!(ctx.locals, saved)
    end
    push!(outer.lines, "    "^outer.indent * "def " * fname * "(" * join((e.code for e in elements), ", ") * "):")
    append!(outer.lines, inner.lines)
    push!(outer.lines, "    "^inner.indent * "return " * result.code)
    _PyVal("jax.vmap(" * fname * ")(" * _jax_args(collections) * ")", Vector{_jax_typeof(result)},
           false, _jax_extent(collections))
end

# Loops. A `for` over a range is `jax.lax.fori_loop` over a function of the
# index and the carry, the outer locals the body assigns (an indexed write
# assigns its array). The range's bounds are static, so the loop lowers to a
# scan that `jax.grad` differentiates; a data-dependent upper bound with a
# static limit (a padded group vector's length) runs to the limit and keeps
# the carry unchanged past the bound.
function _jax_expr(ctx::_JaxContext, ::Val{:for}, ex)
    binding, body = ex.args
    binding isa Expr && binding.head === :(=) && binding.args[1] isa Symbol ||
        _jax_error("the loop binding `", binding, "` has no JAX translation", _jax_where(ctx))
    var = binding.args[1]
    lo, hi, bound = _jax_loop_range(ctx, binding.args[2])
    assigned = Set{Symbol}()
    _jax_assigned_names!(assigned, body)
    carried = sort!([n for n in assigned if haskey(ctx.locals, n) && n !== var])
    initial = [_jax_carry_value(ctx, ctx.locals[n]) for n in carried]
    fname = _jax_fresh(ctx, :loop_body)
    index = _jax_fresh(ctx, var)
    carry = _jax_fresh(ctx, :carry)
    outer = ctx.scope
    inner = _JaxScope(outer)
    inner.indent = outer.indent + 1
    push!(inner.used, index, carry)
    saved = copy(ctx.locals)
    ctx.scope = inner
    updated = try
        names = [_jax_identifier(inner.used, n) for n in carried]
        isempty(names) || _jax_emit!(ctx, "(" * join(names, ", ") * (length(names) == 1 ? ",)" : ")") *
                                         " = " * carry)
        for (n, name, v) in zip(carried, names, initial)
            ctx.locals[n] = _PyVal(name, v.T, false, v.extent, v.limit)
        end
        ctx.locals[var] = _PyVal(index, Int)
        _jax_statements(ctx, body isa Expr && body.head === :block ? body.args : Any[body])
        # Past a data-dependent bound the scalar carry stays as it was; an
        # array written by index there writes only padded entries.
        map(zip(carried, initial)) do (n, v)
            new = ctx.locals[n]
            bound == "" || _jax_is_array(v.T) ? new.code :
                "jnp.where(" * index * " <= " * bound * ", " * new.code * ", " *
                _jax_carry_old(names, carried, n) * ")"
        end
    finally
        ctx.scope = outer
        empty!(ctx.locals); merge!(ctx.locals, saved)
    end
    push!(outer.lines, "    "^outer.indent * "def " * fname * "(" * index * ", " * carry * "):")
    append!(outer.lines, inner.lines)
    push!(outer.lines, "    "^inner.indent * "return (" * join(updated, ", ") *
                       (length(updated) == 1 ? ",)" : ")"))
    result = _jax_fresh(ctx, :loop_carry)
    _jax_emit!(ctx, result * " = jax.lax.fori_loop(" * lo.code * ", " * _jax_plus_one(hi) * ", " *
                    fname * ", (" * join((v.code for v in initial), ", ") *
                    (length(initial) == 1 ? ",))" : "))"))
    for (k, (n, v)) in enumerate(zip(carried, initial))
        ctx.locals[n] = _PyVal(result * "[" * string(k - 1) * "]", v.T, false, v.extent, v.limit)
    end
    _PyVal("None", Nothing, true)
end
_jax_carry_old(names, carried, n) = names[findfirst(==(n), carried)]

# A loop carry enters with a fixed dtype, as `fori_loop` requires.
function _jax_carry_value(ctx::_JaxContext, v::_PyVal)
    _jax_is_array(v.T) && return v
    T = v.T isa Type ? v.T : Float64
    dtype = T <: Bool ? "jnp.bool_" : T <: Integer ? "jnp.int64" : "jnp.float64"
    _PyVal("jnp.asarray(" * v.code * ", dtype=" * dtype * ")", T, false, "", "")
end

# The locals a loop body assigns, including arrays it writes by index.
function _jax_assigned_names!(names::Set{Symbol}, ex)
    ex isa Expr || return names
    ex.head in (:quote, :inert, :->, :function) && return names
    if (ex.head === :(=) || haskey(_JAX_UPDATING, ex.head)) && length(ex.args) == 2
        target = ex.args[1]
        target isa Expr && target.head === :(::) && (target = target.args[1])
        while target isa Expr && target.head === :ref
            target = target.args[1]
        end
        target isa Symbol && push!(names, target)
        target isa Expr && target.head === :tuple &&
            foreach(t -> t isa Symbol && push!(names, t), target.args)
    end
    foreach(a -> _jax_assigned_names!(names, a), ex.args)
    names
end

# A loop range's (first, last, data-dependent bound or ""): `a:b`,
# `eachindex(x)`, `axes(x, 1)`.
function _jax_loop_range(ctx::_JaxContext, iter)
    if iter isa Expr && iter.head === :call && iter.args[1] === :(:) && length(iter.args) == 3
        lo, hi = _jax_expr(ctx, iter.args[2]), _jax_expr(ctx, iter.args[3])
    elseif iter isa Expr && iter.head === :call && length(iter.args) >= 2 &&
           _jax_resolve(ctx, iter.args[1]) in (eachindex, axes)
        x = _jax_expr(ctx, iter.args[2])
        _jax_rank(x.T) == 1 || _jax_error("a loop over the indices of a `", x.T, "`",
                                          _jax_where(ctx))
        lo = _jax_literal(1)
        hi = isempty(x.extent) ? _PyVal(x.code * ".shape[0]", Int, true) :
            _PyVal(x.extent, Int, false, "", x.code * ".shape[0]")
    else
        _jax_error("the loop range `", iter, "` has no JAX translation", _jax_where(ctx))
    end
    lo.static || _jax_error("a loop whose first index depends on data", _jax_where(ctx))
    hi.static && return (lo, hi, "")
    isempty(hi.limit) && _jax_error("a loop whose trip count depends on data has no JAX ",
                                    "translation", _jax_where(ctx))
    (lo, _PyVal(hi.limit, Int, true), hi.code)
end

# A lazy branch: `jax.lax.cond` of one function per arm, so only the taken
# arm runs (under `jax.vmap` JAX itself decides how to batch the region).
function _jax_expr(ctx::_JaxContext, ::Val{:if}, ex)
    condition = _jax_expr(ctx, ex.args[1])
    otherwise = length(ex.args) >= 3 ? ex.args[3] : nothing
    _jax_conditional(ctx, condition, () -> _jax_expr(ctx, ex.args[2]),
                     () -> _jax_expr(ctx, otherwise))
end

# A conditional: a static condition is a Python conditional, decided while JAX
# traces; any other is `jax.lax.cond`, whose arms JAX traces both of.
function _jax_conditional(ctx::_JaxContext, condition::_PyVal, then_build, else_build)
    condition.static || (ctx.traced_branches += 1)
    then_arm, else_arm = try
        (_jax_arm_function(ctx, :then_arm, then_build),
         _jax_arm_function(ctx, :else_arm, else_build))
    finally
        condition.static || (ctx.traced_branches -= 1)
    end
    T = then_arm.T === Union{} ? else_arm.T : else_arm.T === Union{} ? then_arm.T :
        promote_type(_jax_typeof(then_arm), _jax_typeof(else_arm))
    condition.static &&
        return _PyVal("(" * then_arm.code * "() if " * condition.code * " else " *
                      else_arm.code * "())", T, then_arm.static && else_arm.static)
    _PyVal("jax.lax.cond(" * condition.code * ", " * then_arm.code * ", " *
           else_arm.code * ")", T)
end
# `throw` in a branch decided while tracing raises then; a data-dependent
# throw has no counterpart inside a traced JAX function.
function _jax_call_positional(ctx::_JaxContext, ::typeof(throw), args::Vector{_PyVal})
    ctx.traced_branches == 0 || _jax_error(
        "a throw that depends on data has no JAX translation", _jax_where(ctx))
    e = only(args)
    message = e.T isa Type && e.T <: Exception && e.code != "None" ? e.code : repr(string(e.T))
    _jax_emit!(ctx, "raise ValueError(" * message * ")")
    _PyVal("None", Union{}, true)
end
# An exception value: its message, for `throw`.
_jax_call_positional(ctx::_JaxContext, T::Type{<:Exception}, args::Vector{_PyVal}) =
    _PyVal(isempty(args) ? "None" : args[1].code, T, true)

# String interpolation, as Python string concatenation.
function _jax_expr(ctx::_JaxContext, ::Val{:string}, ex)
    parts = [_jax_expr(ctx, a) for a in ex.args]
    _PyVal("\"\".join([" * join(("str(" * p.code * ")" for p in parts), ", ") * "])",
           String, all(p -> p.static, parts))
end
_jax_expr(ctx::_JaxContext, ::Val{:elseif}, ex) = _jax_expr(ctx, Val(:if),
    Expr(:if, ex.args[1] isa Expr && ex.args[1].head === :block ?
         ex.args[1].args[end] : ex.args[1], ex.args[2:end]...))

_jax_typeof(v::_PyVal) = v.T isa Type ? v.T : Any

# A zero-argument nested function whose body `build` translates; its value's
# type travels in the returned `_PyVal`.
function _jax_arm_function(ctx::_JaxContext, hint, build)
    name = _jax_fresh(ctx, hint)
    outer = ctx.scope
    inner = _JaxScope(outer)
    inner.indent = outer.indent + 1
    push!(outer.lines, "    "^outer.indent * "def " * name * "():")
    saved_locals = copy(ctx.locals)
    ctx.scope = inner
    result = try
        build()
    finally
        ctx.scope = outer
        empty!(ctx.locals); merge!(ctx.locals, saved_locals)
    end
    append!(outer.lines, inner.lines)
    push!(outer.lines, "    "^inner.indent * "return " * result.code)
    union!(outer.used, inner.used)
    _PyVal(name, result.T, result.static)
end

function _jax_expr(ctx::_JaxContext, ::Val{:call}, ex)
    head = ex.args[1]
    args = ex.args[2:end]
    kwargs = Pair{Symbol,Any}[]
    positional = Any[]
    for a in args
        if a isa Expr && a.head === :parameters
            for p in a.args
                p isa Expr && p.head === :kw ? push!(kwargs, p.args[1] => p.args[2]) :
                    _jax_error("the keyword argument `", p, "`", _jax_where(ctx))
            end
        elseif a isa Expr && a.head === :kw
            push!(kwargs, a.args[1] => a.args[2])
        else
            push!(positional, a)
        end
    end
    # A dotted operator, `a .+ b`.
    if head isa Symbol && !_jax_has_local(ctx, head) && startswith(String(head), ".") &&
       length(String(head)) > 1
        f = _jax_resolve(ctx, Symbol(String(head)[2:end]))
        return _jax_broadcast(ctx, f, [_jax_expr(ctx, a) for a in positional])
    end
    f = head isa Symbol && _jax_has_local(ctx, head) ?
        _jax_error("calling the local `", head, "` has no JAX translation", _jax_where(ctx)) :
        _jax_resolve(ctx, head)
    # Indexing spelled as a call reads its index expressions as indexing does.
    if _jax_is_indexing(f) && length(positional) >= 2 && isempty(kwargs)
        return _jax_index(ctx, _jax_expr(ctx, positional[1]), positional[2:end])
    end
    values = [_jax_expr(ctx, a) for a in positional]
    kw = [k => _jax_expr(ctx, v) for (k, v) in kwargs]
    _jax_call(ctx, f, values, kw)
end

_jax_is_indexing(f) = f === getindex || f === view || f === _tensorized_getindex ||
    f === _tensorized_view || f === Base.maybeview

_jax_call(ctx::_JaxContext, f, args::Vector{_PyVal}) = _jax_call(ctx, f, args, Pair{Symbol,_PyVal}[])

function _jax_call(ctx::_JaxContext, f, args::Vector{_PyVal}, kw)
    isempty(kw) || _jax_error("keyword arguments to `", f, "` have no JAX translation",
                              _jax_where(ctx))
    _jax_call_positional(ctx, f, args)
end

_jax_call_positional(ctx::_JaxContext, f, args::Vector{_PyVal}) =
    _jax_error("the function `", f, "` (", typeof(f), ") has no JAX translation",
               _jax_where(ctx), ". Give it one with a `ReactiveKernels._jax_call_positional` ",
               "method, or author it as a kernel")

function _jax_result(f, args::Vector{_PyVal}, code::AbstractString)
    T = _jax_return_type(f, [a.T isa Type ? a.T : Any for a in args])
    _PyVal(code, T, !isempty(args) && all(a -> a.static, args),
           _jax_is_array(T) ? _jax_extent(args) : "")
end
_jax_args(args) = join((a.code for a in args), ", ")

# Elementwise numeric functions, as their `jax.numpy` counterparts. Each is
# applied to scalars directly and to arrays elementwise when broadcast.
const _JAX_ELEMENTWISE = Dict{Any,String}(
    exp => "jnp.exp", log => "jnp.log", log1p => "jnp.log1p", expm1 => "jnp.expm1",
    sqrt => "jnp.sqrt", abs => "jnp.abs", abs2 => "jnp.square",
    sin => "jnp.sin", cos => "jnp.cos", tan => "jnp.tan", tanh => "jnp.tanh",
    sinh => "jnp.sinh", cosh => "jnp.cosh", asin => "jnp.arcsin", acos => "jnp.arccos",
    atan => "jnp.arctan", asinh => "jnp.arcsinh", acosh => "jnp.arccosh", atanh => "jnp.arctanh",
    exp2 => "jnp.exp2", log2 => "jnp.log2", log10 => "jnp.log10", cbrt => "jnp.cbrt",
    sign => "jnp.sign", floor => "jnp.floor", ceil => "jnp.ceil", round => "jnp.round",
    isnan => "jnp.isnan", isinf => "jnp.isinf", isfinite => "jnp.isfinite",
    min => "jnp.minimum", max => "jnp.maximum",
    ifelse => "jnp.where", (!) => "jnp.logical_not",
    LogExpFunctions.logaddexp => "jnp.logaddexp",
    LogExpFunctions.logistic => "jax.nn.sigmoid",
    LogExpFunctions.logit => "jsp.logit",
    LogExpFunctions.log1pexp => "jax.nn.softplus",
    LogExpFunctions.xlogy => "jsp.xlogy",
    LogExpFunctions.xlog1py => "jsp.xlog1py",
    LogExpFunctions.log1mexp => "_rk_log1mexp",
    _native_exp => "jnp.exp",
)
# LogExpFunctions functions that only some of its supported versions define.
for (name, py) in ((:log1msq, "_rk_log1msq"), (:logcosh, "_rk_logcosh"),
                   (:log2mexp, "_rk_log2mexp"), (:cexpexp, "_rk_cexpexp"),
                   (:loglogistic, "jax.nn.log_sigmoid"),
                   (:log1mlogistic, "_rk_log1mlogistic"))
    isdefined(LogExpFunctions, name) &&
        (_JAX_ELEMENTWISE[getglobal(LogExpFunctions, name)] = py)
end
# Two-argument `atan` is `arctan2`; the table keys one name per function, so
# its arity selects the name.
const _JAX_BINARY_NAMES = Dict{Any,String}(atan => "jnp.arctan2")

# Small helpers the emitted module defines when it uses them: stable forms of
# LogExpFunctions functions whose `jax.numpy` spelling is not one call.
const _JAX_PRELUDE = Dict{String,String}(
    "_rk_log1mexp" => """
def _rk_log1mexp(x):
    # log(1 - exp(x)) for x <= 0 (LogExpFunctions.log1mexp).
    small = x < -0.6931471805599453
    safe_small = jnp.where(small, x, -1.0)
    safe_large = jnp.where(small, -1.0, x)
    return jnp.where(small, jnp.log1p(-jnp.exp(safe_small)),
                     jnp.log(-jnp.expm1(safe_large)))
""",
    "_rk_log1msq" => """
def _rk_log1msq(x):
    # log(1 - x^2) (LogExpFunctions.log1msq).
    return jnp.log1p(-jnp.square(x))
""",
    "_rk_logcosh" => """
def _rk_logcosh(x):
    # log(cosh(x)) (LogExpFunctions.logcosh).
    a = jnp.abs(x)
    return a + jnp.log1p(jnp.exp(-2.0 * a)) - jnp.log(2.0)
""",
    "_rk_cexpexp" => """
def _rk_cexpexp(x):
    # 1 - exp(-exp(x)) (LogExpFunctions.cexpexp).
    return -jnp.expm1(-jnp.exp(x))
""",
    "_rk_log1mlogistic" => """
def _rk_log1mlogistic(x):
    # log(1 - logistic(x)) (LogExpFunctions.log1mlogistic).
    return jax.nn.log_sigmoid(-x)
""",
    "_rk_log2mexp" => """
def _rk_log2mexp(x):
    # log(2 - exp(x)) (LogExpFunctions.log2mexp).
    return jnp.log1p(-jnp.expm1(x))
""",
)

function _jax_use_prelude!(ctx::_JaxContext, name::AbstractString)
    haskey(_JAX_PRELUDE, name) || return
    _jax_helper!(ctx, name, _JAX_PRELUDE[name])
end

# Define the module-level helper `name` (Python source `text`) once.
function _jax_helper!(ctx::_JaxContext, name::AbstractString, text::AbstractString)
    key = (:helper, name)
    haskey(ctx.mod.helper_names, key) && return name
    ctx.mod.helper_names[key] = name
    push!(ctx.mod.used, name)
    push!(ctx.mod.helpers, rstrip(text))
    name
end

"""
    _jax_elementwise(f) -> Union{Nothing,String,Pair{String,String}}

The `jax.numpy` expression of the scalar function `f`, applied elementwise to
arrays (`"jnp.exp"`), or `name => source` for a module-level helper `name`
that `source` defines, or `nothing`. A package adds a translation for its own
scalar function with a method of this function.
"""
function _jax_elementwise(f)
    name = get(_JAX_ELEMENTWISE, f, nothing)
    name === nothing && return nothing
    haskey(_JAX_PRELUDE, name) ? name => _JAX_PRELUDE[name] : name
end

function _jax_elementwise_name(ctx::_JaxContext, f, n::Int)
    n == 2 && haskey(_JAX_BINARY_NAMES, f) && return _JAX_BINARY_NAMES[f]
    translation = _jax_elementwise(f)
    translation isa Pair || return translation
    _jax_helper!(ctx, first(translation), last(translation))
end

# Arithmetic operators keep Python's infix forms.
const _JAX_INFIX = Dict{Any,String}((+) => "+", (-) => "-", (/) => "/", (<) => "<", (>) => ">",
    (<=) => "<=", (>=) => ">=", (==) => "==", (!=) => "!=", (&) => "&", (|) => "|",
    xor => "^")

function _jax_scalar_or_elementwise(ctx::_JaxContext, f, args::Vector{_PyVal})
    if haskey(_JAX_INFIX, f)
        op = _JAX_INFIX[f]
        length(args) == 1 && f === (-) && return _jax_result(f, args, "(-" * args[1].code * ")")
        length(args) == 1 && f === (+) && return args[1]
        length(args) >= 2 || return nothing
        return _jax_result(f, args, "(" * join((a.code for a in args), " " * op * " ") * ")")
    end
    if f === (*) && any(a -> a.T isa Type && a.T <: AbstractString, args)
        return _PyVal("(" * join((a.code for a in args), " + ") * ")", String,
                      all(a -> a.static, args))
    end
    if f === (*) && count(a -> _jax_is_array(a.T), args) <= 1
        return _jax_result(f, args, "(" * join((a.code for a in args), " * ") * ")")
    end
    if f === (^) && length(args) == 2 && !_jax_is_array(args[1].T)
        return _jax_result(f, args, "(" * args[1].code * " ** " * args[2].code * ")")
    end
    name = _jax_elementwise_name(ctx, f, length(args))
    name === nothing && return nothing
    _jax_result(f, args, name * "(" * _jax_args(args) * ")")
end

# A call on scalars is its elementwise form. A call with an array argument
# has Julia's array semantics (`exp(A)` is the matrix exponential), except the
# operators that act elementwise on arrays too: `+` and `-`, and `/` by a scalar.
function _jax_call_positional(ctx::_JaxContext, f::Function, args::Vector{_PyVal})
    if !any(a -> _jax_is_array(a.T), args) ||
       ((f === (+) || f === (-)) && all(a -> _jax_is_array(a.T), args)) ||
       (f === (/) && length(args) == 2 && !_jax_is_array(args[2].T)) ||
       (f === (*) && count(a -> _jax_is_array(a.T), args) <= 1)
        v = _jax_scalar_or_elementwise(ctx, f, args)
        v === nothing || return v
    end
    _jax_function_call(ctx, f, args)
end

# A call with array semantics, or of a function translated from its own
# graph or source.
function _jax_function_call(ctx::_JaxContext, f, args::Vector{_PyVal})
    v = _jax_traceable_call(ctx, f, args)
    v === nothing || return v
    v = _jax_lowered_call(ctx, f, args)
    v === nothing || return v
    _jax_error("the function `", f, "` has no JAX translation for argument types (",
               join((string(a.T) for a in args), ", "), ")", _jax_where(ctx),
               ". Give it one with a `ReactiveKernels._jax_function_call` method, or ",
               "author it as a kernel")
end

# Julia's `*` on arrays is the matrix product.
function _jax_function_call(ctx::_JaxContext, ::typeof(*), args::Vector{_PyVal})
    length(args) == 2 || return _jax_call(ctx, *, [_jax_call(ctx, *, args[1:2]); args[3:end]])
    a, b = args
    if _jax_is_array(a.T) && _jax_is_array(b.T)
        a.T <: LinearAlgebra.Adjoint{<:Any,<:AbstractVector} && ndims(b.T) == 1 &&
            return _jax_result(*, args, "jnp.dot(" * a.code * ", " * b.code * ")")
        ndims(a.T) == 1 && b.T <: LinearAlgebra.Adjoint{<:Any,<:AbstractVector} &&
            return _jax_result(*, args, "jnp.outer(" * a.code * ", " * b.code * ")")
        return _jax_result(*, args, "(" * a.code * " @ " * b.code * ")")
    end
    _jax_result(*, args, "(" * a.code * " * " * b.code * ")")
end
function _jax_function_call(ctx::_JaxContext, ::typeof(_native_exp), args::Vector{_PyVal})
    x = only(args)
    _jax_is_array(x.T) && ndims(x.T) == 2 ||
        _jax_error("`exp` of a `", x.T, "`", _jax_where(ctx))
    _jax_result(_native_exp, args, "jax.scipy.linalg.expm(" * x.code * ")")
end
_jax_function_call(ctx::_JaxContext, ::typeof(exp), args::Vector{_PyVal}) =
    _jax_function_call(ctx, _native_exp, args)
_jax_function_call(ctx::_JaxContext, ::typeof(adjoint), args::Vector{_PyVal}) =
    _jax_result(adjoint, args, ndims(args[1].T) == 1 ? args[1].code :
                "jnp.transpose(" * args[1].code * ")")
_jax_function_call(ctx::_JaxContext, ::typeof(transpose), args::Vector{_PyVal}) =
    _jax_result(transpose, args, ndims(args[1].T) == 1 ? args[1].code :
                "jnp.transpose(" * args[1].code * ")")
function _jax_function_call(ctx::_JaxContext, ::typeof(^), args::Vector{_PyVal})
    _jax_error("a matrix power has no JAX translation", _jax_where(ctx))
end
function _jax_function_call(ctx::_JaxContext, ::typeof(/), args::Vector{_PyVal})
    _jax_error("a matrix division has no JAX translation", _jax_where(ctx))
end

function _jax_function_call(ctx::_JaxContext, ::typeof(sum), args::Vector{_PyVal})
    if length(args) == 1 && !isempty(args[1].extent) && _jax_rank(args[1].T) == 1
        x = args[1]
        return _PyVal("jnp.sum(jnp.where(" * _jax_mask(x) * ", " * x.code * ", 0.0))", eltype(x.T))
    end
    length(args) == 1 && return _jax_result(sum, args, "jnp.sum(" * args[1].code * ")")
    length(args) == 2 && _jax_is_array(args[2].T) || _jax_error("this `sum` call", _jax_where(ctx))
    mapped = _jax_broadcast(ctx, _jax_function_of(args[1]), [args[2]])
    _PyVal("jnp.sum(" * mapped.code * ")", eltype(mapped.T))
end
for (f, name, neutral) in ((prod, "jnp.prod", "1.0"), (maximum, "jnp.max", "-jnp.inf"),
                           (minimum, "jnp.min", "jnp.inf"))
    @eval function _jax_function_call(ctx::_JaxContext, ::typeof($f), args::Vector{_PyVal})
        length(args) == 1 || _jax_error("this `", $f, "` call", _jax_where(ctx))
        x = args[1]
        isempty(x.extent) || return _PyVal($name * "(jnp.where(" * _jax_mask(x) * ", " * x.code *
                                           ", " * $neutral * "))", eltype(x.T))
        _jax_result($f, args, $name * "(" * x.code * ")")
    end
end
function _jax_function_call(ctx::_JaxContext, ::typeof(vec), args::Vector{_PyVal})
    x = only(args)
    (_jax_rank(x.T) == 1 || _jax_ragged(x.T)) && return x
    _jax_is_array(x.T) || _jax_error("`vec` of a `", x.T, "`", _jax_where(ctx))
    _PyVal("jnp.ravel(" * x.code * ", order=\"F\")", Vector{eltype(x.T)})
end
# Julia reshapes in column-major order.
function _jax_function_call(ctx::_JaxContext, ::typeof(reshape), args::Vector{_PyVal})
    x, dims = args[1], args[2:end]
    length(dims) == 1 && dims[1].T isa Type && dims[1].T <: Tuple &&
        _jax_error("`reshape` with a dimension tuple value", _jax_where(ctx))
    all(d -> d.T isa Type && d.T <: Integer, dims) ||
        _jax_error("`reshape` with dimensions of types (", join((string(d.T) for d in dims), ", "),
                   ")", _jax_where(ctx))
    _PyVal("jnp.reshape(" * x.code * ", (" * join((d.code for d in dims), ", ") *
           (length(dims) == 1 ? ",)" : ")") * ", order=\"F\")", Array{eltype(x.T),length(dims)})
end
function _jax_function_call(ctx::_JaxContext, ::typeof(LogExpFunctions.logsumexp), args::Vector{_PyVal})
    length(args) == 1 || _jax_error("this `logsumexp` call", _jax_where(ctx))
    x = args[1]
    isempty(x.extent) || return _PyVal("jsp.logsumexp(jnp.where(" * _jax_mask(x) * ", " * x.code *
                                       ", -jnp.inf))", eltype(x.T))
    _jax_result(LogExpFunctions.logsumexp, args, "jsp.logsumexp(" * x.code * ")")
end
function _jax_function_call(ctx::_JaxContext, ::typeof(length), args::Vector{_PyVal})
    x = only(args)
    isempty(x.extent) || _jax_ragged(x.T) ||
        return _PyVal(x.extent, Int, false, "", x.code * ".shape[0]")
    _jax_ragged(x.T) && return _PyVal(x.code * ".shape[0]", Int, true)
    x.T isa Type && x.T <: Tuple && return _PyVal(string(fieldcount(x.T)), Int, true)
    _PyVal(x.code * ".size", Int, true)
end
function _jax_function_call(ctx::_JaxContext, ::typeof(size), args::Vector{_PyVal})
    length(args) == 2 || return _PyVal(args[1].code * ".shape", NTuple{_jax_rank(args[1].T),Int}, true)
    d = match(r"^[0-9]+$", args[2].code)
    d === nothing && _jax_error("`size` with a computed dimension", _jax_where(ctx))
    _PyVal(args[1].code * ".shape[" * string(parse(Int, args[2].code) - 1) * "]", Int, true)
end
_jax_function_call(ctx::_JaxContext, ::typeof(identity), args::Vector{_PyVal}) = only(args)
for (f, name) in ((all, "jnp.all"), (any, "jnp.any"))
    @eval _jax_function_call(ctx::_JaxContext, ::typeof($f), args::Vector{_PyVal}) =
        length(args) == 1 ? _PyVal($name * "(" * args[1].code * ")", Bool, args[1].static) :
        _jax_error("this `", $f, "` call", _jax_where(ctx))
end
# Identity comparisons decide while tracing, on static values only.
for (f, op) in (((===), " is "), ((!==), " is not "))
    @eval function _jax_call_positional(ctx::_JaxContext, ::typeof($f), args::Vector{_PyVal})
        a, b = args
        a.T isa Type && b.T isa Type && Base.issingletontype(a.T) && Base.issingletontype(b.T) &&
            return _jax_literal($f(a.T.instance, b.T.instance))
        a.static && b.static ||
            _jax_error("`", $f, "` of values that depend on data", _jax_where(ctx))
        _PyVal("(" * a.code * $op * b.code * ")", Bool, true)
    end
end
_jax_function_call(ctx::_JaxContext, ::typeof(float), args::Vector{_PyVal}) =
    _jax_result(float, args, "jnp.asarray(" * only(args).code * ", dtype=jnp.float64)")
function _jax_function_call(ctx::_JaxContext, ::typeof(LinearAlgebra.dot), args::Vector{_PyVal})
    _jax_result(LinearAlgebra.dot, args, "jnp.vdot(" * _jax_args(args) * ")")
end
_jax_function_call(ctx::_JaxContext, ::typeof(vcat), args::Vector{_PyVal}) =
    _jax_result(vcat, args, "jnp.concatenate([" * join(
        (_jax_is_array(a.T) ? a.code : "jnp.atleast_1d(" * a.code * ")" for a in args), ", ") * "])")
_jax_function_call(ctx::_JaxContext, ::typeof(zero), args::Vector{_PyVal}) =
    _jax_is_array(args[1].T) ? _jax_result(zero, args, "jnp.zeros_like(" * args[1].code * ")") :
    _PyVal("0.0", args[1].T)
_jax_function_call(ctx::_JaxContext, ::typeof(one), args::Vector{_PyVal}) =
    _PyVal("1.0", args[1].T)

_jax_function_call(ctx::_JaxContext, ::typeof(_readonly_gather), args::Vector{_PyVal}) =
    _jax_index_values(ctx, args[1], args[2:end])
for (f, name) in ((ones, "jnp.ones"), (zeros, "jnp.zeros"))
    @eval function _jax_function_call(ctx::_JaxContext, ::typeof($f), args::Vector{_PyVal})
        all(a -> a.T isa Type && a.T <: Integer, args) ||
            _jax_error("`", $f, "` with element type or shape arguments", _jax_where(ctx))
        _PyVal($name * "((" * join((a.code for a in args), ", ") * (length(args) == 1 ? ",)" : ")") *
               ", dtype=jnp.float64)", Array{Float64,length(args)})
    end
end
function _jax_function_call(ctx::_JaxContext, ::typeof(fill), args::Vector{_PyVal})
    x, dims = args[1], args[2:end]
    all(a -> a.T isa Type && a.T <: Integer, dims) || _jax_error("`fill` with a shape tuple", _jax_where(ctx))
    _PyVal("jnp.full((" * join((a.code for a in dims), ", ") * (length(dims) == 1 ? ",)" : ")") *
           ", " * x.code * ")", Array{_jax_eltype(x.T),length(dims)})
end

# Index with already translated index values (an integer or an integer array).
function _jax_index_values(ctx::_JaxContext, x::_PyVal, indices::Vector{_PyVal})
    _jax_is_array(x.T) && length(indices) == ndims(x.T) ||
        _jax_error("a gather from `", x.T, "`", _jax_where(ctx))
    parts = String[]
    kept = 0
    for i in indices
        if i.T isa Type && i.T <: Integer
            push!(parts, _jax_minus_one(i))
        elseif _jax_is_array(i.T) && eltype(i.T) <: Integer && ndims(i.T) == 1
            push!(parts, "(" * i.code * ") - 1"); kept += 1
        else
            _jax_error("the index of type `", i.T, "`", _jax_where(ctx))
        end
    end
    E = eltype(x.T)
    _PyVal(x.code * "[" * join(parts, ", ") * "]", kept == 0 ? E : Array{E,kept})
end

# A range value, `a:b`: the array of its elements.
function _jax_call_positional(ctx::_JaxContext, ::Colon, args::Vector{_PyVal})
    length(args) == 2 || _jax_error("a stepped range has no JAX translation", _jax_where(ctx))
    a, b = args
    _PyVal("jnp.arange(" * a.code * ", " * _jax_plus_one(b) * ")", UnitRange{Int})
end
_jax_plus_one(b::_PyVal) = match(r"^-?[0-9]+$", b.code) === nothing ?
    "(" * b.code * ") + 1" : string(parse(Int, b.code) + 1)
_jax_function_call(ctx::_JaxContext, ::typeof(eachindex), args::Vector{_PyVal}) =
    length(args) == 1 && _jax_is_array(args[1].T) && ndims(args[1].T) == 1 ?
    _PyVal("jnp.arange(1, " * args[1].code * ".shape[0] + 1)", Base.OneTo{Int}) :
    _jax_error("`eachindex` of `", args[1].T, "`", _jax_where(ctx))
function _jax_function_call(ctx::_JaxContext, ::typeof(axes), args::Vector{_PyVal})
    length(args) == 2 || _jax_error("`axes` without a dimension", _jax_where(ctx))
    d = match(r"^[0-9]+$", args[2].code)
    d === nothing && _jax_error("`axes` with a computed dimension", _jax_where(ctx))
    _PyVal("jnp.arange(1, " * args[1].code * ".shape[" * string(parse(Int, args[2].code) - 1) *
           "] + 1)", Base.OneTo{Int})
end
function _jax_function_call(ctx::_JaxContext, ::typeof(collect), args::Vector{_PyVal})
    x = only(args)
    x.T isa Type && x.T <: AbstractRange &&
        return _PyVal(x.code, Vector{eltype(x.T)})
    _jax_is_array(x.T) || _jax_error("`collect` of `", x.T, "`", _jax_where(ctx))
    _PyVal(x.code, Array{eltype(x.T),ndims(x.T)})
end
_jax_function_call(ctx::_JaxContext, ::typeof(cumsum), args::Vector{_PyVal}) =
    length(args) == 1 && _jax_rank(args[1].T) == 1 ?
    _jax_result(cumsum, args, "jnp.cumsum(" * args[1].code * ")") :
    _jax_error("this `cumsum` call", _jax_where(ctx))

# Type-level calls run now, on the values' Julia types.
_jax_type_value(T) = _PyVal("<type>", _JaxCallable(T), true)
_jax_type_of(v::_PyVal) = v.T isa _JaxCallable ? v.T.f : v.T
_jax_call_positional(ctx::_JaxContext, ::typeof(eltype), args::Vector{_PyVal}) =
    _jax_type_value(eltype(_jax_type_of(only(args))))
_jax_call_positional(ctx::_JaxContext, ::typeof(typeof), args::Vector{_PyVal}) =
    _jax_type_value(only(args).T)
_jax_call_positional(ctx::_JaxContext, ::typeof(promote_type), args::Vector{_PyVal}) =
    _jax_type_value(promote_type(map(_jax_type_of, args)...))
function _jax_call_positional(ctx::_JaxContext, f::Union{typeof(one),typeof(zero)},
                              args::Vector{_PyVal})
    x = only(args)
    T = x.T isa _JaxCallable ? x.T.f : x.T
    T isa Type && T <: Number && !(x.T isa _JaxCallable) && !x.static &&
        return _jax_result(f, args, (f === one ? "jnp.ones_like(" : "jnp.zeros_like(") * x.code * ")")
    _jax_is_array(T) && !(x.T isa _JaxCallable) && return _jax_function_call(ctx, f, args)
    T isa Type && T <: Number || _jax_error("`", f, "` of `", T, "`", _jax_where(ctx))
    _jax_literal(f(T))
end
_jax_value(ctx::_JaxContext, ::UndefInitializer, hint) = _PyVal("<undef>", UndefInitializer, true)

# The size of an allocation: a static integer, or a data-dependent one at its
# static bound (the array is then padded, its length the value).
function _jax_allocation_size(ctx::_JaxContext, n::_PyVal)
    n.static && return (n.code, "")
    isempty(n.limit) || return (n.limit, n.code)
    _jax_error("an array whose length depends on data has no JAX translation",
               _jax_where(ctx))
end

# A Julia type called as a function: a conversion.
function _jax_call_positional(ctx::_JaxContext, T::Type, args::Vector{_PyVal})
    isempty(args) && Base.issingletontype(T) && return _PyVal("None", T)
    if T <: AbstractArray && !isempty(args) && args[1].T === UndefInitializer
        dims = args[2:end]
        length(dims) == 1 || _jax_error("allocating a multidimensional array", _jax_where(ctx))
        size, extent = _jax_allocation_size(ctx, only(dims))
        E = T isa UnionAll ? Float64 : eltype(T)
        dtype = E <: Bool ? "jnp.bool_" : E <: Integer ? "jnp.int64" : "jnp.float64"
        return _PyVal("jnp.zeros((" * size * ",), dtype=" * dtype * ")", Vector{E}, false, extent)
    end
    if T <: AbstractArray && length(args) == 1 && _jax_is_array(args[1].T)
        E = T isa UnionAll ? eltype(Base.unwrap_unionall(T)) : eltype(T)
        E isa TypeVar && (E = eltype(args[1].T))
        dtype = E <: Bool ? "jnp.bool_" : E <: Integer ? "jnp.int64" : "jnp.float64"
        return _PyVal("jnp.asarray(" * args[1].code * ", dtype=" * dtype * ")",
                      Array{E,ndims(args[1].T)})
    end
    if T <: AbstractFloat && length(args) == 1
        return _PyVal("jnp.asarray(" * args[1].code * ", dtype=jnp.float64)", T)
    end
    if T <: Integer && length(args) == 1
        return _PyVal("jnp.asarray(" * args[1].code * ", dtype=jnp.int64)", T)
    end
    _jax_error("constructing a `", T, "` has no JAX translation", _jax_where(ctx))
end

# The callable a value stands for, for `sum(f, x)`.
_jax_function_of(v::_PyVal) = v.T isa _JaxCallable ? v.T.f :
    v.T isa Type && v.T <: Function && Base.issingletontype(v.T) ?
    v.T.instance : _jax_error("`", v.code, "` is not a function")

# A dotted call `f.(args...)`: Julia broadcasting, which aligns leading
# dimensions, as NumPy broadcasting, which aligns trailing ones, by giving every
# lower-rank array operand trailing singleton axes up to the result's rank.
function _jax_broadcast(ctx::_JaxContext, f, args::Vector{_PyVal})
    R = maximum((_jax_rank(a.T) for a in args); init = 0)
    padded = map(args) do a
        r = _jax_rank(a.T)
        0 < r < R ? _PyVal(a.code * "[" * join(fill("...", 1), "") *
                           repeat(", None", R - r) * "]", a.T) : a
    end
    eltypes = Any[_jax_eltype(a.T) for a in args]
    E = _jax_return_type(f, eltypes)
    T = R == 0 ? E : Array{E,R}
    if haskey(_JAX_INFIX, f) || f === (*) || f === (^) ||
       _jax_elementwise_name(ctx, f, length(args)) !== nothing
        scalars = [_PyVal(p.code, eltype_or_self(p.T)) for p in padded]
        v = f === (*) ? _PyVal("(" * join((p.code for p in padded), " * ") * ")", E) :
            f === (^) ? _PyVal("(" * padded[1].code * " ** " * padded[2].code * ")", E) :
            _jax_scalar_or_elementwise(ctx, f, scalars)
        v === nothing && _jax_error("broadcasting `", f, "`", _jax_where(ctx))
        return _PyVal(v.code, T, false, R == 0 ? "" : _jax_extent(args))
    end
    # Any other scalar function: a module-level function translated once over
    # scalar arguments, vectorized over the operands.
    name = _jax_scalar_function!(ctx, f, eltypes)
    _PyVal("jnp.vectorize(" * name * ")(" * _jax_args(padded) * ")", T, false,
           R == 0 ? "" : _jax_extent(args))
end

# A module-level Python function computing `f` on scalars of `types`.
function _jax_scalar_function!(ctx::_JaxContext, f, types)
    key = (:scalar, objectid(f), Tuple(types))
    haskey(ctx.mod.helper_names, key) && return ctx.mod.helper_names[key]
    hint = Symbol("_rk_", replace(string(nameof_or(f)), r"[^A-Za-z0-9_]" => "_"))
    name = _jax_identifier(ctx.mod.used, hint)
    ctx.mod.helper_names[key] = name
    scope = _JaxScope()
    params = [_jax_identifier(scope.used, Symbol(:x, i)) for i in eachindex(types)]
    inner = _JaxContext(ctx.mod, scope, ctx.recipe, ctx.source_module,
                        Dict{Symbol,_PyVal}(), nothing)
    v = _jax_call(inner, f, [_PyVal(p, T) for (p, T) in zip(params, types)])
    lines = String["def " * name * "(" * join(params, ", ") * "):"]
    append!(lines, scope.lines)
    push!(lines, "    return " * v.code)
    push!(ctx.mod.helpers, join(lines, "\n"))
    name
end
nameof_or(f) = f isa Function && Base.issingletontype(typeof(f)) ? nameof(f) :
    f isa ScalarDerivativeRule ? typeof(f).parameters[1] : :function
eltype_or_self(@nospecialize(T)) = T isa Type && T <: AbstractArray ? eltype(T) : T


# A source operation called directly (an embedded kernel's operation spliced
# into another recipe's source, or one step of a derivative rule): its
# runtime-generated body, whose globals are qualified, over the arguments.
function _jax_call_positional(ctx::_JaxContext, op::_KernelSourceOp, args::Vector{_PyVal})
    op.f isa _KernelBranch && _jax_error("an embedded lazy branch called as a value",
                                         _jax_where(ctx))
    params, body = _jax_op_body(op.f)
    length(params) == length(args) ||
        _jax_error("an embedded operation takes ", length(params), " arguments, not ",
                   length(args), _jax_where(ctx))
    v = _jax_source_expr(ctx, body, params, args, _jax_source_module(op), _jax_captures(op.f))
    _jax_bind!(ctx, :value, v)
end

const _JAX_RGF = RuntimeGeneratedFunctions
# The parameters and body of a source function: its runtime-generated body
# when it has one (its first parameter is the captures), else its source.
function _jax_op_body(f::_KernelSourceFunction)
    ex = _JAX_RGF.get_expression(f.runtime_f)
    ex isa Expr && ex.head === :-> || _jax_error("an unexpected generated body `", ex, "`")
    params = ex.args[1] isa Expr ? Symbol[a for a in ex.args[1].args] : Symbol[ex.args[1]]
    params[2:end], ex.args[2]
end
_jax_op_body(f::_KernelSplit) = _jax_op_body(f.call)
_jax_op_body(f::_KernelReduction) = _jax_op_body(f.call)
_jax_op_body(f) = _jax_error("an embedded operation `", typeof(f), "` without its source")

# A derivative rule called as a function: its primal cut, which `jax.grad`
# differentiates; the rule's partials are not needed.
function _jax_call_positional(ctx::_JaxContext,
        rule::ScalarDerivativeRule{Name,N,Inputs,Primal}, args::Vector{_PyVal}) where {Name,N,Inputs,Primal}
    signature, steps, ret = Primal
    env = Dict{Symbol,_PyVal}(zip(signature, args))
    for (output, index, arguments) in steps
        op = rule.primal_ops[index]
        inputs = _PyVal[a isa Symbol ? env[a] : _jax_literal(a) for a in arguments]
        v = _jax_call(ctx, op, inputs)
        if output isa Symbol
            env[output] = v
        else
            bound = _jax_bind!(ctx, :outputs, v)
            for (k, o) in enumerate(output)
                env[o] = _PyVal(bound.code * "[" * string(k - 1) * "]",
                                v.T isa Type && v.T <: Tuple ? fieldtype(v.T, k) : Any)
            end
        end
    end
    T = _jax_return_type(rule, Any[a.T for a in args])
    ret isa Symbol ? _PyVal(env[ret].code, T) :
        _PyVal(_jax_tuple_code([env[r] for r in ret]), T)
end

# A prepared kernel called as a function (a distribution endpoint a helper
# calls): a module-level function translated from its plan, once per kernel
# and argument types.
function _jax_call_positional(ctx::_JaxContext, k::PreparedKernel, args::Vector{_PyVal})
    p = k.plan
    length(p.have) == length(args) ||
        _jax_error("a prepared kernel of ", length(p.have), " inputs called with ",
                   length(args), _jax_where(ctx))
    types = Any[a.T for a in args]
    key = (:kernel, objectid(k), types)
    name = get(ctx.mod.helper_names, key, nothing)
    if name === nothing
        hint = length(p.want) == 1 ? only(p.want).name : :kernel
        name = _jax_identifier(ctx.mod.used, Symbol("_rk_", hint))
        ctx.mod.helper_names[key] = name
        lines, f = _jax_function!(ctx.mod, name, p, Tuple(types),
                                  length(p.want) == 1 ? only(p.want).name : Tuple(v.name for v in p.want);
                                  reserve = false)
        push!(ctx.mod.helpers, join(lines, "\n"))
        ctx.mod.helper_names[(:kernel_type, name)] = f
    end
    f = ctx.mod.helper_names[(:kernel_type, name)]
    T = length(f.outputs) == 1 ? last(only(f.outputs)) : Tuple{(last(o) for o in f.outputs)...}
    _jax_bind!(ctx, :value, _PyVal(name * "(" * _jax_args(args) * ")", T))
end

# A `@traceable` helper: its authored body over the arguments, as the compiled
# path reads it. Its locals are scoped to the call.
function _jax_traceable_call(ctx::_JaxContext, f, args::Vector{_PyVal})
    types = Any[a.T for a in args]
    all(T -> T isa Type, types) || return nothing
    source = _traceable_source(typeof(f), types...)
    source === nothing && return nothing
    names, body, mod = source
    length(names) == length(args) || return nothing
    v = _jax_source_expr(ctx, body, collect(Symbol, names), args, mod, nothing)
    _jax_bind!(ctx, :value, v)
end

# A plain Julia helper with no translation of its own: its lowered code, when
# that is straight-line (calls, constants and locals; no loops or branches).
const _JAX_LOWERING_STACK = Any[]

function _jax_lowered_call(ctx::_JaxContext, f, args::Vector{_PyVal})
    f isa Function || return nothing
    types = Any[a.T for a in args]
    all(T -> T isa Type, types) || return nothing
    key = (f, Tuple(types))
    key in _JAX_LOWERING_STACK && _jax_error("the function `", f, "` calls itself with argument ",
        "types (", join(string.(types), ", "), ") in its lowered code, which has no JAX ",
        "translation", _jax_where(ctx))
    push!(_JAX_LOWERING_STACK, key)
    try
        return _jax_lowered_call_body(ctx, f, args, types)
    finally
        pop!(_JAX_LOWERING_STACK)
    end
end

function _jax_lowered_call_body(ctx::_JaxContext, f, args::Vector{_PyVal}, types)
    infos = try
        Base.code_lowered(f, Tuple{types...})
    catch
        return nothing
    end
    length(infos) == 1 || return nothing
    ci = only(infos)
    any(st -> st isa Core.GotoNode || st isa Core.GotoIfNot ||
              (st isa Expr && st.head in (:enter, :leave, :pop_exception)), ci.code) &&
        _jax_error("the function `", f, "` has loops or branches, so its lowered code is ",
                   "not translated", _jax_where(ctx), ". Author it as a kernel, ",
                   "or give it a `ReactiveKernels._jax_function_call` method")
    mod = parentmodule(f)
    ssa = Vector{Any}(undef, length(ci.code))
    slots = Dict{Int,_PyVal}(i + 1 => a for (i, a) in enumerate(args))
    value(x) = x isa Core.SSAValue ? ssa[x.id] :
        x isa Core.SlotNumber ? (haskey(slots, x.id) ? slots[x.id] :
                                 _jax_error("an unassigned local in `", f, "`")) :
        x isa Core.Argument ? slots[x.n] :
        x isa GlobalRef ? _jax_global_value(ctx, getglobal(x.mod, x.name), x.name) :
        x isa QuoteNode && x.value isa Symbol ? _PyVal(":" * String(x.value), Symbol) :
        x isa QuoteNode ? _jax_global_value(ctx, x.value, :constant) :
        x isa Number ? _jax_literal(x) :
        _jax_global_value(ctx, x, :constant)
    for (i, st) in enumerate(ci.code)
        if st isa Core.ReturnNode
            return st.val isa Core.SSAValue || st.val isa Core.SlotNumber ||
                   st.val isa Core.Argument ? value(st.val) : value(st.val)
        elseif st isa Expr && st.head === :(=)
            v = value_of_rhs(ctx, st.args[2], value, mod)
            slots[(st.args[1]::Core.SlotNumber).id] = v
            ssa[i] = v
        elseif st isa Expr && st.head === :call
            ssa[i] = value_of_rhs(ctx, st, value, mod)
        elseif st isa Expr && st.head in (:meta, :code_coverage_effect, :inbounds, :boundscheck)
            ssa[i] = nothing
        elseif st === nothing || st isa LineNumberNode
            ssa[i] = nothing
        else
            ssa[i] = value(st)
        end
    end
    _jax_error("the lowered code of `", f, "` does not return", _jax_where(ctx))
end
function value_of_rhs(ctx::_JaxContext, rhs, value, mod)
    rhs isa Expr && rhs.head === :call || return value(rhs)
    callee = rhs.args[1]
    g = callee isa GlobalRef ? getglobal(callee.mod, callee.name) :
        callee isa Core.SSAValue || callee isa Core.SlotNumber || callee isa Core.Argument ?
            _jax_callable(value(callee)) :
        callee isa QuoteNode ? callee.value : callee
    _jax_bind!(ctx, :t, _jax_call(ctx, g, _PyVal[value(a) for a in rhs.args[2:end]]))
end
# A global read in lowered code: callables pass as themselves, data as values.
_jax_global_value(ctx::_JaxContext, x, hint) = x isa Union{Function,Type,PreparedKernel,Module} ?
    _PyVal("<callable>", _JaxCallable{typeof(x)}(x)) : _jax_value(ctx, x, hint)
struct _JaxCallable{F}
    f::F
end
_jax_callable(v::_PyVal) = v.T isa _JaxCallable ? v.T.f :
    _jax_error("`", v.code, "` is called but is not a callable")

# The calls Julia's lowering writes for ordinary syntax: dotted calls, literal
# powers, tuples, field reads and type applications.
_jax_call_positional(ctx::_JaxContext, ::typeof(Base.broadcasted), args::Vector{_PyVal}) =
    _jax_broadcast(ctx, _jax_function_of(args[1]), args[2:end])
_jax_call_positional(ctx::_JaxContext, ::typeof(Base.materialize), args::Vector{_PyVal}) = only(args)
_jax_call_positional(ctx::_JaxContext, ::typeof(tuple), args::Vector{_PyVal}) =
    _PyVal(_jax_tuple_code(args), Tuple{(a.T for a in args)...}, false, _jax_tuple_extent(args))
function _jax_call_positional(ctx::_JaxContext, ::typeof(Base.literal_pow), args::Vector{_PyVal})
    f, x, p = args
    p.T isa Type && p.T <: Val && !_jax_is_array(x.T) ||
        _jax_error("a literal power of `", x.T, "`", _jax_where(ctx))
    n = p.T.parameters[1]
    _PyVal("(" * x.code * " ** " * string(n) * ")", _jax_return_type(^, Any[x.T, typeof(n)]))
end
function _jax_call_positional(ctx::_JaxContext, ::typeof(getproperty), args::Vector{_PyVal})
    x, field = args
    name = Symbol(strip(field.code, ['"', ':']))
    x.T isa Type && x.T <: NamedTuple || _jax_error("the field read `", name, "` on `", x.T, "`",
                                                    _jax_where(ctx))
    k = findfirst(==(name), fieldnames(x.T))
    k === nothing && _jax_error("`", x.T, "` has no field `", name, "`", _jax_where(ctx))
    _PyVal(x.code * "[" * string(k - 1) * "]", fieldtype(x.T, k))
end
function _jax_call_positional(ctx::_JaxContext, ::typeof(Core.apply_type), args::Vector{_PyVal})
    parts = map(args) do a
        a.T isa _JaxCallable && return a.T.f
        m = match(r"^-?[0-9]+$", a.code)
        m === nothing && _jax_error("a type applied to a computed parameter", _jax_where(ctx))
        parse(Int, a.code)
    end
    T = Core.apply_type(parts...)
    _PyVal("<type>", _JaxCallable(T))
end
_jax_call_positional(ctx::_JaxContext, c::_JaxCallable, args::Vector{_PyVal}) =
    _jax_call(ctx, c.f, args)

# --- programs -----------------------------------------------------------------------

# The Python value of graph value `v` in the current scope chain.
function _jax_lookup(ctx::_JaxContext, g::Graph, v::Value)
    id = canon_id(g, v.id)
    scope = ctx.scope
    while scope !== nothing
        haskey(scope.values, id) && return scope.values[id]
        scope = scope.parent
    end
    _jax_error("the value `", v.name, "` is read before it is computed")
end

function _jax_define!(ctx::_JaxContext, g::Graph, v::Value, val::_PyVal; bind::Bool = true)
    id = canon_id(g, v.id)
    if bind
        name = _jax_fresh(ctx, v.name)
        _jax_emit!(ctx, name * " = " * val.code)
        val = _PyVal(name, val.T, val.static, val.extent)
    end
    ctx.scope.values[id] = val
    val
end

# Emit every recipe of plan `p` into the current scope.
function _jax_plan!(ctx::_JaxContext, p::Plan)
    for recipe in p.recipes
        inputs = _PyVal[_jax_lookup(ctx, p.graph, v) for v in recipe.inputs]
        ctx.recipe = recipe
        outputs = _jax_recipe(ctx, p, recipe, recipe.op, inputs)
        length(outputs) == length(recipe.outputs) ||
            _jax_error("recipe produced ", length(outputs), " values for ",
                       length(recipe.outputs), " outputs", _jax_where(ctx))
        for (v, val) in zip(recipe.outputs, outputs)
            id = canon_id(p.graph, v.id)
            # A collateral output never replaces a value already computed.
            any(s -> haskey(s.values, id), _jax_scope_chain(ctx.scope)) && continue
            _jax_define!(ctx, p.graph, v, val)
        end
    end
end
_jax_scope_chain(scope) = scope === nothing ? _JaxScope[] : [scope; _jax_scope_chain(scope.parent)]

# A multi-output recipe returns a tuple; split it into one value per output.
function _jax_split_outputs(ctx::_JaxContext, recipe::Recipe, v::_PyVal)
    length(recipe.outputs) == 1 && return [v]
    bound = _jax_bind!(ctx, :outputs, v)
    [_PyVal(bound.code * "[" * string(i - 1) * "]",
            v.T isa Type && v.T <: Tuple ? fieldtype(v.T, i) : Any)
     for i in eachindex(recipe.outputs)]
end

# An operation without a recipe-specific translation.
_jax_recipe(ctx::_JaxContext, p::Plan, recipe::Recipe, op, inputs) =
    _jax_error("the operation `", typeof(op), "` has no JAX translation", _jax_where(ctx))

# A bare function identity (`+`, `sum`, `log`): its call on the inputs.
_jax_recipe(ctx::_JaxContext, p::Plan, recipe::Recipe, op::Function, inputs) =
    _jax_split_outputs(ctx, recipe, _jax_call(ctx, op, inputs))

# An operation synthesized from authored source: translate that source over
# the recipe's inputs, named as the source names them.
function _jax_recipe(ctx::_JaxContext, p::Plan, recipe::Recipe, op::_KernelSourceOp, inputs)
    f = op.f
    f isa _KernelBranch && return _jax_split_outputs(ctx, recipe,
        _jax_branch(ctx, recipe, f, inputs))
    _has_source(recipe) || _jax_error("a source operation without its source", _jax_where(ctx))
    v = _jax_source_expr(ctx, recipe.source, _recipe_source_names(recipe), inputs,
                         _jax_source_module(op), _jax_source_captures(op))
    _jax_split_outputs(ctx, recipe, v)
end

_jax_source_module(op::_KernelSourceOp) = _jax_closure_module(_source_closure(op.f))
function _jax_closure_module(f)
    f isa Function || return Main
    ms = methods(f)
    isempty(ms) ? parentmodule(typeof(f)) : first(ms).module
end
_jax_source_captures(op::_KernelSourceOp) = _jax_captures(op.f)
_jax_captures(f::_KernelSourceFunction) = f.captures
_jax_captures(f) = nothing

# Translate `source` with `names[i]` bound to `inputs[i]`.
function _jax_source_expr(ctx::_JaxContext, source, names, inputs, mod::Module, captures)
    saved = (ctx.locals, ctx.source_module, ctx.captures)
    ctx.locals = Dict{Symbol,_PyVal}(zip(names, inputs))
    ctx.source_module = mod
    ctx.captures = captures
    try
        return _jax_expr(ctx, source)
    finally
        ctx.locals, ctx.source_module, ctx.captures = saved
    end
end

# A lazy branch recipe: its condition, and one function per arm. An arm that
# carries its own plan (a plate cell's arm, `then_spec`/`else_spec`) is that
# plan; otherwise the arm is its authored source.
function _jax_branch(ctx::_JaxContext, recipe::Recipe, b::_KernelBranch, inputs)
    source = recipe.source
    names = _recipe_source_names(recipe)
    source isa Expr && source.head === :if ||
        _jax_error("a branch whose source is not an `if`", _jax_where(ctx))
    env = Dict{Symbol,_PyVal}(zip(names, inputs))
    mod = _jax_source_module(recipe.op)
    condition = _jax_source_expr(ctx, source.args[1], names, inputs, mod, nothing)
    arm(spec, ex, hint) = spec isa Plan ?
        _jax_arm_function(ctx, hint, () -> _jax_arm_plan(ctx, spec, env)) :
        _jax_arm_function(ctx, hint, () -> _jax_source_expr(ctx, ex, names, inputs, mod, nothing))
    then_arm = arm(b.then_spec, source.args[2], :then_arm)
    else_arm = arm(b.else_spec, source.args[3], :else_arm)
    T = promote_type(_jax_typeof(then_arm), _jax_typeof(else_arm))
    _PyVal("jax.lax.cond(" * condition.code * ", " * then_arm.code * ", " *
           else_arm.code * ")", T)
end

# An arm's plan, its HAVE ports bound by name to the branch's arguments.
function _jax_arm_plan(ctx::_JaxContext, p::Plan, env::Dict{Symbol,_PyVal})
    for v in p.have
        haskey(env, v.name) || _jax_error("a branch arm reads `", v.name,
                                          "`, which the branch does not pass", _jax_where(ctx))
        ctx.scope.values[canon_id(p.graph, v.id)] = env[v.name]
    end
    recipe = ctx.recipe
    try
        _jax_plan!(ctx, p)
    finally
        ctx.recipe = recipe
    end
    _jax_lookup(ctx, p.graph, only(p.want))
end

# A nested function over plan `p`: its HAVE values are the parameters, its
# WANT values the result. `types` are the parameters' Julia types; a parameter
# with `extents[i]` true is padded and takes its length (or lengths) as an
# extra parameter after it. A WANT with an extent returns its length too.
# Returns the function name, the WANT values and which of them return a length.
function _jax_plan_function!(ctx::_JaxContext, hint, p::Plan, types;
                             extents = fill(false, length(p.have)))
    name = _jax_fresh(ctx, hint)
    outer = ctx.scope
    inner = _JaxScope(outer)
    inner.indent = outer.indent + 1
    names = String[]
    for (i, v) in enumerate(p.have)
        param = _jax_identifier(inner.used, v.name)
        push!(names, param)
        extent = ""
        if extents[i]
            extent = _jax_identifier(inner.used, Symbol(v.name, :_n))
            push!(names, extent)
        end
        inner.values[canon_id(p.graph, v.id)] = _PyVal(param, types[i], false, extent)
    end
    recipe = ctx.recipe
    ctx.scope = inner
    wants = try
        _jax_plan!(ctx, p)
        [_jax_lookup(ctx, p.graph, v) for v in p.want]
    finally
        ctx.scope = outer
        ctx.recipe = recipe
    end
    returned = String[]
    for w in wants
        push!(returned, w.code)
        isempty(w.extent) || push!(returned, w.extent)
    end
    push!(outer.lines, "    "^outer.indent * "def " * name * "(" * join(names, ", ") * "):")
    append!(outer.lines, inner.lines)
    push!(outer.lines, "    "^inner.indent * "return " * (length(returned) == 1 ?
        returned[1] : "(" * join(returned, ", ") * ")"))
    name, wants, Bool[!isempty(w.extent) for w in wants]
end

# A plate: the cell as a function, mapped over the batched arguments with
# `jax.vmap` (one mapped axis per broadcast dimension). Shared arguments
# (`Ref` operands, and scalars, which repeat) are not mapped. A ragged
# argument is mapped row by row with its lengths; a padded group vector maps
# its entries, and the plate's values are padded the same way.
function _jax_recipe(ctx::_JaxContext, p::Plan, recipe::Recipe,
                     op::_AuthoredPlateOp{K,A}, inputs) where {K,A}
    body = op.kernel.plan
    # A tuple of numbers is a plate axis too; it is mapped as a stacked array.
    inputs = [!(i in A) && _jax_numeric_tuple(v.T) ?
              _PyVal("jnp.stack(" * v.code * ")", Vector{eltype(v.T)}) : v
              for (i, v) in enumerate(inputs)]
    mapped = Bool[!(i in A) && _jax_is_array(v.T) for (i, v) in enumerate(inputs)]
    any(mapped) || _jax_error("a plate without a batched array argument", _jax_where(ctx))
    ranks = unique(_jax_rank(inputs[i].T) for i in eachindex(inputs) if mapped[i])
    length(ranks) == 1 || _jax_error("a plate over arrays of different ranks (",
        join(string.(ranks), ", "), ") has no JAX translation yet", _jax_where(ctx))
    R = only(ranks)
    # Arguments whose cell value is padded: a ragged array's rows, and the
    # padded entries of an array of tuples (a shared cell plate's values).
    ragged = Bool[!isempty(v.extent) && (_jax_ragged(v.T) || (_jax_is_array(v.T) &&
                  eltype(v.T) <: Tuple) || !mapped[i]) && !(mapped[i] && eltype(v.T) <: Number)
                  for (i, v) in enumerate(inputs)]
    any(ragged[i] && mapped[i] for i in eachindex(inputs)) && R != 1 &&
        _jax_error("a ragged plate of rank ", R, _jax_where(ctx))
    # A mapped padded group vector: the plate runs over its entries.
    padded = [mapped[i] && !ragged[i] && !isempty(v.extent) for (i, v) in enumerate(inputs)]
    cell_types = Any[mapped[i] ? eltype(v.T) : _jax_unref(v.T) for (i, v) in enumerate(inputs)]
    cell_extents = Bool[ragged[i] for i in eachindex(inputs)]
    # A lazy branch in the cell stays a `lax.cond`. Under `vmap`, JAX batches a
    # `lax.cond` whose condition differs between cells by evaluating both arms
    # and selecting; each lane's untaken arm reads its inputs through
    # `stop_gradient`, so a NaN or Inf it computes reaches neither the selected
    # value nor the gradient (measured with JAX 0.12.0.1: the batched jaxpr
    # selects `stop_gradient` inputs per arm; a vmapped `cond(x > 0, log(x), 0)`
    # at x = 0 has gradient 0, where `jnp.where` gives NaN).
    cell, wants, with_length = _jax_plan_function!(ctx, :cell, body, cell_types;
                                                   extents = cell_extents)
    want = only(wants)
    E = _jax_typeof(want)
    axes = String[]
    args = String[]
    for (i, v) in enumerate(inputs)
        push!(axes, mapped[i] ? "0" : "None"); push!(args, v.code)
        if ragged[i]
            push!(axes, mapped[i] ? "0" : "None"); push!(args, v.extent)
        end
    end
    axes_text = "(" * join(axes, ", ") * (length(axes) == 1 ? ",)" : ")")
    call = cell
    for _ in 1:R
        call = "jax.vmap(" * call * ", in_axes=" * axes_text * ")"
    end
    call *= "(" * join(args, ", ") * ")"
    if E <: Tuple && !(E <: NamedTuple)
        R == 1 || _jax_error("a plate of rank ", R, " whose cell returns a tuple", _jax_where(ctx))
        result = _jax_fresh(ctx, :plate_value)
        if only(with_length)
            lengths = _jax_fresh(ctx, :plate_lengths)
            _jax_emit!(ctx, result * ", " * lengths * " = " * call)
            return [_PyVal(result, Vector{E}, false, lengths)]
        end
        _jax_emit!(ctx, result * " = " * call)
        return [_PyVal(result, Vector{E})]
    end
    if E <: AbstractVector
        R == 1 || _jax_error("a plate of rank ", R, " whose cell returns a vector", _jax_where(ctx))
        result = _jax_fresh(ctx, :plate_value)
        lengths = _jax_fresh(ctx, :plate_lengths)
        if only(with_length)
            _jax_emit!(ctx, result * ", " * lengths * " = " * call)
        else
            _jax_emit!(ctx, result * " = " * call)
            _jax_emit!(ctx, lengths * " = jnp.full((" * result * ".shape[0],), " * result *
                            ".shape[1])")
        end
        return [_PyVal(result, Vector{E}, false, lengths)]
    end
    E <: Number || _jax_error("a plate whose cell returns `", E, "` has no JAX translation yet",
                              _jax_where(ctx))
    extent = _jax_extent(inputs[padded])
    [_PyVal(call, Array{E,R}, false, extent)]
end
_jax_numeric_tuple(@nospecialize(T)) = T isa Type && T <: Tuple && fieldcount(T) > 0 &&
    isconcretetype(eltype(T)) && eltype(T) <: Number
_jax_unref(@nospecialize(T)) = T isa Type && T <: Base.RefValue ? eltype(T) : T
_jax_unref_val(v::_PyVal) = v

# A scan: the step as a function of the carry and one element of each
# sequence, run by `jax.lax.scan`. Shared operands are read by the step's
# closure.
function _jax_recipe(ctx::_JaxContext, p::Plan, recipe::Recipe,
                     op::_AuthoredScanOp{K,A,I,H}, inputs) where {K,A,I,H}
    H && _jax_error("a scan with `history` has no JAX translation yet", _jax_where(ctx))
    body = op.kernel.plan
    iterated = [i for i in 2:length(inputs) if !(i in A)]
    shared = [i for i in 2:length(inputs) if i in A]
    types = Any[inputs[1].T; [eltype(inputs[i].T) for i in iterated]; [inputs[i].T for i in shared]]
    # The step is called with (carry, xs); unpack xs and bind shared operands.
    step = _jax_fresh(ctx, :step)
    carry = _jax_fresh(ctx, :carry)
    xs = _jax_fresh(ctx, :xs)
    outer = ctx.scope
    inner = _JaxScope(outer)
    inner.indent = outer.indent + 1
    push!(inner.used, carry, xs)
    have = body.have
    inner.values[canon_id(body.graph, have[1].id)] = _PyVal(carry, types[1])
    unpack = String[]
    for (k, i) in enumerate(iterated)
        name = _jax_identifier(inner.used, have[1 + k].name)
        push!(unpack, name)
        inner.values[canon_id(body.graph, have[1 + k].id)] = _PyVal(name, types[1 + k])
    end
    for (k, i) in enumerate(shared)
        inner.values[canon_id(body.graph, have[1 + length(iterated) + k].id)] = inputs[i]
    end
    recipe_saved = ctx.recipe
    ctx.scope = inner
    wants = try
        isempty(unpack) || _jax_emit!(ctx, join(unpack, ", ") * (length(unpack) == 1 ? "," : "") *
                                            " = " * xs)
        _jax_plan!(ctx, body)
        [_jax_lookup(ctx, body.graph, v) for v in body.want]
    finally
        ctx.scope = outer
        ctx.recipe = recipe_saved
    end
    push!(outer.lines, "    "^outer.indent * "def " * step * "(" * carry * ", " * xs * "):")
    append!(outer.lines, inner.lines)
    push!(outer.lines, "    "^inner.indent * "return " * wants[1].code * ", " * wants[2].code)
    sequences = "(" * join((inputs[i].code for i in iterated), ", ") *
                (length(iterated) == 1 ? ",)" : ")")
    result = _jax_fresh(ctx, :scan_result)
    _jax_emit!(ctx, "_, " * result * " = jax.lax.scan(" * step * ", " * inputs[1].code *
                    ", " * sequences * ")")
    E = _jax_typeof(wants[2])
    if I
        _jax_emit!(ctx, result * " = jnp.concatenate([jnp.atleast_1d(" * inputs[1].code *
                        "), " * result * "])")
    end
    E <: Number || _jax_error("a scan whose step outputs `", E, "` has no JAX translation yet",
                              _jax_where(ctx))
    [_PyVal(result, Vector{E})]
end

# --- module -------------------------------------------------------------------------

"""
    JAXFunction

One function of a [`JAXSource`](@ref) module: its Python `name`, its
parameters `inputs` and its results `outputs` (each `name => Julia type`).
"""
struct JAXFunction
    name::String
    inputs::Vector{Pair{String,Any}}
    outputs::Vector{Pair{String,Any}}
end

"""
    JAXSource

The Python module [`jax_source`](@ref) emits: `text` is the module source and
`functions` the [`JAXFunction`](@ref)s it defines. `write(path, src)` saves the
module and `print(src)` shows it.
"""
struct JAXSource
    text::String
    functions::Vector{JAXFunction}
end
Base.print(io::IO, src::JAXSource) = print(io, src.text)
Base.show(io::IO, ::MIME"text/plain", src::JAXSource) = print(io, src.text)
Base.write(io::IO, src::JAXSource) = write(io, src.text)
Base.write(path::AbstractString, src::JAXSource) = write(path, src.text)

# The host-side helper that turns a ragged input (one array per group) into
# the padded array and lengths vector the functions take.
const _JAX_PAD_RAGGED = """
def pad_ragged(arrays):
    \"\"\"Pad a list of 1-D arrays into one 2-D array and its lengths vector.

    Each row repeats its last entry up to the longest length, so every padded
    slot holds a valid value; the functions mask padded entries by length.
    Pass the two results as the input and its `_lengths` input.
    \"\"\"
    import numpy as np
    rows = [np.asarray(a) for a in arrays]
    lengths = np.asarray([r.shape[0] for r in rows], dtype=np.int64)
    width = max([1] + [int(n) for n in lengths])
    padded = np.stack([np.pad(r, (0, width - r.shape[0]), mode="edge") if r.shape[0] > 0
                       else np.zeros(width) for r in rows])
    return padded, lengths
"""

const _JAX_HEADER = """
import jax
import jax.numpy as jnp
import jax.scipy.linalg
import jax.scipy.special as jsp

jax.config.update("jax_enable_x64", True)
"""

"""
    jax_source(spec::KernelSpec; have, want, types = nothing, name = :kernel) -> JAXSource
    jax_source(spec::KernelSpec; have, functions, types = nothing, constants = (;),
               doc = nothing) -> JAXSource
    jax_source(p::Plan; types = nothing, name = :kernel) -> JAXSource

Translate the program that computes `want` from `have` into a Python module
whose function `name` takes the HAVE values, in order, and returns the WANT
value (a tuple for several WANTs), computed with `jax.numpy`. With
`functions`, a `NamedTuple` of function name => WANT, the module defines one
function per entry over the same HAVE values; a WANT is a value name, a tuple
of them (the function returns a tuple) or a `NamedTuple` of key => value name
(it returns a `dict`). `constants`, a `NamedTuple`, adds module-level
constants (numbers, strings, symbols, vectors, tuples, named tuples and
dictionaries of these), and `doc` replaces the module docstring.

`types` gives each HAVE value's Julia type, as a `NamedTuple` by port name or
a tuple in HAVE order; by default the ports' declared types are used, and
they must be concrete. Value ranks follow from these types, so the module is
generic in array sizes: one module serves every size of the same program.

The translation keeps the program's structure. A plate is `jax.vmap` of its
cell, a scan is `jax.lax.scan` of its step, a lazy branch is `jax.lax.cond`
of its arms, and nothing is unrolled per element. Arrays keep Julia's
dimension order, Julia's broadcasting rules are kept, and integer index data
stay 1-based. Gradients are `jax.grad` of a function. A construct without a
translation raises a [`JAXEmissionError`](@ref) naming its recipe.
"""
function jax_source(spec::KernelSpec; have = _KERNEL_DEFAULT_BOUNDARY,
                    want = nothing, functions = nothing, types = nothing,
                    name = :kernel, constants = NamedTuple(), doc = nothing)
    functions === nothing && want === nothing && (want = _KERNEL_DEFAULT_BOUNDARY)
    if functions === nothing
        functions = NamedTuple{(Symbol(name),)}((want,))
    else
        want === nothing || throw(ArgumentError("jax_source: pass `want` or `functions`, not both"))
    end
    functions isa NamedTuple && !isempty(functions) || throw(ArgumentError(
        "jax_source: `functions` must be a nonempty NamedTuple of function name => WANT"))
    plans = [fname => (plan(spec; have, want = _jax_want_names(w)), w)
             for (fname, w) in pairs(functions)]
    _jax_module_source(plans, types, constants, doc)
end

jax_source(p::Plan; types = nothing, name = :kernel, constants = NamedTuple(),
           doc = nothing) =
    _jax_module_source([Symbol(name) => (p, Tuple(v.name for v in p.want))],
                       types, constants, doc)

_jax_want_names(w::Symbol) = (w,)
_jax_want_names(w::Tuple) = w
_jax_want_names(w::NamedTuple) = Tuple(values(w))
_jax_want_names(w) = throw(ArgumentError(
    "jax_source: a function's WANT is a value name, a tuple of names or a NamedTuple; got `$w`"))

function _jax_module_source(plans, types, constants, doc)
    mod = _JaxModule()
    union!(mod.used, ("jax", "jnp", "jsp"))
    constant_lines = String[]
    for (key, value) in pairs(constants)
        cname = _jax_identifier(mod.used, key)
        push!(constant_lines, cname * " = " * _jax_py_literal(value))
    end
    bodies = String[]
    functions = JAXFunction[]
    for (fname, (p, w)) in plans
        lines, f = _jax_function!(mod, fname, p, types, w)
        push!(bodies, join(lines, "\n"))
        push!(functions, f)
    end
    any(f -> any(io -> _jax_ragged(last(io)), f.inputs), functions) &&
        pushfirst!(mod.helpers, rstrip(_JAX_PAD_RAGGED))
    lines = String[]
    docstring = doc === nothing ?
        "Generated by ReactiveKernels.jax_source; do not edit." : String(doc)
    push!(lines, "\"\"\"" * replace(docstring, "\"\"\"" => "\\\"\\\"\\\"") * "\"\"\"")
    push!(lines, rstrip(_JAX_HEADER))
    isempty(constant_lines) || (push!(lines, ""); append!(lines, constant_lines))
    for block in [mod.helpers; bodies]
        push!(lines, "")
        push!(lines, "")
        push!(lines, block)
    end
    JAXSource(join(lines, "\n") * "\n", functions)
end

# One function `fname` over plan `p`, returning WANT `w` in its shape.
function _jax_function!(mod::_JaxModule, fname, p::Plan, types, w; reserve::Bool = true)
    argtypes = _jax_have_types(p, types)
    name = reserve ? _jax_identifier(mod.used, fname) : String(fname)
    scope = _JaxScope()
    ctx = _JaxContext(mod, scope, nothing, Main, Dict{Symbol,_PyVal}(), nothing)
    params = String[]
    inputs = Pair{String,Any}[]
    for (v, T) in zip(p.have, argtypes)
        param = _jax_identifier(scope.used, v.name)
        push!(params, param); push!(inputs, param => T)
        extent = ""
        if _jax_ragged(T)
            extent = _jax_identifier(scope.used, Symbol(v.name, :_lengths))
            push!(params, extent); push!(inputs, extent => Vector{Int})
        end
        scope.values[canon_id(p.graph, v.id)] = _PyVal(param, T, false, extent)
    end
    _jax_plan!(ctx, p)
    wants = [_jax_returned(ctx, _jax_lookup(ctx, p.graph, v)) for v in p.want]
    returned = if w isa NamedTuple
        "{" * join(("\"" * string(k) * "\": " * v.code for (k, v) in zip(keys(w), wants)), ", ") * "}"
    elseif w isa Symbol
        only(wants).code
    else
        _jax_tuple_code(wants)
    end
    lines = String["def " * name * "(" * join(params, ", ") * "):"]
    append!(lines, scope.lines)
    push!(lines, "    return " * returned)
    lines, JAXFunction(name, inputs, [String(v.name) => x.T for (v, x) in zip(p.want, wants)])
end

# A returned named tuple is a `dict` keyed by its field names.
function _jax_returned(ctx::_JaxContext, v::_PyVal)
    v.T isa Type && v.T <: NamedTuple && fieldcount(v.T) > 0 || return v
    bound = _jax_bind!(ctx, :result, v)
    names = fieldnames(v.T)
    _PyVal("{" * join(("\"" * string(n) * "\": " * bound.code * "[" * string(i - 1) * "]"
                       for (i, n) in enumerate(names)), ", ") * "}", v.T)
end

# Module-level metadata as Python literals.
_jax_py_literal(x::Union{Bool,Integer,AbstractFloat,Nothing}) = _jax_scalar_meta(x)
_jax_scalar_meta(x::Bool) = x ? "True" : "False"
_jax_scalar_meta(x::Integer) = string(x)
_jax_scalar_meta(x::AbstractFloat) = _jax_scalar_text(x)
_jax_scalar_meta(::Nothing) = "None"
_jax_py_literal(x::Union{AbstractString,Symbol}) = repr(String(x))
# A vector of pairs is an ordered `dict`.
_jax_py_literal(x::AbstractVector{<:Pair}) =
    "{" * join((_jax_py_literal(k) * ": " * _jax_py_literal(v) for (k, v) in x), ", ") * "}"
_jax_py_literal(x::Union{AbstractVector,Tuple}) =
    (x isa Tuple ? "(" : "[") * join((_jax_py_literal(v) for v in x), ", ") *
    (x isa Tuple ? (length(x) == 1 ? ",)" : ")") : "]")
_jax_py_literal(x::NamedTuple) =
    "{" * join((repr(string(k)) * ": " * _jax_py_literal(v) for (k, v) in pairs(x)), ", ") * "}"
_jax_py_literal(x::AbstractDict) =
    "{" * join((_jax_py_literal(k) * ": " * _jax_py_literal(v) for (k, v) in x), ", ") * "}"
_jax_py_literal(x) = throw(ArgumentError("jax_source: no Python literal for a `$(typeof(x))` constant"))

function _jax_have_types(p::Plan, types)
    names = [v.name for v in p.have]
    argtypes = if types === nothing
        Any[valtype(v) for v in p.have]
    elseif types isa NamedTuple
        Any[haskey(types, n) ? types[n] : valtype(v) for (n, v) in zip(names, p.have)]
    else
        length(types) == length(p.have) || throw(ArgumentError(
            "jax_source: $(length(types)) types for $(length(p.have)) HAVE values"))
        Any[types...]
    end
    for (n, T) in zip(names, argtypes)
        _concrete_value_type(T) || throw(ArgumentError(
            "jax_source: the HAVE value `$n` has type `$T`; pass a concrete type with `types`"))
    end
    argtypes
end
