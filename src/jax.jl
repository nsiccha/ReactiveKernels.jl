# JAX source emission (todo `1l3nni7`; route: agent call `0rnbx63`).
#
# `jax_source` translates a planned program into a Python module whose
# function computes the plan's WANT values from its HAVE values with
# `jax.numpy`. The translation is structural: the plan is walked in execution
# order, an ordinary recipe becomes one assignment translated from its authored
# right-hand side, a plate becomes `jax.vmap` of its cell, a scan becomes
# `jax.lax.scan` of its step and a lazy branch becomes `jax.lax.cond` of its
# arms. Nothing is unrolled per element (`CONTRIBUTING.md`, "Preserve
# data-dependent iteration"), and sizes are read from the inputs, so one
# emitted module serves every data size of the same program. Gradients are
# `jax.grad` of the emitted callable; scalar derivative rules contribute
# activity-selected cuts of their mathematical graph. A construct the emitter cannot translate raises a
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
#
# `padded` records which Julia axes of an array of rank two or more its
# `extent` pads, as a bitmask (bit `k - 1` for axis `k`): a padded vector
# broadcast against a matrix pads the result's first axis, `I(n)` at a padded
# size both. Zero where no translation recorded them; a vector pads its one
# axis and an adjoint vector its second, by type (`_jax_padded_axes`).
# For a tuple, or a vector of tuples (a tuple-valued plate's values), whose
# `extent` is a tuple of its fields' lengths, `padded` records instead which
# fields have one (bit `k - 1` for field `k`, `_jax_padded_fields`): the
# others are `None` there, and no length at all.
#
# `origin` records where a data integer comes from (`_JaxOrigin`), so a loop
# over a range whose ends are entries of the prepared data has a static
# number of steps (`_jax_trip_bound`).
struct _PyVal
    code::String
    T::Any
    static::Bool
    extent::String
    limit::String
    padded::Int
    origin::Any
    # A tuple's individual length expressions. Keeping them separate lets
    # projection retain a field's prepared-input provenance, even when other
    # fields have different ragged lengths. Empty for non-tuple values.
    field_extents::Tuple{Vararg{String}}
end
_PyVal(code::AbstractString, @nospecialize(T), static::Bool, extent::AbstractString,
       limit::AbstractString, padded::Int, origin) =
    _PyVal(String(code), T, static, String(extent), String(limit), padded, origin, ())
_PyVal(code::AbstractString, @nospecialize(T), static::Bool, extent::AbstractString,
       limit::AbstractString, padded::Int) =
    _PyVal(String(code), T, static, String(extent), String(limit), padded, nothing)
# An integer array of the prepared data, `prepared[key]` (`position ===
# nothing`), or an entry of one, `scale * prepared[key][position] + offset`:
# `position` is the gather's index code, or a token naming a plate cell's
# entry, and `scale` and `offset` static integers (`2 * n[i] + 1` of two
# interleaved reads per entry). Entries of two arrays at one position are
# entries of one record (`starts[r]:stops[r]`), so their difference is
# bounded by the arrays' largest difference, which `prepare_data` computes.
struct _JaxOrigin
    key::String
    position::Union{Nothing,String}
    offset::String
    path::String
    scale::String
end
_JaxOrigin(key::String, position::Union{Nothing,String}, offset::String, path::String = "") =
    _JaxOrigin(key, position, offset, path, "1")
# A data-only cell recipe that can be moved to prepare_data if a later loop
# needs a bound on its integer result. The existing host split does the work.
struct _JaxCellData
    plate::Int
    recipe::Int
end
# An integer array, including a ragged array of integer arrays. Bounds are
# computed over its prepared representation, not over a Julia exemplar.
_jax_integer_data(@nospecialize(T)) = T isa Type &&
    (T <: Integer || T <: AbstractArray && _jax_integer_data(eltype(T)))
# Selecting a row keeps its original prepared array and records the row's
# position. A later scalar selection completes that position. This lets two
# ends in the same row/scan step share a span bound without confusing entries
# read from different rows.
function _jax_selected_origin(source::_PyVal, @nospecialize(T), position::String)
    o = source.origin
    o isa _JaxCellData && _jax_integer_data(T) && return o
    o isa _JaxOrigin && o.position === nothing && _jax_integer_data(T) || return nothing
    path = isempty(o.path) ? position : repr((o.path, position))
    T <: Integer ? _JaxOrigin(o.key, path, "0") : _JaxOrigin(o.key, nothing, "0", path)
end
# An element of a loop's or generator's range: at least `lo` and at most
# `hi`, its ends, so a range from or to it is bounded by theirs (`for s in
# starts[r]:t` with `t` of `starts[r]:stops[r]`, `for j in i:n` with `i` of
# `1:n`).
struct _JaxBetween
    lo::_PyVal
    hi::_PyVal
end
# The value whose static bounds bound `v` from below (above): the first
# (last) end of the range `v` is an element of, or `v` itself.
_jax_lower_end(v::_PyVal) = _jax_lower_end(v, v.origin)
_jax_upper_end(v::_PyVal) = _jax_upper_end(v, v.origin)
_jax_lower_end(v::_PyVal, origin) = v
_jax_upper_end(v::_PyVal, origin) = v
_jax_lower_end(v::_PyVal, origin::_JaxBetween) = _jax_lower_end(origin.lo)
_jax_upper_end(v::_PyVal, origin::_JaxBetween) = _jax_upper_end(origin.hi)
# A range element's index value `index`, between `lo` and `hi`.
_jax_range_index(index::AbstractString, lo::_PyVal, hi::_PyVal) =
    _PyVal(index, Int, false, "", "", 0, _JaxBetween(lo, hi))
_jax_with_origin(v::_PyVal, origin) =
    _PyVal(v.code, v.T, v.static, v.extent, v.limit, v.padded, origin, v.field_extents)
struct _JaxCallable{F}
    f::F
end
# Values assembled while translating retain their fields as translated values,
# not Julia exemplars. In particular, numeric closure captures remain inputs.
struct _JaxClosure <: Function
    F::Type
    captures::Vector{_PyVal}
end
struct _JaxAggregate
    T::Type
    fields::Vector{_PyVal}
end
_jax_callable_type(f) = Core.Typeof(f)
_jax_callable_type(f::_JaxClosure) = f.F
_jax_argument_type(v::_PyVal) = v.T isa _JaxCallable ? _jax_callable_type(v.T.f) :
    v.T isa _JaxAggregate ? v.T.T : v.T
_jax_translation_only(v::_PyVal) = v.T isa Union{_JaxCallable,_JaxAggregate}
_PyVal(code::AbstractString, @nospecialize(T)) = _PyVal(String(code), T, false, "", "", 0)
_PyVal(code::AbstractString, @nospecialize(T), static::Bool) =
    _PyVal(String(code), T, static, "", "", 0)
_PyVal(code::AbstractString, @nospecialize(T), static::Bool, extent::AbstractString) =
    _PyVal(String(code), T, static, String(extent), "", 0)
_PyVal(code::AbstractString, @nospecialize(T), static::Bool, extent::AbstractString,
       limit::AbstractString) = _PyVal(String(code), T, static, String(extent), String(limit), 0)
_jax_static(v::_PyVal) = _PyVal(v.code, v.T, true, v.extent, v.limit, v.padded, v.origin, v.field_extents)
_jax_ragged(@nospecialize(T)) = T isa Type && T <: AbstractArray && eltype(T) isa Type &&
    eltype(T) <: AbstractArray
function _jax_extent(args)
    for a in args
        isempty(a.extent) || return a.extent
    end
    ""
end
# A padded vector's valid-entry mask, over the leading axis of its stored
# array: a vector of tuples is held as a Python tuple of columns, whose first
# column carries that axis (`_jax_mapped_array`).
_jax_mask(v::_PyVal) = "(jnp.arange(" * _py_atom(_jax_mapped_array(v)) * ".shape[0]) < " *
                       v.extent * ")"
# A vector or an adjoint (transposed) vector: stored as one 1-D array.
_jax_vector_like(@nospecialize(T)) = T isa Type && (T <: AbstractVector ||
    T <: Union{LinearAlgebra.Adjoint{<:Any,<:AbstractVector},
               LinearAlgebra.Transpose{<:Any,<:AbstractVector}})
# The Julia axes a padded value's `extent` pads, as a bitmask: those recorded
# in `padded`, else a vector's one axis and an adjoint vector's second. Zero
# for a value that is not padded, a ragged value (its lengths pad each row),
# or an array whose padded axes no translation recorded.
function _jax_padded_axes(v::_PyVal)
    isempty(v.extent) && return 0
    v.padded == 0 || _jax_tuple_valued(v.T) || return v.padded
    T = v.T
    T isa Type && !_jax_ragged(T) || return 0
    T <: AbstractVector && return 1
    _jax_vector_like(T) && return 2
    0
end
# The padded axes of an elementwise combination of `args` (a broadcast aligns
# their leading axes, as `+` of equal shapes does): the union of theirs when
# one extent pads them all; zero (not recorded) when operands are padded by
# different extents or an operand's padded axes are not known.
function _jax_padded_union(args)
    axes = 0
    extent = ""
    for a in args
        isempty(a.extent) && continue
        m = _jax_padded_axes(a)
        m == 0 && return 0
        isempty(extent) || extent == a.extent || return 0
        extent = a.extent
        axes |= m
    end
    axes
end
# Operands padded along the same axes by different extents: two per-row
# arguments broadcast together (`(x .- m) ./ s` with `x` and `s` ragged rows
# of one cell), each with its own lengths. Julia's broadcast gives each such
# axis one length, the operands' length other than one (one when all are
# one). The operands are returned with that length as their common extent,
# `_rk_broadcast_length(...)` of their extents (sorted, so every combination
# of the same rows reads alike and combines again by string equality), an
# expression of their extents, valid wherever they are. Lengths Julia
# refuses to broadcast (two lengths other than one differ) are not checked:
# a check is a host callback in every evaluation (several times the cost of
# a small model's gradient), and a padded row broadcast against an unpadded
# array is not checked either. Operands padding different axes by different
# extents (`x .* s'`) would need an extent per axis: they are returned as
# they are, so a broadcast of them records no padded axes and a reduction of
# it raises.
const _JAX_BROADCAST_LENGTH = """
def _rk_broadcast_length(*lengths):
    # Julia's broadcast length along an axis its operands pad: the length
    # other than one, or one when every operand has length one.
    n = lengths[0]
    for m in lengths[1:]:
        n = jnp.where(n == 1, m, n)
    return n
"""
function _jax_common_extent(ctx, args::Vector{_PyVal})
    ctx.host === nothing || return args
    padded = [a for a in args if !isempty(a.extent)]
    length(unique(a.extent for a in padded)) > 1 || return args
    masks = unique(_jax_padded_axes(a) for a in padded)
    length(masks) == 1 && only(masks) != 0 || return args
    parts = String[]
    for a in padded
        append!(parts, get(ctx.mod.helper_names, (:broadcast_length, a.extent), [a.extent]))
    end
    sort!(unique!(parts))
    name = _jax_helper!(ctx, "_rk_broadcast_length", _JAX_BROADCAST_LENGTH)
    extent = name * "(" * join(parts, ", ") * ")"
    ctx.mod.helper_names[(:broadcast_length, extent)] = parts
    [isempty(a.extent) ? a : _PyVal(a.code, a.T, a.static, extent, "", a.padded, a.origin, a.field_extents)
     for a in args]
end
# The padded axes an elementwise function of `x` keeps: `x`'s, unless `x` is
# an adjoint vector, whose result (typed a matrix) is stored with one axis.
_jax_adjoint_vector(@nospecialize(T)) = _jax_vector_like(T) && !(T <: AbstractVector)
_jax_padded_like(x::_PyVal) = _jax_adjoint_vector(x.T) ? 0 : _jax_padded_axes(x)
# The padded axes of a broadcast of rank `R` over `args`, when its stored
# array has all `R` axes: an operand other than an adjoint vector gives it
# them (an adjoint vector alone keeps its one stored axis).
_jax_broadcast_padded(args, R::Int) =
    R >= 2 && any(a -> _jax_rank(a.T) > 0 && !_jax_adjoint_vector(a.T), args) ?
    _jax_padded_union(args) : 0
# An elementwise result over arrays of one shape (`A + B`, `-A`, `c * A`)
# keeps their padded axes.
function _jax_elementwise_result(f, args::Vector{_PyVal}, code::AbstractString)
    v = _jax_result(f, args, code)
    _jax_is_array(v.T) && !_jax_vector_like(v.T) || return v
    _PyVal(v.code, v.T, v.static, v.extent, v.limit, _jax_padded_union(args))
end
# The valid-entry mask of a padded array over its padded axes `axes` (a
# subset of `_jax_padded_axes(x)`), shaped to broadcast against the stored
# array; `nothing` when its padded axes are not known. `x` is read once per
# axis, so a caller binds it to a name first.
function _jax_padded_mask(x::_PyVal, axes::Int = _jax_padded_axes(x))
    axes == 0 && return nothing
    _jax_vector_like(x.T) && return _jax_mask(x)
    R = _jax_rank(x.T)
    parts = String[]
    for k in 1:R
        axes & (1 << (k - 1)) == 0 && continue
        push!(parts, "(jnp.arange(" * _py_atom(x.code) * ".shape[" * string(k - 1) * "]) < " *
                     x.extent * ")[" * join((j == k ? ":" : "None" for j in 1:R), ", ") * "]")
    end
    join(parts, " & ")
end
# The code of `v` as a reduction reads it: a padded array's padding replaced
# by the reduction's identity `neutral`. An array whose padded axes are not
# known is read whole, as before they were recorded; `_jax_padded_reduced`
# refuses it instead.
function _jax_masked(v::_PyVal, neutral::AbstractString)
    isempty(v.extent) && return v.code
    mask = _jax_vector_like(v.T) ? _jax_mask(v) : _jax_padded_mask(v)
    mask === nothing && return v.code
    "jnp.where(" * mask * ", " * v.code * ", " * neutral * ")"
end
# `v` (an array) as a whole-array reduction reads it, padding replaced by
# `neutral`; a padded array whose padded axes are not known raises, since
# reading its padding as entries gives a wrong value.
function _jax_padded_reduced(ctx, f, v::_PyVal, neutral::AbstractString)
    isempty(v.extent) && return v.code
    (_jax_vector_like(v.T) || !_jax_is_array(v.T)) &&
        return "jnp.where(" * _jax_mask(v) * ", " * v.code * ", " * neutral * ")"
    _jax_padded_axes(v) == 0 && _jax_error("`", f, "` of a padded `", v.T,
        "` whose padded axes are not known", _jax_where(ctx))
    x = _jax_bind!(ctx, :padded, v)
    "jnp.where(" * _jax_padded_mask(x) * ", " * x.code * ", " * neutral * ")"
end

# The Python keywords and module names an emitted identifier must not take.
const _JAX_RESERVED = Set{String}(split("""
    False None True and as assert async await break class continue def del elif
    else except finally for from global if import in is lambda nonlocal not or
    pass raise return try while with yield jax jnp jsp lax np math tuple list dict len
    range sum min max abs str int float bool print type object id iter next map zip
    all any round pow filter sorted set""" ))

# One Python function body under construction. `links` maps an input of a
# nested plan (a branch arm's, a plate cell's unmapped argument) to the
# `(plan, value)` of the enclosing plan it is the same value as, so the
# stable rewrites find that value's definition (`_jax_stable_definition`).
mutable struct _JaxScope
    lines::Vector{String}
    indent::Int
    used::Set{String}
    values::Dict{ValueId,_PyVal}
    parent::Union{Nothing,_JaxScope}
    links::Dict{ValueId,Any}
    hints::Dict{ValueId,Symbol}
    named_hints::Dict{Symbol,Symbol}
    flat_positions::Dict{String,String}
end
_JaxScope(parent::Union{Nothing,_JaxScope} = nothing) =
    _JaxScope(String[], 1, parent === nothing ? Set{String}() : copy(parent.used),
              Dict{ValueId,_PyVal}(), parent, Dict{ValueId,Any}(), Dict{ValueId,Symbol}(),
              parent === nothing ? Dict{Symbol,Symbol}() : parent.named_hints,
              parent === nothing ? Dict{String,String}() : copy(parent.flat_positions))

# An authored lexical region of an otherwise transparent graph. The authoring
# frontend supplies its explicit boundary and names; RK owns the graph cut and
# the executable Python callable. This metadata does not change native plans.
struct _JaxSourceRegion
    name::Symbol
    inputs::Vector{Symbol}
    outputs::Vector{Symbol}
    names::Dict{Symbol,Symbol}
    doc::String
end

# The whole module: its helper definitions and module-level constants.
mutable struct _JaxModule
    helpers::Vector{String}
    helper_names::Dict{Any,Any}
    used::Set{String}
    counter::Int
    # Why `prepare_data` leaves a data-only recipe (by id in the program's
    # graph, `host_graph`) to the functions: its host translation's error.
    host_reasons::Dict{Int,JAXEmissionError}
    host_graph::Any
    # The names `jax_source`'s `definitions` give a value of: a data-only
    # recipe of one of them that neither side translates takes its value from
    # the definition instead (`_JaxPromote`).
    host_definitions::Set{Symbol}
    # The module those definitions' globals resolve in (`jax_source`'s
    # `definitions_module`).
    definitions_module::Module
    # Why `prepare_data` leaves a plate cell's data-only recipe to the
    # functions, by (plate recipe id, cell recipe id) (`_jax_cell_split`,
    # jax_host.jl).
    cell_reasons::Dict{Tuple{Int,Int},JAXEmissionError}
    cell_data::Set{Tuple{Int,Int}}
    # The recipes being emitted, outermost first, each with its plan; and for
    # an emission error, that chain where it was raised (`_jax_plan_recipe!`).
    path::Vector{Tuple{Plan,Recipe}}
    failure_paths::IdDict{Any,Vector{Tuple{Plan,Recipe}}}
    regions::Vector{_JaxSourceRegion}
end
_JaxModule() = _JaxModule(String[], Dict{Any,Any}(), Set{String}(), 0,
                          Dict{Int,JAXEmissionError}(), nothing, Set{Symbol}(), Main,
                          Dict{Tuple{Int,Int},JAXEmissionError}(), Set{Tuple{Int,Int}}(),
                          Tuple{Plan,Recipe}[],
                          IdDict{Any,Vector{Tuple{Plan,Recipe}}}(), _JaxSourceRegion[])

# Translation state: the module, the current function scope, the current
# recipe (for messages) and the module that resolves its source's globals.
# `host` is `nothing` for JAX code and the host state (`jax_host.jl`) for the
# NumPy code of `prepare_data`.
mutable struct _JaxContext
    mod::_JaxModule
    scope::_JaxScope
    recipe::Any
    source_module::Module
    locals::Dict{Symbol,_PyVal}
    captures::Any
    traced_branches::Int
    host::Any
end
_JaxContext(mod, scope, recipe, source_module, locals, captures) =
    _JaxContext(mod, scope, recipe, source_module, locals, captures, 0, nothing)

# A value known while the code runs its Python control flow: a static value
# under JAX, any value on the host.
_jax_known(ctx::_JaxContext, v::_PyVal) = ctx.host !== nothing || v.static

_jax_where(ctx::_JaxContext) = ctx.recipe isa Recipe ?
    " in recipe `" * _recipe_line(ctx.recipe) * "`" : ""

_jax_emit!(ctx::_JaxContext, line::AbstractString) =
    push!(ctx.scope.lines, "    "^ctx.scope.indent * line)

# A fresh Python identifier derived from `hint`.
# A generated name without the counters that differ between builds of one
# program: a gensym's `#<n>` and a value id's long numeric suffix
# (`scale_binding_641723669287000073`). Names then follow the program's
# structure alone, so one declaration emits one module text, and the
# numbering of repeated names (`cell_2`) follows the emission order.
_jax_stable_name(hint) =
    replace(replace(String(Symbol(hint)), r"#\d+" => ""), r"_\d{6,}(?=_|$)" => "")

# A generated module-level helper's name: an identifier with one leading
# underscore, so the module's public names are its documented functions
# alone. It drops an `rk_` prefix (`_standardized`): the fixed helpers own the
# `_rk_` names (`_rk_isequal` and its host twin `_rk_isequal_np`), which they
# take without consulting `used`, so a generated `_rk_isequal_np` shadowed
# one and recursed.
function _jax_private_identifier(used::Set{String}, hint)
    base = replace(_jax_identifier(Set{String}(), hint), r"^rk_" => "")
    candidate = "_" * base
    n = 1
    while candidate in used
        n += 1
        candidate = string("_", base, "_", n)
    end
    push!(used, candidate)
    candidate
end

function _jax_identifier(used::Set{String}, hint)
    text = replace(_jax_stable_name(hint), r"[^A-Za-z0-9_]" => "_")
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

# An input's Python name: its own name when that is a free Python identifier
# (so `prepare_data(**data)` takes the data's own keys), else a fresh one.
function _jax_param_name(used::Set{String}, name; stable::Bool = true)
    text = stable ? _jax_stable_name(name) : String(Symbol(name))
    if occursin(r"^[A-Za-z_][A-Za-z0-9_]*$", text) && !(text in _JAX_RESERVED) && !(text in used)
        push!(used, text)
        return text
    end
    _jax_identifier(used, name)
end

# A data value's key at the module's data interface: `prepare_data`'s keyword
# and the dict it returns, `**prepared`, and the metadata (`DATA`, `INPUTS`,
# `PREPARED`, `UNTRANSLATED`). It is the value's own name when that is an
# identifier, also one Python reserves (`class`, `lambda`, `id`): only the
# Python local that holds it is respelled (`_jax_param_name`), so a caller
# binds the data by the names the model reads (snag jax-source-renam-c36033de).
# A name that is no identifier (a submodel's `r.g`) takes a fresh one (`r_g`).
function _jax_key_name(used::Set{String}, name; stable::Bool = true)
    text = stable ? _jax_stable_name(name) : String(Symbol(name))
    if occursin(r"^[A-Za-z_][A-Za-z0-9_]*$", text) && !(text in used)
        push!(used, text)
        return text
    end
    _jax_identifier(used, name)
end

# A helper may be emitted after locals in its caller. Reserve against both
# sets so a local result cannot shadow a later callable with the same name.
function _jax_named_function(ctx::_JaxContext, hint; private::Bool = false)
    used = union(ctx.scope.used, ctx.mod.used)
    name = private ? _jax_private_identifier(used, hint) : _jax_identifier(used, hint)
    push!(ctx.mod.used, name)
    name
end

function _jax_body_hint(ctx, recipe, suffix)
    value = first(recipe.outputs)
    name = get(ctx.scope.hints, value.id, value.name)
    Symbol(get(ctx.scope.named_hints, name, name), :_, suffix)
end

function _jax_fresh(ctx::_JaxContext, hint)
    hint = get(ctx.scope.named_hints, Symbol(hint), hint)
    name = _jax_identifier(union(ctx.scope.used, ctx.mod.used), hint)
    push!(ctx.scope.used, name)
    name
end

# Bind `value` to a Python local holding `v`; a plain name is reused, and a
# type or callable value (`Val{2}` of a lowered literal power) exists only
# while translating.
function _jax_bind!(ctx::_JaxContext, hint, v::_PyVal)
    (_jax_translation_only(v) || occursin(r"^[A-Za-z_][A-Za-z0-9_]*$", v.code)) && return v
    name = _jax_fresh(ctx, hint)
    _jax_emit!(ctx, name * " = " * v.code)
    _PyVal(name, v.T, v.static, v.extent, v.limit, v.padded, v.origin, v.field_extents)
end

# --- types and ranks ----------------------------------------------------------------

_jax_rank(@nospecialize(T)) = T isa Type && T <: AbstractArray ? ndims(T) : 0
_jax_is_array(@nospecialize(T)) = T isa Type && T <: AbstractArray
# A logical index: an array of `Bool`s (`BitVector`, `Vector{Bool}`).
_jax_logical(@nospecialize(T)) = _jax_is_array(T) && eltype(T) === Bool
_jax_is_scalar(@nospecialize(T)) = T isa Type && T <: Number
_jax_eltype(@nospecialize(T)) = T isa Type && T <: AbstractArray ? eltype(T) : T

function _jax_return_type(f, argtypes)
    T = try
        Core.Compiler.return_type(Tuple{_jax_callable_type(f), argtypes...})
    catch
        Any
    end
    T
end

# --- Python operator precedence ---------------------------------------------------
#
# Infix expressions are printed with only the parentheses Python's grammar
# needs. An operand's precedence is that of its loosest operator outside
# brackets; an atom (a name, a literal, a call, a subscript) binds tightest.

const _PY_PRECEDENCE = Dict{String,Int}(
    "if" => 1, "or" => 2, "and" => 3, "not" => 4,
    "<" => 5, ">" => 5, "<=" => 5, ">=" => 5, "==" => 5, "!=" => 5, "is" => 5,
    "|" => 6, "^" => 7, "&" => 8, "+" => 10, "-" => 10,
    "*" => 11, "/" => 11, "@" => 11, "//" => 11, "%" => 11,
    "unary" => 12, "**" => 13)
const _PY_ATOM = 100

function _py_precedence(code::AbstractString)
    lowest = _PY_ATOM
    depth = 0
    quote_char = nothing
    previous = :start   # what precedes position i: :start, :operand or :operator
    chars = collect(code)
    i = 1
    while i <= length(chars)
        c = chars[i]
        if quote_char !== nothing
            c == '\\' && (i += 2; continue)
            c == quote_char && (quote_char = nothing; previous = :operand)
            i += 1; continue
        end
        if c == '"' || c == '\''
            quote_char = c; i += 1; continue
        end
        if c in ('(', '[', '{')
            depth += 1; i += 1; continue
        elseif c in (')', ']', '}')
            depth -= 1; previous = :operand; i += 1; continue
        end
        if depth == 0
            two = i < length(chars) ? string(c, chars[i + 1]) : ""
            if two in ("**", "<=", ">=", "==", "!=", "//")
                lowest = min(lowest, _PY_PRECEDENCE[two]); previous = :operator; i += 2; continue
            elseif c == '-' || c == '+'
                lowest = min(lowest, previous === :operand ? 10 : 12)
                previous = :operator; i += 1; continue
            elseif c in ('*', '/', '@', '%', '<', '>', '|', '^', '&')
                lowest = min(lowest, _PY_PRECEDENCE[string(c)]); previous = :operator; i += 1; continue
            elseif isletter(c) || c == '_'
                j = i
                while j <= length(chars) && (isletter(chars[j]) || isdigit(chars[j]) || chars[j] == '_')
                    j += 1
                end
                word = String(chars[i:j-1])
                if word in ("if", "else", "or", "and", "not", "is", "in", "lambda")
                    lowest = min(lowest, get(_PY_PRECEDENCE, word, 1)); previous = :operator
                else
                    previous = :operand
                end
                i = j; continue
            end
        end
        isspace(c) || (previous = :operand)
        i += 1
    end
    lowest
end

# `a op b` (or `op a` for unary minus), parenthesizing operands as needed.
function _py_infix(op::AbstractString, operands::AbstractVector{<:AbstractString})
    p = _PY_PRECEDENCE[op]
    if op == "**"
        length(operands) == 2 || error("** takes two operands")
        left = _py_precedence(operands[1]) <= p ? "(" * operands[1] * ")" : operands[1]
        right = _py_precedence(operands[2]) < p ? "(" * operands[2] * ")" : operands[2]
        return left * " ** " * right
    end
    parts = map(enumerate(operands)) do (k, x)
        q = _py_precedence(x)
        # Python's binary operators associate left, so a later operand of the
        # same level keeps its parentheses: `*`, `/` and `@` share one level
        # (`a * (B @ w)` is not `(a * B) @ w`), and floating-point `+` and `*`
        # are not associative either. Comparisons do not chain as Julia's
        # binary calls mean them.
        needs = q < p || (k > 1 && q == p) || (p == 5 && q == 5)
        needs ? "(" * x * ")" : x
    end
    join(parts, " " * op * " ")
end
# An operand of a postfix (`.shape`, `[i]`), parenthesized unless an atom.
_py_atom(x::AbstractString) = _py_precedence(x) < _PY_ATOM ? "(" * x * ")" : x
_py_negate(x::AbstractString) = (_py_precedence(x) < _PY_PRECEDENCE["unary"] ? "-(" * x * ")" : "-" * x)

# --- literals -----------------------------------------------------------------------

_jax_literal(x::Bool) = _PyVal(x ? "True" : "False", Bool, true)
_jax_literal(x::Integer) = _PyVal(string(x), typeof(x), true)
function _jax_literal(x::AbstractFloat)
    isnan(x) && return _PyVal("jnp.nan", typeof(x), true)
    isinf(x) && return _PyVal(x > 0 ? "jnp.inf" : "-jnp.inf", typeof(x), true)
    _PyVal(repr(Float64(x)), typeof(x), true)
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
    dtype = _jax_dtype(T)
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
function _jax_literal(x::Tuple)
    parts = [_jax_literal(v) for v in x]
    _PyVal("(" * join((p.code for p in parts), ", ") * (length(x) == 1 ? ",)" : ")"), typeof(x),
           all(p -> p.static, parts))
end
_jax_literal(x::NamedTuple) = (t = _jax_literal(Tuple(x)); _PyVal(t.code, typeof(x), t.static))
_jax_literal(x) = _jax_error("no JAX literal for a value of type `", typeof(x), "`")

# A constant the source reads (a captured array, a module constant): small
# values inline, arrays become module-level constants.
function _jax_constant(ctx::_JaxContext, hint, x)
    x isa AbstractArray || return _jax_literal(x)
    host = ctx.host !== nothing
    name = get(ctx.mod.helper_names, (:constant, objectid(x), host), nothing)
    if name === nothing
        name = _jax_identifier(ctx.mod.used, uppercase(string(hint)) * (host ? "_NP" : ""))
        line = name * " = " * _jax_literal(x).code
        push!(ctx.mod.helpers, host ? _host_text(line) : line)
        ctx.mod.helper_names[(:constant, objectid(x), host)] = name
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
    # A type computed by a call (`Dict{eltype(v),Int}`).
    a isa Expr && a.head === :call && return _jax_callable(_jax_expr(ctx, a))
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
    haskey(ctx.locals, s) && return ctx.host === nothing ? ctx.locals[s] :
        _host_read(ctx, ctx.locals[s])
    s === :nothing && return _PyVal("None", Nothing, true)
    s === :Inf && return _jax_literal(Inf)
    s === :NaN && return _jax_literal(NaN)
    _jax_value(ctx, _jax_resolve(ctx, s), s)
end

# A Julia value met in source (a resolved global or capture).
_jax_value(ctx::_JaxContext, x::Union{Number,Nothing,AbstractString,Symbol}, hint) = _jax_literal(x)
_jax_value(ctx::_JaxContext, x::AbstractArray{<:Real}, hint) = _jax_constant(ctx, hint, x)
_jax_value(ctx::_JaxContext, x::Tuple, hint) = _jax_literal(x)
# A function passed as a value (`sum(abs2, z)`): its type carries it, and only
# a call that takes a function argument reads it.
_jax_value(ctx::_JaxContext, x::Function, hint) =
    Base.issingletontype(typeof(x)) ? _PyVal("<function " * string(hint) * ">", typeof(x)) :
    _jax_error("the closure `", hint, "` passed as a value has no JAX translation",
               _jax_where(ctx))
# `:` as a value (a `reshape` dimension, also inside a tuple) is Python's
# `None`; the calls that take it read its type.
_jax_value(ctx::_JaxContext, ::Colon, hint) = _PyVal("None", Colon, true)
# `:` as a value, also as lowered code passes it (a callable global).
_jax_is_colon(v::_PyVal) = v.T === Colon || v.T isa _JaxCallable{Colon}
# A type passed as a value (`ones(Float64, n)`, `Float64.(x)`): static, read
# by the calls that take a type argument.
_jax_value(ctx::_JaxContext, x::Type, hint) = _PyVal("<type " * string(x) * ">", Type{x}, true)
# A singleton (`RoundDown`, `Val(:logit)`) carries everything in its type.
_jax_value(ctx::_JaxContext, x, hint) = Base.issingletontype(typeof(x)) ?
    _PyVal("None", typeof(x), true) :
    _jax_error("the value `", hint, "` of type `", typeof(x), "` has no JAX translation",
               _jax_where(ctx))
_jax_is_singleton_value(v::_PyVal) = v.T isa Type && v.T !== Nothing && !(v.T <: Function) &&
    Base.issingletontype(v.T)
# The type a type value carries: `Type{T}` from source, or a callable type from
# lowered code and type-level calls (`Float64` read as a global, a `where T`
# parameter, `promote_type(...)`), as `clamp`'s `convert(T, x)` passes it.
_jax_type_of_value(v::_PyVal) = v.T isa _JaxCallable ? (v.T.f isa Type ? v.T.f : nothing) :
    v.T isa DataType && v.T <: Type && length(v.T.parameters) == 1 ?
        v.T.parameters[1] : nothing
_jax_dtype(@nospecialize(E)) = E <: Bool ? "jnp.bool_" : E <: Integer ? _jax_int_dtype() :
    E <: AbstractFloat ? "jnp." * _jax_dtype_name(E) : "jnp." * _jax_float_name()

# The module's float type, `jax_source`'s `float_type`: the dtype of every
# float value the program does not type `Float32` explicitly (Julia's default
# `Float64`, a sampler boundary's values, data). It is set for the task that
# emits one module, so modules emitted concurrently do not share it.
const _JAX_FLOAT_TYPE_KEY = :reactivekernels_jax_float_type
_jax_float_type() = get(task_local_storage(), _JAX_FLOAT_TYPE_KEY, Float64)::DataType
_jax_float_name() = _jax_float_type() === Float32 ? "float32" : "float64"
# A float32 module does not turn on JAX's 64-bit mode (`_jax_needs_x64`), so
# its integers are JAX's default integer, Python's `int` as a dtype: int32,
# or int64 where another module turned 64-bit mode on, as `jnp.arange` and
# the integer data are there.
_jax_int_dtype() = _jax_float_type() === Float32 ? "int" : "jnp.int64"
function _jax_with_float_type(f, float_type)
    float_type in (Float64, Float32) || throw(ArgumentError(
        "jax_source: `float_type` is `Float64` or `Float32`; got `$float_type`"))
    task_local_storage(f, _JAX_FLOAT_TYPE_KEY, float_type)
end

_jax_expr(ctx::_JaxContext, ::Val{H}, ex) where {H} =
    _jax_error("the construct `", H, "` (`", ex, "`) has no JAX translation yet",
               _jax_where(ctx))

_jax_expr(ctx::_JaxContext, ::Val{:block}, ex) = _jax_statements(ctx, ex.args)
# A subtype test of types known while emitting (`eltype(x) <: Number`).
_jax_expr(ctx::_JaxContext, ::Val{:<:}, ex) =
    _jax_literal(_jax_type_operand(ctx, ex.args[1]) <: _jax_type_operand(ctx, ex.args[2]))
function _jax_type_operand(ctx::_JaxContext, ex)
    v = _jax_expr(ctx, ex)
    v.T isa _JaxCallable && v.T.f isa Type && return v.T.f
    T = _jax_type_of_value(v)
    T isa Type && return T
    v.T isa Type && v.T <: Type && return v.T.parameters[1]
    _jax_error("the subtype test of `", ex, "`, which is not a type", _jax_where(ctx))
end
_jax_expr(ctx::_JaxContext, ::Val{:(::)}, ex) = _jax_expr(ctx, ex.args[1])
_jax_expr(ctx::_JaxContext, ::Val{:return}, ex) = ctx.host === nothing ?
    _jax_expr(ctx, ex.args[1]) : _host_return(ctx, ex)
_jax_expr(ctx::_JaxContext, ::Val{:break}, ex) = ctx.host === nothing ?
    _jax_error("`break` has no JAX translation", _jax_where(ctx)) : _host_jump(ctx, "break")
_jax_expr(ctx::_JaxContext, ::Val{:continue}, ex) = ctx.host === nothing ?
    _jax_error("`continue` has no JAX translation", _jax_where(ctx)) : _host_jump(ctx, "continue")
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
# A statement sequence; its value is its last statement's. A statement whose
# value is discarded (one not last, or any in a loop body) is an effect.
function _jax_statements(ctx::_JaxContext, statements; value::Bool = true)
    result = _PyVal("None", Nothing)
    last = findlast(s -> !(s isa LineNumberNode), statements)
    for (k, statement) in enumerate(statements)
        statement isa LineNumberNode && continue
        result = value && k == last ? _jax_statement(ctx, statement) : _jax_effect(ctx, statement)
    end
    result
end

# A statement translated for its effect only: an `if` there carries just the
# locals its arms assign, so arms whose values differ in type still combine.
function _jax_effect(ctx::_JaxContext, ex)
    ex isa Expr && ex.head === :block && return _jax_statements(ctx, ex.args; value = false)
    ex isa Expr && ex.head in (:if, :elseif) && return _jax_if(ctx, ex, false)
    _jax_statement(ctx, ex)
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
    ctx.host === nothing || lhs isa Expr && lhs.head === :tuple ||
        return _host_assign!(ctx, lhs, v)
    if lhs isa Symbol
        # A type or callable value exists only while translating.
        _jax_translation_only(v) && (ctx.locals[lhs] = v; return)
        name = _jax_fresh(ctx, lhs)
        _jax_emit!(ctx, name * " = " * v.code)
        ctx.locals[lhs] = _PyVal(name, v.T, v.static, v.extent, v.limit, v.padded, v.origin, v.field_extents)
        return
    end
    if lhs isa Expr && lhs.head === :ref && lhs.args[1] isa Symbol && haskey(ctx.locals, lhs.args[1])
        # `A[i] = v`: a functional update rebinding `A`.
        target = ctx.locals[lhs.args[1]]
        indexed = _jax_index(ctx, target, lhs.args[2:end]; target = true)
        position = indexed.code[ncodeunits(target.code)+1:end]
        name = _jax_fresh(ctx, lhs.args[1])
        # In a loop's steps past its bound the entry keeps its value.
        mask = _jax_write_mask(ctx, lhs.args[1], target, lhs.args[2:end])
        value = mask === nothing ? v.code :
            "jnp.where(" * mask * ", " * v.code * ", " * _py_atom(target.code) * position * ")"
        _jax_emit!(ctx, name * " = " * _py_atom(target.code) * ".at" * position * ".set(" * value * ")")
        ctx.locals[lhs.args[1]] = _PyVal(name, target.T, false, target.extent, target.limit,
                                         target.padded)
        return
    end
    if lhs isa Expr && lhs.head === :tuple
        bound = _jax_bind!(ctx, :tuple, v)
        for (i, target) in enumerate(lhs.args)
            T = v.T isa Type && v.T <: Tuple && i <= fieldcount(v.T) ? fieldtype(v.T, i) : Any
            _jax_destructure!(ctx, target, _PyVal(_py_atom(bound.code) * "[" * string(i - 1) * "]", T,
                                                  bound.static))
        end
        return
    end
    _jax_error("the assignment target `", lhs, "` has no JAX translation", _jax_where(ctx))
end

function _jax_expr(ctx::_JaxContext, ::Val{:let}, ex)
    ctx.host === nothing || return _host_let(ctx, ex)
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
    values = _PyVal[_jax_expr(ctx, a) for a in ex.args]
    _jax_call_positional(ctx, tuple, values)
end
# The extents of a tuple's padded elements, as a Python tuple (`None` for an
# element that is not padded), or "" when none is; and which elements are
# padded, as `padded` records them for a tuple.
_jax_tuple_extent(values) = all(v -> isempty(v.extent), values) ? "" :
    "(" * join((isempty(v.extent) ? "None" : v.extent for v in values), ", ") *
    (length(values) == 1 ? ",)" : ")")
_jax_tuple_fields(values) =
    foldl((m, (k, v)) -> isempty(v.extent) ? m : m | 1 << (k - 1), enumerate(values); init = 0)
_jax_tuple_valued(@nospecialize(T)) = T isa Type && (T <: Tuple ||
    T <: AbstractArray && eltype(T) isa Type && eltype(T) <: Tuple)
# The fields of a tuple (or of a vector of tuples' entries) that have a
# length in its extent, as a bitmask; zero when none was recorded, in which
# case every field with an array type is taken to have one.
_jax_padded_fields(v::_PyVal) = isempty(v.extent) || !_jax_tuple_valued(v.T) ? 0 : v.padded
# The extent of a tuple's field `k` of type `T`: its entry of the tuple's
# extent when it has a length, else none ("").
function _jax_field_extent(x::_PyVal, k::Int, @nospecialize(T))
    (isempty(x.extent) || !_jax_is_array(T)) && return ""
    isempty(x.field_extents) || return x.field_extents[k]
    fields = _jax_padded_fields(x)
    fields == 0 || fields & (1 << (k - 1)) != 0 || return ""
    _py_atom(x.extent) * "[" * string(k - 1) * "]"
end
_jax_tuple_code(values) = "(" * join((v.code for v in values), ", ") *
                          (length(values) == 1 ? ",)" : ")")

_jax_expr(ctx::_JaxContext, ::Val{:vect}, ex) =
    _jax_function_call(ctx, Base.vect, _PyVal[_jax_expr(ctx, a) for a in ex.args])
function _jax_function_call(ctx::_JaxContext, ::typeof(Base.vect), values::Vector{_PyVal})
    T = isempty(values) ? Float64 : promote_type((_jax_eltype(v.T) for v in values)...)
    isempty(values) && ctx.host !== nothing && return _host_list(ctx, Vector{T})
    isempty(values) && return _PyVal("jnp.zeros((0,), dtype=" * _jax_dtype(T) * ")", Vector{T})
    _PyVal("jnp.stack([" * join((v.code for v in values), ", ") * "])", Vector{T})
end

# `x[i...]`, or a typed array literal `T[a, b]`.
function _jax_expr(ctx::_JaxContext, ::Val{:ref}, ex)
    head = ex.args[1]
    if head isa Symbol && !_jax_has_local(ctx, head) && isdefined(ctx.source_module, head) &&
       getglobal(ctx.source_module, head) isa Type
        T = getglobal(ctx.source_module, head)
        values = [_jax_expr(ctx, a) for a in ex.args[2:end]]
        dtype = _jax_dtype(T)
        isempty(values) && ctx.host !== nothing && return _host_list(ctx, Vector{T})
        isempty(values) && return _PyVal("jnp.zeros((0,), dtype=" * dtype * ")", Vector{T})
        return _PyVal("jnp.asarray([" * join((v.code for v in values), ", ") *
                      "], dtype=" * dtype * ")", Vector{T})
    end
    x = ctx.host === nothing ? _jax_expr(ctx, head) : _host_raw(ctx, head)
    if x.T isa _JaxCallable && x.T.f isa Type
        # A typed array literal whose element type is computed (`eltype(c)[]`).
        T = x.T.f
        values = [_jax_expr(ctx, a) for a in ex.args[2:end]]
        isempty(values) && ctx.host !== nothing && return _host_list(ctx, Vector{T})
        isempty(values) && return _PyVal("jnp.zeros((0,), dtype=" * _jax_dtype(T) * ")", Vector{T})
        return _PyVal("jnp.asarray([" * join((v.code for v in values), ", ") * "], dtype=" *
                      _jax_dtype(T) * ")", Vector{T})
    end
    _jax_index(ctx, x, ex.args[2:end])
end

# Indices that Julia requires to be 1 (past an array's rank, or of a
# number): a literal 1 needs nothing, any other index a check that it is 1
# (while tracing when it is known then, else when the program runs).
function _jax_unit_indices!(ctx::_JaxContext, indices, what::AbstractString)
    for index in indices
        v = _jax_expr(ctx, index)
        v.T isa Type && v.T <: Integer ||
            _jax_error("indexing ", what, " by a `", v.T, "`", _jax_where(ctx))
        v.code == "1" && continue
        _jax_call(ctx, _runtime_check,
                  [_PyVal("(" * v.code * " == 1)", Bool, _jax_known(ctx, v)),
                   _PyVal(repr("BoundsError: an index of " * what * " must be 1"), BoundsError)])
    end
end

# Index `x` with Julia indices `indices` (source expressions). An assignment
# `target` reads the position only, so its gather carries no extent.
function _jax_index(ctx::_JaxContext, x::_PyVal, indices; target::Bool = false)
    if !target && x.origin isa _JaxHistoryLayout
        read = _jax_history_read!(ctx, x, indices)
        read === nothing || return read
    end
    if ctx.host !== nothing
        v = _host_index(ctx, x, indices)
        v === nothing || return v
    end
    if x.T isa Type && x.T <: Tuple
        length(indices) == 1 && indices[1] isa Integer ||
            _jax_error("a tuple is indexed with a literal integer only", _jax_where(ctx))
        k = indices[1]
        T = fieldtype(x.T, k)
        return _PyVal(_py_atom(x.code) * "[" * string(k - 1) * "]", T, x.static,
                      _jax_field_extent(x, k, T))
    end
    # A number indexed by ones is itself (`s[1, 1, 1]`, as Julia's).
    if x.T isa Type && x.T <: Number
        _jax_unit_indices!(ctx, indices, "a number")
        return x
    end
    _jax_is_array(x.T) || _jax_error("indexing a value of type `", x.T, "`", _jax_where(ctx))
    # A vector of (named) numbers is the tuple of its columns (a scan's
    # stacked outputs): an element, or a slice, reads every column at it.
    if _jax_columns(x.T)
        length(indices) == 1 ||
            _jax_error("indexing a vector of tuples with several indices", _jax_where(ctx))
        E = eltype(x.T)
        parts = [_jax_index(ctx, _jax_column(x, k), indices) for k in 1:fieldcount(E)]
        # An element reads each field's value; a slice, each field's column.
        element = all(k -> parts[k].T == fieldtype(E, k), 1:fieldcount(E))
        number = findfirst(k -> fieldtype(E, k) <: Number, 1:fieldcount(E))
        return _PyVal(_jax_tuple_code(parts), element ? E : Vector{E}, false,
                      element || number === nothing ? "" : parts[number].extent)
    end
    N = ndims(x.T)
    # Linear indexing reads the array in Julia's column-major order. A `Bool`
    # mask of the array's shape (`A[A .> 0]`) selects in that order too.
    if length(indices) == 1 && N > 1 && !_jax_ragged(x.T) && isempty(x.extent)
        flat = _PyVal("jnp.ravel(" * x.code * ", order=\"F\")", Vector{eltype(x.T)}, x.static)
        _jax_written_slice(indices[1]) && return _jax_index(ctx, flat, indices; target)
        i = _jax_bound_expr(ctx, flat, 1, indices[1])
        if _jax_logical(i.T) && ndims(i.T) > 1
            ndims(i.T) == N && isempty(i.extent) ||
                _jax_error("a rank-", ndims(i.T), " `Bool` mask of a rank-", N, " array",
                           _jax_where(ctx))
            _jax_call(ctx, _runtime_check,
                      [_PyVal("(jnp.shape(" * x.code * ") == jnp.shape(" * i.code * "))", Bool, true),
                       _PyVal(repr("BoundsError: a `Bool` mask must have the shape of the array it indexes"),
                              BoundsError)])
            i = _PyVal("jnp.ravel(" * i.code * ", order=\"F\")", Vector{Bool}, i.static)
        end
        return _jax_index(ctx, flat, [i]; target)
    end
    # Julia accepts indices past the rank that are 1 (`v[i, 1, 1]`): one
    # known while tracing is checked then, and dropped.
    if length(indices) > N >= 1 && !_jax_ragged(x.T) && isempty(x.extent)
        _jax_unit_indices!(ctx, indices[N+1:end], "a rank-" * string(N) * " array past its rank")
        indices = indices[1:N]
    end
    length(indices) == N ||
        (length(indices) == 1 && N == 1) ||
        _jax_error("indexing a rank-", N, " array with ", length(indices),
                   " indices", _jax_where(ctx))
    parts = String[]
    values = Union{Nothing,_PyVal}[]
    kept = 0
    for (dim, index) in enumerate(indices)
        part, keeps, value = _jax_index_part(ctx, x, dim, index; target)
        push!(parts, part)
        push!(values, value)
        kept += keeps
    end
    E = eltype(x.T)
    T = kept == 0 ? E : Array{E,kept}
    if _jax_ragged(x.T)
        isempty(x.extent) && _jax_error("a ragged array without its lengths", _jax_where(ctx))
        # One group's row and length, or several groups' rows and lengths.
        return _PyVal(_py_atom(x.code) * "[" * join(parts, ", ") * "]", T, false,
                      x.extent * "[" * join(parts, ", ") * "]", "", kept == 0 ? 1 : 0,
                      kept == 0 && !target ? _jax_selected_origin(x, T, join(parts, ", ")) : nothing)
    end
    if !isempty(x.extent) && kept > 0
        # A gather from a padded group vector by an index vector: the indices
        # are valid entries, and the result is padded as the index is.
        gather = length(indices) == 1 ? values[1] : nothing
        gather !== nothing && _jax_is_array(gather.T) && eltype(gather.T) <: Integer ||
            _jax_error("slicing a padded group vector has no JAX translation", _jax_where(ctx))
        return _PyVal(_py_atom(x.code) * "[" * join(parts, ", ") * "]", T, false, gather.extent)
    end
    v = _PyVal(_py_atom(x.code) * "[" * join(parts, ", ") * "]", T, false,
               target ? "" : _jax_gather_extent(ctx, values, kept))
    # An entry of a prepared integer array keeps where it comes from, so a
    # loop over a range it ends has a static number of steps.
    kept == 0 && !target ? _jax_with_origin(v, _jax_selected_origin(x, T, join(parts, ", "))) : v
end
# The extent of a gather: the lengths of its index vector, when that is padded
# (a ragged row, or a range to a data length), since the gathered entries are
# padded as the index is and reductions of them mask the rest. A padded index
# beside another kept dimension would pad an inner axis, which nothing masks.
function _jax_gather_extent(ctx::_JaxContext, values, kept::Int)
    padded = [v for v in values if v !== nothing && _jax_is_array(v.T) && !isempty(v.extent)]
    isempty(padded) && return ""
    kept == 1 || _jax_error("a gather by a padded (ragged) index vector or a `Bool` mask ",
                            "beside another kept dimension has no JAX translation",
                            _jax_where(ctx))
    only(padded).extent
end

# A literal colon or a written range `a:b`, which `_jax_index_part` slices
# from the source expression (never a value it evaluates first).
_jax_written_slice(index) = index === :(:) || index == :(Colon()) ||
    index isa Expr && index.head === :call && index.args[1] === :(:) && length(index.args) == 3

# One index position: the Python index text, whether it keeps a dimension,
# and the index's value (`nothing` for a literal colon or a written range).
function _jax_index_part(ctx::_JaxContext, x::_PyVal, dim::Int, index; target::Bool = false)
    index === :(:) && return (":", 1, nothing)
    index == :(Colon()) && return (":", 1, nothing)
    if index isa Expr && index.head === :call && index.args[1] === :(:) &&
       length(index.args) == 3
        lo = _jax_bound_expr(ctx, x, dim, index.args[2])
        hi = _jax_bound_expr(ctx, x, dim, index.args[3])
        # Ends that are data with a static number of elements (a stop with a
        # static limit, or ends `_jax_trip_bound` bounds): a gather by the
        # range's elements, padded (`_jax_call_positional(::Colon)`).
        if !(_jax_known(ctx, lo) && _jax_known(ctx, hi)) &&
           (_jax_known(ctx, lo) && !isempty(hi.limit) || _jax_trip_bound(ctx, lo, hi) !== nothing)
            i = _jax_call_positional(ctx, Colon(), [lo, hi])
            return (_py_infix("-", [i.code, "1"]), 1, i)
        end
        return (_jax_minus_one(lo) * ":" * hi.code, 1, nothing)
    end
    i = _jax_bound_expr(ctx, x, dim, index)
    i.T === Colon && return (":", 1, i)
    # `Bool <: Integer`, but a `Bool` index is no position: an array selects
    # with a mask of them, and refuses a single one, as Julia does.
    i.T === Bool && _jax_bool_index_error(ctx, index)
    _jax_logical(i.T) && return _jax_mask_part(ctx, x, i; target)
    _jax_is_scalar(i.T) && i.T <: Integer && return (_jax_minus_one(i), 0, i)
    if _jax_is_array(i.T) && eltype(i.T) <: Integer
        ndims(i.T) == 1 || _jax_error("a gather by a rank-", ndims(i.T), " index array",
                                      _jax_where(ctx))
        return (_py_infix("-", [i.code, "1"]), 1, i)
    end
    if i.T isa Type && i.T <: AbstractUnitRange
        m = match(r"^jnp\.arange\((.*), (.*)\)$", i.code)
        m === nothing || return (_jax_minus_one(_PyVal(m.captures[1], Int)) * ":" *
                                 _jax_minus_one(_PyVal(m.captures[2], Int)), 1, i)
        return (_py_infix("-", [i.code, "1"]), 1, i)
    end
    _jax_error("the index `", index, "` (of type `", i.T, "`) has no JAX translation",
               _jax_where(ctx))
end

_jax_bool_index_error(ctx::_JaxContext, index) =
    _jax_error("indexing an array by the `Bool` `", index, "` (Julia refuses it: invalid index)",
               _jax_where(ctx))

# A logical index (a `Bool` vector, `A[mask, j]`): the entries where it is
# true, in order. The host takes NumPy's boolean mask, which also refuses a
# mask of another length, as Julia does. JAX shapes are static, so the device
# gathers at a padded position vector instead: the true positions, then the
# last of them repeated (padding repeats the last valid entry), with their
# count as the extent that reductions mask by. A write's padding is past the
# end, where JAX drops the update, so an all-false mask writes nothing. A
# padded mask counts only its valid entries.
function _jax_mask_part(ctx::_JaxContext, x::_PyVal, i::_PyVal; target::Bool = false)
    ndims(i.T) == 1 || _jax_error("a rank-", ndims(i.T), " `Bool` mask beside other indices",
                                  _jax_where(ctx))
    ctx.host === nothing || return ("np.asarray(" * i.code * ", dtype=bool)", 1, i)
    _jax_ragged(x.T) && _jax_error("a `Bool` mask of the groups of a ragged value has no ",
                                   "JAX translation", _jax_where(ctx))
    valid = isempty(i.extent) ? i.code : "jnp.logical_and(" * i.code * ", " * _jax_mask(i) * ")"
    mask = _jax_bind!(ctx, :mask, _PyVal(valid, Vector{Bool}))
    n = _py_atom(mask.code) * ".shape[0]"
    count = _jax_bind!(ctx, :count, _PyVal("jnp.sum(" * mask.code * ")", Int))
    fill = target ? n : "jnp.max(jnp.where(" * mask.code * ", jnp.arange(" * n * "), 0), initial=0)"
    positions = _jax_bind!(ctx, :positions,
        _PyVal("jnp.nonzero(" * mask.code * ", size=" * n * ", fill_value=" * fill * ")[0]",
               Vector{Int}, false, count.code))
    (positions.code, 1, positions)
end

# An index expression in which `end` and `begin` name the bounds of `x`'s
# dimension `dim`.
function _jax_bound_expr(ctx::_JaxContext, x::_PyVal, dim::Int, index)
    # A history access may already have resolved its index while checking
    # the storage layout. Reuse it without repeating any emitted effects.
    index isa _PyVal && return index
    index === :end && return _PyVal(_jax_dim_code(ctx, x, dim), Int, true)
    index === :begin && return _PyVal("1", Int)
    if index isa Expr && (_jax_mentions(index, :end) || _jax_mentions(index, :begin))
        saved = copy(ctx.locals)
        try
            ctx.locals[:end] = _PyVal(_jax_dim_code(ctx, x, dim), Int, true)
            ctx.locals[:begin] = _PyVal("1", Int)
            return _jax_expr(ctx, index)
        finally
            empty!(ctx.locals); merge!(ctx.locals, saved)
        end
    end
    _jax_expr(ctx, index)
end
# The length of `x`'s dimension `dim`; a host value may be a Python list.
_jax_dim_code(ctx::_JaxContext, x::_PyVal, dim::Int) = ctx.host === nothing ?
    _py_atom(x.code) * ".shape[" * string(dim - 1) * "]" :
    dim == 1 ? "len(" * x.code * ")" : "np.shape(" * x.code * ")[" * string(dim - 1) * "]"
_jax_mentions(ex, s::Symbol) = ex === s ||
    (ex isa Expr && !(ex.head === :ref) && any(a -> _jax_mentions(a, s), ex.args))

function _jax_minus_one(i::_PyVal)
    m = match(r"^-?[0-9]+$", i.code)
    m === nothing ? _py_infix("-", [i.code, "1"]) : string(parse(Int, i.code) - 1)
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
        return _PyVal(_py_atom(x.code) * "[" * string(k - 1) * "]", fieldtype(x.T, k))
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
# traced left side combines both sides, and a throw on the right side is a
# check masked by the left (`_jax_call_positional(ctx, throw, ...)`).
_jax_expr(ctx::_JaxContext, ::Val{:&&}, ex) = _jax_short_circuit(ctx, ex, false)
_jax_expr(ctx::_JaxContext, ::Val{:||}, ex) = _jax_short_circuit(ctx, ex, true)
function _jax_short_circuit(ctx::_JaxContext, ex, is_or::Bool)
    ctx.host === nothing || return _host_short_circuit(ctx, ex, is_or)
    a = _jax_expr(ctx, ex.args[1])
    if a.static
        rest = () -> _jax_expr(ctx, ex.args[2])
        decided = () -> _jax_literal(is_or)
        return is_or ? _jax_conditional(ctx, a, decided, rest) :
                       _jax_conditional(ctx, a, rest, decided)
    end
    # The right side runs where Julia's would, so a check in it (or a throw,
    # which is one) is masked by that: `a` for `&&`, `!a` for `||`.
    ctx.traced_branches += 1
    b = try
        _jax_with_guard(() -> _jax_expr(ctx, ex.args[2]), ctx.scope,
                        is_or ? "jnp.logical_not(" * a.code * ")" : a.code)
    finally
        ctx.traced_branches -= 1
    end
    # A right side that throws (`scale >= 0 || throw(...)`) is its check, and
    # the value is the left side's wherever the module does not raise.
    b.T === Union{} && return a
    _PyVal((is_or ? "jnp.logical_or(" : "jnp.logical_and(") * a.code * ", " * b.code * ")", Bool)
end

# `map(xs...) do args... body end`: the body as a function of one element of
# each collection, vectorized over them (padded collections stay padded).
function _jax_expr(ctx::_JaxContext, ::Val{:do}, ex)
    ctx.host === nothing || return _host_do(ctx, ex)
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

# Generators and comprehensions: the element expression as a function of the
# iteration variable, `jax.vmap`ped over the range (a range with a
# data-dependent bound runs to its static limit; reductions mask the rest).
# Reductions combine the mapped values; `init` is added (or multiplied) in.
const _JAX_GENERATOR_REDUCTIONS = Dict{Any,Tuple{String,String}}(
    sum => ("jnp.sum", "0.0"), prod => ("jnp.prod", "1.0"),
    maximum => ("jnp.max", "-jnp.inf"), minimum => ("jnp.min", "jnp.inf"),
    any => ("jnp.any", "False"), all => ("jnp.all", "True"))

# Whether a range from `lo` to `hi`, Python integers known while JAX traces,
# has elements: `true` or `false` when both are literals, else the Python
# condition. JAX traces a loop's body, and a `vmap`ped element, even for no
# steps, and one that gathers from an empty array then fails (`a[r]` in
# `for r in 1:length(a)` with `a` empty), so an empty range runs neither.
function _jax_nonempty(lo::AbstractString, hi::AbstractString)
    a, b = tryparse(Int, lo), tryparse(Int, hi)
    a !== nothing && b !== nothing && return a <= b
    _py_infix("<=", [lo, hi])
end
# `target = call` where `nonempty` (`_jax_nonempty`) holds, else `target =
# empty`, as Python decides while JAX traces.
function _jax_emit_unless_empty!(ctx::_JaxContext, target::AbstractString, call::AbstractString,
                                 empty::AbstractString, nonempty)
    nonempty === true && return _jax_emit!(ctx, target * " = " * call)
    nonempty === false && return _jax_emit!(ctx, target * " = " * empty)
    _jax_emit!(ctx, target * " = " * call * " if " * nonempty * " else " * empty)
end
# A loop's initial carry, bound to a name when the loop may not run (both
# branches of `_jax_emit_unless_empty!` read it).
function _jax_initial_carry(ctx::_JaxContext, initial, nonempty)
    code = "(" * join((v.code for v in initial), ", ") * (length(initial) == 1 ? ",)" : ")")
    nonempty === true ? code : _jax_bind!(ctx, :carry_init, _PyVal(code, Any)).code
end
# The values of a `vmap` over no elements: `E`s, and the filter's mask.
_jax_no_values(@nospecialize(E), filtered::Bool) =
    filtered ? "(jnp.zeros((0,), dtype=" * _jax_dtype(E) * "), jnp.zeros((0,), dtype=jnp.bool_))" :
    "jnp.zeros((0,), dtype=" * _jax_dtype(E) * ")"

# A loop or generator range `_jax_loop_range` reads: `a:b`, `eachindex(x)`,
# `axes(x, 1)`.
_jax_is_range_form(ctx::_JaxContext, iter) =
    iter isa Expr && iter.head === :call &&
    (iter.args[1] === :(:) && length(iter.args) == 3 ||
     length(iter.args) >= 2 && _jax_resolve(ctx, iter.args[1]) in (eachindex, axes))

# A generator over a vector's elements (`o.square for o in out`): `jax.vmap`
# of the element over the vector, or over the columns of a vector of tuples;
# a padded vector masks its padding as a filter does.
function _jax_collection_values(ctx::_JaxContext, body, var::Symbol, iter, condition)
    xs = _jax_expr(ctx, iter)
    _jax_rank(xs.T) == 1 && !_jax_ragged(xs.T) ||
        _jax_error("a generator over a `", xs.T, "`", _jax_where(ctx))
    element = _jax_fresh(ctx, var)
    fname = _jax_fresh(ctx, :element)
    outer = ctx.scope
    inner = _JaxScope(outer)
    inner.indent = outer.indent + 1
    push!(inner.used, element)
    saved = copy(ctx.locals)
    ctx.scope = inner
    value, keep = try
        ctx.locals[var] = _PyVal(element, eltype(xs.T))
        (_jax_expr(ctx, body), condition === nothing ? nothing : _jax_expr(ctx, condition))
    finally
        ctx.scope = outer
        empty!(ctx.locals); merge!(ctx.locals, saved)
    end
    push!(outer.lines, "    "^outer.indent * "def " * fname * "(" * element * "):")
    append!(outer.lines, inner.lines)
    push!(outer.lines, "    "^inner.indent * "return " *
          (keep === nothing ? value.code : "(" * value.code * ", " * keep.code * ")"))
    E = _jax_typeof(value)
    E <: Number || _jax_error("a generator whose elements are `", E, "`", _jax_where(ctx))
    # An empty collection maps no element (`_jax_nonempty`).
    xs = _jax_bind!(ctx, :collection, xs)
    first_column = _jax_columns(xs.T) ? _jax_column(xs, 1) : xs
    nonempty = _py_infix("<", ["0", _py_atom(first_column.code) * ".shape[0]"])
    values = _jax_fresh(ctx, :values)
    masks = String[]
    mapped = "jax.vmap(" * fname * ")(" * xs.code * ")"
    if keep === nothing
        _jax_emit_unless_empty!(ctx, values, mapped, _jax_no_values(E, false), nonempty)
    else
        kept = _jax_fresh(ctx, :kept)
        _jax_emit_unless_empty!(ctx, values * ", " * kept, mapped, _jax_no_values(E, true), nonempty)
        push!(masks, kept)
    end
    if !isempty(xs.extent)
        n = _jax_function_call(ctx, length, [xs])
        push!(masks, "(jnp.arange(" * n.limit * ") < " * n.code * ")")
    end
    _PyVal(values, Vector{E}), masks
end

function _jax_generator_values(ctx::_JaxContext, gen::Expr)
    body, spec = gen.args[1], gen.args[2]
    condition = nothing
    if spec isa Expr && spec.head === :filter
        condition, spec = spec.args[1], spec.args[2]
    end
    spec isa Expr && spec.head === :(=) && spec.args[1] isa Symbol ||
        _jax_error("the generator `", gen, "` has no JAX translation", _jax_where(ctx))
    var = spec.args[1]
    _jax_is_range_form(ctx, spec.args[2]) ||
        return (_jax_collection_values(ctx, body, var, spec.args[2], condition)..., nothing)
    lo, hi, bound, steps = _jax_loop_range(ctx, spec.args[2])
    # Ends that are data (`steps`): the element at step `k` is the range's
    # `lo + k` (clamped to `hi`), Julia's while `k <= hi - lo`. It is a lazy
    # branch, `jax.lax.cond`, whose other arm is a zero the reduction masks,
    # so a later step's reads reach neither value nor derivative (as a
    # stepped `for`, `_jax_stepped_for`).
    stepped = !isempty(steps)
    reach = stepped ? _jax_bind!(ctx, :reach, _PyVal(_py_infix("-", [hi.code, lo.code]), Int)) :
        nothing
    index = _jax_fresh(ctx, stepped ? :step : var)
    fname = _jax_fresh(ctx, :element)
    outer = ctx.scope
    inner = _JaxScope(outer)
    inner.indent = outer.indent + 1
    push!(inner.used, index)
    saved = copy(ctx.locals)
    ctx.scope = inner
    run = stepped ? _JaxScope(inner) : inner
    returned = try
        if stepped
            kept = _jax_fresh(ctx, :kept)
            _jax_emit!(ctx, kept * " = " * _py_infix("<=", [index, reach.code]))
            run_name = _jax_fresh(ctx, :element_body)
            run.indent = inner.indent + 1
            ctx.scope = run
            element_index = _jax_fresh(ctx, var)
            _jax_emit!(ctx, element_index * " = jnp.minimum(" * _py_infix("+", [lo.code, index]) *
                            ", " * hi.code * ")")
            ctx.locals[var] = _jax_range_index(element_index, lo, hi)
        else
            ctx.locals[var] = _jax_range_index(index, lo, hi)
        end
        element = () -> (_jax_expr(ctx, body),
                         condition === nothing ? nothing : _jax_expr(ctx, condition))
        guard = stepped ? kept : isempty(bound) ? "" : index * " <= " * bound
        value, keep = isempty(guard) ? element() : _jax_with_guard(element, run, guard)
        if stepped
            E = _jax_typeof(value)
            E <: Number || _jax_error("a generator whose elements are `", E, "`", _jax_where(ctx))
            # Both arms return the element's type, and a filter's `Bool`.
            arm = "jnp.asarray(" * value.code * ", dtype=" * _jax_dtype(E) * ")"
            zero = "jnp.zeros((), dtype=" * _jax_dtype(E) * ")"
            if keep !== nothing
                arm = "(" * arm * ", jnp.asarray(" * keep.code * ", dtype=jnp.bool_))"
                zero = "(" * zero * ", jnp.zeros((), dtype=jnp.bool_))"
            end
            ctx.scope = inner
            push!(inner.lines, "    "^inner.indent * "def " * run_name * "():")
            append!(inner.lines, run.lines)
            push!(inner.lines, "    "^run.indent * "return " * arm)
            (value, keep, "jax.lax.cond(" * kept * ", " * run_name * ", lambda: " * zero * ")")
        else
            (value, keep, keep === nothing ? value.code : "(" * value.code * ", " * keep.code * ")")
        end
    finally
        ctx.scope = outer
        empty!(ctx.locals); merge!(ctx.locals, saved)
    end
    value, keep, element_code = returned
    push!(outer.lines, "    "^outer.indent * "def " * fname * "(" * index * "):")
    append!(outer.lines, inner.lines)
    push!(outer.lines, "    "^inner.indent * "return " * element_code)
    indices = _jax_fresh(ctx, stepped ? :steps : :indices)
    _jax_emit!(ctx, indices * " = jnp.arange(" *
                    (stepped ? steps : lo.code * ", " * _jax_plus_one(hi)) * ")")
    mapped = "jax.vmap(" * fname * ")(" * indices * ")"
    E = _jax_typeof(value)
    E <: Number || _jax_error("a generator whose elements are `", E, "`", _jax_where(ctx))
    # An empty range maps no element (`_jax_nonempty`).
    nonempty = stepped ? _jax_nonempty("1", steps) : _jax_nonempty(lo.code, hi.code)
    values = _jax_fresh(ctx, :values)
    masks = String[]
    if keep === nothing
        _jax_emit_unless_empty!(ctx, values, mapped, _jax_no_values(E, false), nonempty)
    else
        kept = _jax_fresh(ctx, :kept)
        _jax_emit_unless_empty!(ctx, values * ", " * kept, mapped, _jax_no_values(E, true), nonempty)
        push!(masks, kept)
    end
    # The number of Julia's elements, a comprehension's length.
    count = if stepped
        push!(masks, _py_infix("<=", [indices, reach.code]))
        "jnp.maximum(" * _py_infix("+", [reach.code, "1"]) * ", 0)"
    elseif !isempty(bound)
        push!(masks, _py_infix("<=", [indices, bound]))
        _jax_range_count(lo, _PyVal(bound, Int))
    else
        ""
    end
    _PyVal(values, Vector{E}), masks, count
end

function _jax_generator_reduce(ctx::_JaxContext, f, gen::Expr, kwargs)
    ctx.host === nothing || return _host_generator_reduce(ctx, f, gen, kwargs)
    values, masks, _ = _jax_generator_values(ctx, gen)
    reduce, neutral = _JAX_GENERATOR_REDUCTIONS[f]
    operand = isempty(masks) ? values.code :
        "jnp.where(" * join(masks, " & ") * ", " * values.code * ", " * neutral * ")"
    E = eltype(values.T)
    result = _PyVal(reduce * "(" * operand * ")", f in (any, all) ? Bool : E)
    for (k, v) in kwargs
        k === :init && f in (sum, prod) ||
            _jax_error("the keyword `", k, "` of `", f, "` over a generator", _jax_where(ctx))
        init = _jax_expr(ctx, v)
        result = _PyVal(_py_infix(f === sum ? "+" : "*", [init.code, result.code]),
                        promote_type(_jax_typeof(init), E))
    end
    result
end

function _jax_expr(ctx::_JaxContext, ::Val{:comprehension}, ex)
    gen = only(ex.args)
    gen isa Expr && gen.head === :generator ||
        _jax_error("the comprehension `", ex, "` has no JAX translation", _jax_where(ctx))
    ctx.host === nothing || return _host_comprehension(ctx, gen)
    gen.args[2] isa Expr && gen.args[2].head === :filter &&
        _jax_error("a filtered comprehension has a data-dependent length", _jax_where(ctx))
    values, masks, count = _jax_generator_values(ctx, gen)
    isempty(masks) && return values
    if count === nothing
        # The range's data-dependent bound is the vector's length.
        _, hi, bound, _ = _jax_loop_range(ctx, gen.args[2].args[2])
        count = bound
    end
    # A range with a data-dependent end: the vector is padded to its static
    # length, and its length is the number of the range's elements.
    _PyVal(values.code, values.T, false, count)
end

# A `while` loop: `jax.lax.while_loop` over the outer locals the body assigns.
# JAX differentiates it forward only; `jax.grad` through it raises JAX's own
# error, so a differentiated program writes the loop as a `for` over a range.
function _jax_expr(ctx::_JaxContext, ::Val{:while}, ex)
    ctx.host === nothing || return _host_while(ctx, ex)
    counted = _jax_counted_while(ctx, ex)
    counted === nothing || return _jax_expr(ctx, counted)
    condition_ex, body = ex.args
    assigned = Set{Symbol}()
    _jax_assigned_names!(assigned, body)
    carried = sort!([n for n in assigned if haskey(ctx.locals, n)])
    isempty(carried) && _jax_error("a `while` loop that assigns no outer local", _jax_where(ctx))
    initial = [_jax_carry_value(ctx, ctx.locals[n]) for n in carried]
    carry_code = "(" * join((v.code for v in initial), ", ") * (length(initial) == 1 ? ",)" : ")")
    function loop_function(hint, build)
        fname = _jax_fresh(ctx, hint)
        carry = _jax_fresh(ctx, :carry)
        outer = ctx.scope
        inner = _JaxScope(outer)
        inner.indent = outer.indent + 1
        push!(inner.used, carry)
        saved = copy(ctx.locals)
        ctx.scope = inner
        result = try
            names = [_jax_identifier(inner.used, n) for n in carried]
            _jax_emit!(ctx, "(" * join(names, ", ") * (length(names) == 1 ? ",)" : ")") * " = " * carry)
            for (n, name, v) in zip(carried, names, initial)
                ctx.locals[n] = _PyVal(name, v.T)
            end
            build()
        finally
            ctx.scope = outer
            empty!(ctx.locals); merge!(ctx.locals, saved)
        end
        push!(outer.lines, "    "^outer.indent * "def " * fname * "(" * carry * "):")
        append!(outer.lines, inner.lines)
        push!(outer.lines, "    "^inner.indent * "return " * result)
        fname
    end
    cond_f = loop_function(:loop_condition, () -> _jax_expr(ctx, condition_ex).code)
    body_f = loop_function(:loop_body, () -> begin
        _jax_statements(ctx, body isa Expr && body.head === :block ? body.args : Any[body];
                        value = false)
        "(" * join((ctx.locals[n].code for n in carried), ", ") * (length(carried) == 1 ? ",)" : ")")
    end)
    result = _jax_fresh(ctx, :loop_carry)
    _jax_emit!(ctx, result * " = jax.lax.while_loop(" * cond_f * ", " * body_f * ", " * carry_code * ")")
    for (k, (n, v)) in enumerate(zip(carried, initial))
        ctx.locals[n] = _PyVal(result * "[" * string(k - 1) * "]", v.T)
    end
    _PyVal("None", Nothing, true)
end

# A counted `while i < n` (or `<=`) whose body ends in its only assignment
# of `i`, `i = i + 1`, over a static `n`: the `for` over `i:(n - 1)` that sets
# `i` first, which `jax.grad` reverses (a `while_loop` only goes forward).
# The loop leaves `i` at the value the `while` would.
function _jax_counted_while(ctx::_JaxContext, ex)
    condition, body = ex.args
    condition isa Expr && condition.head === :call && length(condition.args) == 3 &&
        condition.args[1] in (:<, :<=) && condition.args[2] isa Symbol || return nothing
    i, limit = condition.args[2], condition.args[3]
    haskey(ctx.locals, i) && _jax_typeof(ctx.locals[i]) <: Integer || return nothing
    statements = body isa Expr && body.head === :block ? body.args : Any[body]
    last = findlast(st -> !(st isa LineNumberNode), statements)
    last === nothing && return nothing
    step = statements[last]
    step == :($i = $i + 1) || step == :($i += 1) || return nothing
    assigned = Set{Symbol}()
    foreach(st -> _jax_assigned_names!(assigned, st), statements[1:last-1])
    i in assigned && return nothing
    n = _jax_expr(ctx, limit)
    n.static && _jax_typeof(n) <: Integer || return nothing
    k = gensym(i)
    hi = condition.args[1] === :< ? :($limit - 1) : limit
    Expr(:for, :($k = $i:$hi), Expr(:block, :($i = $k), statements...))
end

# Loops. A `for` over a range is `jax.lax.fori_loop` over a function of the
# index and the carry, the outer locals the body assigns (an indexed write
# assigns its array). The range's bounds are static, so the loop lowers to a
# scan that `jax.grad` differentiates; a data-dependent upper bound with a
# static limit (a padded group vector's length) runs to the limit and keeps
# the carry unchanged past the bound (`_JaxLoopMask`): an array the body
# rebinds is selected whole, and an indexed write keeps the entry it would
# change, except a write at the loop index into an array padded by the same
# length, whose entries past the bound are padding. Ends that are data with a
# static number of steps (`_jax_loop_range`) run those steps from the first
# element; past the last element a step repeats the last (as padding does)
# and keeps the whole carry, arrays included, unchanged.
function _jax_expr(ctx::_JaxContext, ::Val{:for}, ex)
    ctx.host === nothing || return _host_for(ctx, ex)
    binding, body = ex.args
    binding isa Expr && binding.head === :(=) && binding.args[1] isa Symbol ||
        _jax_error("the loop binding `", binding, "` has no JAX translation", _jax_where(ctx))
    var = binding.args[1]
    lo, hi, bound, steps = _jax_loop_range(ctx, binding.args[2])
    isempty(steps) || return _jax_stepped_for(ctx, var, body, lo, hi, steps)
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
            ctx.locals[n] = _PyVal(name, v.T, false, v.extent, v.limit, v.padded)
        end
        ctx.locals[var] = _jax_range_index(index, lo, hi)
        statements = body isa Expr && body.head === :block ? body.args : Any[body]
        kept = index * " <= " * bound
        # Arrays the body rebinds (not only writes by index) are selected
        # whole past the bound; the others keep their entries at each write.
        rebound = isempty(bound) ? Set{Symbol}() : _jax_rebound_names!(Set{Symbol}(), body)
        written = Set{Symbol}(n for (n, v) in zip(carried, initial)
                              if _jax_is_array(v.T) && !(n in rebound))
        isempty(bound) ? _jax_statements(ctx, statements; value = false) :
            _jax_with_guard(inner, kept) do
                _jax_with_loop_mask(() -> _jax_statements(ctx, statements; value = false),
                                    _JaxLoopMask(inner, kept, index, bound, written))
            end
        map(zip(carried, initial)) do (n, v)
            new = ctx.locals[n]
            bound == "" ? new.code : _jax_carry_select(ctx, kept, new.code,
                                                       _jax_carry_old(names, carried, n), v.T;
                                                       arrays = n in rebound)
        end
    finally
        ctx.scope = outer
        empty!(ctx.locals); merge!(ctx.locals, saved)
    end
    push!(outer.lines, "    "^outer.indent * "def " * fname * "(" * index * ", " * carry * "):")
    append!(outer.lines, inner.lines)
    push!(outer.lines, "    "^inner.indent * "return (" * join(updated, ", ") *
                       (length(updated) == 1 ? ",)" : ")"))
    # An empty range runs no loop: the carry keeps its initial values
    # (`_jax_nonempty`).
    nonempty = _jax_nonempty(lo.code, hi.code)
    init = _jax_initial_carry(ctx, initial, nonempty)
    result = _jax_fresh(ctx, :loop_carry)
    _jax_emit_unless_empty!(ctx, result, "jax.lax.fori_loop(" * lo.code * ", " * _jax_plus_one(hi) *
                            ", " * fname * ", " * init * ")", init, nonempty)
    for (k, (n, v)) in enumerate(zip(carried, initial))
        ctx.locals[n] = _PyVal(result * "[" * string(k - 1) * "]", v.T, false, v.extent, v.limit,
                               v.padded)
    end
    _PyVal("None", Nothing, true)
end
_jax_carry_old(names, carried, n) = names[findfirst(==(n), carried)]

# A `for` over a range whose ends are data, `lo:hi`, taking a static number
# of `steps` (`_jax_loop_range`): step `k` is a lazy branch, `jax.lax.cond`,
# that runs the body at `lo + k` (clamped to `hi`) when `k <= hi - lo`,
# Julia's elements, and otherwise keeps the carry. A later step does not run
# the body (CONTRIBUTING.md, "Lazy branches"): its reads at an element Julia
# never reaches cannot reach the value or the derivative. Under `jax.vmap`
# (a plate cell) JAX runs both arms and selects, reading the untaken one's
# inputs through `stop_gradient`, so a non-finite value there reaches neither.
function _jax_stepped_for(ctx::_JaxContext, var::Symbol, body, lo::_PyVal, hi::_PyVal,
                          steps::AbstractString)
    assigned = Set{Symbol}()
    _jax_assigned_names!(assigned, body)
    carried = sort!([n for n in assigned if haskey(ctx.locals, n) && n !== var])
    initial = [_jax_carry_value(ctx, ctx.locals[n]) for n in carried]
    # The steps that are Julia's: `k <= last - first`.
    reach = _jax_bind!(ctx, :reach, _PyVal(_py_infix("-", [hi.code, lo.code]), Int))
    fname = _jax_fresh(ctx, :loop_body)
    step = _jax_fresh(ctx, :step)
    carry = _jax_fresh(ctx, :carry)
    outer = ctx.scope
    inner = _JaxScope(outer)
    inner.indent = outer.indent + 1
    push!(inner.used, step, carry)
    saved = copy(ctx.locals)
    ctx.scope = inner
    try
        kept = _jax_fresh(ctx, :kept)
        _jax_emit!(ctx, kept * " = " * _py_infix("<=", [step, reach.code]))
        # The step's body, a function of the carry, which the branch runs.
        run_name = _jax_fresh(ctx, :step_body)
        run_carry = _jax_fresh(ctx, :carry)
        run = _JaxScope(inner)
        run.indent = inner.indent + 1
        push!(run.used, run_carry)
        ctx.scope = run
        names = [_jax_identifier(run.used, n) for n in carried]
        isempty(names) || _jax_emit!(ctx, "(" * join(names, ", ") * (length(names) == 1 ? ",)" : ")") *
                                         " = " * run_carry)
        for (n, name, v) in zip(carried, names, initial)
            ctx.locals[n] = _PyVal(name, v.T, false, v.extent, v.limit, v.padded)
        end
        index = _jax_fresh(ctx, var)
        _jax_emit!(ctx, index * " = jnp.minimum(" * _py_infix("+", [lo.code, step]) * ", " *
                        hi.code * ")")
        ctx.locals[var] = _jax_range_index(index, lo, hi)
        statements = body isa Expr && body.head === :block ? body.args : Any[body]
        _jax_with_guard(() -> _jax_statements(ctx, statements; value = false), run, kept)
        # The arms return the carry's types (each value coerced as it entered).
        updated = [_jax_carry_value(ctx, _PyVal(ctx.locals[n].code, v.T)).code
                   for (n, v) in zip(carried, initial)]
        ctx.scope = inner
        push!(inner.lines, "    "^inner.indent * "def " * run_name * "(" * run_carry * "):")
        append!(inner.lines, run.lines)
        push!(inner.lines, "    "^run.indent * "return (" * join(updated, ", ") *
                           (length(updated) == 1 ? ",)" : ")"))
        _jax_emit!(ctx, "return jax.lax.cond(" * kept * ", " * run_name * ", lambda c: c, " *
                        carry * ")")
    finally
        ctx.scope = outer
        empty!(ctx.locals); merge!(ctx.locals, saved)
    end
    push!(outer.lines, "    "^outer.indent * "def " * fname * "(" * step * ", " * carry * "):")
    append!(outer.lines, inner.lines)
    nonempty = _jax_nonempty("1", steps)
    init = _jax_initial_carry(ctx, initial, nonempty)
    result = _jax_fresh(ctx, :loop_carry)
    _jax_emit_unless_empty!(ctx, result, "jax.lax.fori_loop(0, " * steps * ", " * fname * ", " *
                            init * ")", init, nonempty)
    for (k, (n, v)) in enumerate(zip(carried, initial))
        ctx.locals[n] = _PyVal(result * "[" * string(k - 1) * "]", v.T, false, v.extent, v.limit,
                               v.padded)
    end
    _PyVal("None", Nothing, true)
end

# A loop carry enters with a fixed dtype, as `fori_loop` requires. A tuple
# (or `NamedTuple`) stays a Python tuple, its elements coerced one by one, so
# it has the structure the loop body and a branch's arms return.
# A scan's initial carry. A float32 module types it as a loop's carry is: a
# literal (`level[1] = 0.0`) is float64 under 64-bit mode, and the carry would
# stay float64 through the step's float32 arithmetic.
_jax_scan_init(ctx::_JaxContext, v::_PyVal) =
    _jax_float_type() === Float64 ? v : _jax_carry_value(ctx, v)
function _jax_carry_value(ctx::_JaxContext, v::_PyVal)
    _jax_is_array(v.T) && return v
    if _jax_is_tuple(v.T)
        t = _jax_bind!(ctx, :carried, v).code
        parts = [_jax_carry_value(ctx, _PyVal(_py_atom(t) * "[" * string(k - 1) * "]",
                                              fieldtype(v.T, k)))
                 for k in 1:fieldcount(v.T)]
        return _PyVal(_jax_tuple_code(parts), v.T, false, v.extent, v.limit, v.padded)
    end
    T = v.T isa Type ? v.T : Float64
    _PyVal("jnp.asarray(" * v.code * ", dtype=" * _jax_dtype(T) * ")", T, false, "", "")
end

# Past a loop's data-dependent bound (`kept` is false there) the carry keeps
# its value `old`, of type `T`: a scalar is selected, a tuple element by
# element. An array passes as it is: past a padded length's bound each write
# by index keeps its entry or writes padding (`_JaxLoopMask`). With
# `arrays` (an array the body rebinds) it is selected too.
function _jax_carry_select(ctx::_JaxContext, kept::AbstractString, new::AbstractString,
                           old::AbstractString, @nospecialize(T); arrays::Bool = false)
    _jax_is_array(T) && !arrays && return new
    _jax_is_tuple(T) || return "jnp.where(" * kept * ", " * new * ", " * old * ")"
    t = _jax_bind!(ctx, :carried, _PyVal(new, T)).code
    _jax_tuple_code([_PyVal(_jax_carry_select(ctx, kept, _py_atom(t) * "[" * string(k - 1) * "]",
                                              _py_atom(old) * "[" * string(k - 1) * "]",
                                              fieldtype(T, k); arrays), Any)
                     for k in 1:fieldcount(T)])
end

# The locals a loop body binds to a new value, `x = ...` or `(x, y) = ...`
# (not an indexed write `x[i] = ...`).
function _jax_rebound_names!(names::Set{Symbol}, ex)
    ex isa Expr || return names
    ex.head in (:quote, :inert, :->, :function) && return names
    if (ex.head === :(=) || haskey(_JAX_UPDATING, ex.head)) && length(ex.args) == 2
        target = ex.args[1]
        target isa Expr && target.head === :(::) && (target = target.args[1])
        target isa Symbol && push!(names, target)
        target isa Expr && target.head === :tuple &&
            foreach(t -> t isa Symbol && push!(names, t), target.args)
    end
    foreach(a -> _jax_rebound_names!(names, a), ex.args)
    names
end

# A loop running past its data-dependent bound, a padded length: in its
# `scope`, steps past `bound` (where `kept` is false) leave the outer arrays
# it only writes by index (`written`) as they are, except entries along an
# axis that `bound` pads at the loop `index`, which are padding.
struct _JaxLoopMask
    scope::_JaxScope
    kept::String
    index::String
    bound::String
    written::Set{Symbol}
end
const _JAX_LOOP_MASKS = _JaxLoopMask[]
function _jax_with_loop_mask(f, mask::_JaxLoopMask)
    push!(_JAX_LOOP_MASKS, mask)
    try
        return f()
    finally
        pop!(_JAX_LOOP_MASKS)
    end
end
# The condition under which the write `name[indices...] = ...` to `target`
# runs as Julia's would (`nothing`: always): the conjunction of the masks of
# the loops it runs in whose steps past their bound would change an entry.
function _jax_write_mask(ctx::_JaxContext, name::Symbol, target::_PyVal, indices)
    masks = String[]
    for m in _JAX_LOOP_MASKS
        name in m.written && _jax_scope_within(ctx.scope, m.scope) || continue
        _jax_writes_padding(ctx, target, indices, m) && continue
        push!(masks, m.kept)
    end
    isempty(masks) ? nothing : foldl((a, b) -> "jnp.logical_and(" * a * ", " * b * ")", masks)
end
# Whether `target[indices...]` is, at every step past the loop's bound, a
# padded entry: an index is the loop's own, along an axis padded by its bound.
function _jax_writes_padding(ctx::_JaxContext, target::_PyVal, indices, m::_JaxLoopMask)
    target.extent == m.bound || return false
    axes = _jax_padded_axes(target)
    any(enumerate(indices)) do (k, i)
        i isa Symbol && haskey(ctx.locals, i) && ctx.locals[i].code == m.index &&
            axes & (1 << (k - 1)) != 0
    end
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

# A loop range: `a:b`, `eachindex(x)`, `axes(x, 1)`, as
# `(first, last, bound, steps)`.
#
# - Static ends: the loop runs from `first` to `last`; `bound` and `steps`
#   are empty.
# - A stop that is data with a static limit (a padded group's length): the
#   loop runs from `first` to `last`, the limit, and Julia's elements are
#   those up to `bound`, the stop.
# - Ends that are data with a static number of steps (`_jax_trip_bound`:
#   entries of the prepared data, `starts[r]:stops[r]`): the loop takes
#   `steps` steps, element `first + k` at step `k`, and Julia's elements are
#   those up to `last`; `first` and `last` are bound to names.
function _jax_loop_range(ctx::_JaxContext, iter)
    if iter isa Expr && iter.head === :call && iter.args[1] === :(:) && length(iter.args) == 3
        lo, hi = _jax_expr(ctx, iter.args[2]), _jax_expr(ctx, iter.args[3])
    elseif iter isa Expr && iter.head === :call && length(iter.args) >= 2 &&
           _jax_resolve(ctx, iter.args[1]) in (eachindex, axes)
        x = _jax_expr(ctx, iter.args[2])
        lo = _jax_literal(1)
        if _jax_resolve(ctx, iter.args[1]) === axes && length(iter.args) == 3
            hi = _jax_function_call(ctx, size, [x, _jax_expr(ctx, iter.args[3])])
        else
            _jax_rank(x.T) == 1 || _jax_error("a loop over the indices of a `", x.T, "`",
                                              _jax_where(ctx))
            hi = isempty(x.extent) ? _PyVal(_py_atom(x.code) * ".shape[0]", Int, true) :
                _PyVal(x.extent, Int, false, "", _py_atom(x.code) * ".shape[0]")
        end
    else
        _jax_error("the loop range `", iter, "` has no JAX translation", _jax_where(ctx))
    end
    lo.static && hi.static && return (lo, hi, "", "")
    lo.static && !isempty(hi.limit) && return (lo, _PyVal(hi.limit, Int, true), hi.code, "")
    steps = _jax_trip_bound(ctx, lo, hi)
    steps === nothing && _jax_error("a loop whose ", lo.static ? "trip count" : "first index",
        " depends on data has no JAX translation: its number of steps needs a static ",
        "bound. Ends that are entries of integer arrays of the prepared data ",
        "(`jax_source(...; data)`) have one", _jax_where(ctx))
    (_jax_bind!(ctx, :first, lo), _jax_bind!(ctx, :last, hi), "", steps)
end

# A static number of steps for a range from `lo` to `hi`, at least each
# range's length, as Python code (`nothing` when none is known). Entries of
# two prepared arrays at one position (`starts[r]:stops[r]`) take the arrays'
# largest difference, so the loop runs to the longest record's length;
# otherwise the bound is a static upper bound of `hi` (a static value, a
# padded length's limit, or the largest entry of its prepared array) less a
# static lower bound of `lo` (a static value, or the least entry of its array).
function _jax_trip_bound(ctx::_JaxContext, lo::_PyVal, hi::_PyVal)
    lo, hi = _jax_lower_end(lo), _jax_upper_end(hi)
    a, b = lo.origin, hi.origin
    for o in (a, b)
        o isa _JaxCellData || continue
        throw(_CellHostNeeded(o.plate, o.recipe, JAXEmissionError(
            "a data-only integer used as a loop end needs a bound from `prepare_data`" * _jax_where(ctx))))
    end
    # Ends at one position with one positive scale differ by that scale times
    # the arrays' difference there, at most that times their largest one.
    count = if a isa _JaxOrigin && b isa _JaxOrigin && a.position !== nothing &&
               a.position == b.position && a.scale == b.scale && _jax_static_sign(a.scale) > 0
        span = _jax_scaled(a.scale, _jax_prepared_bound!(ctx, :span, a.key, b.key))
        _jax_offset(_jax_offset(span, "+", b.offset), "-", a.offset)
    else
        upper = hi.static ? hi.code : !isempty(hi.limit) ? hi.limit :
            b isa _JaxOrigin && b.position !== nothing ? _jax_entry_bound(ctx, b, true) : nothing
        lower = lo.static ? lo.code : a isa _JaxOrigin && a.position !== nothing ?
            _jax_entry_bound(ctx, a, false) : nothing
        upper === nothing || lower === nothing ? nothing : _py_infix("-", [upper, lower])
    end
    count === nothing ? nothing : "max(" * _py_infix("+", [count, "1"]) * ", 0)"
end
# A static upper (`upper`) or lower bound of a prepared entry `o`,
# `scale * prepared[key][i] + offset`: the scale times the array's largest
# entry, or its least where the scale is negative (either where its sign is
# known only while tracing), plus the offset.
function _jax_entry_bound(ctx::_JaxContext, o::_JaxOrigin, upper::Bool)
    sign = _jax_static_sign(o.scale)
    extreme(kind) = _jax_scaled(o.scale, _jax_prepared_bound!(ctx, kind, o.key))
    bound = sign > 0 ? extreme(upper ? :max : :min) :
        sign < 0 ? extreme(upper ? :min : :max) :
        (upper ? "max(" : "min(") * extreme(:max) * ", " * extreme(:min) * ")"
    _jax_offset(bound, "+", o.offset)
end
# `code op offset`, a static integer offset; nothing added for a zero one.
_jax_offset(code::AbstractString, op::AbstractString, offset::AbstractString) =
    offset == "0" ? code : _py_infix(op, [code, offset])
# `scale * code`, a static integer scale; nothing multiplied for a unit one.
_jax_scaled(scale::AbstractString, code::AbstractString) =
    scale == "1" ? code : _py_infix("*", [scale, code])
# The sign of a static integer's code when it is a literal, else 0 (known
# only while tracing).
function _jax_static_sign(code::AbstractString)
    m = match(r"^-?[0-9]+$", code)
    m === nothing ? 0 : sign(parse(Int, code))
end
# `x op y` of static integers' codes, folded when both are literals.
function _jax_static_integer_op(op::AbstractString, x::AbstractString, y::AbstractString)
    literal = r"^-?[0-9]+$"
    if occursin(literal, x) && occursin(literal, y)
        a, b = parse(Int, x), parse(Int, y)
        return string(op == "+" ? a + b : op == "-" ? a - b : a * b)
    end
    _py_infix(op, [x, y])
end

# The name of a static bound of prepared integer arrays `keys` (`:min` and
# `:max` of one array's entries, `:span` the largest difference of the second
# array's entries less the first's at one position), which `prepare_data`
# computes and returns with the data (`_jax_prepared_bounds`).
function _jax_prepared_bound!(ctx::_JaxContext, kind::Symbol, keys::String...)
    key = (:prepared_bound, kind, keys)
    haskey(ctx.mod.helper_names, key) && return ctx.mod.helper_names[key]
    base = "_rk_" * string(kind) * "_" * join(keys, "_")
    taken = Set{String}(v for (k, v) in ctx.mod.helper_names
                        if k isa Tuple && first(k) === :prepared_bound)
    name, n = base, 1
    while name in taken || name in ctx.mod.used
        n += 1
        name = base * "_" * string(n)
    end
    push!(ctx.mod.used, name)
    ctx.mod.helper_names[key] = name
end
# The prepared bounds requested so far, in name order: `(name, kind, keys)`.
_jax_prepared_bounds(mod::_JaxModule) =
    sort!([(v, k[2], k[3]) for (k, v) in mod.helper_names
           if k isa Tuple && first(k) === :prepared_bound]; by = first)

# The number of elements of a range from `lo` to `hi`, `hi - lo + 1`, as code.
_jax_range_count(lo::_PyVal, hi::_PyVal) =
    lo.code == "1" ? hi.code : _py_infix("-", [_jax_plus_one(hi), lo.code])

# A lazy branch: `jax.lax.cond` of one function per arm, so only the taken
# arm runs (under `jax.vmap` JAX itself decides how to batch the region).
_jax_expr(ctx::_JaxContext, ::Val{:if}, ex) = _jax_if(ctx, ex, true)

# An `if`; with `value = false` its arms' values are discarded.
function _jax_if(ctx::_JaxContext, ex, value::Bool)
    ex.head === :elseif && (ex = Expr(:if, ex.args[1] isa Expr && ex.args[1].head === :block ?
                                      ex.args[1].args[end] : ex.args[1], ex.args[2:end]...))
    ctx.host === nothing || return _host_if(ctx, ex, value)
    condition = _jax_expr(ctx, ex.args[1])
    otherwise = length(ex.args) >= 3 ? ex.args[3] : nothing
    translate(body) = value ? _jax_expr(ctx, body) :
        (_jax_effect(ctx, body); _PyVal("None", Nothing))
    # Outer locals an arm assigns leave the branch with it: each arm returns
    # its value followed by those locals, and the branch rebinds them.
    assigned = Set{Symbol}()
    foreach(a -> _jax_assigned_names!(assigned, a), ex.args[2:end])
    carried = sort!([n for n in assigned if haskey(ctx.locals, n)])
    isempty(carried) && return _jax_conditional(ctx, condition,
        () -> translate(ex.args[2]), () -> translate(otherwise))
    before = Dict(n => ctx.locals[n] for n in carried)
    arm(body) = () -> begin
        v = translate(body)
        # An arm that throws carries nothing out: the branch is the other arm.
        v.T === Union{} && return v
        # A carried number is an array (`_jax_carry_value`) of the type the
        # arm gives it, which the branch promotes with the other arm's
        # (`_jax_branch_type`). So is an integer value known while tracing:
        # it may be a prepared `StaticInt`, which is not a JAX value.
        parts = [v.T === Nothing ? "None" :
                 v.static && v.T isa Type && v.T <: Integer && !(v.T <: Bool) ?
                 _jax_carry_value(ctx, v).code : v.code]
        types = Any[_jax_typeof(v)]
        for n in carried
            c = ctx.locals[n]
            push!(parts, _jax_carry_value(ctx, c).code)
            push!(types, _jax_is_scalar(c.T) && isconcretetype(c.T) ? c.T : before[n].T)
        end
        _PyVal("(" * join(parts, ", ") * ")", Tuple{types...})
    end
    branch = _jax_conditional(ctx, condition, arm(ex.args[2]), arm(otherwise))
    branch.T === Union{} && return branch
    result = _jax_bind!(ctx, :branch, branch)
    for (k, n) in enumerate(carried)
        b = before[n]
        T = _jax_is_tuple(result.T) && result.T <: Tuple && _jax_is_scalar(fieldtype(result.T, k + 1)) ?
            fieldtype(result.T, k + 1) : b.T
        ctx.locals[n] = _PyVal(_py_atom(result.code) * "[" * string(k) * "]", T, false, b.extent,
                               b.limit, b.padded)
    end
    _PyVal(_py_atom(result.code) * "[0]", result.T isa Type && result.T <: Tuple ? fieldtype(result.T, 1) : Any)
end

# An arm of a conditional, translated but not yet emitted: the name hint of
# its zero-argument nested function, the scope holding its body, and its
# value. The function is named only when it is emitted as one: a branch
# whose arms are expressions names nothing (`_jax_emit_arm!`).
struct _JaxArm
    hint::Symbol
    scope::_JaxScope
    value::_PyVal
end

# An emitted arm: what `jax.lax.cond` takes (`callable`: a nested function's
# name, or `lambda: <expression>` for an arm without statements) and the code
# that evaluates it in place (`call`), with its value's type and staticness.
struct _JaxEmittedArm
    callable::String
    call::String
    lambda::Bool
    T::Any
    static::Bool
end

# A conditional: a static condition is a Python conditional, decided while JAX
# traces; any other is `jax.lax.cond`, whose arms JAX traces both of.
#
# An arm of a static condition may raise: the condition is a fact of the
# shapes and literals, so its arm raises exactly when every run of it would.
# Inside a data-dependent branch it raises while JAX traces that branch, even
# where the native program would not take it.
function _jax_conditional(ctx::_JaxContext, condition::_PyVal, then_build, else_build)
    ctx.host === nothing || return _host_conditional(ctx, condition, then_build, else_build)
    # A condition known while emitting (`n === nothing` on a singleton) takes
    # its arm alone: the other is not translated, as Julia never runs it.
    if condition.code in ("True", "False")
        saved = copy(ctx.locals)
        try
            return (condition.code == "True" ? then_build : else_build)()
        finally
            empty!(ctx.locals); merge!(ctx.locals, saved)
        end
    end
    traced = ctx.traced_branches
    ctx.traced_branches = condition.static ? 0 : traced + 1
    effects = _JAX_EFFECTS[]
    then_built, else_built = try
        condition.static ? (_jax_arm_function(ctx, :then_arm, then_build),
                            _jax_arm_function(ctx, :else_arm, else_build)) :
            (_jax_with_guard(() -> _jax_arm_function(ctx, :then_arm, then_build),
                             ctx.scope, condition.code),
             _jax_with_guard(() -> _jax_arm_function(ctx, :else_arm, else_build),
                             ctx.scope, "jnp.logical_not(" * condition.code * ")"))
    finally
        ctx.traced_branches = traced
    end
    A, B = then_built.value.T, else_built.value.T
    T = A === Union{} ? B : B === Union{} ? A :
        _jax_branch_type(_jax_typeof(then_built.value), _jax_typeof(else_built.value))
    # A Python conditional returns the taken arm's value as Julia does.
    condition.static && return _jax_static_branch!(ctx, condition, then_built, else_built, T)
    # A `jax.lax.cond`'s arms return the branch's type (`_jax_branch_value`).
    # An arm that throws (`scale >= 0 ? scale : throw(...)`) has no value: it
    # is its check, masked by its condition, and the branch is the other arm.
    # Where the check passes the other arm is the one Julia takes; where it
    # fails the module raises, so nothing computed there is returned. Both
    # arms' statements run in place, the check first.
    if A === Union{} || B === Union{}
        throwing, valued = A === Union{} ? (then_built, else_built) : (else_built, then_built)
        _jax_run_arm!(ctx, throwing)
        value = _jax_run_arm!(ctx, valued)
        valued.value.T === Union{} && return _PyVal("None", Union{}, true)
        return _jax_bind!(ctx, :value, _PyVal(value, T))
    end
    selected = _jax_selection(ctx, condition, then_built, else_built, T)
    selected === nothing || return selected
    then_arm = _jax_emit_arm!(ctx, then_built, T)
    else_arm = _jax_emit_arm!(ctx, else_built, T)
    branch = _PyVal("jax.lax.cond(" * condition.code * ", " * _jax_typed_arm(then_arm, T) * ", " *
                    _jax_typed_arm(else_arm, T) * ")", T)
    # An arm with a runtime check runs even when the branch value is unused.
    _JAX_EFFECTS[] == effects ? branch : _jax_bind!(ctx, :branch, branch)
end
# A static arm's value, a Python literal (`return 1.0`) or a NumPy scalar, is
# float64 under 64-bit mode, and `jax.lax.cond` refuses arms of different
# dtypes: where the branch's float dtype is another (a float32 module, a
# `Float32` program), the arm returns its value as that dtype.
function _jax_typed_arm(arm::_JaxEmittedArm, @nospecialize(T))
    dtype = arm.static ? _jax_static_arm_dtype(T) : nothing
    dtype === nothing && return arm.callable
    arm.lambda ? "lambda: jnp.asarray(" * arm.call * ", dtype=" * dtype * ")" :
        "(lambda: jnp.asarray(" * arm.call * ", dtype=" * dtype * "))"
end
# The dtype a static arm's value takes in a branch of type `T`: `nothing`
# where its Python literal already has it.
function _jax_static_arm_dtype(@nospecialize(T))
    E = _jax_eltype(T)
    E isa Type && E <: AbstractFloat && (T === E || T <: AbstractArray) || return nothing
    dtype = _jax_dtype(E)
    dtype == "jnp.float64" ? nothing : dtype
end

# A branch whose condition JAX traces and whose arms are each an atom, a
# literal or a value already bound (`s == 0 && y == m ? 1.0 : s`), selects
# between them: `jnp.where(condition, a, b)`. Both arms are valid values
# already computed, so selecting one evaluates nothing that the conditional
# would not, and its value and derivative are the conditional's: the untaken
# arm receives a zero cotangent from either form. An arm with any statement
# or computation stays lazy in `jax.lax.cond` (CONTRIBUTING, lazy branches).
function _jax_selection(ctx::_JaxContext, condition::_PyVal, then_built::_JaxArm,
                        else_built::_JaxArm, @nospecialize(T))
    T isa Type && isconcretetype(T) && T <: _JaxBranchReal || return nothing
    arms = (then_built, else_built)
    all(arm -> isempty(arm.scope.lines) && _jax_selectable(arm.value), arms) || return nothing
    values = map(arms) do arm
        outer = ctx.scope
        ctx.scope = arm.scope
        try
            _jax_branch_value(ctx, arm.value, T)
        finally
            ctx.scope = outer
        end
    end
    # A conversion is an expression, never a statement; anything else keeps
    # the conditional.
    all(arm -> isempty(arm.scope.lines), arms) || return nothing
    a, b = values
    # Two Python literals select a value of the default float dtype: give one
    # the branch's own (`_jax_typed_arm`).
    dtype = a.static && b.static ? _jax_static_arm_dtype(T) : nothing
    a_code = dtype === nothing ? a.code : "jnp.asarray(" * a.code * ", dtype=" * dtype * ")"
    _PyVal("jnp.where(" * condition.code * ", " * a_code * ", " * b.code * ")", T)
end
# An arm value `jnp.where` may select: a scalar literal or a bound name (a
# dotted constant such as `jnp.inf` too), possibly negated.
_jax_selectable(v::_PyVal) =
    isempty(v.extent) && isempty(v.limit) && v.T isa Type && v.T <: _JaxBranchReal &&
    occursin(r"^-?(?:[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*|(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)$",
             v.code) && !(v.code in ("None", "-None"))

# A conditional decided while JAX traces: Python's own, over the arms'
# statements in place. Arms without statements (which therefore cannot raise)
# are a conditional expression `(a if condition else b)`. An arm that raises
# runs in program order, even when the branch's value is unused (a guard
# `n == k || throw(...)` is `if not (n == k): raise ...`).
function _jax_static_branch!(ctx::_JaxContext, condition::_PyVal, then_built::_JaxArm,
                             else_built::_JaxArm, @nospecialize(T))
    a, b = then_built.value, else_built.value
    plain(arm) = isempty(arm.scope.lines)
    if plain(then_built) && plain(else_built) && a.T !== Union{} && b.T !== Union{}
        return _PyVal(_py_conditional(a.code, condition.code, b.code), T, a.static && b.static)
    end
    # An arm without statements whose value is unused, or that the other arm
    # leaves as the branch's only value, needs no block of its own.
    nothing_value(v) = v.T === Union{} || v.code == "None"
    valued = [arm for arm in (then_built, else_built) if !nothing_value(arm.value)]
    result = if isempty(valued)
        nothing
    elseif length(valued) == 1 && plain(only(valued)) &&
            (only(valued) === then_built ? b : a).T === Union{}
        # The other arm raises: where this one is taken, its expression is
        # the branch's value, evaluated after the check.
        only(valued).value
    else
        _PyVal(_jax_fresh(ctx, :value), T, a.static && b.static)
    end
    named = result !== nothing && !(length(valued) == 1 && result === only(valued).value)
    block(arm) = begin
        lines = copy(arm.scope.lines)
        if named && arm.value.T !== Union{}
            push!(lines, "    "^arm.scope.indent * result.code * " = " * arm.value.code)
        end
        lines
    end
    then_lines, else_lines = block(then_built), block(else_built)
    if isempty(then_lines) && isempty(else_lines)
        # Nothing runs in either arm.
    elseif isempty(then_lines)
        _jax_emit!(ctx, "if " * _py_not(condition.code) * ":")
        append!(ctx.scope.lines, else_lines)
    else
        _jax_emit!(ctx, "if " * condition.code * ":")
        append!(ctx.scope.lines, then_lines)
        if !isempty(else_lines)
            _jax_emit!(ctx, "else:")
            append!(ctx.scope.lines, else_lines)
        end
    end
    result === nothing &&
        return _PyVal("None", a.T === Union{} && b.T === Union{} ? Union{} : Nothing, true)
    result.T === T ? result : _PyVal(result.code, T, result.static, result.extent, result.limit,
                                     result.padded, result.origin, result.field_extents)
end
# Python's `a if condition else b`, parenthesized as one atom.
_py_conditional(a::AbstractString, condition::AbstractString, b::AbstractString) =
    "(" * _py_operand(a, "if") * " if " * _py_operand(condition, "if") * " else " *
    _py_operand(b, "if") * ")"
_py_operand(x::AbstractString, op::AbstractString) =
    _py_precedence(x) <= _PY_PRECEDENCE[op] ? "(" * x * ")" : x
# `not condition`: a comparison or looser operand keeps its parentheses, as a
# reader takes `not a == b` for `(not a) == b`.
_py_not(x::AbstractString) =
    "not " * (_py_precedence(x) <= _PY_PRECEDENCE["<"] ? "(" * x * ")" : x)
# `throw` in a branch decided while tracing raises then. Under conditions that
# depend on data it is `_runtime_check(false, e)` there: a check masked by
# those conditions, which raises when the module runs on values that take the
# throwing arm, as the native program throws on them.
function _jax_call_positional(ctx::_JaxContext, ::typeof(throw), args::Vector{_PyVal})
    message = _jax_exception_message(only(args))
    if ctx.traced_branches == 0
        _jax_emit!(ctx, "raise ValueError(" * message * ")")
    else
        _jax_masked_check(ctx, nothing, message)
    end
    _PyVal("None", Union{}, true)
end
_jax_exception_message(e::_PyVal) =
    e.T isa Type && e.T <: Exception && e.code != "None" ? e.code : repr(string(e.T))
# A domain check (`_runtime_check(valid, error)`), retained at execution: a
# check decided while tracing raises then; any other is a host callback that
# raises when JAX runs it, which `jax.jit`, `vmap` and `grad` keep. Under a
# data-dependent condition the check holds where the condition does not:
# JAX evaluates both arms of a vmapped condition, and a padded loop's padded
# steps, so an unmasked check would fail for lanes that never take it.
const _JAX_CHECK = """
def _rk_check(valid, message):
    # ReactiveKernels._runtime_check: raises when the program runs on values
    # that fail the check.
    def check(v):
        import numpy as np
        if not np.all(v):
            raise ValueError(message)
    # A plate checks all active lanes with one callback. Re-entering checked
    # batches outer vmap axes too; an empty axis executes no checks.
    @jax.custom_batching.custom_vmap
    def checked(v):
        jax.debug.callback(check, v)
        return ()
    @checked.def_vmap
    def checked_batch(axis_size, in_batched, v):
        if axis_size:
            checked(jnp.all(v))
        return (), ()
    checked(valid)
"""
function _jax_call_positional(ctx::_JaxContext, ::typeof(_runtime_check), args::Vector{_PyVal})
    valid, e = args
    message = _jax_exception_message(e)
    ctx.host === nothing || return _host_check(ctx, valid, message)
    if valid.static
        error_arm = () -> _jax_call(ctx, throw, [_PyVal(message, ErrorException)])
        _jax_conditional(ctx, valid, () -> _PyVal("None", Nothing), error_arm)
        return _PyVal("None", Nothing, true)
    end
    _jax_masked_check(ctx, valid, message)
    _PyVal("None", Nothing, true)
end
# A check of `valid` (`nothing`: a throw, which no value passes) where the
# code runs: it holds where any data-dependent condition the code runs under
# is false.
function _jax_masked_check(ctx::_JaxContext, valid::Union{Nothing,_PyVal}, message::AbstractString)
    guards = String[]
    for (scope, guard) in _JAX_GUARDS
        _jax_scope_within(ctx.scope, scope) || _jax_error(
            valid === nothing ? "a throw" : "a runtime check",
            " in a helper called under a condition that depends on data ",
            "has no JAX translation", _jax_where(ctx))
        push!(guards, guard)
    end
    taken = isempty(guards) ? nothing :
        foldl((a, b) -> "jnp.logical_and(" * a * ", " * b * ")", guards)
    held = if valid === nothing
        # Every traced condition records its guard, so a throw has one.
        taken === nothing && _jax_error("a throw that depends on data has no JAX translation",
                                        _jax_where(ctx))
        _jax_not(taken)
    else
        taken === nothing ? valid.code : "jnp.logical_or(" * valid.code * ", " * _jax_not(taken) * ")"
    end
    name = _jax_helper!(ctx, "_rk_check", _JAX_CHECK)
    _jax_emit!(ctx, name * "(" * held * ", " * message * ")")
    _JAX_EFFECTS[] += 1
    nothing
end
# `jnp.logical_not` of `code`, cancelling a negation `code` already is (the
# mask of an `else` arm, whose guard is the negated condition).
function _jax_not(code::AbstractString)
    prefix = "jnp.logical_not("
    if startswith(code, prefix) && endswith(code, ")")
        inner = code[nextind(code, lastindex(prefix)):prevind(code, lastindex(code))]
        _jax_balanced(inner) && return inner
    end
    prefix * code * ")"
end
# Whether every parenthesis in `code` closes within it (so `code` is one
# operand, not the tail of `a) and (b`).
function _jax_balanced(code::AbstractString)
    depth = 0
    for c in code
        depth += c == '(' ? 1 : c == ')' ? -1 : 0
        depth < 0 && return false
    end
    depth == 0
end

# Runtime checks emitted so far: a branch whose arms emit one is kept.
const _JAX_EFFECTS = Ref(0)

# The data-dependent conditions the code being emitted runs under, each with
# the scope whose code may read it (a runtime check is masked by them).
const _JAX_GUARDS = Tuple{_JaxScope,String}[]
function _jax_with_guard(f, scope::_JaxScope, guard::AbstractString)
    push!(_JAX_GUARDS, (scope, guard))
    try
        return f()
    finally
        pop!(_JAX_GUARDS)
    end
end
_jax_scope_within(s::_JaxScope, ancestor::_JaxScope) =
    s === ancestor || (s.parent !== nothing && _jax_scope_within(s.parent, ancestor))
# An exception value: its message, for `throw` and `_runtime_check`: the first
# string argument (`DomainError(x, "message")` reads its second), else a first
# argument known while tracing, else none, which names the type. A traced
# value is never the message: a check's message is a constant of the module.
function _jax_call_positional(ctx::_JaxContext, T::Type{<:Exception}, args::Vector{_PyVal})
    i = findfirst(a -> a.T isa Type && a.T <: AbstractString, args)
    message = i !== nothing ? args[i].code :
        !isempty(args) && args[1].static ? args[1].code : "None"
    _PyVal(message, T, true)
end

# String calls and interpolation share the same translation, including in
# exception messages. Do not lower `string` into Julia's printing machinery.
function _jax_call_positional(ctx::_JaxContext, ::typeof(string), parts::Vector{_PyVal})
    _PyVal("\"\".join([" * join(("str(" * p.code * ")" for p in parts), ", ") * "])",
           String, all(p -> p.static, parts))
end

# Python strings are host values, including while JAX traces a shape guard.
# Keep the iterable in the emitted program: its length never expands source.
function _jax_call_positional(ctx::_JaxContext, ::typeof(join), args::Vector{_PyVal})
    1 <= length(args) <= 3 || _jax_error("`join` expects an iterable and up to two delimiters", _jax_where(ctx))
    foreach(a -> _jax_python_value(ctx, a), args)
    delimiter = length(args) >= 2 ? args[2].code : "\"\""
    values = "(str(_rk_item) for _rk_item in " * args[1].code * ")"
    if length(args) < 3
        return _PyVal(_py_atom(delimiter) * ".join(" * values * ")", String, true)
    end
    helper = _jax_helper!(ctx, "_rk_join", raw"""
def _rk_join(items, delimiter, last):
    parts = list(items)
    if len(parts) < 2:
        return delimiter.join(parts)
    return delimiter.join(parts[:-1]) + last + parts[-1]
""")
    _PyVal(helper * "(" * values * ", " * delimiter * ", " * args[3].code * ")", String, true)
end

function _jax_expr(ctx::_JaxContext, ::Val{:generator}, ex)
    element, clauses, _, _ = _host_generator(ctx, ex; column_major = true)
    _PyVal("(" * element * " " * clauses * ")", Base.Generator, true)
end
_jax_expr(ctx::_JaxContext, ::Val{:string}, ex) =
    _jax_call_positional(ctx, string, _PyVal[_jax_expr(ctx, a) for a in ex.args])
_jax_expr(ctx::_JaxContext, ::Val{:elseif}, ex) = _jax_expr(ctx, Val(:if),
    Expr(:if, ex.args[1] isa Expr && ex.args[1].head === :block ?
         ex.args[1].args[end] : ex.args[1], ex.args[2:end]...))

_jax_typeof(v::_PyVal) = v.T isa Type ? v.T : Any

# Translate an arm's body `build` in a scope of its own.
function _jax_arm_function(ctx::_JaxContext, hint, build)
    outer = ctx.scope
    inner = _JaxScope(outer)
    inner.indent = outer.indent + 1
    saved_locals = copy(ctx.locals)
    ctx.scope = inner
    result = try
        build()
    finally
        ctx.scope = outer
        empty!(ctx.locals); merge!(ctx.locals, saved_locals)
    end
    union!(outer.used, inner.used)
    _JaxArm(Symbol(hint), inner, result)
end

# Emit an arm, returning its value as type `T` when `T` is given
# (`_jax_branch_value`): an arm with statements is a nested function, one
# without is `lambda: <expression>`, which names nothing.
function _jax_emit_arm!(ctx::_JaxContext, arm::_JaxArm, @nospecialize(T) = nothing)
    outer = ctx.scope
    result = arm.value
    if T !== nothing
        ctx.scope = arm.scope
        try
            result = _jax_branch_value(ctx, result, T)
        finally
            ctx.scope = outer
        end
        union!(outer.used, arm.scope.used)
    end
    if isempty(arm.scope.lines)
        return _JaxEmittedArm("lambda: " * result.code, result.code, true, result.T, result.static)
    end
    name = _jax_fresh(ctx, arm.hint)
    push!(outer.lines, "    "^outer.indent * "def " * name * "():")
    append!(outer.lines, arm.scope.lines)
    push!(outer.lines, "    "^arm.scope.indent * "return " * result.code)
    _JaxEmittedArm(name, name * "()", false, result.T, result.static)
end

# Run an arm unconditionally where the branch is: its statements in place,
# one level out, returning its value's code. An arm whose statements do not
# move out cleanly (a line that is not one statement at the arm's level) runs
# as its nested function, called here.
function _jax_run_arm!(ctx::_JaxContext, arm::_JaxArm)
    prefix = "    "^arm.scope.indent
    if all(l -> startswith(l, prefix) && !occursin('\n', l), arm.scope.lines)
        for line in arm.scope.lines
            push!(ctx.scope.lines, line[5:end])
        end
        return arm.value.code
    end
    emitted = _jax_emit_arm!(ctx, arm)
    arm.value.T === Union{} && _jax_emit!(ctx, emitted.call)
    emitted.call
end

# The type of a branch whose arms have types `A` and `B`: Julia's promotion,
# element by element for tuples of one length (an `if` returning its value
# with the locals it assigns), where `promote_type` would give `Real`.
function _jax_branch_type(@nospecialize(A), @nospecialize(B))
    A isa Type && B isa Type || return Any
    A <: Tuple && B <: Tuple && isconcretetype(A) && isconcretetype(B) &&
        fieldcount(A) == fieldcount(B) &&
        return Tuple{(_jax_branch_type(fieldtype(A, k), fieldtype(B, k)) for k in 1:fieldcount(A))...}
    promote_type(A, B)
end

# An arm's value `v` as the branch's type `T`. JAX requires both arms of a
# `jax.lax.cond` to return one structure of one dtype, where Julia's arms may
# differ in type (`s == 0 ? 1.0 : s` with an integer `s`). A number whose type
# is not `T` becomes an array of `T`'s dtype, and so does an integer known
# while tracing: it may be a prepared `StaticInt`, a pytree node without
# leaves where the other arm returns an array. An array converts its element
# type, and a tuple converts element by element.
function _jax_branch_value(ctx::_JaxContext, v::_PyVal, @nospecialize(T))
    _jax_branch_converts(_jax_typeof(v), v.static, T) || return v
    S = _jax_typeof(v)
    if S <: Tuple
        t = _jax_bind!(ctx, :arm_value, v)
        parts = [_jax_branch_value(ctx, _PyVal(_py_atom(t.code) * "[" * string(k - 1) * "]",
                                               fieldtype(S, k), false), fieldtype(T, k))
                 for k in 1:fieldcount(T)]
        return _PyVal(_jax_tuple_code(parts), T)
    end
    E = T <: AbstractArray ? eltype(T) : T
    _PyVal("jnp.asarray(" * v.code * ", dtype=" * _jax_dtype(E) * ")", T)
end
function _jax_branch_converts(@nospecialize(S), static::Bool, @nospecialize(T))
    isconcretetype(T) && isconcretetype(S) || return false
    if T <: Tuple
        return S <: Tuple && fieldcount(S) == fieldcount(T) &&
            any(_jax_branch_converts(fieldtype(S, k), false, fieldtype(T, k)) for k in 1:fieldcount(T))
    elseif T <: _JaxBranchReal
        return S <: _JaxBranchReal && (S !== T || static && T <: Integer && !(T <: Bool))
    elseif T <: AbstractArray
        return S <: AbstractArray && ndims(S) == ndims(T) && eltype(S) <: _JaxBranchReal &&
            isconcretetype(eltype(T)) && eltype(T) <: _JaxBranchReal && eltype(S) !== eltype(T)
    end
    false
end
# The numbers whose dtype `_jax_dtype` names.
const _JaxBranchReal = Union{Integer,AbstractFloat}

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
    f = head isa Symbol && _jax_has_local(ctx, head) ? _jax_local_callable(ctx, head) :
        _jax_resolve(ctx, head)
    # Host calls that read a list or dictionary in place (`push!`, `haskey`).
    if ctx.host !== nothing
        v = _host_call(ctx, f, positional, kwargs)
        v === nothing || return v
    end
    # A reduction over a generator, `sum(f(i) for i in r; init = x)`.
    if length(positional) == 1 && positional[1] isa Expr && positional[1].head === :generator &&
       haskey(_JAX_GENERATOR_REDUCTIONS, f)
        return _jax_generator_reduce(ctx, f, positional[1], kwargs)
    end
    # Indexing spelled as a call reads its index expressions as indexing does.
    if _jax_is_indexing(f) && length(positional) >= 2 && isempty(kwargs)
        return _jax_index(ctx, _jax_expr(ctx, positional[1]), positional[2:end])
    end
    values = _PyVal[_jax_expr(ctx, a) for a in positional]
    kw = Pair{Symbol,_PyVal}[k => _jax_expr(ctx, v) for (k, v) in kwargs]
    _jax_call(ctx, f, values, kw)
end

# A local called as a function: a function value without captures (an
# anonymous `(i, j) -> i` passed as an argument), or a type value.
function _jax_local_callable(ctx::_JaxContext, name::Symbol)
    v = ctx.locals[name]
    v.T isa _JaxCallable && return v.T.f
    v.T isa Type && v.T <: Function && Base.issingletontype(v.T) && return v.T.instance
    _jax_error("calling the local `", name, "` has no JAX translation", _jax_where(ctx))
end

_jax_is_indexing(f) = f === getindex || f === view || f === Base.maybeview

_jax_call(ctx::_JaxContext, f, args::Vector{_PyVal}) = _jax_call(ctx, f, args, Pair{Symbol,_PyVal}[])

function _jax_call(ctx::_JaxContext, f, args::Vector{_PyVal}, kw)
    isempty(kw) || _jax_error("keyword arguments to `", f, "` have no JAX translation",
                              _jax_where(ctx))
    _jax_call_positional(ctx, f, args)
end
# `sum(x; init)` and `prod(x; init)`: the reduction combined with `init`,
# which is also the value of an empty `x` (RKPPL's per-group counts).
# `dims = d` (with `sum`, `prod`, `maximum`, `minimum`) reduces axis `d` and
# keeps it, of length one, as Julia does.
const _JAX_AXIS_REDUCTIONS = Dict{Any,String}(sum => "sum", prod => "prod",
                                              maximum => "max", minimum => "min")
function _jax_call(ctx::_JaxContext,
                   f::Union{typeof(sum),typeof(prod),typeof(maximum),typeof(minimum)},
                   args::Vector{_PyVal}, kw)
    isempty(kw) && return _jax_call_positional(ctx, f, args)
    names = first.(kw)
    if names == [:dims]
        x, d = only(args), last(only(kw))
        _jax_is_array(x.T) && _jax_known(ctx, d) && d.T isa Type && d.T <: Integer &&
            (isempty(x.extent) || _jax_padded_dims_reducible(ctx, x, d)) ||
            _jax_error("`", f, "` over `dims` of a `", x.T, "`", _jax_where(ctx))
        isempty(x.extent) || return _jax_padded_dims_reduction(ctx, f, x, parse(Int, d.code))
        np = ctx.host === nothing ? "jnp." : "np."
        return _PyVal(np * _JAX_AXIS_REDUCTIONS[f] * "(" * x.code * ", axis=" *
                      _py_infix("-", [d.code, "1"]) * ", keepdims=True)", x.T)
    end
    f in (sum, prod) && names == [:init] ||
        _jax_error("keyword arguments ", join(names, ", "), " to `", f,
                   "` have no JAX translation", _jax_where(ctx))
    f === sum && length(args) == 2 && _jax_fixed_tuple_type(args[2]) !== nothing &&
        return _jax_tuple_sum(ctx, _jax_function_of(args[1]), args[2], last(only(kw)))
    reduced = _jax_call_positional(ctx, f, args)
    _jax_call(ctx, f === sum ? (+) : (*), [last(only(kw)), reduced])
end
# A padded array reduces over `dims = d` when its padded axes are known, it is
# stored with its own axes (not an adjoint vector's one) and `d` is an integer
# literal, so the emitter knows whether `d` is a padded axis.
function _jax_padded_dims_reducible(ctx::_JaxContext, x::_PyVal, d::_PyVal)
    ctx.host === nothing && _jax_padded_axes(x) != 0 &&
        (_jax_rank(x.T) == 1 || !_jax_vector_like(x.T)) || return false
    k = tryparse(Int, d.code)
    k !== nothing && 1 <= k <= _jax_rank(x.T)
end
# `f(x; dims = k)` of a padded array. Along a padded axis the padding is
# replaced by `f`'s identity, so the reduction reads the valid entries only,
# and the result is no longer padded along `k`; along another axis it keeps
# its padding, which other reductions mask.
const _JAX_REDUCTION_IDENTITIES = Dict{Any,String}(sum => "0.0", prod => "1.0",
                                                   maximum => "-jnp.inf", minimum => "jnp.inf")
function _jax_padded_dims_reduction(ctx::_JaxContext, f, x::_PyVal, k::Int)
    axes = _jax_padded_axes(x)
    bit = 1 << (k - 1)
    x = _jax_bind!(ctx, :padded, x)
    read = axes & bit == 0 ? x.code :
        "jnp.where(" * _jax_padded_mask(x, bit) * ", " * x.code * ", " *
        _JAX_REDUCTION_IDENTITIES[f] * ")"
    rest = axes & ~bit
    _PyVal("jnp." * _JAX_AXIS_REDUCTIONS[f] * "(" * read * ", axis=" * string(k - 1) *
           ", keepdims=True)", x.T, false, rest == 0 ? "" : x.extent, "", rest)
end

# `dropdims(x; dims = d)`: `x` without its axis `d`, of length one.
function _jax_call(ctx::_JaxContext, ::typeof(dropdims), args::Vector{_PyVal}, kw)
    x = only(args)
    length(kw) == 1 && first(only(kw)) === :dims ||
        _jax_error("`dropdims` without one `dims` keyword", _jax_where(ctx))
    d = last(only(kw))
    _jax_is_array(x.T) && isempty(x.extent) && !_jax_ragged(x.T) && _jax_known(ctx, d) &&
        d.T isa Type && d.T <: Integer ||
        _jax_error("`dropdims` over `dims` of a `", x.T, "`", _jax_where(ctx))
    np = ctx.host === nothing ? "jnp." : "np."
    _PyVal(np * "squeeze(" * x.code * ", axis=" * _py_infix("-", [d.code, "1"]) * ")",
           Array{eltype(x.T),ndims(x.T) - 1}, x.static)
end

# `cumsum(x; dims = d)`: the running sum along axis `d`, a static integer, as
# the axis reductions above take it.
function _jax_call(ctx::_JaxContext, ::typeof(cumsum), args::Vector{_PyVal}, kw)
    isempty(kw) && return _jax_call_positional(ctx, cumsum, args)
    first.(kw) == [:dims] || _jax_error("keyword arguments ", join(first.(kw), ", "),
        " to `cumsum` have no JAX translation", _jax_where(ctx))
    x, d = only(args), last(only(kw))
    _jax_is_array(x.T) && isempty(x.extent) && !_jax_ragged(x.T) && _jax_known(ctx, d) &&
        d.T isa Type && d.T <: Integer ||
        _jax_error("`cumsum` over `dims` of a `", x.T, "`", _jax_where(ctx))
    np = ctx.host === nothing ? "jnp." : "np."
    _PyVal(np * "cumsum(" * x.code * ", axis=" * _py_infix("-", [d.code, "1"]) * ")", x.T)
end

# `diff(x; dims = d)` along a literal axis `d`: a vector's differences
# (`_jax_function_call(::typeof(diff))`, which Base's `diff(x)` calls), or an
# unpadded array's along that axis.
function _jax_call(ctx::_JaxContext, ::typeof(diff), args::Vector{_PyVal}, kw)
    isempty(kw) && return _jax_call_positional(ctx, diff, args)
    first.(kw) == [:dims] || _jax_error("keyword arguments ", join(first.(kw), ", "),
        " to `diff` have no JAX translation", _jax_where(ctx))
    x, d = only(args), last(only(kw))
    _jax_known(ctx, d) && d.T isa Type && d.T <: Integer && occursin(r"^[0-9]+$", d.code) ||
        _jax_error("`diff` over a computed `dims`", _jax_where(ctx))
    _jax_rank(x.T) == 1 && d.code == "1" && return _jax_function_call(ctx, diff, [x])
    _jax_is_array(x.T) && isempty(x.extent) && !_jax_ragged(x.T) ||
        _jax_error("`diff` over `dims` of a `", x.T, "`", _jax_where(ctx))
    np = ctx.host === nothing ? "jnp." : "np."
    _PyVal(np * "diff(" * x.code * ", axis=" * _py_infix("-", [d.code, "1"]) * ")", x.T)
end

# `stack(rows; dims)` of a vector of equal-length vectors, held as the 2-D
# array of its rows: a matrix with those rows (`dims = 1`) or columns. On the
# host a plate of vectors is a list of arrays, which `np.stack` joins.
function _jax_call(ctx::_JaxContext, ::typeof(stack), args::Vector{_PyVal}, kw)
    x = only(args)
    T = x.T
    T isa Type && T <: AbstractVector && eltype(T) <: AbstractVector{<:Number} ||
        _jax_error("`stack` of a `", T, "`", _jax_where(ctx))
    dims = isempty(kw) ? "2" : (only(kw).first === :dims ? only(kw).second.code :
        _jax_error("the keyword `", only(kw).first, "` of `stack`", _jax_where(ctx)))
    dims in ("1", "2") || _jax_error("`stack` along dimension ", dims, _jax_where(ctx))
    ctx.host === nothing ||
        return _PyVal("np.stack(" * x.code * ", axis=" * (dims == "1" ? "0" : "1") * ")",
                      Matrix{eltype(eltype(T))}, x.static)
    x = _jax_uniform_rows(ctx, x, "stack: the vectors differ in length")
    _PyVal(dims == "1" ? x.code : "jnp.transpose(" * x.code * ")", Matrix{eltype(eltype(T))})
end
# A vector of arrays of one size, stored one array per row along the leading
# axis, as `stack` and `reduce(hcat, ·)` read it. A ragged one is checked when
# JAX runs it: every row must fill the padded width, which is the longest
# row's length. Rows of different lengths are Julia's `DimensionMismatch`;
# rows of one length shorter than that width would leave padding in the
# result, whose shape JAX fixes while tracing.
function _jax_uniform_rows(ctx::_JaxContext, x::_PyVal, message::AbstractString)
    isempty(x.extent) && return x
    rows = _jax_bind!(ctx, :rows, x)
    _jax_call(ctx, _runtime_check,
              [_PyVal("jnp.all(" * _py_atom(rows.extent) * " == " * rows.code * ".shape[1])", Bool),
               _PyVal(repr(message * " (each must fill the padded width, the longest length)"),
                      DimensionMismatch)])
    rows
end
# `reduce(hcat, columns)`: the columns side by side, as Julia's
# `reduce(hcat, ::AbstractVector{<:AbstractVecOrMat})` builds them; a vector of
# matrices gives each matrix's columns in turn. Under JAX a vector of vectors
# is stored one vector per row, so the matrix is its transpose, and a vector
# of `r×c` matrices, stored `(n, r, c)`, is `(r, n*c)`. On the host, NumPy's
# `column_stack` does the same for a list of arrays or a stored array.
function _jax_reduce_hcat(ctx::_JaxContext, x::_PyVal)
    T = x.T
    C = T isa Type && T <: AbstractVector ? eltype(T) : nothing
    C isa Type && (C <: AbstractVector || C <: AbstractMatrix) && eltype(C) <: Number ||
        _jax_error("`reduce(hcat, x)` of a `", T, "` (a vector of numeric vectors or ",
                   "matrices translates; of numbers, `permutedims(x)` is the row)", _jax_where(ctx))
    E = eltype(C)
    ctx.host === nothing ||
        return _PyVal("np.column_stack([np.asarray(c) for c in " * x.code * "])", Matrix{E}, true)
    v = _jax_bind!(ctx, :columns,
                   _jax_uniform_rows(ctx, x, "reduce(hcat, columns): the columns differ in length"))
    # No columns is Julia's error too (`reduce` over an empty collection); the
    # count is a shape, so the check raises while JAX traces.
    _jax_call(ctx, _runtime_check, [_PyVal(_py_atom(v.code) * ".shape[0] > 0", Bool, true),
        _PyVal(repr("reduce(hcat, columns): reducing over an empty collection is not allowed"),
               ArgumentError)])
    C <: AbstractVector && return _PyVal("jnp.transpose(" * v.code * ")", Matrix{E})
    _PyVal("jnp.reshape(jnp.transpose(" * v.code * ", (1, 0, 2)), (" * v.code * ".shape[1], " *
           v.code * ".shape[0] * " * v.code * ".shape[2]))", Matrix{E})
end

_jax_call_positional(ctx::_JaxContext, f, args::Vector{_PyVal}) =
    _jax_error("the function `", f, "` (", typeof(f), ") has no JAX translation",
               _jax_where(ctx), ". Give it one with a `ReactiveKernels._jax_call_positional` ",
               "method, or author it as a kernel")

# A result is static when its operands are. Under `jax.jit` a `jax.numpy` call
# is staged even on constants, so its value would be traced and could not
# decide a Python conditional: on static operands the call is computed with
# NumPy instead, as host code is (a guard `isfinite(jitter) || throw(...)` of
# a literal `jitter`). A call with no NumPy spelling stays staged.
function _jax_result(f, args::Vector{_PyVal}, code::AbstractString)
    T = _jax_return_type(f, [a.T isa Type ? a.T : Any for a in args])
    static = !isempty(args) && all(a -> a.static, args)
    if static && _jax_stages(code, args)
        host = _host_static_text(code)
        host === nothing ? (static = false) : (code = _host_float_value(host, T))
    end
    _PyVal(code, T, static, _jax_is_array(T) ? _jax_extent(args) : "")
end
# NumPy computes a static float as float64, a strongly typed value that turns
# the JAX arithmetic it meets float64: in a float32 module it is rounded to
# the module's float type (`np.float32(np.log(2 * np.pi))`).
function _host_float_value(code::AbstractString, @nospecialize(T))
    E = _jax_eltype(T)
    E isa Type && E <: AbstractFloat || return code
    dtype = _host_dtype(E)
    dtype == "np.float64" && return code
    T === E ? dtype * "(" * code * ")" : "np.asarray(" * code * ", dtype=" * dtype * ")"
end
function _jax_stages(code::AbstractString, args::Vector{_PyVal})
    operation = foldl((c, a) -> isempty(a.code) ? c : replace(c, a.code => ""), args; init = code)
    occursin(r"\b(?:jnp|jsp|jax)\.(?!(?:pi|e|inf|nan)\b)", operation)
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
    rem => "jnp.fmod", mod => "jnp.mod",
    ifelse => "jnp.where", (!) => "jnp.logical_not",
    # Both `logaddexp`s, RK's tie-exact one (mixture and log-sum-exp folds,
    # the hidden-Markov forward pass) included: see `_rk_logaddexp` below.
    LogExpFunctions.logaddexp => "_rk_logaddexp",
    _tie_exact_logaddexp => "_rk_logaddexp",
    LogExpFunctions.logistic => "jax.nn.sigmoid",
    LogExpFunctions.logit => "jsp.logit",
    LogExpFunctions.log1pexp => "jax.nn.softplus",
    LogExpFunctions.xlogy => "jsp.xlogy",
    LogExpFunctions.xlog1py => "jsp.xlog1py",
    LogExpFunctions.log1mexp => "_rk_log1mexp",
    LogExpFunctions.loglogistic => "jax.nn.log_sigmoid",
    LogExpFunctions.log1mlogistic => "_rk_log1mlogistic",
    _native_exp => "jnp.exp",
    sinpi => "_rk_sinpi", cospi => "_rk_cospi",
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
# its arity selects the name. Two-argument `hypot` is `jnp.hypot` (Base's own
# promotes mixed arguments through a tuple, which has no translation).
const _JAX_BINARY_NAMES = Dict{Any,String}(atan => "jnp.arctan2", hypot => "jnp.hypot")

# Small helpers the emitted module defines when it uses them: stable forms of
# LogExpFunctions functions whose `jax.numpy` spelling is not one call.
const _JAX_PRELUDE = Dict{String,String}(
    "_rk_log1mlogistic" => """
def _rk_log1mlogistic(x):
    # log(1 - logistic(x)) (LogExpFunctions.log1mlogistic).
    return -jax.nn.softplus(x)
""",
    # `_tie_exact_logaddexp` (src/codegen.jl) as JAX selections, so ordinary
    # `jax.grad` gives its derivatives: (1/2, 1/2) at a finite tie, and at an
    # infinite tie, whose value is `x`, (1, 0). `jnp.logaddexp` has the
    # same values, but its derivative at a tie of two `-inf`s (a hidden-Markov
    # state that no move reaches) is NaN, and reverse mode multiplies even a
    # zero cotangent by it, so one such term made every gradient entry NaN
    # (snag jax-grad-of-the-3dac142a). Both arms see finite operands at a
    # tie, so the unselected one never makes a NaN.
    "_rk_logaddexp" => """
def _rk_logaddexp(x, y):
    # log(exp(x) + exp(y)), as ReactiveKernels' tie-exact logaddexp.
    tie = jnp.isinf(x) & (x == y)
    first = x >= y
    high = jnp.where(tie, 0.0, jnp.where(first, x, y))
    low = jnp.where(tie, 0.0, jnp.where(first, y, x))
    return jnp.where(tie, x, high + jnp.log1p(jnp.exp(low - high)))
""",
    # The value of `LogExpFunctions.logsumexp`, shifted by the maximum as
    # ReactiveKernels' `_rk_logsumexp` (src/stable_rewrites.jl) is, and the
    # maximum itself when that is not finite (-inf for no entries). That
    # maximum is rebuilt from comparisons, so it has no derivative: every
    # entry -inf (a hand-written forward pass over a state no move reaches)
    # gives the derivative 0, as native Enzyme's of `logsumexp` is, where
    # `jsp.logsumexp`'s is NaN. The shift is differentiated, so at finite
    # entries its two terms cancel to the softmax. The unselected sum sees
    # no infinite exponent, so it never makes a NaN.
    "_rk_logsumexp" => """
def _rk_logsumexp(x):
    # log(sum(exp(x))), shifted by the maximum.
    top = jnp.max(x, initial=-jnp.inf)
    finite = jnp.isfinite(top)
    shift = jnp.where(finite, top, 0.0)
    total = jnp.sum(jnp.exp(jnp.where(finite, x - shift, -jnp.inf)))
    edge = jnp.where(top > 0, jnp.inf, jnp.where(top < 0, -jnp.inf, jnp.nan))
    return jnp.where(finite, shift + jnp.log(jnp.where(finite, total, 1.0)), edge)
""",
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
    "_rk_sinpi" => """
def _rk_sinpi(x):
    # sin(pi * x) (Base.sinpi), its argument reduced to [-1, 1] first, so a
    # large x (a day count) keeps full precision.
    return jnp.sin(jnp.pi * (x - 2.0 * jnp.round(x / 2.0)))
""",
    "_rk_cospi" => """
def _rk_cospi(x):
    # cos(pi * x) (Base.cospi), its argument reduced to [-1, 1] first.
    return jnp.cos(jnp.pi * (x - 2.0 * jnp.round(x / 2.0)))
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

# `fma(x, y, z)` rounds `x * y + z` once. JAX has no fused multiply-add, and
# XLA contracts `x * y + z` only sometimes (on CPU under `jax.jit`, never
# eagerly), so a floating-point `fma` calls a module helper that rounds once
# however it runs (snag jax-scalar-fma-h-ad6e34c6). Each text is the helper's
# JAX form with `BARRIER(` for `jax.lax.optimization_barrier(`, which keeps
# XLA from fusing a rounded product into a later sum, and `STOP(` for
# `jax.lax.stop_gradient(`; the host twin in plain NumPy needs neither.
# The rounded value carries no derivative. The helper subtracts a term that is
# exactly +0 and has the derivatives of `x * y + z`, so ordinary `jax.grad`
# differentiates `fma` as Julia does, (y, x, 1), with no derivative rule.
# Non-finite operands give IEEE `fma`'s value: `z` for finite `x` and `y`,
# otherwise `x * y + z`.
const _JAX_FMA_TAIL = """
    value = jnp.where(ok, value, jnp.where(finite, z, x * y + z))
    xs = jnp.where(ok, x, 0.0)
    ys = jnp.where(ok, y, 0.0)
    zs = jnp.where(ok, z, 0.0)
    return value - (((STOP(xs) - xs) * ys + STOP(xs) * (STOP(ys) - ys)) + (STOP(zs) - zs))
"""
const _JAX_FMA = Dict{DataType,Tuple{String,String}}(
    Float64 => ("_rk_fma", """
def _rk_fma(x, y, z):
    # x * y + z rounded once (Base.fma of Float64). The product of the
    # mantissas, scaled to [0.5, 1) so no partial term under- or overflows, is
    # exact (Dekker), z is added exactly, and the remaining tail is rounded to
    # odd before the one rounding to nearest (Boldo and Melquiond, IEEE Trans.
    # Comput. 57(4), 2008).
    x = jnp.asarray(x, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    z = jnp.asarray(z, dtype=jnp.float64)
    finite = jnp.isfinite(x) & jnp.isfinite(y)
    ok = finite & jnp.isfinite(z)
    a = jnp.where(ok, STOP(x), 1.0)
    b = jnp.where(ok, STOP(y), 1.0)
    c = jnp.where(ok, STOP(z), 1.0)
    ma, ea = jnp.frexp(a)
    mb, eb = jnp.frexp(b)
    mc, ec = jnp.frexp(c)
    k = ea + eb
    d = ec - k
    near = jnp.abs(d) <= 110
    # Far below the product, z only decides a tie: any small value of its sign does.
    cs = jnp.where(near, jnp.ldexp(c, jnp.where(near, -k, 0)), jnp.copysign(2.0 ** -200, c))
    split = 1.5 * 2.0 ** 26
    ah = BARRIER(ma + split) - split
    al = ma - ah
    bh = BARRIER(mb + split) - split
    bl = mb - bh
    ph = BARRIER(ma * mb)
    pl = ((ah * bh - ph) + ah * bl + al * bh) + al * bl
    th = cs + ph
    tb = th - cs
    tl = (cs - (th - tb)) + (ph - tb)
    s = tl + pl
    sb = s - tl
    se = (tl - (s - sb)) + (pl - sb)
    even = jnp.fmod(jnp.ldexp(jnp.frexp(s)[0], 53), 2.0) == 0
    v = jnp.where((se != 0) & even, jnp.nextafter(s, jnp.copysign(jnp.inf, se)), s)
    r = th + v
    # A subnormal result rounds again when scaled back. Where r lies on a
    # midpoint of that coarser grid, the exact residual's sign decides.
    rb = r - th
    re = (th - (r - rb)) + (v - rb)
    rest = jnp.where(re != 0, jnp.sign(re), jnp.where(v == s, jnp.sign(se), -jnp.sign(se)))
    tiny = jnp.frexp(r)[1] + k <= -1022
    t = jnp.ldexp(r, jnp.where(tiny, k + 1075, 0))
    tie = tiny & (rest != 0) & (t == jnp.floor(t)) & (jnp.fmod(t, 2.0) != 0)
    r = jnp.where(tie, jnp.nextafter(r, jnp.copysign(jnp.inf, rest)), r)
    value = jnp.where(d > 110, c, jnp.ldexp(r, k))
    value = jnp.where(mc == 0, a * b, value)
    value = jnp.where((ma == 0) | (mb == 0), a * b + c, value)
""" * _JAX_FMA_TAIL),
    Float32 => ("_rk_fma32", """
def _rk_fma32(x, y, z):
    # x * y + z rounded once (Base.fma of Float32): the float64 product is
    # exact, and the float64 sum rounded to odd rounds to float32 correctly.
    x = jnp.asarray(x, dtype=jnp.float32)
    y = jnp.asarray(y, dtype=jnp.float32)
    z = jnp.asarray(z, dtype=jnp.float32)
    finite = jnp.isfinite(x) & jnp.isfinite(y)
    ok = finite & jnp.isfinite(z)
    a = jnp.asarray(STOP(x), dtype=jnp.float64)
    b = jnp.asarray(STOP(y), dtype=jnp.float64)
    c = jnp.asarray(STOP(z), dtype=jnp.float64)
    p = a * b
    s = p + c
    sb = s - p
    se = (p - (s - sb)) + (c - sb)
    even = jnp.fmod(jnp.ldexp(jnp.frexp(s)[0], 53), 2.0) == 0
    odd = jnp.isfinite(s) & (se != 0) & even
    value = jnp.asarray(jnp.where(odd, jnp.nextafter(s, jnp.copysign(jnp.inf, se)), s),
                        dtype=jnp.float32)
""" * _JAX_FMA_TAIL),
)

# The `fma` helper for `T` values: its NumPy twin on static operands or in
# host preparation, so a static value stays static, otherwise its JAX form.
function _jax_fma_helper!(ctx::_JaxContext, ::Type{T}, static::Bool) where {T}
    name, text = _JAX_FMA[T]
    (static || ctx.host !== nothing) &&
        return _host_helper!(ctx, name, replace(text, "BARRIER(" => "(", "STOP(" => "("))
    _jax_helper!(ctx, name, replace(text, "BARRIER(" => "jax.lax.optimization_barrier(",
                                    "STOP(" => "jax.lax.stop_gradient("))
end

function _jax_use_prelude!(ctx::_JaxContext, name::AbstractString)
    haskey(_JAX_PRELUDE, name) || return
    _jax_helper!(ctx, name, _JAX_PRELUDE[name])
end

# Define the module-level helper `name` (Python source `text`) once.
function _jax_helper!(ctx::_JaxContext, name::AbstractString, text::AbstractString)
    ctx.host === nothing || return _host_helper!(ctx, name, text)
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
arrays when broadcast (`"jnp.exp"`), or `name => source` for a module-level helper `name`
that `source` defines, or `nothing`. A package adds a translation for its own
scalar function with a method of this function.
An ordinary array call uses `_jax_function_call`: `exp(A)` and `exp.(A)`
have different Julia semantics. See the JAX source guide's package translation
contract for the whole-call hook, result metadata and host preparation.
"""
function _jax_elementwise(f)
    name = get(_JAX_ELEMENTWISE, f, nothing)
    name === nothing && return nothing
    haskey(_JAX_PRELUDE, name) ? name => _JAX_PRELUDE[name] : name
end

"""
    _jax_elementwise_call(ctx, f, args) -> Union{Nothing,_PyVal}

The translation of the scalar function `f` called on `args` when it depends
on the arguments themselves, not only on `f` (a literal order selecting a
`jax.scipy.special` function), or `nothing`. It is applied to scalars and, in
a broadcast, elementwise to arrays, so its code must act elementwise. A
package adds one for its own function with a method of this function.
"""
_jax_elementwise_call(ctx::_JaxContext, f, args::Vector{_PyVal}) = nothing

# `x * y + z` as the source spelling translates. Julia's `muladd` may or may
# not fuse, so this is its translation, and XLA may fuse it in turn; integer
# `fma` is this too.
_jax_mul_add(ctx::_JaxContext, args::Vector{_PyVal}) =
    _jax_call(ctx, +, [_jax_call(ctx, *, args[1:2]), args[3]])
_jax_real_operands(args::Vector{_PyVal}) =
    length(args) == 3 && all(a -> a.T isa Type && a.T <: Real, args)
_jax_elementwise_call(ctx::_JaxContext, ::typeof(muladd), args::Vector{_PyVal}) =
    _jax_real_operands(args) ? _jax_mul_add(ctx, args) : nothing

# Julia's `fma` of real numbers promotes them; integer `fma` is `x * y + z`.
function _jax_elementwise_call(ctx::_JaxContext, ::typeof(fma), args::Vector{_PyVal})
    _jax_real_operands(args) || return nothing
    T = promote_type((a.T for a in args)...)
    T <: Integer && return _jax_mul_add(ctx, args)
    # A float32 module's default float is float32, so its `fma` rounds once
    # to float32 (`_rk_fma32`, which turns on 64-bit mode for its float64
    # product: `_jax_needs_x64`).
    T <: AbstractFloat && T !== Float32 && _jax_float_type() === Float32 && (T = Float32)
    haskey(_JAX_FMA, T) ||
        _jax_error("`fma` of `", T, "` values has no JAX translation (`Float64`, `Float32` ",
                   "and integer `fma` do)", _jax_where(ctx))
    name = _jax_fma_helper!(ctx, T, all(a -> a.static, args))
    _jax_result(fma, args, name * "(" * _jax_args(args) * ")")
end

function _jax_elementwise_name(ctx::_JaxContext, f, args::Vector{_PyVal})
    length(args) == 2 && haskey(_JAX_BINARY_NAMES, f) && return _JAX_BINARY_NAMES[f]
    translation = _jax_elementwise(f)
    translation isa Pair || return translation
    name, text = translation
    # On static operands the helper's NumPy twin keeps the value static, as
    # `_jax_result` does for a `jax.numpy` call.
    !isempty(args) && all(a -> a.static, args) && _host_static_text(text) !== nothing &&
        return _host_helper!(ctx, name, text)
    _jax_helper!(ctx, name, text)
end

# Arithmetic operators keep Python's infix forms.
const _JAX_INFIX = Dict{Any,String}((+) => "+", (-) => "-", (/) => "/", (<) => "<", (>) => ">",
    (<=) => "<=", (>=) => ">=", (==) => "==", (!=) => "!=", (&) => "&", (|) => "|",
    xor => "^")

# A comparison of integer literals (a rank, `size(v, 2) == 1` of a vector)
# is known while emitting, so a conditional on it takes its arm alone.
const _JAX_COMPARISONS = ((==), (!=), (<), (<=), (>), (>=))
_jax_int_literal(v::_PyVal) = occursin(r"^-?[0-9]+$", v.code)
function _jax_scalar_or_elementwise(ctx::_JaxContext, f, args::Vector{_PyVal})
    any(c -> c === f, _JAX_COMPARISONS) && length(args) == 2 && all(_jax_int_literal, args) &&
        return _jax_literal(f(parse(Int, args[1].code), parse(Int, args[2].code)))
    args = _jax_float_operands(ctx, f, args)
    any(a -> !a.static, args) && (args = map(a -> _jax_float_static(ctx, a), args))
    if haskey(_JAX_INFIX, f)
        op = _JAX_INFIX[f]
        length(args) == 1 && f === (-) &&
            return _jax_elementwise_result(f, args, _py_negate(args[1].code))
        length(args) == 1 && f === (+) && return args[1]
        length(args) >= 2 || return nothing
        f in ((+), (-)) && all(a -> _jax_is_array(a.T), args) &&
            (args = _jax_common_extent(ctx, args))
        return _jax_elementwise_result(f, args, _py_infix(op, [a.code for a in args]))
    end
    if f === (*) && any(a -> a.T isa Type && a.T <: AbstractString, args)
        return _PyVal(_py_infix("+", [a.code for a in args]), String,
                      all(a -> a.static, args))
    end
    if f === (*) && count(a -> _jax_is_array(a.T), args) <= 1
        return _jax_elementwise_result(f, args, _py_infix("*", [a.code for a in args]))
    end
    if f === (^) && length(args) == 2 && !_jax_is_array(args[1].T)
        return _jax_result(f, args, _py_infix("**", [args[1].code, args[2].code]))
    end
    v = _jax_elementwise_call(ctx, f, args)
    v === nothing || return v
    name = _jax_elementwise_name(ctx, f, args)
    name === nothing && return nothing
    _jax_result(f, args, name * "(" * _jax_args(args) * ")")
end

# A float that a float32 module computes from traced integers alone (`log(k)`,
# `i / j`, `0.5 * k`) would be JAX's default float, float64 under 64-bit
# mode: the integers are cast to the module's float first, as Julia's
# promotion makes them `Float64`, the module's float type. A float operand
# beside them (`x ^ n`) keeps them integers.
function _jax_float_operands(ctx::_JaxContext, f, args::Vector{_PyVal})
    ctx.host === nothing && _jax_float_type() !== Float64 || return args
    integer(a) = (E = _jax_eltype(a.T); E isa Type && E <: Integer && !(E <: Bool))
    any(a -> !a.static && integer(a), args) || return args
    all(a -> a.static || integer(a), args) || return args
    T = _jax_return_type(f, [a.T isa Type ? a.T : Any for a in args])
    E = _jax_eltype(T)
    E isa Type && E <: AbstractFloat || return args
    map(args) do a
        (a.static || !integer(a)) && return a
        F = a.T <: AbstractArray ? Array{Float64,ndims(a.T)} : Float64
        _PyVal("jnp.asarray(" * a.code * ", dtype=" * _jax_dtype(Float64) * ")", F, false,
               a.extent, a.limit, a.padded, nothing)
    end
end

# A static float that NumPy computed (a host helper's value, `np.log(...)`) is
# float64, a strongly typed value that turns the float32 arithmetic it meets
# float64 under 64-bit mode: where it meets a traced value in a float32
# module, it is rounded to the module's float. A Python literal is weakly
# typed and follows the traced value as it is.
# A Python literal (weakly typed), or a value already rounded to float32.
const _JAX_WEAK_OR_SINGLE = Regex("^(?:" * join((raw"-?[0-9.]+(?:[eE][-+]?[0-9]+)?",
    raw"-?jnp\.(?:inf|nan|pi|e)", raw"float\('-?(?:inf|nan)'\)", raw"np\.float32\(.*",
    raw"np\.asarray\(.*dtype=np\.float32\)"), "|") * ")\$")
function _jax_float_static(ctx::_JaxContext, a::_PyVal)
    a.static && ctx.host === nothing && _jax_float_type() !== Float64 || return a
    occursin(_JAX_WEAK_OR_SINGLE, a.code) && return a
    code = _host_float_value(a.code, a.T)
    code == a.code ? a : _PyVal(code, a.T, true, a.extent, a.limit, a.padded, a.origin, a.field_extents)
end

# A call on scalars is its elementwise form. A call with an array argument
# has Julia's array semantics (`exp(A)` is the matrix exponential), except the
# operators that act elementwise on arrays too: `+` and `-`, and `/` by a scalar.
function _jax_call_positional(ctx::_JaxContext, f::Function, args::Vector{_PyVal})
    # A type or singleton argument (`floor(Int, x)`, `rem(x, y, RoundDown)`)
    # selects a method, not an elementwise form.
    any(a -> _jax_type_of_value(a) !== nothing || _jax_is_singleton_value(a), args) &&
        return _jax_function_call(ctx, f, args)
    if !any(a -> _jax_is_array(a.T), args) ||
       ((f === (+) || f === (-)) && all(a -> _jax_is_array(a.T), args)) ||
       (f === (/) && length(args) == 2 && !_jax_is_array(args[2].T)) ||
       (f === (*) && count(a -> _jax_is_array(a.T), args) <= 1)
        v = _jax_scalar_or_elementwise(ctx, f, args)
        v === nothing || return _jax_shifted_limit(f, args, v)
    end
    _jax_history_index_result(f, args, _jax_function_call(ctx, f, args))
end

# `j ± c` of an integer `j` with a static bound and a static integer `c`
# keeps a bound (`j + c` at most the bound plus `c`, `j - c` at most the bound
# less `c`), so a loop up to it still runs to a static limit, and ranges and
# arrays it sizes pad to the width Julia's have when `j` is at its bound:
# `x[2:n] .- x[1:(n - 1)]` of a padded row's length `n` pads both alike.
function _jax_shifted_limit(f, args::Vector{_PyVal}, v::_PyVal)
    # A history index can also be a prepared integer used as a loop bound.
    # Propagate the ordinary integer origin before adding its lag proof.
    values = map(_jax_history_value, args)
    _jax_history_index_result(f, args, _jax_shifted_integer_limit(f, values, v))
end
function _jax_shifted_integer_limit(f, args::Vector{_PyVal}, v::_PyVal)
    isempty(v.limit) && !v.static || return v
    all(a -> a.T isa Type && a.T <: Integer, args) || return v
    f === (*) && return _jax_scaled_integer_limit(args, v)
    # `-j` is `-1 * j`.
    f === (-) && length(args) == 1 && return _jax_scaled_integer_limit([_jax_literal(-1), args[1]], v)
    (f === (+) || f === (-)) && length(args) == 2 || return v
    a, b = args
    entry(x) = x.origin isa _JaxOrigin && x.origin.position !== nothing
    # An entry of the prepared data shifted by a static integer keeps its
    # origin, shifted: `stops[r] - 1` is at most the largest stop less one.
    if entry(a) && b.static
        o = a.origin
        offset = o.offset == "0" && f === (+) ? b.code :
            _py_infix(f === (+) ? "+" : "-", [o.offset, b.code])
        return _jax_with_origin(v, _JaxOrigin(o.key, o.position, offset, o.path, o.scale))
    elseif entry(b) && a.static
        # `c - j` is `-1 * j + c`.
        o = f === (-) ? _jax_scaled_origin(b.origin, "-1") : b.origin
        return _jax_with_origin(v, _JaxOrigin(o.key, o.position, _jax_offset(a.code, "+", o.offset),
                                              o.path, o.scale))
    end
    if !isempty(a.limit) && b.static && (f === (+) || occursin(r"^[0-9]+$", b.code))
        limit = f === (-) ? _jax_limit_minus(a.limit, b.code) : _py_infix("+", [a.limit, b.code])
    elseif f === (+) && !isempty(b.limit) && a.static
        limit = _py_infix("+", [b.limit, a.code])
    else
        return v
    end
    _PyVal(v.code, v.T, v.static, v.extent, limit, v.padded, v.origin, v.field_extents)
end
# `k * j` of static integers `k` and one integer `j` with a static bound
# keeps a bound: an entry of the prepared data keeps its origin, scaled
# (`2 * n[i]` is at most twice the largest entry), and a nonnegative literal
# scales a static limit (`2 * length(r)` of a padded row).
function _jax_scaled_integer_limit(args::Vector{_PyVal}, v::_PyVal)
    count(a -> !a.static, args) == 1 && length(args) >= 2 || return v
    j = only(a for a in args if !a.static)
    k = foldl((x, y) -> _jax_static_integer_op("*", x, y), (a.code for a in args if a.static))
    j.origin isa _JaxOrigin && j.origin.position !== nothing &&
        return _jax_with_origin(v, _jax_scaled_origin(j.origin, k))
    !isempty(j.limit) && occursin(r"^[0-9]+$", k) || return v
    _PyVal(v.code, v.T, v.static, v.extent, _jax_scaled(k, j.limit), v.padded, v.origin, v.field_extents)
end
# `k` times an entry's origin: its scale and offset times `k`.
_jax_scaled_origin(o::_JaxOrigin, k::AbstractString) =
    _JaxOrigin(o.key, o.position, o.offset == "0" ? "0" : _jax_static_integer_op("*", k, o.offset),
               o.path, o.scale == "1" ? k : _jax_static_integer_op("*", k, o.scale))
# The static bound `limit - c` of `j - c`, a literal `c`: never below zero,
# since it sizes arrays and ranges.
function _jax_limit_minus(limit::AbstractString, c::AbstractString)
    c == "0" && return String(limit)
    occursin(r"^[0-9]+$", limit) && return string(max(parse(Int, limit) - parse(Int, c), 0))
    "max(" * _py_infix("-", [limit, c]) * ", 0)"
end

"""
    _jax_function_call(ctx::_JaxContext, f, args::Vector{_PyVal}) -> _PyVal

Translate a positional call, including one with Julia's whole-array semantics.
A package specializes on `typeof(f)` for its own function. Each argument carries
its Python expression (`code`) and Julia type (`T`); return a translated `_PyVal`,
usually from `_jax_result(f, args, expression)`, not a string or `nothing`.
The default method translates recorded/provider source or straight-line lowered
code. It is reached after any applicable scalar/elementwise mapping.

The hook also runs for host `prepare_data` (`ctx.host !== nothing`). The JAX
source guide documents host spelling, helper definitions, static operands and
ragged metadata, with the executable `examples/jax_translations.jl` example.
"""
_jax_function_call(ctx::_JaxContext, f, args::Vector{_PyVal}) =
    _jax_traceable_or_lowered(ctx, f, args)
function _jax_traceable_or_lowered(ctx::_JaxContext, f, args::Vector{_PyVal})
    # Recorded/provider source preserves the authored diagnostic even when
    # Julia infers that the helper never returns.
    v = _jax_traceable_call(ctx, f, args)
    v === nothing || return v
    # A call Julia infers never returns is an error path (an out-of-line
    # `throw_mismatch(...)`): a Python `raise`, under the same rule as `throw`.
    if all(a -> a.T isa Type, args) && _jax_return_type(f, Any[a.T for a in args]) === Union{}
        hasmethod(f, Tuple{(a.T for a in args)...}) ||
            _jax_error("the function `", f, "` has no applicable Julia method for argument types (",
                       join((string(a.T) for a in args), ", "), ")", _jax_where(ctx))
        return _jax_call(ctx, throw, [_PyVal(repr(string(nameof_or(f), " threw")), ErrorException)])
    end
    v = _jax_named_lowered_call(ctx, f, args)
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
            return _jax_result(*, args, "jnp.dot(" * _jax_masked(a, "0") * ", " *
                                        _jax_masked(b, "0") * ")")
        if ndims(a.T) == 1 && b.T <: LinearAlgebra.Adjoint{<:Any,<:AbstractVector}
            v = _jax_result(*, args, "jnp.outer(" * a.code * ", " * b.code * ")")
            # Padded as `a .* b` is: `a` along the rows, `b` along the columns.
            return _PyVal(v.code, v.T, v.static, v.extent, v.limit, _jax_padded_union(args))
        end
        return _jax_result(*, args, _py_infix("@", [a.code, b.code]))
    end
    _jax_result(*, args, _py_infix("*", [a.code, b.code]))
end
function _jax_function_call(ctx::_JaxContext, ::typeof(_native_exp), args::Vector{_PyVal})
    x = only(args)
    _jax_is_array(x.T) && ndims(x.T) == 2 ||
        _jax_error("`exp` of a `", x.T, "`", _jax_where(ctx))
    _jax_result(_native_exp, args, "jax.scipy.linalg.expm(" * x.code * ")")
end
_jax_function_call(ctx::_JaxContext, ::typeof(exp), args::Vector{_PyVal}) =
    _jax_function_call(ctx, _native_exp, args)

# RK's factorization uses the lower triangle only, even when the ignored
# upper entries are nonfinite. JAX otherwise symmetrizes its input. Both
# libraries supply their ordinary derivatives; no backend-specific rule is
# authored here. Padding is an independent identity block.
function _jax_function_call(ctx::_JaxContext, ::typeof(rk_cholesky_lower), args::Vector{_PyVal})
    A = _jax_bind!(ctx, :covariance, only(args))
    n = _jax_function_call(ctx, size, [A, _jax_literal(1)])
    m = _jax_function_call(ctx, size, [A, _jax_literal(2)])
    _jax_call(ctx, _runtime_check,
        [_jax_call(ctx, ==, [n, m]),
         _PyVal(repr("rk_cholesky_lower needs a square matrix"), ArgumentError)])
    host = _jax_known(ctx, A)
    api = host ? "np" : "jnp"
    dtype = host ? _host_dtype(Float64) : _jax_dtype(Float64)
    code = api * ".asarray(" * A.code * ", dtype=" * dtype * ")"
    extent = ""
    if ctx.host === nothing && !isempty(A.extent)
        # The two stored axes can have different capacities although their
        # valid square block has one size. Extend storage, never source code.
        width = _jax_bind!(ctx, :width, _PyVal("max(" * A.code * ".shape)", Int, true))
        code = "jnp.pad(" * code * ", ((0, " * width.code * " - " * A.code *
               ".shape[0]), (0, " * width.code * " - " * A.code * ".shape[1])))"
        mask = _jax_bind!(ctx, :valid_rows,
            _PyVal("jnp.arange(" * width.code * ") < " * n.code, Vector{Bool}))
        code = "jnp.where(" * mask.code * "[:, None] & " * mask.code *
               "[None, :], " * code * ", jnp.eye(" * width.code * ", dtype=" * dtype * "))"
        extent = n.code
    end
    code = api * ".linalg.cholesky(" * code * (host ? ")" : ", symmetrize_input=False)")
    T = _jax_return_type(rk_cholesky_lower, [A.T])
    L = _jax_bind!(ctx, :factor, _PyVal(code, T, host, extent, "", isempty(extent) ? 0 : 3))
    _jax_call(ctx, _runtime_check,
        [_PyVal(api * ".all(" * api * ".diagonal(" * L.code * ") > 0)", Bool, host),
         _PyVal(repr("rk_cholesky_lower: a pivot is not positive definite"), PosDefException)])
    L
end

function _jax_function_call(ctx::_JaxContext, ::typeof(\), args::Vector{_PyVal})
    length(args) == 2 && args[1].T isa Type && args[1].T <: LinearAlgebra.LowerTriangular &&
        _jax_rank(args[2].T) in (1, 2) || return _jax_traceable_or_lowered(ctx, \, args)
    L, b = args
    # One _PyVal extent cannot describe independently padded rows and
    # columns. A padded RHS row axis shares L's extent by the size check.
    ctx.host === nothing && !isempty(L.extent) && _jax_padded_axes(b) == 2 &&
        L.extent != b.extent && _jax_error(
            "a triangular solve with independently padded rows and columns", _jax_where(ctx))
    n = _jax_function_call(ctx, size, [L, _jax_literal(1)])
    m = _jax_function_call(ctx, size, [b, _jax_literal(1)])
    _jax_call(ctx, _runtime_check, [_jax_call(ctx, ==, [n, m]),
        _PyVal(repr("triangular solve: incompatible dimensions"), DimensionMismatch)])
    factor, rhs = L.code, b.code
    if ctx.host === nothing && (!isempty(L.extent) || !isempty(b.extent))
        L = _jax_bind!(ctx, :factor, L)
        b = _jax_bind!(ctx, :rhs, b)
        factor = _jax_masked(L, "jnp.eye(" * L.code * ".shape[0], dtype=" * L.code * ".dtype)")
        rhs = _jax_masked(b, "0")
        width = L.code * ".shape[0]"
        padding = "(0, max(0, " * width * " - " * b.code * ".shape[0]))"
        _jax_rank(b.T) == 2 && (padding = "(" * padding * ", (0, 0))")
        rhs = "jnp.pad(" * rhs * ", " * padding * ")[:" * width * "]"
    end
    # SciPy's optional full-array finite check would read an ignored upper
    # triangle; JAX already checks only the selected triangle.
    code = "jax.scipy.linalg.solve_triangular(" * factor * ", " * rhs *
           ", lower=True, check_finite=False)"
    result = _jax_result(\, args, code)
    padded = _jax_padded_axes(b) | (isempty(L.extent) ? 0 : 1)
    _PyVal(result.code, result.T, result.static, result.extent, "", padded)
end

function _jax_function_call(ctx::_JaxContext, ::typeof(LinearAlgebra.diag), args::Vector{_PyVal})
    length(args) == 1 && _jax_rank(args[1].T) == 2 ||
        return _jax_traceable_or_lowered(ctx, LinearAlgebra.diag, args)
    A = only(args)
    result = _jax_result(LinearAlgebra.diag, args, "jnp.diag(" * A.code * ")")
    extent = if ctx.host === nothing && !isempty(A.extent)
        n = _jax_function_call(ctx, size, [A, _jax_literal(1)])
        m = _jax_function_call(ctx, size, [A, _jax_literal(2)])
        n.code == m.code ? n.code : "jnp.minimum(" * n.code * ", " * m.code * ")"
    else
        ""
    end
    _PyVal(result.code, result.T, result.static, extent)
end

function _jax_function_call(ctx::_JaxContext, ::typeof(LinearAlgebra.diagm), args::Vector{_PyVal})
    length(args) == 1 && _jax_rank(args[1].T) == 1 ||
        return _jax_traceable_or_lowered(ctx, LinearAlgebra.diagm, args)
    result = _jax_result(LinearAlgebra.diagm, args, "jnp.diag(" * only(args).code * ")")
    _PyVal(result.code, result.T, result.static, result.extent, "", isempty(result.extent) ? 0 : 3)
end

# The native body's product companion computes Base's `*` (core.jl).
_jax_function_call(ctx::_JaxContext, ::typeof(_native_mul), args::Vector{_PyVal}) =
    _jax_function_call(ctx, *, args)
# A transposed padded matrix is padded along the swapped axes.
for f in (adjoint, transpose)
    @eval function _jax_function_call(ctx::_JaxContext, ::typeof($f), args::Vector{_PyVal})
        x = args[1]
        ndims(x.T) == 1 && return _jax_result($f, args, x.code)
        v = _jax_result($f, args, "jnp.transpose(" * x.code * ")")
        m = ndims(x.T) == 2 ? _jax_padded_like(x) : 0
        _PyVal(v.code, v.T, v.static, v.extent, v.limit, (m & 1) << 1 | (m & 2) >> 1)
    end
end
function _jax_function_call(ctx::_JaxContext, ::typeof(^), args::Vector{_PyVal})
    _jax_error("a matrix power has no JAX translation", _jax_where(ctx))
end
# `permutedims` reorders axes (a vector becomes a one-row matrix); a
# permutation is a static tuple of 1-based axes.
function _jax_function_call(ctx::_JaxContext, ::typeof(permutedims), args::Vector{_PyVal})
    x = args[1]
    R = _jax_rank(x.T)
    E = _jax_eltype(x.T)
    if length(args) == 1
        R == 1 && return _PyVal(_py_atom(x.code) * "[None, :]", Matrix{E})
        R == 2 && return _PyVal("jnp.transpose(" * x.code * ")", Matrix{E})
        _jax_error("`permutedims` of a rank-", R, " array without a permutation", _jax_where(ctx))
    end
    perm = args[2]
    _jax_known(ctx, perm) && perm.T isa Type && perm.T <: NTuple{R,Integer} ||
        _jax_error("`permutedims` by a permutation that is not a static tuple", _jax_where(ctx))
    axes = "tuple(a - 1 for a in " * perm.code * ")"
    _PyVal("jnp.transpose(" * x.code * ", " * axes * ")", Array{E,R})
end
# `eachrow(M)` / `eachcol(M)`: a matrix's slices, one per entry, as a plate's
# collection. The slices are the matrix's leading axis (rows) or its
# transpose's (columns), so the plate's `jax.vmap` hands each cell one slice.
function _jax_function_call(ctx::_JaxContext, f::Union{typeof(eachrow),typeof(eachcol)},
                            args::Vector{_PyVal})
    length(args) == 1 && _jax_rank(args[1].T) == 2 ||
        _jax_error("`", f, "` of a value that is not one matrix has no JAX translation",
                   _jax_where(ctx))
    x = only(args)
    code = f === eachrow ? x.code : "jnp.transpose(" * x.code * ")"
    _PyVal(code, Vector{Vector{_jax_eltype(x.T)}})
end
function _jax_function_call(ctx::_JaxContext, ::typeof(/), args::Vector{_PyVal})
    _jax_error("a matrix division has no JAX translation", _jax_where(ctx))
end

function _jax_function_call(ctx::_JaxContext, ::typeof(sum), args::Vector{_PyVal})
    # Fixed structural lag windows reduce a numeric tuple. JAX reductions
    # require an array; tuples of arrays have different Julia sum semantics.
    if length(args) == 1 && _jax_numeric_tuple(args[1].T)
        return _jax_result(sum, args, "jnp.sum(jnp.asarray(" * args[1].code * "))")
    end
    if length(args) == 1 && !isempty(args[1].extent) && _jax_rank(args[1].T) == 1
        x = args[1]
        return _PyVal("jnp.sum(jnp.where(" * _jax_mask(x) * ", " * x.code * ", 0.0))", eltype(x.T))
    end
    # A padded array of rank two or more: its valid entries, along each
    # padded axis.
    length(args) == 1 && !isempty(args[1].extent) && _jax_is_array(args[1].T) &&
        !_jax_ragged(args[1].T) &&
        return _PyVal("jnp.sum(" * _jax_padded_reduced(ctx, sum, args[1], "0.0") * ")",
                      eltype(args[1].T))
    length(args) == 1 && return _jax_result(sum, args, "jnp.sum(" * args[1].code * ")")
    length(args) == 2 && _jax_fixed_tuple_type(args[2]) !== nothing &&
        return _jax_tuple_sum(ctx, _jax_function_of(args[1]), args[2], nothing)
    length(args) == 2 && _jax_is_array(args[2].T) || _jax_error("this `sum` call", _jax_where(ctx))
    mapped = _jax_broadcast(ctx, _jax_function_of(args[1]), [args[2]])
    # `f` of a padded array's padding is not part of the sum.
    _jax_ragged(args[2].T) && return _PyVal("jnp.sum(" * _jax_masked(mapped, "0") * ")",
                                            eltype(mapped.T))
    masked = _PyVal(mapped.code, mapped.T, false, args[2].extent, "", _jax_padded_axes(args[2]))
    _PyVal("jnp.sum(" * _jax_padded_reduced(ctx, sum, masked, "0") * ")", eltype(mapped.T))
end
# The type of a tuple value whose length its type fixes (a Vararg formal's
# trailing arguments, a tuple literal), also one holding a function, or
# `nothing`.
function _jax_fixed_tuple_type(v::_PyVal)
    T = v.T isa _JaxAggregate ? v.T.T : v.T
    _jax_is_tuple(T) && T <: Tuple ? T : nothing
end
# `sum(g, t[; init])` of such a tuple: `g` of each element, added in order
# from `init`, as Julia folds a tuple. The fold follows the tuple's fixed
# length, as `afoldl` folds a fixed argument list, never a data length.
function _jax_tuple_sum(ctx::_JaxContext, g, t::_PyVal, init)
    total = init
    for k in 1:fieldcount(_jax_fixed_tuple_type(t))
        v = _jax_call(ctx, g, [_jax_tuple_element(ctx, t, _PyVal(string(k), Int, true))])
        total = total === nothing ? v : _jax_call(ctx, +, [total, v])
    end
    total === nothing && _jax_error("`sum` of an empty tuple without `init`", _jax_where(ctx))
    total
end
for (f, name, neutral) in ((prod, "jnp.prod", "1.0"), (maximum, "jnp.max", "-jnp.inf"),
                           (minimum, "jnp.min", "jnp.inf"))
    @eval function _jax_function_call(ctx::_JaxContext, ::typeof($f), args::Vector{_PyVal})
        length(args) == 1 || _jax_error("this `", $f, "` call", _jax_where(ctx))
        x = args[1]
        isempty(x.extent) || return _PyVal($name * "(" * _jax_padded_reduced(ctx, $f, x, $neutral) *
                                           ")", eltype(x.T))
        _jax_result($f, args, $name * "(" * x.code * ")")
    end
end
# `div(a, b)` of integers truncates toward zero, as Julia's does (lowered
# code reaches it as `checked_sdiv_int`): Python integers when both are
# known while tracing, else `jax.lax.div`.
function _jax_function_call(ctx::_JaxContext, ::typeof(div), args::Vector{_PyVal})
    length(args) == 2 && all(a -> a.T isa Type && a.T <: Integer, args) ||
        return _jax_traceable_or_lowered(ctx, div, args)
    a, b = args
    if _jax_known(ctx, a) && _jax_known(ctx, b)
        code = "(abs(" * a.code * ") // abs(" * b.code * ") * (1 if (" * a.code * " >= 0) == (" *
               b.code * " >= 0) else -1))"
        return _PyVal(code, Int, true)
    end
    ctx.host === nothing || _jax_error("`div` of host values that are not known", _jax_where(ctx))
    _PyVal("jax.lax.div(jnp.asarray(" * a.code * ", dtype=" * _jax_dtype(Int) * "), jnp.asarray(" *
           b.code * ", dtype=" * _jax_dtype(Int) * "))", Int)
end

# `x in t` for a scalar `x` and a tuple of scalars `t`: Python's `in` when
# both are known while tracing (a shape test `size(x, 1) in (1, K)` decides
# then), else `jnp.isin`.
function _jax_function_call(ctx::_JaxContext, ::typeof(in), args::Vector{_PyVal})
    length(args) == 2 || return _jax_traceable_or_lowered(ctx, in, args)
    x, t = args
    scalar(T) = T isa Type && (T <: Number || T <: Bool)
    t.T isa Type && t.T <: Tuple && !(t.T <: NamedTuple) && isconcretetype(t.T) &&
        scalar(x.T) && all(i -> scalar(fieldtype(t.T, i)), 1:fieldcount(t.T)) ||
        return _jax_traceable_or_lowered(ctx, in, args)
    m = match(r"^\((-?[0-9]+(?:, -?[0-9]+)*),?\)$", t.code)
    _jax_int_literal(x) && m !== nothing &&
        return _jax_literal(parse(Int, x.code) in parse.(Int, split(m.captures[1], ", ")))
    _jax_known(ctx, x) && _jax_known(ctx, t) &&
        return _PyVal("(" * x.code * " in " * t.code * ")", Bool, x.static && t.static)
    ctx.host === nothing || _jax_error("`in` of host values that are not known", _jax_where(ctx))
    _PyVal("jnp.isin(" * x.code * ", jnp.asarray(" * t.code * "))", Bool)
end

# A matrix literal of numbers (`[a b; c d]`, lowered `hvcat((2, 2), a, b, c, d)`)
# with a literal row shape.
function _jax_function_call(ctx::_JaxContext, ::typeof(hvcat), args::Vector{_PyVal})
    m = match(r"^\(([0-9]+(?:, [0-9]+)*),?\)$", args[1].code)
    xs = args[2:end]
    m !== nothing && all(a -> a.T isa Type && a.T <: Number, xs) ||
        return _jax_traceable_or_lowered(ctx, hvcat, args)
    widths = parse.(Int, split(m.captures[1], ", "))
    allequal(widths) && sum(widths) == length(xs) ||
        _jax_error("`hvcat` with rows of unequal widths", _jax_where(ctx))
    rows = [xs[sum(widths[1:i-1]; init = 0)+1:sum(widths[1:i])] for i in eachindex(widths)]
    np = ctx.host === nothing ? "jnp." : "np."
    code = np * "array([" * join(("[" * join((x.code for x in r), ", ") * "]" for r in rows), ", ") * "])"
    _PyVal(code, Matrix{promote_type((x.T for x in xs)...)}, all(x -> x.static, xs))
end

# A recipe's computation through RK's stable algebraic rewrites
# (`stable_rewrites.jl`, on unless `simplify = false`), as native preparation
# rewrites its inlined body: the same rule catalogue, over the recipe's
# expression. Each input that a single-output recipe of this plan computed is
# bound to that recipe's expression, so a composition across statements
# (`v = exp(a)` read as `log(v)`) is found as native finds it, and the
# rewrite reads the producer's operand where this scope holds it. Returns the
# rewritten expression with the names and values it reads (the producers'
# operands appended) and the module its free names resolve in, or `nothing`
# when no rule applies. The helpers a rewrite introduces translate from their
# recorded sources.
function _jax_stable_rewrite(ctx::_JaxContext, p::Plan, recipe::Recipe, inputs)
    _simplify() || return nothing
    view = _jax_stable_view(recipe)
    view === nothing && return nothing
    # A function's call over its inputs composes only with their definitions.
    _jax_stable_rewrite(ctx, p, view, recipe.inputs, inputs; alone = !(recipe.op isa Function))
end
# The same for expression `view.expr` over the values `operands` of plan `p`
# (Python values `inputs`), named `view.names`: a recipe's, or a lazy arm's
# expression over its branch's operands. `alone`: whether the expression can
# be rewritten without definitions.
function _jax_stable_rewrite(ctx::_JaxContext, p::Plan, view, operands, inputs; alone::Bool)
    params = Symbol[view.names...]
    values = _PyVal[inputs...]
    definitions = Dict{Symbol,Any}()
    for (name, v) in zip(view.names, operands)
        # A parameter the source assigns again has no single definition.
        name in view.assigned && continue
        d = _jax_stable_definition(ctx, p, v, length(definitions) + 1)
        d === nothing && continue
        definitions[name] = d[1]
        append!(params, d[2])
        append!(values, d[3])
    end
    alone || !isempty(definitions) || return nothing
    ast = Expr(:->, Expr(:tuple, params...), Expr(:block, view.expr))
    rewritten = _stable_rewrites(ast, (), definitions; parts = false)
    rewritten === ast && return nothing
    body = rewritten.args[2]
    body = length(body.args) == 1 ? only(body.args) : body
    (expr = _jax_unqualify(body, view.mod, view.assigned, view.names), names = params,
     values = values, mod = view.mod)
end

# A recipe's computation as the stable rewrites read it: its expression, the
# names it reads its inputs by (one per input, in order), the names it assigns
# itself and the module its free names resolve in; else `nothing`. A source
# operation's expression is its source, each called name qualified in that
# module, so a rule matches the function the name means there (a module's own
# `log` is not `Base.log`). A function operation's (the statement
# `total = log(v)`) is the call of that function over its inputs.
function _jax_stable_view(recipe::Recipe)
    op = recipe.op
    if op isa _KernelSourceOp
        _has_source(recipe) && recipe.source isa Expr || return nothing
        names = _recipe_source_names(recipe)
        length(names) == length(recipe.inputs) || return nothing
        return _jax_stable_view(recipe.source, names, _jax_source_module(op))
    elseif op isa Function
        names = Symbol[Symbol("#input#", i) for i in eachindex(recipe.inputs)]
        return (expr = Expr(:call, op, names...), names = names, assigned = Set{Symbol}(),
                mod = @__MODULE__)
    end
    nothing
end

# Source expression `ex` over `names` in module `mod`, as above.
function _jax_stable_view(ex, names, mod::Module)
    assigned = Set{Symbol}(keys(_rw_count!(Dict{Symbol,Int}(), ex)))
    expr = _jax_qualify_callees(ex, mod, union(Set{Symbol}(names), assigned))
    (expr = expr, names = names, assigned = assigned, mod = mod)
end

# The expression that computed `v`, when a single-output recipe of `p` with a
# view (`_jax_stable_view`) did and this scope holds that recipe's inputs: its
# input names renamed apart (the `k`-th definition), with those names and
# their values; else `nothing`. A source operation's lazy branch or split is no
# definition. An input of a nested plan (an arm's, a cell's unmapped argument)
# is the enclosing plan's value it is linked to, whose operands the nested
# Python function reads as a closure.
function _jax_stable_definition(ctx::_JaxContext, p::Plan, v::Value, k::Int)
    r = get(p.producer, canon_id(p.graph, v.id), nothing)
    if r === nothing
        link = _jax_link(ctx, p.graph, v)
        return link === nothing ? nothing : _jax_stable_definition(ctx, link[1], link[2], k)
    end
    r isa Recipe && length(r.outputs) == 1 || return nothing
    r.op isa _KernelSourceOp && !(r.op.f isa _KernelSource) && return nothing
    view = _jax_stable_view(r)
    view === nothing && return nothing
    values = _PyVal[]
    for u in r.inputs
        val = _jax_find(ctx, p.graph, u)
        val === nothing && return nothing
        push!(values, val)
    end
    fresh = Symbol[Symbol("#stable_definition#", k, "#", i) for i in eachindex(view.names)]
    _rw_substitute(view.expr, Dict{Symbol,Any}(zip(view.names, fresh))), fresh, values
end

# `ex` with each called name (`f(x)`, `f.(x)`) that `bound` does not bind
# qualified in `mod`, so the rewrite matches the function the source calls.
function _jax_qualify_callees(ex, mod::Module, bound)
    ex isa Expr || return ex
    (ex.head in _RW_OPAQUE || ex.head in _RW_CLOSURES) && return ex
    args = Any[_jax_qualify_callees(a, mod, bound) for a in ex.args]
    called = ex.head === :call && !isempty(args) ||
             ex.head === :. && length(args) == 2 && args[2] isa Expr && args[2].head === :tuple
    called && args[1] isa Symbol && !(args[1] in bound) && isdefined(mod, args[1]) &&
        (args[1] = GlobalRef(mod, args[1]))
    Expr(ex.head, args...)
end
# The rewritten source as the translation reads it: a name of `mod` that the
# source does not bind itself is spelled bare again (the translation resolves
# it in `mod`, and reads some calls, such as a range's `:`, by their spelling).
_jax_unqualify(ex, mod::Module, assigned, names) =
    _jax_unqualify(ex, mod, union(Set{Symbol}(names), assigned))
function _jax_unqualify(ex, mod::Module, bound::Set{Symbol})
    ex isa GlobalRef && return ex.mod === mod && !(ex.name in bound) ? ex.name : ex
    ex isa Expr && !(ex.head in _RW_OPAQUE || ex.head in _RW_CLOSURES) || return ex
    Expr(ex.head, Any[_jax_unqualify(a, mod, bound) for a in ex.args]...)
end
# RK's tie-exact `logsumexp` of a stable rewrite (`_rk_log_sum_of`), as
# `LogExpFunctions.logsumexp` translates.
_jax_function_call(ctx::_JaxContext, ::typeof(_rk_logsumexp), args::Vector{_PyVal}) =
    _jax_function_call(ctx, LogExpFunctions.logsumexp, args)
# `_rw_float(xs...)`: the float type of the operands' promotion, known while
# emitting.
_jax_function_call(ctx::_JaxContext, ::typeof(_rw_float), args::Vector{_PyVal}) =
    _PyVal("None", Type{float(promote_type((a.T for a in args)...))}, true)

# `merge(a, b)` of named tuples (tuples of their fields, in order): each
# field of the merged type from the last argument that has it.
function _jax_function_call(ctx::_JaxContext, ::typeof(merge), args::Vector{_PyVal})
    all(a -> a.T isa Type && a.T <: NamedTuple && isconcretetype(a.T), args) ||
        return _jax_traceable_or_lowered(ctx, merge, args)
    T = _jax_return_type(merge, Any[a.T for a in args])
    T isa Type && T <: NamedTuple && isconcretetype(T) ||
        _jax_error("`merge` of `", join((a.T for a in args), "`, `"), "`", _jax_where(ctx))
    fields = map(fieldnames(T)) do name
        a = args[findlast(a -> name in fieldnames(a.T), args)]
        _PyVal(_py_atom(a.code) * "[" * string(findfirst(==(name), fieldnames(a.T)) - 1) * "]",
               fieldtype(a.T, name))
    end
    _PyVal(_jax_tuple_code(collect(fields)), T)
end

function _jax_function_call(ctx::_JaxContext, ::typeof(vec), args::Vector{_PyVal})
    x = only(args)
    (_jax_rank(x.T) == 1 || _jax_ragged(x.T)) && return x
    _jax_is_array(x.T) || _jax_error("`vec` of a `", x.T, "`", _jax_where(ctx))
    E = eltype(x.T)
    isempty(x.extent) && return _PyVal("jnp.ravel(" * x.code * ", order=\"F\")", Vector{E})
    # An adjoint vector is stored as its vector, padded alike.
    _jax_adjoint_vector(x.T) && return _PyVal(x.code, Vector{E}, false, x.extent)
    # A padded array: its valid entries first, in Julia's column-major order
    # (a stable sort on validity), the number of them the extent, and the
    # last of them repeated as the padding, as a padded vector's is.
    _jax_padded_axes(x) == 0 && _jax_error("`vec` of a padded `", x.T,
        "` whose padded axes are not known", _jax_where(ctx))
    x = _jax_bind!(ctx, :padded, x)
    valid = _jax_fresh(ctx, :valid)
    _jax_emit!(ctx, valid * " = jnp.ravel(jnp.broadcast_to(" * _jax_padded_mask(x) * ", " *
                    x.code * ".shape), order=\"F\")")
    count = _jax_fresh(ctx, :count)
    _jax_emit!(ctx, count * " = jnp.sum(" * valid * ")")
    entries = _jax_fresh(ctx, :entries)
    _jax_emit!(ctx, entries * " = jnp.ravel(" * x.code * ", order=\"F\")[jnp.argsort(" *
                    "jnp.logical_not(" * valid * "), stable=True)]")
    _PyVal("jnp.where(jnp.arange(" * entries * ".shape[0]) < " * count * ", " * entries * ", " *
           entries * "[" * count * " - 1])", Vector{E}, false, count)
end
# Julia reshapes in column-major order. A `Colon` dimension (`reshape(v, 1, :)`)
# is the remaining length, NumPy's `-1`. The dimensions may be one tuple
# (`reshape(v, size(w))`, `reshape(v, (1, :))`).
function _jax_function_call(ctx::_JaxContext, ::typeof(reshape), args::Vector{_PyVal})
    x, dims = args[1], args[2:end]
    isempty(x.extent) || _jax_error("`reshape` of a padded (ragged) value", _jax_where(ctx))
    packed = length(dims) == 1 && dims[1].T isa Type && dims[1].T <: Tuple
    types = packed ? (isconcretetype(dims[1].T) ? fieldtypes(dims[1].T) : nothing) :
        Tuple(_jax_is_colon(d) ? Colon : d.T for d in dims)
    types !== nothing && all(T -> T === Colon || T isa Type && T <: Integer, types) ||
        _jax_error("`reshape` with dimensions of types ",
                   packed ? string(dims[1].T) : "(" * join((string(d.T) for d in dims), ", ") * ")",
                   _jax_where(ctx))
    count(T -> T === Colon, types) <= 1 ||
        _jax_error("`reshape` with more than one `Colon` dimension (Julia raises a ",
                   "`DimensionMismatch`)", _jax_where(ctx))
    # A tuple holds `:` as `None` (`_jax_value`).
    shape = !packed ? "(" * join((_jax_is_colon(d) ? "-1" : d.code for d in dims), ", ") *
                     (length(dims) == 1 ? ",)" : ")") :
        Colon in types ? "tuple(-1 if d is None else d for d in " * dims[1].code * ")" :
        dims[1].code
    _PyVal("jnp.reshape(" * x.code * ", " * shape * ", order=\"F\")",
           Array{eltype(x.T),length(types)})
end
function _jax_function_call(ctx::_JaxContext, ::typeof(LogExpFunctions.logsumexp), args::Vector{_PyVal})
    length(args) == 1 || _jax_error("this `logsumexp` call", _jax_where(ctx))
    x = args[1]
    # A traced value takes the module's `_rk_logsumexp`: `jsp.logsumexp`'s
    # derivative is NaN when every entry is -inf (see that helper). Host and
    # static values, never differentiated, keep SciPy's.
    name = !x.static && ctx.host === nothing ?
        _jax_helper!(ctx, "_rk_logsumexp", _JAX_PRELUDE["_rk_logsumexp"]) : "jsp.logsumexp"
    isempty(x.extent) || return _PyVal(name * "(" * _jax_padded_reduced(ctx, LogExpFunctions.logsumexp,
                                                                     x, "-jnp.inf") * ")", eltype(x.T))
    _jax_result(LogExpFunctions.logsumexp, args, name * "(" * x.code * ")")
end
function _jax_function_call(ctx::_JaxContext, ::typeof(length), args::Vector{_PyVal})
    x = only(args)
    ctx.host === nothing || return _host_length(ctx, x)
    _jax_columns(x.T) && return _jax_function_call(ctx, length, [_jax_column(x, 1)])
    isempty(x.extent) || _jax_ragged(x.T) ||
        return _PyVal(x.extent, Int, false, "", _py_atom(x.code) * ".shape[0]")
    _jax_ragged(x.T) && return _PyVal(_py_atom(x.code) * ".shape[0]", Int, true)
    x.T isa Type && x.T <: Tuple && return _PyVal(string(fieldcount(x.T)), Int, true)
    _PyVal(_py_atom(x.code) * ".size", Int, true)
end
# `size(x, d)` of a literal dimension, or of one that is static while JAX traces
# (an integer a helper returns, `size(U, _slice_dims(o))`); a dimension that
# is data has no static shape entry.
function _jax_function_call(ctx::_JaxContext, ::typeof(size), args::Vector{_PyVal})
    x = args[1]
    ctx.host === nothing && !isempty(x.extent) && _jax_is_array(x.T) && !_jax_ragged(x.T) &&
        return _jax_padded_size(ctx, x, args[2:end])
    shape = ctx.host === nothing ? _py_atom(args[1].code) * ".shape" : "np.shape(" * args[1].code * ")"
    length(args) == 2 || return _PyVal(shape, NTuple{_jax_rank(args[1].T),Int}, true)
    d = args[2]
    T = args[1].T
    # Past an array's rank Julia's `size` is 1 (a vector is a 1-column
    # matrix), and so is every size of a number.
    if occursin(r"^[0-9]+$", d.code)
        k = parse(Int, d.code)
        T isa Type && (T <: Number || T <: AbstractArray && k > _jax_rank(T)) &&
            return _PyVal("1", Int, true)
        return _PyVal(shape * "[" * string(k - 1) * "]", Int, true)
    end
    _jax_known(ctx, d) && d.T isa Type && d.T <: Integer ||
        _jax_error("`size` with a computed dimension", _jax_where(ctx))
    T isa Type && T <: Number && return _PyVal("1", Int, true)
    entry = shape * "[" * _py_infix("-", [d.code, "1"]) * "]"
    T isa Type && T <: AbstractArray || return _PyVal(entry, Int, true)
    _PyVal("(" * entry * " if " * _py_atom(d.code) * " <= " * string(_jax_rank(T)) *
           " else 1)", Int, true)
end
# `size` of a padded value (a ragged row in its cell, an allocation at a data
# length): along an axis its extent pads, its length, the extent, bounded by
# the stored size, as `length` gives it; along another axis, the stored size.
# The padded width is not a size Julia has, so a value whose padded axes are
# not known is refused.
function _jax_padded_size(ctx::_JaxContext, x::_PyVal, dims::Vector{_PyVal})
    axes = _jax_padded_axes(x)
    axes == 0 && _jax_error("`size` of a padded `", x.T, "` whose padded axes are not known",
                            _jax_where(ctx))
    R = _jax_rank(x.T)
    function entry(k)
        k > R && return _PyVal("1", Int, true)
        # An adjoint vector is stored as its one (second) axis.
        adjoint = _jax_adjoint_vector(x.T)
        adjoint && k == 1 && return _PyVal("1", Int, true)
        stored = _py_atom(x.code) * ".shape[" * string(adjoint ? 0 : k - 1) * "]"
        axes & (1 << (k - 1)) == 0 && return _PyVal(stored, Int, true)
        _PyVal(x.extent, Int, false, "", stored)
    end
    if isempty(dims)
        parts = _PyVal[entry(k) for k in 1:R]
        return _PyVal(_jax_tuple_code(parts), NTuple{R,Int}, false, "", _jax_tuple_limit(parts))
    end
    d = only(dims)
    occursin(r"^[0-9]+$", d.code) ||
        _jax_error("`size` of a padded value along a computed dimension", _jax_where(ctx))
    entry(parse(Int, d.code))
end
# `lastindex` (the lowered `end` of `x[a:end]` in a plain helper): the length
# or the dimension's size, static as `end` in source is. A padded (ragged)
# value's dimension size is its padding's, so that form is refused.
function _jax_function_call(ctx::_JaxContext, ::typeof(lastindex), args::Vector{_PyVal})
    length(args) == 1 && return _jax_function_call(ctx, length, args)
    isempty(args[1].extent) && !_jax_ragged(args[1].T) || _jax_error(
        "`lastindex` along a dimension of a ragged value", _jax_where(ctx))
    _jax_function_call(ctx, size, args)
end
_jax_function_call(ctx::_JaxContext, ::typeof(identity), args::Vector{_PyVal}) = only(args)
# `rem(x, y, mode)`: the remainder of the division rounded by `mode`.
function _jax_function_call(ctx::_JaxContext, ::typeof(rem), args::Vector{_PyVal})
    length(args) == 2 && return _jax_result(rem, args, "jnp.fmod(" * _jax_args(args) * ")")
    length(args) == 3 && args[3].T isa Type && args[3].T <: RoundingMode ||
        return _jax_traceable_or_lowered(ctx, rem, args)
    x, y = args[1].code, args[2].code
    code = args[3].T === RoundingMode{:Down} ? "jnp.mod(" * x * ", " * y * ")" :
        args[3].T === RoundingMode{:ToZero} ? "jnp.fmod(" * x * ", " * y * ")" :
        args[3].T === RoundingMode{:Up} ? _py_infix("-", [x, _py_infix("*", [y, "jnp.ceil(" *
            _py_infix("/", [x, y]) * ")"])]) :
        args[3].T === RoundingMode{:Nearest} ? _py_infix("-", [x, _py_infix("*", [y, "jnp.round(" *
            _py_infix("/", [x, y]) * ")"])]) :
        _jax_error("`rem` with the rounding mode `", args[3].T, "`", _jax_where(ctx))
    _jax_result(rem, args, code)
end

# Rounding to an integer type, `floor(Int, x)`.
for (f, name) in ((trunc, "jnp.trunc"), (floor, "jnp.floor"), (ceil, "jnp.ceil"),
                  (round, "jnp.round"))
    @eval function _jax_function_call(ctx::_JaxContext, ::typeof($f), args::Vector{_PyVal})
        T = length(args) == 2 ? _jax_type_of_value(args[1]) : nothing
        T isa Type && T <: Integer ||
            return _jax_traceable_or_lowered(ctx, $f, args)
        x = args[2]
        _PyVal("jnp.asarray(" * $name * "(" * x.code * "), dtype=" * _jax_dtype(T) * ")",
               _jax_is_array(x.T) ? Array{T,ndims(x.T)} : T, false, x.extent, "",
               _jax_padded_like(x))
    end
end
_jax_call_positional(ctx::_JaxContext, ::typeof(typeassert), args::Vector{_PyVal}) = args[1]
# Zero predicates over built-in real scalars are pure and total, so testing
# every entry preserves all/any's short-circuit result and observable behavior.
# Do not apply this reduction to arbitrary predicates: those may throw or have
# effects in entries Julia would not visit.
function _jax_zero_predicate_reduction(ctx::_JaxContext, f, name, args)
    length(args) == 2 || _jax_error("this `", f, "` call", _jax_where(ctx))
    predicate = _jax_function_of(args[1])
    predicate_type = _jax_callable_type(predicate)
    x = args[2]
    E = _jax_eltype(x.T)
    E in (Bool, Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64,
          Float16, Float32, Float64) &&
        (predicate_type === typeof(iszero) || predicate_type === typeof(!iszero)) ||
        _jax_error("this `", f, "` predicate call", _jax_where(ctx))
    comparison = predicate_type === typeof(iszero) ? (==) : (!=)
    mapped = _jax_broadcast(ctx, comparison, [x, _jax_literal(0)])
    code = _jax_padded_reduced(ctx, f, mapped, f === all ? "True" : "False")
    text = name * "(" * code * ")"
    x.static && (text = _host_static_text(text))
    _PyVal(text, Bool, x.static)
end
for (f, name) in ((all, "jnp.all"), (any, "jnp.any"))
    @eval _jax_function_call(ctx::_JaxContext, ::typeof($f), args::Vector{_PyVal}) =
        length(args) == 1 ? _PyVal($name * "(" * args[1].code * ")", Bool, args[1].static) :
        _jax_zero_predicate_reduction(ctx, $f, $name, args)
end
# Identity comparisons decide while tracing, on static values only.
for (f, op) in (((===), " is "), ((!==), " is not "))
    @eval function _jax_call_positional(ctx::_JaxContext, ::typeof($f), args::Vector{_PyVal})
        a, b = args
        a.T isa Type && b.T isa Type && Base.issingletontype(a.T) && Base.issingletontype(b.T) &&
            return _jax_literal($f(a.T.instance, b.T.instance))
        _jax_known(ctx, a) && _jax_known(ctx, b) ||
            _jax_error("`", $f, "` of values that depend on data", _jax_where(ctx))
        _PyVal("(" * a.code * $op * b.code * ")", Bool, true)
    end
end
function _jax_function_call(ctx::_JaxContext, ::typeof(float), args::Vector{_PyVal})
    E = float(_jax_eltype(only(args).T))
    _jax_result(float, args, "jnp.asarray(" * only(args).code * ", dtype=" * _jax_dtype(E) * ")")
end
# `convert(T, x)`: `x` itself when it is a `T` already (`convert(::Type{T}, x::T)`
# returns `x`), else, to a concrete float or integer type, the conversion `T(x)`.
function _jax_function_call(ctx::_JaxContext, ::typeof(convert), args::Vector{_PyVal})
    T = length(args) == 2 ? _jax_type_of_value(args[1]) : nothing
    x = args[end]
    T isa Type && x.T isa Type || return _jax_traceable_or_lowered(ctx, convert, args)
    x.T <: T && return x
    isconcretetype(T) && (T <: AbstractFloat || T <: Integer && T !== Bool) && x.T <: Real &&
        return _jax_call_positional(ctx, T, [x])
    _jax_traceable_or_lowered(ctx, convert, args)
end
function _jax_function_call(ctx::_JaxContext, ::typeof(LinearAlgebra.dot), args::Vector{_PyVal})
    _jax_result(LinearAlgebra.dot, args,
                "jnp.vdot(" * join((_jax_masked(a, "0") for a in args), ", ") * ")")
end
function _jax_function_call(ctx::_JaxContext, ::typeof(vcat), args::Vector{_PyVal})
    v = _jax_result(vcat, args, "jnp.concatenate([" * join(
        (_jax_is_array(a.T) ? a.code : "jnp.atleast_1d(" * a.code * ")" for a in args), ", ") * "])")
    padded = findall(a -> !isempty(a.extent), args)
    isempty(padded) && return v
    # A padded vector last (a scan's steps after its seeds): the result's
    # length is the entries before it plus its own; padding elsewhere would
    # sit between valid entries.
    padded == [length(args)] && _jax_rank(args[end].T) == 1 ||
        _jax_error("`vcat` with a padded (ragged) value before other values", _jax_where(ctx))
    before = String[_jax_is_array(a.T) ? _py_atom(a.code) * ".shape[0]" : "1" for a in args[1:end-1]]
    _PyVal(v.code, v.T, v.static, _py_infix("+", [before; args[end].extent]))
end
# `reduce(vcat, cells)`: the cells' entries in order. On the host, a list of
# arrays concatenated; under JAX an equal-length vector of vectors is a matrix,
# read row by row, and a ragged one's padded entries are dropped: its valid
# positions number its total length, the flat positions of the prepared input
# whose lengths its own are (a static shape). A runtime check confirms they
# add up.
function _jax_call_positional(ctx::_JaxContext, ::typeof(reduce), args::Vector{_PyVal})
    # The reducer is a value: its type at the program level, a callable
    # (`_JaxCallable`) inside a helper's lowered body.
    op = length(args) == 2 ? _jax_argument_type(args[1]) : nothing
    op === typeof(hcat) && return _jax_reduce_hcat(ctx, args[2])
    op === typeof(vcat) || return _jax_function_call(ctx, reduce, args)
    x = args[2]
    T = x.T
    T isa Type && T <: AbstractVector && eltype(T) isa Type && eltype(T) <: AbstractVector ||
        return _jax_function_call(ctx, reduce, args)
    E = eltype(eltype(T))
    ctx.host === nothing || return _PyVal("np.concatenate([np.atleast_1d(r) for r in " *
        x.code * "])", Vector{E}, true)
    isempty(x.extent) && return _PyVal("jnp.reshape(" * x.code * ", (-1,))", Vector{E})
    flat = _jax_flat_positions(ctx, x.extent)
    flat === nothing && _jax_error("flattening a ragged value whose lengths are not those of a ",
        "prepared ragged input (`prepare_data`) has no JAX translation", _jax_where(ctx))
    padded = _jax_bind!(ctx, :cells, x)
    lengths = _jax_bind!(ctx, :lengths, _PyVal(x.extent, Vector{Int}))
    mask = "(jnp.arange(" * padded.code * ".shape[1])[None, :] < " * lengths.code * "[:, None])"
    _jax_call(ctx, _runtime_check, [_PyVal("jnp.sum(" * lengths.code * ") == " * flat * ".shape[0]", Bool),
        _PyVal(repr("reduce(vcat, cells): the cells' lengths do not add up to their input's total"),
               ArgumentError)])
    positions = _jax_fresh(ctx, :positions)
    _jax_emit!(ctx, positions * " = jnp.nonzero(jnp.ravel(" * mask * "), size=" * flat *
                    ".shape[0])[0]")
    _PyVal("jnp.ravel(" * padded.code * ")[" * positions * "]", Vector{E})
end
# The flat positions of the prepared ragged input whose lengths `extent` reads.
function _jax_flat_positions(ctx::_JaxContext, extent::AbstractString)
    found = String[]
    for (key, flat) in ctx.scope.flat_positions
        occursin(Regex("(?<![.\\w])\\Q" * key * "\\E(?!\\w)"), extent) && push!(found, flat)
    end
    length(unique(found)) == 1 ? only(unique(found)) : nothing
end

_jax_function_call(ctx::_JaxContext, ::typeof(zero), args::Vector{_PyVal}) =
    _jax_is_array(args[1].T) ? _jax_result(zero, args, "jnp.zeros_like(" * args[1].code * ")") :
    _PyVal("0.0", args[1].T)
_jax_function_call(ctx::_JaxContext, ::typeof(one), args::Vector{_PyVal}) =
    _PyVal("1.0", args[1].T)

_jax_function_call(ctx::_JaxContext, ::typeof(_readonly_gather), args::Vector{_PyVal}) =
    _jax_index_values(ctx, args[1], args[2:end])
# Normalize a shape tuple before checking allocation sizes or deciding rank.
function _jax_dimension_values(dims::Vector{_PyVal})
    if length(dims) == 1 && dims[1].T isa Type && dims[1].T <: Tuple
        shape = only(dims)
        return [_PyVal(_py_atom(shape.code) * "[" * string(k - 1) * "]",
                       fieldtype(shape.T, k), shape.static, "",
                       isempty(shape.limit) ? "" : _py_atom(shape.limit) * "[" * string(k - 1) * "]")
                for k in 1:fieldcount(shape.T)]
    end
    dims
end
# The dimensions of an allocation, given as integers or as one tuple of them.
function _jax_dims(ctx::_JaxContext, f, dims::Vector{_PyVal})
    dims = _jax_dimension_values(dims)
    all(a -> a.T isa Type && a.T <: Integer, dims) ||
        _jax_error("`", f, "` with shape arguments of types ", Tuple(a.T for a in dims),
                   _jax_where(ctx))
    "(" * join((a.code for a in dims), ", ") * (length(dims) == 1 ? ",)" : ")"), length(dims)
end
# A vector of a data-dependent length is allocated at its static bound and
# padded, its length the value (as `Vector{T}(undef, n)`).
function _jax_vector_dims(ctx::_JaxContext, f, dims::Vector{_PyVal})
    dims = _jax_dimension_values(dims)
    length(dims) == 1 && dims[1].T isa Type && dims[1].T <: Integer || return nothing
    size, extent = _jax_allocation_size(ctx, only(dims))
    isempty(extent) ? nothing : ("(" * size * ",)", extent)
end
for (f, name) in ((ones, "jnp.ones"), (zeros, "jnp.zeros"))
    @eval function _jax_function_call(ctx::_JaxContext, ::typeof($f), args::Vector{_PyVal})
        E = isempty(args) ? nothing : _jax_type_of_value(args[1])
        E === nothing ? (E = Float64) : (args = args[2:end])
        padded = _jax_vector_dims(ctx, $f, args)
        padded === nothing ||
            return _PyVal($name * "(" * padded[1] * ", dtype=" * _jax_dtype(E) * ")", Vector{E},
                          false, padded[2])
        shape, N = _jax_dims(ctx, $f, args)
        _PyVal($name * "(" * shape * ", dtype=" * _jax_dtype(E) * ")", Array{E,N})
    end
end
function _jax_function_call(ctx::_JaxContext, ::typeof(fill), args::Vector{_PyVal})
    x, dims = args[1], args[2:end]
    E = _jax_eltype(x.T)
    padded = _jax_vector_dims(ctx, fill, dims)
    padded === nothing || return _PyVal("jnp.full(" * padded[1] * ", " * x.code * ", dtype=" *
                                        _jax_dtype(E) * ")", Vector{E}, false, padded[2])
    shape, N = _jax_dims(ctx, fill, dims)
    _PyVal("jnp.full(" * shape * ", " * x.code * ", dtype=" * _jax_dtype(E) * ")", Array{E,N})
end
# `hcat` of scalars, vectors (columns) and matrices.
function _jax_function_call(ctx::_JaxContext, ::typeof(hcat), args::Vector{_PyVal})
    columns = map(args) do a
        r = _jax_rank(a.T)
        r == 0 ? "jnp.reshape(" * a.code * ", (1, 1))" :
        r == 1 ? _py_atom(a.code) * "[:, None]" :
        r == 2 ? a.code : _jax_error("`hcat` of a rank-", r, " array", _jax_where(ctx))
    end
    E = promote_type((_jax_eltype(a.T) for a in args)...)
    _PyVal("jnp.concatenate([" * join(columns, ", ") * "], axis=1)", Matrix{E})
end

# Index with already translated index values (an integer, an integer array
# or a `Bool` mask).
function _jax_index_values(ctx::_JaxContext, x::_PyVal, indices::Vector{_PyVal})
    _jax_is_array(x.T) && length(indices) == ndims(x.T) ||
        _jax_error("a gather from `", x.T, "`", _jax_where(ctx))
    parts = String[]
    # A mask gathers at its positions, which carry the gather's extent.
    gathered = copy(indices)
    kept = 0
    for (k, i) in enumerate(indices)
        if _jax_is_colon(i)
            # A whole axis; a ragged value's padding would show through it.
            isempty(x.extent) ||
                _jax_error("a whole-axis slice of a ragged value", _jax_where(ctx))
            push!(parts, ":"); kept += 1
        elseif i.T === Bool
            _jax_bool_index_error(ctx, i.code)
        elseif _jax_logical(i.T)
            part, _, gathered[k] = _jax_mask_part(ctx, x, i)
            push!(parts, part); kept += 1
        elseif i.T isa Type && i.T <: Integer
            push!(parts, _jax_minus_one(i))
        elseif _jax_is_array(i.T) && eltype(i.T) <: Integer && ndims(i.T) == 1
            push!(parts, _py_infix("-", [i.code, "1"])); kept += 1
        else
            _jax_error("the index of type `", i.T, "`", _jax_where(ctx))
        end
    end
    E = eltype(x.T)
    code = _py_atom(x.code) * "[" * join(parts, ", ") * "]"
    T = kept == 0 ? E : Array{E,kept}
    # Rows of a ragged value keep their lengths, as `x[i]` does (`_jax_index`).
    if _jax_ragged(x.T)
        isempty(x.extent) && _jax_error("a ragged array without its lengths", _jax_where(ctx))
        return _PyVal(code, T, false, x.extent * "[" * join(parts, ", ") * "]", "", kept == 0 ? 1 : 0)
    end
    _PyVal(code, T, false, _jax_gather_extent(ctx, gathered, kept))
end

# A range value, `a:b`: the array of its elements. JAX arrays have static
# shapes, so an end that is data needs a static limit (a padded group's
# length); otherwise the range is refused here rather than emitted as a
# `jnp.arange` that JAX rejects when it traces the module.
function _jax_call_positional(ctx::_JaxContext, ::Colon, args::Vector{_PyVal})
    a, b = args[1], args[end]
    limited = !isempty(b.limit) && (length(args) == 2 || args[2].code == "1")
    unit = length(args) == 2 || args[2].code == "1"
    # Ends that are data with a static number of steps (`_jax_trip_bound`,
    # entries of the prepared data): that many elements from `a`, the ones
    # past `b` repeating `b` as padding does, and the range's length as the
    # extent that reductions and gathers mask by.
    if !(_jax_known(ctx, a) && (_jax_known(ctx, b) || limited)) && unit
        steps = _jax_trip_bound(ctx, a, b)
        if steps !== nothing
            a, b = _jax_bind!(ctx, :first, a), _jax_bind!(ctx, :last, b)
            count = "jnp.maximum(" * _py_infix("+", [_py_infix("-", [b.code, a.code]), "1"]) * ", 0)"
            return _jax_bind!(ctx, :range, _PyVal("jnp.minimum(" *
                _py_infix("+", [a.code, "jnp.arange(" * steps * ")"]) * ", " * b.code * ")",
                UnitRange{Int}, false, count))
        end
    end
    _jax_known(ctx, a) && (_jax_known(ctx, b) || limited) ||
        _jax_error("a range whose length depends on data has no JAX translation (its ",
                   "stop needs a static limit, its start and step must be static, or its ",
                   "ends entries of integer arrays of the prepared data)", _jax_where(ctx))
    if length(args) == 3
        a, step, b = args
        step.code == "1" && return _jax_call_positional(ctx, Colon(), [a, b])
        m = match(r"^-?[1-9][0-9]*$", step.code)
        m === nothing && _jax_error("a range whose step is not an integer literal",
                                    _jax_where(ctx))
        stop = parse(Int, step.code) > 0 ? _jax_plus_one(b) : _py_infix("-", [b.code, "1"])
        return _PyVal("jnp.arange(" * a.code * ", " * stop * ", " * step.code * ")",
                      StepRange{Int,Int})
    end
    a, b = args
    # A stop that is data with a static limit (a padded group's length): the
    # range runs to the limit, and its extent is the number of entries up to
    # the stop, as a padded value's (reductions and scans mask the rest). Past
    # the stop it repeats the stop, as padding repeats a row's last entry, so
    # a gather by it (`x[1:(n - 1)]`) pads with its own last entry.
    if !_jax_known(ctx, b) && !isempty(b.limit) && _jax_known(ctx, a)
        b = _jax_bind!(ctx, :last, b)
        count = _py_infix("-", [_jax_plus_one(b), a.code])
        return _PyVal("jnp.minimum(jnp.arange(" * a.code * ", " *
                      _jax_plus_one(_PyVal(b.limit, Int)) * "), " * b.code * ")",
                      UnitRange{Int}, false, count)
    end
    _jax_with_origin(_PyVal("jnp.arange(" * a.code * ", " * _jax_plus_one(b) * ")", UnitRange{Int}),
                     _JaxScanRange(a.code))
end
_jax_plus_one(b::_PyVal) = match(r"^-?[0-9]+$", b.code) === nothing ?
    _py_infix("+", [b.code, "1"]) : string(parse(Int, b.code) + 1)
_jax_function_call(ctx::_JaxContext, ::typeof(eachindex), args::Vector{_PyVal}) =
    length(args) == 1 && _jax_is_array(args[1].T) && ndims(args[1].T) == 1 ?
    _jax_axis(ctx, args[1], 1) :
    _jax_error("`eachindex` of `", args[1].T, "`", _jax_where(ctx))
function _jax_function_call(ctx::_JaxContext, ::typeof(axes), args::Vector{_PyVal})
    # On the host, `axes(x)` is the tuple of its ranges (`_host_axes`).
    ctx.host === nothing || length(args) != 1 || return _host_axes(ctx, only(args))
    length(args) == 2 || _jax_error("`axes` without a dimension", _jax_where(ctx))
    d = match(r"^[0-9]+$", args[2].code)
    d === nothing && _jax_error("`axes` with a computed dimension", _jax_where(ctx))
    _jax_axis(ctx, args[1], parse(Int, args[2].code))
end
# An axis of a padded row has the row's valid length, even though its
# storage spans the padded width. Scans and reductions must mask that tail.
function _jax_axis(ctx::_JaxContext, x::_PyVal, dim::Int)
    extent = ""
    if ctx.host === nothing && !isempty(x.extent) && !_jax_ragged(x.T)
        size = _jax_padded_size(ctx, x, [_PyVal(string(dim), Int, true)])
        isempty(size.limit) || (extent = size.code)
    end
    _PyVal("jnp.arange(1, " * _jax_dim_code(ctx, x, dim) * " + 1)",
           Base.OneTo{Int}, false, extent)
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
# `diff(x)` of a vector: the differences of neighbouring entries. A padded
# row's differences are padded too, with one valid entry fewer (none for an
# empty or one-entry row); their padding repeats the last valid difference, as
# a padded row's repeats its last entry, so a function of it stays finite
# where the differences are in its domain.
function _jax_function_call(ctx::_JaxContext, ::typeof(diff), args::Vector{_PyVal})
    x = only(args)
    _jax_is_array(x.T) && _jax_rank(x.T) == 1 && !_jax_ragged(x.T) ||
        _jax_error("`diff` of a `", x.T, "`", _jax_where(ctx))
    np = ctx.host === nothing ? "jnp." : "np."
    isempty(x.extent) && return _jax_result(diff, args, np * "diff(" * x.code * ")")
    ctx.host === nothing ||
        _jax_error("`diff` of a padded vector in `prepare_data`", _jax_where(ctx))
    x = _jax_bind!(ctx, :x, x)
    n = _jax_bind!(ctx, :n, _PyVal("jnp.maximum(" * _py_infix("-", [x.extent, "1"]) * ", 0)", Int))
    stored = _py_infix("-", [_py_atom(x.code) * ".shape[0]", "1"])
    code = "jnp.diff(" * x.code * ")[jnp.minimum(jnp.arange(" * stored * "), jnp.maximum(" *
        _py_infix("-", [n.code, "1"]) * ", 0))]"
    _PyVal(code, _jax_return_type(diff, [x.T]), false, n.code)
end

# Type-level calls run now, on the values' Julia types.
_jax_type_value(T) = _PyVal("<type>", _JaxCallable(T), true)
_jax_type_of(v::_PyVal) = v.T isa _JaxCallable ? v.T.f : v.T
_jax_call_positional(ctx::_JaxContext, ::typeof(eltype), args::Vector{_PyVal}) =
    _jax_type_value(eltype(_jax_type_of(only(args))))
_jax_call_positional(ctx::_JaxContext, ::typeof(typeof), args::Vector{_PyVal}) =
    _jax_type_value(_jax_argument_type(only(args)))
function _jax_call_positional(ctx::_JaxContext, ::typeof(isa), args::Vector{_PyVal})
    x, target = args
    S, T = _jax_argument_type(x), _jax_type_of_value(target)
    if S isa Type && T isa Type
        S <: T && return _jax_literal(true)
        # An abstract annotation is an upper bound. Only disjoint types prove
        # false; a partial overlap still needs a runtime type test.
        typeintersect(S, T) === Union{} && return _jax_literal(false)
    end
    _jax_error("`isa` cannot be decided from the Julia argument type `", S,
               "` and target type `", T, "`", _jax_where(ctx))
end
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
    _jax_known(ctx, n) && return (n.code, "")
    isempty(n.limit) || return (n.limit, n.code)
    _jax_error("an array whose length depends on data has no JAX translation",
               _jax_where(ctx))
end

# `I(n)` (a `UniformScaling` called with a size): the n×n identity, scaled.
# When `n` is data with a static bound (a ragged cell's length) the matrix is
# allocated at the bound, its rows and columns past `n` padding the extent
# marks, as a padded vector's are.
function _jax_call_positional(ctx::_JaxContext, J::LinearAlgebra.UniformScaling,
                              args::Vector{_PyVal})
    length(args) == 1 && args[1].T isa Type && args[1].T <: Integer ||
        _jax_error("`I` called with `", join((a.T for a in args), ", "), "`", _jax_where(ctx))
    size, extent = _jax_allocation_size(ctx, only(args))
    np = ctx.host === nothing ? "jnp." : "np."
    # Its entries are the default float, the module's float type in a float32 one.
    dtype = _jax_float_type() === Float64 ? "" :
        ", dtype=" * (ctx.host === nothing ? _jax_dtype(Float64) : _host_dtype(Float64))
    code = np * "eye(" * size * dtype * ")"
    J.λ == true || (code = _py_infix("*", [_jax_literal(J.λ).code, code]))
    _PyVal(code, typeof(J(1)), false, extent, "", isempty(extent) ? 0 : 3)
end

# Fixed arguments are captured runtime values, just as in anonymous closures.
function _jax_call_positional(ctx::_JaxContext, T::Union{Type{Base.Fix1},Type{Base.Fix2}},
                              args::Vector{_PyVal})
    length(args) == 2 || return _jax_error("constructing `", T, "` requires two arguments",
                                         _jax_where(ctx))
    # Source passes singleton functions by their type; lowered calls carry a
    # callable. Normalize that field before Base's Fix1/Fix2 body reads it.
    captures = [_PyVal("<callable>", _JaxCallable(_jax_function_of(args[1]))), args[2]]
    F = T{_jax_argument_type(args[1]),_jax_argument_type(args[2])}
    _PyVal("<closure>", _JaxCallable(_JaxClosure(F, captures)), false)
end

# A Julia type called as a function: a conversion.
function _jax_call_positional(ctx::_JaxContext, T::Type, args::Vector{_PyVal})
    isempty(args) && Base.issingletontype(T) && return _PyVal("None", T)
    if (T === Ref || T === Base.RefValue) && length(args) == 1
        x = _jax_bind!(ctx, :referenced, only(args))
        R = Base.RefValue{_jax_argument_type(x)}
        return _PyVal("<ref>", _JaxAggregate(R, [x]), x.static)
    end
    # `Val(:name)`, `Val(2)`: the singleton of a parameter known while emitting.
    T === Val && length(args) == 1 && return _PyVal("None", Val{_jax_static_parameter(ctx, only(args))})
    # A keyword call's arguments in lowered code, `NamedTuple{names}((a, b))`:
    # the tuple's value under those names.
    if T <: NamedTuple && length(args) == 1 && _jax_is_tuple(args[1].T) &&
       !(args[1].T <: NamedTuple)
        names = Base.unwrap_unionall(T).parameters[1]
        names isa Tuple && fieldcount(args[1].T) == length(names) &&
            return _PyVal(args[1].code, NamedTuple{names,args[1].T}, args[1].static)
    end
    if ctx.host !== nothing
        v = _host_construct(ctx, T, args)
        v === nothing || return v
    end
    # `Diagonal(v)` of a vector: the dense diagonal matrix (padded alike).
    if T === LinearAlgebra.Diagonal && length(args) == 1 && args[1].T isa Type &&
       args[1].T <: AbstractVector{<:Number}
        v = only(args)
        return _PyVal("jnp.diag(" * v.code * ")", LinearAlgebra.Diagonal{eltype(v.T),v.T}, false,
                      v.extent, "", isempty(v.extent) ? 0 : 3)
    end
    if T <: AbstractArray && !isempty(args) && args[1].T === UndefInitializer
        dims = _jax_dimension_values(args[2:end])
        E = eltype(T)
        E isa TypeVar && (E = Float64)
        dtype = _jax_dtype(E)
        padded = _jax_vector_dims(ctx, T, dims)
        padded === nothing ||
            return _PyVal("jnp.zeros(" * padded[1] * ", dtype=" * dtype * ")", Vector{E}, false, padded[2])
        shape, N = _jax_dims(ctx, T, dims)
        # JAX requires concrete dimensions. A vector's bounded length is handled
        # above; independently padded matrix axes need richer extent metadata.
        all(d -> _jax_known(ctx, d), dims) ||
            _jax_error("a multidimensional allocation whose shape depends on data has no JAX translation",
                       _jax_where(ctx))
        return _PyVal("jnp.zeros(" * shape * ", dtype=" * dtype * ")", Array{E,N})
    end
    if T <: AbstractArray && length(args) == 1 && _jax_is_array(args[1].T)
        E = T isa UnionAll ? eltype(Base.unwrap_unionall(T)) : eltype(T)
        E isa TypeVar && (E = eltype(args[1].T))
        dtype = _jax_dtype(E)
        return _PyVal("jnp.asarray(" * args[1].code * ", dtype=" * dtype * ")",
                      Array{E,ndims(args[1].T)})
    end
    if T <: AbstractFloat && length(args) == 1
        return _PyVal("jnp.asarray(" * args[1].code * ", dtype=" * _jax_dtype(T) * ")", T)
    end
    if T <: Integer && length(args) == 1
        return _PyVal("jnp.asarray(" * args[1].code * ", dtype=" * _jax_dtype(Int) * ")", T)
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
    referenced = [a.T isa _JaxAggregate && a.T.T <: Ref for a in args]
    values = [ref ? only(a.T.fields) : a for (a, ref) in zip(args, referenced)]
    # A numeric type broadcast over a value converts it, `Float64.(x)`.
    if f isa Type && f <: Number && length(args) == 1 && !any(referenced)
        x = only(args)
        return _PyVal("jnp.asarray(" * x.code * ", dtype=" * _jax_dtype(f) * ")",
                      R == 0 ? f : Array{f,R}, false, x.extent, "", _jax_padded_like(x))
    end
    # The result's extent and padded axes: rows padded by different lengths
    # along the same axes take Julia's broadcast length (`_jax_common_extent`).
    meta = _jax_common_extent(ctx, args)
    padded = map(values, referenced) do a, ref
        r = _jax_rank(a.T)
        !ref && 0 < r < R ? _PyVal(_py_atom(a.code) * "[" * join(fill("...", 1), "") *
                           repeat(", None", R - r) * "]", a.T) : a
    end
    eltypes = Any[ref ? _jax_argument_type(a) : _jax_eltype(_jax_argument_type(a))
                 for (a, ref) in zip(values, referenced)]
    E = _jax_return_type(f, eltypes)
    T = R == 0 ? E : Array{E,R}
    scalars = [ref || _jax_translation_only(p) ? p : _PyVal(p.code, eltype_or_self(p.T))
               for (p, ref) in zip(padded, referenced)]
    # Ref arguments are whole scalar objects even when their payload is an
    # array. A captured callable and these objects live in the enclosing
    # Python scope, so their values and padding metadata are not lost in a
    # module-level helper or accidentally vectorized by NumPy/JAX.
    if any(referenced) || f isa _JaxClosure || any(_jax_translation_only, scalars)
        R == 0 && return _jax_call(ctx, f, scalars)
        code, E = _jax_captured_broadcast(ctx, f, scalars, padded, referenced)
        return _PyVal(code, Array{E,R}, false, _jax_extent(meta), "",
                      _jax_broadcast_padded(meta, R))
    end
    v = _jax_elementwise_call(ctx, f, scalars)
    v === nothing || return _PyVal(v.code, T, false, R == 0 ? "" : _jax_extent(meta), "",
                                   _jax_broadcast_padded(meta, R))
    if haskey(_JAX_INFIX, f) || f === (*) || f === (^) ||
       _jax_elementwise_name(ctx, f, scalars) !== nothing
        any(a -> !a.static, padded) && (padded = map(a -> _jax_float_static(ctx, a), padded))
        v = f === (*) ? _PyVal(_py_infix("*", [p.code for p in padded]), E) :
            f === (^) ? _PyVal(_py_infix("**", [padded[1].code, padded[2].code]), E) :
            _jax_scalar_or_elementwise(ctx, f, scalars)
        v === nothing && _jax_error("broadcasting `", f, "`", _jax_where(ctx))
        return _PyVal(v.code, T, false, R == 0 ? "" : _jax_extent(meta), "",
                      _jax_broadcast_padded(meta, R))
    end
    # A function of each vector of a padded vector of vectors (`Diagonal.(rows)`
    # of a ragged plate's per-cell arguments) maps over the rows, each a padded
    # vector of its length, as a plate maps its cells: Julia applies `f` to
    # each row, not to each number.
    if length(args) == 1 && ctx.host === nothing && _jax_ragged(only(args).T) &&
       ndims(only(args).T) == 1 && eltype(only(args).T) <: AbstractVector{<:Number} &&
       !isempty(only(args).extent)
        x = only(args)
        outer = ctx.scope
        fname = _jax_fresh(ctx, :row_function)
        inner = _JaxScope(outer)
        inner.indent = outer.indent + 1
        row = _jax_identifier(inner.used, :row)
        n = _jax_identifier(inner.used, :row_length)
        saved = copy(ctx.locals)
        ctx.scope = inner
        v = try
            _jax_call(ctx, f, [_PyVal(row, eltype(x.T), false, n)])
        finally
            ctx.scope = outer
            empty!(ctx.locals); merge!(ctx.locals, saved)
        end
        push!(outer.lines, "    "^outer.indent * "def " * fname * "(" * row * ", " * n * "):")
        append!(outer.lines, inner.lines)
        push!(outer.lines, "    "^inner.indent * "return " * v.code)
        return _PyVal("jax.vmap(" * fname * ")(" * x.code * ", " * x.extent * ")", Vector{v.T},
                      false, x.extent)
    end
    # Any other scalar function: a module-level function translated once over
    # scalar arguments, vectorized over the operands.
    name = _jax_scalar_function!(ctx, f, eltypes)
    # NumPy's `vectorize` needs the result type for empty arrays.
    ctx.host === nothing || return _PyVal("np.vectorize(" * name * ", otypes=[" *
        _host_dtype(E) * "])(" * _jax_args(padded) * ")", T)
    _PyVal("jnp.vectorize(" * name * ")(" * _jax_args(padded) * ")", T, false,
           R == 0 ? "" : _jax_extent(meta), "", _jax_broadcast_padded(meta, R))
end

function _jax_captured_broadcast(ctx::_JaxContext, f, scalars, padded, referenced)
    excluded = referenced .| _jax_translation_only.(scalars)
    outer = ctx.scope
    inner = _JaxScope(outer)
    inner.indent = outer.indent + 1
    name = _jax_fresh(ctx, :elementwise)
    params = [_jax_identifier(inner.used, Symbol(:element, k)) for k in findall(!, excluded)]
    inputs = copy(scalars)
    for (k, param) in zip(findall(!, excluded), params)
        inputs[k] = _PyVal(param, scalars[k].T, ctx.host !== nothing)
    end
    saved = copy(ctx.locals)
    ctx.scope = inner
    v = try
        _jax_call(ctx, f, inputs)
    finally
        ctx.scope = outer
        empty!(ctx.locals); merge!(ctx.locals, saved)
    end
    push!(outer.lines, "    "^outer.indent * "def " * name * "(" * join(params, ", ") * "):")
    append!(outer.lines, inner.lines)
    push!(outer.lines, "    "^inner.indent * "return " * v.code)
    union!(outer.used, inner.used)
    vectorize = ctx.host === nothing ? "jnp.vectorize(" * name * ")" :
        "np.vectorize(" * name * ", otypes=[" * _host_dtype(v.T) * "])"
    vectorize * "(" * _jax_args(padded[.!excluded]) * ")", v.T
end

# A dotted literal power in lowered code, `broadcasted(literal_pow, ^, x,
# Val(n))` (`x .^ 2` in a plain helper): the elementwise power, as
# `literal_pow` of a scalar is.
function _jax_broadcast(ctx::_JaxContext, f::typeof(Base.literal_pow), args::Vector{_PyVal})
    length(args) == 3 && args[3].T isa Type && args[3].T <: Val ||
        _jax_error("broadcasting `", f, "`", _jax_where(ctx))
    x, n = args[2], args[3].T.parameters[1]
    R = _jax_rank(x.T)
    E = _jax_return_type(^, Any[_jax_eltype(x.T), typeof(n)])
    _PyVal(_py_infix("**", [x.code, string(n)]), R == 0 ? E : Array{E,R}, false,
           R == 0 ? "" : x.extent, "", _jax_padded_like(x))
end

# A module-level Python function computing `f` on scalars of `types`.
function _jax_scalar_function!(ctx::_JaxContext, f, types)
    host = ctx.host !== nothing
    key = (:scalar, objectid(f), Tuple(types), host)
    haskey(ctx.mod.helper_names, key) && return ctx.mod.helper_names[key]
    named = _jax_vectorized_named_function!(ctx, f, types)
    named === nothing || return (ctx.mod.helper_names[key] = named)
    hint = Symbol("_rk_", replace(string(nameof_or(f)), r"[^A-Za-z0-9_]" => "_"), host ? "_np" : "")
    name = _jax_private_identifier(ctx.mod.used, hint)
    ctx.mod.helper_names[key] = name
    scope = _JaxScope()
    params = [_jax_identifier(scope.used, Symbol(:x, i)) for i in eachindex(types)]
    inner = _JaxContext(ctx.mod, scope, ctx.recipe, ctx.source_module,
                        Dict{Symbol,_PyVal}(), nothing)
    host && (inner.host = _HostState(Set{String}(params)))
    v = _jax_call(inner, f, [_PyVal(p, T) for (p, T) in zip(params, types)])
    lines = String["def " * name * "(" * join(params, ", ") * "):"]
    append!(lines, scope.lines)
    push!(lines, "    return " * v.code)
    push!(ctx.mod.helpers, host ? _host_helper_text(ctx, join(lines, "\n")) : join(lines, "\n"))
    name
end
nameof_or(f) = f isa Function && Base.issingletontype(typeof(f)) ? nameof(f) :
    f isa Union{ScalarDerivativeRule,DerivativeRule} ? typeof(f).parameters[1] : :function
eltype_or_self(@nospecialize(T)) = T isa Type && T <: AbstractArray ? eltype(T) : T


# A source operation called directly (an embedded kernel's operation spliced
# into another recipe's source, or one step of a derivative rule): its
# runtime-generated body, whose globals are qualified, over the arguments,
# with the operation's module for the names the body leaves to it.
_jax_call_positional(ctx::_JaxContext, op::_KernelSourceOp, args::Vector{_PyVal}) =
    _jax_source_call(ctx, op.f, args, _kernel_endpoint_source_module(op))
# A source record called without its operation: a lowering's callables table
# holds the record, not the `_KernelSourceOp` around it (`_lowering_tables`),
# so a derivative rule's primal cut calls its records directly.
_jax_call_positional(ctx::_JaxContext,
                     f::Union{_KernelSource,_KernelSplit,_KernelReduction,_KernelBranch},
                     args::Vector{_PyVal}) =
    _jax_source_call(ctx, f, args, _jax_record_module(f))
# The module whose names a record's generated body leaves unqualified: the
# context its companion was compiled in, else the module of its authored code.
_jax_record_module(f::Union{_KernelSplit,_KernelReduction}) = _jax_record_module(f.call)
_jax_record_module(f::_KernelSource) = _jax_companion_module(f.companion, f)
_jax_record_module(f) = _kernel_endpoint_source_module(f)
_jax_companion_module(c::_PrecompileWarmFunction, f) = _jax_companion_module(c.f, f)
_jax_companion_module(::RuntimeGeneratedFunctions.RuntimeGeneratedFunction{A,C,M},
                      f) where {A,C,M} = parentmodule(M)
_jax_companion_module(_, f) = _kernel_endpoint_source_module(f)
function _jax_source_call(ctx::_JaxContext,
        f::Union{_KernelSource,_KernelSplit,_KernelReduction}, args::Vector{_PyVal}, mod)
    params, body, captures = _jax_op_body(f)
    length(params) == length(args) ||
        _jax_error("an embedded operation takes ", length(params), " arguments, not ",
                   length(args), _jax_where(ctx))
    v = _jax_source_expr(ctx, body, params, args, mod, captures)
    _jax_bind!(ctx, :value, v)
end
# A lazy branch called as a value (a cell's branch spliced into its plate's
# body): its condition over the arguments at `CI`, and an arm over those at
# `TI` or `EI`, as a conditional that keeps the arms lazy.
function _jax_source_call(ctx::_JaxContext, b::_KernelBranch{CI,TI,EI},
                          args::Vector{_PyVal}, mod) where {CI,TI,EI}
    condition = _jax_source_call(ctx, b.condition, args[collect(Int, CI)], mod)
    v = _jax_conditional(ctx, condition,
                         () -> _jax_source_call(ctx, b.then_arm, args[collect(Int, TI)], mod),
                         () -> _jax_source_call(ctx, b.else_arm, args[collect(Int, EI)], mod))
    _jax_bind!(ctx, :value, v)
end
_jax_source_call(ctx::_JaxContext, f, args::Vector{_PyVal}, mod) = _jax_call(ctx, f, args)

# The parameters, body and captures of a source function: its
# runtime-generated body when it has one, else its source. The generated
# body's first parameter is the captures environment, which a leading `let`
# unpacks (`name = Core.getfield(environment, :c)`, `_kernel_source_function`);
# the translation reads each such name as that capture instead.
function _jax_op_body(f::_KernelSource)
    f.companion === nothing &&
        _jax_error("an embedded operation whose source Julia owns has no generated body")
    ex = _program_expression(f.companion)
    ex isa Expr && ex.head === :-> || _jax_error("an unexpected generated body `", ex, "`")
    params = ex.args[1] isa Expr ? Symbol[a for a in ex.args[1].args] : Symbol[ex.args[1]]
    body, captures = _jax_unpack_captures(ex.args[2], first(params), f.captures)
    params[2:end], body, captures
end
_jax_op_body(f::Union{_KernelSplit,_KernelReduction}) = _jax_op_body(f.call)
function _jax_unpack_captures(body, environment::Symbol, captures)
    inner = body
    while inner isa Expr && inner.head === :block &&
          count(a -> !(a isa LineNumberNode), inner.args) == 1
        inner = only(a for a in inner.args if !(a isa LineNumberNode))
    end
    inner isa Expr && inner.head === :let || return body, captures
    bindings = inner.args[1]
    list = bindings isa Expr && bindings.head === :block ?
        Any[b for b in bindings.args if !(b isa LineNumberNode)] : Any[bindings]
    reads = map(list) do b
        b isa Expr && b.head === :(=) && b.args[1] isa Symbol || return nothing
        r = b.args[2]
        Meta.isexpr(r, :call, 3) && r.args[2] === environment && r.args[3] isa QuoteNode &&
            (r.args[1] === getfield || r.args[1] == GlobalRef(Core, :getfield) ||
             r.args[1] === :getfield) || return nothing
        b.args[1] => r.args[3].value::Symbol
    end
    any(isnothing, reads) && return body, captures
    names = Tuple(first.(reads))
    inner.args[2], NamedTuple{names}(Tuple(getfield(captures, c) for c in last.(reads)))
end
_jax_op_body(f) = _jax_error("an embedded operation `", typeof(f), "` without its source")

# Scalar rules retain their mathematical derivative authority in JAX. The
# adapter contains only activity selection and the scalar chain rule.
_jax_call_positional(ctx::_JaxContext,
        rule::ScalarDerivativeRule{Name,N,Inputs,Primal}, args::Vector{_PyVal}) where {Name,N,Inputs,Primal} =
    _jax_scalar_rule(ctx, rule, args)
_jax_call_positional(ctx::_JaxContext,
        rule::DerivativeRule{Name,N,Inputs,Primal}, args::Vector{_PyVal}) where {Name,N,Inputs,Primal} =
    _jax_rule_primal(ctx, rule, Primal, args)

function _jax_rule_primal(ctx::_JaxContext, rule, Primal, args::Vector{_PyVal})
    value = _jax_rule_body(ctx, Primal, rule.primal_ops, args)
    T = _jax_return_type(rule, Any[a.T for a in args])
    # A primal of static arguments is NumPy's float64 value: rounded in a
    # float32 module (`_jax_float_static`), as the rule's result is not static.
    value.static && (value = _jax_float_static(ctx, _PyVal(value.code, T, true, "", "", 0)))
    _PyVal(value.code, T)
end

function _jax_rule_body(ctx::_JaxContext, spec, ops, args::Vector{_PyVal})
    signature, steps, ret = spec
    env = Dict{Symbol,_PyVal}(zip(signature, args))
    for (output, index, arguments) in steps
        op = ops[index]
        inputs = _PyVal[a isa Symbol ? env[a] : _jax_literal(a) for a in arguments]
        v = _jax_call(ctx, op, inputs)
        if output isa Symbol
            env[output] = v
        else
            bound = _jax_bind!(ctx, :outputs, v)
            for (k, o) in enumerate(output)
                env[o] = _PyVal(_py_atom(bound.code) * "[" * string(k - 1) * "]",
                                v.T isa Type && v.T <: Tuple ? fieldtype(v.T, k) : Any)
            end
        end
    end
    ret isa Symbol && return env[ret]
    values = _PyVal[env[r] for r in ret]
    _PyVal(_jax_tuple_code(values), Tuple{(v.T for v in values)...})
end

function _jax_rule_function!(ctx, name, spec, ops, types)
    scope = _JaxScope()
    params = [_jax_identifier(scope.used, Symbol(:x, i)) for i in eachindex(types)]
    inner = _JaxContext(ctx.mod, scope, ctx.recipe, ctx.source_module,
                        Dict{Symbol,_PyVal}(), nothing)
    result = _jax_rule_body(inner, spec, ops,
                           [_PyVal(p, T) for (p, T) in zip(params, types)])
    lines = ["def " * name * "(" * join(params, ", ") * "):"]
    append!(lines, scope.lines)
    push!(lines, "    return " * result.code)
    join(lines, "\n")
end

function _jax_scalar_rule(ctx::_JaxContext,
        rule::ScalarDerivativeRule{Name,N,Inputs,Primal,Cuts},
        args::Vector{_PyVal}) where {Name,N,Inputs,Primal,Cuts}
    # Data preparation and literal/shape expressions need ordinary host values.
    (ctx.host !== nothing || all(a -> a.static, args)) &&
        return _jax_rule_primal(ctx, rule, Primal, args)
    types = Tuple(a.T for a in args)
    key = (:scalar_rule, objectid(rule), types)
    name = get(ctx.mod.helper_names, key, nothing)
    if name === nothing
        name = _jax_private_identifier(ctx.mod.used, Symbol("_rk_", Name))
        ctx.mod.helper_names[key] = name
        primal = _jax_rule_function!(ctx, name, Primal, rule.primal_ops, types)
        push!(ctx.mod.helpers, "@jax.custom_jvp\n" * primal)

        # Joint recipes often serve several activity masks. Emit their body
        # once, with the union of its requested outputs; each mask selects
        # only its own partials. No numerical algorithm lives in this adapter.
        groups = Dict{Any,Vector{Int}}()
        for mask in eachindex(Cuts)
            signature, steps, _ = Cuts[mask]
            push!(get!(() -> Int[], groups,
                       (signature, steps, rule.cut_ops[mask])), mask)
        end
        calls = Dict{Int,Tuple{String,Vector{Symbol}}}()
        for masks in sort!(collect(values(groups)); by = first)
            mask = first(masks)
            outputs = unique(Symbol[r for m in masks for r in Cuts[m][3]])
            signature, steps, _ = Cuts[mask]
            cutname = _jax_private_identifier(ctx.mod.used, Symbol(name, "_partials_", mask))
            push!(ctx.mod.helpers, _jax_rule_function!(ctx, cutname,
                (signature, steps, Tuple(outputs)), rule.cut_ops[mask], types))
            for m in masks
                calls[m] = (cutname, outputs)
            end
        end
        jvp = _jax_private_identifier(ctx.mod.used, Symbol(name, "_jvp"))
        lines = ["def " * jvp * "(primals, tangents):",
                 "    active = tuple(not isinstance(t, jax.custom_derivatives.SymbolicZero) for t in tangents)",
                 "    mask = sum((1 << i) for i, a in enumerate(active) if a)",
                 "    if mask == 0:",
                 "        value = " * name * "(*primals)",
                 "        return value, jnp.zeros_like(value)"]
        for mask in eachindex(Cuts)
            cutname, outputs = calls[mask]
            ret = Cuts[mask][3]
            indices = [i for i in 1:N if (mask >> (i - 1)) & 1 == 1]
            terms = ["values[" * string(findfirst(==(ret[k + 1]), outputs) - 1) *
                     "] * tangents[" * string(i - 1) * "]" for (k, i) in enumerate(indices)]
            append!(lines, ["    if mask == " * string(mask) * ":",
                "        values = " * cutname * "(*primals)",
                "        return values[" * string(findfirst(==(ret[1]), outputs) - 1) *
                "], " * join(terms, " + ")])
        end
        push!(lines, name * ".defjvp(" * jvp * ", symbolic_zeros=True)")
        push!(ctx.mod.helpers, join(lines, "\n"))
    end
    _jax_bind!(ctx, :value,
        _PyVal(name * "(" * _jax_args(args) * ")", _jax_return_type(rule, collect(types))))
end

# A prepared kernel called as a function (a distribution endpoint a helper
# calls): a module-level function translated from its plan, once per kernel
# and argument types.
function _jax_call_positional(ctx::_JaxContext, k::PreparedKernel, args::Vector{_PyVal},
                              hint::Symbol = length(k.plan.want) == 1 ? only(k.plan.want).name : :kernel)
    p = k.plan
    length(p.have) == length(args) ||
        _jax_error("a prepared kernel of ", length(p.have), " inputs called with ",
                   length(args), _jax_where(ctx))
    types = Any[a.T for a in args]
    host = ctx.host !== nothing
    guard = host ? nothing : _jax_active_condition(ctx)
    key = (:kernel, objectid(k), types, host, guard !== nothing, ctx.traced_branches > 0)
    name = get(ctx.mod.helper_names, key, nothing)
    if name === nothing
        name = _jax_private_identifier(ctx.mod.used, Symbol("_rk_", hint, host ? "_np" : ""))
        ctx.mod.helper_names[key] = name
        before = _JAX_EFFECTS[]
        lines, f = _jax_function!(ctx.mod, name, p, Tuple(types),
                                  length(p.want) == 1 ? only(p.want).name : Tuple(v.name for v in p.want);
                                  reserve = false, host, active = guard !== nothing,
                                  traced_branches = ctx.traced_branches)
        ctx.mod.helper_names[(:kernel_effects, name)] = _JAX_EFFECTS[] - before
        push!(ctx.mod.helpers, host ? _host_helper_text(ctx, join(lines, "\n")) : join(lines, "\n"))
        ctx.mod.helper_names[(:kernel_type, name)] = f
    else
        _JAX_EFFECTS[] += ctx.mod.helper_names[(:kernel_effects, name)]
    end
    f = ctx.mod.helper_names[(:kernel_type, name)]
    T = length(f.outputs) == 1 ? last(only(f.outputs)) : Tuple{(last(o) for o in f.outputs)...}
    passed = String[a.code for a in args]
    guard === nothing || push!(passed, guard)
    _jax_bind!(ctx, :value, _PyVal(name * "(" * join(passed, ", ") * ")", T))
end

# A prepared child a source calls (a distribution's `standardized` step):
# its prepared kernel, named after the child.
_jax_call_positional(ctx::_JaxContext, c::_KernelPreparedChild, args::Vector{_PyVal}) =
    _jax_call_positional(ctx, c.kernel, args, c.name)

# An authored kernel called as a function, which Julia runs as
# `prepare(spec)(args...)` (a binder's data-only definition `m = mean_of(x)`
# calling a program-level `@kernel`): that prepared kernel, prepared once per
# module. Called with no arguments, a kernel is a graph view, not a value; and
# a kernel with keyword arguments or defaults fills them by its signature.
function _jax_call_positional(ctx::_JaxContext, spec::KernelSpec, args::Vector{_PyVal})
    isempty(args) && _jax_error("the kernel `", spec, "` called with no arguments is a ",
                                "graph view, not a value", _jax_where(ctx))
    k = get!(() -> prepare(spec), ctx.mod.helper_names, (:prepared_spec, objectid(spec)))
    k isa PreparedKernel ||
        _jax_error("the kernel `", spec, "` has keyword arguments or defaults; its call ",
                   "has no JAX translation", _jax_where(ctx))
    _jax_call_positional(ctx, k, args)
end

# A `@traceable` helper, or a method whose source a registered provider
# supplies: its authored body over the arguments, as the compiled
# path reads it. Its locals are scoped to the call.
function _jax_traceable_call(ctx::_JaxContext, f, args::Vector{_PyVal})
    source = _jax_traceable_body(ctx, f, args)
    source === nothing && return nothing
    names, body, mod, inputs = source
    ctx.host === nothing ||
        return _host_function_call(ctx, f, names, body, mod, args, inputs)
    _jax_authored_function_call(ctx, f, names, body, mod, inputs)
end

# `f`'s recorded or provided source over `args`: `(names, body, module,
# inputs)`, or `nothing`.
function _jax_traceable_body(ctx::_JaxContext, f, args::Vector{_PyVal})
    # A function passed as an argument dispatches on its own type (a stable
    # rewrite's `_rk_log_of(exp, x)` selects its `::typeof(exp)` method).
    types = Any[_jax_argument_type(a) for a in args]
    all(T -> T isa Type, types) || return nothing
    source = _traceable_source(typeof(f), types...)
    # A method a package generates: its source from a registered provider.
    source === nothing && (source = _provided_source(f, types))
    source === nothing && return nothing
    names, body, mod = source
    length(source) >= 5 && (body = _jax_static_parameters(body, source[5]))
    # A Vararg source's last formal is the tuple of the trailing arguments.
    inputs = _traceable_arguments(source, args, rest -> _jax_call_positional(ctx, tuple, rest))
    inputs === nothing && return nothing
    collect(Symbol, names), body, mod, inputs
end

# Whether a call of `f` reaches the default translation (recorded source or
# lowered code): no translation hook has a method for `f` itself.
function _jax_default_translation(f)
    F = typeof(f)
    which(_jax_call, Tuple{_JaxContext,F,Vector{_PyVal},Any}) ===
        which(_jax_call, Tuple{_JaxContext,Any,Vector{_PyVal},Any}) &&
    which(_jax_call_positional, Tuple{_JaxContext,F,Vector{_PyVal}}) ===
        which(_jax_call_positional, Tuple{_JaxContext,Function,Vector{_PyVal}}) &&
    which(_jax_function_call, Tuple{_JaxContext,F,Vector{_PyVal}}) ===
        which(_jax_function_call, Tuple{_JaxContext,Any,Vector{_PyVal}})
end

# A broadcast of a helper with a named function (`toy_mean.(a, b, t)` of an
# authored `toy_mean`) vectorizes that function itself when it takes exactly
# the scalar arguments, rather than a private wrapper around a call of it.
# Otherwise `nothing` (a function with no named form, or one that needs
# extra bound/guard arguments, a tuple-valued result or its caller's scope).
function _jax_vectorized_named_function!(ctx::_JaxContext, f, types)
    ctx.host === nothing && isempty(_JAX_GUARDS) && f isa Function &&
        Base.issingletontype(typeof(f)) && _jax_default_translation(f) || return nothing
    scope = _JaxScope()
    inner = _JaxContext(ctx.mod, scope, ctx.recipe, ctx.source_module,
                        Dict{Symbol,_PyVal}(), nothing)
    params = [_jax_identifier(scope.used, Symbol(:x, i)) for i in eachindex(types)]
    args = [_PyVal(p, T) for (p, T) in zip(params, types)]
    defined = nothing
    source = _jax_traceable_body(inner, f, args)
    if source !== nothing
        names, body, mod, inputs = source
        emit(c, arguments) = _jax_source_expr(c, body, names, arguments, mod, nothing)
        defined = _jax_named_function!(inner, f, names, inputs, mod, emit,
                                       "from its authored source")
    elseif !_jax_internal_function(f) &&
           _jax_return_type(f, Any[types...]) !== Union{}
        m = try
            which(Tuple{typeof(f), types...})
        catch
            nothing
        end
        if m !== nothing && !m.isva && length(Base.method_argnames(m)) == length(types) + 1 &&
           length(try _jax_lowered_code(f, types) catch; () end) == 1
            emit_lowered(c, arguments) = _jax_lowered_call(c, f, arguments)
            defined = _jax_named_function!(inner, f, Base.method_argnames(m)[2:end], args,
                                           ctx.source_module, emit_lowered,
                                           "from its Julia definition")
        end
    end
    defined === nothing && return nothing
    name, v, bounds, passed, flat, _, _ = defined
    passed == params && isempty(bounds) && flat === nothing && isempty(v.extent) &&
        isempty(v.limit) && v.origin === nothing || return nothing
    name
end

# Source-backed helpers keep their authored names and formal arguments. The
# extra extent/bound arguments are the same runtime representation used by
# plates; none is a value baked into the module from a particular dataset.
function _jax_authored_function_call(ctx, f, names, body, source_module, inputs)
    emit(inner, arguments) = _jax_source_expr(inner, body, names, arguments, source_module, nothing)
    inline() = _jax_source_expr(ctx, body, names, inputs, source_module, nothing)
    _jax_named_function_call(ctx, f, names, inputs, source_module, emit, inline,
                             "from its authored source")
end

# A plain helper without recorded source, outside Julia and ReactiveKernels:
# its straight-line lowered code becomes a named function as well, with the
# method's argument names and the function's docstring, so an authored helper
# reads as one whether or not it is `@traceable`. Julia's and ReactiveKernels'
# own functions, a Vararg method and a closure stay translated in place.
function _jax_named_lowered_call(ctx::_JaxContext, f, args::Vector{_PyVal})
    ctx.host === nothing && f isa Function && Base.issingletontype(typeof(f)) &&
        !_jax_internal_function(f) || return nothing
    types = Any[_jax_argument_type(a) for a in args]
    all(T -> T isa Type, types) || return nothing
    m = try
        which(Tuple{typeof(f), types...})
    catch
        return nothing
    end
    m.isva && return nothing
    names = Base.method_argnames(m)[2:end]
    length(names) == length(args) || return nothing
    infos = try
        _jax_lowered_code(f, types)
    catch
        return nothing
    end
    length(infos) == 1 || return nothing
    emit(inner, arguments) = _jax_lowered_call(inner, f, arguments)
    inline() = _jax_lowered_call(ctx, f, args)
    _jax_named_function_call(ctx, f, names, args, ctx.source_module, emit, inline,
                             "from its Julia definition")
end

# Julia's own functions (Base, Core, the standard libraries) and those of
# ReactiveKernels' packages and extensions: translated in place, or, with
# recorded source (a stable rewrite's `_rk_div_by_of`), a private helper.
function _jax_internal_function(f)
    m = Base.moduleroot(parentmodule(typeof(f)))
    (m === Base || m === Core) && return true
    path = pathof(m)
    path !== nothing && startswith(path, Sys.STDLIB) && return true
    dir, root = pkgdir(m), pkgdir(@__MODULE__)
    dir !== nothing && root !== nothing && (dir == root || startswith(dir, root * "/"))
end
# A helper a reader does not call: an emitter-internal one, or one whose
# authored name already marks it private (`_scaled`).
_jax_private_function(f) = startswith(String(nameof_or(f)), "_") || _jax_internal_function(f)

# The value of `f` over `inputs` through its named function, defining that
# function on its first use (`emit` translates the body in the function's
# scope); `inline()` translates it in place when its value exists only during
# translation (a type, a closure).
function _jax_named_function_call(ctx, f, names, inputs, source_module, emit, inline, origin)
    defined = _jax_named_function!(ctx, f, names, inputs, source_module, emit, origin)
    defined === nothing && return inline()
    name, v, bounds, passed, flat, identity, lengths = defined
    call = name * "(" * join([passed; bounds], ", ") * ")"
    if isempty(v.extent) && isempty(v.limit) && v.origin === nothing
        return _jax_bind!(ctx, name, _PyVal(call, v.T, v.static))
    end
    result = _jax_bind!(ctx, name, _PyVal(call, Any))
    _jax_identity_return(_jax_function_return!(ctx, v, flat, result.code), identity, lengths,
                         result.code)
end

# A returned length that is one of the function's own length arguments (the
# field of a retained tuple passed through, `getfield(pair, 2)`): the caller
# reads its own expression for that length, which keeps the length's prepared
# flat positions for a later `reduce(vcat, cells)`. `identity` pairs each
# packed result length with the function's expression it returns.
function _jax_identity_extents(v, result)
    out = Pair{String,String}[]
    for (e, p) in zip((v.extent, v.field_extents...), (result.extent, result.field_extents...))
        isempty(e) || isempty(p) || push!(out, p => e)
    end
    out
end
# Each length expression of a function's arguments (`inner`, its parameters)
# mapped to the caller's for the same length (`outer`, what it passed).
function _jax_argument_extents!(out, inner::_PyVal, outer::_PyVal)
    for (a, b) in zip((inner.extent, inner.limit, inner.field_extents...),
                      (outer.extent, outer.limit, outer.field_extents...))
        isempty(a) || isempty(b) || (out[a] = b)
    end
    if inner.T isa _JaxAggregate && outer.T isa _JaxAggregate
        foreach((x, y) -> _jax_argument_extents!(out, x, y), inner.T.fields, outer.T.fields)
    elseif inner.T isa _JaxCallable && outer.T isa _JaxCallable &&
           inner.T.f isa _JaxClosure && outer.T.f isa _JaxClosure
        foreach((x, y) -> _jax_argument_extents!(out, x, y), inner.T.f.captures,
                outer.T.f.captures)
    end
    out
end
# The caller's expression for the function's length expression `e`: an
# argument's own length, or a field of one (`pair_lengths[1]` of a tuple whose
# fields share one lengths value), else `nothing`.
function _jax_caller_length(lengths, e)
    haskey(lengths, e) && return lengths[e]
    m = match(r"^([A-Za-z_][A-Za-z0-9_]*)((?:\[[0-9]+\])+)$", e)
    m === nothing && return nothing
    base = get(lengths, m[1], nothing)
    base === nothing ? nothing : _py_atom(base) * m[2]
end
function _jax_identity_return(v, identity, lengths, returned)
    sub = Dict{String,String}()
    for (p, e) in identity
        caller = _jax_caller_length(lengths, e)
        caller === nothing || (sub[replace(p, "__rk_function_result__" => returned)] = caller)
    end
    isempty(sub) && return v
    swap(e) = get(sub, e, e)
    _PyVal(v.code, v.T, v.static, swap(v.extent), v.limit, v.padded, v.origin,
           map(swap, v.field_extents))
end

# `f`'s named function for arguments like `inputs`: `(name, value, bounds,
# passed, flat, identity, lengths)`, the call's arguments being `passed` then
# `bounds` (`identity` and `lengths` for `_jax_identity_return`), or `nothing`
# when the value exists only during translation.
function _jax_named_function!(ctx, f, names, inputs, source_module, emit, origin)
    # Integer provenance and history access proofs refer to the surrounding
    # loop/scan. Keep those helpers lexically nested so the scan can still
    # resolve its access placeholders and prove a bounded history carry.
    contextual = any(_jax_contextual_value, inputs)
    key = (:authored_function, objectid(f),
           Tuple((a.T, a.static, !isempty(a.extent), !isempty(a.limit), a.padded, a.origin,
                  Tuple(!isempty(e) for e in a.field_extents),
                  Tuple(_jax_flat_positions(ctx, e) !== nothing
                        for e in (a.extent, a.field_extents...)))
                 for a in inputs), !isempty(_JAX_GUARDS), ctx.traced_branches > 0,
           contextual ? (ctx.scope, Tuple((a.code, a.extent, a.limit) for a in inputs),
                         Tuple(last.(copy(_JAX_GUARDS)))) : nothing)
    cached = get(ctx.mod.helper_names, key, nothing)
    scope = _JaxScope(contextual ? ctx.scope : nothing)
    params, passed = String[], String[]
    arguments = contextual ? inputs :
        [_jax_function_input!(ctx, scope, params, passed, n, a) for (n, a) in zip(names, inputs)]
    guard = contextual ? nothing : _jax_function_guard!(ctx, scope, params, passed)
    if cached === nothing
        inner = _JaxContext(ctx.mod, scope, ctx.recipe, source_module,
                            Dict{Symbol,_PyVal}(), nothing)
        inner.traced_branches = ctx.traced_branches
        before = _JAX_EFFECTS[]
        v = if contextual
            emit(inner, arguments)
        else
            _jax_with_function_guard(scope, guard) do
                emit(inner, arguments)
            end
        end
        effects = _JAX_EFFECTS[] - before
        # Types and callable closures belong to translation. Their captured
        # runtime leaves still need the caller's lexical names.
        (_jax_translation_only(v) || _jax_type_of_value(v) !== nothing) && return nothing
        name = _jax_named_function(ctx, nameof_or(f); private = _jax_private_function(f))
        text = join(scope.lines, "\n") * "\n" * v.code * "\n" * v.extent * "\n" * v.limit
        bounds = contextual ? String[] : [bn for (bn, _, _) in _jax_prepared_bounds(ctx.mod)
                  if occursin(Regex("\\b" * bn * "\\b"), text)]
        append!(params, bounds)
        binding = Base.Docs.Binding(parentmodule(typeof(f)), nameof_or(f))
        documented = any(Base.Docs.modules) do m
            metadata = Base.Docs.meta(m; autoinit = false)
            metadata !== nothing && haskey(metadata, binding)
        end
        doc = documented ? string(Base.Docs.doc(f)) :
              "Compute `$(nameof_or(f))` " * origin * "."
        signature = [n * ": ArrayLike" for n in params]
        lines = ["def " * name * "(" * join(signature, ", ") * ") -> " *
                 (isempty(v.extent) && isempty(v.limit) && v.origin === nothing ?
                    _jax_return_annotation(:value, [v]) : "tuple") * ":",
                 _jax_docstring(doc, "    ")]
        append!(lines, scope.lines)
        returned, result, flat = _jax_function_result(inner, v)
        identity = _jax_identity_extents(v, result)
        push!(lines, "    return " * returned)
        if contextual
            foreach(line -> _jax_emit!(ctx, line), lines)
        else
            push!(ctx.mod.helpers, join(lines, "\n"))
        end
        cached = (name, result, bounds, effects, flat, identity)
        ctx.mod.helper_names[key] = cached
    else
        # A reused check still prevents its caller's branch from being
        # discarded, even though this call emits no new helper definition.
        _JAX_EFFECTS[] += cached[4]
    end
    name, v, bounds, _, flat, identity = cached
    lengths = Dict{String,String}()
    contextual || foreach((a, b) -> _jax_argument_extents!(lengths, a, b), arguments, inputs)
    (name, v, bounds, passed, flat, identity, lengths)
end

_jax_contextual_value(v::_PyVal) = v.origin !== nothing || _jax_contextual_type(v.T)
_jax_contextual_type(T) = false
_jax_contextual_type(T::_JaxAggregate) = any(_jax_contextual_value, T.fields)
_jax_contextual_type(T::_JaxCallable) = T.f isa _JaxClosure && any(_jax_contextual_value, T.f.captures)

# A returned value's extents and integer proofs are part of its execution
# boundary too. Return their numerical leaves explicitly, rather than losing
# the proof or retaining a name local to the helper's body.
function _jax_function_result(ctx, v)
    isempty(v.extent) && isempty(v.limit) && v.origin === nothing && return v.code, v, nothing
    parts = String[]
    pack(code) = isempty(code) ? "" : begin
        push!(parts, code)
        "__rk_function_result__[" * string(length(parts) - 1) * "]"
    end
    result = _jax_map_value(v, pack)
    flat = Pair{String,String}[]
    extents = isempty(v.field_extents) ? (v.extent,) : v.field_extents
    returned_extents = isempty(v.field_extents) ? (result.extent,) : result.field_extents
    for (extent, returned_extent) in zip(extents, returned_extents)
        isempty(extent) && continue
        positions = _jax_flat_positions(ctx, extent)
        positions === nothing || push!(flat, returned_extent => pack(positions))
    end
    "(" * join(parts, ", ") * (length(parts) == 1 ? ",)" : ")"), result, flat
end
function _jax_function_return!(ctx, v, flat, returned)
    code(s) = replace(s, "__rk_function_result__" => returned)
    result = _jax_map_value(v, code)
    if flat !== nothing
        for (extent, positions) in flat
            ctx.scope.flat_positions[code(extent)] = code(positions)
        end
    end
    result
end
_jax_map_value(v::_PyVal, code) = _PyVal(code(v.code), v.T, v.static,
    code(v.extent), code(v.limit), v.padded, _jax_map_origin(v.origin, code),
    map(code, v.field_extents))
_jax_map_origin(origin, code) = origin
_jax_map_origin(o::_JaxOrigin, code) =
    _JaxOrigin(o.key, o.position, code(o.offset), o.path, code(o.scale))
_jax_map_origin(o::_JaxBetween, code) =
    _JaxBetween(_jax_map_value(o.lo, code), _jax_map_value(o.hi, code))

# A module-level function cannot close over a caller's branch/plate mask.
# Pass that mask explicitly, then translate the body in its own guard scope.
function _jax_active_condition(ctx)
    isempty(_JAX_GUARDS) && return nothing
    guards = String[]
    for (parent, guard) in _JAX_GUARDS
        _jax_scope_within(ctx.scope, parent) ||
            _jax_error("a function's active condition is outside its caller's scope", _jax_where(ctx))
        push!(guards, guard)
    end
    foldl((a, b) -> "jnp.logical_and(" * a * ", " * b * ")", guards)
end
function _jax_function_guard!(ctx, scope, params, passed)
    guard = _jax_active_condition(ctx)
    guard === nothing && return nothing
    name = _jax_identifier(scope.used, :active)
    push!(params, name)
    push!(passed, guard)
    name
end
function _jax_with_function_guard(f, scope, guard)
    saved = copy(_JAX_GUARDS)
    empty!(_JAX_GUARDS)
    try
        guard === nothing ? f() : _jax_with_guard(f, scope, guard)
    finally
        empty!(_JAX_GUARDS)
        append!(_JAX_GUARDS, saved)
    end
end

_jax_function_input!(ctx, scope, params, passed, hint, v::_PyVal) =
    _jax_function_input!(ctx, scope, params, passed, hint, v, v.T)
function _jax_function_input!(ctx, scope, params, passed, hint, v::_PyVal, T)
    if T isa Type && T <: Function && Base.issingletontype(T)
        return _PyVal(v.code, _JaxCallable(T.instance), true)
    end
    type = _jax_type_of_value(v)
    type === nothing || return _PyVal(v.code, _JaxCallable(type), true)
    union!(scope.used, ctx.mod.used)
    name = _jax_identifier(scope.used, hint)
    push!(params, name); push!(passed, v.code)
    extent = isempty(v.extent) ? "" : _jax_identifier(scope.used, Symbol(hint, :_lengths))
    limit = isempty(v.limit) ? "" : _jax_identifier(scope.used, Symbol(hint, :_bound))
    function carry_flat(source, target, hint)
        flat = _jax_flat_positions(ctx, source)
        if flat !== nothing
            flat_name = _jax_identifier(scope.used, Symbol(hint, :_flat))
            push!(params, flat_name); push!(passed, flat)
            scope.flat_positions[target] = flat_name
        end
    end
    if !isempty(extent)
        push!(params, extent); push!(passed, v.extent)
        isempty(v.field_extents) && carry_flat(v.extent, extent, hint)
    end
    field_extents = Tuple(begin
        if isempty(source)
            ""
        else
            target = extent * "[" * string(i - 1) * "]"
            carry_flat(source, target, Symbol(hint, :_, i))
            target
        end
    end for (i, source) in enumerate(v.field_extents))
    if !isempty(limit)
        push!(params, limit); push!(passed, v.limit)
    end
    _PyVal(name, T, v.static, extent, limit, v.padded, v.origin, field_extents)
end
function _jax_function_input!(ctx, scope, params, passed, hint, v::_PyVal, T::_JaxCallable)
    T.f isa _JaxClosure || return v
    captures = [_jax_function_input!(ctx, scope, params, passed, Symbol(hint, :_, i), a)
                for (i, a) in enumerate(T.f.captures)]
    _PyVal(v.code, _JaxCallable(_JaxClosure(T.f.F, captures)), v.static)
end
function _jax_function_input!(ctx, scope, params, passed, hint, v::_PyVal, T::_JaxAggregate)
    fields = [_jax_function_input!(ctx, scope, params, passed, Symbol(hint, :_, i), a)
              for (i, a) in enumerate(T.fields)]
    _PyVal(v.code, _JaxAggregate(T.T, fields), v.static)
end

# `body` with the `where` parameters its method's call binds (`statics`, from
# `_traceable_source`) spliced in as values: `exp(k * float(x))` of a
# `::Val{k}` argument reads the literal `k`. A name the body assigns stays.
function _jax_static_parameters(body, statics)
    isempty(statics) && return body
    assigned = keys(_rw_count!(Dict{Symbol,Int}(), body))
    subst = Dict{Symbol,Any}(k => v for (k, v) in pairs(statics)
                             if v !== nothing && !(k in assigned))
    isempty(subst) ? body : _rw_substitute(body, subst)
end

# A plain Julia helper with no translation of its own: its lowered code, when
# that is straight-line or has terminal forward branches. Loops and joins
# still require recorded source, whose structured control flow is retained.
const _JAX_LOWERING_STACK = Any[]

function _jax_lowered_call(ctx::_JaxContext, f, args::Vector{_PyVal})
    f isa Function || return nothing
    # A function, type or module passed on as an argument (lowered code reads
    # globals as `_JaxCallable`s) dispatches on its own type.
    types = Any[_jax_argument_type(a) for a in args]
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

_jax_lowered_code(f, types) = Base.code_lowered(f, Tuple{types...})
function _jax_lowered_code(f::_JaxClosure, types)
    m = which(Tuple{f.F, types...})
    Base.hasgenerator(m) && return Core.CodeInfo[]
    [Base.uncompressed_ast(m)]
end
function _jax_call_positional(ctx::_JaxContext, f::_JaxClosure, args::Vector{_PyVal})
    v = _jax_lowered_call(ctx, f, args)
    v === nothing && _jax_error("the captured callable `", f.F, "` has no JAX translation",
                               _jax_where(ctx))
    v
end

function _jax_lowered_call_body(ctx::_JaxContext, f, args::Vector{_PyVal}, types)
    infos = try
        _jax_lowered_code(f, types)
    catch
        return nothing
    end
    length(infos) == 1 || return nothing
    ci = only(infos)
    any(enumerate(ci.code)) do (i, st)
        st isa Core.GotoNode || (st isa Core.GotoIfNot && st.dest <= i) ||
            (st isa Expr && st.head in (:enter, :leave, :pop_exception))
    end &&
        _jax_error("the function `", f, "` has loops or branches, so its lowered code is ",
                   "not translated", _jax_where(ctx), ". Define it with `@traceable` (its ",
                   "authored body is translated), author it as a kernel, or give it a ",
                   "`ReactiveKernels._jax_function_call` method; for a method ",
                   "a package generates, the package supplies its source with ",
                   "`ReactiveKernels.register_source_provider!`")
    mod = parentmodule(_jax_callable_type(f))
    ssa = Vector{Any}(undef, length(ci.code))
    # A call whose value is read once stays an expression at that read, as an
    # authored body composes it (`level + trend * t`, not three temporaries); a
    # value read more than once, or never (its call may be an effect), is
    # bound. An assignment binds the authored local's name.
    uses = zeros(Int, length(ci.code))
    count_uses(x) = x isa Core.SSAValue ? (uses[x.id] += 1; nothing) :
        x isa Expr ? foreach(count_uses, x.args) : nothing
    for st in ci.code
        st isa Core.ReturnNode ? (isdefined(st, :val) && count_uses(st.val)) :
        st isa Core.GotoIfNot ? count_uses(st.cond) : count_uses(st)
    end
    m = try
        which(Tuple{_jax_callable_type(f), types...})
    catch
        nothing
    end
    # A Vararg method's last slot holds its trailing arguments as one tuple.
    # `splats` keeps them as values, so a splat of that slot (`g(args...)`,
    # lowered `Core._apply_iterate`) passes them on as they came.
    nfixed = m !== nothing && m.isva ? Int(m.nargs) - 2 : length(args)
    slots = Dict{Int,_PyVal}(i + 1 => a for (i, a) in enumerate(args[1:nfixed]))
    slots[1] = _PyVal("<callable>", _JaxCallable(f))
    splats = Dict{Int,Vector{_PyVal}}()
    fields = Dict{Int,Vector{_PyVal}}()
    if nfixed < length(args) || (m !== nothing && m.isva)
        rest = args[nfixed+1:end]
        # A function or type among them has no Python value, so the tuple
        # exists only to be splatted (`value` refuses any other read of it).
        any(a -> a.T isa _JaxCallable, rest) ||
            (slots[nfixed + 2] = _jax_call_positional(ctx, tuple, rest))
        splats[nfixed + 2] = rest
    end
    # The method's static parameters (`where T`), as the call's types bind them.
    sparams = try
        ccall(:jl_type_intersection_with_env, Any, (Any, Any),
              Tuple{_jax_callable_type(f), types...}, m.sig)[2]
    catch
        Core.svec()
    end
    value(x) = x isa Core.SSAValue ? ssa[x.id] :
        x isa Expr && x.head === :boundscheck ? _jax_literal(true) :
        x isa Expr && x.head === :static_parameter ?
            (x.args[1] <= length(sparams) && !(sparams[x.args[1]] isa TypeVar) ?
             _jax_global_value(ctx, sparams[x.args[1]], :parameter) :
             _jax_error("an unbound static parameter in `", f, "`", _jax_where(ctx))) :
        x isa Core.SlotNumber ? (haskey(slots, x.id) ? slots[x.id] :
                                 haskey(splats, x.id) ?
                                 _jax_error("the Vararg arguments of `", f, "` hold a function ",
                                            "or type and are read other than splatted",
                                            _jax_where(ctx)) :
                                 _jax_error("an unassigned local in `", f, "`")) :
        x isa Core.Argument ? slots[x.n] :
        x isa Expr && x.head === :new ? _jax_lowered_new(ctx, value(x.args[1]),
                                                       _PyVal[value(a) for a in x.args[2:end]]) :
        x isa GlobalRef ? _jax_global_value(ctx, getglobal(x.mod, x.name), x.name) :
        x isa QuoteNode ? _jax_global_value(ctx, x.value, :constant) :
        x isa Number ? _jax_literal(x) :
        _jax_global_value(ctx, x, :constant)
    function region(start)
        for i in start:length(ci.code)
            st = ci.code[i]
            if st isa Core.ReturnNode
                return isdefined(st, :val) ? value(st.val) : _PyVal("None", Union{}, true)
            elseif st isa Core.GotoIfNot
                # A terminal branch (for example a shape guard) has no loop or
                # control-flow join. Translate each returning region under the
                # ordinary source conditional, including its lazy error effects.
                condition = value(st.cond)
                saved_ssa, saved_slots = copy(ssa), copy(slots)
                saved_splats, saved_fields = copy(splats), copy(fields)
                arm(next) = () -> begin
                    copyto!(ssa, saved_ssa)
                    empty!(slots); merge!(slots, saved_slots)
                    empty!(splats); merge!(splats, saved_splats)
                    empty!(fields); merge!(fields, saved_fields)
                    region(next)
                end
                return _jax_conditional(ctx, condition, arm(i + 1), arm(Int(st.dest)))
            elseif st isa Expr && st.head === :(=)
                id = (st.args[1]::Core.SlotNumber).id
                hint = _jax_slot_hint(ci, id)
                # An expression a single-read call left unbound is bound here.
                v = _jax_bind!(ctx, hint, value_of_rhs(ctx, st.args[2], value, mod, splats,
                                                       fields; hint))
                slots[id] = v
                delete!(splats, id)
                ssa[i] = v
            elseif st isa Expr && st.head === :call
                ssa[i] = value_of_rhs(ctx, st, value, mod, splats, fields;
                                      hint = uses[i] == 1 ? nothing : :t)
                _jax_lowered_fields!(fields, i, st, value)
            elseif st isa Expr && st.head in (:meta, :code_coverage_effect, :inbounds, :boundscheck)
                ssa[i] = nothing
            elseif st === nothing || st isa LineNumberNode || st isa Core.NewvarNode
                ssa[i] = nothing
            else
                ssa[i] = value(st)
            end
        end
        _jax_error("the lowered code of `", f, "` does not return", _jax_where(ctx))
    end
    region(1)
end
function value_of_rhs(ctx::_JaxContext, rhs, value, mod, splats = Dict{Int,Vector{_PyVal}}(),
                      fields = Dict{Int,Vector{_PyVal}}(); hint = :t)
    rhs isa Expr && rhs.head === :call || return value(rhs)
    callee = rhs.args[1]
    g = callee isa GlobalRef ? getglobal(callee.mod, callee.name) :
        callee isa Core.SSAValue || callee isa Core.SlotNumber || callee isa Core.Argument ?
            _jax_callable(value(callee)) :
        callee isa QuoteNode ? callee.value : callee
    bind(v) = hint === nothing ? v : _jax_bind!(ctx, hint, v)
    g === Core._apply_iterate &&
        return bind(_jax_lowered_splat(ctx, rhs.args[2:end], value, splats))
    g === Core.kwcall &&
        return bind(_jax_lowered_kwcall(ctx, rhs.args[2:end], value, fields))
    bind(_jax_call(ctx, g, _PyVal[value(a) for a in rhs.args[2:end]]))
end
# The name a lowered assignment binds: its authored local's, when it has one.
function _jax_slot_hint(ci, id)
    names = ci.slotnames
    id <= length(names) || return :t
    name = String(names[id])
    occursin(r"^[A-Za-z_][A-Za-z0-9_]*$", name) ? Symbol(name) : :t
end
# `f(args...; kw...)` in lowered code, `Core.kwcall(NamedTuple{names}(values),
# f, args...)`: the call with those keyword arguments, as source spells it.
# `fields` holds the values of the tuples the lowered code built, so each
# keyword argument is the value it was given, not an element read of a tuple.
function _jax_lowered_kwcall(ctx::_JaxContext, args, value, fields)
    nt = value(args[1])
    nt.T isa Type && nt.T <: NamedTuple && _jax_is_tuple(nt.T) ||
        _jax_error("a keyword call whose keyword arguments are not known while emitting",
                   _jax_where(ctx))
    names = fieldnames(nt.T)
    values = args[1] isa Core.SSAValue && haskey(fields, args[1].id) ? fields[args[1].id] :
        [_jax_tuple_element(ctx, nt, _PyVal(string(k), Int, true)) for k in eachindex(names)]
    kw = Pair{Symbol,_PyVal}[n => v for (n, v) in zip(names, values)]
    _jax_call(ctx, _jax_function_of(value(args[2])), _PyVal[value(a) for a in args[3:end]], kw)
end
# The values a lowered `Core.tuple(...)` call (and a `NamedTuple{names}` of
# it) was built from, by statement.
function _jax_lowered_fields!(fields, i::Int, st::Expr, value)
    callee = st.args[1]
    if callee isa GlobalRef && callee.mod === Core && callee.name === :tuple
        fields[i] = _PyVal[value(a) for a in st.args[2:end]]
    elseif callee isa Core.SSAValue && length(st.args) == 2 && st.args[2] isa Core.SSAValue &&
           haskey(fields, st.args[2].id)
        T = value(callee).T
        T isa _JaxCallable && T.f isa Type && T.f <: NamedTuple &&
            (fields[i] = fields[st.args[2].id])
    end
    fields
end
# `g(xs...)` in lowered code, `Core._apply_iterate(iterate, g, xs...)`: each
# splatted collection is a tuple of known length, a Vararg method's trailing
# arguments (`splats`) or a tuple value, and its elements are `g`'s arguments.
function _jax_lowered_splat(ctx::_JaxContext, args, value, splats)
    _jax_function_of(value(args[1])) === iterate ||
        _jax_error("a splat that iterates with `", args[1], "` has no JAX translation",
                   _jax_where(ctx))
    g = _jax_function_of(value(args[2]))
    elements = _PyVal[]
    for x in args[3:end]
        id = x isa Core.SlotNumber ? x.id : x isa Core.Argument ? x.n : 0
        if haskey(splats, id)
            append!(elements, splats[id])
            continue
        end
        v = value(x)
        if v.T isa _JaxAggregate && v.T.T <: Tuple
            append!(elements, v.T.fields)
            continue
        end
        v.T isa Type && v.T <: Tuple && _jax_is_tuple(v.T) ||
            _jax_error("splatting a `", v.T, "` has no JAX translation", _jax_where(ctx))
        for k in 1:fieldcount(v.T)
            push!(elements, _jax_tuple_element(ctx, v, _PyVal(string(k), Int, true)))
        end
    end
    _jax_call(ctx, g, elements)
end
# `%new(T, captures...)` retains the translated captures. No example values
# are constructed, and no captured numerical value is specialized into source.
function _jax_lowered_new(ctx::_JaxContext, T::_PyVal, captures::Vector{_PyVal})
    F = T.T isa _JaxCallable ? T.T.f : nothing
    F isa DataType && F <: Function && !ismutabletype(F) && fieldcount(F) == length(captures) ||
        _jax_error("constructing a `", F, "` in lowered code has no JAX translation",
                   _jax_where(ctx))
    isempty(captures) && Base.issingletontype(F) && return _PyVal("None", F, true)
    _PyVal("<closure>", _JaxCallable(_JaxClosure(F, captures)), false)
end
# A global read in lowered code: callables pass as themselves, data as values.
_jax_global_value(ctx::_JaxContext, x, hint) = x isa Union{Function,Type,PreparedKernel,Module} ?
    _PyVal("<callable>", _JaxCallable{typeof(x)}(x)) : _jax_value(ctx, x, hint)
_jax_callable(v::_PyVal) = v.T isa _JaxCallable ? v.T.f :
    _jax_error("`", v.code, "` is called but is not a callable")

# The calls Julia's lowering writes for ordinary syntax: dotted calls, literal
# powers, tuples, field reads and type applications.
_jax_call_positional(ctx::_JaxContext, ::typeof(Base.broadcasted), args::Vector{_PyVal}) =
    _jax_broadcast(ctx, _jax_function_of(args[1]), args[2:end])
_jax_call_positional(ctx::_JaxContext, ::typeof(Base.materialize), args::Vector{_PyVal}) = only(args)
# `afoldl` folds the fixed argument list, not an array's data-derived length.
# Each call still goes through the scalar/source/lowered translation machinery.
function _jax_call_positional(ctx::_JaxContext, ::typeof(Base.afoldl), args::Vector{_PyVal})
    f = _jax_function_of(args[1])
    foldl(args[3:end]; init = args[2]) do acc, x
        _jax_call(ctx, f, [acc, x])
    end
end
function _jax_call_positional(ctx::_JaxContext, ::typeof(tuple), args::Vector{_PyVal})
    T = Tuple{(_jax_argument_type(a) for a in args)...}
    any(_jax_translation_only, args) &&
        return _PyVal("<tuple>", _JaxAggregate(T, args), all(a -> a.static, args))
    _PyVal(_jax_tuple_code(args), T, all(a -> a.static, args),
           _jax_tuple_extent(args), _jax_tuple_limit(args), _jax_tuple_fields(args),
           nothing, Tuple(a.extent for a in args))
end
# A shape tuple may carry a padded vector's length through a local or helper.
# Its bound refers to the padded input shape, not an observed length literal.
_jax_tuple_limit(args) = any(a -> !isempty(a.limit), args) &&
    all(a -> a.T isa Type && a.T <: Integer && (a.static || !isempty(a.limit)), args) ?
    _jax_tuple_code([_PyVal(a.static ? a.code : a.limit, a.T) for a in args]) : ""
function _jax_call_positional(ctx::_JaxContext, ::typeof(Base.literal_pow), args::Vector{_PyVal})
    f, x, p = args
    p.T isa Type && p.T <: Val && !_jax_is_array(x.T) ||
        _jax_error("a literal power of `", x.T, "`", _jax_where(ctx))
    n = p.T.parameters[1]
    _PyVal(_py_infix("**", [x.code, string(n)]), _jax_return_type(^, Any[x.T, typeof(n)]))
end
# A symbol known while emitting: a quoted `:name` in lowered code, or a symbol
# constant such as a static parameter's (a Python string literal).
function _jax_static_symbol(ctx::_JaxContext, v::_PyVal)
    v.T === Symbol || _jax_error("`", v.code, "` is not a symbol known while emitting",
                                 _jax_where(ctx))
    Symbol(startswith(v.code, ':') ? v.code[2:end] : Meta.parse(v.code)::String)
end
# A module's global, read while emitting as lowered code reads it: functions,
# types and modules pass as themselves, data as values (`_jax_global_value`).
function _jax_module_global(ctx::_JaxContext, m::Module, name::Symbol, read)
    isdefined(m, name) ||
        _jax_error("`", name, "` is not defined in module `", m, "`", _jax_where(ctx))
    _jax_global_value(ctx, read(m, name), name)
end
function _jax_call_positional(ctx::_JaxContext, ::typeof(getproperty), args::Vector{_PyVal})
    x, field = args
    x.T isa _JaxCallable && x.T.f isa _JaxClosure &&
        return _jax_function_call(ctx, getproperty, args)
    name = _jax_static_symbol(ctx, field)
    # `StanBlocks.jcall` in lowered code is `getproperty(StanBlocks, :jcall)`.
    x.T isa _JaxCallable{Module} && return _jax_module_global(ctx, x.T.f, name, getproperty)
    x.T isa Type && x.T <: NamedTuple || _jax_error("the field read `", name, "` on `", x.T, "`",
                                                    _jax_where(ctx))
    k = findfirst(==(name), fieldnames(x.T))
    k === nothing && _jax_error("`", x.T, "` has no field `", name, "`", _jax_where(ctx))
    _PyVal(_py_atom(x.code) * "[" * string(k - 1) * "]", fieldtype(x.T, k))
end
# `getglobal(M, name)` of a module and a name known while emitting (a method's
# static parameters `where {M,name}`).
function _jax_call_positional(ctx::_JaxContext, ::typeof(getglobal), args::Vector{_PyVal})
    length(args) == 2 && args[1].T isa _JaxCallable{Module} ||
        _jax_error("`getglobal` of a module not known while emitting", _jax_where(ctx))
    _jax_module_global(ctx, args[1].T.f, _jax_static_symbol(ctx, args[2]), getglobal)
end
# A tuple element by a static index (lowered `getindex`/`getfield`, and
# `first`/`last`), or an array's first or last element.
function _jax_tuple_element(ctx::_JaxContext, x::_PyVal, i::_PyVal)
    m = match(r"^[0-9]+$", i.code)
    m === nothing && _jax_error("a tuple read at a computed index", _jax_where(ctx))
    k = parse(Int, i.code)
    x.T isa _JaxAggregate && return x.T.fields[k]
    _PyVal(_py_atom(x.code) * "[" * string(k - 1) * "]", fieldtype(x.T, k), x.static,
           _jax_field_extent(x, k, fieldtype(x.T, k)))
end
_jax_is_tuple(@nospecialize(T)) = T isa Type && T <: Union{Tuple,NamedTuple} && isconcretetype(T)
# Indexing called with values (lowered code). A view is the gather: JAX
# arrays are values, as the indexing spelling `view(x, i)` already reads.
function _jax_call_positional(ctx::_JaxContext,
        f::Union{typeof(getindex),typeof(getfield),typeof(view),typeof(Base.maybeview)},
        args::Vector{_PyVal})
    x = args[1]
    if x.T isa _JaxCallable && x.T.f isa _JaxClosure && f === getfield
        closure = x.T.f
        field = args[2]
        k = field.T === Symbol ? findfirst(==(_jax_static_symbol(ctx, field)), fieldnames(closure.F)) :
            _jax_int_literal(field) ? parse(Int, field.code) : nothing
        k === nothing && _jax_error("a closure field not known while emitting", _jax_where(ctx))
        return closure.captures[k]
    end
    if x.T isa _JaxAggregate
        x.T.T <: Tuple && length(args) == 2 && return _jax_tuple_element(ctx, x, args[2])
        x.T.T <: Ref && f === getindex && length(args) == 1 && return only(x.T.fields)
    end
    _jax_is_tuple(x.T) && length(args) >= 2 && return _jax_tuple_element(ctx, x, args[2])
    f !== getfield && _jax_is_array(x.T) && return _jax_index_values(ctx, x, args[2:end])
    _jax_function_call(ctx, f, args)
end
for (f, position) in ((first, 1), (last, 0))
    @eval function _jax_function_call(ctx::_JaxContext, ::typeof($f), args::Vector{_PyVal})
        length(args) == 1 || return _jax_traceable_or_lowered(ctx, $f, args)
        x = only(args)
        if _jax_is_tuple(x.T)
            k = $position == 1 ? 1 : fieldcount(x.T)
            return _jax_tuple_element(ctx, x, _PyVal(string(k), Int, true))
        end
        _jax_is_array(x.T) && ndims(x.T) == 1 && !_jax_ragged(x.T) ||
            return _jax_traceable_or_lowered(ctx, $f, args)
        # A padded vector's last entry is the one at its length.
        isempty(x.extent) || $position == 1 ||
            return _PyVal(_py_atom(x.code) * "[" * _py_infix("-", [x.extent, "1"]) * "]",
                          eltype(x.T))
        _PyVal(_py_atom(x.code) * ($position == 1 ? "[0]" : "[-1]"), eltype(x.T))
    end
end

function _jax_call_positional(ctx::_JaxContext, ::typeof(Core.apply_type), args::Vector{_PyVal})
    T = Core.apply_type(map(a -> _jax_static_parameter(ctx, a), args)...)
    _PyVal("<type>", _JaxCallable(T))
end
# A type parameter known while emitting: a type, function or module, a symbol
# (`Val{:name}`), a tuple of symbols (a keyword call's `NamedTuple{(:a, :b)}`)
# or an integer literal.
function _jax_static_parameter(ctx::_JaxContext, a::_PyVal)
    a.T isa _JaxCallable && return a.T.f
    a.T isa Type && a.T <: Function && Base.issingletontype(a.T) && return a.T.instance
    a.T === Symbol && return _jax_static_symbol(ctx, a)
    if a.static && a.T isa Type && a.T <: Tuple{Vararg{Symbol}}
        ex = Meta.parse(a.code)
        ex isa Expr && ex.head === :tuple && all(x -> x isa String, ex.args) &&
            return Tuple(Symbol.(ex.args))
    end
    m = match(r"^-?[0-9]+$", a.code)
    m === nothing && _jax_error("a type applied to a computed parameter", _jax_where(ctx))
    parse(Int, a.code)
end
_jax_call_positional(ctx::_JaxContext, c::_JaxCallable, args::Vector{_PyVal}) =
    _jax_call(ctx, c.f, args)

# --- programs -----------------------------------------------------------------------

# The Python value of graph value `v` in the current scope chain.
function _jax_lookup(ctx::_JaxContext, g::Graph, v::Value)
    val = _jax_find(ctx, g, v)
    val === nothing && _jax_error("the value `", v.name, "` is read before it is computed")
    val
end
# The `(plan, value)` of an enclosing plan that the nested plan's input `v` is
# (`_JaxScope`'s `links`), or `nothing`.
function _jax_link(ctx::_JaxContext, g::Graph, v::Value)
    id = canon_id(g, v.id)
    scope = ctx.scope
    while scope !== nothing
        haskey(scope.links, id) && return scope.links[id]
        scope = scope.parent
    end
    nothing
end
# The Python value of `v` in the current scope chain, or `nothing`.
function _jax_find(ctx::_JaxContext, g::Graph, v::Value)
    id = canon_id(g, v.id)
    scope = ctx.scope
    while scope !== nothing
        haskey(scope.values, id) && return scope.values[id]
        scope = scope.parent
    end
    nothing
end

function _jax_define!(ctx::_JaxContext, g::Graph, v::Value, val::_PyVal; bind::Bool = true)
    id = canon_id(g, v.id)
    if ctx.host === nothing && !val.static && val.origin === nothing &&
       _jax_integer_data(val.T) && length(ctx.mod.path) == 2
        key = (last(ctx.mod.path[1]).id, last(ctx.mod.path[2]).id)
        key in ctx.mod.cell_data && (val = _jax_with_origin(val, _JaxCellData(key...)))
    end
    if bind
        name = _jax_fresh(ctx, get(ctx.scope.hints, id, v.name))
        _jax_emit!(ctx, name * " = " * val.code)
        val = _PyVal(name, val.T, val.static, val.extent, val.limit, val.padded, val.origin, val.field_extents)
    end
    ctx.scope.values[id] = val
    val
end

# Emit every recipe of plan `p` into the current scope.
function _jax_plan!(ctx::_JaxContext, p::Plan)
    if ctx.host === nothing && !isempty(ctx.mod.regions)
        # Lexical calls also occur in plate/scan bodies and lazy arms. Their
        # own plans keep the call boundary; the outer graph alone cannot see
        # a return value produced inside one of these bodies.
        labels = [:all]
        demand = Dict(r.id => labels for r in p.recipes)
        available = Dict(id => labels for id in keys(ctx.scope.values))
        for v in p.have
            available[canon_id(p.graph, v.id)] = labels
        end
        _jax_demanded_recipes!(ctx, p, demand, labels, "\"all\"", available)
    else
        for recipe in p.recipes
            _jax_plan_recipe!(ctx, p, recipe)
        end
    end
    # The plan's input checks (`InputCheck`), once its values are known: each
    # translates as its function's JAX translation, usually a runtime check.
    for c in p.checks
        _jax_call(ctx, c.f, _PyVal[a isa Value ? _jax_lookup(ctx, p.graph, a) :
                                   _PyVal(repr(a), typeof(a), true) for a in c.args])
    end
end
# Each emitted recipe is on `ctx.mod.path` while it is emitted; an emission
# error records the path where it was raised, innermost recipe last, so the
# caller can tell a plate cell's recipe from its plate (`_jax_cell_failure`).
function _jax_plan_recipe!(ctx::_JaxContext, p::Plan, recipe::Recipe)
    path = ctx.mod.path
    push!(path, (p, recipe))
    try
        _jax_plan_recipe_reasoned!(ctx, p, recipe)
    catch e
        e isa JAXEmissionError && !haskey(ctx.mod.failure_paths, e) &&
            (ctx.mod.failure_paths[e] = copy(path))
        rethrow()
    finally
        pop!(path)
    end
end
function _jax_plan_recipe_reasoned!(ctx::_JaxContext, p::Plan, recipe::Recipe)
    ctx.host === nothing && p.graph === ctx.mod.host_graph &&
        haskey(ctx.mod.host_reasons, recipe.id) || return _jax_plan_recipe_body!(ctx, p, recipe)
    # A data-only recipe `prepare_data` left to the functions: when they cannot
    # compute it either, the error names both reasons, unless a definition
    # gives its values (`_JaxPromote`).
    try
        _jax_plan_recipe_body!(ctx, p, recipe)
    catch e
        e isa JAXEmissionError || rethrow()
        _jax_promotable(ctx.mod.host_definitions, recipe) && throw(_JaxPromote(recipe.id))
        throw(JAXEmissionError(rstrip(e.message, '.') * ". The recipe reads data alone, " *
                               "and `prepare_data` does not compute it on the host either: " *
                               ctx.mod.host_reasons[recipe.id].message))
    end
end
function _jax_plan_recipe_body!(ctx::_JaxContext, p::Plan, recipe::Recipe)
    inputs = _PyVal[_jax_lookup(ctx, p.graph, v) for v in recipe.inputs]
    ctx.recipe = recipe
    first_line = length(ctx.scope.lines) + 1
    outputs = _jax_recipe(ctx, p, recipe, recipe.op, inputs)
    # On the host, a list a recipe ends with is the array it stands for.
    ctx.host === nothing || (outputs = [_host_read(ctx, o) for o in outputs])
    length(outputs) == length(recipe.outputs) ||
        _jax_error("recipe produced ", length(outputs), " values for ",
                   length(recipe.outputs), " outputs", _jax_where(ctx))
    for (v, val) in zip(recipe.outputs, outputs)
        id = canon_id(p.graph, v.id)
        # A collateral output never replaces a value already computed.
        any(s -> haskey(s.values, id), _jax_scope_chain(ctx.scope)) && continue
        _jax_define!(ctx, p.graph, v, val)
    end
    ctx.host === nothing || _host_lines!(ctx, first_line)
end
_jax_scope_chain(scope) = scope === nothing ? _JaxScope[] : [scope; _jax_scope_chain(scope.parent)]

# A multi-output recipe returns a tuple; split it into one value per output.
function _jax_split_outputs(ctx::_JaxContext, recipe::Recipe, v::_PyVal)
    length(recipe.outputs) == 1 && return [v]
    bound = _jax_bind!(ctx, :outputs, v)
    [_PyVal(_py_atom(bound.code) * "[" * string(i - 1) * "]",
            v.T isa Type && v.T <: Tuple ? fieldtype(v.T, i) : Any)
     for i in eachindex(recipe.outputs)]
end

# An operation without a recipe-specific translation.
_jax_recipe(ctx::_JaxContext, p::Plan, recipe::Recipe, op, inputs) =
    _jax_error("the operation `", typeof(op), "` has no JAX translation", _jax_where(ctx))

# A bare function identity (`+`, `sum`, `log`): its call on the inputs.
function _jax_recipe(ctx::_JaxContext, p::Plan, recipe::Recipe, op::Function, inputs)
    # A call whose operands' definitions compose with it (`log(v)` of
    # `v = exp(a)`): the stable form native computes.
    stable = _jax_stable_rewrite(ctx, p, recipe, inputs)
    stable === nothing && return _jax_split_outputs(ctx, recipe, _jax_call(ctx, op, inputs))
    _jax_split_outputs(ctx, recipe, _jax_source_expr(ctx, stable.expr, stable.names,
                                                     stable.values, stable.mod, nothing))
end

# An operation synthesized from authored source: translate that source over
# the recipe's inputs, named as the source names them.
function _jax_recipe(ctx::_JaxContext, p::Plan, recipe::Recipe, op::_KernelSourceOp, inputs)
    f = op.f
    f isa _KernelBranch && return _jax_split_outputs(ctx, recipe,
        _jax_branch(ctx, p, recipe, f, inputs))
    _has_source(recipe) || _jax_error("a source operation without its source", _jax_where(ctx))
    stable = _jax_stable_rewrite(ctx, p, recipe, inputs)
    v = stable === nothing ?
        _jax_source_expr(ctx, recipe.source, _recipe_source_names(recipe), inputs,
                         _jax_source_module(op), _jax_source_captures(op)) :
        _jax_source_expr(ctx, stable.expr, stable.names, stable.values, stable.mod,
                         _jax_source_captures(op))
    _jax_split_outputs(ctx, recipe, v)
end

_jax_source_module(op::_KernelSourceOp) = _kernel_endpoint_source_module(op)
function _jax_closure_module(f)
    f isa Function || return Main
    ms = methods(f)
    isempty(ms) ? parentmodule(typeof(f)) : first(ms).module
end
_jax_source_captures(op::_KernelSourceOp) = _jax_captures(op.f)
_jax_captures(f::_KernelSource) = f.captures
_jax_captures(f::Union{_KernelSplit,_KernelReduction}) = _jax_captures(f.call)
_jax_captures(f) = nothing

# Translate `source` with `names[i]` bound to `inputs[i]`.
function _jax_source_expr(ctx::_JaxContext, source, names, inputs, mod::Module, captures)
    saved = (ctx.locals, ctx.source_module, ctx.captures, ctx.host)
    ctx.locals = Dict{Symbol,_PyVal}(zip(names, inputs))
    ctx.source_module = mod
    ctx.captures = captures
    # On the host a source is a frame of its own: its locals take fresh names.
    ctx.host === nothing || (ctx.host = _HostState())
    try
        return _jax_expr(ctx, source)
    finally
        ctx.locals, ctx.source_module, ctx.captures, ctx.host = saved
    end
end

# A lazy branch recipe: its condition, and one function per arm. An arm that
# carries its own plan (a plate cell's arm, `then_spec`/`else_spec`) is that
# plan; otherwise the arm is its authored source. `a && b` and `a || b` are
# lazy branches whose second operand is an arm.
function _jax_branch(ctx::_JaxContext, p::Plan, recipe::Recipe, b::_KernelBranch, inputs)
    source = recipe.source
    names = _recipe_source_names(recipe)
    source isa Expr && source.head in (:if, :&&, :||) ||
        _jax_error("a branch whose source is not an `if`, `&&` or `||`", _jax_where(ctx))
    then_ex, else_ex = source.head === :if ? (source.args[2], source.args[3]) :
        source.head === :&& ? (source.args[2], false) : (true, source.args[2])
    env = Dict{Symbol,_PyVal}(zip(names, inputs))
    mod = _jax_source_module(recipe.op)
    # The branch's source closes over the values its record captured (a
    # distribution's `prepared standardized` child, say).
    captures = _jax_captures(b.call)
    condition = _jax_source_expr(ctx, source.args[1], names, inputs, mod, captures)
    # Each arm input is the branch's input of its name, for the stable rewrites.
    links = Dict{Symbol,Any}(n => (p, v) for (n, v) in zip(names, recipe.inputs)
                             if count(==(n), names) == 1)
    arm(spec, ex) = spec isa Plan ? () -> _jax_arm_plan(ctx, spec, env, links) :
        () -> _jax_arm_expr(ctx, p, recipe, ex, names, inputs, mod, captures)
    _jax_conditional(ctx, condition, arm(b.then_spec, then_ex), arm(b.else_spec, else_ex))
end

# An arm written as an expression over the branch's arguments, through the
# stable rewrites with the branch's operands' definitions (`log1p(-p)` of
# `p = logistic(a)` before the branch), as native rewrites a leaf arm. It runs
# when its arm is emitted, so the rewrite reads only values computed before
# the branch.
function _jax_arm_expr(ctx::_JaxContext, p::Plan, recipe::Recipe, ex, names, inputs, mod,
                       captures)
    stable = _simplify() && ex isa Expr && length(names) == length(recipe.inputs) ?
        _jax_stable_rewrite(ctx, p, _jax_stable_view(ex, names, mod), recipe.inputs, inputs;
                            alone = true) : nothing
    stable === nothing && return _jax_source_expr(ctx, ex, names, inputs, mod, captures)
    _jax_source_expr(ctx, stable.expr, stable.names, stable.values, stable.mod, captures)
end

# An arm's plan, its HAVE ports bound by name to the branch's arguments and
# linked to the branch's inputs (`links`, by name).
function _jax_arm_plan(ctx::_JaxContext, p::Plan, env::Dict{Symbol,_PyVal},
                       links::Dict{Symbol,Any} = Dict{Symbol,Any}())
    for v in p.have
        haskey(env, v.name) || _jax_error("a branch arm reads `", v.name,
                                          "`, which the branch does not pass", _jax_where(ctx))
        ctx.scope.values[canon_id(p.graph, v.id)] = env[v.name]
        haskey(links, v.name) && (ctx.scope.links[canon_id(p.graph, v.id)] = links[v.name])
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
# A parameter with `statics[i]` true is known while JAX traces (its caller
# passes it a value that is). A padded tuple parameter's `fields[i]` records
# which of its fields its lengths give (`_jax_padded_fields`).
# Returns the function name, the WANT values and which of them return a length.
function _jax_plan_function!(ctx::_JaxContext, hint, p::Plan, types;
                             extents = fill(false, length(p.have)),
                             statics = fill(false, length(p.have)),
                             links = fill(nothing, length(p.have)),
                             origins = fill(nothing, length(p.have)),
                             fields = fill(0, length(p.have)),
                             active_cells::Bool = false)
    name = _jax_fresh(ctx, hint)
    outer = ctx.scope
    inner = _JaxScope(outer)
    inner.indent = outer.indent + 1
    names = String[]
    for (i, v) in enumerate(p.have)
        hint = get(inner.named_hints, v.name, v.name)
        param = _jax_identifier(inner.used, hint)
        push!(names, param)
        extent = ""
        if extents[i]
            extent = _jax_identifier(inner.used, Symbol(hint, :_n))
            push!(names, extent)
        end
        inner.values[canon_id(p.graph, v.id)] = _PyVal(param, types[i], statics[i], extent, "",
                                                       fields[i], origins[i])
        links[i] === nothing || (inner.links[canon_id(p.graph, v.id)] = links[i])
    end
    active = active_cells ? _jax_identifier(inner.used, :active_cell) : ""
    active_cells && push!(names, active)
    recipe = ctx.recipe
    ctx.scope = inner
    wants = try
        if active_cells
            _jax_with_guard(() -> _jax_plan!(ctx, p), inner, active)
        else
            _jax_plan!(ctx, p)
        end
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
    name, wants, Bool[!isempty(w.extent) for w in wants], names
end

# Positions of plate cells' entries (`_JaxOrigin`): one per plate translated.
const _JAX_CELL_POSITIONS = Ref(0)

# A plate: the cell as a function, mapped over the batched arguments with
# `jax.vmap` (one mapped axis per broadcast dimension). Shared arguments
# (`Ref` operands, and scalars, which repeat) are not mapped. A ragged
# argument is mapped row by row with its lengths; a padded group vector maps
# its entries, and the plate's values are padded the same way.
function _jax_recipe(ctx::_JaxContext, p::Plan, recipe::Recipe,
                     op::_AuthoredPlateOp{K,A}, inputs) where {K,A}
    ctx.host === nothing || return _host_plate(ctx, recipe, op, inputs)
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
    # A tuple-valued plate's entries keep which fields have a length: a field
    # without one is `None` in the cell's lengths, and has no extent there.
    cell_fields = Int[!ragged[i] ? 0 : _jax_tuple_valued(v.T) ? _jax_padded_fields(v) :
                      mapped[i] && _jax_ragged(v.T) ? 1 : _jax_padded_axes(v)
                      for (i, v) in enumerate(inputs)]
    # `jax.vmap` hands an argument it does not map (`in_axes` `None`) to the
    # cell unchanged, also under `jax.jit`: a Python or NumPy value stays
    # one, and a `StaticInt` of the prepared data stays static (measured with
    # JAX 0.12.0.1). So a shared argument known while tracing is known in the
    # cell too: a range, an allocation, a loop or a shape test on a prepared
    # integer (`sum(b * x[t] for t in 2:H)`, `fill(b, H)`) is static there as
    # it is outside the plate, also in a recipe that reads a parameter.
    cell_statics = Bool[!mapped[i] && v.static for (i, v) in enumerate(inputs)]
    # A lazy branch in the cell stays a `lax.cond`. Under `vmap`, JAX batches a
    # `lax.cond` whose condition differs between cells by evaluating both arms
    # and selecting; each lane's untaken arm reads its inputs through
    # `stop_gradient`, so a NaN or Inf it computes reaches neither the selected
    # value nor the gradient (measured with JAX 0.12.0.1: the batched jaxpr
    # selects `stop_gradient` inputs per arm; a vmapped `cond(x > 0, log(x), 0)`
    # at x = 0 has gradient 0, where `jnp.where` gives NaN).
    # An unmapped argument the cell reads unchanged is the plate's input, for
    # the stable rewrites (a law's `log(scale)` of a `scale = exp(u)` computed
    # outside the plate).
    cell_links = Any[!mapped[i] && !ragged[i] && cell_types[i] === v.T ? (p, recipe.inputs[i]) :
                     nothing for (i, v) in enumerate(inputs)]
    # A prepared integer array keeps its origin in the cell: shared, as the
    # array; mapped, as its entry at the cell's position, one position for
    # every argument the plate maps (`starts[i]:stops[i]` as `a:b` of a cell
    # over `starts` and `stops`).
    position = string("cell ", _JAX_CELL_POSITIONS[] += 1)
    cell_origins = Any[mapped[i] ? _jax_selected_origin(v, cell_types[i], position) : v.origin
                       for (i, v) in enumerate(inputs)]
    masks = [_jax_padded_mask(v) for v in inputs[padded]]
    masked = any(padded) && all(m -> m !== nothing, masks)
    effects_before = _JAX_EFFECTS[]
    cell, wants, with_length, cell_params = _jax_plan_function!(ctx, _jax_body_hint(ctx, recipe, :cell), body, cell_types;
                                                                extents = cell_extents,
                                                                statics = cell_statics,
                                                                links = cell_links,
                                                                origins = cell_origins,
                                                                fields = cell_fields,
                                                                active_cells = masked)
    # A check cannot be run on padding unless its valid cells are known.
    # Pure cells without checks retain their existing translation.
    any(padded) && !masked && _JAX_EFFECTS[] != effects_before &&
        _jax_error("runtime checks in a padded plate whose valid axes are unknown",
                   _jax_where(ctx))
    want = only(wants)
    E = _jax_typeof(want)
    axes = String[]
    args = String[]
    # Tuple lengths are a structural tuple of columns, not a row array.
    # Selecting one field commutes with vmap; retain that field's own outer
    # expression when translating a cell's length back into this scope.
    field_lengths = Dict{String,Tuple{String,String}}()
    for (i, v) in enumerate(inputs)
        push!(axes, mapped[i] ? "0" : "None"); push!(args, v.code)
        if ragged[i]
            push!(axes, mapped[i] ? "0" : "None"); push!(args, v.extent)
            if _jax_tuple_valued(v.T)
                T = v.T <: Tuple ? v.T : eltype(v.T)
                for k in 1:fieldcount(T)
                    extent = _jax_field_extent(v, k, fieldtype(T, k))
                    isempty(extent) && continue
                    field_lengths[cell_params[length(args)] * "[" * string(k - 1) * "]"] =
                        (extent, last(axes))
                end
            end
        end
    end
    if masked
        # Padding is storage, not a native cell. Include its activity in
        # runtime-check guards before batching those checks across lanes.
        # The mask spans the `R` mapped axes of the argument's stored array
        # (a scan's tuple step outputs: their first column), so each cell
        # receives one flag.
        masks = ["jnp.broadcast_to(" * mask * ", " * _py_atom(_jax_mapped_array(v)) *
                 ".shape[:" * string(R) * "])" for (v, mask) in zip(inputs[padded], masks)]
        push!(axes, "0")
        push!(args, foldl((a, b) -> "jnp.logical_and(" * a * ", " * b * ")", masks))
    end
    axes_text = "(" * join(axes, ", ") * (length(axes) == 1 ? ",)" : ")")
    call = cell
    for _ in 1:R
        call = "jax.vmap(" * call * ", in_axes=" * axes_text * ")"
    end
    call *= "(" * join(args, ", ") * ")"
    # vmap traces its cell even with no mapped entries. An indexed read in
    # that body can then fail on an empty shared array although Julia runs
    # no cells. Shape tests are known to the backend while tracing; branch
    # before calling vmap, without reading or evaluating a cell. Only equal
    # mapped shapes take this path, preserving existing mismatch checks.
    # Scan outputs with tuple-valued cells are stored column by column, and
    # a tuple-valued plate's values field by field; their first column (or
    # field) carries the mapped shape.
    shapes = [_py_atom(_jax_mapped_array(v)) * ".shape[:" * string(R) * "]"
              for (i, v) in enumerate(inputs) if mapped[i]]
    empty = "0 in " * first(shapes)
    for shape in shapes[2:end]
        empty *= " and " * shape * " == " * first(shapes)
    end
    if E <: Tuple && !(E <: NamedTuple)
        R == 1 || _jax_error("a plate of rank ", R, " whose cell returns a tuple", _jax_where(ctx))
        if all(F -> F <: Number || F <: AbstractArray{<:Number}, fieldtypes(E))
            columns = [_PyVal("jnp.zeros((" * join(fill("0", F <: Number ? 1 : ndims(F) + 1),
                ", ") * ",), dtype=" * _jax_dtype(F <: Number ? F : eltype(F)) * ")", F)
                for F in fieldtypes(E)]
            empty_value = _jax_tuple_code(columns)
            if only(with_length)
                lengths = [_PyVal(isempty(_jax_field_extent(want, k, F)) ? "None" :
                    "jnp.zeros((0,), dtype=" * _jax_dtype(Int) * ")", Any)
                    for (k, F) in enumerate(fieldtypes(E))]
                empty_value = "(" * empty_value * ", " * _jax_tuple_code(lengths) * ")"
            end
            call = "(" * empty_value * " if " * empty * " else " * call * ")"
        end
        result = _jax_fresh(ctx, :plate_value)
        if only(with_length)
            lengths = _jax_fresh(ctx, :plate_lengths)
            _jax_emit!(ctx, result * ", " * lengths * " = " * call)
            extents = Tuple(begin
                extent = _jax_field_extent(want, k, fieldtype(E, k))
                outer = _jax_outer_extent(extent, cell_params, args, axes; field_lengths)
                outer === nothing ? lengths * "[" * string(k - 1) * "]" : outer
            end for k in 1:fieldcount(E))
            return [_PyVal(result, Vector{E}, false, lengths, "", _jax_padded_fields(want),
                           nothing, extents)]
        end
        _jax_emit!(ctx, result * " = " * call)
        return [_PyVal(result, Vector{E})]
    end
    # A cell returning a vector (or a square matrix, a law's covariance in a
    # ragged cell) stacks along a new leading axis; a padded cell result
    # carries its length (its size) per cell.
    if E <: AbstractVector || E <: AbstractMatrix
        R == 1 || _jax_error("a plate of rank ", R, " whose cell returns an array", _jax_where(ctx))
        if eltype(E) <: Number
            empty_value = "jnp.zeros((" * join(fill("0", ndims(E) + 1), ", ") *
                          "), dtype=" * _jax_dtype(eltype(E)) * ")"
            only(with_length) && (empty_value = "(" * empty_value *
                ", jnp.zeros((0,), dtype=" * _jax_dtype(Int) * "))")
            call = "(" * empty_value * " if " * empty * " else " * call * ")"
        end
        result = _jax_fresh(ctx, :plate_value)
        lengths = _jax_fresh(ctx, :plate_lengths)
        if only(with_length)
            _jax_emit!(ctx, result * ", " * lengths * " = " * call)
            # The cell's length in terms of the plate's arguments (an input's
            # lengths, or their gather by the plate index) keeps their link
            # to the input, whose flat positions a flattening reads.
            outer = _jax_outer_extent(want.extent, cell_params, args, axes; field_lengths)
            outer === nothing || return [_PyVal(result, Vector{E}, false, outer)]
        else
            _jax_emit!(ctx, result * " = " * call)
            _jax_emit!(ctx, lengths * " = jnp.full((" * result * ".shape[0],), " * result *
                            ".shape[1])")
        end
        return [_PyVal(result, Vector{E}, false, lengths)]
    end
    E <: Number || _jax_error("a plate whose cell returns `", E, "` has no JAX translation yet",
                              _jax_where(ctx))
    call = "(jnp.zeros(" * first(shapes) * ", dtype=" * _jax_dtype(E) * ") if " *
           empty * " else " * call * ")"
    extent = _jax_extent(inputs[padded])
    [_PyVal(call, Array{E,R}, false, extent)]
end
# The stored array whose leading axes are a mapped plate argument's: the
# argument itself, or, for a vector of tuples (a Python tuple of stacked
# arrays, one per field), its first field's.
function _jax_mapped_array(v::_PyVal)
    _jax_columns(v.T) && return _jax_column(v, 1).code
    code, E = v.code, _jax_is_array(v.T) ? eltype(v.T) : Any
    while E isa Type && E <: Tuple && isconcretetype(E) && fieldcount(E) > 0
        code, E = _py_atom(code) * "[0]", fieldtype(E, 1)
    end
    code
end
# A cell's extent expression `extent`, over its parameters `params`, as the
# plate's: each parameter replaced by its argument (`args`, mapped where `axes`
# is "0"), or `nothing` when it reads anything else or reads a mapped argument
# by more than its value (`row.shape`, `row[i]`).
function _jax_outer_extent(extent::AbstractString, params, args, axes;
                           field_lengths = Dict{String,Tuple{String,String}}())
    mapping = Dict(zip(params, zip(args, axes)))
    out = IOBuffer()
    last = 1
    for m in eachmatch(r"(?<![.\w])[A-Za-z_]\w*", extent)
        print(out, SubString(extent, last, prevind(extent, m.offset)))
        last = m.offset + ncodeunits(m.match)
        if m.match == "jnp"
            print(out, m.match)
            continue
        end
        field = match(r"^\[[0-9]+\]", SubString(extent, last))
        key = field === nothing ? "" : m.match * field.match
        if haskey(field_lengths, key)
            arg, axis = field_lengths[key]
            last += ncodeunits(field.match)
        else
            haskey(mapping, m.match) || return nothing
            arg, axis = mapping[m.match]
        end
        next = last <= ncodeunits(extent) ? extent[last] : ' '
        axis == "0" && next in ('.', '[') && return nothing
        print(out, _py_atom(arg))
    end
    print(out, SubString(extent, last))
    String(take!(out))
end

_jax_numeric_namedtuple(@nospecialize(T)) = T isa Type && T <: NamedTuple && isconcretetype(T) &&
    fieldcount(T) > 0 && all(F -> F <: Number, fieldtypes(T))
# Named numbers and numeric vectors (a scan step's output `(lpd, w)` with
# `w::Vector`), stacked field by field.
_jax_stacked_namedtuple(@nospecialize(T)) = T isa Type && T <: NamedTuple &&
    isconcretetype(T) && fieldcount(T) > 0 &&
    all(F -> F <: Number || F <: AbstractVector{<:Number} && isconcretetype(F), fieldtypes(T))
# A vector of named numbers (or numeric vectors) or of numeric tuples is held
# as the tuple of its columns, as `jax.lax.scan` stacks a step's outputs; a
# vector field's column is the dense array of its rows, one per element.
_jax_columns(@nospecialize(T)) = T isa Type && T <: AbstractVector &&
    (_jax_stacked_namedtuple(eltype(T)) || _jax_numeric_tuple(eltype(T)))
function _jax_column(x::_PyVal, k::Integer)
    F = fieldtype(eltype(x.T), k)
    code = _py_atom(x.code) * "[" * string(k - 1) * "]"
    # A vector field's rows all have the stacked width.
    _PyVal(code, Vector{F}, x.static, F <: Number ? x.extent :
           "jnp.full((" * code * ".shape[0],), " * code * ".shape[1])")
end
_jax_numeric_tuple(@nospecialize(T)) = T isa Type && T <: Tuple && isconcretetype(T) && fieldcount(T) > 0 &&
    isconcretetype(eltype(T)) && eltype(T) <: Number
_jax_unref(@nospecialize(T)) = T isa Type && T <: Base.RefValue ? eltype(T) : T
_jax_unref_val(v::_PyVal) = v

# A scan: the step as a function of the carry and one element of each
# sequence, run by `jax.lax.scan`. Shared operands are read by the step's
# closure.
function _jax_recipe(ctx::_JaxContext, p::Plan, recipe::Recipe,
                     op::_AuthoredScanOp{K,A,I,H}, inputs) where {K,A,I,H}
    if H
        ctx.host === nothing ||
            _jax_error("a scan with `history` has no host translation yet", _jax_where(ctx))
        return _jax_history_scan(ctx, recipe, op, inputs)
    end
    ctx.host === nothing || return _host_scan(ctx, recipe, op, inputs)
    body = op.kernel.plan
    iterated = [i for i in 2:length(inputs) if !(i in A)]
    shared = [i for i in 2:length(inputs) if i in A]
    types = Any[inputs[1].T; [eltype(inputs[i].T) for i in iterated]; [inputs[i].T for i in shared]]
    # The step is called with (carry, xs); unpack xs and bind shared operands.
    step = _jax_fresh(ctx, _jax_body_hint(ctx, recipe, :step))
    carry = _jax_fresh(ctx, :carry)
    xs = _jax_fresh(ctx, :xs)
    outer = ctx.scope
    inner = _JaxScope(outer)
    inner.indent = outer.indent + 1
    push!(inner.used, carry, xs)
    have = body.have
    inner.values[canon_id(body.graph, have[1].id)] = _PyVal(carry, types[1])
    unpack = String[]
    entry_position = string("scan ", _JAX_CELL_POSITIONS[] += 1)
    for (k, i) in enumerate(iterated)
        name = _jax_identifier(inner.used, have[1 + k].name)
        push!(unpack, name)
        inner.values[canon_id(body.graph, have[1 + k].id)] =
            _jax_scan_element(name, inputs[i], entry_position)
    end
    for (k, i) in enumerate(shared)
        inner.values[canon_id(body.graph, have[1 + length(iterated) + k].id)] = inputs[i]
    end
    # A sequence padded past a data-dependent length (a group's range, run to
    # its static limit) is scanned to the limit: past the length the carry
    # stays as it was, and the outputs there are padding, which the result's
    # extent masks, as a padded group's loop does.
    padded_inputs = [i for i in iterated if !isempty(inputs[i].extent)]
    extent = isempty(padded_inputs) ? "" : inputs[first(padded_inputs)].extent
    position = isempty(extent) ? "" : _jax_identifier(inner.used, :position)
    isempty(position) || push!(unpack, position)
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
    next_carry = wants[1].code
    if !isempty(position)
        C = _jax_typeof(wants[1])
        live = position * " < " * extent
        # Any carry (a number, an array, the named numbers of several carried
        # values) keeps its previous value past the length, leaf by leaf.
        next_carry = C <: Number || _jax_is_array(C) ?
            "jnp.where(" * live * ", " * wants[1].code * ", " * carry * ")" :
            "jax.tree_util.tree_map(lambda new, old: jnp.where(" * live * ", new, old), " *
            wants[1].code * ", " * carry * ")"
    end
    push!(outer.lines, "    "^outer.indent * "def " * step * "(" * carry * ", " * xs * "):")
    append!(outer.lines, inner.lines)
    push!(outer.lines, "    "^inner.indent * "return " * next_carry * ", " * wants[2].code)
    seqs = String[inputs[i].code for i in iterated]
    isempty(position) || push!(seqs, "jnp.arange(" * _py_atom(inputs[iterated[1]].code) * ".shape[0])")
    sequences = "(" * join(seqs, ", ") * (length(seqs) == 1 ? ",)" : ")")
    result = _jax_fresh(ctx, :scan_result)
    E = _jax_typeof(wants[2])
    call = "jax.lax.scan(" * step * ", " * _jax_scan_init(ctx, inputs[1]).code * ", " *
           sequences * ")[1]"
    # JAX otherwise traces the step even for zero steps, including indexed
    # reads of an empty shared row. Numeric outputs have a known empty shape.
    empty = E <: Number ? _jax_no_values(E, false) :
        _jax_numeric_namedtuple(E) || _jax_numeric_tuple(E) ?
        "(" * join((_jax_no_values(F, false) for F in fieldtypes(E)), ", ") * ",)" : nothing
    if empty === nothing
        _jax_emit!(ctx, result * " = " * call)
    else
        steps = _jax_mapped_array(inputs[first(iterated)]) * ".shape[0]"
        _jax_emit_unless_empty!(ctx, result, call, empty, _jax_nonempty("1", steps))
    end
    if I
        _jax_emit!(ctx, result * " = jnp.concatenate([jnp.atleast_1d(" * inputs[1].code *
                        "), " * result * "])")
        isempty(extent) || (extent = _py_infix("+", [extent, "1"]))
    end
    # A step output of named numbers stacks field by field: the vector of
    # named tuples is the tuple of its columns, as `jax.lax.scan` returns it,
    # and a plate over it maps every column.
    E <: Number || (_jax_numeric_namedtuple(E) || _jax_numeric_tuple(E)) && !I ||
        _jax_stacked_namedtuple(E) && !I && isempty(extent) ||
        _jax_error("a scan whose step outputs `", E, "` has no JAX translation yet",
                   _jax_where(ctx))
    # Padded outputs carry the extent; a vector of named numbers is the tuple
    # of its columns, each padded alike.
    [_PyVal(result, Vector{E}, false, extent)]
end

# One element of a scan's sequence, as its step reads it. An index of
# `eachindex(x)`/`axes(x, d)` (`Base.OneTo`) is at most the sequence's length,
# a static bound, so a loop or generator up to it runs to that bound, masked.
function _jax_scan_element(name::AbstractString, seq::_PyVal, position::String)
    E = eltype(seq.T)
    seq.T isa Type && seq.T <: Base.OneTo ||
        return _jax_with_origin(_PyVal(name, E), _jax_selected_origin(seq, E, position))
    _PyVal(name, E, false, "", _py_atom(seq.code) * ".shape[0]")
end

# A scan with `history` (its step reads the outputs written so far, `earlier`):
# `jax.lax.scan` over the sequences and the step index, carrying the step's
# carry and the result vector. That vector starts as `h₀` filled to the
# sequences' length, or as the leading entries followed by zeros; step `j`
# writes entry `offset + j` (`.at[].set`, stored as the vector's element
# type). Bounded-lag reads use a rolling carry; all other uses retain the
# full vector. Outputs are stacked by scan separately from that carry.
function _jax_history_scan(ctx::_JaxContext, recipe::Recipe,
                           op::_AuthoredScanOp{K,A,I,H}, inputs) where {K,A,I,H}
    body = op.kernel.plan
    history, args = inputs[end], inputs[1:end-1]
    iterated = [i for i in 2:length(args) if !(i in A)]
    shared = [i for i in 2:length(args) if i in A]
    isempty(iterated) && _jax_error("a history scan without a sequence", _jax_where(ctx))
    any(i -> !isempty(args[i].extent), iterated) &&
        _jax_error("a history scan over a padded sequence has no JAX translation yet",
                   _jax_where(ctx))
    HT = history.T
    E = HT isa Type && HT <: Number ? HT :
        HT isa Type && HT <: AbstractVector{<:Number} ? eltype(HT) :
        _jax_error("a scan `history` of type `", HT, "` has no JAX translation yet",
                   _jax_where(ctx))
    dtype = _jax_dtype(E)
    steps = _py_atom(args[iterated[1]].code) * ".shape[0]"
    offset = HT <: Number ? "0" : _py_atom(history.code) * ".shape[0]"
    step = _jax_fresh(ctx, _jax_body_hint(ctx, recipe, :step))
    state = _jax_fresh(ctx, :state)
    xs = _jax_fresh(ctx, :xs)
    outer = ctx.scope
    inner = _JaxScope(outer)
    inner.indent = outer.indent + 1
    push!(inner.used, state, xs)
    carry = _jax_identifier(inner.used, :carry)
    earlier = _jax_identifier(inner.used, :earlier)
    position = _jax_identifier(inner.used, :position)
    history_token = _jax_identifier(inner.used, :history_value)
    layout = _JaxHistoryLayout(history_token, earlier, position, offset,
                               HT <: Number ? history.code : "0", Tuple{String,String,_JaxHistoryIndex}[])
    have = body.have
    inner.values[canon_id(body.graph, have[1].id)] = _PyVal(carry, args[1].T)
    unpack = String[]
    entry_position = string("scan ", _JAX_CELL_POSITIONS[] += 1)
    for (k, i) in enumerate(iterated)
        name = _jax_identifier(inner.used, have[1 + k].name)
        push!(unpack, name)
        inner.values[canon_id(body.graph, have[1 + k].id)] =
            _jax_history_element(ctx, name, args[i], layout, entry_position)
    end
    push!(unpack, position)
    for (k, i) in enumerate(shared)
        inner.values[canon_id(body.graph, have[1 + length(iterated) + k].id)] = args[i]
    end
    inner.values[canon_id(body.graph, have[end].id)] =
        _jax_with_origin(_PyVal(history_token, Vector{E}), layout)
    recipe_saved = ctx.recipe
    ctx.scope = inner
    wants = try
        _jax_emit!(ctx, carry * ", " * earlier * " = " * state)
        _jax_emit!(ctx, join(unpack, ", ") * (length(unpack) == 1 ? "," : "") * " = " * xs)
        _jax_plan!(ctx, body)
        [_jax_lookup(ctx, body.graph, v) for v in body.want]
    finally
        ctx.scope = outer
        ctx.recipe = recipe_saved
    end
    _jax_typeof(wants[2]) <: Number || _jax_error("a history scan whose step outputs `",
        _jax_typeof(wants[2]), "` has no JAX translation yet", _jax_where(ctx))
    storage = _jax_history_layout!(ctx, layout, inner, wants, steps, history, dtype)
    wants = storage.wants
    push!(outer.lines, "    "^outer.indent * "def " * step * "(" * state * ", " * xs * "):")
    append!(outer.lines, inner.lines)
    output = _jax_identifier(inner.used, :output)
    push!(outer.lines, "    "^inner.indent * output * " = jnp.asarray(" * wants[2].code * ", dtype=" * dtype * ")")
    at = _py_infix("+", [offset, position])
    at = "((" * at * ") % " * storage.width * " if " * storage.use_compact * " else " * at * ")"
    push!(outer.lines, "    "^inner.indent * "return (" * wants[1].code * ", " * earlier *
          ".at[" * at * "].set(" * output * ")), " * output)
    seqs = String[[args[i].code for i in iterated]; "jnp.arange(" * steps * ")"]
    result = _jax_fresh(ctx, :scan_result)
    full = _jax_fresh(ctx, :history_result)
    init = _jax_scan_init(ctx, args[1]).code
    outputs = _jax_fresh(ctx, :scan_outputs)
    _jax_emit!(ctx, "(_, " * full * "), " * outputs * " = jax.lax.scan(" * step * ", (" * init *
                    ", " * storage.buffer * "), (" * join(seqs, ", ") * ")) if " * steps *
                    " else ((" * init * ", " * storage.buffer * "), jnp.zeros((0,), dtype=" * dtype * "))")
    compact_result = HT <: Number ? outputs :
        "jnp.concatenate([jnp.asarray(" * history.code * ", dtype=" * dtype * "), " * outputs * "])"
    _jax_emit!(ctx, result * " = " * compact_result * " if " * storage.use_compact * " else " * full)
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
`functions` the [`JAXFunction`](@ref)s it defines. `prepared` maps each value
`prepare_data` returns, by its name in the program, to its key in that
`dict` (a name that is no Python identifier, such as a submodel's
`b_g.z`, has another key). `write(path, src)` saves the module and
`print(src)` shows it.
"""
struct JAXSource
    text::String
    functions::Vector{JAXFunction}
    prepared::Dict{Symbol,String}
end
JAXSource(text::AbstractString, functions) = JAXSource(text, functions, Dict{Symbol,String}())
Base.print(io::IO, src::JAXSource) = print(io, src.text)
Base.show(io::IO, ::MIME"text/plain", src::JAXSource) = print(io, src.text)
Base.write(io::IO, src::JAXSource) = write(io, src.text)
Base.write(path::AbstractString, src::JAXSource) = write(path, src.text)

# The host-side helper that turns a ragged input (one array per group) into
# the padded array and lengths vector the functions take.
const _JAX_PAD_RAGGED = """
def pad_ragged(arrays: list, rank=None) -> tuple:
    \"\"\"Pad arrays along their first axis and return the array and lengths.

    Each row repeats its last entry up to the longest length, so every padded
    slot holds a valid value; the functions mask padded entries by length.
    Pass the two results as the input and its `_lengths` input.
    Trailing dimensions must agree. For an empty list of matrix cells,
    pass `rank=2`; otherwise rank is inferred (or defaults to 1).
    \"\"\"
    import numpy as np
    rows = [np.asarray(a) for a in arrays]
    rank = rows[0].ndim if rank is None and rows else 1 if rank is None else rank
    tail = rows[0].shape[1:] if rows else (1,) * (rank - 1)
    if any(r.ndim != rank or r.shape[1:] != tail for r in rows):
        raise ValueError("ragged cells need their declared rank and common trailing dimensions")
    kinds = [r.dtype for r in rows if r.shape[0] > 0]
    dtype = np.result_type(*kinds) if kinds else np.float64
    rows = [r.astype(dtype) for r in rows]
    lengths = np.asarray([r.shape[0] for r in rows], dtype=np.int64)
    width = max([1] + [int(n) for n in lengths])
    padded = np.stack([np.pad(r, ((0, width - r.shape[0]),) + ((0, 0),) * (rank - 1),
                             mode="edge") if r.shape[0] > 0
                       else np.zeros((width,) + tail, dtype=dtype) for r in rows]) if rows \
        else np.zeros((0, width) + tail, dtype=dtype)
    return padded, lengths
"""

const _JAX_HEADER = """
from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import jax.scipy.special as jsp
from jax.typing import ArrayLike

jax.config.update("jax_enable_x64", True)
"""
const _JAX_X64_LINE = "\njax.config.update(\"jax_enable_x64\", True)\n"
# Whether the module turns on JAX's 64-bit mode: for float64 values, and in a
# float32 module only for the float64 product of `fma`'s once-rounding helper.
# Without it the module's floats are float32 and its integers int32 even
# where a dtype is left to JAX.
_jax_needs_x64(mod::_JaxModule) =
    _jax_float_type() === Float64 || haskey(mod.helper_names, (:helper, "_rk_fma32"))

"""
    jax_source(spec::KernelSpec; have, want, types = nothing, name = :kernel,
               float_type = Float64) -> JAXSource
    jax_source(spec::KernelSpec; have, functions, types = nothing, constants = (;),
               doc = nothing, float_type = Float64) -> JAXSource
    jax_source(p::Plan; types = nothing, name = :kernel, doc = nothing,
               float_type = Float64) -> JAXSource

Translate the program that computes `want` from `have` into a Python module
whose function `name` takes the HAVE values, in order, and returns the WANT
value (a tuple for several WANTs), computed with `jax.numpy`. With
`functions`, a `NamedTuple` of function name => WANT, the module defines one
function per entry over the same HAVE values; a WANT is a value name, a tuple
of them (the function returns a tuple) or a `NamedTuple` of key => value name
(it returns a `dict`). `constants`, a `NamedTuple`, adds module-level
constants (numbers, strings, symbols, vectors, tuples, named tuples and
dictionaries of these).
`function_docs`, a `NamedTuple` of function name => text, gives functions
their docstrings; the others state what they compute. Module functions
annotate their parameters (`ArrayLike`) and results.

The module docstring starts with the generated line `Generated by
ReactiveKernels.jax_source; do not edit.` `doc` adds the author's description
of the program after it, separated by a blank line: a string, or a Julia
docstring such as `@doc TOY_KERNEL` (its Markdown text). A docstring written
above a `const` binding of a kernel belongs to the binding, not to the kernel
value `jax_source` receives, so the caller passes it. Docstrings keep their
text as written: backslashes and quotes are escaped in the Python source.

`types` gives each HAVE value's Julia type, as a `NamedTuple` by port name or
a tuple in HAVE order; by default the ports' declared types are used, and
they must be concrete. Value ranks follow from these types, so the module is
generic in array sizes: one module serves every size of the same program.

`float_type` (`Float64`, the default, or `Float32`) is the dtype of every
float value the program does not type `Float32` itself: Julia's default
`Float64` values, the inputs and data among them, are `float32` in a module
emitted with `float_type = Float32`, for an accelerator whose float64 is slow.
Such a module casts its float inputs to `float32`, `prepare_data` returns
`float32` data, and the dtype metadata says so. It does not turn on JAX's
64-bit mode (`jax_enable_x64`), so its device integers are JAX's default
(`int32`), unless it rounds an `fma` once, which takes a float64 product; it
stays float32 where another module turned that mode on. Its results carry
float32 rounding: compare it with the `Float64` module of the same program at
a tolerance relative to `eps(Float32)` and the size of the value.

The translation keeps the program's structure. A plate is `jax.vmap` of its
cell, a scan is `jax.lax.scan` of its step, a lazy branch is `jax.lax.cond`
of its arms, and nothing is unrolled per element. Arrays keep Julia's
dimension order, Julia's broadcasting rules are kept, and integer index data
stay 1-based. Gradients are `jax.grad` of a function. A construct without a
translation raises a [`JAXEmissionError`](@ref) naming its recipe.

A `for` loop, generator, comprehension or slice over a range takes a static
number of steps, so `jax.grad` reverses it. Static ends and a stop that is a
padded row's length have one. So do ends that are entries of integer arrays
of the prepared data (`data`): for `starts[r]:stops[r]`, entries of two
arrays at one position (also a plate cell's entries of both), it is the
longest such range; for one end that is data, the largest stop less the least
start. `prepare_data` computes these bounds and returns them with the data,
under keys starting with `_rk_`. A range from or to an element of another
such range, or of a static range (`for j in i:n`), is bounded by that range's
ends. A step past a range's stop is a lazy branch (`jax.lax.cond`) that
leaves the loop's values unchanged and reads nothing, so Julia's values and
derivatives, empty ranges included, are kept; the work is the bound for
every range, so a long range among short ones costs each of them its length.
A start that is a traced input, such as a positional `Int`, has no static
bound and raises. A loop or generator over no elements runs no JAX loop, so
its body's reads from empty arrays are never traced.
"""
function jax_source(spec::KernelSpec; have = _KERNEL_DEFAULT_BOUNDARY,
                    want = nothing, functions = nothing, types = nothing,
                    name = :kernel, constants = NamedTuple(), doc = nothing, data = nothing,
                    definitions = nothing, definitions_module::Module = Main, keep = (),
                    function_docs = NamedTuple(), float_type = Float64,
                    source_regions = _JaxSourceRegion[], module_docstring = nothing)
    docstring = _jax_module_docstring(doc, module_docstring)
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
    _jax_with_float_type(float_type) do
        _jax_module_source(plans, types, constants, docstring; data, definitions,
                           definitions_module, keep, function_docs, source_regions)
    end
end

jax_source(p::Plan; types = nothing, name = :kernel, constants = NamedTuple(),
           doc = nothing, data = nothing, definitions = nothing,
           definitions_module::Module = Main, keep = (), function_docs = NamedTuple(),
           float_type = Float64) =
    _jax_with_float_type(float_type) do
        _jax_module_source([Symbol(name) => (p, Tuple(v.name for v in p.want))],
                           types, constants, _jax_module_docstring(doc, nothing);
                           data, definitions, definitions_module, keep, function_docs)
    end

# The module docstring: the generated line naming the generator, the author's
# description `doc` (a string, or docstring Markdown such as `@doc X`, as its
# text), then the generator's further `sections`, separated by blank lines.
function _jax_compose_docstring(generator::AbstractString, doc, sections::AbstractString...)
    # Leading blank lines go; a first line's indentation (a docstring's
    # indented signature) stays.
    authored = doc === nothing ? "" : replace(rstrip(string(doc)), r"\A(?:[ \t]*\n)+" => "")
    join(filter(!isempty, String[generator, authored, sections...]), "\n\n")
end

# `module_docstring` (internal) is a complete docstring a frontend composed
# with `_jax_compose_docstring` under its own generator line, with its own
# sections (ReactiveKernelsPPL's data names and statement docstrings).
function _jax_module_docstring(doc, module_docstring)
    module_docstring === nothing &&
        return _jax_compose_docstring("Generated by ReactiveKernels.jax_source; do not edit.", doc)
    doc === nothing || throw(ArgumentError(
        "jax_source: `module_docstring` is the complete module docstring; pass it or `doc`, not both"))
    String(module_docstring)
end

_jax_want_names(w::Symbol) = (w,)
_jax_want_names(w::Tuple) = w
# A `NamedTuple` WANT may nest (`(lv = (sigma = :x, level = :y), s = :s)`): the
# function returns nested `dict`s; its values are the leaves, in order.
_jax_want_names(w::NamedTuple) = Tuple(n for v in values(w) for n in _jax_want_names(v))
_jax_want_names(w) = throw(ArgumentError(
    "jax_source: a function's WANT is a value name, a tuple of names or a NamedTuple; got `$w`"))

# With `data`, the module's data-only prefix is `prepare_data` (`jax_host.jl`);
# a recipe without a host translation is left to the JAX functions.
#
# `definitions`, an ordered collection of `name => expression`, computes data
# HAVE values on the host from the other data HAVE values, before the prefix:
# the module's `prepare_data` then takes the data those expressions read in
# place of the values they define (a binder's data-only definitions, which
# would otherwise be values only Julia computes). A definition that reads its
# own name refines the raw value the caller passes under that name, which the
# definitions before it read. A definition without a host
# translation, or one given as a `JAXEmissionError` (the reason it has none),
# stays a `prepare_data` input, named with that reason in the module constant
# `UNTRANSLATED`. The definitions' globals (a call head such as `plate`, a
# bare name no definition or data value binds) resolve in
# `definitions_module`: a binder passes the module it evaluates them in, so
# the module it emits does not depend on what its caller imported. `keep`
# names data-only values `prepare_data` also returns (the functions take them
# too), for a caller that reads them.
function _jax_module_source(plans, types, constants, docstring::String; data = nothing,
                            definitions = nothing, definitions_module::Module = Main,
                            keep = (), function_docs = NamedTuple(),
                            source_regions = _JaxSourceRegion[])
    if data === nothing
        definitions === nothing || isempty(definitions) || throw(ArgumentError(
            "jax_source: `definitions` compute data values, so they need `data`"))
        isempty(keep) || throw(ArgumentError(
            "jax_source: `keep` names values of `prepare_data`, so it needs `data`"))
        return _jax_module_text(plans, types, constants, docstring, nothing;
                                function_docs, source_regions)
    end
    data isa NamedTuple || data isa Tuple && all(d -> d isa Symbol, data) ||
        throw(ArgumentError("jax_source: `data` is a tuple of HAVE names or a NamedTuple " *
                            "of their values"))
    if !isempty(keep)
        p = first(plans).second[1]
        ports = data isa NamedTuple ? keys(data) : data
        bound = Value[v for v in p.have if v.name in ports]
        retained = Value[]
        for name in keep
            push!(retained, _jax_keep_value(p.graph, name))
        end
        # Retained metadata has its own data-only cut. The function cuts may
        # omit it, or select an inverse producer that reads a live parameter.
        # A `nothing` WANT shape marks this as a host cut, not a Python function.
        kept = try
            plan(p.graph; have = bound, want = retained)
        catch e
            e isa PlanningError || rethrow()
            throw(ArgumentError("jax_source: `keep` values must be computable from data alone: " *
                                sprint(showerror, e)))
        end
        plans = [plans; gensym(:keep) => (kept, nothing)]
    end
    # A definition of a data value replaces that `prepare_data` input; one of
    # another name is a host local the later definitions read.
    defs = definitions === nothing ? nothing :
        Pair{Symbol,Any}[Symbol(first(d)) => last(d) for d in definitions]
    excluded = Set{Int}()
    # Why each excluded recipe has no host translation: the module's comment
    # names it, and so does an error the functions then raise in that recipe.
    reasons = Dict{Int,JAXEmissionError}()
    # A definition given as a `JAXEmissionError` (its reason) is not translated.
    untranslated = Dict{Symbol,JAXEmissionError}(n => ex for (n, ex) in something(defs, ())
                                                 if ex isa JAXEmissionError)
    # Data-only recipes whose values the definitions give instead, as data
    # inputs of the program: each one's recipe has no host translation, and
    # the functions cannot compute it either, or a value `keep` names reads it
    # (`g = prep(labels)`, whose field `g.n` sizes a block).
    promoted = Set{Int}()
    defined = Set{Symbol}(first(d) for d in something(defs, ()))
    # A plate cell's data-only recipes the functions cannot compute, by plate
    # recipe id: `prepare_data` computes them for every cell
    # (`_jax_cell_split`, jax_host.jl). Those without a host translation
    # either, with the reason, by (plate, cell recipe).
    cells = Dict{Int,Set{Int}}()
    cell_reasons = Dict{Tuple{Int,Int},JAXEmissionError}()
    while true
        try
            return _jax_module_text(plans, types, constants, docstring,
                                    (data, excluded, defs, untranslated, keep, reasons,
                                     promoted, cells, cell_reasons, definitions_module);
                                    function_docs, source_regions)
        catch e
            if e isa _HostUnsupportedDefinition && !haskey(untranslated, e.name)
                untranslated[e.name] = e.error
            elseif e isa _HostUnsupported && !(e.recipe in excluded)
                push!(excluded, e.recipe)
                reasons[e.recipe] = e.error
            elseif e isa _JaxPromote && !(e.recipe in promoted)
                push!(promoted, e.recipe)
            elseif e isa _KeepUnavailable
                more = setdiff(_jax_upstream_recipes(plans, e.name, excluded, defined), promoted)
                if isempty(more)
                    upstream = sort!(collect(_jax_upstream_recipes(plans, e.name, excluded, nothing)))
                    for id in upstream
                        haskey(reasons, id) || continue
                        throw(JAXEmissionError("the retained data value `$(e.name)` has no host " *
                            "translation: " * reasons[id].message))
                    end
                    throw(ArgumentError(
                        "jax_source: `keep` names `$(e.name)`, which is no data-only value of the program"))
                end
                union!(promoted, more)
            elseif e isa _CellHostNeeded
                # Each cell recipe is split off once; one that fails again is
                # the functions' error.
                (e.recipe in get(cells, e.plate, ()) ||
                 haskey(cell_reasons, (e.plate, e.recipe))) && throw(e.error)
                push!(get!(Set{Int}, cells, e.plate), e.recipe)
            elseif e isa _HostUnsupportedCell
                pending = get(cells, e.plate, Set{Int}())
                any(r -> r in pending, e.recipes) || throw(e.error)
                for r in e.recipes
                    delete!(pending, r)
                    cell_reasons[(e.plate, r)] = e.error
                end
            else
                rethrow()
            end
        end
    end
end

# Keep names refer to authored values, including structural-CSE aliases whose
# producer and prepared argument use a different name.
function _jax_keep_value(g::Graph, name::Symbol)
    found = [v for v in values(g.values) if v.name === name]
    isempty(found) && throw(ArgumentError("jax_source: unknown `keep` value `$name`"))
    only(found)
end

# A data-only recipe left to the functions (`_HostUnsupported`) that they
# cannot compute either, and whose values all have definitions.
struct _JaxPromote <: Exception
    recipe::Int
end
# A value `keep` names that no data-only recipe computes.
struct _KeepUnavailable <: Exception
    name::Symbol
end
_jax_promotable(defined, r::Recipe) = !isempty(r.outputs) && all(o -> o.name in defined, r.outputs)

# The `excluded` recipes upstream of the values named `name` in `plans` whose
# values all have definitions (`defined`), or all excluded upstream recipes
# when `defined === nothing` (to report the underlying host failure).
function _jax_upstream_recipes(plans, name::Symbol, excluded, defined)
    found = Set{Int}()
    for (_, (p, _)) in plans
        g = p.graph
        producer = Dict{ValueId,Recipe}()
        for r in p.recipes, o in r.outputs
            get!(producer, canon_id(g, o.id), r)
        end
        queue = [canon_id(g, _jax_keep_value(g, name).id)]
        seen = Set{ValueId}()
        while !isempty(queue)
            id = pop!(queue)
            id in seen && continue
            push!(seen, id)
            r = get(producer, id, nothing)
            r === nothing && continue
            r.id in excluded && (defined === nothing || _jax_promotable(defined, r)) && push!(found, r.id)
            foreach(v -> push!(queue, canon_id(g, v.id)), r.inputs)
        end
    end
    found
end

const _JAX_HOST_HEADER = """
import numpy as np
import scipy.special as ssp
"""

function _jax_module_text(plans, types, constants, docstring::String, prepare;
                          function_docs = NamedTuple(), source_regions = _JaxSourceRegion[])
    mod = _JaxModule()
    append!(mod.regions, source_regions)
    union!(mod.used, ("jax", "jnp", "jsp", "np", "ssp", "scipy"))
    bodies = String[]
    functions = JAXFunction[]
    if prepare === nothing
        _jax_functions!(mod, bodies, functions, plans, types; function_docs)
    else
        data, excluded, defs, untranslated, keep, reasons, promoted, cells, cell_reasons,
            mod.definitions_module = prepare
        merge!(mod.host_reasons, reasons)
        merge!(mod.cell_reasons, cell_reasons)
        mod.host_graph = first(plans).second[1].graph
        defs === nothing || union!(mod.host_definitions, (first(d) for d in defs))
        meta = _jax_prepared_functions!(mod, bodies, functions, plans, types, data, excluded,
                                        defs, untranslated; keep, function_docs, promoted, cells)
        constants = merge(constants, meta)
    end
    constant_lines = String[]
    for (key, value) in pairs(constants)
        cname = _jax_identifier(mod.used, key)
        push!(constant_lines, cname * " = " * _jax_py_literal(value))
    end
    prepare === nothing && any(f -> any(io -> _jax_ragged(last(io)), f.inputs), functions) &&
        pushfirst!(mod.helpers, rstrip(replace(_JAX_PAD_RAGGED, "np.float64" => "np." * _jax_float_name())))
    lines = String[_jax_docstring(docstring, "")]
    header = rstrip(_jax_needs_x64(mod) ? _JAX_HEADER : replace(_JAX_HEADER, _JAX_X64_LINE => ""))
    # The host code: `prepare_data`, and static values the functions compute
    # with NumPy.
    host = prepare !== nothing || any(b -> occursin(r"(?<![.\w])(?:np|ssp|scipy)\.",
                                                    replace(b, _PY_STRING => "\"\"")),
                                      [mod.helpers; bodies])
    if host
        header = rstrip(_JAX_HOST_HEADER) * "\n" * header
        any(b -> occursin("scipy.linalg.", b), [mod.helpers; bodies]) &&
            (header = "import scipy.linalg\n" * header)
    end
    push!(lines, header)
    isempty(constant_lines) || (push!(lines, ""); append!(lines, constant_lines))
    for block in [mod.helpers; bodies]
        push!(lines, "")
        push!(lines, "")
        push!(lines, block)
    end
    prepared = Dict{Symbol,String}(k[2] => v for (k, v) in mod.helper_names
                                   if k isa Tuple && length(k) == 2 && k[1] === :prepared_name)
    JAXSource(join(lines, "\n") * "\n", functions, prepared)
end

# The symbols an expression reads, in order of first appearance (a call's
# function included; callers keep the ones they know). A field read `g.n`
# reads `g`; a dotted call `cos.(angle)` reads `cos` and its arguments.
function _jax_symbols_read!(out::Vector{Symbol}, ex)
    if ex isa Symbol
        ex in out || push!(out, ex)
    elseif ex isa Expr
        field = ex.head === :. && length(ex.args) == 2 && ex.args[2] isa QuoteNode
        foreach(a -> _jax_symbols_read!(out, a), field ? ex.args[1:1] : ex.args)
    end
    out
end

# `prepare_data` and the functions over its results; returns the metadata
# constants `INPUTS` (the functions' parameters) and `PREPARED` (each
# prepared input's dtype, rank and raggedness).
function _jax_prepared_functions!(mod::_JaxModule, bodies, functions, plans, types, data,
                                  excluded, defs = nothing,
                                  untranslated = Dict{Symbol,JAXEmissionError}(); keep = (),
                                  function_docs = NamedTuple(), promoted = Set{Int}(),
                                  cells = Dict{Int,Set{Int}}())
    ports = data isa NamedTuple ? Set(keys(data)) : Set(data)
    # A promoted recipe's values (`_JaxPromote`) are data inputs of every
    # function, which its definitions give: the recipe is not run.
    if !isempty(promoted)
        values = Value[]
        seen = Set{ValueId}()
        for (_, (p, _)) in plans, r in p.recipes
            r.id in promoted || continue
            for o in r.outputs
                cid = canon_id(p.graph, o.id)
                cid in seen && continue
                push!(seen, cid)
                push!(values, o)
            end
        end
        plans = [fname => (_partial_subplan(p, [p.have; values], p.want,
                                            Recipe[r for r in p.recipes if !(r.id in promoted)]), w)
                 for (fname, (p, w)) in plans]
        ports = union(ports, (v.name for v in values))
    end
    split = _jax_prepare_split(plans, ports, excluded; keep, cells)
    p1 = first(plans).second[1]
    # Raw data enters host code, so its types need not be concrete (a column
    # with missing values); the functions' inputs must be.
    have_types = Dict{Symbol,Any}()
    for (k, v) in enumerate(p1.have)
        have_types[v.name] = data isa NamedTuple && haskey(data, v.name) ? typeof(data[v.name]) :
            types isa NamedTuple && haskey(types, v.name) ? types[v.name] :
            types isa Tuple && k <= length(types) ? types[k] : valtype(v)
    end
    for v in split.remaining
        _concrete_value_type(have_types[v.name]) || throw(ArgumentError(
            "jax_source: the HAVE value `$(v.name)` has type `$(have_types[v.name])`; pass a " *
            "concrete type with `types`"))
    end
    # The definitions the host computes, in order: those with a translation
    # that a data input reads, directly or through other definitions (one of
    # a value the program computes is read only once it is promoted).
    needed = Set{Symbol}(v.name for v in split.raw)
    if defs !== nothing
        by_name = Dict{Symbol,Any}(defs)
        queue = collect(needed)
        while !isempty(queue)
            nm = pop!(queue)
            haskey(untranslated, nm) && continue
            ex = get(by_name, nm, nothing)
            ex === nothing && continue
            for r in _jax_symbols_read!(Symbol[], ex)
                haskey(by_name, r) && !(r in needed) && (push!(needed, r); push!(queue, r))
            end
        end
    end
    computed = defs === nothing ? Pair{Symbol,Any}[] :
        Pair{Symbol,Any}[d for d in defs if !haskey(untranslated, first(d)) && first(d) in needed]
    defined = Set{Symbol}(first.(computed))
    # A definition that reads its own name refines the value the caller
    # passes under that name (`y => fill(y)`, a response whose missing entries
    # are `None`): that raw value stays a parameter, taken as the caller gives
    # it, and the definitions before the refinement read it.
    refined = Set{Symbol}(n for (n, ex) in computed if n in _jax_symbols_read!(Symbol[], ex))
    params = Any[v for v in split.raw if !(v.name in defined) || v.name in refined]
    raw_types = Any[v.name in refined ? Any : have_types[v.name] for v in params]
    # A definition may read a raw value the caller passes that is no input of
    # the program (`_rkppl_value_r => :r`, when the program reads `r` only
    # through its bound copy): a parameter too, typed by its value in `data`.
    if data isa NamedTuple
        inputs = Set{Symbol}(v.name for v in p1.have)
        extra = Symbol[]
        for (_, ex) in computed
            _jax_symbols_read!(extra, ex)
        end
        for r in extra
            r in inputs || r in defined || !haskey(data, r) || any(v -> v.name === r, params) ||
                (push!(params, (; name = r)); push!(raw_types, typeof(data[r])))
        end
    end
    used = Set{String}()
    remaining_names = [_jax_param_name(used, v.name) for v in split.remaining]
    prepared_names = String[]
    for v in split.constants
        name = _jax_key_name(used, v.name)
        push!(used, name * "_lengths", name * "_flat")
        push!(prepared_names, name)
        mod.helper_names[(:prepared_name, v.name)] = name
    end
    for name in keep
        cid = canon_id(p1.graph, _jax_keep_value(p1.graph, name).id)
        index = findfirst(v -> canon_id(p1.graph, v.id) == cid, split.constants)
        index === nothing || (mod.helper_names[(:prepared_name, name)] = prepared_names[index])
    end
    # The boundary values' types: those of one binding's values, when `data`
    # gives them; else those the translation infers. Plate cells' values that
    # only `prepare_data` computes (`_JaxCellPrefix`) take the translation's.
    native = nothing
    if data isa NamedTuple && !isempty(split.constants)
        g = split.plan.graph
        hosted = Set{ValueId}(canon_id(g, o.id) for r in split.prefix
                              if r.op isa _JaxCellPrefix for o in r.outputs)
        computed_natively = Value[v for v in split.constants if !(canon_id(g, v.id) in hosted)]
        native_plan = isempty(hosted) ? split.plan :
            _partial_subplan(split.plan, split.raw, computed_natively,
                             Recipe[r for r in split.prefix if !(r.op isa _JaxCellPrefix)])
        native = try
            values = isempty(computed_natively) ? () :
                _prepare_plan(native_plan)((data[v.name] for v in split.raw)...)
            found = Dict{ValueId,Any}(canon_id(g, v.id) => typeof(x) for (v, x) in
                zip(computed_natively, _partial_prefix_values(computed_natively, values)))
            Any[get(found, canon_id(g, v.id), nothing) for v in split.constants]
        catch
            nothing
        end
    end
    lines, prepared_types, param_names = _jax_prepare_function!(mod, split, raw_types,
        prepared_names, native;
        params, definitions = [(n, ex, get(have_types, n, nothing)) for (n, ex) in computed],
        untranslated)
    push!(bodies, join(lines, "\n"))
    prepare_lines, prepare_body = lines, length(bodies)
    # The inputs by the names `prepare_data` takes them (a name that is no
    # Python identifier, such as a submodel's `v.mask`, is spelled `v_mask`).
    push!(functions, JAXFunction("prepare_data",
        [n => T for (n, T) in zip(param_names, raw_types)],
        [n => T for (n, T) in zip(prepared_names, prepared_types)]))
    prepare_function = length(functions)
    all_types = Any[[have_types[v.name] for v in split.remaining]; prepared_types]
    all_names = [remaining_names; prepared_names]
    for (_, (rp, _)) in split.residuals
        _jax_cell_data!(mod, split, rp)
    end
    try
        _jax_functions!(mod, bodies, functions, split.residuals, Tuple(all_types);
                        names = all_names, function_docs,
                        keyword_from = length(remaining_names))
    catch e
        e isa JAXEmissionError || rethrow()
        # A plate cell's data-only recipe the functions cannot compute:
        # `prepare_data` computes it for every cell instead.
        for (_, (rp, _)) in split.residuals
            failure = _jax_cell_failure(mod, split, rp, e)
            failure === nothing || throw(failure)
        end
        rethrow()
    end
    # Static bounds of prepared integer arrays that ranges over their entries
    # read (`_jax_trip_bound`): `prepare_data` computes and returns them too.
    bounds = _jax_prepared_bounds(mod)
    if !isempty(bounds)
        bodies[prepare_body] = join(_jax_prepare_bounds_lines(mod, prepare_lines, bounds), "\n")
        f = functions[prepare_function]
        functions[prepare_function] = JAXFunction(f.name, f.inputs,
            [f.outputs; [name => Int for (name, _, _) in bounds]])
        append!(prepared_names, (name for (name, _, _) in bounds))
        prepared_types = Any[prepared_types; fill(Int, length(bounds))]
    end
    meta = (INPUTS = Tuple(first(io) for io in functions[2].inputs),
            PREPARED = [n => (dtype = _jax_dtype_name(T), ndim = _jax_meta_rank(T),
                              ragged = _jax_ragged(T))
                        for (n, T) in zip(prepared_names, prepared_types)])
    defs === nothing && return meta
    # With definitions, `DATA` describes `prepare_data`'s own parameters, and
    # `UNTRANSLATED` the definitions it takes as inputs (none when every
    # definition has a host translation).
    merge(meta, (DATA = [n => (dtype = _jax_dtype_name(T), ndim = _jax_meta_rank(T),
                               ragged = _jax_ragged(T))
                         for (n, T) in zip(param_names, raw_types)],
                 UNTRANSLATED = Dict{String,String}(String(n) => untranslated[n].message
                     for (n, _) in defs if haskey(untranslated, n) &&
                         any(v -> v.name === n, params))))
end

# Plans with the same boundary and compatible producer assignments can share
# one executable body. Different choices of a multi-output producer remain
# separate: choosing a cheaper cut for one entry must not change another.
function _jax_compatible_plans(a::Plan, b::Plan)
    a.graph === b.graph && a.have == b.have || return false
    # Host preparation can replace a recipe's operator and inputs while
    # retaining its graph id. Only the same executable recipe can be shared.
    all(!haskey(b.producer, id) || (b.producer[id].id == r.id &&
        b.producer[id].inputs == r.inputs && b.producer[id].op === r.op)
        for (id, r) in a.producer)
end

function _jax_functions!(mod, bodies, functions, plans, types;
                         names = nothing, keyword_from = nothing,
                         function_docs = NamedTuple())
    public_names = Dict(fname => _jax_identifier(mod.used, fname) for (fname, _) in plans)
    groups = Vector{eltype(plans)}[]
    for entry in plans
        index = findfirst(group -> all(_jax_compatible_plans(entry.second[1], e.second[1])
                                       for e in group), groups)
        index === nothing ? push!(groups, [entry]) : push!(groups[index], entry)
    end
    emitted = Dict{Symbol,JAXFunction}()
    for group in groups
        if length(group) == 1 && isempty(mod.regions)
            fname, (p, w) = only(group)
            lines, f = _jax_function!(mod, fname, p, types, w; names, keyword_from,
                name_override = public_names[fname],
                                      doc = get(function_docs, fname, nothing))
            push!(bodies, join(lines, "\n"))
            emitted[fname] = f
            continue
        end
        first_plan = first(group).second[1]
        g = first_plan.graph
        producer = merge((p.producer for (_, (p, _)) in group)...)
        selected = unique(Int[r.id for (_, (p, _)) in group for r in p.recipes])
        recipes = Dict(r.id => r for (_, (p, _)) in group for r in p.recipes)
        order = _topo(g, selected, Set(canon_id(g, v.id) for v in first_plan.have))
        order === nothing && error("compatible JAX plans have no joint execution order")
        all_wants = unique(v -> canon_id(g, v.id), Value[v for (_, (q, _)) in group for v in q.want])
        checks = unique(InputCheck[c for (_, (q, _)) in group for c in q.checks])
        p = Plan(g, first_plan.have, all_wants, Recipe[recipes[r] for r in order],
                 producer, 0.0, Recipe[], checks)
        labels = [fname for (fname, _) in group]
        selections = Dict(fname => Set(r.id for r in q.recipes) for (fname, (q, _)) in group)
        demand = Dict(r.id => [fname for fname in labels if r.id in selections[fname]]
                      for r in p.recipes)
        available = Dict(canon_id(g, v.id) => labels for v in p.have)
        outputs = Dict{Symbol,Vector{_PyVal}}()
        selector = "_entry"
        input_names = names === nothing ? String.(getfield.(p.have, :name)) : names
        while selector in input_names
            selector *= "_"
        end
        # The checks every entry runs first, in one order, run once before
        # the entries' own blocks: at the same point of each entry's run.
        entry_checks = [e.second[1].checks for e in group]
        shared_checks = 0
        while all(cs -> length(cs) > shared_checks, entry_checks) &&
              allequal(cs[shared_checks + 1] for cs in entry_checks)
            shared_checks += 1
        end
        emit = function(ctx)
            _jax_demanded_recipes!(ctx, p, demand, labels, selector, available)
            for c in first(group).second[1].checks[1:shared_checks]
                _jax_call(ctx, c.f, _PyVal[a isa Value ? _jax_lookup(ctx, g, a) :
                    _PyVal(repr(a), typeof(a), true) for a in c.args])
            end
            for (fname, (q, w)) in group
                _jax_emit!(ctx, "if " * selector * " == " * repr(String(fname)) * ":")
                ctx.scope.indent += 1
                for c in q.checks[shared_checks + 1:end]
                    _jax_call(ctx, c.f, _PyVal[a isa Value ? _jax_lookup(ctx, g, a) :
                        _PyVal(repr(a), typeof(a), true) for a in c.args])
                end
                wants = [_jax_returned(ctx, _jax_lookup(ctx, g, v); arrays = true) for v in q.want]
                outputs[fname] = wants
                _jax_emit!(ctx, "return " * _jax_result_code(w, wants))
                ctx.scope.indent -= 1
            end
            nothing, [_jax_lookup(ctx, g, v) for v in p.want]
        end
        lines, shared = _jax_function!(mod, :_model, p, types, first(group).second[2];
            names, keyword_from, emit, selector,
            doc = "Shared model computation. The generated entry wrappers select a static output.")
        push!(bodies, join(lines, "\n"))
        shown = keyword_from === nothing ? shared.inputs : shared.inputs[1:keyword_from]
        signature = [pn * ": " * _jax_annotation(T) for (pn, T) in shown]
        args = String[first(io) for io in shown]
        if keyword_from !== nothing
            push!(signature, "**prepared: ArrayLike")
            push!(args, "**prepared")
        end
        for (fname, (q, w)) in group
            wants = outputs[fname]
            name = public_names[fname]
            wrapper = ["def " * name * "(" * join(signature, ", ") * ") -> " *
                       _jax_return_annotation(w, wants) * ":",
                _jax_docstring(something(get(function_docs, fname, nothing), _jax_default_doc(w)), "    "),
                "    return " * shared.name * "(" * join([repr(String(fname)); args], ", ") * ")"]
            push!(bodies, join(wrapper, "\n"))
            emitted[fname] = JAXFunction(name, copy(shared.inputs),
                [String(v.name) => x.T for (v, x) in zip(q.want, wants)])
        end
    end
    append!(functions, (emitted[fname] for (fname, _) in plans))
end

# A float32 module casts its float inputs, so a caller's float64 NumPy arrays
# and scalars, strongly typed in JAX, do not turn its arithmetic float64.
function _jax_input_dtype(@nospecialize(T))
    _jax_float_type() === Float64 && return ""
    E = T
    while E isa Type && E <: AbstractArray
        E = eltype(E)
    end
    E isa Type && E <: AbstractFloat ? ", dtype=" * _jax_dtype(E) : ""
end

# A value's dtype and rank, as module metadata describes an input.
_jax_dtype_name(::Type{<:AbstractArray{T}}) where {T} = _jax_dtype_name(T)
_jax_dtype_name(::Type{<:AbstractArray{<:AbstractArray{T}}}) where {T} = _jax_dtype_name(T)
_jax_dtype_name(::Type{Bool}) = "bool"
_jax_dtype_name(::Type{<:Integer}) = "int64"
_jax_dtype_name(::Type{Float32}) = "float32"
# Every other float is the module's float type (`jax_source`'s `float_type`).
_jax_dtype_name(::Type{<:AbstractFloat}) = _jax_float_name()
# Non-numeric values (level labels) cross as Python objects.
_jax_dtype_name(@nospecialize(T)) = "object"
_jax_meta_rank(@nospecialize(T)) = T isa Type && T <: AbstractArray ? ndims(T) : 0

# One function `fname` over plan `p`, returning WANT `w` in its shape.
function _jax_function!(mod::_JaxModule, fname, p::Plan, types, w; reserve::Bool = true,
                        host::Bool = false, names = nothing, doc = nothing,
                        keyword_from::Union{Nothing,Int} = nothing,
                        emit = nothing, selector::Union{Nothing,String} = nothing,
                        name_override::Union{Nothing,AbstractString} = nothing,
                        active::Bool = false, traced_branches::Int = 0)
    argtypes = _jax_have_types(p, types)
    name = name_override === nothing ?
        (reserve ? _jax_identifier(mod.used, fname) : String(fname)) : name_override
    scope = _JaxScope()
    selector === nothing || push!(scope.used, selector)
    ctx = _JaxContext(mod, scope, nothing, Main, Dict{Symbol,_PyVal}(), nothing)
    host && (ctx.host = _HostState())
    ctx.traced_branches = traced_branches
    params = String[]
    inputs = Pair{String,Any}[]
    # Each input's Python local, beside its key in `inputs`. A prepared input
    # (after the first `keyword_from`) is read from `**prepared` by its key
    # (`_jax_key_name`) into a local, respelled when Python reserves the key.
    input_locals = String[]
    for (k, (v, T)) in enumerate(zip(p.have, argtypes))
        keyed = names !== nothing && keyword_from !== nothing && k > keyword_from
        key = names === nothing ? _jax_identifier(scope.used, v.name) :
            keyed ? names[k] : (push!(scope.used, names[k]); names[k])
        param = keyed ? _jax_param_name(scope.used, key; stable = false) : key
        push!(params, param); push!(inputs, key => T); push!(input_locals, param)
        if host
            # Host inputs are NumPy arrays; a ragged input is a list of them.
            scope.values[canon_id(p.graph, v.id)] = _PyVal(param, T, true)
            _jax_ragged(T) ? _jax_emit!(ctx, param * " = [np.asarray(a) for a in " * param * "]") :
                _jax_is_array(T) && eltype(T) <: Number &&
                _jax_emit!(ctx, param * " = np.asarray(" * param * ")")
            continue
        end
        extent = flat = ""
        if _jax_ragged(T)
            extent_key = names === nothing ? "" : key * "_lengths"
            extent = names === nothing ? _jax_identifier(scope.used, Symbol(v.name, :_lengths)) :
                keyed ? _jax_param_name(scope.used, extent_key; stable = false) :
                (push!(scope.used, extent_key); extent_key)
            push!(params, extent); push!(inputs, (names === nothing ? extent : extent_key) => Vector{Int})
            push!(input_locals, extent)
            if names !== nothing
                # A prepared ragged input's flat positions; their count, the
                # total length, is the static length of its flattening.
                flat_key = key * "_flat"
                flat = keyed ? _jax_param_name(scope.used, flat_key; stable = false) :
                    (push!(scope.used, flat_key); flat_key)
                push!(params, flat); push!(inputs, flat_key => Vector{Int}); push!(input_locals, flat)
                scope.flat_positions[extent] = flat
            end
        end
        # A prepared integer is a `StaticInt` (`prepare_data`), an `int` under
        # `jax.jit` too, so it is known while tracing (a shape test on it
        # decides then).
        static = names !== nothing && T isa Type && T <: Integer
        # A prepared integer array: ranges over its entries have static
        # bounds that `prepare_data` computes (`_jax_trip_bound`).
        prepared = keyword_from !== nothing && k > keyword_from &&
            T isa Type && T <: AbstractArray && _jax_integer_data(T)
        # Its origin names the array by its key in the prepared data, which
        # `prepare_data` reads to compute the bound.
        scope.values[canon_id(p.graph, v.id)] = _PyVal(param, T, static, extent, "", 0,
            prepared ? _JaxOrigin(key, nothing, "0") : nothing)
        # Callers may pass NumPy arrays; JAX indexes only its own arrays by a
        # traced index.
        if _jax_is_array(T) && (eltype(T) <: Number || _jax_ragged(T))
            _jax_emit!(ctx, param * " = jnp.asarray(" * param * _jax_input_dtype(T) * ")")
            isempty(extent) || _jax_emit!(ctx, extent * " = jnp.asarray(" * extent * ")")
            haskey(scope.flat_positions, extent) && names !== nothing &&
                _jax_emit!(ctx, flat * " = jnp.asarray(" * flat * ")")
        elseif T isa Type && T <: AbstractFloat && !isempty(_jax_input_dtype(T))
            _jax_emit!(ctx, param * " = jnp.asarray(" * param * _jax_input_dtype(T) * ")")
        end
    end
    guard = active ? _jax_identifier(scope.used, :active) : nothing
    active && push!(params, guard)
    returned, wants = _jax_with_function_guard(scope, guard) do
        if emit === nothing
            _jax_plan!(ctx, p)
            values = [_jax_returned(ctx, _jax_lookup(ctx, p.graph, v); arrays = reserve && !host)
                      for v in p.want]
            _jax_result_code(w, values), values
        else
            emit(ctx)
        end
    end
    # A module function is annotated and documented; a helper is not. With
    # `keyword_from`, the inputs after the first `keyword_from` are the
    # prepared data, taken as `**prepared` (user GO `08jy2dh`): the signature
    # names only what the caller supplies, and the body reads its inputs.
    if reserve
        shown = keyword_from === nothing ? inputs : inputs[1:keyword_from]
        signature = [pn * ": " * _jax_annotation(T) for (pn, T) in shown]
        selector === nothing || pushfirst!(signature, selector * ": str")
        keyword_from === nothing || push!(signature, "**prepared: ArrayLike")
        annotation = selector === nothing ? _jax_return_annotation(w, wants) : "Any"
        lines = String["def " * name * "(" * join(signature, ", ") * ") -> " * annotation * ":"]
        push!(lines, _jax_docstring(something(doc, _jax_default_doc(w)), "    "))
        if keyword_from !== nothing
            # The static bounds of prepared arrays the body reads (ranges over
            # their entries), which `prepare_data` returns with the data.
            body = join(scope.lines, "\n")
            for (bn, _, _) in _jax_prepared_bounds(mod)
                occursin(Regex("\\b" * bn * "\\b"), body) || continue
                push!(inputs, bn => Int); push!(input_locals, bn)
            end
            for ((pn, _), local_name) in zip(inputs[keyword_from+1:end], input_locals[keyword_from+1:end])
                push!(lines, "    " * local_name * " = prepared[\"" * pn * "\"]")
            end
        end
    else
        lines = String["def " * name * "(" * join(params, ", ") * "):"]
    end
    append!(lines, scope.lines)
    returned === nothing || push!(lines, "    return " * returned)
    lines, JAXFunction(name, inputs, [String(v.name) => x.T for (v, x) in zip(p.want, wants)])
end

_jax_result_code(w, wants) = w isa NamedTuple ?
    _jax_nested_dict(w, Iterators.Stateful(wants)) :
    w isa Symbol ? only(wants).code : _jax_tuple_code(wants)

# Python annotations of a module function's parameters and result: numbers
# and arrays are JAX's `ArrayLike` (NumPy arrays and Python numbers too), a
# function's result is a `jax.Array`, or the `dict`/`tuple` it returns.
_jax_annotation(@nospecialize(T)) = T isa Type &&
    (T <: Number || T <: AbstractArray && (eltype(T) <: Number || _jax_ragged(T))) ?
    "ArrayLike" : "Any"
function _jax_return_annotation(w, wants)
    w isa NamedTuple && return "dict"
    w isa Symbol || return "tuple"
    T = only(wants).T
    T isa Type && T <: NamedTuple ? "dict" : T isa Type && T <: Tuple ? "tuple" :
        T isa Type && (T <: Number || T <: AbstractArray) ? "jax.Array" : "Any"
end
_jax_default_doc(w::Symbol) = "Compute `" * string(w) * "` from the inputs."
_jax_default_doc(w::Tuple) = "Compute " * join(("`" * string(x) * "`" for x in w), ", ") *
    " from the inputs, as a tuple."
_jax_default_doc(w::NamedTuple) = "Compute a dict of " *
    join(("`" * string(k) * "`" for k in keys(w)), ", ") * " from the inputs."
# A Python docstring at `indent`, its text kept as written. Backslashes are
# escaped (an authored `\theta` is no tab, `\sigma` no invalid escape), and so
# is a quote that could close the string: one followed by another quote, or
# the text's last character.
function _jax_docstring(text::AbstractString, indent::AbstractString)
    body = replace(replace(rstrip(text), "\\" => "\\\\"), r"\"(?=\"|\z)" => "\\\"")
    replace(indent * "\"\"\"" * replace(body, "\n" => "\n" * indent) *
            (occursin('\n', body) ? "\n" * indent : "") * "\"\"\"", r"\n[ ]+\n" => "\n\n")
end

# A (nested) `NamedTuple` WANT as `dict`s of the values `wants` yields in order.
# A padded (ragged) value comes with its lengths, under `<key>_lengths`.
function _jax_nested_dict(w::NamedTuple, wants)
    entries = String[]
    for (k, v) in pairs(w)
        if v isa NamedTuple
            push!(entries, "\"" * string(k) * "\": " * _jax_nested_dict(v, wants))
            continue
        end
        x = popfirst!(wants)
        push!(entries, "\"" * string(k) * "\": " * x.code)
        isempty(x.extent) || push!(entries, "\"" * string(k) * "_lengths\": " * x.extent)
    end
    "{" * join(entries, ", ") * "}"
end

# Public numerical results are arrays, including constants and nested leaves.
# Helpers and host preparation keep static scalars usable as shapes and bounds.
# A returned named tuple is a `dict` keyed by its field names.
function _jax_returned(ctx::_JaxContext, v::_PyVal; arrays::Bool = false)
    v.T === Union{} && return v # an unconditional diagnostic has no returned fields
    _jax_translation_only(v) && _jax_error("returning a callable or reference as a module output ",
                                          "has no JAX translation", _jax_where(ctx))
    if arrays && v.T isa Type &&
            (v.T <: Number || v.T <: AbstractArray && eltype(v.T) <: Number)
        dtype = v.T <: Union{Bool,Integer,AbstractFloat} ? ", dtype=" * _jax_dtype(v.T) :
            _jax_input_dtype(v.T)
        return _PyVal("jnp.asarray(" * v.code * dtype * ")", v.T, false,
                      v.extent, v.limit, v.padded, v.origin, v.field_extents)
    elseif arrays && _jax_is_tuple(v.T)
        bound = _jax_bind!(ctx, :result, v)
        fields = [_jax_returned(ctx, _jax_tuple_element(ctx, bound, _PyVal(string(i), Int));
                               arrays = true) for i in 1:fieldcount(v.T)]
        code = v.T <: NamedTuple ?
            "{" * join((repr(string(n)) * ": " * x.code
                        for (n, x) in zip(fieldnames(v.T), fields)), ", ") * "}" :
            _jax_tuple_code(fields)
        return _PyVal(code, v.T, false, v.extent, v.limit, v.padded, v.origin, v.field_extents)
    end
    v.T isa Type && v.T <: NamedTuple && fieldcount(v.T) > 0 || return v
    bound = _jax_bind!(ctx, :result, v)
    names = fieldnames(v.T)
    _PyVal("{" * join(("\"" * string(n) * "\": " * _py_atom(bound.code) * "[" * string(i - 1) * "]"
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
