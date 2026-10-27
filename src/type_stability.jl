# Preparation-time type-stability check.
#
# RK tries to prepare programs in which every value infers a concrete type. A
# client function whose result Julia infers as a non-concrete type (`Any`, an
# abstract type, a `Union`, or an array of such elements) breaks that from the
# inside: every recipe and plate cell that reads the value dispatches
# dynamically on each evaluation, and native reverse-mode AD can refuse the
# program long into its derivative compilation (snag
# `joint-nlme-enzym-f8d3cb5f`, where one `Any` result made every downstream
# plate dynamic). By default preparation does not fail on such a kernel. It
# warns once per value, naming the value and the recipe that computes it (user
# decision
# `ReactiveKernels:snag.joint-nlme-enzym-f8d3cb5f/decisions/2026-10-12T20-13-15-811-1hui2qi`,
# principle review `…/2026-10-12T20-09-11-062-114x8kz`). A project or a call
# opts into refusing it instead, `type_stability = :error`: a warning scrolls
# past in a long log (snag `type-stability-c-7eeae56d`, user decision
# `ReactiveKernels:snag.type-stability-c-7eeae56d/decisions/2026-10-27T20-42-40-959-1jkb0is`).
#
# The check walks the plan's recipes in execution order from concrete input
# types, inferring each recipe's output types with `promote_op` and recursing
# into plate cells, scan steps and embedded kernels. A recipe whose inputs are
# all concrete but whose output is not is the source of the instability; values
# computed from it, and values computed from an input whose type is not known,
# are not reported.

# Whether a value of type `T` is concretely typed: a concrete type whose array
# elements are concretely typed too (`Vector{Any}` is a concrete container of
# dynamically typed elements).
_concrete_value_type(@nospecialize(T)) =
    T isa DataType && isconcretetype(T) &&
    (!(T <: AbstractArray) || _concrete_value_type(eltype(T)))
# A type used as a value (`T = eltype(x)`) is a singleton of `Type{T}`.
_concrete_value_type(::Type{Type{T}}) where {T} = true

# The value types of a plan, keyed by canonical value id: the HAVE ports'
# `argument_types`, then each recipe's outputs in execution order.
# `recipe_types(recipe, input_types)` gives a recipe's output types from its
# input types, where an unknown input is `Any`.
function _plan_value_types(recipe_types, p::Plan, argument_types)
    types = Dict{ValueId,Any}(canon_id(p.graph, v.id) => T
                          for (v, T) in zip(p.have, argument_types))
    for recipe in p.recipes
        input_types = Any[get(types, canon_id(p.graph, v.id), Any)
                          for v in recipe.inputs]
        for (output, T) in zip(recipe.outputs, recipe_types(recipe, input_types))
            types[canon_id(p.graph, output.id)] = T
        end
    end
    types
end

# The per-output types of a recipe whose operation returns `T`: a multi-output
# operation returns a tuple.
_recipe_output_types(recipe::Recipe, @nospecialize(T)) =
    ntuple(length(recipe.outputs)) do index
        length(recipe.outputs) == 1 ? T :
            T isa DataType && T <: Tuple ? fieldtype(T, index) : Any
    end

# The step operand types of a scan without `history` from its argument types
# `(init, operand...)`: the carry seed, one element of each iterated sequence,
# then the shared operands (`_scan_step_input_types` emits the same types as
# expressions for the generated body).
function _scan_step_argument_types(::_AuthoredScanOp{K,A}, types) where {K,A}
    iterated = [i for i in 2:length(types) if !(i in A)]
    shared = [i for i in 2:length(types) if i in A]
    Any[types[1], (eltype(types[i]) for i in iterated)..., types[shared]...]
end

# The result vector type of a scan without `history` whose step output type
# is `output`.
_scan_result_type(op::_AuthoredScanOp, types, @nospecialize(output)) =
    Vector{_scan_includes_init(op) ? promote_type(types[1], output) : output}

"""
    _UninferredValue

A value of a prepared plan whose inferred type is not concrete although every
input of the recipe that computes it is. `path` names the plate, scan or
embedded-kernel values that enclose the recipe, outermost first.
"""
struct _UninferredValue
    path::Vector{Symbol}
    recipe::Recipe
    value::Value
    input_types::Vector{Any}
    inferred::Any
end

# The type of a recipe's result inferred from concrete input types, or
# `nothing` when inference proves no result (the call always throws, or no
# method is visible in this world): such a value is not reported.
function _inferred_recipe_type(op, input_types)
    T = _promote_op(op, input_types...)
    T === Union{} ? nothing : T
end

# A value whose type is not inferred from its inputs keeps its declaration
# where the generated body declares typed locals (`_declare_typed_output!`),
# and is otherwise unknown.
_declared_value_type(declares::Bool, recipe::Recipe, value::Value) =
    declares && !(recipe.op isa _BoundConstant) &&
    _concrete_value_type(valtype(value)) ? valtype(value) : Any

# Append the uninferred values of plan `p` over HAVE types `argument_types` to
# `found` and return the plan's value types. `declares` is whether the
# generated body declares the plan's typed locals; the body of a plate cell
# does not.
function _uninferred_values!(found::Vector{_UninferredValue}, p::Plan,
                             argument_types, path::Vector{Symbol}, declares::Bool)
    _plan_value_types(p, argument_types) do recipe, input_types
        if !all(_concrete_value_type, input_types)
            return map(v -> _declared_value_type(declares, recipe, v), recipe.outputs)
        end
        # Bound data has its own runtime type; it is an input, not inferred.
        # Traced data runs the tensorized product, which this check does not
        # type.
        if recipe.op isa _BoundConstant
            value = recipe.op.value
            return (_kernel_source_arg_style(value) === Val(:native) ?
                    typeof(value) : Any,)
        end
        before = length(found)
        T = _checked_recipe_type!(found, recipe.op, input_types,
                                  [path; first(recipe.outputs).name])
        T === nothing &&
            return map(v -> _declared_value_type(declares, recipe, v), recipe.outputs)
        map(recipe.outputs, _recipe_output_types(recipe, T)) do value, inferred
            declared = _declared_value_type(declares, recipe, value)
            declared === Any || return declared
            _concrete_value_type(inferred) && return inferred
            length(found) == before && push!(found, _UninferredValue(
                path, recipe, value, input_types, inferred))
            inferred
        end
    end
end

# The result type of operation `op` from concrete input types, after checking
# the body it encloses (a plate cell, a scan step or an embedded kernel) under
# `path`. An enclosing operation's result is typed from its body's walk, as
# its generated body types it, never by inferring the operation's own call:
# that call is a semantic fallback the lowering does not run, and inferring it
# for a wide plate (34 operands) took minutes and tens of gigabytes. `nothing`
# leaves the result unknown.
_checked_recipe_type!(found, op, input_types, path) =
    _inferred_recipe_type(op, input_types)

# The element type of a plate's result: the cell's inferred result, or its
# declared type (`_plate_result_eltype`), as the generated body types the
# buffer (`_plate_body_types!`).
function _checked_plate_element!(found, op::_AuthoredPlateOp{K,A}, input_types,
                                 path) where {K,A}
    cell_types = Any[_plate_argument_type(T, position in A)
                     for (position, T) in enumerate(input_types)]
    body = op.kernel.plan
    types = _uninferred_values!(found, body, cell_types, path, false)
    want = only(body.want)
    _plate_result_eltype(get(types, canon_id(body.graph, want.id), Any), valtype(want))
end

# A plate's pointwise buffer as the native lowering allocates it:
# `_plate_similar_output` over the first batched argument whose type is an
# axis, the element type and the `_plate_broadcast_shape` of the batched
# arguments' axes. A batched argument whose type does not decide whether it is
# an axis leaves the marker to run time, and the result unknown. The shape is
# inferred through RK's fold rather than Base's `combine_axes`, whose splat
# recursion took minutes to infer over a 34-operand plate.
function _checked_recipe_type!(found, op::_AuthoredPlateOp{K,A}, input_types,
                               path) where {K,A}
    element = _checked_plate_element!(found, op, input_types, path)
    _concrete_value_type(element) || return nothing
    batched = Any[T for (position, T) in enumerate(input_types) if !(position in A)]
    marker = nothing
    for T in batched
        class = _static_plate_axis_class(T)
        class === :ambiguous && return nothing
        class === :axis && (marker = T; break)
    end
    marker === nothing && return nothing
    # The generated body's nested type query shares this same allocation
    # inference; neither path infers the wide semantic-fallback operation.
    T = _plate_array_type(element, Val(A), input_types...)
    _concrete_value_type(T) ? T : nothing
end

function _checked_recipe_type!(found, op::_AuthoredPlateCellOp, input_types, path)
    element = _checked_plate_element!(found, op.plate, input_types[2:end], path)
    _concrete_value_type(element) ? element : nothing
end

function _checked_recipe_type!(found, op::_AuthoredScanOp, input_types, path)
    # A history scan's result holds values of the fill's type, or the
    # leading entries' element type.
    _scan_has_history(op) && return _scan_history_result_type(input_types[end])
    step = op.kernel.plan
    types = _uninferred_values!(found, step,
        _scan_step_argument_types(op, input_types), path,
        _needs_embedded_tensorization(step))
    output = get(types, canon_id(step.graph, step.want[2].id), Any)
    _concrete_value_type(output) ? _scan_result_type(op, input_types, output) : nothing
end

# An embedded kernel returns its single WANT, or a tuple of its WANTs.
function _checked_recipe_type!(found, op::PreparedKernel, input_types, path)
    p = op.plan
    types = _uninferred_values!(found, p, input_types, path,
                                _needs_embedded_tensorization(p))
    wants = Any[get(types, canon_id(p.graph, v.id), Any) for v in p.want]
    all(_concrete_value_type, wants) || return nothing
    length(wants) == 1 ? only(wants) : Tuple{wants...}
end

"""
    _uninferred_values(kernel, argument_types) -> Vector{_UninferredValue}

The values of a prepared kernel whose types are not inferred concretely when
its HAVE ports have `argument_types`, one per recipe that introduces such a
type. HAVE ports whose types are not concrete leave what they reach unchecked.
"""
function _uninferred_values(kernel, argument_types)
    p = kernel.plan
    found = _UninferredValue[]
    length(p.have) == length(argument_types) || return found
    _uninferred_values!(found, p, collect(Any, argument_types), Symbol[],
                        _needs_embedded_tensorization(p))
    found
end

# The value as the author knows it: an anonymous cell result is the cell value
# of its plate.
function _uninferred_value_label(u::_UninferredValue)
    enclosing = "`" * join(string.(reverse(u.path)), "` within `") * "`"
    u.value.name === _KERNEL_PLATE_VALUE_PORT && !isempty(u.path) &&
        return "The cell value of " * enclosing
    isempty(u.path) ? "`$(u.value.name)`" : "`$(u.value.name)` in the body of $enclosing"
end

# The recipe computing the value; a cell value's recipe is the cell's result
# expression.
function _uninferred_recipe_text(u::_UninferredValue)
    line = _recipe_line(u.recipe)
    prefix = "$(_KERNEL_PLATE_VALUE_PORT) = "
    u.value.name === _KERNEL_PLATE_VALUE_PORT && startswith(line, prefix) ?
        line[ncodeunits(prefix)+1:end] : line
end

function _uninferred_value_message(u::_UninferredValue)
    inputs = join(string.(u.input_types), ", ")
    "ReactiveKernels: $(_uninferred_value_label(u)) is not concretely typed. " *
    "Its recipe `$(_uninferred_recipe_text(u))` infers `$(u.inferred)` from " *
    "input types ($inputs). Recipes and plate cells that read it dispatch " *
    "at run time on every evaluation, and native reverse-mode AD may refuse " *
    "the prepared program. Make the called function's result inferable, or " *
    "assert its type inside that function (`f(args...) = (...)::T`)."
end

"""
    UninferredValuesError(messages)

Thrown by a preparation checked with `type_stability = :error` when values of
the prepared program do not infer concretely. `messages` holds one message per
recipe that introduces such a type, as the warning of the default
`type_stability = :warn` words it.
"""
struct UninferredValuesError <: Exception
    messages::Vector{String}
end

function Base.showerror(io::IO, e::UninferredValuesError)
    n = length(e.messages)
    print(io, "UninferredValuesError: ", n, n == 1 ? " value" : " values",
          " of the prepared program", n == 1 ? " does" : " do",
          " not infer a concrete type, and this preparation is checked with ",
          "`type_stability = :error`.")
    foreach(message -> print(io, "\n\n", message), e.messages)
end

# What preparation does with an uninferred value: `:warn` logs it once per
# process and continues; `:error` refuses the preparation.
const _TYPE_STABILITY_MODES = (:warn, :error)

_type_stability_mode(mode::Symbol) = mode in _TYPE_STABILITY_MODES ? mode :
    throw(ArgumentError("`type_stability` must be one of " *
                        "$(join(repr.(_TYPE_STABILITY_MODES), ", ")); got $(repr(mode))"))
_type_stability_mode(mode) = throw(ArgumentError(
    "`type_stability` must be a Symbol, one of " *
    "$(join(repr.(_TYPE_STABILITY_MODES), ", ")); got $(repr(mode))"))

# The project's `type_stability` preference, `[ReactiveKernels]
# type_stability = "error"` in its `LocalPreferences.toml`, read at the first
# preparation of a process: a read takes tens of microseconds, as long as
# rebinding a small bound kernel. It is not compiled in, so a new session picks
# up a change without recompiling. `__init__` forgets a value read while an
# image was generated.
const _TYPE_STABILITY_PREFERENCE = Ref{Union{Nothing,Symbol}}(nothing)

function _type_stability_preference()
    mode = _TYPE_STABILITY_PREFERENCE[]
    mode === nothing || return mode
    value = Preferences.load_preference(@__MODULE__, "type_stability", "warn")
    value in ("warn", "error") || throw(ArgumentError(
        "the ReactiveKernels preference `type_stability` must be \"warn\" or " *
        "\"error\"; got $(repr(value))"))
    _TYPE_STABILITY_PREFERENCE[] = Symbol(value)
end

# The task-local key of the mode a `type_stability` keyword sets for the
# preparations its call makes, nested ones included (a child kernel's, a bound
# residual's).
const _TYPE_STABILITY_SCOPE = :ReactiveKernels_type_stability

# The mode of the preparation in progress: the innermost enclosing
# `type_stability` keyword, else the preference.
_type_stability() =
    get(_type_stability_preference, task_local_storage(), _TYPE_STABILITY_SCOPE)

# `f()` with every preparation inside it checked in `mode`; `nothing` keeps the
# enclosing mode.
_with_type_stability(f, ::Nothing) = f()
_with_type_stability(f, mode) =
    task_local_storage(f, _TYPE_STABILITY_SCOPE, _type_stability_mode(mode))

_report_uninferred_values(found, mode::Symbol) =
    _report_uninferred_values(found, Val(mode))

function _report_uninferred_values(found, ::Val{:warn})
    for u in found
        @warn _uninferred_value_message(u) _group = :type_stability maxlog = 1 _id = (
            :uninferred_value, u.path, u.value.name, _uninferred_recipe_text(u),
            u.input_types)
    end
end

_report_uninferred_values(found, ::Val{:error}) =
    isempty(found) || throw(UninferredValuesError(map(_uninferred_value_message, found)))

# Check the values of `kernel` over `argument_types` in the mode of the
# preparation in progress: warn once per process for each uninferred value and
# continue, or refuse the preparation, naming every uninferred value.
function _check_uninferred_values(kernel, argument_types)
    mode = _type_stability()
    _report_uninferred_values(_uninferred_values(kernel, argument_types), mode)
    kernel
end

# Prepared kernels whose HAVE ports have concrete declared types are checked
# from those declarations.
_check_uninferred_values(kernel) =
    _check_uninferred_values(kernel, map(valtype, kernel.plan.have))
