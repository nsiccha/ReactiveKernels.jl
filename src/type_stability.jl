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
        recipe.op isa _BoundConstant && return (typeof(recipe.op.value),)
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
        _owns_array_regions(step))
    output = get(types, canon_id(step.graph, step.want[2].id), Any)
    _concrete_value_type(output) ? _scan_result_type(op, input_types, output) : nothing
end

# An embedded kernel returns its single WANT, or a tuple of its WANTs.
function _checked_recipe_type!(found, op::PreparedKernel, input_types, path)
    p = op.plan
    types = _uninferred_values!(found, p, input_types, path,
                                _owns_array_regions(p))
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
Base.@nospecializeinfer function _uninferred_values(@nospecialize(kernel), argument_types)
    p = getfield(kernel, :plan)::Plan
    found = _UninferredValue[]
    length(p.have) == length(argument_types) || return found
    _uninferred_values!(found, p, collect(Any, argument_types), Symbol[],
                        _owns_array_regions(p))
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
    TypeInstabilityError(messages)

Thrown by a preparation checked with `type_stability = :error` when the
prepared program is not type stable at the checked `type_stability_level`.
`messages` holds one message per finding, as the warning of the default
`type_stability = :warn` words it: a value whose type the program does not
infer concretely, or a method whose code holds non-concrete values or calls
dispatched at run time.
"""
struct TypeInstabilityError <: Exception
    messages::Vector{String}
end

function Base.showerror(io::IO, e::TypeInstabilityError)
    n = length(e.messages)
    print(io, "TypeInstabilityError: the prepared program is not type stable (",
          n, n == 1 ? " finding" : " findings",
          "), and this preparation is checked with `type_stability = :error`.")
    foreach(message -> print(io, "\n\n", message), e.messages)
end

# The type-stability settings. Each is a keyword of `prepare`, `prepare!` and
# every `prepare_ad*`, and a ReactiveKernels preference of the same name; its
# first choice is the default.
#
# - `type_stability`: what preparation does with a finding. `:warn` logs it
#   once per process and continues; `:error` refuses the preparation (snag
#   `type-stability-c-7eeae56d`, user decision `1jkb0is`).
# - `type_stability_level`: how far preparation checks (user decision
#   `ReactiveKernels:snag.type-stability-c-7eeae56d/decisions/2026-10-27T21-12-20-388-04ks7hd`:
#   "everything", in levels "either everything rk, or everything everything").
#   `:values` infers the type of every value of the program. `:rk` also audits
#   the code RK prepares, its generated body and the functions of
#   ReactiveKernels' packages it calls; every other function is a leaf, whose
#   call must be dispatched statically and return a concrete type. `:all` also
#   audits every function that code calls, except Julia's own (Base, Core and
#   the standard libraries), which stay leaves.
_type_stability_choices(::Val{:type_stability}) = (:warn, :error)
_type_stability_choices(::Val{:type_stability_level}) = (:all, :rk, :values)

function _type_stability_choice(setting::Val{S}, value) where {S}
    choices = _type_stability_choices(setting)
    value isa Symbol && value in choices && return value
    throw(ArgumentError("`$S` must be one of $(join(repr.(choices), ", ")); " *
                        "got $(repr(value))"))
end

# The project's preferences (`[ReactiveKernels] type_stability = "error"` in its
# `LocalPreferences.toml`), read at the first preparation of a process: a read
# takes tens of microseconds, as long as rebinding a small bound kernel. They
# are not compiled in, so a new session picks up a change without recompiling.
# `__init__` forgets values read while an image was generated.
const _TYPE_STABILITY_PREFERENCES =
    (Ref{Union{Nothing,Symbol}}(nothing), Ref{Union{Nothing,Symbol}}(nothing))
_type_stability_preference_cache(::Val{:type_stability}) = _TYPE_STABILITY_PREFERENCES[1]
_type_stability_preference_cache(::Val{:type_stability_level}) =
    _TYPE_STABILITY_PREFERENCES[2]

function _type_stability_preference(setting::Val{S}) where {S}
    cache = _type_stability_preference_cache(setting)
    cached = cache[]
    cached === nothing || return cached
    choices = string.(_type_stability_choices(setting))
    value = Preferences.load_preference(@__MODULE__, string(S), first(choices))
    value in choices || throw(ArgumentError(
        "the ReactiveKernels preference `$S` must be one of " *
        "$(join(repr.(choices), ", ")); got $(repr(value))"))
    cache[] = Symbol(value)
end

# The task-local keys of the settings a call's keywords give the preparations it
# makes, nested ones included (a child kernel's, a bound residual's).
_type_stability_scope(::Val{:type_stability}) = :ReactiveKernels_type_stability
_type_stability_scope(::Val{:type_stability_level}) = :ReactiveKernels_type_stability_level

# A setting of the preparation in progress: the innermost enclosing keyword,
# else the preference.
_type_stability_setting(setting::Val) = get(() -> _type_stability_preference(setting),
                                            task_local_storage(),
                                            _type_stability_scope(setting))
_type_stability() = _type_stability_setting(Val(:type_stability))
_type_stability_level() = _type_stability_setting(Val(:type_stability_level))

# `f()` with every preparation inside it checked with the given settings;
# `nothing` keeps the enclosing one.
_with_type_stability(f, mode, level) =
    _with_type_stability_setting(Val(:type_stability), mode) do
        _with_type_stability_setting(f, Val(:type_stability_level), level)
    end
_with_type_stability_setting(f, ::Val, ::Nothing) = f()
_with_type_stability_setting(f, setting::Val, value) =
    task_local_storage(f, _type_stability_scope(setting),
                       _type_stability_choice(setting, value))

# Julia's own modules: Base, Core and the standard libraries.
function _julia_own_module(m::Module)
    root = Base.moduleroot(m)
    (root === Base || root === Core) && return true
    path = pathof(root)
    path !== nothing && startswith(path, Sys.STDLIB)
end

# A module of ReactiveKernels or of a package of its repository
# (ReactiveKernelsPPL, its extensions, ...), whose names all start so.
_rk_module(m::Module) = startswith(string(nameof(Base.moduleroot(m))), "ReactiveKernels")

# The callees the code audit of a level recurses into.
_code_recursion(::Val{:rk}) = m::Method ->
    _rk_module(m.module) || nameof(m.module) === :RuntimeGeneratedFunctions
_code_recursion(::Val{:all}) = m::Method -> !_julia_own_module(m.module)

# The code findings of `f` over `argument_types` at `level`; none at `:values`,
# or where an argument type is not concrete, which leaves inference nothing to
# check.
_level_code_findings(::Val{:values}, f, argument_types) = _CodeFinding[]
function _level_code_findings(level::Val, @nospecialize(f), argument_types)
    all(isconcretetype, argument_types) || return _CodeFinding[]
    _code_findings(f, Tuple(argument_types); recurse = _code_recursion(level))
end
Base.@nospecializeinfer _level_code_findings(level::Symbol, @nospecialize(f), argument_types) =
    _level_code_findings(Val(level), f, argument_types)

# A finding that starts an instability: a value whose operands are concretely
# typed, or a call dispatched at run time. Other findings follow from one.
_code_source(f::_CodeFinding) = f.kind === :root || f.kind === :dynamic

# The code findings grouped by method, in the order the audit met them.
function _code_findings_by_method(found)
    groups = Pair{String,Vector{_CodeFinding}}[]
    for f in found
        index = findfirst(g -> first(g) == f.method, groups)
        index === nothing ? push!(groups, f.method => [f]) : push!(last(groups[index]), f)
    end
    groups
end

function _code_method_description(method::String)
    method == "call" && return "The prepared program's entry call"
    occursin("generated_callfunc", method) &&
        return "The body ReactiveKernels generated for the prepared program"
    "`$method`"
end

function _code_finding_message(method::String, found)
    sources = filter(_code_source, found)
    shown = isempty(sources) ? found : sources
    lines = map(shown) do f
        what = f.kind === :dynamic ? "is a call dispatched at run time" :
            "infers `$(f.type)`"
        "  - %$(f.index) $what: `$(f.statement)`"
    end
    following = length(found) - length(shown)
    follow = following == 1 ? "1 further value follows" :
        "$following further values follow"
    "ReactiveKernels: $(_code_method_description(method)) is not type stable:\n" *
    join(lines, "\n") * (following == 0 ? "" : "\n  and $follow from these.") *
    "\nEvery evaluation pays for the dynamic dispatch, and native reverse-mode AD " *
    "may refuse the prepared program. Make that code type stable; " *
    "`type_stability_level = :values` checks only the types of the program's values."
end

_report_type_stability(values, code, mode::Symbol) =
    _report_type_stability(values, code, Val(mode))

function _report_type_stability(values, code, ::Val{:warn})
    for u in values
        @warn _uninferred_value_message(u) _group = :type_stability maxlog = 1 _id = (
            :uninferred_value, u.path, u.value.name, _uninferred_recipe_text(u),
            u.input_types)
    end
    for (method, found) in _code_findings_by_method(code)
        id = (:code_finding, method, map(f -> (f.kind, f.statement), found))
        @warn _code_finding_message(method, found) _group = :type_stability maxlog = 1 _id = id
    end
end

function _report_type_stability(values, code, ::Val{:error})
    messages = [map(_uninferred_value_message, values);
                [_code_finding_message(method, found)
                 for (method, found) in _code_findings_by_method(code)]]
    isempty(messages) || throw(TypeInstabilityError(messages))
end

# Check `kernel` over `argument_types` with the settings of the preparation in
# progress: the type of every value, and at `:rk` and `:all` the code of the
# prepared program. Warn once per process for each finding and continue, or
# refuse the preparation, naming every finding. The check reads only the
# kernel's plan and calls it through inference, so it compiles once rather than
# once per prepared kernel type: specialized on the type of a large program's
# kernel, its compilation took a third of that program's first read (63 of
# 187 s, Bruno's varyingsource3 simulator; snag `compiled-reader-be3cadbb`).
Base.@nospecializeinfer function _check_uninferred_values(@nospecialize(kernel), argument_types)
    mode, level = _type_stability(), _type_stability_level()
    values = _uninferred_values(kernel, argument_types)
    code = _level_code_findings(level, kernel, argument_types)
    _report_type_stability(values, code, mode)
    nothing
end

# Prepared kernels whose HAVE ports have concrete declared types are checked
# from those declarations.
Base.@nospecializeinfer _check_uninferred_values(@nospecialize(kernel)) =
    _check_uninferred_values(kernel, map(valtype, (getfield(kernel, :plan)::Plan).have))
