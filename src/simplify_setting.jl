# Preparation settings: a project preference, overridden by a preparation
# keyword for every preparation nested inside it. Preparations are cached per
# setting (`graphops.jl`).
#
# - `simplify`: whether a preparation applies the stable algebraic rewrites
#   (`stable_rewrites.jl`) and takes shared factors and terms out of summed
#   plates (`summed_plate_factors.jl`). Snag `rk-stable-algebr-e6eb8f42`, user
#   decision
#   `ReactiveKernels:snag.rk-stable-algebr-e6eb8f42/decisions/2026-11-08T04-13-58-283-17d8927`,
#   option 1: on in every preparation, `simplify = false` opts out.
# - `bound_statistics`: whether a bound preparation evaluates a summed squared
#   residual over bound data from least-squares statistics frozen at binding
#   (`bound_statistics.jl`). Snag `bound-data-suffi-7c8369bd`, user decision
#   `ReactiveKernels:snag.bound-data-suffi-7c8369bd/decisions/2026-11-12T12-04-45-460-1gq53h7`,
#   option 1: on by default, `bound_statistics = false` opts out.

struct _PreparationSetting
    # The preference key (`[ReactiveKernels] <name> = false` in the project's
    # `LocalPreferences.toml`) and the preparation keyword.
    name::String
    # The task-local storage key of an enclosing keyword.
    scope::Symbol
    # The preference, read at the first preparation of a process, as the
    # type-stability preferences are (`type_stability.jl`); `__init__` forgets
    # a value read while an image was generated.
    preference::Base.RefValue{Union{Nothing,Bool}}
end
_PreparationSetting(name::String) =
    _PreparationSetting(name, Symbol(:ReactiveKernels_, name), Ref{Union{Nothing,Bool}}(nothing))

const _SIMPLIFY = _PreparationSetting("simplify")
const _BOUND_STATISTICS = _PreparationSetting("bound_statistics")
const _PREPARATION_SETTINGS = (_SIMPLIFY, _BOUND_STATISTICS)

function _setting_preference(s::_PreparationSetting)
    cached = s.preference[]
    cached === nothing || return cached
    value = Preferences.load_preference(@__MODULE__, s.name, true)
    value isa Bool || throw(ArgumentError(
        "the ReactiveKernels preference `$(s.name)` must be true or false; got $(repr(value))"))
    s.preference[] = value
end

# The setting of the preparation in progress: the innermost enclosing keyword,
# else the preference.
_setting(s::_PreparationSetting) =
    get(() -> _setting_preference(s), task_local_storage(), s.scope)::Bool

# `f()` with every preparation inside it, nested ones included, applying (true)
# or not applying (false) the setting; `nothing` keeps the enclosing setting.
_with_setting(f, ::_PreparationSetting, ::Nothing) = f()
_with_setting(f, s::_PreparationSetting, value::Bool) = task_local_storage(f, s.scope, value)
_with_setting(f, s::_PreparationSetting, value) = throw(ArgumentError(
    "`$(s.name)` must be `true`, `false` or `nothing`; got $(repr(value))"))

# Whether the preparation in progress applies the stable rewrites.
_simplify() = _setting(_SIMPLIFY)
_with_simplify(f, value) = _with_setting(f, _SIMPLIFY, value)
# Whether the preparation in progress uses bound least-squares statistics.
_bound_statistics() = _setting(_BOUND_STATISTICS)
_with_bound_statistics(f, value) = _with_setting(f, _BOUND_STATISTICS, value)

"""
    with_bound_statistics(f, enabled::Bool = true)

Run `f()` with every preparation it makes, nested ones included, using
(`true`) or not using (`false`) bound least-squares statistics for summed
Gaussian plates, whatever the `bound_statistics` preference says. It is the
`bound_statistics` keyword of [`prepare`](@ref) for preparations made through
an API that does not pass it on, such as a PPL's sampler preparation.
"""
with_bound_statistics(f, enabled::Bool = true) = _with_bound_statistics(f, enabled)
# Both preparation keywords.
_with_settings(f, simplify, bound_statistics) =
    _with_simplify(() -> _with_bound_statistics(f, bound_statistics), simplify)
