# The `simplify` setting: whether a preparation applies the stable algebraic
# rewrites (`stable_rewrites.jl`) and takes shared factors and terms out of
# summed plates (`summed_plate_factors.jl`). Snag `rk-stable-algebr-e6eb8f42`,
# user decision
# `ReactiveKernels:snag.rk-stable-algebr-e6eb8f42/decisions/2026-11-08T04-13-58-283-17d8927`,
# option 1: on in every preparation, `simplify = false` opts out.

# The project's preference (`[ReactiveKernels] simplify = false` in its
# `LocalPreferences.toml`), read at the first preparation of a process, as the
# type-stability preferences are (`type_stability.jl`); `__init__` forgets a
# value read while an image was generated.
const _SIMPLIFY_PREFERENCE = Ref{Union{Nothing,Bool}}(nothing)
const _SIMPLIFY_SCOPE = :ReactiveKernels_simplify

function _simplify_preference()
    cached = _SIMPLIFY_PREFERENCE[]
    cached === nothing || return cached
    value = Preferences.load_preference(@__MODULE__, "simplify", true)
    value isa Bool || throw(ArgumentError(
        "the ReactiveKernels preference `simplify` must be true or false; got $(repr(value))"))
    _SIMPLIFY_PREFERENCE[] = value
end

# Whether the preparation in progress applies the stable rewrites: the
# innermost enclosing `simplify` keyword, else the preference.
_simplify() = get(_simplify_preference, task_local_storage(), _SIMPLIFY_SCOPE)::Bool

# `f()` with every preparation inside it, nested ones included, applying (true)
# or not applying (false) the rewrites; `nothing` keeps the enclosing setting.
_with_simplify(f, ::Nothing) = f()
_with_simplify(f, value::Bool) = task_local_storage(f, _SIMPLIFY_SCOPE, value)
_with_simplify(f, value) = throw(ArgumentError(
    "`simplify` must be `true`, `false` or `nothing`; got $(repr(value))"))
