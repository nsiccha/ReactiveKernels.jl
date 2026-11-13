# Benchmark-only ownership refinement. No production method changes until install!.
module ResidualBorrowPrototype
using ReactiveKernels
import Enzyme

# Use Julia's documented deepcopy specialization hook on an owned wrapper. The
# memo is shared across the flat environment, preserving repeated array
# references. Nested containers retain the existing whole-tuple snapshot path.
struct Environment{R,S,F}
    residuals::R
    sources::S
    overwritten::F
    borrow_inputs::Bool
end

_flat_value(value) = isbitstype(typeof(value)) ||
    (value isa Array && isbitstype(eltype(value)))
flat_environment(residuals::Tuple) = all(_flat_value, residuals)

_seed_borrowed!(memo, ::Tuple{}, ::Tuple{}, flags) = nothing
function _seed_borrowed!(memo, residuals::Tuple, sources::Tuple, flags)
    value = first(residuals)
    source = first(sources)
    # A direct dense bits array has no reference-valued element. The backend
    # flag must establish that its memory is unchanged through the reverse.
    if value isa Array && isbitstype(eltype(value)) && source > 0 &&
            !flags[source + 1]
        memo[value] = value
    end
    _seed_borrowed!(memo, Base.tail(residuals), Base.tail(sources), flags)
end

function Base.deepcopy_internal(env::Environment{R}, memo::IdDict) where {R}
    env.borrow_inputs && _seed_borrowed!(memo, env.residuals, env.sources,
        env.overwritten)
    # Preserve each structural tuple position's type. Base's tuple copier uses
    # integer indexing and infers a tuple of unions for heterogeneous residuals.
    retained = map(env.residuals) do value
        isbitstype(typeof(value)) && return value
        (Base.@noinline Base.deepcopy_internal(value, memo))::typeof(value)
    end
    Environment(retained, env.sources, env.overwritten, env.borrow_inputs)
end

@inline function retain(sources, residuals, primal, flags)
    isbitstype(typeof(residuals)) && return residuals
    # A mutable returned value can expose input storage independently of the
    # original argument's write flag. Keep the complete snapshot in that case.
    if !isbitstype(typeof(primal)) || !flat_environment(residuals)
        return Base.@noinline deepcopy(residuals)
    end
    env = Environment(residuals, sources, flags, true)
    saved = Base.@noinline deepcopy(env)
    saved.residuals
end

function install!()
    ext = Base.get_extension(ReactiveKernels, :ReactiveKernelsEnzymeExt)
    @eval ext @inline function _retain_residuals(sources::NTuple{R,Int},
            residuals::NTuple{R,Any}, primal, flags) where {R}
        $(retain)(sources, residuals, primal, flags)
    end
    nothing
end
end
