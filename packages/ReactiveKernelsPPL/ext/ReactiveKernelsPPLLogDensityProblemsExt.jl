module ReactiveKernelsPPLLogDensityProblemsExt

import LogDensityProblems
import ReactiveKernelsPPL
using ReactiveKernelsPPL: SamplerQuery, SliceProblem, JoinedWindowsProblem

# The full-model sampler query (`ReactiveKernelsPPL.prepare_sampler`) as a
# first-order LogDensityProblems target over every packed coordinate.
LogDensityProblems.capabilities(::Type{<:SamplerQuery}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(q::SamplerQuery) = q.layout.total
LogDensityProblems.logdensity(q::SamplerQuery, x) = q(_sampler_point(q, x))
function LogDensityProblems.logdensity_and_gradient(q::SamplerQuery, x)
    # A fresh gradient on every call: a sampler such as WarmupHMC keeps the
    # gradients it is handed in its trajectories.
    gradient = Vector{Float64}(undef, LogDensityProblems.dimension(q))
    value, _ = ReactiveKernelsPPL.sampler_value_and_gradient!(q, gradient, _sampler_point(q, x))
    return value, gradient
end
function _sampler_point(q::SamplerQuery, x::AbstractVector{<:Real})
    length(x) == LogDensityProblems.dimension(q) || throw(DimensionMismatch(
        "sampler point length $(length(x)) ≠ dimension $(LogDensityProblems.dimension(q))"))
    return x
end

# A slice of one bound model as a first-order LogDensityProblems target over its
# compact coordinates (`ReactiveKernelsPPL.slice_problem`).
LogDensityProblems.capabilities(::Type{<:SliceProblem}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(p::SliceProblem) = length(p.coordinates)
LogDensityProblems.logdensity(p::SliceProblem, x) = ReactiveKernelsPPL._slice_problem_value(p, x)
LogDensityProblems.logdensity_and_gradient(p::SliceProblem, x) =
    ReactiveKernelsPPL._slice_problem_value_and_gradient(p, x)

# Windows of one recurrence joined through their boundary state, over the joined
# coordinates (`ReactiveKernelsPPL.joined_problem`).
LogDensityProblems.capabilities(::Type{<:JoinedWindowsProblem}) =
    LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(p::JoinedWindowsProblem) = length(p.gradient)
LogDensityProblems.logdensity(p::JoinedWindowsProblem, x) = p.joined(x)
LogDensityProblems.logdensity_and_gradient(p::JoinedWindowsProblem, x) =
    ReactiveKernelsPPL._joined_problem_value_and_gradient(p, x)

end
