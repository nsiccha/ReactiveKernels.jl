module ReactiveKernelsPPLLogDensityProblemsExt

import LogDensityProblems
import ReactiveKernelsPPL
using ReactiveKernelsPPL: SliceProblem

# A slice of one bound model as a first-order LogDensityProblems target over its
# compact coordinates (`ReactiveKernelsPPL.slice_problem`).
LogDensityProblems.capabilities(::Type{<:SliceProblem}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(p::SliceProblem) = length(p.coordinates)
LogDensityProblems.logdensity(p::SliceProblem, x) = ReactiveKernelsPPL._slice_problem_value(p, x)
LogDensityProblems.logdensity_and_gradient(p::SliceProblem, x) =
    ReactiveKernelsPPL._slice_problem_value_and_gradient(p, x)

end
