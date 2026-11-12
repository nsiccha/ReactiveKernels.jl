module ReactiveKernelsPPLWarmupHMCExt

import ReactiveKernelsPPL as PPL
import WarmupHMC as WH
import LogDensityProblems as LDP

function PPL.adaptive_centering_problem(built, plan::PPL.StructuralPlan, problem, backend;
        cells::NamedTuple, u0 = nothing, on_error = nothing, locality = nothing)
    prepared = PPL._centering_cells(built, plan, cells; u0, on_error, locality)
    LDP.dimension(problem) == prepared.dimension || throw(DimensionMismatch(
        "problem dimension $(LDP.dimension(problem)) ≠ model dimension $(prepared.dimension)"))
    map = _centering_reparametrization(prepared.kernels, prepared.records)
    WH.ReparametrizedProblem(map, problem, backend)
end

function _centering_reparametrization(kernels::K, records) where {K}
    accessor = PPL._CenteringAccessor(kernels, 1, 1)
    identity = WH.Reparametrization(WH.PartiallyCentered(0.0),
        WH.PartiallyCentered(0.0), accessor, accessor)
    pairs = Pair{Int,typeof(identity)}[]
    for cell in records, element in 1:cell.size
        location = PPL._CenteringAccessor(kernels, cell.location, element)
        log_scale = PPL._CenteringAccessor(kernels, cell.log_scale, element)
        push!(pairs, (cell.offset + element - 1) => WH.Reparametrization(
            WH.PartiallyCentered(cell.target), WH.PartiallyCentered(cell.source),
            location, log_scale))
    end
    WH.IndexedReparametrization(pairs)
end

end
