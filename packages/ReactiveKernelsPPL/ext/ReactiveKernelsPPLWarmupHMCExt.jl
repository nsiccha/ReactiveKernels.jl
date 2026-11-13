module ReactiveKernelsPPLWarmupHMCExt

import ReactiveKernelsPPL as PPL
import WarmupHMC as WH
import LogDensityProblems as LDP

function PPL.adaptive_centering_problem(built, plan::PPL.StructuralPlan, problem, backend;
        cells::NamedTuple, u0 = nothing, on_error = nothing)
    prepared = PPL._centering_cells(built, plan, cells; u0, on_error)
    LDP.dimension(problem) == prepared.dimension || throw(DimensionMismatch(
        "problem dimension $(LDP.dimension(problem)) ≠ model dimension $(prepared.dimension)"))
    map = _centering_reparametrization(prepared.kernel, prepared.records)
    WH.ReparametrizedProblem(map, problem, backend)
end

function _centering_reparametrization(kernel, records)
    shared = PPL._CenteringAccessor(Ref(kernel))
    accessor = WH.SharedAccessor(shared, CartesianIndex(1, 1))
    identity = WH.Reparametrization(WH.PartiallyCentered(0.0),
        WH.PartiallyCentered(0.0), accessor, accessor)
    pairs = Pair{Int,typeof(identity)}[]
    for cell in records
        for element in 1:cell.size
            location = WH.SharedAccessor(shared,
                CartesianIndex(cell.location, cell.scalar_location ? 1 : element))
            log_scale = WH.SharedAccessor(shared,
                CartesianIndex(cell.log_scale, cell.scalar_log_scale ? 1 : element))
            push!(pairs, (cell.offset + element - 1) => WH.Reparametrization(
                WH.PartiallyCentered(cell.target), WH.PartiallyCentered(cell.source),
                location, log_scale))
        end
    end
    WH.IndexedReparametrization(pairs)
end

end
