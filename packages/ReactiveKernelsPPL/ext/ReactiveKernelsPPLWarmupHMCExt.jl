module ReactiveKernelsPPLWarmupHMCExt

import ReactiveKernelsPPL as PPL
import WarmupHMC as WH
import LogDensityProblems as LDP

function PPL.adaptive_centering_problem(built, plan::PPL.StructuralPlan, problem, backend;
        cells::NamedTuple = (;), u0 = nothing, on_error = nothing)
    prepared = PPL._centering_cells(built, plan, cells; u0, on_error)
    LDP.dimension(problem) == prepared.dimension || throw(DimensionMismatch(
        "problem dimension $(LDP.dimension(problem)) ≠ model dimension $(prepared.dimension)"))
    map = _centering_reparametrization(prepared.kernel, prepared.records,
        _centering_chains(prepared.chain_kernel, prepared.chains))
    WH.ReparametrizedProblem(map, problem, backend)
end

# One `MarkovChain` per recurrence `markov_chains` marked, on the frame kernel.
function _centering_chains(kernel, records)
    isempty(records) && return ()
    pkgversion(WH) >= v"0.3.1" && isdefined(WH, :MarkovChain) ||
        throw(PPL.ContractValidationError("[centering] Markov chains (`markov_chains`) " *
            "need WarmupHMC 0.3.1 or later; this environment has WarmupHMC $(pkgversion(WH))"))
    shared = PPL._CenteringAccessor(Ref(kernel))
    Tuple(WH.MarkovChain(chain.states;
        log_scale = PPL._ChainAccessor(shared, chain.log_scale),
        coefficient = PPL._ChainAccessor(shared, chain.coef),
        drift = PPL._ChainAccessor(shared, chain.drift),
        initial = PPL._ChainInitial(shared, chain.drift)) for chain in records)
end

# Records arrive ordered by level. Each level reads the shared kernel at its own
# stage, so WarmupHMC evaluates the kernel once per level, after the earlier
# levels are mapped, and those reads see their model values.
function _centering_reparametrization(kernel, records, chains = ())
    shared = PPL._CenteringAccessor(Ref(kernel))
    if any(cell -> cell.stage > 1, records) && pkgversion(WH) < v"0.3.0"
        throw(PPL.ContractValidationError("[centering] cells that read other " *
            "selected cells' latents need WarmupHMC 0.3.0 or later, whose " *
            "accessors read earlier pairs at their model values; this " *
            "environment has WarmupHMC $(pkgversion(WH))"))
    end
    # WarmupHMC 0.2 has no stages, and its two-argument form is stage 1 on 0.3.
    accessor(index, stage) = stage == 1 ? WH.SharedAccessor(shared, index) :
        WH.SharedAccessor(shared, index, stage)
    identity = WH.Reparametrization(WH.PartiallyCentered(0.0),
        WH.PartiallyCentered(0.0), accessor(CartesianIndex(1, 1), 1),
        accessor(CartesianIndex(1, 1), 1))
    pairs = Pair{Int,typeof(identity)}[]
    for cell in records
        for element in 1:cell.size
            location = accessor(
                CartesianIndex(cell.location, cell.scalar_location ? 1 : element), cell.stage)
            log_scale = accessor(
                CartesianIndex(cell.log_scale, cell.scalar_log_scale ? 1 : element), cell.stage)
            push!(pairs, (cell.offset + element - 1) => WH.Reparametrization(
                WH.PartiallyCentered(cell.target), WH.PartiallyCentered(cell.source),
                location, log_scale))
        end
    end
    isempty(chains) ? WH.IndexedReparametrization(pairs) :
        WH.IndexedReparametrization(pairs; chains)
end

end
