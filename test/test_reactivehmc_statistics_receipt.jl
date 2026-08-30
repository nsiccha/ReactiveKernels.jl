using ReactiveKernels
using Test
import TOML

include(joinpath(@__DIR__, "..", "benchmark", "receipts",
                 "validate_reactivehmc_statistics.jl"))

const _StatsPoint = @NamedTuple{pos::Vector{Float64},
                                dham_dpos::Vector{Float64}, pot::Float64}

struct _StatisticsReceiptState <: ReactiveKernels.AbstractNUTSState
    go_forward::Bool
    fwd::_StatsPoint
    energy_error::Float64
    init::NamedTuple{(:pos,), Tuple{Vector{Float64}}}
    step_f::NamedTuple{(:stepsize,), Tuple{Float64}}
    diverged::Bool
    acceptance_rate::Float64
end

ReactiveKernels._traj_fwd_pos(state::_StatisticsReceiptState) = state.fwd.pos
ReactiveKernels._traj_fwd_dpos(state::_StatisticsReceiptState) =
    state.fwd.dham_dpos
ReactiveKernels._traj_fwd_pot(state::_StatisticsReceiptState) = state.fwd.pot
ReactiveKernels.diagnostics(state::_StatisticsReceiptState) =
    (; acceptance_rate=state.acceptance_rate)

_stats_point(pos, dham_dpos, pot) =
    (pos=Float64.(pos), dham_dpos=Float64.(dham_dpos), pot=Float64(pot))
_stats_columns(matrix) = [collect(column) for column in eachcol(matrix)]
_stats_acceptance(dhams) =
    (sum(dham -> min(1.0, exp(dham)), dhams) - 1) / max(1, length(dhams) - 1)

function _stats_event(event; init_pos=zeros(2), stepsize=NaN,
                      diverged=false, acceptance_rate=NaN)
    _StatisticsReceiptState(
        event["go_forward"],
        _stats_point(event["pos"], event["dham_dpos"], event["pot"]),
        event["dham"],
        (; pos=Float64.(init_pos)),
        (; stepsize=Float64(stepsize)),
        diverged,
        acceptance_rate,
    )
end

@testset "ReactiveHMC trajectory/sampling statistics physical receipt" begin
    receipt_path = joinpath(@__DIR__, "..", "benchmark", "receipts",
                            "reactivehmc-statistics-ca9-v1.toml")
    @test isempty(validate_reactivehmc_statistics_receipt(receipt_path))
    receipt = TOML.parsefile(receipt_path)
    @test receipt["pins"]["reactivehmc_revision"] ==
          ReactiveHMCAlgorithmCorpus.UPSTREAM.revision
    source_digests = Dict(ReactiveHMCAlgorithmCorpus.UPSTREAM.source_sha256)
    @test receipt["pins"]["statistics_sha256"] ==
          source_digests["src/statistics.jl"]

    inputs = receipt["inputs"]
    trajectory = trajectory_stats(inputs["dimension"])
    reset!(trajectory, _stats_point(
        inputs["reset_pos"], inputs["reset_dham_dpos"], inputs["reset_pot"]))
    for event in receipt["events"]
        trajectory(_stats_event(event))
    end

    expected_trajectory = receipt["trajectory"]
    @test _stats_columns(trajectory.positions) == expected_trajectory["positions"]
    @test _stats_columns(trajectory.gradients) == expected_trajectory["gradients"]
    @test trajectory.dhams == expected_trajectory["dhams"]
    @test trajectory.pots == expected_trajectory["pots"]
    @test trajectory.idxs == expected_trajectory["idxs"]

    sampling = sampling_stats(trajectory)
    first_sample = receipt["samples"][1]
    sampling(_stats_event(
        Dict("go_forward" => true, "pos" => first_sample["init_pos"],
             "dham_dpos" => zeros(2), "pot" => 0.0, "dham" => 0.0);
        init_pos=first_sample["init_pos"], stepsize=first_sample["stepsize"],
        diverged=first_sample["diverged"],
        acceptance_rate=_stats_acceptance(trajectory.dhams)))

    second = receipt["second_trajectory"]
    reset!(trajectory, _stats_point(
        second["reset_pos"], second["reset_dham_dpos"], second["reset_pot"]))
    trajectory(_stats_event(second))
    second_sample = receipt["samples"][2]
    sampling(_stats_event(
        Dict("go_forward" => true, "pos" => second_sample["init_pos"],
             "dham_dpos" => zeros(2), "pot" => 0.0, "dham" => 0.0);
        init_pos=second_sample["init_pos"], stepsize=second_sample["stepsize"],
        diverged=second_sample["diverged"],
        acceptance_rate=_stats_acceptance(trajectory.dhams)))

    expected_sampling = receipt["sampling"]
    @test _stats_columns(sampling.draws) == expected_sampling["draws"]
    @test sampling.n_steps == expected_sampling["n_steps"]
    @test sampling.stepsizes == expected_sampling["stepsizes"]
    @test sampling.acc_rate ≈ expected_sampling["acc_rate"]
    @test sampling.diverged == expected_sampling["diverged"]
    @test map(_stats_columns, sampling.full_history) ==
          expected_sampling["full_history"]
    @test sampling.full_idxs == expected_sampling["full_idxs"]

    # The first snapshots must remain detached after the live trajectory was
    # reset and overwritten for the second sample.
    @test _stats_columns(sampling.full_history[1]) == expected_trajectory["positions"]
    @test sampling.full_idxs[1] == expected_trajectory["idxs"]
    @test sampling.full_history[1] !== trajectory.positions
    @test sampling.full_idxs[1] !== trajectory.idxs
end
