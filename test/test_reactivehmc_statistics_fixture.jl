using ReactiveKernels
using Test

include(joinpath(@__DIR__, "..", "benchmark",
                 "reactivehmc_statistics_kernel_fixture.jl"))

const RHMC_STATISTICS = ReactiveHMCStatisticsFixture

_normalize_source_newlines(source) =
    replace(replace(source, "\r\n" => "\n"), '\r' => '\n')

@testset "ReactiveHMC fixed-capacity statistics kernel source contract" begin
    skeleton = RHMC_STATISTICS.statistics_state
    registration = ReactiveKernels.kernel_registration(skeleton)
    @test registration.kind == :object_kernel
    @test ReactiveKernels.kernel_port_names(skeleton) == (
        :positions, :gradients, :dhams, :pots, :idxs,
        :draws, :n_steps, :stepsizes, :acc_rate, :diverged,
        :full_history, :full_idxs, :history_counts,
        :dimension, :trajectory_capacity, :sample_capacity,
        :reset_first,
        :first, :count, :sample_count,
        :trajectory_overflow, :sampling_overflow,
    )
    @test map(ir -> ir.id.name, ReactiveKernels.method_irs(skeleton)) ==
          (:min1exp, :reset!, :record_trajectory!, :record_sample!)

    sources = RHMC_STATISTICS.initial_statistics_sources(2, 8, 4)
    @test size(sources.positions) == size(sources.gradients) == (2, 8)
    @test size(sources.draws) == (2, 4)
    @test size(sources.full_history) == (2, 8, 4)
    @test size(sources.full_idxs) == (8, 4)
    @test sources.reset_first == 4
    @test sources.count == sources.sample_count == 0
    @test !sources.trajectory_overflow
    @test !sources.sampling_overflow

    source_path = joinpath(@__DIR__, "..", "benchmark",
                           "reactivehmc_statistics_kernel_fixture.jl")
    source = _normalize_source_newlines(read(source_path, String))
    @test occursin("column = go_forward ? first + count : first - 1", source)
    @test occursin("idxs[column] = count", source)
    @test findfirst("idxs[column] = count", source) <
          findfirst("count += 1", source)
    @test occursin("trajectory_overflow && return trajectory_overflow", source)
    @test occursin("reset_first < 1 || reset_first > trajectory_capacity", source)
    sampling_overflow_source =
        "sampling_overflow = true\n            return sampling_overflow"
    @test occursin(sampling_overflow_source, source)
    crlf_source = replace(source, "\n" => "\r\n")
    @test !occursin(sampling_overflow_source, crlf_source)
    @test occursin(
        sampling_overflow_source, _normalize_source_newlines(crlf_source))
    @test occursin("min1exp(x) = x > zero(x) ? one(x) : exp(x)", source)
    @test occursin("acceptance_sum += min1exp(__self__, dhams[column])", source)
    @test occursin("acceptance_count = count > 1 ? count - 1 : 1", source)
    @test occursin("full_history[index, offset + 1, next_sample]", source)

    compiler_sources = String[]
    for root in (joinpath(@__DIR__, "..", "src"),
                 joinpath(@__DIR__, "..", "ext"))
        isdir(root) || continue
        for (directory, _, files) in walkdir(root), file in files
            endswith(file, ".jl") || continue
            push!(compiler_sources, read(joinpath(directory, file), String))
        end
    end
    compiler_text = join(compiler_sources, '\n')
    @test !occursin("statistics_state", compiler_text)
    @test !occursin("record_trajectory!", compiler_text)
    @test !occursin("record_sample!", compiler_text)
end
