using ReactiveKernels
using Test

if !isdefined(@__MODULE__, :ReactiveHMCStatisticsFixture)
    include(joinpath(@__DIR__, "..", "benchmark",
                     "reactivehmc_statistics_kernel_fixture.jl"))
end

const _RHMC_STATS = ReactiveHMCStatisticsFixture

module _StructuredCompilerNegatives
using ReactiveKernels

@kernel changing_local_type(value=0.0) = begin
    step!(flag) = begin
        carried = value
        if flag
            carried = 1
        end
        value = carried
    end
end

@kernel leaking_branch_local(value=0.0) = begin
    step!(flag) = begin
        if flag
            hidden = one(value)
        end
        value = hidden
    end
end

@kernel shadowing_loop_local(value=0.0) = begin
    step!(n) = begin
        index = zero(n)
        for index in 1:n
            value += index
        end
    end
end

@kernel changing_return_type(value=0.0) = begin
    step!(flag) = begin
        flag && return true
        return value
    end
end
end

function _statistics_kernel_and_state(sources)
    names = propertynames(sources)
    args = ntuple(i -> getproperty(sources, names[i]), 13)
    keyword_names = Tuple(names[14:end])
    keywords = NamedTuple{keyword_names}(ntuple(
        i -> getproperty(sources, keyword_names[i]), length(keyword_names)))
    kernel = ReactiveKernels.compile_stateful(
        _RHMC_STATS.statistics_state, args...; keywords...)
    kernel, kernel(args...; keywords...)
end


_compiled_statistics_state(sources) =
    last(_statistics_kernel_and_state(sources))

function _record_statistics_receipt!(state)
    @test !ReactiveKernels.stateful_call(
        state, Val(:reset!), [0.25, -0.5], [0.2, -0.4], 0.15625)
    events = (
        (true,  [0.4, -0.3],  [0.5, -0.6], 0.125,   -0.1),
        (false, [-0.2, 0.6], [-0.3, 0.7],  0.2,     -0.2),
        (true,  [0.55, -0.1], [0.65, -0.2], 0.15625, -0.05),
    )
    for event in events
        @test !ReactiveKernels.stateful_call(
            state, Val(:record_trajectory!), event...)
    end
    @test !ReactiveKernels.stateful_call(
        state, Val(:record_sample!), [1.0, -1.0], 0.125, false)

    expected_positions = [-0.2 0.25 0.4 0.55; 0.6 -0.5 -0.3 -0.1]
    expected_gradients = [0.3 -0.2 -0.5 -0.65; -0.7 0.4 0.6 0.2]
    @test state.first == 3
    @test state.count == 4
    @test state.positions[:, 3:6] == expected_positions
    @test state.gradients[:, 3:6] == expected_gradients
    @test state.dhams[3:6] == [-0.2, 0.0, -0.1, -0.05]
    @test state.pots[3:6] == [0.2, 0.15625, 0.125, 0.15625]
    @test state.idxs[3:6] == [2, 0, 1, 3]
    @test state.full_history[:, 1:4, 1] == expected_positions
    @test state.full_idxs[1:4, 1] == [2, 0, 1, 3]

    first_snapshot = (
        history=copy(state.full_history[:, :, 1]),
        idxs=copy(state.full_idxs[:, 1]),
        count=state.history_counts[1],
    )
    @test !ReactiveKernels.stateful_call(
        state, Val(:reset!), [0.7, 0.8], [0.75, 0.85], 0.565)
    @test !ReactiveKernels.stateful_call(
        state, Val(:record_trajectory!), true, [0.9, 1.0],
        [0.95, 1.05], 0.905, -1.5)
    @test !ReactiveKernels.stateful_call(
        state, Val(:record_sample!), [-1.0, 2.0], 0.0625, true)

    @test state.draws[:, 1:2] == [1.0 -1.0; -1.0 2.0]
    @test state.n_steps[1:2] == [3, 1]
    @test state.stepsizes[1:2] == [0.125, 0.0625]
    @test state.acc_rate[1:2] ≈
          [0.8915991985382185, 0.2231301601484299] atol=2e-16 rtol=0
    @test state.diverged[1:2] == Bool[false, true]
    @test state.history_counts[1:2] == [4, 2]
    @test state.full_history[:, :, 1] == first_snapshot.history
    @test state.full_idxs[:, 1] == first_snapshot.idxs
    @test state.history_counts[1] == first_snapshot.count
    @test state.full_history[:, 1:2, 2] == [0.7 0.9; 0.8 1.0]
    @test state.full_idxs[1:2, 2] == [0, 1]
    state
end

@testset "generic structured compiler reproduces ReactiveHMC statistics" begin
    state = _compiled_statistics_state(
        _RHMC_STATS.initial_statistics_sources(2, 8, 4))
    _record_statistics_receipt!(state)

    @test_throws MethodError ReactiveKernels.stateful_call(
        state, Val(:reset!), [0.0, 0.0], [0.0, 0.0])
    @test_throws ArgumentError ReactiveKernels.stateful_call(
        state, Val(:record_sample!), "not an array", 0.1, false)
end

@testset "functional structured compiler preserves the same source program" begin
    sources = _RHMC_STATS.initial_statistics_sources(2, 8, 4)
    kernel, native = _statistics_kernel_and_state(sources)
    _record_statistics_receipt!(native)

    _, fresh = _statistics_kernel_and_state(sources)
    state = ReactiveKernels._stateful_snapshot(fresh)
    reset = ReactiveKernels._functionalize_stateful(
        kernel, Val(:reset!); max_iterations=8,
        argument_types=Tuple{Vector{Float64},Vector{Float64},Float64})
    trajectory = ReactiveKernels._functionalize_stateful(
        kernel, Val(:record_trajectory!); max_iterations=8,
        argument_types=Tuple{Bool,Vector{Float64},Vector{Float64},Float64,Float64})
    sample = ReactiveKernels._functionalize_stateful(
        kernel, Val(:record_sample!); max_iterations=8,
        argument_types=Tuple{Vector{Float64},Float64,Bool})

    output = reset(state, [0.25, -0.5], [0.2, -0.4], 0.15625)
    @test output.returned && !output.result && !output.control_overflow
    state = output.state
    for event in (
            (true,  [0.4, -0.3], [0.5, -0.6], 0.125, -0.1),
            (false, [-0.2, 0.6], [-0.3, 0.7], 0.2, -0.2),
            (true,  [0.55, -0.1], [0.65, -0.2], 0.15625, -0.05),
        )
        output = trajectory(state, event...)
        @test output.returned && !output.result && !output.control_overflow
        state = output.state
    end
    output = sample(state, [1.0, -1.0], 0.125, false)
    @test output.returned && !output.result && !output.control_overflow
    state = output.state
    output = reset(state, [0.7, 0.8], [0.75, 0.85], 0.565)
    @test output.returned && !output.result && !output.control_overflow
    state = output.state
    output = trajectory(state, true, [0.9, 1.0], [0.95, 1.05],
                        0.905, -1.5)
    @test output.returned && !output.result && !output.control_overflow
    state = output.state
    output = sample(state, [-1.0, 2.0], 0.0625, true)
    @test output.returned && !output.result && !output.control_overflow
    state = output.state

    for name in propertynames(state)
        @test getfield(state, name) == getproperty(native, name)
    end

    initial_state = ReactiveKernels._stateful_snapshot(fresh)
    too_small = ReactiveKernels._functionalize_stateful(
        kernel, Val(:reset!); max_iterations=1,
        argument_types=Tuple{Vector{Float64},Vector{Float64},Float64})
    bounded = too_small(
        initial_state, [0.25, -0.5], [0.2, -0.4], 0.15625)
    @test bounded.control_overflow
    @test !bounded.returned
    for name in propertynames(initial_state)
        @test getfield(bounded.state, name) == getfield(initial_state, name)
    end

    out_of_bounds = reset(
        initial_state, [0.25], [0.2, -0.4], 0.15625)
    @test out_of_bounds.control_overflow
    @test !out_of_bounds.returned
    for name in propertynames(initial_state)
        @test getfield(out_of_bounds.state, name) == getfield(initial_state, name)
    end
    @test_throws ArgumentError reset(
        initial_state, Float32[0.25, -0.5], [0.2, -0.4], 0.15625)

    @test_throws ReactiveKernels._LLowerReject ReactiveKernels._functionalize_stateful(
        kernel, Val(:record_sample!);
        argument_types=Tuple{Vector{Float64},Float64,Bool})
end

@testset "functional structured capacity result and shape contracts" begin
    sources = _RHMC_STATS.initial_statistics_sources(2, 2, 1)
    kernel, native = _statistics_kernel_and_state(sources)
    state = ReactiveKernels._stateful_snapshot(native)
    reset = ReactiveKernels._functionalize_stateful(
        kernel, Val(:reset!); max_iterations=2,
        argument_types=Tuple{Vector{Float64},Vector{Float64},Float64})
    trajectory = ReactiveKernels._functionalize_stateful(
        kernel, Val(:record_trajectory!); max_iterations=2,
        argument_types=Tuple{Bool,Vector{Float64},Vector{Float64},Float64,Float64})
    output = reset(state, [0.0, 0.0], [0.0, 0.0], 0.0)
    @test output.returned && !output.result && !output.control_overflow
    output = trajectory(output.state, true, [1.0, 2.0], [3.0, 4.0],
                        5.0, 6.0)
    @test output.returned && !output.result && !output.control_overflow
    before = output.state
    output = trajectory(before, true, [7.0, 8.0], [9.0, 10.0],
                        11.0, 12.0)
    @test output.returned && output.result && !output.control_overflow
    @test output.state.trajectory_overflow
    @test output.state.positions == before.positions
    @test output.state.count == before.count

    empty_sources = _RHMC_STATS.initial_statistics_sources(2, 0, 1)
    empty_kernel, empty_native = _statistics_kernel_and_state(empty_sources)
    empty_reset = ReactiveKernels._functionalize_stateful(
        empty_kernel, Val(:reset!); max_iterations=2,
        argument_types=Tuple{Vector{Float64},Vector{Float64},Float64})
    @test_throws ArgumentError empty_reset(
        ReactiveKernels._stateful_snapshot(empty_native),
        [1.0, 2.0], [3.0, 4.0], 5.0)
end

@testset "structured compiler fails closed before partial statistics writes" begin
    trajectory = _compiled_statistics_state(
        _RHMC_STATS.initial_statistics_sources(2, 2, 2))
    @test !ReactiveKernels.stateful_call(
        trajectory, Val(:reset!), [0.0, 0.0], [0.0, 0.0], 0.0)
    @test !ReactiveKernels.stateful_call(
        trajectory, Val(:record_trajectory!), true, [1.0, 2.0],
        [3.0, 4.0], 5.0, 6.0)
    before = (positions=copy(trajectory.positions),
              gradients=copy(trajectory.gradients),
              dhams=copy(trajectory.dhams), pots=copy(trajectory.pots),
              idxs=copy(trajectory.idxs), first=trajectory.first,
              count=trajectory.count)
    @test ReactiveKernels.stateful_call(
        trajectory, Val(:record_trajectory!), true, [7.0, 8.0],
        [9.0, 10.0], 11.0, 12.0)
    @test trajectory.trajectory_overflow
    @test trajectory.positions == before.positions
    @test trajectory.gradients == before.gradients
    @test trajectory.dhams == before.dhams
    @test trajectory.pots == before.pots
    @test trajectory.idxs == before.idxs
    @test trajectory.first == before.first
    @test trajectory.count == before.count

    sampling = _compiled_statistics_state(
        _RHMC_STATS.initial_statistics_sources(2, 2, 1))
    @test !ReactiveKernels.stateful_call(
        sampling, Val(:reset!), [0.0, 0.0], [0.0, 0.0], 0.0)
    @test !ReactiveKernels.stateful_call(
        sampling, Val(:record_sample!), [1.0, 2.0], 0.25, false)
    sample_before = (draws=copy(sampling.draws),
                     n_steps=copy(sampling.n_steps),
                     stepsizes=copy(sampling.stepsizes),
                     acc_rate=copy(sampling.acc_rate),
                     diverged=copy(sampling.diverged),
                     history=copy(sampling.full_history),
                     idxs=copy(sampling.full_idxs),
                     counts=copy(sampling.history_counts),
                     sample_count=sampling.sample_count)
    @test ReactiveKernels.stateful_call(
        sampling, Val(:record_sample!), [3.0, 4.0], 0.5, true)
    @test sampling.sampling_overflow
    @test sampling.draws == sample_before.draws
    @test sampling.n_steps == sample_before.n_steps
    @test sampling.stepsizes == sample_before.stepsizes
    @test sampling.acc_rate == sample_before.acc_rate
    @test sampling.diverged == sample_before.diverged
    @test sampling.full_history == sample_before.history
    @test sampling.full_idxs == sample_before.idxs
    @test sampling.history_counts == sample_before.counts
    @test sampling.sample_count == sample_before.sample_count

    empty_state = _compiled_statistics_state(
        _RHMC_STATS.initial_statistics_sources(2, 0, 1))
    @test ReactiveKernels.stateful_call(
        empty_state, Val(:reset!), [1.0, 2.0], [3.0, 4.0], 5.0)
    @test empty_state.trajectory_overflow
    @test empty_state.count == 0
    @test isempty(empty_state.positions)

    invalid_sources = merge(
        _RHMC_STATS.initial_statistics_sources(2, 2, 1),
        (reset_first=3,))
    invalid = _compiled_statistics_state(invalid_sources)
    invalid_before = copy(invalid.positions)
    @test ReactiveKernels.stateful_call(
        invalid, Val(:reset!), [1.0, 2.0], [3.0, 4.0], 5.0)
    @test invalid.trajectory_overflow
    @test invalid.positions == invalid_before
    @test invalid.count == 0
end

@testset "structured compiler has lexical and dtype-stable local scopes" begin
    changing_kernel = ReactiveKernels.compile_stateful(
        _StructuredCompilerNegatives.changing_local_type, 0.0)
    changing = changing_kernel(0.0)
    @test_throws ArgumentError ReactiveKernels.stateful_call(
        changing, Val(:step!), true)

    leak_error = try
        ReactiveKernels.compile_stateful(
            _StructuredCompilerNegatives.leaking_branch_local, 0.0)
        nothing
    catch error
        error
    end
    @test leak_error isa ReactiveKernels._KernelFactoryReject
    @test occursin("maybe-bound local `hidden`", sprint(showerror, leak_error))
    @test_throws ReactiveKernels._LLowerReject ReactiveKernels.compile_stateful(
        _StructuredCompilerNegatives.shadowing_loop_local, 0.0)

    return_kernel = ReactiveKernels.compile_stateful(
        _StructuredCompilerNegatives.changing_return_type, 0.0)
    @test_throws ReactiveKernels._LLowerReject ReactiveKernels._functionalize_stateful(
        return_kernel, Val(:step!); argument_types=Tuple{Bool})
end
