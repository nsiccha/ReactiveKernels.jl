if !isdefined(@__MODULE__, :ReactiveHMCStatisticsFixture)
    include(joinpath(@__DIR__, "..", "benchmark",
                     "reactivehmc_statistics_kernel_fixture.jl"))
end

const _RHMC_REACTANT_STATS = ReactiveHMCStatisticsFixture

function _rhmc_statistics_kernel(sources)
    names = propertynames(sources)
    positional = ntuple(i -> getproperty(sources, names[i]), 13)
    keyword_names = Tuple(names[14:end])
    keywords = NamedTuple{keyword_names}(ntuple(
        i -> getproperty(sources, keyword_names[i]), length(keyword_names)))
    kernel = ReactiveKernels.compile_stateful(
        _RHMC_REACTANT_STATS.statistics_state, positional...; keywords...)
    kernel, kernel(positional...; keywords...)
end

_rhmc_trace_value(value::BitArray) =
    Reactant.to_rarray(collect(value); track_numbers=true)
_rhmc_trace_value(value) = Reactant.to_rarray(value; track_numbers=true)
_rhmc_trace_state(state) = map(_rhmc_trace_value, state)

_rhmc_host_value(value::Reactant.AbstractConcreteArray) = Array(value)
_rhmc_host_value(value::Reactant.AbstractConcreteNumber) =
    Reactant.to_number(value)
_rhmc_host_value(value) = value

function _rhmc_test_state(actual, expected)
    for name in propertynames(expected)
        got = _rhmc_host_value(getfield(actual, name))
        want = getfield(expected, name)
        if want isa AbstractArray{<:AbstractFloat} || want isa AbstractFloat
            @test got ≈ want atol=128eps(Float64) rtol=0
        else
            @test got == want
        end
    end
end

function _rhmc_test_statistics_receipt(state, receipt)
    trajectory = receipt["trajectory"]
    sampling = receipt["sampling"]
    first_history = reduce(hcat, sampling["full_history"][1])
    second_history = reduce(hcat, sampling["full_history"][2])

    # The live trajectory has already been reset for sample two. The first
    # trajectory must survive only in its detached sample snapshot.
    @test Reactant.to_number(state.first) == 4
    @test Reactant.to_number(state.count) == 2
    @test Reactant.to_number(state.sample_count) == 2
    @test Array(state.positions[:, 4:5]) == [0.7 0.9; 0.8 1.0]
    @test Array(state.gradients[:, 4:5]) == [-0.75 -0.95; -0.85 -1.05]
    @test Array(state.dhams[4:5]) == [0.0, -1.5]
    @test Array(state.pots[4:5]) == [0.565, 0.905]
    @test Array(state.idxs[4:5]) == [0, 1]

    @test Array(state.full_history[:, 1:4, 1]) == first_history
    @test Array(state.full_idxs[1:4, 1]) == sampling["full_idxs"][1]
    @test Array(state.full_history[:, 1:2, 2]) == second_history
    @test Array(state.full_idxs[1:2, 2]) == sampling["full_idxs"][2]
    @test Array(state.history_counts[1:2]) == [4, 2]
    @test first_history == reduce(hcat, trajectory["positions"])
    @test Array(state.draws[:, 1:2]) == reduce(hcat, sampling["draws"])
    @test Array(state.n_steps[1:2]) == sampling["n_steps"]
    @test Array(state.stepsizes[1:2]) == sampling["stepsizes"]
    @test Array(state.acc_rate[1:2]) ≈ sampling["acc_rate"] atol=128eps(Float64) rtol=0
    @test Array(state.diverged[1:2]) == sampling["diverged"]
    @test !Bool(Reactant.to_number(state.trajectory_overflow))
    @test !Bool(Reactant.to_number(state.sampling_overflow))
end

function _rhmc_native_statistics_receipt!(state)
    @test !ReactiveKernels.stateful_call(
        state, Val(:reset!), [0.25, -0.5], [0.2, -0.4], 0.15625)
    for event in (
            (true,  [0.4, -0.3], [0.5, -0.6], 0.125, -0.1),
            (false, [-0.2, 0.6], [-0.3, 0.7], 0.2, -0.2),
            (true,  [0.55, -0.1], [0.65, -0.2], 0.15625, -0.05),
        )
        @test !ReactiveKernels.stateful_call(
            state, Val(:record_trajectory!), event...)
    end
    @test !ReactiveKernels.stateful_call(
        state, Val(:record_sample!), [1.0, -1.0], 0.125, false)
    @test !ReactiveKernels.stateful_call(
        state, Val(:reset!), [0.7, 0.8], [0.75, 0.85], 0.565)
    @test !ReactiveKernels.stateful_call(
        state, Val(:record_trajectory!), true, [0.9, 1.0],
        [0.95, 1.05], 0.905, -1.5)
    @test !ReactiveKernels.stateful_call(
        state, Val(:record_sample!), [-1.0, 2.0], 0.0625, true)
    ReactiveKernels._stateful_snapshot(state)
end

function _rhmc_call_compiled(compiled, state, arguments...)
    traced = map(_rhmc_trace_value, arguments)
    output = compiled(state, traced...)
    @test Bool(_rhmc_host_value(output.returned))
    @test !Bool(_rhmc_host_value(output.result))
    @test !Bool(_rhmc_host_value(output.control_overflow))
    output.state
end

@testset "Reactant executes the source-derived statistics state machine" begin
    sources = _RHMC_REACTANT_STATS.initial_statistics_sources(2, 8, 4)
    kernel, native = _rhmc_statistics_kernel(sources)
    native_expected = _rhmc_native_statistics_receipt!(native)
    receipt = TOML.parsefile(joinpath(
        @__DIR__, "..", "benchmark", "receipts",
        "reactivehmc-statistics-ca9-v1.toml"))
    _, fresh = _rhmc_statistics_kernel(sources)

    reset = ReactiveKernels._functionalize_stateful(
        kernel, Val(:reset!); max_iterations=8,
        argument_types=Tuple{Vector{Float64},Vector{Float64},Float64})
    trajectory = ReactiveKernels._functionalize_stateful(
        kernel, Val(:record_trajectory!); max_iterations=8,
        argument_types=Tuple{Bool,Vector{Float64},Vector{Float64},Float64,Float64})
    sample = ReactiveKernels._functionalize_stateful(
        kernel, Val(:record_sample!); max_iterations=8,
        argument_types=Tuple{Vector{Float64},Float64,Bool})

    state = _rhmc_trace_state(ReactiveKernels._stateful_snapshot(fresh))
    reset_args = map(_rhmc_trace_value,
                     ([0.25, -0.5], [0.2, -0.4], 0.15625))
    trajectory_args = map(_rhmc_trace_value,
        (true, [0.4, -0.3], [0.5, -0.6], 0.125, -0.1))
    sample_args = map(_rhmc_trace_value, ([1.0, -1.0], 0.125, false))
    compiled_reset = @compile reset(state, reset_args...)
    compiled_trajectory = @compile trajectory(state, trajectory_args...)
    compiled_sample = @compile sample(state, sample_args...)

    state = _rhmc_call_compiled(
        compiled_reset, state, [0.25, -0.5], [0.2, -0.4], 0.15625)
    for event in (
            (true,  [0.4, -0.3], [0.5, -0.6], 0.125, -0.1),
            (false, [-0.2, 0.6], [-0.3, 0.7], 0.2, -0.2),
            (true,  [0.55, -0.1], [0.65, -0.2], 0.15625, -0.05),
        )
        state = _rhmc_call_compiled(compiled_trajectory, state, event...)
    end
    state = _rhmc_call_compiled(
        compiled_sample, state, [1.0, -1.0], 0.125, false)
    state = _rhmc_call_compiled(
        compiled_reset, state, [0.7, 0.8], [0.75, 0.85], 0.565)
    state = _rhmc_call_compiled(compiled_trajectory, state, true,
        [0.9, 1.0], [0.95, 1.05], 0.905, -1.5)
    state = _rhmc_call_compiled(
        compiled_sample, state, [-1.0, 2.0], 0.0625, true)

    _rhmc_test_statistics_receipt(state, receipt)
    # Native parity is supplementary: the independent physical receipt above
    # is the authority, so a same-engine bug cannot define expected behavior.
    _rhmc_test_state(state, native_expected)

    # A too-small static bound is an observable compilation failure, not a
    # silently shortened source loop. The whole state rolls back atomically.
    bounded = ReactiveKernels._functionalize_stateful(
        kernel, Val(:reset!); max_iterations=1,
        argument_types=Tuple{Vector{Float64},Vector{Float64},Float64})
    initial = _rhmc_trace_state(ReactiveKernels._stateful_snapshot(fresh))
    compiled_bounded = @compile bounded(initial, reset_args...)
    output = compiled_bounded(initial, reset_args...)
    @test Bool(_rhmc_host_value(output.control_overflow))
    @test !Bool(_rhmc_host_value(output.returned))
    _rhmc_test_state(output.state, ReactiveKernels._stateful_snapshot(fresh))

    # Exercise the Reactant-specific dynamic-index helpers with an active OOB
    # source read. It must report compiler overflow and roll back the whole
    # transition instead of committing a clamped value.
    short_position = _rhmc_trace_value([0.25])
    full_gradient = _rhmc_trace_value([0.2, -0.4])
    traced_potential = _rhmc_trace_value(0.15625)
    compiled_oob = @compile reset(
        initial, short_position, full_gradient, traced_potential)
    output = compiled_oob(
        initial, short_position, full_gradient, traced_potential)
    @test Bool(_rhmc_host_value(output.control_overflow))
    @test !Bool(_rhmc_host_value(output.returned))
    _rhmc_test_state(output.state, ReactiveKernels._stateful_snapshot(fresh))
end

@testset "Reactant preserves source capacity overflow result and atomicity" begin
    sources = _RHMC_REACTANT_STATS.initial_statistics_sources(2, 2, 1)
    kernel, native = _rhmc_statistics_kernel(sources)
    state = _rhmc_trace_state(ReactiveKernels._stateful_snapshot(native))
    reset = ReactiveKernels._functionalize_stateful(
        kernel, Val(:reset!); max_iterations=2,
        argument_types=Tuple{Vector{Float64},Vector{Float64},Float64})
    trajectory = ReactiveKernels._functionalize_stateful(
        kernel, Val(:record_trajectory!); max_iterations=2,
        argument_types=Tuple{Bool,Vector{Float64},Vector{Float64},Float64,Float64})
    reset_args = map(_rhmc_trace_value, ([0.0, 0.0], [0.0, 0.0], 0.0))
    event_args = map(_rhmc_trace_value,
        (true, [1.0, 2.0], [3.0, 4.0], 5.0, 6.0))
    compiled_reset = @compile reset(state, reset_args...)
    compiled_trajectory = @compile trajectory(state, event_args...)
    output = compiled_reset(state, reset_args...)
    @test Bool(_rhmc_host_value(output.returned))
    @test !Bool(_rhmc_host_value(output.result))
    state = output.state
    output = compiled_trajectory(state, event_args...)
    @test Bool(_rhmc_host_value(output.returned))
    @test !Bool(_rhmc_host_value(output.result))
    state = output.state
    before = map(_rhmc_host_value, state)
    overflow_args = map(_rhmc_trace_value,
        (true, [7.0, 8.0], [9.0, 10.0], 11.0, 12.0))
    output = compiled_trajectory(state, overflow_args...)
    @test Bool(_rhmc_host_value(output.returned))
    @test Bool(_rhmc_host_value(output.result))
    @test !Bool(_rhmc_host_value(output.control_overflow))
    @test Bool(_rhmc_host_value(output.state.trajectory_overflow))
    @test Array(output.state.positions) == before.positions
    @test Reactant.to_number(output.state.count) == before.count
end
