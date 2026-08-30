# Fixed-capacity ReactiveKernels compiler fixture for ReactiveHMC.jl
# src/statistics.jl at ca9ea4ca41924bb0e1fadc01c717e1333916aba6.
#
# The ordinary source uses ElasticMatrix and dynamically growing vectors. A
# static-shape compiler cannot preserve those container types, so this fixture
# makes the representation boundary explicit while retaining the mathematical
# callback order: one centered reset entry, direction-dependent prepend/append,
# reveal indices assigned from the pre-increment count, the exact acceptance
# reduction, and detached per-sample history copies. Capacity exhaustion is a
# sticky Bool field/result and returns before any partial write. There are no
# statistics/field-name compiler hooks; every branch, index, loop, and mutation
# remains visible in the authored MethodIR.
module ReactiveHMCStatisticsFixture

using ReactiveKernels

@kernel statistics_state(
    positions,
    gradients,
    dhams,
    pots,
    idxs,
    draws,
    n_steps,
    stepsizes,
    acc_rate,
    diverged,
    full_history,
    full_idxs,
    history_counts;
    dimension,
    trajectory_capacity,
    sample_capacity,
    reset_first,
    first=1,
    count=0,
    sample_count=0,
    trajectory_overflow=false,
    sampling_overflow=false,
) = begin
    min1exp(x) = x > zero(x) ? one(x) : exp(x)

    reset!(pos, dham_dpos, pot) = begin
        trajectory_overflow = trajectory_capacity < 1
        trajectory_overflow && return trajectory_overflow
        first = reset_first
        count = 1
        for index in 1:dimension
            positions[index, first] = pos[index]
            gradients[index, first] = -dham_dpos[index]
        end
        dhams[first] = zero(pot)
        pots[first] = pot
        idxs[first] = 0
        trajectory_overflow
    end

    record_trajectory!(go_forward, pos, dham_dpos, pot, dham) = begin
        trajectory_overflow && return trajectory_overflow
        column = go_forward ? first + count : first - 1
        if count >= trajectory_capacity || column < 1 || column > trajectory_capacity
            trajectory_overflow = true
            return trajectory_overflow
        end
        for index in 1:dimension
            positions[index, column] = pos[index]
            gradients[index, column] = -dham_dpos[index]
        end
        dhams[column] = dham
        pots[column] = pot
        idxs[column] = count
        go_forward || (first = column)
        count += 1
        trajectory_overflow
    end

    record_sample!(init_pos, stepsize, was_diverged) = begin
        if trajectory_overflow || sampling_overflow
            sampling_overflow = true
            return sampling_overflow
        end
        next_sample = sample_count + 1
        if count < 1 || next_sample > sample_capacity
            sampling_overflow = true
            return sampling_overflow
        end

        for index in 1:dimension
            draws[index, next_sample] = init_pos[index]
        end
        n_steps[next_sample] = count - 1
        stepsizes[next_sample] = stepsize
        diverged[next_sample] = was_diverged

        acceptance_sum = zero(stepsize)
        for offset in 0:(count - 1)
            column = first + offset
            acceptance_sum += min1exp(__self__, dhams[column])
            for index in 1:dimension
                full_history[index, offset + 1, next_sample] = positions[index, column]
            end
            full_idxs[offset + 1, next_sample] = idxs[column]
        end
        acceptance_count = count > 1 ? count - 1 : 1
        acc_rate[next_sample] =
            (acceptance_sum - one(acceptance_sum)) / acceptance_count
        history_counts[next_sample] = count
        sample_count = next_sample
        sampling_overflow
    end
end

function initial_statistics_sources(
    dimension::Integer=2,
    trajectory_capacity::Integer=8,
    sample_capacity::Integer=4,
    ::Type{T}=Float64,
) where {T<:AbstractFloat}
    dimension >= 1 || throw(ArgumentError("dimension must be positive"))
    trajectory_capacity >= 0 || throw(ArgumentError(
        "trajectory_capacity must be nonnegative"))
    sample_capacity >= 0 || throw(ArgumentError(
        "sample_capacity must be nonnegative"))
    (
        positions=zeros(T, dimension, trajectory_capacity),
        gradients=zeros(T, dimension, trajectory_capacity),
        dhams=zeros(T, trajectory_capacity),
        pots=zeros(T, trajectory_capacity),
        idxs=zeros(Int, trajectory_capacity),
        draws=zeros(T, dimension, sample_capacity),
        n_steps=zeros(Int, sample_capacity),
        stepsizes=zeros(T, sample_capacity),
        acc_rate=zeros(T, sample_capacity),
        diverged=falses(sample_capacity),
        full_history=zeros(T, dimension, trajectory_capacity, sample_capacity),
        full_idxs=zeros(Int, trajectory_capacity, sample_capacity),
        history_counts=zeros(Int, sample_capacity),
        dimension=Int(dimension),
        trajectory_capacity=Int(trajectory_capacity),
        sample_capacity=Int(sample_capacity),
        reset_first=max(1, 1 + div(trajectory_capacity - 1, 2)),
        first=1,
        count=0,
        sample_count=0,
        trajectory_overflow=false,
        sampling_overflow=false,
    )
end

end # module ReactiveHMCStatisticsFixture
