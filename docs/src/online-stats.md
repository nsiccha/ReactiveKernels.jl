# Incremental and mergeable online statistics

Online mean and variance are a compact example of the boundary between pure
state transitions and reactive orchestration. A `MomentsAccumulator{T}` stores
only `(n, mean, m2)`. Welford's update consumes one observation, while Chan's
parallel formula combines independently processed partitions. Both operations
return new immutable values, so they are ordinary pure recipes rather than
hidden mutation inside the graph.

The complete runnable implementation is
[`examples/online_stats.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/examples/online_stats.jl).
It fixes the storage type to a floating `T`, which keeps empty and singleton
`NaN` results type-stable.

## One source, generated subkernel, and colored plan

The Raw pane below is the literal compact source executed by the docs build.
It declares update/summary and partition-merge fragments with `@kernel`,
composes them by name, and prepares only the streaming query. The Generated
pane is `code_expr(update_kernel)` from that execution, and the Compute DAG is
the selected `Plan` rendered by `visualize`; the unused partition branch is
therefore absent from both generated views.

```@eval
Main.ReactiveKernelsDocs.execute_example(@__MODULE__, raw"""
updates = @kernel begin
    state::MomentsAccumulator{Float64}
    observation::Float64
    updated::MomentsAccumulator{Float64} =
        OnlineStatsExample.update(state, observation)
    average::Float64 = Statistics.mean(updated)
    sample_variance::Float64 = Statistics.var(updated)
    return updated, average, sample_variance
end

partitions = @kernel begin
    left_partition::MomentsAccumulator{Float64}
    right_partition::MomentsAccumulator{Float64}
    merged::MomentsAccumulator{Float64} =
        Base.merge(left_partition, right_partition)
    return merged
end

model = merge(updates, partitions)
update_kernel = prepare(model;
    have = (:state, :observation),
    want = (:updated, :average, :sample_variance))

seed = MomentsAccumulator()
inputs = (seed, 3.0)
output = update_kernel(inputs...)

docs_example = (;
    name = :online_moments_update,
    origin = "compact @kernel update and merge model (build executed)",
    inputs,
    kernel = update_kernel,
    output,
)
"""; setup = Main.ReactiveKernelsDocs.setup_online_stats!)
```

## Streaming and partitioned execution

The accumulator API is intentionally example-owned: `update` advances one
stream and `Base.merge` combines two sufficient-statistic states. Empty states
are exact merge identities.

```julia
data = [1.0, 2.0, 4.0, 8.0]
streaming = OnlineStatsExample.fit(data)

parts = (
    OnlineStatsExample.fit(data[1:1]),
    OnlineStatsExample.fit(data[2:3]),
    OnlineStatsExample.fit(data[4:4]),
)
partitioned = reduce(merge, parts)

@assert Statistics.mean(partitioned) ≈ Statistics.mean(streaming)
@assert Statistics.var(partitioned) ≈ Statistics.var(streaming)
```

Empty means and variances are `NaN`. A singleton has a finite mean, `NaN`
corrected variance, and zero uncorrected variance. Constant and large-offset
streams retain nonnegative `m2`; only a negative value within the explicit
floating-point rounding tolerance is clamped to zero.

## Reactive invalidation is explicit

`ReactiveState` treats `state` and `observation` as authoritative inputs.
Replacing an observation invalidates and recomputes `updated`; it does not
silently fold into the previous result. Advancing a stream means explicitly
promoting the returned accumulator to the next `state` input. Frozen values
remain cut points until `unfreeze!` is called.

## Measurement contract

`kernel_performance_report` warms the generated update kernel, measures
steady-state allocations in one run, and elapsed time in a separate run.
`reactive_performance_report` does the same for the orchestration path, whose
allocation count deliberately includes source versioning, invalidation,
planning-cache lookup, and materialization. Timings are reported observations,
not pass/fail thresholds.

```julia
model = OnlineStatsExample.build_online_stats_graph()
kernel = prepare(model; have = (:state, :observation), want = :updated)

direct = OnlineStatsExample.kernel_performance_report(kernel)
reactive = OnlineStatsExample.reactive_performance_report(model)
```

Run the full walkthrough from the repository root:

```sh
julia --project=. examples/online_stats.jl
```
