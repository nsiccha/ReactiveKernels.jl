# Incremental and mergeable online statistics

Online mean and variance are a compact example of the boundary between pure
state transitions and reactive orchestration. A `MomentsAccumulator{T}` stores
only `(n, mean, m2)`. Welford's update consumes one observation, while Chan's
parallel formula combines independently processed partitions. The same
building block also summarizes HMC acceptance rates, leapfrog-step counts, and
step sizes, alongside an exact divergence count. All operations return new
immutable values, so they are ordinary pure recipes rather than hidden
mutation inside the graph.

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
@kernel updates(state::MomentsAccumulator{Float64}, observation::Float64) = begin
    updated::MomentsAccumulator{Float64} =
        OnlineStatsExample.update(state, observation)
    average::Float64 = Statistics.mean(updated)
    sample_variance::Float64 = Statistics.var(updated)
end

@kernel partitions(left_partition::MomentsAccumulator{Float64},
                   right_partition::MomentsAccumulator{Float64}) = begin
    merged::MomentsAccumulator{Float64} =
        Base.merge(left_partition, right_partition)
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

## HMC sampling diagnostics

`HMCDiagnosticsAccumulator{T}` consumes the scalar transition streams exposed
by ReactiveHMC's sampling statistics: `acc_rate`, `n_steps`, `stepsize`, and
`diverged`. It stores online moments for the first three and an exact divergence
count. Independently summarized chain segments are mergeable, and the empty
state is an exact identity.

The compact block below is again the literal source executed by the docs build.
Its Generated and Compute DAG panes therefore show the actual HMC diagnostics
subkernel, not setup or include plumbing.

```@eval
Main.ReactiveKernelsDocs.execute_example(@__MODULE__, raw"""
diagnostics = @kernel begin
    diagnostics_state::HMCDiagnosticsAccumulator{Float64}
    acceptance_rate_observation::Float64
    leapfrog_steps_observation::Int
    stepsize_observation::Float64
    diverged_observation::Bool
    updated_diagnostics::HMCDiagnosticsAccumulator{Float64} =
        OnlineStatsExample.record_transition(
            diagnostics_state,
            acceptance_rate_observation,
            leapfrog_steps_observation,
            stepsize_observation,
            diverged_observation,
        )
    divergence_percent::Float64 =
        OnlineStatsExample.divergence_percent(updated_diagnostics)
    mean_acceptance_rate::Float64 =
        OnlineStatsExample.mean_acceptance_rate(updated_diagnostics)
    mean_leapfrog_steps::Float64 =
        OnlineStatsExample.mean_leapfrog_steps(updated_diagnostics)
    mean_stepsize::Float64 =
        OnlineStatsExample.mean_stepsize(updated_diagnostics)
    return updated_diagnostics, divergence_percent,
           mean_acceptance_rate, mean_leapfrog_steps, mean_stepsize
end

diagnostics_kernel = prepare(diagnostics;
    have = (:diagnostics_state, :acceptance_rate_observation,
            :leapfrog_steps_observation, :stepsize_observation,
            :diverged_observation),
    want = (:updated_diagnostics, :divergence_percent,
            :mean_acceptance_rate, :mean_leapfrog_steps, :mean_stepsize))

seed = HMCDiagnosticsAccumulator()
inputs = (seed, 0.91, 7, 0.25, false)
output = diagnostics_kernel(inputs...)

docs_example = (;
    name = :hmc_transition_diagnostics,
    origin = "compact @kernel HMC diagnostics reducer (build executed)",
    inputs,
    kernel = diagnostics_kernel,
    output,
)
"""; setup = Main.ReactiveKernelsDocs.setup_online_stats!)
```

```julia
records = [
    (acc_rate=0.91, n_steps=7,  stepsize=0.25, diverged=false),
    (acc_rate=0.83, n_steps=15, stepsize=0.20, diverged=true),
]
diagnostics = OnlineStatsExample.fit_diagnostics(records)

@assert OnlineStatsExample.sample_count(diagnostics) == 2
@assert diagnostics.n_divergent == 1
@assert OnlineStatsExample.divergence_percent(diagnostics) == 50.0
```

This fixed-size reducer intentionally does not approximate rank-normalized
R-hat or bulk/tail ESS: exact versions require retained, ordered draws from
multiple chains. WarmupHMC reports those from its retained sampling history,
while divergence count and percent are represented exactly here. Vector
Welford state used for diagonal-metric adaptation is also separate from this
scalar sampling-diagnostics surface.

Empty means and variances are `NaN`. A singleton has a finite mean, `NaN`
corrected variance, and zero uncorrected variance. Non-finite observations
propagate `NaN` or positive infinity; a negative `m2`, including negative
infinity, is rejected. Constant and large-offset streams retain nonnegative
`m2`; only a finite negative value within the explicit floating-point rounding
tolerance is clamped to zero. Count ratios use widened arithmetic before the
result is stored back in `T`, so narrow storage such as `Float16` remains valid
when a merged count exceeds its largest finite value.

## Reactive invalidation is explicit

`ReactiveState` treats `state` and `observation` as authoritative inputs.
Replacing an observation invalidates and recomputes `updated`; it does not
silently fold into the previous result. Advancing a stream means explicitly
promoting the returned accumulator to the next `state` input. Frozen values
remain cut points until `unfreeze!` is called.

## Mutation-friendly reactive authoring

`MomentsAccumulator` is immutable on purpose: for a three-field sufficient
statistic there is nothing to mutate in place, so pure `set!`/`get!` replacement
is both correct and allocation-cheap. When state is instead a large **mutable**
object — an array, a sampler phase point — where an in-place update avoids
reallocating the whole buffer, ReactiveKernels provides a compiled reactive
surface that keeps the same invalidation, freeze, and checkpoint guarantees.

`prepare_reactive` fixes the have/want boundary once and returns a
[`ReactiveProgram`](api.md); instantiate it to a `CompiledReactiveState`. Inside
a `mutate!` transaction you edit the stored object for a declared source in
place, and downstream slots are invalidated and lazily recomputed automatically
— no manual round-trip through the returned value:

```julia
g = @kernel begin
    weights::Vector{Float64}
    total::Float64 = sum(weights)
    return total
end

program = prepare_reactive(g; have = (:weights,), want = (:total,))
state   = program([1.0, 2.0, 3.0])
total   = statevalue(program, port(g, :total))

get!(state, total)                       # 6.0
mutate!(state, port(g, :weights)) do w   # in-place edit of the owned buffer
    w[1] = 10.0
end
get!(state, total)                       # 15.0, recomputed on demand
```

Derived values live in owned typed slots, so `freeze!`/`unfreeze!` and
`checkpoint` behave exactly as they do for `ReactiveState`: a frozen `total`
stays fixed under later `mutate!` calls, and a checkpoint replays into a fresh
`program(...; frozen = cp)` instance. This is the supported way to get
mutation-friendly (`!!`-style) ergonomics inside a reactive layer. The direct
[`prepare_nonallocating`](nonallocating.md) kernels are a separate,
single-caller optimization whose borrowed caches are not persisted through the
reactive layers.

## Measurement contract

`kernel_performance_report` warms the generated update kernel, measures
steady-state allocations in one run, and elapsed time in a separate run.
`diagnostics_performance_report` applies the same measurement contract to the
HMC diagnostics update kernel.
`reactive_performance_report` does the same for the orchestration path, whose
allocation count deliberately includes source versioning, invalidation,
planning-cache lookup, and materialization. Timings are reported observations,
not pass/fail thresholds.

```julia
model = OnlineStatsExample.build_online_stats_graph()
kernel = prepare(model; have = (:state, :observation), want = :updated)

direct = OnlineStatsExample.kernel_performance_report(kernel)
diagnostics_kernel = prepare(model;
    have = (:diagnostics_state, :acceptance_rate_observation,
            :leapfrog_steps_observation, :stepsize_observation,
            :diverged_observation),
    want = :updated_diagnostics)
diagnostics = OnlineStatsExample.diagnostics_performance_report(
    diagnostics_kernel)
reactive = OnlineStatsExample.reactive_performance_report(model)
```

Run the full walkthrough from the repository root:

```sh
julia --project=. examples/online_stats.jl
```
