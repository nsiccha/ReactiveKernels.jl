# Stateful online statistics and mergeable snapshots

Online statistics are history-dependent, so the primary interface here follows
ReactiveHMC's original shape: a stateful `@reactive` object owns the running
statistics, and ordinary methods mutate those graph-owned sources. Callers use
`update!`, `fit!`, or `record!`; they do **not** receive a complete replacement
state and wire it back as the next input.

Immutable values still have an important role. `MomentsAccumulator{T}` and
`HMCDiagnosticsAccumulator{T}` are compact snapshots for partition boundaries,
persistence, and Chan-style associative merging. Full-state return is therefore
kept in the merge lane, where value semantics are useful, rather than imposed on
the streaming lane.

The complete executable authority is
[`examples/online_stats.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/examples/online_stats.jl).
The `@reactive` definitions displayed below are extracted from that exact file,
which the documentation build also loads.

## Stateful Welford moments

The three authoritative sources are `n`, `mean`, and `m2`. The summary fields
are compiled reactive recipes: changing any source invalidates only the derived
values that depend on it. The Welford recurrence itself lives visibly inside
the object's `update!` method.

```@eval
Main.ReactiveKernelsDocs.render_online_stats_reactive_source(:moments)
```

The ergonomic streaming path keeps one object identity:

```julia
statistics = OnlineStatsExample.online_moments()

OnlineStatsExample.update!(statistics, 1.0) === statistics
OnlineStatsExample.fit!(statistics, [2.0, 4.0, 8.0]) === statistics

Statistics.mean(statistics)
Statistics.var(statistics)
statistics.n
```

The actual compiled dependency graph remains inspectable. It contains the
derived mean and variance recipes over the mutable HAVE sources; `update!` is
the ordinary stateful method that writes those sources and triggers
invalidation.

```julia
program = reactive_program(statistics)
reactive_program(statistics).plan
explain(program.plan)
code_expr(program, statistics.handles.average)
```

Only materialize an immutable value when it is useful as a value:

```julia
snapshot = OnlineStatsExample.snapshot(statistics)
@assert snapshot isa OnlineStatsExample.MomentsAccumulator{Float64}
```

## Partition merging stays pure

Independent partitions naturally produce immutable sufficient-statistic
snapshots. `Base.merge` combines them with Chan's parallel-variance formula;
the empty snapshot is an exact identity. This is the one public `@kernel` lane
that returns a complete state, deliberately scoped to a partition boundary.

```@eval
Main.ReactiveKernelsDocs.execute_example(@__MODULE__, raw"""
@kernel partitions(left_partition::MomentsAccumulator{Float64},
                   right_partition::MomentsAccumulator{Float64}) = begin
    merged::MomentsAccumulator{Float64} =
        Base.merge(left_partition, right_partition)
    merged_average::Float64 = Statistics.mean(merged)
    merged_variance::Float64 = Statistics.var(merged)
    return merged, merged_average, merged_variance
end

# OnlineStatsExample.build_partition_graph() is the reusable source authority.
merge_kernel = prepare(partitions;
    have = (:left_partition, :right_partition),
    want = (:merged, :merged_average, :merged_variance))

inputs = (
    OnlineStatsExample.fit([1.0, 2.0]),
    OnlineStatsExample.fit([4.0, 8.0]),
)
output = merge_kernel(inputs...)

docs_example = (;
    name = :online_moments_partition_merge,
    origin = "immutable Chan merge at an explicit partition boundary",
    inputs,
    kernel = merge_kernel,
    output,
)
"""; setup = Main.ReactiveKernelsDocs.setup_online_stats!)
```

A streamed object and merged partitions agree without making state replacement
the streaming API:

```julia
data = [1.0, 2.0, 4.0, 8.0]

streaming = OnlineStatsExample.online_moments()
OnlineStatsExample.fit!(streaming, data)

parts = (
    OnlineStatsExample.fit(data[1:1]),
    OnlineStatsExample.fit(data[2:3]),
    OnlineStatsExample.fit(data[4:4]),
)
partitioned = reduce(merge, parts)

@assert Statistics.mean(partitioned) ≈ Statistics.mean(streaming)
@assert Statistics.var(partitioned) ≈ Statistics.var(streaming)
```

## Stateful HMC diagnostics

HMC diagnostics follow the same ownership model. The reactive object owns exact
divergence/max-depth counters and five immutable scalar-moment sources.
`record!` validates a transition, computes every next component before the first
write, and then mutates the graph-owned sources. Derived summaries invalidate
lazily. `snapshot` is again reserved for merging or persistence.

```@eval
Main.ReactiveKernelsDocs.render_online_stats_reactive_source(:diagnostics)
```

Each transition is the canonical external sampler record. Step size remains a
separate adaptation value because it is not a field of `NUTSDiagnostics`:

```julia
diagnostics = OnlineStatsExample.online_diagnostics()

transition::NUTSDiagnostics{Float64} =
    NUTSDiagnostics(3, 7, 0.91, false, -0.05)
OnlineStatsExample.record!(diagnostics, transition, 0.25) === diagnostics

records = [
    (diagnostics=NUTSDiagnostics(5, 15, 0.83, true, -6.0), stepsize=0.20),
    (diagnostics=NUTSDiagnostics(2, 5, 0.96, false, -0.01),),
]
OnlineStatsExample.fit!(diagnostics, records)

@assert OnlineStatsExample.sample_count(diagnostics) == 3
@assert diagnostics.n_divergent == 1
@assert OnlineStatsExample.max_tree_depth(diagnostics) == 5
```

`NaN` or `missing` means an unavailable step size and is not folded. A finite
non-positive or infinite step size is invalid telemetry. A non-finite energy
error is accepted only on a divergent transition, where it contributes to the
exact divergence count but not the finite energy-error moments. These checks
happen before graph-owned state is mutated, so a rejected transition leaves the
object unchanged.

The immutable `HMCDiagnosticsAccumulator` remains mergeable across independently
summarized chain segments. It intentionally does not approximate rank-normalized
R-hat or bulk/tail ESS: exact versions require retained, ordered draws from
multiple chains.

## Vector metric adaptation

The scalar example above contains its own visible Welford recurrence. For the
sampler's vector-valued diagonal metric, the example additionally reuses the
external NUTS exemplar's canonical stateful `welford_var` implementation. That
keeps HMC domain APIs outside the ReactiveKernels package while demonstrating
that the sampler and this page share the same adaptation statistic.

```julia
report = OnlineStatsExample.metric_adaptation_report()

@assert report.count == length(OnlineStatsExample.METRIC_ADAPTATION_DRAWS)
@assert report.dimension == 3
@assert all(report.variance .>= 0)
```

## Numeric and measurement contracts

Empty means and variances are `NaN`. A singleton has a finite mean, `NaN`
corrected variance, and zero uncorrected variance. Non-finite observations
propagate `NaN` or positive infinity; negative `m2`, including negative
infinity, is rejected. Count ratios use widened arithmetic before storage back
in `T`, so narrow types such as `Float16` remain valid when merged counts exceed
their largest finite floating value.

The performance helpers separate construction from steady-state work:

```julia
partition_model = OnlineStatsExample.build_partition_graph()
partition_kernel = prepare(partition_model;
    have = (:left_partition, :right_partition), want = :merged)

partition = OnlineStatsExample.partition_performance_report(partition_kernel)
streaming = OnlineStatsExample.reactive_performance_report()
diagnostics = OnlineStatsExample.diagnostics_performance_report()
```

`partition_performance_report` measures the generated immutable merge kernel.
`reactive_performance_report` and `diagnostics_performance_report` measure
stateful method calls, including source writes and reactive invalidation but
excluding object construction and program preparation. Timings are observations,
not pass/fail thresholds.

Run the complete walkthrough from the repository root:

```sh
julia --project=. examples/online_stats.jl
```
