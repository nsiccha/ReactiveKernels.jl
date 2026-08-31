# Online statistics: the ReactiveHMC kernel, compiled

This page uses the actual stateful shape from
[ReactiveHMC.jl at `ca9ea4c`](https://github.com/nsiccha/ReactiveHMC.jl/blob/ca9ea4ca41924bb0e1fadc01c717e1333916aba6/src/adaptation.jl#L47-L59):
one object owns `n`, `mean`, and `var`; `step!` updates a vector; and the matrix
overload folds its columns through the same vector method. ReactiveKernels spells
that object with the one public authoring macro, a method-bearing `@kernel`.

The streaming source authority is
[`ReactiveKernelsStreamingStats`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsStreamingStats/src/ReactiveKernelsStreamingStats.jl);
HMC summaries live downstream in `ReactiveKernelsHMCDiagnostics`. The source
below is extracted from the streaming package during the documentation build.

## Welford as a method-bearing `@kernel`

```@eval
Main.ReactiveKernelsDocs.render_online_stats_welford_source()
```

The only intentional textual difference from ReactiveHMC is that its
`smooth(old, new, w)` helper is expanded to `(1-w)*old + w*new`. That keeps the
same recurrence visible in the captured MethodIR using only the compiler's
supported arithmetic primitives. The field layout, vector method, keyword
weight (`dn`), and matrix-to-vector forwarding all follow the pinned source.

Compile once, instantiate compiler-owned state, and execute either overload:

```julia
statistics = OnlineStatsExample.online_moments(3)

OnlineStatsExample.step!(statistics, [1.0, 2.0, 4.0])
OnlineStatsExample.step!(statistics, [3.0 5.0; 4.0 8.0; 2.0 6.0])

statistics.n
statistics.mean
statistics.var
```

`online_moments` is a small example wrapper around the public compiler flow:

```julia
compiled = compile_stateful(
    OnlineStatsExample.welford_var,
    zeros(3),
)
state = compiled(zeros(3))

ReactiveKernels.stateful_call!(state, Val(:step!), [1.0, 2.0, 4.0])
state.mean
state.var
```

The stateful compiler derives ownership and invalidation from the authored
methods. In particular, the matrix overload's `step!(__self__, xi; kwargs...)`
edge makes the same `n`, `mean`, and `var` writes visible interprocedurally; it is
not a second hand-maintained recurrence.

## Streaming state is not a partition value

ReactiveHMC's Welford object is an in-place, history-dependent metric-adaptation
state. It is not mergeable. Partition reduction is a different operation with a
different representation: `MomentsAccumulator{T}` stores scalar `(n, mean, m2)`
snapshots, and `Base.merge` combines them with Chan's parallel-variance formula.
Keeping this lane separate avoids pretending the sampler's Welford state has an
associative merge that it does not implement.

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

For a scalar stream:

```julia
data = [1.0, 2.0, 4.0, 8.0]
streamed = OnlineStatsExample.fit(data)

parts = (
    OnlineStatsExample.fit(data[1:1]),
    OnlineStatsExample.fit(data[2:3]),
    OnlineStatsExample.fit(data[4:4]),
)
partitioned = reduce(merge, parts)

@assert Statistics.mean(partitioned) ≈ Statistics.mean(streamed)
@assert Statistics.var(partitioned) ≈ Statistics.var(streamed)
```

## HMC transition diagnostics are a separate reducer

`HMCDiagnosticsAccumulator{T}` consumes the external sampler's canonical
`NUTSDiagnostics`: depth, leapfrog count, acceptance rate, divergence, and
energy error. It also accepts step size separately because step size is not a
field of `NUTSDiagnostics`. Independent summaries can be merged; this reducer
does not claim to implement rank-normalized R-hat or bulk/tail ESS, which require
retained ordered draws from multiple chains.

```julia
records = [
    (diagnostics=NUTSDiagnostics(3, 7, 0.91, false, -0.05), stepsize=0.25),
    (diagnostics=NUTSDiagnostics(5, 15, 0.83, true, -6.0), stepsize=0.20),
    (diagnostics=NUTSDiagnostics(2, 5, 0.96, false, -0.01),),
]
diagnostics = OnlineStatsExample.fit_diagnostics(records)

@assert OnlineStatsExample.sample_count(diagnostics) == 3
@assert diagnostics.n_divergent == 1
@assert OnlineStatsExample.max_tree_depth(diagnostics) == 5
```

Run the complete executable example from the repository root:

```sh
julia --project=packages examples/online_stats.jl
```
