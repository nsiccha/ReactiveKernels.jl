using ReactiveKernels
using Statistics
using Test

include(joinpath(@__DIR__, "..", "examples", "online_stats.jl"))
const OSE = OnlineStatsExample
using Main.ReactiveKernelsNUTSExample: NUTSDiagnostics

@testset "OnlineStats-style reactive streaming and mergeable snapshots" begin
    @test isbitstype(OSE.MomentsAccumulator{Float64})
    @test_throws MethodError OSE.MomentsAccumulator(Int)
    @test_throws ArgumentError OSE.MomentsAccumulator{Float64}(-1, 0, 0)
    @test_throws ArgumentError OSE.MomentsAccumulator{Float64}(2, 0, -1)
    @test_throws ArgumentError OSE.MomentsAccumulator{Float64}(2, 0, -Inf)
    @test_throws DomainError OSE._nonnegative_m2(-Inf, Inf)
    @test isnan(OSE._nonnegative_m2(NaN, 1.0))
    @test OSE._nonnegative_m2(Inf, Inf) == Inf
    @test isnan(OSE.MomentsAccumulator{Float64}(2, 0, NaN).m2)
    @test OSE.MomentsAccumulator{Float64}(2, 0, Inf).m2 == Inf

    @testset "empty, singleton, and non-finite semantics" begin
        empty_state = OSE.MomentsAccumulator()
        singleton = @inferred OSE.update(empty_state, 3.0)
        empty32 = OSE.MomentsAccumulator(Float32)
        singleton32 = @inferred OSE.update(empty32, 3)

        @test isnan(mean(empty_state))
        @test isnan(var(empty_state))
        @test isnan(var(empty_state; corrected=false))
        @test mean(singleton) === 3.0
        @test isnan(var(singleton))
        @test var(singleton; corrected=false) === 0.0
        @test mean(empty32) isa Float32
        @test var(empty32) isa Float32
        @test mean(singleton32) === 3.0f0
        @test var(singleton32; corrected=false) === 0.0f0

        nan_state = OSE.update(empty_state, NaN)
        inf_state = OSE.update(empty_state, Inf)
        @test isnan(mean(nan_state))
        @test isnan(var(nan_state))
        @test mean(inf_state) == Inf
        @test isnan(var(inf_state))
        @test isnan(mean(OSE.update(inf_state, 1.0)))
    end

    @testset "streaming updates match stable batch references" begin
        data = Float64[3, -1, 4, 1, 5, -9, 2, 6, 5, 3, 5]
        state = OSE.fit(data)
        @test state.n == length(data)
        @test mean(state) ≈ mean(data)
        @test var(state) ≈ var(data)
        @test var(state; corrected=false) ≈ var(data; corrected=false)
        @test state.m2 >= 0

        constant_data = fill(1.0e12, 1_001)
        constant_state = OSE.fit(constant_data)
        @test mean(constant_state) == 1.0e12
        @test var(constant_state) == 0.0
        @test constant_state.m2 == 0.0

        offset_data = 1.0e12 .+ collect(1.0:1_000.0)
        offset_state = OSE.fit(offset_data)
        @test mean(offset_state) ≈ mean(offset_data)
        @test var(offset_state) ≈ var(offset_data) rtol=1e-12
        @test offset_state.m2 >= 0
    end

    @testset "partition merges are exact identities and associative numerically" begin
        data = sin.(collect(1.0:103.0)) .+ 1.0e6
        ranges = (1:1, 2:4, 5:21, 22:103)
        parts = map(range -> OSE.fit(@view data[range]), ranges)
        empty_state = OSE.MomentsAccumulator()

        @test isequal(merge(empty_state, parts[1]), parts[1])
        @test isequal(merge(parts[1], empty_state), parts[1])

        merged = reduce(merge, parts)
        streamed = OSE.fit(data)
        @test merged.n == streamed.n
        @test mean(merged) ≈ mean(streamed) rtol=1e-14
        @test var(merged) ≈ var(streamed) rtol=1e-10
        @test merged.m2 >= 0

        left_grouped = merge(merge(parts[1], parts[2]),
                             merge(parts[3], parts[4]))
        right_grouped = merge(parts[1],
                              merge(parts[2], merge(parts[3], parts[4])))
        @test mean(left_grouped) ≈ mean(right_grouped) rtol=1e-14
        @test left_grouped.m2 ≈ right_grouped.m2 rtol=1e-10

        # Float16 cannot represent 80_000 as a finite value. Count ratios and
        # the Chan cross term must therefore be evaluated in widened arithmetic
        # before the resulting sufficient statistics are stored back in Float16.
        zeros16 = OSE.fit(Float16, zeros(Float16, 40_000))
        ones16 = OSE.fit(Float16, ones(Float16, 40_000))
        forward16 = @inferred merge(zeros16, ones16)
        reverse16 = @inferred merge(ones16, zeros16)
        for combined in (forward16, reverse16)
            @test combined.n == 80_000
            @test mean(combined) isa Float16
            @test isfinite(mean(combined))
            @test mean(combined) ≈ Float16(0.5)
            @test isfinite(combined.m2)
            @test combined.m2 ≈ Float16(20_000)
            @test @inferred(var(combined)) isa Float16
            @test isfinite(var(combined))
            @test var(combined) ≈ Float16(0.25)
            @test var(combined; corrected=false) ≈ Float16(0.25)
        end
        @test isequal(forward16, reverse16)
        advanced16 = @inferred OSE.update(forward16, one(Float16))
        @test advanced16.n == 80_001
        @test isfinite(advanced16.mean)
        @test isfinite(advanced16.m2)
        @test isfinite(var(advanced16))

        uneven16 = (
            OSE.fit(Float16, zeros(Float16, 30_000)),
            OSE.fit(Float16, ones(Float16, 40_000)),
            OSE.fit(Float16, fill(Float16(2), 10_000)),
        )
        left16 = merge(merge(uneven16[1], uneven16[2]), uneven16[3])
        right16 = merge(uneven16[1], merge(uneven16[2], uneven16[3]))
        @test left16.n == right16.n == 80_000
        @test isfinite(left16.mean) && isfinite(right16.mean)
        @test left16.mean ≈ Float16(0.75) atol=eps(Float16)
        @test right16.mean ≈ Float16(0.75) atol=eps(Float16)
        @test isfinite(left16.m2) && isfinite(right16.m2)
        @test left16.m2 ≈ right16.m2 rtol=Float16(4) * eps(Float16)
        @test isfinite(var(left16)) && isfinite(var(right16))
    end

    @testset "HMC transition diagnostics are fixed-size and mergeable" begin
        empty = OSE.HMCDiagnosticsAccumulator()
        @test isbitstype(typeof(empty))
        @test OSE.sample_count(empty) == 0
        @test OSE.max_tree_depth(empty) == 0
        @test empty.n_divergent == 0
        @test isnan(OSE.divergence_rate(empty))
        @test isnan(OSE.divergence_percent(empty))
        @test isnan(OSE.mean_tree_depth(empty))
        @test isnan(OSE.mean_leapfrog_steps(empty))
        @test isnan(OSE.mean_acceptance_rate(empty))
        @test isnan(OSE.mean_energy_error(empty))
        @test isnan(OSE.mean_stepsize(empty))

        # Acceptance rate is still validated even though it now arrives inside
        # a canonical NUTSDiagnostics record.
        @test_throws DomainError OSE.record_transition(
            empty, NUTSDiagnostics(3, 7, -0.1, false, 0.0), 0.1)
        @test_throws DomainError OSE.record_transition(
            empty, NUTSDiagnostics(3, 7, 1.1, false, 0.0), 0.1)
        @test_throws DomainError OSE.record_transition(
            empty, NUTSDiagnostics(3, 7, NaN, false, 0.0), 0.1)
        # Non-negative tree depth and leapfrog count.
        @test_throws DomainError OSE.record_transition(
            empty, NUTSDiagnostics(-1, 7, 0.8, false, 0.0), 0.1)
        @test_throws DomainError OSE.record_transition(
            empty, NUTSDiagnostics(3, -1, 0.8, false, 0.0), 0.1)

        # Step size: NaN (the default) is an explicit UNAVAILABLE — recorded
        # without error and never folded — while a finite non-positive or
        # infinite value is invalid telemetry and is rejected.
        valid = NUTSDiagnostics(3, 7, 0.8, false, -0.05)
        unavailable = OSE.record_transition(empty, valid)
        @test OSE.sample_count(unavailable) == 1
        @test unavailable.stepsize.n == 0
        @test isnan(OSE.mean_stepsize(unavailable))
        @test OSE.record_transition(empty, valid, NaN).stepsize.n == 0
        available = OSE.record_transition(empty, valid, 0.2)
        @test available.stepsize.n == 1
        @test OSE.mean_stepsize(available) == 0.2
        @test_throws DomainError OSE.record_transition(empty, valid, 0.0)
        @test_throws DomainError OSE.record_transition(empty, valid, -0.5)
        @test_throws DomainError OSE.record_transition(empty, valid, Inf)

        # `missing` is a first-class unavailable step size in BOTH the direct
        # call and the named-tuple fit path: retain the transition, never fold,
        # and leave mean_stepsize NaN.
        missing_direct = OSE.record_transition(empty, valid, missing)
        @test OSE.sample_count(missing_direct) == 1
        @test missing_direct.stepsize.n == 0
        @test isnan(OSE.mean_stepsize(missing_direct))
        missing_fit = OSE.fit_diagnostics([
            (diagnostics=valid, stepsize=missing),
            (diagnostics=NUTSDiagnostics(2, 3, 0.7, false, -0.01), stepsize=0.2),
        ])
        @test OSE.sample_count(missing_fit) == 2
        @test missing_fit.stepsize.n == 1              # only the available one
        @test OSE.mean_stepsize(missing_fit) == 0.2

        # A divergent transition's non-finite energy error is captured by the
        # divergence count, not folded into the energy-error moments.
        diverged = OSE.record_transition(
            empty, NUTSDiagnostics(4, 9, 0.6, true, -Inf), 0.15)
        @test diverged.n_divergent == 1
        @test OSE.sample_count(diverged) == 1
        @test diverged.energy_error.n == 0
        @test isnan(OSE.mean_energy_error(diverged))
        @test OSE.max_tree_depth(diverged) == 4

        # A non-finite energy error on a NON-divergent transition is invalid
        # telemetry and is rejected; a divergent non-finite (incl. NaN) error is
        # accepted count-only.
        @test_throws DomainError OSE.record_transition(
            empty, NUTSDiagnostics(3, 7, 0.8, false, -Inf), 0.15)
        @test_throws DomainError OSE.record_transition(
            empty, NUTSDiagnostics(3, 7, 0.8, false, NaN), 0.15)
        divergent_nan = OSE.record_transition(
            empty, NUTSDiagnostics(4, 9, 0.6, true, NaN), 0.15)
        @test divergent_nan.n_divergent == 1
        @test divergent_nan.energy_error.n == 0
        @test isnan(OSE.mean_energy_error(divergent_nan))

        # Constructor invariants over the retargeted field layout.
        m1 = OSE.update(OSE.MomentsAccumulator(), 0.8)
        e = OSE.MomentsAccumulator()
        @test_throws ArgumentError OSE.HMCDiagnosticsAccumulator{Float64}(
            1, 0, e, e, e, e, e)               # divergences exceed count
        @test_throws ArgumentError OSE.HMCDiagnosticsAccumulator{Float64}(
            0, 0, m1, e, e, e, e)              # per-transition counts disagree
        @test_throws ArgumentError OSE.HMCDiagnosticsAccumulator{Float64}(
            0, -1, e, e, e, e, e)             # negative maximum tree depth
        @test_throws ArgumentError OSE.HMCDiagnosticsAccumulator{Float64}(
            0, 3, e, e, e, e, e)              # empty state, nonzero max depth
        @test_throws ArgumentError OSE.HMCDiagnosticsAccumulator{Float64}(
            0, 0, m1, m1, m1, e, OSE.update(m1, 0.1))  # stepsize count exceeds n

        records = [
            (diagnostics=NUTSDiagnostics(3,  7, 0.91, false, -0.05), stepsize=0.25),
            (diagnostics=NUTSDiagnostics(5, 15, 0.83, true,  -6.0),  stepsize=0.20),
            (diagnostics=NUTSDiagnostics(4,  5, 0.97, false, -0.01), stepsize=0.30),
            (diagnostics=NUTSDiagnostics(6, 31, 0.76, true,  -8.0),  stepsize=0.15),
            (diagnostics=NUTSDiagnostics(3,  9, 0.88, false, -0.02), stepsize=0.22),
            (diagnostics=NUTSDiagnostics(4,  6, 0.94, false, -0.03), stepsize=0.28),
            (diagnostics=NUTSDiagnostics(7, 63, 0.69, true,  -9.0),  stepsize=0.10),
        ]
        diags = [record.diagnostics for record in records]
        fitted = @inferred OSE.fit_diagnostics(records)
        @test OSE.sample_count(fitted) == length(records)
        @test fitted.n_divergent == count(d -> d.diverged, diags)
        @test OSE.max_tree_depth(fitted) == maximum(d -> d.depth, diags)
        @test OSE.divergence_rate(fitted) ≈ mean(d.diverged for d in diags)
        @test OSE.divergence_percent(fitted) ≈ 100 * mean(d.diverged for d in diags)
        @test OSE.mean_acceptance_rate(fitted) ≈ mean(d.acceptance_rate for d in diags)
        @test OSE.mean_tree_depth(fitted) ≈ mean(d.depth for d in diags)
        @test OSE.mean_leapfrog_steps(fitted) ≈ mean(d.n_steps for d in diags)
        @test OSE.mean_energy_error(fitted) ≈ mean(d.energy_error for d in diags)
        @test OSE.mean_stepsize(fitted) ≈ mean(record.stepsize for record in records)
        @test var(fitted.depth) ≈ var([Float64(d.depth) for d in diags])
        @test var(fitted.energy_error) ≈ var([d.energy_error for d in diags])
        @test var(fitted.stepsize) ≈ var([record.stepsize for record in records])

        # Bare NUTSDiagnostics (no paired step size) records an unavailable one.
        bare = OSE.fit_diagnostics(diags)
        @test OSE.sample_count(bare) == length(diags)
        @test bare.stepsize.n == 0
        @test isnan(OSE.mean_stepsize(bare))
        @test OSE.mean_acceptance_rate(bare) ≈ OSE.mean_acceptance_rate(fitted)
        @test OSE.max_tree_depth(bare) == OSE.max_tree_depth(fitted)

        parts = (
            OSE.fit_diagnostics(records[1:1]),
            OSE.fit_diagnostics(records[2:3]),
            OSE.fit_diagnostics(records[4:7]),
        )
        @test isequal(merge(empty, parts[1]), parts[1])
        @test isequal(merge(parts[1], empty), parts[1])
        merged = reduce(merge, parts)
        @test merged.n_divergent == fitted.n_divergent
        @test OSE.sample_count(merged) == OSE.sample_count(fitted)
        @test OSE.max_tree_depth(merged) == OSE.max_tree_depth(fitted)
        @test OSE.mean_acceptance_rate(merged) ≈ OSE.mean_acceptance_rate(fitted)
        @test OSE.mean_tree_depth(merged) ≈ OSE.mean_tree_depth(fitted)
        @test OSE.mean_leapfrog_steps(merged) ≈ OSE.mean_leapfrog_steps(fitted)
        @test OSE.mean_energy_error(merged) ≈ OSE.mean_energy_error(fitted)
        @test OSE.mean_stepsize(merged) ≈ OSE.mean_stepsize(fitted)

        left_grouped = merge(merge(parts[1], parts[2]), parts[3])
        right_grouped = merge(parts[1], merge(parts[2], parts[3]))
        @test left_grouped.n_divergent == right_grouped.n_divergent
        @test OSE.max_tree_depth(left_grouped) == OSE.max_tree_depth(right_grouped)
        @test OSE.mean_acceptance_rate(left_grouped) ≈
              OSE.mean_acceptance_rate(right_grouped)
        @test OSE.mean_energy_error(left_grouped) ≈
              OSE.mean_energy_error(right_grouped)
        @test OSE.mean_stepsize(left_grouped) ≈ OSE.mean_stepsize(right_grouped)

        # Count-derived rates must also avoid converting large valid counts
        # through narrow storage before division.
        moments16 = OSE.MomentsAccumulator{Float16}(40_000, 0.5, 1.0)
        narrow_a = OSE.HMCDiagnosticsAccumulator{Float16}(
            10_000, 5, moments16, moments16, moments16, moments16, moments16)
        narrow_b = OSE.HMCDiagnosticsAccumulator{Float16}(
            20_000, 6, moments16, moments16, moments16, moments16, moments16)
        narrow = @inferred merge(narrow_a, narrow_b)
        @test OSE.sample_count(narrow) == 80_000
        @test narrow.n_divergent == 30_000
        @test OSE.max_tree_depth(narrow) == 6
        @test OSE.divergence_rate(narrow) === Float16(0.375)
        @test OSE.divergence_percent(narrow) === Float16(37.5)
    end

    @testset "ReactiveHMC-shaped method-bearing @kernel moments" begin
        statistics = OSE.online_moments(1)
        @test statistics isa OSE.OnlineMoments
        @test statistics.n == 0.0
        @test statistics.mean == [0.0]
        @test statistics.var == [0.0]

        @test @inferred(OSE.update!(statistics, 1.0)) === statistics
        @test statistics.n == 1.0
        @test mean(statistics) == [1.0]
        @test all(isnan, var(statistics))
        @test var(statistics; corrected=false) == [0.0]

        @test OSE.update!(statistics, 3.0) === statistics
        @test statistics.n == 2.0
        @test mean(statistics) == [2.0]
        @test var(statistics) == [2.0]
        @test OSE.snapshot(statistics) == (n=2.0, mean=[2.0], var=[1.0])

        # Vector and matrix calls execute the two authored overloads.
        fitted = OSE.online_moments(2)
        OSE.step!(fitted, [1.0, 2.0])
        OSE.step!(fitted, [3.0 5.0; 4.0 8.0])
        @test fitted.n == 3.0
        @test fitted.mean ≈ [3.0, 14 / 3]

        scalar = OSE.online_moments(1)
        @test OSE.fit!(scalar, [3.0, -1.0, 4.0, 1.0, 5.0]) === scalar
        @test only(scalar.mean) ≈ mean([3.0, -1.0, 4.0, 1.0, 5.0])

        # Copy detaches compiler-owned state while sharing the compiled kernel.
        clone = copy(scalar)
        @test clone.kernel === scalar.kernel
        OSE.update!(clone, 99.0)
        @test clone.n == scalar.n + 1
        @test scalar.n == 5

        state32 = OSE.online_moments(1, Float32)
        @test OSE.update!(state32, 3) === state32
        @test eltype(state32.mean) === Float32
        @test eltype(mean(state32)) === Float32
        @test eltype(var(state32)) === Float32

        @test OSE.reset!(statistics) === statistics
        @test statistics.n == 0.0
        @test statistics.mean == [0.0]
    end

    @testset "mutable diagnostics convenience wrapper" begin
        statistics = OSE.online_diagnostics()
        @test statistics isa OSE.OnlineDiagnostics
        @test OSE.sample_count(statistics) == 0
        @test isnan(OSE.divergence_percent(statistics))

        valid = NUTSDiagnostics(3, 7, 0.8, false, -0.05)
        @test @inferred(OSE.record!(statistics, valid, 0.2)) === statistics
        @test OSE.sample_count(statistics) == 1
        @test OSE.max_tree_depth(statistics) == 3
        @test OSE.mean_acceptance_rate(statistics) == 0.8
        @test OSE.mean_energy_error(statistics) == -0.05
        @test OSE.mean_stepsize(statistics) == 0.2
        @test OSE.snapshot(statistics) ==
              OSE.record_transition(OSE.HMCDiagnosticsAccumulator(), valid, 0.2)

        # Validate every field before the first graph-owned source write: a bad
        # transition leaves the state unchanged.
        before = OSE.snapshot(statistics)
        @test_throws DomainError OSE.record!(
            statistics, NUTSDiagnostics(4, 9, 1.1, false, 0.0), 0.1)
        @test OSE.snapshot(statistics) == before

        records = [
            (diagnostics=NUTSDiagnostics(3, 7, 0.91, false, -0.05), stepsize=0.25),
            (diagnostics=NUTSDiagnostics(5, 15, 0.83, true, -6.0), stepsize=0.20),
            (diagnostics=NUTSDiagnostics(2, 5, 0.96, false, -0.01),),
        ]
        fitted = OSE.online_diagnostics()
        @test OSE.fit!(fitted, records) === fitted
        @test OSE.snapshot(fitted) == OSE.fit_diagnostics(records)

        clone = copy(fitted)
        OSE.record!(clone, valid, missing)
        @test OSE.sample_count(clone) == OSE.sample_count(fitted) + 1
        @test OSE.sample_count(fitted) == length(records)

        @test OSE.reset!(statistics) === statistics
        @test OSE.sample_count(statistics) == 0
        @test isnan(OSE.mean_stepsize(statistics))
    end

    @testset "metric adaptation uses the method-bearing welford kernel" begin
        report = OSE.metric_adaptation_report()
        @test report.dimension == 3
        @test report.count == length(OSE.METRIC_ADAPTATION_DRAWS)
        @test length(report.mean) == 3
        @test length(report.variance) == 3
        @test all(report.variance .>= 0)

        # The report matches the page's exact ReactiveHMC-shaped kernel folded
        # the same way (this is the sampler's metric-adaptation statistic).
        reference = OSE.online_moments(3)
        for value in OSE.METRIC_ADAPTATION_DRAWS
            OSE.step!(reference, value)
        end
        @test report.mean ≈ reference.mean
        @test report.variance ≈ reference.var

        @test_throws ArgumentError OSE.metric_adaptation_report(Vector{Float64}[])
        @test_throws DimensionMismatch OSE.metric_adaptation_report(
            [[1.0, 2.0, 3.0], [1.0, 2.0]])
    end

    @testset "partition snapshots retain an inferred allocation-free kernel" begin
        model = OSE.build_partition_graph()
        @test model isa KernelSpec
        @test inputs(model) == (model.left_partition, model.right_partition)
        @test outputs(model) ==
              (model.merged, model.merged_average, model.merged_variance)

        merge_kernel = prepare(model;
            have=(:left_partition, :right_partition), want=:merged)
        summary_kernel = prepare(model;
            have=(:left_partition, :right_partition),
            want=(:merged_average, :merged_variance))

        empty_state = OSE.MomentsAccumulator()
        one_state = OSE.fit([1.0])
        two_state = OSE.fit([2.0, 3.0])
        merged = @inferred merge_kernel(one_state, two_state)
        @test merged isa
              OSE.MomentsAccumulator{Float64}
        @test merged == OSE.fit([1.0, 2.0, 3.0])
        @test @inferred(summary_kernel(one_state, two_state)) == (2.0, 1.0)

        partition_report = OSE.partition_performance_report(
            merge_kernel; iterations=10_000)
        @test partition_report.allocated_bytes == 0
        @test partition_report.elapsed_ns > 0
        @test partition_report.nanoseconds_per_merge > 0
        @test partition_report.result.n == 10_000

        diagnostics_report = OSE.diagnostics_performance_report(
            iterations=100)
        @test diagnostics_report.allocated_bytes >= 0
        @test diagnostics_report.elapsed_ns > 0
        @test diagnostics_report.nanoseconds_per_update > 0
        @test OSE.sample_count(diagnostics_report.result) == 100
        @test diagnostics_report.result.n_divergent == fld(100, 31)

        reactive_report = OSE.reactive_performance_report(iterations=100)
        @test reactive_report.allocated_bytes >= 0
        @test reactive_report.elapsed_ns > 0
        @test reactive_report.nanoseconds_per_update > 0
        @test reactive_report.result.n == 100
    end

    @testset "stateful Welford plan and partition DAG remain inspectable" begin
        statistics = OSE.online_moments(3)
        roles = Dict(
            String(slot.path[end]) => ReactiveKernels.kernel_plan_field(
                statistics.kernel.prepared.plan, slot.canon)[1]
            for slot in ReactiveKernels.kernel_plan_slots(statistics.kernel.prepared.plan)
        )
        @test all(roles[name] === :owned for name in ("n", "mean", "var"))
        @test roles["template"] === :shared

        model = OSE.build_partition_graph()
        merge_plan = plan(model;
            have=(:left_partition, :right_partition),
            want=(:merged, :merged_average, :merged_variance))
        view = visualize(merge_plan)
        html = sprint(show, MIME"text/html"(), view)
        dot = dot_source(view)

        @test occursin("class=\"rk-node value have\"", html)
        @test occursin("class=\"rk-node value want\"", html)
        @test occursin("rk-node recipe selected", html)
        @test occursin("#dcfce7", dot)
        @test occursin("#ffedd5", dot)
        @test occursin("#dbeafe", dot)
        @test occursin("merge", dot)
    end

    @testset "public docs show exact pinned @kernel authoring and separate merge" begin
        page = read(joinpath(@__DIR__, "..", "docs", "src", "online-stats.md"),
                    String)
        @test occursin("render_online_stats_welford_source()", page)
        @test occursin("ca9ea4ca41924bb0e1fadc01c717e1333916aba6", page)
        @test occursin("online_moments(3)", page)
        @test occursin("OnlineStatsExample.step!", page)
        @test occursin("compile_stateful", page)
        @test occursin("step!(__self__, xi; kwargs...)", page)
        @test occursin("@kernel partitions(", page)
        @test occursin("NUTSDiagnostics(3, 7, 0.91, false, -0.05)", page)
        @test occursin("Main.ReactiveKernelsDocs.execute_example", page)
        @test !occursin("@" * "reactive", page)
        @test !occursin("include(", page)
        @test !occursin("Graph()", page)
        @test !occursin("value!(", page)
        @test !occursin("add!(", page)
    end
end
