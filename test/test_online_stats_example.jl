using ReactiveKernels
using Statistics
using Test

include(joinpath(@__DIR__, "..", "examples", "online_stats.jl"))
const OSE = OnlineStatsExample

@testset "OnlineStats-style immutable accumulators" begin
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

    @testset "metric adaptation reuses the canonical welford_var estimator" begin
        report = OSE.metric_adaptation_report()
        @test report.dimension == 3
        @test report.count == length(OSE.METRIC_ADAPTATION_DRAWS)
        @test length(report.mean) == 3
        @test length(report.variance) == 3
        @test all(report.variance .>= 0)

        # The estimate matches the canonical welford_var it reuses, folded the
        # same way (this is the sampler's own metric-adaptation statistic).
        reference = welford_var(3)
        for value in OSE.METRIC_ADAPTATION_DRAWS
            step!(reference, value)
        end
        @test report.mean ≈ reference.mean
        @test report.variance ≈ reference.var

        @test_throws ArgumentError OSE.metric_adaptation_report(Vector{Float64}[])
        @test_throws DimensionMismatch OSE.metric_adaptation_report(
            [[1.0, 2.0, 3.0], [1.0, 2.0]])
    end

    @testset "generated state kernels are inferred and allocation-free" begin
        model = OSE.build_online_stats_graph()
        @test model isa KernelSpec
        @test inputs(model) == (model.state, model.observation)
        @test outputs(model) ==
              (model.updated, model.average, model.sample_variance)

        update_kernel = prepare(model;
            have=(:state, :observation), want=:updated)
        merge_kernel = prepare(model;
            have=(:left_partition, :right_partition), want=:merged)
        summary_kernel = prepare(model;
            have=(:state, :observation), want=(:average, :sample_variance))
        diagnostics_kernel = prepare(model;
            have=(:diagnostics_state, :transition, :stepsize_observation),
            want=(:updated_diagnostics, :diagnostic_max_tree_depth,
                  :diagnostic_divergence_percent,
                  :diagnostic_mean_acceptance_rate,
                  :diagnostic_mean_energy_error, :diagnostic_mean_stepsize))

        empty_state = OSE.MomentsAccumulator()
        one_state = @inferred update_kernel(empty_state, 1.0)
        two_state = @inferred update_kernel(one_state, 3.0)
        @test @inferred(merge_kernel(one_state, two_state)) isa
              OSE.MomentsAccumulator{Float64}
        @test @inferred(summary_kernel(one_state, 3.0)) == (2.0, 2.0)

        empty_diagnostics = OSE.HMCDiagnosticsAccumulator()
        diagnostic_result = @inferred diagnostics_kernel(
            empty_diagnostics, NUTSDiagnostics(3, 7, 0.9, false, -0.05), 0.2)
        @test diagnostic_result[1] isa OSE.HMCDiagnosticsAccumulator{Float64}
        @test diagnostic_result[2:end] == (3, 0.0, 0.9, -0.05, 0.2)

        diagnostics_update_kernel = prepare(model;
            have=(:diagnostics_state, :transition, :stepsize_observation),
            want=:updated_diagnostics)
        diagnostics_report = OSE.diagnostics_performance_report(
            diagnostics_update_kernel; iterations=10_000)
        @test diagnostics_report.allocated_bytes == 0
        @test diagnostics_report.elapsed_ns > 0
        @test diagnostics_report.nanoseconds_per_update > 0
        @test OSE.sample_count(diagnostics_report.result) == 10_000
        @test diagnostics_report.result.n_divergent == fld(10_000, 31)

        report = OSE.kernel_performance_report(update_kernel; iterations=10_000)
        @test report.allocated_bytes == 0
        @test report.elapsed_ns > 0
        @test report.nanoseconds_per_update > 0
        @test report.result.n == 10_000

        reactive_report = OSE.reactive_performance_report(model; iterations=20)
        @test reactive_report.allocated_bytes > 0
        @test reactive_report.elapsed_ns > 0
        @test reactive_report.result.n == 20
    end

    @testset "ReactiveState replacement, invalidation, and frozen cut points" begin
        model = OSE.build_online_stats_graph()
        state = ReactiveState(model; materialize=(model.updated,))
        empty_state = OSE.MomentsAccumulator()

        set!(state, model.state, empty_state)
        set!(state, model.observation, 1.0)
        first_state = get!(state, model.updated)
        @test first_state.n == 1

        # Replacing only the observation invalidates the materialized update and
        # recomputes from the same authoritative source state; it is not an
        # implicit mutable fold.
        set!(state, model.observation, 5.0)
        replacement = get!(state, model.updated)
        @test replacement.n == 1
        @test mean(replacement) == 5.0

        # A caller advances the fold explicitly by promoting the returned state
        # to the next authoritative source value.
        set!(state, model.state, first_state)
        advanced = get!(state, model.updated)
        @test advanced.n == 2
        @test mean(advanced) == 3.0

        freeze!(state, model.updated, advanced)
        set!(state, model.state, empty_state)
        set!(state, model.observation, 99.0)
        @test get!(state, model.average) == 3.0
        unfreeze!(state, model.updated)
        @test get!(state, model.average) == 99.0
    end

    @testset "canonical colored DAG exposes update structure" begin
        model = OSE.build_online_stats_graph()
        update_plan = plan(model;
            have=(:state, :observation),
            want=(:updated, :average, :sample_variance))
        view = visualize(update_plan)
        html = sprint(show, MIME"text/html"(), view)
        dot = dot_source(view)

        @test occursin("class=\"rk-node value have\"", html)
        @test occursin("class=\"rk-node value want\"", html)
        @test occursin("rk-node recipe selected", html)
        @test occursin("#dcfce7", dot)
        @test occursin("#ffedd5", dot)
        @test occursin("#dbeafe", dot)
        @test occursin("update", dot)
    end

    @testset "public docs keep a literal declarative source block" begin
        page = read(joinpath(@__DIR__, "..", "docs", "src", "online-stats.md"),
                    String)
        @test occursin("@kernel updates(", page)
        @test occursin("@kernel partitions(", page)
        @test occursin("diagnostics = @kernel begin", page)
        @test occursin("OnlineStatsExample.record_transition", page)
        @test occursin("transition::NUTSDiagnostics{Float64}", page)
        @test occursin("NUTSDiagnostics(3, 7, 0.91, false, -0.05)", page)
        @test occursin("name = :hmc_transition_diagnostics", page)
        @test occursin("metric_adaptation_report", page)
        @test occursin("welford_var", page)
        @test occursin("model = merge(updates, partitions)", page)
        @test occursin("Main.ReactiveKernelsDocs.execute_example", page)
        @test !occursin("include(", page)
        @test !occursin("Graph()", page)
        @test !occursin("value!(", page)
        @test !occursin("add!(", page)
    end
end
