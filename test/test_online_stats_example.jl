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
        @test empty.n_divergent == 0
        @test isnan(OSE.divergence_rate(empty))
        @test isnan(OSE.divergence_percent(empty))
        @test isnan(OSE.mean_acceptance_rate(empty))
        @test isnan(OSE.mean_leapfrog_steps(empty))
        @test isnan(OSE.mean_stepsize(empty))

        @test_throws DomainError OSE.record_transition(empty, -0.1, 3, 0.1, false)
        @test_throws DomainError OSE.record_transition(empty, 1.1, 3, 0.1, false)
        @test_throws DomainError OSE.record_transition(empty, NaN, 3, 0.1, false)
        @test_throws DomainError OSE.record_transition(empty, 0.8, -1, 0.1, false)
        @test_throws DomainError OSE.record_transition(empty, 0.8, 3, 0.0, false)
        @test_throws DomainError OSE.record_transition(empty, 0.8, 3, Inf, false)
        @test_throws ArgumentError OSE.HMCDiagnosticsAccumulator{Float64}(
            1, OSE.MomentsAccumulator(), OSE.MomentsAccumulator(),
            OSE.MomentsAccumulator())
        @test_throws ArgumentError OSE.HMCDiagnosticsAccumulator{Float64}(
            0, OSE.update(OSE.MomentsAccumulator(), 0.8),
            OSE.MomentsAccumulator(), OSE.MomentsAccumulator())

        records = [
            (acc_rate=0.91, n_steps=7,  stepsize=0.25, diverged=false),
            (acc_rate=0.83, n_steps=15, stepsize=0.20, diverged=true),
            (acc_rate=0.97, n_steps=5,  stepsize=0.30, diverged=false),
            (acc_rate=0.76, n_steps=31, stepsize=0.15, diverged=true),
            (acc_rate=0.88, n_steps=9,  stepsize=0.22, diverged=false),
            (acc_rate=0.94, n_steps=6,  stepsize=0.28, diverged=false),
            (acc_rate=0.69, n_steps=63, stepsize=0.10, diverged=true),
        ]
        fitted = @inferred OSE.fit_diagnostics(records)
        @test OSE.sample_count(fitted) == length(records)
        @test fitted.n_divergent == count(record -> record.diverged, records)
        @test OSE.divergence_rate(fitted) ≈ mean(record.diverged for record in records)
        @test OSE.divergence_percent(fitted) ≈
              100 * mean(record.diverged for record in records)
        @test OSE.mean_acceptance_rate(fitted) ≈
              mean(record.acc_rate for record in records)
        @test OSE.mean_leapfrog_steps(fitted) ≈
              mean(record.n_steps for record in records)
        @test OSE.mean_stepsize(fitted) ≈
              mean(record.stepsize for record in records)
        @test var(fitted.acceptance_rate) ≈
              var([record.acc_rate for record in records])
        @test var(fitted.leapfrog_steps) ≈
              var([record.n_steps for record in records])
        @test var(fitted.stepsize) ≈
              var([record.stepsize for record in records])

        parts = (
            OSE.fit_diagnostics(@view records[1:1]),
            OSE.fit_diagnostics(@view records[2:3]),
            OSE.fit_diagnostics(@view records[4:7]),
        )
        @test isequal(merge(empty, parts[1]), parts[1])
        @test isequal(merge(parts[1], empty), parts[1])
        merged = reduce(merge, parts)
        @test merged.n_divergent == fitted.n_divergent
        @test OSE.sample_count(merged) == OSE.sample_count(fitted)
        @test OSE.mean_acceptance_rate(merged) ≈ OSE.mean_acceptance_rate(fitted)
        @test OSE.mean_leapfrog_steps(merged) ≈ OSE.mean_leapfrog_steps(fitted)
        @test OSE.mean_stepsize(merged) ≈ OSE.mean_stepsize(fitted)

        left_grouped = merge(merge(parts[1], parts[2]), parts[3])
        right_grouped = merge(parts[1], merge(parts[2], parts[3]))
        @test left_grouped.n_divergent == right_grouped.n_divergent
        @test OSE.mean_acceptance_rate(left_grouped) ≈
              OSE.mean_acceptance_rate(right_grouped)
        @test OSE.mean_leapfrog_steps(left_grouped) ≈
              OSE.mean_leapfrog_steps(right_grouped)
        @test OSE.mean_stepsize(left_grouped) ≈ OSE.mean_stepsize(right_grouped)

        # Count-derived rates must also avoid converting large valid counts
        # through narrow storage before division.
        moments16 = OSE.MomentsAccumulator{Float16}(40_000, 0.5, 1.0)
        narrow_a = OSE.HMCDiagnosticsAccumulator{Float16}(
            10_000, moments16, moments16, moments16)
        narrow_b = OSE.HMCDiagnosticsAccumulator{Float16}(
            20_000, moments16, moments16, moments16)
        narrow = @inferred merge(narrow_a, narrow_b)
        @test OSE.sample_count(narrow) == 80_000
        @test narrow.n_divergent == 30_000
        @test OSE.divergence_rate(narrow) === Float16(0.375)
        @test OSE.divergence_percent(narrow) === Float16(37.5)
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
            have=(:diagnostics_state, :acceptance_rate_observation,
                  :leapfrog_steps_observation, :stepsize_observation,
                  :diverged_observation),
            want=(:updated_diagnostics, :diagnostic_divergence_percent,
                  :diagnostic_mean_acceptance_rate,
                  :diagnostic_mean_leapfrog_steps, :diagnostic_mean_stepsize))

        empty_state = OSE.MomentsAccumulator()
        one_state = @inferred update_kernel(empty_state, 1.0)
        two_state = @inferred update_kernel(one_state, 3.0)
        @test @inferred(merge_kernel(one_state, two_state)) isa
              OSE.MomentsAccumulator{Float64}
        @test @inferred(summary_kernel(one_state, 3.0)) == (2.0, 2.0)

        empty_diagnostics = OSE.HMCDiagnosticsAccumulator()
        diagnostic_result = @inferred diagnostics_kernel(
            empty_diagnostics, 0.9, 7, 0.2, false)
        @test diagnostic_result[1] isa OSE.HMCDiagnosticsAccumulator{Float64}
        @test diagnostic_result[2:end] == (0.0, 0.9, 7.0, 0.2)

        diagnostics_update_kernel = prepare(model;
            have=(:diagnostics_state, :acceptance_rate_observation,
                  :leapfrog_steps_observation, :stepsize_observation,
                  :diverged_observation),
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
        @test occursin("name = :hmc_transition_diagnostics", page)
        @test occursin("model = merge(updates, partitions)", page)
        @test occursin("Main.ReactiveKernelsDocs.execute_example", page)
        @test !occursin("include(", page)
        @test !occursin("Graph()", page)
        @test !occursin("value!(", page)
        @test !occursin("add!(", page)
    end
end
